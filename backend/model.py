"""
model.py
--------
Component A: Tabular ML Engine (Scikit-Learn StandardScaler + XGBoost Classifier)
Component B: Deep Learning Sequential Risk Layer (PyTorch MLP)

Pipeline flow:
    raw transaction vector
        -> StandardScaler (runtime feature scaling)
        -> XGBoost Classifier -> baseline anomaly probability (ML signal)
        -> [scaled features + ML anomaly signal] concatenated into an augmented vector
        -> PyTorch MLP (2 hidden layers, ReLU, Dropout, Sigmoid)
        -> final fraud probability

No LLMs, no chatbots, no LangGraph, no Federated Learning anywhere in this file.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from config import settings

MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    "transaction_amount",
    "location_divergence_score",
    "device_trust_score",
    "transaction_velocity",
]

# ---------------------------------------------------------------------------
# Synthetic training data generator
#
# In a production system this would be replaced by a real historical
# transactions warehouse pull. For a self-contained resume project, we
# generate a labeled synthetic dataset with a known underlying risk
# function so the pipeline has something non-trivial to learn.
# ---------------------------------------------------------------------------
def _generate_synthetic_dataset(n_samples: int = 20000, seed: int = None):
    seed = seed if seed is not None else settings.RANDOM_SEED
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=4.2, sigma=1.1, size=n_samples)          # skewed, most small, some large
    location_divergence = rng.beta(2, 5, size=n_samples) * 100           # 0-100, mostly low
    device_trust = rng.beta(5, 2, size=n_samples) * 100                  # 0-100, mostly high (trusted)
    velocity = rng.poisson(lam=2.5, size=n_samples).astype(float)        # transactions in last hour

    # --- Edge-case augmentation ---
    # The distributions above rarely produce extreme values in *every*
    # dimension at once (e.g. huge amount + max location divergence +
    # min device trust + max velocity, all together). Without seeing
    # such combinations during training, the DL layer extrapolates
    # poorly on them at inference time. We inject a batch of uniformly
    # sampled full-range combinations (including fully-extreme corners)
    # so the model actually learns the full input space, not just the
    # "typical" region of it.
    n_edge = int(n_samples * 0.15)
    edge_amount = rng.uniform(0, 20000, size=n_edge)
    edge_location = rng.uniform(0, 100, size=n_edge)
    edge_device = rng.uniform(0, 100, size=n_edge)
    edge_velocity = rng.uniform(0, 30, size=n_edge)

    amount = np.concatenate([amount, edge_amount])
    location_divergence = np.concatenate([location_divergence, edge_location])
    device_trust = np.concatenate([device_trust, edge_device])
    velocity = np.concatenate([velocity, edge_velocity])

    # latent risk score used only to generate labels (not visible to model directly)
    latent_risk = (
        0.00035 * amount
        + 0.028 * location_divergence
        - 0.022 * device_trust
        + 0.09 * velocity
        + rng.normal(0, 0.6, size=len(amount))
    )
    threshold = np.quantile(latent_risk, 0.85)
    labels = (latent_risk > threshold).astype(int)

    X = np.column_stack([amount, location_divergence, device_trust, velocity])
    y = labels
    return X, y


class FraudMLP(nn.Module):
    """
    Deep Learning Sequential Risk Layer.

    Ingests the normalized feature vector concatenated with the upstream
    ML anomaly signal, and routes it through two hidden layers with
    ReLU activations and Dropout regularization, terminating in a
    Sigmoid output representing fraud probability in [0, 1].
    """

    def __init__(self, input_dim: int, hidden_dim_1: int = 32, hidden_dim_2: int = 16, dropout_p: float = 0.25):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HybridFraudPipeline:
    """
    Orchestrates the ML -> DL sequential inference pipeline and owns
    the trained artifacts (scaler, XGBoost classifier, PyTorch MLP).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=settings.RANDOM_SEED,
        )
        # input to MLP = 4 scaled tabular features + 1 ML anomaly signal
        self.dl_model = FraudMLP(input_dim=len(FEATURE_NAMES) + 1)
        self._is_fitted = False
        self.evaluation_metrics: dict | None = None

    def fit(self):
        X, y = _generate_synthetic_dataset()

        # --- Held-out test split ---
        # Trained only on X_train/y_train; X_test/y_test are never seen
        # during fitting and exist purely to report honest generalization
        # metrics (precision/recall/F1/ROC-AUC/confusion matrix), not to
        # rubber-stamp training accuracy.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.RANDOM_SEED, stratify=y
        )

        # --- Component A: ML tabular engine ---
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.xgb_model.fit(X_train_scaled, y_train)
        ml_anomaly_signal_train = self.xgb_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)

        # --- Component B: DL sequential risk layer ---
        dl_input_train = np.hstack([X_train_scaled, ml_anomaly_signal_train])
        self._train_dl_layer(dl_input_train, y_train)

        self._is_fitted = True
        self.evaluation_metrics = self._evaluate(X_test, y_test)
        return self

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Computes precision/recall/F1/ROC-AUC/confusion matrix on the held-out
        test split for both the XGBoost stage and the final combined score.
        Fraud detection is an imbalanced classification problem (here an
        85/15 split by construction), so accuracy alone is meaningless —
        recall (catching fraud) and precision (not over-flagging legit
        transactions) are reported separately.
        """
        X_test_scaled = self.scaler.transform(X_test)
        ml_scores = self.xgb_model.predict_proba(X_test_scaled)[:, 1]

        dl_input_test = np.hstack([X_test_scaled, ml_scores.reshape(-1, 1)])
        with torch.no_grad():
            dl_scores = self.dl_model(torch.tensor(dl_input_test, dtype=torch.float32)).squeeze(1).numpy()

        combined_scores = np.maximum(ml_scores, dl_scores)
        combined_preds = (combined_scores >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, combined_preds).ravel()

        return {
            "test_set_size": int(len(y_test)),
            "fraud_rate_in_test_set": round(float(y_test.mean()), 4),
            "precision": round(float(precision_score(y_test, combined_preds)), 4),
            "recall": round(float(recall_score(y_test, combined_preds)), 4),
            "f1_score": round(float(f1_score(y_test, combined_preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, combined_scores)), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    def _train_dl_layer(self, X: np.ndarray, y: np.ndarray, epochs: int = None, lr: float = None, batch_size: int = None):
        epochs = epochs if epochs is not None else settings.DL_EPOCHS
        lr = lr if lr is not None else settings.DL_LEARNING_RATE
        batch_size = batch_size if batch_size is not None else settings.DL_BATCH_SIZE

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        optimizer = torch.optim.Adam(self.dl_model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.dl_model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.dl_model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        self.dl_model.eval()

    def predict(self, feature_vector: list[float]) -> dict:
        """
        Runs a single transaction vector through the full ML -> DL pipeline.
        feature_vector order must match FEATURE_NAMES.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call .fit() first.")

        X = np.array(feature_vector, dtype=float).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        ml_anomaly_score = float(self.xgb_model.predict_proba(X_scaled)[:, 1][0])

        dl_input = np.hstack([X_scaled, [[ml_anomaly_score]]])
        with torch.no_grad():
            dl_input_t = torch.tensor(dl_input, dtype=torch.float32)
            fraud_probability = float(self.dl_model(dl_input_t).item())

        # feature importance snapshot from the XGBoost stage, for the
        # frontend's "Decision Matrix Breakdown" panel
        importances = self.xgb_model.feature_importances_.tolist()
        feature_importance = {
            name: round(float(score), 4) for name, score in zip(FEATURE_NAMES, importances)
        }

        # --- Final combined risk score ---
        # Rather than trusting the DL layer's output in isolation, the
        # final score takes the *maximum* of the ML anomaly score and the
        # DL fraud probability. This is a deliberate safety design choice:
        # if either model independently flags a transaction as high-risk,
        # the system should not silently downgrade it just because the
        # other model disagrees (which matters most on rare/out-of-
        # distribution inputs where one model may extrapolate poorly).
        combined_risk_score = max(ml_anomaly_score, fraud_probability)

        return {
            "ml_anomaly_score": round(ml_anomaly_score, 4),
            "dl_fraud_probability": round(fraud_probability, 4),
            "combined_risk_score": round(combined_risk_score, 4),
            "feature_importance": feature_importance,
        }


# ---------------------------------------------------------------------------
# Module-level singleton, trained once at API startup
# ---------------------------------------------------------------------------
_pipeline_instance: HybridFraudPipeline | None = None


def get_pipeline() -> HybridFraudPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HybridFraudPipeline().fit()
    return _pipeline_instance
