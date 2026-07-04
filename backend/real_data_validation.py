"""
real_data_validation.py
------------------------
Validates the same ML (XGBoost) -> DL (PyTorch MLP) hybrid architecture
used in model.py against a REAL, messy, publicly available fraud dataset
(the Kaggle "Credit Card Fraud Detection" dataset: 284,807 European
cardholder transactions from Sept 2013, 492 of them fraudulent — a
0.17% fraud rate, i.e. a genuinely imbalanced real-world classification
problem, not the synthetic 15%-by-construction dataset used for the
live demo in model.py).

This script exists to answer the interview question "did you ever test
this on real data?" with a yes, and to back it up with honest numbers.

Why the live API demo (model.py) doesn't use this dataset directly:
    The dashboard's 4 input sliders (Amount, Location Divergence,
    Device Trust, Velocity) are interpretable by design so a reviewer
    can play with them live. This dataset's 28 features (V1-V28) are
    PCA-anonymized by Kaggle for privacy and have no interpretable
    meaning, so they can't power an interactive slider demo. This
    script is the "does the underlying architecture actually hold up
    on real, imbalanced, non-interpretable data" proof, kept separate
    from the live product demo.

Usage:
    1. Download creditcard.csv (see backend/data/README.md) and place
       it at backend/data/creditcard.csv
    2. pip install -r requirements-eval.txt
    3. python real_data_validation.py

Outputs:
    - Console: classification report, precision/recall/F1/ROC-AUC/PR-AUC
    - backend/artifacts/real_data_eval/metrics.json
    - backend/artifacts/real_data_eval/confusion_matrix.png
    - backend/artifacts/real_data_eval/roc_curve.png
    - backend/artifacts/real_data_eval/precision_recall_curve.png
    - backend/artifacts/real_data_eval/shap_summary.png
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from model import FraudMLP  # reuse the exact same DL architecture as the live pipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "real_data_eval")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_SEED = 42


def load_real_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. See backend/data/README.md "
            "for download instructions."
        )
    return pd.read_csv(DATA_PATH)


def main():
    print("Loading real dataset...")
    df = load_real_data()
    print(f"Loaded {len(df):,} transactions, {df['Class'].sum():,} labeled fraud "
          f"({df['Class'].mean() * 100:.3f}% fraud rate)")

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    feature_names = df.drop(columns=["Class"]).columns.tolist()

    # Stratified split so the tiny fraud class is proportionally represented
    # in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Class imbalance handling ---
    # At a 0.17% fraud rate, a model optimizing raw accuracy can score
    # 99.8% by predicting "not fraud" every time and never catch a
    # single fraudulent transaction. scale_pos_weight tells XGBoost to
    # weight each fraud example proportionally to how rare it is, so
    # missing a fraud is penalized as heavily as it should be.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"\nClass imbalance in training set: {n_neg:,} legit vs {n_pos:,} fraud "
          f"(scale_pos_weight={scale_pos_weight:.1f})")

    print("\nTraining XGBoost (Component A)...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",  # PR-AUC, the right metric to optimize under heavy imbalance
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )
    xgb_model.fit(X_train_scaled, y_train)
    ml_signal_train = xgb_model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
    ml_signal_test = xgb_model.predict_proba(X_test_scaled)[:, 1].reshape(-1, 1)

    print("Training PyTorch MLP (Component B)...")
    dl_input_train = np.hstack([X_train_scaled, ml_signal_train])
    dl_input_test = np.hstack([X_test_scaled, ml_signal_test])

    dl_model = FraudMLP(input_dim=dl_input_train.shape[1])
    X_t = torch.tensor(dl_input_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Same imbalance problem applies to the DL layer's loss function:
    # weight positive (fraud) examples by the same ratio so rare fraud
    # cases aren't drowned out during backprop.
    pos_weight = torch.tensor([scale_pos_weight], dtype=torch.float32)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=5e-3, weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    dl_model.train()
    for epoch in range(30):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = dl_model(xb)
            raw_loss = criterion(preds, yb)
            weights = torch.where(yb == 1, pos_weight, torch.tensor(1.0))
            loss = (raw_loss * weights).mean()
            loss.backward()
            optimizer.step()
    dl_model.eval()

    with torch.no_grad():
        dl_scores_test = dl_model(torch.tensor(dl_input_test, dtype=torch.float32)).squeeze(1).numpy()

    combined_scores = np.maximum(ml_signal_test.ravel(), dl_scores_test)
    combined_preds = (combined_scores >= 0.5).astype(int)

    # --- Metrics ---
    precision = precision_score(y_test, combined_preds)
    recall = recall_score(y_test, combined_preds)
    f1 = f1_score(y_test, combined_preds)
    roc_auc = roc_auc_score(y_test, combined_scores)
    pr_auc = average_precision_score(y_test, combined_scores)  # more honest than ROC-AUC under heavy imbalance
    tn, fp, fn, tp = confusion_matrix(y_test, combined_preds).ravel()

    metrics = {
        "dataset": "Kaggle Credit Card Fraud Detection (European cardholders, Sept 2013)",
        "n_transactions": int(len(df)),
        "fraud_rate_pct": round(float(df["Class"].mean() * 100), 4),
        "test_set_size": int(len(y_test)),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }

    print("\n" + "=" * 60)
    print("REAL-DATA VALIDATION RESULTS")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print("\nFull classification report:")
    print(classification_report(y_test, combined_preds, target_names=["Legit", "Fraud"]))

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Plots ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        [[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Legit", "Pred Fraud"], yticklabels=["Actual Legit", "Actual Fraud"],
    )
    plt.title("Confusion Matrix — Real Data Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, combined_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Real Data Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, combined_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(rec_curve, prec_curve, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Real Data Validation\n(more informative than ROC under heavy imbalance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()

    # --- SHAP explainability for the XGBoost stage ---
    print("\nComputing SHAP values (this can take a minute)...")
    try:
        import shap
        sample = X_test_scaled[:2000]  # sample for speed
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(sample)
        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot to {ARTIFACT_DIR}/shap_summary.png")
    except Exception as exc:
        print(f"SHAP plot skipped: {exc}")

    print(f"\nAll artifacts saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
