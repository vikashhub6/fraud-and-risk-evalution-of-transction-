"""
main.py
-------
Component C: Unified Inference REST API Gateway

Exposes POST /api/v1/evaluate-fraud which:
  1. Validates incoming transaction metrics with strict Pydantic V2 schemas
  2. Runs the ML (XGBoost) -> DL (PyTorch MLP) sequential pipeline
  3. Returns a structured JSON response with the final Risk Status
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from config import settings
from model import get_pipeline, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Pydantic V2 schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    transaction_amount: float = Field(..., ge=0, le=1_000_000, description="Transaction amount in USD")
    location_divergence_score: float = Field(
        ..., ge=0, le=100, description="0 = transaction location matches user history, 100 = maximally divergent"
    )
    device_trust_score: float = Field(
        ..., ge=0, le=100, description="0 = brand new/untrusted device, 100 = fully trusted known device"
    )
    transaction_velocity: float = Field(
        ..., ge=0, le=100, description="Number of transactions by this account in the last rolling hour"
    )


class RiskBreakdown(BaseModel):
    ml_anomaly_score: float
    dl_fraud_probability: float
    combined_risk_score: float
    feature_importance: dict[str, float]


class FraudEvaluationResponse(BaseModel):
    risk_status: str
    risk_breakdown: RiskBreakdown
    input_echo: TransactionInput


# ---------------------------------------------------------------------------
# App lifecycle: train the hybrid pipeline once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # triggers training / warms the singleton
    yield


app = FastAPI(
    title="Multimodal Financial Fraud Detection & Risk Analytics Pipeline",
    description="Hybrid ML (XGBoost) + DL (PyTorch) fraud evaluation engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # from .env, never hardcoded / never "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _classify_risk(fraud_probability: float) -> str:
    if fraud_probability < settings.RISK_THRESHOLD_SUSPECT:
        return "Safe"
    elif fraud_probability < settings.RISK_THRESHOLD_HIGH:
        return "Suspect"
    else:
        return "High Risk"


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "service": "fraud-detection-pipeline"}


@app.get("/api/v1/model-metrics")
async def model_metrics():
    """
    Returns held-out test set evaluation metrics (precision/recall/F1/
    ROC-AUC/confusion matrix) for the currently trained pipeline. These
    are computed once at training time on data the model never saw
    during fitting, not on the training set.
    """
    pipeline = get_pipeline()
    if pipeline.evaluation_metrics is None:
        raise HTTPException(status_code=503, detail="Evaluation metrics not yet available")
    return pipeline.evaluation_metrics


@app.post("/api/v1/evaluate-fraud", response_model=FraudEvaluationResponse)
async def evaluate_fraud(payload: TransactionInput):
    try:
        pipeline = get_pipeline()
        vector = [
            payload.transaction_amount,
            payload.location_divergence_score,
            payload.device_trust_score,
            payload.transaction_velocity,
        ]
        result = pipeline.predict(vector)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline inference failed: {exc}") from exc

    risk_status = _classify_risk(result["combined_risk_score"])

    return FraudEvaluationResponse(
        risk_status=risk_status,
        risk_breakdown=RiskBreakdown(**result),
        input_echo=payload,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.RELOAD)
