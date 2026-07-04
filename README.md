# Multimodal Financial Fraud Detection & Risk Analytics Pipeline

Hybrid **ML (XGBoost)** + **DL (PyTorch)** fraud evaluation engine with a FastAPI backend
and a React + Tailwind + Recharts frontend.

No LLMs, chatbots, LangGraph, or Federated Learning — pure ML/DL, as specified.

## Architecture

```
Transaction vector
   -> StandardScaler (Scikit-Learn)
   -> XGBoost Classifier            -> ML anomaly score
   -> [scaled features + ML score]
   -> PyTorch MLP (2 hidden layers, ReLU, Dropout, Sigmoid)
   -> Final Fraud Probability -> Risk Status (Safe / Suspect / High Risk)
```

- `backend/model.py` — Component A (ML) + Component B (DL), trained on a
  synthetic-but-labeled dataset at API startup (no external dataset dependency,
  so the project runs out of the box).
- `backend/main.py` — Component C, the FastAPI gateway exposing
  `POST /api/v1/evaluate-fraud`.
- `frontend/src/App.js` — Risk Operations Console: transaction simulator,
  live telemetry gauges, and a decision matrix breakdown.

## Run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 main.py
```

The API starts on `http://localhost:8000`. On startup it trains the XGBoost
+ PyTorch pipeline in-memory (~20-30 seconds) — this is expected, not a hang.

Interactive API docs: `http://localhost:8000/docs`

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/evaluate-fraud \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 4500,
    "location_divergence_score": 78,
    "device_trust_score": 15,
    "transaction_velocity": 9
  }'
```

## Run the frontend

```bash
cd frontend
npm install
npm start
```

Opens on `http://localhost:3000` and calls the backend at
`http://localhost:8000` by default. To point at a different backend URL,
set `REACT_APP_API_BASE` before starting:

```bash
REACT_APP_API_BASE=http://localhost:8000 npm start
```

## Notes for your resume writeup

- The ML stage (XGBoost) produces a baseline anomaly score from raw
  transaction metadata; that score is then *fed into* the DL stage as an
  additional engineered feature — this is the "hybrid ML→DL sequential
  routing" architecture referenced in the brief, not two independent models.
- The synthetic dataset generator (`_generate_synthetic_dataset` in
  `model.py`) uses a hidden latent-risk formula to label transactions, so
  the pipeline has a genuine, non-trivial pattern to learn — feel free to
  swap it for a real dataset (e.g. IEEE-CIS Fraud Detection, or the UCI
  Default of Credit Card Clients dataset repurposed for fraud) if you want
  real-world benchmark numbers to cite.
- SHAP/feature-importance is currently pulled from the XGBoost stage only
  (`feature_importances_`); adding SHAP values proper is a natural v2
  extension if you want deeper explainability on the DL side too.
