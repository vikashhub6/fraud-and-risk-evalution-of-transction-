# Real-data validation dataset

`real_data_validation.py` expects `creditcard.csv` in this folder.

This is the public **Kaggle Credit Card Fraud Detection** dataset:
284,807 real European cardholder transactions from September 2013,
with 492 labeled frauds (a 0.17% fraud rate). Features `V1`-`V28` are
PCA-anonymized by the original publisher for privacy; `Time`, `Amount`,
and `Class` (0 = legit, 1 = fraud) are kept as-is.

The file is ~98MB, so it is **not committed to this repo** (see
`.gitignore`). To regenerate the real-data evaluation artifacts:

1. Download `creditcard.csv` from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   (or any mirror of the same dataset)
2. Place it at `backend/data/creditcard.csv`
3. From `backend/`, run:
   ```
   pip install -r requirements-eval.txt
   python real_data_validation.py
   ```

This regenerates `backend/artifacts/real_data_eval/metrics.json` and
the confusion matrix / ROC / precision-recall / SHAP plots.
