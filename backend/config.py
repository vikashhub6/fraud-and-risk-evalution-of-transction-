"""
config.py
---------
Single source of truth for all configuration. Every value here is read
from environment variables (via a .env file for local dev, or real
environment variables in staging/production). Nothing is hardcoded
anywhere else in the codebase — main.py and model.py both import
their settings from this module.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in this directory, if present


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default


class Settings:
    # --- Server ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _get_int("PORT", 8000)
    RELOAD: bool = _get_bool("RELOAD", False)

    # --- CORS ---
    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]

    # --- Risk thresholds ---
    RISK_THRESHOLD_SUSPECT: float = _get_float("RISK_THRESHOLD_SUSPECT", 0.33)
    RISK_THRESHOLD_HIGH: float = _get_float("RISK_THRESHOLD_HIGH", 0.66)

    # --- Model training ---
    DL_EPOCHS: int = _get_int("DL_EPOCHS", 150)
    DL_LEARNING_RATE: float = _get_float("DL_LEARNING_RATE", 5e-3)
    DL_BATCH_SIZE: int = _get_int("DL_BATCH_SIZE", 512)
    RANDOM_SEED: int = _get_int("RANDOM_SEED", 42)


settings = Settings()

if not settings.ALLOWED_ORIGINS:
    raise RuntimeError(
        "ALLOWED_ORIGINS is not set. Create a .env file (see .env.example) "
        "and set ALLOWED_ORIGINS to your frontend's URL."
    )
