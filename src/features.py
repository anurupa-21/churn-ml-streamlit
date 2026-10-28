"""
Feature engineering and preprocessing for the Telecom Customer Churn dataset.

The same objects are used by train_models.py (at training time) and by app.py
(at inference time), so the uploaded test CSV goes through exactly the same
transformations that the models were trained on.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "churn"

# Columns expected in any input CSV (raw dataset schema, without the target).
RAW_FEATURES = [
    "state",
    "account_length",
    "area_code",
    "phone_number",
    "international_plan",
    "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_day_charge",
    "total_eve_minutes",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_minutes",
    "total_night_calls",
    "total_night_charge",
    "total_intl_minutes",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls",
]

# Identifier column - carries no predictive signal, dropped before modelling.
ID_COLS = ["phone_number"]

# Charge = minutes x fixed tariff, so the charge columns are perfectly
# collinear with the minutes columns. We keep minutes and replace the four
# charge columns with a single total_charge feature.
CHARGE_COLS = [
    "total_day_charge",
    "total_eve_charge",
    "total_night_charge",
    "total_intl_charge",
]

CATEGORICAL_FEATURES = ["state", "area_code"]

NUMERIC_FEATURES = [
    "account_length",
    "international_plan",
    "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_eve_minutes",
    "total_eve_calls",
    "total_night_minutes",
    "total_night_calls",
    "total_intl_minutes",
    "total_intl_calls",
    "number_customer_service_calls",
    # engineered
    "total_charge",
    "total_calls",
    "intl_minutes_per_call",
    "high_service_calls",
]

# Threshold chosen from EDA: churn rate jumps sharply from 4 service calls on.
HIGH_SERVICE_CALL_THRESHOLD = 4


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds engineered features and removes redundant / identifier columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        df["total_charge"] = df[CHARGE_COLS].sum(axis=1)
        df["total_calls"] = (
            df["total_day_calls"]
            + df["total_eve_calls"]
            + df["total_night_calls"]
            + df["total_intl_calls"]
        )
        df["intl_minutes_per_call"] = np.where(
            df["total_intl_calls"] > 0,
            df["total_intl_minutes"] / df["total_intl_calls"].replace(0, np.nan),
            0.0,
        )
        df["high_service_calls"] = (
            df["number_customer_service_calls"] >= HIGH_SERVICE_CALL_THRESHOLD
        ).astype(int)

        df = df.drop(columns=ID_COLS + CHARGE_COLS, errors="ignore")
        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
        return df


def build_column_transformer() -> ColumnTransformer:
    """One-hot encode categoricals, standardise numerics."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def validate_input_frame(df: pd.DataFrame, require_target: bool = True) -> list:
    """Return a list of missing column names (empty list means the frame is valid)."""
    required = RAW_FEATURES + ([TARGET_COL] if require_target else [])
    return [c for c in required if c not in df.columns]
