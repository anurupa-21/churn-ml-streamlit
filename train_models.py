"""
Train and evaluate five classifiers on the Telecom Customer Churn dataset.

Run from the project root:
    python train_models.py

Outputs
-------
test_data.csv                    held-out 20% split (raw features + churn), used by the app
model/<model_key>.joblib         one fitted sklearn Pipeline per classifier
model/metrics.json               test-set metrics + best hyper-parameters per model
model/model_comparison.csv       the comparison table used in README.md
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import compute_metrics, positive_class_scores
from src.features import (
    RAW_FEATURES,
    TARGET_COL,
    ChurnFeatureBuilder,
    build_column_transformer,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "telecom_churn.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

RANDOM_STATE = 2026
TEST_SIZE = 0.20

# key -> (display name, estimator, hyper-parameter grid)
MODEL_SPECS = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "clf__max_depth": [4, 6, 8, 10, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        },
    ),
    "knn": (
        "KNN",
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 9, 15], "clf__weights": ["uniform", "distance"]},
    ),
    "naive_bayes": (
        "Naive Bayes",
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    ),
    "svm": (
        "Support Vector Machine",
        SVC(kernel="rbf", random_state=RANDOM_STATE),  # AUC uses decision_function scores
        {"clf__C": [1.0, 5.0, 10.0], "clf__gamma": ["scale", 0.01]},
    ),
}


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", ChurnFeatureBuilder()),
            ("preprocess", build_column_transformer()),
            ("clf", estimator),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Class balance:\n", df[TARGET_COL].value_counts(normalize=True).round(3))

    X = df[RAW_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Save the untouched held-out split - this is the file uploaded to the app.
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows -> {TEST_DATA_PATH.name}")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for key, (display_name, estimator, grid) in MODEL_SPECS.items():
        start = time.time()
        # Tune on F1 because the positive (churn) class is only ~14% of the data,
        # so plain accuracy would reward predicting "no churn" for everyone.
        search = GridSearchCV(
            build_pipeline(estimator), grid, scoring="f1", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_score = positive_class_scores(best_model, X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)

        joblib.dump(best_model, MODEL_DIR / f"{key}.joblib")
        results[key] = {
            "display_name": display_name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": round(search.best_score_, 4),
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "train_seconds": round(time.time() - start, 1),
        }
        print(
            f"{display_name:<24} "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            + f" | best={results[key]['best_params']}"
        )

    with open(MODEL_DIR / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    table = pd.DataFrame(
        [{"ML Model Name": r["display_name"], **r["test_metrics"]} for r in results.values()]
    )
    table.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print("\nComparison table:\n", table.to_string(index=False))


if __name__ == "__main__":
    main()
