"""
ChurnScope - Streamlit front-end for the Telecom Customer Churn classifiers.

The app only *loads* the five pre-trained pipelines from model/ and evaluates
them on the uploaded test CSV. No model is retrained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.evaluation import METRIC_NAMES, compute_metrics, positive_class_scores
from src.features import RAW_FEATURES, TARGET_COL, validate_input_frame

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_TEST_PATH = APP_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

CLASS_LABELS = ["Stays (0)", "Churns (1)"]
ACCENT = "#0F766E"

st.set_page_config(
    page_title="ChurnScope | Telecom Churn Classifiers",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; }}
      .hero {{
          border-left: 6px solid {ACCENT};
          padding: 0.6rem 1rem;
          background: rgba(15,118,110,0.07);
          border-radius: 6px;
          margin-bottom: 1rem;
      }}
      .hero h1 {{ margin: 0; font-size: 2rem; }}
      .hero p {{ margin: 0.2rem 0 0 0; opacity: 0.8; }}
      div[data-testid="stMetric"] {{
          border: 1px solid rgba(128,128,128,0.25);
          border-radius: 8px;
          padding: 0.5rem 0.8rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_training_report() -> dict:
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return {v["display_name"]: v for v in raw.values()}


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def normalise_target(series: pd.Series) -> pd.Series:
    """Accept 0/1, True/False or yes/no style churn labels."""
    if series.dtype == object or series.dtype == bool:
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
        series = series.astype(str).str.strip().str.lower().map(mapping)
    return series.astype(int)


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_score = positive_class_scores(model, X)
    return {
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": compute_metrics(y, y_pred, y_score),
    }


# ---------------------------------------------------------------- plots
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array(
        [[f"{cm[i, j]}\n({pct[i, j]:.0%})" for j in range(2)] for i in range(2)]
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Greens", cbar=False,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
        linewidths=1, linecolor="white", annot_kws={"fontsize": 11},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc(curves: dict, highlight: str = None):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, (fpr, tpr, auc) in curves.items():
        lw = 2.6 if name == highlight else 1.3
        alpha = 1.0 if (highlight is None or name == highlight) else 0.55
        ax.plot(fpr, tpr, lw=lw, alpha=alpha, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
models = load_models()
training_report = load_training_report()

with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()), index=1)

    st.divider()
    st.subheader("Test data")
    uploaded = st.file_uploader(
        "Upload test CSV (features + `churn` column)", type=["csv"]
    )
    use_sample = st.checkbox(
        "Use the bundled test_data.csv instead",
        value=uploaded is None,
        help="Handy for a quick demo. The upload takes priority when unchecked.",
    )
    if SAMPLE_TEST_PATH.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=SAMPLE_TEST_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

    if model_name in training_report:
        st.divider()
        st.caption("Tuned hyper-parameters (5-fold CV, F1 scoring)")
        st.json(training_report[model_name]["best_params"])


# ---------------------------------------------------------------- header
st.markdown(
    """
    <div class="hero">
      <h1>📡 ChurnScope</h1>
      <p>Will a telecom subscriber leave? Five pre-trained classifiers,
      evaluated live on the test data you upload.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- data
if uploaded is not None and not use_sample:
    df = read_csv(uploaded)
    source_label = f"uploaded file `{uploaded.name}`"
elif use_sample and SAMPLE_TEST_PATH.exists():
    df = read_csv(SAMPLE_TEST_PATH)
    source_label = "bundled `test_data.csv`"
else:
    st.info(
        "Upload a test CSV from the sidebar (or tick *Use the bundled test_data.csv*) "
        "to start the evaluation."
    )
    st.stop()

missing = validate_input_frame(df, require_target=True)
if missing:
    st.error(
        "The CSV is missing required column(s): "
        + ", ".join(f"`{c}`" for c in missing)
        + ". Download the sample file from the sidebar to see the expected schema."
    )
    st.stop()

try:
    y_true = normalise_target(df[TARGET_COL])
except (ValueError, TypeError):
    st.error("Could not interpret the `churn` column - use 0/1, True/False or yes/no.")
    st.stop()

X = df[RAW_FEATURES]

c1, c2, c3 = st.columns(3)
c1.metric("Rows evaluated", f"{len(df):,}")
c2.metric("Actual churners", f"{int(y_true.sum()):,}")
c3.metric("Churn rate", f"{y_true.mean():.1%}")
st.caption(f"Data source: {source_label}")

tab_eval, tab_compare, tab_pred, tab_about = st.tabs(
    ["🔍 Model evaluation", "🏁 Compare all models", "📋 Predictions", "ℹ️ About"]
)

# ---------------------------------------------------------------- tab 1
with tab_eval:
    result = evaluate(models[model_name], X, y_true)
    m = result["metrics"]

    st.subheader(f"{model_name} - evaluation metrics")
    cols = st.columns(len(METRIC_NAMES))
    for col, metric in zip(cols, METRIC_NAMES):
        value = m[metric]
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")

    if np.isnan(m["AUC"]):
        st.warning("Only one class is present in the uploaded data, so AUC cannot be computed.")

    left, right = st.columns(2)
    with left:
        st.pyplot(
            plot_confusion_matrix(y_true, result["y_pred"], f"Confusion matrix - {model_name}"),
            width="stretch",
        )
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, result["y_pred"], labels=[0, 1],
            target_names=CLASS_LABELS, output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
            width="stretch",
        )
        tn, fp, fn, tp = confusion_matrix(y_true, result["y_pred"], labels=[0, 1]).ravel()
        st.markdown(
            f"- **{tp}** churners correctly flagged, **{fn}** churners missed  \n"
            f"- **{fp}** loyal customers wrongly flagged as churners"
        )

    if not np.isnan(m["AUC"]):
        fpr, tpr, _ = roc_curve(y_true, result["y_score"])
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.pyplot(plot_roc({model_name: (fpr, tpr, m["AUC"])}), width="stretch")

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader("All five models on the same test data")
    all_results = {name: evaluate(mdl, X, y_true) for name, mdl in models.items()}
    table = pd.DataFrame(
        [{"ML Model Name": n, **r["metrics"]} for n, r in all_results.items()]
    ).set_index("ML Model Name")

    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#99F6E4"),
        width="stretch",
    )
    best = table["MCC"].idxmax()
    st.success(
        f"Best model on this data by MCC: **{best}** "
        f"(MCC = {table.loc[best, 'MCC']:.4f}, F1 = {table.loc[best, 'F1']:.4f})"
    )

    metric_to_plot = st.radio(
        "Metric to chart", METRIC_NAMES, index=METRIC_NAMES.index("F1"), horizontal=True
    )
    chart_df = table[[metric_to_plot]].sort_values(metric_to_plot)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [ACCENT if n == best else "#94A3B8" for n in chart_df.index]
    ax.barh(chart_df.index, chart_df[metric_to_plot], color=colors)
    for i, v in enumerate(chart_df[metric_to_plot]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(metric_to_plot)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    if len(np.unique(y_true)) > 1:
        curves = {}
        for n, r in all_results.items():
            fpr, tpr, _ = roc_curve(y_true, r["y_score"])
            curves[n] = (fpr, tpr, r["metrics"]["AUC"])
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.pyplot(plot_roc(curves, highlight=best), width="stretch")

# ---------------------------------------------------------------- tab 3
with tab_pred:
    st.subheader(f"Row-level predictions - {model_name}")
    pred_df = df.copy()
    pred_df["predicted_churn"] = result["y_pred"]
    pred_df["correct"] = pred_df["predicted_churn"] == y_true
    show = st.radio(
        "Show", ["All rows", "Misclassified only", "Predicted churners only"], horizontal=True
    )
    if show == "Misclassified only":
        pred_df = pred_df[~pred_df["correct"]]
    elif show == "Predicted churners only":
        pred_df = pred_df[pred_df["predicted_churn"] == 1]
    st.dataframe(pred_df, width="stretch", height=420)
    st.download_button(
        "⬇️ Download these predictions",
        pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------- tab 4
with tab_about:
    st.markdown(
        """
        **Dataset:** Telecom Customer Churn (5,000 subscribers, 20 input features,
        binary target `churn`). Each row describes one customer's plan and usage -
        day / evening / night / international minutes and calls, voicemail and
        international plan flags, and number of customer-service calls.

        **Pipeline (identical for every model):**
        1. Drop the `phone_number` identifier.
        2. Replace the four `*_charge` columns (charge = minutes x fixed tariff)
           with one `total_charge` feature, and add `total_calls`,
           `intl_minutes_per_call` and a `high_service_calls` flag (4+ calls).
        3. One-hot encode `state` and `area_code`, standardise numeric columns.
        4. Classifier, tuned with 5-fold stratified CV on F1.

        Models were trained once on an 80% stratified split and saved with joblib.
        This app never retrains - it only loads the saved pipelines and scores the
        uploaded test data.
        """
    )
    if training_report:
        st.markdown("**Hold-out results recorded at training time**")
        ref = pd.DataFrame(
            {n: v["test_metrics"] for n, v in training_report.items()}
        ).T
        st.dataframe(ref.style.format("{:.4f}"), width="stretch")
