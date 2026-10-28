"""
ChurnScope - Streamlit front-end for the Telecom Customer Churn classifiers.
 
The app only *loads* the five pre-trained pipelines from model/ and evaluates
them on the uploaded test CSV. No model is retrained here.
"""
 
import json
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
 
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
 
MODEL_NOTES = {
    "Logistic Regression": "Linear baseline with balanced class weights. Catches many churners but raises more false alarms.",
    "Decision Tree": "Depth-limited tree that learns rule-like churn patterns such as high bills combined with repeated support calls.",
    "KNN": "Votes using the five most similar customers. Precise, but the loyal majority often outvotes churners.",
    "Naive Bayes": "Gaussian model that assumes independent, normally distributed features - an assumption this data breaks.",
    "Support Vector Machine": "RBF-kernel margin classifier. Learns a curved boundary and keeps false alarms low.",
}
 
METRIC_HELP = {
    "Accuracy": "Share of all customers classified correctly",
    "AUC": "How well scores rank churners above loyal customers",
    "Precision": "Of customers flagged as churners, the share who really left",
    "Recall": "Of customers who really left, the share that was flagged",
    "F1": "Balance of precision and recall",
    "MCC": "Correlation between prediction and truth, robust to class imbalance",
}
 
# Design tokens
INK = "#172554"
PAPER = "#F7F9FC"
SIGNAL = "#0D9488"
DROP = "#E11D48"
AMBER = "#D97706"
SLATE = "#64748B"
LINE = "#E2E8F0"
MODEL_COLORS = {
    "Logistic Regression": "#6366F1",
    "Decision Tree": SIGNAL,
    "KNN": "#0EA5E9",
    "Naive Bayes": "#A855F7",
    "Support Vector Machine": AMBER,
}
 
st.set_page_config(
    page_title="ChurnScope | Telecom Churn Classifiers",
    page_icon="📶",
    layout="wide",
)
 
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap');
 
      html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
          font-family: 'Sora', sans-serif;
      }}
      .block-container {{ padding-top: 3.2rem; max-width: 1240px; }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
 
      /* Sidebar */
      section[data-testid="stSidebar"] {{
          background: {INK};
      }}
      section[data-testid="stSidebar"] * {{ color: #E0E7FF; }}
      section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
      section[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
      section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg,
      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
          color: {INK} !important;
          fill: {INK} !important;
          -webkit-text-fill-color: {INK} !important;
      }}
      section[data-testid="stSidebar"] [data-testid="stSelectbox"] label * {{
          color: #E0E7FF !important;
          -webkit-text-fill-color: #E0E7FF !important;
      }}
      section[data-testid="stSidebar"] .stDownloadButton button {{
          background: transparent;
          border: 1px solid rgba(255,255,255,0.35);
      }}
      .side-model {{
          border-left: 3px solid {SIGNAL};
          padding: 0.2rem 0 0.2rem 0.75rem;
          margin-top: 0.4rem;
          font-size: 0.85rem;
          line-height: 1.45;
          opacity: 0.9;
      }}
      .param-chip {{
          display: inline-block;
          margin: 0.25rem 0.3rem 0 0;
          padding: 0.15rem 0.55rem;
          border-radius: 999px;
          background: rgba(255,255,255,0.1);
          font-size: 0.75rem;
      }}
 
      /* Hero */
      .hero {{
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 2rem;
          padding: 1.6rem 1.8rem;
          border-radius: 18px;
          background: {INK};
          color: #fff;
          margin-bottom: 1.2rem;
      }}
      .hero h1 {{
          color: #fff;
          font-size: 2.5rem;
          font-weight: 700;
          margin: 0;
          padding: 0;
          line-height: 1.05;
      }}
      .hero p {{ margin: 0.5rem 0 0 0; color: #C7D2FE; max-width: 34rem; font-size: 0.98rem; }}
      .hero-signal {{ text-align: right; min-width: 170px; }}
      .hero-signal .lbl {{ color: #C7D2FE; font-size: 0.8rem; }}
      .hero-signal .val {{ font-size: 2.1rem; font-weight: 700; line-height: 1.1; }}
 
      /* Signal bars - the app's signature element */
      .bars {{ display: inline-flex; align-items: flex-end; gap: 4px; height: 34px; }}
      .bars span {{ width: 8px; border-radius: 2px; background: rgba(148,163,184,0.35); }}
      .bars span:nth-child(1) {{ height: 20%; }}
      .bars span:nth-child(2) {{ height: 40%; }}
      .bars span:nth-child(3) {{ height: 60%; }}
      .bars span:nth-child(4) {{ height: 80%; }}
      .bars span:nth-child(5) {{ height: 100%; }}
      .bars.small {{ height: 22px; gap: 3px; }}
      .bars.small span {{ width: 6px; }}
 
      /* Dataset strip */
      .strip {{
          display: flex; flex-wrap: wrap; gap: 2.4rem;
          padding: 0.2rem 0.2rem 1rem 0.2rem;
          border-bottom: 1px solid {LINE};
          margin-bottom: 0.6rem;
      }}
      .strip div b {{ display: block; font-size: 1.45rem; color: {INK}; }}
      .strip div span {{ font-size: 0.82rem; color: {SLATE}; }}
 
      /* Metric meters */
      .meters {{
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 0.8rem;
          margin: 0.4rem 0 1.2rem 0;
      }}
      @media (max-width: 900px) {{ .meters {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
      @media (max-width: 520px) {{ .meters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
      .meter {{
          background: #fff;
          border: 1px solid {LINE};
          border-radius: 12px;
          padding: 0.8rem 0.9rem;
      }}
      .meter .name {{ font-size: 0.82rem; color: {SLATE}; }}
      .meter .row {{ display: flex; justify-content: space-between; align-items: flex-end; margin-top: 0.2rem; }}
      .meter .num {{ font-size: 1.55rem; font-weight: 700; color: {INK}; font-variant-numeric: tabular-nums; }}
      .meter .hint {{ font-size: 0.72rem; color: {SLATE}; margin-top: 0.35rem; line-height: 1.35; }}
 
      .verdict {{
          padding: 0.9rem 1.1rem;
          border-radius: 12px;
          background: #F0FDFA;
          border: 1px solid #99F6E4;
          color: {INK};
          margin: 0.4rem 0 1rem 0;
      }}
 
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {LINE}; }}
      .stTabs [data-baseweb="tab"] {{
          padding: 0.55rem 1rem;
          border-radius: 10px 10px 0 0;
          font-weight: 500;
      }}
      .stTabs [aria-selected="true"] {{ color: {SIGNAL} !important; }}
 
      .step {{ display: flex; gap: 0.9rem; margin: 0.7rem 0; }}
      .step .n {{
          flex: 0 0 30px; height: 30px; border-radius: 50%;
          background: {INK}; color: #fff; display: flex;
          align-items: center; justify-content: center; font-weight: 600; font-size: 0.85rem;
      }}
      .step .t {{ color: #334155; font-size: 0.92rem; line-height: 1.5; padding-top: 0.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ---------------------------------------------------------------- helpers
def signal_bars(value: float, small: bool = False) -> str:
    """Render a 0-1 score as five phone-signal bars."""
    if value is None or np.isnan(value):
        filled, color = 0, SLATE
    else:
        filled = int(np.clip(np.ceil(value * 5 - 1e-9), 0, 5))
        color = SIGNAL if value >= 0.8 else AMBER if value >= 0.6 else DROP
    spans = "".join(
        f'<span style="background:{color}"></span>' if i < filled else "<span></span>"
        for i in range(5)
    )
    return f'<div class="bars{" small" if small else ""}">{spans}</div>'
 
 
def style_fig(fig, height=360, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=110 if legend else 10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Sora, sans-serif", color="#334155", size=12),
        title_font=dict(size=14, color=INK),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig
 
 
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
    return {"y_pred": y_pred, "y_score": y_score, "metrics": compute_metrics(y, y_pred, y_score)}
 
 
# ---------------------------------------------------------------- charts
def confusion_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = ["Stayed", "Churned"]
    text = [[f"<b>{cm[i, j]}</b><br>{pct[i, j]:.0%} of actual" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=pct, x=[f"Predicted {l.lower()}" for l in labels], y=[f"Actually {l.lower()}" for l in labels],
            text=text, texttemplate="%{text}", textfont=dict(size=14),
            colorscale=[[0, "#F1F5F9"], [1, SIGNAL]], showscale=False, zmin=0, zmax=1,
            hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=340, legend=False).update_layout(title="Confusion matrix")
 
 
def score_distribution_fig(y_true, y_score):
    df = pd.DataFrame({"score": y_score, "Actual": np.where(np.asarray(y_true) == 1, "Churned", "Stayed")})
    fig = go.Figure()
    for label, color in [("Stayed", "#94A3B8"), ("Churned", DROP)]:
        fig.add_trace(go.Histogram(x=df.loc[df.Actual == label, "score"], name=label,
                                   marker_color=color, opacity=0.75, nbinsx=30))
    fig.update_layout(barmode="overlay", title="How the model scores each group")
    fig.update_xaxes(title="Model churn score")
    fig.update_yaxes(title="Customers")
    return style_fig(fig, height=340)
 
 
def roc_fig(curves: dict, highlight=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#CBD5E1"),
                             name="Random guess", hoverinfo="skip"))
    for name, (fpr, tpr, auc) in curves.items():
        is_main = highlight is None or name == highlight
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.3f})",
            line=dict(color=MODEL_COLORS[name], width=3.2 if is_main else 1.6),
            opacity=1 if is_main else 0.55,
        ))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return style_fig(fig, height=380).update_layout(title="ROC curve")
 
 
def pr_fig(y_true, y_score, name):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", fill="tozeroy",
                               line=dict(color=MODEL_COLORS[name], width=2.6),
                               fillcolor="rgba(13,148,136,0.08)", name=name))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return style_fig(fig, height=380, legend=False).update_layout(title="Precision vs recall trade-off")
 
 
def radar_fig(table: pd.DataFrame):
    fig = go.Figure()
    cats = METRIC_NAMES + [METRIC_NAMES[0]]
    for name, row in table.iterrows():
        vals = [max(row[m], 0) if not np.isnan(row[m]) else 0 for m in METRIC_NAMES]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats, name=name,
                                      line=dict(color=MODEL_COLORS[name], width=2), opacity=0.85))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], gridcolor=LINE, tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor=LINE), bgcolor="#FFFFFF"),
                      title="Model profiles across all six metrics")
    return style_fig(fig, height=430)
 
 
def metric_bar_fig(table: pd.DataFrame, metric: str, best: str):
    data = table[metric].sort_values()
    fig = go.Figure(go.Bar(
        x=data.values, y=data.index, orientation="h",
        marker_color=[MODEL_COLORS[n] if n == best else "#CBD5E1" for n in data.index],
        text=[f"{v:.3f}" for v in data.values], textposition="outside",
    ))
    fig.update_xaxes(range=[min(0, data.min() - 0.05), 1.1], title=metric)
    return style_fig(fig, height=300, legend=False).update_layout(title=f"{metric} by model")
 
 
# ---------------------------------------------------------------- sidebar
models = load_models()
training_report = load_training_report()
 
with st.sidebar:
    st.markdown("### 📶 ChurnScope")
    st.caption("Pick a model and the test data to score.")
 
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()), index=1)
    st.markdown(f'<div class="side-model">{MODEL_NOTES[model_name]}</div>', unsafe_allow_html=True)
    if model_name in training_report:
        chips = "".join(
            f'<span class="param-chip">{k} = {v}</span>'
            for k, v in training_report[model_name]["best_params"].items()
        )
        st.markdown(chips, unsafe_allow_html=True)
 
    st.divider()
    source = st.radio("Test data", ["Use bundled test_data.csv", "Upload my own CSV"], index=0)
    uploaded = None
    if source == "Upload my own CSV":
        uploaded = st.file_uploader("CSV with the 20 features and a churn column", type=["csv"])
    if SAMPLE_TEST_PATH.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=SAMPLE_TEST_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )
    st.caption("Models were trained once and saved. Uploaded data is only scored, never used for training.")
 
 
# ---------------------------------------------------------------- data
if source == "Upload my own CSV":
    if uploaded is None:
        st.markdown(
            '<div class="hero"><div><h1>ChurnScope</h1>'
            "<p>Upload a test CSV in the sidebar to see how each model performs on it.</p></div></div>",
            unsafe_allow_html=True,
        )
        st.info("Waiting for a CSV. It needs the 20 feature columns and the churn column - "
                "download the sample file in the sidebar to see the exact format.")
        st.stop()
    df = read_csv(uploaded)
    source_label = uploaded.name
else:
    df = read_csv(SAMPLE_TEST_PATH)
    source_label = "test_data.csv (bundled hold-out split)"
 
missing = validate_input_frame(df, require_target=True)
if missing:
    st.error(
        "This CSV is missing required columns: "
        + ", ".join(f"`{c}`" for c in missing)
        + ". Download the sample file from the sidebar to see the expected format."
    )
    st.stop()
 
try:
    y_true = normalise_target(df[TARGET_COL])
except (ValueError, TypeError):
    st.error("The churn column could not be read. Use 0/1, True/False or yes/no values.")
    st.stop()
 
X = df[RAW_FEATURES]
all_results = {name: evaluate(mdl, X, y_true) for name, mdl in models.items()}
result = all_results[model_name]
m = result["metrics"]
 
# ---------------------------------------------------------------- hero
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>ChurnScope</h1>
        <p>Which telecom subscribers are about to leave? Five trained classifiers,
        scored live on {len(df):,} customers from {source_label}.</p>
      </div>
      <div class="hero-signal">
        <div class="lbl">{model_name} signal (MCC)</div>
        <div class="val">{"n/a" if np.isnan(m["MCC"]) else f"{m['MCC']:.3f}"}</div>
        {signal_bars(m["MCC"])}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
st.markdown(
    f"""
    <div class="strip">
      <div><b>{len(df):,}</b><span>customers scored</span></div>
      <div><b>{int(y_true.sum()):,}</b><span>actually churned</span></div>
      <div><b>{y_true.mean():.1%}</b><span>churn rate</span></div>
      <div><b>{len(RAW_FEATURES)}</b><span>input features</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
tab_eval, tab_compare, tab_pred, tab_about = st.tabs(
    ["Model evaluation", "Leaderboard", "Customer predictions", "How it works"]
)
 
# ---------------------------------------------------------------- tab 1
with tab_eval:
    st.subheader(f"{model_name} on this data")
    meters = "".join(
        f"""<div class="meter" title="{METRIC_HELP[k]}">
              <div class="name">{k}</div>
              <div class="row"><div class="num">{"n/a" if np.isnan(m[k]) else f"{m[k]:.3f}"}</div>
              {signal_bars(m[k], small=True)}</div>
              <div class="hint">{METRIC_HELP[k]}</div>
            </div>"""
        for k in METRIC_NAMES
    )
    st.markdown(f'<div class="meters">{meters}</div>', unsafe_allow_html=True)
 
    if np.isnan(m["AUC"]):
        st.warning("Only one class is present in this data, so AUC cannot be calculated.")
 
    tn, fp, fn, tp = confusion_matrix(y_true, result["y_pred"], labels=[0, 1]).ravel()
    st.markdown(
        f'<div class="verdict">Out of <b>{tp + fn}</b> customers who left, the model flagged '
        f'<b>{tp}</b> and missed <b>{fn}</b>. It also raised <b>{fp}</b> false alarms on '
        f"customers who stayed.</div>",
        unsafe_allow_html=True,
    )
 
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(confusion_fig(y_true, result["y_pred"]), width="stretch")
    with c2:
        st.plotly_chart(score_distribution_fig(y_true, result["y_score"]), width="stretch")
 
    st.markdown("**Classification report**")
    report = classification_report(
        y_true, result["y_pred"], labels=[0, 1], target_names=["Stayed (0)", "Churned (1)"],
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(
        report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"})
        .background_gradient(subset=["f1-score"], cmap="BuGn", vmin=0, vmax=2.2),
        width="stretch",
    )
 
    if not np.isnan(m["AUC"]):
        c3, c4 = st.columns(2)
        with c3:
            fpr, tpr, _ = roc_curve(y_true, result["y_score"])
            st.plotly_chart(roc_fig({model_name: (fpr, tpr, m["AUC"])}), width="stretch")
        with c4:
            st.plotly_chart(pr_fig(y_true, result["y_score"], model_name), width="stretch")
 
# ---------------------------------------------------------------- tab 2
with tab_compare:
    table = pd.DataFrame(
        [{"Model": n, **r["metrics"]} for n, r in all_results.items()]
    ).set_index("Model")
    ranked = table.sort_values("MCC", ascending=False)
    best = ranked.index[0]
 
    st.subheader("All five models, same customers")
    st.markdown(
        f'<div class="verdict"><b>{best}</b> leads on this data with MCC {ranked.loc[best, "MCC"]:.3f} '
        f'and F1 {ranked.loc[best, "F1"]:.3f}. Ranking uses MCC because only '
        f"{y_true.mean():.0%} of customers churn, which makes accuracy misleading.</div>",
        unsafe_allow_html=True,
    )
 
    medals = ["🥇", "🥈", "🥉", "4th", "5th"]
    shown = ranked.copy()
    shown.index = [f"{medals[i]}  {n}" for i, n in enumerate(shown.index)]
    st.dataframe(
        shown.style.format("{:.4f}").background_gradient(cmap="BuGn", vmin=0, vmax=2.2),
        width="stretch",
    )
 
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.plotly_chart(radar_fig(table), width="stretch")
    with c2:
        metric_to_plot = st.segmented_control(
            "Compare on", METRIC_NAMES, default="F1", key="cmp_metric"
        ) or "F1"
        st.plotly_chart(metric_bar_fig(table, metric_to_plot, best), width="stretch")
 
    if len(np.unique(y_true)) > 1:
        curves = {}
        for n, r in all_results.items():
            fpr, tpr, _ = roc_curve(y_true, r["y_score"])
            curves[n] = (fpr, tpr, r["metrics"]["AUC"])
        st.plotly_chart(roc_fig(curves, highlight=best), width="stretch")
 
# ---------------------------------------------------------------- tab 3
with tab_pred:
    st.subheader(f"Customer-level predictions from {model_name}")
    pred_df = df.copy()
    pred_df.insert(0, "predicted_churn", result["y_pred"])
    pred_df.insert(1, "result", np.where(pred_df["predicted_churn"] == y_true, "Correct", "Wrong"))
    view = st.segmented_control(
        "Show", ["All customers", "Wrong predictions", "Flagged as churners"],
        default="All customers", key="pred_view",
    ) or "All customers"
    if view == "Wrong predictions":
        pred_df = pred_df[pred_df["result"] == "Wrong"]
    elif view == "Flagged as churners":
        pred_df = pred_df[pred_df["predicted_churn"] == 1]
    st.caption(f"{len(pred_df):,} rows")
 
    def color_result(v):
        return f"color: {SIGNAL}; font-weight: 600" if v == "Correct" else f"color: {DROP}; font-weight: 600"
 
    st.dataframe(pred_df.style.map(color_result, subset=["result"]), width="stretch", height=430)
    st.download_button(
        "Download these predictions",
        pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
 
# ---------------------------------------------------------------- tab 4
with tab_about:
    st.subheader("From raw customer record to churn prediction")
    steps = [
        "The <b>phone_number</b> identifier is dropped - it says nothing about behaviour.",
        "The four <b>charge</b> columns are just minutes times a fixed tariff, so they are replaced "
        "by one <b>total_charge</b>. New features: <b>total_calls</b>, <b>intl_minutes_per_call</b> "
        "and a <b>high_service_calls</b> flag for 4 or more support calls, where churn jumps from about 11% to 44%.",
        "<b>state</b> and <b>area_code</b> are one-hot encoded and every numeric column is standardised.",
        "Each classifier is tuned with 5-fold stratified cross-validation on F1, using 80% of the data.",
        "The fitted pipelines are saved with joblib. This app loads them and scores the uploaded "
        "test data without any retraining.",
    ]
    st.markdown(
        "".join(f'<div class="step"><div class="n">{i}</div><div class="t">{s}</div></div>'
                for i, s in enumerate(steps, 1)),
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Dataset:** Telecom Customer Churn - 5,000 subscribers, 20 input features, "
        "binary target `churn` (14.1% churners)."
    )
    if training_report:
        st.markdown("**Hold-out results recorded at training time**")
        ref = pd.DataFrame({n: v["test_metrics"] for n, v in training_report.items()}).T
        st.dataframe(ref.style.format("{:.4f}"), width="stretch")
 