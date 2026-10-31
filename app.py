"""
ChurnScope - Streamlit front-end for the Telecom Customer Churn classifiers.

The app only *loads* the five pre-trained pipelines from model/ and evaluates
them on the uploaded test CSV. No model is retrained here.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve

from src.evaluation import METRIC_NAMES, compute_metrics
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
    "Decision Tree": "Depth-limited tree that learns rule-like churn patterns such as high bills plus repeated support calls.",
    "KNN": "Votes using the five most similar customers. Precise, but the loyal majority often outvotes churners.",
    "Naive Bayes": "Gaussian model that assumes independent features - an assumption this data breaks.",
    "Support Vector Machine": "RBF-kernel margin classifier. Learns a curved boundary and keeps false alarms low.",
}

METRIC_HELP = {
    "Accuracy": "Customers classified correctly",
    "AUC": "Ranks leavers above stayers",
    "Precision": "Flagged customers who really left",
    "Recall": "Leavers the model flagged",
    "F1": "Balance of precision and recall",
    "MCC": "Agreement, fair under imbalance",
}

# ------------------------------------------------------------- design tokens
VIOLET = "#A855F7"
FUCHSIA = "#EC4899"
AMBER = "#F59E0B"
INK = "#1E1537"
MUTED = "#7C7598"
GRID = "#EFEAF8"
REPORT_CMAP = LinearSegmentedColormap.from_list("churnscope", ["#FFFFFF", "#E9D5FF", "#C084FC"])

MODEL_COLORS = {
    "Decision Tree": VIOLET,
    "Support Vector Machine": AMBER,
    "Logistic Regression": "#60A5FA",
    "KNN": FUCHSIA,
    "Naive Bayes": "#94A3B8",
}

st.set_page_config(
    page_title="ChurnScope | Telecom Churn Classifiers",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

@property --spin { syntax: '<angle>'; initial-value: 0deg; inherits: false; }
@keyframes spin { to { --spin: 360deg; } }
@keyframes draw { from { stroke-dasharray: 0 999; } }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes pulse { 0%,100% { opacity: .55; } 50% { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; } }

:root {
  --violet: #A855F7; --fuchsia: #EC4899; --amber: #F59E0B; --rose: #F43F5E;
  --ink: #1E1537; --body: #4B4468; --muted: #7C7598; --line: #ECE7F6;
  --night: #150D2B; --night-edge: rgba(255,255,255,0.10); --night-muted: #B9AEDB;
  --grad: linear-gradient(100deg, #A855F7 0%, #EC4899 55%, #F59E0B 100%);
  --card-shadow: 0 1px 2px rgba(30,21,55,0.04), 0 18px 40px -24px rgba(91,33,182,0.28);
}
html, body, .stApp, .stMarkdown, button, input, select, textarea, label, p, li { font-family: 'Outfit', system-ui, sans-serif; }

/* light lavender page with soft colour blooms */
.stApp {
  background:
    radial-gradient(900px 500px at 0% 0%, rgba(168,85,247,0.10), transparent 60%),
    radial-gradient(800px 500px at 100% 30%, rgba(236,72,153,0.08), transparent 60%),
    radial-gradient(900px 600px at 40% 100%, rgba(245,158,11,0.07), transparent 60%),
    #F7F5FC;
  background-attachment: fixed;
}
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1280px; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none !important; }

/* ---------- dark sidebar ---------- */
section[data-testid="stSidebar"] {
  background:
    radial-gradient(420px 300px at 0% 0%, rgba(168,85,247,0.35), transparent 60%),
    radial-gradient(420px 320px at 100% 100%, rgba(236,72,153,0.22), transparent 60%),
    #150D2B;
  border-right: 1px solid rgba(168,85,247,0.25);
}
section[data-testid="stSidebar"] * { color: #E9E4FB; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
  color: #1E1537 !important; fill: #6D28D9 !important; -webkit-text-fill-color: #1E1537 !important; font-weight: 600;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #FFFFFF !important; font-weight: 600; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.06); border: 1.5px dashed rgba(236,72,153,0.6); border-radius: 14px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { color: #1E1537 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color: #1E1537 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
.sb-brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1.2rem; }
.sb-logo { width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center; background: var(--grad);
  box-shadow: 0 10px 26px -8px rgba(236,72,153,0.8); }
.sb-name { font-size: 1.35rem; font-weight: 800; color: #fff; line-height: 1; letter-spacing: -0.02em; }
.sb-sub { font-size: .78rem; color: #B9AEDB !important; margin-top: .2rem; }
.sb-step { display: flex; align-items: center; gap: .55rem; margin: 1.1rem 0 .35rem; font-weight: 600; color: #fff; font-size: .95rem; }
.sb-step b { width: 24px; height: 24px; border-radius: 8px; display: grid; place-items: center; font-size: .8rem;
  background: rgba(168,85,247,0.35); border: 1px solid rgba(216,180,254,0.5); }
.sb-card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px;
  padding: .75rem .85rem; font-size: .82rem; line-height: 1.5; color: #DCD3F5 !important; margin-top: .4rem; }
.sb-chip { display: inline-block; margin: .3rem .3rem 0 0; padding: .15rem .55rem; border-radius: 999px; font-size: .72rem;
  background: rgba(236,72,153,0.16); border: 1px solid rgba(244,114,182,0.4); }
.sb-key { display: grid; gap: .35rem; margin-top: .3rem; }
.sb-key div { display: flex; align-items: center; gap: .55rem; font-size: .84rem; }
.sb-key i { width: 22px; height: 6px; border-radius: 999px; display: inline-block; }
.sb-foot { font-size: .74rem; color: #9E93C4 !important; margin-top: 1.4rem; line-height: 1.5; }

/* chart reading guides and colour keys */
.read { display: flex; gap: .6rem; align-items: flex-start; background: #FFFFFF; border: 1px solid var(--line);
  border-left: 4px solid #C084FC; border-radius: 12px; padding: .65rem .85rem; margin: -.3rem 0 1rem;
  color: var(--body); font-size: .84rem; line-height: 1.5; }
.read b { color: var(--ink); }
.read .ic { font-size: 1rem; line-height: 1.3; }
.key { display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; margin: .7rem 0 0; font-size: .8rem; color: var(--body); }
.key span { display: inline-flex; align-items: center; gap: .45rem; }
.key i { display: inline-block; width: 14px; height: 14px; border-radius: 4px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- dark command deck: hero + controls ---------- */
.hero-wrap {
  position: relative; border-radius: 28px; padding: 1.5px; margin-bottom: .4rem;
  box-shadow: 0 30px 70px -34px rgba(91,33,182,0.6);
  background: conic-gradient(from var(--spin), #A855F7, #EC4899, #F59E0B, #A855F7 50%, rgba(168,85,247,0.2) 75%, #A855F7);
  animation: spin 9s linear infinite;
}
.hero {
  position: relative; overflow: hidden; border-radius: 27px; padding: 2.2rem 2.4rem;
  background:
    radial-gradient(620px 320px at 0% 0%, rgba(168,85,247,0.40), transparent 65%),
    radial-gradient(520px 320px at 100% 100%, rgba(236,72,153,0.30), transparent 60%),
    var(--night);
  display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2rem; align-items: center;
}
.hero::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(rgba(255,255,255,0.11) 1px, transparent 1px);
  background-size: 22px 22px;
  mask-image: radial-gradient(ellipse at 80% 50%, black 10%, transparent 70%);
}
.hero .copy { position: relative; z-index: 1; animation: rise .7s ease-out both; }
.live { display: inline-flex; align-items: center; gap: .5rem; font-size: .82rem; color: #F5D0FE; font-weight: 500; }
.live i { width: 8px; height: 8px; border-radius: 50%; background: #F472B6; box-shadow: 0 0 12px #F472B6; animation: pulse 1.8s infinite; }
.hero .title {
  font-size: 3.6rem; font-weight: 800; line-height: 1; margin: .6rem 0 .7rem 0; letter-spacing: -0.035em;
  background: linear-gradient(95deg, #FFFFFF 0%, #E9D5FF 40%, #F9A8D4 75%, #FCD34D 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent;
}
.hero p.lead { color: #DCD3F5; font-size: 1.05rem; line-height: 1.6; margin: 0; max-width: 36rem; }
.kpis { display: flex; flex-wrap: wrap; gap: .7rem; margin-top: 1.5rem; }
.kpi { background: rgba(255,255,255,0.07); border: 1px solid var(--night-edge); border-radius: 16px; padding: .7rem 1rem; min-width: 128px; }
.kpi b { display: block; font-size: 1.6rem; font-weight: 700; color: #fff; line-height: 1.1; font-variant-numeric: tabular-nums; }
.kpi span { font-size: .78rem; color: var(--night-muted); }
.kpi.good b { background: linear-gradient(90deg, #D8B4FE, #F9A8D4); -webkit-background-clip: text; background-clip: text; color: transparent; }
.kpi.bad b { color: #FDA4AF; }
.gauge { position: relative; z-index: 1; text-align: center; }
.gauge svg { filter: drop-shadow(0 0 22px rgba(236,72,153,0.5)); }
.gauge svg text { fill: #FFFFFF; }
.gauge svg circle.track { stroke: rgba(255,255,255,0.12); }
.gauge .cap { color: var(--night-muted); font-size: .85rem; margin-top: .3rem; }
.gauge .cap b { color: #fff; font-weight: 600; }

/* controls form the bottom half of the dark deck */
.st-key-controls {
  background: linear-gradient(180deg, #1D1338 0%, #241747 100%);
  border: 1.5px solid rgba(168,85,247,0.55); border-top: 1px solid var(--night-edge);
  border-radius: 0 0 28px 28px; padding: 1.2rem 2.4rem 1.2rem;
  box-shadow: 0 34px 70px -34px rgba(91,33,182,0.65); margin-bottom: .6rem;
}
.st-key-controls label p, .st-key-controls [data-testid="stWidgetLabel"] p { color: #F5F3FF !important; font-weight: 600; }
.st-key-controls [role="radiogroup"] p { color: #E9E4FB !important; }
.st-key-controls .ctlnote { color: var(--night-muted); }
.st-key-controls [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.06); border: 1.5px dashed rgba(236,72,153,0.6); }
.st-key-controls [data-testid="stFileUploaderDropzone"] span,
.st-key-controls [data-testid="stFileUploaderDropzone"] small { color: #E9E4FB !important; }
.ctlnote { font-size: .82rem; line-height: 1.45; margin-top: .2rem; }
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [role="group"] { border-radius: 12px !important; border: 1px solid rgba(168,85,247,0.5) !important; }
.stDownloadButton button {
  background: var(--grad) !important; color: #fff !important; border: none !important; border-radius: 12px !important;
  font-weight: 600 !important; box-shadow: 0 10px 30px -10px rgba(236,72,153,0.7);
}
.stDownloadButton button p { color: #fff !important; }
.stDownloadButton button:hover { filter: brightness(1.08); }

/* ---------- light content ---------- */
.card { background: #FFFFFF; border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--card-shadow); }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stExpander"] {
  background: #FFFFFF; border: 1px solid var(--line) !important; border-radius: 20px; padding: .5rem; box-shadow: var(--card-shadow);
}
[data-testid="stExpander"] details { border: none !important; }

/* tabs as gradient pills */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {
  gap: .35rem; background: #FFFFFF; border: 1px solid var(--line); box-shadow: var(--card-shadow);
  border-radius: 999px; padding: .35rem; width: fit-content; max-width: 100%; overflow-x: auto; margin-top: .8rem;
}
.stTabs [role="tab"] { border-radius: 999px; padding: .5rem 1.3rem !important; white-space: nowrap; }
.stTabs [role="tab"] p { font-weight: 500; font-size: .95rem; color: var(--muted); }
.stTabs [role="tab"][aria-selected="true"] { background: var(--grad); box-shadow: 0 8px 24px -8px rgba(236,72,153,0.7); }
.stTabs [role="tab"][aria-selected="true"] p { color: #fff !important; font-weight: 600; }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

.sec { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0 .9rem 0; }
.sec .tt { font-size: 1.45rem; font-weight: 700; color: var(--ink); letter-spacing: -0.01em; }
.sec .nt { color: var(--muted); font-size: .88rem; }

/* metric rings */
.rings { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .9rem; }
@media (max-width: 1050px) { .rings { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 560px) {
  .rings { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero { grid-template-columns: 1fr; padding: 1.5rem; } .hero .title { font-size: 2.6rem; }
  .st-key-controls { padding: 1rem 1.2rem; }
}
.ring { padding: 1.1rem .6rem 1rem; text-align: center; animation: rise .6s ease-out both; position: relative; overflow: hidden; }
.ring::after { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; background: var(--grad); opacity: .85; }
.ring:nth-child(2) { animation-delay: .05s; } .ring:nth-child(3) { animation-delay: .1s; }
.ring:nth-child(4) { animation-delay: .15s; } .ring:nth-child(5) { animation-delay: .2s; } .ring:nth-child(6) { animation-delay: .25s; }
.ring svg { filter: drop-shadow(0 6px 14px rgba(168,85,247,0.28)); }
svg circle.arc { animation: draw 1.2s cubic-bezier(.2,.8,.2,1) both; }
.ring svg text { fill: var(--ink); }
.ring svg circle.track { stroke: #F1ECFA; }
.ring .nm { color: var(--ink); font-weight: 600; margin-top: .4rem; }
.ring .hp { color: var(--muted); font-size: .76rem; line-height: 1.35; margin-top: .1rem; }

/* confusion matrix tiles */
.cmx { padding: 1.2rem 1.3rem; height: 100%; }
.cmx .pt { color: var(--ink); font-weight: 600; font-size: 1.1rem; }
.cmx .pn { color: var(--muted); font-size: .82rem; margin-bottom: .9rem; }
.cmx .grid { display: grid; grid-template-columns: 96px 1fr 1fr; gap: 8px; }
.cmx .ax { color: var(--muted); font-size: .78rem; display: flex; align-items: center; }
.cmx .ax.top { justify-content: center; }
.tile { border-radius: 16px; padding: 1rem .9rem; min-height: 118px; }
.tile .n { font-size: 2.3rem; font-weight: 800; line-height: 1; }
.tile .w { font-size: .85rem; font-weight: 600; margin-top: .4rem; }
.tile .p { font-size: .76rem; opacity: .75; margin-top: .1rem; }
.tile.tn { background: #F3F0FA; color: var(--ink); }
.tile.fp { background: linear-gradient(140deg, #FEF3C7, #FFFBEB); color: #92400E; }
.tile.fn { background: linear-gradient(140deg, #FFE4E6, #FFF1F2); color: #9F1239; }
.tile.tp { background: linear-gradient(140deg, #A855F7, #EC4899); color: #fff; box-shadow: 0 14px 34px -14px rgba(236,72,153,0.8); }

/* dark standings board (accent block on the light page) */
.board {
  padding: .5rem 1.6rem; border-radius: 22px; color: #fff;
  background: radial-gradient(500px 220px at 100% 0%, rgba(236,72,153,0.25), transparent 60%),
              radial-gradient(500px 220px at 0% 100%, rgba(168,85,247,0.30), transparent 60%), var(--night);
  box-shadow: 0 30px 60px -30px rgba(91,33,182,0.6);
}
.row { display: grid; grid-template-columns: 44px minmax(170px, 1.2fr) minmax(170px, 2fr) repeat(3, 70px);
       gap: .9rem; align-items: center; padding: .85rem 0; border-top: 1px solid var(--night-edge); }
.row:first-child { border-top: none; }
.row.hd { color: var(--night-muted); font-size: .76rem; padding: .8rem 0 .5rem; }
.rk { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-weight: 700; color: #fff; background: rgba(255,255,255,0.08); }
.row.first .rk { background: var(--grad); box-shadow: 0 8px 22px -6px rgba(236,72,153,0.8); }
.md { color: #fff; font-weight: 600; }
.md small { display: block; color: var(--night-muted); font-weight: 400; font-size: .75rem; margin-top: .1rem; }
.md i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .5rem; }
.track { height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
.track div { height: 100%; border-radius: 999px; }
.num { text-align: right; color: #fff; font-variant-numeric: tabular-nums; }
.row.hd .num { color: var(--night-muted); }
.row.sel { background: linear-gradient(90deg, rgba(168,85,247,0.22), transparent 70%); margin: 0 -1.6rem; padding-left: 1.6rem; padding-right: 1.6rem; }
@media (max-width: 760px) {
  .row { grid-template-columns: 38px 1fr 64px; }
  .row > :nth-child(3), .row > :nth-child(5), .row > :nth-child(6) { display: none; }
}

/* pipeline */
.flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .9rem; }
@media (max-width: 950px) { .flow { grid-template-columns: 1fr; } }
.step { padding: 1.1rem; }
.step .no { font-size: 2rem; font-weight: 800; line-height: 1;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.step .ft { color: var(--ink); font-weight: 600; font-size: 1.05rem; margin: .4rem 0 .35rem; }
.step p { color: var(--body); font-size: .85rem; line-height: 1.5; margin: 0; }
.step p b { color: var(--ink); font-weight: 600; }
.step.dark { background: var(--night); border: none; }
.step.dark .ft { color: #fff; } .step.dark p { color: #DCD3F5; } .step.dark p b { color: #fff; }

.waiting { padding: 1.3rem 1.5rem; color: var(--ink); margin-top: 1rem; }
.waiting code { background: #F3E8FF; color: #6B21A8; padding: .1rem .4rem; border-radius: 6px; font-size: .78rem; }
.foot { color: var(--muted); font-size: .8rem; text-align: center; margin-top: 3rem; }
</style>
"""


def html(markup: str) -> None:
    """Render HTML without Markdown treating indented lines as code blocks."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


html(CSS)


# ------------------------------------------------------------- visual helpers
def ring(value: float, size: int = 96, stroke: int = 9, uid: str = "r") -> str:
    """Gradient ring gauge for a 0-1 score (negative MCC draws as empty)."""
    r = (size - stroke) / 2
    c = 2 * math.pi * r
    frac = 0 if np.isnan(value) else float(np.clip(value, 0, 1))
    txt = "n/a" if np.isnan(value) else f"{value:.3f}"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-label="{txt}">
      <defs><linearGradient id="g{uid}" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#A855F7"/><stop offset="60%" stop-color="#EC4899"/>
        <stop offset="100%" stop-color="#F59E0B"/></linearGradient></defs>
      <circle class="track" cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#F1ECFA" stroke-width="{stroke}"/>
      <circle class="arc" cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="url(#g{uid})"
        stroke-width="{stroke}" stroke-linecap="round" stroke-dasharray="{c * frac:.2f} {c:.2f}"
        transform="rotate(-90 {size/2} {size/2})"/>
      <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" font-family="Outfit, sans-serif"
        font-weight="700" font-size="{size * 0.2:.1f}">{txt}</text>
    </svg>"""


def explain(text: str) -> None:
    """Plain-language guide under a chart so a first-time viewer can read it."""
    html(f'<div class="read"><span class="ic">💡</span><div>{text}</div></div>')


def section(title: str, note: str = "") -> None:
    html(f'<div class="sec"><div class="tt">{title}</div><div class="nt">{note}</div></div>')


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def style_fig(fig, height=380, legend=True, title=None):
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=18, t=56 if title else 18, b=100 if legend else 18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#4B4468", size=12),
        title=dict(text=title, x=0.02, font=dict(size=17, color=INK)) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yref="container", yanchor="bottom", y=0.02, x=0.01,
                    font=dict(size=11, color="#4B4468"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1B1233", bordercolor=VIOLET, font=dict(color="#FFFFFF", family="Outfit")),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#DDD5EE")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#DDD5EE")
    return fig


def glow_line(fig, x, y, color, name, width=3.0, hover=None, opacity=1.0):
    """Neon line: a wide faint halo trace underneath the real line."""
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color=hex_rgba(color, 0.18 * opacity), width=width * 4)))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, opacity=opacity,
                             line=dict(color=color, width=width), hovertemplate=hover))


# ------------------------------------------------------------- data + models
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


def churn_score(model, X) -> np.ndarray:
    """0-1 churn score: predicted probability, or a sigmoid of the SVM margin.

    The sigmoid is monotonic, so AUC and ROC are identical to using the raw margin.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(X)))


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    score = churn_score(model, X)
    return {"y_pred": y_pred, "score": score, "metrics": compute_metrics(y, y_pred, score)}


# ------------------------------------------------------------- charts
def gains_fig(y_true, score, name):
    """Cumulative gains: share of churners reached when contacting the riskiest customers first."""
    order = np.argsort(-score)
    hits = np.cumsum(np.asarray(y_true)[order])
    total = max(hits[-1], 1)
    x = np.concatenate([[0], np.arange(1, len(order) + 1) / len(order)])
    y = np.concatenate([[0], hits / total])
    at20 = y[int(round(0.2 * len(order)))]
    color = MODEL_COLORS[name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random contact (baseline)",
                             line=dict(color="#B7AECF", dash="dot", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tonexty", fillcolor=hex_rgba(color, 0.16),
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    glow_line(fig, x, y, color, f"{name} (contact riskiest first)",
              hover="Contact top %{x:.0%}<br>Reach %{y:.0%} of churners<extra></extra>")
    fig.add_trace(go.Scatter(x=[0.2], y=[at20], mode="markers", name="Top 20% point", hoverinfo="skip",
                             marker=dict(size=13, color=FUCHSIA, line=dict(color="#fff", width=2))))
    fig.add_annotation(x=0.2, y=at20, ax=80, ay=46, text=f"<b>Top 20%</b> → {at20:.0%} of churners",
                       showarrow=True, arrowcolor="#F9A8D4", font=dict(size=12, color="#fff"),
                       bgcolor="rgba(21,13,43,0.92)", bordercolor=FUCHSIA, borderpad=5)
    fig.update_xaxes(title="Customers contacted, riskiest first", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Churners reached", tickformat=".0%", range=[0, 1.04])
    return style_fig(fig, height=400, title="Retention campaign reach")


def roc_fig(curves: dict, selected: str, height=400):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess (AUC 0.5)",
                             line=dict(color="#B7AECF", dash="dot", width=1.5), hoverinfo="skip"))
    for name, (fpr, tpr, auc) in curves.items():
        main = name == selected
        if len(curves) == 1:
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", fill="tozeroy",
                                     fillcolor=hex_rgba(MODEL_COLORS[name], 0.14),
                                     line=dict(width=0), hoverinfo="skip", showlegend=False))
        glow_line(fig, fpr, tpr, MODEL_COLORS[name], f"{name} ({auc:.3f})",
                  width=3.2 if main else 1.8, opacity=1 if main else 0.65,
                  hover=name + "<br>FPR %{x:.2f}, TPR %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.03])
    fig = style_fig(fig, height=height, title="ROC curve")
    if len(curves) > 1:
        fig.update_layout(margin=dict(b=130))
    return fig


def butterfly_fig(y_true, score):
    """Mirrored histogram: churners above the axis, loyal customers below (share of each group)."""
    actual = np.asarray(y_true)
    bins = np.linspace(0, 1, 21)
    mids = (bins[:-1] + bins[1:]) / 2
    up, _ = np.histogram(score[actual == 1], bins=bins)
    down, _ = np.histogram(score[actual == 0], bins=bins)
    up = up / max(up.sum(), 1)
    down = down / max(down.sum(), 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids, y=up, name="Customers who churned (above the line)", width=0.042,
                         marker=dict(color=FUCHSIA, cornerradius=4),
                         hovertemplate="Score %{x:.2f}<br>%{y:.1%} of churners<extra></extra>"))
    fig.add_trace(go.Bar(x=mids, y=-down, name="Customers who stayed (below the line)", width=0.042,
                         marker=dict(color="rgba(124,117,152,0.45)", cornerradius=4),
                         customdata=down, hovertemplate="Score %{x:.2f}<br>%{customdata:.1%} of stayers<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dot", line_color="#B7AECF")
    fig.add_annotation(x=0.5, y=1.0, xref="x", yref="paper", text="threshold 0.5", showarrow=False,
                       font=dict(color=MUTED, size=11), yanchor="bottom")
    fig.update_layout(barmode="relative", bargap=0.05)
    fig.update_xaxes(title="Churn score", range=[-0.03, 1.03])
    fig.update_yaxes(title="Share of group", tickformat=".0%", zeroline=True,
                     zerolinecolor="#B7AECF")
    return style_fig(fig, height=420, title="Churn score by actual outcome")


def pr_tradeoff_fig(y_true, score, name):
    """Precision, recall and F1 as the decision threshold moves from 0 to 1."""
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    best = int(np.argmax(f1))
    fig = go.Figure()
    glow_line(fig, thresholds, precision, VIOLET, "Precision (flagged customers who really left)",
              hover="Threshold %{x:.2f}<br>Precision %{y:.3f}<extra></extra>")
    glow_line(fig, thresholds, recall, FUCHSIA, "Recall (leavers the model catches)",
              hover="Threshold %{x:.2f}<br>Recall %{y:.3f}<extra></extra>")
    fig.add_trace(go.Scatter(x=thresholds, y=f1, mode="lines", name="F1 (balance of both)",
                             line=dict(color=AMBER, width=2.4, dash="dash"),
                             hovertemplate="Threshold %{x:.2f}<br>F1 %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[thresholds[best]], y=[f1[best]], mode="markers", name="Best F1 threshold",
                             marker=dict(size=13, color=AMBER, symbol="star", line=dict(color="#fff", width=1.5)),
                             hovertemplate="Best F1 %{y:.3f} at threshold %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dot", line_color="#B7AECF")
    fig.add_annotation(x=0.5, y=1.02, yref="y", text="threshold used: 0.5", showarrow=False,
                       font=dict(color=MUTED, size=11), yanchor="bottom")
    fig.update_xaxes(title="Decision threshold on the churn score", range=[0, 1])
    fig.update_yaxes(title="Score", range=[0, 1.1])
    return style_fig(fig, height=430, title="Precision-recall trade-off")


def flower_fig(table: pd.DataFrame, selected: str):
    """Polar bar chart: one petal per metric, one layer of petals per model."""
    fig = go.Figure()
    order = table[METRIC_NAMES].clip(lower=0).fillna(0).mean(axis=1).sort_values(ascending=False).index
    for name in order:
        vals = [0 if np.isnan(table.loc[name, k]) else max(table.loc[name, k], 0) for k in METRIC_NAMES]
        main = name == selected
        fig.add_trace(go.Barpolar(
            r=vals, theta=METRIC_NAMES, name=name,
            marker=dict(color=hex_rgba(MODEL_COLORS[name], 0.9 if main else 0.6),
                        line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate=name + "<br>%{theta}: %{r:.3f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", barmode="overlay",
                   radialaxis=dict(range=[0, 1], gridcolor=GRID, linecolor=GRID,
                                   tickfont=dict(size=9, color=MUTED), angle=90),
                   angularaxis=dict(gridcolor=GRID, linecolor="#DDD5EE",
                                    tickfont=dict(size=13, color=INK), direction="clockwise")),
    )
    return style_fig(fig, height=470, title="Metric flower")


# ------------------------------------------------------------- controls
models = load_models()
training_report = load_training_report()

hero_slot = st.container()

with st.sidebar:
    html(f"""<div class="sb-brand"><div class="sb-logo">📡</div>
             <div><div class="sb-name">ChurnScope</div><div class="sb-sub">Telecom churn classifiers</div></div></div>""")
    html('<div class="sb-step"><b>1</b>Choose a classifier</div>')
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()), index=1, label_visibility="collapsed")
    params = training_report.get(model_name, {}).get("best_params", {})
    chips = "".join(f'<span class="sb-chip">{k} = {v}</span>' for k, v in params.items())
    html(f'<div class="sb-card">{MODEL_NOTES[model_name]}<div>{chips}</div></div>')

    html('<div class="sb-step"><b>2</b>Pick the test customers</div>')
    source = st.radio("Test customers", ["Bundled test_data.csv", "Upload a CSV"], index=0,
                      label_visibility="collapsed")
    uploaded = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Test CSV (20 features + churn)", type=["csv"])
    st.download_button("Download sample test_data.csv", data=SAMPLE_TEST_PATH.read_bytes(),
                       file_name="test_data.csv", mime="text/csv", width="stretch")

    st.divider()
    html('<div class="sb-step" style="margin-top:.2rem">Model colour key</div>')
    html('<div class="sb-key">' + "".join(
        f'<div><i style="background:{c};box-shadow:0 0 10px {c}"></i>{n}</div>' for n, c in MODEL_COLORS.items()
    ) + "</div>")
    html('<div class="sb-foot">The same colour is used for each model in every chart. Models were trained '
         'once and saved; uploaded data is only scored, never used for training.</div>')

# ------------------------------------------------------------- load test data
if source == "Upload a CSV" and uploaded is None:
    with hero_slot:
        html("""
        <div class="hero-wrap"><div class="hero"><div class="copy">
          <div class="live"><i></i>Waiting for test data</div>
          <div class="title">ChurnScope</div>
          <p class="lead">Upload a CSV from the sidebar and all five classifiers will score it instantly.</p>
        </div></div></div>""")
    cols = " ".join(f"<code>{c}</code>" for c in RAW_FEATURES + [TARGET_COL])
    html(f'<div class="card waiting" style="margin-top:1.2rem"><b>The CSV needs these 21 columns</b>, '
         f"the same format as test_data.csv:<br><br>{cols}</div>")
    st.stop()

if source == "Upload a CSV":
    df = read_csv(uploaded)
    source_label = uploaded.name
else:
    df = read_csv(SAMPLE_TEST_PATH)
    source_label = "test_data.csv"

missing = validate_input_frame(df, require_target=True)
if missing:
    st.error("This CSV is missing required columns: " + ", ".join(f"`{c}`" for c in missing)
             + ". Download test_data.csv to see the expected format.")
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
tn, fp, fn, tp = confusion_matrix(y_true, result["y_pred"], labels=[0, 1]).ravel()
leavers = tp + fn

# ------------------------------------------------------------- hero
with hero_slot:
    html(f"""
    <div class="hero-wrap"><div class="hero">
      <div class="copy">
        <div class="live"><i></i>Live evaluation on {len(df):,} customers from {source_label}</div>
        <div class="title">ChurnScope</div>
        <p class="lead">Which telecom subscribers are about to cancel? Five trained classifiers
        compete on the same customers. You are viewing <b style="color:#fff">{model_name}</b>.</p>
        <div class="kpis">
          <div class="kpi"><b>{len(df):,}</b><span>customers scored</span></div>
          <div class="kpi"><b>{leavers:,}</b><span>actually cancelled ({y_true.mean():.1%})</span></div>
          <div class="kpi good"><b>{tp:,}</b><span>caught by the model</span></div>
          <div class="kpi bad"><b>{fn:,}</b><span>slipped through</span></div>
        </div>
      </div>
      <div class="gauge">
        {ring(m["MCC"], size=170, stroke=14, uid="hero")}
        <div class="cap"><b>MCC</b> for {model_name}</div>
      </div>
    </div></div>
    """)

tab_over, tab_arena, tab_cust, tab_pipe = st.tabs(["Overview", "Model arena", "Customers", "Pipeline"])

# ------------------------------------------------------------- overview
with tab_over:
    section(f"{model_name}: evaluation metrics", "Hover a card for its definition")
    cards = "".join(
        f'<div class="card ring" title="{k}: {METRIC_HELP[k]}">{ring(m[k], uid=k)}'
        f'<div class="nm">{k}</div><div class="hp">{METRIC_HELP[k]}</div></div>'
        for k in METRIC_NAMES
    )
    html(f'<div class="rings">{cards}</div>')
    explain("Each ring fills up to the score: a <b>full ring is 1.0</b> (perfect), an empty ring is 0. "
            "The colour runs <b>violet → pink → amber</b> along the filled part. MCC is the fairest single "
            "number here because only a small share of customers churn.")
    if np.isnan(m["AUC"]):
        st.warning("Only one class is present in this data, so AUC cannot be calculated.")

    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        def pct(a, b):
            return f"{a / b:.0%} of this row" if b else "-"

        html(f"""
        <div class="card cmx">
          <div class="pt">Confusion matrix</div>
          <div class="pn">Rows are what really happened, columns are what the model said.</div>
          <div class="grid">
            <div></div><div class="ax top">Predicted stay</div><div class="ax top">Predicted churn</div>
            <div class="ax">Actually stayed</div>
            <div class="tile tn"><div class="n">{tn}</div><div class="w">Correctly kept</div><div class="p">{pct(tn, tn + fp)}</div></div>
            <div class="tile fp"><div class="n">{fp}</div><div class="w">False alarm</div><div class="p">{pct(fp, tn + fp)}</div></div>
            <div class="ax">Actually churned</div>
            <div class="tile fn"><div class="n">{fn}</div><div class="w">Missed churner</div><div class="p">{pct(fn, leavers)}</div></div>
            <div class="tile tp"><div class="n">{tp}</div><div class="w">Caught churner</div><div class="p">{pct(tp, leavers)}</div></div>
          </div>
          <div class="key">
            <span><i style="background:#E9E4F5"></i>Correct, stayed</span>
            <span><i style="background:#FCD34D"></i>False alarm</span>
            <span><i style="background:#FDA4AF"></i>Missed churner</span>
            <span><i style="background:linear-gradient(140deg,#A855F7,#EC4899)"></i>Correct, churned</span>
          </div>
        </div>""")
        explain("Big numbers on the <b>diagonal</b> (top-left and bottom-right) are correct predictions. "
                "The <b>rose</b> tile is the costly mistake: customers who left without being flagged.")
    with right:
        if len(np.unique(y_true)) > 1:
            st.plotly_chart(gains_fig(y_true, result["score"], model_name), width="stretch")
            explain("If the retention team calls customers from the <b>highest churn score down</b>, the coloured "
                    "line shows what share of all churners they reach. The further it sits above the "
                    "<b>dotted random line</b>, the more useful the model is.")

    left, right = st.columns(2, gap="medium")
    with left:
        if not np.isnan(m["AUC"]):
            fpr, tpr, _ = roc_curve(y_true, result["score"])
            st.plotly_chart(roc_fig({model_name: (fpr, tpr, m["AUC"])}, model_name), width="stretch")
            explain("A curve that hugs the <b>top-left corner</b> separates churners from loyal customers well. "
                    "The shaded area is the <b>AUC</b>; the dotted diagonal is a coin toss (0.5).")
    with right:
        if len(np.unique(y_true)) > 1:
            st.plotly_chart(pr_tradeoff_fig(y_true, result["score"], model_name), width="stretch")
            explain("Moving the threshold <b>right</b> makes the model stricter: <b>precision</b> (violet) usually "
                    "rises while <b>recall</b> (pink) falls. The amber star marks the threshold with the best F1; "
                    "the dotted line is the 0.5 threshold the app uses.")

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        st.plotly_chart(butterfly_fig(y_true, result["score"]), width="stretch")
        explain("<b>Pink bars above</b> the line are customers who really churned; <b>grey bars below</b> are "
                "those who stayed. A good model pushes pink to the right and grey to the left of the 0.5 line.")
    with right:
        section("Classification report", "Per-class precision, recall and F1")
        report = classification_report(
            y_true, result["y_pred"], labels=[0, 1], target_names=["Stayed (0)", "Churned (1)"],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T.rename(columns={"f1-score": "f1"})
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"})
            .background_gradient(subset=["precision", "recall", "f1"], cmap=REPORT_CMAP, vmin=0.3, vmax=1.25),
            width="stretch",
        )
        explain("Darker violet cells mean higher scores. <b>Support</b> is the number of customers in each class. "
                "The <b>Churned (1)</b> row matters most, because those are the customers the business wants to keep.")

# ------------------------------------------------------------- model arena
with tab_arena:
    table = pd.DataFrame([{"Model": n, **r["metrics"]} for n, r in all_results.items()]).set_index("Model")
    ranked = table.sort_values("MCC", ascending=False)
    section("Standings", f"Ranked by MCC, because only {y_true.mean():.0%} of customers churn")
    rows = ['<div class="row hd"><div></div><div>Model</div><div>MCC</div>'
            '<div class="num">F1</div><div class="num">AUC</div><div class="num">Recall</div></div>']
    for i, (name, r) in enumerate(ranked.iterrows(), 1):
        w = max(r["MCC"], 0) * 100
        c = MODEL_COLORS[name]
        cls = " ".join(x for x in ["first" if i == 1 else "", "sel" if name == model_name else ""] if x)
        sel_note = ", selected" if name == model_name else ""
        rows.append(f"""
        <div class="row {cls}">
          <div class="rk">{i}</div>
          <div class="md"><i style="background:{c};box-shadow:0 0 10px {c}"></i>{name}
            <small>MCC {r["MCC"]:.3f}{sel_note}</small></div>
          <div class="track"><div style="width:{w:.1f}%;background:linear-gradient(90deg,{hex_rgba(c, 0.45)},{c});
            box-shadow:0 0 14px {hex_rgba(c, 0.7)}"></div></div>
          <div class="num">{r["F1"]:.3f}</div><div class="num">{r["AUC"]:.3f}</div><div class="num">{r["Recall"]:.3f}</div>
        </div>""")
    html(f'<div class="card board">{"".join(rows)}</div>')
    st.write("")
    explain("Each bar shows the model's <b>MCC</b> in its own colour (see the colour key in the sidebar). "
            "The highlighted row is the classifier currently selected in the sidebar.")

    st.write("")
    left, right = st.columns([1, 1.1], gap="medium")
    with left:
        st.plotly_chart(flower_fig(ranked, model_name), width="stretch")
        explain("Each <b>petal</b> is one metric and its length is the score (0 at the centre, 1 at the rim). "
                "Petals of different models are layered, so a model that reaches further out on every "
                "petal is better all round.")
    with right:
        if len(np.unique(y_true)) > 1:
            curves = {}
            for n, r in all_results.items():
                fpr, tpr, _ = roc_curve(y_true, r["score"])
                curves[n] = (fpr, tpr, r["metrics"]["AUC"])
            st.plotly_chart(roc_fig(curves, model_name, height=470), width="stretch")
            explain("All five models on one ROC chart. The <b>thick line</b> is the selected model; the number "
                    "in brackets in the legend is each model's AUC.")

    with st.expander("Exact values for every model and metric"):
        st.dataframe(ranked.style.format("{:.4f}"), width="stretch")

# ------------------------------------------------------------- customers
with tab_cust:
    section(f"Customer-level predictions from {model_name}", "Riskiest customers first")
    score = result["score"]
    pred_df = df.copy()
    pred_df.insert(0, "risk", np.select([score >= 0.7, score >= 0.3], ["🔴 High", "🟠 Medium"], "🟢 Low"))
    pred_df.insert(1, "churn_score_pct", np.round(score * 100, 1))
    pred_df.insert(2, "predicted_churn", result["y_pred"])
    pred_df.insert(3, "correct", result["y_pred"] == y_true.to_numpy())
    pred_df = pred_df.sort_values("churn_score_pct", ascending=False)

    view = st.radio("Show", ["All customers", "High risk only", "Wrong predictions only"],
                    horizontal=True, label_visibility="collapsed")
    if view == "High risk only":
        pred_df = pred_df[pred_df["risk"].str.contains("High")]
    elif view == "Wrong predictions only":
        pred_df = pred_df[~pred_df["correct"]]

    st.dataframe(
        pred_df, width="stretch", height=460, hide_index=True,
        column_config={
            "risk": st.column_config.TextColumn("Risk"),
            "churn_score_pct": st.column_config.NumberColumn(
                "Churn score", format="%.1f%%",
                help="Predicted probability of churn (SVM: sigmoid of the decision margin)"),
            "predicted_churn": st.column_config.NumberColumn("Predicted"),
            "correct": st.column_config.CheckboxColumn("Correct"),
            "churn": st.column_config.NumberColumn("Actual"),
        },
    )
    st.download_button(
        f"Download {len(pred_df):,} rows as CSV",
        pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------- pipeline
with tab_pipe:
    section("From customer record to churn prediction", "The same pipeline runs inside every saved model")
    steps = [
        ("Clean", "Drop the <b>phone_number</b> identifier; it says nothing about behaviour."),
        ("Engineer", "Replace the four collinear charge columns with <b>total_charge</b>; add "
                     "<b>total_calls</b>, <b>intl_minutes_per_call</b> and a <b>4+ support calls</b> flag."),
        ("Encode", "One-hot encode <b>state</b> and <b>area_code</b>; standardise every numeric column."),
        ("Tune", "Grid search with 5-fold stratified cross-validation on F1, using 80% of the data."),
        ("Score", "Saved pipelines are loaded here and score your test data. Nothing is retrained."),
    ]
    html('<div class="flow">' + "".join(
        f'<div class="card step"><div class="no">0{i}</div><div class="ft">{t}</div><p>{d}</p></div>'
        for i, (t, d) in enumerate(steps, 1)) + "</div>")

    st.write("")
    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        html("""
        <div class="card step dark" style="padding:1.4rem">
          <div class="ft">Dataset</div>
          <p>Telecom Customer Churn: <b>5,000</b> subscribers and <b>20</b> input features covering plans,
          day / evening / night / international usage and support calls. Target <b>churn</b> marks the
          <b>14.1%</b> who cancelled. An 80 / 20 stratified split produced the 1,000-row
          <b>test_data.csv</b> used here.</p>
        </div>""")
    with right:
        calls = pd.DataFrame({"calls": df["number_customer_service_calls"].clip(upper=6), "churn": y_true})
        rate = calls.groupby("calls")["churn"].mean()
        labels = [f"{int(c)}{'+' if c == 6 else ''}" for c in rate.index]
        low = rate.index < 4
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[l for l, k in zip(labels, low) if k], y=rate.values[low], name="0-3 support calls",
                             marker=dict(color="#D8CCF3", cornerradius=8),
                             text=[f"{v:.0%}" for v in rate.values[low]], textposition="outside",
                             textfont=dict(color=INK)))
        fig.add_trace(go.Bar(x=[l for l, k in zip(labels, low) if not k], y=rate.values[~low],
                             name="4 or more support calls", marker=dict(color=FUCHSIA, cornerradius=8),
                             text=[f"{v:.0%}" for v in rate.values[~low]], textposition="outside",
                             textfont=dict(color=INK)))
        fig.update_xaxes(title="Customer service calls", type="category")
        fig.update_yaxes(title="Churn rate", tickformat=".0%", range=[0, max(rate.max() * 1.25, 0.1)])
        fig.update_xaxes(categoryorder="array", categoryarray=labels)
        st.plotly_chart(style_fig(fig, height=380, title="Churn jumps after the 4th support call"), width="stretch")
        explain("Each bar is the share of customers who churned, grouped by how many times they called "
                "customer service. From the <b>4th call</b> (pink) churn jumps from about 10% to over 40%, "
                "which is why the pipeline adds a 4+ calls flag.")

    if training_report:
        with st.expander("Hold-out results recorded at training time"):
            ref = pd.DataFrame({n: v["test_metrics"] for n, v in training_report.items()}).T
            st.dataframe(ref.style.format("{:.4f}"), width="stretch")

html('<div class="foot">ChurnScope, M.Tech AIML Machine Learning Assignment 2. '
     'Models are loaded from disk and never retrained on uploaded data.</div>')
