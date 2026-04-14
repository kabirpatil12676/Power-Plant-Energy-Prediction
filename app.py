"""
Power Plant Energy Output Prediction — Streamlit Web Application
A professional predictive analytics dashboard for Combined Cycle Power Plant energy output.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Power Plant Energy Predictor",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #0d0d14;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111120;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(120deg, #1a1a3e 0%, #0f2027 50%, #1a1a3e 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
    line-height: 1.15 !important;
    margin: 0 0 0.6rem 0 !important;
}
.hero p {
    color: rgba(255,255,255,0.55) !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
    margin: 0 !important;
    max-width: 600px;
}

/* ── Prediction Card ── */
.pred-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 18px;
    padding: 2.2rem 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(99,102,241,0.2);
    position: relative;
    overflow: hidden;
    margin: 0.5rem 0 1.5rem 0;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.15) 0%, transparent 60%);
}
.pred-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.pred-value {
    font-size: 3.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -2px;
}
.pred-unit {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.4rem;
    font-weight: 400;
}

/* ── Input Summary Cards ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.input-card:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-2px);
}
.input-card .ic-label {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.input-card .ic-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
}
.input-card .ic-unit {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
    margin-left: 4px;
}

/* ── Section Header ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1.2rem 0;
}
.sec-header .sec-line {
    width: 4px;
    height: 22px;
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
    border-radius: 4px;
    flex-shrink: 0;
}
.sec-header h3 {
    color: #e2e8f0 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.2px;
}

/* ── Metric Stat Cards ── */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    transition: border-color 0.2s ease;
}
.stat-card:hover {
    border-color: rgba(99,102,241,0.35);
}
.stat-card .sc-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 0.2rem 0;
    letter-spacing: -0.5px;
}
.stat-card .sc-label {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.stat-card .sc-hint {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.25);
    margin-top: 0.2rem;
}

/* ── Sidebar inputs ── */
.sidebar-section-title {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    margin-bottom: 0.8rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 22px;
    background: transparent;
    border: none;
    color: rgba(255,255,255,0.45);
    font-size: 0.88rem;
    font-weight: 500;
    transition: color 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.75);
    background: rgba(255,255,255,0.04);
}
.stTabs [aria-selected="true"] {
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.1) !important;
    border-bottom: 2px solid #6366f1 !important;
    font-weight: 600 !important;
}

/* ── Feature Table ── */
.feat-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 12px;
    overflow: hidden;
    font-size: 0.875rem;
}
.feat-table thead tr {
    background: rgba(99,102,241,0.12);
}
.feat-table th {
    color: #a5b4fc;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.feat-table td {
    padding: 11px 16px;
    color: rgba(255,255,255,0.7);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feat-table tbody tr:hover td {
    background: rgba(255,255,255,0.02);
}
.feat-table .tag-input {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 10px;
    display: inline-block;
}
.feat-table .tag-target {
    background: rgba(139,92,246,0.15);
    color: #c4b5fd;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 10px;
    display: inline-block;
}

/* ── Info pill in sidebar ── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.8rem;
}
.info-row .ir-key { color: rgba(255,255,255,0.4); }
.info-row .ir-val { color: rgba(255,255,255,0.85); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Model Definition ──────────────────────────────────────────────────────────
class PowerPlantANN(nn.Module):
    def __init__(self, input_dim=4):
        super(PowerPlantANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.LeakyReLU(), nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


# ── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = PowerPlantANN(input_dim=4)
    m.load_state_dict(torch.load("models/best_ann_model.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("powerplant_data.csv").drop_duplicates()

@st.cache_data
def load_metadata():
    with open("models/metadata.json", "r") as f:
        return json.load(f)


# ── Plotly Helpers ────────────────────────────────────────────────────────────
C  = ["#6366f1", "#8b5cf6", "#a78bfa", "#4f46e5", "#7c3aed", "#c4b5fd"]
BG = "rgba(0,0,0,0)"

def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.7)", size=12),
        margin=dict(l=40, r=30, t=48, b=36),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )
    return fig

def sec(title):
    st.markdown(
        f'<div class="sec-header"><div class="sec-line"></div><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, scaler, at, v, ap, rh):
    x = np.array([[at, v, ap, rh]])
    xs = scaler.transform(x)
    xt = torch.tensor(xs, dtype=torch.float32)
    with torch.no_grad():
        return round(model(xt).item(), 2)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists("models/best_ann_model.pt"):
        st.error("""
        ⚠️ Model files not found. Please run `python train.py` first to 
        generate the trained model artifacts in the `models/` directory.
        See README.md for setup instructions.
        """)
        st.stop()

    model    = load_model()
    scaler   = load_scaler()
    df       = load_data()
    metadata = load_metadata()
    ranges   = metadata["feature_ranges"]

    # ────────────────────── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sidebar-section-title">Input Parameters</p>', unsafe_allow_html=True)

        at = st.slider(
            "Temperature (AT)",
            min_value=float(ranges["AT"]["min"]),
            max_value=float(ranges["AT"]["max"]),
            value=float(ranges["AT"]["mean"]),
            step=0.1,
            help="Ambient Temperature in degrees Celsius",
            format="%.1f °C",
        )
        v = st.slider(
            "Exhaust Vacuum (V)",
            min_value=float(ranges["V"]["min"]),
            max_value=float(ranges["V"]["max"]),
            value=float(ranges["V"]["mean"]),
            step=0.1,
            help="Exhaust Vacuum in cmHg",
            format="%.1f cmHg",
        )
        ap = st.slider(
            "Ambient Pressure (AP)",
            min_value=float(ranges["AP"]["min"]),
            max_value=float(ranges["AP"]["max"]),
            value=float(ranges["AP"]["mean"]),
            step=0.1,
            help="Ambient Pressure in millibar",
            format="%.1f mbar",
        )
        rh = st.slider(
            "Relative Humidity (RH)",
            min_value=float(ranges["RH"]["min"]),
            max_value=float(ranges["RH"]["max"]),
            value=float(ranges["RH"]["mean"]),
            step=0.1,
            help="Relative Humidity as a percentage",
            format="%.1f %%",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section-title">Model Info</p>', unsafe_allow_html=True)
        ann_cfg = metadata["ann_config"]
        ds_info = metadata["dataset"]
        rows = [
            ("Dataset",    f"{ds_info['total_samples']:,} samples"),
            ("Parameters", f"{ann_cfg['total_params']:,}"),
            ("Best Epoch", str(ann_cfg["best_epoch"])),
            ("Optimizer",  ann_cfg.get("optimizer", "Adam")),
            ("Device",     "CPU (Streamlit)"),
        ]
        for k, v_val in rows:
            st.markdown(
                f'<div class="info-row"><span class="ir-key">{k}</span><span class="ir-val">{v_val}</span></div>',
                unsafe_allow_html=True,
            )

    # ────────────────────── Hero ─────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">Deep Learning &nbsp;|&nbsp; Regression</div>
        <h1>Power Plant Energy Predictor</h1>
        <p>Predict the net hourly electrical energy output of a Combined Cycle Power Plant using ambient sensor variables and an ANN trained on the UCI dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # ────────────────────── Prediction ───────────────────────────────────────
    output = predict(model, scaler, at, v, ap, rh)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Predicted Energy Output</div>
            <div class="pred-value">{output:.2f}</div>
            <div class="pred-unit">Megawatts per hour (MW)</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Temperature",       f"{at:.1f}", "°C"),
        (c2, "Exhaust Vacuum",    f"{v:.1f}",  "cmHg"),
        (c3, "Ambient Pressure",  f"{ap:.1f}", "mbar"),
        (c4, "Relative Humidity", f"{rh:.1f}", "%"),
    ]
    for col, label, val, unit in cards:
        with col:
            st.markdown(f"""
            <div class="input-card">
                <div class="ic-label">{label}</div>
                <div class="ic-value">{val}<span class="ic-unit">{unit}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────── Tabs ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Exploratory Analysis", "Model Performance", "Training History", "Dataset"])

    # ═══════════════════ TAB 1: EDA ══════════════════════════════════════════
    with tab1:
        sec("Correlation Matrix")
        corr = df.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale=[[0, "#312e81"], [0.5, "#0d0d14"], [1, "#6366f1"]],
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=13, color="rgba(255,255,255,0.9)"),
            zmin=-1, zmax=1,
            colorbar=dict(title="r", tickfont=dict(color="rgba(255,255,255,0.6)")),
        ))
        fig_corr.update_layout(title="Pearson Correlation Coefficients")
        style_fig(fig_corr, height=480)
        st.plotly_chart(fig_corr, use_container_width=True)

        sec("Feature Distributions")
        fig_dist = make_subplots(rows=1, cols=4,
            subplot_titles=["Temperature (AT)", "Vacuum (V)", "Pressure (AP)", "Humidity (RH)"])
        for i, col_name in enumerate(["AT", "V", "AP", "RH"]):
            fig_dist.add_trace(
                go.Histogram(x=df[col_name], nbinsx=40, marker_color=C[i],
                             opacity=0.85, name=col_name, showlegend=False),
                row=1, col=i+1,
            )
        fig_dist.update_layout(title="Distribution of Input Features")
        style_fig(fig_dist, height=370)
        st.plotly_chart(fig_dist, use_container_width=True)

        sec("Feature vs. Energy Output")
        fc1, fc2 = st.columns(2)
        feat_cols_ui = [fc1, fc2, fc1, fc2]
        feat_meta = [
            ("AT", "Temperature (°C)"),
            ("V",  "Exhaust Vacuum (cmHg)"),
            ("AP", "Ambient Pressure (mbar)"),
            ("RH", "Relative Humidity (%)"),
        ]
        for idx, (col_key, col_label) in enumerate(feat_meta):
            with feat_cols_ui[idx]:
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=df[col_key], y=df["PE"], mode="markers",
                    marker=dict(color=C[idx], size=3.5, opacity=0.35),
                    name="Observations",
                ))
                z = np.polyfit(df[col_key], df["PE"], 1)
                xr = np.linspace(df[col_key].min(), df[col_key].max(), 120)
                fig_sc.add_trace(go.Scatter(
                    x=xr, y=np.poly1d(z)(xr), mode="lines",
                    line=dict(color="rgba(255,255,255,0.6)", width=1.8, dash="dash"),
                    name="Linear Trend",
                ))
                fig_sc.update_layout(
                    title=f"{col_label} vs. Energy Output",
                    xaxis_title=col_label, yaxis_title="PE (MW)", showlegend=False,
                )
                style_fig(fig_sc, height=360)
                st.plotly_chart(fig_sc, use_container_width=True)

        sec("Energy Output Distribution")
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Histogram(
            x=df["PE"], nbinsx=55, marker_color="#6366f1",
            opacity=0.85, name="PE Distribution",
        ))
        fig_pe.add_vline(
            x=df["PE"].mean(), line_dash="dot", line_color="rgba(255,255,255,0.5)",
            annotation_text=f"Mean: {df['PE'].mean():.1f} MW",
            annotation_font_color="rgba(255,255,255,0.6)",
        )
        fig_pe.update_layout(
            title="Distribution of Net Hourly Energy Output (PE)",
            xaxis_title="Energy Output (MW)", yaxis_title="Count",
        )
        style_fig(fig_pe, height=380)
        st.plotly_chart(fig_pe, use_container_width=True)

        sec("Box Plots — Spread and Outliers")
        fig_box = make_subplots(rows=1, cols=5,
            subplot_titles=["AT (°C)", "V (cmHg)", "AP (mbar)", "RH (%)", "PE (MW)"])
        for i, col_name in enumerate(["AT", "V", "AP", "RH", "PE"]):
            fig_box.add_trace(
                go.Box(y=df[col_name], marker_color=C[i], name=col_name,
                       showlegend=False, boxpoints="outliers",
                       line=dict(width=1.5), fillcolor=f"rgba({','.join(str(x) for x in bytes.fromhex(C[i][1:]))},0.2)"),
                row=1, col=i+1,
            )
        fig_box.update_layout(title="Box Plots for All Variables")
        style_fig(fig_box, height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    # ═══════════════════ TAB 2: Model Performance ═════════════════════════════
    with tab2:
        results = metadata["model_results"]
        ann_m   = results.get("ANN (PyTorch)", {})

        sec("ANN Model — Test Set Metrics")
        m_cols = st.columns(5)
        metric_items = [
            ("MAE",     ann_m.get("MAE", 0),   "Mean Absolute Error",     "Lower is better"),
            ("MSE",     ann_m.get("MSE", 0),   "Mean Squared Error",      "Lower is better"),
            ("RMSE",    ann_m.get("RMSE", 0),  "Root Mean Squared Error", "Lower is better"),
            ("R² Score",ann_m.get("R2", 0),    "Coefficient of Determination", "Higher is better"),
            ("MAPE",    f"{ann_m.get('MAPE',0)}%", "Mean Abs. Percentage Error", "Lower is better"),
        ]
        for col, (name, val, desc, hint) in zip(m_cols, metric_items):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="sc-label">{name}</div>
                    <div class="sc-value">{val}</div>
                    <div class="sc-hint">{hint}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        sec("R\u00b2 Score — Model Comparison")
        model_names = list(results.keys())
        r2_scores   = [results[m]["R2"] for m in model_names]
        bar_colors  = [C[0] if m == "ANN (PyTorch)" else C[1] for m in model_names]

        fig_r2 = go.Figure(data=[go.Bar(
            x=model_names, y=r2_scores,
            marker=dict(color=bar_colors, opacity=0.9,
                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
            text=[f"{v:.4f}" for v in r2_scores],
            textposition="outside",
            textfont=dict(size=12, color="rgba(255,255,255,0.7)"),
        )])
        fig_r2.update_layout(
            title="R\u00b2 Score — All Models (Test Set)",
            yaxis_title="R\u00b2 Score",
            yaxis_range=[min(r2_scores) - 0.03, 1.01],
        )
        style_fig(fig_r2, height=430)
        st.plotly_chart(fig_r2, use_container_width=True)

        sec("Error Metrics — Detailed Comparison")
        e_c1, e_c2 = st.columns(2)

        with e_c1:
            mae_scores = [results[m]["MAE"] for m in model_names]
            fig_mae = go.Figure(data=[go.Bar(
                x=model_names, y=mae_scores,
                marker=dict(color=C[:len(model_names)], opacity=0.9),
                text=[f"{v:.3f}" for v in mae_scores],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.6)"),
            )])
            fig_mae.update_layout(title="Mean Absolute Error (MAE)", yaxis_title="MAE")
            style_fig(fig_mae, height=370)
            st.plotly_chart(fig_mae, use_container_width=True)

        with e_c2:
            rmse_scores = [results[m]["RMSE"] for m in model_names]
            fig_rmse = go.Figure(data=[go.Bar(
                x=model_names, y=rmse_scores,
                marker=dict(color=C[:len(model_names)], opacity=0.9),
                text=[f"{v:.3f}" for v in rmse_scores],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.6)"),
            )])
            fig_rmse.update_layout(title="Root Mean Squared Error (RMSE)", yaxis_title="RMSE")
            style_fig(fig_rmse, height=370)
            st.plotly_chart(fig_rmse, use_container_width=True)

        sec("Complete Results Table")
        results_df = pd.DataFrame(results).T.reset_index()
        results_df.columns = ["Model", "MAE", "MSE", "RMSE", "R2", "MAPE (%)"]
        st.dataframe(
            results_df.style
                .highlight_min(subset=["MAE","MSE","RMSE","MAPE (%)"], color="rgba(99,102,241,0.25)")
                .highlight_max(subset=["R2"], color="rgba(99,102,241,0.25)")
                .format({"MAE":"{:.4f}","MSE":"{:.4f}","RMSE":"{:.4f}","R2":"{:.4f}","MAPE (%)":"{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

        feat_imp = metadata.get("feature_importance", {})
        if feat_imp:
            sec("Feature Importance — Random Forest")
            fn = list(feat_imp.keys())
            fv = list(feat_imp.values())
            fig_fi = go.Figure(data=[go.Bar(
                x=fv, y=fn, orientation="h",
                marker=dict(color=fv, colorscale=[[0,"#312e81"],[1,"#6366f1"]],
                            opacity=0.9, line=dict(color="rgba(255,255,255,0.08)", width=1)),
                text=[f"{v:.4f}" for v in fv], textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.6)"),
            )])
            fig_fi.update_layout(
                title="Feature Importance Scores (Random Forest)",
                xaxis_title="Importance Score",
                yaxis=dict(autorange="reversed"),
            )
            style_fig(fig_fi, height=320)
            st.plotly_chart(fig_fi, use_container_width=True)

        shap_imp = metadata.get("shap_importance", {})
        if shap_imp:
            sec("SHAP Feature Importance — XGBoost")
            sn = list(shap_imp.keys())
            sv = list(shap_imp.values())
            fig_shap = go.Figure(data=[go.Bar(
                x=sv, y=sn, orientation="h",
                marker=dict(color=sv, colorscale=[[0,"#312e81"],[1,"#6366f1"]],
                            opacity=0.9, line=dict(color="rgba(255,255,255,0.08)", width=1)),
                text=[f"{v:.4f}" for v in sv], textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.6)"),
            )])
            fig_shap.update_layout(
                title="Mean Absolute SHAP Values (XGBoost)",
                xaxis_title="Mean |SHAP Value|",
                yaxis=dict(autorange="reversed"),
            )
            style_fig(fig_shap, height=320)
            st.plotly_chart(fig_shap, use_container_width=True)

    # ═══════════════════ TAB 3: Training History ══════════════════════════════
    with tab3:
        history     = metadata.get("training_history", {})
        train_loss  = history.get("train_losses", [])
        val_loss    = history.get("val_losses", [])
        lr_hist     = history.get("lr_history", [])
        best_ep     = metadata["ann_config"]["best_epoch"]

        if train_loss and val_loss:
            eps = list(range(1, len(train_loss) + 1))

            sec("Loss Curves — Training and Validation")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=eps, y=train_loss, mode="lines", name="Training Loss",
                line=dict(color="#6366f1", width=2.2),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
            ))
            fig_loss.add_trace(go.Scatter(
                x=eps, y=val_loss, mode="lines", name="Validation Loss",
                line=dict(color="#a78bfa", width=2.2),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.06)",
            ))
            fig_loss.add_vline(
                x=best_ep, line_dash="dot",
                line_color="rgba(255,255,255,0.35)",
                annotation_text=f"Best Epoch: {best_ep}",
                annotation_font_color="rgba(255,255,255,0.5)",
                annotation_position="top right",
            )
            fig_loss.update_layout(
                title="Training and Validation Loss Over Epochs",
                xaxis_title="Epoch", yaxis_title="MSE Loss",
            )
            style_fig(fig_loss, height=460)
            st.plotly_chart(fig_loss, use_container_width=True)

            sec("Convergence Detail — Final Epochs")
            start = len(train_loss) // 3
            fig_zoom = go.Figure()
            fig_zoom.add_trace(go.Scatter(
                x=eps[start:], y=train_loss[start:], mode="lines", name="Training Loss",
                line=dict(color="#6366f1", width=2),
            ))
            fig_zoom.add_trace(go.Scatter(
                x=eps[start:], y=val_loss[start:], mode="lines", name="Validation Loss",
                line=dict(color="#a78bfa", width=2),
            ))
            fig_zoom.update_layout(
                title="Loss Convergence (Zoomed — Later Epochs)",
                xaxis_title="Epoch", yaxis_title="MSE Loss",
            )
            style_fig(fig_zoom, height=380)
            st.plotly_chart(fig_zoom, use_container_width=True)

            if lr_hist:
                sec("Learning Rate Schedule")
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=eps, y=lr_hist, mode="lines+markers", name="Learning Rate",
                    line=dict(color="#818cf8", width=2),
                    marker=dict(size=3, color="#6366f1"),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                ))
                fig_lr.update_layout(
                    title="Learning Rate Schedule (ReduceLROnPlateau)",
                    xaxis_title="Epoch", yaxis_title="Learning Rate",
                )
                style_fig(fig_lr, height=340)
                st.plotly_chart(fig_lr, use_container_width=True)

        sec("Model Architecture")
        arch = metadata.get("ann_config", {})
        st.code(f"""
PowerPlantANN — Architecture Summary
=====================================
Input Layer     : 4 features (AT, V, AP, RH)
Hidden Layer 1  : Linear(4 → 256) + BatchNorm1d + LeakyReLU + Dropout(0.15)
Hidden Layer 2  : Linear(256 → 128) + BatchNorm1d + LeakyReLU + Dropout(0.15)
Hidden Layer 3  : Linear(128 → 64) + BatchNorm1d + LeakyReLU + Dropout(0.15)
Hidden Layer 4  : Linear(64 → 32) + BatchNorm1d + LeakyReLU + Dropout(0.15)
Output Layer    : Linear(32 → 1) — Energy Output (MW)

────────────────────────────────────────
Total Parameters  : {arch.get("total_params", 0):,}
Optimizer         : {arch.get("optimizer", "Adam")} (lr={arch.get("learning_rate", 0.001)})
LR Scheduler      : ReduceLROnPlateau (factor=0.3, patience=15)
Early Stopping    : patience=30
Best Epoch        : {arch.get("best_epoch", "N/A")}
""", language="text")

    # ═══════════════════ TAB 4: Dataset ══════════════════════════════════════
    with tab4:
        sec("Feature Descriptions")
        st.markdown("""
        <table class="feat-table">
            <thead>
                <tr>
                    <th>Feature</th><th>Symbol</th><th>Description</th><th>Unit</th><th>Type</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Temperature</td><td>AT</td>
                    <td>Ambient temperature measured at intake</td><td>°C</td>
                    <td><span class="tag-input">Input</span></td>
                </tr>
                <tr>
                    <td>Exhaust Vacuum</td><td>V</td>
                    <td>Vacuum pressure at turbine exhaust</td><td>cmHg</td>
                    <td><span class="tag-input">Input</span></td>
                </tr>
                <tr>
                    <td>Ambient Pressure</td><td>AP</td>
                    <td>Atmospheric pressure at plant level</td><td>mbar</td>
                    <td><span class="tag-input">Input</span></td>
                </tr>
                <tr>
                    <td>Relative Humidity</td><td>RH</td>
                    <td>Moisture content of ambient air</td><td>%</td>
                    <td><span class="tag-input">Input</span></td>
                </tr>
                <tr>
                    <td>Net Energy Output</td><td>PE</td>
                    <td>Net hourly electrical energy output of plant</td><td>MW</td>
                    <td><span class="tag-target">Target</span></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        sec("Dataset Statistics")
        ds = metadata["dataset"]
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        for col, (lbl, val) in zip(
            [d_c1, d_c2, d_c3, d_c4],
            [
                ("Total Samples",  f"{ds['total_samples']:,}"),
                ("Input Features", "4  —  Numerical"),
                ("Target Variable","PE  —  Continuous"),
                ("Missing Values", "0"),
            ]
        ):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="sc-label">{lbl}</div>
                    <div class="sc-value" style="font-size:1.3rem">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        sec("Sample Records")
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)

        sec("Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

        sec("Dataset Source")
        st.markdown("""
        <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.18);
                    border-radius:12px;padding:1.4rem 1.6rem;">
            <p style="color:#a5b4fc;font-weight:600;font-size:0.85rem;
                      text-transform:uppercase;letter-spacing:1.5px;margin:0 0 0.6rem 0;">
                UCI Machine Learning Repository
            </p>
            <p style="color:rgba(255,255,255,0.65);font-size:0.88rem;line-height:1.7;margin:0 0 0.8rem 0;">
                Combined Cycle Power Plant Dataset — 9,568 hourly readings collected over 6 years
                (2006–2011) when the plant operated at full load. Each row captures the mean
                ambient conditions across one hour of operation.
            </p>
            <p style="color:rgba(255,255,255,0.4);font-size:0.78rem;font-style:italic;margin:0;">
                Reference: P. Tufekci, "Prediction of full load electrical power output of a base load
                operated combined cycle power plant using machine learning methods,"
                Int. Journal of Electrical Power &amp; Energy Systems, 2014.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0;
                border-top:1px solid rgba(255,255,255,0.05);">
        <p style="color:rgba(255,255,255,0.2);font-size:0.78rem;margin:0;letter-spacing:0.5px;">
            Power Plant Energy Predictor &nbsp;|&nbsp;
            Built by Kabir Patil &nbsp;|&nbsp;
            PyTorch &amp; Streamlit &nbsp;|&nbsp;
            UCI CCPP Dataset
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
