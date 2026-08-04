"""
HealthPulse — Longitudinal Biomarker Trend Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HealthPulse",
    page_icon="assets/stethoscope.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME & CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ──
       "Salina" is a paid Fontfabric typeface, not available on a free CDN,
       so the HealthPulse wordmark uses Playfair Display instead — a free
       Google Font with the same high-contrast serif character. Everything
       else uses Montserrat. ── */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

    /* ── Background: soft off-white, lets the vibrant cards do the work ── */
    .stApp {
        background: linear-gradient(180deg, #F6F5FF 0%, #EFECFF 100%);
    }
    .stApp::before {
        content: '';
        position: fixed; inset: 0; z-index: -1; pointer-events: none;
        background-image:
            repeating-linear-gradient(0deg, rgba(124,58,237,0.05) 0 1px, transparent 1px 46px),
            repeating-linear-gradient(90deg, rgba(124,58,237,0.05) 0 1px, transparent 1px 46px);
        opacity: .6;
    }
    .block-container { padding: 1.5rem 2rem 2rem; }

    /* ── Animated ECG pulse lines, fixed behind all content ── */
    .hp-pulse-bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none; }
    .hp-pulse-row { position: absolute; left: 0; right: 0; height: 130px; overflow: hidden; }
    .hp-pulse-row.hp-top { top: 6%; opacity: .12; transform: scale(0.85); }
    .hp-pulse-row.hp-mid { top: 46%; opacity: .15; }
    .hp-pulse-row.hp-bottom { bottom: 4%; opacity: .18; }
    .hp-pulse-track { display: flex; width: 200%; height: 100%; animation: hpPulseScroll 9s linear infinite; }
    .hp-pulse-row.hp-top .hp-pulse-track { animation-duration: 13s; animation-direction: reverse; }
    .hp-pulse-svg { width: 50%; height: 100%; flex-shrink: 0; display: block; }
    .hp-pulse-svg path { stroke: #7C3AED !important; }
    @keyframes hpPulseScroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

    /* ── Page headings — bigger, bolder, vibrant accent ── */
    h1, h2, h3 {
        font-weight: 800 !important;
        color: #3B1E88 !important;
        letter-spacing: -0.01em;
    }
    .block-container h3 {
        font-size: 1.9rem !important;
        margin-bottom: 0.25rem !important;
        background: linear-gradient(90deg, #7C3AED, #4C1D95);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Sidebar: dark navy/purple, MoneyFlow-style ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1B4B 0%, #15122E 100%);
        border-right: none;
    }
    /* ── Logo wordmark — Playfair Display, bigger ── */
    [data-testid="stSidebar"] .hp-logo-text {
        font-family: 'Playfair Display', serif;
        font-weight: 800; font-size: 1.55rem; color: #FFFFFF;
        margin-top: 4px; letter-spacing: 0.01em;
    }
    [data-testid="stSidebar"] .hp-sidebar-subtitle {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.85rem; font-weight: 600; color: #9C97D4;
        margin-top: -0.35rem; margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] .hp-sidebar-footer {
        font-family: 'Montserrat', sans-serif;
        font-size: 0.8rem; font-weight: 500; color: #9C97D4;
    }
    [data-testid="stSidebar"] hr { border-top: 1px solid rgba(255,255,255,0.12) !important; }
    [data-testid="stSidebar"] .section-label {
        color: #9C97D4 !important; font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] * { font-family: 'Montserrat', sans-serif; }

    /* ── Icon-based nav buttons ── */
    [data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: transparent;
        border: none;
        border-radius: 10px;
        text-align: left;
        justify-content: flex-start;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #C9C6F0 !important;
        padding: 0.55rem 0.8rem;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: rgba(124,58,237,0.25);
        color: #FFFFFF !important;
    }
    /* active page — rendered with type="primary" */
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: #7C3AED !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 12px rgba(124,58,237,0.55);
        border: none;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #6D28D9 !important;
    }

    /* ── Sidebar selectboxes — keep light for contrast against the dark panel ── */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: #FFFFFF; border-radius: 8px; font-size: 0.95rem;
    }

    /* ── Bordered component "cards" (st.container(border=True)) — shadowed box
         around each dashboard/trends/alerts/prediction component ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-radius: 14px !important;
        border: 0.5px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 4px 16px rgba(76,29,149,0.10);
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }

    /* ── Metric cards — bold values, vibrant top-border accents ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 14px;
        border: 0.5px solid rgba(0,0,0,0.06);
        border-top: 4px solid #7C3AED;
        box-shadow: 0 4px 16px rgba(76,29,149,0.10);
        padding: 0.9rem 1.1rem;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] { border-top-color: #7C3AED; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-top-color: #16A34A; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-top-color: #F97316; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-top-color: #2563EB; }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(5) [data-testid="stMetric"] { border-top-color: #EC4899; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #6B21A8; font-weight: 700; }
    [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 800; color: #211547; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; font-weight: 600; }

    /* ── Cards ── */
    .hp-card {
        background: #ffffff;
        border-radius: 14px;
        border: 0.5px solid rgba(0,0,0,0.06);
        box-shadow: 0 4px 16px rgba(76,29,149,0.10);
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }

    /* ── Dashboard greeting line (item 4) ── */
    .hp-greeting {
        font-size: 0.95rem; font-weight: 600; color: #6B21A8;
        margin-bottom: 0.6rem;
    }

    /* ── Alert banners — bolder, more saturated ── */
    .alert-high {
        background: #FEE2E2; border: 1px solid #F87171;
        border-left: 5px solid #DC2626;
        border-radius: 10px; padding: 0.85rem 1.1rem;
        color: #7F1D1D; font-size: 0.92rem; font-weight: 600; line-height: 1.5;
    }
    .alert-medium {
        background: #FEF3C7; border: 1px solid #FBBF24;
        border-left: 5px solid #D97706;
        border-radius: 10px; padding: 0.85rem 1.1rem;
        color: #78350F; font-size: 0.92rem; font-weight: 600; line-height: 1.5;
    }
    .alert-ok {
        background: #DCFCE7; border: 1px solid #4ADE80;
        border-left: 5px solid #16A34A;
        border-radius: 10px; padding: 0.85rem 1.1rem;
        color: #14532D; font-size: 0.92rem; font-weight: 600; line-height: 1.5;
    }

    /* ── Badges — bold pill shape ── */
    .badge-high { background:#DC2626; color:#ffffff; padding:3px 11px; border-radius:999px; font-size:0.72rem; font-weight:700; }
    .badge-med  { background:#D97706; color:#ffffff; padding:3px 11px; border-radius:999px; font-size:0.72rem; font-weight:700; }
    .badge-low  { background:#16A34A; color:#ffffff; padding:3px 11px; border-radius:999px; font-size:0.72rem; font-weight:700; }

    /* ── Section headers — brighter, still restrained ── */
    .section-label {
        font-size: 0.74rem; font-weight: 700; color: #6B21A8;
        letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.5rem;
    }

    /* ── Divider ── */
    .hp-divider { border: none; border-top: 1px solid rgba(124,58,237,0.15); margin: 0.75rem 0; }

    /* ── Hide streamlit default elements, but keep the sidebar expand/collapse
         controls visible — confirmed via live DOM inspection that both
         stExpandSidebarButton and stSidebarCollapseButton live inside the same
         <header> element, so a blanket header{visibility:hidden} traps the
         sidebar closed with no way to reopen it ── */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; height: 0; }
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        position: fixed !important; top: 8px; left: 8px; z-index: 999999 !important;
    }
    .stDeployButton { display: none; }

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ANIMATED BACKGROUND (blue gradient + ECG pulse, matches the brand theme)
# ─────────────────────────────────────────────
st.markdown("""
<div class="hp-pulse-bg">
  <div class="hp-pulse-row hp-top"><div class="hp-pulse-track">
    <svg class="hp-pulse-svg" viewBox="0 0 800 120" preserveAspectRatio="none"><path d="M0,60 L60,60 L100,68 L140,52 L180,60 L260,60 L300,20 L340,100 L380,25 L420,60 L500,60 L540,68 L580,52 L620,60 L700,60 L740,20 L780,100 L800,60" fill="none" stroke="#ffffff" stroke-width="3"/></svg>
    <svg class="hp-pulse-svg" viewBox="0 0 800 120" preserveAspectRatio="none"><path d="M0,60 L60,60 L100,68 L140,52 L180,60 L260,60 L300,20 L340,100 L380,25 L420,60 L500,60 L540,68 L580,52 L620,60 L700,60 L740,20 L780,100 L800,60" fill="none" stroke="#ffffff" stroke-width="3"/></svg>
  </div></div>
  <div class="hp-pulse-row hp-mid"><div class="hp-pulse-track">
    <svg class="hp-pulse-svg" viewBox="0 0 900 140" preserveAspectRatio="none"><path d="M0,70 L80,70 L95,55 L110,85 L125,65 L140,72 L160,70 L220,70 L240,20 L255,120 L270,25 L290,70 L360,70 L375,58 L390,82 L405,62 L420,70 L440,70 L500,70 L515,18 L530,122 L545,25 L565,70 L640,70 L655,57 L670,83 L685,63 L700,70 L900,70" fill="none" stroke="#ffffff" stroke-width="3.5"/></svg>
    <svg class="hp-pulse-svg" viewBox="0 0 900 140" preserveAspectRatio="none"><path d="M0,70 L80,70 L95,55 L110,85 L125,65 L140,72 L160,70 L220,70 L240,20 L255,120 L270,25 L290,70 L360,70 L375,58 L390,82 L405,62 L420,70 L440,70 L500,70 L515,18 L530,122 L545,25 L565,70 L640,70 L655,57 L670,83 L685,63 L700,70 L900,70" fill="none" stroke="#ffffff" stroke-width="3.5"/></svg>
  </div></div>
  <div class="hp-pulse-row hp-bottom"><div class="hp-pulse-track">
    <svg class="hp-pulse-svg" viewBox="0 0 900 140" preserveAspectRatio="none"><path d="M0,75 L70,75 L88,62 L104,90 L120,68 L136,78 L160,75 L210,75 L228,22 L242,128 L258,28 L278,75 L340,75 L358,60 L374,88 L390,66 L406,75 L430,75 L480,75 L498,20 L512,130 L528,26 L548,75 L610,75 L628,61 L644,89 L660,67 L676,75 L900,75" fill="none" stroke="#ffffff" stroke-width="4"/></svg>
    <svg class="hp-pulse-svg" viewBox="0 0 900 140" preserveAspectRatio="none"><path d="M0,75 L70,75 L88,62 L104,90 L120,68 L136,78 L160,75 L210,75 L228,22 L242,128 L258,28 L278,75 L340,75 L358,60 L374,88 L390,66 L406,75 L430,75 L480,75 L498,20 L512,130 L528,26 L548,75 L610,75 L628,61 L644,89 L660,67 L676,75 L900,75" fill="none" stroke="#ffffff" stroke-width="4"/></svg>
  </div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df  = pd.read_csv("healthpulse_dataset.csv")
    usr = pd.read_csv("user_summary.csv")
    return df, usr

@st.cache_resource
def load_models():
    clf    = joblib.load("models/risk_classifier.pkl")
    scaler = joblib.load("models/scaler.pkl")
    iso    = joblib.load("models/isolation_forest.pkl")
    return clf, scaler, iso

df, user_summary = load_data()
clf, scaler, iso = load_models()

# NOTE: uacr is intentionally excluded from the model's feature set.
# uacr is the variable used to *define* the KDIGO risk label, so it is never
# passed into the trained models (scaler/clf/iso all expect exactly 5 inputs).
# It's still shown/used as a separate rule-based clinical check below.
FEATURES   = ['creatinine', 'albumin', 'glucose', 'ph', 'specific_gravity']
RISK_COLOR = {0: '#4CAF50', 1: '#EF9F27', 2: '#E24B4A'}
RISK_LABEL = {0: 'Low',     1: 'Medium', 2: 'High'}
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def sliding_window_alert(user_df, col, window=3, z_thresh=2.0, baseline_months=2):
    vals = user_df.sort_values('month')[col].values
    if len(vals) < baseline_months + window:
        return []
    baseline_mean = np.mean(vals[:baseline_months])
    baseline_std  = np.std(vals) + 1e-6
    alerts = []
    for i in range(len(vals) - window + 1):
        window_mean = np.mean(vals[i:i+window])
        z_score     = abs(window_mean - baseline_mean) / baseline_std
        if z_score > z_thresh:
            alerts.append({
                'months':     f'{i+1}–{i+window}',
                'z_score':    round(z_score, 2),
                'direction':  'HIGH ↑' if window_mean > baseline_mean else 'LOW ↓',
                'window_mean': round(window_mean, 2),
                'baseline':    round(baseline_mean, 2)
            })
    return alerts


def generate_alerts(user_df):
    user_df  = user_df.sort_values('month')
    latest   = user_df.iloc[-1]
    alerts   = []

    # Determine adaptive albumin threshold based on overall trend
    alb_vals  = user_df['albumin'].values
    alb_slope = np.polyfit(np.arange(len(alb_vals)), alb_vals, 1)[0]
    pct_change = alb_slope / (alb_vals[0] + 1e-6) * 100
    alb_z_thresh = 1.5 if pct_change > 5 else 2.0

    # KDIGO population thresholds
    if latest['uacr'] >= 300:
        alerts.append({'severity': 'HIGH', 'icon': '🔴',
            'message': f'uACR is **{latest["uacr"]:.0f} mg/g** — macroalbuminuria (above KDIGO 300 threshold). This indicates significant kidney damage. Consult a nephrologist immediately.'})
    elif latest['uacr'] >= 30:
        alerts.append({'severity': 'MEDIUM', 'icon': '🟡',
            'message': f'uACR is **{latest["uacr"]:.0f} mg/g** — microalbuminuria range (30–300 mg/g). Early kidney stress detected. Monitor monthly and consult a doctor.'})

    # Sliding window albumin
    alb_alerts = sliding_window_alert(user_df, 'albumin', baseline_months=2, z_thresh=alb_z_thresh)
    if alb_alerts:
        worst = max(alb_alerts, key=lambda x: x['z_score'])
        spike_ph = user_df[(user_df['month'] >= 3) & (user_df['month'] <= 5)]['ph'].mean()
        if spike_ph > 6.8:
            alerts.append({'severity': 'MEDIUM', 'icon': '🟡',
                'message': f'Albumin spike in months **{worst["months"]}** combined with elevated pH ({spike_ph:.1f}) — strongly suggests a UTI episode. Monitor for symptoms.'})
        else:
            alerts.append({'severity': 'MEDIUM', 'icon': '🟡',
                'message': f'Albumin steadily rising — months **{worst["months"]}** (z={worst["z_score"]:.2f}). May indicate early kidney stress. Monitor monthly.'})

    # Glucose rising
    glu_alerts = sliding_window_alert(user_df, 'glucose', z_thresh=2.5, baseline_months=2)
    if glu_alerts:
        alerts.append({'severity': 'MEDIUM', 'icon': '🟡',
            'message': 'Urine glucose rising consistently. This may indicate developing insulin resistance. Consider an HbA1c blood test.'})

    if not alerts:
        alerts.append({'severity': 'OK', 'icon': '🟢',
            'message': 'All biomarkers are stable and within normal range. Keep up your health routine!'})

    return alerts


def get_trend_arrow(trend):
    if trend == 'rising':  return '↑', '#E24B4A'
    if trend == 'falling': return '↓', '#2196F3'
    return '→', '#4CAF50'


def render_metric_card(icon, icon_bg, label, value, sub, spark_series=None, spark_color="#7C3AED"):
    """Renders one metric as a bordered card: big value, label, sub-text,
    and an optional gradient-filled sparkline underneath."""
    with st.container(border=True):
        st.markdown(f"""
        <div style='font-size:0.78rem;font-weight:700;color:#6B21A8;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.3rem'>{label}</div>
        <div style='font-size:1.55rem;font-weight:800;color:#211547;line-height:1.2'>{value}</div>
        <div style='font-size:0.78rem;color:#6B7280;margin-top:2px'>{sub}</div>
        """, unsafe_allow_html=True)

        if spark_series is not None and len(spark_series) >= 2:
            spark = go.Figure(go.Scatter(
                y=list(spark_series), mode='lines',
                line=dict(color=spark_color, width=2),
                fill='tozeroy', fillcolor=spark_color.replace(')', ',0.15)').replace('rgb', 'rgba') if spark_color.startswith('rgb') else _hex_to_rgba(spark_color, 0.15),
                hoverinfo='skip'
            ))
            spark.update_layout(
                height=40, margin=dict(l=0, r=0, t=4, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                showlegend=False
            )
            st.plotly_chart(spark, use_container_width=True, config={'displayModeBar': False})


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def personal_z_scores(user_df, col='uacr'):
    vals     = user_df.sort_values('month')[col].values
    baseline = np.mean(vals[:3])
    bstd     = np.std(vals[:3]) + 1e-6
    return [(v - baseline) / bstd for v in vals]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    logo_col, title_col = st.columns([1, 4])
    with logo_col:
        st.image("assets/nav_logo.png", width=36)
    with title_col:
        st.markdown("<div class='hp-logo-text'>HealthPulse</div>", unsafe_allow_html=True)
    st.markdown("<div class='hp-sidebar-subtitle'>Longitudinal Biomarker Dashboard</div>", unsafe_allow_html=True)
    st.divider()

    # Icon-based nav. st.radio can't render raster images inside its option
    # labels (only emoji/markdown text), so the nav is built from buttons +
    # session_state instead — the `page` variable still ends up holding the
    # exact same strings ("🏠  Dashboard", etc.) that the rest of the app
    # already checks against, so nothing downstream needed to change.
    nav_items = [
        ("assets/nav_dashboard.png", "Dashboard",         "🏠  Dashboard"),
        ("assets/nav_trends.png",    "Biomarker Trends",  "📈  Biomarker Trends"),
        ("assets/nav_alerts.png",    "Alerts",             "🔔  Alerts"),
        ("assets/nav_predict.png",   "Live Prediction",    "🔮  Live Prediction"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = nav_items[0][2]

    for icon_path, label, page_value in nav_items:
        icon_col, btn_col = st.columns([1, 5])
        with icon_col:
            st.image(icon_path, width=22)
        with btn_col:
            is_active = st.session_state.page == page_value
            if st.button(label, key=f"nav_{label}",
                         type="primary" if is_active else "secondary",
                         width="stretch"):
                st.session_state.page = page_value
                st.rerun()

    page = st.session_state.page

    st.divider()

    # User selector
    st.markdown("<div class='section-label'>Select user</div>", unsafe_allow_html=True)

    # Group users by condition for easy demo
    condition_filter = st.selectbox(
        "Filter by condition",
        ["All", "healthy", "early_ckd", "moderate_ckd",
         "uti_episode", "diabetic_risk", "ckd_plus_diabetes"],
        label_visibility="collapsed"
    )

    if condition_filter == "All":
        user_ids = user_summary['user_id'].tolist()
    else:
        user_ids = user_summary[user_summary['condition'] == condition_filter]['user_id'].tolist()

    selected_user = st.selectbox("User ID", user_ids, label_visibility="collapsed")

    # Show user info
    usr_info = user_summary[user_summary['user_id'] == selected_user].iloc[0]
    risk_col = {'Low':'#4CAF50','Medium':'#EF9F27','High':'#E24B4A'}.get(usr_info['latest_risk'],'grey')

    st.markdown(f"""
    <div class='hp-card' style='margin-top:0.5rem'>
        <div style='font-size:0.8rem;font-weight:500;color:#2C2C2A'>{selected_user}</div>
        <div style='font-size:0.72rem;color:#5F5E5A;margin:3px 0'>
            Age {usr_info['age']} · {usr_info['gender']} · {usr_info['condition'].replace('_',' ').title()}
        </div>
        <hr class='hp-divider'>
        <div style='font-size:0.72rem;color:#5F5E5A'>Risk level</div>
        <div style='font-size:1rem;font-weight:500;color:{risk_col}'>{usr_info['latest_risk']}</div>
        <div style='font-size:0.7rem;color:#888780;margin-top:3px'>uACR {usr_info['latest_uacr']} mg/g</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='hp-sidebar-footer'>Model: KDIGO 2022 clinical thresholds</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD SELECTED USER DATA
# ─────────────────────────────────────────────
user_df   = df[df['user_id'] == selected_user].sort_values('month').reset_index(drop=True)
user_info = user_summary[user_summary['user_id'] == selected_user].iloc[0]
alerts    = generate_alerts(user_df)

# ═════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════
if page == "🏠  Dashboard":

    st.markdown(f"<div class='hp-greeting'>Hi, tracking <b>{selected_user}</b> · Last entry: Month {int(user_df['month'].max())}</div>", unsafe_allow_html=True)
    st.markdown(f"### Dashboard — {selected_user}")
    st.markdown(f"<div style='font-size:0.8rem;color:#888780;margin-top:-0.75rem;margin-bottom:1rem'>{user_info['condition'].replace('_',' ').title()} · Age {user_info['age']} · {user_info['months_of_data']} months of data</div>", unsafe_allow_html=True)

    # ── Top alert banner ──
    with st.container(border=True):
        top_alert = alerts[0]
        css_class = {'HIGH':'alert-high','MEDIUM':'alert-medium','OK':'alert-ok'}.get(top_alert['severity'],'alert-ok')
        st.markdown(f"<div class='{css_class}'>{top_alert['icon']} {top_alert['message']}</div>", unsafe_allow_html=True)

    # ── Metrics ──
    latest = user_df.iloc[-1]
    arrow, arr_col = get_trend_arrow(user_info['uacr_trend'])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("⚠️", "#EDE4FF", "Risk level", user_info['latest_risk'],
                            "KDIGO-based classification", spark_series=None)
    with col2:
        render_metric_card("🧪", "#DCFCE7", "uACR", f"{latest['uacr']:.0f} mg/g",
                            f"{arrow} {user_info['uacr_trend']}",
                            spark_series=user_df.sort_values('month')['uacr'].tolist(), spark_color="#16A34A")
    with col3:
        render_metric_card("💧", "#FFEDD5", "Albumin", f"{latest['albumin']:.0f} mg/L", "trend, last N tests",
                            spark_series=user_df.sort_values('month')['albumin'].tolist(), spark_color="#F97316")
    with col4:
        render_metric_card("⚗️", "#DBEAFE", "Creatinine", f"{latest['creatinine']:.2f} mg/dL", "trend, last N tests",
                            spark_series=user_df.sort_values('month')['creatinine'].tolist(), spark_color="#2563EB")
    with col5:
        render_metric_card("📅", "#FCE7F3", "Tests logged", user_info['months_of_data'], "months of history", spark_series=None)


    # ── Charts row ──
    with st.container(border=True):
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("<div class='section-label'>uACR — 12 month trend</div>", unsafe_allow_html=True)

            fig = go.Figure()

            # Personal baseline band
            baseline_mean = user_df['uacr'].values[:3].mean()
            baseline_std  = user_df['uacr'].values[:3].std()
            months_x      = user_df['month'].tolist()

            fig.add_hrect(y0=baseline_mean - baseline_std,
                          y1=baseline_mean + baseline_std,
                          fillcolor="#1D9E75", opacity=0.08,
                          annotation_text="Personal baseline",
                          annotation_font_size=10,
                          annotation_font_color="#0F6E56",
                          line_width=0)

            # Threshold lines
            fig.add_hline(y=30,  line_dash="dash", line_color="#EF9F27",
                          line_width=1.5, annotation_text="30 mg/g (microalbuminuria)",
                          annotation_font_size=10, annotation_font_color="#854F0B")
            fig.add_hline(y=300, line_dash="dash", line_color="#E24B4A",
                          line_width=1.5, annotation_text="300 mg/g (macroalbuminuria)",
                          annotation_font_size=10, annotation_font_color="#A32D2D")

            # Main uACR line
            fig.add_trace(go.Scatter(
                x=months_x, y=user_df['uacr'],
                mode='lines+markers',
                name='uACR',
                line=dict(color='#1D9E75', width=2.5),
                marker=dict(size=6, color='#1D9E75'),
                fill='tozeroy', fillcolor='rgba(29,158,117,0.12)',
                hovertemplate='Month %{x}<br>uACR: %{y:.1f} mg/g<extra></extra>'
            ))

            # Anomaly markers
            anomaly_df = user_df[user_df['anomaly'] == 1]
            if not anomaly_df.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_df['month'], y=anomaly_df['uacr'],
                    mode='markers', name='Anomaly',
                    marker=dict(size=10, color='#EF9F27',
                                symbol='circle', line=dict(color='white', width=2)),
                    hovertemplate='Month %{x}<br>uACR: %{y:.1f} mg/g<br>⚠️ Anomaly flagged<extra></extra>'
                ))

            fig.update_layout(
                height=240, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                           ticktext=MONTH_NAMES, gridcolor='rgba(0,0,0,0.04)',
                           title=None, showline=False),
                yaxis=dict(gridcolor='rgba(0,0,0,0.04)', title='mg/g', showline=False),
                legend=dict(orientation='h', y=-0.2, font=dict(size=11)),
                font=dict(family='Inter', color='#2C2C2A', size=11)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("<div class='section-label'>Latest biomarker values</div>", unsafe_allow_html=True)
            biomarkers = {
                'uACR':        (latest['uacr'],           'mg/g'),
                'Albumin':     (latest['albumin'],         'mg/L'),
                'Creatinine':  (latest['creatinine'],      'mg/dL'),
                'Glucose':     (latest['glucose'],         'mmol/L'),
                'pH':          (latest['ph'],              ''),
                'Sp. gravity': (latest['specific_gravity'],''),
            }
            for name, (val, unit) in biomarkers.items():
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:5px 0;border-bottom:0.5px solid rgba(0,0,0,0.05)'>
                    <div style='font-size:0.78rem;color:#5F5E5A'>{name}</div>
                    <div style='font-size:0.82rem;font-weight:500;color:#2C2C2A'>
                        {val} <span style='font-size:0.68rem;color:#888780'>{unit}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Anomaly timeline ──
    with st.container(border=True):
        st.markdown("<div class='section-label'>Sliding window anomaly timeline</div>", unsafe_allow_html=True)

        uacr_vals = user_df['uacr'].tolist()
        anom_vals = user_df['anomaly'].tolist()
        max_v     = max(uacr_vals) if max(uacr_vals) > 0 else 1

        bar_colors = []
        for i, (v, a) in enumerate(zip(uacr_vals, anom_vals)):
            if a == 1:   bar_colors.append('#F09595')
            elif v > 20: bar_colors.append('#FAEEDA')
            else:        bar_colors.append('#C8E6C9')

        fig2 = go.Figure(go.Bar(
            x=MONTH_NAMES, y=uacr_vals,
            marker_color=bar_colors,
            hovertemplate='%{x}<br>uACR: %{y:.1f} mg/g<extra></extra>',
            width=0.7
        ))
        fig2.update_layout(
            height=140, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showline=False),
            yaxis=dict(showgrid=False, showline=False, visible=False),
            showlegend=False,
            font=dict(family='Inter', color='#2C2C2A', size=11)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        <div style='display:flex;gap:16px;font-size:0.72rem;color:#5F5E5A'>
            <div>🟩 Normal</div>
            <div>🟨 Watch (uACR > 20)</div>
            <div>🟥 Anomaly flagged by Isolation Forest</div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 2 — BIOMARKER TRENDS
# ═════════════════════════════════════════════
elif page == "📈  Biomarker Trends":

    st.markdown(f"### Biomarker Trends — {selected_user}")

    # Biomarker selector
    selected_bio = st.selectbox(
        "Select biomarker",
        ["uACR", "Albumin", "Creatinine", "Glucose", "pH", "Specific gravity"],
        label_visibility="collapsed"
    )

    col_map = {
        "uACR": "uacr", "Albumin": "albumin",
        "Creatinine": "creatinine", "Glucose": "glucose",
        "pH": "ph", "Specific gravity": "specific_gravity"
    }
    unit_map = {
        "uACR": "mg/g", "Albumin": "mg/L", "Creatinine": "mg/dL",
        "Glucose": "mmol/L", "pH": "", "Specific gravity": ""
    }
    threshold_map = {
        "uACR":         [(30, "#EF9F27", "30 — microalbuminuria"), (300, "#E24B4A", "300 — macroalbuminuria")],
        "Albumin":      [(30, "#EF9F27", "30 mg/L — elevated")],
        "Creatinine":   [(40, "#2196F3", "40 mg/dL — low boundary"), (300, "#EF9F27", "300 mg/dL — high boundary")],
        "Glucose":      [(2.8, "#EF9F27", "2.8 mmol/L — elevated")],
        "pH":           [(7.0, "#EF9F27", "7.0 — alkaline (UTI risk)"), (4.5, "#2196F3", "4.5 — acidic")],
        "Specific gravity": [(1.010, "#E24B4A", "1.010 — low (CKD risk)"), (1.030, "#EF9F27", "1.030 — high")],
    }

    col     = col_map[selected_bio]
    unit    = unit_map[selected_bio]
    vals    = user_df[col].tolist()
    months  = user_df['month'].tolist()
    b_mean  = np.mean(vals[:3])
    b_std   = np.std(vals[:3]) + 1e-6
    z_scores = [(v - b_mean) / b_std for v in vals]

    # ── Main trend chart ──
    with st.container(border=True):
        fig3 = go.Figure()

        # Baseline band
        fig3.add_hrect(y0=b_mean - b_std, y1=b_mean + b_std,
                       fillcolor="#1D9E75", opacity=0.08,
                       annotation_text="Personal baseline", annotation_font_size=10,
                       annotation_font_color="#0F6E56", line_width=0)

        # Thresholds
        for thresh_val, thresh_col, thresh_label in threshold_map.get(selected_bio, []):
            fig3.add_hline(y=thresh_val, line_dash="dash",
                           line_color=thresh_col, line_width=1.5,
                           annotation_text=thresh_label,
                           annotation_font_size=10,
                           annotation_font_color=thresh_col)

        # Trend line
        slope = np.polyfit(np.arange(len(vals)), vals, 1)
        trend_y = [slope[0] * i + slope[1] for i in range(len(vals))]
        fig3.add_trace(go.Scatter(
            x=months, y=trend_y, mode='lines', name='Trend',
            line=dict(color='#1D9E75', width=1, dash='dot'), opacity=0.5
        ))

        # Main line
        fig3.add_trace(go.Scatter(
            x=months, y=vals, mode='lines+markers',
            name=selected_bio,
            line=dict(color='#1D9E75', width=2.5),
            marker=dict(size=7, color='#1D9E75'),
            fill='tozeroy', fillcolor='rgba(29,158,117,0.12)',
            hovertemplate=f'Month %{{x}}<br>{selected_bio}: %{{y:.2f}} {unit}<extra></extra>'
        ))

        # Anomalies
        anomaly_df = user_df[user_df['anomaly'] == 1]
        if not anomaly_df.empty:
            fig3.add_trace(go.Scatter(
                x=anomaly_df['month'], y=anomaly_df[col],
                mode='markers', name='Anomaly',
                marker=dict(size=11, color='#EF9F27',
                            symbol='circle', line=dict(color='white', width=2)),
                hovertemplate=f'Month %{{x}}<br>{selected_bio}: %{{y:.2f}} {unit}<br>⚠️ Anomaly<extra></extra>'
            ))

        fig3.update_layout(
            height=280, margin=dict(l=0, r=10, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                       ticktext=MONTH_NAMES, gridcolor='rgba(0,0,0,0.04)', title=None),
            yaxis=dict(gridcolor='rgba(0,0,0,0.04)', title=f'{selected_bio} ({unit})'),
            legend=dict(orientation='h', y=-0.18, font=dict(size=11)),
            font=dict(family='Inter', color='#2C2C2A', size=11)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Stats row ──
    with st.container(border=True):
        arrow, arr_col = get_trend_arrow(user_info['uacr_trend'])
        latest_val = user_df[col].iloc[-1]
        max_z      = max(abs(z) for z in z_scores)
        slope_pct  = (slope[0] / (vals[0] + 1e-6)) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Latest value", f"{latest_val:.2f} {unit}")
        with c2:
            st.metric("Personal baseline", f"{b_mean:.2f} {unit}")
        with c3:
            st.metric("Max z-score", f"{max_z:.2f}", "deviation from your baseline")
        with c4:
            st.metric("Slope", f"{slope_pct:+.1f}% / month")

    # ── Z-score chart ──
    with st.container(border=True):
        st.markdown("<div class='section-label'>Personal baseline deviation (z-score)</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem;color:#888780;margin-bottom:0.5rem'>Z-score > 2.0 = significantly above your own normal</div>", unsafe_allow_html=True)

        z_colors = ['#E24B4A' if abs(z) > 2 else '#EF9F27' if abs(z) > 1 else '#4CAF50' for z in z_scores]
        fig4 = go.Figure(go.Bar(
            x=MONTH_NAMES, y=z_scores, marker_color=z_colors, width=0.6,
            hovertemplate='Month: %{x}<br>Z-score: %{y:.2f}<extra></extra>'
        ))
        fig4.add_hline(y=2.0,  line_dash="dash", line_color="#E24B4A", line_width=1,
                       annotation_text="z=2.0 threshold", annotation_font_size=9)
        fig4.add_hline(y=-2.0, line_dash="dash", line_color="#2196F3", line_width=1)
        fig4.add_hline(y=0,    line_color="rgba(0,0,0,0.1)", line_width=1)
        fig4.update_layout(
            height=160, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(0,0,0,0.04)', title='Z-score'),
            showlegend=False, font=dict(family='Inter', color='#2C2C2A', size=11)
        )
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 3 — ALERTS
# ═════════════════════════════════════════════
elif page == "🔔  Alerts":

    st.markdown(f"### Alerts — {selected_user}")
    st.markdown(f"<div style='font-size:0.8rem;color:#888780;margin-top:-0.75rem;margin-bottom:1.25rem'>{len(alerts)} alert(s) generated · {user_info['condition'].replace('_',' ').title()}</div>", unsafe_allow_html=True)

    # ── All alerts ──
    with st.container(border=True):
        for alert in alerts:
            css = {'HIGH':'alert-high','MEDIUM':'alert-medium','OK':'alert-ok'}.get(alert['severity'],'alert-ok')
            st.markdown(f"<div class='{css}' style='margin-bottom:0.75rem'>{alert['icon']} <strong>[{alert['severity']}]</strong> {alert['message']}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='hp-divider'>", unsafe_allow_html=True)

    # ── All users alert summary ──
    with st.container(border=True):
        st.markdown("<div class='section-label'>Alert summary across all 500 users</div>", unsafe_allow_html=True)

        risk_counts = df.groupby('condition')['risk_label'].value_counts().unstack(fill_value=0)
        for col_name in ['Low','Medium','High']:
            if col_name not in risk_counts.columns:
                risk_counts[col_name] = 0

        fig5 = px.bar(
            risk_counts.reset_index(),
            x='condition', y=['Low','Medium','High'],
            color_discrete_map={'Low':'#4CAF50','Medium':'#EF9F27','High':'#E24B4A'},
            barmode='stack',
            labels={'condition':'Condition','value':'Count','variable':'Risk'},
        )
        fig5.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(gridcolor='rgba(0,0,0,0.04)', title='Number of readings'),
            legend=dict(orientation='h', y=-0.2, font=dict(size=11)),
            font=dict(family='Inter', color='#2C2C2A', size=11)
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Anomaly rates ──
    with st.container(border=True):
        st.markdown("<div class='section-label' style='margin-top:0.5rem'>Anomaly detection rate by condition (Isolation Forest)</div>", unsafe_allow_html=True)
        anom_rates = df.groupby('condition')['anomaly'].mean().sort_values(ascending=False) * 100

        fig6 = go.Figure(go.Bar(
            x=anom_rates.values.round(1),
            y=anom_rates.index,
            orientation='h',
            marker_color=['#E24B4A' if c != 'healthy' else '#4CAF50' for c in anom_rates.index],
            text=[f'{v:.1f}%' for v in anom_rates.values],
            textposition='outside',
            hovertemplate='%{y}<br>Anomaly rate: %{x:.1f}%<extra></extra>'
        ))
        fig6.update_layout(
            height=220, margin=dict(l=0, r=40, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, title='% readings flagged', range=[0, 110]),
            yaxis=dict(showgrid=False, title=None),
            font=dict(family='Inter', color='#2C2C2A', size=11)
        )
        st.plotly_chart(fig6, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTION
# ═════════════════════════════════════════════
elif page == "🔮  Live Prediction":

    st.markdown("### Live Risk Prediction")
    st.markdown("<div style='font-size:0.8rem;color:#888780;margin-top:-0.75rem;margin-bottom:1.25rem'>Enter biomarker values manually and get an instant risk prediction from the trained Random Forest model.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Input biomarker values</div>", unsafe_allow_html=True)
    with st.container(border=True):

        col1, col2, col3 = st.columns(3)
        with col1:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=10.0, max_value=500.0, value=120.0, step=5.0)
            # Capped at 220 (training data max is ~214.4 mg/L) so the demo stays
            # within the range the model actually learned from — above this, the
            # Random Forest's predictions can disagree with the KDIGO rule check
            # since it's extrapolating beyond anything it was trained on.
            albumin    = st.number_input("Albumin (mg/L)",     min_value=0.1,  max_value=220.0, value=5.0,   step=0.5,
                                          help="Capped near the training data's observed range for reliable predictions.")
        with col2:
            glucose = st.number_input("Glucose (mmol/L)", min_value=0.0, max_value=50.0, value=0.5, step=0.1)
            ph      = st.number_input("pH",               min_value=4.5, max_value=8.5,  value=6.0, step=0.1)
        with col3:
            sg   = st.number_input("Specific gravity",   min_value=1.001, max_value=1.035, value=1.020, step=0.001, format="%.3f")
            # uACR is computed from creatinine + albumin (not typed independently) so it
            # can't be entered inconsistently with the other two values, and it is used
            # only for the rule-based KDIGO check below — never fed into the ML models.
            uacr = albumin / (creatinine / 100 + 1e-6)
            st.metric("uACR (computed)", f"{uacr:.1f} mg/g")


    if st.button("🔮  Predict risk level", use_container_width=True):
        # Only the 5 features the models were actually trained on go into the
        # scaler/classifier/isolation forest — uacr is deliberately excluded
        # since it's the variable that defines the KDIGO risk label itself.
        input_vals = np.array([[creatinine, albumin, glucose, ph, sg]])
        input_s    = scaler.transform(input_vals)

        # Risk classifier
        risk_pred  = clf.predict(input_s)[0]
        risk_proba = clf.predict_proba(input_s)[0]

        # Anomaly detector
        anom_pred  = iso.predict(input_s)[0]
        anom_score = iso.score_samples(input_s)[0]
        is_anomaly = anom_pred == -1

        # KDIGO clinical check (rule-based, uses the computed uacr directly)
        if uacr >= 300:   kdigo = ("Macroalbuminuria", "#E24B4A", "High clinical risk")
        elif uacr >= 30:  kdigo = ("Microalbuminuria", "#EF9F27", "Medium clinical risk")
        else:             kdigo = ("Normal range",     "#4CAF50", "Low clinical risk")

        st.markdown("<br>", unsafe_allow_html=True)

        # Results
        r1, r2, r3 = st.columns(3)

        with r1:
            risk_name  = RISK_LABEL[risk_pred]
            risk_color = RISK_COLOR[risk_pred]
            confidence = risk_proba[risk_pred] * 100
            with st.container(border=True):
                st.markdown("<div class='section-label' style='text-align:center'>Random Forest prediction</div>", unsafe_allow_html=True)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={'suffix': '%', 'font': {'size': 26, 'color': risk_color}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 0, 'showticklabels': False},
                        'bar': {'color': risk_color, 'thickness': 0.28},
                        'bgcolor': 'rgba(0,0,0,0.04)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(76,175,80,0.15)'},
                            {'range': [40, 70], 'color': 'rgba(239,159,39,0.15)'},
                            {'range': [70, 100], 'color': 'rgba(226,75,74,0.15)'},
                        ],
                    }
                ))
                gauge.update_layout(
                    height=160, margin=dict(l=20, r=20, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#2C2C2A')
                )
                st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
                st.markdown(f"""
                <div style='text-align:center'>
                    <div style='font-size:1.3rem;font-weight:700;color:{risk_color};margin:-0.5rem 0 0.3rem'>{risk_name} risk</div>
                    <div style='font-size:0.75rem;color:#5F5E5A'>
                        Low: {risk_proba[0]:.1%} &nbsp;|&nbsp;
                        Med: {risk_proba[1]:.1%} &nbsp;|&nbsp;
                        High: {risk_proba[2]:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            anom_text  = "Anomaly detected" if is_anomaly else "Within normal range"
            anom_color = "#E24B4A" if is_anomaly else "#4CAF50"
            anom_icon  = "⚠️" if is_anomaly else "✅"
            st.markdown(f"""
            <div class='hp-card' style='text-align:center;border-top:3px solid {anom_color}'>
                <div class='section-label'>Isolation Forest</div>
                <div style='font-size:1.4rem;font-weight:600;color:{anom_color};margin:0.5rem 0'>{anom_icon} {anom_text}</div>
                <div style='font-size:0.75rem;color:#5F5E5A'>Anomaly score: {anom_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class='hp-card' style='text-align:center;border-top:3px solid {kdigo[1]}'>
                <div class='section-label'>KDIGO 2022 clinical</div>
                <div style='font-size:1.4rem;font-weight:600;color:{kdigo[1]};margin:0.5rem 0'>{kdigo[0]}</div>
                <div style='font-size:0.75rem;color:#5F5E5A'>{kdigo[2]}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability gauge ──
        with st.container(border=True):
            st.markdown("<div class='section-label'>Prediction confidence breakdown</div>", unsafe_allow_html=True)
            fig7 = go.Figure(go.Bar(
                x=['Low Risk', 'Medium Risk', 'High Risk'],
                y=[risk_proba[0]*100, risk_proba[1]*100, risk_proba[2]*100],
                marker_color=['#4CAF50','#EF9F27','#E24B4A'],
                text=[f'{p:.1f}%' for p in risk_proba*100],
                textposition='outside',
                width=0.5,
                hovertemplate='%{x}<br>Confidence: %{y:.1f}%<extra></extra>'
            ))
            fig7.update_layout(
                height=200, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, range=[0, 115], title='Confidence (%)'),
                showlegend=False,
                font=dict(family='Inter', color='#2C2C2A', size=11)
            )
            st.plotly_chart(fig7, use_container_width=True)

        # ── Clinical interpretation ──
        with st.container(border=True):
            st.markdown("<div class='section-label'>Clinical interpretation</div>", unsafe_allow_html=True)

            interp_lines = []
            if uacr >= 300:
                interp_lines.append("🔴 **uACR ≥ 300 mg/g** — macroalbuminuria. Significant kidney damage. Immediate nephrology referral recommended.")
            elif uacr >= 30:
                interp_lines.append("🟡 **uACR 30–300 mg/g** — microalbuminuria. Early kidney stress. Monthly monitoring and medical consultation advised.")
            else:
                interp_lines.append("🟢 **uACR < 30 mg/g** — normal range. No signs of albuminuria.")
            if glucose > 2.8:
                interp_lines.append("🟡 **Elevated glucose** — consider HbA1c blood test to rule out diabetes.")
            if ph > 7.0:
                interp_lines.append("🔵 **Alkaline pH** — elevated pH can indicate UTI. Monitor for symptoms.")
            if sg < 1.010:
                interp_lines.append("🟡 **Low specific gravity** — dilute urine may indicate poor kidney concentrating ability.")
            if not interp_lines:
                interp_lines.append("🟢 All values within normal clinical ranges.")

            for line in interp_lines:
                st.markdown(f"- {line}")
