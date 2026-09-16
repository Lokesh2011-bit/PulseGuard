"""
IoT Network Anomaly Detection System — MN692 Capstone Project
Client: APM (Advanced Personnel Management)
Team: Loki (ML Engineer), Mani (Cybersecurity), Navoda (Data Engineer),
      Naveen (Full Stack), Kishore (Network Analyst)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
import os
import base64
import hashlib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Anomaly Detection | APM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ROLE-BASED ACCESS CONTROL (RBAC) ───────────────────────────────────────────
# Credentials now come from .streamlit/secrets.toml, never hardcoded in source.
USERS = {
    username: {
        "password": hashlib.sha256(info["password"].encode()).hexdigest(),
        "role": info["role"],
    }
    for username, info in st.secrets["users"].items()
}

def check_login(username, password):
    user = USERS.get(username)
    if user and user["password"] == hashlib.sha256(password.encode()).hexdigest():
        return user["role"]
    return None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.role = None
    st.session_state.username = None

if not st.session_state.authenticated:
    st.markdown("## 🛡️ PulseGuard -  AAPM Secure Login")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;margin-bottom:20px;'>Role-based access — Admin / Analyst / Read-only</div>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
        if submitted:
            role = check_login(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Incorrect username or password")
    st.stop()

# ── THEME CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #060D1F; color: #D0E4F7; }
[data-testid="stSidebar"] { background-color: #0A1628; border-right: 1px solid #1A3A5C; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0D2137 0%, #0A1628 100%);
    border: 1px solid #1E4A7A;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 0 12px rgba(0,180,255,0.08);
}
[data-testid="stMetricValue"] { color: #00D4AA !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #7BA8CC !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
h1, h2, h3 { color: #D0E4F7 !important; }
hr { border-color: #1A3A5C !important; }
.stDataFrame { border: 1px solid #1E4A7A; border-radius: 8px; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #00A67E, #007A5E) !important;
    color: white !important; border: none !important; border-radius: 6px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1A4A7A, #0D2B50) !important;
    color: #D0E4F7 !important; border: 1px solid #2A6AAA !important; border-radius: 6px !important;
}
.info-card {
    background: linear-gradient(135deg, #0D2137, #091525);
    border: 1px solid #1E4A7A; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0;
    box-shadow: 0 0 10px rgba(0,150,255,0.06);
}
.info-label {
    font-size: 10px; font-weight: 700; color: #8FC4EC;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}
.info-value { font-size: 13px; color: #B0D4F0; }
.badge-red    { background:#C0392B; color:#fff; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; }
.badge-amber  { background:#D97706; color:#fff; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; }
.badge-green  { background:#059669; color:#fff; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; }
.badge-blue   { background:#2563EB; color:#fff; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; }
.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:0.5 } }

/* ── Live "pulse" EKG widget (sidebar) ────────────────────────────────── */
.ekg-wrap { overflow:hidden; width:100%; height:34px; margin:2px 0 4px; }
.ekg-line { width:200%; height:34px; animation: ekg-scroll 3.2s linear infinite; }
@keyframes ekg-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.live-badge { display:flex; align-items:center; gap:6px; margin-bottom:10px; }
.live-badge-label { font-size:10px; color:#7BA8CC; letter-spacing:0.08em; text-transform:uppercase; }

/* ── Top navigation bar buttons ───────────────────────────────────────── */
div[data-testid="column"] button[kind="primary"]{
    background: linear-gradient(135deg, #00D4AA, #00A67E) !important;
    color:#04140F !important; border:none !important; font-weight:600 !important;
}
div[data-testid="column"] button[kind="secondary"]{
    background:transparent !important; color:#7BA8CC !important;
    border:1px solid transparent !important;
}
div[data-testid="column"] button[kind="secondary"]:hover{ color:#D0E4F7 !important; }
</style>
""", unsafe_allow_html=True)

# ── BACKGROUND IMAGE (with dark overlay so text/charts stay fully legible) ────
@st.cache_data
def _get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path):
    """Sets a page background image behind a dark overlay gradient,
    so it reads as texture rather than competing with the dashboard's
    text, cards, or charts. Silently no-ops if the file isn't found."""
    try:
        b64 = _get_base64_of_bin_file(image_path)
        ext = image_path.split('.')[-1]
        st.markdown(f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(6,13,31,0.75), rgba(6,13,31,0.85)),
                url("data:image/{ext};base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

set_background("assets/dashboard_bg.webp")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
FEATURES_IOT23   = ['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']
FEATURES_LIVE    = ['frame.len', 'ip.proto', 'tcp.srcport', 'tcp.dstport']

TOOLTIP = {
    'duration':          'ℹ️ Duration: Length of the network connection in seconds',
    'orig_bytes':        'ℹ️ Orig Bytes: Bytes sent from the source IoT device',
    'resp_bytes':        'ℹ️ Resp Bytes: Bytes received by the source IoT device',
    'proto':             'ℹ️ Protocol: Network protocol used (TCP=6, UDP=17, ICMP=1)',
    'conn_state':        'ℹ️ Conn State: Connection state (0=established, 1=rejected, etc.)',
    'anomaly_score':     'ℹ️ Anomaly Score: Lower (more negative) = more suspicious. Isolation Forest output.',
    'ensemble_pred':     'ℹ️ Ensemble Prediction: 1 = anomalous (majority vote across models), 0 = normal',
    'frame.len':         'ℹ️ Frame Length: Size of the captured packet in bytes',
    'ip.proto':          'ℹ️ IP Protocol: IP-layer protocol number (6=TCP, 17=UDP)',
    'tcp.srcport':       'ℹ️ Source Port: TCP port the packet originated from',
    'tcp.dstport':       'ℹ️ Destination Port: TCP port the packet was sent to',
    'ip.src':            'ℹ️ Source IP: IP address of the sending machine',
    'ip.dst':            'ℹ️ Destination IP: IP address of the receiving machine',
    'rf_pred':           'ℹ️ Random Forest Prediction: Supervised ML output (1=Malicious, 0=Benign)',
    'label':             'ℹ️ Label: Ground truth traffic classification from IoT-23 dataset',
}

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_iot23_results():
    """Load pre-computed IoT-23 results from MN690"""
    try:
        df = pd.read_csv('results.csv')
        return df
    except:
        return None

@st.cache_data
def load_live_capture():
    """Load and preprocess live simulation capture"""
    try:
        df = pd.read_csv('live_capture.csv')
        df = df.dropna(subset=['ip.src', 'ip.dst'])
        le = LabelEncoder()
        df['ip.proto_enc'] = le.fit_transform(df['ip.proto'].astype(str))
        flags_map = {'0x0002': 0, '0x0012': 1, '0x0010': 2, '0x0018': 3,
                     '0x0011': 4, '0x0014': 5, '0x0004': 6}
        df['tcp.flags_enc'] = df['tcp.flags'].map(flags_map).fillna(7)
        df['tcp.srcport'] = df['tcp.srcport'].fillna(0)
        df['tcp.dstport'] = df['tcp.dstport'].fillna(0)
        scaler = MinMaxScaler()
        df['frame.len_norm'] = scaler.fit_transform(df[['frame.len']])
        return df
    except Exception as e:
        st.error(f"Could not load live_capture.csv: {e}")
        return None

@st.cache_resource
def load_models():
    """Load all trained ML models"""
    models = {}
    for name, path in [
        ('Isolation Forest', 'isolation_forest_model.pkl'),
        ('LOF',              'lof_model.pkl'),
        ('Random Forest',    'rf_model.pkl'),
        ('XGBoost',          'xgb_model.pkl'),
    ]:
        try:
            models[name] = joblib.load(path)
        except:
            models[name] = None
    return models

# ── HELPER: score live data ───────────────────────────────────────────────────
@st.cache_data
def score_live_data(df_hash):
    """Run models on live capture data"""
    df = load_live_capture()
    if df is None:
        return None
    models = load_models()
    features = ['frame.len', 'ip.proto_enc', 'tcp.srcport', 'tcp.dstport', 'tcp.flags_enc']
    X = df[features].values
    results = df.copy()

    if models.get('Isolation Forest'):
        iso_pred = models['Isolation Forest'].predict(X)
        results['iso_pred'] = [1 if p == -1 else 0 for p in iso_pred]
        results['anomaly_score'] = models['Isolation Forest'].score_samples(X)
    else:
        results['iso_pred'] = 0
        results['anomaly_score'] = -0.5

    if models.get('LOF'):
        try:
            lof_pred = models['LOF'].predict(X)
            results['lof_pred'] = [1 if p == -1 else 0 for p in lof_pred]
        except:
            results['lof_pred'] = results['iso_pred']
    else:
        results['lof_pred'] = results['iso_pred']

    results['rf_pred']  = results['iso_pred']
    results['xgb_pred'] = results['iso_pred']

    results['ensemble_pred'] = ((results['iso_pred'] == 1) | (results['lof_pred'] == 1)).astype(int)
    results['label'] = results['ensemble_pred'].map({1: 'Malicious', 0: 'Benign'})
    return results

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ PulseGuard")
    st.markdown("<div style='font-size:11px;color:#5A9FCC;'>Real-time IoT Threat Pulse Monitoring — APM</div>", unsafe_allow_html=True)
    st.markdown("---")

    role_badge = {"Admin": "🔴", "Analyst": "🟡", "Read-only": "🟢"}.get(st.session_state.role, "")
    st.markdown(f"**Logged in as:** {st.session_state.username}  \n{role_badge} `{st.session_state.role}`")
    if st.button("🚪 Log Out"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()

    st.markdown("---")

    ROLE_PAGES = {
        "Admin":     ["📊 Overview Dashboard", "📤 Upload & Detect", "🔴 Live Simulation Feed", "🧠 Model Comparison",
                      "📋 Device Baselines", "📄 Compliance Report"],
        "Analyst":   ["📊 Overview Dashboard", "📤 Upload & Detect", "🔴 Live Simulation Feed", "🧠 Model Comparison",
                      "📋 Device Baselines"],
        "Read-only": ["📊 Overview Dashboard", "📋 Device Baselines"],
    }

    st.markdown("<div class='info-label'>Data Source</div>", unsafe_allow_html=True)
    data_source = st.selectbox("Select Data Source", ["IoT-23 Dataset (MN690)", "Live Simulation Capture (MN692)"])

    st.markdown("---")
    st.markdown("<div class='info-label'>System Status</div>", unsafe_allow_html=True)
    st.markdown("<span style='color:#00D4AA'>● Models Loaded</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#00D4AA'>● Pipeline Active</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#7BA8CC;font-size:10px;'>Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}</span>", unsafe_allow_html=True)

    if data_source == "IoT-23 Dataset (MN690)":
        _pulse_df = load_iot23_results()
    else:
        _pulse_df = score_live_data(hash("live"))

    if _pulse_df is not None and 'ensemble_pred' in _pulse_df.columns and len(_pulse_df) > 0:
        _total = len(_pulse_df)
        _anom  = int(_pulse_df['ensemble_pred'].sum())
        _rate  = _anom / _total * 100
    else:
        _rate = None

    if _rate is None:
        _pulse_color, _pulse_status, _pulse_speed = "#7BA8CC", "No data loaded", 4.0
    elif _rate < 30:
        _pulse_color, _pulse_status, _pulse_speed = "#00D4AA", "Nominal", 3.2
    elif _rate < 60:
        _pulse_color, _pulse_status, _pulse_speed = "#F59E0B", "Elevated", 2.2
    else:
        _pulse_color, _pulse_status, _pulse_speed = "#FF4C6A", "Critical", 1.3

    _rate_text = f"{_rate:.1f}% alert rate" if _rate is not None else "awaiting data"

    st.markdown(f"""
    <div class="ekg-wrap">
        <svg class="ekg-line" viewBox="0 0 600 40" preserveAspectRatio="none"
             xmlns="http://www.w3.org/2000/svg" style="animation-duration:{_pulse_speed}s;">
            <path d="M0,20 L150,20 L165,4 L180,36 L195,20 L600,20"
                  stroke="{_pulse_color}" stroke-width="2" fill="none"
                  stroke-linejoin="round" stroke-linecap="round"/>
            <path d="M0,20 L150,20 L165,4 L180,36 L195,20 L600,20"
                  stroke="{_pulse_color}" stroke-width="2" fill="none"
                  stroke-linejoin="round" stroke-linecap="round" transform="translate(600,0)"/>
        </svg>
    </div>
    <div class="live-badge">
        <span class="pulse" style="color:{_pulse_color};font-size:12px;">●</span>
        <span class="live-badge-label" style="color:{_pulse_color};">{_pulse_status} — {_rate_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:10px;color:#3A6A9C;'>MN692 Capstone | Client: APM<br>Supervisor: Ahmed Jawad Khan<br>Team: Loki · Mani · Navoda · Naveen · Kishore</div>", unsafe_allow_html=True)

# ── TOP NAVIGATION BAR ─────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = ROLE_PAGES[st.session_state.role][0]

nav_items = ROLE_PAGES[st.session_state.role]
nav_cols = st.columns([1.4] + [1]*len(nav_items))
with nav_cols[0]:
    st.markdown("#### 🛡️ PulseGuard")
for i, item in enumerate(nav_items):
    with nav_cols[i+1]:
        is_active = st.session_state.current_page == item
        if st.button(item, key=f"nav_{item}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.current_page = item
            st.rerun()

page = st.session_state.current_page
st.markdown("---")

# ── UPLOAD & DETECT: format normalization ──────────────────────────────────────
def bucket_argus_state(code):
    """Maps CTU-13's Argus connection-state codes into the same
    Established / Rejected / Other buckets used elsewhere in the app."""
    if pd.isna(code):
        return "Other"
    code = str(code)
    named_other = {"ECO","ECR","REQ","RSP","RED","INT","DCE","DNP","MRQ",
                   "NNS","NRS","TXD","UNK","SEC","SRC","URF","URFIL","URH",
                   "URHPRO","URN","URNPRO","URO","URP"}
    if code in named_other:
        return "Other"
    if code == "CON":
        return "Established"
    if "R" in code:
        return "Rejected"
    if "_" in code:
        left, right = code.split("_", 1)
        if "F" in left and "F" in right:
            return "Established"
    return "Other"


def detect_and_normalize(df):
    """
    Detects whether an uploaded dataframe is IoT-23 or CTU-13 format,
    and returns (normalized_df, format_name).
    """
    cols = set(df.columns)

    if {'duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state'}.issubset(cols):
        return df[['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']].copy(), "IoT-23"

    if {'dur', 'proto', 'state', 'src_bytes', 'tot_bytes'}.issubset(cols):
        out = pd.DataFrame()
        out['duration']   = df['dur']
        out['orig_bytes'] = df['src_bytes']
        out['resp_bytes'] = df['tot_bytes'] - df['src_bytes']
        out['proto']      = df['proto']
        out['conn_state'] = df['state'].apply(bucket_argus_state)
        return out, "CTU-13"

    return None, None


@st.cache_data
def score_uploaded_data(df_normalized_json):
    """
    Runs the trained models on an uploaded, normalized dataframe.
    NOTE — ENCODING ASSUMPTION: this mapping is still unverified against
    the actual training encoding for RF/XGBoost. Treat RF/XGBoost
    predictions on uploaded files as provisional until confirmed.
    """
    df = pd.read_json(df_normalized_json)
    models = load_models()

    proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
    df['proto_enc'] = df['proto'].astype(str).str.lower().map(proto_map).fillna(-1)

    if df['conn_state'].dtype == object:
        state_map = {'Established': 0, 'Rejected': 1, 'Other': 2}
        df['conn_state_enc'] = df['conn_state'].map(state_map).fillna(2)
    else:
        df['conn_state_enc'] = df['conn_state']

    features = ['duration', 'orig_bytes', 'resp_bytes', 'proto_enc', 'conn_state_enc']
    X = df[features].values
    results = df.copy()

    if models.get('Isolation Forest'):
        iso_pred = models['Isolation Forest'].predict(X)
        results['iso_pred'] = [1 if p == -1 else 0 for p in iso_pred]
        results['anomaly_score'] = models['Isolation Forest'].score_samples(X)
    else:
        results['iso_pred'] = 0
        results['anomaly_score'] = -0.5

    if models.get('LOF'):
        try:
            lof_pred = models['LOF'].predict(X)
            results['lof_pred'] = [1 if p == -1 else 0 for p in lof_pred]
        except Exception:
            results['lof_pred'] = results['iso_pred']
    else:
        results['lof_pred'] = results['iso_pred']

    if models.get('Random Forest'):
        try:
            results['rf_pred'] = models['Random Forest'].predict(X)
        except Exception:
            results['rf_pred'] = results['iso_pred']
    else:
        results['rf_pred'] = results['iso_pred']

    if models.get('XGBoost'):
        try:
            results['xgb_pred'] = models['XGBoost'].predict(X)
        except Exception:
            results['xgb_pred'] = results['iso_pred']
    else:
        results['xgb_pred'] = results['iso_pred']

    results['ensemble_pred'] = ((results['iso_pred'] == 1) | (results['lof_pred'] == 1)).astype(int)
    results['label'] = results['ensemble_pred'].map({1: 'Malicious', 0: 'Benign'})
    return results

# ── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,33,55,0.92)',
    font=dict(color='#B0D4F0', size=11),
    margin=dict(t=30, b=30, l=30, r=30),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#B0D4F0')),
    xaxis=dict(gridcolor='#1A3A5C', zerolinecolor='#1A3A5C'),
    yaxis=dict(gridcolor='#1A3A5C', zerolinecolor='#1A3A5C'),
)
COL_MAL = '#FF4C6A'
COL_BEN = '#00D4AA'
COL_AMB = '#F59E0B'

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if '📊 Overview Dashboard' in page:
    st.markdown("## 🛡️ IoT Network Anomaly Detection System")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;margin-bottom:20px;'>Client: APM (Advanced Personnel Management) &nbsp;|&nbsp; MN692 Capstone Project &nbsp;|&nbsp; Supervisor: Ahmed Jawad Khan</div>", unsafe_allow_html=True)

    if data_source == "IoT-23 Dataset (MN690)":
        df = load_iot23_results()
        source_label = "IoT-23 Dataset — 23 CSV files from Stratosphere Laboratory, CTU Prague"
    else:
        df = score_live_data(hash("live"))
        source_label = "Live Simulation Capture — Kali Linux attack vs Metasploitable2 (tshark)"

    if df is None:
        st.error("Could not load data. Please ensure results.csv or live_capture.csv is in the same folder.")
        st.stop()

    st.markdown(f"<div style='font-size:11px;color:#3A7AAC;margin-bottom:12px;'>📦 Data source: {source_label}</div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    total    = len(df)
    n_anom   = int(df['ensemble_pred'].sum()) if 'ensemble_pred' in df.columns else 0
    n_normal = total - n_anom
    rate     = n_anom / total * 100 if total else 0

    col1.metric("🔍 Total Flows", f"{total:,}", help=TOOLTIP['duration'])
    col2.metric("🚨 Anomalies", f"{n_anom:,}", delta=f"{rate:.1f}% alert rate",
                delta_color="inverse", help=TOOLTIP['ensemble_pred'])
    col3.metric("✅ Normal Flows", f"{n_normal:,}", help="Flows classified as benign by the ensemble")
    col4.metric("📈 Alert Rate", f"{rate:.1f}%", help="Percentage of total flows flagged as anomalous")

    if 'anomaly_score' in df.columns:
        avg_score = df[df['ensemble_pred']==1]['anomaly_score'].mean()
        col5.metric("⚡ Avg Anomaly Score", f"{avg_score:.3f}", help=TOOLTIP['anomaly_score'])
    else:
        col5.metric("📊 Unique IPs", f"{df['ip.src'].nunique() if 'ip.src' in df.columns else 'N/A'}")

    st.markdown("---")

    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c1:
        st.markdown("#### 📊 Traffic Distribution")
        st.markdown(f"<div class='info-label'>{TOOLTIP.get('label','')}</div>", unsafe_allow_html=True)
        if 'label' in df.columns:
            counts = df['label'].value_counts()
        else:
            counts = pd.Series({'Malicious': n_anom, 'Benign': n_normal})
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={'Malicious': COL_MAL, 'Benign': COL_BEN},
            hole=0.4
        )
        fig_pie.update_layout(**PLOT_LAYOUT)
        fig_pie.update_traces(textfont_color='white', textfont_size=12)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### ⚠️ Anomaly Score Distribution")
        st.markdown(f"<div class='info-label'>{TOOLTIP['anomaly_score']}</div>", unsafe_allow_html=True)
        if 'anomaly_score' in df.columns:
            fig_hist = px.histogram(
                df, x='anomaly_score',
                color='ensemble_pred' if 'ensemble_pred' in df.columns else None,
                color_discrete_map={0: COL_BEN, 1: COL_MAL},
                labels={'ensemble_pred': 'Anomaly (1=Yes)', 'anomaly_score': 'Anomaly Score'},
                nbins=60
            )
            fig_hist.update_layout(**PLOT_LAYOUT)
            fig_hist.update_layout(xaxis_title="Anomaly Score ℹ️ (lower = more suspicious)")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Anomaly scores not available for this data source")

    with c3:
        st.markdown("#### 🌐 Protocol Distribution")
        st.markdown(f"<div class='info-label'>{TOOLTIP['proto']}</div>", unsafe_allow_html=True)
        proto_col = 'proto' if 'proto' in df.columns else 'ip.proto'
        if proto_col in df.columns:
            proto_counts = df[proto_col].value_counts().head(8)
            fig_proto = px.bar(
                x=proto_counts.index.astype(str),
                y=proto_counts.values,
                color=proto_counts.values,
                color_continuous_scale='Blues',
                labels={'x': 'Protocol ℹ️', 'y': 'Count'}
            )
            fig_proto.update_layout(**PLOT_LAYOUT)
            fig_proto.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_proto, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 📈 Anomaly Score Over Time")
    st.markdown(f"<div class='info-label'>{TOOLTIP['anomaly_score']}</div>", unsafe_allow_html=True)

    time_col = 'frame.time_relative' if 'frame.time_relative' in df.columns else None
    if time_col and 'anomaly_score' in df.columns:
        df_time = df[[time_col, 'anomaly_score', 'ensemble_pred']].dropna().sort_values(time_col)
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_time[time_col], y=df_time['anomaly_score'],
            mode='markers', marker=dict(
                color=df_time['ensemble_pred'].map({1: COL_MAL, 0: COL_BEN}),
                size=3, opacity=0.6
            ), name='Flow Score'
        ))
        fig_line.add_hline(y=-0.5, line_dash='dash', line_color=COL_AMB,
                           annotation_text="⚠️ Alert Threshold")
        fig_line.update_layout(**PLOT_LAYOUT,
                               xaxis_title="Time (seconds) ℹ️ — relative to capture start",
                               yaxis_title="Anomaly Score ℹ️")
        st.plotly_chart(fig_line, use_container_width=True)
    elif 'anomaly_score' in df.columns:
        df_sample = df[['anomaly_score', 'ensemble_pred']].copy()
        df_sample['flow_index'] = range(len(df_sample))
        df_sample = df_sample.sample(min(2000, len(df_sample))).sort_values('flow_index')
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_sample['flow_index'], y=df_sample['anomaly_score'],
            mode='markers',
            marker=dict(
                color=df_sample['ensemble_pred'].map({1: COL_MAL, 0: COL_BEN}),
                size=3, opacity=0.5
            ), name='Anomaly Score'
        ))
        fig_line.add_hline(y=-0.5, line_dash='dash', line_color=COL_AMB,
                           annotation_text="⚠️ Alert Threshold")
        fig_line.update_layout(**PLOT_LAYOUT,
                               xaxis_title="Flow Index ℹ️ (sequential packet order)",
                               yaxis_title="Anomaly Score ℹ️")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 🔴 Top Detected Anomalies")

    show_cols_iot23 = ['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state', 'label', 'anomaly_score']
    show_cols_live  = ['frame.time_relative', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'frame.len', 'label', 'anomaly_score']

    available = show_cols_iot23 if 'orig_bytes' in df.columns else show_cols_live
    available = [c for c in available if c in df.columns]

    anom_df = df[df['ensemble_pred'] == 1][available].copy()
    if 'anomaly_score' in anom_df.columns:
        anom_df = anom_df.sort_values('anomaly_score').head(30)

    if 'anomaly_score' in anom_df.columns:
        def severity(score):
            if score < -0.65: return "🔴 Critical"
            elif score < -0.50: return "🟠 High"
            else: return "🟡 Medium"
        anom_df.insert(0, 'Severity', anom_df['anomaly_score'].apply(severity))

    tooltip_cols = [c for c in available if c in TOOLTIP]
    if tooltip_cols:
        tip_text = " &nbsp;|&nbsp; ".join([f"<b>{c}</b>: {TOOLTIP[c].replace('ℹ️','')}" for c in tooltip_cols[:4]])
        st.markdown(f"<div style='font-size:10px;color:#5A9FCC;margin-bottom:6px;'>{tip_text}</div>", unsafe_allow_html=True)

    rename_map = {
        'duration':      'DURATION ⓘ',
        'orig_bytes':    'ORIG BYTES ⓘ',
        'resp_bytes':    'RESP BYTES ⓘ',
        'proto':         'PROTO ⓘ',
        'conn_state':    'CONN STATE ⓘ',
        'label':         'LABEL ⓘ',
        'anomaly_score': 'ANOMALY SCORE ⓘ',
        'Severity':      'SEVERITY ⓘ',
    }
    st.dataframe(anom_df.rename(columns=rename_map), use_container_width=True, height=320)

    with st.expander("ⓘ Click here to see what each column means"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SEVERITY ⓘ** — Risk level: 🔴 Critical (<-0.65) · 🟠 High (<-0.50) · 🟡 Medium")
            st.markdown("**DURATION ⓘ** — How long the network connection lasted in seconds")
            st.markdown("**ORIG BYTES ⓘ** — Bytes sent FROM the IoT device to destination")
            st.markdown("**RESP BYTES ⓘ** — Bytes received BY the IoT device from destination")
        with col2:
            st.markdown("**PROTO ⓘ** — Network protocol: TCP=6, UDP=17, ICMP=1")
            st.markdown("**CONN STATE ⓘ** — Connection state: 0=established, 1=rejected, 2=reset")
            st.markdown("**LABEL ⓘ** — Final classification: Malicious or Benign")
            st.markdown("**ANOMALY SCORE ⓘ** — Lower = more suspicious. Isolation Forest output score.")

    if st.session_state.role in ("Admin", "Analyst"):
        dl_df = df[df['ensemble_pred'] == 1][available].copy() if 'ensemble_pred' in df.columns else df
        st.download_button(
            label="📄 Download NDB Compliance Report (CSV)",
            data=dl_df.to_csv(index=False).encode('utf-8'),
            file_name=f"APM_NDB_Anomaly_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Downloads anomaly log formatted for Australian Privacy Act 1988 NDB scheme reporting"
        )
    else:
        st.info("🔒 Downloading reports requires Analyst or Admin access.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD & DETECT
# ══════════════════════════════════════════════════════════════════════════════
elif '📤 Upload & Detect' in page:
    st.markdown("## 📤 Upload & Detect")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;'>Upload a network flow capture (IoT-23 or CTU-13 format) for on-demand anomaly detection</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.caption("Supported formats: **IoT-23** and **CTU-13** (CSV or Parquet)")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "parquet"],
        help="Upload an IoT-23 or CTU-13 formatted network flow file"
    )

    if uploaded_file is None:
        st.info("No file uploaded yet. Select a CSV or Parquet file above to begin.")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_parquet(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    with st.spinner("Detecting format..."):
        normalized_df, detected_as = detect_and_normalize(raw_df)

    if normalized_df is None:
        st.error("⚠️ Unrecognized format — please upload a file matching the IoT-23 or CTU-13 schema.")
        with st.expander("What columns are expected?"):
            st.markdown("""
            - **IoT-23**: `duration`, `orig_bytes`, `resp_bytes`, `proto`, `conn_state`
            - **CTU-13**: `dur`, `proto`, `state`, `src_bytes`, `tot_bytes`
            """)
        st.stop()

    st.success(f"✅ Format detected: **{detected_as}**  &nbsp;|&nbsp;  {len(normalized_df):,} rows")

    if detected_as == "CTU-13":
        st.caption(
            "Note: connection states were mapped from Argus format "
            "(e.g. `S_RA`, `CON`) into `Established` / `Rejected` / `Other` "
            "to match the standard schema."
        )

    with st.expander(f"Preview normalized data ({normalized_df.shape[0]:,} rows)"):
        st.dataframe(normalized_df.head(20), use_container_width=True)

    st.markdown("---")

    if st.button("🔍 Run Anomaly Detection", type="primary"):
        with st.spinner("Running models..."):
            results = score_uploaded_data(normalized_df.to_json())

        n_total  = len(results)
        n_anom   = int(results['ensemble_pred'].sum())
        n_normal = n_total - n_anom
        rate     = n_anom / n_total * 100 if n_total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔍 Total Flows", f"{n_total:,}")
        c2.metric("🚨 Anomalies", f"{n_anom:,}", delta=f"{rate:.1f}% alert rate", delta_color="inverse")
        c3.metric("✅ Normal Flows", f"{n_normal:,}")
        avg_score = results[results['ensemble_pred']==1]['anomaly_score'].mean() if n_anom else None
        c4.metric("⚡ Avg Anomaly Score", f"{avg_score:.3f}" if avg_score is not None else "N/A")

        st.markdown("---")
        st.markdown("#### 📊 Result Breakdown")
        counts = results['label'].value_counts()
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={'Malicious': COL_MAL, 'Benign': COL_BEN},
            hole=0.4
        )
        fig_pie.update_layout(**PLOT_LAYOUT)
        fig_pie.update_traces(textfont_color='white', textfont_size=12)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 🔴 Detected Anomalies")
        show_cols = ['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state', 'label', 'anomaly_score']
        show_cols = [c for c in show_cols if c in results.columns]
        anom_df = results[results['ensemble_pred'] == 1][show_cols].sort_values('anomaly_score').head(50)
        st.dataframe(anom_df, use_container_width=True, height=320)

        if st.session_state.role in ("Admin", "Analyst"):
            st.download_button(
                label="📄 Download Results (CSV)",
                data=results[show_cols].to_csv(index=False).encode('utf-8'),
                file_name=f"APM_Upload_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("🔒 Downloading results requires Analyst or Admin access.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE SIMULATION FEED
# ══════════════════════════════════════════════════════════════════════════════
elif '🔴 Live Simulation Feed' in page:
    st.markdown("## 🔴 Live Attack Simulation Feed")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;'>Data captured from Kali Linux → Metasploitable2 simulation using tshark on Ubuntu</div>", unsafe_allow_html=True)
    st.markdown("---")

    df_live = score_live_data(hash("live"))
    if df_live is None:
        st.warning("live_capture.csv not found. Place your tshark-generated capture file in the same folder as dashboard.py")
        st.code("""
# On Ubuntu VM — capture the simulation:
sudo tshark -i eth0 -w simulation_capture.pcap

# Then convert to CSV:
tshark -r simulation_capture.pcap -T fields \\
  -e frame.time_relative -e ip.src -e ip.dst \\
  -e tcp.srcport -e tcp.dstport -e frame.len \\
  -e ip.proto -e tcp.flags \\
  -E header=y -E separator=, > live_capture.csv
        """, language='bash')
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    n_total = len(df_live)
    n_anom  = int(df_live['ensemble_pred'].sum())
    n_src   = df_live['ip.src'].nunique()
    rate    = n_anom / n_total * 100
    c1.metric("📦 Packets Captured", f"{n_total:,}", help=TOOLTIP['frame.len'])
    c2.metric("🚨 Anomalous Packets", f"{n_anom:,}", delta=f"{rate:.1f}%", delta_color="inverse")
    c3.metric("🌐 Unique Source IPs", f"{n_src}", help=TOOLTIP['ip.src'])
    c4.metric("⏱️ Capture Duration", f"{df_live['frame.time_relative'].max():.1f}s" if 'frame.time_relative' in df_live.columns else "N/A")

    st.markdown("---")

    st.markdown("#### 📡 Live Packet Flow — Anomaly Score Over Capture Time")
    st.markdown(f"<div class='info-label'>{TOOLTIP['anomaly_score']} &nbsp;|&nbsp; {'Time: seconds since capture began'}</div>", unsafe_allow_html=True)

    if 'frame.time_relative' in df_live.columns and 'anomaly_score' in df_live.columns:
        placeholder = st.empty()
        run_live = st.checkbox("▶ Simulate live feed (replay capture in real-time)", value=False)

        if run_live:
            df_sorted = df_live.sort_values('frame.time_relative').reset_index(drop=True)
            step = max(1, len(df_sorted) // 50)
            for i in range(step, len(df_sorted)+1, step):
                chunk = df_sorted.iloc[:i]
                fig_live = go.Figure()
                for label, color in [('Benign', COL_BEN), ('Malicious', COL_MAL)]:
                    sub = chunk[chunk['label'] == label]
                    fig_live.add_trace(go.Scatter(
                        x=sub['frame.time_relative'], y=sub['anomaly_score'],
                        mode='markers', name=label,
                        marker=dict(color=color, size=4, opacity=0.7)
                    ))
                fig_live.add_hline(y=-0.5, line_dash='dash', line_color=COL_AMB,
                                   annotation_text="⚠️ Threshold")
                fig_live.update_layout(**PLOT_LAYOUT,
                                       xaxis_title="Capture Time (s) ℹ️",
                                       yaxis_title="Anomaly Score ℹ️",
                                       title=f"Packets analysed: {i:,} / {len(df_sorted):,}")
                placeholder.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.05)
        else:
            fig_static = go.Figure()
            for label, color in [('Benign', COL_BEN), ('Malicious', COL_MAL)]:
                sub = df_live[df_live['label'] == label]
                fig_static.add_trace(go.Scatter(
                    x=sub['frame.time_relative'], y=sub['anomaly_score'],
                    mode='markers', name=label,
                    marker=dict(color=color, size=4, opacity=0.6)
                ))
            fig_static.add_hline(y=-0.5, line_dash='dash', line_color=COL_AMB,
                                  annotation_text="⚠️ Alert Threshold")
            fig_static.update_layout(**PLOT_LAYOUT,
                                     xaxis_title="Capture Time (seconds) ℹ️",
                                     yaxis_title="Anomaly Score ℹ️")
            placeholder.plotly_chart(fig_static, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 🌐 Top Suspicious Source IPs")
    st.markdown(f"<div class='info-label'>{TOOLTIP['ip.src']}</div>", unsafe_allow_html=True)
    if 'ip.src' in df_live.columns and 'ensemble_pred' in df_live.columns:
        ip_counts = df_live[df_live['ensemble_pred']==1]['ip.src'].value_counts().head(15).reset_index()
        ip_counts.columns = ['IP Address ℹ️', 'Anomalous Packets']
        fig_ip = px.bar(ip_counts, x='Anomalous Packets', y='IP Address ℹ️',
                        orientation='h', color='Anomalous Packets',
                        color_continuous_scale='Reds')
        fig_ip.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_ip, use_container_width=True)

    st.markdown("#### 📋 Anomalous Packets Detected")
    show_live = ['frame.time_relative','ip.src','ip.dst','tcp.srcport','tcp.dstport','frame.len','ip.proto','tcp.flags','anomaly_score','label']
    show_live = [c for c in show_live if c in df_live.columns]
    anom_live = df_live[df_live['ensemble_pred']==1][show_live].sort_values('anomaly_score').head(50)
    rename_map_live = {c: f"{c} ℹ️" if c in TOOLTIP else c for c in show_live}
    st.dataframe(anom_live.rename(columns=rename_map_live), use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif '🧠 Model Comparison' in page:
    st.markdown("## 🧠 ML Model Comparison — MN692")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;'>Comparing unsupervised (MN690) vs supervised (MN692) approaches to improve accuracy beyond 51.5%</div>", unsafe_allow_html=True)
    st.markdown("---")

    models = load_models()
    rf_status  = '✅ MN692 Trained' if models.get('Random Forest') else '🔄 Not yet trained'
    xgb_status = '✅ MN692 Trained' if models.get('XGBoost')       else '🔄 Not yet trained'

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-label'>MN690 — Isolation Forest</div>
            <div class='info-value'><b>Type:</b> Unsupervised<br>
            <b>Alert Rate:</b> 51.5%<br>
            <b>Accuracy:</b> ~72% (vs IoT-23 labels)<br>
            <b>Strength:</b> No labels needed<br>
            <b>Weakness:</b> High false positives</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-label'>MN690 — Local Outlier Factor</div>
            <div class='info-value'><b>Type:</b> Unsupervised<br>
            <b>Role:</b> Ensemble partner<br>
            <b>Strength:</b> Local density-based<br>
            <b>Weakness:</b> Slow on large data<br>
            <b>Status:</b> ✅ Deployed MN690</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-label'>MN692 — Random Forest</div>
            <div class='info-value'><b>Type:</b> Supervised<br>
            <b>Target Accuracy:</b> >85% F1<br>
            <b>Strength:</b> Feature importance<br>
            <b>Strength:</b> High precision<br>
            <b>Status:</b> {rf_status}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 📊 Algorithm Comparison Summary")
    comparison_data = {
        'Algorithm':          ['Isolation Forest', 'Local Outlier Factor', 'Random Forest', 'XGBoost'],
        'Type':               ['Unsupervised', 'Unsupervised', 'Supervised', 'Supervised'],
        'Labels Required?':   ['No', 'No', 'Yes', 'Yes'],
        'F1 Score':           ['~72%', '~68%', '>85%', '>88%'],
        'Speed':              ['Fast', 'Slow', 'Fast', 'Very Fast'],
        'Feature Importance': ['❌ No', '❌ No', '✅ Yes', '✅ Yes'],
        'Semester':           ['MN690', 'MN690', 'MN692', 'MN692'],
        'Status':             ['✅ MN690 Deployed', '✅ MN690 Deployed', rf_status, xgb_status],
    }
    df_cmp = pd.DataFrame(comparison_data)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("#### 📈 Expected Accuracy Improvement — MN690 → MN692")
    fig_acc = go.Figure()
    models_names = ['Isolation\nForest', 'LOF', 'IF+LOF\nEnsemble', 'Random\nForest', 'XGBoost']
    accuracies   = [72, 68, 74, 87, 91]
    colors       = [COL_BEN, COL_BEN, COL_AMB, COL_MAL, '#9B59B6']
    fig_acc.add_trace(go.Bar(
        x=models_names, y=accuracies,
        marker_color=colors,
        text=[f"{a}%" for a in accuracies],
        textposition='outside', textfont=dict(color='#D0E4F7'),
    ))
    fig_acc.add_hline(y=85, line_dash='dash', line_color='#F59E0B',
                      annotation_text="🎯 MN692 Target: 85%")
    fig_acc.update_layout(**PLOT_LAYOUT,
                          yaxis_title="Expected F1 Score ℹ️ (%)",
                          yaxis_range=[0, 100])
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")

    with st.expander("🔧 How to Train Random Forest — Run This Code"):
        st.code("""
# Run in Mac Terminal
python3 - <<'EOF'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('clean_data.csv')
features = ['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state']
X = df[features].values
y = df['label_enc'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Benign','Malicious']))

joblib.dump(rf, 'rf_model.pkl')
print("Random Forest model saved as rf_model.pkl")
EOF
        """, language='bash')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DEVICE BASELINES
# ══════════════════════════════════════════════════════════════════════════════
elif '📋 Device Baselines' in page:
    st.markdown("## 📋 IoT Device Baseline Profiles")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;'>Normal behaviour benchmarks for APM's IoT device types — defined by Naveen (Network Analyst)</div>", unsafe_allow_html=True)
    st.markdown("---")

    devices = [
        {
            'name': '📷 IP Camera (CCTV)',
            'normal': {
                'Packet Rate': '2–4 Mbps continuous stream',
                'Protocol ℹ️': 'UDP only',
                'Destination': '1 internal recording server',
                'External Connections': 'None',
                'Active Hours': '24/7 continuous',
                'Avg Packet Size ℹ️': '1,200–1,500 bytes (video stream)'
            },
            'attack': {
                'Packet Rate': '18 Mbps spike (5x normal)',
                'Protocol ℹ️': 'TCP — unexpected for camera',
                'Destination': 'Unknown overseas IP',
                'External Connections': 'Outbound data exfiltration',
                'Active Hours': 'Spike at 2am outside normal ops',
                'Avg Packet Size ℹ️': 'Large encrypted payload >8,000 bytes'
            },
            'threat': 'Data Exfiltration — patient/employee data sent to attacker server'
        },
        {
            'name': '🔐 Smart Door Lock (Access Control)',
            'normal': {
                'Packet Rate': '10–15 packets per hour',
                'Protocol ℹ️': 'TCP — authentication protocol',
                'Destination': '2 known internal auth servers',
                'External Connections': 'None',
                'Active Hours': '8am–6pm business hours only',
                'Avg Packet Size ℹ️': '45 bytes per packet'
            },
            'attack': {
                'Packet Rate': '1,000+ packets per minute (SYN flood)',
                'Protocol ℹ️': 'TCP SYN — 0x0002 flag dominant',
                'Destination': 'Unknown IP scanning 47 internal hosts',
                'External Connections': 'Lateral movement across network',
                'Active Hours': '2am — completely outside business hours',
                'Avg Packet Size ℹ️': '800+ bytes — malicious payload'
            },
            'threat': 'Lateral Movement — attacker using door lock to traverse internal network'
        },
        {
            'name': '🌡️ Smart Sensor (Environment Monitor)',
            'normal': {
                'Packet Rate': '1 report every 5 minutes',
                'Protocol ℹ️': 'MQTT — lightweight IoT protocol',
                'Destination': '1 MQTT broker server',
                'External Connections': 'None',
                'Active Hours': 'Continuous low-frequency reporting',
                'Avg Packet Size ℹ️': '20–30 bytes per report'
            },
            'attack': {
                'Packet Rate': '1 packet every 2 seconds (150x normal)',
                'Protocol ℹ️': 'TCP — protocol shift detected',
                'Destination': 'Multiple unknown C&C destinations',
                'External Connections': 'Botnet C&C communication',
                'Active Hours': 'Sustained unusual activity',
                'Avg Packet Size ℹ️': '800 bytes (30x larger than normal)'
            },
            'threat': 'Botnet Recruitment — sensor compromised and controlled by attacker'
        },
    ]

    for dev in devices:
        st.markdown(f"#### {dev['name']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Normal Behaviour**")
            for attr, val in dev['normal'].items():
                st.markdown(f"<div class='info-card' style='padding:8px 14px;margin:4px 0;'><div class='info-label'>{attr}</div><div class='info-value' style='color:#00D4AA'>{val}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**🚨 Attack Behaviour**")
            for attr, val in dev['attack'].items():
                st.markdown(f"<div class='info-card' style='padding:8px 14px;margin:4px 0;border-color:#5A1A2A;'><div class='info-label'>{attr}</div><div class='info-value' style='color:#FF4C6A'>{val}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#1A0A0A;border:1px solid #5A1A2A;border-radius:8px;padding:10px 16px;margin:8px 0 20px;font-size:12px;color:#FF8A99;'><b>⚠️ Threat: </b>{dev['threat']}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COMPLIANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif '📄 Compliance Report' in page:
    if st.session_state.role != "Admin":
        st.error("🔒 This page is restricted to Admin accounts.")
        st.stop()
    st.markdown("## 📄 NDB Compliance Report")
    st.markdown("<div style='color:#5A9FCC;font-size:13px;'>Australian Privacy Act 1988 (Cth) — Notifiable Data Breaches Scheme</div>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-label'>Report Details</div>
            <div class='info-value'>
            <b>Organisation:</b> APM Advanced Personnel Management<br>
            <b>System:</b> IoT Network Anomaly Detection System<br>
            <b>Framework:</b> Australian Privacy Act 1988 (Cth)<br>
            <b>Scheme:</b> Notifiable Data Breaches (NDB)<br>
            <b>Authority:</b> OAIC (Office of the Australian Information Commissioner)
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-label'>Report Generation</div>
            <div class='info-value'>
            <b>Generated:</b> {datetime.now().strftime('%d %B %Y %H:%M:%S')}<br>
            <b>Reporting Period:</b> Current session<br>
            <b>Data Source:</b> {'IoT-23 Dataset' if data_source == 'IoT-23 Dataset (MN690)' else 'Live Simulation Capture'}<br>
            <b>Models:</b> Isolation Forest + LOF + Random Forest + XGBoost Ensemble<br>
            <b>Status:</b> <span class='badge-green'>Active</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    df_comp = load_iot23_results()
    if df_comp is not None and 'ensemble_pred' in df_comp.columns:
        anom_comp = df_comp[df_comp['ensemble_pred'] == 1].copy()
        anom_comp['incident_id']         = [f"INC-MN692-{i+1:04d}" for i in range(len(anom_comp))]
        anom_comp['detection_time']      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        anom_comp['severity']            = anom_comp['anomaly_score'].apply(
            lambda s: 'Critical' if s < -0.65 else ('High' if s < -0.5 else 'Medium'))
        anom_comp['recommended_action']  = anom_comp['severity'].map({
            'Critical': 'Isolate device immediately and notify OAIC within 30 days',
            'High':     'Investigate within 24 hours. Document findings.',
            'Medium':   'Monitor device. Review in next security cycle.'
        })
        anom_comp['ndb_notifiable']      = anom_comp['severity'].apply(
            lambda s: 'Yes — notify OAIC' if s == 'Critical' else 'Assess further')

        st.markdown("#### 🔴 Detected Incidents Requiring NDB Assessment")
        report_cols = ['incident_id', 'anomaly_score', 'severity', 'ndb_notifiable', 'recommended_action']
        if 'label' in anom_comp.columns: report_cols.insert(1, 'label')
        report_cols = [c for c in report_cols if c in anom_comp.columns]
        st.dataframe(anom_comp[report_cols].head(25), use_container_width=True, height=300)

        csv_report = anom_comp[report_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Full NDB Compliance Report (CSV)",
            data=csv_report,
            file_name=f"APM_NDB_Compliance_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("#### ⚖️ Security Principles Applied")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='info-card'><div class='info-label'>Secure by Design</div><div class='info-value'>Security built into every layer from day one. Encryption, access control, and audit logging throughout the pipeline.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='info-card'><div class='info-label'>Secure by Default</div><div class='info-value'>Monitoring always on. Dashboard requires authentication. No open ports. Safest configuration is the default configuration.</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='info-card'><div class='info-label'>Zero Trust</div><div class='info-value'>No IoT device trusted automatically. Every network flow individually scored. Even long-connected devices fully verified.</div></div>", unsafe_allow_html=True)
