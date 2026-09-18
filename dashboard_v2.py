
"""
PulseGuard — IoT Network Anomaly Detection System
Real-time anomaly detection for IoT network traffic.
"""

import streamlit as st
st.caption('BUILD CHECK: v-fix-20260918-1600')
import hashlib
import pandas as pd
try:
    with open('results.csv', 'rb') as _f:
        _filehash = hashlib.md5(_f.read()).hexdigest()[:8]
    _diag_df = pd.read_csv('results.csv')
    _diag_rate = _diag_df['ensemble_pred'].mean() * 100
    st.caption(f'DIAG: file_hash={_filehash} rows={len(_diag_df)} live_alert_rate={_diag_rate:.2f}%')
except Exception as _e:
    st.caption(f'DIAG ERROR: {_e}')

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

# ── BRAND LOGO (base64, loaded once at module level) ──────────────────────────
from pathlib import Path as _Path
_LOGO_B64 = __import__('base64').b64encode(
    _Path('assets/pulseguard_logo.png').read_bytes()
).decode()

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
    st.markdown(
        """
        <style>
        /* Center the login form */
        [data-testid="stForm"] {
            max-width: 400px;
            margin: 0 auto 0 auto;
            border: 1px solid #2A2F3A;
            border-radius: 4px;
            padding: 32px 28px;
            background: #15181F;
        }
        /* Brand header block above the form */
        .pg-login-brand-wrap {
            max-width: 400px;
            margin: 8px auto 0 auto;
            text-align: center;
        }
        .pg-login-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 28px;
            letter-spacing: -0.01em;
            color: #E9EBEF;
            margin: 16px 0 6px 0;
        }
        .pg-login-sub {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: #8B93A1;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 20px;
        }
        .pg-login-rule {
            border: none !important;
            border-top: 1px solid #2A2F3A !important;
            display: block !important;
            width: 400px !important;
            max-width: 400px !important;
            margin: 24px auto 32px auto !important;
            padding: 0 !important;
            background: transparent !important;
            box-sizing: border-box !important;
        }
        /* Role hint below the form */
        .pg-login-hint {
            max-width: 400px;
            margin: 18px auto 0 auto;
            text-align: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 10.5px;
            color: #5B6270;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Centered logo
    _lc1, _lc2, _lc3 = st.columns([1, 0.25, 1])
    with _lc2:
        st.image('assets/pulseguard_logo.png', width='stretch')

    # Centered brand text + rule
    st.markdown(
        '''<div class="pg-login-brand-wrap">
        <div class="pg-login-brand">PulseGuard</div>
        <div class="pg-login-sub">IoT Network Anomaly Detection</div>
        </div>
        <hr class="pg-login-rule">''',
        unsafe_allow_html=True,
    )

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
    st.markdown(
        '<div class="pg-login-hint">Role-based access · Admin &middot; Analyst &middot; Read-only</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── THEME CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
    --bg:         #0E1015;
    --surface:    #15181F;
    --surface-2:  #1B1F28;
    --line:       #2A2F3A;
    --text:       #E9EBEF;
    --text-mute:  #8B93A1;
    --text-dim:   #5B6270;
    --signal:     #FF6B35;
    --signal-dim: #8A3D1F;
    --calm:       #3ECF8E;
    --alert:      #EF4444;
    --amber:      #F59E0B;
}
html, body, [class*="css"], .stApp, .stMarkdown, .stMetric,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
.stDataFrame, .stButton > button, .stDownloadButton > button,
input, textarea, select, button {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp { background: var(--bg) !important; color: var(--text); }
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
    padding: 16px !important;
    box-shadow: none !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text) !important;
    font-size: 1.75rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-mute) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stDataFrame {
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
.stButton > button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--signal) !important;
    color: var(--signal) !important;
}
.stDownloadButton > button,
[data-testid="stDownloadButton"] > button,
.stDownloadButton button,
button[kind="secondary"] {
    background: var(--signal) !important;
    color: #12100D !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 500 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] > button:hover,
.stDownloadButton button:hover,
button[kind="secondary"]:hover {
    background: #ff7d4d !important;
    color: #12100D !important;
}
.info-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: none;
}
.info-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.info-value { font-size: 13.5px; color: var(--text); line-height: 1.55; }
.badge-red   { background: var(--alert);  color: #fff; padding: 2px 9px; border-radius: 2px; font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-amber { background: var(--amber);  color: #fff; padding: 2px 9px; border-radius: 2px; font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-green { background: var(--calm);   color: #08150F; padding: 2px 9px; border-radius: 2px; font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-blue  { background: #2563EB;       color: #fff; padding: 2px 9px; border-radius: 2px; font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.ekg-wrap { overflow: hidden; width: 100%; height: 34px; margin: 2px 0 4px; }
.ekg-line { width: 200%; height: 34px; animation: ekg-scroll 3.2s linear infinite; }
@keyframes ekg-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.live-badge { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }
.live-badge-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--text-mute); letter-spacing: 0.1em; text-transform: uppercase; }
div[data-testid="column"] button[kind="primary"]{
    background: transparent !important;
    color: var(--signal) !important;
    border: none !important;
    border-bottom: 2px solid var(--signal) !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="column"] button[kind="secondary"]{
    background: transparent !important;
    color: var(--text-mute) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="column"] button[kind="secondary"]:hover{
    color: var(--text) !important;
    border-bottom-color: var(--line) !important;
}
.pg-brand { display: flex; align-items: center; gap: 10px; }
.pg-brand img { width: 26px; height: 26px; }
.pg-brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 17px;
    letter-spacing: -0.01em;
    color: var(--text);
}
.pg-credits { font-size: 11px; color: var(--text-dim); line-height: 1.6; }
.pg-credits img { width: 22px; height: 22px; margin-bottom: 4px; }
.pg-credits summary { cursor: pointer; outline: none; }
.pg-credits[open] summary { margin-bottom: 8px; }
.pg-credits-body { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; color: var(--text-dim); }
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

# Background removed — flat brand colour (#0E1015) via CSS above

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
    'ensemble_pred':     'ℹ️ Ensemble Prediction: 1 = malicious (Random Forest AND XGBoost both agree), 0 = normal',
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
    """Load pre-computed IoT-23 results"""
    try:
        df = pd.read_csv('results.csv')
        return df
    except:
        return None

@st.cache_resource
def load_models():
    """Load all trained ML models"""
    models = {}
    for name, path in [
        ('Isolation Forest', 'models_new_v2/isolation_forest_model.pkl'),
        ('LOF',              'models_new_v2/lof_model.pkl'),
        ('Random Forest',    'models_new_v2/rf_model.pkl'),
        ('XGBoost',          'models_new_v2/xgb_model.pkl'),
    ]:
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            models[name] = None
            st.warning(f"⚠️ Failed to load {name}: {e}")
    return models

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="pg-brand"><img src="data:image/png;base64,{_LOGO_B64}" alt=""><span class="pg-brand-name">PulseGuard</span></div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#8B93A1;'>IoT Anomaly Detection</div>", unsafe_allow_html=True)
    st.markdown("---")

    role_badge = {"Admin": "🔴", "Analyst": "🟡", "Read-only": "🟢"}.get(st.session_state.role, "")
    st.markdown(f"**Logged in as:** {st.session_state.username}  \n{role_badge} `{st.session_state.role}`")
    if st.button("🚪 Log Out"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
        # Clear any upload state so a fresh login starts from the baseline
        st.session_state.pop('last_upload_rate', None)
        st.session_state.pop('uploaded_results', None)
        st.session_state.pop('uploaded_filename', None)
        st.rerun()

    st.markdown("---")

    ROLE_PAGES = {
        "Admin":     ["Overview Dashboard", "Upload & Detect", "Model Comparison",
                      "Device Baselines", "Compliance Report"],
        "Analyst":   ["Overview Dashboard", "Upload & Detect", "Model Comparison",
                      "Device Baselines"],
        "Read-only": ["Overview Dashboard", "Device Baselines"],
    }

    st.markdown("<div class='info-label'>Data Source</div>", unsafe_allow_html=True)
    data_source = st.selectbox("Select Data Source", ["IoT-23 Dataset"])

    st.markdown("---")
    st.markdown("<div class='info-label'>System Status</div>", unsafe_allow_html=True)
    st.markdown("<span style='color:#3ECF8E'>● Models Loaded</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#3ECF8E'>● Pipeline Active</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#8B93A1;font-size:10px;'>Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}</span>", unsafe_allow_html=True)

    # Pulse reflects the currently-displayed dataset. If an upload is
    # active in session state, use its rate. Otherwise fall back to the
    # reference results file (baseline).
    if 'uploaded_results' in st.session_state and 'last_upload_rate' in st.session_state:
        _rate = st.session_state['last_upload_rate']
    else:
        _pulse_df = load_iot23_results()

        if _pulse_df is not None and 'ensemble_pred' in _pulse_df.columns and len(_pulse_df) > 0:
            _total = len(_pulse_df)
            _anom  = int(_pulse_df['ensemble_pred'].sum())
            _rate  = _anom / _total * 100
        else:
            _rate = None

    if _rate is None:
        _pulse_color, _pulse_status, _pulse_speed = "#8B93A1", "No data loaded", 4.0
    elif _rate < 30:
        _pulse_color, _pulse_status, _pulse_speed = "#3ECF8E", "Nominal", 3.2
    elif _rate < 60:
        _pulse_color, _pulse_status, _pulse_speed = "#F59E0B", "Elevated", 2.2
    else:
        _pulse_color, _pulse_status, _pulse_speed = "#EF4444", "Critical", 1.3

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
    st.markdown(
        f'''<details class="pg-credits">
        <summary><img src="data:image/png;base64,{_LOGO_B64}" alt="PulseGuard"></summary>
        <div class="pg-credits-body">
        PulseGuard v2.0 · IoT Anomaly Detection<br><br>
        Built by Lokesh, Mani, Navoda, Naveen &amp; Kishore<br>
        Client: APM (Advanced Personnel Management)<br>
        MN692 Capstone · Supervisor: Ahmed Jawad Khan
        </div>
        </details>''',
        unsafe_allow_html=True
    )

# ── TOP NAVIGATION BAR ─────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = ROLE_PAGES[st.session_state.role][0]

nav_items = ROLE_PAGES[st.session_state.role]
nav_cols = st.columns([1.4] + [1]*len(nav_items))
with nav_cols[0]:
    st.markdown("#### PulseGuard")
for i, item in enumerate(nav_items):
    with nav_cols[i+1]:
        is_active = st.session_state.current_page == item
        if st.button(item, key=f"nav_{item}",
                     type="primary" if is_active else "secondary",
                     width='stretch'):
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


# ── Label encodings used at TRAINING time (verified against clean_data.csv) ──
# These are the exact integer codes the RF / XGB / IF / LOF models were fit on.
PROTO_MAP      = {'icmp': 0, 'tcp': 1, 'udp': 2}
CONN_STATE_MAP = {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTR': 3, 'S0': 4,
                  'S1': 5, 'S2': 6, 'S3': 7, 'SF': 8, 'SHR': 9}


# ── MinMaxScaler parameters recovered from the training pipeline ──────────
# clean_data.csv was produced by applying MinMaxScaler to three numeric
# features. Without applying the SAME scaler at scoring time, the models
# receive raw byte counts up to 1.7e9 instead of the [0,1] values they
# were trained on — every row then looks anomalous.
#
# Recovered by pairing raw combined_data.csv with clean_data.csv row-by-row
# and solving raw = min + clean * (max - min) via least squares. Verified:
# reconstruction error < 3e-15 on all three features. See evaluation report §X.
SCALER_MIN = {
    'duration':   0.0,
    'orig_bytes': 0.0,
    'resp_bytes': 0.0,
}
SCALER_MAX = {
    'duration':   78_840.32931,
    'orig_bytes': 1_744_830_458.0,
    'resp_bytes': 336_516_351.0,
}


def apply_training_scaler(df):
    """Apply the same MinMaxScaler used during training.
    Must run BEFORE building X, and AFTER numeric coercion."""
    for col in ['duration', 'orig_bytes', 'resp_bytes']:
        mn, mx = SCALER_MIN[col], SCALER_MAX[col]
        if mx > mn:
            df[col] = (df[col] - mn) / (mx - mn)
        else:
            df[col] = 0.0
    return df


def encode_conn_state(code):
    """Maps a raw Zeek conn.log state string to the same integer code
    the models were trained on (verified 1-to-1 against clean_data.csv).
    Unknown states fall back to OTH=0, the 'everything else' bucket."""
    if pd.isna(code):
        return 0
    return CONN_STATE_MAP.get(str(code).upper(), 0)


def detect_and_normalize(df):
    """
    Detects whether an uploaded dataframe is IoT-23, CTU-13, or raw
    Zeek/Bro conn.log format, and returns (normalized_df, format_name).
    """
    cols = set(df.columns)

    zeek_signature = {'ts', 'uid', 'id.orig_h', 'id.resp_h'}
    if zeek_signature.issubset(cols) and {'duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state'}.issubset(cols):
        out = pd.DataFrame()
        out['duration']   = pd.to_numeric(df['duration'].replace('-', pd.NA), errors='coerce').fillna(0)
        out['orig_bytes'] = pd.to_numeric(df['orig_bytes'].replace('-', pd.NA), errors='coerce').fillna(0)
        out['resp_bytes'] = pd.to_numeric(df['resp_bytes'].replace('-', pd.NA), errors='coerce').fillna(0)
        out['proto']      = df['proto']
        out['conn_state'] = df['conn_state'].apply(encode_conn_state)
        return out, "Zeek/Bro conn.log"

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


def score_uploaded_data(df):
    """
    Runs the trained models on an uploaded, normalized dataframe.
    NOTE — ENCODING ASSUMPTION: this mapping is still unverified against
    the actual training encoding for RF/XGBoost. Treat RF/XGBoost
    predictions on uploaded files as provisional until confirmed.
    """
    df = df.copy()
    models = load_models()

    # Encode proto using the TRAINING-time label codes (verified against
    # clean_data.csv). If already numeric, trust it; otherwise apply PROTO_MAP.
    if pd.api.types.is_numeric_dtype(df['proto']):
        df['proto_enc'] = df['proto'].astype(int)
    else:
        df['proto_enc'] = (df['proto'].astype(str).str.lower()
                            .map(PROTO_MAP).fillna(0).astype(int))

    # Encode conn_state using the TRAINING-time label codes.
    if pd.api.types.is_numeric_dtype(df['conn_state']):
        df['conn_state_enc'] = df['conn_state'].astype(int)
    else:
        df['conn_state_enc'] = (df['conn_state'].astype(str).str.upper()
                                 .map(CONN_STATE_MAP).fillna(0).astype(int))

    features = ['duration', 'orig_bytes', 'resp_bytes', 'proto_enc', 'conn_state_enc']

    # Force EVERY feature numeric BEFORE building X — this is what stops the
    # silent 'Established' string from reaching RF/XGB via category/object dtypes.
    for _col in features:
        df[_col] = pd.to_numeric(df[_col], errors='coerce').fillna(0)

    # Apply the training-time MinMaxScaler. Without this, raw byte counts
    # (up to ~1.7e9) reach models trained on values in [0, 1] — every row
    # is out-of-distribution and flagged as anomalous. See evaluation report §X.
    df = apply_training_scaler(df)

    # Drop half-open connections: rows where every byte field is 0. These are
    # Zeek S0/OTH entries — SYN attempts with no reply, no payload. Standard
    # NIDS practice: filter them before scoring so the model sees real flows.
    _before = len(df)
    df = df[~((df['duration'] == 0) &
              (df['orig_bytes'] == 0) &
              (df['resp_bytes'] == 0))].reset_index(drop=True)
    _dropped = _before - len(df)
    if _dropped > 0:
        print(f"NOTE: dropped {_dropped:,} half-open rows (no payload) before scoring.")

    if len(df) == 0:
        print("WARN: all rows were filtered out as half-open. Nothing to score.")
        return pd.DataFrame(columns=features + ['iso_pred', 'lof_pred', 'rf_pred',
                                                 'xgb_pred', 'anomaly_score',
                                                 'ensemble_pred', 'label'])

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

    results['ensemble_pred'] = ((results['rf_pred'] == 1) & (results['xgb_pred'] == 1)).astype(int)
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
COL_MAL = '#EF4444'
COL_BEN = '#3ECF8E'
COL_AMB = '#F59E0B'

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if 'Overview Dashboard' in page:
    st.markdown("## IoT Network Anomaly Detection System")
    st.markdown("<div style='color:#8B93A1;font-size:13px;margin-bottom:20px;'>Client: APM (Advanced Personnel Management)</div>", unsafe_allow_html=True)

    if "uploaded_results" in st.session_state:
        df = st.session_state["uploaded_results"]
        source_label = f"Uploaded file — {st.session_state['uploaded_filename']}"
        if st.button("🔄 Clear uploaded data and return to default dataset"):
            st.session_state.pop("uploaded_results", None)
            st.session_state.pop("uploaded_filename", None)
            st.session_state.pop("last_upload_rate", None)
            st.rerun()
    elif data_source == "IoT-23 Dataset":
        df = load_iot23_results()
        source_label = "IoT-23 Dataset — 23 CSV files from Stratosphere Laboratory, CTU Prague"

    if df is None:
        st.error("Could not load data. Please ensure results.csv or live_capture.csv is in the same folder.")
        st.stop()

    st.markdown(f"<div style='font-size:11px;color:#3A7AAC;margin-bottom:12px;'>📦 Data source: {source_label}</div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    total    = len(df)
    n_anom   = int(df['ensemble_pred'].sum()) if 'ensemble_pred' in df.columns else 0
    n_normal = total - n_anom
    rate     = n_anom / total * 100 if total else 0

    col1.metric("Total Flows", f"{total:,}", help=TOOLTIP['duration'])
    col2.metric("Anomalies", f"{n_anom:,}", delta=f"{rate:.1f}% alert rate",
                delta_color="inverse", help=TOOLTIP['ensemble_pred'])
    col3.metric("Normal Flows", f"{n_normal:,}", help="Flows classified as benign by the ensemble")
    col4.metric("Alert Rate", f"{rate:.1f}%", help="Percentage of total flows flagged as anomalous")

    if 'anomaly_score' in df.columns:
        avg_score = df[df['ensemble_pred']==1]['anomaly_score'].mean()
        col5.metric("Avg Anomaly Score", f"{avg_score:.3f}", help=TOOLTIP['anomaly_score'])
    else:
        col5.metric("📊 Unique IPs", f"{df['ip.src'].nunique() if 'ip.src' in df.columns else 'N/A'}")

    st.markdown("---")

    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c1:
        st.markdown("#### Traffic Distribution")
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
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.markdown("#### Anomaly Score Distribution")
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
            st.plotly_chart(fig_hist, width='stretch')
        else:
            st.info("Anomaly scores not available for this data source")

    with c3:
        st.markdown("#### Protocol Distribution")
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
            st.plotly_chart(fig_proto, width='stretch')

    st.markdown("---")

    st.markdown("#### Anomaly Score Over Time")
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
        st.plotly_chart(fig_line, width='stretch')
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
        st.plotly_chart(fig_line, width='stretch')

    st.markdown("---")

    st.markdown("#### Top Detected Anomalies")

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
        st.markdown(f"<div style='font-size:10px;color:#8B93A1;margin-bottom:6px;'>{tip_text}</div>", unsafe_allow_html=True)

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
    st.dataframe(anom_df.rename(columns=rename_map), width='stretch', height=320)

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
elif 'Upload & Detect' in page:
    st.markdown("## Upload & Detect")
    st.markdown("<div style='color:#8B93A1;font-size:13px;'>Upload a network flow capture (IoT-23 or CTU-13 format) for on-demand anomaly detection</div>", unsafe_allow_html=True)
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
        st.dataframe(normalized_df.head(20), width='stretch')

    st.markdown("---")

    if st.button("🔍 Run Anomaly Detection", type="primary"):
        with st.spinner("Running models..."):
            results = score_uploaded_data(normalized_df)

        st.session_state["uploaded_results"] = results
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.success("✅ This dataset is now driving the Overview Dashboard too. Switch pages to see it.")

        n_total  = len(results)
        n_anom   = int(results['ensemble_pred'].sum())
        n_normal = n_total - n_anom
        rate     = n_anom / n_total * 100 if n_total else 0
        st.session_state['last_upload_rate'] = rate

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flows", f"{n_total:,}")
        c2.metric("Anomalies", f"{n_anom:,}", delta=f"{rate:.1f}% alert rate", delta_color="inverse")
        c3.metric("Normal Flows", f"{n_normal:,}")
        avg_score = results[results['ensemble_pred']==1]['anomaly_score'].mean() if n_anom else None
        c4.metric("Avg Anomaly Score", f"{avg_score:.3f}" if avg_score is not None else "N/A")

        st.markdown("---")
        st.markdown("#### Result Breakdown")
        counts = results['label'].value_counts()
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={'Malicious': COL_MAL, 'Benign': COL_BEN},
            hole=0.4
        )
        fig_pie.update_layout(**PLOT_LAYOUT)
        fig_pie.update_traces(textfont_color='white', textfont_size=12)
        st.plotly_chart(fig_pie, width='stretch')

        st.markdown("#### Detected Anomalies")
        show_cols = ['duration', 'orig_bytes', 'resp_bytes', 'proto', 'conn_state', 'label', 'anomaly_score']
        show_cols = [c for c in show_cols if c in results.columns]
        anom_df = results[results['ensemble_pred'] == 1][show_cols].sort_values('anomaly_score').head(50)
        st.dataframe(anom_df, width='stretch', height=320)

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
elif 'Model Comparison' in page:
    st.markdown("## ML Model Comparison")
    st.markdown("<div style='color:#8B93A1;font-size:13px;'>Unsupervised anomaly detection and supervised classification working together.</div>", unsafe_allow_html=True)
    st.markdown("---")

    models = load_models()
    rf_status  = 'Trained' if models.get('Random Forest') else 'Not yet trained'
    xgb_status = 'Trained' if models.get('XGBoost')       else 'Not yet trained'

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-label'>Unsupervised — Isolation Forest</div>
            <div class='info-value'>
            <b>Role:</b> Primary anomaly detector<br>
            <b>Strength:</b> No labels required; fast<br>
            <b>Weakness:</b> Sensitive to contamination setting<br>
            <b>Status:</b> Deployed
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-label'>Unsupervised — Local Outlier Factor</div>
            <div class='info-value'>
            <b>Role:</b> Comparison model (not used in final ensemble decision)<br>
            <b>Strength:</b> Local density-based detection<br>
            <b>Weakness:</b> Slower on large data<br>
            <b>Status:</b> Deployed
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-label'>Supervised — Random Forest</div>
            <div class='info-value'>
            <b>Role:</b> Classifier (ensemble decision-maker)<br>
            <b>Strength:</b> Feature importance; high precision<br>
            <b>Weakness:</b> Requires labelled training data<br>
            <b>Status:</b> {rf_status}
            </div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-label'>Supervised — XGBoost</div>
            <div class='info-value'>
            <b>Role:</b> Classifier (ensemble decision-maker)<br>
            <b>Strength:</b> Fast inference on tabular data<br>
            <b>Weakness:</b> Requires labelled training data<br>
            <b>Status:</b> {xgb_status}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Algorithm Comparison Summary")
    comparison_data = {
        'Algorithm':          ['Isolation Forest', 'Local Outlier Factor', 'Random Forest', 'XGBoost'],
        'Type':               ['Unsupervised', 'Unsupervised', 'Supervised', 'Supervised'],
        'Labels Required?':   ['No', 'No', 'Yes', 'Yes'],
        'F1 Score':           ['see report', 'see report', 'see report', 'see report'],
        'Speed':              ['Fast', 'Slow', 'Fast', 'Very Fast'],
        'Feature Importance': ['❌ No', '❌ No', '✅ Yes', '✅ Yes'],
        'Stage':              ['Base', 'Base', 'Supervised', 'Supervised'],
        'Status':             ['Deployed', 'Deployed', rf_status, xgb_status],
    }
    df_cmp = pd.DataFrame(comparison_data)
    st.dataframe(df_cmp, width='stretch', hide_index=True)

    st.markdown("---")

    st.markdown("#### Model Evaluation")
    st.markdown(
        "<div style='color:#8B93A1;font-size:13px;margin-bottom:14px;'>"
        "F1 scores computed live from the reference results file "
        "(per-model predictions vs. ground-truth labels)."
        "</div>",
        unsafe_allow_html=True,
    )

    # Compute F1 for each model from results.csv (real data, live computation)
    _ref = load_iot23_results()
    if _ref is not None and 'label_enc' in _ref.columns:
        from sklearn.metrics import f1_score as _f1
        _y = _ref['label_enc'].values
        _model_cols = [
            ('Isolation Forest', 'iso_pred'),
            ('Local Outlier Factor', 'lof_pred'),
            ('Random Forest', 'rf_pred'),
            ('XGBoost', 'xgb_pred'),
            ('Ensemble (RF AND XGBoost)', 'ensemble_pred'),
        ]
        _rows = []
        for _name, _col in _model_cols:
            if _col in _ref.columns:
                try:
                    _rows.append((_name, float(_f1(_y, _ref[_col].values)) * 100))
                except Exception:
                    pass

        if _rows:
            import plotly.graph_objects as _go
            _names  = [r[0] for r in _rows]
            _scores = [r[1] for r in _rows]
            _colors = [COL_BEN if s >= 85 else (COL_AMB if s >= 70 else COL_MAL) for s in _scores]

            _fig = _go.Figure()
            _fig.add_trace(_go.Bar(
                x=_names, y=_scores,
                marker_color=_colors,
                text=[f"{s:.1f}%" for s in _scores],
                textposition='outside',
                textfont=dict(color='#E9EBEF', family='IBM Plex Mono'),
            ))
            _fig.add_hline(
                y=85, line_dash='dash', line_color=COL_AMB,
                annotation_text="Target: 85%",
                annotation_font=dict(color='#8B93A1', family='IBM Plex Mono', size=11),
            )
            _fig.update_layout(
                **PLOT_LAYOUT,
                yaxis_title="F1 Score (%)",
                yaxis_range=[0, 105],
                showlegend=False,
            )
            st.plotly_chart(_fig, width='stretch')

            st.markdown(
                f"<div style='color:#5B6270;font-size:11px;font-family:IBM Plex Mono,monospace;margin-top:6px;'>"
                f"Computed on {len(_ref):,} reference flows · "
                f"Target threshold 85% F1 · "
                f"Higher is better"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Reference prediction columns not found in results.csv.")
    else:
        st.info("No reference results available for evaluation.")

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DEVICE BASELINES
# ══════════════════════════════════════════════════════════════════════════════
elif 'Device Baselines' in page:
    st.markdown("## IoT Device Baseline Profiles")
    st.markdown("<div style='color:#8B93A1;font-size:13px;'>Normal behaviour benchmarks for APM's IoT device types — defined by the PulseGuard team</div>", unsafe_allow_html=True)
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
                st.markdown(f"<div class='info-card' style='padding:8px 14px;margin:4px 0;'><div class='info-label'>{attr}</div><div class='info-value' style='color:#3ECF8E'>{val}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**🚨 Attack Behaviour**")
            for attr, val in dev['attack'].items():
                st.markdown(f"<div class='info-card' style='padding:8px 14px;margin:4px 0;border-color:#3A1A1A;'><div class='info-label'>{attr}</div><div class='info-value' style='color:#EF4444'>{val}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.35);border-left:3px solid #EF4444;border-radius:4px;padding:11px 16px;margin:8px 0 20px;font-size:12px;color:#EF4444;font-family:IBM Plex Mono,monospace;'><b>Threat:</b> {dev['threat']}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COMPLIANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif 'Compliance Report' in page:
    if st.session_state.role != "Admin":
        st.error("🔒 This page is restricted to Admin accounts.")
        st.stop()
    st.markdown("## NDB Compliance Report")
    st.markdown("<div style='color:#8B93A1;font-size:13px;'>Australian Privacy Act 1988 (Cth) — Notifiable Data Breaches Scheme</div>", unsafe_allow_html=True)
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
            <b>Data Source:</b> IoT-23 Dataset<br>
            <b>Models compared:</b> Isolation Forest, LOF, Random Forest, XGBoost<br>
            <b>Ensemble decision:</b> Random Forest AND XGBoost<br>
            <b>Status:</b> <span class='badge-green'>Active</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    df_comp = load_iot23_results()
    if df_comp is not None and 'ensemble_pred' in df_comp.columns:
        anom_comp = df_comp[df_comp['ensemble_pred'] == 1].copy()
        anom_comp['incident_id']         = [f"INC-{i+1:04d}" for i in range(len(anom_comp))]
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

        st.markdown("#### Detected Incidents Requiring NDB Assessment")
        report_cols = ['incident_id', 'anomaly_score', 'severity', 'ndb_notifiable', 'recommended_action']
        if 'label' in anom_comp.columns: report_cols.insert(1, 'label')
        report_cols = [c for c in report_cols if c in anom_comp.columns]
        st.dataframe(anom_comp[report_cols].head(25), width='stretch', height=300)

        csv_report = anom_comp[report_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Full NDB Compliance Report (CSV)",
            data=csv_report,
            file_name=f"APM_NDB_Compliance_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("#### Security Principles Applied")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='info-card'><div class='info-label'>Secure by Design</div><div class='info-value'>Security built into every layer from day one. Encryption, access control, and audit logging throughout the pipeline.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='info-card'><div class='info-label'>Secure by Default</div><div class='info-value'>Monitoring always on. Dashboard requires authentication. No open ports. Safest configuration is the default configuration.</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='info-card'><div class='info-label'>Zero Trust</div><div class='info-value'>No IoT device trusted automatically. Every network flow individually scored. Even long-connected devices fully verified.</div></div>", unsafe_allow_html=True)
