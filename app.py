"""
Multi-Agent Investment Analysis System
Main Streamlit Application

This is the main entry point for the investment analysis system.
Provides a web interface for stock analysis, portfolio recommendations, and backtesting.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime, timedelta
import os
from pathlib import Path
import yaml
import json
import time

# Setup page config
st.set_page_config(
    page_title="Total Insights Investing",
    page_icon="TI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS - Modern Investment Platform Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #3b5998;
    --primary-light: #5b7bb3;
    --primary-bg: #eef2f9;
    --primary-dark: #2c4a73;
    --success: #10b981;
    --success-bg: #ecfdf5;
    --warning: #f59e0b;
    --warning-bg: #fffbeb;
    --danger: #ef4444;
    --danger-bg: #fef2f2;
    --text: #111827;
    --text-secondary: #6b7280;
    --text-muted: #9ca3af;
    --border: #e5e7eb;
    --border-light: #f3f4f6;
    --surface: #ffffff;
    --bg: #f8fafc;
    --raised: #f9fafb;
    --shadow-xs: 0 1px 2px rgba(0,0,0,0.03);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.03);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -4px rgba(0,0,0,0.03);
    --r: 8px;
    --r-lg: 12px;
    --r-xl: 16px;
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== Global ===== */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp > header { background: transparent !important; }
* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }

h1, h2, h3, h4, h5, h6 {
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
h1 { font-size: 1.75rem !important; font-weight: 700 !important; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1rem !important; }
p, li, span, div { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }

.block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-light) !important; margin: 0.75rem 0 !important; }

/* ===== Metric Cards ===== */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--shadow-xs) !important;
    transition: var(--transition);
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    border-color: var(--primary) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] svg { width: 12px !important; height: 12px !important; }

/* ===== Tabs - Pill Navigation ===== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--r-xl) !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 2px !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--r-lg) !important;
    padding: 0.5rem 1.15rem !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    transition: var(--transition);
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(59,89,152,0.25) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ===== Buttons ===== */
.stButton > button {
    border-radius: var(--r) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.25rem !important;
    border: none !important;
    box-shadow: var(--shadow-sm) !important;
    transition: var(--transition);
    letter-spacing: -0.01em;
}
.stButton > button:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-light)) !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
    box-shadow: 0 4px 12px rgba(59,89,152,0.3) !important;
}
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-weight: 500 !important;
    transition: var(--transition);
}
.stDownloadButton > button:hover {
    background: var(--bg) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* ===== Expanders ===== */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 0.5rem !important;
    transition: var(--transition);
}
[data-testid="stExpander"]:hover {
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
}

/* ===== Alerts ===== */
.stAlert {
    border-radius: var(--r-lg) !important;
    border-left-width: 3px !important;
    font-size: 0.85rem !important;
}

/* ===== Inputs ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    caret-color: var(--primary) !important;
    padding: 0.6rem 0.875rem !important;
    font-size: 0.875rem !important;
    transition: var(--transition);
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-bg) !important;
}
.stSelectbox > div > div { border-radius: var(--r) !important; }

/* ===== Data Tables ===== */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ===== Dividers ===== */
.stMarkdown hr {
    border: none !important;
    border-top: 1px solid var(--border-light) !important;
    margin: 1.25rem 0 !important;
}

/* ===== Progress Bars ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
    border-radius: 999px !important;
}

/* ===== Forms ===== */
[data-testid="stForm"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
    padding: 1.25rem !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ===== Score Badge ===== */
.score-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 72px; height: 72px;
    border-radius: 50%;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    box-shadow: var(--shadow-md);
}
.score-badge.excellent { background: linear-gradient(135deg, #10b981, #059669); }
.score-badge.good { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.score-badge.moderate { background: linear-gradient(135deg, #f59e0b, #d97706); }
.score-badge.poor { background: linear-gradient(135deg, #ef4444, #dc2626); }

/* ===== Columns ===== */
[data-testid="column"] { padding: 0 0.375rem !important; }

/* ===== Selectbox / Multiselect ===== */
.stMultiSelect [data-baseweb="tag"] {
    background: var(--primary-bg) !important;
    color: var(--primary) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* ===== Smooth Scrolling ===== */
html { scroll-behavior: smooth; }

/* ===== Plotly Chart Containers ===== */
.js-plotly-plot { border-radius: var(--r-lg) !important; }
.js-plotly-plot .plotly .modebar { opacity: 0; transition: opacity 0.2s ease; }
.js-plotly-plot:hover .plotly .modebar { opacity: 1; }

/* ===== Toast / Notification ===== */
[data-testid="stToast"] { border-radius: var(--r-lg) !important; box-shadow: var(--shadow-lg) !important; }

/* ===== Plotly Charts — force white container ===== */
.js-plotly-plot,
.js-plotly-plot .plot-container,
.js-plotly-plot .svg-container {
    background: #ffffff !important;
    border-radius: var(--r-lg) !important;
}

/* ===== Code blocks ===== */
[data-testid="stCode"],
pre, code,
.stCodeBlock pre {
    background: #f8fafc !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}

/* ===== Streamlit native charts (Vega-Lite) ===== */
.vega-embed,
.vega-embed .chart-wrapper {
    background: #ffffff !important;
}

/* ===== Hide Streamlit Branding ===== */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stStatusWidget"] { visibility: hidden; }
[data-testid="stDecoration"] { background: none !important; display: none !important; }

/* =================================================================
   DARK-MODE IMMUNITY
   Streamlit's BaseWeb widgets inherit the OS color scheme via
   prefers-color-scheme.  On macOS Dark Mode the browser sends
   "dark" and BaseWeb flips backgrounds/text.  These overrides
   ensure every widget renders in light mode regardless of OS.
   ================================================================= */

/* --- Global containers --- */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stBottom"] {
    background-color: var(--surface) !important;
    color: var(--text) !important;
}

/* --- Text everywhere --- */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stMarkdown strong, .stMarkdown em, .stMarkdown a {
    color: var(--text) !important;
}

/* --- Buttons (non-primary default to white bg + dark text) --- */
.stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button p,
.stButton > button span {
    color: inherit !important;
}
/* Primary buttons keep gradient + white text */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-light)) !important;
    color: white !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
    color: white !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[data-testid="baseButton-primary"] span {
    color: white !important;
}

/* --- All BaseWeb inputs --- */
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="input"] > div,
[data-baseweb="base-input"] > div,
[data-baseweb="input"] input,
[data-baseweb="base-input"] input,
[data-baseweb="textarea"],
[data-baseweb="textarea"] textarea {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
    caret-color: var(--text) !important;
}

/* Input wrapper divs (Streamlit's extra wrapper) */
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea > div > div {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
}

/* --- Selectbox / Dropdown --- */
[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div,
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
}
/* Dropdown arrow icon */
[data-baseweb="select"] svg {
    color: var(--text-secondary) !important;
    fill: var(--text-secondary) !important;
}
/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[data-baseweb="list"] {
    background: var(--surface) !important;
    color: var(--text) !important;
}
[data-baseweb="menu-item"],
[data-baseweb="option"],
[role="option"] {
    background: var(--surface) !important;
    color: var(--text) !important;
}
[data-baseweb="menu-item"]:hover,
[data-baseweb="option"]:hover,
[role="option"]:hover {
    background: var(--primary-bg) !important;
    color: var(--primary) !important;
}

/* --- Date Input --- */
[data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stDateInput"] [data-baseweb="base-input"],
[data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stDateInput"] input {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
}
/* Calendar popover */
[data-baseweb="datepicker"],
[data-baseweb="calendar"],
[data-baseweb="calendar"] > div,
[data-baseweb="month-header"],
[data-baseweb="calendar-header"] {
    background: var(--surface) !important;
    color: var(--text) !important;
}
[data-baseweb="calendar"] button {
    color: var(--text) !important;
    background: transparent !important;
}
[data-baseweb="calendar"] button:hover {
    background: var(--primary-bg) !important;
}
[data-baseweb="calendar"] [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* --- Radio / Checkbox labels --- */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] > div > label,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p {
    color: var(--text) !important;
}

/* --- Widget labels --- */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stLabel p {
    color: var(--text-secondary) !important;
}

/* --- Alerts / Notifications --- */
.stAlert,
.stAlert > div,
[data-baseweb="notification"] {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.stAlert p, .stAlert span, .stAlert li,
.stAlert strong, .stAlert em {
    color: var(--text) !important;
}


/* --- Multiselect tags --- */
.stMultiSelect [data-baseweb="tag"] {
    background: var(--primary-bg) !important;
    color: var(--primary) !important;
}

/* --- Tooltip / Popover --- */
[data-baseweb="tooltip"],
[data-baseweb="tooltip"] > div {
    background: var(--surface) !important;
    color: var(--text) !important;
}

/* --- Tab panel content --- */
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    color: var(--text) !important;
}

/* --- Progress bar background --- */
.stProgress > div {
    background: var(--border-light) !important;
}

/* --- Expander toggle icon --- */
[data-testid="stExpander"] svg {
    color: var(--text-secondary) !important;
    fill: var(--text-secondary) !important;
}


/* --- Toggle / Switch --- */
[data-baseweb="toggle"] span { background: var(--border) !important; }
[data-baseweb="toggle"][aria-checked="true"] span,
[data-baseweb="toggle"] input:checked + span { background: var(--primary) !important; }

/* --- Slider: force blue accent --- */
/* Thumb knob */
[data-baseweb="slider"] [role="slider"] {
    background-color: #3b5998 !important;
    border-color: #3b5998 !important;
}
/* Thumb value label color */
[data-baseweb="slider"] [role="slider"] div {
    color: #3b5998 !important;
    background-color: transparent !important;
}

/* --- Number input stepper buttons --- */
.stNumberInput button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
.stNumberInput button:hover {
    background: var(--bg) !important;
}

/* --- Dataframe header row --- */
[data-testid="stDataFrame"] thead th {
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* =================================================================
   CATCH-ALL: OS-level dark mode override
   Even if Streamlit or BaseWeb injects dark-scheme styles via media
   query, we force our light palette everywhere.
   ================================================================= */
@media (prefers-color-scheme: dark) {
    .stApp,
    .stApp > header,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stBottom"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        color-scheme: light !important;
    }
    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="list"],
    [data-baseweb="tooltip"],
    [data-baseweb="tooltip"] > div {
        background: var(--surface) !important;
        color: var(--text) !important;
    }
    .js-plotly-plot,
    .js-plotly-plot .plot-container,
    .js-plotly-plot .svg-container,
    .vega-embed, .vega-embed .chart-wrapper {
        background: #ffffff !important;
    }
    [data-testid="stCode"], pre, code, .stCodeBlock pre {
        background: #f8fafc !important;
        color: var(--text) !important;
    }
}

/* Slider track — override BaseWeb orange accent with TI blue */
div[data-baseweb="slider"] [class*="InnerTrack"],
div[data-baseweb="slider"] [class*="Track"] > div:first-child {
    background: #3b5998 !important;
}
div[data-baseweb="slider"] [class*="Thumb"],
div[data-baseweb="slider"] [class*="InnerThumb"] {
    background: #3b5998 !important;
    border-color: #3b5998 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly Light Theme
# ---------------------------------------------------------------------------
CHART_COLORS = ["#3b5998", "#0ea371", "#3b82f6", "#d97706", "#ec4899",
                "#8b5cf6", "#06b6d4", "#f43f5e", "#10b981", "#5b7bb3"]

_chart_template = dict(
    layout=dict(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                  color="#111827", size=13),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(font=dict(size=15, color="#111827")),
        xaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb",
                   tickfont=dict(color="#6b7280", size=11),
                   title_font=dict(color="#6b7280", size=12), zeroline=False),
        yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb",
                   tickfont=dict(color="#6b7280", size=11),
                   title_font=dict(color="#6b7280", size=12), zeroline=False),
        legend=dict(font=dict(color="#6b7280", size=12),
                    bgcolor="#ffffff", bordercolor="#ffffff"),
        margin=dict(l=40, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor="white", bordercolor="#e5e7eb",
                        font=dict(color="#111827", size=13)),
        colorway=CHART_COLORS,
    )
)
pio.templates["invest_light"] = pio.templates["plotly_white"]
pio.templates["invest_light"].update(_chart_template)
pio.templates.default = "invest_light"



# Import system components
from dotenv import load_dotenv
from utils.config_loader import get_config_loader
from utils.logger import setup_logging, get_disclosure_logger
from utils.tier_manager import TierManager
from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator
from engine.backtest import BacktestEngine

# Import OpenAI at module level to avoid circular dependency issues
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(os.getenv('LOG_LEVEL', 'INFO'))

# Suppress noisy WebSocket errors from Streamlit (these are harmless)
import logging
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('tornado.websocket').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)

# Create logger instance
import logging
logger = logging.getLogger(__name__)


def initialize_system():
    """Initialize the system components."""
    if st.session_state.initialized:
        return True

    # Initialize tier manager
    if 'tier_manager' not in st.session_state:
        st.session_state.tier_manager = TierManager()
    tier = st.session_state.tier_manager

    # Check API keys via tier manager
    if not tier.get_api_key('OPENAI_API_KEY'):
        st.error("OPENAI_API_KEY not found. Please set it in .env file or provide your own key in the sidebar.")
        return False

    if not tier.get_api_key('ALPHA_VANTAGE_API_KEY'):
        st.warning("ALPHA_VANTAGE_API_KEY not found. Some features may be limited.")

    try:
        # Initialize components
        st.session_state.config_loader = get_config_loader()

        # Use Enhanced Data Provider with tier-resolved keys
        st.session_state.data_provider = EnhancedDataProvider(
            alpha_vantage_key=tier.get_api_key('ALPHA_VANTAGE_API_KEY'),
            news_api_key=tier.get_api_key('NEWS_API_KEY'),
            polygon_key=tier.get_api_key('POLYGON_API_KEY'),
        )

        # Load configurations
        model_config = st.session_state.config_loader.load_model_config()
        ips_config = st.session_state.config_loader.load_ips()

        # Initialize AI clients for advanced features
        openai_client = None
        gemini_api_key = None

        try:
            if OpenAI is not None:
                openai_client = OpenAI(api_key=tier.get_api_key('OPENAI_API_KEY'))
                st.session_state.openai_client = openai_client
            else:
                st.warning("OpenAI library not available. Please install: pip install openai")
        except Exception as e:
            st.warning(f"OpenAI client initialization failed: {e}")

        gemini_api_key = tier.get_api_key('GEMINI_API_KEY')
        if gemini_api_key:
            st.session_state.gemini_api_key = gemini_api_key
        else:
            st.warning("GEMINI_API_KEY not found. Portfolio AI selection will be limited.")

        # Initialize orchestrator with enhanced data provider and AI clients
        st.session_state.orchestrator = PortfolioOrchestrator(
            model_config=model_config,
            ips_config=ips_config,
            enhanced_data_provider=st.session_state.data_provider,
            openai_client=openai_client,
            gemini_api_key=gemini_api_key
        )
        
        
        # Initialize analysis time tracking (simple historical average)
        if 'analysis_times' not in st.session_state:
            st.session_state.analysis_times = []  # List of historical analysis times in seconds
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return False


def _render_privacy_policy_page():
    """Render the privacy policy as a full in-app page.
    
    Served at ?page=privacy so the URL lives on the app's own domain,
    satisfying Google OAuth verification requirements.
    """
    st.markdown(
        '<style>.block-container{padding-top:2.5rem !important;max-width:800px !important;}</style>',
        unsafe_allow_html=True,
    )
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;">
    <div style="background:linear-gradient(135deg,#2c4a73,#3b5998);color:white;font-weight:700;font-size:1rem;
                width:38px;height:38px;border-radius:10px;display:flex;
                align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(44,74,115,0.25);
                letter-spacing:-0.02em;">TI</div>
    <div>
        <div style="font-size:1.2rem;font-weight:700;color:#111827;letter-spacing:-0.03em;line-height:1.2;">
            Total Insights Investing</div>
        <div style="font-size:0.75rem;color:#9ca3af;font-weight:400;letter-spacing:0.01em;">
            Privacy Policy</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Read the markdown file and render it
    _pp_path = Path(__file__).parent / "PRIVACY_POLICY.md"
    if _pp_path.exists():
        st.markdown(_pp_path.read_text(), unsafe_allow_html=False)
    else:
        st.error("Privacy policy file not found.")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#9ca3af;font-size:0.8rem;padding:16px 0;">'
        '&copy; 2026 Total Insights Investing'
        '</div>',
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""

    # ---- Privacy policy page (must be before OAuth / system init) ----
    # Served at ?page=privacy so the URL is on the app's own domain.
    if st.query_params.get("page") == "privacy":
        _render_privacy_policy_page()
        return

#     # ---- Google OAuth callback (must run FIRST, before any display) ----
#     # When Google redirects back with ?code=..., exchange it for a token.
#     # This is at the top so it works even if session_state was lost.
#     _handle_google_oauth_callback()

    # Initialize tier manager early (before system init) so sidebar shows
    if 'tier_manager' not in st.session_state:
        st.session_state.tier_manager = TierManager()
    # Tier sidebar removed — API keys resolved from .env / Streamlit Secrets

    # Initialize session state keys (must be inside main() so a valid session exists)
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.data_provider = None
        st.session_state.orchestrator = None
        st.session_state.config_loader = None

    # Initialize system first (needed for analysis execution path)
    if not initialize_system():
        st.stop()

    # ---- Analysis history (session-level, last 5 results) ----
    if '_analysis_history' not in st.session_state:
        st.session_state['_analysis_history'] = []

    # Sidebar: Recent Analyses + Compare
    _hist = st.session_state['_analysis_history']
    if _hist:
        st.sidebar.markdown(
            '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'
            'color:#9ca3af;margin-bottom:8px;font-weight:600;">Recent Analyses</div>',
            unsafe_allow_html=True,
        )
        for _hi, _hentry in enumerate(_hist):
            _hticker = _hentry['ticker']
            _hscore = _hentry.get('blended_score', 0)
            _htime = _hentry.get('timestamp', '')
            if st.sidebar.button(
                f"{_hticker}  \u2022  {_hscore:.0f}/100  \u2022  {_htime}",
                key=f"hist_{_hi}",
                use_container_width=True,
            ):
                st.session_state['_display_result'] = _hentry['display_data']
                st.rerun()

        # Compare feature — available when 2+ results in history
        if len(_hist) >= 2:
            st.sidebar.markdown(
                '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'
                'color:#9ca3af;margin:12px 0 6px 0;font-weight:600;">Compare Stocks</div>',
                unsafe_allow_html=True,
            )
            _compare_tickers = st.sidebar.multiselect(
                "Select stocks",
                options=[h['ticker'] for h in _hist],
                default=[h['ticker'] for h in _hist[:2]],
                key="_compare_select",
                label_visibility="collapsed",
            )
            if len(_compare_tickers) >= 2:
                if st.sidebar.button("Compare Selected", use_container_width=True, type="primary", key="compare_btn"):
                    _compare_results = []
                    for _ct in _compare_tickers:
                        for _ch in _hist:
                            if _ch['ticker'] == _ct:
                                _cr = _ch['display_data'].get('result')
                                if _cr:
                                    _compare_results.append(_cr)
                                break
                    if len(_compare_results) >= 2:
                        st.session_state['_display_result'] = {
                            'mode': 'multi', 'results': _compare_results, 'failed': [],
                        }
                        st.rerun()

        st.sidebar.markdown("---")

    # ---- Results display path (persisted across reruns) ----
    # When results are stored in session state (after analysis completed),
    # re-render them even on widget-triggered reruns (e.g. Google auth button).
    if '_display_result' in st.session_state:
        _dr = st.session_state['_display_result']

        # Same CSS and header as the analysis path — combined with back button
        # for fastest possible rendering (single HTML block before any widgets)
        st.markdown(
            '<style>'
            '.block-container{padding-top:2.5rem !important;opacity:1 !important;}'
            '</style>',
            unsafe_allow_html=True,
        )

        # Back to Home at top of results page (render before anything else)
        if st.button("← Back to Home Page", type="secondary", key="back_home_top"):
            if '_analysis_params' in st.session_state:
                del st.session_state['_analysis_params']
            if '_display_result' in st.session_state:
                del st.session_state['_display_result']
            _cleanup_display_result_backup()
            st.rerun()

        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
            <div style="background:linear-gradient(135deg,#2c4a73,#3b5998);color:white;font-weight:700;font-size:1rem;
                        width:38px;height:38px;border-radius:10px;display:flex;
                        align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(44,74,115,0.25);
                        letter-spacing:-0.02em;">TI</div>
            <div>
                <div style="font-size:1.2rem;font-weight:700;color:#111827;letter-spacing:-0.03em;line-height:1.2;">
                    Total Insights Investing</div>
                <div style="font-size:0.75rem;color:#9ca3af;font-weight:400;letter-spacing:0.01em;">
                    Multi-Agent Investment Research Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if _dr['mode'] == 'single':
            display_stock_analysis(_dr['result'])
        else:
            display_multiple_stock_analysis(_dr['results'], _dr.get('failed', []))
        return

    # ---- Analysis execution path ----
    # Handle analysis BEFORE rendering any form elements so the entire UI
    # (header, form, tabs, weights, etc.) is replaced by ONLY the progress card.
    if '_analysis_params' in st.session_state:
        _ap = st.session_state.pop('_analysis_params')

        # CSS: extra top padding + prevent Streamlit running-state dimming
        st.markdown(
            '<style>'
            '.block-container{padding-top:2.5rem !important;opacity:1 !important;}'
            '</style>',
            unsafe_allow_html=True,
        )

        # Branded header so user has a visual anchor
        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
            <div style="background:linear-gradient(135deg,#2c4a73,#3b5998);color:white;font-weight:700;font-size:1rem;
                        width:38px;height:38px;border-radius:10px;display:flex;
                        align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(44,74,115,0.25);
                        letter-spacing:-0.02em;">TI</div>
            <div>
                <div style="font-size:1.2rem;font-weight:700;color:#111827;letter-spacing:-0.03em;line-height:1.2;">
                    Total Insights Investing</div>
                <div style="font-size:0.75rem;color:#9ca3af;font-weight:400;letter-spacing:0.01em;">
                    Multi-Agent Investment Research Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Create empty placeholders at positions that match old render elements
        # (tabs, form, inputs, etc.) so Streamlit diffs them away instead of
        # leaving them visible as stale/dimmed content.
        for _ in range(10):
            st.empty()

        # Run analysis — only the progress card appears below the header
        _execute_analysis(
            _ap['mode'],
            _ap.get('ticker'),
            _ap.get('tickers'),
            _ap['date'],
            _ap['weights'],
            regime_modulation=_ap.get('regime_modulation', False),
            regime_sensitivity=_ap.get('regime_sensitivity', 'moderate'),
        )

        # Don't render anything else
        return

    # ---- Normal rendering path (no analysis running) ----

    # Branded header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
        <div style="background:linear-gradient(135deg,#2c4a73,#3b5998);color:white;font-weight:700;font-size:1rem;
                    width:38px;height:38px;border-radius:10px;display:flex;
                    align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(44,74,115,0.25);
                    letter-spacing:-0.02em;">TI</div>
        <div>
            <div style="font-size:1.2rem;font-weight:700;color:#111827;letter-spacing:-0.03em;line-height:1.2;">
                Total Insights Investing</div>
            <div style="font-size:0.75rem;color:#9ca3af;font-weight:400;letter-spacing:0.01em;">
                Multi-Agent Investment Research Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top navigation tabs
#     tab_stock, tab_portfolio, tab_config, tab_status = st.tabs([
#         "Stock Analysis",
#         "Portfolio Recs",
#         "Configuration",
#         "System Status",
#     ])
# 
#     with tab_stock:
#         stock_analysis_page()
#     with tab_portfolio:
#         portfolio_recommendations_page()
#     with tab_config:
#         configuration_page()
#     with tab_status:
#         system_status_and_ai_disclosure_page()

    # Only Stock Analysis mode (other tabs commented out)
    stock_analysis_page()

    # Purpose description & privacy link (required for Google OAuth verification)
    st.markdown(
        '<div style="color:#6b7280;font-size:0.85rem;margin:24px 0 12px 0;line-height:1.5;text-align:center;">'
        'Total Insights Investing is a multi-agent investment research '
        'platform that analyzes stocks using AI-powered value, growth, macro, risk, '
        'and sentiment agents. It is designed for educational and informational purposes only, and does not provide financial advice. '
        '<a href="?page=privacy" target="_self" style="color:#3b5998;">Privacy Policy</a>'
        '</div>',
        unsafe_allow_html=True,
    )



def _execute_analysis(analysis_mode, ticker, tickers, analysis_date, agent_weights,
                      regime_modulation=False, regime_sensitivity="moderate"):
    """Execute stock analysis with progress display.
    
    This is extracted from the button handler so it can be called
    from the rerun path (form hidden) or directly.
    """
    import threading, time, math

    # Create empty slots for progress display
    progress_slot = st.empty()

    # Agent step definitions for the progress display
    # Clean inline SVG icons (Lucide-style, 14x14, stroke=currentColor)
    _SVG = {
        "data":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/></svg>',
        "value": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
        "growth":'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "macro": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        "risk":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "sent":  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 002-2V4a2 2 0 00-2-2H8a2 2 0 00-2 2v16a2 2 0 01-2 2zm0 0a2 2 0 01-2-2v-9c0-1.1.9-2 2-2h2"/><line x1="10" y1="8" x2="18" y2="8"/><line x1="10" y1="12" x2="18" y2="12"/><line x1="10" y1="16" x2="14" y2="16"/></svg>',
        "blend": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg>',
        "check": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    }

    # Step ranges must match the orchestrator's exact milestone percentages:
    # Data: 0-42, then 5 agents IN PARALLEL across 42-98, then blend 98-100
    _AGENT_STEPS = [
        {"key": "data",      "label": "Data Gathering",     "svg": "data",  "range": (0, 42),  "parallel": False},
        {"key": "value",     "label": "Value",              "svg": "value", "range": (42, 98), "parallel": True},
        {"key": "growth",    "label": "Growth",             "svg": "growth","range": (42, 98), "parallel": True},
        {"key": "macro",     "label": "Macro Regime",       "svg": "macro", "range": (42, 98), "parallel": True},
        {"key": "risk",      "label": "Risk",               "svg": "risk",  "range": (42, 98), "parallel": True},
        {"key": "sentiment", "label": "Sentiment",          "svg": "sent",  "range": (42, 98), "parallel": True},
        {"key": "blend",     "label": "Score Blending",     "svg": "blend", "range": (98, 100), "parallel": False},
    ]

    # Slate-blue palette for progress UI
    _SLATE_600 = "#3b5998"
    _SLATE_500 = "#5b7bb3"
    _SLATE_200 = "#dce4f0"
    _SLATE_100 = "#eef2f9"

    # Determine the ticker label for the progress card header
    _progress_ticker_label = ''
    if analysis_mode == 'Single Stock' and ticker:
        _progress_ticker_label = f' — {ticker}'
    elif analysis_mode == 'Multiple Stocks' and tickers:
        _progress_ticker_label = f' — {len(tickers)} stocks'

    def _render_progress(slot, bar_pct, message, remaining_secs=None, step_pct=None, completed_steps=None):
        """Render a professional analysis progress card with agent steps and countdown."""
        import re as _re
        bar_pct = max(0.0, min(100.0, float(bar_pct)))
        bar_pct_int = int(bar_pct)  # for the HTML width
        sp = int(step_pct) if step_pct is not None else bar_pct_int
        sp = max(0, min(100, sp))

        # --- Time remaining label ---
        if remaining_secs is not None and bar_pct < 100:
            rs = max(0, int(remaining_secs))
            if rs >= 60:
                time_label = f"~{rs // 60}m {rs % 60:02d}s remaining"
            else:
                time_label = f"~{rs}s remaining" if rs > 0 else "finishing up..."
        elif bar_pct >= 100:
            time_label = "Complete"
        else:
            time_label = "estimating..."

        # Strip ~Xs ETA suffix from message (already shown in timer)
        clean_msg = _re.sub(r'\s*~\d+(?:m\s+\d+)?s\s*$', '', message)

        # During agent phase, show which agents are still running
        if 42 <= sp < 98 and completed_steps is not None:
            _agent_labels = {
                'value': 'Value', 'growth': 'Growth', 'macro': 'Macro',
                'risk': 'Risk', 'sentiment': 'Sentiment',
            }
            _still_running = [lbl for key, lbl in _agent_labels.items()
                              if key not in completed_steps]
            _done_count = sum(1 for k in _agent_labels if k in completed_steps)
            if _still_running:
                clean_msg = f"Running {', '.join(_still_running)}... [{_done_count}/{len(_agent_labels)}]"

        # Separate sequential and parallel steps
        seq_steps = [s for s in _AGENT_STEPS if not s.get("parallel")]
        par_steps = [s for s in _AGENT_STEPS if s.get("parallel")]
        # Are we in the parallel agent phase?
        in_agent_phase = 42 <= sp < 98
        agent_phase_done = sp >= 98 or (completed_steps and all(
            s["key"] in completed_steps for s in par_steps))

        def _step_row(step, is_done, is_active):
            badge_bg, badge_fg, icon_svg, label_color, label_weight, row_bg = (
                ("#ecfdf5", "#059669", _SVG["check"], "#374151", "400", "transparent") if is_done else
                (_SLATE_200, _SLATE_600, _SVG[step["svg"]], "#111827", "600", _SLATE_100) if is_active else
                ("#f3f4f6", "#9ca3af", _SVG[step["svg"]], "#9ca3af", "400", "transparent")
            )
            pulse = (
                f'<span style="width:6px;height:6px;border-radius:50%;'
                f'background:{_SLATE_600};display:inline-block;margin-left:auto;'
                f'animation:_prog_pulse 1.4s ease-in-out infinite"></span>'
            ) if is_active else ""
            return (
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'padding:5px 8px;border-radius:6px;background:{row_bg};'
                f'transition:background 0.3s ease">'
                f'<span style="display:flex;align-items:center;justify-content:center;'
                f'width:26px;height:26px;border-radius:6px;font-size:13px;'
                f'background:{badge_bg};color:{badge_fg};flex-shrink:0">{icon_svg}</span>'
                f'<span style="font-size:13px;color:{label_color};'
                f'font-weight:{label_weight}">{step["label"]}</span>'
                f'{pulse}'
                f'</div>'
            )

        steps_html = ""

        # --- Data step ---
        data_step = seq_steps[0]  # data
        data_done = (completed_steps and "data" in completed_steps) or sp >= 42
        data_active = sp < 42 and not data_done
        steps_html += _step_row(data_step, data_done, data_active)

        # --- Parallel agents group ---
        par_label_color = _SLATE_600 if in_agent_phase else ("#059669" if agent_phase_done else "#9ca3af")
        par_label_weight = "600" if in_agent_phase else "400"
        par_completed_count = sum(1 for s in par_steps if completed_steps and s["key"] in completed_steps)
        par_total = len(par_steps)
        par_counter = f" ({par_completed_count}/{par_total})" if in_agent_phase or (completed_steps and par_completed_count > 0 and not agent_phase_done) else ""

        steps_html += (
            f'<div style="margin:8px 0 4px 0;padding:6px 8px;border-radius:8px;'
            f'background:{"#f8faff" if in_agent_phase else "transparent"};'
            f'border:1px solid {"#dce4f0" if in_agent_phase else "transparent"};'
            f'transition:all 0.3s ease">'
            f'<div style="font-size:11px;color:{par_label_color};font-weight:{par_label_weight};'
            f'margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em">'
            f'⇉ Running in parallel{par_counter}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px 16px">'
        )
        for pstep in par_steps:
            p_done = completed_steps is not None and pstep["key"] in completed_steps
            p_active = in_agent_phase and not p_done and not agent_phase_done
            steps_html += _step_row(pstep, p_done, p_active)
        steps_html += '</div></div>'

        # --- Blend step ---
        blend_step = seq_steps[1]  # blend
        blend_done = (completed_steps and "blend" in completed_steps) or sp >= 100 or bar_pct >= 100
        blend_active = sp >= 98 and not blend_done
        steps_html += _step_row(blend_step, blend_done, blend_active)

        # --- Card assembly ---
        slot.markdown(
            f'<style>'
            f'@keyframes _prog_pulse{{0%,100%{{opacity:.3;transform:scale(.8)}}50%{{opacity:1;transform:scale(1.1)}}}}'
            f'@keyframes _prog_bar{{from{{background-position:0 0}}to{{background-position:30px 0}}}}'
            f'</style>'
            f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;'
            f'padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.06);margin:10px 0;'
            f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif">'

            # Header row
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'margin-bottom:14px">'
            f'<span style="font-size:15px;font-weight:700;color:#111827;letter-spacing:-0.01em">'
            f'Analyzing{_progress_ticker_label}</span>'
            f'<span style="font-size:13px;color:#6b7280;font-weight:500;'
            f'font-variant-numeric:tabular-nums">{time_label}</span>'
            f'</div>'

            # Progress bar (with animated stripe when active)
            f'<div style="width:100%;background:#f3f4f6;border-radius:99px;'
            f'overflow:hidden;height:5px;margin-bottom:16px">'
            f'<div style="width:{bar_pct:.1f}%;height:100%;border-radius:99px;'
            f'background-color:{_SLATE_600};'
            f'{"background-image:linear-gradient(-45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent);background-size:30px 30px;animation:_prog_bar .8s linear infinite;" if 0 < bar_pct < 100 else ""}'
            f'{"background-color:#10b981;" if bar_pct >= 100 else ""}'
            f'transition:width 0.15s linear"></div></div>'

            # Step layout
            f'<div style="display:flex;flex-direction:column;gap:2px;'
            f'margin-bottom:14px">{steps_html}</div>'

            # Status message
            f'<div style="font-size:14px;color:#6b7280;border-top:1px solid #f0f1f3;'
            f'padding-top:10px;line-height:1.5">{clean_msg}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    def _log_phase_times(phase_times):
        """Append measured phase/step durations to data/step_times.json for future calibration.

        Accepts both legacy 3-phase keys (data_gather, agents, blend, total)
        and per-step keys (fundamentals, price_history, benchmark,
        value_agent, growth_momentum_agent, macro_regime_agent,
        risk_agent, sentiment_agent, agents_wall).
        """
        try:
            import json as _json
            _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'step_times.json')
            if os.path.exists(_path):
                with open(_path, 'r') as f:
                    store = _json.load(f)
            else:
                store = {"step_times": {}, "metadata": {}}

            st_data = store.setdefault("step_times", {})

            # Legacy phase keys (kept for backward compatibility)
            if phase_times.get('data_gather') is not None:
                st_data.setdefault("1", []).append(round(phase_times['data_gather'], 3))
            if phase_times.get('agents') is not None:
                st_data.setdefault("2", []).append(round(phase_times['agents'], 3))
            if phase_times.get('blend') is not None:
                st_data.setdefault("3", []).append(round(phase_times['blend'], 3))
            if phase_times.get('total') is not None:
                st_data.setdefault("total", []).append(round(phase_times['total'], 3))

            # Per-step keys (new granular timing)
            per_step_keys = [
                'fundamentals', 'price_history', 'benchmark',
                'value_agent', 'growth_momentum_agent',
                'macro_regime_agent', 'risk_agent', 'sentiment_agent',
                'agents_wall',
            ]
            for key in per_step_keys:
                if phase_times.get(key) is not None:
                    st_data.setdefault(key, []).append(round(phase_times[key], 3))

            # Cap each list at 100 most recent entries
            for k in st_data:
                if len(st_data[k]) > 100:
                    st_data[k] = st_data[k][-100:]

            store["metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "total_samples": sum(len(v) for v in st_data.values()),
            }

            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, 'w') as f:
                _json.dump(store, f, indent=2)

            # Reload learned phases so the NEXT analysis benefits immediately
            from engine.portfolio_orchestrator import _load_learned_phase_durations
            PortfolioOrchestrator._learned_phases = _load_learned_phase_durations()
        except Exception:
            pass  # timing log is best-effort, never block analysis

    def _run_with_smooth_progress(slot, orchestrator, tick, date_s, weights=None,
                                  regime_modulation=False, regime_sensitivity="moderate"):
        """Run analysis with a simple linear countdown timer.

        Timer starts at the learned estimated total and decrements in real-time.
        At phase boundaries (data complete / agents complete) the countdown
        snaps to match estimated remaining time.  After analysis
        completes, measured phase times are appended to step_times.json
        so future runs start with better estimates.
        """

        _prog = {
            'mile_pct': 0.0,
            'mile_msg': 'Initializing...',
            'done': False,
            'result': None,
            'error': None,
        }

        # Track which agent steps have completed (for parallel execution)
        _completed_steps = set()

        # Map of keywords to detect agent completions from orchestrator messages
        _STEP_COMPLETE_MAP = {
            'value':     ['value agent', 'value complete'],
            'growth':    ['growth agent', 'growth complete'],
            'macro':     ['macro regime agent', 'macro complete'],
            'risk':      ['risk agent', 'risk complete'],
            'sentiment': ['sentiment agent', 'articles analyzed'],
            'blend':     ['blending', 'analysis complete'],
        }

        # Phase transition timestamps (for logging & step rendering)
        _phase_ts = {
            'start': None,
            'data_done': None,   # milestone pct crosses 42
            'agents_done': None, # milestone pct crosses 98
            'end': None,         # milestone pct crosses 100
        }

        # --- Per-step expected timing (for rate-based recalibration) ---
        _lp = orchestrator._learned_phases if hasattr(orchestrator, '_learned_phases') else {}
        _est_data_wall  = _lp.get('data_gather', 10.0)
        # Use 75th-percentile of actual parallel wall time (bottleneck agent).
        # This is more accurate than max(individual medians) because agent
        # durations are highly variable and the bottleneck is what matters.
        _est_agents_wall = _lp.get('agents_wall_p75', _lp.get('agents', 16.0))
        _est_blend = _lp.get('blend', 0.1)

        # Actual step-completion timestamps (elapsed seconds from start)
        _step_done_at = {}

        # Map from orchestrator message keywords → step key
        _DATA_STEP_KEYWORDS = {
            'fundamentals': 'fundamentals',
            'price history': 'price_history',
            'benchmark': 'benchmark',
        }
        _AGENT_STEP_KEYWORDS = {
            'value': 'value_agent',
            'growth': 'growth_momentum_agent',
            'macro regime': 'macro_regime_agent',
            'macro': 'macro_regime_agent',
            'risk': 'risk_agent',
            'sentiment': 'sentiment_agent',
        }

        def _on_milestone(pct, message):
            _prog['mile_pct'] = pct
            _prog['mile_msg'] = message
            now_ts = time.time()
            elapsed = now_ts - _phase_ts['start'] if _phase_ts['start'] else 0

            # Phase transition timestamps
            if pct >= 42 and _phase_ts['data_done'] is None:
                _phase_ts['data_done'] = now_ts
            if pct >= 98 and _phase_ts['agents_done'] is None:
                _phase_ts['agents_done'] = now_ts
            if pct >= 100 and _phase_ts['end'] is None:
                _phase_ts['end'] = now_ts

            msg_lower = message.lower()

            # --- Detect data sub-step completions ---
            if 'received' in msg_lower:
                for kw, step_key in _DATA_STEP_KEYWORDS.items():
                    if kw in msg_lower and step_key not in _step_done_at:
                        _step_done_at[step_key] = elapsed
                        break

            # Mark all data sub-steps done when data phase completes
            if pct >= 42:
                for k in ('fundamentals', 'price_history', 'benchmark'):
                    if k not in _step_done_at:
                        _step_done_at[k] = elapsed
                _completed_steps.add('data')

            # --- Detect agent completions ---
            if any(w in msg_lower for w in ('complete', 'analyzed', 'score', 'failed')):
                for kw, step_key in _AGENT_STEP_KEYWORDS.items():
                    if kw in msg_lower and step_key not in _step_done_at:
                        _step_done_at[step_key] = elapsed
                        break
                # UI step tracking
                for sk, kws in _STEP_COMPLETE_MAP.items():
                    if any(kw in msg_lower for kw in kws):
                        _completed_steps.add(sk)

            # Mark all agents done when agent phase completes
            if pct >= 98:
                for k in ('value_agent', 'growth_momentum_agent',
                          'macro_regime_agent', 'risk_agent', 'sentiment_agent'):
                    if k not in _step_done_at:
                        _step_done_at[k] = elapsed

            if pct >= 100:
                _completed_steps.add('blend')
                if 'blend' not in _step_done_at:
                    _step_done_at['blend'] = elapsed

        def _bg():
            try:
                _prog['result'] = orchestrator.analyze_stock(
                    ticker=tick,
                    analysis_date=date_s,
                    agent_weights=weights,
                    progress_callback=_on_milestone,
                    regime_modulation=regime_modulation,
                    regime_sensitivity=regime_sensitivity,
                )
            except Exception as e:
                _prog['error'] = e
            finally:
                _prog['done'] = True

        thread = threading.Thread(target=_bg, daemon=True)
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            add_script_run_ctx(thread, get_script_run_ctx())
        except Exception:
            pass
        thread.start()

        # ─── Simple linear countdown timer ───
        # Starts from the learned average total time (recent 5 runs).
        # At phase-boundary milestones, the countdown is snapped
        # to match the estimated remaining time so it lands near 0
        # when analysis completes.
        _avg_total = _lp.get('avg_total', _est_data_wall + _est_agents_wall + _est_blend + 2.0)
        _countdown = _avg_total
        _data_snapped = False
        _agents_snapped = False
        display_pct = 0.0
        last_render = 0.0
        last_tick = time.time()
        start_wall = time.time()
        _phase_ts['start'] = start_wall

        while not _prog['done']:
            now = time.time()
            dt = now - last_tick
            last_tick = now
            elapsed = now - start_wall

            msg = _prog['mile_msg']
            mp = _prog['mile_pct']

            # ── Simple linear countdown ──
            _countdown -= dt

            # ── One-time snap when data phase completes (mp >= 42) ──
            if mp >= 42 and not _data_snapped:
                _data_snapped = True
                agents_remaining = _est_agents_wall + _est_blend + 2.0
                if _countdown > agents_remaining + 3:
                    _countdown = agents_remaining + 2
                elif _countdown < agents_remaining * 0.3:
                    # Data was very slow — timer too low, bump it up gently
                    _countdown = agents_remaining * 0.6

            # ── One-time snap when agent phase completes (mp >= 98) ──
            if mp >= 98 and not _agents_snapped:
                _agents_snapped = True
                _countdown = min(_countdown, 2.0)

            # ── Floor ──
            _countdown = max(0.0, _countdown)

            display_remaining = _countdown

            # ── Progress bar percentage ──
            est_total = elapsed + max(display_remaining, 0.5)
            target_pct = min(99.0, (elapsed / est_total) * 100.0)

            if target_pct > display_pct:
                pct_gap = target_pct - display_pct
                lerp = min(0.8, 0.25 + pct_gap * 0.01)
                display_pct += pct_gap * lerp
            display_pct = max(0.0, min(99.0, display_pct))

            # Render at ~10 fps
            if now - last_render >= 0.10:
                if mp >= 42:
                    _completed_steps.add('data')

                _render_progress(slot, display_pct, msg,
                                 remaining_secs=display_remaining,
                                 step_pct=mp,
                                 completed_steps=_completed_steps if _completed_steps else None)
                last_render = now

            time.sleep(0.05)

        if _prog['error']:
            raise _prog['error']

        # --- Log phase times to step_times.json for future calibration ---
        # Use per-step timings from the orchestrator result if available
        result = _prog['result']
        step_timings = result.get('step_timings', {}) if isinstance(result, dict) else {}

        total_time = time.time() - start_wall
        phase_times = {'total': total_time}

        # Legacy phase times from wall-clock transitions
        if _phase_ts['data_done']:
            phase_times['data_gather'] = _phase_ts['data_done'] - start_wall
        if _phase_ts['data_done'] and _phase_ts['agents_done']:
            phase_times['agents'] = _phase_ts['agents_done'] - _phase_ts['data_done']
        if _phase_ts['agents_done'] and _phase_ts['end']:
            phase_times['blend'] = _phase_ts['end'] - _phase_ts['agents_done']

        # Merge in per-step timings from orchestrator (more accurate)
        for key in ('fundamentals', 'price_history', 'benchmark',
                    'value_agent', 'growth_momentum_agent',
                    'macro_regime_agent', 'risk_agent', 'sentiment_agent',
                    'agents_wall', 'blend'):
            if key in step_timings:
                phase_times[key] = step_timings[key]

        _log_phase_times(phase_times)

        return result

    # Use average total time from step_times.json for the initial countdown display
    _lp = st.session_state.orchestrator._learned_phases if hasattr(st.session_state, 'orchestrator') and hasattr(st.session_state.orchestrator, '_learned_phases') else {}
    _initial_est = _lp.get('avg_total', _lp.get('total', 30.0))

    _render_progress(progress_slot, 0, "Initializing analysis…",
                     remaining_secs=_initial_est)

    # Handle single or multiple stock analysis
    if analysis_mode == "Single Stock":
        try:
            _render_progress(progress_slot, 0, "Starting analysis…",
                             remaining_secs=_initial_est)

            start_time = time.time()

            # Convert analysis_date to string format
            if isinstance(analysis_date, (datetime, type(datetime.now().date()))):
                date_str = analysis_date.strftime('%Y-%m-%d') if hasattr(analysis_date, 'strftime') else str(analysis_date)
            elif isinstance(analysis_date, tuple) and len(analysis_date) > 0:
                date_str = analysis_date[0].strftime('%Y-%m-%d') if hasattr(analysis_date[0], 'strftime') else str(analysis_date[0])
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')

            # Quick ticker validation before running full analysis
            try:
                import yfinance as yf
                _fast = yf.Ticker(ticker).fast_info
                if not getattr(_fast, 'last_price', None):
                    progress_slot.empty()
                    st.error(f"**'{ticker}'** doesn't appear to be a valid ticker symbol. Please check and try again.")
                    return
            except Exception:
                pass  # If validation itself fails, let the analysis proceed normally

            # Run with smooth progress interpolation (background thread + 10fps polling)
            orchestrator = st.session_state.orchestrator
            result = _run_with_smooth_progress(
                progress_slot, orchestrator, ticker, date_str, agent_weights,
                regime_modulation=regime_modulation,
                regime_sensitivity=regime_sensitivity,
            )

            # Track timing
            end_time = time.time()
            analysis_duration = end_time - start_time
            st.session_state.analysis_times.append(analysis_duration)

            if len(st.session_state.analysis_times) > 50:
                st.session_state.analysis_times = st.session_state.analysis_times[-50:]

            actual_minutes = int(analysis_duration // 60)
            actual_seconds = int(analysis_duration % 60)
            _render_progress(progress_slot, 100,
                             f"Analysis complete — {actual_minutes}m {actual_seconds:02d}s",
                             remaining_secs=0)

            time.sleep(0.5)
            progress_slot.empty()

            if 'error' in result:
                if result['error'] == 'ticker_not_found':
                    _render_ticker_not_found(result.get('ticker', params.get('ticker', '')))
                else:
                    _failed_ticker = result.get('ticker', ticker)
                    st.error(f"**Analysis failed for {_failed_ticker}**")
                    with st.expander("Error details", expanded=True):
                        st.code(str(result['error']))
                        st.caption("Possible causes: API rate limit, network issue, or invalid data. Try again in a moment.")
                return

            # Store agent scores for delta tracking across analyses
            if '_score_history' not in st.session_state:
                st.session_state['_score_history'] = {}
            _result_ticker = result.get('ticker', ticker)
            # Move current stored scores to _prev_score_history before overwriting
            if '_prev_score_history' not in st.session_state:
                st.session_state['_prev_score_history'] = {}
            if _result_ticker in st.session_state['_score_history']:
                st.session_state['_prev_score_history'][_result_ticker] = st.session_state['_score_history'][_result_ticker]
            st.session_state['_score_history'][_result_ticker] = result.get('agent_scores', {})

            # Persist results so page survives widget-triggered reruns
            _display_data = {'mode': 'single', 'result': result}
            st.session_state['_display_result'] = _display_data

            # Append to analysis history (most recent first, max 5)
            _analysis_history = st.session_state.setdefault('_analysis_history', [])
            _analysis_history.insert(0, {
                'ticker': _result_ticker,
                'blended_score': result.get('blended_score', result.get('final_score', 0)),
                'timestamp': datetime.now().strftime('%H:%M'),
                'display_data': _display_data,
            })
            # Remove older duplicate of same ticker
            _seen = set()
            _deduped = []
            for _ah in _analysis_history:
                if _ah['ticker'] not in _seen:
                    _seen.add(_ah['ticker'])
                    _deduped.append(_ah)
            st.session_state['_analysis_history'] = _deduped[:5]

            # Display results
            display_stock_analysis(result)

        except Exception as e:
            # Clear progress indicator on error
            progress_slot.empty()
            st.error(f"Analysis failed: {e}")
    
    else:
        # Multiple stocks analysis — use the same progress card as single stock
        # but with an outer wrapper showing overall batch progress.

        # Clear the outer progress_slot rendered before the mode check
        progress_slot.empty()

        # Validate all tickers upfront before running any analysis
        _invalid_tickers = []
        _valid_tickers = []
        for _t in tickers:
            try:
                import yfinance as yf
                _fi = yf.Ticker(_t).fast_info
                if not getattr(_fi, 'last_price', None):
                    _invalid_tickers.append(_t)
                else:
                    _valid_tickers.append(_t)
            except Exception:
                _valid_tickers.append(_t)  # Let analysis decide if validation fails

        if _invalid_tickers:
            st.warning(f"Skipping invalid ticker(s): {', '.join(f'**{t}**' for t in _invalid_tickers)}")

        tickers = _valid_tickers
        if not tickers:
            progress_slot.empty()
            st.error("No valid tickers to analyze.")
            return

        results = []
        failed_tickers = []
        batch_start_time = time.time()

        # Per-stock time estimate from history
        if st.session_state.analysis_times:
            avg_time_per_stock = sum(st.session_state.analysis_times) / len(st.session_state.analysis_times)
        else:
            avg_time_per_stock = _lp.get('avg_total', 30.0)

        total_stocks = len(tickers)

        # Create slots ONCE outside the loop so no extra placeholders accumulate
        _batch_header_slot = st.empty()
        _stock_detail_slot = st.empty()

        for idx, stock_ticker in enumerate(tickers):
            stock_start_time = time.time()

            # Compute overall batch remaining estimate
            completed_count = idx
            if completed_count > 0:
                elapsed_batch = time.time() - batch_start_time
                per_stock_actual = elapsed_batch / completed_count
                remaining_batch = per_stock_actual * (total_stocks - idx)
            else:
                remaining_batch = avg_time_per_stock * total_stocks

            # Update the progress card header to show batch context
            _progress_ticker_label_stock = f' — {stock_ticker} ({idx + 1}/{total_stocks})'

            # Override the outer _progress_ticker_label for this stock
            # We need a small wrapper that patches the header for multi-stock
            _saved_label = _progress_ticker_label

            def _render_multi_progress(slot, bar_pct, message, remaining_secs=None, step_pct=None, completed_steps=None):
                """Wrapper that renders single-stock progress card + batch header."""
                import re as _re

                # Build a batch header above the card
                _completed = idx
                _elapsed = time.time() - batch_start_time
                if _completed > 0:
                    _per_stock = _elapsed / _completed
                    _rem = _per_stock * (total_stocks - idx)
                    if bar_pct >= 100:
                        _rem = _per_stock * (total_stocks - idx - 1)
                else:
                    _rem = avg_time_per_stock * (total_stocks - idx)

                _rm = int(_rem // 60)
                _rs = int(_rem % 60)
                _batch_time_str = f"{_rm}m {_rs:02d}s" if _rm > 0 else f"{_rs}s"

                # Completed stocks mini-badges
                _badges = ""
                for _ci, _ct in enumerate(tickers[:idx]):
                    _was_ok = any(r.get('ticker') == _ct for r in results)
                    _bc = "#10b981" if _was_ok else "#ef4444"
                    _badges += (f'<span style="display:inline-block;padding:2px 8px;'
                                f'border-radius:4px;font-size:11px;font-weight:600;'
                                f'background:{_bc}22;color:{_bc};margin-right:4px">{_ct}</span>')

                # Pending stocks
                for _pi, _pt in enumerate(tickers[idx+1:]):
                    _badges += (f'<span style="display:inline-block;padding:2px 8px;'
                                f'border-radius:4px;font-size:11px;font-weight:500;'
                                f'background:#f3f4f6;color:#9ca3af;margin-right:4px">{_pt}</span>')

                # Overall progress fraction
                _overall_pct = (idx / total_stocks) * 100
                if bar_pct >= 100:
                    _overall_pct = ((idx + 1) / total_stocks) * 100

                # Render batch header
                slot.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;'
                    f'padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.06);margin:10px 0;'
                    f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif">'

                    # Batch header
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'margin-bottom:6px">'
                    f'<span style="font-size:15px;font-weight:700;color:#111827;letter-spacing:-0.01em">'
                    f'Analyzing {stock_ticker} ({idx + 1}/{total_stocks})</span>'
                    f'<span style="font-size:13px;color:#6b7280;font-weight:500;'
                    f'font-variant-numeric:tabular-nums">{_batch_time_str} remaining</span>'
                    f'</div>'

                    # Stock badges
                    f'<div style="margin-bottom:12px;line-height:1.8">{_badges}</div>'

                    # Overall batch progress bar
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">'
                    f'<span style="font-size:11px;color:#9ca3af;white-space:nowrap">Overall</span>'
                    f'<div style="flex:1;background:#f3f4f6;border-radius:99px;'
                    f'overflow:hidden;height:4px">'
                    f'<div style="width:{_overall_pct:.1f}%;height:100%;border-radius:99px;'
                    f'background:{"#10b981" if _overall_pct >= 100 else "#3b5998"};'
                    f'transition:width 0.3s ease"></div></div>'
                    f'<span style="font-size:11px;color:#6b7280;font-weight:500;'
                    f'font-variant-numeric:tabular-nums">{int(_overall_pct)}%</span>'
                    f'</div>'

                    f'<div style="border-top:1px solid #f0f1f3;padding-top:12px"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Now render the per-stock progress card using the original _render_progress
                # but into a second slot below the batch header
                _render_progress(
                    _stock_detail_slot, bar_pct, message,
                    remaining_secs=remaining_secs,
                    step_pct=step_pct,
                    completed_steps=completed_steps
                )

            # Initial render
            _render_multi_progress(_batch_header_slot, 0, "Initializing analysis…",
                                   remaining_secs=avg_time_per_stock)

            try:
                # Convert analysis_date to string format
                if isinstance(analysis_date, (datetime, type(datetime.now().date()))):
                    date_str = analysis_date.strftime('%Y-%m-%d') if hasattr(analysis_date, 'strftime') else str(analysis_date)
                elif isinstance(analysis_date, tuple) and len(analysis_date) > 0:
                    date_str = analysis_date[0].strftime('%Y-%m-%d') if hasattr(analysis_date[0], 'strftime') else str(analysis_date[0])
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')

                # Run with smooth progress — dynamic phase-aware timer

                _prog = {
                    'mile_pct': 0.0,
                    'mile_msg': 'Initializing...',
                    'done': False,
                    'result': None,
                    'error': None,
                }
                _completed_steps_m = set()

                _STEP_COMPLETE_MAP_M = {
                    'value':     ['value agent', 'value complete'],
                    'growth':    ['growth agent', 'growth complete'],
                    'macro':     ['macro regime agent', 'macro complete'],
                    'risk':      ['risk agent', 'risk complete'],
                    'sentiment': ['sentiment agent', 'articles analyzed'],
                    'blend':     ['blending', 'analysis complete'],
                }

                # Phase transition timestamps for dynamic adjustment & logging
                _phase_ts_m = {
                    'start': None, 'data_done': None,
                    'agents_done': None, 'end': None,
                }

                # --- Per-step expected timing (multi-point recalibration) ---
                _lp_m = st.session_state.orchestrator._learned_phases if hasattr(st.session_state.orchestrator, '_learned_phases') else {}
                _est_data_wall_m = _lp_m.get('data_gather', 10.0)
                _est_agents_wall_m = _lp_m.get('agents_wall_p75', _lp_m.get('agents', 16.0))
                _est_blend_m = _lp_m.get('blend', 0.1)
                _step_done_at_m = {}

                _DATA_KW_M = {'fundamentals': 'fundamentals', 'price history': 'price_history', 'benchmark': 'benchmark'}
                _AGENT_KW_M = {'value': 'value_agent', 'growth': 'growth_momentum_agent',
                               'macro regime': 'macro_regime_agent', 'macro': 'macro_regime_agent',
                               'risk': 'risk_agent', 'sentiment': 'sentiment_agent'}

                def _on_milestone_m(pct, message):
                    _prog['mile_pct'] = pct
                    _prog['mile_msg'] = message
                    now_ts = time.time()
                    elapsed = now_ts - _phase_ts_m['start'] if _phase_ts_m['start'] else 0
                    if pct >= 42 and _phase_ts_m['data_done'] is None:
                        _phase_ts_m['data_done'] = now_ts
                    if pct >= 98 and _phase_ts_m['agents_done'] is None:
                        _phase_ts_m['agents_done'] = now_ts
                    if pct >= 100 and _phase_ts_m['end'] is None:
                        _phase_ts_m['end'] = now_ts
                    msg_lower = message.lower()
                    if 'received' in msg_lower:
                        for kw, sk in _DATA_KW_M.items():
                            if kw in msg_lower and sk not in _step_done_at_m:
                                _step_done_at_m[sk] = elapsed
                                break
                    if pct >= 42:
                        for k in ('fundamentals', 'price_history', 'benchmark'):
                            if k not in _step_done_at_m:
                                _step_done_at_m[k] = elapsed
                        _completed_steps_m.add('data')
                    if any(w in msg_lower for w in ('complete', 'analyzed', 'score', 'failed')):
                        for kw, sk in _AGENT_KW_M.items():
                            if kw in msg_lower and sk not in _step_done_at_m:
                                _step_done_at_m[sk] = elapsed
                                break
                        for sk, kws in _STEP_COMPLETE_MAP_M.items():
                            if any(kw in msg_lower for kw in kws):
                                _completed_steps_m.add(sk)
                    if pct >= 98:
                        for k in ('value_agent', 'growth_momentum_agent',
                                  'macro_regime_agent', 'risk_agent', 'sentiment_agent'):
                            if k not in _step_done_at_m:
                                _step_done_at_m[k] = elapsed
                    if pct >= 100:
                        _completed_steps_m.add('blend')
                        if 'blend' not in _step_done_at_m:
                            _step_done_at_m['blend'] = elapsed

                def _bg_m():
                    try:
                        orchestrator = st.session_state.orchestrator
                        _prog['result'] = orchestrator.analyze_stock(
                            ticker=stock_ticker,
                            analysis_date=date_str,
                            agent_weights=agent_weights,
                            progress_callback=_on_milestone_m,
                            regime_modulation=regime_modulation,
                            regime_sensitivity=regime_sensitivity,
                        )
                    except Exception as e:
                        _prog['error'] = e
                    finally:
                        _prog['done'] = True

                thread = threading.Thread(target=_bg_m, daemon=True)
                try:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
                    add_script_run_ctx(thread, get_script_run_ctx())
                except Exception:
                    pass
                thread.start()

                # ─── Simple linear countdown timer (same as single-stock) ───
                _countdown_m = _est_data_wall_m + _est_agents_wall_m + _est_blend_m + 2.0
                _data_snapped_m = False
                _agents_snapped_m = False
                display_pct_m = 0.0
                last_render_m = 0.0
                last_tick_m = time.time()
                start_wall_m = time.time()
                _phase_ts_m['start'] = start_wall_m

                while not _prog['done']:
                    now = time.time()
                    dt = now - last_tick_m
                    last_tick_m = now
                    elapsed = now - start_wall_m
                    msg = _prog['mile_msg']
                    mp = _prog['mile_pct']

                    # ── Simple linear countdown ──
                    _countdown_m -= dt

                    # ── One-time snap when data phase completes ──
                    if mp >= 42 and not _data_snapped_m:
                        _data_snapped_m = True
                        agents_remaining = _est_agents_wall_m + _est_blend_m + 2.0
                        if _countdown_m > agents_remaining + 3:
                            _countdown_m = agents_remaining + 2
                        elif _countdown_m < agents_remaining * 0.3:
                            _countdown_m = agents_remaining * 0.6

                    # ── One-time snap when agent phase completes ──
                    if mp >= 98 and not _agents_snapped_m:
                        _agents_snapped_m = True
                        _countdown_m = min(_countdown_m, 2.0)

                    # ── Floor ──
                    _countdown_m = max(0.0, _countdown_m)

                    display_remaining_m = _countdown_m

                    # ── Progress bar percentage ──
                    est_total_m = elapsed + max(display_remaining_m, 0.5)
                    target_pct_m = min(99.0, (elapsed / est_total_m) * 100.0)

                    if target_pct_m > display_pct_m:
                        pct_gap_m = target_pct_m - display_pct_m
                        lerp_m = min(0.8, 0.25 + pct_gap_m * 0.01)
                        display_pct_m += pct_gap_m * lerp_m
                    display_pct_m = max(0.0, min(99.0, display_pct_m))

                    if now - last_render_m >= 0.10:
                        if mp >= 42:
                            _completed_steps_m.add('data')

                        _render_multi_progress(
                            _batch_header_slot, display_pct_m, msg,
                            remaining_secs=display_remaining_m,
                            step_pct=mp,
                            completed_steps=_completed_steps_m if _completed_steps_m else None
                        )
                        last_render_m = now

                    time.sleep(0.05)

                if _prog['error']:
                    raise _prog['error']

                result = _prog['result']

                # Log phase times for this stock (with per-step data)
                step_timings_m = result.get('step_timings', {}) if isinstance(result, dict) else {}
                _total_m = time.time() - start_wall_m
                _pt_m = {'total': _total_m}
                if _phase_ts_m['data_done']:
                    _pt_m['data_gather'] = _phase_ts_m['data_done'] - start_wall_m
                if _phase_ts_m['data_done'] and _phase_ts_m['agents_done']:
                    _pt_m['agents'] = _phase_ts_m['agents_done'] - _phase_ts_m['data_done']
                if _phase_ts_m['agents_done'] and _phase_ts_m['end']:
                    _pt_m['blend'] = _phase_ts_m['end'] - _phase_ts_m['agents_done']
                # Merge per-step timings from orchestrator
                for key in ('fundamentals', 'price_history', 'benchmark',
                            'value_agent', 'growth_momentum_agent',
                            'macro_regime_agent', 'risk_agent', 'sentiment_agent',
                            'agents_wall', 'blend'):
                    if key in step_timings_m:
                        _pt_m[key] = step_timings_m[key]
                _log_phase_times(_pt_m)

                # Track time for this stock
                stock_duration = time.time() - stock_start_time
                st.session_state.analysis_times.append(stock_duration)
                if len(st.session_state.analysis_times) > 50:
                    st.session_state.analysis_times = st.session_state.analysis_times[-50:]

                # Update per-stock estimate for next stock
                avg_time_per_stock = sum(st.session_state.analysis_times[-5:]) / len(st.session_state.analysis_times[-5:])

                # Show completion briefly
                actual_m = int(stock_duration // 60)
                actual_s = int(stock_duration % 60)
                _render_multi_progress(
                    _batch_header_slot, 100,
                    f"{stock_ticker} complete — {actual_m}m {actual_s:02d}s",
                    remaining_secs=0
                )
                time.sleep(0.3)

                # Clear this stock's slots
                _batch_header_slot.empty()
                _stock_detail_slot.empty()

                if 'error' in result:
                    failed_tickers.append((stock_ticker, result['error']))
                else:
                    results.append(result)

            except Exception as e:
                _batch_header_slot.empty()
                _stock_detail_slot.empty()
                failed_tickers.append((stock_ticker, str(e)))

        # Final batch summary
        batch_duration = time.time() - batch_start_time
        bm = int(batch_duration // 60)
        bs = int(batch_duration % 60)
        st.success(f"Batch complete — {len(results)}/{total_stocks} stocks analyzed in {bm}m {bs:02d}s")
        if failed_tickers:
            for ft, fe in failed_tickers:
                if fe == 'ticker_not_found':
                    _render_ticker_not_found(ft)
                else:
                    st.error(f"**{ft}**: {fe}")
        time.sleep(0.5)
        
        # Display results summary
        if results:
            # Persist results so page survives widget-triggered reruns
            _display_data = {
                'mode': 'multi', 'results': results, 'failed': failed_tickers,
            }
            st.session_state['_display_result'] = _display_data

            # Append each successful result to analysis history
            _analysis_history = st.session_state.setdefault('_analysis_history', [])
            for _mr in results:
                _mt = _mr.get('ticker', '?')
                _analysis_history.insert(0, {
                    'ticker': _mt,
                    'blended_score': _mr.get('blended_score', _mr.get('final_score', 0)),
                    'timestamp': datetime.now().strftime('%H:%M'),
                    'display_data': {'mode': 'single', 'result': _mr},
                })
            # Deduplicate (keep most recent per ticker) and cap at 5
            _seen = set()
            _deduped = []
            for _ah in _analysis_history:
                if _ah['ticker'] not in _seen:
                    _seen.add(_ah['ticker'])
                    _deduped.append(_ah)
            st.session_state['_analysis_history'] = _deduped[:5]

            display_multiple_stock_analysis(results, failed_tickers)
        else:
            st.error("All analyses failed!")
            for ticker_name, error_msg in failed_tickers:
                st.error(f"**{ticker_name}**: {error_msg}")


def stock_analysis_page():
    """Single or multiple stock analysis page."""
    import threading, math

    st.markdown("---")

    # Analysis mode selection
    analysis_mode = st.radio(
        "Analysis Mode",
        options=["Single Stock", "Multiple Stocks"],
        horizontal=True,
        help="Choose to analyze one stock or multiple stocks at once"
    )

    if analysis_mode == "Multiple Stocks":
        st.caption("**Note:** Multi-stock analysis takes longer since each stock is analyzed individually. "
                   "Consider starting with a single stock first to test your configuration.")
    
    if analysis_mode == "Single Stock":
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
    else:
        ticker_input = st.text_area(
            "Stock Ticker Symbols",
            value="AAPL MSFT GOOGL",
            height=100,
            help="Enter multiple ticker symbols separated by spaces, commas, or line breaks (e.g., AAPL MSFT GOOGL or AAPL, MSFT, GOOGL)"
        )
        # Parse tickers - handle spaces, commas, and newlines, remove duplicates
        import re
        ticker_list = [t.strip().upper() for t in re.split(r'[,\s\n]+', ticker_input) if t.strip()]
        # Remove duplicates while preserving order
        seen = set()
        tickers = []
        for t in ticker_list:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
        ticker = None  # Not used in multi mode

    # Always use today's date
    analysis_date = datetime.now()
    
    # Weight preset
    st.markdown("### Agent Weights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight_preset = st.selectbox(
            "Choose Weight Configuration:",
            options=["equal_weights", "theory_based", "custom_weights"],
            format_func=lambda x: {
                "equal_weights": "Equal Weights",
                "theory_based": "Theory Based",
                "custom_weights": "Custom Weights",
            }[x],
            help="Select how agent weights should be configured for this analysis"
        )
        
        # Store weight preset in session state for use in display functions
        st.session_state.weight_preset = weight_preset
    
    # Initialize session state for custom weights
    if 'custom_agent_weights' not in st.session_state:
        st.session_state.custom_agent_weights = {
            'value': 0.5,
            'growth_momentum': 0.5,
            'macro_regime': 0.5,
            'risk': 0.5,
            'sentiment': 0.5
        }
    
    # Handle weight preset selection
    agent_weights = None
    if weight_preset == "custom_weights":
        st.caption("Adjust sliders to set each agent's influence on the final score. Higher = more influence.")

        # Slider track color is handled via global CSS (see st.markdown style block at top)

        @st.fragment
        def _weight_slider_fragment():
            weight_cols = st.columns(5)
            agents = ['value', 'growth_momentum', 'macro_regime', 'risk', 'sentiment']
            agent_labels = ['Value', 'Growth', 'Macro Regime', 'Risk', 'Sentiment']
            agent_tips = {
                'value':           'P/E, P/B, DCF intrinsic-value metrics',
                'growth_momentum': 'Revenue growth, earnings trends, price momentum',
                'macro_regime':    'Interest rates, inflation, economic cycle',
                'risk':            'Volatility, drawdown, debt-level risk',
                'sentiment':       'News tone, analyst ratings, social buzz',
            }
            for i, (agent, label) in enumerate(zip(agents, agent_labels)):
                slider_key = f"custom_weight_{agent}"
                # Initialize the widget key from our weights dict only on first run
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = min(st.session_state.custom_agent_weights[agent], 1.0)
                with weight_cols[i]:
                    st.slider(
                        label,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.05,
                        key=slider_key,
                        help=agent_tips[agent]
                    )
                # Sync back from widget key to our weights dict
                st.session_state.custom_agent_weights[agent] = st.session_state[slider_key]
            # Show current weight distribution
            total_weight = sum(st.session_state.custom_agent_weights.values())
            percentages = {k: (v/total_weight)*100 for k, v in st.session_state.custom_agent_weights.items()}
            dist_cols = st.columns(5)
            for i, (agent, pct) in enumerate(percentages.items()):
                with dist_cols[i]:
                    st.metric(agent_labels[i], f"{pct:.1f}%", help=agent_tips[agents[i]])

        _weight_slider_fragment()

        agent_weights = st.session_state.custom_agent_weights.copy()
        st.session_state.locked_custom_weights = agent_weights
        st.session_state.regime_modulation_enabled = False

    elif weight_preset == "theory_based":
        # ── Theory Based preset ────────────────────────────────────────────────────────────
        # Weights grounded in multi-factor asset pricing & behavioral finance
        st.info(
            "**Theory Based Weighting**\n\n"
            "Weights are grounded in multi-factor asset pricing theory "
            "(Fama & French, 1992; Carhart, 1997) and behavioral finance "
            "research (Shiller, 2000; Thaler, 2015). Value and momentum "
            "receive the largest allocations as compensated structural risk "
            "premia with decades of cross-market evidence. Growth quality "
            "captures profitability and reinvestment efficiency. Sentiment "
            "serves as a constrained behavioral overlay.\n\n"
            "Weights shift automatically based on the detected macroeconomic "
            "regime \u2014 expansion, recession, inflation, or disinflation \u2014 "
            "following regime-switching research (Ang & Bekaert, 2002). "
            "Your selected investment horizon tilts the value\u2013momentum "
            "balance per empirical holding-period evidence: value strengthens "
            "over longer horizons, momentum over shorter ones."
        )

        # Configurable sub-settings
        theory_cols = st.columns(3)
        with theory_cols[0]:
            theory_horizon = st.selectbox(
                "Investment Horizon",
                options=["short", "medium", "long"],
                index=1,
                format_func=lambda x: {
                    "short":  "Short-term (3\u20136 months)",
                    "medium": "Medium-term (6\u201318 months)",
                    "long":   "Long-term (18+ months)",
                }[x],
                help=(
                    "Jegadeesh & Titman (1993): momentum is strongest at 6\u201312 months. "
                    "Fama & French (1992): value premia strengthen beyond 18 months."
                ),
            )
        with theory_cols[1]:
            theory_risk_fw = st.selectbox(
                "Risk Framework",
                options=["capital", "risk_contribution"],
                index=0,
                format_func=lambda x: {
                    "capital":           "Capital-weighted",
                    "risk_contribution": "Risk-contribution-weighted",
                }[x],
                help=(
                    "Capital-weighted: allocate by conviction. "
                    "Risk-contribution (Maillard et al., 2010): adjusts for "
                    "momentum\u2019s higher volatility so each factor contributes "
                    "similar portfolio risk."
                ),
            )
        with theory_cols[2]:
            theory_regime_sens = st.selectbox(
                "Regime Sensitivity",
                options=["conservative", "moderate", "aggressive"],
                index=1,
                format_func=lambda x: {
                    "conservative": "Conservative (\u00b15pp)",
                    "moderate":     "Moderate (\u00b110pp)",
                    "aggressive":   "Aggressive (\u00b115pp)",
                }[x],
                help=(
                    "How strongly macro regime shifts tilt the weights. "
                    "Based on Ang & Bekaert (2002) and Ilmanen (2011)."
                ),
            )

        # Weight lookup table: (horizon, risk_framework) -> weights
        _THEORY_WEIGHTS = {
            ("long",   "capital"):           {"value": 0.45, "growth_momentum": 0.30, "macro_regime": 0.05, "risk": 0.05, "sentiment": 0.15},
            ("long",   "risk_contribution"): {"value": 0.40, "growth_momentum": 0.30, "macro_regime": 0.05, "risk": 0.10, "sentiment": 0.15},
            ("medium", "capital"):           {"value": 0.35, "growth_momentum": 0.40, "macro_regime": 0.05, "risk": 0.05, "sentiment": 0.15},
            ("medium", "risk_contribution"): {"value": 0.30, "growth_momentum": 0.35, "macro_regime": 0.05, "risk": 0.10, "sentiment": 0.20},
            ("short",  "capital"):           {"value": 0.25, "growth_momentum": 0.45, "macro_regime": 0.05, "risk": 0.05, "sentiment": 0.20},
            ("short",  "risk_contribution"): {"value": 0.20, "growth_momentum": 0.40, "macro_regime": 0.05, "risk": 0.10, "sentiment": 0.25},
        }

        agent_weights = _THEORY_WEIGHTS[(theory_horizon, theory_risk_fw)]

        # Display the base allocation
        st.markdown("**Base Factor Allocation** *(before regime adjustment)*")
        tw_total = sum(agent_weights.values())
        tw_cols = st.columns(5)
        tw_labels = ["Value", "Growth / Mom.", "Macro Regime", "Risk", "Sentiment"]
        for i, (agent_key, label) in enumerate(zip(
            ["value", "growth_momentum", "macro_regime", "risk", "sentiment"], tw_labels
        )):
            with tw_cols[i]:
                pct = (agent_weights[agent_key] / tw_total) * 100
                st.metric(label, f"{pct:.0f}%")

        # Store theory settings for downstream use
        st.session_state.theory_settings = {
            "horizon": theory_horizon,
            "risk_framework": theory_risk_fw,
            "regime_sensitivity": theory_regime_sens,
        }
        st.session_state.regime_modulation_enabled = True
        st.session_state.locked_theory_weights = agent_weights.copy()

        # Clear custom-weight state so display doesn't confuse the two
        if 'locked_custom_weights' in st.session_state:
            del st.session_state['locked_custom_weights']

    else:  # equal_weights
        # Always use truly equal weights (1.0 each), overriding orchestrator defaults
        agent_weights = {
            'value': 1.0,
            'growth_momentum': 1.0,
            'macro_regime': 1.0,
            'risk': 1.0,
            'sentiment': 1.0
        }
        # Clear any previously locked custom weights so the display is consistent
        if 'locked_custom_weights' in st.session_state:
            del st.session_state['locked_custom_weights']
        st.session_state.regime_modulation_enabled = False
    
    st.markdown("---")
    
    if st.button("Run Analysis", type="primary"):
        # Validation
        import re as _re_val
        if analysis_mode == "Single Stock":
            if not ticker:
                st.error("Please enter a ticker symbol.")
                return
            if not _re_val.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', ticker):
                st.error(f"**Invalid ticker: '{ticker}'** — Ticker symbols should be 1-5 letters "
                         f"(e.g., AAPL, MSFT, BRK.B). Please check your input and try again.")
                return
        else:
            if not tickers:
                st.error("Please enter at least one ticker symbol.")
                return
            invalid_tickers = [t for t in tickers if not _re_val.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', t)]
            if invalid_tickers:
                st.error(f"**Invalid ticker(s): {', '.join(invalid_tickers)}** — "
                         f"Ticker symbols should be 1-5 letters (e.g., AAPL, MSFT, BRK.B). "
                         f"Please check your input and try again.")
                return

        # Store analysis parameters and rerun to hide form
        st.session_state['_analysis_params'] = {
            'mode': analysis_mode,
            'ticker': ticker if analysis_mode == "Single Stock" else None,
            'tickers': tickers if analysis_mode == "Multiple Stocks" else None,
            'date': analysis_date,
            'weights': agent_weights,
            'regime_modulation': st.session_state.get('regime_modulation_enabled', False),
            'regime_sensitivity': (st.session_state.get('theory_settings') or {}).get('regime_sensitivity', 'moderate'),
        }
        st.rerun()


# # ---------------------------------------------------------------------------
# # Google Sheets / Docs export helpers  (OAuth 2.0 via authlib)
# # ---------------------------------------------------------------------------
# from authlib.integrations.requests_client import OAuth2Session as _OAuth2Session
# 
# _GOOGLE_SCOPES = [
#     'openid',
#     'https://www.googleapis.com/auth/userinfo.email',
#     'https://www.googleapis.com/auth/userinfo.profile',
#     'https://www.googleapis.com/auth/spreadsheets',
#     'https://www.googleapis.com/auth/documents',
#     'https://www.googleapis.com/auth/drive',
# ]
# 
# _GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
# _GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
# 
# # Paths for persisting state across the OAuth redirect (session may be lost)
# _OAUTH_TOKEN_PATH = os.path.join(os.path.dirname(__file__), 'data', 'google_oauth_token.json')
# _OAUTH_DISPLAY_RESULT_PATH = os.path.join(os.path.dirname(__file__), 'data', '_display_result_backup.json')
# 
# 
# def _save_display_result_to_disk():
#     """Persist _display_result to disk so it survives an OAuth redirect."""
#     dr = st.session_state.get('_display_result')
#     if not dr:
#         return
#     try:
#         with open(_OAUTH_DISPLAY_RESULT_PATH, 'w') as f:
#             json.dump(dr, f, default=str)
#     except Exception:
#         pass
# 
# 
# def _restore_display_result_from_disk():
#     """Restore _display_result from disk if session was lost after OAuth redirect."""
#     if '_display_result' in st.session_state:
#         return  # already have it
#     if not os.path.exists(_OAUTH_DISPLAY_RESULT_PATH):
#         return
#     try:
#         with open(_OAUTH_DISPLAY_RESULT_PATH, 'r') as f:
#             dr = json.load(f)
#         st.session_state['_display_result'] = dr
#     except Exception:
#         pass
# 
# 
def _cleanup_display_result_backup():
    """No-op stub — OAuth backup file logic is disabled."""
    pass
# 
# 
# def _save_oauth_token_to_disk(token, user_info=None):
#     """Persist OAuth token to disk so it survives session loss."""
#     try:
#         data = {'token': token}
#         if user_info:
#             data['user'] = user_info
#         with open(_OAUTH_TOKEN_PATH, 'w') as f:
#             json.dump(data, f, default=str)
#     except Exception:
#         pass
# 
# 
# def _load_oauth_token_from_disk():
#     """Load a previously saved OAuth token from disk into session_state."""
#     if 'google_token' in st.session_state:
#         return  # already loaded
#     if not os.path.exists(_OAUTH_TOKEN_PATH):
#         return
#     try:
#         with open(_OAUTH_TOKEN_PATH, 'r') as f:
#             data = json.load(f)
#         st.session_state['google_token'] = data['token']
#         if 'user' in data:
#             st.session_state['google_user'] = data['user']
#     except Exception:
#         pass
# 
# 
# def _google_oauth_cfg():
#     """Return (client_id, client_secret, redirect_uri) from secrets, or Nones."""
#     try:
#         g = st.secrets["google"]
#         cid = g["client_id"]
#         csecret = g["client_secret"]
#         redirect = g.get("redirect_uri", "http://localhost:8501")
#         if cid.startswith("YOUR_"):
#             return None, None, None
#         return cid, csecret, redirect
#     except Exception:
#         return None, None, None
# 
# 
# def _handle_google_oauth_callback():
#     """If the URL contains a Google OAuth `code` param, exchange it for a token.
# 
#     Called at the top of main() so it runs before any display logic.
#     Restores analysis results from disk if the session was lost during redirect.
#     """
#     code = st.query_params.get("code")
#     if not code:
#         # No callback — but try to load a previously saved token from disk
#         _load_oauth_token_from_disk()
#         return
#     if "google_token" in st.session_state:
#         # Already exchanged — just clear leftover params
#         st.query_params.clear()
#         return
# 
#     cid, csecret, redirect = _google_oauth_cfg()
#     if not cid:
#         return
# 
#     try:
#         session = _OAuth2Session(
#             cid, csecret,
#             redirect_uri=redirect,
#         )
#         token = session.fetch_token(
#             _GOOGLE_TOKEN_URL,
#             code=code,
#         )
#         st.session_state["google_token"] = token
# 
#         # Fetch basic user info for display
#         user_info = session.get(
#             "https://www.googleapis.com/oauth2/v1/userinfo"
#         ).json()
#         st.session_state["google_user"] = user_info
# 
#         # Persist token to disk (survives session loss)
#         _save_oauth_token_to_disk(token, user_info)
# 
#         # Restore analysis results if session was lost during the redirect
#         _restore_display_result_from_disk()
# 
#         # Clear the code from the URL so it doesn't get reused
#         st.query_params.clear()
#         st.rerun()
#     except Exception as exc:
#         # Still try to restore results even if auth fails
#         _restore_display_result_from_disk()
#         st.query_params.clear()
#         st.warning(f"Google sign-in failed: {exc}")
# 
# 
# def _get_google_creds():
#     """Return google.oauth2.credentials.Credentials built from the OAuth2 token, or None."""
#     token_data = st.session_state.get("google_token")
#     if not token_data:
#         return None
# 
#     try:
#         from google.oauth2.credentials import Credentials
#         cid, csecret, _ = _google_oauth_cfg()
#         creds = Credentials(
#             token=token_data["access_token"],
#             refresh_token=token_data.get("refresh_token"),
#             token_uri=_GOOGLE_TOKEN_URL,
#             client_id=cid,
#             client_secret=csecret,
#             scopes=_GOOGLE_SCOPES,
#         )
#         return creds
#     except Exception:
#         return None
# 
# 
# def _render_google_sign_in(key_suffix=""):
#     """Show a 'Sign in with Google' button if the user is not yet authenticated.
# 
#     Returns True if the user IS signed in, False otherwise.
#     """
#     if "google_token" in st.session_state:
#         user = st.session_state.get("google_user", {})
#         name = user.get("name", "Google User")
#         email = user.get("email", "")
#         st.caption(f"Signed in as **{name}** ({email})")
#         if st.button("Sign out of Google", key=f"google_sign_out_{key_suffix}"):
#             for k in ("google_token", "google_user"):
#                 st.session_state.pop(k, None)
#             # Remove persisted token from disk too
#             try:
#                 if os.path.exists(_OAUTH_TOKEN_PATH):
#                     os.remove(_OAUTH_TOKEN_PATH)
#             except Exception:
#                 pass
#             _cleanup_display_result_backup()
#             st.rerun()
#         return True
# 
#     cid, csecret, redirect = _google_oauth_cfg()
#     if not cid:
#         st.warning(
#             "Google export requires OAuth 2.0 credentials.  "
#             "Add `[google]` client_id / client_secret to `.streamlit/secrets.toml`.  "
#             "[Setup guide](https://console.cloud.google.com/apis/credentials)"
#         )
#         return False
# 
#     # Save analysis results to disk BEFORE the user navigates away,
#     # so we can restore them if the session is lost during the redirect.
#     _save_display_result_to_disk()
# 
#     session = _OAuth2Session(
#         cid, csecret,
#         scope=" ".join(_GOOGLE_SCOPES),
#         redirect_uri=redirect,
#     )
#     auth_url, _ = session.create_authorization_url(
#         _GOOGLE_AUTH_URL,
#         access_type="offline",
#         prompt="consent",
#     )
# 
#     st.markdown(f'<a href="{auth_url}" target="_self" style="'
#                 f'display:inline-block;padding:8px 20px;background:#4285F4;'
#                 f'color:white;border-radius:4px;text-decoration:none;'
#                 f'font-weight:500;">Sign in with Google</a>',
#                 unsafe_allow_html=True)
#     return False
# 
# 
# def _share_file_anyone(creds, file_id):
#     """Make a Drive file accessible to anyone with the link."""
#     try:
#         from googleapiclient.discovery import build
#         drive = build('drive', 'v3', credentials=creds)
#         drive.permissions().create(
#             fileId=file_id,
#             body={'type': 'anyone', 'role': 'writer'},
#             fields='id',
#         ).execute()
#     except Exception:
#         pass  # non-fatal — file still works for service account
# 
# 
# 
# def _build_report_text(result):
#     """Build a plain-text report from analysis result (fallback / Sheets use)."""
#     f = result['fundamentals']
#     ticker = result['ticker']
#     lines = []
#     lines.append(f"Investment Analysis: {ticker}")
#     lines.append(f"{f.get('name', 'N/A')}")
#     lines.append(f"Date: {datetime.now().strftime('%B %d, %Y')}")
#     lines.append(f"Sector: {f.get('sector', 'N/A')}")
#     lines.append(f"Price: ${f.get('price', 0):.2f}")
#     lines.append("")
#     lines.append(f"Final Score: {result['final_score']:.1f}/100")
#     lines.append(f"Recommendation: {result.get('recommendation', 'N/A')}")
#     lines.append("")
#     lines.append("Agent Breakdown:")
#     for agent, score in result.get('agent_scores', {}).items():
#         lines.append(f"  {agent.replace('_', ' ').title()}: {score:.1f}/100")
#     lines.append("")
#     lines.append("Key Metrics:")
#     mc = f.get('market_cap', 0)
#     lines.append(f"  Market Cap: ${mc/1e9:.2f}B" if mc else "  Market Cap: N/A")
#     pe = f.get('pe_ratio')
#     lines.append(f"  P/E Ratio: {pe:.1f}" if pe else "  P/E Ratio: N/A")
#     beta = f.get('beta')
#     lines.append(f"  Beta: {beta:.2f}" if beta else "  Beta: N/A")
#     dy = f.get('dividend_yield')
#     if dy:
#         lines.append(f"  Dividend Yield: {dy*100:.2f}%")
#     lines.append("")
#     lines.append("Agent Analysis:")
#     for agent, rationale in result.get('agent_rationales', {}).items():
#         if rationale:
#             lines.append(f"\n{agent.replace('_', ' ').title()}:")
#             lines.append(rationale)
#     return "\n".join(lines)
# 
# 
# def _build_formatted_doc_content(result, base_index=1, tab_id=None):
#     """Build text and Google Docs API formatting requests for a richly formatted report.
# 
#     Returns (full_text, format_requests) where format_requests is a list
#     of Google Docs API batchUpdate request dicts for styling (headings, bold,
#     colored scores, hyperlinks).  All body text is set to Times New Roman.
#     Key Metrics and Agent Scores are rendered as inline tables.
#     """
#     import re as _re_fmt
#     f = result['fundamentals']
#     ticker = result['ticker']
#     timestamp = datetime.now().strftime('%B %d, %Y')
#     score = result['final_score']
#     rec = result.get('recommendation', 'N/A')
# 
#     # --- Accumulate text with position tracking ---
#     _offset = [0]  # mutable counter
#     _ranges = []    # (start_offset, end_offset, style)
#     _parts = []
# 
#     def add(text, style='normal'):
#         _parts.append(text)
#         s = _offset[0]
#         _offset[0] += len(text)
#         if style != 'normal':
#             _ranges.append((s, _offset[0], style))
# 
#     # ── Title ──
#     add(f"Investment Analysis: {ticker}\n", 'title')
#     add(f"{f.get('name', 'N/A')}  •  {f.get('sector', 'N/A')}  •  {timestamp}\n", 'subtitle')
#     add("\n")
# 
#     # ── Summary ──
#     add("Summary\n", 'heading2')
#     score_style = 'score_good' if score >= 60 else 'score_bad'
#     add(f"Final Score: {score:.1f}/100", score_style)
#     add("   |   ")
#     rec_style = 'rec_good' if rec and any(w in rec.lower() for w in ('buy', 'outperform', 'overweight', 'accumulate')) else (
#         'rec_bad' if rec and any(w in rec.lower() for w in ('sell', 'underperform', 'underweight', 'reduce')) else 'bold')
#     add(f"Recommendation: {rec}\n", rec_style)
#     add("\n")
# 
#     # ── Key Metrics (as text table) ──
#     add("Key Metrics\n", 'heading2')
#     metrics = []
#     price = f.get('price', 0)
#     if price:
#         metrics.append(("Price", f"${price:.2f}"))
#     mc = f.get('market_cap', 0)
#     if mc:
#         metrics.append(("Market Cap", f"${mc/1e9:.2f}B"))
#     pe = f.get('pe_ratio')
#     if pe:
#         metrics.append(("P/E Ratio", f"{pe:.1f}"))
#     beta_val = f.get('beta')
#     if beta_val:
#         metrics.append(("Beta", f"{beta_val:.2f}"))
#     dy = f.get('dividend_yield')
#     if dy:
#         metrics.append(("Dividend Yield", f"{dy*100:.2f}%"))
#     ev = f.get('enterprise_value')
#     if ev and ev > 0:
#         metrics.append(("Enterprise Value", f"${ev/1e9:.2f}B"))
#     pb = f.get('pb_ratio')
#     if pb:
#         metrics.append(("P/B Ratio", f"{pb:.2f}"))
#     # Render as two-column aligned text block
#     if metrics:
#         col_w = max(len(m[0]) for m in metrics) + 4
#         add("Metric".ljust(col_w) + "Value\n", 'table_header')
#         add("─" * (col_w + 16) + "\n")
#         for i, (label, value) in enumerate(metrics):
#             add(label.ljust(col_w), 'metric_label')
#             add(f"{value}\n")
#     add("\n")
# 
#     # ── Agent Scores (as text table) ──
#     add("Agent Scores\n", 'heading2')
#     agent_scores_list = list(result.get('agent_scores', {}).items())
#     if agent_scores_list:
#         name_w = max(len(n.replace('_', ' ').title()) for n, _ in agent_scores_list) + 4
#         add("Agent".ljust(name_w) + "Score".ljust(14) + "Rating\n", 'table_header')
#         add("─" * (name_w + 24) + "\n")
#         for agent_name, agent_score in agent_scores_list:
#             display_name = agent_name.replace('_', ' ').title()
#             a_style = 'score_good' if agent_score >= 60 else 'score_bad'
#             rating = "Strong" if agent_score >= 75 else ("Good" if agent_score >= 60 else ("Fair" if agent_score >= 40 else "Weak"))
#             add(f"{display_name.ljust(name_w)}", 'metric_label')
#             add(f"{agent_score:.1f}/100".ljust(14), a_style)
#             add(f"{rating}\n")
#     add("\n")
# 
#     # ── Detailed Analysis ──
#     add("Detailed Analysis\n", 'heading2')
#     for agent_name, rationale in result.get('agent_rationales', {}).items():
#         if rationale:
#             display_name = agent_name.replace('_', ' ').title()
#             add(f"{display_name}\n", 'heading3')
#             rat_text = str(rationale).strip()
#             add(f"{rat_text}\n\n")
# 
#     full_text = "".join(_parts)
# 
#     # --- Generate formatting requests ---
#     _GREEN  = {'red': 0.067, 'green': 0.533, 'blue': 0.227}   # #119938
#     _RED    = {'red': 0.776, 'green': 0.165, 'blue': 0.165}   # #C62A2A
#     _GRAY   = {'red': 0.42,  'green': 0.45,  'blue': 0.49}    # #6B737D
#     _BLUE   = {'red': 0.063, 'green': 0.376, 'blue': 0.784}   # #1060C8
#     _NAVY_BG = {'red': 0.11, 'green': 0.22, 'blue': 0.43}
#     _TIMES  = 'Times New Roman'
# 
#     def _rng(s, e):
#         r = {'startIndex': base_index + s, 'endIndex': base_index + e}
#         if tab_id:
#             r['tabId'] = tab_id
#         return r
# 
#     fmt = []
# 
#     # ── Set entire document text to Times New Roman ──
#     fmt.append({'updateTextStyle': {
#         'range': _rng(0, len(full_text)),
#         'textStyle': {'weightedFontFamily': {'fontFamily': _TIMES}},
#         'fields': 'weightedFontFamily',
#     }})
# 
#     for s, e, style in _ranges:
#         rng = _rng(s, e)
#         if style == 'title':
#             fmt.append({'updateParagraphStyle': {
#                 'range': rng,
#                 'paragraphStyle': {'namedStyleType': 'HEADING_1'},
#                 'fields': 'namedStyleType',
#             }})
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'weightedFontFamily': {'fontFamily': _TIMES},
#                               'fontSize': {'magnitude': 18, 'unit': 'PT'}},
#                 'fields': 'weightedFontFamily,fontSize',
#             }})
#         elif style == 'subtitle':
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'italic': True,
#                               'foregroundColor': {'color': {'rgbColor': _GRAY}},
#                               'fontSize': {'magnitude': 11, 'unit': 'PT'},
#                               'weightedFontFamily': {'fontFamily': _TIMES}},
#                 'fields': 'italic,foregroundColor,fontSize,weightedFontFamily',
#             }})
#         elif style == 'heading2':
#             fmt.append({'updateParagraphStyle': {
#                 'range': rng,
#                 'paragraphStyle': {'namedStyleType': 'HEADING_2',
#                                    'spaceAbove': {'magnitude': 14, 'unit': 'PT'},
#                                    'spaceBelow': {'magnitude': 6, 'unit': 'PT'}},
#                 'fields': 'namedStyleType,spaceAbove,spaceBelow',
#             }})
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'weightedFontFamily': {'fontFamily': _TIMES},
#                               'fontSize': {'magnitude': 14, 'unit': 'PT'}},
#                 'fields': 'weightedFontFamily,fontSize',
#             }})
#         elif style == 'heading3':
#             fmt.append({'updateParagraphStyle': {
#                 'range': rng,
#                 'paragraphStyle': {'namedStyleType': 'HEADING_3',
#                                    'spaceAbove': {'magnitude': 10, 'unit': 'PT'}},
#                 'fields': 'namedStyleType,spaceAbove',
#             }})
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'weightedFontFamily': {'fontFamily': _TIMES},
#                               'fontSize': {'magnitude': 12, 'unit': 'PT'}},
#                 'fields': 'weightedFontFamily,fontSize',
#             }})
#         elif style == 'table_header':
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'bold': True,
#                               'foregroundColor': {'color': {'rgbColor': _NAVY_BG}},
#                               'fontSize': {'magnitude': 10, 'unit': 'PT'},
#                               'weightedFontFamily': {'fontFamily': _TIMES}},
#                 'fields': 'bold,foregroundColor,fontSize,weightedFontFamily',
#             }})
#         elif style in ('bold', 'metric_label'):
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'bold': True,
#                               'weightedFontFamily': {'fontFamily': _TIMES}},
#                 'fields': 'bold,weightedFontFamily',
#             }})
#         elif style in ('score_good', 'rec_good'):
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'bold': True,
#                               'foregroundColor': {'color': {'rgbColor': _GREEN}},
#                               'weightedFontFamily': {'fontFamily': _TIMES}},
#                 'fields': 'bold,foregroundColor,weightedFontFamily',
#             }})
#         elif style in ('score_bad', 'rec_bad'):
#             fmt.append({'updateTextStyle': {
#                 'range': rng,
#                 'textStyle': {'bold': True,
#                               'foregroundColor': {'color': {'rgbColor': _RED}},
#                               'weightedFontFamily': {'fontFamily': _TIMES}},
#                 'fields': 'bold,foregroundColor,weightedFontFamily',
#             }})
# 
#     # --- Convert URLs in text to clickable hyperlinks ---
#     _url_re = _re_fmt.compile(r'https?://[^\s\)\]\}>,"]+')
#     for m in _url_re.finditer(full_text):
#         url = m.group().rstrip('.')  # strip trailing period if any
#         rng = _rng(m.start(), m.start() + len(url))
#         fmt.append({'updateTextStyle': {
#             'range': rng,
#             'textStyle': {'link': {'url': url},
#                           'foregroundColor': {'color': {'rgbColor': _BLUE}},
#                           'underline': True,
#                           'weightedFontFamily': {'fontFamily': _TIMES}},
#             'fields': 'link,foregroundColor,underline,weightedFontFamily',
#         }})
# 
#     return full_text, fmt
# 
# 
# def _export_to_sheets(creds, result, spreadsheet_id=None, custom_name=None):
#     """Export analysis to Google Sheets. Returns (spreadsheet_id, url)."""
#     import gspread
#     from gspread_formatting import (
#         format_cell_range, CellFormat, TextFormat, Color, Border, Borders,
#         set_column_widths, set_row_height,
#     )
#     gc = gspread.authorize(creds)
#     f = result['fundamentals']
#     ticker = result['ticker']
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
# 
#     if spreadsheet_id:
#         sh = gc.open_by_key(spreadsheet_id)
#         ws_title = f"{ticker} {datetime.now().strftime('%m/%d')}"
#         try:
#             ws = sh.add_worksheet(title=ws_title, rows=60, cols=10)
#         except Exception:
#             ws = sh.add_worksheet(title=f"{ws_title}_{int(time.time())%10000}", rows=60, cols=10)
#     else:
#         title = custom_name if custom_name else f"{ticker} Analysis - {timestamp}"
#         sh = gc.create(title)
#         ws = sh.sheet1
#         ws.update_title(f"{ticker} Analysis")
# 
#     # ── Colour palette ──
#     _WHITE = Color(1, 1, 1)
#     _DARK  = Color(0.15, 0.18, 0.22)
#     _NAVY  = Color(0.11, 0.22, 0.43)       # header bg
#     _LIGHT_BLUE = Color(0.85, 0.91, 0.97)  # alternating row
#     _GREEN = Color(0.07, 0.53, 0.23)
#     _RED   = Color(0.78, 0.17, 0.17)
#     _LIGHT_GRAY = Color(0.94, 0.94, 0.94)
# 
#     _thin_border = Border("SOLID", width=1, color=Color(0.8, 0.8, 0.8))
#     _borders_all = Borders(top=_thin_border, bottom=_thin_border, left=_thin_border, right=_thin_border)
# 
#     score_val = result['final_score']
#     rec = result.get('recommendation', 'N/A')
# 
#     # ── Build rows ──
#     rows = [
#         [f"{ticker} – Investment Analysis", "", "", "", ""],
#         [f.get('name', 'N/A'), "", f"Sector: {f.get('sector', 'N/A')}", "", f"Date: {timestamp}"],
#         [],
#         ["SUMMARY", "", "", "", ""],
#         ["Final Score", f"{score_val:.1f} / 100", "", "Recommendation", rec],
#         [],
#         ["KEY METRICS", "", "", "", ""],
#         ["Metric", "Value", "", "Metric", "Value"],
#     ]
# 
#     # Build two-column metric pairs
#     metric_pairs = []
#     price = f.get('price', 0)
#     if price:
#         metric_pairs.append(("Price", f"${price:.2f}"))
#     mc = f.get('market_cap', 0)
#     if mc:
#         metric_pairs.append(("Market Cap", f"${mc/1e9:.2f}B"))
#     pe = f.get('pe_ratio')
#     if pe:
#         metric_pairs.append(("P/E Ratio", f"{pe:.1f}"))
#     beta_val = f.get('beta')
#     if beta_val:
#         metric_pairs.append(("Beta", f"{beta_val:.2f}"))
#     dy = f.get('dividend_yield')
#     if dy:
#         metric_pairs.append(("Dividend Yield", f"{dy*100:.2f}%"))
#     ev = f.get('enterprise_value')
#     if ev and ev > 0:
#         metric_pairs.append(("Enterprise Value", f"${ev/1e9:.2f}B"))
#     pb = f.get('pb_ratio')
#     if pb:
#         metric_pairs.append(("P/B Ratio", f"{pb:.2f}"))
# 
#     # Pair metrics into left/right columns
#     half = (len(metric_pairs) + 1) // 2
#     for i in range(half):
#         left = metric_pairs[i] if i < len(metric_pairs) else ("", "")
#         right = metric_pairs[i + half] if (i + half) < len(metric_pairs) else ("", "")
#         rows.append([left[0], left[1], "", right[0], right[1]])
#     metrics_end_row = len(rows)
# 
#     rows.append([])
#     agent_header_row = len(rows) + 1
#     rows.append(["AGENT SCORES", "", "", "", ""])
#     rows.append(["Agent", "Score", "Rating", "", ""])
#     agent_start_row = len(rows) + 1
#     for agent_name, agent_score in result.get('agent_scores', {}).items():
#         display_name = agent_name.replace('_', ' ').title()
#         rating = "Strong" if agent_score >= 75 else ("Good" if agent_score >= 60 else ("Fair" if agent_score >= 40 else "Weak"))
#         rows.append([display_name, round(agent_score, 1), rating, "", ""])
#     agent_end_row = len(rows)
# 
#     rows.append([])
#     analysis_header_row = len(rows) + 1
#     rows.append(["DETAILED ANALYSIS", "", "", "", ""])
#     for agent_name, rationale in result.get('agent_rationales', {}).items():
#         if rationale:
#             rows.append([agent_name.replace('_', ' ').title()])
#             for line in str(rationale).split('\n'):
#                 if line.strip():
#                     rows.append([line.strip()])
#             rows.append([])
#     total_rows = len(rows)
# 
#     ws.update(range_name='A1', values=rows)
# 
#     # ── Formatting ──
#     try:
#         # Column widths
#         set_column_widths(ws, [('A', 180), ('B', 140), ('C', 140), ('D', 180), ('E', 180)])
# 
#         # Row 1 – Title bar
#         set_row_height(ws, '1', 42)
#         format_cell_range(ws, 'A1:E1', CellFormat(
#             backgroundColor=_NAVY,
#             textFormat=TextFormat(bold=True, fontSize=16, foregroundColor=_WHITE,
#                                  fontFamily='Arial'),
#             horizontalAlignment='LEFT',
#             verticalAlignment='MIDDLE',
#         ))
# 
#         # Row 2 – Subtitle
#         format_cell_range(ws, 'A2:E2', CellFormat(
#             backgroundColor=_LIGHT_GRAY,
#             textFormat=TextFormat(fontSize=10, foregroundColor=_DARK, fontFamily='Arial'),
#             horizontalAlignment='LEFT',
#         ))
# 
#         # Section headers (SUMMARY, KEY METRICS, AGENT SCORES, DETAILED ANALYSIS)
#         _section_fmt = CellFormat(
#             backgroundColor=Color(0.22, 0.32, 0.52),
#             textFormat=TextFormat(bold=True, fontSize=12, foregroundColor=_WHITE, fontFamily='Arial'),
#             horizontalAlignment='LEFT',
#             borders=_borders_all,
#         )
#         format_cell_range(ws, 'A4:E4', _section_fmt)
#         format_cell_range(ws, 'A7:E7', _section_fmt)
#         format_cell_range(ws, f'A{agent_header_row}:E{agent_header_row}', _section_fmt)
#         format_cell_range(ws, f'A{analysis_header_row}:E{analysis_header_row}', _section_fmt)
# 
#         # Summary row (row 5)
#         format_cell_range(ws, 'A5:E5', CellFormat(
#             textFormat=TextFormat(bold=True, fontSize=11, fontFamily='Arial'),
#             borders=_borders_all,
#             backgroundColor=_LIGHT_BLUE,
#         ))
#         # Color score
#         score_color = _GREEN if score_val >= 60 else _RED
#         format_cell_range(ws, 'B5', CellFormat(
#             textFormat=TextFormat(bold=True, fontSize=11, foregroundColor=score_color, fontFamily='Arial'),
#         ))
# 
#         # Metric table header (row 8)
#         format_cell_range(ws, 'A8:E8', CellFormat(
#             textFormat=TextFormat(bold=True, fontSize=10, foregroundColor=_WHITE, fontFamily='Arial'),
#             backgroundColor=Color(0.33, 0.43, 0.60),
#             borders=_borders_all,
#         ))
#         # Metric rows with alternating colours
#         for r_idx in range(9, metrics_end_row + 1):
#             bg = _LIGHT_BLUE if (r_idx % 2 == 1) else _WHITE
#             format_cell_range(ws, f'A{r_idx}:E{r_idx}', CellFormat(
#                 backgroundColor=bg,
#                 textFormat=TextFormat(fontSize=10, fontFamily='Arial'),
#                 borders=_borders_all,
#             ))
#             # Bold metric labels
#             format_cell_range(ws, f'A{r_idx}', CellFormat(
#                 textFormat=TextFormat(bold=True, fontSize=10, fontFamily='Arial'),
#             ))
#             format_cell_range(ws, f'D{r_idx}', CellFormat(
#                 textFormat=TextFormat(bold=True, fontSize=10, fontFamily='Arial'),
#             ))
# 
#         # Agent scores table header
#         agent_hdr = agent_start_row - 1
#         format_cell_range(ws, f'A{agent_hdr}:E{agent_hdr}', CellFormat(
#             textFormat=TextFormat(bold=True, fontSize=10, foregroundColor=_WHITE, fontFamily='Arial'),
#             backgroundColor=Color(0.33, 0.43, 0.60),
#             borders=_borders_all,
#         ))
#         # Agent score rows with conditional coloring
#         for r_idx in range(agent_start_row, agent_end_row + 1):
#             bg = _LIGHT_BLUE if (r_idx % 2 == 1) else _WHITE
#             format_cell_range(ws, f'A{r_idx}:E{r_idx}', CellFormat(
#                 backgroundColor=bg,
#                 textFormat=TextFormat(fontSize=10, fontFamily='Arial'),
#                 borders=_borders_all,
#             ))
#             format_cell_range(ws, f'A{r_idx}', CellFormat(
#                 textFormat=TextFormat(bold=True, fontSize=10, fontFamily='Arial'),
#             ))
# 
#         # Color agent scores (column B) green/red based on value
#         for r_idx in range(agent_start_row, agent_end_row + 1):
#             cell_val = ws.acell(f'B{r_idx}').value
#             try:
#                 v = float(cell_val)
#                 clr = _GREEN if v >= 60 else _RED
#             except (TypeError, ValueError):
#                 clr = _DARK
#             format_cell_range(ws, f'B{r_idx}', CellFormat(
#                 textFormat=TextFormat(bold=True, fontSize=10, foregroundColor=clr, fontFamily='Arial'),
#             ))
# 
#         # Analysis section – bold agent names
#         for r_idx in range(analysis_header_row + 1, total_rows + 1):
#             format_cell_range(ws, f'A{r_idx}:E{r_idx}', CellFormat(
#                 textFormat=TextFormat(fontSize=10, fontFamily='Arial'),
#             ))
# 
#         # Freeze top row
#         ws.freeze(rows=1)
# 
#     except Exception:
#         pass
# 
#     _share_file_anyone(creds, sh.id)
#     return sh.id, sh.url
# 
# 
# def _export_to_docs(creds, result, document_id=None, insert_mode='page_break', custom_name=None):
#     """Export analysis to Google Docs with rich formatting. Returns (doc_id, url).
# 
#     insert_mode: 'page_break' inserts a page break then the report (new page).
#                  'append' appends to the bottom of the document with a separator.
#                  'doc_tab' creates a new document tab within the same document.
#     """
#     from googleapiclient.discovery import build
#     service = build('docs', 'v1', credentials=creds)
#     drive_service = build('drive', 'v3', credentials=creds)
# 
#     ticker = result['ticker']
#     timestamp = datetime.now().strftime('%B %d, %Y')
# 
#     if document_id:
#         if insert_mode == 'doc_tab':
#             # ── Create a new document tab via addDocumentTab ──
#             tab_title = f"{ticker} Analysis - {datetime.now().strftime('%m/%d')}"
#             try:
#                 resp = service.documents().batchUpdate(
#                     documentId=document_id,
#                     body={'requests': [{'addDocumentTab': {
#                         'tabProperties': {'title': tab_title}
#                     }}]}
#                 ).execute()
#                 new_tab_id = resp['replies'][0]['addDocumentTab']['tabProperties']['tabId']
# 
#                 # Build formatted text + formatting requests targeting the new tab
#                 report_text, fmt_requests = _build_formatted_doc_content(
#                     result, base_index=1, tab_id=new_tab_id)
# 
#                 # Insert text then apply formatting
#                 all_requests = [{'insertText': {
#                     'location': {'index': 1, 'tabId': new_tab_id},
#                     'text': report_text,
#                 }}]
#                 all_requests.extend(fmt_requests)
#                 service.documents().batchUpdate(
#                     documentId=document_id, body={'requests': all_requests}
#                 ).execute()
# 
#             except Exception as tab_err:
#                 import logging
#                 logging.getLogger(__name__).warning(
#                     f"Document tab creation failed ({tab_err}), falling back to page break."
#                 )
#                 return _export_to_docs(creds, result, document_id=document_id,
#                                        insert_mode='page_break', custom_name=custom_name)
# 
#             url = f"https://docs.google.com/document/d/{document_id}/edit"
#             return document_id, url
# 
#         # ── Export to existing document (page_break or append) ──
#         doc = service.documents().get(documentId=document_id).execute()
#         end_index = doc['body']['content'][-1]['endIndex'] - 1
# 
#         if insert_mode == 'page_break':
#             # Phase 1: insert the page break
#             service.documents().batchUpdate(documentId=document_id, body={'requests': [
#                 {'insertText': {'location': {'index': end_index}, 'text': '\n'}},
#                 {'insertPageBreak': {'location': {'index': end_index + 1}}},
#             ]}).execute()
# 
#             # Re-fetch to get the new insertion point
#             doc = service.documents().get(documentId=document_id).execute()
#             new_end = doc['body']['content'][-1]['endIndex'] - 1
# 
#             # Phase 2: insert formatted text + styling
#             report_text, fmt_requests = _build_formatted_doc_content(
#                 result, base_index=new_end)
#             all_requests = [{'insertText': {
#                 'location': {'index': new_end}, 'text': report_text
#             }}]
#             all_requests.extend(fmt_requests)
#             service.documents().batchUpdate(
#                 documentId=document_id, body={'requests': all_requests}
#             ).execute()
#         else:
#             # Append with separator
#             separator = "\n\n" + "=" * 60 + "\n\n"
#             base = end_index + len(separator)
#             report_text, fmt_requests = _build_formatted_doc_content(
#                 result, base_index=base)
#             all_requests = [{'insertText': {
#                 'location': {'index': end_index},
#                 'text': separator + report_text
#             }}]
#             all_requests.extend(fmt_requests)
#             service.documents().batchUpdate(
#                 documentId=document_id, body={'requests': all_requests}
#             ).execute()
# 
#         url = f"https://docs.google.com/document/d/{document_id}/edit"
#         return document_id, url
#     else:
#         # ── Create new document ──
#         title = custom_name if custom_name else f"{ticker} Investment Analysis - {timestamp}"
#         doc = service.documents().create(body={'title': title}).execute()
#         doc_id = doc['documentId']
# 
#         report_text, fmt_requests = _build_formatted_doc_content(
#             result, base_index=1)
#         all_requests = [{'insertText': {
#             'location': {'index': 1}, 'text': report_text
#         }}]
#         all_requests.extend(fmt_requests)
#         service.documents().batchUpdate(
#             documentId=doc_id, body={'requests': all_requests}
#         ).execute()
# 
#         _share_file_anyone(creds, doc_id)
# 
#         url = f"https://docs.google.com/document/d/{doc_id}/edit"
#         return doc_id, url
# 
# 
# def _export_multi_to_sheets(creds, results, spreadsheet_id=None, custom_name=None):
#     """Export multi-stock comparison to Google Sheets."""
#     import gspread
#     from gspread_formatting import (
#         format_cell_range, CellFormat, TextFormat, Color, Border, Borders,
#         set_column_widths, set_row_height,
#     )
#     gc = gspread.authorize(creds)
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
# 
#     if spreadsheet_id:
#         sh = gc.open_by_key(spreadsheet_id)
#         ws_title = f"Comparison {datetime.now().strftime('%m/%d')}"
#         try:
#             ws = sh.add_worksheet(title=ws_title, rows=50, cols=15)
#         except Exception:
#             ws = sh.add_worksheet(title=f"{ws_title}_{int(time.time())%10000}", rows=50, cols=15)
#     else:
#         title = custom_name if custom_name else f"Stock Comparison - {timestamp}"
#         sh = gc.create(title)
#         ws = sh.sheet1
#         ws.update_title("Comparison")
# 
#     # Colour palette
#     _WHITE = Color(1, 1, 1)
#     _NAVY  = Color(0.11, 0.22, 0.43)
#     _LIGHT_BLUE = Color(0.85, 0.91, 0.97)
#     _GREEN = Color(0.07, 0.53, 0.23)
#     _RED   = Color(0.78, 0.17, 0.17)
#     _thin = Border("SOLID", width=1, color=Color(0.8, 0.8, 0.8))
#     _borders_all = Borders(top=_thin, bottom=_thin, left=_thin, right=_thin)
# 
#     # Header
#     headers = ['Ticker', 'Name', 'Final Score', 'Recommendation', 'Price',
#                'Sector', 'Value', 'Growth', 'Macro', 'Risk', 'Sentiment']
#     rows = [headers]
# 
#     sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
#     for r in sorted_results:
#         f = r['fundamentals']
#         scores = r.get('agent_scores', {})
#         rows.append([
#             r['ticker'],
#             f.get('name', 'N/A'),
#             round(r['final_score'], 1),
#             r.get('recommendation', 'N/A'),
#             f"${f.get('price', 0):.2f}",
#             f.get('sector', 'N/A'),
#             round(scores.get('value_agent', 0), 1),
#             round(scores.get('growth_momentum_agent', 0), 1),
#             round(scores.get('macro_regime_agent', 0), 1),
#             round(scores.get('risk_agent', 0), 1),
#             round(scores.get('sentiment_agent', 0), 1),
#         ])
# 
#     ws.update(range_name='A1', values=rows)
# 
#     # ── Formatting ──
#     try:
#         total_rows = len(rows)
# 
#         # Column widths
#         set_column_widths(ws, [
#             ('A', 90), ('B', 200), ('C', 100), ('D', 140), ('E', 100),
#             ('F', 140), ('G', 80), ('H', 80), ('I', 80), ('J', 80), ('K', 90),
#         ])
# 
#         # Header row
#         set_row_height(ws, '1', 38)
#         format_cell_range(ws, 'A1:K1', CellFormat(
#             backgroundColor=_NAVY,
#             textFormat=TextFormat(bold=True, fontSize=11, foregroundColor=_WHITE, fontFamily='Arial'),
#             horizontalAlignment='CENTER',
#             verticalAlignment='MIDDLE',
#             borders=_borders_all,
#         ))
# 
#         # Data rows with alternating colours
#         for r_idx in range(2, total_rows + 1):
#             bg = _LIGHT_BLUE if (r_idx % 2 == 0) else _WHITE
#             format_cell_range(ws, f'A{r_idx}:K{r_idx}', CellFormat(
#                 backgroundColor=bg,
#                 textFormat=TextFormat(fontSize=10, fontFamily='Arial'),
#                 borders=_borders_all,
#                 verticalAlignment='MIDDLE',
#             ))
#             # Bold ticker
#             format_cell_range(ws, f'A{r_idx}', CellFormat(
#                 textFormat=TextFormat(bold=True, fontSize=10, fontFamily='Arial'),
#             ))
# 
#         # Color final scores (column C) and agent scores (G-K) green/red
#         score_cols = ['C', 'G', 'H', 'I', 'J', 'K']
#         for r_idx in range(2, total_rows + 1):
#             for col in score_cols:
#                 cell_val = ws.acell(f'{col}{r_idx}').value
#                 try:
#                     v = float(cell_val)
#                     clr = _GREEN if v >= 60 else _RED
#                 except (TypeError, ValueError):
#                     continue
#                 format_cell_range(ws, f'{col}{r_idx}', CellFormat(
#                     textFormat=TextFormat(bold=True, fontSize=10, foregroundColor=clr, fontFamily='Arial'),
#                 ))
# 
#         # Freeze header row
#         ws.freeze(rows=1)
# 
#     except Exception:
#         pass
# 
#     _share_file_anyone(creds, sh.id)
# 
#     return sh.id, sh.url
# 
# 
# def _list_drive_files(creds, mime_type):
#     """List recent files of given MIME type from user's Drive."""
#     from googleapiclient.discovery import build
#     service = build('drive', 'v3', credentials=creds)
#     query = f"mimeType='{mime_type}' and trashed=false"
#     results = service.files().list(
#         q=query, pageSize=20, orderBy='modifiedTime desc',
#         fields='files(id, name, modifiedTime)',
#     ).execute()
#     return results.get('files', [])
# 
# 
# def _render_google_export(result):
#     """Render the Google Sheets/Docs export UI for a single stock."""
#     ticker = result['ticker']
#     exp_key = f"_gexp_open_{ticker}"
#     is_expanded = st.session_state.get(exp_key, False)
#     with st.expander("Export to Google Sheets / Docs", expanded=is_expanded):
#         # Track that the user opened the expander (persists across reruns)
#         if not is_expanded:
#             st.session_state[exp_key] = True
# 
#         if not _render_google_sign_in(key_suffix=ticker):
#             return
# 
#         creds = _get_google_creds()
#         if creds is None:
#             st.error("Could not build credentials from token.")
#             return
# 
#         # ---- Authenticated: show export options ----
#         col_s, col_d = st.columns(2)
# 
#         with col_s:
#             st.markdown("**Google Sheets**")
#             sheets_mode = st.radio(
#                 "Destination", ["New Spreadsheet", "Existing Spreadsheet"],
#                 key=f"sheets_mode_{ticker}", horizontal=True
#             )
#             sheet_id = None
#             sheets_custom_name = None
#             if sheets_mode == "Existing Spreadsheet":
#                 try:
#                     files = _list_drive_files(creds, 'application/vnd.google-apps.spreadsheet')
#                     if files:
#                         options = {gf['name']: gf['id'] for gf in files}
#                         selected = st.selectbox("Choose spreadsheet", list(options.keys()),
#                                                 key=f"sheets_pick_{ticker}")
#                         sheet_id = options[selected]
#                     else:
#                         st.caption("No spreadsheets found. A new one will be created.")
#                 except Exception:
#                     st.caption("Could not list files. A new one will be created.")
#             else:
#                 default_name = f"{ticker} Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
#                 sheets_custom_name = st.text_input(
#                     "Spreadsheet name", value=default_name,
#                     key=f"sheets_name_{ticker}"
#                 )
# 
#             if st.button("Export to Sheets", key=f"export_sheets_{ticker}", use_container_width=True):
#                 with st.spinner("Exporting to Google Sheets..."):
#                     try:
#                         sid, url = _export_to_sheets(creds, result, spreadsheet_id=sheet_id,
#                                                      custom_name=sheets_custom_name)
#                         st.success(f"Exported! [Open Spreadsheet]({url})")
#                     except Exception as e:
#                         st.error(f"Export failed: {e}")
# 
#         with col_d:
#             st.markdown("**Google Docs**")
#             docs_mode = st.radio(
#                 "Destination", ["New Document", "Existing Document"],
#                 key=f"docs_mode_{ticker}", horizontal=True
#             )
#             doc_id = None
#             docs_insert_mode = 'page_break'
#             docs_custom_name = None
#             if docs_mode == "Existing Document":
#                 try:
#                     files = _list_drive_files(creds, 'application/vnd.google-apps.document')
#                     if files:
#                         options = {gf['name']: gf['id'] for gf in files}
#                         selected = st.selectbox("Choose document", list(options.keys()),
#                                                 key=f"docs_pick_{ticker}")
#                         doc_id = options[selected]
#                     else:
#                         st.caption("No documents found. A new one will be created.")
#                 except Exception:
#                     st.caption("Could not list files. A new one will be created.")
# 
#                 docs_insert_mode = st.radio(
#                     "Insert method",
#                     ["New document tab", "New page (page break)", "Append to bottom"],
#                     key=f"docs_insert_{ticker}", horizontal=True,
#                     help="New document tab: creates a separate tab in the same doc. "
#                          "Page break: adds content on a new page. "
#                          "Append: adds to the bottom of the current content."
#                 )
#                 if 'tab' in docs_insert_mode.lower():
#                     docs_insert_mode = 'doc_tab'
#                 elif 'New page' in docs_insert_mode:
#                     docs_insert_mode = 'page_break'
#                 else:
#                     docs_insert_mode = 'append'
#             else:
#                 default_doc_name = f"{ticker} Investment Analysis - {datetime.now().strftime('%B %d, %Y')}"
#                 docs_custom_name = st.text_input(
#                     "Document name", value=default_doc_name,
#                     key=f"docs_name_{ticker}"
#                 )
# 
#             if st.button("Export to Docs", key=f"export_docs_{ticker}", use_container_width=True):
#                 with st.spinner("Exporting to Google Docs..."):
#                     try:
#                         did, url = _export_to_docs(creds, result, document_id=doc_id,
#                                                    insert_mode=docs_insert_mode,
#                                                    custom_name=docs_custom_name)
#                         st.success(f"Exported! [Open Document]({url})")
#                     except Exception as e:
#                         st.error(f"Export failed: {e}")
# 
# 
# def _render_google_export_multi(results):
#     """Render the Google Sheets/Docs export UI for multi-stock comparison."""
#     exp_key = "_gexp_open_multi"
#     is_expanded = st.session_state.get(exp_key, False)
#     with st.expander("Export Comparison to Google Sheets / Docs", expanded=is_expanded):
#         if not is_expanded:
#             st.session_state[exp_key] = True
# 
#         if not _render_google_sign_in(key_suffix="multi"):
#             return
# 
#         creds = _get_google_creds()
#         if creds is None:
#             st.error("Could not build credentials from token.")
#             return
# 
#         # ---- Authenticated ----
#         col_s, col_d = st.columns(2)
# 
#         with col_s:
#             st.markdown("**Google Sheets (Comparison Table)**")
#             sheets_mode = st.radio(
#                 "Destination", ["New Spreadsheet", "Existing Spreadsheet"],
#                 key="sheets_mode_multi", horizontal=True
#             )
#             sheet_id = None
#             sheets_custom_name = None
#             if sheets_mode == "Existing Spreadsheet":
#                 try:
#                     files = _list_drive_files(creds, 'application/vnd.google-apps.spreadsheet')
#                     if files:
#                         options = {gf['name']: gf['id'] for gf in files}
#                         selected = st.selectbox("Choose spreadsheet", list(options.keys()),
#                                                 key="sheets_pick_multi")
#                         sheet_id = options[selected]
#                     else:
#                         st.caption("No spreadsheets found.")
#                 except Exception:
#                     st.caption("Could not list files.")
#             else:
#                 tickers_str = ", ".join(r['ticker'] for r in results[:5])
#                 default_name = f"Stock Comparison ({tickers_str}) - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
#                 sheets_custom_name = st.text_input(
#                     "Spreadsheet name", value=default_name,
#                     key="sheets_name_multi"
#                 )
# 
#             if st.button("Export Comparison to Sheets", key="export_sheets_multi", use_container_width=True):
#                 with st.spinner("Exporting comparison to Google Sheets..."):
#                     try:
#                         sid, url = _export_multi_to_sheets(creds, results, spreadsheet_id=sheet_id,
#                                                            custom_name=sheets_custom_name)
#                         st.success(f"Exported! [Open Spreadsheet]({url})")
#                     except Exception as e:
#                         st.error(f"Export failed: {e}")
# 
#         with col_d:
#             st.markdown("**Google Docs (Full Reports)**")
#             docs_mode = st.radio(
#                 "Destination", ["New Document", "Existing Document"],
#                 key="docs_mode_multi", horizontal=True
#             )
#             doc_id = None
#             docs_insert_mode_multi = 'page_break'
#             docs_custom_name = None
#             if docs_mode == "Existing Document":
#                 try:
#                     files = _list_drive_files(creds, 'application/vnd.google-apps.document')
#                     if files:
#                         options = {gf['name']: gf['id'] for gf in files}
#                         selected = st.selectbox("Choose document", list(options.keys()),
#                                                 key="docs_pick_multi")
#                         doc_id = options[selected]
#                     else:
#                         st.caption("No documents found.")
#                 except Exception:
#                     st.caption("Could not list files.")
# 
#                 docs_insert_mode_multi = st.radio(
#                     "Insert method",
#                     ["New document tab", "New page (page break)", "Append to bottom"],
#                     key="docs_insert_multi", horizontal=True,
#                     help="New document tab: creates a separate tab in the same doc. "
#                          "Page break: adds content on a new page. "
#                          "Append: adds to the bottom of the current content."
#                 )
#                 if 'tab' in docs_insert_mode_multi.lower():
#                     docs_insert_mode_multi = 'doc_tab'
#                 elif 'New page' in docs_insert_mode_multi:
#                     docs_insert_mode_multi = 'page_break'
#                 else:
#                     docs_insert_mode_multi = 'append'
#             else:
#                 tickers_str = ", ".join(r['ticker'] for r in results[:5])
#                 default_doc_name = f"Investment Analysis ({tickers_str}) - {datetime.now().strftime('%B %d, %Y')}"
#                 docs_custom_name = st.text_input(
#                     "Document name", value=default_doc_name,
#                     key="docs_name_multi"
#                 )
# 
#             if st.button("Export Reports to Docs", key="export_docs_multi", use_container_width=True):
#                 with st.spinner("Exporting reports to Google Docs..."):
#                     try:
#                         first = True
#                         created_doc_id = doc_id
#                         _ins_mode = docs_insert_mode_multi
#                         for r in sorted(results, key=lambda x: x['final_score'], reverse=True):
#                             if first and not created_doc_id:
#                                 created_doc_id, url = _export_to_docs(creds, r, document_id=None,
#                                                                        custom_name=docs_custom_name)
#                                 first = False
#                             else:
#                                 _, url = _export_to_docs(creds, r, document_id=created_doc_id,
#                                                          insert_mode=_ins_mode)
#                                 first = False
#                         st.success(f"Exported {len(results)} reports! [Open Document]({url})")
#                     except Exception as e:
#                         st.error(f"Export failed: {e}")
# 
# 
def _render_ticker_not_found(ticker: str):
    """Render a friendly, informative error when a ticker has no real data."""
    st.markdown(
        f"""
        <div style="
            background:#fff8f8;border:1.5px solid #fca5a5;border-radius:12px;
            padding:22px 26px;margin:16px 0;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        ">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <span style="font-size:22px;">⚠️</span>
            <span style="font-size:17px;font-weight:700;color:#b91c1c;">
              No data found for &ldquo;{ticker}&rdquo;
            </span>
          </div>
          <p style="color:#374151;margin:0 0 10px 0;font-size:14px;">
            We couldn&rsquo;t retrieve reliable market data for this ticker.
            This typically happens because:
          </p>
          <ul style="color:#374151;font-size:13px;margin:0 0 12px 0;padding-left:20px;line-height:1.8;">
            <li>The ticker symbol is <strong>misspelled</strong> (e.g. <code>APPL</code> instead of <code>AAPL</code>)</li>
            <li>The stock is <strong>delisted</strong> or no longer actively trading</li>
            <li>It&rsquo;s an <strong>OTC / pink-sheet / penny stock</strong> with limited data coverage</li>
            <li>It&rsquo;s a <strong>very new listing</strong> or a private company</li>
            <li>The symbol belongs to a <strong>non-US exchange</strong> not covered by our data providers</li>
          </ul>
          <p style="color:#6b7280;font-size:13px;margin:0;">
            <strong>Try:</strong> &nbsp;
            Double-check the symbol on a site like Yahoo Finance or Google Finance,
            or use a well-known US stock (e.g. AAPL, MSFT, GOOGL, JPM) to verify the system is working.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_multi_stock_pdf_report(results: list) -> bytes:
    """Generate a multi-stock comparison PDF report using ReportLab."""
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )

    # ---- Colour palette ----
    C_DARK      = HexColor('#1a1a2e')
    C_BRAND     = HexColor('#3b5998')
    C_GREEN     = HexColor('#059669')
    C_L_GREEN   = HexColor('#10b981')
    C_YELLOW    = HexColor('#d97706')
    C_ORANGE    = HexColor('#f97316')
    C_RED       = HexColor('#ef4444')
    C_GRAY_BG   = HexColor('#f8f9fa')
    C_GRAY_LINE = HexColor('#e5e7eb')
    C_GRAY_MID  = HexColor('#6b7280')

    def _sc(s):
        if s >= 65:   return C_GREEN
        if s >= 55:   return C_L_GREEN
        if s >= 45:   return C_YELLOW
        if s >= 35:   return C_ORANGE
        return C_RED

    def _sl(s):
        if s >= 65:   return 'STRONG BUY'
        if s >= 55:   return 'BUY'
        if s >= 45:   return 'HOLD'
        if s >= 35:   return 'UNDERPERFORM'
        return 'SELL'

    def _fmt_price(p):
        return f'${p:,.2f}' if p and p > 0 else 'N/A'

    def _fmt_cap(mc):
        if mc and mc >= 1e12: return f'${mc/1e12:.1f}T'
        if mc and mc >= 1e9:  return f'${mc/1e9:.1f}B'
        if mc and mc > 0:     return f'${mc/1e6:.0f}M'
        return 'N/A'

    sorted_results = sorted(results, key=lambda r: float(r.get('final_score', 0)), reverse=True)
    report_date = datetime.now().strftime('%B %d, %Y')
    n = len(sorted_results)

    # ---- Styles ----
    styles = getSampleStyleSheet()
    S = lambda name, **kw: ParagraphStyle(name, parent=styles['Normal'], **kw)
    sTitle      = S('msTitle',   fontSize=20, fontName='Helvetica-Bold', textColor=C_DARK, leading=26, spaceAfter=4)
    sSub        = S('msSub',     fontSize=10, textColor=C_GRAY_MID, leading=14, spaceAfter=3)
    sDate       = S('msDate',    fontSize=9,  textColor=C_GRAY_MID, spaceAfter=8)
    sSecHdr     = S('msSecHdr',  fontSize=13, fontName='Helvetica-Bold', textColor=C_BRAND, spaceBefore=14, spaceAfter=4)
    sWhiteBold  = S('msWB',      fontSize=12, fontName='Helvetica-Bold', textColor=white)
    sWhiteBoldR = S('msWBR',     fontSize=12, fontName='Helvetica-Bold', textColor=white, alignment=TA_RIGHT)
    sWhiteSub   = S('msWSub',    fontSize=9,  textColor=white, leading=12)

    # ---- Document ----
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
        title='Multi-Stock Comparison Report',
    )
    story = []

    # === HEADER ===
    story.append(Paragraph('Multi-Stock Comparison Report', sTitle))
    story.append(Paragraph(f'{n} stock{"s" if n != 1 else ""} analyzed', sSub))
    story.append(Paragraph(f'Report generated: {report_date}', sDate))
    story.append(HRFlowable(width='100%', thickness=2, color=C_BRAND, spaceAfter=10))

    # === RANKING TABLE ===
    story.append(Paragraph('Rankings', sSecHdr))
    rank_badges = ['#1', '#2', '#3'] + [f'#{i}' for i in range(4, n + 1)]
    rank_rows = [['Rank', 'Ticker', 'Score', 'Signal', 'Sector', 'Price', 'Market Cap']]
    for i, r in enumerate(sorted_results):
        fs  = float(r.get('final_score', 0))
        fund = r.get('fundamentals', {})
        rank_rows.append([
            rank_badges[i],
            r.get('ticker', ''),
            f'{fs:.1f}',
            _sl(fs),
            str(fund.get('sector', 'N/A') or 'N/A')[:20],
            _fmt_price(fund.get('price')),
            _fmt_cap(fund.get('market_cap')),
        ])
    rk_col_w = [0.5*inch, 0.75*inch, 0.6*inch, 1.2*inch, 1.85*inch, 0.9*inch, 1.2*inch]
    rk_table  = Table(rank_rows, colWidths=rk_col_w)
    rk_style  = [
        ('BACKGROUND',    (0, 0), (-1, 0), C_BRAND),
        ('TEXTCOLOR',     (0, 0), (-1, 0), white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('GRID',          (0, 0), (-1, -1), 0.4, C_GRAY_LINE),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('FONTNAME',      (1, 1), (1, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (1, 1), (1, -1), C_DARK),
    ]
    for i, r in enumerate(sorted_results, start=1):
        fs  = float(r.get('final_score', 0))
        c   = _sc(fs)
        bg  = C_GRAY_BG if i % 2 == 0 else white
        rk_style += [
            ('BACKGROUND', (3, i), (3, i), c),
            ('TEXTCOLOR',  (3, i), (3, i), white),
            ('FONTNAME',   (3, i), (3, i), 'Helvetica-Bold'),
            ('TEXTCOLOR',  (2, i), (2, i), c),
            ('FONTNAME',   (2, i), (2, i), 'Helvetica-Bold'),
        ]
        for col in [0, 1, 4, 5, 6]:
            rk_style.append(('BACKGROUND', (col, i), (col, i), bg))
    rk_table.setStyle(TableStyle(rk_style))
    story.append(rk_table)
    story.append(Spacer(1, 12))

    # === FINAL SCORE BAR CHART ===
    story.append(Paragraph('Final Score Comparison', sSecHdr))
    try:
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        _cw = 7 * inch
        _ch = 2.0 * inch
        _d  = Drawing(_cw, _ch)
        _bc = VerticalBarChart()
        _bc.x      = 48
        _bc.y      = 30
        _bc.width  = _cw - 68
        _bc.height = _ch - 46
        _tickers_chart = [r.get('ticker', '') for r in sorted_results]
        _fscores_chart = [float(r.get('final_score', 0)) for r in sorted_results]
        _bc.data = [_fscores_chart]
        _bc.categoryAxis.categoryNames = _tickers_chart
        _bc.valueAxis.valueMin  = 0
        _bc.valueAxis.valueMax  = 100
        _bc.valueAxis.valueStep = 25
        _bar_w = max(12, min(35, int((_cw - 120) / max(n, 1))))
        _bc.barWidth = _bar_w
        for _bi, _bs in enumerate(_fscores_chart):
            _bc.bars[0, _bi].fillColor = _sc(_bs)
        _bc.categoryAxis.labels.fontName = 'Helvetica-Bold'
        _bc.categoryAxis.labels.fontSize = 9
        _bc.valueAxis.labels.fontName    = 'Helvetica'
        _bc.valueAxis.labels.fontSize    = 8
        _d.add(_bc)
        story.append(_d)
        story.append(Spacer(1, 8))
    except Exception:
        pass  # chart optional

    # === AGENT SCORES COMPARISON TABLE ===
    story.append(Paragraph('Agent Scores by Stock', sSecHdr))
    _agent_keys_ms = [
        ('value_agent',           'Value'),
        ('growth_momentum_agent', 'Growth / Momentum'),
        ('macro_regime_agent',    'Macro Regime'),
        ('risk_agent',            'Risk'),
        ('sentiment_agent',       'Sentiment'),
    ]
    _ticker_hdrs = [r.get('ticker', '') for r in sorted_results]
    agent_tbl_rows = [['Agent'] + _ticker_hdrs]
    for key, label in _agent_keys_ms:
        row = [label]
        for r in sorted_results:
            s = float(r.get('agent_scores', {}).get(key, 0))
            row.append(f'{s:.1f}')
        agent_tbl_rows.append(row)
    _label_col_w = 1.5 * inch
    _score_col_w = (7.0 * inch - _label_col_w) / max(n, 1)
    at_col_w   = [_label_col_w] + [_score_col_w] * n
    agent_tbl  = Table(agent_tbl_rows, colWidths=at_col_w)
    at_style   = [
        ('BACKGROUND',    (0, 0), (-1, 0), C_BRAND),
        ('TEXTCOLOR',     (0, 0), (-1, 0), white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('FONTNAME',      (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('GRID',          (0, 0), (-1, -1), 0.4, C_GRAY_LINE),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('ALIGN',         (1, 1), (-1, -1), 'CENTER'),
    ]
    for ri, (key, _) in enumerate(_agent_keys_ms, start=1):
        bg = C_GRAY_BG if ri % 2 == 0 else white
        at_style.append(('BACKGROUND', (0, ri), (0, ri), bg))
        for ci, r in enumerate(sorted_results, start=1):
            s  = float(r.get('agent_scores', {}).get(key, 0))
            at_style += [
                ('TEXTCOLOR',  (ci, ri), (ci, ri), _sc(s)),
                ('FONTNAME',   (ci, ri), (ci, ri), 'Helvetica-Bold'),
                ('BACKGROUND', (ci, ri), (ci, ri), bg),
            ]
    agent_tbl.setStyle(TableStyle(at_style))
    story.append(agent_tbl)
    story.append(Spacer(1, 14))

    # === INDIVIDUAL STOCK CARDS ===
    story.append(Paragraph('Individual Stock Summaries', sSecHdr))
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_GRAY_LINE, spaceAfter=6))
    for r in sorted_results:
        ticker_ms   = r.get('ticker', '')
        fund_ms     = r.get('fundamentals', {})
        fs_ms       = float(r.get('final_score', 0))
        c_ms        = _sc(fs_ms)
        _raw_ms     = str(fund_ms.get('name', ticker_ms) or ticker_ms)
        company_ms  = _raw_ms.split('\n')[0][:70]
        rationale_ms = str(r.get('rationale', '')).strip()
        # Truncate rationale to ~600 chars
        if len(rationale_ms) > 600:
            rationale_ms = rationale_ms[:597] + '...'
        # Header banner
        card_hdr = Table([[
            Paragraph(f'<b>{ticker_ms}</b>', sWhiteBold),
            Paragraph(f'<b>{fs_ms:.1f} / 100 \u2014 {_sl(fs_ms)}</b>', sWhiteBoldR),
        ]], colWidths=[3.5*inch, 3.5*inch])
        card_hdr.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), c_ms),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ]))
        # Mini metrics row
        pe_ms   = fund_ms.get('pe_ratio')
        beta_ms = fund_ms.get('beta')
        mini_data = [[
            'P/E',   f'{pe_ms:.1f}'   if pe_ms   else 'N/A',
            'Beta',  f'{beta_ms:.2f}' if beta_ms  else 'N/A',
            'Price', _fmt_price(fund_ms.get('price')),
            'Mkt Cap', _fmt_cap(fund_ms.get('market_cap')),
        ]]
        mini_tbl = Table(mini_data, colWidths=[
            0.55*inch, 0.85*inch, 0.55*inch, 0.85*inch,
            0.55*inch, 1.15*inch, 0.75*inch, 1.75*inch
        ])
        mini_tbl.setStyle(TableStyle([
            ('FONTSIZE',      (0, 0), (-1, -1), 8),
            ('FONTNAME',      (0, 0), (0, 0),   'Helvetica-Bold'),
            ('FONTNAME',      (2, 0), (2, 0),   'Helvetica-Bold'),
            ('FONTNAME',      (4, 0), (4, 0),   'Helvetica-Bold'),
            ('FONTNAME',      (6, 0), (6, 0),   'Helvetica-Bold'),
            ('TEXTCOLOR',     (0, 0), (0, 0),   C_GRAY_MID),
            ('TEXTCOLOR',     (2, 0), (2, 0),   C_GRAY_MID),
            ('TEXTCOLOR',     (4, 0), (4, 0),   C_GRAY_MID),
            ('TEXTCOLOR',     (6, 0), (6, 0),   C_GRAY_MID),
            ('BACKGROUND',    (0, 0), (-1, -1), C_GRAY_BG),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ]))
        rat_para = Paragraph(
            rationale_ms,
            S(f'rat_{ticker_ms}', fontSize=9, leading=13, textColor=C_DARK, spaceAfter=2)
        )
        story.append(KeepTogether([
            card_hdr,
            Spacer(1, 2),
            Paragraph(company_ms, S(f'cn_{ticker_ms}', fontSize=9, textColor=C_GRAY_MID,
                                    spaceBefore=2, spaceAfter=4)),
            mini_tbl,
            Spacer(1, 5),
            rat_para,
            Spacer(1, 12),
        ]))

    # === FOOTER ===
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_GRAY_LINE, spaceBefore=8, spaceAfter=4))
    story.append(Paragraph(
        'This report is generated by the Total Insights Investment Analysis System. '
        'For informational purposes only \u2014 not financial advice.',
        S('ms_footer', fontSize=7, textColor=C_GRAY_MID, alignment=TA_CENTER)
    ))
    doc.build(story)
    return buf.getvalue()


def generate_pdf_report(result: dict) -> bytes:
    """Generate a formatted PDF investment analysis report using ReportLab."""
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )

    # ---- Colour palette ----
    C_DARK      = HexColor('#1a1a2e')
    C_BRAND     = HexColor('#3b5998')
    C_GREEN     = HexColor('#059669')
    C_L_GREEN   = HexColor('#10b981')
    C_YELLOW    = HexColor('#d97706')
    C_ORANGE    = HexColor('#f97316')
    C_RED       = HexColor('#ef4444')
    C_GRAY_BG   = HexColor('#f8f9fa')
    C_GRAY_LINE = HexColor('#e5e7eb')
    C_GRAY_MID  = HexColor('#6b7280')

    def score_color(s):
        if s >= 65:   return C_GREEN
        if s >= 55:   return C_L_GREEN
        if s >= 45:   return C_YELLOW
        if s >= 35:   return C_ORANGE
        return C_RED

    def score_label(s):
        if s >= 65:   return 'STRONG BUY'
        if s >= 55:   return 'BUY'
        if s >= 45:   return 'HOLD'
        if s >= 35:   return 'UNDERPERFORM'
        return 'SELL'

    # ---- Data extraction ----
    ticker        = result.get('ticker', '')
    fund          = result.get('fundamentals', {})
    # Use only the first line / first 80 chars of the name to avoid overflow
    _raw_name     = str(fund.get('name', ticker) or ticker)
    company_name  = _raw_name.split('\n')[0][:80]
    final_score   = float(result.get('final_score', 0))
    agent_scores  = result.get('agent_scores', {})
    agent_rats    = result.get('agent_rationales', {})
    recommendation = score_label(final_score)
    s_color       = score_color(final_score)
    report_date   = datetime.now().strftime('%B %d, %Y')

    agent_display = {
        'value_agent':           'Value',
        'growth_momentum_agent': 'Growth / Momentum',
        'macro_regime_agent':    'Macro Regime',
        'risk_agent':            'Risk',
        'sentiment_agent':       'Sentiment',
    }

    # ---- Styles ----
    styles = getSampleStyleSheet()
    S = lambda name, **kw: ParagraphStyle(name, parent=styles['Normal'], **kw)

    sTitle   = S('sTitle',   fontSize=22, fontName='Helvetica-Bold',
                 textColor=C_DARK,  leading=28, spaceAfter=6)
    sSub     = S('sSub',     fontSize=11, textColor=C_GRAY_MID, leading=16, spaceAfter=3)
    sDate    = S('sDate',    fontSize=9,  textColor=C_GRAY_MID, spaceAfter=8)
    sSecHdr  = S('sSecHdr',  fontSize=13, fontName='Helvetica-Bold',
                 textColor=C_BRAND, spaceBefore=14, spaceAfter=4)
    sBody    = S('sBody',    fontSize=9,  leading=13, spaceAfter=4)
    sCaption = S('sCaption', fontSize=8,  textColor=C_GRAY_MID, spaceAfter=2)
    sAssess  = S('sAssess',  fontSize=14, fontName='Helvetica-Bold',
                 textColor=white, alignment=TA_CENTER)
    sAgentHdr= S('sAgentHdr',fontSize=10, fontName='Helvetica-Bold',
                 textColor=C_DARK, spaceBefore=10, spaceAfter=2)

    # ---- Document ----
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
        title=f"{ticker} Investment Analysis",
    )

    story = []

    # === HEADER ===
    story.append(Paragraph(f"{ticker} — Investment Analysis", sTitle))
    story.append(Paragraph(company_name, sSub))
    story.append(Paragraph(f"Report generated: {report_date}", sDate))
    story.append(HRFlowable(width='100%', thickness=2, color=C_BRAND, spaceAfter=10))

    # === OVERALL ASSESSMENT BANNER ===
    banner_data = [[
        Paragraph(
            f"Overall Assessment: <b>{recommendation}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;Score: <b>{final_score:.1f} / 100</b>",
            sAssess
        )
    ]]
    banner = Table(banner_data, colWidths=[7*inch])
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), s_color),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(banner)
    story.append(Spacer(1, 12))

    # === KEY METRICS TABLE ===
    story.append(Paragraph("Key Metrics", sSecHdr))

    def fmt_val(v, prefix='', suffix='', decimals=2, scale=1):
        if v and v != 0:
            return f"{prefix}{v*scale:.{decimals}f}{suffix}"
        return 'N/A'

    price     = fund.get('price')
    pe        = fund.get('pe_ratio')
    beta      = fund.get('beta')
    eps       = fund.get('eps')
    div_yield = fund.get('dividend_yield')
    low52     = fund.get('week_52_low')
    high52    = fund.get('week_52_high')
    mktcap    = fund.get('market_cap')

    if div_yield and div_yield != 0:
        div_str = f"{div_yield*100:.2f}%" if div_yield < 1 else f"{div_yield:.2f}%"
    else:
        div_str = 'N/A'

    if mktcap:
        if mktcap >= 1e12:   mkt_str = f"${mktcap/1e12:.1f}T"
        elif mktcap >= 1e9:  mkt_str = f"${mktcap/1e9:.1f}B"
        else:                 mkt_str = f"${mktcap/1e6:.0f}M"
    else:
        mkt_str = 'N/A'

    metrics = [
        ['Metric', 'Value', 'Metric', 'Value'],
        ['Current Price',   fmt_val(price, '$'), 'P/E Ratio',      fmt_val(pe, decimals=1)],
        ['EPS',             fmt_val(eps, '$'),   'Beta',           fmt_val(beta)],
        ['Dividend Yield',  div_str,             'Market Cap',     mkt_str],
        ['52-Week Low',     fmt_val(low52, '$'), '52-Week High',   fmt_val(high52, '$')],
    ]

    col_w = [2.0*inch, 1.5*inch, 2.0*inch, 1.5*inch]
    m_table = Table(metrics, colWidths=col_w)
    m_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND',    (0, 0), (-1, 0), C_BRAND),
        ('TEXTCOLOR',     (0, 0), (-1, 0), white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        # Data rows
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [C_GRAY_BG, white]),
        # Label columns bold
        ('FONTNAME',      (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',      (2, 1), (2, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0, 1), (0, -1), C_DARK),
        ('TEXTCOLOR',     (2, 1), (2, -1), C_DARK),
        # Grid
        ('GRID',          (0, 0), (-1, -1), 0.4, C_GRAY_LINE),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 12))

    # === AGENT SCORES TABLE ===
    story.append(Paragraph("Agent Scores", sSecHdr))

    score_rows = [['Agent', 'Score', 'Signal']]
    for key, label in agent_display.items():
        s = float(agent_scores.get(key, 50))
        lbl = score_label(s)
        score_rows.append([label, f"{s:.1f}", lbl])

    s_col_w = [3.5*inch, 1.5*inch, 2.0*inch]
    s_table = Table(score_rows, colWidths=s_col_w)

    s_style = [
        ('BACKGROUND',    (0, 0), (-1, 0), C_BRAND),
        ('TEXTCOLOR',     (0, 0), (-1, 0), white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('GRID',          (0, 0), (-1, -1), 0.4, C_GRAY_LINE),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
    ]
    # Colour the signal cell per score
    for i, key in enumerate(agent_display.keys(), start=1):
        s = float(agent_scores.get(key, 50))
        c = score_color(s)
        s_style += [
            ('BACKGROUND',  (2, i), (2, i), c),
            ('TEXTCOLOR',   (2, i), (2, i), white),
            ('FONTNAME',    (2, i), (2, i), 'Helvetica-Bold'),
        ]
        # Score column: colour text
        s_style += [('TEXTCOLOR', (1, i), (1, i), c),
                    ('FONTNAME',  (1, i), (1, i), 'Helvetica-Bold')]
        # Alternating row bg for agent name col
        bg = C_GRAY_BG if i % 2 == 0 else white
        s_style.append(('BACKGROUND', (0, i), (0, i), bg))
        s_style.append(('BACKGROUND', (1, i), (1, i), bg))

    s_table.setStyle(TableStyle(s_style))
    story.append(s_table)
    story.append(Spacer(1, 14))

    # === AGENT SCORE BAR CHART ===
    story.append(Paragraph("Agent Score Overview", sSecHdr))
    try:
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        _cw = 7 * inch
        _ch = 1.85 * inch
        _d = Drawing(_cw, _ch)
        _bc = VerticalBarChart()
        _bc.x = 45
        _bc.y = 28
        _bc.width = _cw - 65
        _bc.height = _ch - 40
        _agent_keys_chart = [
            'value_agent', 'growth_momentum_agent', 'macro_regime_agent',
            'risk_agent', 'sentiment_agent'
        ]
        _agent_labels_chart = ['Value', 'Growth', 'Macro', 'Risk', 'Sentiment']
        _scores_chart = [float(agent_scores.get(k, 0)) for k in _agent_keys_chart]
        _bc.data = [_scores_chart]
        _bc.categoryAxis.categoryNames = _agent_labels_chart
        _bc.valueAxis.valueMin = 0
        _bc.valueAxis.valueMax = 100
        _bc.valueAxis.valueStep = 25
        for _bi, _bs in enumerate(_scores_chart):
            _bc.bars[0, _bi].fillColor = score_color(_bs)
        _bc.barWidth = 30
        _bc.categoryAxis.labels.fontName = 'Helvetica'
        _bc.categoryAxis.labels.fontSize = 9
        _bc.valueAxis.labels.fontName = 'Helvetica'
        _bc.valueAxis.labels.fontSize = 8
        _d.add(_bc)
        story.append(_d)
        story.append(Spacer(1, 8))
    except Exception:
        pass  # chart is optional – skip silently if drawing fails

    # === AGENT RATIONALES ===
    story.append(Paragraph("Agent Analysis Details", sSecHdr))
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_GRAY_LINE, spaceAfter=6))

    for key, label in agent_display.items():
        s = float(agent_scores.get(key, 50))
        rat = agent_rats.get(key, '')
        if not rat:
            continue

        c = score_color(s)
        # Agent header bar
        hdr_data = [[
            Paragraph(f"<b>{label}</b>", S('ah', fontSize=10, fontName='Helvetica-Bold', textColor=white)),
            Paragraph(f"<b>{s:.1f} / 100 — {score_label(s)}</b>",
                      S('as', fontSize=10, fontName='Helvetica-Bold',
                        textColor=white, alignment=TA_CENTER)),
        ]]
        hdr_table = Table(hdr_data, colWidths=[3.5*inch, 3.5*inch])
        hdr_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), c),
            ('TOPPADDING',    (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether([hdr_table]))

        # Rationale text
        rat_clean = str(rat).replace("\\n", "\n").strip()
        for line in rat_clean.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Bold markdown lines
            if line.startswith('**') and line.endswith('**'):
                line_text = line.strip('*')
                story.append(Paragraph(f"<b>{line_text}</b>",
                                       S('rl', fontSize=9, leading=13,
                                         textColor=C_DARK, spaceBefore=3)))
            else:
                story.append(Paragraph(line,
                                       S('rb', fontSize=9, leading=13,
                                         textColor=C_DARK, spaceAfter=1)))
        story.append(Spacer(1, 8))

    # === FOOTER ===
    story.append(HRFlowable(width='100%', thickness=0.5, color=C_GRAY_LINE, spaceBefore=12, spaceAfter=4))
    story.append(Paragraph(
        "This report is generated by the Total Insights Investment Analysis System. "
        "For informational purposes only — not financial advice.",
        S('footer', fontSize=7, textColor=C_GRAY_MID, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


@st.cache_data(ttl=3600)
def _get_market_context() -> dict:
    """Fetch key market indicators (cached 1 hour)."""
    try:
        import yfinance as yf
        _sp = yf.Ticker("^GSPC").fast_info
        _vix = yf.Ticker("^VIX").fast_info
        return {
            'sp500': getattr(_sp, 'last_price', None),
            'vix': getattr(_vix, 'last_price', None),
        }
    except Exception:
        return {}


def display_stock_analysis(result: dict, show_back_button: bool = True):
    """Display detailed stock analysis results with enhanced rationales."""

    # Market context one-liner banner
    _mkt = _get_market_context()
    _analysis_date = result.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))
    _mkt_parts = [f"Analysis Date: {_analysis_date}"]
    if _mkt.get('sp500'):
        _mkt_parts.append(f"S&P 500: {_mkt['sp500']:,.0f}")
    if _mkt.get('vix'):
        _mkt_parts.append(f"VIX: {_mkt['vix']:.1f}")
    st.caption("  ·  ".join(_mkt_parts))

    # Header with company info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{result['ticker']} - Investment Analysis")
        if 'name' in result['fundamentals']:
            st.caption(result['fundamentals']['name'])
    with col2:
        # Score badge — styled to match design system
        final_score = result['final_score']
        _badge_color = (
            '#059669' if final_score >= 65 else
            '#10b981' if final_score >= 55 else
            '#f59e0b' if final_score >= 45 else
            '#f97316' if final_score >= 35 else
            '#ef4444'
        )
        st.markdown(
            f'<div style="background:{_badge_color};color:white;font-weight:700;font-size:1.15rem;'
            f'padding:10px 16px;border-radius:10px;text-align:center;margin-top:6px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.12);letter-spacing:-0.01em;">'
            f'{final_score:.1f}/100</div>',
            unsafe_allow_html=True,
        )
    
    # --- Overall Assessment ---
    final_score = result['final_score']
    recommendation = result.get('recommendation', '')
    if final_score >= 65:
        _assess_label = 'STRONG BUY'
        _assess_color = '#059669'
        _assess_desc = f'Strong score of {final_score:.1f} indicating excellent investment potential with favorable risk/reward profile.'
    elif final_score >= 55:
        _assess_label = 'BUY'
        _assess_color = '#10b981'
        _assess_desc = f'Solid score of {final_score:.1f} indicating good investment potential with acceptable risk/reward profile.'
    elif final_score >= 45:
        _assess_label = 'HOLD'
        _assess_color = '#f59e0b'
        _assess_desc = f'Moderate score of {final_score:.1f} suggesting a neutral outlook. Monitor for changes before acting.'
    elif final_score >= 35:
        _assess_label = 'UNDERPERFORM'
        _assess_color = '#f97316'
        _assess_desc = f'Below-average score of {final_score:.1f} indicating limited upside and elevated risk.'
    else:
        _assess_label = 'SELL'
        _assess_color = '#ef4444'
        _assess_desc = f'Low score of {final_score:.1f} indicating poor investment potential with unfavorable risk/reward profile.'
    
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;'
        f'padding:18px 24px;margin:12px 0 20px 0;box-shadow:0 1px 4px rgba(0,0,0,0.06);'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif">'
        f'<div style="font-size:13px;color:#6b7280;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:8px">Overall Assessment</div>'
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<span style="background:{_assess_color};color:#ffffff;font-weight:700;font-size:14px;'
        f'padding:4px 14px;border-radius:6px;letter-spacing:0.03em">{_assess_label}</span>'
        f'<span style="font-size:14px;color:#374151;line-height:1.5">{_assess_desc}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )
    
    # Show which weights were used for this analysis
    weight_preset = st.session_state.get('weight_preset', 'equal_weights')
    if weight_preset == 'theory_based' and 'locked_theory_weights' in st.session_state:
        with st.expander("Theory Based Weights Used in This Analysis", expanded=True):
            theory_s = st.session_state.get('theory_settings', {})

            # ── Settings row ──────────────────────────────────────────────
            _hz_map = {
                "short":  "Short-term (3–6 mo)",
                "medium": "Medium-term (6–18 mo)",
                "long":   "Long-term (18+ mo)",
            }
            _rf_map = {
                "capital":           "Capital-weighted",
                "risk_contribution": "Risk-contribution-weighted",
            }
            _rs_map = {
                "conservative": "Conservative (±5pp)",
                "moderate":     "Moderate (±10pp)",
                "aggressive":   "Aggressive (±15pp)",
            }
            _hz = _hz_map.get(theory_s.get('horizon'), "—")
            _rf = _rf_map.get(theory_s.get('risk_framework'), "—")
            _rs = _rs_map.get(theory_s.get('regime_sensitivity'), "—")

            _sc1, _sc2, _sc3 = st.columns(3)
            with _sc1:
                st.metric(
                    label="Investment Horizon",
                    value=_hz,
                    help="How far ahead this analysis is optimized for. "
                         "Longer horizons favor value and quality factors; "
                         "shorter horizons lean on momentum and sentiment.",
                )
            with _sc2:
                st.metric(
                    label="Weighting Method",
                    value=_rf,
                    help="How agent weights are sized relative to each other. "
                         "Capital-weighted gives equal budget to each agent. "
                         "Risk-contribution-weighted scales each agent so that "
                         "higher-volatility signals have less influence on the final score.",
                )
            with _sc3:
                st.metric(
                    label="Regime Sensitivity",
                    value=_rs,
                    help="How much the macro regime is allowed to shift agent weights. "
                         "±5pp (conservative) makes small adjustments; "
                         "±15pp (aggressive) can significantly re-tilt toward growth "
                         "in expansions or toward value/risk in downturns.",
                )

            st.markdown("---")

            # ── Agent descriptions for tooltips ───────────────────────────
            _agent_help = {
                'value':          "Scores the stock on traditional valuation metrics — P/E ratio, EV/EBITDA, and free cash flow yield. A high weight here means cheap stocks score better.",
                'growth_momentum':"Scores earnings and revenue growth plus recent price momentum. A high weight here rewards fast-growing companies with rising share prices.",
                'macro_regime':   "Adjusts the score based on the current economic environment (expansion, recession, high inflation, etc.). Its base weight is small because it mainly shifts the other agents via regime detection.",
                'risk':           "Penalizes high volatility, high beta, and large drawdowns. A high weight here makes the final score more conservative and safety-focused.",
                'sentiment':      "Scores based on recent news, analyst revisions, and earnings surprises. A high weight here means market narrative and near-term catalysts matter more.",
            }

            # ── Base weights ──────────────────────────────────────────────
            base_weights = st.session_state.get('locked_theory_weights', {})
            base_total = sum(base_weights.values()) or 1
            _agent_labels = {
                'value':          'Value',
                'growth_momentum':'Growth & Momentum',
                'macro_regime':   'Macro Regime',
                'risk':           'Risk',
                'sentiment':      'Sentiment',
            }

            st.markdown("**Base Allocation** — how each agent is weighted before any macro adjustment")
            _bcols = st.columns(5)
            for i, (k, lbl) in enumerate(_agent_labels.items()):
                with _bcols[i]:
                    st.metric(
                        label=lbl,
                        value=f"{(base_weights.get(k, 0) / base_total) * 100:.0f}%",
                        help=_agent_help[k],
                    )

            # ── Regime-adjusted weights ───────────────────────────────────
            detected_regime = result.get('detected_regime')
            adj_weights = result.get('regime_adjusted_weights')
            if detected_regime and adj_weights:
                regime_display = detected_regime.replace('_', ' ').title()
                _regime_help = {
                    'Expansion':     "The economy is growing — GDP and employment are rising. The system tilts toward growth and momentum.",
                    'Recession':     "Economic contraction — GDP is falling. The system tilts toward value and defensive/low-risk stocks.",
                    'Stagflation':   "High inflation with slow growth. The system favors value and real assets over growth.",
                    'High Inflation':"Prices rising fast. The system reduces growth exposure and increases value weight.",
                }.get(regime_display, "The detected macro environment that drove the weight adjustment below.")

                st.markdown("---")
                _reg_col, _ = st.columns([2, 3])
                with _reg_col:
                    st.metric(
                        label="Detected Macro Regime",
                        value=regime_display,
                        help=_regime_help,
                    )

                adj_total = sum(adj_weights.values()) or 1
                _adj_key_map = {
                    'value_agent':           'value',
                    'growth_momentum_agent': 'growth_momentum',
                    'macro_regime_agent':    'macro_regime',
                    'risk_agent':            'risk',
                    'sentiment_agent':       'sentiment',
                }
                st.markdown("**Regime-Adjusted Allocation** — final weights used to compute the score")
                _acols = st.columns(5)
                for i, (ak, sk) in enumerate(_adj_key_map.items()):
                    with _acols[i]:
                        base_pct = (base_weights.get(sk, 0) / base_total) * 100
                        adj_pct  = (adj_weights.get(ak, 0) / adj_total) * 100
                        delta    = adj_pct - base_pct
                        st.metric(
                            label=_agent_labels[sk],
                            value=f"{adj_pct:.1f}%",
                            delta=f"{delta:+.1f}pp vs base",
                            help=f"{_agent_help[sk]}\n\n"
                                 f"Base: {base_pct:.0f}% → Regime-adjusted: {adj_pct:.1f}% "
                                 f"({'increased' if delta > 0 else 'decreased' if delta < 0 else 'unchanged'} "
                                 f"by {abs(delta):.1f}pp due to {regime_display} regime).",
                        )

            st.caption(
                "Weights derived from Fama–French (1992), Carhart (1997), and regime-switching "
                "research (Ang & Bekaert, 2002). The final score is a pure weighted average — "
                "no upside multiplier is applied in theory mode."
            )

    elif weight_preset == 'custom_weights' and 'locked_custom_weights' in st.session_state:
        with st.expander("Custom Weights Used in This Analysis", expanded=False):
            st.info("This analysis used custom agent weights to calculate the final score.")
            
            custom_weights = st.session_state.get('locked_custom_weights', {})
            agent_scores = result.get('agent_scores', {})
            
            if custom_weights and agent_scores:
                st.write("**Weight Distribution & Score Contributions:**")
                
                # Calculate total weight and weighted contributions
                total_weight = sum(custom_weights.values())
                
                # Create a detailed breakdown
                breakdown_data = []
                for agent_key in ['value', 'growth_momentum', 'macro_regime', 'risk', 'sentiment']:
                    # Map to agent score keys
                    score_key = f"{agent_key}_agent"
                    score = agent_scores.get(score_key, 50)
                    weight = custom_weights.get(agent_key, 1.0)
                    contribution = score * weight
                    percentage = (weight / total_weight) * 100
                    
                    breakdown_data.append({
                        'Agent': agent_key.replace('_', ' ').title(),
                        'Weight': f"{weight:.1f}x",
                        'Score': f"{score:.1f}",
                        'Contribution': f"{contribution:.1f}",
                        'Influence': f"{percentage:.1f}%"
                    })
                
                df = pd.DataFrame(breakdown_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Calculate and show final score calculation
                weighted_sum = sum(agent_scores.get(f"{k}_agent", 50) * v for k, v in custom_weights.items())
                calculated_final = weighted_sum / total_weight
                actual_final = result.get('final_score', calculated_final)
                
                st.write(f"**Final Score Calculation:**")
                st.code(f"""
                Weighted Sum = {weighted_sum:.2f}
                Total Weight = {total_weight:.2f}
                Blended Score = {weighted_sum:.2f} / {total_weight:.2f} = {calculated_final:.2f}
                Final Score   = {actual_final:.2f}  (after upside/risk adjustments)
                """)
                
                st.caption("Higher weights mean that agent's score had MORE influence on the final score.")
    

    
    # Enhanced key metrics section with modern card-style layout
    st.markdown("### Key Investment Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        final_score = result['final_score']
        delta_color = "normal" if final_score >= 65 else "inverse" if final_score < 45 else "off"
        st.metric("Final Score", f"{final_score:.1f}/100")
    with col2:
        price_value = result['fundamentals'].get('price')
        st.metric("Current Price", f"${price_value:.2f}" if price_value and price_value != 0 else "N/A")
    with col3:
        pe_ratio = result['fundamentals'].get('pe_ratio')
        st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio and pe_ratio != 0 else "N/A", help="Price-to-Earnings ratio: stock price divided by earnings per share")
    with col4:
        beta = result['fundamentals'].get('beta')
        st.metric("Beta", f"{beta:.2f}" if beta and beta != 0 else "N/A", help="Measures stock volatility vs. the market. >1 = more volatile, <1 = less volatile")
    
    # Additional Enhanced Metrics Row
    col5, col6, col7, col8, col9 = st.columns(5)
    with col5:
        div_yield = result['fundamentals'].get('dividend_yield')
        # Dividend yield can be a decimal (0.02 = 2%) or already a percentage (2.0 = 2%)
        if div_yield and div_yield != 0:
            # If it's a small decimal, multiply by 100, otherwise use as-is
            display_yield = div_yield * 100 if div_yield < 1 else div_yield
            st.metric("Dividend Yield", f"{display_yield:.2f}%")
        else:
            st.metric("Dividend Yield", "N/A")
    with col6:
        eps = result['fundamentals'].get('eps')
        if eps and eps != 0:
            st.metric("EPS", f"${eps:.2f}", help="Earnings Per Share: company profit divided by outstanding shares")
        else:
            st.metric("EPS", "N/A", help="Earnings Per Share: company profit divided by outstanding shares")
    with col7:
        week_52_low = result['fundamentals'].get('week_52_low')
        week_52_high = result['fundamentals'].get('week_52_high')
        if week_52_low and week_52_high:
            st.metric("52W Low", f"${week_52_low:.2f}")
        else:
            st.metric("52W Low", "N/A")
    with col8:
        week_52_low = result['fundamentals'].get('week_52_low')
        week_52_high = result['fundamentals'].get('week_52_high')
        if week_52_low and week_52_high:
            st.metric("52W High", f"${week_52_high:.2f}")
        else:
            st.metric("52W High", "N/A")
    with col9:
        market_cap = result['fundamentals'].get('market_cap')
        if market_cap:
            if market_cap >= 1e12:
                st.metric("Market Cap", f"${market_cap/1e12:.1f}T")
            elif market_cap >= 1e9:
                st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
            else:
                st.metric("Market Cap", f"${market_cap/1e6:.0f}M")
        else:
            st.metric("Market Cap", "N/A")
    
    # 52-Week Range Visualization
    week_52_low = result['fundamentals'].get('week_52_low')
    week_52_high = result['fundamentals'].get('week_52_high')
    current_price = result['fundamentals'].get('price')

    if week_52_low and week_52_high and current_price:
        st.subheader("52-Week Price Range")

        # Calculate position of current price within the range
        price_position = (current_price - week_52_low) / (week_52_high - week_52_low)
        price_position = max(0, min(1, price_position))  # Clamp between 0 and 1

        # Determine color based on position
        if price_position >= 0.80:
            bar_color = "#0ea371"
            position_text = "Near 52W High"
        elif price_position >= 0.60:
            bar_color = "#3b82f6"
            position_text = "Upper Range"
        elif price_position >= 0.40:
            bar_color = "#d97706"
            position_text = "Mid Range"
        elif price_position >= 0.20:
            bar_color = "#ea580c"
            position_text = "Lower Range"
        else:
            bar_color = "#dc2626"
            position_text = "Near 52W Low"

        # Single HTML bar with prices embedded at each end and current price marker
        bar_html = f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 600; font-size: 14px; color: #dc2626;">${week_52_low:.2f}</span>
                <span style="font-weight: 600; font-size: 13px; color: #1a1a2e;">Current: ${current_price:.2f}</span>
                <span style="font-weight: 600; font-size: 14px; color: #0ea371;">${week_52_high:.2f}</span>
            </div>
            <div style="position: relative; width: 100%; height: 36px; background-color: #f0f0f5; border-radius: 8px; overflow: visible; border: 1px solid #e5e7eb;">
                <div style="position: absolute; left: 0; top: 0; width: {price_position*100}%; height: 100%; background: {bar_color}; border-radius: 8px 0 0 8px; opacity: 0.85;"></div>
                <div style="position: absolute; left: {price_position*100}%; top: 50%; transform: translate(-50%, -50%); width: 3px; height: 44px; background-color: #1a1a2e; z-index: 10; border-radius: 2px;"></div>
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        # Position info
        st.markdown(f"**{position_text}** - {price_position*100:.1f}% of 52-week range")
    
    # ========== COMPREHENSIVE SCORE ANALYSIS SECTION ==========
    _weight_preset_display = st.session_state.get('weight_preset', 'equal_weights')
    if _weight_preset_display != 'equal_weights':
        st.markdown("---")
        st.markdown("### Score Analysis & Agent Breakdown")

    if _weight_preset_display != 'equal_weights':
     with st.expander("Detailed Breakdown", expanded=False):
        # Get agent scores and weights
        agent_scores = result.get('agent_scores', {})
        blended_score = result.get('blended_score', result.get('final_score', 0))
        
        # Determine which weights were used
        weight_preset = st.session_state.get('weight_preset', 'equal_weights')
        if weight_preset == 'theory_based' and 'locked_theory_weights' in st.session_state:
            weights_used = st.session_state.locked_theory_weights
            weights_source = "Theory Based"
            # If regime-adjusted weights are available, show those instead
            adj_w = result.get('regime_adjusted_weights')
            if adj_w:
                weights_used = {k.replace('_agent', ''): v for k, v in adj_w.items()}
                weights_source = f"Theory Based (regime-adjusted: {(result.get('detected_regime') or 'unknown').replace('_', ' ').title()})"
        elif weight_preset == 'custom_weights' and 'locked_custom_weights' in st.session_state:
            weights_used = st.session_state.locked_custom_weights
            weights_source = "Custom Weights"
        else:
            # Equal weights: every agent has the same weight
            weights_used = {
                'value': 1.0,
                'growth_momentum': 1.0,
                'macro_regime': 1.0,
                'risk': 1.0,
                'sentiment': 1.0
            }
            weights_source = "Equal Weights"
        
        st.write(f"**Weights Source:** {weights_source}")
        st.write("---")
        
        # Calculate weight breakdown
        total_weighted_score = 0
        total_weight = 0
        breakdown_data = []
        
        agent_order = ['value_agent', 'growth_momentum_agent', 'macro_regime_agent', 'risk_agent', 'sentiment_agent']
        agent_labels = {
            'value_agent': 'Value',
            'growth_momentum_agent': 'Growth/Momentum',
            'macro_regime_agent': 'Macro Regime',
            'risk_agent': 'Risk',
            'sentiment_agent': 'Sentiment'
        }
        
        for agent_key in agent_order:
            # Check if the agent was skipped (ETF) or data was unavailable
            agent_result = result.get('agent_results', {}).get(agent_key)
            _agent_skipped = agent_result is None
            _data_unavailable = (agent_result or {}).get('data_unavailable', False)

            score = agent_scores.get(agent_key, 50)

            weight_key = agent_key
            # Get weight - try exact key first, then simplified key
            weight = weights_used.get(weight_key, 1.0)
            if weight == 1.0 and '_agent' in weight_key:
                simplified_key = weight_key.replace('_agent', '')
                weight = weights_used.get(simplified_key, 1.0)

            if _agent_skipped:
                # Agent was not run (e.g. Value/Growth for ETFs)
                breakdown_data.append({
                    'Agent': agent_labels.get(agent_key, agent_key),
                    'Score': 'N/A',
                    'Weight': '0.00x',
                    'Weighted Score': '—',
                    'Influence': '—'
                })
            elif _data_unavailable:
                # Agent ran but data was unavailable (sentiment fallback)
                breakdown_data.append({
                    'Agent': agent_labels.get(agent_key, agent_key) + ' (no data)',
                    'Score': '—',
                    'Weight': '0.00x',
                    'Weighted Score': '—',
                    'Influence': 'redistributed'
                })
            else:
                weighted_contribution = score * weight
                total_weighted_score += weighted_contribution
                total_weight += weight

                breakdown_data.append({
                    'Agent': agent_labels.get(agent_key, agent_key),
                    'Score': f"{score:.1f}",
                    'Weight': f"{weight:.2f}x",
                    'Weighted Score': f"{weighted_contribution:.2f}",
                    'Influence': f"{(weight/sum(w for w in [weights_used.get(k, 1.0) for k in agent_order]))*100:.1f}%"
                })
        
        # Display breakdown table
        st.write("**Individual Agent Contributions:**")
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
        
        # Show calculation
        st.write("---")
        st.write("**Final Score Calculation:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weighted Sum", f"{total_weighted_score:.2f}")
        with col2:
            st.metric("Total Weight", f"{total_weight:.2f}")
        with col3:
            calculated_score = total_weighted_score / total_weight if total_weight > 0 else 50
            st.metric("Blended Score", f"{calculated_score:.2f}", help="Weighted average of all agent scores before upside multiplier")
        
        # Show formula
        actual_final = result.get('final_score', calculated_score)
        _is_theory = weight_preset == 'theory_based'
        _score_note = "(pure weighted average \u2014 upside multiplier disabled)" if _is_theory else "(after upside/risk adjustments)"
        st.code(f"""
Formula: Blended Score = Weighted Sum / Total Weight
         Blended Score = {total_weighted_score:.2f} / {total_weight:.2f} = {calculated_score:.2f}
         Final Score   = {actual_final:.2f}  {_score_note}
        """)
        
        # Weight impact analysis (when custom or theory weights differ from equal)
        is_custom = weight_preset == 'custom_weights' and 'locked_custom_weights' in st.session_state
        is_theory = weight_preset == 'theory_based' and 'locked_theory_weights' in st.session_state
        if is_custom or is_theory:
            st.write("---")
            st.write("**Weight Impact Analysis:**")
            
            # Calculate equal weight score for comparison
            equal_weight_score = sum(float(agent_scores.get(k, 50)) for k in agent_order) / len(agent_order)
            weight_effect = calculated_score - equal_weight_score
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Equal Weight Score", f"{equal_weight_score:.2f}", 
                         help="Score if all agents had equal influence (weight 1.0)")
            with col2:
                st.metric("Weight Effect", f"{weight_effect:+.2f}", 
                         help="How much your custom weights shifted the score vs. equal weights",
                         delta=f"{weight_effect:+.2f}")
            
            if abs(weight_effect) > 0.5:
                _w_label = "Theory-based" if is_theory else "Custom"
                if weight_effect > 0:
                    st.success(f"{_w_label} weights INCREASED the score by {weight_effect:.2f} points by emphasizing higher-scoring agents")
                else:
                    st.warning(f"{_w_label} weights DECREASED the score by {abs(weight_effect):.2f} points by emphasizing lower-scoring agents")
            else:
                _w_label = "Theory-based" if is_theory else "Custom"
                st.info(f"{_w_label} weights had minimal impact on the final score")
        
        # Visual representation
        st.write("---")
        st.write("**Visual Weight Distribution:**")
        
        # Create bar chart of weights
        chart_data = pd.DataFrame({
            'Agent': [agent_labels.get(k, k) for k in agent_order],
            'Weight': [weights_used.get(k, weights_used.get(k.replace('_agent', ''), 1.0)) for k in agent_order],
            'Score': [agent_scores.get(k, 50) for k in agent_order]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            fig_w = go.Figure(go.Bar(
                x=chart_data['Agent'], y=chart_data['Weight'],
                marker_color="#3b5998",
                text=[f"{w:.1f}x" for w in chart_data['Weight']],
                textposition='auto'
            ))
            fig_w.update_layout(yaxis_title="Weight", height=300, showlegend=False,
                                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption("Agent Weights (Higher = More Influence)")
        with col2:
            fig_s = go.Figure(go.Bar(
                x=chart_data['Agent'], y=chart_data['Score'],
                marker_color=[get_gradient_color(s) for s in chart_data['Score']],
                text=[f"{s:.0f}" for s in chart_data['Score']],
                textposition='auto'
            ))
            fig_s.update_layout(yaxis_title="Score", yaxis_range=[0, 100], height=300,
                                showlegend=False, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption("Agent Scores (0-100)")
    
    # Enhanced Agent Analysis Section
    st.markdown("---")
    st.markdown("### Agent Analysis Details")
    
    # Display enhanced agent rationales with collaboration
    display_enhanced_agent_rationales(result)
    
#     # Google Sheets / Docs export
#     _render_google_export(result)

    # ---- PDF Download ----
    st.markdown("---")
    st.markdown("### Download Report")
    try:
        pdf_bytes = generate_pdf_report(result)
        ticker_safe = result.get('ticker', 'analysis').upper()
        pdf_filename = f"{ticker_safe}_Investment_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        st.download_button(
            label="⬇  Download PDF Report",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            type="primary",
            key="download_pdf_report",
        )
    except Exception as _pdf_err:
        st.warning(f"PDF generation unavailable: {_pdf_err}")

    # Back to home
    if show_back_button:
        st.markdown("---")
        _ticker_key = result.get('ticker', 'single').replace('.', '_')
        if st.button("← Back to Home Page", type="secondary", use_container_width=True,
                     key=f"back_home_{_ticker_key}"):
            if '_analysis_params' in st.session_state:
                del st.session_state['_analysis_params']
            if '_display_result' in st.session_state:
                del st.session_state['_display_result']
            _cleanup_display_result_backup()
            st.rerun()


def display_multiple_stock_analysis(results: list, failed_tickers: list):
    """Display analysis results for multiple stocks in a comparison table."""
    
    st.success(f"Successfully analyzed {len(results)} stock{'s' if len(results) != 1 else ''}")
    
    if failed_tickers:
        st.warning(f"Failed to analyze {len(failed_tickers)} stock{'s' if len(failed_tickers) != 1 else ''}")
        with st.expander("View Failed Tickers", expanded=False):
            for ticker_name, error_msg in failed_tickers:
                st.error(f"**{ticker_name}**: {error_msg}")
    
    # Summary comparison
    st.markdown("---")
    st.markdown("### Comparison")
    
    # Prepare data for comparison table
    comparison_data = []
    for result in results:
        row = {
            'Ticker': result['ticker'],
            'Score': round(float(result['final_score']), 1),
            'Signal': result.get('recommendation', 'N/A'),
            'Value': round(float(result.get('agent_scores', {}).get('value_agent', 0)), 1),
            'Growth': round(float(result.get('agent_scores', {}).get('growth_momentum_agent', 0)), 1),
            'Macro': round(float(result.get('agent_scores', {}).get('macro_regime_agent', 0)), 1),
            'Risk': round(float(result.get('agent_scores', {}).get('risk_agent', 0)), 1),
            'Sentiment': round(float(result.get('agent_scores', {}).get('sentiment_agent', 0)), 1),
        }
        comparison_data.append(row)

    comparison_data = sorted(comparison_data, key=lambda x: x['Score'], reverse=True)

    import pandas as pd
    df_display = pd.DataFrame(comparison_data)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format='%.1f'),
            'Value': st.column_config.ProgressColumn('Value', min_value=0, max_value=100, format='%.1f'),
            'Growth': st.column_config.ProgressColumn('Growth', min_value=0, max_value=100, format='%.1f'),
            'Macro': st.column_config.ProgressColumn('Macro', min_value=0, max_value=100, format='%.1f'),
            'Risk': st.column_config.ProgressColumn('Risk', min_value=0, max_value=100, format='%.1f'),
            'Sentiment': st.column_config.ProgressColumn('Sentiment', min_value=0, max_value=100, format='%.1f'),
        },
    )

    # Agent scores comparison chart
    st.markdown("---")
    _agent_categories = ['Value', 'Growth', 'Macro', 'Risk', 'Sentiment']
    _agent_keys = ['value_agent', 'growth_momentum_agent', 'macro_regime_agent', 'risk_agent', 'sentiment_agent']
    fig_bar = go.Figure()
    for result in results:
        fig_bar.add_trace(go.Bar(
            name=result['ticker'],
            x=_agent_categories,
            y=[result['agent_scores'].get(k, 0) for k in _agent_keys],
            text=[f"{result['agent_scores'].get(k, 0):.1f}" for k in _agent_keys],
            textposition='auto',
        ))
    fig_bar.update_layout(
        barmode='group',
        yaxis_range=[0, 100],
        yaxis_title='Score',
        height=360,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Export buttons
    csv = df_display.to_csv(index=False)
    _btn_col1, _btn_col2 = st.columns(2)
    with _btn_col1:
        st.download_button(
            label='Download CSV',
            data=csv,
            file_name=f"comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True,
        )
    with _btn_col2:
        try:
            _multi_pdf = generate_multi_stock_pdf_report(results)
            st.download_button(
                label='Download PDF Report',
                data=_multi_pdf,
                file_name=f"comparison_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime='application/pdf',
                use_container_width=True,
                key='download_multi_pdf',
            )
        except Exception as _mpdf_err:
            st.warning(f'PDF export unavailable: {_mpdf_err}')

    # Individual stock details
    st.markdown("---")
    st.markdown("### Stock Details")
    
    tabs = st.tabs([result['ticker'] for result in results])
    
    for idx, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            display_stock_analysis(result, show_back_button=False)

#     # Google Sheets / Docs export (multi-stock comparison)
#     _render_google_export_multi(results)

    # Back to home
    st.markdown("---")
    if st.button("← Back to Home Page", type="secondary", use_container_width=True, key="back_home_multi"):
        if '_analysis_params' in st.session_state:
            del st.session_state['_analysis_params']
        if '_display_result' in st.session_state:
            del st.session_state['_display_result']
        _cleanup_display_result_backup()
        st.rerun()


def get_gradient_color(score: float) -> str:
    """Generate a polished gradient color based on score (0-100).

    Uses a 5-stop palette of professionally chosen colours:
      0-25  Deep rose-red   → Warm coral
      25-50 Warm coral      → Amber
      50-70 Amber           → Soft teal-green
      70-100 Soft teal-green → Rich emerald
    """
    s = max(0.0, min(100.0, float(score)))

    # Palette stops: (score, R, G, B)
    stops = [
        (0,   200,  60,  60),   # Deep rose-red
        (25,  224, 108,  72),   # Warm coral
        (50,  217, 170,  62),   # Amber / warm gold
        (70,   52, 179, 136),   # Soft teal-green
        (100,  16, 152,  96),   # Rich emerald
    ]

    # Find the two surrounding stops and interpolate
    for i in range(len(stops) - 1):
        lo_s, lo_r, lo_g, lo_b = stops[i]
        hi_s, hi_r, hi_g, hi_b = stops[i + 1]
        if s <= hi_s:
            t = (s - lo_s) / (hi_s - lo_s) if hi_s != lo_s else 0.0
            r = int(lo_r + (hi_r - lo_r) * t)
            g = int(lo_g + (hi_g - lo_g) * t)
            b = int(lo_b + (hi_b - lo_b) * t)
            return f"rgb({r},{g},{b})"

    # Fallback (score == 100)
    return f"rgb({stops[-1][1]},{stops[-1][2]},{stops[-1][3]})"


def get_agent_specific_context(agent_key: str, result: dict) -> dict:
    """Get agent-specific context and key metrics for display."""
    
    fundamentals = result.get('fundamentals', {})
    data = result.get('data', {})
    context = {}
    
    if agent_key == 'value_agent':
        context.update({
            'P/E Ratio': f"{fundamentals.get('pe_ratio'):.1f}" if fundamentals.get('pe_ratio') else 'N/A',
            'Market Cap': f"${fundamentals.get('market_cap', 0)/1e9:.1f}B" if fundamentals.get('market_cap') else 'N/A',
            'Dividend Yield': f"{fundamentals.get('dividend_yield', 0)*100:.2f}%" if fundamentals.get('dividend_yield') else 'N/A',
            'Price': f"${fundamentals.get('price'):.2f}" if fundamentals.get('price') else 'N/A'
        })
    
    elif agent_key == 'growth_momentum_agent':
        context.update({
            'Current Price': f"${fundamentals.get('price'):.2f}" if fundamentals.get('price') else 'N/A',
            '52-Week High': f"${fundamentals.get('week_52_high'):.2f}" if fundamentals.get('week_52_high') else 'N/A',
            '52-Week Low': f"${fundamentals.get('week_52_low'):.2f}" if fundamentals.get('week_52_low') else 'N/A',
            'Volume': f"{fundamentals.get('volume', 'N/A'):,.0f}" if fundamentals.get('volume') else 'N/A'
        })
    
    elif agent_key == 'risk_agent':
        context.update({
            'Beta': f"{fundamentals.get('beta', 'N/A'):.2f}" if fundamentals.get('beta') else 'N/A',
            'Market Cap': f"${fundamentals.get('market_cap', 0)/1e9:.1f}B" if fundamentals.get('market_cap') else 'N/A',
            'Sector': f"{fundamentals.get('sector', 'Unknown')}",
            'Volatility': f"{data.get('volatility', 0)*100:.1f}%" if data.get('volatility') else 'N/A'
        })
    
    elif agent_key == 'sentiment_agent':
        # Get actual displayed article count from sentiment agent details
        agent_results = result.get('agent_results', {})
        sentiment_details = agent_results.get('sentiment_agent', {}).get('details', {})
        article_details_list = sentiment_details.get('article_details', [])
        news_count = len(article_details_list) if article_details_list else sentiment_details.get('num_articles', 0)
        context.update({
            'News Articles Analyzed': f"{news_count}",
            'Sector': f"{fundamentals.get('sector', 'Unknown')}",
            'Recent Price': f"${fundamentals.get('price'):.2f}" if fundamentals.get('price') else 'N/A'
        })
    
    elif agent_key == 'macro_regime_agent':
        context.update({
            'Sector': f"{fundamentals.get('sector', 'Unknown')}",
            'Market Cap Category': get_market_cap_category(fundamentals.get('market_cap', 0)),
            'Beta': f"{fundamentals.get('beta', 'N/A'):.2f}" if fundamentals.get('beta') else 'N/A'
        })
    
    # Remove None values and empty strings
    return {k: v for k, v in context.items() if v is not None and v != 'N/A' and str(v).strip()}


def get_market_cap_category(market_cap: float) -> str:
    """Categorize market cap size."""
    if not market_cap or market_cap == 0:
        return 'Unknown'
    elif market_cap > 200_000_000_000:  # > $200B
        return 'Large Cap'
    elif market_cap > 10_000_000_000:   # $10B - $200B
        return 'Mid Cap'
    else:                               # < $10B
        return 'Small Cap'


def display_enhanced_agent_rationales(result: dict):
    """Display enhanced agent rationales with detailed analysis and collaboration."""
    
    agent_scores = result['agent_scores']
    agent_rationales = result['agent_rationales']

    # Create agent names from keys
    agent_names = [key.replace('_', ' ').title() for key in agent_scores.keys()]
    
    # Agent collaboration results
    collaboration_results = get_agent_collaboration(result)
    
    # Display agent scores chart
    st.write("**Agent Score Overview**")
    
    # Create bar chart with gradient colors 
    fig = go.Figure()
    gradient_colors = [get_gradient_color(score) for score in agent_scores.values()]
    
    fig.add_trace(go.Bar(
        x=agent_names,
        y=list(agent_scores.values()),
        marker_color=gradient_colors,
        text=[f"{s:.1f}" for s in agent_scores.values()],
        textposition='auto',
        name='Scores'
    ))
    
    fig.update_layout(
        title="Agent Analysis Scores",
        xaxis_title="",
        yaxis_title="Score",
        yaxis_range=[0, 100],
        height=350,
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual Agent Rationales Section
    st.write("---")
    st.write("**Individual Agent Analysis**")
    
    # Key mapping from display agent key to orchestrator key (used for agent_details lookup)
    _dq_key_map = {
        'value_agent': 'value',
        'growth_momentum_agent': 'growth_momentum',
        'risk_agent': 'risk',
        'sentiment_agent': 'sentiment',
        'macro_regime_agent': 'macro_regime',
    }

    # Retrieve previous scores for this ticker (if any)
    _result_ticker = result.get('ticker', '')
    _prev_scores = st.session_state.get('_prev_score_history', {}).get(_result_ticker, {})

    # Create detailed rationale display for each agent
    for i, (agent_key, agent_name) in enumerate(zip(agent_scores.keys(), agent_names)):
        score = agent_scores[agent_key]
        rationale = agent_rationales.get(agent_key, "Analysis not available")

        # Resolve data_quality from agent details
        _orch_key = _dq_key_map.get(agent_key, agent_key)
        _agent_det = result.get('agent_details', {}).get(_orch_key, {})
        data_quality = _agent_det.get('data_quality', 1.0)

        # Score delta vs previous run
        _prev_score = _prev_scores.get(agent_key)
        if _prev_score is not None:
            _delta = score - _prev_score
            _delta_str = f" ▲ +{_delta:.1f}" if _delta > 0 else (f" ▼ {_delta:.1f}" if _delta < 0 else " ━ no change")
        else:
            _delta_str = ""

        # Create expandable section for each agent
        with st.expander(f"**{agent_name}** - Score: {score:.1f}/100{_delta_str}", expanded=False):
            col1, col2 = st.columns([1, 3])

            with col1:
                # Score display with gradient color
                score_color = get_gradient_color(score)
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {score_color}, {score_color}aa);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                ">
                    <h2 style="margin: 0; color: white;">{score:.1f}</h2>
                    <p style="margin: 0; color: white;">out of 100</p>
                </div>
                """, unsafe_allow_html=True)

                # Score interpretation
                if score >= 80:
                    st.success("**Excellent**\nStrong positive signals")
                elif score >= 65:
                    st.info("**Good**\nPositive with minor concerns")
                elif score >= 50:
                    st.warning("**Moderate**\nMixed signals")
                elif score >= 35:
                    st.error("**Concerning**\nSeveral negative factors")
                else:
                    st.error("**Poor**\nSignificant issues identified")

                # Data quality warning
                if data_quality < 0.6:
                    st.caption("⚠️ Limited data — score may be less reliable")
            
            with col2:
                # TL;DR: bold first meaningful sentence of the rationale
                if isinstance(rationale, str) and rationale.strip():
                    _clean_rat = rationale.replace("\\n", "\n").strip()
                    # Find first non-header, non-bullet sentence
                    _tldr = ""
                    for _line in _clean_rat.split('\n'):
                        _line = _line.strip().lstrip('•-* #').strip()
                        if _line and not _line.startswith('**') and len(_line) > 20:
                            # Take up to first period
                            _first_period = _line.find('.')
                            _tldr = _line[:_first_period + 1] if _first_period > 0 else _line
                            break
                    if _tldr:
                        st.markdown(f"**{_tldr}**")

                st.write("**Detailed Analysis:**")

                # Display the rationale with proper formatting
                if isinstance(rationale, str) and rationale.strip():
                    formatted_rationale = rationale.replace("\\n", "\n").strip()

                    # Split into paragraphs; render markdown lines as markdown, plain text as write
                    paragraphs = [p.strip() for p in formatted_rationale.split('\n') if p.strip()]

                    for paragraph in paragraphs:
                        if paragraph.startswith(('**', '##', '•', '-', '*')):
                            st.markdown(paragraph)
                        else:
                            st.write(paragraph)
                else:
                    st.info("Detailed rationale not available for this agent.")
                
                # Add agent-specific context based on agent type
                agent_context = get_agent_specific_context(agent_key, result)
                if agent_context:
                    st.write("**Key Metrics:**")
                    for key, value in agent_context.items():
                        if value is not None:
                            st.write(f"• **{key}**: {value}")




def get_agent_collaboration(result: dict) -> dict:
    """Generate collaboration insights between agents."""
    collaboration = {}

    agent_scores = result['agent_scores']

    # Value vs Growth/Momentum collaboration
    if 'value_agent' in agent_scores and 'growth_momentum_agent' in agent_scores:
        value_score = agent_scores['value_agent']
        growth_score = agent_scores['growth_momentum_agent']

        if abs(value_score - growth_score) < 10:
            collaboration['value_agent'] = f"Growth agent agrees (score: {growth_score:.1f}) - balanced value/growth profile"
            collaboration['growth_momentum_agent'] = f"Value agent concurs (score: {value_score:.1f}) - well-balanced investment"
        elif value_score > growth_score + 15:
            collaboration['value_agent'] = f"Growth agent disagrees (score: {growth_score:.1f}) - potential value trap concern"
            collaboration['growth_momentum_agent'] = f"Value agent sees opportunity (score: {value_score:.1f}) - momentum may be lacking"
        else:
            collaboration['value_agent'] = f"Growth agent is more optimistic (score: {growth_score:.1f}) - growth may justify premium"
            collaboration['growth_momentum_agent'] = f"Value agent is cautious (score: {value_score:.1f}) - high growth expectations"
    
    # Risk vs Sentiment collaboration
    if 'risk_agent' in agent_scores and 'sentiment_agent' in agent_scores:
        risk_score = agent_scores['risk_agent']
        sentiment_score = agent_scores['sentiment_agent']
        
        if risk_score > 70 and sentiment_score > 70:
            collaboration['risk_agent'] = f"Sentiment agent confirms (score: {sentiment_score:.1f}) - low risk supported by positive sentiment"
            collaboration['sentiment_agent'] = f"Risk agent validates (score: {risk_score:.1f}) - positive sentiment backed by solid risk profile"
        elif risk_score < 50 and sentiment_score < 50:
            collaboration['risk_agent'] = f"Sentiment agent agrees (score: {sentiment_score:.1f}) - high risk confirmed by negative sentiment"
            collaboration['sentiment_agent'] = f"Risk agent concurs (score: {risk_score:.1f}) - negative sentiment reflects real risks"
    
    return collaboration


def get_detailed_agent_analysis(agent_key: str, result: dict) -> str:
    """Generate detailed analysis for each agent based on available data."""
    
    fundamentals = result['fundamentals']
    ticker = result['ticker']
    score = result['agent_scores'].get(agent_key, 0)
    
    # Map display agent keys to orchestrator agent keys
    agent_key_mapping = {
        'value_agent': 'value',
        'growth_momentum_agent': 'growth_momentum', 
        'risk_agent': 'risk',
        'sentiment_agent': 'sentiment',
        'macro_regime_agent': 'macro_regime'
    }
    
    # Use the mapped key for agent details lookup
    mapped_key = agent_key_mapping.get(agent_key, agent_key)
    
    if agent_key == 'value_agent':
        value_details = result.get('agent_details', {}).get(mapped_key, {})
        component_scores = value_details.get('component_scores', {})
        
        # Use the ACTUAL agent score, not the passed score parameter
        actual_value_score = result['agent_scores'].get(agent_key, score)
        
        pe_ratio = value_details.get('pe_ratio', fundamentals.get('pe_ratio', 0))
        price = fundamentals.get('price', 0)
        sector = fundamentals.get('sector', 'Unknown')
        pe_discount = value_details.get('pe_discount_pct', 0)
        # Get dividend yield, treating 0 as None
        div_yield = value_details.get('dividend_yield_pct', fundamentals.get('dividend_yield'))
        if div_yield == 0:
            div_yield = None
        ev_ebitda = value_details.get('ev_ebitda', 'N/A')
        fcf_yield = value_details.get('fcf_yield_pct', 0)
        
        pe_score = component_scores.get('pe_score', 50)
        ev_score = component_scores.get('ev_ebitda_score', 50)
        fcf_score = component_scores.get('fcf_yield_score', 50)
        yield_score = component_scores.get('shareholder_yield_score', 50)
        
        analysis = f"""
**Comprehensive Value Analysis for {ticker}:**

**Current Valuation Overview:**
Trading at ${price:.2f} with a {actual_value_score:.1f}/100 value score, representing {'exceptional value' if actual_value_score >= 80 else 'good value' if actual_value_score >= 70 else 'fair value' if actual_value_score >= 50 else 'poor value'} opportunity.

**Detailed Valuation Metrics:**

**1. P/E Ratio Analysis** (Score: {pe_score:.1f}/100)
- Current P/E: {pe_ratio:.1f}x
- Sector Premium/Discount: {pe_discount:+.1f}%
- Assessment: {'Excellent discount' if pe_discount > 15 else 'Good discount' if pe_discount > 5 else 'Fair pricing' if pe_discount > -5 else 'Premium pricing' if pe_discount > -15 else 'Significant premium'}
- Implication: {'Strong value signal' if pe_score >= 70 else 'Moderate value signal' if pe_score >= 50 else 'Overvaluation concern'}

**2. EV/EBITDA Multiple** (Score: {ev_score:.1f}/100)
- Current EV/EBITDA: {ev_ebitda if ev_ebitda != 'N/A' else 'Data unavailable'}
- {'Attractive enterprise valuation' if ev_score >= 70 else 'Reasonable valuation' if ev_score >= 50 else 'Expensive enterprise valuation' if ev_ebitda != 'N/A' else 'Unable to assess enterprise value'}

**3. Free Cash Flow Yield** (Score: {fcf_score:.1f}/100)
- FCF Yield: {fcf_yield:.1f}%
- {'Excellent cash generation' if fcf_yield > 8 else 'Good cash flow' if fcf_yield > 5 else 'Moderate cash generation' if fcf_yield > 2 else 'Weak cash flow' if fcf_yield > 0 else 'Negative cash flow'}
- Cash return to investors: {'Very attractive' if fcf_score >= 70 else 'Decent' if fcf_score >= 50 else 'Concerning'}

**4. Dividend Yield & Shareholder Returns** (Score: {yield_score:.1f}/100)
- Dividend Yield: {f'{div_yield*100:.1f}%' if div_yield else 'N/A (likely growth-focused company)'}
- Income Potential: {'High income' if div_yield and div_yield > 0.04 else 'Moderate income' if div_yield and div_yield > 0.02 else 'Low/No dividend income (growth reinvestment strategy)' if div_yield and div_yield > 0 else 'No dividend (growth-focused)'}

**Value Investment Thesis:**
{'Strong value opportunity with multiple attractive metrics supporting potential outperformance' if actual_value_score >= 75 else
 'Reasonable value play with selective attractive characteristics' if actual_value_score >= 60 else
 'Mixed value signals requiring careful analysis' if actual_value_score >= 45 else
 'Limited value appeal with overvaluation concerns'}

**Sector Context ({sector}):**
{sector} sector valuation comparison shows this stock is {'significantly undervalued' if pe_discount > 20 else 'moderately undervalued' if pe_discount > 10 else 'fairly valued' if pe_discount > -10 else 'overvalued'} relative to peers.

**Investment Strategy Implications:**
- **Value Style:** {'Core value holding' if actual_value_score >= 70 else 'Opportunistic value play' if actual_value_score >= 50 else 'Value trap risk'}
- **Time Horizon:** {'Long-term value realization expected' if actual_value_score >= 60 else 'Extended holding period may be required'}
- **Risk/Reward:** {'Favorable risk-adjusted returns potential' if actual_value_score >= 65 else 'Balanced risk/reward profile' if actual_value_score >= 50 else 'Higher risk relative to value potential'}
"""
    
    elif agent_key == 'growth_momentum_agent':
        # Use the ACTUAL agent score, not the passed score parameter
        actual_growth_score = result['agent_scores'].get(agent_key, score)
        
        beta = fundamentals.get('beta', 1.0)
        sector = fundamentals.get('sector', 'Unknown')
        
        analysis = f"""
**Growth & Momentum Analysis for {ticker}:**

Beta coefficient of {beta:.2f} indicates {'higher' if beta > 1.2 else 'moderate' if beta > 0.8 else 'lower'} 
volatility relative to market. {sector} sector positioning provides context for growth expectations.

**Growth Indicators:**
- Revenue Growth: Analyzing recent quarters for acceleration/deceleration
- Earnings Growth: Tracking EPS progression and guidance
- Market Share: Competitive position and expansion opportunities

**Momentum Factors:**
- Technical indicators suggest {'strong positive' if actual_growth_score >= 70 else 'mixed' if actual_growth_score >= 50 else 'weak'} momentum
- Volume and price action analysis
- Relative strength vs sector and market

**Growth Score Reasoning:**
{'Excellent growth trajectory with strong momentum indicators' if actual_growth_score >= 70 else
 'Moderate growth potential with some momentum factors' if actual_growth_score >= 50 else
 'Limited growth visibility and weak momentum signals'}

**Forward Outlook:**
Growth sustainability depends on continued market expansion, competitive advantages, 
and management execution of strategic initiatives.
"""
    
    elif agent_key == 'risk_agent':
        risk_details = result.get('agent_details', {}).get(mapped_key, {})
        component_scores = risk_details.get('component_scores', {})
        
        # Use the ACTUAL agent score, not the passed score parameter
        actual_risk_score = result['agent_scores'].get(agent_key, score)
        
        beta = risk_details.get('beta', fundamentals.get('beta', 1.0))
        sector = fundamentals.get('sector', 'Unknown')
        is_low_risk = risk_details.get('is_low_risk_asset', False)
        risk_boost = risk_details.get('risk_boost_applied', 0)
        volatility = risk_details.get('volatility_pct')
        max_drawdown = risk_details.get('max_drawdown_pct')
        
        vol_score = component_scores.get('volatility_score', 50)
        beta_score = component_scores.get('beta_score', 50)
        dd_score = component_scores.get('drawdown_score', 50)
        div_score = component_scores.get('diversification_score', 50)
        
        analysis = f"""
**Comprehensive Risk Assessment for {ticker}:**

{'**Large-Cap Classification:** Recognized as inherently lower risk due to institutional size, market liquidity, and regulatory oversight.' if is_low_risk else '**Standard Risk Assessment:** Evaluated using traditional risk metrics without size-based adjustments.'}

**Detailed Risk Metrics:**
- **Market Beta:** {beta:.2f} (Score: {beta_score:.1f}/100)
  - {'Market-neutral positioning' if abs(beta - 1.0) < 0.2 else f'{"Higher" if beta > 1.2 else "Lower"} volatility than market'}
  - Systematic risk exposure {'well-controlled' if beta_score >= 70 else 'moderate' if beta_score >= 50 else 'concerning'}

- **Price Volatility:** {f'{volatility:.1f}%' if volatility else 'N/A'} annualized (Score: {vol_score:.1f}/100)
  - {'Low volatility suggests stable price action' if volatility and volatility < 20 else 'Moderate volatility within normal range' if volatility and volatility < 35 else 'High volatility indicates price instability' if volatility else 'Volatility data unavailable'}

- **Maximum Drawdown:** {f'{max_drawdown:.1f}%' if max_drawdown else 'N/A'} (Score: {dd_score:.1f}/100)
  - {'Excellent downside protection' if max_drawdown and max_drawdown > -10 else 'Good risk management' if max_drawdown and max_drawdown > -20 else 'Concerning downside risk' if max_drawdown else 'Historical drawdown data unavailable'}

- **Portfolio Diversification:** Score {div_score:.1f}/100
  - {'Strong diversification benefit for portfolio construction' if div_score >= 70 else 'Moderate diversification value' if div_score >= 50 else 'Limited diversification benefit'}

{'**Institutional Risk Adjustment:** +' + str(risk_boost) + ' points applied recognizing large-cap stability, liquidity advantages, and reduced default risk' if risk_boost > 0 else ''}

**Risk Assessment Summary:**
Based on the {actual_risk_score:.1f}/100 risk score, this asset is classified as {'extremely low risk' if actual_risk_score >= 90 else 'low risk' if actual_risk_score >= 70 else 'moderate risk' if actual_risk_score >= 50 else 'high risk'}. 

**Investment Implications:**
- **Position Sizing:** {'Suitable for large allocations in conservative portfolios' if actual_risk_score >= 80 else 'Appropriate for moderate allocations' if actual_risk_score >= 60 else 'Requires careful position sizing'}
- **Portfolio Role:** {'Core holding providing stability' if is_low_risk else 'Strategic allocation based on risk tolerance'}
- **Monitoring:** {'Standard quarterly review sufficient' if actual_risk_score >= 70 else 'Monthly monitoring recommended' if actual_risk_score >= 50 else 'Active monitoring required'}

**Sector Context ({sector}):**
Technology sector typically exhibits moderate to high volatility but offers growth potential. {'This large-cap position provides sector exposure with reduced volatility' if is_low_risk else 'Standard sector risk characteristics apply'}.
"""
    
    elif agent_key == 'sentiment_agent':
        # Use the ACTUAL agent score, not the passed score parameter
        actual_sentiment_score = result['agent_scores'].get(agent_key, score)
        
        # Get detailed sentiment analysis including articles
        sentiment_details = result.get('agent_details', {}).get(mapped_key, {})
        article_details = sentiment_details.get('article_details', [])
        key_events = sentiment_details.get('key_events', [])
        num_articles = len(article_details) if article_details else sentiment_details.get('num_articles', 0)
        
        analysis = f"""
**Market Sentiment Analysis for {ticker}:**

Analyzed {num_articles} recent articles to assess market sentiment and narrative trends.

**Key Events Detected:**
{', '.join(key_events) if key_events else 'No significant events detected in recent news coverage'}

**Sentiment Score Interpretation:**
{'Very positive sentiment with broad bullish consensus across news sources' if actual_sentiment_score >= 70 else
 'Mixed sentiment with balanced perspectives in media coverage' if actual_sentiment_score >= 50 else
 'Negative sentiment with bearish undertones in recent reporting'}

**Recent News Articles:**"""
        
        if article_details:
            # Rank articles: credible sources first, then most recent
            _TIER_1_SOURCES = {'bloomberg', 'reuters', 'wsj', 'wall street journal',
                               'financial times', 'ft', 'nytimes', 'new york times', 'economist'}
            _TIER_2_SOURCES = {'cnbc', 'barrons', "barron's", 'marketwatch', 'yahoo finance',
                               'yahoo', "investor's business daily", 'investors'}

            def _article_rank(art):
                src = (art.get('source', '') or '').lower()
                if any(s in src for s in _TIER_1_SOURCES):
                    tier = 0
                elif any(s in src for s in _TIER_2_SOURCES):
                    tier = 1
                else:
                    tier = 2
                # Newest first within tier: negate ISO-style date string for desc sort
                pub = art.get('published_at', '') or '0000'
                return (tier, pub)

            # Sort by tier ascending, then date descending (reverse within same tier)
            from itertools import groupby
            ranked_articles = []
            temp = sorted(article_details, key=_article_rank)
            for _tier, group in groupby(temp, key=lambda a: _article_rank(a)[0]):
                tier_list = list(group)
                tier_list.sort(key=lambda a: a.get('published_at', '') or '', reverse=True)
                ranked_articles.extend(tier_list)

            for i, article in enumerate(ranked_articles, 1):
                preview = article.get('preview', '')
                preview_section = f"\n- **Preview:** \"{preview}\"" if preview else ""
                analysis += f"""

**Article {i}: {article['source']}**
- **Title:** {article['title']}
- **Published:** {article['published_at']}{preview_section}
- **Link:** {article['url'] if article['url'] else 'No link available'}
"""
        else:
            analysis += "\n\nNo detailed article information available. Analysis based on headline sentiment only."
        
        analysis += f"""

**Market Implications:**
- News sentiment {'supports' if actual_sentiment_score >= 60 else 'challenges' if actual_sentiment_score <= 40 else 'provides mixed signals for'} current stock valuation
- Media narrative {'reinforces positive' if actual_sentiment_score >= 70 else 'creates neutral' if actual_sentiment_score >= 50 else 'contributes to negative'} investor expectations
- Contrarian opportunities may exist if sentiment reaches extreme levels

**Risk Considerations:**
Sentiment can shift rapidly based on new developments. Monitor for narrative changes that could impact investor perception.
"""
    
    elif agent_key == 'macro_regime_agent':
        # Use the ACTUAL agent score, not the passed score parameter
        actual_macro_score = result['agent_scores'].get(agent_key, score)
        
        analysis = f"""
**Macroeconomic Environment Analysis:**

Current macroeconomic regime assessment and impact on {ticker}:

**Economic Cycle Position:**
- GDP growth trends and economic expansion/contraction signals
- Interest rate environment and Federal Reserve policy stance
- Inflation trends and purchasing power considerations

**Market Regime Analysis:**
- Bull/bear market characteristics and typical sector rotation patterns
- Volatility environment and risk appetite indicators
- Currency trends and international trade considerations

**Sector-Specific Macro Factors:**
- How current macro environment affects {fundamentals.get('sector', 'this')} sector
- Regulatory environment and policy changes
- Global supply chain and commodity price impacts

**Macro Score Rationale:**
{'Favorable macro environment supporting stock performance' if actual_macro_score >= 70 else
 'Mixed macro signals requiring careful monitoring' if actual_macro_score >= 50 else
 'Challenging macro headwinds affecting outlook'}

**Forward-Looking Indicators:**
Monitor leading indicators for regime changes that could impact positioning.
"""
    
    else:
        analysis = f"""
**Comprehensive Analysis for {agent_key.replace('_', ' ').title()}:**

Detailed assessment pending enhanced agent implementation. 
Current score of {score:.1f} reflects preliminary analysis.

Please refer to the basic rationale above for current insights.
"""
    
    return analysis.strip()


def extract_key_factors(agent_key: str, result: dict) -> list:
    """Extract key factors that influenced each agent's decision."""
    
    fundamentals = result['fundamentals']
    score = result['agent_scores'].get(agent_key, 0)
    
    factors = []
    
    if agent_key == 'value_agent':
        pe_ratio = fundamentals.get('pe_ratio', 0)
        if pe_ratio and pe_ratio < 15:
            factors.append(f"Low P/E ratio ({pe_ratio:.1f}) suggests undervaluation")
        elif pe_ratio and pe_ratio > 25:
            factors.append(f"High P/E ratio ({pe_ratio:.1f}) indicates premium valuation")
        
        dividend_yield = fundamentals.get('dividend_yield')
        # Treat 0 as None for dividend yield (0 means no data, not 0% yield)
        if dividend_yield and dividend_yield > 0.03:  # 3% as decimal (0.03)
            factors.append(f"Attractive dividend yield ({dividend_yield*100:.1f}%)")
    
    elif agent_key == 'growth_momentum_agent':
        beta = fundamentals.get('beta', 1.0)
        if beta > 1.5:
            factors.append(f"High beta ({beta:.2f}) indicates strong market sensitivity")
        elif beta < 0.5:
            factors.append(f"Low beta ({beta:.2f}) suggests defensive characteristics")
    
    elif agent_key == 'risk_agent':
        beta = fundamentals.get('beta', 1.0)
        if beta < 0.8:
            factors.append("Low beta suggests reduced portfolio risk")
        elif beta > 1.5:
            factors.append("High beta increases portfolio volatility")
    
    # Add score-based factors
    if score >= 80:
        factors.append("Exceptionally strong fundamentals in this category")
    elif score >= 70:
        factors.append("Strong positive indicators across key metrics")
    elif score < 40:
        factors.append("Significant concerns in multiple areas")
    
    return factors


# def portfolio_recommendations_page():
#     """Portfolio recommendation page with AI-powered selection."""
#     st.header("AI-Powered Portfolio Recommendations")
#     st.write("Multi-stage AI selection using OpenAI o3 and Gemini 2.5 Pro to identify optimal stocks.")
#     st.markdown("---")
#     
#     # Challenge context input
#     st.subheader("Investment Challenge Context")
#     challenge_context = st.text_area(
#         "Describe the investment challenge, goals, and requirements:",
#         value="""Generate an optimal diversified portfolio that maximizes risk-adjusted returns 
# while adhering to the Investment Policy Statement constraints.
# Focus on high-quality companies with strong fundamentals and growth potential.""",
#         height=120,
#         help="Provide detailed context about the investment challenge"
#     )
#     
#     st.markdown("---")
#     
#     # Configuration options
#     with st.expander("Portfolio Configuration", expanded=True):
#         col1, col2 = st.columns(2)
#         
#         with col1:
#             num_positions = st.number_input(
#                 "Target Portfolio Positions",
#                 min_value=3,
#                 max_value=20,
#                 value=5,
#                 help="Target number of holdings in portfolio (up to 20 for diversified growth)"
#             )
#     
#     # Advanced options
#     with st.expander("Investment Focus & Strategy"):
#         col1, col2 = st.columns(2)
#         
#         with col1:
#             st.markdown("**Investment Focus**")
#             focus_value = st.checkbox("Emphasize Value (Undervalued stocks)", value=False)
#             focus_growth = st.checkbox("Emphasize Growth & Momentum", value=False)
#             focus_upside = st.checkbox("Emphasize Potential Upside (High-growth niche stocks)", value=False, 
#                                       help="Discover small-cap and emerging companies with massive growth potential")
#             focus_dividend = st.checkbox("Emphasize Dividend Income", value=False)
#             focus_lowrisk = st.checkbox("Emphasize Low Volatility", value=False)
#         
#         with col2:
#             st.markdown("**Portfolio Strategy**")
#             sector_constraint = st.selectbox(
#                 "Sector Diversification",
#                 ["No Preference", "Tech-Heavy", "Tech-Light", "Diversified Only"],
#                 help="Control sector concentration"
#             )
#             
#             market_cap_pref = st.selectbox(
#                 "Market Cap Preference",
#                 ["All Market Caps (Best opportunities anywhere)", 
#                  "Small & Mid Cap Focus (Higher growth potential)", 
#                  "Large Cap Focus (Established companies)",
#                  "Mix of All Sizes"],
#                 index=0,
#                 help="Define which company sizes to prioritize"
#             )
#     
#     # Build custom instructions from advanced options
#     custom_instructions = []
#     if focus_value:
#         custom_instructions.append("Prioritize value stocks with low P/E ratios, strong fundamentals, and attractive valuations.")
#     if focus_growth:
#         custom_instructions.append("Seek high-growth companies with strong revenue acceleration and momentum indicators.")
#     if focus_upside:
#         custom_instructions.append("CRITICAL: Discover hidden gems - small-cap, mid-cap, and emerging companies with MASSIVE growth potential. Look beyond well-known names. Seek niche players, disruptors, and companies in high-growth sectors (AI, biotech, clean energy, fintech, SaaS, semiconductors). Market cap is NOT a constraint - find the best opportunities regardless of size.")
#     if focus_dividend:
#         custom_instructions.append("Include dividend-paying stocks with sustainable yields above 2%.")
#     if focus_lowrisk:
#         custom_instructions.append("Favor low-beta stocks with reduced volatility and defensive characteristics.")
#     
#     if sector_constraint == "Tech-Heavy":
#         custom_instructions.append("Allocate 40-60% to technology sector stocks.")
#     elif sector_constraint == "Tech-Light":
#         custom_instructions.append("Limit technology sector exposure to 20% maximum.")
#     elif sector_constraint == "Diversified Only":
#         custom_instructions.append("Ensure no single sector exceeds 25% of portfolio weight.")
#     
#     if market_cap_pref == "Small & Mid Cap Focus (Higher growth potential)":
#         custom_instructions.append("Focus primarily on small-cap ($300M-$2B) and mid-cap ($2B-$10B) companies with high growth potential.")
#     elif market_cap_pref == "Large Cap Focus (Established companies)":
#         custom_instructions.append("Focus on large-cap companies ($10B+) with established market positions.")
#     elif market_cap_pref == "Mix of All Sizes":
#         custom_instructions.append("Include a balanced mix of small-cap, mid-cap, and large-cap companies.")
#     else:  # All Market Caps
#         custom_instructions.append("Consider companies of ALL sizes - from small-cap emerging players to mega-cap leaders. Find the best opportunities regardless of market capitalization.")
#     
#     # Append custom instructions to challenge context
#     if custom_instructions:
#         challenge_context += "\n\nAdditional Requirements:\n" + "\n".join(f"- {inst}" for inst in custom_instructions)
#     
#     # AI-powered ticker selection
#     tickers = None
#     st.info("""
#     **AI Selection Process:**
#     1. OpenAI o3 selects 20 best tickers
#     2. Gemini 2.5 Pro selects 20 best tickers
#     3. Aggregate to 40 unique candidates
#     4. Generate 4-sentence rationale for each
#     5. Run 3 rounds of top-5 selection
#     6. Consolidate to final 5 tickers
#     7. Full analysis on all final selections
#     """)
#     
#     if st.button("Generate Portfolio", type="primary", use_container_width=True):
#         with st.spinner("Running AI-powered portfolio generation..."):
#             try:
#                 result = st.session_state.orchestrator.recommend_portfolio(
#                     challenge_context=challenge_context,
#                     tickers=tickers,
#                     num_positions=num_positions
#                 )
# 
#                 # Store result in session state
#                 st.session_state.portfolio_result = result
# 
#                 # Display results
#                 display_portfolio_recommendations(result)
# 
#             except Exception as e:
#                 st.error(f"Portfolio generation failed: {e}")
#                 import traceback
#                 st.code(traceback.format_exc())


# def display_portfolio_recommendations(result: dict):
#     """Display portfolio recommendations with AI selection details."""
#     
#     portfolio = result['portfolio']
#     summary = result['summary']
#     selection_log = result.get('selection_log', {})
#     
#     if not portfolio:
#         st.warning("No stocks found in universe")
#         return
#     
#     # AI Selection Summary (if available)
#     if not selection_log.get('manual_selection', False):
#         st.subheader("AI Selection Process")
#         
#         with st.expander("View AI Selection Details", expanded=False):
#             stages = selection_log.get('stages', [])
#             
#             for stage_info in stages:
#                 stage = stage_info.get('stage', 'Unknown')
#                 
#                 if stage == 'openai_initial_selection':
#                     st.markdown("#### 1. OpenAI Initial Selection")
#                     tickers = stage_info.get('tickers', [])
#                     st.write(f"Selected {len(tickers)} tickers: {', '.join(tickers)}")
#                 
#                 elif stage == 'gemini_initial_selection':
#                     st.markdown("#### 2. Gemini 2.5 Pro Initial Selection")
#                     tickers = stage_info.get('tickers', [])
#                     st.write(f"Selected {len(tickers)} tickers: {', '.join(tickers)}")
#                 
#                 elif stage == 'aggregation':
#                     st.markdown("#### 3. Aggregation")
#                     count = stage_info.get('count', 0)
#                     st.write(f"Total unique candidates: **{count}** tickers")
#                 
#                 elif stage == 'rationale_generation':
#                     st.markdown("#### 4. Rationale Generation")
#                     rationales = stage_info.get('ticker_rationales', {})
#                     st.write(f"Generated 4-sentence rationales for {len(rationales)} tickers")
#                 
#                 elif stage == 'final_selection_rounds':
#                     st.markdown("#### 5. Final Selection Rounds")
#                     round_1 = stage_info.get('round_1', [])
#                     round_2 = stage_info.get('round_2', [])
#                     round_3 = stage_info.get('round_3', [])
#                     
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.write("**Round 1:**")
#                         st.write(", ".join(round_1))
#                     with col2:
#                         st.write("**Round 2:**")
#                         st.write(", ".join(round_2))
#                     with col3:
#                         st.write("**Round 3:**")
#                         st.write(", ".join(round_3))
#                 
#                 elif stage == 'final_consolidation':
#                     st.markdown("#### 6. Final Consolidation")
#                     unique = stage_info.get('unique_finalists', [])
#                     final = stage_info.get('final_5', [])
#                     st.write(f"Unique finalists: {len(unique)} → Final selection: **{len(final)}**")
#                     st.success(f"Final tickers: {', '.join(final)}")
#             
#             # Download log
#             import json
#             log_json = json.dumps(selection_log, indent=2)
#             st.download_button(
#                 label="Download Full Selection Log (JSON)",
#                 data=log_json,
#                 file_name=f"ai_selection_log_{result['analysis_date']}.json",
#                 mime="application/json"
#             )
#         
#         # Download complete archives section
#         st.markdown("---")
#         st.subheader("Complete Archives")
#         st.write("Download all portfolio selection logs and archives from the system.")
#         
#         import os
#         import zipfile
#         from io import BytesIO
#         
#         # Check if portfolio_selection_logs directory exists
#         logs_dir = "portfolio_selection_logs"
#         if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
#             log_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
#             
#             if log_files:
#                 col1, col2 = st.columns([2, 1])
#                 
#                 with col1:
#                     st.info(f"Found **{len(log_files)}** archived portfolio selection(s)")
#                 
#                 with col2:
#                     # Create ZIP file in memory
#                     zip_buffer = BytesIO()
#                     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
#                         # Add all JSON files
#                         for log_file in log_files:
#                             file_path = os.path.join(logs_dir, log_file)
#                             with open(file_path, 'r') as f:
#                                 zip_file.writestr(log_file, f.read())
#                         
#                         # Add README if exists
#                         readme_path = os.path.join(logs_dir, 'README.md')
#                         if os.path.exists(readme_path):
#                             with open(readme_path, 'r') as f:
#                                 zip_file.writestr('README.md', f.read())
#                     
#                     zip_buffer.seek(0)
#                     
#                     st.download_button(
#                         label="Download All Archives (ZIP)",
#                         data=zip_buffer.getvalue(),
#                         file_name=f"portfolio_archives_{result['analysis_date']}.zip",
#                         mime="application/zip",
#                         use_container_width=True,
#                         help="Download all portfolio selection logs as a ZIP file"
#                     )
#                 
#                 # Show list of available archives
#                 with st.expander("View Available Archives", expanded=False):
#                     for log_file in sorted(log_files, reverse=True):
#                         file_path = os.path.join(logs_dir, log_file)
#                         file_size = os.path.getsize(file_path)
#                         file_size_kb = file_size / 1024
#                         
#                         col1, col2, col3 = st.columns([3, 1, 1])
#                         with col1:
#                             st.text(f"{log_file}")
#                         with col2:
#                             st.text(f"{file_size_kb:.1f} KB")
#                         with col3:
#                             # Individual download
#                             with open(file_path, 'r') as f:
#                                 st.download_button(
#                                     label="Download",
#                                     data=f.read(),
#                                     file_name=log_file,
#                                     mime="application/json",
#                                     key=f"download_{log_file}"
#                                 )
#             else:
#                 st.info("No archived selections found yet. Generate a portfolio to create archives.")
#         else:
#             st.warning("Portfolio selection logs directory not found.")
#     
#     st.markdown("---")
#     
#     # Summary metrics
#     st.subheader("Portfolio Summary")
#     
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.metric("Total Positions", summary['num_positions'])
#     with col2:
#         st.metric("Invested Capital", f"{summary['total_weight_pct']:.1f}%")
#     with col3:
#         st.metric("Average Score", f"{summary['avg_score']:.1f}")
#     with col4:
#         st.metric("Selection Method", summary.get('selection_method', 'N/A'))
#     with col5:
#         st.metric("Analyzed", f"{result.get('total_analyzed', 0)}")
#     
#     # Holdings table with AI rationales
#     st.subheader("Portfolio Holdings")
#     
#     for i, holding in enumerate(portfolio, 1):
#         with st.expander(f"{i}. {holding['ticker']} - {holding['name']} ({holding['sector']})", expanded=False):
#             col1, col2 = st.columns([1, 2])
#             
#             with col1:
#                 st.metric("Final Score", f"{holding['final_score']:.1f}/100")
#                 st.metric("Weight", f"{holding['target_weight_pct']:.1f}%")
#                 st.metric("Recommendation", holding['recommendation'])
#             
#             with col2:
#                 st.markdown("**AI Rationale:**")
#                 st.write(holding['rationale'])
#     
#     # Detailed table
#     st.subheader("Holdings Table")
#     df = pd.DataFrame(portfolio)
#     df = df[['ticker', 'name', 'sector', 'final_score', 'target_weight_pct']]
#     df.columns = ['Ticker', 'Name', 'Sector', 'Score', 'Weight %']
# 
#     st.dataframe(
#         df,
#         use_container_width=True,
#         hide_index=True,
#         column_config={
#             "Score": st.column_config.ProgressColumn(
#                 "Score",
#                 help="Final composite score",
#                 min_value=0,
#                 max_value=100,
#             ),
#         }
#     )
#     
#     # Sector allocation
#     st.subheader("Sector Allocation")
#     
#     sector_data = summary['sector_exposure']
#     fig = go.Figure(data=[go.Pie(
#         labels=list(sector_data.keys()),
#         values=list(sector_data.values()),
#         hole=.3,
#         textinfo='label+percent',
#         marker=dict(colors=CHART_COLORS)
#     )])
#     
#     fig.update_layout(height=400, showlegend=True,
#                        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
#     st.plotly_chart(fig, use_container_width=True)
#     
#     # Export
#     st.subheader("Export Portfolio")
#     
#     col1, col2 = st.columns(2)
# 
#     with col1:
#         # Export basic CSV
#         csv = df.to_csv(index=False)
#         st.download_button(
#             label="Download Portfolio CSV",
#             data=csv,
#             file_name=f"portfolio_recommendations_{result['analysis_date']}.csv",
#             mime="text/csv",
#             use_container_width=True
#         )
# 
#     with col2:
#         # Export full analysis JSON
#         import json
#         full_data = {
#             'portfolio': portfolio,
#             'summary': summary,
#             'analysis_date': result['analysis_date'],
#             'selection_log': selection_log
#         }
#         json_data = json.dumps(full_data, indent=2, default=str)
#         st.download_button(
#             label="Download Full Analysis (JSON)",
#             data=json_data,
#             file_name=f"portfolio_full_analysis_{result['analysis_date']}.json",
#             mime="application/json",
#             use_container_width=True
#         )
    



# def system_status_and_ai_disclosure_page():
#     """Combined system status and AI disclosure page."""
#     st.header("System Status & AI Disclosure")
#     st.write("Monitor system health, data provider status, and AI usage information.")
#     st.markdown("---")
#     
#     tab1, tab2 = st.tabs(["System Status", "AI Usage Disclosure"])
#     
#     with tab1:
#         st.subheader("Data Provider Status")
#         
#         # Check if data provider is available
#         if not st.session_state.data_provider:
#             st.error("Data provider not initialized. Please restart the application.")
#             return
#         
#         data_provider = st.session_state.data_provider
#         
#         # Display Data Provider Information
#         st.write("**Provider Information**")
#         col1, col2, col3 = st.columns(3)
#         
#         with col1:
#             st.metric("Provider Type", "Enhanced Data Provider")
#         
#         with col2:
#             # Check if provider has premium services
#             has_polygon = hasattr(data_provider, 'polygon_client') and data_provider.polygon_client is not None
#             has_gemini = bool(os.getenv('GEMINI_API_KEY'))
#             premium_count = sum([has_polygon, has_gemini])
#             st.metric("Premium Services", f"{premium_count}/2 Available")
#         
#         with col3:
#             # Check cache directory
#             cache_dir = Path("data/cache")
#             cache_exists = cache_dir.exists()
#             st.metric("Cache Status", "Available" if cache_exists else "Not Found")
#         
#         # API Keys Status
#         st.markdown("---")
#         st.write("**API Keys Status**")
#         
#         api_keys_status = {
#             "Alpha Vantage": bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
#             "OpenAI": bool(os.getenv('OPENAI_API_KEY')),
#             "Polygon.io": bool(os.getenv('POLYGON_API_KEY')),
#             "Gemini AI": bool(os.getenv('GEMINI_API_KEY')),
#             "NewsAPI": bool(os.getenv('NEWSAPI_KEY')),
#             "IEX Cloud": bool(os.getenv('IEX_TOKEN'))
#         }
#         
#         cols = st.columns(3)
#         for i, (service, available) in enumerate(api_keys_status.items()):
#             with cols[i % 3]:
#                 icon = "" if available else ""
#                 status_text = "Available" if available else "Missing"
#                 st.write(f"{icon} **{service}**: {status_text}")
#         
#         # Provider Capabilities
#         st.markdown("---")
#         st.write("**Provider Capabilities**")
#         
#         capabilities = {
#             "Stock Price Data": True,
#             "Fundamentals Data": True,
#             "News & Sentiment": bool(os.getenv('NEWSAPI_KEY')),
#             "Premium Price Data": bool(os.getenv('POLYGON_API_KEY')),
#             "AI-Enhanced Analysis": bool(os.getenv('GEMINI_API_KEY')),
#             "52-Week Range Verification": True,
#             "Multi-Source Fallback": True
#         }
#         
#         col1, col2 = st.columns(2)
#         for i, (capability, available) in enumerate(capabilities.items()):
#             with col1 if i % 2 == 0 else col2:
#                 icon = "" if available else ""
#                 st.write(f"{icon} {capability}")
#         
#         # Cache Information
#         if cache_exists:
#             st.markdown("---")
#             st.write("**Cache Information**")
#             try:
#                 cache_files = list(cache_dir.glob("*"))
#                 total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
#                 total_size_mb = total_size / (1024 * 1024)
#                 
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Cache Files", len(cache_files))
#                 with col2:
#                     st.metric("Total Size", f"{total_size_mb:.1f} MB")
#                 with col3:
#                     # Show newest cache file age
#                     if cache_files:
#                         newest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
#                         current_time = datetime.now().timestamp()
#                         age_hours = (current_time - newest_file.stat().st_mtime) / 3600
#                         st.metric("Newest Cache", f"{age_hours:.1f} hours ago")
#             except Exception as e:
#                 st.warning(f"Could not read cache information: {e}")
#         
#         # Data Source Test
#         st.markdown("---")
#         st.write("**Test Data Sources**")
#         
#         test_ticker = st.text_input("Test ticker:", value="AAPL")
#         
#         if st.button("Test All Data Sources"):
#             with st.spinner("Testing data sources..."):
#                 results = {}
#                 
#                 # Test price data
#                 try:
#                     if hasattr(st.session_state.data_provider, 'get_price_history_enhanced'):
#                         price_data = st.session_state.data_provider.get_price_history_enhanced(
#                             test_ticker, "2024-01-01", "2024-12-31"
#                         )
#                     else:
#                         price_data = st.session_state.data_provider.get_price_history(
#                             test_ticker, "2024-01-01", "2024-12-31"
#                         )
#                     
#                     if not price_data.empty:
#                         results['Price Data'] = f"{len(price_data)} days of data"
#                         if 'SYNTHETIC_DATA' in price_data.columns:
#                             results['Price Data'] += " (Synthetic)"
#                     else:
#                         results['Price Data'] = "No data"
#                         
#                 except Exception as e:
#                     results['Price Data'] = f"Error: {str(e)}"
#                 
#                 # Test fundamentals
#                 try:
#                     if hasattr(st.session_state.data_provider, 'get_fundamentals_enhanced'):
#                         fund_data = st.session_state.data_provider.get_fundamentals_enhanced(test_ticker)
#                     else:
#                         fund_data = st.session_state.data_provider.get_fundamentals(test_ticker)
#                     
#                     if fund_data:
#                         results['Fundamentals'] = f"{len(fund_data)} data points"
#                         if fund_data.get('estimated'):
#                             results['Fundamentals'] += " (Estimated)"
#                     else:
#                         results['Fundamentals'] = "No data"
#                         
#                 except Exception as e:
#                     results['Fundamentals'] = f"Error: {str(e)}"
#                 
#                 # Display results
#                 for source, result in results.items():
#                     st.write(f"**{source}:** {result}")
#         
#         # Clear Cache
#         if st.button("Clear Cache", help="Clear cached data to force fresh API calls"):
#             cache_dir = Path("data/cache")
#             if cache_dir.exists():
#                 import shutil
#                 shutil.rmtree(cache_dir)
#                 cache_dir.mkdir(parents=True, exist_ok=True)
#                 st.success("Cache cleared!")
#             else:
#                 st.info("No cache to clear")
#     
#     with tab2:
#         st.subheader("AI Usage Disclosure")
#         
#         disclosure_logger = get_disclosure_logger()
#         summary = disclosure_logger.get_disclosure_summary()
#         
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total API Calls", summary['total_calls'])
#         with col2:
#             st.metric("Total Tokens", f"{summary['total_tokens']:,}")
#         with col3:
#             st.metric("Estimated Cost", f"${summary['total_cost_usd']:.2f}")
#         
#         st.write(f"**Tools Used:** {', '.join(summary.get('tools_used', []))}")
#         st.write(f"**Log File:** `{summary.get('log_file', 'N/A')}`")
#         
#         # Download log
#         log_file = summary.get('log_file', '')
#         if log_file and Path(log_file).exists():
#             with open(log_file, 'r') as f:
#                 log_data = f.read()
#             
#             st.download_button(
#                 label="Download Disclosure Log",
#                 data=log_data,
#                 file_name="ai_disclosure_log.jsonl",
#                 mime="application/json"
#             )
#         
#         st.info("""
#         **For Works Cited:**
#         
#         This system uses the following APIs/tools:
#         - OpenAI o3 for agent reasoning and portfolio selection, Gemini 2.5 Pro for AI-powered ticker discovery and real-time data synthesis. 
#         - Polygon.io for market data
#         - Alpha Vantage for fundamental data and macroeconomic indicators
#         - NewsAPI for news sentiment analysis 
#         
#         All API calls are logged with timestamps, purposes, and token usage for full disclosure.
#         """)
#         
#         # Premium Setup Guide
#         st.markdown("---")
#         st.subheader("Premium Setup Guide")
#         
#         with st.expander("View Premium API Setup Instructions"):
#             st.markdown("""
#             ### Recommended Premium APIs for Production
#             
#             **For reliable data access without rate limits:**
#             
#             1. **IEX Cloud** ($9/month) - Excellent US stock data
#                - Add to .env: `IEX_TOKEN=your_token_here`
#                - Get token: https://iexcloud.io/
#             
#             2. **Alpha Vantage Premium** ($49.99/month) - Comprehensive fundamentals  
#                - Upgrade your existing key at: https://www.alphavantage.co/premium/
#                - 1200 calls/minute vs 5 calls/minute free
#             
#             3. **Polygon.io** ($99/month) - Professional grade data
#                - Add to .env: `POLYGON_API_KEY=your_key_here` 
#                - Get key: https://polygon.io/
#             
#             **Total recommended cost: ~$60/month for rock-solid data access**
#             
#             ### Current Free Tier Limitations:
#             - Alpha Vantage: 5 calls/minute, 500/day
#             - NewsAPI: 100 requests/day
#             
#             ### Testing vs Production:
#             - Free tier works fine for testing and development
#             - Premium recommended for live trading or intensive analysis
#             """)


# def configuration_page():
#     """Configuration management page."""
#     st.header("System Configuration")
#     st.write("Manage analysis constraints and model parameters.")
#     st.markdown("---")
# 
#     tab1, tab2, tab3 = st.tabs(["Analysis Configuration", "Agent Weights", "Timing Analytics"])
# 
#     with tab1:
#         st.subheader("Analysis Configuration")
#         st.write("Configure analysis constraints and parameters.")
#         
#         # Load current IPS
#         ips = st.session_state.config_loader.load_ips()
#         
#         col1, col2 = st.columns(2)
#         
#         with col1:
#             st.write("**Position & Sector Constraints:**")
#             max_position = st.number_input(
#                 "Max Single Position (%)", 
#                 value=float(ips.get('position_limits', {}).get('max_position_pct', 10.0)), 
#                 min_value=1.0, 
#                 max_value=50.0
#             )
#             max_sector = st.number_input(
#                 "Max Sector Allocation (%)", 
#                 value=float(ips.get('position_limits', {}).get('max_sector_pct', 30.0)), 
#                 min_value=10.0, 
#                 max_value=100.0
#             )
#             
#             st.write("**Price & Market Cap:**")
#             min_price = st.number_input(
#                 "Min Stock Price ($)", 
#                 value=float(ips.get('universe', {}).get('min_price', 1.0)), 
#                 min_value=0.0
#             )
#             min_market_cap = st.number_input(
#                 "Min Market Cap ($B)", 
#                 value=float(ips.get('universe', {}).get('min_market_cap', 1000000000)) / 1000000000, 
#                 min_value=0.0
#             )
#         
#         with col2:
#             st.write("**Risk Parameters:**")
#             min_beta = st.number_input(
#                 "Min Beta", 
#                 value=float(ips.get('portfolio_constraints', {}).get('beta_min', 0.7)), 
#                 min_value=0.0, 
#                 max_value=3.0
#             )
#             max_beta = st.number_input(
#                 "Max Beta", 
#                 value=float(ips.get('portfolio_constraints', {}).get('beta_max', 1.3)), 
#                 min_value=0.0, 
#                 max_value=3.0
#             )
#             max_volatility = st.number_input(
#                 "Max Portfolio Volatility (%)", 
#                 value=float(ips.get('portfolio_constraints', {}).get('max_portfolio_volatility', 18.0)), 
#                 min_value=0.0, 
#                 max_value=50.0
#             )
#         
#         st.write("**Excluded Sectors:**")
#         current_exclusions = ips.get('exclusions', {}).get('sectors', [])
#         excluded_sectors = st.multiselect(
#             "Select sectors to exclude",
#             options=["Energy", "Financials", "Healthcare", "Technology", "Consumer Staples", "Consumer Discretionary", 
#                     "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services", "Tobacco", "Weapons"],
#             default=current_exclusions
#         )
#         
#         if st.button("Save Configuration"):
#             # Update IPS with proper structure
#             if 'position_limits' not in ips:
#                 ips['position_limits'] = {}
#             ips['position_limits']['max_position_pct'] = max_position
#             ips['position_limits']['max_sector_pct'] = max_sector
#             
#             if 'universe' not in ips:
#                 ips['universe'] = {}
#             ips['universe']['min_price'] = min_price
#             ips['universe']['min_market_cap'] = min_market_cap * 1000000000
#             
#             if 'portfolio_constraints' not in ips:
#                 ips['portfolio_constraints'] = {}
#             ips['portfolio_constraints']['beta_min'] = min_beta
#             ips['portfolio_constraints']['beta_max'] = max_beta
#             ips['portfolio_constraints']['max_portfolio_volatility'] = max_volatility
#             
#             if 'exclusions' not in ips:
#                 ips['exclusions'] = {}
#             ips['exclusions']['sectors'] = excluded_sectors
# 
#             st.session_state.config_loader.save_ips(ips)
#             st.success("Configuration saved!")
# 
#     with tab2:
#         st.subheader("Agent Weights")
#         st.write("Adjust how much each agent influences the final score.")
#         
#         # Load current weights
#         model_config = st.session_state.config_loader.load_model_config()
#         weights = model_config['agent_weights']
#         
#         new_weights = {}
#         for agent, weight in weights.items():
#             new_weights[agent] = st.slider(
#                 f"{agent.replace('_', ' ').title()}",
#                 min_value=0.0,
#                 max_value=3.0,
#                 value=float(weight),
#                 step=0.1,
#                 help=f"Current weight: {weight}"
#             )
#         
#         if st.button("Save Agent Weights"):
#             st.session_state.config_loader.update_model_weights(new_weights)
#             st.success("Agent weights updated!")
#             st.info("System will be reinitialized on next analysis.")
#             st.session_state.initialized = False
#     
#     with tab3:
#         st.subheader("Analysis Timing Analytics")
#         st.write("Deep insights into step-level timing data collected from all analyses.")
#         
#         if hasattr(st.session_state, 'step_time_manager'):
#             manager = st.session_state.step_time_manager
#             
#             # Summary statistics
#             col1, col2, col3 = st.columns(3)
#             
#             total_samples = sum(len(manager.step_times.get(i, [])) for i in range(1, 11))
#             all_stats = manager.get_all_stats()
#             
#             with col1:
#                 st.metric("Total Data Points", f"{total_samples:,}")
#             
#             with col2:
#                 steps_tracked = len(all_stats)
#                 st.metric("Steps Tracked", f"{steps_tracked}/10")
#             
#             with col3:
#                 if all_stats:
#                     avg_analysis_time = sum(s['avg'] for s in all_stats.values())
#                     st.metric("Est. Analysis Time", f"{avg_analysis_time:.1f}s")
#                 else:
#                     st.metric("Est. Analysis Time", "No data")
#             
#             st.markdown("---")
#             
#             # Detailed step breakdown
#             st.subheader("Step-by-Step Breakdown")
#             
#             step_names = {
#                 1: "Data Gathering - Fundamentals",
#                 2: "Data Gathering - Market Data",
#                 3: "Value Agent Analysis",
#                 4: "Growth Agent Analysis",
#                 5: "Macro Regime Agent Analysis",
#                 6: "Risk Agent Analysis",
#                 7: "Sentiment Agent Analysis",
#                 8: "Score Blending",
#                 9: "Finalizing",
#                 10: "Final Analysis"
#             }
#             
#             if all_stats:
#                 for step in sorted(all_stats.keys()):
#                     stats = all_stats[step]
#                     name = step_names.get(step, f"Step {step}")
#                     
#                     with st.expander(f"**{name}**", expanded=False):
#                         col1, col2, col3, col4 = st.columns(4)
#                         
#                         with col1:
#                             st.metric("Samples", stats['count'])
#                             st.metric("Average", f"{stats['avg']:.2f}s")
#                         
#                         with col2:
#                             st.metric("Median", f"{stats['median']:.2f}s")
#                             st.metric("Std Dev", f"{stats['std_dev']:.2f}s")
#                         
#                         with col3:
#                             st.metric("Minimum", f"{stats['min']:.2f}s")
#                             st.metric("Maximum", f"{stats['max']:.2f}s")
#                         
#                         with col4:
#                             st.metric("25th %ile", f"{stats['p25']:.2f}s")
#                             st.metric("75th %ile", f"{stats['p75']:.2f}s")
#                 
#                 st.markdown("---")
#                 
#                 # Export option
#                 if st.button("Export Timing Data"):
#                     import pandas as pd
#                     from datetime import datetime
#                     
#                     export_data = []
#                     for step, stats in all_stats.items():
#                         export_data.append({
#                             'Step': step,
#                             'Name': step_names.get(step, f"Step {step}"),
#                             'Count': stats['count'],
#                             'Average': stats['avg'],
#                             'Median': stats['median'],
#                             'Std_Dev': stats['std_dev'],
#                             'Min': stats['min'],
#                             'Max': stats['max'],
#                             'P25': stats['p25'],
#                             'P75': stats['p75']
#                         })
#                     
#                     df = pd.DataFrame(export_data)
#                     csv_data = df.to_csv(index=False)
#                     
#                     st.download_button(
#                         label="Download Timing Data CSV",
#                         data=csv_data,
#                         file_name=f"timing_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                         mime="text/csv"
#                     )
#             else:
#                 st.info("No timing data available yet. Run some analyses to collect timing statistics.")
#         else:
#             st.warning("Step time manager not initialized.")



if __name__ == "__main__":
    main()
