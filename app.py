"""
Wharton Investing Challenge - Multi-Agent Investment Analysis System
Main Streamlit Application

This is the main entry point for the investment analysis system.
Provides a web interface for stock analysis, portfolio recommendations, and backtesting.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from pathlib import Path
import yaml
import json

# Setup page config
st.set_page_config(
    page_title="Wharton Investment Analysis System",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Import system components
from dotenv import load_dotenv
from utils.config_loader import get_config_loader
from utils.logger import setup_logging, get_disclosure_logger
from utils.qa_system import QASystem, RecommendationType
from utils.google_sheets_integration import get_sheets_integration
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

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_provider = None
    st.session_state.orchestrator = None
    st.session_state.config_loader = None
    st.session_state.client_data = None
    st.session_state.qa_system = None
    st.session_state.sheets_integration = None
    st.session_state.sheets_enabled = False
    st.session_state.sheets_auto_update = False
    st.session_state.show_sheets_export = False


def get_client_profile_weights(client_name: str) -> dict:
    """Derive agent weights from detailed client profile characteristics and specific metrics."""
    from utils.client_profile_manager import ClientProfileManager
    
    profile_manager = ClientProfileManager()
    profile_data = profile_manager.load_client_profile(client_name)
    
    if not profile_data:
        # Default equal weights if profile not found
        return {
            'value': 1.0,
            'growth_momentum': 1.0,
            'macro_regime': 1.0,
            'risk': 1.0,
            'sentiment': 1.0
        }
    
    profile_text = profile_data.get('profile_text', '').lower()
    ips_data = profile_data.get('ips_data', {})
    weights = {'value': 1.0, 'growth_momentum': 1.0, 'macro_regime': 1.0, 'risk': 1.0, 'sentiment': 1.0}
    
    # SPECIFIC METRIC-BASED ANALYSIS
    
    # 1. Risk Tolerance Analysis (from exact risk_tolerance field)
    risk_tolerance = ips_data.get('risk_tolerance', 'moderate')
    if risk_tolerance == 'conservative' or 'conservative' in profile_text:
        weights['value'] = 1.6      # Strong emphasis on value metrics
        weights['risk'] = 1.9       # Very high risk assessment priority
        weights['growth_momentum'] = 0.5  # De-emphasize momentum plays
        weights['sentiment'] = 0.4  # Ignore market sentiment noise
        weights['macro_regime'] = 1.3     # Monitor macro for safety
    elif risk_tolerance == 'aggressive' or any(word in profile_text for word in ['aggressive', 'high-risk', 'growth']):
        weights['growth_momentum'] = 1.9  # Prioritize growth opportunities
        weights['sentiment'] = 1.6        # Use market sentiment for timing
        weights['value'] = 0.6           # Lower focus on traditional value
        weights['risk'] = 0.7           # Accept higher risk for returns
        weights['macro_regime'] = 1.1   # Standard macro monitoring
    
    # 2. Time Horizon Analysis (from exact time_horizon_years)
    time_horizon = ips_data.get('time_horizon_years', 5)
    if time_horizon >= 10:  # Long-term (10+ years)
        weights['value'] *= 1.3          # Value compounds over time
        weights['growth_momentum'] *= 1.2 # Growth more important long-term
        weights['sentiment'] *= 0.6      # Less relevant for long horizons
        weights['macro_regime'] *= 0.9   # Macro cycles matter less long-term
    elif time_horizon <= 3:  # Short-term (3 years or less)
        weights['sentiment'] *= 1.4      # Market timing more critical
        weights['macro_regime'] *= 1.4   # Economic cycles highly relevant
        weights['risk'] *= 1.3          # Downside protection crucial
        weights['growth_momentum'] *= 0.8 # Less time for growth to materialize
    
    # 3. Return Requirements Analysis (from required_growth_rate)
    required_return = ips_data.get('required_growth_rate', 8.0)
    if required_return >= 12.0:  # High return requirement (12%+)
        weights['growth_momentum'] *= 1.5 # Need strong growth stocks
        weights['sentiment'] *= 1.3      # Use sentiment for alpha generation
        weights['value'] *= 0.8         # Value may not meet return needs
        weights['risk'] *= 0.9         # Must accept higher risk
    elif required_return <= 6.0:  # Conservative return target (6% or less)
        weights['value'] *= 1.4         # Focus on undervalued, stable stocks
        weights['risk'] *= 1.5         # Prioritize capital preservation
        weights['growth_momentum'] *= 0.7 # Don't need high growth
        weights['sentiment'] *= 0.7     # Less aggressive positioning
    
    # 4. Drawdown Requirements Analysis (from annual_drawdown)
    has_drawdowns = ips_data.get('annual_drawdown', 0) > 0
    if has_drawdowns:
        weights['risk'] *= 1.4          # Liquidity and stability critical
        weights['value'] *= 1.2        # Need reliable dividend/cash flow
        weights['macro_regime'] *= 1.2  # Economic conditions affect liquidity
        weights['growth_momentum'] *= 0.8 # Can't afford volatile growth stocks
        weights['sentiment'] *= 0.8     # Less speculation allowed
    
    # 5. Mission/ESG Focus Analysis (from foundation_mission and focus_areas)
    has_mission = bool(ips_data.get('foundation_mission') or ips_data.get('focus_areas'))
    if has_mission:
        weights['risk'] *= 1.2          # ESG screening may limit universe
        weights['value'] *= 1.1        # Need sustainable business models
        weights['macro_regime'] *= 1.1  # Policy/regulation impacts ESG stocks
        weights['sentiment'] *= 0.9     # Less focus on pure market dynamics
    
    # 6. Capital Size Analysis (from initial_capital)
    initial_capital = ips_data.get('initial_capital', 100000)
    if initial_capital >= 1000000:  # Large portfolios ($1M+)
        weights['macro_regime'] *= 1.2  # Macro factors more impactful
        weights['risk'] *= 1.1         # Institutional-level risk management
        weights['sentiment'] *= 0.9     # Less tactical, more strategic
    elif initial_capital <= 100000:  # Smaller portfolios (<$100K)
        weights['growth_momentum'] *= 1.2 # Need higher growth to build wealth
        weights['sentiment'] *= 1.1      # More tactical opportunities
        weights['value'] *= 0.9         # May not have luxury of patience
    
    # 7. Organization Type Analysis (from organization field)
    org_type = ips_data.get('organization', '').lower()
    if 'foundation' in org_type or 'nonprofit' in org_type:
        weights['risk'] *= 1.3          # Fiduciary responsibility
        weights['value'] *= 1.2        # Need sustainable returns
        weights['growth_momentum'] *= 0.9 # Less aggressive growth seeking
        weights['sentiment'] *= 0.7     # Avoid speculative positioning
    
    # 8. Professional Background Analysis (from background field)
    background = ips_data.get('background', '').lower()
    if any(word in background for word in ['finance', 'investment', 'business']):
        weights['macro_regime'] *= 1.2  # Sophisticated macro understanding
        weights['sentiment'] *= 1.1     # Can handle market timing
    elif any(word in background for word in ['teacher', 'government', 'nonprofit']):
        weights['risk'] *= 1.3          # Conservative, steady approach
        weights['value'] *= 1.2        # Prefer understandable investments
        weights['sentiment'] *= 0.8     # Less comfort with market speculation
    
    # Normalize weights and ensure they're reasonable
    for key in weights:
        weights[key] = max(0.3, min(2.5, weights[key]))  # Clamp between 0.3 and 2.5
    
    return weights


def initialize_system():
    """Initialize the system components."""
    if st.session_state.initialized:
        return True
    
    # Check API keys
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    if not os.getenv('ALPHA_VANTAGE_API_KEY'):
        st.warning("⚠️ ALPHA_VANTAGE_API_KEY not found. Some features may be limited.")
    
    try:
        # Initialize components
        st.session_state.config_loader = get_config_loader()
        
        # Use Enhanced Data Provider with fallbacks
        st.session_state.data_provider = EnhancedDataProvider()
        
        # Load configurations
        model_config = st.session_state.config_loader.load_model_config()
        ips_config = st.session_state.config_loader.load_ips()
        
        # Initialize AI clients for advanced features
        openai_client = None
        perplexity_client = None
        
        try:
            if OpenAI is not None:
                openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                st.session_state.openai_client = openai_client
            else:
                st.warning("⚠️ OpenAI library not available. Please install: pip install openai")
        except Exception as e:
            st.warning(f"⚠️ OpenAI client initialization failed: {e}")
        
        try:
            if OpenAI is not None:
                perplexity_client = OpenAI(
                    api_key=os.getenv('PERPLEXITY_API_KEY'),
                    base_url="https://api.perplexity.ai"
                )
                st.session_state.perplexity_client = perplexity_client
            else:
                st.warning("⚠️ OpenAI library not available for Perplexity. Please install: pip install openai")
        except Exception as e:
            st.warning(f"⚠️ Perplexity client initialization failed: {e}")
        
        # Initialize orchestrator with enhanced data provider and AI clients
        st.session_state.orchestrator = PortfolioOrchestrator(
            model_config=model_config,
            ips_config=ips_config,
            enhanced_data_provider=st.session_state.data_provider,
            openai_client=openai_client,
            perplexity_client=perplexity_client
        )
        
        # Initialize QA system
        st.session_state.qa_system = QASystem()
        
        # Initialize Step Time Manager for persistent step-level timing
        from utils.step_time_manager import StepTimeManager
        if 'step_time_manager' not in st.session_state:
            st.session_state.step_time_manager = StepTimeManager()
            print(st.session_state.step_time_manager.get_summary())
        
        # Initialize analysis time tracking
        if 'analysis_times' not in st.session_state:
            st.session_state.analysis_times = []  # List of historical analysis times in seconds
        
        # Initialize current analysis tracking
        if 'current_analysis_start' not in st.session_state:
            st.session_state.current_analysis_start = None
        if 'current_step_start' not in st.session_state:
            st.session_state.current_step_start = None
        if 'last_step' not in st.session_state:
            st.session_state.last_step = 0
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        return False


def main():
    """Main application entry point."""
    
    # Header
    st.title("Wharton Investment Analysis System")
    st.markdown("**Multi-Agent Investment Research Platform**")
    st.markdown("---")
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Check for stocks due for weekly review and show notification
    if st.session_state.qa_system:
        stocks_due = st.session_state.qa_system.get_stocks_due_for_review()
        if stocks_due:
            st.sidebar.warning(f"⏰ {len(stocks_due)} stock(s) due for weekly review")
            st.sidebar.info("Visit QA & Learning Center to conduct reviews")
    
    # Google Sheets Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Google Sheets Integration")
    
    # Ensure Google Sheets session state variables exist (must be first!)
    if 'sheets_integration' not in st.session_state:
        st.session_state.sheets_integration = get_sheets_integration()
    if 'sheets_enabled' not in st.session_state:
        st.session_state.sheets_enabled = False
    if 'sheets_auto_update' not in st.session_state:
        st.session_state.sheets_auto_update = False
    
    sheets_integration = st.session_state.sheets_integration
    
    # Safety check: ensure sheets_integration is not None
    if sheets_integration is None:
        sheets_integration = get_sheets_integration()
        st.session_state.sheets_integration = sheets_integration
    
    if sheets_integration and sheets_integration.enabled:
        st.sidebar.success("✅ Google Sheets API Ready")
        
        # Auto-connect if Sheet ID is in .env and not yet connected
        env_sheet_id = os.getenv('GOOGLE_SHEET_ID', '')
        if env_sheet_id and not st.session_state.sheets_enabled and sheets_integration.sheet is None:
            with st.spinner("Auto-connecting to Google Sheet..."):
                if sheets_integration.connect_to_sheet(env_sheet_id):
                    st.session_state.sheets_enabled = True
                    st.sidebar.success(f"✅ Auto-connected from .env!")
                    
                    # Auto-sync existing QA analyses on first connection
                    with st.spinner("📤 Syncing QA analyses to Google Sheets..."):
                        if sync_all_archives_to_sheets():
                            st.sidebar.success("✅ QA analyses synced! New portfolios will sync automatically.")
                        else:
                            st.sidebar.info("ℹ️ No QA analyses to sync yet")
        
        # Sheet ID input (shows current value from .env or manual entry)
        sheet_id = st.sidebar.text_input(
            "Google Sheet ID",
            value=env_sheet_id,
            help="Enter the Sheet ID from the URL (or set GOOGLE_SHEET_ID in .env)",
            key="sheet_id_input"
        )
        
        # Manual connect button (only if not already connected)
        if sheet_id and sheets_integration.sheet is None:
            if st.sidebar.button("🔗 Connect to Sheet"):
                with st.spinner("Connecting..."):
                    if sheets_integration.connect_to_sheet(sheet_id):
                        st.sidebar.success(f"✅ Connected!")
                        st.session_state.sheets_enabled = True
                        # Save to env for persistence
                        os.environ['GOOGLE_SHEET_ID'] = sheet_id
                        
                        # Auto-sync existing QA analyses
                        with st.spinner("📤 Syncing QA analyses to Google Sheets..."):
                            if sync_all_archives_to_sheets():
                                st.sidebar.success("✅ QA analyses synced! New portfolios will sync automatically.")
                            else:
                                st.sidebar.info("ℹ️ No QA analyses to sync yet")
                    else:
                        st.sidebar.error("❌ Connection failed")
            
            # Auto-update toggle
            if st.session_state.sheets_enabled:
                st.session_state.sheets_auto_update = st.sidebar.checkbox(
                    "🔄 Auto-update on analysis",
                    value=st.session_state.sheets_auto_update,
                    help="Automatically push results to Google Sheets"
                )
                
                if sheets_integration.sheet:
                    sheet_url = sheets_integration.get_sheet_url()
                    if sheet_url:
                        st.sidebar.markdown(f"[📄 Open Sheet]({sheet_url})")
                    
                    # Manual sync QA analyses button
                    if st.sidebar.button("🔄 Sync QA Analyses Now"):
                        with st.spinner("📤 Syncing QA analyses..."):
                            if sync_all_archives_to_sheets():
                                st.sidebar.success("✅ QA analyses synced!")
                            else:
                                st.sidebar.info("ℹ️ No QA analyses to sync")
    else:
        st.sidebar.warning("⚙️ Not configured (optional)")
        with st.sidebar.expander("📖 Setup Instructions (Optional)"):
            st.markdown("""
            **Google Sheets integration is optional** - the app works fully without it.
            
            To enable automatic portfolio syncing to Google Sheets:
            
            1. Create a [Google Cloud project](https://console.cloud.google.com)
            2. Enable the Google Sheets API
            3. Create a service account
            4. Download the credentials JSON file
            5. Save it as `google_credentials.json` in the project root
            6. Create a Google Sheet and share it with the service account email
            7. Add the Sheet ID to `.env` as `GOOGLE_SHEET_ID=your_sheet_id`
            
            **Benefits:**
            - Auto-sync all analyses to Google Sheets
            - Track portfolio history over time
            - Share results with clients/team
            - Advanced filtering and charting
            
            **Without it:**
            - All analysis still works perfectly
            - Results shown in the app
            - QA system still tracks everything
            
            [Full Setup Guide](https://docs.gspread.org/en/latest/oauth2.html)
            """)
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.title("NAVIGATION")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Select Analysis Mode:",
        ["Stock Analysis", "Portfolio Recommendations", "Portfolio Management", "QA & Learning Center", "System Configuration", "System Status & AI Disclosure"]
    )
    
    # Route to appropriate page
    if page == "Stock Analysis":
        stock_analysis_page()
    elif page == "Portfolio Recommendations":
        portfolio_recommendations_page()
    elif page == "Portfolio Management":
        portfolio_management_page()
    elif page == "QA & Learning Center":
        qa_learning_center_page()
    elif page == "System Configuration":
        configuration_page()
    elif page == "System Status & AI Disclosure":
        system_status_and_ai_disclosure_page()


def stock_analysis_page():
    """Single or multiple stock analysis page."""
    st.header("Stock Analysis")
    st.write("Evaluate individual securities or analyze multiple stocks at once using multi-agent investment research methodology.")
    st.markdown("---")
    
    # Analysis mode selection
    analysis_mode = st.radio(
        "Analysis Mode",
        options=["Single Stock", "Multiple Stocks"],
        horizontal=True,
        help="Choose to analyze one stock or multiple stocks at once"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
    
    with col2:
        analysis_date = st.date_input(
            "Analysis Date",
            value=datetime.now(),
            help="Date for analysis (leave as today for latest data)"
        )
    
    # Weight preset
    st.markdown("### ⚖️ Agent Weights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight_preset = st.selectbox(
            "Choose Weight Configuration:",
            options=["equal_weights", "custom_weights", "client_profile_weights"],
            format_func=lambda x: {
                "equal_weights": "1. Equal Weights",
                "custom_weights": "2. Custom Weights", 
                "client_profile_weights": "3. Client Profile Weights"
            }[x],
            help="Select how agent weights should be configured for this analysis"
        )
        
        # Store weight preset in session state for use in display functions
        st.session_state.weight_preset = weight_preset
    
    # Initialize session state for custom weights
    if 'custom_agent_weights' not in st.session_state:
        st.session_state.custom_agent_weights = {
            'value': 1.0,
            'growth_momentum': 1.0,
            'macro_regime': 1.0,
            'risk': 1.0,
            'sentiment': 1.0
        }
    
    # Handle weight preset selection
    agent_weights = None
    if weight_preset == "custom_weights":
        with col2:
            if st.button("🔧 Configure Custom Weights"):
                st.session_state.show_custom_weights = not st.session_state.get('show_custom_weights', False)
        
        if st.session_state.get('show_custom_weights', False):
            st.info("""
            **📊 Custom Weights Explanation:**
            
            These weights control **how much each agent's score influences the final score**.
            
            - **Higher weight (2.0)** = Agent's opinion has MORE influence on final score
            - **Lower weight (0.5)** = Agent's opinion has LESS influence on final score  
            - **Weight of 1.0** = Standard/equal influence
            
            **Example:** If Value Agent = 2.0 and Growth Agent = 0.5:
            - Final score will be heavily weighted toward value metrics
            - Growth metrics will have less impact on the final score
            
            **Important:** Agents still score independently (0-100). Weights only affect how 
            those scores are combined into the final score.
            """)
            
            st.write("**Configure Custom Agent Weights:**")
            weight_cols = st.columns(5)
            
            agents = ['value', 'growth_momentum', 'macro_regime', 'risk', 'sentiment']
            agent_labels = ['Value', 'Growth/Momentum', 'Macro Regime', 'Risk', 'Sentiment']
            
            for i, (agent, label) in enumerate(zip(agents, agent_labels)):
                with weight_cols[i]:
                    st.session_state.custom_agent_weights[agent] = st.slider(
                        label,
                        min_value=0.0,
                        max_value=2.0,
                        value=st.session_state.custom_agent_weights[agent],
                        step=0.1,
                        key=f"custom_weight_{agent}",
                        help=f"Weight for {label} agent's contribution to final score"
                    )
            
            # Show current weight distribution
            st.write("**Current Weight Distribution:**")
            total_weight = sum(st.session_state.custom_agent_weights.values())
            percentages = {k: (v/total_weight)*100 for k, v in st.session_state.custom_agent_weights.items()}
            
            dist_cols = st.columns(5)
            for i, (agent, pct) in enumerate(percentages.items()):
                with dist_cols[i]:
                    st.metric(agent_labels[i], f"{pct:.1f}%", help=f"This agent's influence: {pct:.1f}%")
            
            # 🆕 IMPROVEMENT #2: Save/Load Weight Presets
            st.markdown("---")
            col_save, col_load = st.columns(2)
            
            with col_save:
                st.write("**💾 Save Current Weights as Preset:**")
                preset_name = st.text_input("Preset Name", placeholder="e.g., Aggressive Growth", key="save_preset_name")
                if st.button("💾 Save Preset", key="save_preset_btn"):
                    if preset_name:
                        if 'saved_weight_presets' not in st.session_state:
                            st.session_state.saved_weight_presets = {}
                        st.session_state.saved_weight_presets[preset_name] = st.session_state.custom_agent_weights.copy()
                        # Persist to disk
                        import json
                        presets_file = Path("data/weight_presets.json")
                        presets_file.parent.mkdir(exist_ok=True)
                        with open(presets_file, 'w') as f:
                            json.dump(st.session_state.saved_weight_presets, f, indent=2)
                        st.success(f"✅ Saved preset: {preset_name}")
                    else:
                        st.warning("Please enter a preset name")
            
            with col_load:
                st.write("**📂 Load Saved Preset:**")
                # Load presets from disk if not in session state
                if 'saved_weight_presets' not in st.session_state:
                    import json
                    presets_file = Path("data/weight_presets.json")
                    if presets_file.exists():
                        with open(presets_file, 'r') as f:
                            st.session_state.saved_weight_presets = json.load(f)
                    else:
                        st.session_state.saved_weight_presets = {}
                
                if st.session_state.saved_weight_presets:
                    preset_to_load = st.selectbox("Select Preset", options=list(st.session_state.saved_weight_presets.keys()), key="load_preset_select")
                    if st.button("📂 Load Preset", key="load_preset_btn"):
                        st.session_state.custom_agent_weights = st.session_state.saved_weight_presets[preset_to_load].copy()
                        st.success(f"✅ Loaded preset: {preset_to_load}")
                        st.rerun()
                else:
                    st.info("No saved presets yet")
            
            # Lock in weights button
            st.markdown("---")
            if st.button("🔒 Lock In Custom Weights", type="primary"):
                st.session_state.locked_custom_weights = st.session_state.custom_agent_weights.copy()
                st.success("✅ Custom weights locked in! These will be used for analysis.")
                st.session_state.show_custom_weights = False
        
        # Use locked custom weights if available
        if 'locked_custom_weights' in st.session_state:
            agent_weights = st.session_state.locked_custom_weights
            
            # Show which weights are active
            with st.expander("⚖️ Active Custom Weights", expanded=False):
                st.write("**These custom weights will be applied to your analysis:**")
                total_weight = sum(agent_weights.values())
                cols = st.columns(5)
                agent_labels_dict = {
                    'value': 'Value',
                    'growth_momentum': 'Growth/Momentum',
                    'macro_regime': 'Macro Regime',
                    'risk': 'Risk',
                    'sentiment': 'Sentiment'
                }
                for i, (agent, weight) in enumerate(agent_weights.items()):
                    with cols[i]:
                        pct = (weight / total_weight) * 100
                        st.metric(agent_labels_dict.get(agent, agent), f"{weight:.1f}x", delta=f"{pct:.1f}%")
    
    elif weight_preset == "client_profile_weights":
        with col2:
            # Load available client profiles
            from utils.client_profile_manager import ClientProfileManager
            profile_manager = ClientProfileManager()
            available_profiles = profile_manager.list_client_profiles()
            
            if available_profiles:
                profile_names = [p['client_name'] for p in available_profiles]
                selected_profile = st.selectbox(
                    "Select Client Profile:",
                    options=profile_names,
                    help="Choose a client profile to derive agent weights"
                )
                
                if st.button("📋 Apply Profile Weights"):
                    agent_weights = get_client_profile_weights(selected_profile)
                    st.success(f"✅ Applied weights based on {selected_profile} profile!")
            else:
                st.warning("⚠️ No client profiles found. Please create a client profile in System Configuration first.")
    
    else:  # equal_weights
        with col2:
            if st.button("⚖️ Apply Equal Weights"):
                agent_weights = {
                    'value': 1.0,
                    'growth_momentum': 1.0,
                    'macro_regime': 1.0,
                    'risk': 1.0,
                    'sentiment': 1.0
                }
                st.success("✅ Equal weights applied!")
    
    st.markdown("---")
    
    if st.button("Run Analysis", type="primary"):
        # Validation
        if analysis_mode == "Single Stock":
            if not ticker:
                st.error("Please enter a ticker symbol")
                return
        else:
            if not tickers:
                st.error("Please enter at least one ticker symbol")
                return
        
        # Create detailed progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize progress tracking in session state
        if 'analysis_progress' not in st.session_state:
            st.session_state.analysis_progress = {
                'step': 0,
                'total_steps': 10,
                'current_status': 'Starting analysis...',
                'progress_bar': None,
                'status_text': None
            }
        
        # Store progress components in session state for orchestrator access
        st.session_state.analysis_progress['progress_bar'] = progress_bar
        st.session_state.analysis_progress['status_text'] = status_text
        
        # Handle single or multiple stock analysis
        if analysis_mode == "Single Stock":
            # Single stock analysis (existing logic)
            try:
                import time
                
                # Initialize step tracking for this analysis
                st.session_state.current_analysis_start = time.time()
                st.session_state.current_step_start = time.time()
                st.session_state.last_step = 0
                
                # Calculate estimated time
                if st.session_state.analysis_times:
                    avg_time = sum(st.session_state.analysis_times) / len(st.session_state.analysis_times)
                    est_minutes = int(avg_time // 60)
                    est_seconds = int(avg_time % 60)
                    status_text.text(f"🚀 Starting analysis... (Est. {est_minutes}m {est_seconds}s)")
                else:
                    status_text.text("🚀 Starting analysis... (Est. ~30-40s)")
                
                progress_bar.progress(0)
                
                # Track start time
                start_time = time.time()
                
                # Convert analysis_date to string format
                # Handle both date object and potential tuple from date_input
                if isinstance(analysis_date, (datetime, type(datetime.now().date()))):
                    date_str = analysis_date.strftime('%Y-%m-%d') if hasattr(analysis_date, 'strftime') else str(analysis_date)
                elif isinstance(analysis_date, tuple) and len(analysis_date) > 0:
                    date_str = analysis_date[0].strftime('%Y-%m-%d') if hasattr(analysis_date[0], 'strftime') else str(analysis_date[0])
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                # Run analysis with optional agent weights
                result = st.session_state.orchestrator.analyze_stock(
                    ticker=ticker,
                    analysis_date=date_str,
                    agent_weights=agent_weights
                )
                
                # Track end time and store
                end_time = time.time()
                analysis_duration = end_time - start_time
                st.session_state.analysis_times.append(analysis_duration)
                
                # Keep only last 50 times for better estimates
                if len(st.session_state.analysis_times) > 50:
                    st.session_state.analysis_times = st.session_state.analysis_times[-50:]
                
                # Final completion with actual time
                actual_minutes = int(analysis_duration // 60)
                actual_seconds = int(analysis_duration % 60)
                status_text.text(f"✅ Analysis complete! (Took {actual_minutes}m {actual_seconds}s)")
                progress_bar.progress(100)
                
                # Clear progress indicators after a brief moment
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                    return
                
                # Display results
                display_stock_analysis(result)
                
            except Exception as e:
                # Clear progress indicators on error
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {e}")
        
        else:
            # Multiple stocks analysis
            import time
            
            st.info(f"🔍 Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
            
            # Create overall progress tracking
            overall_progress = st.progress(0)
            overall_status = st.empty()
            time_estimate_display = st.empty()
            
            results = []
            failed_tickers = []
            batch_start_time = time.time()
            
            # Initial time estimate
            if st.session_state.analysis_times:
                avg_time = sum(st.session_state.analysis_times) / len(st.session_state.analysis_times)
            else:
                avg_time = 35  # Default estimate in seconds
            
            total_est_seconds = int(avg_time * len(tickers))
            est_minutes = total_est_seconds // 60
            est_seconds = total_est_seconds % 60
            time_estimate_display.info(f"⏱️ Initial estimate: ~{est_minutes}m {est_seconds}s for {len(tickers)} stocks")
            
            for idx, stock_ticker in enumerate(tickers):
                stock_start_time = time.time()
                
                # Calculate dynamic time remaining using actual batch performance
                completed_count = idx  # Number of stocks completed so far
                if completed_count > 0:
                    # Calculate elapsed time and stocks per minute
                    elapsed_time = time.time() - batch_start_time
                    stocks_per_minute = completed_count / (elapsed_time / 60)
                    
                    # Calculate remaining time based on actual rate
                    remaining_stocks = len(tickers) - idx
                    est_remaining_minutes = remaining_stocks / stocks_per_minute if stocks_per_minute > 0 else 0
                    est_minutes = int(est_remaining_minutes)
                    est_seconds = int((est_remaining_minutes - est_minutes) * 60)
                    
                    overall_status.text(f"📊 Analyzing {stock_ticker} ({idx + 1} of {len(tickers)}) - Est. {est_minutes}m {est_seconds}s remaining (Rate: {stocks_per_minute:.1f} stocks/min)")
                else:
                    # Use historical average for first stock
                    remaining_stocks = len(tickers) - idx
                    est_remaining_seconds = int(avg_time * remaining_stocks)
                    est_minutes = est_remaining_seconds // 60
                    est_seconds = est_remaining_seconds % 60
                    overall_status.text(f"📊 Analyzing {stock_ticker} ({idx + 1} of {len(tickers)}) - Est. {est_minutes}m {est_seconds}s remaining")
                
                # Create progress tracking for individual stock
                stock_progress_bar = st.progress(0)
                stock_status_text = st.empty()
                
                # Initialize step tracking for this stock
                st.session_state.current_analysis_start = time.time()
                st.session_state.current_step_start = time.time()
                st.session_state.last_step = 0
                
                # Re-initialize progress tracking in session state for this stock
                st.session_state.analysis_progress = {
                    'step': 0,
                    'total_steps': 10,
                    'current_status': 'Starting analysis...',
                    'progress_bar': stock_progress_bar,
                    'status_text': stock_status_text
                }
                
                try:
                    # Convert analysis_date to string format
                    # Handle both date object and potential tuple from date_input
                    if isinstance(analysis_date, (datetime, type(datetime.now().date()))):
                        date_str = analysis_date.strftime('%Y-%m-%d') if hasattr(analysis_date, 'strftime') else str(analysis_date)
                    elif isinstance(analysis_date, tuple) and len(analysis_date) > 0:
                        date_str = analysis_date[0].strftime('%Y-%m-%d') if hasattr(analysis_date[0], 'strftime') else str(analysis_date[0])
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    # Run analysis for this stock
                    result = st.session_state.orchestrator.analyze_stock(
                        ticker=stock_ticker,
                        analysis_date=date_str,
                        agent_weights=agent_weights
                    )
                    
                    # Track time for this stock
                    stock_end_time = time.time()
                    stock_duration = stock_end_time - stock_start_time
                    st.session_state.analysis_times.append(stock_duration)
                    
                    # Keep only last 50 times
                    if len(st.session_state.analysis_times) > 50:
                        st.session_state.analysis_times = st.session_state.analysis_times[-50:]
                    
                    # Clear individual progress indicators
                    stock_progress_bar.empty()
                    stock_status_text.empty()
                    
                    if 'error' in result:
                        failed_tickers.append((stock_ticker, result['error']))
                    else:
                        results.append(result)
                        
                        # 🔧 FIX: Automatically log each successful analysis to QA archive
                        # This ensures ALL analyzed stocks (not just individually clicked ones) get archived
                        if st.session_state.get('qa_system'):
                            try:
                                qa_system = st.session_state.qa_system
                                recommendation_type = _determine_recommendation_type(result['final_score'])
                                
                                # Create comprehensive rationale
                                agent_rationales = result.get('agent_rationales', {})
                                current_date = datetime.now().strftime('%Y-%m-%d')
                                final_rationale = f"Investment analysis for {result['ticker']} completed on {current_date}"
                                if 'client_layer_agent' in agent_rationales:
                                    final_rationale = agent_rationales['client_layer_agent'][:500] + "..." if len(agent_rationales.get('client_layer_agent', '')) > 500 else agent_rationales.get('client_layer_agent', final_rationale)
                                
                                # Log complete analysis automatically for multi-stock batch
                                analysis_id = qa_system.log_complete_analysis(
                                    ticker=result['ticker'],
                                    price=result['fundamentals'].get('price', 0),
                                    recommendation=recommendation_type,
                                    confidence_score=result['final_score'],
                                    final_rationale=final_rationale,
                                    agent_scores=result.get('agent_scores', {}),
                                    agent_rationales=agent_rationales,
                                    key_factors=[],  # Will be populated later if needed
                                    fundamentals=result.get('fundamentals', {}),
                                    market_data=result.get('market_data', {}),
                                    sector=result['fundamentals'].get('sector'),
                                    market_cap=result['fundamentals'].get('market_cap')
                                )
                                
                                if analysis_id:
                                    print(f"🔧 DEBUG: Auto-logged {stock_ticker} to QA archive with ID: {analysis_id}")
                                    
                            except Exception as e:
                                print(f"🔧 WARNING: Could not auto-log {stock_ticker} to QA archive: {e}")
                    
                    # Update time estimate with actual batch performance
                    completed = idx + 1
                    remaining = len(tickers) - completed
                    if completed > 0 and remaining > 0:
                        elapsed_total = time.time() - batch_start_time
                        stocks_per_minute = completed / (elapsed_total / 60)
                        est_remaining_minutes = remaining / stocks_per_minute if stocks_per_minute > 0 else 0
                        est_minutes = int(est_remaining_minutes)
                        est_seconds = int((est_remaining_minutes - est_minutes) * 60)
                        time_estimate_display.info(f"⏱️ Updated estimate: ~{est_minutes}m {est_seconds}s remaining ({completed}/{len(tickers)} complete, {stocks_per_minute:.1f} stocks/min)")
                    
                except Exception as e:
                    stock_progress_bar.empty()
                    stock_status_text.empty()
                    failed_tickers.append((stock_ticker, str(e)))
                
                # Update overall progress
                overall_progress.progress((idx + 1) / len(tickers))
            
            # Clear overall progress and show final time
            batch_end_time = time.time()
            total_duration = batch_end_time - batch_start_time
            total_minutes = int(total_duration // 60)
            total_seconds = int(total_duration % 60)
            overall_status.text(f"✅ Batch analysis complete! (Total time: {total_minutes}m {total_seconds}s)")
            time_estimate_display.success(f"🎉 Analyzed {len(results)} stocks successfully in {total_minutes}m {total_seconds}s")
            time.sleep(1.5)
            overall_progress.empty()
            overall_status.empty()
            time_estimate_display.empty()
            
            # Display results summary
            if results:
                display_multiple_stock_analysis(results, failed_tickers)
            else:
                st.error("❌ All analyses failed!")
                for ticker_name, error_msg in failed_tickers:
                    st.error(f"**{ticker_name}**: {error_msg}")


def display_stock_analysis(result: dict):
    """Display detailed stock analysis results with enhanced rationales."""
    
    # Automatically log complete analysis to archive (avoid duplicates)
    if st.session_state.get('qa_system'):
        try:
            qa_system = st.session_state.qa_system
            ticker = result['ticker']
            
            # Check if this analysis was already logged recently (within last 5 minutes)
            # This prevents duplicate logging in multi-stock analysis when users click tabs
            analysis_archive = qa_system.get_analysis_archive()
            recently_logged = False
            
            if ticker in analysis_archive:
                latest_analysis = analysis_archive[ticker][0]  # Most recent is first 
                time_diff = datetime.now() - latest_analysis.timestamp
                if time_diff.total_seconds() < 300:  # 5 minutes
                    recently_logged = True
                    print(f"🔧 DEBUG: {ticker} already logged recently, skipping duplicate")
            
            if not recently_logged:
                recommendation_type = _determine_recommendation_type(result['final_score'])
                
                # Create comprehensive rationale
                agent_rationales = result.get('agent_rationales', {})
                current_date = datetime.now().strftime('%Y-%m-%d')
                final_rationale = f"Investment analysis for {ticker} completed on {current_date}"
                if 'client_layer_agent' in agent_rationales:
                    final_rationale = agent_rationales['client_layer_agent'][:500] + "..." if len(agent_rationales.get('client_layer_agent', '')) > 500 else agent_rationales.get('client_layer_agent', final_rationale)
                
                # Log complete analysis
                analysis_id = qa_system.log_complete_analysis(
                    ticker=ticker,
                    price=result['fundamentals'].get('price', 0),
                    recommendation=recommendation_type,
                    confidence_score=result['final_score'],
                    final_rationale=final_rationale,
                    agent_scores=result.get('agent_scores', {}),
                    agent_rationales=agent_rationales,
                    key_factors=[],  # Will be populated later
                    fundamentals=result.get('fundamentals', {}),
                    market_data=result.get('market_data', {}),
                    sector=result['fundamentals'].get('sector'),
                    market_cap=result['fundamentals'].get('market_cap')
                )
                
                if analysis_id:
                    # Add a small indicator that analysis was saved
                    st.info(f"📚 Analysis automatically saved to archive (ID: {analysis_id})")
                    print(f"🔧 DEBUG: Logged {ticker} to QA archive with ID: {analysis_id}")
                    
        except Exception as e:
            st.warning(f"⚠️ Could not auto-save analysis: {e}")
            print(f"🔧 ERROR: Auto-save failed for {result.get('ticker', 'unknown')}: {e}")
    
    # Header with company info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{result['ticker']} - Investment Analysis")
        if 'name' in result['fundamentals']:
            st.caption(result['fundamentals']['name'])
        # Display analysis date in MM/DD/YYYY format
        if 'analysis_date' in result:
            try:
                date_obj = datetime.strptime(result['analysis_date'], '%Y-%m-%d')
                formatted_date = date_obj.strftime('%m/%d/%Y')
                st.caption(f"Analysis Date: {formatted_date}")
            except:
                st.caption(f"Analysis Date: {result['analysis_date']}")
    with col2:
        # Eligibility badge
        if result['eligible']:
            st.success("✅ ELIGIBLE")
            st.caption("This stock meets all client investment criteria")
        else:
            st.error("❌ NOT ELIGIBLE")
            st.caption("This stock violates one or more client criteria")
    
    # Show which weights were used for this analysis
    weight_preset = st.session_state.get('weight_preset', 'equal_weights')
    if weight_preset == 'custom_weights' and 'locked_custom_weights' in st.session_state:
        with st.expander("⚖️ Custom Weights Used in This Analysis", expanded=False):
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
                
                st.write(f"**Final Score Calculation:**")
                st.code(f"""
                Weighted Sum = {weighted_sum:.2f}
                Total Weight = {total_weight:.2f}
                Final Score = {weighted_sum:.2f} / {total_weight:.2f} = {calculated_final:.2f}
                """)
                
                st.caption("💡 Higher weights mean that agent's score had MORE influence on the final score.")
    
    elif weight_preset == 'client_profile_weights':
        st.info("ℹ️ This analysis used weights derived from the selected client profile.")
    
    # 🆕 IMPROVEMENT #7: Side-by-Side Comparison with Previous Analysis
    ticker = result['ticker']
    qa_system = st.session_state.get('qa_system')
    
    if qa_system and ticker in qa_system.all_analyses:
        analyses = qa_system.all_analyses[ticker]
        if len(analyses) >= 2:
            # Get the most recent previous analysis
            sorted_analyses = sorted(analyses, key=lambda x: x.timestamp, reverse=True)
            previous = sorted_analyses[1] if len(sorted_analyses) > 1 else None
            
            if previous:
                st.info(f"📊 Previous analysis available from {previous.timestamp.strftime('%b %d, %Y')}")
                
                with st.expander("🔄 Compare with Previous Analysis", expanded=False):
                    st.markdown("#### Side-by-Side Comparison")
                    
                    col_prev, col_curr = st.columns(2)
                    
                    with col_prev:
                        st.markdown("**📅 Previous** ({})".format(previous.timestamp.strftime('%m/%d/%y')))
                        st.metric("Score", f"{previous.confidence_score:.1f}/100")
                        st.metric("Recommendation", previous.recommendation.value.upper())
                        st.metric("Price", f"${previous.price_at_analysis:.2f}")
                    
                    with col_curr:
                        st.markdown("**📅 Current** ({})".format(datetime.now().strftime('%m/%d/%y')))
                        st.metric("Score", f"{result['final_score']:.1f}/100")
                        rec_type = _determine_recommendation_type(result['final_score'])
                        st.metric("Recommendation", rec_type.value.upper())
                        st.metric("Price", f"${result['fundamentals'].get('price', 0):.2f}")
                    
                    # Change analysis
                    st.markdown("---")
                    st.markdown("**📈 Changes Over Time**")
                    
                    score_change = result['final_score'] - previous.confidence_score
                    price_change = result['fundamentals'].get('price', 0) - previous.price_at_analysis
                    price_change_pct = (price_change / previous.price_at_analysis * 100) if previous.price_at_analysis > 0 else 0
                    days_between = (datetime.now() - previous.timestamp).days
                    
                    col_ch1, col_ch2, col_ch3 = st.columns(3)
                    with col_ch1:
                        st.metric("Score Change", f"{score_change:+.1f} points", 
                                 delta=f"{score_change:+.1f}", 
                                 delta_color="normal" if score_change > 0 else "inverse")
                    with col_ch2:
                        st.metric("Price Change", f"${price_change:+.2f}",
                                 delta=f"{price_change_pct:+.1f}%")
                    with col_ch3:
                        st.metric("Time Between", f"{days_between} days")
                    
                    # Agent-level changes
                    if hasattr(previous, 'agent_scores') and previous.agent_scores:
                        st.write("**Agent Score Changes:**")
                        agent_changes = []
                        for agent in result['agent_scores'].keys():
                            if agent in previous.agent_scores:
                                change = result['agent_scores'][agent] - previous.agent_scores[agent]
                                agent_changes.append({
                                    'Agent': agent.replace('_', ' ').title(),
                                    'Previous': f"{previous.agent_scores[agent]:.1f}",
                                    'Current': f"{result['agent_scores'][agent]:.1f}",
                                    'Change': f"{change:+.1f}",
                                    'Direction': '📈' if change > 0 else '📉' if change < 0 else '➡️'
                                })
                        
                        change_df = pd.DataFrame(agent_changes)
                        st.dataframe(change_df, use_container_width=True, hide_index=True)
                        
                        # Highlight biggest changes
                        biggest_changes = sorted(agent_changes, key=lambda x: abs(float(x['Change'])), reverse=True)[:3]
                        st.write("**Biggest Changes:**")
                        for item in biggest_changes:
                            if abs(float(item['Change'])) > 5:
                                st.write(f"• {item['Direction']} **{item['Agent']}**: {item['Change']} points")
    
    # Enhanced key metrics section with modern card-style layout
    st.markdown("### 📊 Key Investment Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        final_score = result['final_score']
        delta_color = "normal" if final_score >= 70 else "inverse" if final_score < 50 else "off"
        st.metric("Final Score", f"{final_score:.1f}/100", help="Overall investment recommendation score")
    with col2:
        price_value = result['fundamentals'].get('price')
        st.metric("Current Price", f"${price_value:.2f}" if price_value and price_value != 0 else "N/A")
    with col3:
        pe_ratio = result['fundamentals'].get('pe_ratio')
        st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio and pe_ratio != 0 else "N/A", help="Price-to-Earnings ratio")
    with col4:
        beta = result['fundamentals'].get('beta')
        st.metric("Beta", f"{beta:.2f}" if beta and beta != 0 else "N/A", help="Market volatility coefficient")
    
    # Additional Enhanced Metrics Row
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        div_yield = result['fundamentals'].get('dividend_yield')
        # Dividend yield can be a decimal (0.02 = 2%) or already a percentage (2.0 = 2%)
        if div_yield and div_yield != 0:
            # If it's a small decimal, multiply by 100, otherwise use as-is
            display_yield = div_yield * 100 if div_yield < 1 else div_yield
            st.metric("Dividend Yield", f"{display_yield:.2f}%", help="Annual dividend yield percentage")
        else:
            st.metric("Dividend Yield", "N/A", help="Annual dividend yield percentage")
    with col6:
        eps = result['fundamentals'].get('eps')
        if eps and eps != 0:
            st.metric("EPS", f"${eps:.2f}", help="Earnings per share")
        else:
            st.metric("EPS", "N/A", help="Earnings per share")
    with col7:
        week_52_low = result['fundamentals'].get('week_52_low')
        week_52_high = result['fundamentals'].get('week_52_high')
        if week_52_low and week_52_high:
            st.metric("52W Range", f"${week_52_low:.2f}-${week_52_high:.2f}", help="52-week price range")
        else:
            st.metric("52W Range", "N/A", help="52-week price range")
    with col8:
        market_cap = result['fundamentals'].get('market_cap')
        if market_cap:
            if market_cap >= 1e12:
                st.metric("Market Cap", f"${market_cap/1e12:.1f}T", help="Market capitalization")
            elif market_cap >= 1e9:
                st.metric("Market Cap", f"${market_cap/1e9:.1f}B", help="Market capitalization")
            else:
                st.metric("Market Cap", f"${market_cap/1e6:.0f}M", help="Market capitalization")
        else:
            st.metric("Market Cap", "N/A", help="Market capitalization")
    
    # 52-Week Range Visualization
    week_52_low = result['fundamentals'].get('week_52_low')
    week_52_high = result['fundamentals'].get('week_52_high')
    current_price = result['fundamentals'].get('price')
    
    if week_52_low and week_52_high and current_price:
        st.subheader("📈 52-Week Price Range")
        
        # Create a simple visualization using Streamlit's native components
        col1, col2, col3 = st.columns([1, 8, 1])
        
        with col1:
            st.metric("52W Low", f"${week_52_low:.2f}")
        
        with col2:
            # Calculate position of current price within the range
            price_position = (current_price - week_52_low) / (week_52_high - week_52_low)
            price_position = max(0, min(1, price_position))  # Clamp between 0 and 1
            
            # Create a visual representation with a fully shaded bar
            st.markdown("**Current Price Position in 52-Week Range:**")
            
            # Determine color based on position
            if price_position >= 0.80:
                bar_color = "#00cc00"  # Green - near high
                position_text = "Near 52W High 🚀"
            elif price_position >= 0.60:
                bar_color = "#66cc00"  # Yellow-green
                position_text = "Upper Range"
            elif price_position >= 0.40:
                bar_color = "#ffaa00"  # Orange
                position_text = "Mid Range"
            elif price_position >= 0.20:
                bar_color = "#ff6600"  # Orange-red
                position_text = "Lower Range"
            else:
                bar_color = "#cc0000"  # Red - near low
                position_text = "Near 52W Low 📉"
            
            # Create a fully shaded horizontal bar using HTML/CSS
            bar_html = f"""
            <div style="position: relative; width: 100%; height: 40px; background-color: #e0e0e0; border-radius: 8px; overflow: hidden; margin: 10px 0;">
                <div style="position: absolute; left: 0; top: 0; width: {price_position*100}%; height: 100%; background: linear-gradient(90deg, {bar_color} 0%, {bar_color} 100%); border-radius: 8px 0 0 8px;"></div>
                <div style="position: absolute; left: {price_position*100}%; top: 50%; transform: translate(-50%, -50%); width: 3px; height: 45px; background-color: #000; z-index: 10;"></div>
                <div style="position: absolute; left: {price_position*100}%; top: -30px; transform: translateX(-50%); font-weight: bold; font-size: 14px; color: #000; white-space: nowrap;">
                    ${current_price:.2f}
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
            
            # Position info
            st.markdown(f"**{position_text}** • {price_position*100:.1f}% of 52-week range")
            
        with col3:
            st.metric("52W High", f"${week_52_high:.2f}")
    
    # ========== COMPREHENSIVE SCORE ANALYSIS SECTION ==========
    st.markdown("---")
    st.markdown("### ⚖️ Score Analysis & Agent Breakdown")
    
    with st.expander("📊 Detailed Breakdown", expanded=True):
        # Get agent scores and weights
        agent_scores = result.get('agent_scores', {})
        blended_score = result.get('blended_score', result.get('final_score', 0))
        
        # Determine which weights were used
        weight_preset = st.session_state.get('weight_preset', 'equal_weights')
        if weight_preset == 'custom_weights' and 'locked_custom_weights' in st.session_state:
            weights_used = st.session_state.locked_custom_weights
            weights_source = "Custom Weights"
        else:
            # Get default weights from orchestrator
            orchestrator = st.session_state.get('orchestrator')
            if orchestrator:
                weights_used = orchestrator.agent_weights
            else:
                weights_used = {
                    'value_agent': 0.25,
                    'growth_momentum_agent': 0.20,
                    'macro_regime_agent': 0.15,
                    'risk_agent': 0.25,
                    'sentiment_agent': 0.15
                }
            weights_source = "Default IPS Weights"
        
        st.write(f"**Weights Source:** {weights_source}")
        st.write("---")
        
        # Calculate weight breakdown
        total_weighted_score = 0
        total_weight = 0
        breakdown_data = []
        
        agent_order = ['value_agent', 'growth_momentum_agent', 'macro_regime_agent', 'risk_agent', 'sentiment_agent']
        agent_labels = {
            'value_agent': '💎 Value',
            'growth_momentum_agent': '📈 Growth/Momentum',
            'macro_regime_agent': '🌍 Macro Regime',
            'risk_agent': '⚠️ Risk',
            'sentiment_agent': '📰 Sentiment'
        }
        
        for agent_key in agent_order:
            score = agent_scores.get(agent_key, 50)
            
            # Map agent key to weight key
            if agent_key == 'value_agent':
                weight_key = 'value_agent'
            elif agent_key == 'growth_momentum_agent':
                weight_key = 'growth_momentum_agent'
            elif agent_key == 'macro_regime_agent':
                weight_key = 'macro_regime_agent'
            elif agent_key == 'risk_agent':
                weight_key = 'risk_agent'
            elif agent_key == 'sentiment_agent':
                weight_key = 'sentiment_agent'
            else:
                weight_key = agent_key
            
            # Get weight - try exact key first, then simplified key
            weight = weights_used.get(weight_key, 1.0)
            if weight == 1.0 and '_agent' in weight_key:
                simplified_key = weight_key.replace('_agent', '')
                weight = weights_used.get(simplified_key, 1.0)
            
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
            st.metric("Weighted Sum", f"{total_weighted_score:.2f}", help="Sum of all weighted scores")
        with col2:
            st.metric("Total Weight", f"{total_weight:.2f}", help="Sum of all weights")
        with col3:
            calculated_score = total_weighted_score / total_weight if total_weight > 0 else 50
            st.metric("Blended Score", f"{calculated_score:.2f}", help="Weighted average of all agent scores")
        
        # Show formula
        st.code(f"""
Formula: Blended Score = Weighted Sum / Total Weight
         Blended Score = {total_weighted_score:.2f} / {total_weight:.2f} = {calculated_score:.2f}
        """)
        
        # Weight impact analysis
        st.write("---")
        st.write("**Weight Impact Analysis:**")
        
        # Calculate equal weight score for comparison
        equal_weight_score = sum(float(agent_scores.get(k, 50)) for k in agent_order) / len(agent_order)
        weight_effect = calculated_score - equal_weight_score
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Equal Weight Score", f"{equal_weight_score:.2f}", 
                     help="What the score would be if all weights were 1.0")
        with col2:
            st.metric("Weight Effect", f"{weight_effect:+.2f}", 
                     help="How much the custom weights changed the score",
                     delta=f"{weight_effect:+.2f}")
        
        if abs(weight_effect) > 0.5:
            if weight_effect > 0:
                st.success(f"✅ Custom weights INCREASED the score by {weight_effect:.2f} points by emphasizing higher-scoring agents")
            else:
                st.warning(f"⚠️ Custom weights DECREASED the score by {abs(weight_effect):.2f} points by emphasizing lower-scoring agents")
        else:
            st.info("ℹ️ Custom weights had minimal impact on the final score")
        
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
            st.bar_chart(chart_data.set_index('Agent')['Weight'], use_container_width=True)
            st.caption("Agent Weights (Higher = More Influence)")
        with col2:
            st.bar_chart(chart_data.set_index('Agent')['Score'], use_container_width=True)
            st.caption("Agent Scores (0-100)")
    
    # Enhanced Agent Analysis Section
    st.markdown("---")
    st.markdown("### 🤖 Agent Analysis Details")
    
    # Display enhanced agent rationales with collaboration
    display_enhanced_agent_rationales(result)
    
    # Comprehensive rationale
    st.markdown("---")
    st.markdown("### 📋 Investment Rationale")
    
    with st.expander("View Full Report", expanded=False):
        # Get the comprehensive rationale from the result
        comprehensive_rationale = result.get('rationale', '')
        
        if comprehensive_rationale:
            # Display in a code block for better formatting
            st.text(comprehensive_rationale)
            
            # Add download button for the rationale
            st.download_button(
                label="📥 Download Complete Rationale (TXT)",
                data=comprehensive_rationale,
                file_name=f"{result['ticker']}_comprehensive_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                help="Download the complete investment analysis report"
            )
        else:
            st.warning("Comprehensive rationale not available")
        
        # Add a summary section
        st.write("---")
        st.write("**Key Takeaways:**")
        
        # Extract some key points
        agent_scores = result.get('agent_scores', {})
        final_score = result.get('final_score', 0)
        eligible = result.get('eligible', False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Strengths:**")
            strengths = [f"• {k.replace('_agent', '').replace('_', ' ').title()}: {v:.1f}/100" 
                        for k, v in agent_scores.items() if v >= 70]
            if strengths:
                for strength in strengths:
                    st.success(strength)
            else:
                st.info("No exceptional strengths identified")
        
        with col2:
            st.write("**Concerns:**")
            concerns = [f"• {k.replace('_agent', '').replace('_', ' ').title()}: {v:.1f}/100" 
                       for k, v in agent_scores.items() if v < 50]
            if concerns:
                for concern in concerns:
                    st.error(concern)
            else:
                st.success("No major concerns identified")
        
        # Overall assessment
        st.write("---")
        st.write("**Overall Assessment:**")
        
        if not eligible:
            st.error(f"❌ **NOT RECOMMENDED** - While the analysis score is {final_score:.1f}, this investment does not meet client suitability criteria.")
        elif final_score >= 80:
            st.success(f"✅ **STRONG BUY** - Excellent score of {final_score:.1f} with compelling fundamentals and strong multi-factor support.")
        elif final_score >= 70:
            st.success(f"✅ **BUY** - Strong score of {final_score:.1f} indicating good investment potential with favorable risk/reward profile.")
        elif final_score >= 60:
            st.info(f"⚖️ **HOLD** - Moderate score of {final_score:.1f}. Suitable for holding if already owned, but not a priority for new positions.")
        elif final_score >= 40:
            st.warning(f"⚠️ **WEAK HOLD** - Below-average score of {final_score:.1f}. Consider for portfolio review or reduction.")
        else:
            st.error(f"❌ **SELL** - Low score of {final_score:.1f} with significant concerns. Not recommended for client portfolio.")
    
    # Client validation summary
    st.markdown("---")
    st.markdown("### ✅ Client Suitability")
    if result['eligible']:
        st.success("**Approved** - Meets all suitability requirements")
    else:
        st.error("**Not Suitable** - Does not meet requirements")
        if result['client_layer'].get('violations'):
            for violation in result['client_layer']['violations']:
                st.write(f"• {violation}")
    
    # Export functionality
    with st.expander("📥 Export Analysis", expanded=False):
        agent_scores = result['agent_scores']
        export_data = {
            'Ticker': [result['ticker']],
            'Name': [result['fundamentals'].get('name', 'N/A')],
            'Eligible': [result['eligible']],
            'Final Score': [result['final_score']],
            'Sector': [result['fundamentals'].get('sector', 'N/A')],
            'Price': [result['fundamentals'].get('price', 0)],
            **{f"{agent.replace('_', ' ').title()}_Score": [score] for agent, score in agent_scores.items()}
        }
        
        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        current_timestamp = datetime.now().strftime('%Y%m%d')
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.download_button(
                label="� Download CSV Report",
                data=csv,
                file_name=f"{result['ticker']}_analysis_{current_timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # Generate detailed markdown report
            ticker = result['ticker']
            report = f"""# Investment Analysis: {ticker}
## {result['fundamentals'].get('name', 'N/A')}

**Date:** {datetime.now().strftime('%B %d, %Y')}
**Sector:** {result['fundamentals'].get('sector', 'N/A')}
**Price:** ${result['fundamentals'].get('price', 0):.2f}

---

## Score: {result['final_score']:.1f}/100
**Status:** {'✅ APPROVED' if result['eligible'] else '❌ NOT APPROVED'}

---

## Agent Breakdown
"""
            for agent, score in agent_scores.items():
                agent_name = agent.replace('_', ' ').title()
                report += f"- **{agent_name}:** {score:.1f}/100\n"
            
            report += f"\n---\n\n## Key Metrics\n"
            report += f"- **Market Cap:** ${result['fundamentals'].get('market_cap', 0)/1e9:.2f}B\n"
            report += f"- **P/E Ratio:** {result['fundamentals'].get('pe_ratio', 'N/A')}\n"
            report += f"- **Beta:** {result['fundamentals'].get('beta', 'N/A')}\n"
            
            if result['fundamentals'].get('dividend_yield'):
                report += f"- **Dividend Yield:** {result['fundamentals']['dividend_yield']*100:.2f}%\n"
            
            report += f"\n---\n\n## Agent Analysis\n"
            for agent, rationale in result.get('agent_rationales', {}).items():
                if rationale:
                    agent_name = agent.replace('_', ' ').title()
                    report += f"### {agent_name}\n{rationale}\n\n"
            
            report += f"\n---\n*Wharton Investment Analysis System*\n"
            
            st.download_button(
                label="� Download Full Report",
                data=report,
                file_name=f"{ticker}_report_{current_timestamp}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    # QA System Integration - Log for Learning
    st.markdown("---")
    st.markdown("### 🎯 Performance Tracking")
    print(f"🔧 DEBUG: *** REACHED QA SECTION FOR {result['ticker']} ***")
    
    # Check if QA system is available
    qa_system = st.session_state.get('qa_system')
    print(f"🔧 DEBUG: QA system check - Available: {qa_system is not None}")
    if not qa_system:
        st.warning("⚠️ QA System unavailable. Restart app to enable performance tracking.")
    else:
        try:
            # Show current recommendation details
            recommendation_type = _determine_recommendation_type(result['final_score'])
            confidence_score = result['final_score']
            
            # Check if already logged
            already_logged = result['ticker'] in qa_system.recommendations if qa_system else False
            print(f"🔧 DEBUG: Already logged check for {result['ticker']}: {already_logged}")
            if already_logged:
                st.info(f"ℹ️ {result['ticker']} is currently tracked")
            
            st.write(f"**{recommendation_type.value.upper()}** ({confidence_score:.1f}/100)")
            print(f"🔧 DEBUG: About to render button for {result['ticker']}")
            if st.button("🎯 Track Ticker for QA Monitoring", type="primary", use_container_width=True, key=f"track_btn_{result['ticker']}"):
                print(f"🔧 DEBUG: *** BUTTON CLICKED *** Track Ticker button clicked for {result['ticker']}")
                print(f"🔧 DEBUG: *** BUTTON PROCESSING STARTED ***")
                st.success(f"🎯 Button clicked for {result['ticker']}! Processing...")
                try:
                    ticker = result['ticker']
                    print(f"🔧 DEBUG: Processing ticker: {ticker}")
                    
                    # Check if this ticker already exists in the analysis archive
                    analysis_archive = qa_system.get_analysis_archive()
                    print(f"🔧 DEBUG: Analysis archive keys: {list(analysis_archive.keys())}")
                    
                    if ticker in analysis_archive and analysis_archive[ticker]:
                        print(f"🔧 DEBUG: Found {ticker} in analysis archive with {len(analysis_archive[ticker])} analyses")
                        # Use the most recent analysis data (same as Recent Analysis Activity)
                        most_recent_analysis = analysis_archive[ticker][0]  # Already sorted by timestamp desc
                        print(f"🔧 DEBUG: Using analysis from {most_recent_analysis.timestamp}")
                        
                        # Extract data from the existing analysis
                        price = most_recent_analysis.price_at_analysis
                        recommendation_type = most_recent_analysis.recommendation
                        confidence_score = most_recent_analysis.confidence_score
                        rationale = most_recent_analysis.final_rationale
                        agent_scores = most_recent_analysis.agent_scores
                        key_factors = most_recent_analysis.key_factors
                        sector = most_recent_analysis.sector
                        market_cap = most_recent_analysis.market_cap
                        
                        st.info(f"📊 Using existing analysis data from {most_recent_analysis.timestamp.strftime('%m/%d/%Y %H:%M')}")
                    else:
                        # Fallback to current result data if no archive entry exists
                        price = result['fundamentals'].get('price', 100.0)
                        agent_scores = result.get('agent_scores', {})
                        
                        # Create simple rationale
                        current_date = datetime.now().strftime('%Y-%m-%d')
                        rationale = f"Investment analysis for {ticker} completed on {current_date}"
                        
                        # Get basic factors
                        key_factors = ["Financial metrics", "Market analysis", "Risk assessment"]
                        
                        # Get stock info
                        sector = result['fundamentals'].get('sector', 'Unknown')
                        market_cap = result['fundamentals'].get('market_cap')
                    
                    # Log the recommendation (this moves it from analysis archive to tracked tickers)
                    print(f"🔧 DEBUG: About to log recommendation for {ticker} with price {price}")
                    success = qa_system.log_recommendation(
                        ticker=ticker,
                        price=price,
                        recommendation=recommendation_type,
                        confidence_score=confidence_score,
                        rationale=rationale,
                        agent_scores=agent_scores,
                        key_factors=key_factors,
                        sector=sector,
                        market_cap=market_cap
                    )
                    print(f"🔧 DEBUG: log_recommendation returned: {success}")
                    
                    if success:
                        # Force reload QA data from storage to ensure consistency
                        qa_system.recommendations = qa_system._load_recommendations()
                        qa_system.all_analyses = qa_system._load_all_analyses()
                        
                        # Update session state to refresh data
                        st.session_state.qa_system = qa_system
                        
                        # Debug: Show current recommendations count
                        current_count = len(qa_system.get_tracked_tickers())
                        st.success(f"✅ Successfully added {ticker} to tracked tickers!")
                        st.info(f"📊 Total tracked tickers: {current_count}")
                        
                        # Show the actual tickers for verification
                        if current_count > 0:
                            tickers_list = qa_system.get_tracked_tickers()
                            st.info(f"📋 Currently tracking: {', '.join(tickers_list)}")
                            
                        # Provide clear debugging information
                        st.success("🎯 Ticker is now being monitored in the QA system!")
                        st.info("💡 Go to the QA & Learning Center → 🎯 Tracked Tickers tab to see your analysis.")
                        # Since we can't programmatically change radio selection, show clear instruction
                        st.markdown("**👈 Click 'QA & Learning Center' in the sidebar, then go to the '🎯 Tracked Tickers' tab!**")
                        # Removed st.rerun() to prevent page refresh that loses analysis results
                    else:
                        st.error("❌ Failed to log recommendation. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error logging recommendation: {str(e)}")
                    
        except Exception as e:
            st.warning(f"⚠️ Error in QA section: {str(e)}")
    
    # Personal notes
    st.markdown("---")
    st.markdown("### 📝 Notes")
    
    # Initialize notes storage in session state
    if 'analysis_notes' not in st.session_state:
        st.session_state.analysis_notes = {}
        # Try to load from disk
        notes_file = Path("data/analysis_notes.json")
        if notes_file.exists():
            import json
            with open(notes_file, 'r') as f:
                st.session_state.analysis_notes = json.load(f)
    
    ticker = result['ticker']
    note_key = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
    
    # Get existing note if any
    existing_note = st.session_state.analysis_notes.get(note_key, "")
    
    col_note1, col_note2 = st.columns([3, 1])
    
    with col_note1:
        note_text = st.text_area(
            f"Add notes for {ticker} analysis",
            value=existing_note,
            height=100,
            placeholder="e.g., Strong fundamentals but concerned about sector headwinds. Consider for portfolio if price dips below $150...",
            help="Your personal notes are saved automatically and linked to this ticker and date"
        )
    
    with col_note2:
        st.write(" ")  # Spacing
        st.write(" ")  # Spacing
        if st.button("💾 Save Note", type="primary", use_container_width=True, key=f"save_note_{note_key}"):
            st.session_state.analysis_notes[note_key] = note_text
            # Save to disk
            import json
            notes_file = Path("data/analysis_notes.json")
            notes_file.parent.mkdir(exist_ok=True)
            with open(notes_file, 'w') as f:
                json.dump(st.session_state.analysis_notes, f, indent=2)
            st.success("✅ Note saved!")
        
        if st.button("🗑️ Clear Note", use_container_width=True, key=f"clear_note_{note_key}"):
            if note_key in st.session_state.analysis_notes:
                del st.session_state.analysis_notes[note_key]
                import json
                notes_file = Path("data/analysis_notes.json")
                with open(notes_file, 'w') as f:
                    json.dump(st.session_state.analysis_notes, f, indent=2)
                st.success("✅ Note cleared!")
                st.rerun()
    
    # Show historical notes for this ticker
    ticker_notes = {k: v for k, v in st.session_state.analysis_notes.items() if k.startswith(f"{ticker}_") and v.strip()}
    if len(ticker_notes) > 1:
        with st.expander(f"📚 View {len(ticker_notes)} Historical Notes for {ticker}"):
            for note_k in sorted(ticker_notes.keys(), reverse=True):
                date_str = note_k.split('_')[1]
                formatted_date = datetime.strptime(date_str, '%Y%m%d').strftime('%B %d, %Y')
                st.write(f"**{formatted_date}:**")
                st.info(ticker_notes[note_k])
                st.markdown("---")


def display_multiple_stock_analysis(results: list, failed_tickers: list):
    """Display analysis results for multiple stocks in a comparison table."""
    
    st.success(f"✅ Successfully analyzed {len(results)} stock{'s' if len(results) != 1 else ''}")
    
    if failed_tickers:
        st.warning(f"⚠️ Failed to analyze {len(failed_tickers)} stock{'s' if len(failed_tickers) != 1 else ''}")
        with st.expander("View Failed Tickers", expanded=False):
            for ticker_name, error_msg in failed_tickers:
                st.error(f"**{ticker_name}**: {error_msg}")
    
    # Summary comparison
    st.markdown("---")
    st.markdown("### 📊 Comparison")
    
    # Prepare data for comparison table
    comparison_data = []
    for result in results:
        row = {
            'Ticker': result['ticker'],
            'Final Score': result['final_score'],
            'Eligible': '✅' if result['eligible'] else '❌',
            'Price': result['fundamentals'].get('price', 0),
            'Market Cap': result['fundamentals'].get('market_cap', 0),
            'Sector': result['fundamentals'].get('sector', 'N/A'),
            'Value Score': result.get('agent_scores', {}).get('value_agent', 0),
            'Growth Score': result.get('agent_scores', {}).get('growth_momentum_agent', 0),
            'Macro Score': result.get('agent_scores', {}).get('macro_regime_agent', 0),
            'Risk Score': result.get('agent_scores', {}).get('risk_agent', 0),
            'Sentiment Score': result.get('agent_scores', {}).get('sentiment_agent', 0),
            'Client Score': result.get('agent_scores', {}).get('client_layer_agent', 0),
        }
        comparison_data.append(row)
    
    # Sort by final score (descending)
    comparison_data = sorted(comparison_data, key=lambda x: x['Final Score'], reverse=True)
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    
    # Format numeric columns
    df['Final Score'] = df['Final Score'].round(1)
    df['Price'] = df['Price'].apply(lambda x: f"${x:,.2f}")
    df['Market Cap'] = df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.2f}M")
    df['Value Score'] = df['Value Score'].round(1)
    df['Growth Score'] = df['Growth Score'].round(1)
    df['Macro Score'] = df['Macro Score'].round(1)
    df['Risk Score'] = df['Risk Score'].round(1)
    df['Sentiment Score'] = df['Sentiment Score'].round(1)
    df['Client Score'] = df['Client Score'].round(1)
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export to CSV button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison (CSV)",
        data=csv,
        file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Visual comparison
    st.markdown("---")
    st.markdown("### 📊 Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent Scores Comparison Bar Chart
        st.write("**Agent Scores Comparison**")
        agent_categories = ['Value', 'Growth', 'Macro', 'Risk', 'Sentiment']
        
        fig_bar = go.Figure()
        for result in results:
            scores = [
                result['agent_scores'].get('value_agent', 0),
                result['agent_scores'].get('growth_momentum_agent', 0),
                result['agent_scores'].get('macro_regime_agent', 0),
                result['agent_scores'].get('risk_agent', 0),
                result['agent_scores'].get('sentiment_agent', 0)
            ]
            fig_bar.add_trace(go.Bar(
                name=result['ticker'],
                x=agent_categories,
                y=scores,
                text=[f"{s:.1f}" for s in scores],
                textposition='auto'
            ))
        
        fig_bar.update_layout(
            barmode='group',
            yaxis_range=[0, 100],
            yaxis_title="Score",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Radar Chart for Multi-Stock Comparison
        st.write("**Multi-Dimensional Comparison**")
        
        fig_radar = go.Figure()
        
        for result in results:
            scores = [
                result['agent_scores'].get('value_agent', 0),
                result['agent_scores'].get('growth_momentum_agent', 0),
                result['agent_scores'].get('macro_regime_agent', 0),
                result['agent_scores'].get('risk_agent', 0),
                result['agent_scores'].get('sentiment_agent', 0),
                result['agent_scores'].get('value_agent', 0)  # Close the polygon
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=scores,
                theta=['Value', 'Growth', 'Macro', 'Risk', 'Sentiment', 'Value'],
                fill='toself',
                name=result['ticker']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Final Score Ranking
    st.write("**Final Score Ranking**")
    fig_final = go.Figure()
    
    tickers = [r['ticker'] for r in results]
    final_scores = [r['final_score'] for r in results]
    colors = [get_gradient_color(score) for score in final_scores]
    
    fig_final.add_trace(go.Bar(
        x=tickers,
        y=final_scores,
        marker_color=colors,
        text=[f"{s:.1f}" for s in final_scores],
        textposition='auto',
        name='Final Score'
    ))
    
    fig_final.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Final Score",
        xaxis_title="Stock",
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig_final, use_container_width=True)
    
    # Portfolio insights
    st.markdown("---")
    st.markdown("### 🎯 Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sector Diversification**")
        
        # Calculate sector distribution
        sector_counts = {}
        sector_scores = {}
        for result in results:
            sector = result['fundamentals'].get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(result['final_score'])
        
        # Create pie chart
        fig_sector = go.Figure(data=[go.Pie(
            labels=list(sector_counts.keys()),
            values=list(sector_counts.values()),
            hole=.3,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig_sector.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Sector concentration warning
        max_sector_pct = max(sector_counts.values()) / len(results) * 100
        if max_sector_pct > 40:
            st.warning(f"⚠️ High concentration: {max_sector_pct:.0f}% in one sector")
        elif max_sector_pct > 30:
            st.info(f"ℹ️ Moderate concentration: {max_sector_pct:.0f}% in one sector")
        else:
            st.success(f"✅ Well diversified: Max {max_sector_pct:.0f}% in any sector")
    
    with col2:
        st.write("**Risk Distribution Matrix**")
        
        # Create risk/score scatter plot
        risk_scores = [r['agent_scores'].get('risk_agent', 50) for r in results]
        final_scores = [r['final_score'] for r in results]
        tickers = [r['ticker'] for r in results]
        market_caps = [r['fundamentals'].get('market_cap', 0) for r in results]
        
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Scatter(
            x=risk_scores,
            y=final_scores,
            mode='markers+text',
            text=tickers,
            textposition='top center',
            marker=dict(
                size=[max(10, min(30, mc/1e10)) for mc in market_caps],  # Size by market cap
                color=final_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            hovertemplate='<b>%{text}</b><br>Risk: %{x:.1f}<br>Score: %{y:.1f}<extra></extra>'
        ))
        
        # Add quadrant lines
        fig_risk.add_hline(y=70, line_dash="dash", line_color="gray", opacity=0.5)
        fig_risk.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig_risk.add_annotation(x=85, y=85, text="High Score<br>Low Risk", showarrow=False, opacity=0.5)
        fig_risk.add_annotation(x=55, y=85, text="High Score<br>High Risk", showarrow=False, opacity=0.5)
        fig_risk.add_annotation(x=85, y=55, text="Low Score<br>Low Risk", showarrow=False, opacity=0.5)
        fig_risk.add_annotation(x=55, y=55, text="Low Score<br>High Risk", showarrow=False, opacity=0.5)
        
        fig_risk.update_layout(
            xaxis_title="Risk Score (Higher = Safer)",
            yaxis_title="Final Score",
            xaxis_range=[0, 100],
            yaxis_range=[0, 100],
            height=350
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk summary
        high_risk_count = sum(1 for r in risk_scores if r < 50)
        if high_risk_count > len(results) * 0.5:
            st.warning(f"⚠️ {high_risk_count}/{len(results)} stocks are high risk")
        else:
            st.success(f"✅ Balanced risk: {high_risk_count}/{len(results)} high risk stocks")
    
    # Sector performance breakdown
    st.write("**Sector Performance Summary**")
    sector_summary = []
    for sector, scores in sector_scores.items():
        sector_summary.append({
            'Sector': sector,
            'Count': len(scores),
            'Avg Score': sum(scores) / len(scores),
            'Max Score': max(scores),
            'Min Score': min(scores)
        })
    
    sector_df = pd.DataFrame(sector_summary).sort_values('Avg Score', ascending=False)
    sector_df['Avg Score'] = sector_df['Avg Score'].round(1)
    sector_df['Max Score'] = sector_df['Max Score'].round(1)
    sector_df['Min Score'] = sector_df['Min Score'].round(1)
    
    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    
    # Individual stock details
    st.markdown("---")
    st.markdown("### 📋 Stock Details")
    
    tabs = st.tabs([result['ticker'] for result in results])
    
    for idx, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            display_stock_analysis(result)


def _determine_recommendation_type(final_score: float) -> RecommendationType:
    """Determine recommendation type based on final score."""
    if final_score >= 80:
        return RecommendationType.STRONG_BUY
    elif final_score >= 65:
        return RecommendationType.BUY
    elif final_score >= 45:
        return RecommendationType.HOLD
    elif final_score >= 30:
        return RecommendationType.SELL
    else:
        return RecommendationType.STRONG_SELL


def get_gradient_color(score: float) -> str:
    """Generate gradient color based on score value (0-100).
    Red (low) -> Yellow (medium) -> Green (high)"""
    
    # Normalize score to 0-1 range
    normalized = max(0, min(100, score)) / 100
    
    if normalized <= 0.5:
        # Red to Yellow gradient (0-50)
        ratio = normalized * 2  # 0 to 1
        red = 255
        green = int(255 * ratio)
        blue = 0
    else:
        # Yellow to Green gradient (50-100)
        ratio = (normalized - 0.5) * 2  # 0 to 1
        red = int(255 * (1 - ratio))
        green = 255
        blue = 0
    
    return f"rgb({red},{green},{blue})"


def get_agent_specific_context(agent_key: str, result: dict) -> dict:
    """Get agent-specific context and key metrics for display."""
    
    fundamentals = result.get('fundamentals', {})
    data = result.get('data', {})
    context = {}
    
    if agent_key == 'value_agent':
        context.update({
            'P/E Ratio': f"{fundamentals.get('pe_ratio', 'N/A')}" if fundamentals.get('pe_ratio') else 'N/A',
            'Market Cap': f"${fundamentals.get('market_cap', 0)/1e9:.1f}B" if fundamentals.get('market_cap') else 'N/A',
            'Dividend Yield': f"{fundamentals.get('dividend_yield', 0)*100:.2f}%" if fundamentals.get('dividend_yield') else 'N/A',
            'Price': f"${fundamentals.get('price', 'N/A')}" if fundamentals.get('price') else 'N/A'
        })
    
    elif agent_key == 'growth_momentum_agent':
        context.update({
            'Current Price': f"${fundamentals.get('price', 'N/A')}" if fundamentals.get('price') else 'N/A',
            '52-Week High': f"${fundamentals.get('week_52_high', 'N/A')}" if fundamentals.get('week_52_high') else 'N/A',
            '52-Week Low': f"${fundamentals.get('week_52_low', 'N/A')}" if fundamentals.get('week_52_low') else 'N/A',
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
        news_count = len(result.get('news', []))
        context.update({
            'News Articles Analyzed': f"{news_count}",
            'Sector': f"{fundamentals.get('sector', 'Unknown')}",
            'Recent Price': f"${fundamentals.get('price', 'N/A')}" if fundamentals.get('price') else 'N/A'
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
    
    # Exclude client layer agent from individual rationales - it has its own section at bottom
    filtered_scores = {k: v for k, v in agent_scores.items() if k != 'client_layer_agent'}
    filtered_rationales = {k: v for k, v in agent_rationales.items() if k != 'client_layer_agent'}
    
    # Create agent names from filtered keys
    agent_names = [key.replace('_', ' ').title() for key in filtered_scores.keys()]
    
    # Agent collaboration results
    collaboration_results = get_agent_collaboration(result)
    
    # Display agent scores chart
    st.write("**� Agent Score Overview**")
    
    # Create bar chart with gradient colors (excluding client layer agent)
    fig = go.Figure()
    gradient_colors = [get_gradient_color(score) for score in filtered_scores.values()]
    
    fig.add_trace(go.Bar(
        x=agent_names,
        y=list(filtered_scores.values()),
        marker_color=gradient_colors,
        text=[f"{s:.1f}" for s in filtered_scores.values()],
        textposition='auto',
        name='Scores'
    ))
    
    fig.update_layout(
        title="Agent Analysis Scores",
        xaxis_title="",
        yaxis_title="Score",
        yaxis_range=[0, 100],
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual Agent Rationales Section
    st.write("---")
    st.write("**🧠 Individual Agent Analysis**")
    
    # Create detailed rationale display for each agent (excluding client layer agent)
    for i, (agent_key, agent_name) in enumerate(zip(filtered_scores.keys(), agent_names)):
        score = filtered_scores[agent_key]
        rationale = filtered_rationales.get(agent_key, "Analysis not available")
        
        # Create expandable section for each agent
        with st.expander(f"**{agent_name}** - Score: {score:.1f}/100", expanded=False):
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
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                ">
                    <h2 style="margin: 0; color: white;">{score:.1f}</h2>
                    <p style="margin: 0; color: white;">out of 100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Score interpretation
                if score >= 80:
                    st.success("🟢 **Excellent**\nStrong positive signals")
                elif score >= 65:
                    st.info("🔵 **Good**\nPositive with minor concerns")
                elif score >= 50:
                    st.warning("🟡 **Moderate**\nMixed signals")
                elif score >= 35:
                    st.error("🟠 **Concerning**\nSeveral negative factors")
                else:
                    st.error("🔴 **Poor**\nSignificant issues identified")
            
            with col2:
                st.write("**Detailed Analysis:**")
                
                # Display the rationale with proper formatting
                if isinstance(rationale, str) and rationale.strip():
                    # Clean up and format the rationale text
                    formatted_rationale = rationale.replace("\\n", "\n").strip()
                    
                    # Split into paragraphs for better readability
                    paragraphs = [p.strip() for p in formatted_rationale.split('\n') if p.strip()]
                    
                    for paragraph in paragraphs:
                        if paragraph.startswith('**') or paragraph.startswith('##'):
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
    
    # Client Fit Analysis (single section at the bottom)
    st.write("---")
    with st.expander("🎯 Client Fit Analysis", expanded=False):
        # Get ticker from result data
        ticker = result.get('ticker', result.get('symbol', 'UNKNOWN'))
        client_fit = analyze_client_fit(ticker, result)
        
        # Overall fit indicator
        fit_score = client_fit['fit_score']
        overall_fit = client_fit['overall_fit']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if overall_fit == 'excellent':
                st.success(f"🎯 **Excellent Fit**\n\n{fit_score:.0f}/100")
            elif overall_fit == 'good':
                st.success(f"✅ **Good Fit**\n\n{fit_score:.0f}/100")
            elif overall_fit == 'moderate':
                st.warning(f"⚖️ **Partial Fit**\n\n{fit_score:.0f}/100")
            elif overall_fit == 'poor':
                st.error(f"⚠️ **Poor Fit**\n\n{fit_score:.0f}/100")
            else:
                st.error(f"🚫 **Incompatible**\n\n{fit_score:.0f}/100")
        
        with col2:
            st.write("**Suitability Assessment:**")
            
            if overall_fit == 'excellent':
                st.write("This investment aligns exceptionally well with the client's profile, restrictions, and risk tolerance.")
            elif overall_fit == 'good':
                st.write("This investment fits well within the client's parameters with only minor considerations.")
            elif overall_fit == 'moderate': 
                st.write("**Mixed suitability** - has both positive attributes and concerning aspects that require careful evaluation.")
            elif overall_fit == 'poor':
                st.write("This investment has **significant conflicts** with the client's investment criteria and risk profile.")
            else:
                st.write("This investment is **fundamentally incompatible** with the client's investment guidelines and should be avoided.")
        
        # Comprehensive IPS Compliance Analysis
        st.write("**📋 Comprehensive IPS Compliance Analysis:**")
        try:
            analysis = generate_comprehensive_ips_compliance_analysis(ticker, result, client_fit, st.session_state.client_data)
            
            # Score Breakdown Section
            st.write("**📊 Fit Score Breakdown:**")
            score_breakdown = analysis.get('fit_score_breakdown', {})
            base_score = score_breakdown.get('base_score', 50)
            adjustments = score_breakdown.get('adjustments', [])
            final_score = score_breakdown.get('final_calculated_score', client_fit.get('fit_score', 50))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Base Score", f"{base_score}/100")
            with col2:
                total_adj = sum(adj[1] for adj in adjustments)
                st.metric("Total Adjustments", f"{total_adj:+.0f} points", delta=total_adj)
            
            # Show individual adjustments
            if adjustments:
                st.write("**Adjustment Details:**")
                for factor, impact in adjustments:
                    if impact > 0:
                        st.success(f"✅ {factor}: +{impact} points")
                    else:
                        st.error(f"❌ {factor}: {impact} points")
            
            # IPS Compliance Details
            ips_compliance = analysis.get('ips_compliance_detailed', {})
            
            # Fully Compliant Items
            fully_compliant = ips_compliance.get('fully_compliant', [])
            if fully_compliant:
                st.write("**✅ Fully IPS Compliant:**")
                for item in fully_compliant:
                    st.success(f"**{item['constraint']}**: {item['compliance']}")
                    if 'benefit' in item:
                        st.write(f"   💡 {item['benefit']}")
            
            # Partially Compliant Items
            partially_compliant = ips_compliance.get('partially_compliant', [])
            if partially_compliant:
                st.write("**⚖️ Partially Compliant (With Conditions):**")
                for item in partially_compliant:
                    st.warning(f"**{item['constraint']}**: {item['status']}")
                    if 'conditions' in item:
                        st.write(f"   📋 Conditions: {item['conditions']}")
            
            # Requires Attention Items
            requires_attention = ips_compliance.get('requires_attention', [])
            if requires_attention:
                st.write("**⚠️ Requires Attention:**")
                for item in requires_attention:
                    st.warning(f"**{item['constraint']}**: {item['concern']}")
                    if 'mitigation' in item:
                        st.write(f"   🔧 Mitigation: {item['mitigation']}")
            
            # Non-Compliant Items (Major Issues)
            non_compliant = ips_compliance.get('non_compliant', [])
            if non_compliant:
                st.write("**❌ IPS Violations (Non-Compliant):**")
                for item in non_compliant:
                    st.error(f"**{item['constraint']}**: {item['violation']}")
                    st.write(f"   🚨 Impact: {item['impact']}")
            
            # Investment Constraints Analysis
            constraints = analysis.get('investment_constraints_analysis', {})
            if constraints:
                st.write("**📏 Investment Constraints:**")
                for constraint_type, details in constraints.items():
                    if isinstance(details, dict):
                        st.write(f"**{constraint_type.replace('_', ' ').title()}:**")
                        for key, value in details.items():
                            st.write(f"   • {key.replace('_', ' ').title()}: {value}")
            
            # Score Explanation
            score_explanation = analysis.get('score_explanation', {})
            if score_explanation:
                st.write("**🔍 Score Explanation:**")
                
                if 'why_this_score' in score_explanation:
                    st.info(score_explanation['why_this_score'])
                
                if 'why_not_higher' in score_explanation:
                    st.write("**Why the score isn't higher:**")
                    for reason in score_explanation['why_not_higher']:
                        st.write(f"   • {reason}")
                
                if 'why_not_lower' in score_explanation:
                    st.write("**Why the score isn't lower:**")
                    for reason in score_explanation['why_not_lower']:
                        st.write(f"   • {reason}")
                
                if 'key_factors' in score_explanation:
                    st.write("**Key factors affecting the score:**")
                    for factor in score_explanation['key_factors']:
                        st.write(f"   • {factor}")
            
            # Final Recommendation
            recommendation = analysis.get('recommendation_rationale', 'Assessment incomplete')
            st.write("**🎯 Final Investment Recommendation:**")
            
            if 'NOT SUITABLE' in recommendation:
                st.error(f"🚫 {recommendation}")
            elif 'PROCEED WITH CAUTION' in recommendation:
                st.warning(f"⚠️ {recommendation}")
            elif 'SUITABLE' in recommendation:
                st.success(f"✅ {recommendation}")
            elif 'CONDITIONALLY SUITABLE' in recommendation:
                st.warning(f"⚖️ {recommendation}")
            else:
                st.error(f"❌ {recommendation}")
                
        except Exception as e:
            st.error(f"Error generating comprehensive IPS analysis: {str(e)}")
            st.write("**Debug Info:**")
            st.write(f"Client data available: {st.session_state.client_data is not None}")
            st.write(f"Client fit data: {client_fit}")
            import traceback
            st.code(traceback.format_exc())


def analyze_client_fit(ticker: str, result: dict, client_data: dict | None = None) -> dict:
    """
    Analyze how well a stock fits the client's investment restrictions and preferences using
    advanced LLM analysis with complete client profile and all agent scores and rationales.
    """
    
    if not client_data:
        # Try to get client data from session state or default profile
        if 'selected_profile' in st.session_state:
            try:
                from utils.client_profile_manager import ClientProfileManager
                profile_manager = ClientProfileManager()
                client_data = profile_manager.load_client_profile(st.session_state['selected_profile'])
            except:
                client_data = {}
        else:
            client_data = {}
    
    # If still no client data, create default
    if not client_data:
        client_data = {
            'risk_tolerance': 'moderate',
            'investment_style': 'balanced',
            'time_horizon': 'medium',
            'restricted_sectors': [],
            'max_position_pct': 5,
            'return_expectation': 'moderate'
        }
    
    # Extract comprehensive analysis data
    agent_scores = result.get('agent_scores', {})
    agent_rationales = result.get('agent_rationales', {})
    fundamentals = result.get('fundamentals', {})
    final_score = result.get('final_score', 50)
    
    # Generate comprehensive client fit analysis using OpenAI
    try:
        openai_client = st.session_state.get('openai_client')
        if openai_client:
            return _generate_llm_client_fit_analysis(
                ticker, client_data, agent_scores, agent_rationales, 
                fundamentals, final_score, openai_client
            )
    except Exception as e:
        print(f"Warning: LLM client fit analysis failed: {e}")
    
    # Fallback to rule-based analysis if LLM fails
    return _generate_fallback_client_fit_analysis(
        ticker, client_data, agent_scores, fundamentals, final_score
    )


def _generate_llm_client_fit_analysis(
    ticker: str, 
    client_data: dict, 
    agent_scores: dict, 
    agent_rationales: dict,
    fundamentals: dict,
    final_score: float,
    openai_client
) -> dict:
    """Generate comprehensive client fit analysis using OpenAI with complete context."""
    
    # Prepare comprehensive prompt with all available data
    client_profile_text = _format_client_profile_for_llm(client_data)
    agent_analysis_text = _format_agent_analysis_for_llm(agent_scores, agent_rationales)
    stock_fundamentals_text = _format_stock_fundamentals_for_llm(ticker, fundamentals, final_score)
    
    prompt = f"""You are an expert investment advisor conducting a comprehensive client suitability analysis.

TASK: Analyze how well the stock {ticker} fits this specific client's investment profile, constraints, and objectives.

CLIENT PROFILE:
{client_profile_text}

COMPREHENSIVE STOCK ANALYSIS:
{stock_fundamentals_text}

DETAILED AGENT ANALYSIS:
{agent_analysis_text}

ANALYSIS REQUIREMENTS:
1. Provide a numerical fit score (0-100) based on comprehensive suitability analysis
2. Identify specific positive factors that align with client requirements
3. Identify specific negative factors that conflict with client constraints
4. Provide neutral factors that are neither strongly positive nor negative
5. Give an overall fit assessment (excellent/good/moderate/poor/incompatible)
6. Include specific IPS compliance considerations
7. Address risk tolerance alignment, investment style match, sector restrictions, position sizing
8. Consider time horizon compatibility and return expectations

OUTPUT FORMAT (JSON):
{{
    "fit_score": <0-100 numerical score>,
    "overall_fit": "<excellent|good|moderate|poor|incompatible>",
    "positive_factors": [
        "Specific positive alignment factor 1",
        "Specific positive alignment factor 2"
    ],
    "negative_factors": [
        "Specific concern or constraint violation 1", 
        "Specific concern or constraint violation 2"
    ],
    "neutral_factors": [
        "Neutral factor 1",
        "Neutral factor 2"
    ],
    "ips_compliance": {{
        "fully_compliant": ["Compliant area 1", "Compliant area 2"],
        "requires_attention": ["Area needing attention 1"],
        "violations": ["Violation 1 if any"]
    }},
    "recommendation": "Detailed investment recommendation with specific position sizing and monitoring requirements",
    "key_risks": ["Primary risk 1", "Primary risk 2"],
    "monitoring_requirements": ["Monitor factor 1", "Monitor factor 2"]
}}

Ensure your analysis is thorough, specific, and directly addresses the client's unique profile and constraints."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        import json
        analysis_text = response.choices[0].message.content
        
        # Extract JSON from response
        if '```json' in analysis_text:
            json_start = analysis_text.find('```json') + 7
            json_end = analysis_text.find('```', json_start)
            analysis_text = analysis_text[json_start:json_end]
        elif '{' in analysis_text:
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            analysis_text = analysis_text[json_start:json_end]
        
        analysis = json.loads(analysis_text)
        
        # Validate and normalize the response
        return _validate_and_normalize_llm_response(analysis, ticker)
        
    except Exception as e:
        print(f"Error: LLM client fit analysis failed: {e}")
        raise


def _format_client_profile_for_llm(client_data: dict) -> str:
    """Format client profile data for LLM analysis."""
    profile_text = []
    
    if client_data.get('name'):
        profile_text.append(f"Client: {client_data['name']}")
    
    profile_text.append(f"Risk Tolerance: {client_data.get('risk_tolerance', 'moderate').title()}")
    profile_text.append(f"Investment Style: {client_data.get('investment_style', 'balanced').title()}")
    profile_text.append(f"Time Horizon: {client_data.get('time_horizon', 'medium').title()}")
    profile_text.append(f"Return Expectation: {client_data.get('return_expectation', 'moderate').title()}")
    
    if client_data.get('restricted_sectors'):
        sectors = client_data['restricted_sectors']
        if isinstance(sectors, str):
            sectors = [s.strip() for s in sectors.split(',')]
        profile_text.append(f"Restricted Sectors: {', '.join(sectors)}")
    
    profile_text.append(f"Maximum Position Size: {client_data.get('max_position_pct', 5)}%")
    
    if client_data.get('market_cap_preference'):
        profile_text.append(f"Market Cap Preference: {client_data['market_cap_preference']}")
    
    if client_data.get('esg_preference'):
        profile_text.append(f"ESG Preference: {client_data['esg_preference']}")
    
    if client_data.get('income_requirement'):
        profile_text.append(f"Income Requirement: {client_data['income_requirement']}")
    
    if client_data.get('liquidity_needs'):
        profile_text.append(f"Liquidity Needs: {client_data['liquidity_needs']}")
    
    return '\n'.join(profile_text)


def _format_agent_analysis_for_llm(agent_scores: dict, agent_rationales: dict) -> str:
    """Format agent analysis data for LLM."""
    analysis_text = []
    
    agent_names = {
        'value_agent': 'Value Analysis',
        'growth_momentum_agent': 'Growth & Momentum Analysis', 
        'macro_regime_agent': 'Macro Economic Analysis',
        'risk_agent': 'Risk Analysis',
        'sentiment_agent': 'Market Sentiment Analysis'
    }
    
    for agent_key, display_name in agent_names.items():
        score = agent_scores.get(agent_key, 50)
        rationale = agent_rationales.get(agent_key, "Analysis not available")
        
        analysis_text.append(f"\n{display_name}: {score:.1f}/100")
        analysis_text.append(f"Rationale: {rationale}")
    
    return '\n'.join(analysis_text)


def _format_stock_fundamentals_for_llm(ticker: str, fundamentals: dict, final_score: float) -> str:
    """Format stock fundamentals for LLM analysis."""
    fund_text = [f"Stock: {ticker}"]
    fund_text.append(f"Overall Analysis Score: {final_score:.1f}/100")
    
    if fundamentals.get('name'):
        fund_text.append(f"Company: {fundamentals['name']}")
    
    if fundamentals.get('sector'):
        fund_text.append(f"Sector: {fundamentals['sector']}")
    
    if fundamentals.get('price'):
        fund_text.append(f"Current Price: ${fundamentals['price']:.2f}")
    
    if fundamentals.get('market_cap'):
        mc = fundamentals['market_cap']
        if mc > 1_000_000_000:
            fund_text.append(f"Market Cap: ${mc/1_000_000_000:.1f}B")
        else:
            fund_text.append(f"Market Cap: ${mc/1_000_000:.0f}M")
    
    if fundamentals.get('pe_ratio'):
        fund_text.append(f"P/E Ratio: {fundamentals['pe_ratio']:.1f}")
    
    if fundamentals.get('beta'):
        fund_text.append(f"Beta: {fundamentals['beta']:.2f}")
    
    if fundamentals.get('dividend_yield'):
        fund_text.append(f"Dividend Yield: {fundamentals['dividend_yield']*100:.2f}%")
    
    if fundamentals.get('week_52_low') and fundamentals.get('week_52_high'):
        fund_text.append(f"52-Week Range: ${fundamentals['week_52_low']:.2f} - ${fundamentals['week_52_high']:.2f}")
    
    return '\n'.join(fund_text)


def _validate_and_normalize_llm_response(analysis: dict, ticker: str) -> dict:
    """Validate and normalize LLM response to ensure consistency."""
    
    # Ensure required keys exist
    normalized = {
        'fit_score': max(0, min(100, analysis.get('fit_score', 50))),
        'overall_fit': analysis.get('overall_fit', 'moderate'),
        'positive_factors': analysis.get('positive_factors', []),
        'negative_factors': analysis.get('negative_factors', []),
        'neutral_factors': analysis.get('neutral_factors', []),
        'ips_compliance': analysis.get('ips_compliance', {}),
        'recommendation': analysis.get('recommendation', f"Analysis completed for {ticker}"),
        'key_risks': analysis.get('key_risks', []),
        'monitoring_requirements': analysis.get('monitoring_requirements', [])
    }
    
    # Validate overall_fit values
    valid_fits = ['excellent', 'good', 'moderate', 'poor', 'incompatible']
    if normalized['overall_fit'] not in valid_fits:
        normalized['overall_fit'] = 'moderate'
    
    # Ensure IPS compliance structure
    normalized['ips_compliance'] = {
        'fully_compliant': normalized['ips_compliance'].get('fully_compliant', []),
        'requires_attention': normalized['ips_compliance'].get('requires_attention', []),
        'violations': normalized['ips_compliance'].get('violations', [])
    }
    
    return normalized


def _generate_fallback_client_fit_analysis(
    ticker: str, 
    client_data: dict, 
    agent_scores: dict, 
    fundamentals: dict,
    final_score: float
) -> dict:
    """Fallback rule-based analysis if LLM fails."""
    
    # Basic rule-based analysis
    risk_score = agent_scores.get('risk_agent', 50) or 50
    value_score = agent_scores.get('value_agent', 50) or 50
    growth_score = agent_scores.get('growth_momentum_agent', 50) or 50
    
    # Get stock data for sector/industry analysis
    sector = fundamentals.get('sector', 'Unknown')
    market_cap = fundamentals.get('market_cap') or 0
    
    fit_score = 50  # Start at neutral
    positive_factors = []
    negative_factors = []
    neutral_factors = []
    
    # Risk tolerance analysis
    risk_tolerance = client_data.get('risk_tolerance', 'moderate').lower()
    if risk_tolerance == 'low' and risk_score > 70:
        negative_factors.append(f"High risk score ({risk_score:.1f}) conflicts with conservative risk tolerance")
        fit_score -= 15
    elif risk_tolerance == 'high' and risk_score < 40:
        negative_factors.append(f"Low risk score ({risk_score:.1f}) may not meet aggressive growth expectations")
        fit_score -= 10
    elif risk_tolerance == 'moderate' and 40 <= risk_score <= 70:
        positive_factors.append(f"Risk profile ({risk_score:.1f}) aligns with moderate risk tolerance")
        fit_score += 10
    
    # Investment style analysis
    investment_style = client_data.get('investment_style', 'balanced').lower()
    if investment_style == 'value' and value_score > 65:
        positive_factors.append(f"Strong value characteristics ({value_score:.1f}) match value investment style")
        fit_score += 15
    elif investment_style == 'growth' and growth_score > 65:
        positive_factors.append(f"Strong growth potential ({growth_score:.1f}) aligns with growth investment style")
        fit_score += 15
    elif investment_style == 'balanced':
        avg_score = (value_score + growth_score) / 2
        if avg_score > 55:
            positive_factors.append(f"Balanced value/growth profile ({avg_score:.1f}) suits balanced approach")
            fit_score += 10
    
    # Sector restrictions
    restricted_sectors = client_data.get('restricted_sectors', [])
    if restricted_sectors and isinstance(restricted_sectors, str):
        restricted_sectors = [s.strip().lower() for s in restricted_sectors.split(',')]
    elif isinstance(restricted_sectors, list):
        restricted_sectors = [s.lower() for s in restricted_sectors]
    
    if restricted_sectors and sector.lower() in restricted_sectors:
        negative_factors.append(f"Stock is in restricted sector: {sector}")
        fit_score -= 30
    
    # Overall score considerations
    if final_score > 70:
        positive_factors.append(f"Strong overall analysis score ({final_score:.1f})")
        fit_score += 10
    elif final_score < 40:
        negative_factors.append(f"Weak overall analysis score ({final_score:.1f})")
        fit_score -= 15
    
    # Determine overall fit
    fit_score = max(0, min(100, fit_score))
    
    if fit_score >= 75:
        overall_fit = 'high'
    elif fit_score >= 60:
        overall_fit = 'good'  
    elif fit_score >= 40:
        overall_fit = 'moderate'
    elif fit_score >= 25:
        overall_fit = 'poor'
    else:
        overall_fit = 'incompatible'
    
    # Add some neutral factors if we don't have many
    if len(positive_factors) + len(negative_factors) < 3:
        neutral_factors.append(f"Market cap: ${market_cap/1_000_000:.0f}M")
        neutral_factors.append(f"Sector: {sector}")
    
    fit_analysis = {
        'fit_score': fit_score,
        'overall_fit': overall_fit,
        'positive_factors': positive_factors,
        'negative_factors': negative_factors,
        'neutral_factors': neutral_factors,
        'ips_compliance': {
            'fully_compliant': ["Basic suitability assessment completed"],
            'requires_attention': [],
            'violations': []
        },
        'recommendation': f"Fallback analysis completed for {ticker}. Consider detailed manual review.",
        'key_risks': ["Limited analysis depth", "Rule-based assessment only"],
        'monitoring_requirements': ["Regular portfolio review", "Performance monitoring"]
    }
    
    return fit_analysis


def generate_comprehensive_ips_compliance_analysis(ticker: str, result: dict, client_fit: dict, client_data: dict | None = None) -> dict:
    """Generate comprehensive IPS compliance analysis with detailed score breakdown and specific rationale."""
    
    if not client_data:
        if 'selected_profile' in st.session_state:
            try:
                from utils.client_profile_manager import ClientProfileManager
                profile_manager = ClientProfileManager()
                client_data = profile_manager.load_client_profile(st.session_state['selected_profile'])
            except:
                client_data = {}
        else:
            client_data = {}
    
    # Extract comprehensive metrics for detailed analysis
    agent_scores = result.get('agent_scores', {})
    risk_score = agent_scores.get('risk_agent', 50) or 50
    value_score = agent_scores.get('value_agent', 50) or 50
    growth_score = agent_scores.get('growth_momentum_agent', 50) or 50
    sentiment_score = agent_scores.get('sentiment_agent', 50) or 50
    macro_score = agent_scores.get('macro_regime_agent', 50) or 50
    
    stock_data = result.get('data', {})
    fundamentals = result.get('fundamentals', {})
    final_score = result.get('final_score', 50)
    
    # Extract detailed financial metrics
    sector = (stock_data.get('sector') or fundamentals.get('sector') or 'Unknown')
    market_cap = fundamentals.get('market_cap') or 0
    pe_ratio = fundamentals.get('pe_ratio')
    beta = fundamentals.get('beta')
    # Treat 0 as None for dividend_yield (0 means no dividend data, not 0% yield)
    dividend_yield = fundamentals.get('dividend_yield')
    if dividend_yield == 0:
        dividend_yield = None
    price = fundamentals.get('price', 0)
    
    # Initialize comprehensive analysis structure
    analysis = {
        'fit_score_breakdown': {
            'base_score': 50,
            'adjustments': [],
            'final_calculated_score': client_fit.get('fit_score', 50)
        },
        'ips_compliance_detailed': {
            'fully_compliant': [],
            'partially_compliant': [],
            'non_compliant': [],
            'requires_attention': []
        },
        'score_explanation': {
            'why_this_score': '',
            'why_not_higher': [],
            'why_not_lower': [],
            'key_factors': []
        },
        'investment_constraints_analysis': {},
        'recommendation_rationale': ''
    }
    
    # COMPREHENSIVE IPS COMPLIANCE ANALYSIS
    
    # 1. RISK TOLERANCE COMPLIANCE ANALYSIS
    risk_tolerance = client_data.get('risk_tolerance', 'moderate').lower()
    
    if risk_tolerance == 'conservative':
        if risk_score > 80:
            analysis['ips_compliance_detailed']['non_compliant'].append({
                'constraint': 'Conservative Risk Tolerance',
                'violation': f'Risk score {risk_score:.1f}/100 significantly exceeds conservative limits',
                'impact': 'Major compliance violation - unsuitable for conservative investor',
                'score_impact': -20
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Risk Tolerance Violation', -20))
        elif risk_score > 65:
            analysis['ips_compliance_detailed']['requires_attention'].append({
                'constraint': 'Conservative Risk Tolerance',
                'concern': f'Risk score {risk_score:.1f}/100 at upper boundary of conservative tolerance',
                'mitigation': 'Reduced position size (max 3-4% allocation) and enhanced monitoring required',
                'score_impact': -10
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Risk Tolerance Concern', -10))
        elif risk_score > 50:
            analysis['ips_compliance_detailed']['partially_compliant'].append({
                'constraint': 'Conservative Risk Tolerance',
                'status': f'Risk score {risk_score:.1f}/100 within acceptable range with conditions',
                'conditions': 'Standard position sizing (max 5% allocation) with quarterly reviews',
                'score_impact': -5
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Risk Tolerance Conditional', -5))
        else:
            analysis['ips_compliance_detailed']['fully_compliant'].append({
                'constraint': 'Conservative Risk Tolerance',
                'compliance': f'Risk score {risk_score:.1f}/100 fully compliant with conservative parameters',
                'benefit': 'Suitable for core conservative portfolio allocation up to 7%',
                'score_impact': +5
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Risk Tolerance Match', +5))
    
    elif risk_tolerance == 'aggressive':
        if risk_score < 40:
            analysis['ips_compliance_detailed']['partially_compliant'].append({
                'constraint': 'Aggressive Risk Tolerance',
                'status': f'Risk score {risk_score:.1f}/100 below aggressive targets but acceptable',
                'note': 'May not maximize return potential given risk tolerance',
                'score_impact': -5
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Below Risk Target', -5))
        else:
            analysis['ips_compliance_detailed']['fully_compliant'].append({
                'constraint': 'Aggressive Risk Tolerance',
                'compliance': f'Risk score {risk_score:.1f}/100 aligns with aggressive risk parameters',
                'benefit': 'Suitable for growth-oriented allocations up to 10%',
                'score_impact': +8
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Risk Tolerance Match', +8))
    
    # 2. INVESTMENT STYLE COMPLIANCE
    investment_style = client_data.get('investment_style', 'balanced').lower()
    
    if 'value' in investment_style:
        if value_score > 75:
            # Build metrics string with available data
            metrics_parts = []
            if pe_ratio:
                metrics_parts.append(f'P/E ratio {pe_ratio:.1f}x')
            if dividend_yield:
                metrics_parts.append(f'dividend yield {dividend_yield*100:.1f}%')
            metrics_str = ', '.join(metrics_parts) if metrics_parts else 'Strong value characteristics confirmed'
            
            analysis['ips_compliance_detailed']['fully_compliant'].append({
                'constraint': 'Value Investment Style',
                'compliance': f'Value score {value_score:.1f}/100 strongly matches value mandate',
                'metrics': metrics_str,
                'score_impact': +12
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Value Style Strong Match', +12))
        elif value_score > 55:
            analysis['ips_compliance_detailed']['partially_compliant'].append({
                'constraint': 'Value Investment Style',
                'status': f'Value score {value_score:.1f}/100 provides moderate value characteristics',
                'conditions': 'Acceptable for blended value approach with reduced allocation',
                'score_impact': +5
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Value Style Moderate', +5))
        else:
            analysis['ips_compliance_detailed']['requires_attention'].append({
                'constraint': 'Value Investment Style',
                'concern': f'Value score {value_score:.1f}/100 insufficient for value mandate',
                'recommendation': 'Consider alternative value opportunities or style drift analysis',
                'score_impact': -8
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Value Style Mismatch', -8))
    
    # 3. SECTOR RESTRICTIONS COMPLIANCE
    restricted_sectors = client_data.get('restricted_sectors', [])
    if isinstance(restricted_sectors, str):
        restricted_sectors = [s.strip() for s in restricted_sectors.split(',')]
    
    if restricted_sectors:
        sector_violation = any(sector.lower() in rs.lower() or rs.lower() in sector.lower() 
                              for rs in restricted_sectors if rs.strip())
        if sector_violation:
            analysis['ips_compliance_detailed']['non_compliant'].append({
                'constraint': 'Sector Restrictions',
                'violation': f'{sector} sector explicitly prohibited in IPS',
                'restricted_list': ', '.join(restricted_sectors),
                'impact': 'Absolute exclusion - cannot be held regardless of other merits',
                'score_impact': -50  # Major penalty
            })
            analysis['fit_score_breakdown']['adjustments'].append(('Sector Restriction Violation', -50))
        else:
            analysis['ips_compliance_detailed']['fully_compliant'].append({
                'constraint': 'Sector Restrictions',
                'compliance': f'{sector} sector approved - not in restricted list',
                'cleared_restrictions': ', '.join(restricted_sectors),
                'score_impact': 0
            })
    
    # 4. POSITION SIZE CONSTRAINTS
    max_position = client_data.get('max_position_pct', 5)
    if risk_score > 70:
        recommended_position = min(max_position, 3)
        analysis['investment_constraints_analysis']['position_sizing'] = {
            'max_allowed': f'{max_position}%',
            'recommended': f'{recommended_position}%',
            'rationale': f'Reduced from {max_position}% due to elevated risk score ({risk_score:.1f})',
            'compliance_status': 'Requires reduced sizing'
        }
        analysis['fit_score_breakdown']['adjustments'].append(('Position Size Constraint', -3))
    else:
        analysis['investment_constraints_analysis']['position_sizing'] = {
            'max_allowed': f'{max_position}%',
            'recommended': f'{max_position}%',
            'rationale': f'Standard sizing appropriate given risk score ({risk_score:.1f})',
            'compliance_status': 'Fully compliant'
        }
    
    # 5. LIQUIDITY REQUIREMENTS
    if market_cap < 1_000_000_000:  # Less than $1B
        analysis['ips_compliance_detailed']['requires_attention'].append({
            'constraint': 'Liquidity Requirements',
            'concern': f'${market_cap/1e6:.0f}M market cap may present liquidity constraints',
            'mitigation': 'Reduced position size and extended trading timeframes required',
            'score_impact': -8
        })
        analysis['fit_score_breakdown']['adjustments'].append(('Liquidity Concern', -8))
    elif market_cap > 10_000_000_000:  # Greater than $10B
        analysis['ips_compliance_detailed']['fully_compliant'].append({
            'constraint': 'Liquidity Requirements',
            'compliance': f'${market_cap/1e9:.1f}B market cap provides excellent liquidity',
            'benefit': 'Enables flexible position sizing and efficient execution',
            'score_impact': +3
        })
        analysis['fit_score_breakdown']['adjustments'].append(('Liquidity Advantage', +3))
    
    # CALCULATE FINAL SCORE AND EXPLANATIONS
    base_score = analysis['fit_score_breakdown']['base_score']
    total_adjustments = sum(adj[1] for adj in analysis['fit_score_breakdown']['adjustments'])
    calculated_score = max(0, min(100, base_score + total_adjustments))
    
    analysis['fit_score_breakdown']['final_calculated_score'] = calculated_score
    
    # DETAILED SCORE EXPLANATION
    actual_fit_score = client_fit.get('fit_score', calculated_score)
    
    analysis['score_explanation']['why_this_score'] = f"""
    The {actual_fit_score:.0f}/100 client fit score reflects a comprehensive IPS compliance analysis:
    
    BASE SCORE: {base_score}/100 (neutral starting point)
    ADJUSTMENTS: {total_adjustments:+.0f} points from IPS factors
    CALCULATED: {calculated_score:.0f}/100
    """
    
    # Why not higher?
    if actual_fit_score < 80:
        negative_adjustments = [adj for adj in analysis['fit_score_breakdown']['adjustments'] if adj[1] < 0]
        analysis['score_explanation']['why_not_higher'] = [
            f"{factor}: {impact:+.0f} points" for factor, impact in negative_adjustments
        ]
    
    # Why not lower?  
    if actual_fit_score > 40:
        positive_adjustments = [adj for adj in analysis['fit_score_breakdown']['adjustments'] if adj[1] > 0]
        analysis['score_explanation']['why_not_lower'] = [
            f"{factor}: {impact:+.0f} points" for factor, impact in positive_adjustments  
        ]
    
    # Key factors
    significant_factors = [adj for adj in analysis['fit_score_breakdown']['adjustments'] if abs(adj[1]) >= 5]
    analysis['score_explanation']['key_factors'] = [
        f"{factor} ({impact:+.0f})" for factor, impact in significant_factors
    ]
    
    # FINAL RECOMMENDATION
    compliance_violations = len(analysis['ips_compliance_detailed']['non_compliant'])
    attention_items = len(analysis['ips_compliance_detailed']['requires_attention'])
    
    if compliance_violations > 0:
        analysis['recommendation_rationale'] = f"NOT SUITABLE: {compliance_violations} IPS violation(s) present. Major constraints prevent investment regardless of financial merits."
    elif attention_items > 2:
        analysis['recommendation_rationale'] = f"PROCEED WITH CAUTION: {attention_items} items require attention. Enhanced due diligence and modified parameters necessary."
    elif actual_fit_score >= 70:
        analysis['recommendation_rationale'] = f"SUITABLE: Strong IPS alignment with score of {actual_fit_score}/100. Standard investment process and sizing appropriate."
    elif actual_fit_score >= 50:
        analysis['recommendation_rationale'] = f"CONDITIONALLY SUITABLE: Moderate fit ({actual_fit_score}/100) with specific conditions. Reduced sizing and enhanced monitoring recommended."
    else:
        analysis['recommendation_rationale'] = f"POOR FIT: Low compatibility ({actual_fit_score}/100) with client IPS. Consider alternatives better aligned with constraints."
    
    return analysis


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

{'🏛️ **Large-Cap Classification:** Recognized as inherently lower risk due to institutional size, market liquidity, and regulatory oversight.' if is_low_risk else '**Standard Risk Assessment:** Evaluated using traditional risk metrics without size-based adjustments.'}

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
        num_articles = sentiment_details.get('num_articles', 0)
        
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
            for i, article in enumerate(article_details, 1):
                analysis += f"""

**Article {i}: {article['source']}**
- **Title:** {article['title']}
- **Published:** {article['published_at']}
- **Description:** {article['description']}
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


def portfolio_recommendations_page():
    """Portfolio recommendation page with AI-powered selection."""
    st.header("🎯 AI-Powered Portfolio Recommendations")
    st.write("Multi-stage AI selection using OpenAI and Perplexity to identify optimal stocks.")
    st.markdown("---")
    
    # Challenge context input
    st.subheader("📋 Investment Challenge Context")
    challenge_context = st.text_area(
        "Describe the investment challenge, goals, and requirements:",
        value="""Generate an optimal diversified portfolio that maximizes risk-adjusted returns 
while adhering to the client's Investment Policy Statement constraints.
Focus on high-quality companies with strong fundamentals and growth potential.""",
        height=120,
        help="Provide detailed context about the investment challenge"
    )
    
    st.markdown("---")
    
    # Configuration options
    with st.expander("⚙️ Portfolio Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            selection_mode = st.selectbox(
                "Selection Mode",
                ["AI-Powered Selection (Recommended)", "Manual Ticker Input"],
                help="AI-Powered uses OpenAI + Perplexity to select best tickers across ALL market caps"
            )
        
        with col2:
            num_positions = st.number_input(
                "Target Portfolio Positions",
                min_value=3,
                max_value=20,
                value=5,
                help="Target number of holdings in portfolio (up to 20 for diversified growth)"
            )
    
    # Advanced options
    with st.expander("🎛️ Investment Focus & Strategy"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Investment Focus**")
            focus_value = st.checkbox("Emphasize Value (Undervalued stocks)", value=False)
            focus_growth = st.checkbox("Emphasize Growth & Momentum", value=False)
            focus_upside = st.checkbox("Emphasize Potential Upside (High-growth niche stocks)", value=False, 
                                      help="Discover small-cap and emerging companies with massive growth potential")
            focus_dividend = st.checkbox("Emphasize Dividend Income", value=False)
            focus_lowrisk = st.checkbox("Emphasize Low Volatility", value=False)
        
        with col2:
            st.markdown("**Portfolio Strategy**")
            sector_constraint = st.selectbox(
                "Sector Diversification",
                ["No Preference", "Tech-Heavy", "Tech-Light", "Diversified Only"],
                help="Control sector concentration"
            )
            
            market_cap_pref = st.selectbox(
                "Market Cap Preference",
                ["All Market Caps (Best opportunities anywhere)", 
                 "Small & Mid Cap Focus (Higher growth potential)", 
                 "Large Cap Focus (Established companies)",
                 "Mix of All Sizes"],
                index=0,
                help="Define which company sizes to prioritize"
            )
    
    # Build custom instructions from advanced options
    custom_instructions = []
    if focus_value:
        custom_instructions.append("Prioritize value stocks with low P/E ratios, strong fundamentals, and attractive valuations.")
    if focus_growth:
        custom_instructions.append("Seek high-growth companies with strong revenue acceleration and momentum indicators.")
    if focus_upside:
        custom_instructions.append("CRITICAL: Discover hidden gems - small-cap, mid-cap, and emerging companies with MASSIVE growth potential. Look beyond well-known names. Seek niche players, disruptors, and companies in high-growth sectors (AI, biotech, clean energy, fintech, SaaS, semiconductors). Market cap is NOT a constraint - find the best opportunities regardless of size.")
    if focus_dividend:
        custom_instructions.append("Include dividend-paying stocks with sustainable yields above 2%.")
    if focus_lowrisk:
        custom_instructions.append("Favor low-beta stocks with reduced volatility and defensive characteristics.")
    
    if sector_constraint == "Tech-Heavy":
        custom_instructions.append("Allocate 40-60% to technology sector stocks.")
    elif sector_constraint == "Tech-Light":
        custom_instructions.append("Limit technology sector exposure to 20% maximum.")
    elif sector_constraint == "Diversified Only":
        custom_instructions.append("Ensure no single sector exceeds 25% of portfolio weight.")
    
    if market_cap_pref == "Small & Mid Cap Focus (Higher growth potential)":
        custom_instructions.append("Focus primarily on small-cap ($300M-$2B) and mid-cap ($2B-$10B) companies with high growth potential.")
    elif market_cap_pref == "Large Cap Focus (Established companies)":
        custom_instructions.append("Focus on large-cap companies ($10B+) with established market positions.")
    elif market_cap_pref == "Mix of All Sizes":
        custom_instructions.append("Include a balanced mix of small-cap, mid-cap, and large-cap companies.")
    else:  # All Market Caps
        custom_instructions.append("Consider companies of ALL sizes - from small-cap emerging players to mega-cap leaders. Find the best opportunities regardless of market capitalization.")
    
    # Append custom instructions to challenge context
    if custom_instructions:
        challenge_context += "\n\nAdditional Requirements:\n" + "\n".join(f"- {inst}" for inst in custom_instructions)
    
    # Manual ticker input (optional)
    if selection_mode == "Manual Ticker Input":
        custom_tickers = st.text_area(
            "Enter Tickers (comma-separated):",
            value="AAPL, MSFT, GOOGL, AMZN, NVDA",
            help="Enter stock tickers separated by commas"
        )
        tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
    else:
        tickers = None  # Will use AI selection
        st.info("""
        🤖 **AI Selection Process:**
        1. OpenAI selects 20 best tickers
        2. Perplexity selects 20 best tickers  
        3. Aggregate to 40 unique candidates
        4. Generate 4-sentence rationale for each
        5. Run 3 rounds of top-5 selection
        6. Consolidate to final 5 tickers
        7. Full analysis on all final selections
        """)
    
    if st.button("🚀 Generate Portfolio", type="primary", use_container_width=True):
        with st.spinner("🤖 Running AI-powered portfolio generation..."):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate recommendations
                status_text.text("Stage 1/4: AI Ticker Selection (Searching ALL market caps for best opportunities)...")
                progress_bar.progress(25)
                
                result = st.session_state.orchestrator.recommend_portfolio(
                    challenge_context=challenge_context,
                    tickers=tickers,
                    num_positions=num_positions
                )
                
                status_text.text("Stage 2/4: Analyzing Stocks...")
                progress_bar.progress(50)
                
                status_text.text("Stage 3/4: Constructing Portfolio...")
                progress_bar.progress(75)
                
                status_text.text("Stage 4/4: Finalizing Recommendations...")
                progress_bar.progress(100)
                
                status_text.text("✅ Portfolio generation complete!")
                
                # Store result in session state
                st.session_state.portfolio_result = result
                
                # Log ALL analyzed stocks to QA archive (every stock gets same treatment as individual analysis)
                status_text.text("📝 Logging all analyzed stocks to QA archive...")
                all_analyses = result.get('all_analyses', [])
                for analysis in all_analyses:
                    try:
                        # Convert analysis dict to StockAnalysis object if needed
                        if not hasattr(analysis, 'ticker'):
                            # It's already a dict, convert to StockAnalysis-like object
                            from types import SimpleNamespace
                            analysis_obj = SimpleNamespace(**analysis)
                            # Convert nested dicts
                            if 'fundamentals' in analysis and isinstance(analysis['fundamentals'], dict):
                                analysis_obj.fundamentals = SimpleNamespace(**analysis['fundamentals'])
                            if 'agent_scores' in analysis and isinstance(analysis['agent_scores'], dict):
                                analysis_obj.agent_scores = analysis['agent_scores']
                            if 'agent_rationales' in analysis and isinstance(analysis['agent_rationales'], dict):
                                analysis_obj.agent_rationales = analysis['agent_rationales']
                            if 'recommendation' in analysis:
                                # Convert string recommendation to RecommendationType enum
                                from utils.qa_system import RecommendationType
                                rec_str = analysis['recommendation'].upper()
                                # Map orchestrator recommendations to QA system types
                                if 'STRONG BUY' in rec_str or 'STRONG_BUY' in rec_str:
                                    analysis_obj.recommendation = RecommendationType.STRONG_BUY
                                elif 'BUY' in rec_str:
                                    analysis_obj.recommendation = RecommendationType.BUY
                                elif 'STRONG SELL' in rec_str or 'STRONG_SELL' in rec_str:
                                    analysis_obj.recommendation = RecommendationType.STRONG_SELL
                                elif 'SELL' in rec_str or 'AVOID' in rec_str:
                                    analysis_obj.recommendation = RecommendationType.SELL
                                elif 'HOLD' in rec_str:
                                    analysis_obj.recommendation = RecommendationType.HOLD
                                else:
                                    # Default to HOLD if unclear
                                    analysis_obj.recommendation = RecommendationType.HOLD
                            analysis = analysis_obj
                        
                        # Log to QA system using correct method
                        if hasattr(st.session_state, 'qa_system') and st.session_state.qa_system:
                            # Extract values from either dict or SimpleNamespace
                            ticker = getattr(analysis, 'ticker', None) or analysis.get('ticker') if isinstance(analysis, dict) else getattr(analysis, 'ticker', 'UNKNOWN')
                            
                            # Get fundamentals first to extract price
                            fundamentals = getattr(analysis, 'fundamentals', {}) if hasattr(analysis, 'fundamentals') else analysis.get('fundamentals', {}) if isinstance(analysis, dict) else {}
                            
                            # Extract price from fundamentals (portfolio analysis stores it there)
                            if isinstance(fundamentals, dict):
                                fund_dict = fundamentals
                            elif hasattr(fundamentals, '__dict__'):
                                fund_dict = fundamentals.__dict__
                            elif hasattr(fundamentals, 'get'):
                                fund_dict = fundamentals
                            else:
                                fund_dict = {}
                            
                            price = fund_dict.get('price', 0) if fund_dict else 0
                            
                            # Get confidence score from final_score (portfolio analysis uses this)
                            confidence = getattr(analysis, 'final_score', 0) if hasattr(analysis, 'final_score') else analysis.get('final_score', 0) if isinstance(analysis, dict) else 0
                            
                            recommendation = getattr(analysis, 'recommendation', RecommendationType.HOLD)
                            final_rationale = getattr(analysis, 'rationale', 'Portfolio generation analysis') if hasattr(analysis, 'rationale') else analysis.get('rationale', 'Portfolio generation analysis') if isinstance(analysis, dict) else 'Portfolio generation analysis'
                            agent_scores = getattr(analysis, 'agent_scores', {}) if hasattr(analysis, 'agent_scores') else analysis.get('agent_scores', {}) if isinstance(analysis, dict) else {}
                            agent_rationales = getattr(analysis, 'agent_rationales', {}) if hasattr(analysis, 'agent_rationales') else analysis.get('agent_rationales', {}) if isinstance(analysis, dict) else {}
                            
                            st.session_state.qa_system.log_complete_analysis(
                                ticker=ticker,
                                price=price,
                                recommendation=recommendation,
                                confidence_score=confidence,
                                final_rationale=final_rationale,
                                agent_scores=agent_scores,
                                agent_rationales=agent_rationales,
                                key_factors=[],
                                fundamentals=fund_dict,
                                sector=fund_dict.get('sector'),
                                market_cap=fund_dict.get('market_cap')
                            )
                    except Exception as e:
                        # Get ticker safely for error message
                        ticker = 'unknown'
                        if isinstance(analysis, dict):
                            ticker = analysis.get('ticker', 'unknown')
                        elif hasattr(analysis, 'ticker'):
                            ticker = analysis.ticker
                        st.warning(f"Failed to log {ticker} to QA archive: {e}")
                
                status_text.text(f"✅ Logged {len(all_analyses)} analyses to QA archive")
                
                # Auto-update Google Sheets if enabled
                if st.session_state.sheets_auto_update and st.session_state.sheets_integration.sheet:
                    status_text.text("📊 Updating Google Sheets...")
                    
                    # Update both QA Analyses sheet (all stocks) and Portfolio Recommendations sheet (selected only)
                    sheets_success = update_google_sheets_portfolio(result)
                    
                    # Also update QA analyses with all analyzed stocks
                    if hasattr(st.session_state, 'qa_system') and st.session_state.qa_system:
                        qa_archive = st.session_state.qa_system.get_analysis_archive()
                        update_google_sheets_qa_analyses(qa_archive)
                    
                    if sheets_success:
                        st.success("✅ Google Sheets updated successfully!")
                    else:
                        st.warning("⚠️ Google Sheets update failed (see logs)")
                
                # Display results
                display_portfolio_recommendations(result)
                
            except Exception as e:
                st.error(f"❌ Portfolio generation failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def display_portfolio_recommendations(result: dict):
    """Display portfolio recommendations with AI selection details."""
    
    portfolio = result['portfolio']
    summary = result['summary']
    selection_log = result.get('selection_log', {})
    
    if not portfolio:
        st.warning("No eligible stocks found in universe")
        return
    
    # AI Selection Summary (if available)
    if not selection_log.get('manual_selection', False):
        st.subheader("🤖 AI Selection Process")
        
        with st.expander("View AI Selection Details", expanded=False):
            stages = selection_log.get('stages', [])
            
            for stage_info in stages:
                stage = stage_info.get('stage', 'Unknown')
                
                if stage == 'openai_initial_selection':
                    st.markdown("#### 1️⃣ OpenAI Initial Selection")
                    tickers = stage_info.get('tickers', [])
                    st.write(f"Selected {len(tickers)} tickers: {', '.join(tickers)}")
                
                elif stage == 'perplexity_initial_selection':
                    st.markdown("#### 2️⃣ Perplexity Initial Selection")
                    tickers = stage_info.get('tickers', [])
                    st.write(f"Selected {len(tickers)} tickers: {', '.join(tickers)}")
                
                elif stage == 'aggregation':
                    st.markdown("#### 3️⃣ Aggregation")
                    count = stage_info.get('count', 0)
                    st.write(f"Total unique candidates: **{count}** tickers")
                
                elif stage == 'rationale_generation':
                    st.markdown("#### 4️⃣ Rationale Generation")
                    rationales = stage_info.get('ticker_rationales', {})
                    st.write(f"Generated 4-sentence rationales for {len(rationales)} tickers")
                
                elif stage == 'final_selection_rounds':
                    st.markdown("#### 5️⃣ Final Selection Rounds")
                    round_1 = stage_info.get('round_1', [])
                    round_2 = stage_info.get('round_2', [])
                    round_3 = stage_info.get('round_3', [])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Round 1:**")
                        st.write(", ".join(round_1))
                    with col2:
                        st.write("**Round 2:**")
                        st.write(", ".join(round_2))
                    with col3:
                        st.write("**Round 3:**")
                        st.write(", ".join(round_3))
                
                elif stage == 'final_consolidation':
                    st.markdown("#### 6️⃣ Final Consolidation")
                    unique = stage_info.get('unique_finalists', [])
                    final = stage_info.get('final_5', [])
                    st.write(f"Unique finalists: {len(unique)} → Final selection: **{len(final)}**")
                    st.success(f"✅ Final tickers: {', '.join(final)}")
            
            # Download log
            import json
            log_json = json.dumps(selection_log, indent=2)
            st.download_button(
                label="📥 Download Full Selection Log (JSON)",
                data=log_json,
                file_name=f"ai_selection_log_{result['analysis_date']}.json",
                mime="application/json"
            )
        
        # Download complete archives section
        st.markdown("---")
        st.subheader("📦 Complete Archives")
        st.write("Download all portfolio selection logs and archives from the system.")
        
        import os
        import zipfile
        from io import BytesIO
        
        # Check if portfolio_selection_logs directory exists
        logs_dir = "portfolio_selection_logs"
        if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
            
            if log_files:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info(f"📊 Found **{len(log_files)}** archived portfolio selection(s)")
                
                with col2:
                    # Create ZIP file in memory
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Add all JSON files
                        for log_file in log_files:
                            file_path = os.path.join(logs_dir, log_file)
                            with open(file_path, 'r') as f:
                                zip_file.writestr(log_file, f.read())
                        
                        # Add README if exists
                        readme_path = os.path.join(logs_dir, 'README.md')
                        if os.path.exists(readme_path):
                            with open(readme_path, 'r') as f:
                                zip_file.writestr('README.md', f.read())
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download All Archives (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"portfolio_archives_{result['analysis_date']}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        help="Download all portfolio selection logs as a ZIP file"
                    )
                
                # Show list of available archives
                with st.expander("📋 View Available Archives", expanded=False):
                    for log_file in sorted(log_files, reverse=True):
                        file_path = os.path.join(logs_dir, log_file)
                        file_size = os.path.getsize(file_path)
                        file_size_kb = file_size / 1024
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.text(f"📄 {log_file}")
                        with col2:
                            st.text(f"{file_size_kb:.1f} KB")
                        with col3:
                            # Individual download
                            with open(file_path, 'r') as f:
                                st.download_button(
                                    label="⬇️",
                                    data=f.read(),
                                    file_name=log_file,
                                    mime="application/json",
                                    key=f"download_{log_file}"
                                )
            else:
                st.info("📭 No archived selections found yet. Generate a portfolio to create archives.")
        else:
            st.warning("📂 Portfolio selection logs directory not found.")
    
    st.markdown("---")
    
    # Summary metrics
    st.subheader("📊 Portfolio Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Positions", summary['num_positions'])
    with col2:
        st.metric("Invested Capital", f"{summary['total_weight_pct']:.1f}%")
    with col3:
        st.metric("Average Score", f"{summary['avg_score']:.1f}")
    with col4:
        st.metric("Selection Method", summary.get('selection_method', 'N/A'))
    with col5:
        st.metric("Analyzed", f"{result.get('total_analyzed', 0)}")
    
    # Holdings table with AI rationales
    st.subheader("📈 Portfolio Holdings")
    
    for i, holding in enumerate(portfolio, 1):
        with st.expander(f"{i}. {holding['ticker']} - {holding['name']} ({holding['sector']})", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Final Score", f"{holding['final_score']:.1f}/100")
                st.metric("Weight", f"{holding['target_weight_pct']:.1f}%")
                st.metric("Recommendation", holding['recommendation'])
            
            with col2:
                st.markdown("**AI Rationale:**")
                st.write(holding['rationale'])
    
    # Detailed table
    st.subheader("📋 Holdings Table")
    df = pd.DataFrame(portfolio)
    df = df[['ticker', 'name', 'sector', 'final_score', 'target_weight_pct', 'eligible']]
    df.columns = ['Ticker', 'Name', 'Sector', 'Score', 'Weight %', 'Eligible']
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                help="Final composite score",
                min_value=0,
                max_value=100,
            ),
            "Eligible": st.column_config.CheckboxColumn(
                "Eligible",
                help="Meets IPS requirements"
            ),
        }
    )
    
    # Sector allocation
    st.subheader("🥧 Sector Allocation")
    
    sector_data = summary['sector_exposure']
    fig = go.Figure(data=[go.Pie(
        labels=list(sector_data.keys()),
        values=list(sector_data.values()),
        hole=.3,
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.subheader("📥 Export Portfolio")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export basic CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Portfolio CSV",
            data=csv,
            file_name=f"portfolio_recommendations_{result['analysis_date']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export full analysis JSON
        import json
        full_data = {
            'portfolio': portfolio,
            'summary': summary,
            'analysis_date': result['analysis_date'],
            'selection_log': selection_log
        }
        json_data = json.dumps(full_data, indent=2, default=str)
        st.download_button(
            label="Download Full Analysis (JSON)",
            data=json_data,
            file_name=f"portfolio_full_analysis_{result['analysis_date']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Manual Google Sheets export
        if st.session_state.sheets_integration.sheet:
            if st.button("📊 Push to Google Sheets", use_container_width=True):
                with st.spinner("Updating Google Sheets..."):
                    success = update_google_sheets_portfolio(result)
                    if success:
                        st.success("✅ Updated!")
                        sheet_url = st.session_state.sheets_integration.get_sheet_url()
                        if sheet_url:
                            st.markdown(f"[📄 Open Sheet]({sheet_url})")
                    else:
                        st.error("❌ Update failed")
        else:
            st.info("Connect to Google Sheet in sidebar")


# Backtesting functionality removed as requested


def parse_client_profile_with_ai(client_profile_text: str) -> dict | None:
    """Use OpenAI to parse client profile text into structured IPS.""" 
    import os
    
    # Check if OpenAI is available
    if OpenAI is None:
        st.error("OpenAI library not installed. Cannot parse client profile with AI.")
        return parse_client_profile_fallback(client_profile_text)
    
    # Check if OpenAI key is available
    if not os.getenv('OPENAI_API_KEY'):
        st.error("OpenAI API key not found. Cannot parse client profile with AI.")
        return parse_client_profile_fallback(client_profile_text)
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    system_prompt = """You are an expert investment advisor. Parse the client profile text and extract key information to create an Investment Policy Statement (IPS).

Return a JSON object with this exact structure:
{
  "client": {
    "name": "extracted or 'Client'",
    "risk_tolerance": "low, moderate, or high",
    "time_horizon_years": number (1-30),
    "cash_buffer_pct": number (3-20)
  },
  "position_limits": {
    "max_position_pct": number (3-15),
    "max_sector_pct": number (15-50), 
    "max_industry_pct": number (10-30)
  },
  "exclusions": {
    "sectors": ["list of excluded sectors"],
    "tickers": ["list of excluded tickers"],
    "esg_screens": ["list of ESG exclusions"]
  },
  "portfolio_constraints": {
    "beta_min": number (0.5-1.0),
    "beta_max": number (1.0-2.0)
  },
  "universe": {
    "min_price": number (1-10),
    "min_avg_daily_volume": number (500000-5000000)
  }
}

Use reasonable defaults if information is missing. Be conservative with risk settings unless explicitly stated otherwise."""

    user_prompt = f"Parse this client profile:\n\n{client_profile_text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        import json
        response_content = response.choices[0].message.content
        
        # Check if response is empty or None
        if not response_content:
            st.error("AI parsing error: Empty response from OpenAI")
            return None
        
        response_content = response_content.strip()
        
        # Try to extract JSON from response (in case there's extra text)
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            st.error(f"AI parsing error: No valid JSON found in response")
            st.code(f"Response received: {response_content[:200]}...")
            return None
        
        json_content = response_content[json_start:json_end]
        
        try:
            parsed_data = json.loads(json_content)
        except json.JSONDecodeError as json_error:
            st.error(f"AI parsing error: Invalid JSON format - {json_error}")
            st.code(f"JSON content: {json_content[:200]}...")
            return None
        
        # Validate that we got the expected structure
        if not isinstance(parsed_data, dict):
            st.error("AI parsing error: Response is not a JSON object")
            return None
        
        # Merge with default IPS structure
        default_ips = st.session_state.config_loader.load_ips()
        
        # Update with parsed values
        for section, values in parsed_data.items():
            if section in default_ips:
                if isinstance(values, dict):
                    default_ips[section].update(values)
                else:
                    default_ips[section] = values
            else:
                st.warning(f"Unknown section '{section}' in parsed data - skipping")
        
        st.success("✅ Client profile parsed successfully!")
        return default_ips
        
    except Exception as e:
        st.error(f"AI parsing error: {e}")
        st.warning("💡 Tip: Make sure your OpenAI API key is valid and you have sufficient credits")
        # Try fallback parsing
        st.info("🔄 Attempting fallback parsing method...")
        return parse_client_profile_fallback(client_profile_text)


def parse_client_profile_fallback(client_profile_text: str) -> dict:
    """Fallback method to parse client profile using keyword matching."""
    import re
    
    # Load default IPS
    default_ips = st.session_state.config_loader.load_ips()
    
    text_lower = client_profile_text.lower()
    
    # Parse risk tolerance
    if any(word in text_lower for word in ['conservative', 'low risk', 'safety', 'capital preservation']):
        default_ips['client']['risk_tolerance'] = 'low'
        default_ips['client']['cash_buffer_pct'] = 15
        default_ips['position_limits']['max_position_pct'] = 5
    elif any(word in text_lower for word in ['aggressive', 'high risk', 'growth', 'speculative']):
        default_ips['client']['risk_tolerance'] = 'high' 
        default_ips['client']['cash_buffer_pct'] = 5
        default_ips['position_limits']['max_position_pct'] = 10
    else:
        default_ips['client']['risk_tolerance'] = 'moderate'
        default_ips['client']['cash_buffer_pct'] = 10
        default_ips['position_limits']['max_position_pct'] = 7
    
    # Parse time horizon
    years_match = re.search(r'(\d+)\s*year', text_lower)
    if years_match:
        years = int(years_match.group(1))
        default_ips['client']['time_horizon_years'] = min(max(years, 1), 30)
    
    # Parse exclusions
    if 'esg' in text_lower or 'sustainable' in text_lower or 'ethical' in text_lower:
        default_ips['exclusions']['esg_screens'] = ['tobacco', 'weapons', 'fossil_fuels']
    
    if 'no tobacco' in text_lower or 'tobacco free' in text_lower:
        default_ips['exclusions']['sectors'].append('tobacco')
    
    if 'no crypto' in text_lower or 'cryptocurrency' in text_lower:
        default_ips['exclusions']['tickers'].extend(['COIN', 'MSTR', 'RIOT'])
    
    # Extract client name if mentioned
    name_match = re.search(r'(?:client|name|i am|my name is)\s+(?:is\s+)?([a-zA-Z\s]+)', text_lower)
    if name_match:
        name = name_match.group(1).strip().title()
        if len(name) < 50:  # Reasonable name length
            default_ips['client']['name'] = name
    
    st.info("✅ Used fallback parsing - review settings carefully")
    return default_ips


def display_backtest_results(results: dict):
    """Display backtest results."""
    
    metrics = results['metrics']
    benchmark = results['benchmark']
    
    st.success("✅ Backtest Complete!")
    st.write(results['performance_summary'])
    
    # Metrics comparison
    st.subheader("Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strategy**")
        st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        st.metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    
    with col2:
        st.write("**Benchmark (S&P 500)**")
        if benchmark:
            st.metric("Total Return", f"{benchmark['total_return_pct']:.2f}%")
            st.metric("Annualized Return", f"{benchmark['annualized_return_pct']:.2f}%")
            st.metric("Volatility", f"{benchmark['volatility_pct']:.2f}%")
            st.metric("Max Drawdown", f"{benchmark['max_drawdown_pct']:.2f}%")
            
            # Calculate alpha/IR
            if metrics and benchmark:
                alpha = metrics['annualized_return_pct'] - benchmark['annualized_return_pct']
                st.metric("Alpha vs Benchmark", f"{alpha:+.2f}%")
    
    # Equity curve
    st.subheader("💹 Equity Curve")
    
    portfolio_history = results['portfolio_history']
    dates = [p['date'] for p in portfolio_history]
    values = [p['total_value'] for p in portfolio_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.subheader("📥 Export Results")
    
    df = pd.DataFrame(portfolio_history)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="Download Backtest Results CSV",
        data=csv,
        file_name=f"backtest_{results['config']['start_date']}_to_{results['config']['end_date']}.csv",
        mime="text/csv"
    )


def portfolio_management_page():
    """Portfolio Management page - smart allocation analysis using existing QA archives."""
    st.header("📊 Portfolio Management")
    st.write("Analyze your current portfolio and get smart allocation recommendations using existing analysis archives.")
    st.markdown("---")
    
    # Portfolio Storage Functions
    def save_portfolio(portfolio_name, holdings):
        """Save portfolio to local storage."""
        portfolio_file = "data/saved_portfolios.json"
        os.makedirs("data", exist_ok=True)
        
        try:
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    saved_portfolios = json.load(f)
            else:
                saved_portfolios = {}
            
            saved_portfolios[portfolio_name] = {
                'holdings': holdings,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(portfolio_file, 'w') as f:
                json.dump(saved_portfolios, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Error saving portfolio: {e}")
            return False
    
    def load_saved_portfolios():
        """Load saved portfolios from local storage."""
        portfolio_file = "data/saved_portfolios.json"
        try:
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Error loading portfolios: {e}")
            return {}
    
    def map_to_broad_sector(sector_string):
        """Map specific sector/industry names to broad sector categories."""
        if not sector_string:
            return 'Unknown'
        
        sector_lower = sector_string.lower()
        
        # Technology & Software
        if any(term in sector_lower for term in [
            'technology', 'software', 'tech', 'semiconductor', 'chip', 'hardware',
            'electronic', 'connector', 'component', 'circuit', 'computer', 'internet', 
            'cloud', 'cyber', 'ai', 'artificial intelligence', 'data', 'it services', 
            'telecom equipment', 'networking', 'storage', 'processor', 'digital',
            'automation', 'robotics', 'iot', 'sensor', 'microchip', 'integrated circuit'
        ]):
            return 'Technology'
        
        # Healthcare & Pharma
        if any(term in sector_lower for term in [
            'healthcare', 'health', 'pharmaceutical', 'pharma', 'biotech', 'medical',
            'drug', 'hospital', 'insurance', 'managed care', 'life sciences', 'diagnostic'
        ]):
            return 'Healthcare'
        
        # Financial Services
        if any(term in sector_lower for term in [
            'financ', 'bank', 'insurance', 'investment', 'capital', 'asset management',
            'wealth', 'credit', 'lending', 'mortgage', 'securities', 'brokerage', 'fintech'
        ]):
            return 'Finance'
        
        # Consumer (Discretionary & Staples)
        if any(term in sector_lower for term in [
            'consumer', 'retail', 'restaurant', 'hotel', 'leisure', 'entertainment',
            'media', 'apparel', 'automotive', 'auto', 'food', 'beverage', 'household',
            'personal products', 'e-commerce', 'luxury', 'gaming'
        ]):
            return 'Consumer'
        
        # Industrials & Manufacturing
        if any(term in sector_lower for term in [
            'industrial', 'manufactur', 'aerospace', 'defense', 'construction',
            'machinery', 'equipment', 'transport', 'logistics', 'shipping', 'airlines',
            'rail', 'engineering', 'building'
        ]):
            return 'Industrials'
        
        # Energy & Utilities
        if any(term in sector_lower for term in [
            'energy', 'oil', 'gas', 'petroleum', 'renewable', 'solar', 'wind',
            'utility', 'utilities', 'electric', 'power', 'coal', 'nuclear'
        ]):
            return 'Energy'
        
        # Real Estate
        if any(term in sector_lower for term in [
            'real estate', 'reit', 'property', 'realty', 'commercial real estate',
            'residential', 'office', 'warehouse', 'data center'
        ]):
            return 'Real Estate'
        
        # Materials & Chemicals
        if any(term in sector_lower for term in [
            'materials', 'chemical', 'mining', 'metal', 'steel', 'aluminum',
            'copper', 'paper', 'packaging', 'commodity', 'resources'
        ]):
            return 'Materials'
        
        # Communication Services
        if any(term in sector_lower for term in [
            'communication', 'telecom', 'wireless', 'cable', 'satellite',
            'broadcasting', 'social media', 'advertising'
        ]):
            return 'Communication Services'
        
        # If no match, return original but capitalized
        return sector_string.title()
    
    def get_latest_analysis_for_ticker(qa_system, ticker):
        """Get the latest analysis for a specific ticker."""
        try:
            archive = qa_system.get_analysis_archive()
            if ticker in archive and archive[ticker]:
                latest = archive[ticker][0]  # First item is most recent
                
                # Map specific sector to broad category
                original_sector = latest.sector if latest.sector else 'Unknown'
                broad_sector = map_to_broad_sector(original_sector)
                
                return {
                    'timestamp': latest.timestamp.strftime('%Y-%m-%d %H:%M'),
                    'recommendation': latest.recommendation.value if hasattr(latest.recommendation, 'value') else str(latest.recommendation),
                    'price_target': latest.expected_target_price,
                    'confidence': latest.confidence_score,
                    'rationale': latest.final_rationale,
                    'key_points': latest.key_factors,
                    'sectors': [broad_sector],
                    'original_sector': original_sector  # Keep original for reference
                }
            return None
        except Exception as e:
            st.error(f"Error getting analysis for {ticker}: {e}")
            return None
    
    # Portfolio Input Section
    with st.container():
        st.subheader("💼 Current Portfolio")
        
        # Initialize portfolio in session state
        if 'portfolio_holdings' not in st.session_state:
            st.session_state.portfolio_holdings = {}
        
        # Portfolio Management Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Load saved portfolios
            saved_portfolios = load_saved_portfolios()
            if saved_portfolios:
                selected_portfolio = st.selectbox(
                    "Load Saved Portfolio:",
                    [""] + list(saved_portfolios.keys()),
                    help="Select a previously saved portfolio"
                )
                
                if selected_portfolio and st.button("📂 Load Portfolio"):
                    st.session_state.portfolio_holdings = saved_portfolios[selected_portfolio]['holdings']
                    st.success(f"Loaded portfolio: {selected_portfolio}")
                    st.rerun()
        
        with col2:
            # Save current portfolio
            if st.session_state.portfolio_holdings:
                save_name = st.text_input("Portfolio Name:", placeholder="My Portfolio")
                if st.button("💾 Save Portfolio") and save_name:
                    if save_portfolio(save_name, st.session_state.portfolio_holdings):
                        st.success(f"Saved: {save_name}")
        
        with col3:
            # Clear portfolio button
            if st.session_state.portfolio_holdings and st.button("🗑️ Clear Portfolio"):
                st.session_state.portfolio_holdings = {}
                st.rerun()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input method selection
            input_method = st.radio(
                "Portfolio Input Method:",
                ["Manual Entry", "Upload CSV"],
                horizontal=True
            )
            
            if input_method == "Manual Entry":
                with st.form("portfolio_entry"):
                    st.write("**Add Holdings:**")
                    ticker_input = st.text_input("Stock Ticker", placeholder="e.g., AAPL").upper()
                    allocation_input = st.number_input("Allocation %", min_value=0.0, max_value=100.0, step=0.1)
                    
                    if st.form_submit_button("Add to Portfolio"):
                        if ticker_input and allocation_input > 0:
                            st.session_state.portfolio_holdings[ticker_input] = allocation_input
                            st.success(f"Added {ticker_input}: {allocation_input}%")
                        else:
                            st.error("Please enter valid ticker and allocation")
            
            else:  # CSV Upload
                uploaded_file = st.file_uploader(
                    "Upload Portfolio CSV (Ticker, Allocation%)",
                    type=['csv'],
                    help="CSV should have columns: Ticker, Allocation"
                )
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'Ticker' in df.columns and 'Allocation' in df.columns:
                            for _, row in df.iterrows():
                                st.session_state.portfolio_holdings[row['Ticker'].upper()] = float(row['Allocation'])
                            st.success(f"Loaded {len(df)} holdings from CSV")
                        else:
                            st.error("CSV must have 'Ticker' and 'Allocation' columns")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
        
        with col2:
            st.write("**Analysis Mode:**")
            analysis_mode = st.selectbox(
                "Choose Analysis Mode:",
                ["Client Fit Analysis", "Custom Specifications"],
                help="Client Fit uses predetermined risk profiles, Custom allows detailed specifications"
            )
            
            # Initialize variables with defaults
            client_profile = None
            target_sectors = []
            risk_tolerance = 5
            growth_focus = 5
            
            if analysis_mode == "Client Fit Analysis":
                client_profile = st.selectbox(
                    "Client Risk Profile:",
                    ["Conservative", "Moderate", "Aggressive", "Growth-Focused"],
                    help="Predetermined allocation strategies based on risk tolerance"
                )
            else:
                st.write("**Custom Parameters:**")
                target_sectors = st.multiselect(
                    "Preferred Sectors:",
                    ["Technology", "Healthcare", "Finance", "Consumer", "Energy", "Real Estate", "Utilities"]
                )
                risk_tolerance = st.slider("Risk Tolerance", 1, 10, 5)
                growth_focus = st.slider("Growth vs Value", 1, 10, 5, help="1=Value focused, 10=Growth focused")
    
    # Display Current Portfolio
    if st.session_state.portfolio_holdings:
        st.subheader("📈 Current Holdings")
        
        portfolio_df = pd.DataFrame([
            {"Ticker": ticker, "Allocation %": allocation}
            for ticker, allocation in st.session_state.portfolio_holdings.items()
        ])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(portfolio_df, use_container_width=True)
        
        with col2:
            total_allocation = sum(st.session_state.portfolio_holdings.values())
            st.metric("Total Allocation", f"{total_allocation:.1f}%")
            
            if total_allocation != 100:
                st.warning(f"Portfolio not fully allocated ({100-total_allocation:.1f}% remaining)")
            else:
                st.success("Portfolio fully allocated ✓")
    
    # Analysis Section
    if st.session_state.portfolio_holdings and st.button("🔍 Analyze Portfolio", type="primary"):
        with st.spinner("Analyzing portfolio using existing archives..."):
            
            # Load QA system if not already loaded
            if not st.session_state.qa_system:
                st.session_state.qa_system = QASystem()
            
            qa_system = st.session_state.qa_system
            
            # Get archive data for current holdings
            portfolio_analysis = {}
            missing_analysis = []
            
            for ticker in st.session_state.portfolio_holdings.keys():
                archive_data = get_latest_analysis_for_ticker(qa_system, ticker)
                if archive_data:
                    portfolio_analysis[ticker] = archive_data
                else:
                    missing_analysis.append(ticker)
            
            # Perform deep portfolio analysis
            deep_analysis = perform_deep_portfolio_analysis(
                st.session_state.portfolio_holdings,
                portfolio_analysis,
                analysis_mode,
                client_profile if analysis_mode == "Client Fit Analysis" else None,
                target_sectors if analysis_mode == "Custom Specifications" else [],
                risk_tolerance,
                growth_focus
            )
            
            # Display Analysis Results
            st.subheader("📊 Comprehensive Portfolio Analysis")
            
            if portfolio_analysis:
                # Create tabs for different analysis views
                # Consolidated from 8 to 5 tabs for better UX and performance
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Holdings & Risk", "Recommendations", "News & Events", "Optimization"])
                
                with tab1:
                    st.write("### 📊 Portfolio Overview & Key Metrics")
                    
                    # Top-level metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Holdings", len(st.session_state.portfolio_holdings))
                    with col2:
                        avg_confidence = sum(a.get('confidence', 0) for a in portfolio_analysis.values()) / len(portfolio_analysis) if portfolio_analysis else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    with col3:
                        st.metric("Sectors Covered", len(deep_analysis['sectors']))
                    with col4:
                        st.metric("Risk Score", f"{deep_analysis['risk_score']:.1f}/10")
                    
                    st.markdown("---")
                    
                    # Portfolio composition
                    st.write("**Portfolio Composition:**")
                    composition_df = pd.DataFrame([
                        {
                            "Ticker": ticker,
                            "Allocation %": allocation,
                            "Recommendation": portfolio_analysis[ticker].get('recommendation', 'N/A'),
                            "Confidence": f"{portfolio_analysis[ticker].get('confidence', 0):.0f}%",
                            "Sector": ', '.join(portfolio_analysis[ticker].get('sectors', ['Unknown']))[:30]
                        }
                        for ticker, allocation in st.session_state.portfolio_holdings.items()
                        if ticker in portfolio_analysis
                    ])
                    st.dataframe(composition_df, use_container_width=True)
                    
                    # Allocation visualization
                    st.write("**Allocation Breakdown:**")
                    fig = px.pie(
                        values=list(st.session_state.portfolio_holdings.values()),
                        names=list(st.session_state.portfolio_holdings.keys()),
                        title="Portfolio Allocation by Stock"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sector allocation
                    if deep_analysis['sector_allocation']:
                        st.write("**Sector Allocation:**")
                        sector_fig = px.pie(
                            values=list(deep_analysis['sector_allocation'].values()),
                            names=list(deep_analysis['sector_allocation'].keys()),
                            title="Portfolio Allocation by Sector"
                        )
                        st.plotly_chart(sector_fig, use_container_width=True)
                    
                    # Key insights
                    st.write("**Key Portfolio Insights:**")
                    for insight in deep_analysis['key_insights']:
                        st.info(f"💡 {insight}")
                
                with tab2:
                    st.write("### 🔍 Deep Dive: Individual Holdings Analysis")
                    
                    for ticker, analysis in portfolio_analysis.items():
                        allocation = st.session_state.portfolio_holdings[ticker]
                        
                        with st.expander(f"**{ticker}** - {allocation}% allocation ({deep_analysis['holding_ratings'].get(ticker, 'Unknown')} positioning)", expanded=False):
                            # Header metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Recommendation", analysis.get('recommendation', 'N/A'))
                            with col2:
                                st.metric("Confidence", f"{analysis.get('confidence', 0):.0f}%")
                            with col3:
                                if 'price_target' in analysis and analysis['price_target']:
                                    st.metric("Price Target", f"${analysis['price_target']:.2f}")
                            with col4:
                                st.metric("Analysis Date", analysis.get('timestamp', 'Unknown')[:10])
                            
                            st.markdown("---")
                            
                            # Detailed analysis
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.write("**📋 Full Rationale:**")
                                if 'rationale' in analysis and analysis['rationale']:
                                    st.write(analysis['rationale'])
                                else:
                                    st.write("No detailed rationale available.")
                                
                                st.write("**🎯 Key Investment Points:**")
                                if 'key_points' in analysis and analysis['key_points']:
                                    for i, point in enumerate(analysis['key_points'], 1):
                                        st.write(f"{i}. {point}")
                                else:
                                    st.write("No key points available.")
                            
                            with col2:
                                st.write("**📊 Position Context:**")
                                st.write(f"• **Portfolio Weight:** {allocation}% ({deep_analysis['holding_ratings'].get(ticker, 'Unknown')} position)")
                                
                                # Show broad sector and original industry if different
                                broad_sector = ', '.join(analysis.get('sectors', ['Unknown']))
                                original_sector = analysis.get('original_sector', '')
                                if original_sector and original_sector != broad_sector and original_sector != 'Unknown':
                                    st.write(f"• **Sector:** {broad_sector} ({original_sector})")
                                else:
                                    st.write(f"• **Sector:** {broad_sector}")
                                
                                # Calculate potential impact
                                if 'price_target' in analysis and analysis['price_target']:
                                    upside = ((analysis['price_target'] / 100) - 1) * 100  # Simplified calculation
                                    portfolio_impact = (allocation / 100) * upside
                                    st.write(f"• **Potential Upside:** ~{upside:.1f}%")
                                    st.write(f"• **Portfolio Impact:** ~{portfolio_impact:.2f}%")
                                
                                st.write("**🔍 Role in Portfolio:**")
                                role = deep_analysis['holding_roles'].get(ticker, "Balanced position contributing to diversification")
                                st.write(role)
                            
                            st.markdown("---")
                            
                            # Risk assessment for this holding
                            st.write("**⚠️ Individual Risk Assessment:**")
                            risk_factors = deep_analysis['individual_risks'].get(ticker, ["Standard market risk applies"])
                            for factor in risk_factors:
                                st.write(f"• {factor}")
                
                with tab3:
                    st.write("### ⚖️ Portfolio Risk Analysis")
                    
                    # Overall risk metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Risk Score", f"{deep_analysis['risk_score']:.1f}/10", 
                                 help="Based on diversification, concentration, and volatility factors")
                    with col2:
                        st.metric("Diversification Score", f"{deep_analysis['diversification_score']:.1f}/10",
                                 help="Higher is better - measures spread across sectors and positions")
                    with col3:
                        st.metric("Concentration Risk", deep_analysis['concentration_level'],
                                 help="Measures if portfolio is too heavily weighted in few positions")
                    
                    st.markdown("---")
                    
                    # Risk breakdown
                    st.write("**📊 Risk Factor Breakdown:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Concentration Analysis:**")
                        for risk in deep_analysis['concentration_risks']:
                            st.warning(f"⚠️ {risk}")
                        
                        if not deep_analysis['concentration_risks']:
                            st.success("✅ No significant concentration risks detected")
                        
                        st.write("**Position Sizing:**")
                        for ticker, allocation in sorted(st.session_state.portfolio_holdings.items(), 
                                                        key=lambda x: x[1], reverse=True):
                            risk_level = "🔴 High" if allocation > 20 else "🟡 Medium" if allocation > 10 else "🟢 Low"
                            st.write(f"• {ticker}: {allocation}% - {risk_level}")
                    
                    with col2:
                        st.write("**Sector Risk Exposure:**")
                        for sector, alloc in sorted(deep_analysis['sector_allocation'].items(), 
                                                   key=lambda x: x[1], reverse=True):
                            risk_level = "🔴 High" if alloc > 40 else "🟡 Medium" if alloc > 25 else "🟢 Low"
                            st.write(f"• {sector}: {alloc:.1f}% - {risk_level}")
                        
                        st.write("**Portfolio-Level Risks:**")
                        for risk in deep_analysis['portfolio_risks']:
                            st.warning(f"⚠️ {risk}")
                        
                        if not deep_analysis['portfolio_risks']:
                            st.success("✅ Portfolio structure appears sound")
                    
                    st.markdown("---")
                    
                    # Risk mitigation suggestions
                    st.write("**🛡️ Risk Mitigation Strategies:**")
                    for strategy in deep_analysis['risk_mitigation']:
                        st.info(f"💡 {strategy}")
                
                with tab4:
                    st.write("### 🎯 Sector Analysis & Diversification")
                    
                    # Sector breakdown
                    st.write("**Sector Diversification Analysis:**")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Current Sector Allocation:**")
                        sector_df = pd.DataFrame([
                            {"Sector": sector, "Allocation %": f"{alloc:.1f}%", "Holdings": count}
                            for sector, (alloc, count) in deep_analysis['sector_details'].items()
                        ]).sort_values("Allocation %", ascending=False)
                        st.dataframe(sector_df, use_container_width=True)
                        
                        st.write("**Diversification Quality:**")
                        st.metric("Sector Diversity Score", f"{deep_analysis['sector_diversity_score']:.1f}/10")
                        
                        if deep_analysis['diversification_score'] >= 7:
                            st.success("✅ Well-diversified across sectors")
                        elif deep_analysis['diversification_score'] >= 5:
                            st.warning("⚠️ Moderate diversification - consider broadening")
                        else:
                            st.error("🔴 Limited diversification - high sector concentration risk")
                    
                    with col2:
                        st.write("**Missing/Underrepresented Sectors:**")
                        for sector in deep_analysis['missing_sectors']:
                            st.write(f"• **{sector}** - Not represented in portfolio")
                        
                        for sector, reason in deep_analysis['underweight_sectors'].items():
                            st.write(f"• **{sector}** - Underweight: {reason}")
                        
                        if not deep_analysis['missing_sectors'] and not deep_analysis['underweight_sectors']:
                            st.success("✅ Good sector coverage")
                    
                    st.markdown("---")
                    
                    # Correlation analysis
                    st.write("**📊 Correlation & Overlap Analysis:**")
                    st.write(deep_analysis['correlation_analysis'])
                
                with tab5:
                    st.write("### 💡 Smart Allocation Recommendations")
                    
                    # Overall assessment
                    if deep_analysis['overall_quality'] == 'Excellent':
                        st.success(f"✅ **Portfolio Quality: {deep_analysis['overall_quality']}**")
                    elif deep_analysis['overall_quality'] == 'Good':
                        st.info(f"👍 **Portfolio Quality: {deep_analysis['overall_quality']}**")
                    else:
                        st.warning(f"⚠️ **Portfolio Quality: {deep_analysis['overall_quality']}**")
                    
                    st.write(deep_analysis['quality_explanation'])
                    
                    st.markdown("---")
                    
                    # Detailed recommendations
                    st.write("**🎯 Specific Action Items:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Immediate Adjustments:**")
                        if deep_analysis['immediate_actions']:
                            for i, action in enumerate(deep_analysis['immediate_actions'], 1):
                                st.error(f"{i}. {action}")
                        else:
                            st.success("✅ No immediate adjustments needed")
                        
                        st.write("**Short-term Optimizations (1-3 months):**")
                        for i, action in enumerate(deep_analysis['short_term_actions'], 1):
                            st.warning(f"{i}. {action}")
                    
                    with col2:
                        st.write("**Long-term Strategy (3-12 months):**")
                        for i, action in enumerate(deep_analysis['long_term_actions'], 1):
                            st.info(f"{i}. {action}")
                        
                        st.write("**Monitoring Priorities:**")
                        for i, priority in enumerate(deep_analysis['monitoring_priorities'], 1):
                            st.write(f"{i}. {priority}")
                    
                    st.markdown("---")
                    
                    # Rebalancing suggestions
                    if deep_analysis['rebalancing_suggestions']:
                        st.write("**⚖️ Suggested Rebalancing:**")
                        rebal_df = pd.DataFrame(deep_analysis['rebalancing_suggestions'])
                        st.dataframe(rebal_df, use_container_width=True)
                
                with tab2:  # Growth & Value merged into Holdings & Risk
                    st.write("### 📈 Growth & Value Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Growth vs Value Composition:**")
                        st.metric("Growth Allocation", f"{deep_analysis['growth_allocation']:.1f}%")
                        st.metric("Value Allocation", f"{deep_analysis['value_allocation']:.1f}%")
                        st.metric("Blend Allocation", f"{deep_analysis['blend_allocation']:.1f}%")
                        
                        st.write("**Style Analysis:**")
                        st.write(deep_analysis['style_analysis'])
                    
                    with col2:
                        st.write("**Expected Return Profile:**")
                        st.write(deep_analysis['return_profile'])
                        
                        st.write("**Investment Horizon Fit:**")
                        st.write(deep_analysis['time_horizon_fit'])
                        
                        if analysis_mode == "Client Fit Analysis" and client_profile:
                            st.write(f"**Profile Alignment ({client_profile}):**")
                            st.write(deep_analysis['profile_alignment'])
                    
                    st.markdown("---")
                    
                    st.write("**📊 Holdings by Investment Style:**")
                    for style, holdings in deep_analysis['holdings_by_style'].items():
                        with st.expander(f"{style} Holdings ({len(holdings)} stocks)"):
                            for holding in holdings:
                                st.write(f"• {holding['ticker']}: {holding['allocation']}% - {holding['reason']}")
                
                with tab4:  # News & Market Context
                    st.write("### 📰 News & Market Context Analysis")
                    st.info("💡 Real-time analysis powered by Perplexity AI")
                    
                    # Analysis type selector
                    analysis_type = st.radio(
                        "Select Analysis Type:",
                        ["Portfolio News", "Macro Overview", "Individual Stock"],
                        horizontal=True
                    )
                    
                    # Portfolio News Analysis (default)
                    if analysis_type == "Portfolio News":
                        with st.spinner("🔍 Analyzing recent news and market developments..."):
                            if 'portfolio_news_cache' not in st.session_state:
                                news_analysis = get_portfolio_news_analysis(
                                    list(st.session_state.portfolio_holdings.keys()),
                                    deep_analysis
                                )
                                st.session_state.portfolio_news_cache = news_analysis
                            else:
                                news_analysis = st.session_state.portfolio_news_cache
                        
                        st.markdown(news_analysis)
                        if st.button("🔄 Refresh News", key="refresh_news"):
                            with st.spinner("Refreshing news analysis..."):
                                news_analysis = get_portfolio_news_analysis(
                                    list(st.session_state.portfolio_holdings.keys()),
                                    deep_analysis
                                )
                                st.session_state.portfolio_news_cache = news_analysis
                                st.rerun()
                    
                    # Macro Market Overview
                    elif analysis_type == "Macro Overview":
                        if st.button("🌍 Analyze Macro Environment", type="primary"):
                            with st.spinner("Analyzing global macro environment..."):
                                macro_overview = get_macro_market_overview(deep_analysis)
                                st.session_state.macro_overview_cache = macro_overview
                        
                        if 'macro_overview_cache' in st.session_state:
                            st.markdown("#### 🌍 Macro Market Environment")
                            st.markdown(st.session_state.macro_overview_cache)
                    
                    # Individual Stock Analysis
                    elif analysis_type == "Individual Stock":
                        selected_ticker = st.selectbox(
                            "Select ticker for detailed analysis:",
                            list(st.session_state.portfolio_holdings.keys()),
                            key="news_ticker_select"
                        )
                        
                        if st.button(f"� Analyze {selected_ticker}", type="primary"):
                            with st.spinner(f"Analyzing {selected_ticker} in detail..."):
                                ticker_news = get_individual_ticker_news_analysis(
                                    selected_ticker,
                                    st.session_state.portfolio_holdings[selected_ticker],
                                    portfolio_analysis.get(selected_ticker, {})
                                )
                                st.session_state[f'ticker_news_{selected_ticker}'] = ticker_news
                        
                        # Display cached analysis
                        if f'ticker_news_{selected_ticker}' in st.session_state:
                            st.markdown(st.session_state[f'ticker_news_{selected_ticker}'])
                    
                    st.markdown("---")
                    
                    st.write("### � Sector-Level Market Trends")
                    # Group holdings by sector for detailed analysis
                    holdings_by_sector = {}
                    for ticker, data in portfolio_analysis.items():
                        for sector in data.get('sectors', ['Unknown']):
                            if sector not in holdings_by_sector:
                                holdings_by_sector[sector] = []
                            holdings_by_sector[sector].append(ticker)
                    
                    for sector in deep_analysis['sectors']:
                        sector_tickers = holdings_by_sector.get(sector, [])
                        sector_allocation = deep_analysis['sector_allocation'].get(sector, 0)
                        
                        with st.expander(f"📊 {sector} - {sector_allocation:.1f}% allocation ({len(sector_tickers)} holdings)"):
                            st.write(f"**Holdings:** {', '.join(sector_tickers)}")
                            
                            if st.button(f"Get Real-Time {sector} Analysis", key=f"sector_{sector}"):
                                with st.spinner(f"Analyzing {sector} sector with Perplexity AI..."):
                                    sector_analysis = get_sector_specific_analysis(
                                        sector,
                                        sector_tickers,
                                        deep_analysis
                                    )
                                    st.markdown(sector_analysis)
                            else:
                                # Show basic trend
                                st.write(deep_analysis['sector_trends'].get(sector, "Click button above for detailed analysis"))
                    
                    st.markdown("---")
                    
                    # Upcoming Events Calendar
                    st.write("### 📅 Upcoming Events & Catalysts")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        event_timeframe = st.selectbox("Timeframe:", ["Next 7 Days", "Next 14 Days", "Next 30 Days"], key="event_timeframe")
                    with col2:
                        event_format = st.selectbox("Format:", ["Detailed", "Summary Table"], key="event_format")
                    
                    # Initialize or update event cache in session state
                    cache_key = f"events_{event_timeframe}_{event_format}"
                    
                    # Button to fetch/refresh events
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        fetch_events = st.button("🔍 Load Events", type="primary", key="fetch_events_btn")
                    with col2:
                        if cache_key in st.session_state:
                            st.caption(f"✓ Loaded for {event_timeframe} ({event_format})")
                        else:
                            st.caption("👆 Click to load events")
                    
                    # Fetch events only when button clicked
                    if fetch_events:
                        try:
                            with st.spinner(f"Scanning {event_timeframe.lower()} for portfolio events..."):
                                upcoming_events = get_portfolio_upcoming_events(
                                    list(st.session_state.portfolio_holdings.keys()),
                                    event_timeframe,
                                    deep_analysis,
                                    event_format
                                )
                                st.session_state[cache_key] = upcoming_events
                                st.success(f"✅ Events loaded for {event_timeframe}!")
                        except Exception as e:
                            st.error(f"Error loading events: {e}")
                            logger.error(f"Event calendar error: {e}")
                    
                    # Display events if they exist in cache
                    if cache_key in st.session_state:
                        st.markdown(st.session_state[cache_key])
                    else:
                        st.info(f"� Click 'Load Events' to fetch events for {event_timeframe} ({event_format})")
                    
                    st.markdown("---")
                
                with tab5:  # Optimization
                    st.write("### 🚀 Portfolio Optimization Suggestions")
                    
                    st.write("**New Position Recommendations:**")
                    
                    # Generate detailed suggestions
                    suggestions = generate_detailed_portfolio_suggestions(
                        st.session_state.portfolio_holdings,
                        portfolio_analysis,
                        deep_analysis,
                        analysis_mode,
                        client_profile if analysis_mode == "Client Fit Analysis" else None
                    )
                    
                    if suggestions:
                        for i, suggestion in enumerate(suggestions, 1):
                            with st.expander(f"**{i}. {suggestion['ticker']}** - {suggestion['company']} ({suggestion['sector']})", expanded=i<=2):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write("**Investment Thesis:**")
                                    st.write(suggestion['detailed_rationale'])
                                    
                                    st.write("**Why This Stock for Your Portfolio:**")
                                    st.write(suggestion['portfolio_fit'])
                                    
                                    st.write("**Key Strengths:**")
                                    for strength in suggestion['strengths']:
                                        st.write(f"• {strength}")
                                    
                                    st.write("**Considerations:**")
                                    for consideration in suggestion['considerations']:
                                        st.write(f"• {consideration}")
                                
                                with col2:
                                    st.metric("Suggested Allocation", f"{suggestion['suggested_allocation']}%")
                                    st.metric("Priority Level", suggestion['priority'])
                                    st.metric("Risk Level", suggestion['risk_level'])
                                    st.metric("Expected Timeline", suggestion['timeline'])
                                    
                                    st.write("**Fills Gap:**")
                                    st.write(suggestion['fills_gap'])
                    
                    st.markdown("---")
                    
                    st.write("**Exit/Trim Considerations:**")
                    if deep_analysis['trim_candidates']:
                        for candidate in deep_analysis['trim_candidates']:
                            st.warning(f"⚠️ **{candidate['ticker']}**: {candidate['reason']}")
                    else:
                        st.success("✅ No immediate trim candidates identified")
                    
                    st.markdown("---")
                    
                    st.write("**Alternative Scenarios:**")
                    for scenario in deep_analysis['alternative_scenarios']:
                        with st.expander(f"Scenario: {scenario['name']}"):
                            st.write(f"**Objective:** {scenario['objective']}")
                            st.write(f"**Approach:** {scenario['approach']}")
                            st.write("**Suggested Changes:**")
                            for change in scenario['changes']:
                                st.write(f"• {change}")
            
            # Handle missing analysis
            if missing_analysis:
                st.warning(f"⚠️ Missing analysis for: {', '.join(missing_analysis)}")
                st.write("These stocks haven't been analyzed yet. Run individual analysis first to get complete portfolio insights.")


def perform_deep_portfolio_analysis(holdings, analysis_data, mode, profile, target_sectors, risk_tolerance, growth_focus):
    """Perform comprehensive deep-dive portfolio analysis."""
    
    total_allocation = sum(holdings.values())
    num_holdings = len(holdings)
    
    # Initialize analysis dictionary
    analysis = {
        'sectors': set(),
        'sector_allocation': {},
        'sector_details': {},
        'risk_score': 0,
        'diversification_score': 0,
        'concentration_level': 'Low',
        'concentration_risks': [],
        'portfolio_risks': [],
        'risk_mitigation': [],
        'holding_ratings': {},
        'holding_roles': {},
        'individual_risks': {},
        'key_insights': [],
        'missing_sectors': [],
        'underweight_sectors': {},
        'sector_diversity_score': 0,
        'correlation_analysis': '',
        'overall_quality': 'Good',
        'quality_explanation': '',
        'immediate_actions': [],
        'short_term_actions': [],
        'long_term_actions': [],
        'monitoring_priorities': [],
        'rebalancing_suggestions': [],
        'growth_allocation': 0,
        'value_allocation': 0,
        'blend_allocation': 0,
        'style_analysis': '',
        'return_profile': '',
        'time_horizon_fit': '',
        'profile_alignment': '',
        'holdings_by_style': {},
        'market_context': '',
        'sector_trends': {},
        'potential_catalysts': [],
        'external_risks': [],
        'trim_candidates': [],
        'alternative_scenarios': []
    }
    
    # Analyze sector allocation
    sector_counts = {}
    for ticker, data in analysis_data.items():
        ticker_sectors = data.get('sectors', ['Unknown'])
        allocation = holdings.get(ticker, 0)
        
        for sector in ticker_sectors:
            analysis['sectors'].add(sector)
            analysis['sector_allocation'][sector] = analysis['sector_allocation'].get(sector, 0) + allocation
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    # Calculate sector details
    for sector in analysis['sectors']:
        analysis['sector_details'][sector] = (
            analysis['sector_allocation'].get(sector, 0),
            sector_counts.get(sector, 0)
        )
    
    # Position sizing analysis
    max_position = max(holdings.values()) if holdings else 0
    top_3_allocation = sum(sorted(holdings.values(), reverse=True)[:3]) if len(holdings) >= 3 else total_allocation
    
    # Calculate risk score (0-10, higher = riskier)
    risk_factors = []
    
    if max_position > 25:
        risk_factors.append(3.0)
        analysis['concentration_risks'].append(f"Single position exceeds 25% ({max_position:.1f}%)")
    elif max_position > 20:
        risk_factors.append(2.0)
    elif max_position > 15:
        risk_factors.append(1.0)
    
    if top_3_allocation > 60:
        risk_factors.append(2.5)
        analysis['concentration_risks'].append(f"Top 3 positions represent {top_3_allocation:.1f}% of portfolio")
    elif top_3_allocation > 50:
        risk_factors.append(1.5)
    
    if num_holdings < 5:
        risk_factors.append(2.0)
        analysis['portfolio_risks'].append(f"Limited diversification with only {num_holdings} holdings")
    elif num_holdings < 8:
        risk_factors.append(1.0)
    
    if len(analysis['sectors']) < 3:
        risk_factors.append(2.5)
        analysis['portfolio_risks'].append(f"Concentrated in {len(analysis['sectors'])} sector(s)")
    elif len(analysis['sectors']) < 5:
        risk_factors.append(1.0)
    
    analysis['risk_score'] = min(10, sum(risk_factors))
    
    # Diversification score (0-10, higher = better)
    div_score = 0
    if num_holdings >= 10:
        div_score += 3
    elif num_holdings >= 7:
        div_score += 2
    elif num_holdings >= 5:
        div_score += 1
    
    if len(analysis['sectors']) >= 6:
        div_score += 3
    elif len(analysis['sectors']) >= 4:
        div_score += 2
    elif len(analysis['sectors']) >= 3:
        div_score += 1
    
    if max_position <= 15:
        div_score += 2
    elif max_position <= 20:
        div_score += 1
    
    if top_3_allocation <= 40:
        div_score += 2
    elif top_3_allocation <= 50:
        div_score += 1
    
    analysis['diversification_score'] = div_score
    analysis['sector_diversity_score'] = min(10, (len(analysis['sectors']) / 8.0) * 10)
    
    # Concentration level
    if max_position > 25 or top_3_allocation > 60:
        analysis['concentration_level'] = 'High'
    elif max_position > 15 or top_3_allocation > 45:
        analysis['concentration_level'] = 'Medium'
    else:
        analysis['concentration_level'] = 'Low'
    
    # Rate each holding
    for ticker, allocation in holdings.items():
        if allocation > 20:
            analysis['holding_ratings'][ticker] = 'Overweight/Core'
            analysis['holding_roles'][ticker] = f"Core position driving {allocation:.1f}% of portfolio returns. High impact on overall performance."
        elif allocation > 10:
            analysis['holding_ratings'][ticker] = 'Core'
            analysis['holding_roles'][ticker] = f"Significant core holding contributing {allocation:.1f}% to portfolio diversification and returns."
        elif allocation > 5:
            analysis['holding_ratings'][ticker] = 'Balanced'
            analysis['holding_roles'][ticker] = f"Balanced position at {allocation:.1f}% providing sector exposure and diversification."
        else:
            analysis['holding_ratings'][ticker] = 'Satellite'
            analysis['holding_roles'][ticker] = f"Satellite position at {allocation:.1f}% for tactical exposure or emerging opportunities."
        
        # Individual risks
        risks = []
        if allocation > 20:
            risks.append(f"High concentration risk - represents {allocation:.1f}% of portfolio")
        
        ticker_data = analysis_data.get(ticker, {})
        confidence = ticker_data.get('confidence', 0)
        if confidence < 60:
            risks.append(f"Lower confidence analysis ({confidence:.0f}%) - may warrant additional due diligence")
        
        recommendation = ticker_data.get('recommendation', '')
        if 'sell' in recommendation.lower():
            risks.append("Current recommendation is SELL - consider reviewing position")
        
        analysis['individual_risks'][ticker] = risks if risks else ["Standard market risk applies"]
    
    # Risk mitigation strategies
    if analysis['risk_score'] > 6:
        analysis['risk_mitigation'].append("Consider reducing largest positions to under 15% each")
        analysis['risk_mitigation'].append("Add holdings in underrepresented sectors to improve diversification")
    if num_holdings < 8:
        analysis['risk_mitigation'].append(f"Increase number of holdings from {num_holdings} to at least 8-10 for better risk distribution")
    if len(analysis['sectors']) < 5:
        analysis['risk_mitigation'].append("Expand sector coverage to include defensive sectors like Healthcare or Consumer Staples")
    if not analysis['risk_mitigation']:
        analysis['risk_mitigation'].append("Portfolio structure is sound - maintain current diversification levels")
    
    # Key insights
    analysis['key_insights'].append(f"Portfolio contains {num_holdings} holdings across {len(analysis['sectors'])} sector(s)")
    analysis['key_insights'].append(f"Largest position is {max_position:.1f}% - " + 
                                   ("within optimal range" if max_position <= 20 else "consider trimming for risk management"))
    
    avg_confidence = sum(data.get('confidence', 0) for data in analysis_data.values()) / len(analysis_data) if analysis_data else 0
    analysis['key_insights'].append(f"Average analysis confidence: {avg_confidence:.0f}% - " + 
                                   ("high conviction" if avg_confidence > 70 else "moderate conviction" if avg_confidence > 60 else "lower conviction"))
    
    # Missing sectors analysis
    all_sectors = {'Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy', 'Real Estate', 'Utilities', 'Industrials', 'Materials'}
    analysis['missing_sectors'] = list(all_sectors - analysis['sectors'])
    
    # Underweight sectors
    for sector, (alloc, count) in analysis['sector_details'].items():
        if alloc < 10 and count == 1:
            analysis['underweight_sectors'][sector] = f"Only {alloc:.1f}% from single holding"
    
    # Correlation analysis
    if len(analysis['sectors']) >= 5:
        analysis['correlation_analysis'] = f"Portfolio shows good sector diversification across {len(analysis['sectors'])} sectors, suggesting lower correlation risk. This should help reduce volatility during market corrections."
    else:
        analysis['correlation_analysis'] = f"Limited to {len(analysis['sectors'])} sectors - holdings may move together during market volatility. Consider adding uncorrelated sectors."
    
    # Overall quality assessment
    if analysis['diversification_score'] >= 8 and analysis['risk_score'] <= 4:
        analysis['overall_quality'] = 'Excellent'
        analysis['quality_explanation'] = "Portfolio demonstrates excellent diversification with well-balanced positions and appropriate risk management."
    elif analysis['diversification_score'] >= 6 and analysis['risk_score'] <= 6:
        analysis['overall_quality'] = 'Good'
        analysis['quality_explanation'] = "Portfolio is well-structured with good diversification, though some optimization opportunities exist."
    else:
        analysis['overall_quality'] = 'Needs Improvement'
        analysis['quality_explanation'] = "Portfolio would benefit from improved diversification and risk management adjustments."
    
    # Action items
    if max_position > 25:
        analysis['immediate_actions'].append(f"Trim largest position ({max_position:.1f}%) to under 20% to reduce concentration risk")
    
    if num_holdings < 5:
        analysis['immediate_actions'].append("Add at least 2-3 more positions to improve diversification")
    
    if len(analysis['sectors']) < 3:
        analysis['short_term_actions'].append("Expand into at least 2 additional sectors")
    
    if 'Healthcare' not in analysis['sectors']:
        analysis['short_term_actions'].append("Consider adding Healthcare exposure for defensive diversification")
    
    if total_allocation < 95:
        analysis['short_term_actions'].append(f"Deploy remaining {100-total_allocation:.1f}% capital according to recommendations")
    
    analysis['long_term_actions'].append("Regularly rebalance to maintain target allocations")
    analysis['long_term_actions'].append("Monitor earnings reports and update analysis quarterly")
    analysis['long_term_actions'].append("Review and adjust sector exposure based on economic cycle")
    
    # Monitoring priorities
    for ticker, allocation in sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:3]:
        analysis['monitoring_priorities'].append(f"{ticker} ({allocation:.1f}%) - Top position requiring regular monitoring")
    
    # Rebalancing suggestions
    if max_position > 20:
        target = 15
        excess = max_position - target
        top_ticker = max(holdings.items(), key=lambda x: x[1])[0]
        analysis['rebalancing_suggestions'].append({
            'Action': 'Trim',
            'Ticker': top_ticker,
            'Current %': f"{max_position:.1f}%",
            'Target %': f"{target:.1f}%",
            'Change': f"-{excess:.1f}%"
        })
    
    # Style analysis
    growth_count = 0
    value_count = 0
    for ticker, data in analysis_data.items():
        rec = data.get('recommendation', '').lower()
        if 'growth' in rec or data.get('confidence', 0) > 75:
            growth_count += holdings.get(ticker, 0)
        elif 'value' in rec:
            value_count += holdings.get(ticker, 0)
    
    analysis['growth_allocation'] = growth_count
    analysis['value_allocation'] = value_count
    analysis['blend_allocation'] = total_allocation - growth_count - value_count
    
    if growth_count > 60:
        analysis['style_analysis'] = "Growth-oriented portfolio with high allocation to appreciation-focused stocks. Suitable for longer time horizons and higher risk tolerance."
    elif value_count > 60:
        analysis['style_analysis'] = "Value-oriented portfolio emphasizing fundamentals and downside protection. More conservative approach."
    else:
        analysis['style_analysis'] = "Balanced blend of growth and value, providing exposure to both appreciation and stability."
    
    # Return profile
    avg_confidence = sum(data.get('confidence', 0) for data in analysis_data.values()) / len(analysis_data) if analysis_data else 0
    if avg_confidence > 70:
        analysis['return_profile'] = "High conviction portfolio with strong return potential based on analysis confidence levels. Expect above-market returns if thesis plays out."
    else:
        analysis['return_profile'] = "Moderate conviction portfolio with market-level return expectations. Focus on consistency and risk management."
    
    # Time horizon
    if growth_count > 50:
        analysis['time_horizon_fit'] = "Best suited for 3-5+ year investment horizon given growth orientation"
    else:
        analysis['time_horizon_fit'] = "Suitable for 1-3 year horizon with quarterly rebalancing"
    
    # Profile alignment
    if mode == "Client Fit Analysis" and profile:
        if profile == "Conservative" and analysis['risk_score'] <= 4:
            analysis['profile_alignment'] = "Well-aligned with conservative profile - appropriate risk levels and diversification"
        elif profile == "Aggressive" and analysis['risk_score'] >= 6:
            analysis['profile_alignment'] = "Matches aggressive profile with concentrated positions and growth focus"
        else:
            analysis['profile_alignment'] = f"Moderate alignment with {profile} profile - some adjustments recommended"
    
    # Holdings by style
    analysis['holdings_by_style'] = {
        'Growth': [],
        'Value': [],
        'Blend': []
    }
    
    for ticker, allocation in holdings.items():
        ticker_data = analysis_data.get(ticker, {})
        confidence = ticker_data.get('confidence', 0)
        
        if confidence > 75:
            analysis['holdings_by_style']['Growth'].append({
                'ticker': ticker,
                'allocation': allocation,
                'reason': f"High confidence ({confidence:.0f}%) growth candidate"
            })
        elif confidence > 60:
            analysis['holdings_by_style']['Blend'].append({
                'ticker': ticker,
                'allocation': allocation,
                'reason': f"Moderate confidence ({confidence:.0f}%) balanced position"
            })
        else:
            analysis['holdings_by_style']['Value'].append({
                'ticker': ticker,
                'allocation': allocation,
                'reason': f"Value/defensive play with standard risk profile"
            })
    
    # Market context
    analysis['market_context'] = "Current market environment requires balanced approach with attention to sector rotation and risk management. Maintain diversification across defensive and growth sectors."
    
    # Sector trends
    for sector in analysis['sectors']:
        analysis['sector_trends'][sector] = f"Monitor {sector} sector for earnings trends, regulatory changes, and competitive dynamics affecting portfolio holdings."
    
    # Potential catalysts
    analysis['potential_catalysts'].append("Earnings reports from major holdings")
    analysis['potential_catalysts'].append("Sector rotation opportunities based on economic data")
    analysis['potential_catalysts'].append("Market volatility creating entry points for underweight sectors")
    
    # External risks
    analysis['external_risks'].append("Market-wide correction affecting all holdings")
    analysis['external_risks'].append("Sector-specific headwinds in concentrated areas")
    analysis['external_risks'].append("Macroeconomic changes (rates, inflation) impacting valuations")
    
    # Trim candidates
    for ticker, allocation in holdings.items():
        ticker_data = analysis_data.get(ticker, {})
        if allocation > 20:
            analysis['trim_candidates'].append({
                'ticker': ticker,
                'reason': f"Overweight at {allocation:.1f}% - consider trimming to 15-18% for risk management"
            })
        
        rec = ticker_data.get('recommendation', '').lower()
        if 'sell' in rec:
            analysis['trim_candidates'].append({
                'ticker': ticker,
                'reason': f"Current analysis recommends SELL - evaluate exit strategy"
            })
    
    # Alternative scenarios
    analysis['alternative_scenarios'] = [
        {
            'name': 'Conservative Shift',
            'objective': 'Reduce portfolio risk and volatility',
            'approach': 'Increase defensive sectors (Healthcare, Utilities) to 30%, trim growth positions',
            'changes': [
                'Add 2-3 defensive healthcare names',
                'Trim top 2 growth positions by 3-5% each',
                'Target max position size of 12%'
            ]
        },
        {
            'name': 'Growth Acceleration',
            'objective': 'Maximize upside potential for longer-term gains',
            'approach': 'Concentrate in high-conviction growth ideas',
            'changes': [
                'Increase top 3 positions to 20-25% each',
                'Exit lower conviction positions',
                'Focus on Technology and innovation themes'
            ]
        },
        {
            'name': 'Income Focus',
            'objective': 'Generate steady dividend income',
            'approach': 'Rotate into dividend aristocrats and stable payers',
            'changes': [
                'Add 3-4 dividend-focused positions',
                'Target 3-4% portfolio yield',
                'Balance growth and income (60/40 split)'
            ]
        }
    ]
    
    return analysis


def generate_client_fit_recommendations(holdings, analysis_data, profile):
    """Generate recommendations based on client risk profile."""
    total_allocation = sum(holdings.values())
    
    # Define profile-based allocation strategies
    profile_strategies = {
        "Conservative": {"tech_max": 25, "growth_focus": 3, "risk_tolerance": 3},
        "Moderate": {"tech_max": 40, "growth_focus": 5, "risk_tolerance": 5},
        "Aggressive": {"tech_max": 60, "growth_focus": 8, "risk_tolerance": 8},
        "Growth-Focused": {"tech_max": 70, "growth_focus": 9, "risk_tolerance": 7}
    }
    
    strategy = profile_strategies.get(profile, profile_strategies["Moderate"])
    
    # Analyze current allocation vs profile
    tech_allocation = 0
    for ticker in holdings.keys():
        ticker_sectors = analysis_data.get(ticker, {}).get('sectors', [])
        if ticker_sectors and any(sector in ticker_sectors for sector in ['Technology', 'Software']):
            tech_allocation += holdings[ticker]
    
    recommendations = {
        'status': 'good',
        'message': f"Your portfolio aligns well with a {profile} investment profile.",
        'adjustments': []
    }
    
    if tech_allocation > strategy['tech_max']:
        recommendations['status'] = 'adjust'
        recommendations['adjustments'].append(
            f"Consider reducing technology allocation from {tech_allocation:.1f}% to under {strategy['tech_max']}%"
        )
    
    if total_allocation < 95:
        recommendations['adjustments'].append(
            f"Consider allocating the remaining {100-total_allocation:.1f}% to complete your portfolio"
        )
    
    return recommendations


def generate_custom_recommendations(holdings, analysis_data, target_sectors, risk_tolerance, growth_focus):
    """Generate recommendations based on custom specifications."""
    recommendations = {
        'status': 'good',
        'message': "Your portfolio aligns with your custom specifications.",
        'adjustments': []
    }
    
    # Add custom logic based on parameters
    if risk_tolerance < 5 and len(holdings) < 5:
        recommendations['adjustments'].append(
            "For conservative risk tolerance, consider adding more diversification with 5+ holdings"
        )
    
    if growth_focus > 7:
        growth_heavy = 0
        for ticker in holdings.keys():
            ticker_tags = analysis_data.get(ticker, {}).get('tags', [])
            if ticker_tags and 'Growth' in ticker_tags:
                growth_heavy += holdings[ticker]
        if growth_heavy < 60:
            recommendations['adjustments'].append(
                "For high growth focus, consider increasing allocation to growth stocks"
            )
    
    return recommendations


def get_portfolio_news_analysis(tickers, deep_analysis=None):
    """Get comprehensive real-time news analysis using Perplexity AI."""
    
    # Check if Perplexity is available
    if not hasattr(st.session_state, 'perplexity_client') or st.session_state.perplexity_client is None:
        return """
**Market Environment Assessment:**

⚠️ Real-time news analysis requires Perplexity API access. Using general market context instead.

Your portfolio of {0} holdings shows diversification across multiple sectors. Monitor earnings reports, 
regulatory changes, and competitive dynamics in your key sectors for optimal risk management.

**Recommendation:** Enable Perplexity API for comprehensive real-time market intelligence and news analysis.
""".format(len(tickers))
    
    try:
        # Build comprehensive prompt for Perplexity
        ticker_list = ', '.join(tickers)
        sector_info = ""
        
        if deep_analysis and deep_analysis['sector_allocation']:
            top_sectors = sorted(deep_analysis['sector_allocation'].items(), key=lambda x: x[1], reverse=True)[:3]
            sector_info = f"\nTop sector exposures: {', '.join([f'{s[0]} ({s[1]:.1f}%)' for s in top_sectors])}"
        
        prompt = f"""Analyze the current market environment and recent news (past 7 days) for this investment portfolio:

Holdings: {ticker_list}
Total positions: {len(tickers)}{sector_info}

Provide a comprehensive analysis covering:

1. **Overall Market Context**: Current market conditions, trends, and sentiment affecting these holdings
2. **Company-Specific News**: Recent earnings, announcements, or significant developments for each ticker
3. **Sector Trends**: Industry dynamics and rotation patterns affecting the portfolio sectors
4. **Risk Factors**: Immediate concerns, regulatory issues, or competitive threats
5. **Opportunities**: Positive catalysts, favorable trends, or underappreciated developments
6. **Portfolio Impact Assessment**: How recent news affects the overall portfolio positioning

Focus on actionable insights and be specific about which holdings are impacted by each development. 
Rate the overall news sentiment as Positive, Neutral, or Negative with brief justification.

Format the response in clear sections with bullet points for easy reading."""

        # Call Perplexity API
        response = st.session_state.perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert financial analyst with access to real-time market data and news. Provide comprehensive, actionable market analysis for investment portfolios."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more factual analysis
            max_tokens=2000   # Allow for comprehensive response
        )
        
        analysis_text = response.choices[0].message.content.strip()
        
        # Add metadata
        analysis_header = f"""## 📰 Real-Time Portfolio News & Market Analysis
*Analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} using Perplexity AI with real-time data*

---

"""
        
        return analysis_header + analysis_text
        
    except Exception as e:
        st.error(f"Error fetching real-time news analysis: {e}")
        # Fallback to basic analysis
        return f"""
**Market Environment Assessment:**

⚠️ Unable to fetch real-time analysis. Error: {str(e)}

Your portfolio of {len(tickers)} holdings includes: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}

**General Recommendations:**
• Monitor earnings reports from your holdings
• Stay alert to sector-specific regulatory changes
• Review position sizes during high volatility
• Consider setting price alerts for largest positions

**Next Steps:** Check API configuration or try again later for comprehensive real-time analysis.
"""


def get_portfolio_upcoming_events(tickers, timeframe, deep_analysis, format_type="Detailed"):
    """Get comprehensive upcoming events calendar for portfolio holdings using Perplexity."""
    
    if not hasattr(st.session_state, 'perplexity_client') or st.session_state.perplexity_client is None:
        return """
## 📅 Upcoming Events Calendar

⚠️ Real-time event tracking requires Perplexity API access.

**Manual Tracking Recommended:**
- Check each company's investor relations page for earnings dates
- Monitor SEC filings for material events
- Set up Google Alerts for company-specific news

Enable Perplexity API for automated event tracking and analysis.
"""
    
    try:
        ticker_list = ', '.join(tickers)
        days_map = {
            "Next 7 Days": 7,
            "Next 14 Days": 14,
            "Next 30 Days": 30
        }
        days = days_map.get(timeframe, 14)
        
        # Build portfolio context
        sector_info = ""
        if deep_analysis and deep_analysis['sector_allocation']:
            top_sectors = sorted(deep_analysis['sector_allocation'].items(), key=lambda x: x[1], reverse=True)[:3]
            sector_info = f"\nKey sectors: {', '.join([f'{s[0]}' for s in top_sectors])}"
        
        # Adjust prompt based on format
        if format_type == "Summary Table":
            format_instructions = """
Format as a clean table:

| Date | Ticker | Event Type | Description | Impact |
|------|--------|------------|-------------|--------|
| MM/DD | XXX | Earnings | Q4 earnings report | High |

Sort chronologically. Use impact levels: 🔴 High, 🟡 Medium, 🟢 Low.
Include only confirmed events with specific dates. If date is TBD, note in description."""
        else:
            format_instructions = """
Format the response as:

## 📅 Upcoming Events Calendar ({0}-Day Outlook)

### [TICKER] - Company Name
**[Date]** - Event Type
- Description and details
- Expected impact: High/Medium/Low
- What to watch for

Group events chronologically and highlight HIGH IMPACT events. Include specific dates and times when available. 
If no significant events are scheduled for a ticker, note that it's in a "quiet period."

Be specific with dates and details. Flag any events where timing is uncertain.""".format(days)
        
        prompt = f"""Create a comprehensive upcoming events calendar for these portfolio holdings: {ticker_list}

Timeframe: Next {days} days from today ({datetime.now().strftime('%Y-%m-%d')}){sector_info}

For EACH ticker, identify and list:

1. **Earnings Reports**:
   - Exact date and time (if announced)
   - Expected EPS vs consensus
   - Key metrics to watch
   - Historical earnings reaction patterns

2. **Product Launches & Announcements**:
   - New product releases or major updates
   - Strategic announcements scheduled
   - Conference presentations or keynotes

3. **Regulatory & Legal Events**:
   - FDA decisions or regulatory approvals
   - Court dates or legal proceedings
   - Compliance deadlines

4. **Corporate Actions**:
   - Dividend ex-dates and payment dates
   - Stock splits or special dividends
   - Shareholder meetings
   - Executive presentations or conferences

5. **Industry Events**:
   - Major industry conferences where company will present
   - Competitor events that could impact holdings
   - Sector-wide regulatory decisions

6. **Economic Indicators**:
   - Macro events that could significantly impact the portfolio sectors
   - Fed meetings, jobs reports, inflation data

{format_instructions}"""

        response = st.session_state.perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert portfolio manager and event tracker. Provide accurate, date-specific information about upcoming corporate and market events. Use real-time data to identify exact dates and details."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Very low for factual accuracy on dates
            max_tokens=3000   # Extended for comprehensive calendar
        )
        
        events_text = response.choices[0].message.content.strip()
        
        # Add header with metadata
        header = f"""*Event calendar generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} via Perplexity AI with real-time data*

**Portfolio Overview:** {len(tickers)} holdings across {len(deep_analysis['sectors'])} sectors

---

"""
        
        # Add summary section
        footer = """

---

### 📊 Event Impact Summary

**Key Recommendations:**
- Set calendar reminders for all earnings dates
- Review positions before high-impact events
- Consider hedging strategies for concentrated positions ahead of binary events
- Monitor pre-earnings option activity for sentiment signals
- Have a plan for both upside and downside scenarios

**Risk Management:**
- Avoid adding to positions immediately before earnings
- Consider reducing position size ahead of uncertain regulatory decisions
- Stay liquid to take advantage of post-event volatility
- Track analyst estimate revisions leading up to events

💡 **Pro Tip:** The most important events are often the ones you don't know about. This calendar helps you stay ahead of portfolio-moving catalysts.
"""
        
        return header + events_text + footer
        
    except Exception as e:
        st.error(f"Error fetching upcoming events: {e}")
        return f"""
## 📅 Upcoming Events Calendar

⚠️ Unable to fetch real-time event calendar. Error: {str(e)}

**Manual Tracking Steps:**
1. Visit each company's investor relations website
2. Check earnings calendar sites (Yahoo Finance, Seeking Alpha)
3. Set up news alerts for your holdings
4. Monitor SEC Form 8-K filings for material events

Please check API configuration or try again later.
"""


def get_macro_market_overview(deep_analysis):
    """Get comprehensive macro market overview using Perplexity."""
    
    if not hasattr(st.session_state, 'perplexity_client') or st.session_state.perplexity_client is None:
        return """
**Macro Market Overview**

⚠️ Real-time macro analysis requires Perplexity API access.

Monitor key indicators: Fed policy, inflation data, GDP growth, and sector rotation trends for optimal portfolio positioning.
"""
    
    try:
        # Build portfolio context
        sector_breakdown = ", ".join([f"{s}: {a:.1f}%" for s, a in sorted(deep_analysis['sector_allocation'].items(), key=lambda x: x[1], reverse=True)[:4]])
        
        prompt = f"""Provide a comprehensive macro market overview and its implications for an investment portfolio:

Portfolio Context:
- Portfolio composition: {sector_breakdown}
- Risk score: {deep_analysis['risk_score']:.1f}/10
- Diversification across {len(deep_analysis['sectors'])} sectors

Analyze:

1. **Current Macro Environment**:
   - Federal Reserve policy stance and interest rate trajectory
   - Inflation trends and their market impact
   - Economic growth indicators (GDP, employment, consumer spending)
   - Global economic considerations (China, Europe, emerging markets)

2. **Market Regime Analysis**:
   - Current market regime (Bull/Bear/Sideways) and volatility levels
   - Risk-on vs risk-off sentiment
   - Sector rotation patterns observed recently
   - Equity market valuations (P/E ratios, earnings trends)

3. **Asset Class Dynamics**:
   - Equities vs bonds vs commodities performance
   - Dollar strength and currency impacts
   - Credit spreads and financial conditions
   - Alternative assets (real estate, crypto) if relevant

4. **Key Risks & Opportunities**:
   - Top macro risks to monitor (recession, inflation, geopolitical)
   - Potential positive catalysts ahead
   - Which market segments appear attractive/unattractive

5. **Portfolio Positioning Implications**:
   - How should portfolios be positioned given current macro backdrop?
   - Defensive vs offensive positioning recommendations
   - Specific sector/style tilts that make sense now
   - Risk management considerations

Be data-driven and specific. Include recent economic data points and market moves."""

        response = st.session_state.perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a macro strategist and economist with expertise in Federal Reserve policy, economic indicators, and portfolio positioning across market cycles."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        analysis = response.choices[0].message.content.strip()
        
        header = f"""*Real-time macro analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} via Perplexity AI*

---

"""
        return header + analysis
        
    except Exception as e:
        return f"Unable to fetch macro analysis: {str(e)}"


def get_sector_specific_analysis(sector, holdings_in_sector, deep_analysis):
    """Get detailed sector-specific analysis using Perplexity."""
    
    if not hasattr(st.session_state, 'perplexity_client') or st.session_state.perplexity_client is None:
        return f"""
### {sector} Sector Analysis

Your portfolio has {len(holdings_in_sector)} holding(s) in this sector representing {deep_analysis['sector_allocation'].get(sector, 0):.1f}% allocation.

⚠️ Real-time sector analysis requires Perplexity API access.
"""
    
    try:
        ticker_list = ', '.join(holdings_in_sector)
        allocation = deep_analysis['sector_allocation'].get(sector, 0)
        
        prompt = f"""Provide comprehensive {sector} sector analysis relevant to these holdings: {ticker_list}

Portfolio Context:
- Sector allocation: {allocation:.1f}%
- Number of holdings: {len(holdings_in_sector)}

Analyze:

1. **Current Sector Trends** (Past Month):
   - Industry momentum and performance vs broader market
   - Key sector-wide developments affecting all players
   - Regulatory or policy changes impacting the sector

2. **Competitive Dynamics**:
   - Market leaders vs laggards
   - Emerging competitive threats or consolidation trends
   - Technology disruption or innovation affecting the sector

3. **Economic & Market Factors**:
   - Macroeconomic drivers (rates, GDP, consumer spending, etc.)
   - Seasonal or cyclical considerations
   - Supply chain or commodity price impacts

4. **Forward Outlook**:
   - Consensus sector outlook for next 3-6 months
   - Key events or catalysts on the horizon
   - Potential headwinds or tailwinds

5. **Portfolio Implications**:
   - Is {allocation:.1f}% allocation appropriate? (Overweight/Underweight/Neutral)
   - Which holdings in this sector appear strongest/weakest?
   - Recommended adjustments or additions in this sector

Be specific with data points and actionable insights."""

        response = st.session_state.perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sector specialist analyst with expertise in industry trends, competitive dynamics, and macroeconomic impacts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Unable to fetch sector analysis: {str(e)}"


def get_individual_ticker_news_analysis(ticker, allocation, analysis_data):
    """Get detailed news analysis for a specific ticker using Perplexity."""
    
    if not hasattr(st.session_state, 'perplexity_client') or st.session_state.perplexity_client is None:
        return f"""
### {ticker} - Deep Dive News Analysis

⚠️ Real-time news analysis requires Perplexity API access.

**Position Details:**
- Portfolio Allocation: {allocation}%
- Recommendation: {analysis_data.get('recommendation', 'N/A')}
- Latest Analysis: {analysis_data.get('timestamp', 'Unknown')}

Enable Perplexity API for comprehensive real-time news and market intelligence.
"""
    
    try:
        # Build detailed prompt for individual ticker
        prompt = f"""Provide a comprehensive news and market analysis for {ticker}:

Position Context:
- Portfolio weight: {allocation}%
- Current recommendation: {analysis_data.get('recommendation', 'N/A')}
- Analysis date: {analysis_data.get('timestamp', 'Unknown')}

Provide detailed analysis covering:

1. **Breaking News & Recent Developments** (past 7 days):
   - Earnings reports, guidance, or analyst updates
   - Product launches, partnerships, or acquisitions
   - Management changes or strategic announcements
   - Any regulatory or legal developments

2. **Stock Performance & Market Reaction**:
   - Recent price action and volume trends
   - Analyst rating changes and price target adjustments
   - Institutional buying/selling activity if notable
   - Technical levels and momentum indicators

3. **Competitive Landscape**:
   - Industry trends affecting the company
   - Competitor moves or market share dynamics
   - Disruptive threats or emerging opportunities

4. **Forward Outlook**:
   - Upcoming catalysts (earnings date, product releases, etc.)
   - Consensus expectations and potential surprises
   - Risk factors to monitor in coming weeks/months

5. **Portfolio Recommendation**:
   - Given the {allocation}% allocation, should this position be: Maintained, Increased, Decreased, or Exited?
   - Specific price levels or events to watch
   - Suggested action items for the investor

Be specific with dates, numbers, and actionable insights. Rate overall sentiment as Bullish, Neutral, or Bearish."""

        # Call Perplexity API
        response = st.session_state.perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert equity analyst with access to real-time market data, news, and filings. Provide detailed, actionable stock analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,  # Very low for factual accuracy
            max_tokens=2500   # Extended for comprehensive analysis
        )
        
        analysis_text = response.choices[0].message.content.strip()
        
        # Add header with metadata
        header = f"""## 🔍 {ticker} - Comprehensive News Deep Dive

**Position Details:**
- Portfolio Allocation: {allocation}%
- Recommendation: {analysis_data.get('recommendation', 'N/A')}
- Confidence: {analysis_data.get('confidence', 0):.0f}%
- Analysis Date: {analysis_data.get('timestamp', 'Unknown')}

*Real-time analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} via Perplexity AI*

---

"""
        
        return header + analysis_text
        
    except Exception as e:
        st.error(f"Error fetching ticker news analysis: {e}")
        return f"""
### {ticker} - News Analysis

⚠️ Unable to fetch real-time analysis. Error: {str(e)}

**Position Details:**
- Portfolio Allocation: {allocation}%
- Recommendation: {analysis_data.get('recommendation', 'N/A')}

Please check API configuration or try again later.
"""


def generate_portfolio_suggestions(holdings, analysis_data, mode, profile=None):
    """Generate basic suggestions for new portfolio additions."""
    suggestions = []
    
    # Get current sectors
    current_sectors = set()
    for ticker, data in analysis_data.items():
        ticker_sectors = data.get('sectors', [])
        if ticker_sectors:
            current_sectors.update(ticker_sectors)
    
    # Sample suggestions based on what's missing
    if 'Healthcare' not in current_sectors:
        suggestions.append({
            'ticker': 'JNJ',
            'company': 'Johnson & Johnson',
            'rationale': 'Adds healthcare diversification with defensive characteristics and dividend income.',
            'suggested_allocation': 8,
            'sector': 'Healthcare'
        })
    
    if 'Finance' not in current_sectors:
        suggestions.append({
            'ticker': 'JPM',
            'company': 'JPMorgan Chase',
            'rationale': 'Strong financial sector exposure with solid fundamentals and rising rate environment benefits.',
            'suggested_allocation': 6,
            'sector': 'Finance'
        })
    
    return suggestions[:2]  # Return top 2 suggestions


def generate_detailed_portfolio_suggestions(holdings, analysis_data, deep_analysis, mode, profile=None):
    """Generate detailed, in-depth suggestions for portfolio additions."""
    suggestions = []
    
    current_sectors = deep_analysis['sectors']
    missing_sectors = deep_analysis['missing_sectors']
    
    # Suggestion templates with detailed rationales
    detailed_suggestions = {
        'Healthcare': {
            'ticker': 'UNH',
            'company': 'UnitedHealth Group',
            'sector': 'Healthcare',
            'suggested_allocation': 8,
            'priority': 'High',
            'risk_level': 'Medium',
            'timeline': '1-3 years',
            'fills_gap': 'Healthcare sector exposure',
            'detailed_rationale': """UnitedHealth Group is the largest healthcare company by revenue, offering diversified 
exposure through both insurance (UnitedHealthcare) and healthcare services (Optum). The company demonstrates consistent 
growth with strong fundamentals, making it an ideal defensive addition to growth-oriented portfolios.""",
            'portfolio_fit': """Adding UNH would provide defensive healthcare exposure that performs well during market 
volatility. Its ~2% dividend yield adds income generation while maintaining growth potential. The stock typically has 
low correlation with technology and financial sectors, improving your portfolio's risk-adjusted returns.""",
            'strengths': [
                'Market leader with 15%+ revenue growth',
                'Defensive characteristics reduce portfolio volatility',
                'Strong cash flow generation supports dividend growth',
                'Aging demographics provide long-term tailwind',
                'Diversified business model reduces single-point risk'
            ],
            'considerations': [
                'Regulatory risk from potential healthcare reform',
                'Lower beta means less upside in bull markets',
                'Large cap limits explosive growth potential'
            ]
        },
        'Finance': {
            'ticker': 'JPM',
            'company': 'JPMorgan Chase',
            'sector': 'Finance',
            'suggested_allocation': 7,
            'priority': 'High',
            'risk_level': 'Medium',
            'timeline': '2-4 years',
            'fills_gap': 'Financial sector exposure',
            'detailed_rationale': """JPMorgan Chase represents best-in-class financial services exposure with dominant 
market positions across investment banking, consumer banking, and asset management. The company's fortress balance 
sheet and diversified revenue streams provide stability while benefiting from rising interest rate environments.""",
            'portfolio_fit': """JPM adds cyclical exposure that benefits from economic growth while maintaining quality 
fundamentals. Its 2.5%+ dividend yield enhances portfolio income. Financial sector exposure provides diversification 
from technology while capturing economic expansion themes.""",
            'strengths': [
                'Industry-leading ROE consistently above 15%',
                'Diversified revenue across multiple business lines',
                'Benefits from rising interest rate environment',
                'Strong capital position exceeds regulatory requirements',
                'Track record of growing dividends and buybacks'
            ],
            'considerations': [
                'Cyclical exposure increases during recessions',
                'Regulatory requirements limit aggressive growth',
                'Interest rate sensitivity cuts both ways'
            ]
        },
        'Consumer': {
            'ticker': 'COST',
            'company': 'Costco Wholesale',
            'sector': 'Consumer',
            'suggested_allocation': 6,
            'priority': 'Medium',
            'risk_level': 'Low',
            'timeline': '3-5 years',
            'fills_gap': 'Consumer defensive exposure',
            'detailed_rationale': """Costco's membership-based warehouse model creates recurring revenue and customer 
loyalty that transcends economic cycles. The company's pricing power and operational efficiency drive consistent 
same-store sales growth while maintaining industry-leading margins.""",
            'portfolio_fit': """COST provides defensive consumer exposure with growth characteristics. The stock performs 
well across market cycles due to its value proposition. Adds stability without sacrificing growth potential, making 
it ideal for balanced portfolio construction.""",
            'strengths': [
                'Membership model creates predictable recurring revenue',
                'Defensive characteristics with growth potential',
                'Strong balance sheet with minimal debt',
                'Consistent market share gains',
                'Inflation-resistant business model'
            ],
            'considerations': [
                'Premium valuation limits margin of safety',
                'Lower dividend yield than traditional consumer staples',
                'International expansion carries execution risk'
            ]
        },
        'Energy': {
            'ticker': 'XOM',
            'company': 'Exxon Mobil',
            'sector': 'Energy',
            'suggested_allocation': 5,
            'priority': 'Medium',
            'risk_level': 'Medium-High',
            'timeline': '1-2 years',
            'fills_gap': 'Energy and inflation hedge',
            'detailed_rationale': """ExxonMobil provides integrated energy exposure with upstream and downstream 
operations creating natural hedges. As energy transitions, XOM's massive cash flows fund both traditional operations 
and clean energy investments, positioning it for long-term relevance.""",
            'portfolio_fit': """Energy exposure serves as inflation hedge and portfolio diversifier with low correlation 
to technology. XOM's 3.5%+ dividend yield enhances portfolio income while commodity exposure provides protection 
during inflationary periods.""",
            'strengths': [
                'Integrated model provides operational flexibility',
                'Strong dividend history (40+ years of growth)',
                'Benefits from energy transition investments',
                'Inflation hedge through commodity exposure',
                'Massive scale provides competitive advantages'
            ],
            'considerations': [
                'Energy transition risks to long-term business model',
                'Commodity price volatility affects earnings',
                'ESG concerns may limit investor base',
                'Cyclical nature increases portfolio volatility'
            ]
        },
        'Real Estate': {
            'ticker': 'PLD',
            'company': 'Prologis',
            'sector': 'Real Estate',
            'suggested_allocation': 5,
            'priority': 'Low',
            'risk_level': 'Medium',
            'timeline': '2-5 years',
            'fills_gap': 'Real estate and logistics exposure',
            'detailed_rationale': """Prologis is the global leader in logistics real estate, owning warehouses essential 
to e-commerce fulfillment. The secular trend toward online retail creates sustained demand for modern distribution 
facilities, while the REIT structure provides tax-efficient income.""",
            'portfolio_fit': """PLD adds real estate diversification with growth characteristics tied to e-commerce 
expansion. The 2.5-3% dividend yield from REIT structure enhances portfolio income while providing inflation protection 
through real asset exposure.""",
            'strengths': [
                'Market leader in high-demand logistics real estate',
                'Secular e-commerce trends drive sustained demand',
                'REIT structure provides tax-efficient income',
                'Modern portfolio commands premium rents',
                'Global diversification reduces geographic risk'
            ],
            'considerations': [
                'Interest rate sensitivity affects REIT valuations',
                'High valuation limits near-term upside',
                'Economic slowdown impacts leasing activity',
                'Competition from new supply in hot markets'
            ]
        },
        'Technology': {
            'ticker': 'MSFT',
            'company': 'Microsoft',
            'sector': 'Technology',
            'suggested_allocation': 10,
            'priority': 'High',
            'risk_level': 'Medium',
            'timeline': '3-5 years',
            'fills_gap': 'Quality technology exposure',
            'detailed_rationale': """Microsoft combines growth and quality with dominant positions in cloud computing 
(Azure), productivity software (Office 365), and emerging AI applications. The company's subscription-based model 
generates predictable recurring revenue while maintaining strong margins.""",
            'portfolio_fit': """MSFT provides quality technology exposure with lower volatility than pure-play tech names. 
Strong fundamentals and consistent execution make it suitable as a core holding. Cloud and AI exposure position the 
portfolio for multi-year secular trends.""",
            'strengths': [
                'Leader in cloud computing with Azure growth',
                'Subscription model creates recurring revenue',
                'Strong competitive moats across products',
                'AI integration across product suite',
                'Consistent capital returns through dividends and buybacks'
            ],
            'considerations': [
                'Large size limits growth rate potential',
                'Valuation can contract during bear markets',
                'Cloud competition from AWS and Google',
                'Regulatory scrutiny on big tech'
            ]
        }
    }
    
    # Select suggestions based on missing sectors and portfolio needs
    suggestion_priority = []
    
    # High priority: Missing defensive sectors
    if 'Healthcare' in missing_sectors:
        suggestion_priority.append(detailed_suggestions['Healthcare'])
    
    # High priority: Missing financial exposure
    if 'Finance' in missing_sectors or deep_analysis['sector_allocation'].get('Finance', 0) < 10:
        suggestion_priority.append(detailed_suggestions['Finance'])
    
    # Medium priority: Consumer defensive
    if 'Consumer' in missing_sectors or deep_analysis['sector_allocation'].get('Consumer', 0) < 5:
        suggestion_priority.append(detailed_suggestions['Consumer'])
    
    # Add technology if underweight
    if deep_analysis['sector_allocation'].get('Technology', 0) < 20 and len(holdings) > 3:
        suggestion_priority.append(detailed_suggestions['Technology'])
    
    # Inflation hedges
    if 'Energy' in missing_sectors:
        suggestion_priority.append(detailed_suggestions['Energy'])
    
    # Diversification
    if 'Real Estate' in missing_sectors and len(holdings) >= 7:
        suggestion_priority.append(detailed_suggestions['Real Estate'])
    
    # Return top 3-4 suggestions
    return suggestion_priority[:4] if len(suggestion_priority) > 3 else suggestion_priority


def qa_learning_center_page():
    """QA & Learning Center page - tracks recommendation performance and enables model improvement."""
    st.header("🎯 QA & Learning Center")
    st.write("Track recommendation performance, conduct weekly reviews, and improve analysis through learning.")
    st.markdown("---")
    
    if not st.session_state.qa_system:
        st.error("QA System not initialized. Please restart the application.")
        return
    
    qa_system = st.session_state.qa_system
    
    # Add refresh and export buttons
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([2, 2, 2, 2])
    
    with col_btn1:
        if st.button("🔄 Refresh QA Data", help="Reload data from storage", use_container_width=True):
            qa_system.recommendations = qa_system._load_recommendations()
            qa_system.all_analyses = qa_system._load_all_analyses()
            qa_system.reviews = qa_system._load_reviews()
            st.success("QA data refreshed!")
            st.rerun()
    
    # 🆕 IMPROVEMENT #9: Batch Export All Tracked Stocks
    with col_btn2:
        tracked_tickers = qa_system.get_tracked_tickers()
        if tracked_tickers and len(tracked_tickers) > 0:
            if st.button(f"📥 Export All ({len(tracked_tickers)} stocks)", use_container_width=True):
                st.session_state.show_batch_export = True
        else:
            st.button("📥 Export All (No data)", disabled=True, use_container_width=True)
    
    # Google Sheets Export with Price Fetching
    with col_btn3:
        analysis_archive = qa_system.get_analysis_archive()
        sheets_enabled = st.session_state.get('sheets_enabled', False)
        if analysis_archive and sheets_enabled:
            if st.button("📊 Sync to Sheets", help="Export to Google Sheets with price options", use_container_width=True):
                st.session_state.show_sheets_export = True
                st.rerun()
        elif not sheets_enabled:
            st.button("📊 Sync to Sheets (Connect first)", disabled=True, use_container_width=True)
        else:
            st.button("📊 Sync to Sheets (No data)", disabled=True, use_container_width=True)
    
    with col_btn4:
        analysis_archive = qa_system.get_analysis_archive()
        if analysis_archive and len(analysis_archive) > 0:
            # Generate comprehensive export
            import pandas as pd
            from datetime import datetime
            export_rows = []
            for ticker, analyses in analysis_archive.items():
                for analysis in analyses:
                    row = {
                        'Ticker': ticker,
                        'Date': analysis.timestamp.strftime('%Y-%m-%d %H:%M'),
                        'Recommendation': analysis.recommendation.value,
                        'Score': analysis.confidence_score,
                        'Price': analysis.price_at_analysis,
                        'Sector': analysis.sector if hasattr(analysis, 'sector') else 'N/A',
                        'Market Cap': analysis.market_cap if hasattr(analysis, 'market_cap') else 'N/A'
                    }
                    
                    # Add agent scores if available
                    if hasattr(analysis, 'agent_scores') and analysis.agent_scores:
                        for agent, score in analysis.agent_scores.items():
                            row[f"{agent.replace('_', ' ').title()} Score"] = score
                    
                    export_rows.append(row)
            
            df_export = pd.DataFrame(export_rows)
            csv_export = df_export.to_csv(index=False)
            
            st.download_button(
                label=f"📊 Download All Data ({len(export_rows)} analyses)",
                data=csv_export,
                file_name=f"all_analyses_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Show batch export interface if requested
    if st.session_state.get('show_batch_export', False):
        with st.expander("📦 Batch Export Options", expanded=True):
            st.write("**Select what to include in the export:**")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                include_rationales = st.checkbox("Include Agent Rationales", value=False)
                include_fundamentals = st.checkbox("Include Fundamental Data", value=True)
                include_agent_scores = st.checkbox("Include Agent Scores", value=True)
            
            with col_exp2:
                export_format = st.radio("Export Format", ["CSV", "JSON", "Markdown Report"])
                date_filter = st.selectbox("Date Range", ["All Time", "Last 30 Days", "Last 90 Days", "Last Year"])
            
            if st.button("🎯 Generate Batch Export", type="primary"):
                from datetime import datetime, timedelta
                
                # Apply date filter
                cutoff_date = None
                if date_filter == "Last 30 Days":
                    cutoff_date = datetime.now() - timedelta(days=30)
                elif date_filter == "Last 90 Days":
                    cutoff_date = datetime.now() - timedelta(days=90)
                elif date_filter == "Last Year":
                    cutoff_date = datetime.now() - timedelta(days=365)
                
                # Generate export data
                export_data = []
                for ticker, analyses in analysis_archive.items():
                    for analysis in analyses:
                        if cutoff_date and analysis.timestamp < cutoff_date:
                            continue
                        
                        row = {
                            'Ticker': ticker,
                            'Analysis Date': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'Recommendation': analysis.recommendation.value.upper(),
                            'Confidence Score': analysis.confidence_score,
                            'Price at Analysis': analysis.price_at_analysis
                        }
                        
                        if include_fundamentals:
                            row['Sector'] = analysis.sector if hasattr(analysis, 'sector') else 'N/A'
                            row['Market Cap'] = analysis.market_cap if hasattr(analysis, 'market_cap') else 'N/A'
                        
                        if include_agent_scores and hasattr(analysis, 'agent_scores') and analysis.agent_scores:
                            for agent, score in analysis.agent_scores.items():
                                row[f"{agent.replace('_', ' ').title()} Score"] = score
                        
                        if include_rationales and hasattr(analysis, 'agent_rationales') and analysis.agent_rationales:
                            for agent, rationale in analysis.agent_rationales.items():
                                if rationale:
                                    row[f"{agent.replace('_', ' ').title()} Rationale"] = rationale[:1000]  # Allow full rationale
                        
                        export_data.append(row)
                
                # Generate file based on format
                if export_format == "CSV":
                    import pandas as pd
                    df = pd.DataFrame(export_data)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download CSV ({len(export_data)} analyses)",
                        data=csv_data,
                        file_name=f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "JSON":
                    import json
                    json_data = json.dumps(export_data, indent=2, default=str)
                    st.download_button(
                        label=f"📥 Download JSON ({len(export_data)} analyses)",
                        data=json_data,
                        file_name=f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:  # Markdown Report
                    md_report = f"# Investment Analysis Batch Export\n\n"
                    md_report += f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n"
                    md_report += f"**Total Analyses:** {len(export_data)}\n\n---\n\n"
                    
                    for item in export_data:
                        md_report += f"## {item['Ticker']}\n\n"
                        for key, value in item.items():
                            if key != 'Ticker':
                                md_report += f"- **{key}:** {value}\n"
                        md_report += "\n---\n\n"
                    
                    st.download_button(
                        label=f"📥 Download Markdown ({len(export_data)} analyses)",
                        data=md_report,
                        file_name=f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                st.success(f"✅ Exported {len(export_data)} analyses!")
            
            if st.button("❌ Close"):
                st.session_state.show_batch_export = False
                st.rerun()
    
    # Show Google Sheets export interface if requested
    if st.session_state.get('show_sheets_export', False):
        with st.expander("📊 Google Sheets Export with Price Fetching", expanded=True):
            st.write("**Export all analyses to Google Sheets with optional current price fetching**")
            
            # Get analysis archive
            analysis_archive = qa_system.get_analysis_archive()
            
            # Call the update function with UI enabled
            try:
                if update_google_sheets_qa_analyses(analysis_archive, show_price_ui=True):
                    st.success("✅ Successfully exported to Google Sheets!")
                else:
                    st.error("❌ Failed to export to Google Sheets")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                import traceback
                st.exception(e)
            
            if st.button("❌ Close", key="close_sheets_export"):
                st.session_state.show_sheets_export = False
                st.rerun()
    
    # Get data for display
    qa_summary = qa_system.get_qa_summary()
    tracked_tickers = qa_system.get_tracked_tickers()
    analysis_archive = qa_system.get_analysis_archive()
    analysis_stats = qa_system.get_analysis_stats()
    
    # Debug info
    st.sidebar.write(f"**Debug Info:**")
    st.sidebar.write(f"Tracked tickers: {len(tracked_tickers)}")
    st.sidebar.write(f"Analyses: {len(analysis_archive)}")
    if tracked_tickers:
        st.sidebar.write(f"Tickers: {', '.join(tracked_tickers)}")
    
    # Create tabs for different QA views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🎯 Tracked Tickers",
        "📚 Complete Archives", 
        "📈 Weekly Reviews", 
        "🧠 Learning Insights",
        "🔬 Performance Analysis"
    ])
    
    with tab1:
        st.subheader("📊 System Dashboard")
        
        # 🆕 IMPROVEMENT #4: Smart Review Alerts
        if analysis_archive:
            from datetime import datetime, timedelta
            
            # Calculate stocks needing review
            stocks_needing_review = []
            stocks_with_changes = []
            
            for ticker, analyses in analysis_archive.items():
                if len(analyses) > 0:
                    sorted_analyses = sorted(analyses, key=lambda x: x.timestamp, reverse=True)
                    latest = sorted_analyses[0]
                    days_since = (datetime.now() - latest.timestamp).days
                    
                    # Alert if not analyzed in 30+ days
                    if days_since >= 30:
                        stocks_needing_review.append((ticker, days_since, latest.confidence_score))
                    
                    # Alert if significant score change in recent analyses
                    if len(sorted_analyses) >= 2:
                        score_change = latest.confidence_score - sorted_analyses[1].confidence_score
                        if abs(score_change) > 15:
                            stocks_with_changes.append((ticker, score_change, latest.confidence_score))
            
            if stocks_needing_review or stocks_with_changes:
                st.warning("### ⚠️ Smart Alerts")
                
                if stocks_needing_review:
                    with st.expander(f"🔔 {len(stocks_needing_review)} Stock(s) Need Re-Analysis (30+ days old)", expanded=True):
                        for ticker, days, score in sorted(stocks_needing_review, key=lambda x: x[1], reverse=True):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.write(f"**{ticker}**")
                            with col2:
                                st.write(f"⏰ {days} days since last analysis")
                            with col3:
                                st.write(f"Score: {score:.1f}")
                
                if stocks_with_changes:
                    with st.expander(f"📊 {len(stocks_with_changes)} Stock(s) with Significant Score Changes", expanded=True):
                        for ticker, change, current_score in sorted(stocks_with_changes, key=lambda x: abs(x[1]), reverse=True):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.write(f"**{ticker}**")
                            with col2:
                                if change > 0:
                                    st.success(f"📈 +{change:.1f} points")
                                else:
                                    st.error(f"📉 {change:.1f} points")
                            with col3:
                                st.write(f"Now: {current_score:.1f}")
                
                st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Analyses", analysis_stats['total_analyses'])
        
        with col2:
            st.metric("Unique Tickers", analysis_stats['unique_tickers'])
        
        with col3:
            st.metric("QA Recommendations", qa_summary['total_recommendations'])
        
        with col4:
            st.metric("Recent Activity", f"{analysis_stats['recent_activity_count']} (30d)")
        
        with col5:
            avg_confidence = analysis_stats['avg_confidence_score']
            st.metric("Avg Confidence", f"{avg_confidence:.1f}/100")
        
        # Helper function to generalize sector names
        def generalize_sector(sector_name):
            """Simplify long sector names into broader categories."""
            sector = sector_name.upper()
            
            # Technology & Software
            if any(word in sector for word in ['SOFTWARE', 'COMPUTER', 'ELECTRONIC', 'SEMICONDUCTOR', 'TECHNOLOGY']):
                return 'Technology'
            # Healthcare & Pharma
            elif any(word in sector for word in ['PHARMACEUTICAL', 'BIOLOGICAL', 'MEDICAL', 'DRUG', 'HEALTH']):
                return 'Healthcare'
            # Finance
            elif any(word in sector for word in ['BANK', 'FINANCE', 'INSURANCE', 'INVESTMENT', 'CREDIT']):
                return 'Financials'
            # Consumer
            elif any(word in sector for word in ['RETAIL', 'CONSUMER', 'RESTAURANT', 'APPAREL', 'CLOTHING']):
                return 'Consumer'
            # Industrial
            elif any(word in sector for word in ['INDUSTRIAL', 'MACHINERY', 'EQUIPMENT', 'MANUFACTURING', 'CONSTRUCTION']):
                return 'Industrials'
            # Energy & Utilities
            elif any(word in sector for word in ['ENERGY', 'OIL', 'GAS', 'ELECTRIC', 'UTILITY', 'POWER']):
                return 'Energy & Utilities'
            # Services
            elif any(word in sector for word in ['SERVICES', 'CONSULTING', 'MANAGEMENT']):
                return 'Services'
            # Transportation
            elif any(word in sector for word in ['MOTOR', 'VEHICLES', 'AUTO', 'TRANSPORTATION', 'AEROSPACE']):
                return 'Transportation'
            # Materials
            elif any(word in sector for word in ['CHEMICAL', 'MINING', 'METAL', 'MATERIAL']):
                return 'Materials'
            # Communication
            elif any(word in sector for word in ['TELECOM', 'COMMUNICATION', 'MEDIA', 'BROADCASTING']):
                return 'Communication'
            # Real Estate
            elif any(word in sector for word in ['REAL ESTATE', 'REIT', 'PROPERTY']):
                return 'Real Estate'
            else:
                return 'Other'
        
        # Charts
        if analysis_stats['total_analyses'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recommendation Breakdown")
                rec_data = analysis_stats['recommendation_breakdown']
                if rec_data:
                    import plotly.express as px
                    
                    # Simple grouping: Bullish, Neutral, Bearish
                    bullish = rec_data.get('strong_buy', 0) + rec_data.get('buy', 0)
                    neutral = rec_data.get('hold', 0)
                    bearish = rec_data.get('sell', 0) + rec_data.get('strong_sell', 0)
                    
                    simple_data = {
                        'Bullish': bullish,
                        'Neutral': neutral,
                        'Bearish': bearish
                    }
                    
                    # Filter out zeros
                    simple_data = {k: v for k, v in simple_data.items() if v > 0}
                    
                    if simple_data:
                        # Simple bar chart
                        fig = px.bar(
                            x=list(simple_data.keys()),
                            y=list(simple_data.values()),
                            color=list(simple_data.keys()),
                            color_discrete_map={
                                'Bullish': '#22c55e',
                                'Neutral': '#eab308',
                                'Bearish': '#ef4444'
                            },
                            text=list(simple_data.values())
                        )
                        
                        fig.update_traces(
                            textposition='outside',
                            textfont=dict(size=14, color='#333')
                        )
                        
                        fig.update_layout(
                            showlegend=False,
                            xaxis_title="",
                            yaxis_title="Number of Analyses",
                            height=350,
                            margin=dict(l=40, r=20, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sector Distribution")
                sector_data = analysis_stats['sector_breakdown']
                if sector_data:
                    import plotly.express as px
                    
                    # Generalize sectors
                    generalized_sectors = {}
                    for sector, count in sector_data.items():
                        gen_sector = generalize_sector(sector)
                        generalized_sectors[gen_sector] = generalized_sectors.get(gen_sector, 0) + count
                    
                    # Sort and get top sectors
                    sorted_sectors = sorted(generalized_sectors.items(), key=lambda x: x[1], reverse=True)
                    
                    # Limit to top 8 sectors, group rest as "Other"
                    if len(sorted_sectors) > 8:
                        top_sectors = dict(sorted_sectors[:8])
                        other_count = sum(count for _, count in sorted_sectors[8:])
                        if other_count > 0:
                            top_sectors['Other'] = other_count
                    else:
                        top_sectors = dict(sorted_sectors)
                    
                    # Simple pie chart
                    fig = px.pie(
                        values=list(top_sectors.values()),
                        names=list(top_sectors.keys()),
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='label+percent',
                        textfont=dict(size=12),
                        marker=dict(line=dict(color='white', width=2))
                    )
                    
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        ),
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=80)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Biggest Changes Section
        st.markdown("---")
        st.subheader("🔥 Biggest Score Changes")
        st.write("Stocks with the largest score differences between their latest and previous analyses")
        
        if analysis_archive:
            from datetime import datetime
            
            # Find stocks with multiple analyses and calculate changes
            change_data = []
            for ticker, analyses in analysis_archive.items():
                if len(analyses) >= 2:
                    # Sort by timestamp to get latest and previous
                    sorted_analyses = sorted(analyses, key=lambda x: x.timestamp, reverse=True)
                    latest = sorted_analyses[0]
                    previous = sorted_analyses[1]
                    
                    score_change = latest.confidence_score - previous.confidence_score
                    rec_changed = latest.recommendation.value != previous.recommendation.value
                    
                    change_data.append({
                        'ticker': ticker,
                        'score_change': score_change,
                        'latest_score': latest.confidence_score,
                        'previous_score': previous.confidence_score,
                        'latest_rec': latest.recommendation.value,
                        'previous_rec': previous.recommendation.value,
                        'rec_changed': rec_changed,
                        'price_at_analysis': latest.price_at_analysis,
                        'days_ago': (datetime.now() - previous.timestamp).days,
                        'latest_date': latest.timestamp,
                        'previous_date': previous.timestamp
                    })
            
            if change_data:
                # Sort by absolute score change
                change_data.sort(key=lambda x: abs(x['score_change']), reverse=True)
                top_changes = change_data[:5]
                
                for item in top_changes:
                    change_icon = "📈" if item['score_change'] > 0 else "📉" if item['score_change'] < 0 else "➡️"
                    rec_icon = "🟢" if item['latest_rec'] in ['strong_buy', 'buy'] else "🔴" if item['latest_rec'] in ['strong_sell', 'sell'] else "🟡"
                    rec_change_text = f" (was {item['previous_rec'].upper()})" if item['rec_changed'] else ""
                    
                    # Price at time of analysis (always show)
                    price_text = f" | Analysis Price: ${item['price_at_analysis']:.2f}"
                    
                    col1, col2, col3 = st.columns([3, 4, 5])
                    with col1:
                        st.write(f"**{item['ticker']}**")
                    with col2:
                        st.write(f"{change_icon} **{item['score_change']:+.1f}** pts ({item['previous_score']:.1f} → {item['latest_score']:.1f})")
                    with col3:
                        st.write(f"{rec_icon} {item['latest_rec'].upper()}{rec_change_text}{price_text}")
                
                st.caption(f"Showing top {len(top_changes)} stocks with largest score changes")
                st.caption("💡 Tip: Use the Google Sheets export to track current prices with auto-refresh")
            else:
                st.info("No stocks with multiple analyses yet. Re-analyze stocks to see changes.")
        else:
            st.info("No analyses performed yet.")
        
        # Recent Analysis Activity with Ticker Names
        st.markdown("---")
        st.subheader("📈 Recent Analysis Activity")
        
        if analysis_archive:
            # Get recent analyses (last 5)
            all_recent_analyses = []
            for ticker, analyses in analysis_archive.items():
                for analysis in analyses:
                    all_recent_analyses.append((ticker, analysis))
            
            # Sort by timestamp (most recent first)
            all_recent_analyses.sort(key=lambda x: x[1].timestamp, reverse=True)
            recent_analyses = all_recent_analyses[:5]
            
            if recent_analyses:
                for ticker, analysis in recent_analyses:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    with col1:
                        st.write(f"**📊 {ticker}**")
                    with col2:
                        rec_color = "🟢" if analysis.recommendation.value in ['strong_buy', 'buy'] else "🔴" if analysis.recommendation.value in ['strong_sell', 'sell'] else "🟡"
                        st.write(f"{rec_color} {analysis.recommendation.value.upper()}")
                    with col3:
                        st.write(f"**{analysis.confidence_score:.1f}**/100")
                    with col4:
                        st.write(f"*{analysis.timestamp.strftime('%m/%d %H:%M')}*")
                
                st.caption(f"Showing {len(recent_analyses)} most recent analyses")
            else:
                st.info("No recent analyses to display")
        else:
            st.info("No analyses performed yet. Start analyzing stocks to see activity here.")
        
        # Performance tracking (if reviews exist)
        if qa_summary['total_reviews'] > 0:
            st.markdown("---")
            st.subheader("🎯 QA Performance Tracking")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                better_count = qa_summary['performance_stats']['better']
                total_reviews = qa_summary['total_reviews']
                success_rate = (better_count / total_reviews * 100) if total_reviews > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col2:
                st.metric("Total Reviews", qa_summary['total_reviews'])
            
            with col3:
                stocks_due = len(qa_summary['stocks_due_for_review'])
                st.metric("Due for Review", stocks_due)
    
    with tab2:
        st.subheader("🎯 Tracked Tickers in QA System")
        st.write("These tickers are currently being tracked for performance against recommendations.")
        
        if tracked_tickers:
            # Display tracked tickers in a nice grid
            cols_per_row = 4
            for i in range(0, len(tracked_tickers), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, ticker in enumerate(tracked_tickers[i:i+cols_per_row]):
                    with cols[j]:
                        st.info(f"📈 **{ticker}**")
            
            st.markdown("---")
            st.write(f"**Total: {len(tracked_tickers)} tickers being tracked**")
            
            # Show recommendation breakdown for tracked tickers
            if st.session_state.qa_system.recommendations:
                rec_types = {}
                for rec in st.session_state.qa_system.recommendations.values():
                    rec_type = rec.recommendation.value
                    rec_types[rec_type] = rec_types.get(rec_type, 0) + 1
                
                st.subheader("Recommendation Types")
                for rec_type, count in rec_types.items():
                    st.write(f"• **{rec_type.upper()}**: {count} ticker(s)")
        else:
            st.info("No tickers currently being tracked in QA system. Analyze stocks and log them to QA to start tracking.")
    
    with tab3:
        st.subheader("📚 Complete Analysis Archives")
        st.write("All analyses performed, organized by ticker with expandable details.")
        
        # Export all analyses to CSV button
        if analysis_archive:
            import pandas as pd
            from io import StringIO
            
            # Prepare comprehensive CSV data
            csv_rows = []
            for ticker, analyses in analysis_archive.items():
                for analysis in analyses:
                    row = {
                        'Ticker': ticker,
                        'Analysis Date': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'Recommendation': analysis.recommendation.value.upper(),
                        'Confidence Score': f"{analysis.confidence_score:.1f}",
                        'Price at Analysis': f"${analysis.price_at_analysis:.2f}",
                        'Current Price': f"${analysis.current_price:.2f}" if hasattr(analysis, 'current_price') and analysis.current_price else 'N/A',
                        'Price Change %': f"{((analysis.current_price - analysis.price_at_analysis) / analysis.price_at_analysis * 100):.2f}%" if hasattr(analysis, 'current_price') and analysis.current_price else 'N/A',
                    }
                    
                    # Add fundamentals data
                    if analysis.fundamentals:
                        for key, value in analysis.fundamentals.items():
                            if value is not None:
                                formatted_key = key.replace('_', ' ').title()
                                if isinstance(value, float):
                                    row[formatted_key] = f"{value:.2f}"
                                else:
                                    row[formatted_key] = str(value)
                    
                    # Add agent scores if available
                    if hasattr(analysis, 'agent_scores') and analysis.agent_scores:
                        for agent, score in analysis.agent_scores.items():
                            agent_name = agent.replace('_', ' ').title()
                            row[f"{agent_name} Score"] = f"{score:.1f}"
                    
                    # Add key factors as single field
                    if analysis.key_factors:
                        row['Key Factors'] = ' | '.join(analysis.key_factors)
                    
                    # Add rationales
                    if analysis.agent_rationales:
                        for agent, rationale in analysis.agent_rationales.items():
                            if rationale and rationale.strip():
                                agent_name = agent.replace('_', ' ').title()
                                # Clean rationale for CSV (remove newlines, limit length)
                                clean_rationale = ' '.join(rationale.split())
                                if len(clean_rationale) > 500:
                                    clean_rationale = clean_rationale[:497] + '...'
                                row[f"{agent_name} Rationale"] = clean_rationale
                    
                    csv_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(csv_rows)
            
            # Reorder columns for better readability
            priority_cols = ['Ticker', 'Analysis Date', 'Recommendation', 'Confidence Score', 
                           'Price at Analysis', 'Current Price', 'Price Change %']
            other_cols = [col for col in df.columns if col not in priority_cols]
            df = df[priority_cols + other_cols]
            
            # Convert to CSV
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Download button
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info(f"📊 **{len(csv_rows)}** total analyses across **{len(analysis_archive)}** tickers ready for export")
            with col2:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"complete_qa_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download all analyses as a comprehensive CSV file"
                )
            with col3:
                # Google Sheets export
                if st.session_state.sheets_integration.sheet:
                    if st.button("📊 Push to Sheets", use_container_width=True):
                        with st.spinner("Updating..."):
                            success = update_google_sheets_qa_analyses(analysis_archive)
                            if success:
                                st.success("✅ Updated!")
                                sheet_url = st.session_state.sheets_integration.get_sheet_url()
                                if sheet_url:
                                    st.markdown(f"[📄 Open Sheet]({sheet_url})")
                            else:
                                st.error("❌ Update failed")
                else:
                    st.info("Connect in sidebar")
            
            st.markdown("---")
        
        if analysis_archive:
            # Search and filter options
            col1, col2 = st.columns([2, 1])
            with col1:
                search_ticker = st.text_input("🔍 Search ticker:", placeholder="Enter ticker symbol...")
            with col2:
                sort_option = st.selectbox("Sort by:", ["Most Recent", "Ticker A-Z", "Confidence Score"])
            
            # Filter and sort archive
            filtered_archive = analysis_archive
            if search_ticker:
                filtered_archive = {k: v for k, v in analysis_archive.items() 
                                  if search_ticker.upper() in k.upper()}
            
            # Sort the archive
            if sort_option == "Ticker A-Z":
                sorted_tickers = sorted(filtered_archive.keys())
            elif sort_option == "Confidence Score":
                sorted_tickers = sorted(filtered_archive.keys(), 
                                      key=lambda t: max(a.confidence_score for a in filtered_archive[t]), 
                                      reverse=True)
            else:  # Most Recent
                sorted_tickers = sorted(filtered_archive.keys(), 
                                      key=lambda t: max(a.timestamp for a in filtered_archive[t]), 
                                      reverse=True)
            
            # Display archives
            for ticker in sorted_tickers:
                analyses = filtered_archive[ticker]
                
                with st.expander(f"📊 **{ticker}** ({len(analyses)} analysis{'es' if len(analyses) != 1 else ''})"):
                    # Add a "Delete All" button at the top of each ticker's expander
                    col_del1, col_del2 = st.columns([4, 1])
                    with col_del2:
                        if st.button(f"🗑️ Delete All", key=f"delete_all_{ticker}", help=f"Delete all analyses for {ticker}"):
                            if qa_system.delete_all_analyses_for_ticker(ticker):
                                # Auto-sync to Google Sheets if enabled
                                if st.session_state.get('sheets_enabled', False) and st.session_state.get('sheets_auto_update', False):
                                    analysis_archive = qa_system.get_analysis_archive()
                                    update_google_sheets_qa_analyses(analysis_archive, show_price_ui=False)
                                st.success(f"✅ Deleted all analyses for {ticker}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete analyses for {ticker}")
                    
                    # 🆕 IMPROVEMENT #3: Historical Trend Analysis
                    if len(analyses) > 1:
                        st.markdown("### 📈 Historical Trend Analysis")
                        
                        # Sort analyses by timestamp
                        sorted_analyses = sorted(analyses, key=lambda x: x.timestamp)
                        
                        # Prepare data for trend chart
                        dates = [a.timestamp for a in sorted_analyses]
                        confidence_scores = [a.confidence_score for a in sorted_analyses]
                        
                        # Get agent scores if available
                        has_agent_scores = hasattr(sorted_analyses[0], 'agent_scores') and sorted_analyses[0].agent_scores
                        
                        if has_agent_scores:
                            # Multi-line chart with all agent scores
                            fig_trend = go.Figure()
                            
                            # Add confidence score
                            fig_trend.add_trace(go.Scatter(
                                x=dates,
                                y=confidence_scores,
                                mode='lines+markers',
                                name='Final Score',
                                line=dict(width=3, color='blue'),
                                marker=dict(size=10)
                            ))
                            
                            # Add agent scores
                            agent_names = {
                                'value_agent': 'Value',
                                'growth_momentum_agent': 'Growth',
                                'risk_agent': 'Risk',
                                'sentiment_agent': 'Sentiment',
                                'macro_regime_agent': 'Macro'
                            }
                            
                            for agent_key, agent_name in agent_names.items():
                                agent_scores = [a.agent_scores.get(agent_key, None) for a in sorted_analyses if hasattr(a, 'agent_scores') and a.agent_scores]
                                if agent_scores and all(s is not None for s in agent_scores):
                                    fig_trend.add_trace(go.Scatter(
                                        x=dates,
                                        y=agent_scores,
                                        mode='lines+markers',
                                        name=agent_name,
                                        line=dict(width=2)
                                    ))
                            
                            fig_trend.update_layout(
                                title=f"{ticker} Score Trends Over Time",
                                xaxis_title="Analysis Date",
                                yaxis_title="Score",
                                yaxis_range=[0, 100],
                                height=400,
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                            
                            # Score change analysis
                            if len(sorted_analyses) >= 2:
                                latest = sorted_analyses[-1]
                                previous = sorted_analyses[-2]
                                score_change = latest.confidence_score - previous.confidence_score
                                days_between = (latest.timestamp - previous.timestamp).days
                                
                                col_change1, col_change2, col_change3 = st.columns(3)
                                with col_change1:
                                    st.metric("Latest Score", f"{latest.confidence_score:.1f}", delta=f"{score_change:+.1f}")
                                with col_change2:
                                    st.metric("Days Since Last", days_between)
                                with col_change3:
                                    if abs(score_change) > 10:
                                        st.warning(f"⚠️ Significant change: {score_change:+.1f} points")
                                    elif score_change > 0:
                                        st.success(f"✅ Improving: {score_change:+.1f} points")
                                    else:
                                        st.info(f"📉 Declining: {score_change:.1f} points")
                        else:
                            # Simple confidence score trend
                            fig_simple = go.Figure()
                            fig_simple.add_trace(go.Scatter(
                                x=dates,
                                y=confidence_scores,
                                mode='lines+markers',
                                name='Score',
                                line=dict(width=3),
                                marker=dict(size=10)
                            ))
                            fig_simple.update_layout(
                                title=f"{ticker} Score Trend",
                                xaxis_title="Date",
                                yaxis_title="Score",
                                yaxis_range=[0, 100],
                                height=300
                            )
                            st.plotly_chart(fig_simple, use_container_width=True)
                    
                    st.markdown("---")
                    
                    for i, analysis in enumerate(analyses):
                        st.markdown(f"### Analysis #{i+1}")
                        
                        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                        with col1:
                            st.write(f"**Date:** {analysis.timestamp.strftime('%Y-%m-%d %H:%M')}")
                        with col2:
                            st.write(f"**Recommendation:** {analysis.recommendation.value.upper()}")
                        with col3:
                            st.write(f"**Confidence:** {analysis.confidence_score:.1f}/100")
                        with col4:
                            st.write(f"**Price:** ${analysis.price_at_analysis:.2f}")
                        with col5:
                            # Delete button for this specific analysis
                            unique_key = f"delete_{ticker}_{analysis.timestamp.timestamp()}"
                            if st.button("🗑️", key=unique_key, help="Delete this analysis"):
                                # Delete this specific analysis
                                if qa_system.delete_analysis(ticker, analysis.timestamp):
                                    # Auto-sync to Google Sheets if enabled
                                    if st.session_state.get('sheets_enabled', False) and st.session_state.get('sheets_auto_update', False):
                                        analysis_archive = qa_system.get_analysis_archive()
                                        update_google_sheets_qa_analyses(analysis_archive, show_price_ui=False)
                                    st.success(f"✅ Deleted analysis from {analysis.timestamp.strftime('%Y-%m-%d %H:%M')} for {ticker}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete analysis")
                        
                        # Show rationales with collapsible sections
                        if analysis.agent_rationales:
                            st.markdown("**Agent Rationales:**")
                            for agent, rationale in analysis.agent_rationales.items():
                                if rationale and rationale.strip():
                                    st.markdown(f"**🤖 {agent.replace('_', ' ').title()}:**")
                                    st.write(rationale)
                                    st.markdown("---")
                        
                        # Show key factors
                        if analysis.key_factors:
                            st.markdown("**Key Factors:**")
                            for factor in analysis.key_factors:
                                st.write(f"• {factor}")
                        
                        # Show fundamentals summary
                        if analysis.fundamentals:
                            st.markdown("**📊 Fundamentals Data:**")
                            cols = st.columns(2)
                            fund_items = [(k, v) for k, v in analysis.fundamentals.items() if v is not None]
                            for idx, (key, value) in enumerate(fund_items):
                                with cols[idx % 2]:
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        if i < len(analyses) - 1:
                            st.markdown("---")
        else:
            st.info("No analyses in archive yet. Perform stock analyses to build your archive.")
    
    with tab4:
        st.subheader("📈 Weekly Reviews")
        
        # Check for stocks due for review
        stocks_due = qa_summary['stocks_due_for_review']
        
        if stocks_due:
            st.warning(f"⏰ {len(stocks_due)} stock(s) are due for weekly review")
            
            for ticker in stocks_due:
                with st.expander(f"Review {ticker} - Due for Weekly Check"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Stock:** {ticker}")
                        if ticker in qa_system.recommendations:
                            rec = qa_system.recommendations[ticker]
                            st.write(f"**Original Recommendation:** {rec.recommendation.value.upper()}")
                            st.write(f"**Original Price:** ${rec.price_at_recommendation:.2f}")
                            st.write(f"**Date:** {rec.timestamp.strftime('%Y-%m-%d')}")
                            current_time = datetime.now()
                            st.write(f"**Days Since:** {(current_time - rec.timestamp).days}")
                    
                    with col2:
                        if st.button(f"Conduct Review", key=f"review_{ticker}"):
                            with st.spinner(f"Conducting performance review for {ticker}..."):
                                try:
                                    # Get current price
                                    data_provider = st.session_state.data_provider
                                    current_data = data_provider.get_stock_data(ticker)
                                    
                                    if current_data and 'price' in current_data:
                                        current_price = current_data['price']
                                        
                                        # Conduct review
                                        openai_client = st.session_state.get('openai_client')
                                        review = qa_system.conduct_performance_review(
                                            ticker, current_price, openai_client
                                        )
                                        
                                        if review:
                                            st.success(f"✅ Review completed for {ticker}")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to complete review for {ticker}")
                                    else:
                                        st.error(f"Could not fetch current price for {ticker}")
                                        
                                except Exception as e:
                                    st.error(f"Error conducting review: {e}")
        else:
            st.success("✅ All tracked stocks are up to date with their weekly reviews")
        
        # Display recent reviews with details
        if qa_summary['latest_reviews']:
            st.subheader("Recent Detailed Reviews")
            
            for review_data in qa_summary['latest_reviews'][:3]:  # Show top 3
                ticker = review_data['ticker']
                if ticker in qa_system.reviews:
                    latest_review = qa_system.reviews[ticker][-1]  # Most recent review
                    
                    with st.expander(f"{ticker} - Review from {latest_review.review_date.strftime('%Y-%m-%d')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Performance Metrics:**")
                            st.write(f"Price Change: {latest_review.price_change_pct:.2f}%")
                            st.write(f"Performance vs Prediction: {latest_review.performance_vs_prediction}")
                            st.write(f"Analysis Accuracy: {latest_review.analysis_accuracy}")
                        
                        with col2:
                            st.write("**Price Details:**")
                            st.write(f"Original: ${latest_review.original_recommendation.price_at_recommendation:.2f}")
                            st.write(f"Current: ${latest_review.current_price:.2f}")
                            st.write(f"Change: ${latest_review.price_change_absolute:.2f}")
                        
                        if latest_review.lessons_learned:
                            st.write("**Lessons Learned:**")
                            for lesson in latest_review.lessons_learned:
                                st.write(f"• {lesson}")
                        
                        if latest_review.unforeseen_factors:
                            st.write("**Unforeseen Factors:**")
                            for factor in latest_review.unforeseen_factors:
                                st.write(f"• {factor}")
    
    with tab5:
        st.subheader("🧠 Learning Insights")
        
        insights = qa_summary['insights']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if insights['common_mistakes']:
                st.write("**🚨 Common Mistakes to Avoid:**")
                for mistake in insights['common_mistakes'][-5:]:  # Show last 5
                    st.write(f"• {mistake}")
            
            if insights['improvement_patterns']:
                st.write("**📈 Key Improvements Identified:**")
                for improvement in insights['improvement_patterns'][-5:]:
                    st.write(f"• {improvement}")
        
        with col2:
            if insights['successful_strategies']:
                st.write("**✅ Successful Strategies:**")
                for strategy in insights['successful_strategies'][-5:]:
                    st.write(f"• {strategy}")
            
            if insights['market_lessons']:
                st.write("**📚 Market Lessons:**")
                for lesson in insights['market_lessons'][-5:]:
                    st.write(f"• {lesson}")
        
        # Learning insights for future analysis
        st.subheader("Insights for Future Analysis")
        learning_text = qa_system.get_learning_insights_for_analysis()
        st.text_area(
            "Copy this text to include in future analysis prompts:",
            learning_text,
            height=200
        )
        
        # Manual review interface
        st.markdown("---")
        st.subheader("Manual Review Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Conduct Manual Review:**")
            
            # Select stock for manual review
            available_stocks = list(qa_system.recommendations.keys())
            if available_stocks:
                selected_stock = st.selectbox("Select stock for review:", available_stocks)
                
                if selected_stock:
                    rec = qa_system.recommendations[selected_stock]
                    st.write(f"**Original Recommendation:** {rec.recommendation.value.upper()}")
                    st.write(f"**Original Price:** ${rec.price_at_recommendation:.2f}")
                    st.write(f"**Date:** {rec.timestamp.strftime('%Y-%m-%d')}")
                    
                    current_price = st.number_input(
                        "Enter current price:",
                        min_value=0.01,
                        value=rec.price_at_recommendation,
                        step=0.01
                    )
                    
                    if st.button("Conduct Manual Review"):
                        with st.spinner("Conducting review..."):
                            openai_client = st.session_state.get('openai_client')
                            review = qa_system.conduct_performance_review(
                                selected_stock, current_price, openai_client
                            )
                            
                            if review:
                                st.success("✅ Manual review completed")
                                st.rerun()
            else:
                st.info("No recommendations logged yet. Analyze some stocks first!")
        
        with col2:
            st.write("**QA System Statistics:**")
            st.json({
                "Total Recommendations": qa_summary['total_recommendations'],
                "Total Reviews": qa_summary['total_reviews'],
                "Stocks Due for Review": len(qa_summary['stocks_due_for_review']),
                "Success Rate": f"{(qa_summary['performance_stats']['better'] / max(qa_summary['total_reviews'], 1) * 100):.1f}%"
            })
    
    with tab6:
        st.subheader("🔬 Performance Analysis & Model Improvement")
        st.write("**Analyze stocks that moved significantly to identify patterns and improve the model.**")
        st.markdown("---")
        
        # Initialize Performance Analysis Engine
        try:
            from utils.performance_analysis_engine import PerformanceAnalysisEngine
            
            if 'performance_engine' not in st.session_state:
                data_provider = st.session_state.data_provider
                openai_client = st.session_state.get('openai_client')
                perplexity_client = st.session_state.get('perplexity_client')
                st.session_state.performance_engine = PerformanceAnalysisEngine(
                    data_provider, openai_client, perplexity_client
                )
            
            engine = st.session_state.performance_engine
            
            # Info box about data source
            sheets_connected = st.session_state.get('sheets_integration') and st.session_state.sheets_integration.sheet
            if sheets_connected:
                st.info("📊 **Using Google Sheets data**: Analysis will use 'Percent Change' from your connected sheet for faster and more accurate movement detection.")
            else:
                st.info("📈 **Using price history**: Analysis will fetch historical prices to calculate movements. Connect Google Sheets for faster analysis!")
            
            # Configuration section
            st.write("### 📅 Analysis Period Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                analysis_preset = st.selectbox(
                    "Time Period:",
                    ["Last 7 Days", "Last 14 Days", "Last 30 Days", "Last 90 Days", "Custom Range"]
                )
            
            # Calculate dates based on preset
            from datetime import datetime, timedelta
            end_date = datetime.now()
            
            if analysis_preset == "Last 7 Days":
                start_date = end_date - timedelta(days=7)
            elif analysis_preset == "Last 14 Days":
                start_date = end_date - timedelta(days=14)
            elif analysis_preset == "Last 30 Days":
                start_date = end_date - timedelta(days=30)
            elif analysis_preset == "Last 90 Days":
                start_date = end_date - timedelta(days=90)
            else:  # Custom Range
                with col2:
                    start_date = st.date_input("Start Date:", value=end_date - timedelta(days=30))
                with col3:
                    end_date = st.date_input("End Date:", value=end_date)
            
            # Convert dates to strings properly
            from datetime import date
            if isinstance(start_date, str):
                start_date_str = start_date
            elif isinstance(start_date, (datetime, date)):
                start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            elif isinstance(start_date, tuple) and len(start_date) > 0:
                first_date = start_date[0]
                start_date_str = first_date.strftime('%Y-%m-%d') if hasattr(first_date, 'strftime') else str(first_date)
            else:
                start_date_str = str(start_date)
            
            if isinstance(end_date, str):
                end_date_str = end_date
            elif isinstance(end_date, (datetime, date)):
                end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            elif isinstance(end_date, tuple) and len(end_date) > 0:
                first_date = end_date[0]
                end_date_str = first_date.strftime('%Y-%m-%d') if hasattr(first_date, 'strftime') else str(first_date)
            else:
                end_date_str = str(end_date)
            
            # Debug options
            with st.expander("🔧 Advanced Options"):
                debug_mode = st.checkbox("Enable debug logging (shows all stocks and parsing details)", value=True)
                custom_threshold = st.slider("Minimum movement threshold (%)", min_value=1.0, max_value=50.0, value=15.0, step=0.5)
                st.info(f"Will analyze stocks that moved ≥ {custom_threshold}%")
            
            st.markdown("---")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                run_analysis_btn = st.button(
                    "🚀 Run Performance Analysis",
                    type="primary",
                    use_container_width=True,
                    help="Analyze stock movements and generate model recommendations"
                )
            
            with col2:
                view_history_btn = st.button(
                    "📜 View Analysis History",
                    use_container_width=True,
                    help="View previous performance analyses"
                )
            
            with col3:
                view_recommendations_btn = st.button(
                    "💡 View Recommendations",
                    use_container_width=True,
                    help="View model improvement recommendations"
                )
            
            # Run analysis
            if run_analysis_btn:
                with st.spinner(f"🔍 Analyzing stock performance from {start_date_str} to {end_date_str}..."):
                    try:
                        # Get Google Sheets integration if available
                        sheets_integration = st.session_state.get('sheets_integration')
                        
                        # Check if we can get stocks from sheets
                        if not sheets_integration or not sheets_integration.sheet:
                            st.warning("⚠️ Google Sheets not connected. This feature requires Google Sheets to identify stocks with significant movements.")
                            st.info("💡 Go to System Configuration → Google Sheets Integration to connect your sheet.")
                        else:
                            # Get available worksheets first
                            try:
                                worksheets = [ws.title for ws in sheets_integration.sheet.worksheets()]
                                st.info(f"� Available worksheets: {', '.join(worksheets)}")
                                
                                # Check if required worksheet exists
                                required_names = ['Historical Price Analysis', 'Portfolio Analysis', 'Price Analysis']
                                has_required = any(name in worksheets for name in required_names)
                                
                                if not has_required:
                                    st.error(f"❌ Missing required worksheet!")
                                    st.error(f"Please create a worksheet named one of: **{', '.join(required_names)}**")
                                    st.error(f"Current worksheets: {', '.join(worksheets)}")
                                    st.info("💡 The worksheet must have columns: **Ticker**, **Percent Change** (or **Price at Analysis** + **Price**)")
                                else:
                                    st.info(f"�🔍 Scanning all stocks in Google Sheets for movements ≥{custom_threshold}%...")
                                    
                                    # Run comprehensive analysis (will auto-detect stocks with significant movement from sheets)
                                    report = engine.analyze_performance_period(
                                        start_date_str,
                                        end_date_str,
                                        tickers=None,  # Don't filter by tracked - analyze ALL stocks
                                        qa_system=qa_system,
                                        sheets_integration=sheets_integration,
                                        min_threshold=custom_threshold
                                    )
                                    
                                    # Store in session state
                                    st.session_state.latest_performance_report = report
                                    
                                    st.success("✅ Performance analysis complete!")
                                    st.rerun()
                            except Exception as check_error:
                                st.warning(f"Could not check worksheets: {check_error}")
                                # Continue anyway - let the engine handle it
                                st.info(f"🔍 Scanning all stocks in Google Sheets for movements ≥{custom_threshold}%...")
                                
                                report = engine.analyze_performance_period(
                                    start_date_str,
                                    end_date_str,
                                    tickers=None,
                                    qa_system=qa_system,
                                    sheets_integration=sheets_integration,
                                    min_threshold=custom_threshold
                                )
                                
                                st.session_state.latest_performance_report = report
                                st.success("✅ Performance analysis complete!")
                                st.rerun()
                    
                    except KeyError as e:
                        st.error(f"❌ Data error: Missing required field {e}")
                        st.info("💡 Tip: Ensure your Google Sheets has the required columns (Ticker, Percent Change, or Price at Analysis + Price)")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        with st.expander("🔍 Debug Information"):
                            import traceback
                            st.code(traceback.format_exc())
                        st.info("💡 Common fixes: Check API keys, verify tracked stocks exist, ensure date range is valid")
            
            # Display results if available
            if 'latest_performance_report' in st.session_state and st.session_state.latest_performance_report:
                report = st.session_state.latest_performance_report
                
                # Handle case where analysis found no movements
                if report.get('status') == 'no_movements':
                    st.warning("⚠️ " + report.get('message', 'No significant stock movements detected in this period'))
                    st.info("💡 Try expanding the date range or lowering movement thresholds")
                    return
                
                st.markdown("---")
                st.write("## 📊 Analysis Results")
                
                # Executive Summary - with safe access
                if 'executive_summary' in report:
                    st.info(f"**Executive Summary:** {report['executive_summary']}")
                
                # Key Metrics - with safe access
                if 'summary' in report:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Movements", report['summary'].get('total_movements', 0))
                    with col2:
                        st.metric("Analyses Completed", report['summary'].get('analyses_completed', 0))
                    with col3:
                        st.metric("Top Gainers", report['summary'].get('top_gainers_count', 0))
                    with col4:
                        st.metric("Top Losers", report['summary'].get('top_losers_count', 0))
                
                # Top Movers tabs
                st.markdown("---")
                movers_tab1, movers_tab2 = st.tabs(["📈 Top Gainers", "📉 Top Losers"])
                
                with movers_tab1:
                    if report['top_gainers']:
                        st.write("### 🚀 Stocks with Largest Price Increases")
                        
                        for i, movement in enumerate(report['top_gainers'][:10], 1):
                            with st.expander(f"#{i} {movement['ticker']} (+{movement['price_change_pct']:.2f}%) - {movement['magnitude'].upper()}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Price Movement:**")
                                    st.write(f"- Start Price: ${movement['start_price']:.2f}")
                                    st.write(f"- End Price: ${movement['end_price']:.2f}")
                                    st.write(f"- Change: +${movement['price_change_abs']:.2f} (+{movement['price_change_pct']:.2f}%)")
                                    if movement.get('volume_change_pct'):
                                        st.write(f"- Volume Change: {movement['volume_change_pct']:+.1f}%")
                                
                                with col2:
                                    st.write(f"**Details:**")
                                    st.write(f"- Sector: {movement.get('sector', 'N/A')}")
                                    if movement.get('market_cap'):
                                        market_cap_b = movement['market_cap'] / 1e9
                                        st.write(f"- Market Cap: ${market_cap_b:.1f}B")
                                    st.write(f"- Period: {movement['start_date']} to {movement['end_date']}")
                                
                                # Find corresponding analysis
                                analysis = next((a for a in report['analyses'] if a['ticker'] == movement['ticker']), None)
                                if analysis:
                                    st.write(f"**🔍 Root Cause Analysis:**")
                                    st.write(f"**Catalyst:** {analysis['catalyst_summary']}")
                                    st.write(f"**Confidence:** {analysis['confidence']:.0f}%")
                                    
                                    if analysis['root_causes']:
                                        st.write("**Primary Drivers:**")
                                        for cause in analysis['root_causes']:
                                            st.write(f"  • {cause}")
                                    
                                    # Show flags
                                    flags = []
                                    if analysis['earnings_related']:
                                        flags.append("📊 Earnings")
                                    if analysis['news_driven']:
                                        flags.append("📰 News")
                                    if analysis['sector_driven']:
                                        flags.append("🏢 Sector")
                                    if analysis['fundamental_change']:
                                        flags.append("📈 Fundamental")
                                    if flags:
                                        st.write(f"**Flags:** {' | '.join(flags)}")
                    else:
                        st.info("No significant gainers in this period")
                
                with movers_tab2:
                    if report['top_losers']:
                        st.write("### 📉 Stocks with Largest Price Decreases")
                        
                        for i, movement in enumerate(report['top_losers'][:10], 1):
                            with st.expander(f"#{i} {movement['ticker']} ({movement['price_change_pct']:.2f}%) - {movement['magnitude'].upper()}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Price Movement:**")
                                    st.write(f"- Start Price: ${movement['start_price']:.2f}")
                                    st.write(f"- End Price: ${movement['end_price']:.2f}")
                                    st.write(f"- Change: ${movement['price_change_abs']:.2f} ({movement['price_change_pct']:.2f}%)")
                                    if movement.get('volume_change_pct'):
                                        st.write(f"- Volume Change: {movement['volume_change_pct']:+.1f}%")
                                
                                with col2:
                                    st.write(f"**Details:**")
                                    st.write(f"- Sector: {movement.get('sector', 'N/A')}")
                                    if movement.get('market_cap'):
                                        market_cap_b = movement['market_cap'] / 1e9
                                        st.write(f"- Market Cap: ${market_cap_b:.1f}B")
                                    st.write(f"- Period: {movement['start_date']} to {movement['end_date']}")
                                
                                # Find corresponding analysis
                                analysis = next((a for a in report['analyses'] if a['ticker'] == movement['ticker']), None)
                                if analysis:
                                    st.write(f"**🔍 Root Cause Analysis:**")
                                    st.write(f"**Catalyst:** {analysis['catalyst_summary']}")
                                    st.write(f"**Confidence:** {analysis['confidence']:.0f}%")
                                    
                                    if analysis['root_causes']:
                                        st.write("**Primary Drivers:**")
                                        for cause in analysis['root_causes']:
                                            st.write(f"  • {cause}")
                                    
                                    # Show flags
                                    flags = []
                                    if analysis['earnings_related']:
                                        flags.append("📊 Earnings")
                                    if analysis['news_driven']:
                                        flags.append("📰 News")
                                    if analysis['sector_driven']:
                                        flags.append("🏢 Sector")
                                    if analysis['fundamental_change']:
                                        flags.append("📈 Fundamental")
                                    if flags:
                                        st.write(f"**Flags:** {' | '.join(flags)}")
                    else:
                        st.info("No significant losers in this period")
                
                # Model Recommendations
                if report['recommendations']:
                    st.markdown("---")
                    st.write("## 💡 Model Improvement Recommendations")
                    st.write(f"**Generated {report['summary']['recommendations_generated']} recommendations** based on observed patterns")
                    
                    # Filter by priority
                    priority_filter = st.multiselect(
                        "Filter by Priority:",
                        ["critical", "high", "medium", "low"],
                        default=["critical", "high"]
                    )
                    
                    filtered_recommendations = [
                        r for r in report['recommendations']
                        if r['priority'] in priority_filter
                    ]
                    
                    for i, rec in enumerate(filtered_recommendations, 1):
                        priority_emoji = {
                            'critical': '🚨',
                            'high': '⚠️',
                            'medium': '💡',
                            'low': 'ℹ️'
                        }
                        
                        emoji = priority_emoji.get(rec['priority'], '💡')
                        
                        with st.expander(f"{emoji} [{rec['priority'].upper()}] {rec['specific_change']}", expanded=(rec['priority'] in ['critical', 'high'])):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Recommendation:** {rec['specific_change']}")
                                st.write(f"**Rationale:** {rec['rationale']}")
                                st.write(f"**Expected Impact:** {rec['expected_impact']}")
                                
                                if rec['supporting_evidence']:
                                    st.write("**Supporting Evidence:**")
                                    for evidence in rec['supporting_evidence'][:3]:
                                        st.write(f"  • {evidence}")
                                
                                st.write("**Implementation Steps:**")
                                for step in rec['implementation_steps']:
                                    st.write(f"  {step}")
                            
                            with col2:
                                st.metric("Confidence", f"{rec['confidence']:.0f}%")
                                st.write(f"**Category:** {rec['category']}")
                                st.write(f"**Affected Agents:**")
                                for agent in rec['affected_agents']:
                                    st.write(f"  • {agent}")
                                
                                # Action buttons
                                if st.button(f"✅ Mark as Implemented", key=f"implement_{rec['recommendation_id']}"):
                                    notes = st.text_input("Implementation notes:", key=f"notes_{rec['recommendation_id']}")
                                    engine.mark_recommendation_implemented(rec['recommendation_id'], notes)
                                    st.success("Marked as implemented!")
                
                # Pattern Analysis
                if report.get('patterns'):
                    st.markdown("---")
                    st.write("## 📊 Pattern Analysis")
                    
                    patterns = report['patterns']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Earnings-Related", f"{patterns.get('earnings_frequency', 0)*100:.0f}%")
                        st.metric("News-Driven", f"{patterns.get('news_driven_frequency', 0)*100:.0f}%")
                    
                    with col2:
                        st.metric("Market-Driven", f"{patterns.get('market_driven_frequency', 0)*100:.0f}%")
                        st.metric("Sector-Driven", f"{patterns.get('sector_driven_frequency', 0)*100:.0f}%")
                    
                    with col3:
                        st.metric("Fundamental Change", f"{patterns.get('fundamental_change_frequency', 0)*100:.0f}%")
                        st.metric("Technical Breakout", f"{patterns.get('technical_breakout_frequency', 0)*100:.0f}%")
                    
                    # Direction breakdown
                    st.write("**Movement Direction:**")
                    total_moves = patterns.get('up_movements', 0) + patterns.get('down_movements', 0)
                    if total_moves > 0:
                        up_pct = (patterns.get('up_movements', 0) / total_moves) * 100
                        down_pct = (patterns.get('down_movements', 0) / total_moves) * 100
                        st.write(f"  • Up movements: {patterns.get('up_movements', 0)} ({up_pct:.1f}%)")
                        st.write(f"  • Down movements: {patterns.get('down_movements', 0)} ({down_pct:.1f}%)")
            
            # View history
            elif view_history_btn:
                st.write("### 📜 Analysis History")
                # TODO: Implement history viewing
                st.info("Analysis history feature coming soon")
            
            # View recommendations
            elif view_recommendations_btn:
                st.write("### 💡 All Model Recommendations")
                latest_recs = engine.get_latest_recommendations(20)
                
                if latest_recs:
                    for rec in latest_recs:
                        priority_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '💡', 'low': 'ℹ️'}
                        emoji = priority_emoji.get(rec.get('priority', 'medium'), '💡')
                        
                        with st.expander(f"{emoji} [{rec.get('priority', 'N/A').upper()}] {rec.get('specific_change', 'N/A')}"):
                            st.write(f"**Recommendation ID:** {rec.get('recommendation_id', 'N/A')}")
                            st.write(f"**Category:** {rec.get('category', 'N/A')}")
                            st.write(f"**Rationale:** {rec.get('rationale', 'N/A')}")
                            st.write(f"**Confidence:** {rec.get('confidence', 0):.0f}%")
                else:
                    st.info("No recommendations yet. Run a performance analysis first!")
        
        except ImportError as e:
            st.error(f"❌ Performance Analysis Engine not available: {e}")
            st.info("The performance analysis feature requires the PerformanceAnalysisEngine module.")
        except Exception as e:
            st.error(f"❌ Error initializing Performance Analysis: {e}")
            import traceback
            st.code(traceback.format_exc())


def system_status_and_ai_disclosure_page():
    """Combined system status and AI disclosure page."""
    st.header("🔧 System Status & AI Disclosure")
    st.write("Monitor system health, data provider status, and AI usage information.")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📊 System Status", "🤖 AI Usage Disclosure"])
    
    with tab1:
        st.subheader("📊 Data Provider Status")
        
        # Check if data provider is available
        if not st.session_state.data_provider:
            st.error("❌ Data provider not initialized. Please restart the application.")
            return
        
        data_provider = st.session_state.data_provider
        
        # Display Data Provider Information
        st.write("**Provider Information**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Provider Type", "Enhanced Data Provider")
        
        with col2:
            # Check if provider has premium services
            has_polygon = hasattr(data_provider, 'polygon_client') and data_provider.polygon_client is not None
            has_perplexity = hasattr(data_provider, 'perplexity_client') and data_provider.perplexity_client is not None
            premium_count = sum([has_polygon, has_perplexity])
            st.metric("Premium Services", f"{premium_count}/2 Available")
        
        with col3:
            # Check cache directory
            cache_dir = Path("data/cache")
            cache_exists = cache_dir.exists()
            st.metric("Cache Status", "Available" if cache_exists else "Not Found")
        
        # API Keys Status
        st.markdown("---")
        st.write("**🔑 API Keys Status**")
        
        api_keys_status = {
            "Alpha Vantage": bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
            "OpenAI": bool(os.getenv('OPENAI_API_KEY')),
            "Polygon.io": bool(os.getenv('POLYGON_API_KEY')),
            "Perplexity AI": bool(os.getenv('PERPLEXITY_API_KEY')),
            "NewsAPI": bool(os.getenv('NEWSAPI_KEY')),
            "IEX Cloud": bool(os.getenv('IEX_TOKEN'))
        }
        
        cols = st.columns(3)
        for i, (service, available) in enumerate(api_keys_status.items()):
            with cols[i % 3]:
                icon = "✅" if available else "❌"
                status_text = "Available" if available else "Missing"
                st.write(f"{icon} **{service}**: {status_text}")
        
        # Provider Capabilities
        st.markdown("---")
        st.write("**⚡ Provider Capabilities**")
        
        capabilities = {
            "Stock Price Data": True,
            "Fundamentals Data": True,
            "News & Sentiment": bool(os.getenv('NEWSAPI_KEY')),
            "Premium Price Data": bool(os.getenv('POLYGON_API_KEY')),
            "AI-Enhanced Analysis": bool(os.getenv('PERPLEXITY_API_KEY')),
            "52-Week Range Verification": True,
            "Multi-Source Fallback": True
        }
        
        col1, col2 = st.columns(2)
        for i, (capability, available) in enumerate(capabilities.items()):
            with col1 if i % 2 == 0 else col2:
                icon = "✅" if available else "⚠️"
                st.write(f"{icon} {capability}")
        
        # Cache Information
        if cache_exists:
            st.markdown("---")
            st.write("**💾 Cache Information**")
            try:
                cache_files = list(cache_dir.glob("*"))
                total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
                total_size_mb = total_size / (1024 * 1024)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cache Files", len(cache_files))
                with col2:
                    st.metric("Total Size", f"{total_size_mb:.1f} MB")
                with col3:
                    # Show newest cache file age
                    if cache_files:
                        newest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
                        current_time = datetime.now().timestamp()
                        age_hours = (current_time - newest_file.stat().st_mtime) / 3600
                        st.metric("Newest Cache", f"{age_hours:.1f} hours ago")
            except Exception as e:
                st.warning(f"Could not read cache information: {e}")
        
        # Data Source Test
        st.markdown("---")
        st.write("**🧪 Test Data Sources**")
        
        test_ticker = st.text_input("Test ticker:", value="AAPL")
        
        if st.button("Test All Data Sources"):
            with st.spinner("Testing data sources..."):
                results = {}
                
                # Test price data
                try:
                    if hasattr(st.session_state.data_provider, 'get_price_history_enhanced'):
                        price_data = st.session_state.data_provider.get_price_history_enhanced(
                            test_ticker, "2024-01-01", "2024-12-31"
                        )
                    else:
                        price_data = st.session_state.data_provider.get_price_history(
                            test_ticker, "2024-01-01", "2024-12-31"
                        )
                    
                    if not price_data.empty:
                        results['Price Data'] = f"✅ {len(price_data)} days of data"
                        if 'SYNTHETIC_DATA' in price_data.columns:
                            results['Price Data'] += " (⚠️ Synthetic)"
                    else:
                        results['Price Data'] = "❌ No data"
                        
                except Exception as e:
                    results['Price Data'] = f"❌ Error: {str(e)}"
                
                # Test fundamentals
                try:
                    if hasattr(st.session_state.data_provider, 'get_fundamentals_enhanced'):
                        fund_data = st.session_state.data_provider.get_fundamentals_enhanced(test_ticker)
                    else:
                        fund_data = st.session_state.data_provider.get_fundamentals(test_ticker)
                    
                    if fund_data:
                        results['Fundamentals'] = f"✅ {len(fund_data)} data points"
                        if fund_data.get('estimated'):
                            results['Fundamentals'] += " (⚠️ Estimated)"
                    else:
                        results['Fundamentals'] = "❌ No data"
                        
                except Exception as e:
                    results['Fundamentals'] = f"❌ Error: {str(e)}"
                
                # Display results
                for source, result in results.items():
                    st.write(f"**{source}:** {result}")
        
        # Clear Cache
        if st.button("🗑️ Clear Cache", help="Clear cached data to force fresh API calls"):
            cache_dir = Path("data/cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                st.success("Cache cleared!")
            else:
                st.info("No cache to clear")
    
    with tab2:
        st.subheader("🤖 AI Usage Disclosure")
        
        disclosure_logger = get_disclosure_logger()
        summary = disclosure_logger.get_disclosure_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total API Calls", summary['total_calls'])
        with col2:
            st.metric("Total Tokens", f"{summary['total_tokens']:,}")
        with col3:
            st.metric("Estimated Cost", f"${summary['total_cost_usd']:.2f}")
        
        st.write(f"**Tools Used:** {', '.join(summary.get('tools_used', []))}")
        st.write(f"**Log File:** `{summary.get('log_file', 'N/A')}`")
        
        # Download log
        log_file = summary.get('log_file', '')
        if log_file and Path(log_file).exists():
            with open(log_file, 'r') as f:
                log_data = f.read()
            
            st.download_button(
                label="📥 Download Disclosure Log",
                data=log_data,
                file_name="ai_disclosure_log.jsonl",
                mime="application/json"
            )
        
        st.info("""
        **For Works Cited:**
        
        This system uses the following APIs/tools:
        - OpenAI GPT-4o-mini for agent reasoning and rationale generation, as well as perplexityAI for enforcing real time data retrieval. 
        - yfinance/PolygonIO for market data
        - Alpha Vantage for fundamental data and macroeconomic indicators
        - NewsAPI for news sentiment analysis 
        
        All API calls are logged with timestamps, purposes, and token usage for full disclosure.
        """)
        
        # Premium Setup Guide
        st.markdown("---")
        st.subheader("🔧 Premium Setup Guide")
        
        with st.expander("View Premium API Setup Instructions"):
            st.markdown("""
            ### Recommended Premium APIs for Production
            
            **For reliable data access without rate limits:**
            
            1. **IEX Cloud** ($9/month) - Excellent US stock data
               - Add to .env: `IEX_TOKEN=your_token_here`
               - Get token: https://iexcloud.io/
            
            2. **Alpha Vantage Premium** ($49.99/month) - Comprehensive fundamentals  
               - Upgrade your existing key at: https://www.alphavantage.co/premium/
               - 1200 calls/minute vs 5 calls/minute free
            
            3. **Polygon.io** ($99/month) - Professional grade data
               - Add to .env: `POLYGON_API_KEY=your_key_here` 
               - Get key: https://polygon.io/
            
            **Total recommended cost: ~$60/month for rock-solid data access**
            
            ### Current Free Tier Limitations:
            - yfinance: ~2000 requests/day (rate limited)
            - Alpha Vantage: 5 calls/minute, 500/day
            - NewsAPI: 100 requests/day
            
            ### Testing vs Production:
            - Free tier works fine for testing and development
            - Premium recommended for live trading or intensive analysis
            """)


def configuration_page():
    """Configuration management page."""
    st.header("System Configuration")
    st.write("Manage Investment Policy Statement constraints and model parameters.")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Client Profile Upload", "IPS Configuration", "Agent Weights", "⏱️ Timing Analytics"])
    
    with tab1:
        st.subheader("Client Profile Upload")
        st.write("Configure client requirements and investment objectives (detailed text input)")
        
        # Text input for client profile
        client_profile = st.text_area(
            "Client Profile & Requirements:",
            height=300,
            placeholder="""Paste your client profile here. For example:

The client is a 35-year-old technology executive with a high risk tolerance and 10-year investment horizon. They are seeking aggressive growth and are comfortable with high volatility. The client wants to exclude tobacco and weapons companies from their portfolio.

Key constraints:
- No single position over 10% of portfolio
- Maximum 40% allocation to any single sector
- Prefer growth-oriented companies with strong fundamentals
- ESG considerations: exclude tobacco, weapons, fossil fuels
- Minimum $5 stock price
- Beta range: 0.8 to 1.3 acceptable

The client has expressed interest in technology, healthcare, and renewable energy sectors...""",
            help="Paste the detailed client profile text here. The system will parse this and update the IPS automatically."
        )
        
        if st.button("📝 Parse and Update IPS"):
            if client_profile:
                st.info("Manual parsing of client profile text is not yet implemented. Please configure IPS directly in the next tab.")
            else:
                st.warning("Please enter a client profile first.")
    
    with tab2:
        st.subheader("IPS Configuration")
        st.write("Configure Investment Policy Statement constraints.")
        
        # Load current IPS
        ips = st.session_state.config_loader.load_ips()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Position & Sector Constraints:**")
            max_position = st.number_input(
                "Max Single Position (%)", 
                value=float(ips.get('position_limits', {}).get('max_position_pct', 10.0)), 
                min_value=1.0, 
                max_value=50.0
            )
            max_sector = st.number_input(
                "Max Sector Allocation (%)", 
                value=float(ips.get('position_limits', {}).get('max_sector_pct', 30.0)), 
                min_value=10.0, 
                max_value=100.0
            )
            
            st.write("**Price & Market Cap:**")
            min_price = st.number_input(
                "Min Stock Price ($)", 
                value=float(ips.get('universe', {}).get('min_price', 1.0)), 
                min_value=0.0
            )
            min_market_cap = st.number_input(
                "Min Market Cap ($B)", 
                value=float(ips.get('universe', {}).get('min_market_cap', 1000000000)) / 1000000000, 
                min_value=0.0
            )
        
        with col2:
            st.write("**Risk Parameters:**")
            min_beta = st.number_input(
                "Min Beta", 
                value=float(ips.get('portfolio_constraints', {}).get('beta_min', 0.7)), 
                min_value=0.0, 
                max_value=3.0
            )
            max_beta = st.number_input(
                "Max Beta", 
                value=float(ips.get('portfolio_constraints', {}).get('beta_max', 1.3)), 
                min_value=0.0, 
                max_value=3.0
            )
            max_volatility = st.number_input(
                "Max Portfolio Volatility (%)", 
                value=float(ips.get('portfolio_constraints', {}).get('max_portfolio_volatility', 18.0)), 
                min_value=0.0, 
                max_value=50.0
            )
        
        st.write("**Excluded Sectors:**")
        current_exclusions = ips.get('exclusions', {}).get('sectors', [])
        excluded_sectors = st.multiselect(
            "Select sectors to exclude",
            options=["Energy", "Financials", "Healthcare", "Technology", "Consumer Staples", "Consumer Discretionary", 
                    "Industrials", "Materials", "Real Estate", "Utilities", "Communication Services", "Tobacco", "Weapons"],
            default=current_exclusions
        )
        
        if st.button("💾 Save IPS Configuration"):
            # Update IPS with proper structure
            if 'position_limits' not in ips:
                ips['position_limits'] = {}
            ips['position_limits']['max_position_pct'] = max_position
            ips['position_limits']['max_sector_pct'] = max_sector
            
            if 'universe' not in ips:
                ips['universe'] = {}
            ips['universe']['min_price'] = min_price
            ips['universe']['min_market_cap'] = min_market_cap * 1000000000
            
            if 'portfolio_constraints' not in ips:
                ips['portfolio_constraints'] = {}
            ips['portfolio_constraints']['beta_min'] = min_beta
            ips['portfolio_constraints']['beta_max'] = max_beta
            ips['portfolio_constraints']['max_portfolio_volatility'] = max_volatility
            
            if 'exclusions' not in ips:
                ips['exclusions'] = {}
            ips['exclusions']['sectors'] = excluded_sectors
            
            st.session_state.config_loader.save_ips(ips)
            st.success("✅ IPS configuration saved!")
    
    with tab3:
        st.subheader("Agent Weights")
        st.write("Adjust how much each agent influences the final score.")
        
        # Load current weights
        model_config = st.session_state.config_loader.load_model_config()
        weights = model_config['agent_weights']
        
        new_weights = {}
        for agent, weight in weights.items():
            new_weights[agent] = st.slider(
                f"{agent.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=3.0,
                value=float(weight),
                step=0.1,
                help=f"Current weight: {weight}"
            )
        
        if st.button("Save Agent Weights"):
            st.session_state.config_loader.update_model_weights(new_weights)
            st.success("✅ Agent weights updated!")
            st.info("ℹ️ System will be reinitialized on next analysis.")
            st.session_state.initialized = False
    
    with tab4:
        st.subheader("⏱️ Analysis Timing Analytics")
        st.write("Deep insights into step-level timing data collected from all analyses.")
        
        if hasattr(st.session_state, 'step_time_manager'):
            manager = st.session_state.step_time_manager
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            total_samples = sum(len(manager.step_times.get(i, [])) for i in range(1, 11))
            all_stats = manager.get_all_stats()
            
            with col1:
                st.metric("Total Data Points", f"{total_samples:,}")
            
            with col2:
                steps_tracked = len(all_stats)
                st.metric("Steps Tracked", f"{steps_tracked}/10")
            
            with col3:
                if all_stats:
                    avg_analysis_time = sum(s['avg'] for s in all_stats.values())
                    st.metric("Est. Analysis Time", f"{avg_analysis_time:.1f}s")
                else:
                    st.metric("Est. Analysis Time", "No data")
            
            st.markdown("---")
            
            # Detailed step breakdown
            st.subheader("📊 Step-by-Step Breakdown")
            
            step_names = {
                1: "📥 Data Gathering - Fundamentals",
                2: "📈 Data Gathering - Market Data",
                3: "💰 Value Agent Analysis",
                4: "📊 Growth/Momentum Agent Analysis",
                5: "🌍 Macro Regime Agent Analysis",
                6: "⚠️ Risk Agent Analysis",
                7: "💭 Sentiment Agent Analysis",
                8: "⚖️ Score Blending",
                9: "✅ Client Layer Validation",
                10: "🎯 Final Analysis"
            }
            
            if all_stats:
                for step in sorted(all_stats.keys()):
                    stats = all_stats[step]
                    name = step_names.get(step, f"Step {step}")
                    
                    with st.expander(f"**{name}**", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Samples", stats['count'])
                            st.metric("Average", f"{stats['avg']:.2f}s")
                        
                        with col2:
                            st.metric("Median", f"{stats['median']:.2f}s")
                            st.metric("Std Dev", f"{stats['std_dev']:.2f}s")
                        
                        with col3:
                            st.metric("Minimum", f"{stats['min']:.2f}s")
                            st.metric("Maximum", f"{stats['max']:.2f}s")
                        
                        with col4:
                            st.metric("25th %ile", f"{stats['p25']:.2f}s")
                            st.metric("75th %ile", f"{stats['p75']:.2f}s")
                
                st.markdown("---")
                
                # Export option
                if st.button("📥 Export Timing Data"):
                    import pandas as pd
                    from datetime import datetime
                    
                    export_data = []
                    for step, stats in all_stats.items():
                        export_data.append({
                            'Step': step,
                            'Name': step_names.get(step, f"Step {step}"),
                            'Count': stats['count'],
                            'Average': stats['avg'],
                            'Median': stats['median'],
                            'Std_Dev': stats['std_dev'],
                            'Min': stats['min'],
                            'Max': stats['max'],
                            'P25': stats['p25'],
                            'P75': stats['p75']
                        })
                    
                    df = pd.DataFrame(export_data)
                    csv_data = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Timing Data CSV",
                        data=csv_data,
                        file_name=f"timing_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No timing data available yet. Run some analyses to collect timing statistics.")
        else:
            st.warning("Step time manager not initialized.")


# Duplicate function removed - configuration_page defined above


# Old disclosure_page and data_status_page functions removed - consolidated into system_status_and_ai_disclosure_page


def sync_all_archives_to_sheets() -> bool:
    """
    Sync all existing portfolio and QA archives to Google Sheets on first connection.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        sheets_integration = st.session_state.sheets_integration
        
        if not sheets_integration or not sheets_integration.sheet:
            return False
        
        synced_count = 0
        
        # Note: Portfolio selection logs only contain the selection process, not full portfolio data
        # Full portfolio data with scores/rationales is only available during active analysis
        
        # Sync QA analyses
        qa_system = st.session_state.qa_system
        analysis_archive = qa_system.get_analysis_archive()
        
        if analysis_archive:
            success = update_google_sheets_qa_analyses(analysis_archive)
            if success:
                synced_count += len(analysis_archive)
        
        return synced_count > 0
        
    except Exception as e:
        print(f"Error syncing archives: {e}")
        return False


def update_google_sheets_portfolio(result: dict) -> bool:
    """
    Update Google Sheets with comprehensive portfolio analysis results.
    Creates a detailed "Portfolio Recommendations" sheet with full analysis data.
    
    Args:
        result: Portfolio analysis result dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        sheets_integration = st.session_state.sheets_integration
        
        if not sheets_integration or not sheets_integration.sheet:
            return False
        
        # Get or create "Portfolio Recommendations" worksheet
        try:
            worksheet = sheets_integration.sheet.worksheet("Portfolio Recommendations")
            # Clear existing data
            worksheet.clear()
        except:
            worksheet = sheets_integration.sheet.add_worksheet(title="Portfolio Recommendations", rows=1000, cols=40)
        
        # Header row
        headers = [
            'Ticker', 'Company Name', 'Recommendation', 'Confidence Score', 'Eligible',
            'Price', 'Market Cap', 'Sector', 'Industry',
            'Value Score', 'Growth Score', 'Macro Score', 'Risk Score', 'Sentiment Score', 'Client Layer Score',
            'Value Rationale', 'Growth Rationale', 'Macro Rationale', 'Risk Rationale', 'Sentiment Rationale', 
            'Client Layer Rationale', 'Final Rationale',
            'P/E Ratio', 'P/B Ratio', 'ROE', 'Debt/Equity', 'Beta',
            'Data Sources', 'Key Metrics', 'Risk Assessment',
            'Export Date'
        ]
        
        # Format header
        worksheet.update('A1', [headers])
        worksheet.format('A1:AE1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.8},
            'textFormat': {'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
        })
        
        # Prepare data rows
        rows = []
        
        if 'final_portfolio' in result and result['final_portfolio']:
            for stock in result['final_portfolio']:
                recommendation_type = _determine_recommendation_type(stock['final_score'])
                
                row = [
                    stock['ticker'],
                    stock['fundamentals'].get('name', 'N/A'),
                    recommendation_type.value.upper(),
                    round(stock['final_score'], 1),
                    'Yes' if stock['eligible'] else 'No',
                    stock['fundamentals'].get('price', 0),
                    stock['fundamentals'].get('market_cap', 0),
                    stock['fundamentals'].get('sector', 'N/A'),
                    stock['fundamentals'].get('industry', 'N/A'),
                    round(stock.get('agent_scores', {}).get('value_agent', 0), 1),
                    round(stock.get('agent_scores', {}).get('growth_momentum_agent', 0), 1),
                    round(stock.get('agent_scores', {}).get('macro_regime_agent', 0), 1),
                    round(stock.get('agent_scores', {}).get('risk_agent', 0), 1),
                    round(stock.get('agent_scores', {}).get('sentiment_agent', 0), 1),
                    round(stock.get('agent_scores', {}).get('client_layer_agent', 0), 1),
                    stock.get('agent_rationales', {}).get('value_agent', '')[:1000],
                    stock.get('agent_rationales', {}).get('growth_momentum_agent', '')[:1000],
                    stock.get('agent_rationales', {}).get('macro_regime_agent', '')[:1000],
                    stock.get('agent_rationales', {}).get('risk_agent', '')[:1000],
                    stock.get('agent_rationales', {}).get('sentiment_agent', '')[:1000],
                    stock.get('agent_rationales', {}).get('client_layer_agent', '')[:1000],
                    stock.get('rationale', '')[:1500],
                    stock['fundamentals'].get('pe_ratio', 0),
                    stock['fundamentals'].get('pb_ratio', 0),
                    stock['fundamentals'].get('roe', 0),
                    stock['fundamentals'].get('debt_to_equity', 0),
                    stock['fundamentals'].get('beta', 0),
                    ', '.join(stock.get('data_sources', [])),
                    ', '.join(stock.get('key_metrics', [])),
                    stock.get('risk_assessment', ''),
                    datetime.now().strftime('%Y-%m-%d')
                ]
                rows.append(row)
        
        # Write data
        if rows:
            worksheet.update('A2', rows)
            
            # Auto-resize columns
            for i in range(len(headers)):
                worksheet.format(f'{chr(65+i)}:{chr(65+i)}', {'wrapStrategy': 'WRAP'})
        
        return True
        
    except Exception as e:
        print(f"Error updating Google Sheets portfolio: {e}")
        return False


# Configuration page code ends here
# Google Sheets integration functions defined below
# (Note: sync_all_archives_to_sheets is already defined earlier in file)

    with tab2:
        st.subheader("Investment Policy Statement")        # Load current IPS
        ips = st.session_state.config_loader.load_ips()
        
        # Client information
        st.write("**Client Information**")
        col1, col2 = st.columns(2)
        with col1:
            client_name = st.text_input("Client Name", value=ips['client']['name'])
            
            # Enhanced risk tolerance options to match MTWB Foundation profile
            risk_options = ["low", "moderate", "moderate-aggressive", "high", "very-high"]
            current_risk = ips['client']['risk_tolerance']
            
            # Handle risk tolerance safely
            try:
                risk_index = risk_options.index(current_risk)
            except ValueError:
                # If the current risk isn't in our list, default to moderate and show warning
                st.warning(f"⚠️ Unknown risk tolerance '{current_risk}', defaulting to 'moderate'")
                risk_index = 1  # moderate
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                risk_options,
                index=risk_index,
                help="Risk tolerance level for investment strategy"
            )
        with col2:
            time_horizon = st.number_input(
                "Time Horizon (years)",
                min_value=1,
                max_value=30,
                value=ips['client']['time_horizon_years']
            )
            cash_buffer = st.number_input(
                "Cash Buffer (%)",
                min_value=0,
                max_value=50,
                value=ips['client']['cash_buffer_pct']
            )
        
        # Position limits
        st.write("**Position Limits**")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_position = st.number_input(
                "Max Position (%)",
                min_value=1,
                max_value=50,
                value=ips['position_limits']['max_position_pct']
            )
        with col2:
            max_sector = st.number_input(
                "Max Sector (%)",
                min_value=5,
                max_value=100,
                value=ips['position_limits']['max_sector_pct']
            )
        with col3:
            max_industry = st.number_input(
                "Max Industry (%)",
                min_value=5,
                max_value=100,
                value=ips['position_limits']['max_industry_pct']
            )
        
        if st.button("Save IPS Configuration"):
            # Update IPS
            ips['client']['name'] = client_name
            ips['client']['risk_tolerance'] = risk_tolerance
            ips['client']['time_horizon_years'] = time_horizon
            ips['client']['cash_buffer_pct'] = cash_buffer
            ips['position_limits']['max_position_pct'] = max_position
            ips['position_limits']['max_sector_pct'] = max_sector
            ips['position_limits']['max_industry_pct'] = max_industry
            
            st.session_state.config_loader.save_ips(ips)
            st.success("✅ IPS configuration saved!")
    
    with tab3:
        st.subheader("Agent Weights")
        st.write("Adjust how much each agent influences the final score.")
        
        # Load current weights
        model_config = st.session_state.config_loader.load_model_config()
        weights = model_config['agent_weights']
        
        new_weights = {}
        for agent, weight in weights.items():
            new_weights[agent] = st.slider(
                f"{agent.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=3.0,
                value=float(weight),
                step=0.1,
                help=f"Current weight: {weight}"
            )
        
        if st.button("Save Agent Weights"):
            st.session_state.config_loader.update_model_weights(new_weights)
            st.success("✅ Agent weights updated!")
            st.info("ℹ️ System will be reinitialized on next analysis.")
            st.session_state.initialized = False
    
    with tab4:
        st.subheader("⏱️ Analysis Timing Analytics")
        st.write("Deep insights into step-level timing data collected from all analyses.")
        
        if hasattr(st.session_state, 'step_time_manager'):
            manager = st.session_state.step_time_manager
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            total_samples = sum(len(manager.step_times.get(i, [])) for i in range(1, 11))
            all_stats = manager.get_all_stats()
            
            with col1:
                st.metric("Total Data Points", f"{total_samples:,}")
            
            with col2:
                steps_tracked = len(all_stats)
                st.metric("Steps Tracked", f"{steps_tracked}/10")
            
            with col3:
                if all_stats:
                    avg_analysis_time = sum(s['avg'] for s in all_stats.values())
                    st.metric("Est. Analysis Time", f"{avg_analysis_time:.1f}s")
                else:
                    st.metric("Est. Analysis Time", "No data")
            
            st.markdown("---")
            
            # Detailed step breakdown
            st.subheader("📊 Step-by-Step Breakdown")
            
            step_names = {
                1: "📥 Data Gathering - Fundamentals",
                2: "📈 Data Gathering - Market Data",
                3: "💰 Value Agent Analysis",
                4: "📊 Growth/Momentum Agent Analysis",
                5: "🌍 Macro Regime Agent Analysis",
                6: "⚠️ Risk Agent Analysis",
                7: "💭 Sentiment Agent Analysis",
                8: "⚖️ Score Blending",
                9: "✅ Client Layer Validation",
                10: "🎯 Final Analysis"
            }
            
            if all_stats:
                for step in sorted(all_stats.keys()):
                    stats = all_stats[step]
                    name = step_names.get(step, f"Step {step}")
                    
                    with st.expander(f"**{name}**", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Samples", stats['count'])
                            st.metric("Average", f"{stats['avg']:.2f}s")
                        
                        with col2:
                            st.metric("Median", f"{stats['median']:.2f}s")
                            st.metric("Std Dev", f"{stats['std_dev']:.2f}s")
                        
                        with col3:
                            st.metric("Minimum", f"{stats['min']:.2f}s")
                            st.metric("25th %ile", f"{stats['p25']:.2f}s")
                        
                        with col4:
                            st.metric("Maximum", f"{stats['max']:.2f}s")
                            st.metric("75th %ile", f"{stats['p75']:.2f}s")
                        
                        # Visualization
                        if step in manager.step_times and len(manager.step_times[step]) > 1:
                            import pandas as pd
                            import plotly.graph_objects as go
                            
                            times = manager.step_times[step]
                            
                            # Create histogram
                            fig = go.Figure(data=[go.Histogram(x=times, nbinsx=20)])
                            fig.update_layout(
                                title=f"Distribution of {name} Times",
                                xaxis_title="Duration (seconds)",
                                yaxis_title="Frequency",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Overall timing chart
                st.markdown("---")
                st.subheader("📈 Overall Step Timing Comparison")
                
                import pandas as pd
                import plotly.graph_objects as go
                
                step_data = []
                for step in sorted(all_stats.keys()):
                    stats = all_stats[step]
                    step_data.append({
                        'Step': f"Step {step}",
                        'Name': step_names.get(step, f"Step {step}").split(' ', 1)[1] if step in step_names else f"Step {step}",
                        'Average': stats['avg'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'P25': stats['p25'],
                        'P75': stats['p75']
                    })
                
                df = pd.DataFrame(step_data)
                
                # Box plot style chart
                fig = go.Figure()
                
                for idx, row in df.iterrows():
                    fig.add_trace(go.Box(
                        x=[row['Name']],
                        q1=[row['P25']],
                        median=[row['Average']],
                        q3=[row['P75']],
                        lowerfence=[row['Min']],
                        upperfence=[row['Max']],
                        name=row['Step']
                    ))
                
                fig.update_layout(
                    title="Step Duration Distribution (Box Plot)",
                    xaxis_title="Analysis Step",
                    yaxis_title="Duration (seconds)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("📊 No timing data collected yet. Run some analyses to build the timing model!")
            
            st.markdown("---")
            
            # Raw data download
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Export Timing Data**")
                st.caption("Download raw timing data for external analysis")
            
            with col2:
                if st.button("📥 Download JSON"):
                    import json
                    timing_export = {
                        'step_times': {str(k): v for k, v in manager.step_times.items()},
                        'step_stats': {str(k): v for k, v in manager.step_stats.items()},
                        'exported_at': datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="💾 Save Timing Data",
                        data=json.dumps(timing_export, indent=2),
                        file_name=f"step_timing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # Force save button
            if st.button("💾 Force Save Timing Data to Disk"):
                manager.force_save()
                st.success("✅ Timing data saved to disk!")
        
        else:
            st.warning("⚠️ Step Time Manager not initialized. Please restart the application.")



def update_google_sheets_qa_analyses(analysis_archive: dict, show_price_ui: bool = False) -> bool:
    """
    Update Google Sheets with QA analyses.
    Uses specific column order matching user's format.
    
    Args:
        analysis_archive: Dictionary of analyses by ticker
        show_price_ui: If True, shows UI for fetching current prices (for manual exports)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        sheets_integration = st.session_state.sheets_integration
        
        if not sheets_integration or not sheets_integration.sheet:
            return False
        
        import math
        import pandas as pd
        import time
        
        def safe_float(value, decimals=2):
            """Keep numeric values as numbers, handling NaN and Infinity."""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return None
                # Round to specified decimals but keep as number
                return round(value, decimals)
            # Try to convert string to float
            try:
                num = float(value)
                if math.isnan(num) or math.isinf(num):
                    return None
                return round(num, decimals)
            except (ValueError, TypeError):
                return None
        
        def safe_value(value):
            """Safely convert text values, keeping None for missing data."""
            if value is None:
                return None
            if isinstance(value, str):
                return value if value.strip() else None
            return str(value)
        
        # ENHANCED: Get current prices using ALL Polygon.io features for maximum speed and coverage
        def get_bulk_prices_polygon(tickers):
            """
            Fetch current prices for multiple tickers using Polygon.io with multiple strategies.
            Leverages: Snapshot API, Aggregates API, Reference Data, and parallel processing.
            With unlimited API calls, we maximize speed and coverage.
            """
            import requests
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from datetime import datetime, timedelta
            
            polygon_key = os.getenv('POLYGON_API_KEY')
            if not polygon_key:
                logger.warning("Polygon API key not found, skipping price fetch")
                return {}
            
            prices = {}
            failed_tickers = []
            
            # STRATEGY 1: Snapshot API (fastest for bulk - gets all tickers at once)
            def fetch_all_snapshots():
                """Fetch ALL ticker snapshots at once."""
                all_prices = {}
                try:
                    # Get all tickers snapshot in ONE call
                    url = f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey={polygon_key}'
                    response = requests.get(url, timeout=20)
                    data = response.json()
                    
                    if data.get('status') == 'OK' and data.get('tickers'):
                        ticker_map = {t.upper(): t for t in tickers}  # Handle case sensitivity
                        
                        for ticker_data in data['tickers']:
                            ticker_symbol = ticker_data.get('ticker', '').upper()
                            
                            # Only process tickers we're interested in
                            if ticker_symbol in ticker_map:
                                price = None
                                
                                # Priority 1: Use day close (most recent intraday close)
                                if ticker_data.get('day') and ticker_data['day'].get('c'):
                                    price = float(ticker_data['day']['c'])
                                
                                # Priority 2: Use last trade price (real-time)
                                elif ticker_data.get('lastTrade') and ticker_data['lastTrade'].get('p'):
                                    price = float(ticker_data['lastTrade']['p'])
                                
                                # Priority 3: Use previous day close
                                elif ticker_data.get('prevDay') and ticker_data['prevDay'].get('c'):
                                    price = float(ticker_data['prevDay']['c'])
                                
                                # Priority 4: Use minute aggregates (most recent minute)
                                elif ticker_data.get('min') and ticker_data['min'].get('c'):
                                    price = float(ticker_data['min']['c'])
                                
                                if price and price > 0:
                                    all_prices[ticker_map[ticker_symbol]] = price
                                    logger.info(f"✅ {ticker_symbol}: ${price:.2f}")
                        
                        logger.info(f"Snapshot API fetched {len(all_prices)}/{len(tickers)} prices")
                
                except Exception as e:
                    logger.warning(f"Snapshot API error: {e}")
                
                return all_prices
            
            # STRATEGY 2: Individual ticker snapshots (for specific tickers)
            def fetch_ticker_snapshot(ticker):
                """Fetch snapshot for a single ticker."""
                try:
                    url = f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?apiKey={polygon_key}'
                    response = requests.get(url, timeout=10)
                    data = response.json()
                    
                    if data.get('status') == 'OK' and data.get('ticker'):
                        ticker_data = data['ticker']
                        
                        # Try multiple price sources
                        if ticker_data.get('day') and ticker_data['day'].get('c'):
                            return float(ticker_data['day']['c'])
                        elif ticker_data.get('lastTrade') and ticker_data['lastTrade'].get('p'):
                            return float(ticker_data['lastTrade']['p'])
                        elif ticker_data.get('prevDay') and ticker_data['prevDay'].get('c'):
                            return float(ticker_data['prevDay']['c'])
                        elif ticker_data.get('min') and ticker_data['min'].get('c'):
                            return float(ticker_data['min']['c'])
                
                except Exception as e:
                    logger.debug(f"Snapshot failed for {ticker}: {e}")
                
                return None
            
            # STRATEGY 3: Aggregates (previous close - most reliable)
            def fetch_previous_close(ticker):
                """Fetch previous day close using Aggregates API."""
                try:
                    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={polygon_key}'
                    response = requests.get(url, timeout=10)
                    data = response.json()
                    
                    if data.get('status') == 'OK' and data.get('results') and len(data['results']) > 0:
                        return float(data['results'][0]['c'])
                
                except Exception as e:
                    logger.debug(f"Aggregates failed for {ticker}: {e}")
                
                return None
            
            # STRATEGY 4: Daily open close (today or last trading day)
            def fetch_daily_open_close(ticker):
                """Fetch today's open/close or last trading day."""
                try:
                    # Try today first
                    today = datetime.now().strftime('%Y-%m-%d')
                    url = f'https://api.polygon.io/v1/open-close/{ticker}/{today}?adjusted=true&apiKey={polygon_key}'
                    response = requests.get(url, timeout=10)
                    data = response.json()
                    
                    if data.get('status') == 'OK' and data.get('close'):
                        return float(data['close'])
                    
                    # Try yesterday if today not available
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    url = f'https://api.polygon.io/v1/open-close/{ticker}/{yesterday}?adjusted=true&apiKey={polygon_key}'
                    response = requests.get(url, timeout=10)
                    data = response.json()
                    
                    if data.get('status') == 'OK' and data.get('close'):
                        return float(data['close'])
                
                except Exception as e:
                    logger.debug(f"Daily open-close failed for {ticker}: {e}")
                
                return None
            
            # STEP 1: Try to get ALL prices at once using Snapshot API (fastest)
            logger.info("🚀 Fetching all prices using Snapshot API...")
            prices = fetch_all_snapshots()
            
            # STEP 2: For missing tickers, use parallel individual requests with fallback strategies
            missing_tickers = [t for t in tickers if t not in prices]
            
            if missing_tickers:
                logger.info(f"📊 Fetching {len(missing_tickers)} missing tickers using multiple strategies...")
                
                def fetch_with_fallback(ticker):
                    """Try multiple strategies to get price for a ticker."""
                    # Strategy 1: Individual snapshot
                    price = fetch_ticker_snapshot(ticker)
                    if price:
                        return ticker, price
                    
                    # Strategy 2: Previous close aggregates
                    price = fetch_previous_close(ticker)
                    if price:
                        return ticker, price
                    
                    # Strategy 3: Daily open-close
                    price = fetch_daily_open_close(ticker)
                    if price:
                        return ticker, price
                    
                    return ticker, None
                
                # Use parallel processing with unlimited API calls
                with ThreadPoolExecutor(max_workers=20) as executor:
                    future_to_ticker = {executor.submit(fetch_with_fallback, ticker): ticker for ticker in missing_tickers}
                    
                    for future in as_completed(future_to_ticker):
                        ticker, price = future.result()
                        if price and price > 0:
                            prices[ticker] = price
                            logger.info(f"✅ {ticker}: ${price:.2f}")
                        else:
                            failed_tickers.append(ticker)
            
            # Log results
            success_count = len(prices)
            total_count = len(tickers)
            logger.info(f"✅ Price fetch complete: {success_count}/{total_count} successful")
            
            if failed_tickers:
                logger.warning(f"⚠️ Failed to fetch prices for: {', '.join(failed_tickers)}")
            
            return prices
        
        # Fallback function for individual ticker price fetch (simplified - relies on get_bulk_prices_polygon)
        def get_single_price_polygon(ticker):
            """Fetch price for a single ticker - uses bulk function for consistency."""
            result = get_bulk_prices_polygon([ticker])
            return result.get(ticker, 0)
        
        # Prepare QA data
        qa_data = []
        unique_tickers = list(analysis_archive.keys())
        
        # Automatically fetch current prices
        ticker_prices = {}
        fetch_prices = True  # Always fetch prices by default
        
        # Show price fetching info UI if explicitly requested (for manual exports)
        if show_price_ui:
            # Determine which API to use
            has_polygon = bool(os.getenv('POLYGON_API_KEY'))
            
            if has_polygon:
                # Calculate time estimate for bulk API (MUCH faster!)
                num_batches = (len(unique_tickers) + 14) // 15  # 15 tickers per batch
                estimated_time = max(2, num_batches * 0.5)  # ~0.5s per batch with parallel requests
                api_source = "Polygon.io Snapshot API (Bulk + Parallel)"
            else:
                # Fallback to slower sequential fetch
                estimated_time = len(unique_tickers) * 0.6
                api_source = "Yahoo Finance (Sequential)"
            
            # Show info about price fetching (enabled by default)
            st.info(f"💡 **Fetching current prices from {api_source}** (Est. time: ~{int(estimated_time)}s)")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Price API:** {api_source}")
                if not has_polygon:
                    st.caption("⚠️ Using Yahoo Finance (slow, sequential). Set POLYGON_API_KEY for 10x faster bulk fetching!")
                else:
                    st.caption(f"✅ Using Polygon.io with Unlimited API Calls - Fetching {len(unique_tickers)} prices in parallel!")
            with col2:
                st.write(f"**Est. Time:** ~{int(estimated_time)}s")
            
            # Option to skip price fetching if desired
            skip_prices = st.checkbox(
                f"⏭️ Skip Price Fetching",
                value=False,
                help="Check this to skip fetching new prices. Will use last documented prices from previous exports when available."
            )
            if skip_prices:
                fetch_prices = False
        
        if fetch_prices:
            has_polygon = bool(os.getenv('POLYGON_API_KEY'))
            
            # Add progress indicator for price fetching
            price_progress = st.empty()
            price_status = st.empty()
            
            if has_polygon:
                # FAST PATH: Use Polygon Snapshot API with parallel requests
                price_status.text(f"🚀 Fetching {len(unique_tickers)} prices in parallel using Polygon Snapshot API...")
                
                import time
                start_time = time.time()
                ticker_prices = get_bulk_prices_polygon(unique_tickers)
                elapsed = time.time() - start_time
                
                # Fill in any missing prices with individual calls
                missing_tickers = [t for t in unique_tickers if t not in ticker_prices]
                if missing_tickers:
                    price_status.text(f"Fetching {len(missing_tickers)} remaining tickers individually...")
                    for ticker in missing_tickers:
                        ticker_prices[ticker] = get_single_price_polygon(ticker)
                
                price_status.text(f"✅ Fetched {len(ticker_prices)} prices in {elapsed:.1f}s (Polygon Bulk API)")
                price_progress.progress(1.0)
                
            else:
                # SLOW PATH: Fallback to yfinance sequential fetching
                import yfinance as yf
                for i, ticker in enumerate(unique_tickers):
                    price_status.text(f"Fetching prices... {i+1}/{len(unique_tickers)} ({ticker})")
                    try:
                        stock = yf.Ticker(ticker)
                        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice', 0)
                        ticker_prices[ticker] = current_price if current_price else 0
                        time.sleep(0.6)  # Rate limit for Yahoo
                    except Exception as e:
                        logger.warning(f"Error fetching {ticker}: {e}")
                        ticker_prices[ticker] = 0
                    price_progress.progress((i + 1) / len(unique_tickers))
                
                price_status.text(f"✅ Fetched prices for {len(unique_tickers)} tickers (Yahoo Finance)")
            
            time.sleep(1)
            price_status.empty()
            price_progress.empty()
        
        # Always include price columns for consistency
        include_price_columns = True
        
        # If we didn't fetch new prices, try to get last documented prices from existing sheet
        if not ticker_prices:
            try:
                # Try to get existing data from Google Sheets to preserve last known prices
                existing_worksheet = sheets_integration.sheet.worksheet("QA Analyses")
                existing_data = existing_worksheet.get_all_records()
                
                # Build a map of ticker -> last known price
                last_known_prices = {}
                for row in existing_data:
                    ticker_key = row.get('Ticker', '')
                    current_price_val = row.get('Current Price', 0)
                    if ticker_key and current_price_val and current_price_val != 0:
                        # Keep the most recent (last) price for each ticker
                        last_known_prices[ticker_key] = float(current_price_val)
                
                # Use last known prices when available
                for ticker in unique_tickers:
                    if ticker in last_known_prices:
                        ticker_prices[ticker] = last_known_prices[ticker]
                        
                if last_known_prices:
                    st.info(f"📈 Using last documented prices for {len(last_known_prices)} tickers")
                    logger.info(f"Retrieved last known prices for: {list(last_known_prices.keys())}")
                else:
                    st.info("ℹ️ No previous price data found - Current Price column will be empty")
                        
            except Exception as e:
                logger.warning(f"Could not retrieve last known prices from sheet: {e}")
        
        for ticker, analyses in analysis_archive.items():
            # Get current price from cache (use last known price if no new price fetched)
            current_price = ticker_prices.get(ticker, None)
            
            for analysis in analyses:
                # Extract fundamentals
                fundamentals = analysis.fundamentals if hasattr(analysis, 'fundamentals') and analysis.fundamentals else {}
                
                # Extract agent scores
                agent_scores = analysis.agent_scores if hasattr(analysis, 'agent_scores') and analysis.agent_scores else {}
                
                # Extract agent rationales
                agent_rationales = analysis.agent_rationales if hasattr(analysis, 'agent_rationales') and analysis.agent_rationales else {}
                
                # Calculate price change (use current price if available, otherwise leave blank)
                price_change_pct = None
                if current_price is not None and current_price > 0:
                    price_at_analysis = safe_float(analysis.price_at_analysis, 2) or 0
                    if price_at_analysis > 0:
                        price_change = current_price - price_at_analysis
                        price_change_pct = (price_change / price_at_analysis) * 100
                
                # Build row in exact order specified
                row = {
                    'Ticker': ticker,
                    'Recommendation': analysis.recommendation.value.upper(),
                    'Confidence Score': safe_float(analysis.confidence_score, 1),
                    'Price at Analysis': safe_float(analysis.price_at_analysis, 2),
                    'Price': safe_float(fundamentals.get('price', analysis.price_at_analysis), 2),
                    'Beta': safe_float(fundamentals.get('beta'), 2),
                    'EPS': safe_float(fundamentals.get('eps'), 2),
                    'Week 52 Low': safe_float(fundamentals.get('week_52_low'), 2),
                    'Week 52 High': safe_float(fundamentals.get('week_52_high'), 2),
                    'Is EFT?': safe_value(fundamentals.get('is_etf', 'No')),
                    'Market Cap': safe_float(fundamentals.get('market_cap'), 0),
                    'Summary': (safe_value(fundamentals.get('description', fundamentals.get('name', 'N/A'))) or 'N/A')[:500],
                    'Value Agent Score': safe_float(agent_scores.get('value_agent'), 1),
                    'Growth Momentum Agent Score': safe_float(agent_scores.get('growth_momentum_agent'), 1),
                    'Macro Regime Agent Score': safe_float(agent_scores.get('macro_regime_agent'), 1),
                    'Risk Agent Score': safe_float(agent_scores.get('risk_agent'), 1),
                    'Sentiment Agent Score': safe_float(agent_scores.get('sentiment_agent'), 1),
                    'Client Layer Agent Score': safe_float(agent_scores.get('client_layer_agent'), 1),
                    'Learning Agent Score': safe_float(agent_scores.get('learning_agent'), 1),
                    'Sector': safe_value(fundamentals.get('sector', 'N/A')),
                    'Pe Ratio': safe_float(fundamentals.get('pe_ratio'), 2),
                    'Dividend Yield': safe_float(fundamentals.get('dividend_yield'), 4),
                    'Data Sources': safe_value(', '.join(fundamentals.get('data_sources') or []) if fundamentals.get('data_sources') else 'N/A'),
                    'Key Metrics': safe_value(fundamentals.get('key_metrics', 'N/A')),
                    'Risk Assessment': safe_value(fundamentals.get('risk_assessment', 'N/A')),
                    'Perplexity Analysis': (safe_value(fundamentals.get('perplexity_analysis', 'N/A')) or 'N/A')[:500],
                    'Value Agent Rationale': ' '.join(str(agent_rationales.get('value_agent', 'N/A')).split())[:1000],
                    'Growth Momentum Agent Rationale': ' '.join(str(agent_rationales.get('growth_momentum_agent', 'N/A')).split())[:1000],
                    'Macro Regime Agent Rationale': ' '.join(str(agent_rationales.get('macro_regime_agent', 'N/A')).split())[:1000],
                    'Risk Agent Rationale': ' '.join(str(agent_rationales.get('risk_agent', 'N/A')).split())[:1000],
                    'Sentiment Agent Rationale': ' '.join(str(agent_rationales.get('sentiment_agent', 'N/A')).split())[:1000],
                    'Client Layer Agent Rationale': ' '.join(str(agent_rationales.get('client_layer_agent', 'N/A')).split())[:1000],
                    'Learning Agent Rationale': ' '.join(str(agent_rationales.get('learning_agent', 'N/A')).split())[:1000],
                    'Analysis Date': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Timestamp': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Source': safe_value(fundamentals.get('source', 'N/A')),
                    'Polygon Data': safe_value(fundamentals.get('polygon_data', 'N/A'))
                }
                
                # Always add price columns for consistency
                row['Current Price'] = safe_float(current_price, 2) if current_price is not None else None
                row['Price Change %'] = safe_float(price_change_pct, 2) if price_change_pct is not None else None
                
                qa_data.append(row)
        
        # Create DataFrame with exact column order (always include price columns for consistency)
        column_order = ['Ticker', 'Recommendation', 'Confidence Score', 'Price at Analysis', 'Current Price', 'Price Change %']
        
        column_order.extend([
            'Beta', 'EPS', 'Week 52 Low', 'Week 52 High', 'Is EFT?', 'Market Cap',
            'Value Agent Score', 'Growth Momentum Agent Score', 'Macro Regime Agent Score',
            'Risk Agent Score', 'Sentiment Agent Score', 'Client Layer Agent Score',
            'Summary', 'Learning Agent Score',
            'Sector', 'Pe Ratio', 'Dividend Yield',
            'Perplexity Analysis',
            'Value Agent Rationale', 'Growth Momentum Agent Rationale', 'Macro Regime Agent Rationale',
            'Risk Agent Rationale', 'Sentiment Agent Rationale', 'Client Layer Agent Rationale',
            'Learning Agent Rationale',
            'Analysis Date', 'Export Date', 'Timestamp', 'Source',
            'Data Sources', 'Key Metrics', 'Risk Assessment', 'Polygon Data'
        ])
        
        # Update QA worksheet with ordered data
        return sheets_integration.update_qa_analyses(qa_data, column_order=column_order)
        
    except Exception as e:
        st.error(f"Google Sheets QA update error: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
