# Wharton Investing Challenge - Multi-Agent Investment Analysis System

A sophisticated multi-agent system for investment analysis and portfolio construction, built for the Wharton Investing Challenge.

## 🎯 Overview

This system provides **decision support only** - it produces data, scores, weights, and charts, but **never generates prose for submissions**. All recommendations include:
- Per-agent scores with rationales
- IPS compliance validation
- Explainable scoring methodology
- Full AI usage disclosure for Works Cited

## 🏗️ Architecture

The system uses 7 specialized agents:

1. **Value Agent** - Analyzes valuation metrics (P/E, EV/EBIT, FCF yield, dividends)
2. **Growth/Momentum Agent** - Tracks earnings growth and price momentum
3. **Macro/Regime Agent** - Classifies economic environment and sector tilts
4. **Risk Agent** - Monitors volatility, beta, and diversification
5. **Sentiment Agent** - Analyzes news and narrative risks
6. **Client Layer Agent** - Validates IPS compliance (final gatekeeper)
7. **Learning/QA Agent** - Evaluates performance and tunes parameters

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project directory
cd /Users/arjansingh/Wharton

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add:

```env
# REQUIRED
OPENAI_API_KEY=sk-...
ALPHA_VANTAGE_API_KEY=your_key_here

# OPTIONAL
NEWS_API_KEY=your_key_here
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (free)
- NewsAPI: https://newsapi.org/register (100 req/day free)

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Usage Modes

### 1️⃣ Stock Analysis
Evaluate a specific stock:
1. Select "Stock Analysis" from sidebar
2. Enter ticker (e.g., AAPL)
3. Click "Analyze Stock"
4. View agent scores, rationales, and IPS compliance
5. Download results as CSV

### 2️⃣ Portfolio Recommendations
Generate optimal portfolio:
1. Select "Portfolio Recommendations"
2. Choose universe (S&P 100 or custom tickers)
3. Set target number of positions
4. Click "Generate Portfolio"
5. View holdings, sector allocation, and scores
6. Download portfolio as CSV

### 3️⃣ Backtest
Test strategy performance:
1. Select "Backtest"
2. Set date range (1 year maximum recommended)
3. Click "Run Backtest"
4. View performance metrics vs S&P 500
5. Download results as CSV

### 4️⃣ Configuration
Customize settings:
- **IPS Tab**: Update client profile, risk tolerance, position limits
- **Agent Weights Tab**: Adjust how much each agent influences scores

### 5️⃣ Disclosure Log
Track AI usage for Works Cited:
- View total API calls, tokens, and costs
- Download complete disclosure log
- Required for competition transparency

## 🌐 Enhanced Data Reliability

The system includes **Enhanced Data Provider** with multiple fallbacks to eliminate API rate limiting issues:

### Automatic Fallback Chain:
1. **Premium APIs** (if configured) - IEX Cloud, Polygon.io
2. **Alpha Vantage** with smart retry logic  
3. **yfinance** with multiple user agents and retry
4. **Stale cache** (up to 3 days old)
5. **Synthetic data** (emergency fallback with warnings)

### Premium Setup (Optional but Recommended):
```bash
# Run the premium setup helper
python setup_premium.py

# Or manually add to .env:
IEX_TOKEN=your_iex_token_here          # $9/month - highly recommended
POLYGON_API_KEY=your_polygon_key_here  # $99/month - professional grade
```

**Recommended for production: ~$60/month (IEX + Alpha Vantage Premium)**

### Monitoring:
- **🌐 Data Status** page in Streamlit app
- Real-time API usage tracking
- Cache management tools
- Data source testing

## ⚙️ Configuration Files

### `config/ips.yaml` - Investment Policy Statement
Defines client constraints:
- Risk tolerance
- Position/sector limits
- Exclusions
- Beta bands
- Cash buffer

### `config/model.yaml` - Agent Configuration
Controls agent behavior:
- **Agent Weights**: How much each agent influences final score
- **Thresholds**: Scoring parameters per agent
- **Backtest Settings**: Walk-forward parameters, costs

### `config/universe.yaml` - Stock Universe
Defines investment universe (default: S&P 100)

## 📊 Output Files

All outputs are saved with timestamps for easy tracking:

```
outputs/
├── backtests/          # Backtest results
│   └── backtest_YYYYMMDD_HHMMSS.csv
├── portfolios/         # Portfolio recommendations  
│   └── portfolio_YYYYMMDD.csv
└── analyses/           # Individual stock analyses
    └── TICKER_analysis_YYYYMMDD.csv

logs/
├── app_YYYYMMDD.log                # Application logs
└── ai_disclosure_YYYYMMDD.jsonl    # AI usage for Works Cited

data/
├── cache/              # Cached API responses (auto-managed)
└── history/            # Decision history for learning
    ├── decision_history.jsonl
    └── qa_report_YYYYMMDD_HHMMSS.json
```

## 🎓 For Competition Disclosure

**AI/Tool Usage:**
This system uses AI/tools for analysis (NOT for writing submissions):

1. **OpenAI GPT-4o-mini** - Agent reasoning and rationale generation
2. **yfinance** - Market price data
3. **Alpha Vantage** - Fundamental data, macroeconomic indicators, news
4. **NewsAPI** - News headlines for sentiment analysis

**All usage is logged** in `logs/ai_disclosure_YYYYMMDD.jsonl` with:
- Timestamp
- Tool/model used
- Purpose
- Input/output summaries
- Token counts and costs

**Include in Works Cited:**
```
OpenAI GPT-4o-mini API. Used for multi-agent investment analysis 
reasoning and rationale generation. [DATE]. Total tokens: [X], 
Cost: $[Y]. See disclosure log for details.
```

## 🧪 Backtest Hygiene

The system enforces proper backtesting practices:

✅ **No Look-Ahead Bias**
- Only point-in-time data used
- Walk-forward validation with embargo period
- Fundamentals aligned by filing date

✅ **Trading Costs**
- Commissions and slippage applied (configurable in `ips.yaml`)
- Reports both gross and net returns

✅ **Universe Definition**
- Survivorship-bias-free universe per test date
- Dynamic rebalancing based on configured frequency

## 🔧 Advanced Customization

### Adjust Agent Weights
Edit `config/model.yaml`:

```yaml
agent_weights:
  value: 1.5              # Increase for value focus
  growth_momentum: 1.2    # Increase for growth focus
  macro_regime: 0.8       # Decrease if less macro-aware
  risk: 1.0
  sentiment: 0.5          # Decrease if less news-reactive
```

### Modify Scoring Logic
Each agent is in `agents/` directory:
- `value_agent.py` - Valuation metrics
- `growth_momentum_agent.py` - Growth & momentum
- `macro_regime_agent.py` - Macro analysis
- `risk_agent.py` - Risk metrics
- `sentiment_agent.py` - News sentiment
- `client_layer_agent.py` - IPS validation

### Add Custom Constraints
Edit `config/ips.yaml` to add:
- Excluded sectors/tickers
- ESG screens
- Geographic restrictions
- Liquidity requirements

## 📚 Project Structure

```
/Wharton/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # API key template
├── README.md                  # This file
│
├── config/                    # Configuration files
│   ├── ips.yaml              # Investment Policy Statement
│   ├── model.yaml            # Agent weights & parameters
│   └── universe.yaml         # Stock universe definition
│
├── agents/                    # Agent modules
│   ├── base_agent.py         # Abstract base class
│   ├── value_agent.py        # Value analysis
│   ├── growth_momentum_agent.py
│   ├── macro_regime_agent.py
│   ├── risk_agent.py
│   ├── sentiment_agent.py
│   ├── client_layer_agent.py # IPS validation
│   └── learning_agent.py     # QA & parameter tuning
│
├── engine/                    # Orchestration & backtesting
│   ├── portfolio_orchestrator.py  # Coordinates all agents
│   └── backtest.py           # Walk-forward backtesting
│
├── data/                      # Data layer
│   ├── data_provider.py      # Unified API interface
│   └── cache/                # Cached API responses
│
├── utils/                     # Utilities
│   ├── config_loader.py      # Config management
│   └── logger.py             # Logging & disclosure
│
├── logs/                      # Log files (auto-created)
└── outputs/                   # Results (auto-created)
```

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Ensure `.env` file exists in project root
- Check API key is correctly set in `.env`
- Restart the application after editing `.env`

### "Alpha Vantage API limit reached"
- Free tier: 5 calls/minute, 500/day
- Wait a few minutes or upgrade to premium
- System will fall back to yfinance when possible

### "No data available for ticker"
- Check ticker symbol is correct (use AAPL not Apple)
- Try a more recent analysis date
- Some stocks may have incomplete fundamental data

### Slow backtest performance
- Reduce date range (1 year recommended)
- Use smaller universe (custom tickers instead of S&P 100)
- Check internet connection (data fetching intensive)

## 📞 Support

For issues or questions:
1. Check this README first
2. Review configuration files in `config/`
3. Check logs in `logs/` directory
4. Verify API keys are valid and have remaining quota

## 📄 License

This system is built for educational purposes as part of the Wharton Investing Challenge.

---

**Remember:** This system provides **decision support only**. All written submissions must be your own work. Use the data, scores, and charts as inputs to your analysis, and always disclose AI/tool usage in your Works Cited section.

Good luck with the competition! 🎓📈
