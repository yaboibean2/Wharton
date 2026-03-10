# API Configuration Guide

Total Insights Investing uses multiple data sources with a tiered fallback system.
You do **not** need all of these — the app will gracefully degrade to whichever sources are available.

---

## Quick Start (minimum viable setup)

The only API keys **required** to run the app are:

| Key | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | Powers all 5 agent rationales (required) |
| `PERPLEXITY_API_KEY` | Fallback data source + sentiment news (required) |

Everything else is optional but recommended. **yfinance requires no API key** and is the primary data source for stock metrics and price history.

---

## All API Keys

Add these to your `.env` file in the project root (see `.env.example`).

### Tier 1: Yahoo Finance via yfinance (FREE, no key needed)

- **Key needed:** None
- **What it provides:** Stock price, P/E, EPS, beta, market cap, sector, industry, dividend yield, 52-week range, enterprise value, profit margin, EV/EBITDA, FCF yield, earnings/revenue growth, price history
- **Rate limits:** No formal API key limits. Uses Yahoo Finance web scraping under the hood. Avoid excessive concurrent requests.
- **Setup:** Installed automatically via `pip install -r requirements.txt`
- **Notes:** This is now the **primary data source** for all structured financial metrics. No LLM parsing involved — returns deterministic structured data.

### Tier 1: FRED API (FREE, key needed)

- **Key needed:** `FRED_API_KEY`
- **What it provides:** Yield curve slope (10Y-2Y spread), CPI/inflation YoY, unemployment rate, fed funds rate
- **Rate limits:** 120 requests/minute (very generous)
- **How to get the key:**
  1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
  2. Create a free account
  3. Request an API key (instant approval)
  4. Add `FRED_API_KEY=your_key_here` to your `.env`
- **Notes:** Powers the Macro Regime Agent with real economic data instead of hardcoded defaults. Data is cached for 24 hours. Highly recommended.

### Tier 2: Polygon.io

- **Key needed:** `POLYGON_API_KEY`
- **What it provides:** Backup stock prices, company details, financial statements, 52-week range
- **Rate limits:** Free tier: 5 calls/minute. Paid tiers are faster.
- **How to get the key:**
  1. Go to https://polygon.io/
  2. Sign up for a free account
  3. Copy your API key from the dashboard
  4. Add `POLYGON_API_KEY=your_key_here` to your `.env`
- **Notes:** Secondary data source behind yfinance. Useful as a fallback if Yahoo Finance has issues.

### Tier 3: Perplexity AI

- **Key needed:** `PERPLEXITY_API_KEY`
- **What it provides:** Real-time financial news/sentiment, qualitative analysis, fallback for any metrics yfinance+Polygon both miss
- **Rate limits:** Depends on plan (free tier ~20 req/min)
- **How to get the key:**
  1. Go to https://www.perplexity.ai/settings/api
  2. Sign up and generate an API key
  3. Add `PERPLEXITY_API_KEY=your_key_here` to your `.env`
- **Notes:** Previously the primary data source. Now used as a **last-resort fallback** for structured metrics (only if yfinance AND Polygon both fail). Still used by the Sentiment Agent for real-time news context.

### Required: OpenAI

- **Key needed:** `OPENAI_API_KEY`
- **What it provides:** Agent rationale generation (all 5 agents), portfolio AI selection
- **Models used:** gpt-4o-mini (agents), o3 (portfolio selector)
- **How to get the key:**
  1. Go to https://platform.openai.com/api-keys
  2. Create an API key
  3. Add `OPENAI_API_KEY=your_key_here` to your `.env`
- **Notes:** Required for the app to function. All agent scoring rationales are generated via OpenAI.

### Optional: Google Gemini

- **Key needed:** `GEMINI_API_KEY`
- **What it provides:** Second opinion for portfolio AI selection (used alongside OpenAI)
- **How to get the key:**
  1. Go to https://aistudio.google.com/app/apikey
  2. Generate an API key
  3. Add `GEMINI_API_KEY=your_key_here` to your `.env`
- **Notes:** Optional. If not configured, portfolio selection uses only OpenAI.

### Optional: NewsAPI

- **Key needed:** `NEWS_API_KEY`
- **What it provides:** Additional news headlines for sentiment analysis
- **Rate limits:** Free tier: 100 requests/day
- **How to get the key:**
  1. Go to https://newsapi.org/register
  2. Sign up for free (Developer plan)
  3. Copy your API key
  4. Add `NEWS_API_KEY=your_key_here` to your `.env`
- **Notes:** Supplements Perplexity news. Not required — sentiment agent works without it.

### Optional: Alpha Vantage

- **Key needed:** `ALPHA_VANTAGE_API_KEY`
- **What it provides:** Backup news sentiment, basic market data
- **Rate limits:** Free tier: 25 requests/day (very limited)
- **How to get the key:**
  1. Go to https://www.alphavantage.co/support/#api-key
  2. Claim a free API key
  3. Add `ALPHA_VANTAGE_API_KEY=your_key_here` to your `.env`
- **Notes:** Lowest priority fallback. The free tier is very limited. Not recommended unless you have a paid plan.

---

## Data Source Priority (Fallback Chain)

### Stock Metrics (price, P/E, beta, EPS, market cap, etc.)
```
yfinance (free) → Polygon.io → Perplexity AI (LLM fallback)
```
Perplexity is only invoked if yfinance AND Polygon both fail to provide critical fields.

### Price History (daily OHLCV)
```
Cache (fresh) → yfinance (free) → Polygon.io → Alpha Vantage → Cache (stale) → Synthetic
```

### Macro Economic Data (yield curve, inflation, unemployment)
```
FRED API (free) → Hardcoded estimates
```

### News / Sentiment
```
Perplexity AI (sonar-pro) → NewsAPI → Alpha Vantage News → Template fallback
```

---

## Example `.env` File

```bash
# Required
OPENAI_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...

# Recommended (free)
FRED_API_KEY=your_fred_key_here

# Optional (free tiers available)
POLYGON_API_KEY=your_polygon_key_here
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_av_key_here
GEMINI_API_KEY=your_gemini_key_here
```

## Verifying Your Setup

Run the API test suite to verify connectivity:

```bash
python tests/test_apis.py
```

This will test each configured API and report which ones are working.
