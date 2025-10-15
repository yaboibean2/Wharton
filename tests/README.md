# Test Scripts

This folder contains test scripts for validating various components of the Wharton Investment Analysis System.

## Available Tests

### test_polygon.py
**Purpose:** Verify Polygon.io API connectivity and price fetching

**Usage:**
```bash
python test_polygon.py
```

**What it tests:**
- Polygon.io API key validity
- Price fetching for 5 major tickers (AAPL, MSFT, GOOGL, TSLA, NVDA)
- Response format and data structure
- Success rate

**Expected Output:**
```
================================================================================
POLYGON.IO API TEST
================================================================================
✅ API Key found: jz_jRfDLJN...
Testing with 5 tickers: AAPL, MSFT, GOOGL, TSLA, NVDA

Fetching AAPL...
  Status Code: 200
  ✅ Success!
     Price: $258.02
     Volume: 49,155,614.0
     Timestamp: 1759521600000

... (results for other tickers)

================================================================================
RESULTS: 5/5 successful
================================================================================
✅ All tests passed! Polygon.io API is working correctly.
```

---

### test_ai_portfolio_system.py
**Purpose:** Test the AI portfolio generation system

**Usage:**
```bash
python test_ai_portfolio_system.py
```

**What it tests:**
- Portfolio orchestrator initialization
- Agent analysis execution
- Score blending logic
- Recommendation generation
- Performance metrics

---

### test_custom_weights.py
**Purpose:** Test custom agent weight configurations

**Usage:**
```bash
python test_custom_weights.py
```

**What it tests:**
- Custom weight application
- Agent score calculation
- Weight normalization
- Client profile integration

---

## Running All Tests

To run all tests sequentially:
```bash
cd /Users/arjansingh/Wharton/tests
python test_polygon.py
python test_ai_portfolio_system.py
python test_custom_weights.py
```

## Prerequisites

Before running tests, ensure:
1. Virtual environment is activated: `source .venv/bin/activate`
2. All dependencies are installed: `pip install -r ../requirements.txt`
3. Environment variables are set in `../.env`
4. API keys are valid

## Troubleshooting

**Test fails with "No API key found":**
- Check that `.env` file exists in the parent directory
- Verify the API key is set correctly
- Ensure `.env` is loaded properly

**Test fails with "Connection error":**
- Check internet connectivity
- Verify API endpoint is accessible
- Check for rate limiting

**Test fails with "Import error":**
- Ensure virtual environment is activated
- Run `pip install -r ../requirements.txt`
- Check Python version (requires 3.8+)

## Adding New Tests

To add a new test:
1. Create a new file: `test_<feature_name>.py`
2. Import required modules
3. Write test functions
4. Add clear output messages
5. Update this README

## Note

These tests are for development and debugging purposes. They are not automated unit tests and should be run manually when needed.
