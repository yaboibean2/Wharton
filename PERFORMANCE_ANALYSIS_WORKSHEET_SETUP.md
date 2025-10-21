# Performance Analysis: Worksheet Configuration Fix

## THE PROBLEM

The error shows:
```
Using first worksheet: 'Year by year'
ERROR: the header row in the worksheet contains duplicates: ['']
```

**Root Cause**: The system is reading the WRONG worksheet ("Year by year" instead of "Historical Price Analysis"), and that worksheet has empty/duplicate column headers.

## THE FIX

### 1. Create the Correct Worksheet

You need a worksheet named one of these (in priority order):
1. **"Historical Price Analysis"** ← RECOMMENDED
2. "Portfolio Analysis"
3. "Price Analysis"

### 2. Required Columns

Your worksheet MUST have:

**Option A (Recommended):**
- `Ticker` - Stock symbol (e.g., AAPL, NVDA, TSLA)
- `Percent Change` - Movement since analysis (e.g., 58.97, 50.97, -18.5)

**Option B (Alternative):**
- `Ticker` - Stock symbol
- `Price at Analysis` - Starting price (e.g., 100.50)
- `Price` - Current price (e.g., 159.47)

### 3. Example Worksheet Setup

Create a new worksheet named "Historical Price Analysis" with this structure:

| Ticker | Percent Change | Price at Analysis | Price | Sector | Market Cap |
|--------|---------------|-------------------|-------|---------|------------|
| AAPL   | 58.97         | 100.25            | 159.47| Tech    | 2.5T       |
| NVDA   | 50.97         | 450.30            | 679.76| Tech    | 1.8T       |
| TSLA   | 45.08         | 225.50            | 327.17| Auto    | 800B       |
| INTC   | -22.15        | 45.80             | 35.66 | Tech    | 150B       |

**Important:**
- Column names must be in the FIRST row
- No empty column headers
- No duplicate column headers
- Data starts in row 2

### 4. What You'll See After Fix

When you run the analysis, you should see:
```
✅ Found worksheet: 'Historical Price Analysis'
✅ Fetched 150 rows from Google Sheets
📋 Columns available: ['Ticker', 'Percent Change', 'Price at Analysis', 'Price', 'Sector', 'Market Cap']
📊 Data source: Percent Change column

📊 RAW DATA: AAPL: 58.97 (type: float), NVDA: 50.97 (type: float), ...

Processing 150 rows with min_threshold=15.0%
✅ AAPL: +58.97% - QUALIFIED
✅ NVDA: +50.97% - QUALIFIED
✅ TSLA: +45.08% - QUALIFIED

🎯 ANALYSIS COMPLETE: Identified 13 significant movements
```

## Common Issues

### Issue: "Year by year" worksheet being read

**Problem**: The system can't find "Historical Price Analysis", so it's trying to use the first worksheet.

**Solution**: 
1. Create a new worksheet
2. Name it exactly: **Historical Price Analysis**
3. Or rename your current working worksheet to this name

### Issue: "Duplicate header" error

**Problem**: Your worksheet has:
- Empty column headers (blank cells in row 1)
- Duplicate column names

**Solution**:
1. Go to row 1 of your worksheet
2. Delete any empty columns
3. Ensure every column has a unique name
4. Remove any spaces from end of column names

### Issue: "Missing required 'Ticker' column"

**Problem**: First row doesn't have a "Ticker" column

**Solution**:
- Add "Ticker" as the first column header
- Ensure spelling is exact (capital T)

### Issue: "Missing required data columns"

**Problem**: No percent change or price columns

**Solution**:
Add either:
- Column: "Percent Change"
- OR Columns: "Price at Analysis" + "Price"

## Quick Fix Steps

1. **Open your Google Sheet**

2. **Check worksheets** - Do you have one named "Historical Price Analysis"?
   - YES → Skip to step 4
   - NO → Continue to step 3

3. **Create/Rename worksheet**
   - Option A: Rename existing worksheet to "Historical Price Analysis"
   - Option B: Create new worksheet named "Historical Price Analysis"

4. **Check row 1** - Are all column headers filled and unique?
   - Remove any blank columns
   - Ensure no duplicates
   - Must have: "Ticker" and either "Percent Change" OR ("Price at Analysis" + "Price")

5. **Check your data**
   - Row 1: Headers
   - Row 2+: Your stock data
   - Ticker column: Stock symbols (AAPL, NVDA, etc.)
   - Percent Change column: Numbers like 58.97, 50.97, -18.5

6. **Save** and go back to the app

7. **Run analysis** - You should now see stocks detected

## What the App Will Show

When you click "Run Performance Analysis", you'll see:
```
📊 Available worksheets: Sheet1, Historical Price Analysis, Year by year
✅ Found required worksheet: Historical Price Analysis
🔍 Scanning all stocks in Google Sheets for movements ≥15.0%...
```

If there's still a problem, it will show:
```
❌ Missing required worksheet!
Please create a worksheet named one of: Historical Price Analysis, Portfolio Analysis, Price Analysis
Current worksheets: Sheet1, Year by year
```

## Verification

After creating the worksheet, the logs should show:
1. ✅ Worksheet found
2. ✅ Data fetched
3. ✅ Columns validated
4. 📊 Raw data displayed
5. ✅ Stocks parsed
6. ✅ Movements detected

## Files Modified

1. **utils/performance_analysis_engine.py**
   - Better worksheet detection
   - Duplicate header error handling
   - Column validation
   - Clear error messages

2. **app.py**
   - Shows available worksheets before running
   - Validates worksheet exists
   - Shows helpful error messages

---

**Next Steps:**
1. Create/rename worksheet to "Historical Price Analysis"
2. Ensure first row has: Ticker, Percent Change (and/or prices)
3. No empty columns in row 1
4. Restart Streamlit
5. Try analysis again

Your 58%, 50%, 45% movers should now be detected! 🎯
