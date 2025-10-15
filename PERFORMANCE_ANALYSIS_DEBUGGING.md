# Performance Analysis: Enhanced Debugging & Fixes

## Changes Made

### 1. Added Comprehensive Debug Logging

The system now shows **exactly** what data it's reading from Google Sheets:

```
📊 RAW DATA FROM SHEETS (first 20): AAPL: -18.5%, NVDA: +24.3%, TSLA: +18.1%, ...
```

### 2. Enhanced Data Parsing

**Better handling of edge cases:**
- Empty strings
- "N/A" values
- "nan" strings
- Missing values
- Malformed percentages

**Before:** Silently failed or treated empty as 0
**After:** Logs warnings and skips invalid data

### 3. Added Configurable Threshold

You can now adjust the minimum movement threshold from the UI:
- Default: 15%
- Range: 1% to 50%
- Shows in real-time how many stocks qualify

### 4. Detailed Per-Stock Logging

For each stock, you'll see:
```
✅ NVDA: +24.50% - QUALIFIED (threshold=15.0%)
⚪ MSFT: +8.20% (below 15.0% threshold)
```

### 5. Analysis Summary

At the end, you get a comprehensive summary:
```
🎯 ANALYSIS COMPLETE: Identified 12 significant movements from 150 stocks (threshold: 15.0%)
✅ Top movers: INTC (-22.10%), NVDA (+24.50%), TSLA (+18.20%), AAPL (-16.80%), MSFT (+15.30%)
```

## How to Debug "No Movements Found"

### Step 1: Check What Data Is Being Read

Look for this log line:
```
📊 RAW DATA FROM SHEETS (first 20): AAPL: -18.5%, NVDA: +24.3%, ...
```

**Problem Indicators:**
- `AAPL: N/A` → Percent Change column has no data
- `AAPL: 0` → All stocks showing 0% change
- `AAPL: ` (empty) → Column exists but no values
- No log line at all → Google Sheets not being read

### Step 2: Check Column Names

Look for:
```
Columns available: ['Ticker', 'Percent Change', 'Price at Analysis', 'Price', 'Sector', 'Market Cap']
```

**Required:** At least one of:
- "Percent Change" (preferred)
- "Price Change %"
- "% Change"
- OR both "Price at Analysis" AND "Price"

### Step 3: Check Individual Stock Processing

Enable debug mode to see:
```
AAPL: Parsed '-18.5%' from Percent Change -> -18.5%
AAPL: abs_change=18.5%, min_threshold=15.0%
✅ AAPL: -18.50% - QUALIFIED (threshold=15.0%)
```

**Problem Indicators:**
- `AAPL: Skipping empty/invalid value` → Bad data format
- `AAPL: Failed to parse` → Wrong data type
- `⚪ AAPL: +12.00% (below threshold)` → Valid but doesn't meet threshold

### Step 4: Check Summary

Look for:
```
⚠️ NO MOVEMENTS FOUND meeting 15.0% threshold
   - Analyzed 150 stocks from Google Sheets
   - Try lowering threshold or check if 'Percent Change' column has valid data
```

## Common Issues & Solutions

### Issue 1: "No significant movements" but I see them in sheets

**Possible Causes:**
1. **Wrong format:** Percent Change is text like "down 5%" instead of "-5%"
   - **Fix:** Format as number or "-5.0%"

2. **Wrong scale:** Values are 0.15 instead of 15.0
   - **Fix:** Multiply by 100 (15% not 0.15)

3. **Empty cells:** Some cells are empty
   - **Fix:** Fill with 0 or remove empty rows

4. **Wrong worksheet:** Reading from wrong tab
   - **Fix:** Rename your worksheet to "Historical Price Analysis"

### Issue 2: All stocks show 0% change

**Possible Causes:**
1. **Formula error:** Google Sheets formula isn't calculating
   - **Fix:** Check formula in "Percent Change" column

2. **Price data missing:** "Price at Analysis" or "Price" columns empty
   - **Fix:** Ensure both columns have valid numbers

3. **Date mismatch:** Analysis date is same as current date
   - **Fix:** Wait for prices to update or manually input

### Issue 3: Only getting a few stocks when there are more

**Possible Causes:**
1. **Threshold too high:** 15% is high - only extreme movements qualify
   - **Fix:** Lower threshold to 10% or 5% using the slider

2. **Some invalid data:** Most stocks have bad data, only a few parse correctly
   - **Fix:** Check logs for parsing errors

### Issue 4: Can't see debug logs

**Enable in UI:**
1. Go to Performance Analysis tab
2. Expand "🔧 Advanced Options"
3. Check "Enable debug logging"
4. Look in terminal/logs where Streamlit is running

## Data Format Examples

### ✅ GOOD Formats

**Percent Change column:**
```
-18.5%
+24.3%
-18.5
24.3
```

**Price columns:**
```
Price at Analysis: 150.25
Price: 128.50
→ Calculates: -14.5%
```

### ❌ BAD Formats

```
down 18.5%          → Won't parse (contains text)
-18.5 percent       → Won't parse (contains text)
(18.5%)             → Won't parse (contains parentheses)
-$18.50             → Won't parse (contains $ sign)
N/A                 → Skipped
[empty]             → Skipped
```

## Testing Steps

### 1. Lower the Threshold
Set threshold to **5%** to see if ANY stocks are detected:
- If yes → Your data is good, just no stocks moved >15%
- If no → Data format issue

### 2. Check Raw Data
Look at terminal logs for:
```
📊 RAW DATA FROM SHEETS (first 20): ...
```
This shows exactly what's in your sheet.

### 3. Try Sample Data
Add a test row to your Google Sheet:
```
Ticker: TEST
Percent Change: -50.0%
```

If TEST doesn't show up, there's a reading issue.

### 4. Check Worksheet Name
The code looks for:
1. "Historical Price Analysis" (first)
2. "Portfolio Analysis" (second)
3. "Price Analysis" (third)
4. First worksheet (fallback)

Rename your worksheet to one of these names.

## Advanced Debugging

### Enable Python Logging

In your terminal where Streamlit runs:
```bash
export LOG_LEVEL=DEBUG
streamlit run app.py
```

### Check Logs for Patterns

**Good pattern:**
```
✅ Found worksheet: 'Historical Price Analysis'
✅ Fetched 150 rows from Google Sheets
Processing 150 rows with min_threshold=15.0%
AAPL: Parsed '-18.5%' from Percent Change -> -18.5%
✅ AAPL: -18.50% - QUALIFIED
🎯 ANALYSIS COMPLETE: Identified 12 significant movements
```

**Bad pattern:**
```
⚠️ No sheets_integration or sheet attribute
⚠️ NO MOVEMENTS FOUND meeting 15.0% threshold
```

## UI Changes

### New Advanced Options Panel

Located below the date selector:

**Debug Mode:**
- Shows detailed per-stock processing
- Displays raw data from sheets
- Shows why stocks were included/excluded

**Custom Threshold:**
- Slider from 1% to 50%
- Default: 15%
- Real-time preview of threshold impact

## Files Modified

1. **utils/performance_analysis_engine.py**
   - Enhanced logging throughout
   - Better error handling for edge cases
   - Raw data display
   - Per-stock debug output
   - Summary statistics

2. **app.py**
   - Added debug mode checkbox
   - Added threshold slider
   - Pass custom threshold to engine

## What You'll See Now

### In the UI:
```
🔧 Advanced Options
  ✓ Enable debug logging
  Minimum movement threshold: 15.0%
```

### In the Logs:
```
Starting performance analysis from 2024-10-01 to 2024-10-14
Analysis mode: ALL stocks with ≥15.0% movement
✅ Found worksheet: 'Historical Price Analysis'
✅ Fetched 150 rows from Google Sheets
Columns available: ['Ticker', 'Percent Change', 'Price at Analysis', 'Price', 'Sector', 'Market Cap']
Sample row: {'Ticker': 'AAPL', 'Percent Change': '-18.5%', ...}
📊 RAW DATA FROM SHEETS (first 20): AAPL: -18.5%, NVDA: +24.3%, TSLA: +18.1%, INTC: -22.0%, ...
Processing 150 rows with min_threshold=15.0%
✅ AAPL: -18.50% - QUALIFIED (threshold=15.0%)
✅ NVDA: +24.30% - QUALIFIED (threshold=15.0%)
✅ TSLA: +18.10% - QUALIFIED (threshold=15.0%)
⚪ MSFT: +8.20% (below 15.0% threshold)
...
🎯 ANALYSIS COMPLETE: Identified 12 significant movements from 150 stocks (threshold: 15.0%)
✅ Top movers: INTC (-22.10%), NVDA (+24.30%), TSLA (+18.10%), AAPL (-18.50%), ...
```

## Next Steps

1. **Restart Streamlit** (cache was cleared)
2. **Open Performance Analysis tab**
3. **Enable debug logging** in Advanced Options
4. **Set threshold to 5%** temporarily to see if any stocks are detected
5. **Run analysis**
6. **Check terminal logs** for the detailed output above
7. **Share the log output** if still not working

The logs will now tell you exactly what's happening at each step!

---
**Status**: ✅ **DEBUGGING ENHANCED**
**Date**: October 14, 2025
**Action Required**: Restart Streamlit and check logs
