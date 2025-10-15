# Performance Analysis: Debug Fixes for Data Detection

## Issues Fixed

### 1. "Bad message format / SessionInfo not initialized"
This is a Streamlit initialization timing issue - harmless but annoying. It occurs when the app tries to use session state before it's fully initialized.

### 2. Not Detecting High-Percentage Movements
**Problem**: Stocks with 58%, 50%, 45% movements weren't being detected.

**Root Causes Identified:**
1. **Numeric values not handled**: Google Sheets might be returning numbers (58.97) not strings ("58.97%")
2. **Column name whitespace**: Column names might have leading/trailing spaces
3. **Insufficient logging**: Hard to see what was actually being parsed

## Changes Made

### 1. Handle Numeric Values Directly
**Before:** Only handled string values with conversion
```python
val = str(raw_value).strip().replace('%', '')
price_change_pct = float(val)
```

**After:** Check if value is already a number first
```python
# If it's already a number, use it directly
if isinstance(raw_value, (int, float)):
    price_change_pct = float(raw_value)
    logger.info(f"{ticker}: Direct numeric value: {price_change_pct}%")
    break

# Otherwise parse as string
val = str(raw_value).strip().replace('%', '')
price_change_pct = float(val)
```

### 2. Clean Column Names
**Added automatic column name cleaning:**
```python
# Strip whitespace from all column names
df.columns = df.columns.str.strip()
```

This handles:
- `"Percent Change "` → `"Percent Change"`
- `" Percent Change"` → `"Percent Change"`
- `" Percent Change "` → `"Percent Change"`

### 3. Enhanced Logging

**Show exact column names:**
```python
🔍 EXACT COLUMN NAMES: ['Ticker', 'Percent Change', 'Price at Analysis', ...]
🔍 'Percent Change' in columns: True
```

**Show data types:**
```python
📊 RAW DATA (first 20): AAPL: 58.97 (type: float), NVDA: 50.97 (type: float), ...
```

**Show parsing attempts:**
```python
AAPL: Checking Percent Change='58.97' (type: float)
AAPL: Direct numeric value from Percent Change: 58.97%
AAPL: ✓ Successfully parsed percent change: 58.97%
```

**Show failures:**
```python
TICKER: ❌ No percent change data available after trying all methods - SKIPPED
   Raw 'Percent Change' value: N/A
   Available columns: ['Ticker', 'Percent Change', 'Price', ...]
```

### 4. Better Error Handling

**Skip None explicitly:**
```python
if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ''):
    continue
```

**Handle more exception types:**
```python
except (ValueError, AttributeError, TypeError) as e:
    logger.warning(f"Failed to parse: {e}")
```

## Expected Behavior Now

### When Everything Works:
```
✅ Fetched 150 rows from Google Sheets
📋 Columns available (cleaned): ['Ticker', 'Percent Change', 'Price at Analysis', 'Price', ...]
🔍 EXACT COLUMN NAMES: ['Ticker', 'Percent Change', 'Price at Analysis', 'Price', ...]
🔍 'Percent Change' in columns: True
📊 RAW DATA (first 20): AAPL: 58.97 (type: float), NVDA: 50.97 (type: float), TSLA: 45.08 (type: float), ...

Processing 150 rows with min_threshold=15.0%

AAPL: Checking Percent Change='58.97' (type: float)
AAPL: Direct numeric value from Percent Change: 58.97%
AAPL: ✓ Successfully parsed percent change: 58.97%
✅ AAPL: +58.97% - QUALIFIED (threshold=15.0%)

NVDA: Checking Percent Change='50.97' (type: float)
NVDA: Direct numeric value from Percent Change: 50.97%
NVDA: ✓ Successfully parsed percent change: 50.97%
✅ NVDA: +50.97% - QUALIFIED (threshold=15.0%)

...

🎯 ANALYSIS COMPLETE: Identified 13 significant movements from 150 stocks (threshold: 15.0%)
✅ Top movers: AAPL (+58.97%), NVDA (+50.97%), TSLA (+45.08%), ...
```

### When There's a Problem:
```
TICKER: Checking Percent Change='N/A' (type: str)
TICKER: Skipping empty/invalid value in Percent Change: 'N/A'
TICKER: ❌ No percent change data available after trying all methods - SKIPPED
   Raw 'Percent Change' value: N/A
   Available columns: ['Ticker', 'Percent Change', 'Price at Analysis', 'Price']
```

## Data Format Support

### Now Supports ALL These Formats:

**Numeric (NEW):**
- `58.97` ✅ (as number)
- `50.97` ✅ (as number)
- `-18.5` ✅ (as number)

**String:**
- `"58.97%"` ✅
- `"58.97"` ✅
- `"-18.5%"` ✅
- `"-18.5"` ✅

**With formatting:**
- `"58.97 %"` ✅ (strips spaces)
- `"58.97%"` ✅

## How to Test

1. **Restart Streamlit** (cache cleared)

2. **Run Performance Analysis** with:
   - Threshold set to **5%** (to catch everything)
   - Debug logging enabled

3. **Check Terminal Logs** for:
   ```
   📊 RAW DATA (first 20): ...
   ```
   This shows EXACTLY what values are in your sheet

4. **Look for parsing messages:**
   ```
   TICKER: Direct numeric value from Percent Change: 58.97%
   TICKER: ✓ Successfully parsed percent change: 58.97%
   ```

5. **Check qualification:**
   ```
   ✅ TICKER: +58.97% - QUALIFIED (threshold=5.0%)
   ```

## If Still Not Working

If you still don't see movements, the logs will now show:

1. **Exact column names** - verify "Percent Change" exists
2. **Raw data with types** - verify values are there
3. **Parsing attempts** - see which column it's trying
4. **Failure reasons** - see exactly why it failed

Share those log lines and we can diagnose further!

## Files Modified

**utils/performance_analysis_engine.py:**
- Added numeric value handling (direct float/int support)
- Added column name cleaning (strip whitespace)
- Enhanced logging at every step
- Better error messages
- More robust type checking

---
**Status**: ✅ **READY TO TEST**
**Date**: October 14, 2025
**Action**: Restart Streamlit and run analysis with debug logging enabled
