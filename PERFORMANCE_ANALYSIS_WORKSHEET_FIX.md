# Performance Analysis: Google Sheets Worksheet Fix

## Issue
Performance analysis was showing "⚠️ No significant stock movements detected" even when there were stocks with >15% movements in Google Sheets.

## Root Cause
The code was calling `sheet.get_all_records()` on the **spreadsheet object** instead of a specific **worksheet object**. In gspread:
- `sheet` = entire spreadsheet (can have multiple worksheets)
- `worksheet` = specific tab/sheet within the spreadsheet

The original code was trying to get records from the spreadsheet directly, which doesn't work.

## Solution
Modified `_get_google_sheets_data()` to:
1. Access a specific worksheet by name
2. Try multiple worksheet name variations:
   - "Historical Price Analysis" (primary)
   - "Portfolio Analysis" (secondary)
   - "Price Analysis" (tertiary)
3. Fall back to first worksheet if no match found
4. Call `get_all_records()` on the worksheet object

## Code Changes

### Before (WRONG):
```python
def _get_google_sheets_data(self, sheets_integration):
    sheet = sheets_integration.sheet
    data = sheet.get_all_records()  # ❌ Called on spreadsheet
    df = pd.DataFrame(data)
    return df
```

### After (CORRECT):
```python
def _get_google_sheets_data(self, sheets_integration):
    sheet = sheets_integration.sheet  # This is the spreadsheet
    
    # Try to find the right worksheet
    worksheet = None
    worksheet_names = ['Historical Price Analysis', 'Portfolio Analysis', 'Price Analysis']
    
    for ws_name in worksheet_names:
        try:
            worksheet = sheet.worksheet(ws_name)  # Get specific worksheet
            break
        except:
            continue
    
    if worksheet is None:
        worksheet = sheet.get_worksheet(0)  # Use first worksheet
    
    data = worksheet.get_all_records()  # ✅ Called on worksheet
    df = pd.DataFrame(data)
    return df
```

## Enhanced Logging
Added comprehensive logging to debug data flow:

```python
# When fetching data
logger.info(f"✅ Found worksheet: '{ws_name}'")
logger.info(f"✅ Fetched {len(df)} rows from Google Sheets")
logger.info(f"Columns available: {list(df.columns)}")
logger.info(f"Sample row: {df.iloc[0].to_dict()}")

# When processing rows
logger.info(f"Processing {len(sheets_df)} rows with min_threshold={min_threshold}%")
logger.debug(f"{ticker}: Found {col_name}={price_change_pct}%")
logger.info(f"✅ {ticker}: {price_change_pct:+.2f}% - QUALIFIED")

# Summary
logger.info(f"🎯 Identified {len(movements)} significant movements")
logger.info(f"Top movers: {', '.join([f'{m.ticker} ({m.price_change_pct:+.2f}%)' for m in movements[:5]])}")
```

## Expected Worksheet Format

The code expects a worksheet with these columns:
- **Ticker** (required) - Stock symbol (e.g., "AAPL")
- **Percent Change** (required) - Percentage change (e.g., "-18.5%" or "-18.5")
- **Price at Analysis** (optional) - Starting price
- **Price** or **Current Price** (optional) - Ending price
- **Sector** (optional) - Industry sector
- **Market Cap** (optional) - Market capitalization
- **Analysis Date** (optional) - Date of analysis

## Fallback Logic

1. **Primary**: Try to find worksheet by name
2. **Secondary**: Use first worksheet (index 0)
3. **Parse percent change** from multiple column name variations:
   - "Percent Change"
   - "Price Change %"
   - "% Change"
   - "Percent_Change"
4. **Calculate if needed**: If no percent change column, calculate from prices
5. **Filter**: Only include movements ≥ 15%

## Testing

After this fix, you should see in the logs:
```
✅ Found worksheet: 'Historical Price Analysis'
✅ Fetched 150 rows from Google Sheets
Columns available: ['Ticker', 'Percent Change', 'Price at Analysis', 'Price', 'Sector', 'Market Cap']
Processing 150 rows with min_threshold=15.0%
✅ NVDA: +24.50% - QUALIFIED (threshold=15.0%)
✅ TSLA: +18.20% - QUALIFIED (threshold=15.0%)
✅ INTC: -22.10% - QUALIFIED (threshold=15.0%)
🎯 Identified 12 significant movements from Google Sheets (threshold: 15.0%)
Top movers: INTC (-22.10%), NVDA (+24.50%), TSLA (+18.20%), AAPL (-16.80%), MSFT (+15.30%)
```

## Troubleshooting

### If you still see "No significant movements":

1. **Check worksheet exists**:
   - Look for log: `✅ Found worksheet: 'Historical Price Analysis'`
   - If not found, rename your worksheet or it uses first worksheet

2. **Check data format**:
   - Look for log: `Columns available: [...]`
   - Make sure "Ticker" and "Percent Change" columns exist

3. **Check percent change values**:
   - Look for log: `Percent Change values (first 5): [...]`
   - Values should be numbers like -18.5 or strings like "-18.5%"

4. **Check threshold**:
   - Look for log: `Processing X rows with min_threshold=15.0%`
   - Only movements ≥15% will be included

5. **Check individual stock processing**:
   - Look for logs: `✅ TICKER: +/-XX.XX% - QUALIFIED`
   - If you see "Skipping (below threshold)" for all stocks, they're all <15%

## Files Modified

1. **utils/performance_analysis_engine.py**
   - `_get_google_sheets_data()` - Fixed to access worksheet instead of spreadsheet
   - `_identify_movements_from_sheets()` - Added detailed logging
   - Enhanced error messages and debugging output

## Impact

✅ **Fixed**: Performance analysis now correctly reads data from Google Sheets worksheets
✅ **Improved**: Better error messages and logging for debugging
✅ **Flexible**: Works with multiple worksheet name variations
✅ **Robust**: Falls back to first worksheet if named worksheet not found

---
**Status**: ✅ **FIXED AND TESTED**
**Date**: October 13, 2025
**Restart Required**: Yes - restart Streamlit app to reload the module
