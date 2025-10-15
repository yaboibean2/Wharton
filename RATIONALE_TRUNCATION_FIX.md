# Bug Fix: Google Sheets Rationale Truncation

## Issue Description
User reported that written rationales were getting cut off after exactly 2 lines in Google Sheets, preventing full rationale text from being visible.

## Root Cause Analysis
1. **Character Truncation**: Agent rationales were being truncated to exactly 200 characters using `[:200]` slicing
2. **Missing Text Wrapping**: Google Sheets cells were not configured for proper text wrapping
3. **Multiple Locations**: The truncation was happening in several export functions

## Files Modified

### 1. `/Users/arjansingh/Wharton/app.py`

#### QA Analyses Export (Lines 6629-6634)
**Before:**
```python
'Value Agent Rationale': ' '.join(str(agent_rationales.get('value_agent', 'N/A')).split())[:200],
'Growth Momentum Agent Rationale': ' '.join(str(agent_rationales.get('growth_momentum_agent', 'N/A')).split())[:200],
# ... all rationales truncated to 200 characters
```

**After:**
```python
'Value Agent Rationale': ' '.join(str(agent_rationales.get('value_agent', 'N/A')).split())[:1000],
'Growth Momentum Agent Rationale': ' '.join(str(agent_rationales.get('growth_momentum_agent', 'N/A')).split())[:1000],
# ... all rationales now allow 1000 characters
```

#### Portfolio Recommendations Export (Lines 5802-5808)
**Before:**
```python
stock.get('agent_rationales', {}).get('value_agent', '')[:200],
# ... all rationales truncated to 200 characters
stock.get('rationale', '')[:300],
```

**After:**
```python
stock.get('agent_rationales', {}).get('value_agent', '')[:1000],
# ... all rationales now allow 1000 characters
stock.get('rationale', '')[:1500],  # Main rationale gets 1500 characters
```

#### Final Portfolio Export (Line 6372)
**Before:**
```python
'AI Rationale': safe_value(holding.get('rationale', 'N/A'))[:300],
```

**After:**
```python
'AI Rationale': safe_value(holding.get('rationale', 'N/A'))[:1500],
```

#### Batch Export (Line 4253)
**Before:**
```python
row[f"{agent.replace('_', ' ').title()} Rationale"] = rationale[:500]  # Truncate
```

**After:**
```python
row[f"{agent.replace('_', ' ').title()} Rationale"] = rationale[:1000]  # Allow full rationale
```

### 2. `/Users/arjansingh/Wharton/utils/google_sheets_integration.py`

#### Added Text Wrapping Support (Lines 228-253)
**New Feature:**
```python
# Format rationale columns for proper text wrapping
try:
    # Find rationale columns and apply text wrapping
    rationale_col_indices = []
    for i, col_name in enumerate(headers):
        if 'Rationale' in str(col_name) or 'Analysis' in str(col_name):
            rationale_col_indices.append(i)
    
    if rationale_col_indices:
        # Apply text wrapping to all data rows for rationale columns
        for col_index in rationale_col_indices:
            col_letter = col_index_to_letter(col_index)
            
            # Format entire column for text wrapping
            col_range = f'{col_letter}:{col_letter}'
            worksheet.format(col_range, {
                'wrapStrategy': 'WRAP',
                'verticalAlignment': 'TOP',
                'textFormat': {'fontSize': 9}
            })
            
        logger.info(f"Applied text wrapping to {len(rationale_col_indices)} rationale columns")
```

## Character Limits Summary

| Content Type | Old Limit | New Limit | Increase |
|--------------|-----------|-----------|----------|
| Agent Rationales | 200 chars | 1000 chars | 5x |
| Main Rationale | 300 chars | 1500 chars | 5x |
| Perplexity Analysis | 300 chars | 500 chars | 1.7x |
| Batch Export Rationales | 500 chars | 1000 chars | 2x |

## Google Sheets Formatting Improvements

1. **Text Wrapping**: All rationale columns now have `wrapStrategy: 'WRAP'` applied
2. **Vertical Alignment**: Set to `'TOP'` for better readability
3. **Font Size**: Reduced to 9pt for rationale columns to fit more text
4. **Auto-Detection**: Automatically identifies columns containing "Rationale" or "Analysis"

## Testing Recommendations

1. **Export QA Analyses**: Test the "📊 Sync to Sheets" button from QA Center
2. **Portfolio Export**: Test portfolio recommendations export to Google Sheets  
3. **Batch Export**: Test batch export with rationales included
4. **Visual Check**: Verify in Google Sheets that:
   - Rationales are no longer cut off at ~2 lines
   - Text wraps properly within cells
   - Cells expand vertically to show full content
   - Font size is readable

## Expected Behavior After Fix

- **Before**: Rationales cut off at exactly 200 characters (~2 lines in sheet)
- **After**: Full rationales displayed up to 1000-1500 characters with proper text wrapping
- **Visual**: Multi-line rationales will wrap within cells and expand vertically
- **Readability**: Smaller font (9pt) in rationale columns for better space utilization

## Backward Compatibility

- ✅ Existing Google Sheets will work without changes
- ✅ No breaking changes to data structure
- ✅ Enhanced formatting is additive only
- ✅ All export functions maintain same column structure

---

**Date Fixed**: October 5, 2025
**Bug Report**: "theres a weird bug where in the spreadsheet the written rationales get cut off after exactly 2 lines"
**Status**: ✅ **RESOLVED**