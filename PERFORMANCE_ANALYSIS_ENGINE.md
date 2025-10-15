# Performance Analysis Engine - Complete Documentation

## Overview

The **Performance Analysis Engine** is a sophisticated learning system that continuously improves the investment analysis model by:

1. **Identifying stocks that moved significantly** (both up and down)
2. **Fetching and analyzing news articles** and events that may have driven the movements
3. **Using AI to determine root causes** of why stocks moved the way they did
4. **Generating specific, actionable recommendations** for improving the model
5. **Tracking implementation and impact** of recommendations over time

## Key Features

### 🔍 Comprehensive Movement Analysis
- Automatically detects stocks with significant price movements (>5% threshold)
- Classifies movements as "significant", "major", or "extreme"
- Tracks volume changes alongside price movements
- Analyzes both gainers and losers

### 📰 News Integration
- Fetches relevant news articles from Polygon.io News API
- Captures breaking news, earnings releases, and major events
- Analyzes article sentiment and relevance

### 🤖 AI-Powered Root Cause Analysis
- Uses OpenAI GPT-4 or Perplexity AI to analyze movement drivers
- Identifies whether movements were:
  - **Earnings-related**: Driven by quarterly reports or guidance
  - **News-driven**: Triggered by breaking news or announcements
  - **Market-driven**: Following broader market trends
  - **Sector-driven**: Part of sector rotation
  - **Fundamental changes**: Due to business model shifts
  - **Technical breakouts**: Driven by chart patterns

### 💡 Model Improvement Recommendations
Generates specific, prioritized recommendations:

- **Critical Priority**: Immediate action required (e.g., risk control issues)
- **High Priority**: Important improvements (e.g., agent weight adjustments)
- **Medium Priority**: Beneficial enhancements (e.g., new data sources)
- **Low Priority**: Nice-to-have improvements

Each recommendation includes:
- Specific change to implement
- Detailed rationale with supporting evidence
- Expected impact on model performance
- Step-by-step implementation guide
- Affected agents and confidence score

### 📊 Pattern Recognition
Identifies patterns across multiple movements:
- Frequency of earnings-driven moves
- News sensitivity patterns
- Sector rotation trends
- Market correlation analysis

## How It Works

### Step 1: Movement Detection
```python
movements = engine.analyze_performance_period(
    start_date="2025-10-01",
    end_date="2025-10-13",
    tickers=tracked_tickers,
    qa_system=qa_system
)
```

The engine analyzes all tracked stocks and identifies those with significant price changes.

### Step 2: News Fetching
For each significant movement, the engine fetches relevant news articles from the period.

### Step 3: AI Analysis
The AI analyzes:
- Price movement data
- News articles
- Fundamental data
- Market context

And determines:
- Primary root causes
- Which agents should have caught this
- What the model missed

### Step 4: Recommendation Generation
Based on patterns across all analyzed movements, the engine generates specific recommendations:

**Example Recommendation:**
```
Priority: HIGH
Category: agent_weight
Change: Increase sentiment agent weight by 0.2x
Rationale: 65% of movements were news-driven. Sentiment agent should have stronger influence.
Expected Impact: Faster reaction to breaking news; better capture of sentiment shifts
Confidence: 80%

Implementation Steps:
1. Increase sentiment agent weight from 1.0 to 1.2
2. Add real-time news monitoring for tracked stocks
3. Implement breaking news alerts
4. Add social media sentiment tracking

Affected Agents: sentiment
```

### Step 5: Tracking & Improvement
- Mark recommendations as implemented
- Track performance before/after implementation
- Monitor improvement metrics over time

## Usage Guide

### Accessing the Feature

1. Navigate to **QA & Learning Center** in the sidebar
2. Click on the **"🔬 Performance Analysis"** tab
3. Configure your analysis period
4. Click **"🚀 Run Performance Analysis"**

### Configuration Options

**Time Period Presets:**
- Last 7 Days
- Last 14 Days
- Last 30 Days
- Last 90 Days
- Custom Range

**Analysis Options:**
- Analyze all tracked stocks
- Focus on specific sectors
- Filter by movement magnitude

### Viewing Results

The analysis report includes:

1. **Executive Summary**: Key findings at a glance
2. **Top Gainers**: Stocks with largest price increases
3. **Top Losers**: Stocks with largest price decreases
4. **Root Cause Analysis**: Why each stock moved
5. **Model Recommendations**: Specific improvements to implement
6. **Pattern Analysis**: Trends across all movements

### Acting on Recommendations

1. Review each recommendation
2. Assess priority and confidence
3. Review implementation steps
4. Click "Mark as Implemented" when done
5. Monitor impact over time

## Technical Implementation

### Architecture

```
utils/performance_analysis_engine.py
├── PerformanceAnalysisEngine (main class)
├── StockMovement (data structure)
├── NewsArticle (data structure)
├── MovementAnalysis (data structure)
└── ModelAdjustmentRecommendation (data structure)
```

### Key Methods

**analyze_performance_period()**
- Main entry point
- Orchestrates entire analysis workflow
- Returns comprehensive report

**_identify_significant_movements()**
- Scans tracked stocks for price movements
- Applies threshold filters
- Classifies magnitude

**_fetch_news_for_stock()**
- Integrates with Polygon.io News API
- Fetches articles for date range
- Parses and structures news data

**_analyze_movement()**
- Deep-dive into individual stock movements
- Combines price, news, and fundamental data
- Generates MovementAnalysis object

**_ai_analyze_root_causes()**
- Uses GPT-4 or Perplexity for analysis
- Generates structured JSON response
- Identifies patterns and gaps

**_generate_model_recommendations()**
- Analyzes patterns across movements
- Generates prioritized recommendations
- Creates implementation guides

### Data Storage

All analysis results are stored in:
```
data/performance_analysis/
├── significant_movements.json      # All detected movements
├── movement_analyses.json          # Detailed analyses
├── model_recommendations.json      # Generated recommendations
└── improvement_tracking.json       # Implementation tracking
```

## Configuration

### API Requirements

**Required:**
- Polygon.io API key (for news and price data)
- OpenAI API key OR Perplexity API key (for AI analysis)

**Optional:**
- Additional news APIs for broader coverage

### Thresholds

Default movement thresholds:
```python
movement_thresholds = {
    'significant': 5.0,   # 5% move
    'major': 10.0,        # 10% move
    'extreme': 20.0       # 20% move
}
```

Customize in `PerformanceAnalysisEngine.__init__()`

## Example Workflow

### Weekly Analysis Routine

1. **Monday Morning**: Run analysis for previous week
```
Period: Last 7 Days
Tickers: All tracked (from QA system)
```

2. **Review Results**:
- Check executive summary
- Identify top movers
- Read root cause analyses

3. **Prioritize Recommendations**:
- Focus on CRITICAL and HIGH priority items
- Review supporting evidence
- Assess feasibility

4. **Implement Changes**:
- Update agent weights
- Add new data sources
- Adjust thresholds
- Mark as implemented

5. **Monitor Impact**:
- Track performance metrics
- Compare before/after results
- Refine as needed

### Monthly Deep Dive

1. Run 90-day analysis
2. Identify long-term patterns
3. Make strategic model adjustments
4. Document lessons learned

## Best Practices

### 1. Regular Analysis
- Run weekly analyses consistently
- Don't wait for problems - be proactive
- Track trends over time

### 2. Action-Oriented
- Don't just review - implement recommendations
- Start with high-priority items
- Document what you implement

### 3. Evidence-Based
- Review supporting evidence carefully
- Validate recommendations before implementing
- Test changes on small scale first

### 4. Continuous Improvement
- Track which recommendations worked
- Iterate based on results
- Build knowledge base over time

### 5. Balance Automation & Judgment
- Let AI identify patterns
- Use human judgment for implementation
- Don't blindly follow all recommendations

## Troubleshooting

### "No significant movements found"
- Expand date range
- Lower movement thresholds
- Ensure stocks are being tracked

### "Empty response from AI"
- Check API keys
- Verify rate limits
- Check API client initialization

### "Could not fetch news"
- Verify Polygon.io API key
- Check date range format
- Ensure ticker symbols are valid

### "No recommendations generated"
- Need more data points (>10 movements)
- Patterns may not be strong enough
- Check confidence thresholds

## Performance Considerations

### Speed
- News fetching can take time (10+ stocks)
- AI analysis is rate-limited
- Cache results for faster review

### Costs
- Each AI analysis call costs $0.01-0.05
- News API has rate limits
- Budget accordingly for large analyses

### Accuracy
- AI analysis has 70-90% confidence typically
- Validate critical recommendations
- Cross-reference with multiple sources

## Future Enhancements

### Planned Features
- [ ] Automated weekly analysis scheduling
- [ ] Email/Slack notifications for critical findings
- [ ] Advanced pattern recognition (ML-based)
- [ ] Backtesting recommendation impact
- [ ] Integration with trading execution
- [ ] Sentiment tracking from social media
- [ ] Earnings calendar integration
- [ ] Competitive analysis features

### Community Contributions
We welcome contributions! Focus areas:
- Additional news sources
- Better pattern recognition
- UI improvements
- Documentation enhancements

## Conclusion

The Performance Analysis Engine transforms reactive investing into proactive learning. By systematically analyzing what happened and why, the model continuously improves and adapts to market conditions.

**Key Benefits:**
- ✅ Learn from every market movement
- ✅ Identify model blind spots automatically
- ✅ Get specific, actionable improvements
- ✅ Track progress over time
- ✅ Stay ahead of market changes

Start using it today to make your investment analysis system smarter every week!
