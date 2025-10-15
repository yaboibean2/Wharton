# System Architecture & Flow Diagrams

## 📊 Complete System Pipeline Visualizations

This document provides detailed Mermaid.js diagrams showing exactly what happens in each mode of the investment analysis system.

---

## 🎯 Mode 1: Pre-Determined Stock Analysis

### High-Level Flow

```mermaid
graph TB
    Start([User Enters Stock Ticker]) --> Validate{Valid?}
    Validate -->|No| Error[Display Error]
    Validate -->|Yes| Init[Initialize System]
    
    Init --> LoadConfig[Load Configuration]
    LoadConfig --> FetchData[Fetch Stock Data]
    
    FetchData --> PriceData[Price History]
    FetchData --> Fundamentals[Fundamentals]
    FetchData --> News[News & Sentiment]
    FetchData --> Analyst[Analyst Coverage]
    
    PriceData --> CalcTech[Calculate Technical Indicators]
    Fundamentals --> CalcFund[Calculate Fundamental Metrics]
    
    CalcTech --> Agents[Multi-Agent Analysis]
    CalcFund --> Agents
    News --> Agents
    Analyst --> Agents
    
    Agents --> ValueAgent[Value Agent]
    Agents --> GrowthAgent[Growth Agent]
    Agents --> SentimentAgent[Sentiment Agent]
    Agents --> MacroAgent[Macro Agent]
    Agents --> RiskAgent[Risk Agent]
    
    ValueAgent --> Orchestrator[Portfolio Orchestrator]
    GrowthAgent --> Orchestrator
    SentimentAgent --> Orchestrator
    MacroAgent --> Orchestrator
    RiskAgent --> Orchestrator
    
    Orchestrator --> Composite[Calculate Composite Score]
    Composite --> IPS[Check IPS Alignment]
    IPS --> Final[Final Recommendation]
    
    Final --> Report[Generate Report]
    Report --> Save[Save to QA System]
    Save --> Display[Display in UI]
    
    style Start fill:#e1f5e1
    style Display fill:#e1f5e1
    style Orchestrator fill:#d1ecf1
    style Agents fill:#d1ecf1
```

### Detailed Component Breakdown

**Data Fetching (5-10 seconds)**
- Price History: YFinance OHLCV data (5 years)
- Fundamentals: Company info, sector, market cap, financial metrics
- News: Perplexity AI (7-14 days) or NewsAPI fallback
- Analyst Coverage: Recommendations, target prices, ratings

**Agent Analysis (5-8 seconds)**
- **Value Agent**: P/E, P/B, P/S ratios, PEG, DCF estimates
- **Growth Agent**: Revenue/earnings growth, price momentum, volume trends
- **Sentiment Agent**: News polarity, analyst consensus, price targets
- **Macro Agent**: Sector performance, market cycle, economic indicators
- **Risk Agent**: Volatility (Beta, Std Dev), drawdown, Sharpe/Sortino ratios

**Scoring & Output (2-3 seconds)**
- Weighted composite score from all agents
- IPS alignment check (time horizon, risk tolerance, tax efficiency)
- Final recommendation: STRONG BUY / BUY / HOLD / SELL / STRONG SELL

---

## 🎲 Mode 2: Portfolio Generation

### High-Level Flow

```mermaid
graph TB
    Start([User Clicks Generate]) --> Params[Get Parameters]
    Params --> Validate{Valid?}
    Validate -->|No| Error[Display Error]
    Validate -->|Yes| Universe[Load Stock Universe]
    
    Universe --> Filter[Apply Filters]
    Filter --> Parallel[Parallel Data Fetch]
    
    Parallel --> BulkPrice[Bulk Prices]
    Parallel --> BulkFund[Bulk Fundamentals]
    Parallel --> BulkTech[Bulk Technical]
    
    BulkPrice --> Screen[Pre-Screening]
    BulkFund --> Screen
    BulkTech --> Screen
    
    Screen --> BatchAgents[Batch Agent Analysis]
    
    BatchAgents --> ValueBatch[Value Batch]
    BatchAgents --> GrowthBatch[Growth Batch]
    BatchAgents --> SentBatch[Sentiment Batch]
    BatchAgents --> MacroBatch[Macro Batch]
    BatchAgents --> RiskBatch[Risk Batch]
    
    ValueBatch --> Aggregate[Aggregate Scores]
    GrowthBatch --> Aggregate
    SentBatch --> Aggregate
    MacroBatch --> Aggregate
    RiskBatch --> Aggregate
    
    Aggregate --> Composite[Calculate Composites]
    Composite --> IPSFilter[Apply IPS Filters]
    IPSFilter --> Rank[Rank Stocks]
    
    Rank --> Diversify[Apply Diversification]
    Diversify --> Select[Select Top N]
    Select --> Weighting[Calculate Weights]
    
    Weighting --> QualityCheck{Quality OK?}
    QualityCheck -->|No| Adjust[Adjust Portfolio]
    QualityCheck -->|Yes| Final[Final Portfolio]
    Adjust --> Select
    
    Final --> Metrics[Calculate Portfolio Metrics]
    Metrics --> SavePort[Save Portfolio]
    SavePort --> QATrack[Add to QA Tracking]
    QATrack --> Display[Display in UI]
    
    style Start fill:#e1f5e1
    style Display fill:#e1f5e1
    style Parallel fill:#d1ecf1
    style BatchAgents fill:#d1ecf1
    style Final fill:#fff3cd
```

### Detailed Component Breakdown

**Universe & Filtering (1-2 seconds)**
- Load from config/universe.yaml
- Apply sector, liquidity, data availability filters
- Result: Filtered list of candidate stocks

**Parallel Data Fetching (2-3 seconds)**
- Bulk price snapshot: Polygon.io API (all tickers in one call)
- Bulk fundamentals: ThreadPoolExecutor (10 concurrent workers)
- Bulk technical: Vectorized calculations on cached history

**Batch Agent Analysis (30-60 seconds for 50 stocks)**
- All agents process all stocks in parallel
- Rank stocks by each agent's criteria
- Generate composite scores with agent weights

**Portfolio Construction (5-10 seconds)**
- Apply IPS constraints (risk tolerance, time horizon)
- Diversification rules: Max 30% per sector, min 3 sectors
- Position sizing: Equal weight / Score-based / Risk parity
- Quality checks: Min stocks, max concentration, sector diversity

**Portfolio Metrics**
- Expected return, portfolio risk, Sharpe ratio
- Diversification score, sector allocation
- Individual stock analytics

---

## 📈 Mode 3: QA & Learning Center

### Overall Architecture

```mermaid
graph TB
    Start([User Opens QA Center]) --> LoadQA[Load QA System]
    
    LoadQA --> Tab1[Tab 1: Dashboard]
    LoadQA --> Tab2[Tab 2: Tracked Tickers]
    LoadQA --> Tab3[Tab 3: Archives]
    LoadQA --> Tab4[Tab 4: Reviews]
    LoadQA --> Tab5[Tab 5: Learning Insights]
    LoadQA --> Tab6[Tab 6: Performance Analysis]
    
    Tab1 --> CurrentPrices[Fetch Current Prices]
    CurrentPrices --> CalcPerf[Calculate Performance]
    CalcPerf --> Metrics[Display Metrics]
    
    Tab2 --> TickerList[Display Tracked]
    TickerList --> Alerts[Smart Alerts]
    Alerts --> ReviewFlow[Review Process]
    
    Tab3 --> Archive[Full Archive]
    Archive --> FilterSearch[Filter & Search]
    FilterSearch --> Export[Export Options]
    Export --> SheetsSync[Google Sheets Sync]
    
    Tab4 --> ReviewSchedule[Review Management]
    ReviewSchedule --> Conduct[Conduct Review]
    Conduct --> Document[Document Learning]
    
    Tab5 --> Aggregate[Aggregate Insights]
    Aggregate --> Patterns[Pattern Analysis]
    Patterns --> ModelGaps[Identify Gaps]
    ModelGaps --> Recommend[Recommendations]
    
    Tab6 --> Configure[Configure Analysis]
    Configure --> RunAnalysis[Run Analysis]
    RunAnalysis --> FetchSheets[Fetch Google Sheets]
    FetchSheets --> ParseMove[Parse Movements]
    ParseMove --> ParallelAnal[Parallel Stock Analysis]
    ParallelAnal --> AIAnalysis[AI Root Cause Analysis]
    AIAnalysis --> PatternID[Identify Patterns]
    PatternID --> Autonomous[Autonomous Adjustment]
    Autonomous --> ApplyChanges[Apply Config Changes]
    ApplyChanges --> Results[Display Results]
    
    style Start fill:#e1f5e1
    style Autonomous fill:#fff3cd
    style ApplyChanges fill:#fff3cd
```

### Tab 6: Performance Analysis (Autonomous Learning)

```mermaid
graph TB
    Start([Run Performance Analysis]) --> DateRange[Select Date Range]
    DateRange --> Threshold[Set Movement Threshold]
    Threshold --> FetchData[Fetch Google Sheets Data]
    
    FetchData --> Validate{Valid Sheet?}
    Validate -->|No| ShowError[Show Available Sheets]
    Validate -->|Yes| ParseMovements[Parse Stock Movements]
    
    ParseMovements --> FilterThreshold[Filter by Threshold]
    FilterThreshold --> Dedupe[Deduplicate Tickers]
    Dedupe --> MovementList[Movement List: X up Y down]
    
    MovementList --> ParallelLoop[For Each Stock in Parallel]
    
    ParallelLoop --> FetchNews[Fetch Recent News]
    FetchNews --> Strategy1{Polygon.io}
    Strategy1 -->|Success| Got10[Got 10 articles]
    Strategy1 -->|Fail| Strategy2[get_news_with_sources]
    Strategy2 -->|Success| Got8[Got 8 articles]
    Strategy2 -->|Fail| Strategy3[Perplexity Fast Search]
    Strategy3 --> Got5[Got 5 articles]
    
    Got10 --> GetFund[Get Fundamentals]
    Got8 --> GetFund
    Got5 --> GetFund
    
    GetFund --> AIModel{Choose AI}
    AIModel -->|OpenAI| GPT4[GPT-4 Analysis]
    AIModel -->|Perplexity| Sonar[Sonar-Pro Analysis]
    
    GPT4 --> Analyze[Analyze Root Causes]
    Sonar --> Analyze
    
    Analyze --> ParseJSON[Parse Response]
    ParseJSON --> StockDone[Stock Analysis Complete]
    
    StockDone --> AllDone{All Stocks Done?}
    AllDone -->|No| ParallelLoop
    AllDone -->|Yes| IdentifyPatterns[Identify Patterns]
    
    IdentifyPatterns --> CalcFreq[Calculate Pattern Frequencies]
    CalcFreq --> AutoPhase[AUTONOMOUS ADJUSTMENT]
    
    AutoPhase --> CalcWeights[Calculate Agent Weight Changes]
    CalcWeights --> MissRate[Check Agent Miss Rates]
    MissRate --> PatternAdj[Pattern-Based Adjustments]
    
    PatternAdj --> Earnings{Earnings over 40 percent?}
    PatternAdj --> NewsPattern{News over 40 percent?}
    PatternAdj --> SectorPattern{Sector over 30 percent?}
    
    Earnings -->|Yes| IncValue[Increase Value Agent]
    NewsPattern -->|Yes| IncSent[Increase Sentiment Agent]
    SectorPattern -->|Yes| IncMacro[Increase Macro Agent]
    
    IncValue --> CalcThresh[Calculate Threshold Changes]
    IncSent --> CalcThresh
    IncMacro --> CalcThresh
    
    CalcThresh --> ConfCheck{Avg Confidence?}
    ConfCheck -->|High over 75| Aggressive[More Aggressive Thresholds]
    ConfCheck -->|Low under 50| Conservative[More Conservative Thresholds]
    
    Aggressive --> Backup[Backup config file]
    Conservative --> Backup
    
    Backup --> UpdateYAML[Update config model yaml]
    UpdateYAML --> LogHistory[Log to adjustment history]
    
    LogHistory --> DisplayResults[Display Results]
    DisplayResults --> MoveSummary[Movement Summary]
    DisplayResults --> RootCauses[Root Causes]
    DisplayResults --> AdjustReport[Adjustment Report]
    DisplayResults --> Recommendations[Model Recommendations]
    
    MoveSummary --> SaveAll[Save All Results]
    RootCauses --> SaveAll
    AdjustReport --> SaveAll
    Recommendations --> SaveAll
    
    SaveAll --> FeedbackLoop[Feedback to System]
    FeedbackLoop --> NextRun[Next Portfolio Uses New Weights]
    
    style Start fill:#e1f5e1
    style AutoPhase fill:#fff3cd
    style UpdateYAML fill:#fff3cd
    style FeedbackLoop fill:#d4edda
    style NextRun fill:#d4edda
```

### Autonomous Adjustment Logic

**Agent Weight Adjustment**
- If agent miss rate over 30 percent: Increase weight by 10-25 percent
- Earnings-driven (over 40 percent): Plus 15 percent to Value Agent
- News-driven (over 40 percent): Plus 20 percent to Sentiment Agent
- Sector-driven (over 30 percent): Plus 15 percent to Macro Agent
- Technical (over 20 percent): Plus 10 percent to Growth Agent
- Max adjustment: Plus 25 percent per run

**Threshold Adjustment**
- High confidence (over 75 percent): Lower thresholds (more aggressive)
- Low confidence (under 50 percent): Raise thresholds (more conservative)
- Moderate confidence (50-75 percent): Keep current thresholds

**Safety Features**
- Automatic backup: config model yaml backup with TIMESTAMP
- Full audit trail: adjustment history json
- Revertible changes: Can roll back to any backup
- Capped adjustments: Prevents over-correction

---

## 🔄 Cross-Mode Integration

```mermaid
graph LR
    subgraph Mode1[Mode 1: Single Stock]
        A1[Analyze Stock] --> A2[Generate Recommendation]
        A2 --> A3[Save to QA System]
    end
    
    subgraph Mode2[Mode 2: Portfolio]
        B1[Generate Portfolio] --> B2[Multiple Analyses]
        B2 --> B3[Save All to QA]
    end
    
    subgraph Mode3[Mode 3: QA & Learning]
        C1[Track Performance] --> C2[Analyze Patterns]
        C2 --> C3[Autonomous Adjustment]
        C3 --> C4[Update Config]
    end
    
    A3 -->|Feeds Into| C1
    B3 -->|Feeds Into| C1
    C4 -->|Improves| A1
    C4 -->|Improves| B1
    
    style Mode1 fill:#e3f2fd
    style Mode2 fill:#f3e5f5
    style Mode3 fill:#e8f5e9
```

---

## 📊 Data Flow Architecture

```mermaid
graph TB
    subgraph External[External Data Sources]
        Polygon[Polygon.io: Prices & News]
        YFinance[Yahoo Finance: Prices & Fundamentals]
        Perplexity[Perplexity AI: News & Analysis]
        OpenAI[OpenAI GPT-4: Deep Analysis]
    end
    
    subgraph System[Core System]
        DataProvider[Enhanced Data Provider]
        Cache[data cache directory]
        
        Mode1[Mode 1: Single Stock Analysis]
        Mode2[Mode 2: Portfolio Generation]
        Mode3[Mode 3: QA & Learning]
    end
    
    subgraph Storage[Data Storage]
        QAStorage[data qa system directory]
        PerfStorage[data performance analysis directory]
        Config[config model yaml]
    end
    
    Polygon --> DataProvider
    YFinance --> DataProvider
    Perplexity --> DataProvider
    OpenAI --> Mode3
    
    DataProvider --> Cache
    Cache --> Mode1
    Cache --> Mode2
    
    Mode1 --> QAStorage
    Mode2 --> QAStorage
    
    QAStorage --> Mode3
    Mode3 --> PerfStorage
    Mode3 --> Config
    
    Config --> Mode1
    Config --> Mode2
    
    style System fill:#4CAF50,color:#fff
    style Storage fill:#2196F3,color:#fff
    style External fill:#FF9800,color:#fff
```

---

## 📝 Performance Metrics

| Metric | Mode 1 | Mode 2 | Mode 3 (Performance Analysis) |
|--------|--------|--------|-------------------------------|
| **Typical Duration** | 10-15 seconds | 2-4 minutes | 2-5 minutes |
| **API Calls** | 5-10 | 50-200 | 100-500 |
| **Stocks Analyzed** | 1 | 5-50 | All with over 15 percent movement |
| **Outputs** | 1 recommendation | 1 portfolio | Learning updates plus config changes |
| **Storage Impact** | ~50KB | ~500KB-2MB | ~1-5MB |
| **Autonomous Actions** | None | None | **Yes: Modifies config model yaml** |

---

## 🎯 Key Features Summary

### Mode 1: Single Stock Analysis
- Real-time data from multiple sources
- 5 specialized AI agents
- IPS alignment checking
- Comprehensive reporting
- Automatic QA tracking

### Mode 2: Portfolio Generation
- Parallel bulk processing (50-60 percent faster)
- Smart diversification rules
- Multiple weighting strategies
- Quality checks and constraints
- Full portfolio analytics

### Mode 3: QA & Learning Center
- Continuous performance tracking
- Smart alert system
- Pattern identification
- **Autonomous self-adjustment**
- **Automatically modifies agent weights**
- **Learns from mistakes**
- Google Sheets integration

---

## 🚀 Getting Started

1. **Single Stock Analysis**: Enter ticker → Get recommendation in 10-15 seconds
2. **Portfolio Generation**: Set parameters → Get diversified portfolio in 2-4 minutes  
3. **Performance Analysis**: Run weekly → System learns and improves automatically

The system continuously learns from its performance and automatically adjusts its configuration to improve future predictions. No manual intervention required!

---

*These diagrams are rendered natively on GitHub. View them at: https://github.com/yaboibean2/Wharton/blob/main/SYSTEM_FLOW_DIAGRAMS.md*
