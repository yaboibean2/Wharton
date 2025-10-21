# System Architecture & Flow Diagrams

## 📊 AI Investment System - Complete Pipeline Visualization

This document provides clear, intuitive diagrams showing exactly how the system works in each mode.

---

## 🎯 Mode 1: Single Stock Analysis

**Purpose**: Analyze one stock and get an AI-powered investment recommendation  
**Time**: ~15 seconds  
**Output**: BUY/HOLD/SELL recommendation with detailed reasoning

```mermaid
graph TB
    Start([Enter Stock Ticker]) --> Fetch[Gather Data from Multiple Sources]
    
    Fetch --> Data1[Price History]
    Fetch --> Data2[Company Fundamentals]
    Fetch --> Data3[Recent News]
    Fetch --> Data4[Analyst Ratings]
    
    Data1 --> Analysis[5 AI Agents Analyze Stock]
    Data2 --> Analysis
    Data3 --> Analysis
    Data4 --> Analysis
    
    Analysis --> Agent1[Value Agent<br/>Valuation metrics]
    Analysis --> Agent2[Growth Agent<br/>Growth trajectory]
    Analysis --> Agent3[Sentiment Agent<br/>Market sentiment]
    Analysis --> Agent4[Macro Agent<br/>Market conditions]
    Analysis --> Agent5[Risk Agent<br/>Risk assessment]
    
    Agent1 --> Combine[Combine Scores<br/>Apply Agent Weights]
    Agent2 --> Combine
    Agent3 --> Combine
    Agent4 --> Combine
    Agent5 --> Combine
    
    Combine --> IPS[Check Against<br/>Investment Policy]
    IPS --> Recommend[Generate Recommendation]
    
    Recommend --> Output[Display Results<br/>Save to Tracking]
    
    style Start fill:#4CAF50,color:#fff
    style Analysis fill:#2196F3,color:#fff
    style Combine fill:#FF9800,color:#fff
    style Output fill:#4CAF50,color:#fff
```

### Key Features
- **Data Sources**: YFinance (prices), Perplexity AI (news), Analyst consensus
- **5 AI Agents**: Each specializes in different aspects (value, growth, sentiment, macro, risk)
- **Smart Weighting**: Agent weights from `config/model.yaml` (automatically adjusted by learning system)
- **IPS Compliance**: Checks alignment with time horizon, risk tolerance, tax considerations

---

## 🎲 Mode 2: Portfolio Generation

**Purpose**: Build a diversified portfolio of 5-50 stocks optimized for your goals  
**Time**: ~2-4 minutes  
**Output**: Complete portfolio with weights, metrics, and individual stock analyses

```mermaid
graph TB
    Start([Set Portfolio Parameters<br/>Size, Risk, Sector Focus]) --> Universe[Load Stock Universe<br/>from config]
    
    Universe --> Filter[Filter Candidates<br/>Liquidity, Data Quality]
    
    Filter --> Fetch[Parallel Data Fetch<br/>All Stocks at Once]
    
    Fetch --> Screen[Quick Screen<br/>Price, Volume, Quality]
    
    Screen --> Analyze[Run All 5 Agents<br/>On All Stocks]
    
    Analyze --> Score[Calculate Composite Scores<br/>Rank All Stocks]
    
    Score --> Build[Portfolio Construction]
    
    Build --> Rule1[Apply IPS Rules<br/>Risk Tolerance Match]
    Build --> Rule2[Apply Diversification<br/>Max 30% per Sector]
    Build --> Rule3[Avoid Correlation<br/>Low overlap stocks]
    
    Rule1 --> Select[Select Top N Stocks]
    Rule2 --> Select
    Rule3 --> Select
    
    Select --> Weight[Calculate Position Sizes<br/>Equal or Score-Based]
    
    Weight --> Quality{Quality Check<br/>Meets All Rules?}
    Quality -->|No| Build
    Quality -->|Yes| Final[Final Portfolio Ready]
    
    Final --> Results[Display Results<br/>Track Performance]
    
    style Start fill:#4CAF50,color:#fff
    style Fetch fill:#2196F3,color:#fff
    style Build fill:#FF9800,color:#fff
    style Final fill:#9C27B0,color:#fff
    style Results fill:#4CAF50,color:#fff
```

### Key Features
- **Parallel Processing**: Fetches and analyzes 50+ stocks in ~3 seconds using bulk APIs
- **Smart Diversification**: Automatic sector balancing, correlation checking, size distribution
- **Multiple Weighting**: Choose equal weight, score-based, or risk parity
- **Quality Assurance**: Validates min stocks, max concentration, sector diversity before finalizing

---

## 📈 Mode 3: QA & Learning Center

**Purpose**: Track performance, learn from outcomes, and automatically improve the AI system  
**Time**: Continuous monitoring + weekly learning cycles  
**Output**: Performance metrics, learning insights, autonomous system improvements

### System Overview

```mermaid
graph TB
    Start([QA & Learning Center]) --> Track[Track All Recommendations]
    
    Track --> Monitor[Monitor Performance<br/>Real-time Price Updates]
    
    Monitor --> Tab1[Dashboard<br/>Accuracy Metrics]
    Monitor --> Tab2[Tracked Tickers<br/>Individual Performance]
    Monitor --> Tab3[Archives<br/>Historical Analysis]
    
    Tab1 --> Review[Weekly Review Cycle]
    Tab2 --> Review
    Tab3 --> Review
    
    Review --> Learn[Performance Analysis<br/>Autonomous Learning]
    
    Learn --> Identify[Identify What Worked<br/>and What Did Not]
    
    Identify --> Adjust[Automatically Adjust<br/>Agent Weights & Thresholds]
    
    Adjust --> Improve[Improved Future<br/>Recommendations]
    
    Improve --> Track
    
    style Start fill:#4CAF50,color:#fff
    style Learn fill:#FF9800,color:#fff
    style Adjust fill:#F44336,color:#fff
    style Improve fill:#4CAF50,color:#fff
```

### The Autonomous Learning Engine (Core Innovation)

**Purpose**: Analyze significant stock movements and automatically improve the AI system  
**Time**: ~2-5 minutes per analysis  
**Output**: System automatically updates itself to perform better

```mermaid
graph TB
    Start([Run Performance Analysis]) --> Fetch[Fetch Stock Movement Data<br/>from Google Sheets]
    
    Fetch --> Filter[Find Significant Moves<br/>Over 15 percent threshold]
    
    Filter --> Analyze[For Each Stock Movement]
    
    Analyze --> News[Gather Recent News<br/>Multi-source strategy]
    Analyze --> Context[Get Market Context<br/>Fundamentals & Events]
    
    News --> AI[AI Deep Dive Analysis<br/>Why did this happen?]
    Context --> AI
    
    AI --> Causes[Identify Root Causes<br/>Earnings, News, Sector, Technical]
    
    Causes --> Done{All Stocks<br/>Analyzed?}
    Done -->|No| Analyze
    Done -->|Yes| Patterns[Identify Patterns<br/>Across All Movements]
    
    Patterns --> Learn[Learning Phase<br/>What did we miss?]
    
    Learn --> Miss1{Value Agent<br/>Missed earnings?}
    Learn --> Miss2{Sentiment Agent<br/>Missed news?}
    Learn --> Miss3{Macro Agent<br/>Missed sector trends?}
    
    Miss1 -->|Yes| Adjust[Autonomous Adjustment]
    Miss2 -->|Yes| Adjust
    Miss3 -->|Yes| Adjust
    
    Adjust --> Backup[Backup Current Config<br/>Safety first]
    
    Backup --> Update[Update Agent Weights<br/>and Thresholds]
    
    Update --> Log[Log All Changes<br/>Full audit trail]
    
    Log --> Apply[Apply to Next Analysis<br/>System Improved]
    
    style Start fill:#4CAF50,color:#fff
    style AI fill:#2196F3,color:#fff
    style Learn fill:#FF9800,color:#fff
    style Adjust fill:#F44336,color:#fff
    style Apply fill:#4CAF50,color:#fff
```

### How It Learns (The Magic)

**1. Detection**: Finds stocks that moved significantly (default: over 15 percent)

**2. Investigation**: For each movement, the AI asks "Why did this happen?"
   - Fetches recent news from multiple sources
   - Gets company fundamentals and events
   - Uses GPT-4 or Perplexity to analyze root causes

**3. Pattern Recognition**: Looks across all movements
   - 40 percent earnings-driven → Value Agent needs more weight
   - 40 percent news-driven → Sentiment Agent needs more weight
   - 30 percent sector-driven → Macro Agent needs more weight

**4. Autonomous Adjustment**: Automatically updates `config/model.yaml`
   - Increases weights for agents that missed opportunities
   - Adjusts thresholds based on confidence levels
   - Creates automatic backup before changes

**5. Continuous Improvement**: Next time the system runs
   - Uses updated agent weights
   - Makes better predictions
   - Learns from every cycle

### Safety Features
- ✅ **Automatic backups** before any changes
- ✅ **Capped adjustments** (max 25 percent per run)
- ✅ **Full audit trail** in adjustment_history.json
- ✅ **Revertible** - can roll back to any previous state

---

## 🔄 How Everything Works Together

**The Complete Learning Loop**

```mermaid
graph TB
    User([You Use The System]) --> Action{What Do You Need?}
    
    Action -->|Analyze One Stock| Mode1[Mode 1: Single Stock Analysis]
    Action -->|Build Portfolio| Mode2[Mode 2: Portfolio Generation]
    Action -->|Review Performance| Mode3[Mode 3: QA & Learning]
    
    Mode1 --> Rec1[Get Recommendation]
    Mode2 --> Rec2[Get Portfolio]
    
    Rec1 --> Track[Saved to QA System<br/>Tracked Automatically]
    Rec2 --> Track
    
    Track --> Monitor[System Monitors<br/>Real-time Performance]
    
    Monitor --> Weekly[Weekly: Performance Analysis]
    
    Weekly --> Learn[AI Analyzes Results<br/>What worked? What did not?]
    
    Learn --> Auto[Autonomous Adjustment<br/>System Updates Itself]
    
    Auto --> Better[Improved Agent Weights<br/>Better Predictions]
    
    Better --> Mode1
    Better --> Mode2
    
    Mode3 --> Export[Export Reports<br/>Share Insights]
    
    style User fill:#4CAF50,color:#fff
    style Track fill:#2196F3,color:#fff
    style Learn fill:#FF9800,color:#fff
    style Auto fill:#F44336,color:#fff
    style Better fill:#9C27B0,color:#fff
```

### The Key Innovation: Self-Improvement

1. **You analyze stocks or build portfolios** → System makes recommendations
2. **QA System tracks everything** → Monitors actual outcomes vs predictions
3. **Performance Analysis runs weekly** → AI investigates why stocks moved
4. **System learns automatically** → Updates its own configuration
5. **Future predictions improve** → Gets smarter with every cycle

**This is not static AI. It's a continuously learning system that improves itself.**

---

## � System Comparison

| Feature | Mode 1: Single Stock | Mode 2: Portfolio | Mode 3: QA & Learning |
|---------|---------------------|-------------------|----------------------|
| **Time** | ~15 seconds | ~2-4 minutes | ~2-5 minutes |
| **Input** | 1 ticker symbol | Portfolio parameters | Historical data |
| **Output** | BUY/HOLD/SELL + analysis | Complete portfolio | System improvements |
| **Best For** | Quick stock checks | Building positions | Weekly reviews |
| **Learns?** | No | No | **YES - Updates system** |

---

## 🎯 Data Sources & Technology

```mermaid
graph LR
    Sources[Data Sources] --> System[AI Investment System]
    
    Sources --> A[Yahoo Finance<br/>Price & Fundamentals]
    Sources --> B[Polygon.io<br/>News & Real-time Data]
    Sources --> C[Perplexity AI<br/>News Analysis]
    Sources --> D[OpenAI GPT-4<br/>Deep Analysis]
    
    System --> Out1[Stock Recommendations]
    System --> Out2[Portfolio Suggestions]
    System --> Out3[Learning Insights]
    
    Out1 --> Storage[QA Tracking System]
    Out2 --> Storage
    Out3 --> Config[Auto-Updated Config]
    
    Config --> System
    
    style Sources fill:#FF9800,color:#fff
    style System fill:#4CAF50,color:#fff
    style Config fill:#F44336,color:#fff
```

---

## 🚀 Quick Start Guide

### For Investment Analysis
1. **Analyze a single stock**: Enter ticker → Get AI recommendation in 15 seconds
2. **Build a portfolio**: Set parameters → Get optimized portfolio in 2-4 minutes
3. **Review weekly**: Run Performance Analysis → System improves automatically

### For System Learning
- **Connect Google Sheets**: Track all stocks and their performance
- **Run Performance Analysis**: Weekly or after major market moves
- **Let it learn**: System automatically adjusts weights and improves

### What Makes This Special
- ✅ **5 Specialized AI Agents**: Each expert in different aspects
- ✅ **Parallel Processing**: Fast bulk analysis (50+ stocks in ~3 seconds)
- ✅ **Autonomous Learning**: System improves itself automatically
- ✅ **Full Transparency**: Complete audit trail of all decisions
- ✅ **IPS Compliance**: Respects your investment policy and risk tolerance

---

## � Key Insights

### Why Multiple AI Agents?
Different aspects of investing require different expertise:
- **Value Agent**: Looks at fundamentals and valuation
- **Growth Agent**: Analyzes momentum and trajectory
- **Sentiment Agent**: Interprets news and market sentiment
- **Macro Agent**: Considers broader market conditions
- **Risk Agent**: Assesses volatility and downside protection

### Why Autonomous Learning?
Markets change constantly. What worked last quarter may not work now. The system:
- Detects when agents miss important signals
- Identifies patterns in successful vs failed predictions
- Automatically adjusts agent weights to improve
- Creates backups and audit trails for transparency

### Why Google Sheets Integration?
- Easy to review and share performance data
- Automatic sync of all tracked stocks
- Visual tracking of recommendations vs reality
- Export-friendly for further analysis

---

*View these interactive diagrams on GitHub: https://github.com/yaboibean2/Wharton*
