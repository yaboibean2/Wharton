# Portfolio Selection Logs

This directory contains detailed logs of all AI-powered portfolio selection sessions.

## Log Format

Each session generates a JSON file named:
```
portfolio_selection_YYYYMMDD_HHMMSS.json
```

## Log Contents

Each log file contains:

### Session Metadata
- `timestamp`: Session start time
- `challenge_context`: Investment challenge description
- `client_profile`: Complete client profile with IPS data
- `universe_size`: Size of investment universe considered

### Selection Stages

#### Stage 1: OpenAI Initial Selection
- Selected tickers (20)
- API prompt used
- Model response
- Model version

#### Stage 2: Perplexity Initial Selection
- Selected tickers (20)
- API prompt used
- Model response  
- Model version

#### Stage 3: Aggregation
- Unique tickers count
- List of all candidates

#### Stage 4: Rationale Generation
- 4-sentence rationale for each ticker
- Explanation of strength, benefit, relevance, strategy

#### Stage 5: Final Selection Rounds
- Round 1 top 5 tickers
- Round 2 top 5 tickers
- Round 3 top 5 tickers

#### Stage 6: Final Consolidation
- Unique finalists from all rounds
- Final 5 selected tickers
- Consolidation logic (if more than 5)

## Usage

Logs are automatically created during each portfolio generation session. They provide:
- Complete audit trail
- Reproducibility of results
- Debugging information
- Historical selection patterns
- Model performance tracking

## Retention

Logs are kept indefinitely for:
- Regulatory compliance
- Performance analysis
- Model improvement
- Client reporting
- Audit requirements

## Access

Logs can be downloaded directly from the Portfolio Recommendations page:
- Individual session logs (JSON)
- Full analysis reports (JSON)
- Portfolio summaries (CSV)
