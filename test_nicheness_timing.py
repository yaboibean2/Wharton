#!/usr/bin/env python3
"""
Nicheness Timing Test
=====================
Runs full analyses on ~12 stocks spanning 5 "nicheness" tiers (mega / large /
mid / small / micro) and measures wall-clock time for every pipeline phase.

Results are used to calibrate the per-tier time estimates in
engine/portfolio_orchestrator.py (_TIER_ESTIMATES) and in app.py.

Usage:
    python test_nicheness_timing.py              # run full 12-stock default suite
    python test_nicheness_timing.py --dry-run    # list stocks without running
    python test_nicheness_timing.py AAPL MVIS    # custom tickers (auto-classified)
    python test_nicheness_timing.py --tier mega  # run only a specific tier

Output:
    - Per-ticker timing table (Data / Fundamentals / Agents / Total)
    - Per-tier average table with P75 and P90 columns
    - CALIBRATION BLOCK: ready-to-paste _TIER_ESTIMATES dict for orchestrator.py
    - Tier data written to data/step_times.json under keys like
      total_mega, data_large, agents_micro, etc.
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from statistics import mean, median, stdev
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nicheness_timing")
logger.setLevel(logging.INFO)

from openai import OpenAI
from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator
from utils.config_loader import get_config_loader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STEP_TIMES_PATH = os.path.join(os.path.dirname(__file__), "data", "step_times.json")

# ---------------------------------------------------------------------------
# Test suite — 12 stocks calibrated to cover the full data-speed spectrum.
#
# "Tier" reflects how quickly financial APIs return data for this ticker,
# which correlates strongly with market-cap but also with index membership,
# analyst coverage, and yfinance cache hit rates.
#
# Swap any ticker for a local equivalent if you prefer, but keep the tier
# label accurate so the calibration output remains meaningful.
# ---------------------------------------------------------------------------
TEST_SUITE = [
    # ── Mega-cap  ($100B+, S&P top names, yfinance cache almost always hits) ──
    ("AAPL",  "mega",  "Apple  — $3T, maximum coverage across all APIs"),
    ("MSFT",  "mega",  "Microsoft — $3T, total API saturation"),
    ("NVDA",  "mega",  "NVIDIA — $2T, top coverage + heavy analyst attention"),

    # ── Large-cap  ($10–100B, S&P 500 but not top-tier, occasional API miss) ──
    ("SBUX",  "large", "Starbucks — $90B, consumer staple, solid coverage"),
    ("SQ",    "large", "Block — $40B, fintech, moderate yfinance hit rate"),

    # ── Mid-cap  ($2–10B, some data gaps, Perplexity fallback sometimes triggered) ──
    ("CROX",  "mid",   "Crocs — $5B, consumer discretionary, mid-tier APIs"),
    ("SAIA",  "mid",   "Saia Inc — $4B, trucking, occasional sentiment fallback"),

    # ── Small-cap  ($300M–2B, limited analyst coverage, API fallbacks common) ──
    ("ASAN",  "small", "Asana — $1.5B, SaaS project mgmt, limited data sources"),
    ("LPSN",  "small", "LivePerson — $200M, enterprise AI, sparse coverage"),

    # ── Micro/niche  (<$300M or obscure, triggers full Perplexity / multi-fallback) ──
    ("MVIS",  "micro", "MicroVision — $250M, lidar startup, minimal data APIs"),
    ("DAVE",  "micro", "Dave Inc — $300M, fintech neobank, sparse fundamentals"),
    ("HIMS",  "micro", "Hims & Hers — $1.5B, telehealth, limited traditional data"),
]

TIER_ORDER = ["mega", "large", "mid", "small", "micro"]

# Hard-coded fallback estimates (used when step_times.json has no data for a tier).
# These are conservative — better to over-estimate and finish early than freeze.
_TIER_HARDCODED_DEFAULTS = {
    "mega":  {"total": 25, "data": 6,  "fund": 4,  "agents": 17},
    "large": {"total": 40, "data": 15, "fund": 12, "agents": 18},
    "mid":   {"total": 55, "data": 28, "fund": 24, "agents": 19},
    "small": {"total": 75, "data": 45, "fund": 40, "agents": 21},
    "micro": {"total": 95, "data": 65, "fund": 60, "agents": 23},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_tier_by_market_cap(ticker: str) -> str:
    """Classify a ticker into a data-speed tier using yfinance fast_info."""
    # Static fast path — known mega-caps never need an API call
    _KNOWN_MEGA = {
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "BRK-A",
        "TSLA", "LLY", "JPM", "V", "WMT", "XOM", "UNH", "MA", "AVGO", "PG", "JNJ",
        "HD", "MRK", "ABBV", "CVX", "COST", "NFLX", "KO", "BAC", "ORCL", "CRM",
        "AMD", "TMO", "MCD", "PM", "ACN", "CSCO", "PEP", "GE", "NKE", "ADBE",
        "T", "DIS", "INTC", "TXN", "UPS", "NEE", "QCOM", "DHR", "WFC", "IBM",
    }
    if ticker.upper() in _KNOWN_MEGA:
        return "mega"
    try:
        import yfinance as yf
        fast = yf.Ticker(ticker).fast_info
        mc = getattr(fast, "market_cap", None) or getattr(fast, "marketCap", None)
        if mc:
            if mc >= 100e9: return "mega"
            if mc >= 10e9:  return "large"
            if mc >= 2e9:   return "mid"
            if mc >= 300e6: return "small"
            return "micro"
    except Exception:
        pass
    return "micro"  # conservative: treat unknown as slow


def run_timed_analysis(orchestrator, ticker: str, analysis_date: str):
    """Run full pipeline analysis and return per-phase wall-clock timings."""
    logger.info(f"  ▶  {ticker}")
    milestones: list = []

    def _cb(pct, msg):
        milestones.append((time.time(), pct, msg))

    t_start = time.time()
    try:
        result = orchestrator.analyze_stock(
            ticker=ticker,
            analysis_date=analysis_date,
            progress_callback=_cb,
        )
    except Exception as e:
        logger.error(f"  ✗  {ticker} failed with exception: {e}")
        return None, str(e)

    total_elapsed = time.time() - t_start

    if isinstance(result, dict) and "error" in result:
        logger.error(f"  ✗  {ticker} analysis error: {result['error']}")
        return None, str(result["error"])

    # Derive phase boundaries from milestone percentages
    t0 = milestones[0][0] if milestones else t_start
    t_data = t_agents = t_blend = None
    for ts, pct, _ in milestones:
        if pct >= 42 and t_data is None:    t_data = ts
        if pct >= 98 and t_agents is None:  t_agents = ts
        if pct >= 100 and t_blend is None:  t_blend = ts

    phases: dict = {"total": round(total_elapsed, 2)}
    if t_data:
        phases["data"]   = round(t_data - t0, 2)
    if t_data and t_agents:
        phases["agents"] = round(t_agents - t_data, 2)
    if t_agents and t_blend:
        phases["blend"]  = round(t_blend - t_agents, 2)

    # Richer per-step data from the orchestrator result
    step_timings = result.get("step_timings", {}) if isinstance(result, dict) else {}
    for k in ("fundamentals", "price_history", "benchmark",
              "value_agent", "growth_momentum_agent", "macro_regime_agent",
              "risk_agent", "sentiment_agent", "agents_wall"):
        if k in step_timings:
            phases[k] = round(step_timings[k], 2)

    score = result.get("final_score", "?") if isinstance(result, dict) else "?"
    logger.info(
        f"  ✓  {ticker}  score={score}  "
        f"data={phases.get('data', '?')}s  "
        f"agents_wall={phases.get('agents_wall', phases.get('agents', '?'))}s  "
        f"total={phases['total']}s"
    )
    return phases, None


# ---------------------------------------------------------------------------
# step_times.json I/O
# ---------------------------------------------------------------------------

def load_step_times() -> dict:
    if os.path.exists(STEP_TIMES_PATH):
        with open(STEP_TIMES_PATH, "r") as f:
            return json.load(f)
    return {"step_times": {}, "metadata": {}}


def save_step_times(data: dict):
    os.makedirs(os.path.dirname(STEP_TIMES_PATH), exist_ok=True)
    with open(STEP_TIMES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def update_tier_times(results_by_tier: dict):
    """Merge per-tier timing data into step_times.json under tier-keyed lists."""
    store = load_step_times()
    st = store.setdefault("step_times", {})

    for tier, records in results_by_tier.items():
        for r in records:
            phases = r.get("phases", {})
            for key in ("total", "data", "agents", "fundamentals",
                        "agents_wall", "price_history", "benchmark",
                        "value_agent", "growth_momentum_agent",
                        "macro_regime_agent", "risk_agent", "sentiment_agent"):
                val = phases.get(key)
                if val is not None:
                    st.setdefault(f"{key}_{tier}", []).append(round(val, 3))

    # Keep the most recent 50 samples per tier key
    for k in list(st.keys()):
        if len(st[k]) > 50:
            st[k] = st[k][-50:]

    store["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "total_samples": sum(len(v) for v in st.values()),
        "tier_calibration_run": True,
    }
    save_step_times(store)
    logger.info(f"  Saved per-tier timings to {STEP_TIMES_PATH}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(vals: list, p: float) -> float:
    """Return the p-th percentile (0–1) of a sorted list."""
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def print_results_table(all_results: list):
    """Print per-ticker timing details."""
    print(f"\n{'='*90}")
    print(f"  NICHENESS TIMING RESULTS  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*90}")
    print(f"  {'Tier':<7} {'Ticker':<8} {'Data':>8} {'Fund':>9} {'Agents':>9} {'Total':>9}  Description")
    print(f"  {'-'*80}")
    for r in all_results:
        tier   = r["tier"]
        ticker = r["ticker"]
        desc   = r.get("desc", "")
        ph     = r["phases"]
        data_s   = f"{ph['data']:.1f}s"   if "data"   in ph else "  N/A"
        fund_s   = f"{ph.get('fundamentals', ph.get('data', 0)):.1f}s"
        agents_s = f"{ph.get('agents_wall', ph.get('agents', 0)):.1f}s"
        total_s  = f"{ph['total']:.1f}s"
        print(f"  {tier:<7} {ticker:<8} {data_s:>8} {fund_s:>9} {agents_s:>9} {total_s:>9}  {desc}")


def print_tier_summary(results_by_tier: dict) -> dict:
    """Print per-tier aggregates and return a calibration dict."""
    print(f"\n{'='*90}")
    print(f"  PER-TIER SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Tier':<7} {'N':>3}  {'Data avg':>10}  {'Fund avg':>10}  "
          f"{'Agents avg':>12}  {'Total avg':>10}  {'Total P75':>10}  {'Total P90':>10}")
    print(f"  {'-'*82}")

    calibration = {}
    for tier in TIER_ORDER:
        records = results_by_tier.get(tier, [])
        if not records:
            continue
        n = len(records)
        phases_list = [r["phases"] for r in records]

        def _vals(key):
            return [p[key] for p in phases_list if key in p]

        totals   = _vals("total")
        datas    = _vals("data")
        funds    = _vals("fundamentals") or _vals("data")
        agents   = [p.get("agents_wall") or p.get("agents", 0) for p in phases_list
                    if p.get("agents_wall") or p.get("agents")]

        avg_total   = mean(totals)   if totals   else 0
        avg_data    = mean(datas)    if datas    else 0
        avg_fund    = mean(funds)    if funds    else 0
        avg_agents  = mean(agents)   if agents   else 0
        p75_total   = _pct(totals, 0.75)
        p90_total   = _pct(totals, 0.90)

        print(f"  {tier:<7} {n:>3}  {avg_data:>9.1f}s  {avg_fund:>9.1f}s  "
              f"{avg_agents:>11.1f}s  {avg_total:>9.1f}s  "
              f"{p75_total:>9.1f}s  {p90_total:>9.1f}s")

        # Calibration target: use P90 with a 5% safety buffer, rounded up
        cal_total  = int(p90_total  * 1.05 + 0.5)
        cal_data   = int(avg_data   * 1.15 + 0.5)  # 15% buffer on data phase
        cal_fund   = int(avg_fund   * 1.15 + 0.5)
        cal_agents = int(avg_agents * 1.20 + 0.5)  # 20% buffer on agents

        calibration[tier] = {
            "total":  max(cal_total,  _TIER_HARDCODED_DEFAULTS[tier]["total"]),
            "data":   max(cal_data,   _TIER_HARDCODED_DEFAULTS[tier]["data"]),
            "fund":   max(cal_fund,   _TIER_HARDCODED_DEFAULTS[tier]["fund"]),
            "agents": max(cal_agents, _TIER_HARDCODED_DEFAULTS[tier]["agents"]),
        }

    return calibration


def print_calibration_block(calibration: dict):
    """Print a ready-to-paste Python dict for _TIER_ESTIMATES."""
    # Niche multiplier and detection threshold: derived from tier relative to mega
    niche_config = {
        "mega":  (1.0, 2.5),
        "large": (1.0, 2.0),
        "mid":   (1.1, 1.7),
        "small": (1.3, 1.5),
        "micro": (1.5, 1.3),
    }

    print(f"\n{'='*90}")
    print(f"  CALIBRATION OUTPUT")
    print(f"  Copy this block into engine/portfolio_orchestrator.py → _TIER_ESTIMATES")
    print(f"{'='*90}")
    print(f"\n_TIER_ESTIMATES = {{")
    for tier in TIER_ORDER:
        cal = calibration.get(tier, _TIER_HARDCODED_DEFAULTS.get(tier, {}))
        nm, nt = niche_config.get(tier, (1.0, 2.0))
        print(f"    '{tier}':  {{'total': {cal['total']}, 'data': {cal['data']}, "
              f"'fund': {cal['fund']}, 'agents': {cal['agents']}, "
              f"'niche_mult_start': {nm}, 'niche_threshold': {nt}}},")
    print(f"}}")
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    # Dry run: just print the plan
    if "--dry-run" in args:
        print("\nDRY RUN — stocks that would be analysed:\n")
        for ticker, tier, desc in TEST_SUITE:
            print(f"  [{tier:5}] {ticker:<8}  {desc}")
        print()
        sys.exit(0)

    # Tier filter
    tier_filter = None
    if "--tier" in args:
        idx = args.index("--tier")
        if idx + 1 < len(args):
            tier_filter = args[idx + 1].lower()
            args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    # Custom tickers (any remaining positional args)
    custom_tickers = [a.upper() for a in args if not a.startswith("--")]
    if custom_tickers:
        suite = [(t, classify_tier_by_market_cap(t), "custom") for t in custom_tickers]
        print(f"\nCustom tickers (auto-classified):")
        for ticker, tier, _ in suite:
            print(f"  [{tier:5}] {ticker}")
    else:
        suite = TEST_SUITE

    # Apply tier filter
    if tier_filter:
        suite = [(t, tr, d) for t, tr, d in suite if tr == tier_filter]
        if not suite:
            print(f"No stocks match tier '{tier_filter}'. Valid tiers: {TIER_ORDER}")
            sys.exit(1)

    analysis_date = datetime.now().strftime("%Y-%m-%d")

    # Build orchestrator
    config_loader = get_config_loader()
    model_config  = config_loader.load_model_config()
    ips_config    = config_loader.load_ips()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    data_provider = EnhancedDataProvider()
    orchestrator  = PortfolioOrchestrator(
        model_config=model_config,
        ips_config=ips_config,
        enhanced_data_provider=data_provider,
        openai_client=openai_client,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )

    print(f"\nRunning {len(suite)} stocks  (date: {analysis_date})\n")

    all_results: list = []
    results_by_tier: dict = {t: [] for t in TIER_ORDER}

    for ticker, tier, desc in suite:
        phases, err = run_timed_analysis(orchestrator, ticker, analysis_date)
        if err:
            print(f"  ✗  {ticker} [{tier}] FAILED: {err}")
            continue

        record = {"ticker": ticker, "tier": tier, "desc": desc, "phases": phases}
        all_results.append(record)
        results_by_tier.setdefault(tier, []).append(record)

    if not all_results:
        print("All analyses failed — no timing data collected.")
        sys.exit(1)

    print_results_table(all_results)
    calibration = print_tier_summary(results_by_tier)
    print_calibration_block(calibration)

    update_tier_times(results_by_tier)
    print(f"  ✓  Per-tier timing data written to {STEP_TIMES_PATH}")
    print(f"     Run with --dry-run to verify the next test plan.\n")
