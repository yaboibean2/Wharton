#!/usr/bin/env python3
"""
Pipeline Timing Test
====================
Runs full stock analyses (using the real parallel pipeline) on random tickers
and logs per-phase wall-clock times to data/step_times.json.

The saved timings are used by the Streamlit app to calibrate the progress bar
and "time remaining" countdown.

Usage:
    python test_pipeline_timing.py                     # 3 default tickers
    python test_pipeline_timing.py AAPL MSFT TSLA      # specific tickers
    python test_pipeline_timing.py --random 5           # 5 random from universe
"""

import json
import os
import random
import sys
import time
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("timing_test")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
from openai import OpenAI
from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator
from utils.config_loader import get_config_loader

STEP_TIMES_PATH = os.path.join(os.path.dirname(__file__), "data", "step_times.json")

# Default universe of well-known liquid tickers to draw randoms from
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "LLY", "WMT", "MCD", "TMO",
    "CSCO", "ACN", "ABT", "CRM", "NKE", "DHR", "TXN", "NEE", "UPS",
    "PM", "INTC", "AMD", "QCOM", "LOW", "BA", "CAT", "GS", "SPGI",
    "AMAT", "ISRG", "BKNG", "ADP", "MDLZ",
]


def load_step_times() -> dict:
    """Load existing step_times.json or return empty structure."""
    if os.path.exists(STEP_TIMES_PATH):
        with open(STEP_TIMES_PATH, "r") as f:
            return json.load(f)
    return {"step_times": {}, "metadata": {}}


def save_step_times(data: dict):
    """Persist step_times.json."""
    os.makedirs(os.path.dirname(STEP_TIMES_PATH), exist_ok=True)
    with open(STEP_TIMES_PATH, "w") as f:
        json.dump(data, f, indent=2)


def run_timed_analysis(orchestrator, ticker, analysis_date):
    """
    Run a full analysis through the real orchestrator pipeline and capture
    wall-clock seconds for each phase:
        Phase 0  : Init (negligible)
        Phase 1  : Data gathering
        Phase 2  : Agents (parallel)
        Phase 3  : Score blending / finalize
        Total    : End-to-end
    """
    logger.info(f"▶  {ticker}")

    # Milestone log (populated by progress_callback)
    milestones = []

    def _cb(pct, msg):
        milestones.append((time.time(), pct, msg))

    wall_start = time.time()

    try:
        result = orchestrator.analyze_stock(
            ticker=ticker,
            analysis_date=analysis_date,
            progress_callback=_cb,
        )
    except Exception as e:
        logger.error(f"   ✗  {ticker} failed: {e}")
        return None

    wall_end = time.time()
    total_time = wall_end - wall_start

    if "error" in result:
        logger.error(f"   ✗  {ticker}: {result['error']}")
        return None

    # --- Derive phase timings from milestones ---
    # Phase 1 ends at first milestone with pct >= 42
    # Phase 2 ends at first milestone with pct >= 98
    # Phase 3 ends at pct >= 100

    phase_times = {
        "data_gather": None,
        "agents_parallel": None,
        "blend": None,
        "total": total_time,
    }

    t0 = milestones[0][0] if milestones else wall_start
    t_data_done = None
    t_agents_done = None
    t_blend_done = None

    for ts, pct, msg in milestones:
        if pct >= 42 and t_data_done is None:
            t_data_done = ts
        if pct >= 98 and t_agents_done is None:
            t_agents_done = ts
        if pct >= 100 and t_blend_done is None:
            t_blend_done = ts

    if t_data_done:
        phase_times["data_gather"] = t_data_done - t0
    if t_data_done and t_agents_done:
        phase_times["agents_parallel"] = t_agents_done - t_data_done
    if t_agents_done and t_blend_done:
        phase_times["blend"] = t_blend_done - t_agents_done

    score = result.get("final_score", "?")
    rec = result.get("recommendation", "?")

    logger.info(
        f"   ✓  {ticker}  score={score:.1f}  rec={rec}  "
        f"total={total_time:.1f}s  "
        f"data={phase_times['data_gather']:.1f}s  "
        f"agents={phase_times['agents_parallel']:.1f}s  "
        f"blend={phase_times['blend']:.1f}s"
    )

    return {
        "ticker": ticker,
        "phases": phase_times,
        "milestones": [(round(ts - t0, 2), pct, msg) for ts, pct, msg in milestones],
    }


def update_step_times_from_results(all_results):
    """
    Merge measured timings into data/step_times.json so the app can use them
    for progress estimation.

    Mapping to step_times keys (matching the orchestrator's progress percentages):
        step 1 → data gathering (0→42%)
        step 2 → agent phase (42→98%)  — parallel wall-clock
        step 3 → blend/finalize (98→100%)
    """
    store = load_step_times()
    st = store.setdefault("step_times", {})

    for r in all_results:
        ph = r["phases"]
        if ph["data_gather"] is not None:
            st.setdefault("1", []).append(round(ph["data_gather"], 3))
        if ph["agents_parallel"] is not None:
            st.setdefault("2", []).append(round(ph["agents_parallel"], 3))
        if ph["blend"] is not None:
            st.setdefault("3", []).append(round(ph["blend"], 3))
        if ph["total"] is not None:
            st.setdefault("total", []).append(round(ph["total"], 3))

    # Cap each list at 100 most recent entries
    for k in st:
        if len(st[k]) > 100:
            st[k] = st[k][-100:]

    store["metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "total_samples": sum(len(v) for v in st.values()),
    }

    save_step_times(store)
    logger.info(f"Saved step times to {STEP_TIMES_PATH}")


def print_summary(all_results):
    """Print summary table."""
    print(f"\n{'='*70}")
    print(f"  PIPELINE TIMING RESULTS")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'Data':>8} {'Agents':>8} {'Blend':>8} {'Total':>8}  Score")
    print(f"  {'-'*56}")

    for r in all_results:
        ph = r["phases"]
        ticker = r["ticker"]
        data_s = f"{ph['data_gather']:.1f}s" if ph["data_gather"] else "N/A"
        agent_s = f"{ph['agents_parallel']:.1f}s" if ph["agents_parallel"] else "N/A"
        blend_s = f"{ph['blend']:.1f}s" if ph["blend"] else "N/A"
        total_s = f"{ph['total']:.1f}s"
        # Get score from milestones (last one that says "complete")
        score = "?"
        for _, pct, msg in reversed(r["milestones"]):
            if "complete" in msg.lower() and "/" in msg:
                try:
                    score = msg.split(":")[1].split("/")[0].strip()
                except:
                    pass
                break
        print(f"  {ticker:<8} {data_s:>8} {agent_s:>8} {blend_s:>8} {total_s:>8}  {score}")

    if len(all_results) > 1:
        print(f"  {'-'*56}")
        avg_data = sum(r["phases"]["data_gather"] or 0 for r in all_results) / len(all_results)
        avg_agents = sum(r["phases"]["agents_parallel"] or 0 for r in all_results) / len(all_results)
        avg_blend = sum(r["phases"]["blend"] or 0 for r in all_results) / len(all_results)
        avg_total = sum(r["phases"]["total"] for r in all_results) / len(all_results)
        print(f"  {'AVG':<8} {avg_data:>7.1f}s {avg_agents:>7.1f}s {avg_blend:>7.1f}s {avg_total:>7.1f}s")
        print()
        print(f"  Use these averages to calibrate _expected_elapsed_at() in app.py:")
        print(f"    Data phase:   ~{avg_data:.0f}s  (0→42%)")
        print(f"    Agent phase:  ~{avg_agents:.0f}s  (42→98%, parallel)")
        print(f"    Blend phase:  ~{avg_blend:.0f}s  (98→100%)")
        print(f"    Total:        ~{avg_total:.0f}s")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse args
    args = sys.argv[1:]
    tickers = []

    if "--random" in args:
        idx = args.index("--random")
        n = int(args[idx + 1]) if idx + 1 < len(args) else 3
        tickers = random.sample(DEFAULT_UNIVERSE, min(n, len(DEFAULT_UNIVERSE)))
        logger.info(f"Random tickers: {', '.join(tickers)}")
    elif args:
        tickers = [t.upper().strip() for t in args]
    else:
        tickers = ["AAPL", "GOOGL", "NVDA"]

    analysis_date = datetime.now().strftime("%Y-%m-%d")

    # Build orchestrator
    config_loader = get_config_loader()
    model_config = config_loader.load_model_config()
    ips_config = config_loader.load_ips()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    data_provider = EnhancedDataProvider()

    orchestrator = PortfolioOrchestrator(
        model_config=model_config,
        ips_config=ips_config,
        enhanced_data_provider=data_provider,
        openai_client=openai_client,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )

    logger.info(f"Testing {len(tickers)} tickers: {', '.join(tickers)}")
    logger.info(f"Analysis date: {analysis_date}")
    print()

    all_results = []
    for ticker in tickers:
        r = run_timed_analysis(orchestrator, ticker, analysis_date)
        if r:
            all_results.append(r)

    if all_results:
        print_summary(all_results)
        update_step_times_from_results(all_results)
        logger.info("Done! Step times saved — the app will use these for progress estimation.")
    else:
        logger.error("All analyses failed — no timing data collected.")
