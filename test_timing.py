"""
Test script to measure actual timing across multiple tickers.
"""
import os
import sys
import time
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

from openai import OpenAI
from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator
from utils.config_loader import get_config_loader

def run_timed_analysis(orchestrator, ticker):
    """Run analysis and measure per-phase timing."""
    print(f"\n--- {ticker} ---")

    # Phase 1: Data gathering
    t1 = time.time()
    data = orchestrator._gather_data(ticker, "2025-02-18")
    t1_elapsed = time.time() - t1

    if not data or not data.get('fundamentals'):
        print(f"  Data gathering: {t1_elapsed:.1f}s - NO DATA")
        return None

    price = data['fundamentals'].get('price', 'N/A')
    print(f"  Data gathering: {t1_elapsed:.1f}s (price: ${price})")

    # Phase 2: Each agent
    agent_times = {}
    for agent_name, agent in orchestrator.agents.items():
        t_start = time.time()
        try:
            result = agent.analyze(ticker, data)
            elapsed = time.time() - t_start
            agent_times[agent_name] = elapsed
            score = result.get('score', 'N/A')
            label = agent_name.replace('_agent', '')
            n_articles = len(result.get('details', {}).get('supporting_articles', []))

            # Check sentiment specifically
            if 'sentiment' in agent_name:
                details = result.get('details', {})
                n_articles = details.get('num_articles', 0)
                is_default = 'neutral default' in result.get('rationale', '').lower()
                status = "DEFAULT 50!" if is_default else f"OK ({n_articles} articles)"
                print(f"  {label}: {elapsed:.1f}s score={score} {status}")
            else:
                print(f"  {label}: {elapsed:.1f}s score={score} articles={n_articles}")
        except Exception as e:
            elapsed = time.time() - t_start
            agent_times[agent_name] = elapsed
            print(f"  {agent_name}: {elapsed:.1f}s FAILED: {e}")

    total_agents = sum(agent_times.values())
    total = t1_elapsed + total_agents
    print(f"  TOTAL: {total:.1f}s (data={t1_elapsed:.1f}s, agents={total_agents:.1f}s)")

    return {
        'ticker': ticker,
        'data_gather': t1_elapsed,
        'agents': total_agents,
        'agent_breakdown': agent_times,
        'total': total
    }

if __name__ == "__main__":
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
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "GOOGL", "NVDA"]

    results = []
    for ticker in tickers:
        r = run_timed_analysis(orchestrator, ticker)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        avg_data = sum(r['data_gather'] for r in results) / len(results)
        avg_agents = sum(r['agents'] for r in results) / len(results)
        avg_total = sum(r['total'] for r in results) / len(results)
        print(f"  Avg Data Gather: {avg_data:.1f}s")
        print(f"  Avg Agents:      {avg_agents:.1f}s")
        print(f"  Avg Total:       {avg_total:.1f}s")
        print(f"  Samples:         {len(results)}")
        print(f"{'='*60}")
