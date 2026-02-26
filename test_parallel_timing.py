"""Timing test for parallel agent execution."""
import time, os, sys, yaml
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator

# Load configs
with open('config/model.yaml') as f:
    model_config = yaml.safe_load(f)
with open('config/ips.yaml') as f:
    ips_config = yaml.safe_load(f)

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
data_provider = EnhancedDataProvider()
orch = PortfolioOrchestrator(model_config, ips_config, data_provider, openai_client)

milestones = []
def progress_cb(pct, msg):
    milestones.append((time.time(), pct, msg))

print('Starting timing test with AAPL...')
start = time.time()
try:
    result = orch.analyze_stock(ticker='AAPL', progress_callback=progress_cb)
    elapsed = time.time() - start
    print(f'\nTotal time: {elapsed:.1f}s')
    print(f'Final score: {result.get("final_score", "N/A")}')
    print(f'\nMilestone timeline:')
    for t, pct, msg in milestones:
        rel = t - start
        print(f'  {rel:5.1f}s  [{pct:3.0f}%]  {msg[:80]}')

    # Identify phase boundaries
    data_end = next((t - start for t, p, m in milestones if p >= 42), None)
    agents_end = next((t - start for t, p, m in milestones if p >= 98), None)
    print(f'\nPhase timings:')
    if data_end:
        print(f'  Data gathering: {data_end:.1f}s (estimated: 35s)')
    if data_end and agents_end:
        print(f'  Agent phase:    {agents_end - data_end:.1f}s (estimated: 12s)')
    if agents_end:
        print(f'  Finalization:   {elapsed - agents_end:.1f}s (estimated: 1s)')
except Exception as e:
    elapsed = time.time() - start
    print(f'Error after {elapsed:.1f}s: {e}')
    import traceback
    traceback.print_exc()
