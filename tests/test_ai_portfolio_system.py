"""Live integration test for the AI-powered portfolio recommendation system."""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator
from utils.config_loader import get_config_loader

load_dotenv(PROJECT_ROOT / ".env")


def _missing_required_keys() -> list[str]:
    required = ["OPENAI_API_KEY", "PERPLEXITY_API_KEY", "GEMINI_API_KEY"]
    return [key for key in required if not os.getenv(key)]


def run_ai_portfolio_system_test() -> dict:
    """Run the full live portfolio generation flow and return the result."""
    print("=" * 80)
    print("AI-POWERED PORTFOLIO RECOMMENDATION SYSTEM - TEST")
    print("=" * 80)

    missing_keys = _missing_required_keys()
    if missing_keys:
        raise RuntimeError(f"Missing required API keys: {', '.join(missing_keys)}")

    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    print("\n1. Checking API Keys...")
    print(f"   PASS: OpenAI API Key: {openai_key[:10]}...")
    print(f"   PASS: Perplexity API Key: {perplexity_key[:10]}...")
    print(f"   PASS: Gemini API Key: {gemini_api_key[:10]}...")

    print("\n2. Initializing AI Clients...")
    openai_client = OpenAI(api_key=openai_key)
    print("   PASS: OpenAI client initialized")
    print("   PASS: Perplexity API key available via environment")
    print("   PASS: Gemini API key available for AI selector")

    print("\n3. Initializing System Components...")
    config_loader = get_config_loader()
    model_config = config_loader.load_model_config()
    ips_config = config_loader.load_ips()
    data_provider = EnhancedDataProvider()
    print("   PASS: Config and data provider initialized")

    print("\n4. Initializing Portfolio Orchestrator...")
    orchestrator = PortfolioOrchestrator(
        model_config=model_config,
        ips_config=ips_config,
        enhanced_data_provider=data_provider,
        openai_client=openai_client,
        gemini_api_key=gemini_api_key,
    )
    print("   PASS: Portfolio Orchestrator initialized")
    assert orchestrator.ai_selector, "AI Portfolio Selector not available"
    print("   PASS: AI Portfolio Selector ready")

    print("\n" + "=" * 80)
    print("ALL SYSTEMS READY FOR AI-POWERED PORTFOLIO GENERATION")
    print("=" * 80)

    print("\n5. Testing Portfolio Generation (Manual Tickers)...")
    test_challenge = """
Generate a balanced technology portfolio focusing on established leaders
with strong fundamentals and growth potential. Target moderate risk profile.
"""

    result = orchestrator.recommend_portfolio(
        challenge_context=test_challenge,
        tickers=["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        num_positions=5,
    )

    print("\nPASS: Portfolio Generation Successful!")
    print(f"   -> Portfolio: {len(result['portfolio'])} positions")
    print(f"   -> Average Score: {result['summary']['avg_score']:.1f}")
    print(f"   -> Selection Method: {result['summary']['selection_method']}")

    print("\n   Holdings:")
    for holding in result["portfolio"]:
        print(
            f"     - {holding['ticker']}: "
            f"Score {holding['final_score']:.1f}, "
            f"Weight {holding['target_weight_pct']:.1f}%"
        )

    print("\n" + "=" * 80)
    print("TEST COMPLETE - SYSTEM FULLY FUNCTIONAL")
    print("=" * 80)
    return result


@pytest.mark.integration
@pytest.mark.live_api
def test_ai_portfolio_system_live():
    missing_keys = _missing_required_keys()
    if missing_keys:
        pytest.skip(f"Missing required API keys: {', '.join(missing_keys)}")

    result = run_ai_portfolio_system_test()
    assert result["portfolio"]
    assert len(result["portfolio"]) == 5
    assert "summary" in result


if __name__ == "__main__":
    run_ai_portfolio_system_test()
