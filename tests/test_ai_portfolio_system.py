"""
Test the AI-Powered Portfolio Recommendation System
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from engine.portfolio_orchestrator import PortfolioOrchestrator
from data.enhanced_data_provider import EnhancedDataProvider
from utils.config_loader import get_config_loader
from openai import OpenAI

# Load environment
load_dotenv()

print("=" * 80)
print("AI-POWERED PORTFOLIO RECOMMENDATION SYSTEM - TEST")
print("=" * 80)

# Check API keys
print("\n1. Checking API Keys...")
openai_key = os.getenv('OPENAI_API_KEY')
perplexity_key = os.getenv('PERPLEXITY_API_KEY')

if openai_key:
    print(f"   ✅ OpenAI API Key: {openai_key[:10]}...")
else:
    print("   ❌ OpenAI API Key: NOT FOUND")

if perplexity_key:
    print(f"   ✅ Perplexity API Key: {perplexity_key[:10]}...")
else:
    print("   ❌ Perplexity API Key: NOT FOUND")

if not (openai_key and perplexity_key):
    print("\n⚠️  Missing required API keys. Please set:")
    print("   OPENAI_API_KEY=your_key")
    print("   PERPLEXITY_API_KEY=your_key")
    sys.exit(1)

# Initialize clients
print("\n2. Initializing AI Clients...")
try:
    openai_client = OpenAI(api_key=openai_key)
    print("   ✅ OpenAI client initialized")
except Exception as e:
    print(f"   ❌ OpenAI client failed: {e}")
    sys.exit(1)

try:
    perplexity_client = OpenAI(
        api_key=perplexity_key,
        base_url="https://api.perplexity.ai"
    )
    print("   ✅ Perplexity client initialized")
except Exception as e:
    print(f"   ❌ Perplexity client failed: {e}")
    sys.exit(1)

# Initialize system components
print("\n3. Initializing System Components...")
try:
    config_loader = get_config_loader()
    model_config = config_loader.load_model_config()
    ips_config = config_loader.load_ips()
    data_provider = EnhancedDataProvider()
    print("   ✅ Config and data provider initialized")
except Exception as e:
    print(f"   ❌ Component initialization failed: {e}")
    sys.exit(1)

# Initialize orchestrator
print("\n4. Initializing Portfolio Orchestrator...")
try:
    orchestrator = PortfolioOrchestrator(
        model_config=model_config,
        ips_config=ips_config,
        enhanced_data_provider=data_provider,
        openai_client=openai_client,
        perplexity_client=perplexity_client
    )
    print("   ✅ Portfolio Orchestrator initialized")
    
    if orchestrator.ai_selector:
        print("   ✅ AI Portfolio Selector ready")
    else:
        print("   ❌ AI Portfolio Selector not available")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Orchestrator initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL SYSTEMS READY FOR AI-POWERED PORTFOLIO GENERATION")
print("=" * 80)

# Test with manual tickers (skip AI selection for quick test)
print("\n5. Testing Portfolio Generation (Manual Tickers)...")
test_challenge = """
Generate a balanced technology portfolio focusing on established leaders
with strong fundamentals and growth potential. Target moderate risk profile.
"""

try:
    result = orchestrator.recommend_portfolio(
        challenge_context=test_challenge,
        tickers=["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        num_positions=5
    )
    
    print("\n✅ Portfolio Generation Successful!")
    print(f"   → Portfolio: {len(result['portfolio'])} positions")
    print(f"   → Average Score: {result['summary']['avg_score']:.1f}")
    print(f"   → Selection Method: {result['summary']['selection_method']}")
    
    print("\n   Holdings:")
    for holding in result['portfolio']:
        print(f"     • {holding['ticker']}: Score {holding['final_score']:.1f}, Weight {holding['target_weight_pct']:.1f}%")
    
except Exception as e:
    print(f"\n❌ Portfolio generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("🎉 TEST COMPLETE - SYSTEM FULLY FUNCTIONAL")
print("=" * 80)
print("\nNext Steps:")
print("1. Run the Streamlit app: streamlit run app.py")
print("2. Navigate to 'Portfolio Recommendations'")
print("3. Try AI-Powered Selection mode")
print("4. Review selection logs in portfolio_selection_logs/")
print("\n" + "=" * 80)
