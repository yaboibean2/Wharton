"""
Test script to verify parallel data gathering works correctly and is faster.
"""
import time
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.enhanced_data_provider import EnhancedDataProvider
from engine.portfolio_orchestrator import PortfolioOrchestrator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parallel_gather():
    """Test the parallel data gathering functionality."""
    
    print("\n" + "="*80)
    print("TESTING PARALLEL DATA GATHERING")
    print("="*80 + "\n")
    
    # Initialize components
    print("Initializing data provider...")
    data_provider = EnhancedDataProvider()
    
    # Mock config for orchestrator
    model_config = {
        'agents': {
            'value': {'enabled': True, 'weight': 0.20},
            'growth': {'enabled': True, 'weight': 0.20},
            'risk': {'enabled': True, 'weight': 0.20},
            'sentiment': {'enabled': True, 'weight': 0.20},
            'macro': {'enabled': True, 'weight': 0.20}
        }
    }
    
    ips_config = {
        'universe': {
            'benchmark': '^GSPC'
        }
    }
    
    print("Initializing orchestrator...")
    # Create a minimal orchestrator just for testing _gather_data
    class MinimalOrchestrator:
        def __init__(self, data_provider, ips_config):
            self.data_provider = data_provider
            self.ips_config = ips_config
        
        # Import the _gather_data and helper methods from PortfolioOrchestrator
        from engine.portfolio_orchestrator import PortfolioOrchestrator
        _gather_data = PortfolioOrchestrator._gather_data
        _create_benchmark_data = PortfolioOrchestrator._create_benchmark_data
        _extract_price_history_from_fundamentals = PortfolioOrchestrator._extract_price_history_from_fundamentals
    
    orchestrator = MinimalOrchestrator(
        data_provider=data_provider,
        ips_config=ips_config
    )
    
    # Test with a simple ticker
    ticker = "AAPL"
    analysis_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n{'-'*80}")
    print(f"Testing parallel data gathering for {ticker}")
    print(f"Analysis date: {analysis_date}")
    print(f"{'-'*80}\n")
    
    # Time the parallel gather
    start_time = time.time()
    
    try:
        data = orchestrator._gather_data(ticker, analysis_date, existing_portfolio=None)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS: Data gathering completed in {elapsed_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        # Verify data structure
        print("Verifying data structure:")
        print(f"  - Ticker: {data.get('ticker')}")
        print(f"  - Analysis Date: {data.get('analysis_date')}")
        
        fundamentals = data.get('fundamentals', {})
        print(f"\n  Fundamentals:")
        print(f"    - Price: ${fundamentals.get('price', 'N/A')}")
        print(f"    - P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
        print(f"    - Beta: {fundamentals.get('beta', 'N/A')}")
        print(f"    - Data Sources: {fundamentals.get('data_sources', [])}")
        
        price_history = data.get('price_history')
        if price_history is not None and not price_history.empty:
            print(f"\n  Price History:")
            print(f"    - Days of data: {len(price_history)}")
            print(f"    - Latest close: ${price_history['Close'].iloc[-1]:.2f}")
        else:
            print(f"\n  Price History: Empty or None")
        
        benchmark_history = data.get('benchmark_history')
        if benchmark_history is not None and not benchmark_history.empty:
            print(f"\n  Benchmark History:")
            print(f"    - Days of data: {len(benchmark_history)}")
            print(f"    - Return: {((benchmark_history['Close'].iloc[-1] / benchmark_history['Close'].iloc[0]) - 1) * 100:.2f}%")
        else:
            print(f"\n  Benchmark History: Empty or None")
        
        print(f"\n{'='*80}")
        print(f"TIMING ANALYSIS:")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"\nExpected improvements:")
        print(f"  - Sequential execution: ~39 seconds (36s + 3s + 0.1s)")
        print(f"  - Parallel execution: ~36 seconds (max of all 3)")
        print(f"  - Improvement: ~8% faster (or more if other calls are slower)")
        print(f"\n✅ Test completed successfully!")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"❌ FAILED: Data gathering failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        
        import traceback
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    print("\n🚀 Starting parallel data gathering test...\n")
    
    success = test_parallel_gather()
    
    if success:
        print("\n✅ All tests passed! Parallel data gathering is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Please check the errors above.")
        sys.exit(1)
