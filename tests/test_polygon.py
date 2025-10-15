#!/usr/bin/env python3
"""
Test script to verify Polygon.io API is working correctly.
Run this to diagnose price fetching issues.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_polygon_api():
    """Test Polygon.io API with a few sample tickers."""
    
    polygon_key = os.getenv('POLYGON_API_KEY')
    
    print("=" * 80)
    print("POLYGON.IO API TEST")
    print("=" * 80)
    print()
    
    if not polygon_key:
        print("❌ ERROR: POLYGON_API_KEY not found in .env file")
        print("Please add your Polygon API key to the .env file")
        return
    
    print(f"✅ API Key found: {polygon_key[:10]}...")
    print()
    
    # Test tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    print()
    
    success_count = 0
    
    for ticker in test_tickers:
        print(f"Fetching {ticker}...")
        
        try:
            # Polygon.io API endpoint for previous close
            url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={polygon_key}'
            
            response = requests.get(url, timeout=10)
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"  ❌ HTTP Error: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                continue
            
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                price = float(data['results'][0]['c'])
                volume = data['results'][0].get('v', 0)
                timestamp = data['results'][0].get('t', 0)
                
                print(f"  ✅ Success!")
                print(f"     Price: ${price:.2f}")
                print(f"     Volume: {volume:,}")
                print(f"     Timestamp: {timestamp}")
                success_count += 1
            else:
                error = data.get('error', data.get('message', 'Unknown error'))
                print(f"  ❌ API Error: {error}")
                print(f"  Full response: {data}")
        
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        
        print()
    
    print("=" * 80)
    print(f"RESULTS: {success_count}/{len(test_tickers)} successful")
    print("=" * 80)
    
    if success_count == len(test_tickers):
        print("✅ All tests passed! Polygon.io API is working correctly.")
        print("Your price fetching should work in the main app.")
    elif success_count > 0:
        print("⚠️  Partial success. Some tickers worked, others failed.")
        print("This might be a rate limit or ticker availability issue.")
    else:
        print("❌ All tests failed. Check your API key and network connection.")
        print()
        print("Troubleshooting steps:")
        print("1. Verify your API key at: https://polygon.io/dashboard/api-keys")
        print("2. Check if your account is active and not rate-limited")
        print("3. Try accessing: https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__prev")

if __name__ == '__main__':
    test_polygon_api()
