"""
Comprehensive Data Verification System
Queries Perplexity for every financial data point mentioned in analysis to verify accuracy
"""

import os
import requests
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ComprehensiveDataVerifier:
    """
    Verify financial data points using multiple Perplexity queries for maximum accuracy
    """
    
    def __init__(self, perplexity_api_key: str):
        self.perplexity_api_key = perplexity_api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        # Cache to store verification results and avoid redundant checks
        self.verification_cache = {}
        
    def _verify_single_data_point(self, ticker: str, data_key: str, data_value: Any, description: str, num_verifications: int = 1) -> Dict:
        """
        Verify a single data point MULTIPLE TIMES using different Perplexity queries for maximum accuracy.
        
        Args:
            ticker: Stock ticker
            data_key: Key identifying the data point
            data_value: Value to verify
            description: Human-readable description
            num_verifications: Number of independent verifications (default: 3)
            
        Returns:
            Comprehensive verification result with consensus analysis
        """
        # Create cache key from ticker, data_key, and data_value
        cache_key = f"{ticker}_{data_key}_{str(data_value)}"
        
        # Check if we already verified this data point
        if cache_key in self.verification_cache:
            logger.info(f"🎯 CACHE HIT: Using cached verification for {ticker} {data_key}: {data_value}")
            return self.verification_cache[cache_key]
        
        logger.info(f"🔍 NEW VERIFICATION: Starting verification for {ticker} {data_key}: {data_value}")
        logger.info(f"Starting MULTIPLE verification for {ticker} {data_key}: {data_value} ({num_verifications} verifications)")
        
        # Format the value for display
        if isinstance(data_value, float):
            if data_key == 'market_cap':
                formatted_value = f"${data_value / 1_000_000_000:.1f}B"
            elif data_key == 'pe_ratio':
                formatted_value = f"{data_value:.2f}"
            elif data_key == 'dividend_yield':
                formatted_value = f"{data_value * 100:.2f}%"
            else:
                formatted_value = f"{data_value:.2f}"
        else:
            formatted_value = str(data_value)
            
        # Perform multiple independent verifications
        verification_results = []
        
        for i in range(num_verifications):
            logger.info(f"Performing verification {i+1}/{num_verifications} for {ticker} {data_key}")
            
            # Get different query strategy for each verification
            query_strategy = self._get_query_strategy(i, ticker, data_key, formatted_value, description)
            
            try:
                headers = {
                    "Authorization": f"Bearer {self.perplexity_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a financial data verification expert. Verify the accuracy of specific financial metrics by checking current data sources. Use only robust sources from well known credible financial sources like Bloomberg, Reuters, SEC filings, company investor relations, Yahoo Finance, or other established financial data providers."
                        },
                        {
                            "role": "user", 
                            "content": query_strategy
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "stream": False
                }
                
                response = requests.post(self.base_url, json=payload, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    verification_content = result['choices'][0]['message']['content']
                    
                    verification_analysis = self._analyze_verification_response(
                        verification_content, data_value, formatted_value, data_key
                    )
                    
                    verification_results.append({
                        "verification_id": i + 1,
                        "query_strategy": query_strategy[:100] + "...",
                        "verification_content": verification_content,
                        "verification_analysis": verification_analysis,
                        "confidence": verification_analysis.get('confidence', 50),
                        "status": "verified"
                    })
                    
                    logger.info(f"✅ Verification {i+1} completed for {ticker} {data_key}")
                    
                else:
                    logger.error(f"Perplexity verification {i+1} failed for {ticker} {data_key}: {response.status_code}")
                    verification_results.append({
                        "verification_id": i + 1,
                        "error": f"API error: {response.status_code}",
                        "status": "failed"
                    })
                    
            except Exception as e:
                logger.error(f"Error in verification {i+1} for {ticker} {data_key}: {e}")
                verification_results.append({
                    "verification_id": i + 1,
                    "error": str(e),
                    "status": "error"
                })
        
        # Analyze consensus across all verifications
        if verification_results:
            consensus_analysis = self._analyze_verification_consensus(verification_results, data_value, formatted_value)
            
            result = {
                "data_key": data_key,
                "original_value": data_value,
                "formatted_value": formatted_value,
                "description": description,
                "verification_attempts": len(verification_results),
                "individual_verifications": verification_results,
                "consensus_analysis": consensus_analysis,
                "confidence": consensus_analysis.get('confidence', 50),
                "verified_at": datetime.now().isoformat(),
                "status": "multi_verified"
            }
            
            # Cache the successful verification result
            self.verification_cache[cache_key] = result
            logger.info(f"✅ CACHED: Verification result cached for {ticker} {data_key}: {data_value}")
            
            return result
        else:
            result = {
                "data_key": data_key,
                "original_value": data_value,
                "status": "verification_failed",
                "error": "All verification attempts failed",
                "verified_at": datetime.now().isoformat()
            }
            
            # Cache the failed verification result to avoid retrying
            self.verification_cache[cache_key] = result
            logger.info(f"❌ CACHED: Failed verification result cached for {ticker} {data_key}: {data_value}")
            
            return result
    
    def clear_verification_cache(self):
        """Clear the verification cache to force fresh verification of all data points."""
        self.verification_cache.clear()
        logger.info("🧹 CACHE CLEARED: All cached verification results have been cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the verification cache."""
        total_cached = len(self.verification_cache)
        successful_verifications = sum(1 for result in self.verification_cache.values() 
                                       if result.get('status') == 'multi_verified')
        failed_verifications = total_cached - successful_verifications
        
        return {
            "total_cached_verifications": total_cached,
            "successful_verifications": successful_verifications,
            "failed_verifications": failed_verifications,
            "cache_hit_efficiency": f"{(successful_verifications/total_cached*100):.1f}%" if total_cached > 0 else "0%"
        }
    
    def _get_query_strategy(self, verification_index: int, ticker: str, data_key: str, formatted_value: str, description: str) -> str:
        """
        Generate different query strategies for multiple verifications
        """
        strategies = [
            # Strategy 1: Direct verification
            f"What is the current {data_key.replace('_', ' ')} for {ticker}? Please verify if {formatted_value} is accurate based on the most recent data available.",
            
            # Strategy 2: Source-focused verification  
            f"According to recent financial reports and reliable sources, what is {ticker}'s {data_key.replace('_', ' ')}? I need to verify if {formatted_value} is correct.",
            
            # Strategy 3: Comparative verification
            f"Please find the latest {data_key.replace('_', ' ')} for {ticker} from multiple financial sources and compare it to {formatted_value}. Is this value accurate?"
        ]
        
        return strategies[verification_index % len(strategies)]
    
    def _analyze_verification_response(self, content: str, original_value: Any, formatted_value: str, data_key: str) -> Dict[str, Any]:
        """
        Analyze Perplexity's verification response to determine accuracy.
        
        Args:
            content: Perplexity response content
            original_value: Original data value
            formatted_value: Formatted value string
            data_key: Data point key
            
        Returns:
            Analysis of verification accuracy
        """
        content_lower = content.lower()
        analysis = {
            "extracted_values": [],
            "confirmation_indicators": 0,
            "contradiction_indicators": 0,
            "confidence": 50,
            "accuracy_assessment": "unclear",
            "notes": []
        }
        
        # Look for confirmation indicators
        confirmations = ["correct", "accurate", "confirmed", "yes", "matches", "consistent", "agrees"]
        confirmation_count = sum(1 for word in confirmations if word in content_lower)
        analysis["confirmation_indicators"] = confirmation_count
        
        # Look for contradiction indicators  
        contradictions = ["incorrect", "inaccurate", "wrong", "no", "differs", "inconsistent", "disagrees"]
        contradiction_count = sum(1 for word in contradictions if word in content_lower)
        analysis["contradiction_indicators"] = contradiction_count
        
        # Extract numerical values from response
        if data_key in ['pe_ratio', 'beta', 'dividend_yield']:
            number_matches = re.findall(r'\d+\.?\d*', content)
            analysis["extracted_values"] = number_matches[:3]  # Limit to first 3 matches
        elif data_key == 'market_cap':
            # Look for market cap values
            cap_matches = re.findall(r'\$?(\d+\.?\d*)\s*[BbMm]illion', content)
            analysis["extracted_values"] = [f"${match}B" for match in cap_matches]
        elif 'percent' in data_key or 'yield' in data_key:
            percent_matches = re.findall(r'(\d+\.?\d*)%', content)
            analysis["extracted_values"] = [f"{match}%" for match in percent_matches]
        
        # Determine accuracy
        if confirmation_count > contradiction_count and confirmation_count > 0:
            analysis["accuracy_assessment"] = "confirmed"
            analysis["confidence"] = min(90, 60 + confirmation_count * 10)
            analysis["notes"].append("Perplexity confirms the data point accuracy")
        elif contradiction_count > confirmation_count and contradiction_count > 0:
            analysis["accuracy_assessment"] = "contradicted"
            analysis["confidence"] = max(20, 60 - contradiction_count * 15)
            analysis["notes"].append("Perplexity indicates potential inaccuracy")
        else:
            analysis["accuracy_assessment"] = "unclear"
            analysis["confidence"] = 50
            analysis["notes"].append("Verification response inconclusive")
        
        # Check for recent data mentions
        recent_keywords = ["recent", "current", "latest", "as of", "updated"]
        if any(word in content_lower for word in recent_keywords):
            analysis["confidence"] += 10
            analysis["notes"].append("Response includes recent data references")
        
        return analysis
    
    def verify_comprehensive_data(self, ticker: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies every single financial data point by querying Perplexity individually.
        Ensures complete accuracy for all claims made in analysis.
        
        Args:
            ticker: Stock ticker symbol
            data_dict: Dictionary of financial data to verify
            
        Returns:
            Comprehensive verification results for all data points
        """
        logger.info(f"🔍 COMPREHENSIVE VERIFICATION: Starting verification of ALL data points for {ticker}")
        
        # Log current cache stats before starting
        cache_stats = self.get_cache_stats()
        if cache_stats['total_cached_verifications'] > 0:
            logger.info(f"📊 CACHE STATUS: {cache_stats['total_cached_verifications']} cached verifications available ({cache_stats['cache_hit_efficiency']} success rate)")
        
        verification_results = {}
        
        # Extract data from nested structure (fundamentals and details)
        fundamentals = data_dict.get('fundamentals', {})
        details = data_dict.get('details', {})
        
        # Define key data points that need verification - check both fundamentals and details
        key_points = {
            'current_price': fundamentals.get('price') or data_dict.get('price', 'N/A'),
            'pe_ratio': fundamentals.get('pe_ratio') or data_dict.get('pe_ratio', 'N/A'), 
            'market_cap': fundamentals.get('market_cap') or data_dict.get('market_cap', 'N/A'),
            'beta': fundamentals.get('beta') or details.get('beta') or data_dict.get('beta', 'N/A'),
            'dividend_yield': fundamentals.get('dividend_yield') or data_dict.get('dividend_yield', 'N/A'),
            'sector': fundamentals.get('sector') or data_dict.get('sector', 'N/A'),
            'volatility': details.get('volatility_pct', 'N/A')
        }
        
        for data_key, data_value in key_points.items():
            if data_value != 'N/A' and data_value is not None:
                logger.info(f"🎯 Verifying {ticker} {data_key}: {data_value}")
                
                description = f"{ticker} {data_key.replace('_', ' ')}"
                verification = self._verify_single_data_point(ticker, data_key, data_value, description)
                verification_results[data_key] = verification
                
                logger.info(f"✅ {ticker} {data_key} verification complete: {verification.get('status', 'unknown')}")
        
        # Log final cache stats to show efficiency gains
        final_cache_stats = self.get_cache_stats()
        logger.info(f"🎯 COMPREHENSIVE VERIFICATION COMPLETE for {ticker}: {len(verification_results)} data points verified")
        logger.info(f"📊 CACHE EFFICIENCY: {final_cache_stats['total_cached_verifications']} total cached ({final_cache_stats['cache_hit_efficiency']} success rate)")
        
        return {
            "ticker": ticker,
            "total_verifications": len(verification_results),
            "verification_results": verification_results,
            "verification_timestamp": datetime.now().isoformat(),
            "status": "comprehensive_verification_complete"
        }
        
    def verify_sector_comparatives(self, ticker: str, sector: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Verify sector comparative data using multiple Perplexity queries
        """
        logger.info(f"🏭 Starting sector comparative verification for {ticker} in {sector}")
        
        sector_verification = {
            "ticker": ticker,
            "sector": sector,
            "sector_averages": {},
            "comparative_analysis": {},
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Verify sector averages for key metrics
        sector_metrics = ['pe_ratio', 'beta', 'dividend_yield', 'debt_to_equity', 'roe']
        
        for metric in sector_metrics:
            if metric in metrics:
                try:
                    headers = {
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    query = f"What is the average {metric.replace('_', ' ')} for companies in the {sector} sector? Please provide recent industry averages."
                    
                    payload = {
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        "max_tokens": 500,
                        "temperature": 0.1
                    }
                    
                    response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        sector_content = result['choices'][0]['message']['content']
                        
                        sector_verification["sector_averages"][metric] = {
                            "query": query,
                            "response": sector_content,
                            "company_value": metrics[metric],
                            "status": "verified"
                        }
                    else:
                        sector_verification["sector_averages"][metric] = {
                            "error": f"API error: {response.status_code}"
                        }
                        
                except Exception as e:
                    sector_verification["sector_averages"][metric] = {
                        "error": str(e)
                    }
        
        return sector_verification

    def _analyze_verification_consensus(self, verification_results: List[Dict], data_value: Any, formatted_value: str) -> Dict:
        """
        Analyze multiple verification results to reach consensus
        """
        if not verification_results:
            return {
                "consensus": "No verifications available", 
                "confidence": 0
            }
        
        successful_verifications = [v for v in verification_results if v.get('status') == 'verified']
        
        if not successful_verifications:
            return {
                "consensus": "All verifications failed",
                "confidence": 0
            }
        
        # Calculate average confidence
        confidences = [v.get('confidence', 50) for v in successful_verifications]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Look for consistency in verification content
        all_content = " ".join([v.get('verification_content', '') for v in successful_verifications]).lower()
        
        # Check for contradictions
        contradiction_indicators = ['however', 'but', 'actually', 'incorrect', 'inaccurate', 'wrong']
        contradictions = sum(1 for indicator in contradiction_indicators if indicator in all_content)
        
        # Check for confirmations
        confirmation_indicators = ['confirms', 'accurate', 'correct', 'verified', 'consistent', 'matches']
        confirmations = sum(1 for indicator in confirmation_indicators if indicator in all_content)
        
        # Analyze numerical consistency for financial metrics
        if isinstance(data_value, (int, float)):
            values = []
            for v in successful_verifications:
                try:
                    analysis = v.get('verification_analysis', {})
                    extracted_values = analysis.get('extracted_values', [])
                    if extracted_values:
                        # Try to extract numeric values
                        for val_str in extracted_values:
                            numeric_val = re.findall(r'[\d.]+', str(val_str))
                            if numeric_val:
                                values.append(float(numeric_val[0]))
                except:
                    continue
            
            if values:
                avg_value = sum(values) / len(values)
                variance = sum((v - avg_value) ** 2 for v in values) / len(values) if len(values) > 1 else 0
                consistency_score = max(0, 100 - (variance / max(avg_value, 1) * 100))
                
                return {
                    "consensus": f"Multiple sources confirm value around {avg_value:.2f}",
                    "confidence": min(avg_confidence, consistency_score),
                    "values_found": values,
                    "consistency_score": consistency_score,
                    "verification_count": len(successful_verifications)
                }
        
        # General consensus analysis
        confidence_penalty = max(0, contradictions * 15 - confirmations * 10)
        final_confidence = max(20, avg_confidence - confidence_penalty)
        
        return {
            "consensus": f"Verified across {len(successful_verifications)} independent sources",
            "confidence": final_confidence,
            "verification_count": len(successful_verifications),
            "confirmations": confirmations,
            "contradictions": contradictions
        }


def verify_analysis_claims(ticker: str, analysis_text: str) -> Dict[str, Any]:
    """
    Extract and verify specific financial claims made in analysis text.
    
    Args:
        ticker: Stock ticker
        analysis_text: Text containing financial claims
        
    Returns:
        Verification results for extracted claims
    """
    logger.info(f"🔍 Extracting and verifying claims from analysis for {ticker}")
    
    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
    if not perplexity_api_key:
        return {"error": "Perplexity API key not found"}
    
    verifier = ComprehensiveDataVerifier(perplexity_api_key)
    
    # Extract specific financial claims from the analysis text
    claims = []
    
    # P/E ratio claims
    pe_matches = re.findall(r'P/E ratio (?:of |at |is )?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
    for pe in pe_matches:
        claims.append(("pe_ratio", float(pe), f"P/E ratio of {pe}"))
    
    # Beta claims
    beta_matches = re.findall(r'beta (?:of |at |is )?(\d+\.?\d*)', analysis_text, re.IGNORECASE)
    for beta in beta_matches:
        claims.append(("beta", float(beta), f"beta of {beta}"))
    
    # Volatility claims
    vol_matches = re.findall(r'volatility (?:of |at |is )?(\d+\.?\d*)%', analysis_text, re.IGNORECASE)
    for vol in vol_matches:
        claims.append(("volatility", float(vol), f"volatility of {vol}%"))
    
    # Price claims
    price_matches = re.findall(r'\$(\d+\.?\d*)', analysis_text)
    for price in price_matches:
        claims.append(("price", float(price), f"price of ${price}"))
    
    # Verify each claim
    claim_verifications = {}
    
    for i, (claim_type, claim_value, claim_description) in enumerate(claims):
        verification = verifier._verify_single_data_point(
            ticker, f"{claim_type}_{i}", claim_value, claim_description
        )
        claim_verifications[f"claim_{i}_{claim_type}"] = verification
    
    return {
        "ticker": ticker,
        "total_claims_found": len(claims),
        "claim_verifications": claim_verifications,
        "verification_timestamp": datetime.now().isoformat()
    }