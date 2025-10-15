"""
Portfolio Orchestrator
Coordinates all agents and blends scores to generate final recommendations.
Handles position sizing and portfolio construction.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.value_agent import ValueAgent
from agents.growth_momentum_agent import GrowthMomentumAgent
from agents.macro_regime_agent import MacroRegimeAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.client_layer_agent import ClientLayerAgent
from agents.learning_agent import LearningAgent
from data.enhanced_data_provider import EnhancedDataProvider
from engine.ai_portfolio_selector import AIPortfolioSelector

logger = logging.getLogger(__name__)


class PortfolioOrchestrator:
    """
    Main orchestration engine for the multi-agent investment system.
    Coordinates data gathering, agent analysis, score blending, and portfolio construction.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        ips_config: Dict[str, Any],
        enhanced_data_provider: EnhancedDataProvider,
        openai_client=None,
        perplexity_client=None
    ):
        self.model_config = model_config
        self.ips_config = ips_config
        self.data_provider = enhanced_data_provider
        self.openai_client = openai_client
        self.perplexity_client = perplexity_client
        logger.info("Using Enhanced Data Provider with premium fallbacks")
        
        # Initialize agents with their dependencies
        self.agents = {}
        
        # Core analysis agents
        self.agents['value_agent'] = ValueAgent(model_config, openai_client)
        self.agents['growth_momentum_agent'] = GrowthMomentumAgent(model_config, openai_client)
        self.agents['macro_regime_agent'] = MacroRegimeAgent(model_config, openai_client)
        self.agents['risk_agent'] = RiskAgent(model_config, openai_client)
        self.agents['sentiment_agent'] = SentimentAgent(model_config, openai_client)
        
        # Meta agents
        self.agents['client_layer_agent'] = ClientLayerAgent(model_config, ips_config, openai_client)
        self.agents['learning_agent'] = LearningAgent(model_config, openai_client)
        
        # Initialize AI Portfolio Selector
        if openai_client and perplexity_client:
            self.ai_selector = AIPortfolioSelector(
                openai_client, perplexity_client, ips_config, model_config
            )
            logger.info("AI Portfolio Selector initialized")
        else:
            self.ai_selector = None
            logger.warning("AI Portfolio Selector not available (missing API clients)")
        
        # Agent weights for score blending - HEAVILY FAVOR UPSIDE POTENTIAL
        # Growth/Momentum is the most important for capturing upside
        agent_weights_config = ips_config.get('agent_weights', {})
        self.agent_weights = {
            'value_agent': agent_weights_config.get('value', 0.20),  # Value for reasonable entry
            'growth_momentum_agent': agent_weights_config.get('growth_momentum', 0.40),  # 🚀 DOUBLED - Upside potential is KEY
            'macro_regime_agent': agent_weights_config.get('macro_regime', 0.10),  # Reduced - less important than growth
            'risk_agent': agent_weights_config.get('risk', 0.15),  # Reduced - upside > downside protection
            'sentiment_agent': agent_weights_config.get('sentiment', 0.15)  # Market momentum matters
        }
        
        logger.info(f"Portfolio Orchestrator initialized with UPSIDE-FOCUSED weights: {self.agent_weights}")
    
    def analyze_single_stock(
        self,
        ticker: str,
        analysis_date: str,
        existing_portfolio: List[Dict] = None,
        agent_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single stock using all agents.
        
        Returns complete analysis with scores, rationale, and recommendations.
        """
        logger.info(f"Starting comprehensive analysis for {ticker} as of {analysis_date}")
        
        # Progress tracking function with step-level time tracking
        def update_progress(step, total_steps, message):
            try:
                # Try to import streamlit to check if we're in a Streamlit context
                import streamlit as st
                import time
                
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'analysis_progress'):
                    progress = st.session_state.analysis_progress
                    
                    # Track step completion time using StepTimeManager
                    if hasattr(st.session_state, 'last_step') and hasattr(st.session_state, 'current_step_start'):
                        if step > st.session_state.last_step and st.session_state.current_step_start is not None:
                            # A step was completed, record its time
                            step_duration = time.time() - st.session_state.current_step_start
                            completed_step = st.session_state.last_step
                            
                            # Record in StepTimeManager for persistence
                            if hasattr(st.session_state, 'step_time_manager'):
                                st.session_state.step_time_manager.record_step_time(completed_step, step_duration)
                        
                        # Update current step tracking
                        st.session_state.last_step = step
                        st.session_state.current_step_start = time.time()
                    
                    # Calculate estimated time remaining using StepTimeManager
                    time_remaining_str = ""
                    if hasattr(st.session_state, 'step_time_manager'):
                        remaining_steps = total_steps - step
                        if remaining_steps > 0:
                            # Get remaining step numbers
                            future_steps = list(range(step + 1, total_steps + 1))
                            
                            # Use StepTimeManager to calculate estimate
                            total_est_time = st.session_state.step_time_manager.get_total_estimate(future_steps)
                            
                            est_minutes = int(total_est_time // 60)
                            est_seconds = int(total_est_time % 60)
                            if est_minutes > 0:
                                time_remaining_str = f" - ~{est_minutes}m {est_seconds}s left"
                            else:
                                time_remaining_str = f" - ~{est_seconds}s left"
                    
                    if progress.get('progress_bar') and progress.get('status_text'):
                        progress_percent = int((step / total_steps) * 100)
                        progress['progress_bar'].progress(progress_percent)
                        progress['status_text'].text(f"{message} ({step}/{total_steps}){time_remaining_str}")
            except Exception as e:
                # If not in Streamlit context or any error, just continue
                pass
        
        # Set agent weights for this analysis if provided
        if agent_weights:
            # Map the simplified names to agent names used in the system
            weight_mapping = {
                'value': 'value_agent',
                'growth_momentum': 'growth_momentum_agent',
                'macro_regime': 'macro_regime_agent',
                'risk': 'risk_agent',
                'sentiment': 'sentiment_agent'
            }
            
            # Apply weights
            original_weights = self.agent_weights.copy()
            for simplified_name, weight in agent_weights.items():
                agent_name = weight_mapping.get(simplified_name, simplified_name)
                if agent_name in self.agent_weights:
                    self.agent_weights[agent_name] = weight
        
        # 1. Gather all data
        update_progress(1, 10, f"🔍 Fetching fundamental data from multiple sources for {ticker}")
        data = self._gather_data(ticker, analysis_date, existing_portfolio)
        
        if not data['fundamentals']:
            logger.warning(f"No fundamental data for {ticker}")
            return {
                'ticker': ticker, 
                'error': 'No data available',
                'fundamentals': {},
                'price_history': {},
                'agent_results': {},
                'agent_scores': {},
                'agent_rationales': {},
                'client_result': {'score': 0, 'rationale': 'No data available', 'eligible': False},
                'client_layer': {'score': 0, 'rationale': 'No data available', 'eligible': False},
                'blended_score': 0,
                'final_score': 0,
                'eligible': False
            }
        
        # Show specific extracted values
        fundamentals = data.get('fundamentals', {})
        price = fundamentals.get('price', 'N/A')
        eps = fundamentals.get('eps', 'N/A')
        pe_ratio = fundamentals.get('pe_ratio', 'N/A')
        market_cap = fundamentals.get('market_cap', 'N/A')
        
        # Format market cap for readability
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.1f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.1f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.1f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        update_progress(2, 10, f"📊 Extracted - Price: ${price}, EPS: ${eps}, P/E: {pe_ratio}, Market Cap: {market_cap_str}")
        
        # 2. Phase 1: Run all agents independently
        update_progress(3, 10, f"🤖 Initializing {len(self.agents)} specialized analysis agents")
        agent_results = {}
        agent_count = 0
        total_agents = len(self.agents)
        
        for agent_name, agent in self.agents.items():
            agent_count += 1
            # Update progress for specific agents - ONE update per agent
            if 'value' in agent_name.lower():
                pe_ratio = data['fundamentals'].get('pe_ratio', 'N/A')
                dividend_yield = data['fundamentals'].get('dividend_yield', 'N/A')
                update_progress(3, 10, f"💎 Value Agent: Analyzing P/E {pe_ratio}, dividend yield {dividend_yield}%")
            elif 'growth' in agent_name.lower():
                eps = data['fundamentals'].get('eps', 'N/A')
                revenue_growth = data['fundamentals'].get('revenue_growth', 'N/A')
                update_progress(4, 10, f"📈 Growth Agent: Analyzing EPS ${eps}, revenue growth {revenue_growth}%")
            elif 'macro' in agent_name.lower():
                sector = data['fundamentals'].get('sector', 'Unknown')
                update_progress(5, 10, f"🌍 Macro Agent: Evaluating {sector} sector in current regime")
            elif 'risk' in agent_name.lower():
                beta = data['fundamentals'].get('beta', 'N/A')
                volatility = data.get('details', {}).get('volatility_pct', 'N/A')
                update_progress(6, 10, f"⚖️ Risk Agent: Computing beta {beta}, volatility {volatility}%")
            elif 'sentiment' in agent_name.lower():
                update_progress(7, 10, f"📰 Sentiment Agent: Analyzing news and analyst sentiment")
            
            try:
                result = agent.analyze(ticker, data)
                agent_results[agent_name] = result
                
                # Log results but don't update progress to avoid duplicate step timing
                score = result.get('score', 0)
                logger.info(f"{agent_name}: {result['score']:.1f} - {result['rationale']}")
            except Exception as e:
                logger.error(f"Error in {agent_name} for {ticker}: {e}")
                agent_results[agent_name] = {
                    'score': 50,
                    'rationale': f'Analysis failed: {str(e)}',
                    'details': {}
                }
        
        # 3. Blend scores with weighted averaging
        update_progress(8, 10, f"⚖️ Blending agent scores with weights")
        blended_score = self._blend_scores(agent_results)
        
        # 4. Apply client layer validation
        update_progress(9, 10, f"🔍 Running client suitability validation")
        client_result = self.agents['client_layer_agent'].analyze(ticker, {
            'agent_results': agent_results,
            'blended_score': blended_score,
            'fundamentals': data['fundamentals']
        })
        
        # 5. Final result
        eligibility = "✅ Eligible" if client_result.get('eligible', False) else "❌ Not Eligible"
        update_progress(10, 10, f"✅ Analysis complete: {client_result['score']:.1f}/100, {eligibility}")
        
        # Restore original weights if they were changed
        if agent_weights:
            self.agent_weights = original_weights
        
        # Extract agent scores and rationales for backward compatibility
        agent_scores = {agent_name: result.get('score', 50) for agent_name, result in agent_results.items()}
        agent_rationales = {agent_name: result.get('rationale', 'Analysis not available') for agent_name, result in agent_results.items()}
        
        # Add client layer agent's independent compliance score to agent_scores
        agent_scores['client_layer_agent'] = client_result.get('compliance_score', client_result.get('score', 50))
        agent_rationales['client_layer_agent'] = client_result.get('rationale', 'Client layer analysis')
        
        return {
            'ticker': ticker,
            'analysis_date': analysis_date,
            'agent_results': agent_results,
            'agent_scores': agent_scores,
            'agent_rationales': agent_rationales,
            'blended_score': blended_score,
            'client_result': client_result,
            'client_layer': client_result,  # Alias for backward compatibility
            'final_score': client_result['score'],
            'eligible': client_result.get('eligible', False),
            'recommendation': self._generate_recommendation(client_result['score'], client_result.get('eligible', False)),
            'rationale': self._generate_comprehensive_rationale(ticker, agent_results, client_result, data),
            'fundamentals': data.get('fundamentals', {}),
            'price_history': data.get('price_history', {})
        }
    
    def analyze_stock(
        self,
        ticker: str,
        analysis_date: str = None,
        existing_portfolio: List[Dict] = None,
        agent_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Alias for analyze_single_stock method for backward compatibility.
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Date for analysis (defaults to today)
            existing_portfolio: Existing portfolio for context
            agent_weights: Custom agent weights for this analysis
            
        Returns:
            Complete analysis results
        """
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        return self.analyze_single_stock(
            ticker=ticker,
            analysis_date=analysis_date,
            existing_portfolio=existing_portfolio,
            agent_weights=agent_weights
        )
    
    def _blend_scores(self, agent_results: Dict[str, Dict]) -> float:
        """
        Blend agent scores using configured weights with UPSIDE POTENTIAL MULTIPLIER.
        
        Upside potential is THE MOST IMPORTANT factor. We boost scores based on:
        - Growth momentum (earnings growth, revenue growth, price momentum)
        - Distance from 52-week high (more room to run)
        - Positive sentiment (market recognition of upside)
        """
        # Calculate base weighted score
        total_score = 0
        total_weight = 0
        
        for agent_name, weight in self.agent_weights.items():
            if agent_name in agent_results:
                score = agent_results[agent_name].get('score', 50)
                total_score += score * weight
                total_weight += weight
        
        base_score = total_score / total_weight if total_weight > 0 else 50
        
        # ========== UPSIDE POTENTIAL MULTIPLIER ==========
        # This is THE KEY: boost stocks with high upside potential
        
        upside_multiplier = 1.0  # Start at neutral
        upside_factors = []
        
        # Factor 1: Growth/Momentum score (40% weight - most important!)
        if 'growth_momentum_agent' in agent_results:
            growth_score = agent_results['growth_momentum_agent'].get('score', 50)
            if growth_score >= 80:
                upside_multiplier += 0.15  # +15% boost for exceptional growth
                upside_factors.append(f"Exceptional growth ({growth_score:.0f}/100) → +15% boost")
            elif growth_score >= 70:
                upside_multiplier += 0.10  # +10% boost for strong growth
                upside_factors.append(f"Strong growth ({growth_score:.0f}/100) → +10% boost")
            elif growth_score >= 60:
                upside_multiplier += 0.05  # +5% boost for good growth
                upside_factors.append(f"Good growth ({growth_score:.0f}/100) → +5% boost")
            elif growth_score < 40:
                upside_multiplier -= 0.05  # -5% penalty for weak growth
                upside_factors.append(f"Weak growth ({growth_score:.0f}/100) → -5% penalty")
        
        # Factor 2: Sentiment (market recognizing upside)
        if 'sentiment_agent' in agent_results:
            sentiment_score = agent_results['sentiment_agent'].get('score', 50)
            if sentiment_score >= 75:
                upside_multiplier += 0.08  # +8% boost for very positive sentiment
                upside_factors.append(f"Very positive sentiment ({sentiment_score:.0f}/100) → +8% boost")
            elif sentiment_score >= 65:
                upside_multiplier += 0.05  # +5% boost for positive sentiment
                upside_factors.append(f"Positive sentiment ({sentiment_score:.0f}/100) → +5% boost")
        
        # Factor 3: Value (reasonable entry point amplifies upside)
        if 'value_agent' in agent_results:
            value_score = agent_results['value_agent'].get('score', 50)
            if value_score >= 75:
                upside_multiplier += 0.05  # +5% boost - good value = more upside runway
                upside_factors.append(f"Attractive valuation ({value_score:.0f}/100) → +5% boost")
        
        # Factor 4: Penalize very high risk (extreme risk reduces realizable upside)
        if 'risk_agent' in agent_results:
            risk_score = agent_results['risk_agent'].get('score', 50)
            if risk_score < 30:  # Very high risk
                upside_multiplier -= 0.10  # -10% penalty for extreme risk
                upside_factors.append(f"Extreme risk ({risk_score:.0f}/100) → -10% penalty")
        
        # Apply the multiplier (cap at 1.35 to prevent over-inflation)
        upside_multiplier = min(upside_multiplier, 1.35)
        upside_multiplier = max(upside_multiplier, 0.85)  # Floor at 0.85
        
        final_score = base_score * upside_multiplier
        
        # Log the upside calculation for transparency
        if upside_factors:
            logger.info(f"🚀 UPSIDE MULTIPLIER APPLIED: {upside_multiplier:.2f}x")
            logger.info(f"   Base Score: {base_score:.1f}")
            logger.info(f"   Upside Factors:")
            for factor in upside_factors:
                logger.info(f"     - {factor}")
            logger.info(f"   Final Score: {final_score:.1f} (boosted by {((upside_multiplier - 1) * 100):.0f}%)")
        
        return final_score
    
    def _generate_recommendation(self, score: float, eligible: bool) -> str:
        """Generate investment recommendation based on score and eligibility."""
        if not eligible:
            return "AVOID - Does not meet client criteria"
        elif score >= 80:
            return "STRONG BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 60:
            return "HOLD"
        elif score >= 40:
            return "WEAK HOLD"
        else:
            return "SELL"
    
    def _generate_comprehensive_rationale(self, ticker: str, agent_results: Dict, client_result: Dict, data: Dict) -> str:
        """
        Generate comprehensive investment rationale with complete context.
        Includes all data points, agent scores, weight analysis, and detailed reasoning.
        """
        fundamentals = data.get('fundamentals', {})
        
        # Build comprehensive rationale with all context
        rationale_parts = []
        
        # ========== SECTION 1: EXECUTIVE SUMMARY ==========
        rationale_parts.append("=" * 80)
        rationale_parts.append(f"COMPREHENSIVE INVESTMENT ANALYSIS: {ticker}")
        rationale_parts.append("=" * 80)
        
        # Company overview
        company_name = fundamentals.get('name', ticker)
        sector = fundamentals.get('sector', 'Unknown')
        
        rationale_parts.append(f"\n📊 COMPANY OVERVIEW:")
        rationale_parts.append(f"Company: {company_name}")
        rationale_parts.append(f"Sector: {sector}")
        rationale_parts.append(f"Ticker: {ticker}")
        
        # ========== SECTION 2: KEY FINANCIAL METRICS ==========
        rationale_parts.append(f"\n💰 KEY FINANCIAL METRICS:")
        
        price = fundamentals.get('price')
        if price:
            rationale_parts.append(f"Current Price: ${price:.2f}")
        
        market_cap = fundamentals.get('market_cap')
        if market_cap:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
            rationale_parts.append(f"Market Cap: {market_cap_str}")
        
        pe_ratio = fundamentals.get('pe_ratio')
        if pe_ratio:
            rationale_parts.append(f"P/E Ratio: {pe_ratio:.2f}")
        
        eps = fundamentals.get('eps')
        if eps:
            rationale_parts.append(f"EPS: ${eps:.2f}")
        
        beta = fundamentals.get('beta')
        if beta:
            rationale_parts.append(f"Beta: {beta:.2f}")
        
        dividend_yield = fundamentals.get('dividend_yield')
        if dividend_yield:
            rationale_parts.append(f"Dividend Yield: {dividend_yield*100:.2f}%")
        
        week_52_low = fundamentals.get('week_52_low')
        week_52_high = fundamentals.get('week_52_high')
        if week_52_low and week_52_high:
            rationale_parts.append(f"52-Week Range: ${week_52_low:.2f} - ${week_52_high:.2f}")
            if price:
                position = ((price - week_52_low) / (week_52_high - week_52_low)) * 100
                rationale_parts.append(f"Position in Range: {position:.1f}%")
        
        volume = fundamentals.get('volume')
        if volume:
            rationale_parts.append(f"Average Volume: {volume:,.0f}")
        
        # ========== SECTION 3: MULTI-AGENT ANALYSIS BREAKDOWN ==========
        rationale_parts.append(f"\n🤖 MULTI-AGENT ANALYSIS (Detailed Breakdown):")
        rationale_parts.append("=" * 80)
        
        # Order agents for presentation
        agent_order = ['value_agent', 'growth_momentum_agent', 'macro_regime_agent', 'risk_agent', 'sentiment_agent']
        agent_labels = {
            'value_agent': '💎 VALUE ANALYSIS',
            'growth_momentum_agent': '📈 GROWTH & MOMENTUM ANALYSIS',
            'macro_regime_agent': '🌍 MACROECONOMIC ANALYSIS',
            'risk_agent': '⚠️ RISK ASSESSMENT',
            'sentiment_agent': '📰 MARKET SENTIMENT ANALYSIS'
        }
        
        for agent_name in agent_order:
            if agent_name in agent_results:
                result = agent_results[agent_name]
                score = result.get('score', 50)
                rationale = result.get('rationale', 'Analysis not available')
                details = result.get('details', {})
                
                rationale_parts.append(f"\n{agent_labels.get(agent_name, agent_name.upper())}:")
                rationale_parts.append(f"Score: {score:.2f}/100")
                
                # Add specific metrics from details if available
                if details:
                    rationale_parts.append("Key Metrics:")
                    for key, value in details.items():
                        if value is not None and key not in ['rationale', 'score']:
                            if isinstance(value, float):
                                rationale_parts.append(f"  • {key.replace('_', ' ').title()}: {value:.2f}")
                            else:
                                rationale_parts.append(f"  • {key.replace('_', ' ').title()}: {value}")
                
                rationale_parts.append(f"\nDetailed Analysis:")
                rationale_parts.append(f"{rationale}")
                rationale_parts.append("-" * 80)
        
        # ========== SECTION 4: WEIGHT ANALYSIS & SCORE CALCULATION ==========
        rationale_parts.append(f"\n⚖️ WEIGHT ANALYSIS & FINAL SCORE CALCULATION:")
        rationale_parts.append("=" * 80)
        
        # Calculate weighted contributions
        total_weighted_score = 0
        total_weight = 0
        weight_breakdown = []
        
        for agent_name in agent_order:
            if agent_name in agent_results:
                score = agent_results[agent_name].get('score', 50)
                weight = self.agent_weights.get(agent_name, 1.0)
                weighted_contribution = score * weight
                
                total_weighted_score += weighted_contribution
                total_weight += weight
                
                weight_breakdown.append({
                    'agent': agent_name,
                    'score': score,
                    'weight': weight,
                    'contribution': weighted_contribution
                })
        
        # Show weight distribution
        rationale_parts.append(f"\nAgent Weight Distribution:")
        for item in weight_breakdown:
            agent_display = item['agent'].replace('_agent', '').replace('_', ' ').title()
            weight = item['weight']
            percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
            rationale_parts.append(f"  • {agent_display}: {weight:.2f}x ({percentage:.1f}% influence)")
        
        # Show calculation breakdown
        rationale_parts.append(f"\nScore Calculation Breakdown:")
        for item in weight_breakdown:
            agent_display = item['agent'].replace('_agent', '').replace('_', ' ').title()
            score = item['score']
            weight = item['weight']
            contribution = item['contribution']
            rationale_parts.append(f"  • {agent_display}: {score:.2f} × {weight:.2f} = {contribution:.2f}")
        
        # Final calculation
        blended_score = total_weighted_score / total_weight if total_weight > 0 else 50
        rationale_parts.append(f"\nWeighted Sum: {total_weighted_score:.2f}")
        rationale_parts.append(f"Total Weight: {total_weight:.2f}")
        rationale_parts.append(f"Blended Score: {total_weighted_score:.2f} / {total_weight:.2f} = {blended_score:.2f}")
        
        # Explain weight impact
        rationale_parts.append(f"\nWeight Impact Analysis:")
        
        # Find highest and lowest weighted agents
        max_weight_item = max(weight_breakdown, key=lambda x: x['weight'])
        min_weight_item = min(weight_breakdown, key=lambda x: x['weight'])
        
        max_agent = max_weight_item['agent'].replace('_agent', '').replace('_', ' ').title()
        min_agent = min_weight_item['agent'].replace('_agent', '').replace('_', ' ').title()
        
        if max_weight_item['weight'] > 1.0:
            rationale_parts.append(f"  • {max_agent} has the highest weight ({max_weight_item['weight']:.2f}x), giving it")
            rationale_parts.append(f"    MORE influence on the final score. Its score of {max_weight_item['score']:.2f}")
            rationale_parts.append(f"    contributes {max_weight_item['contribution']:.2f} points to the weighted sum.")
        
        if min_weight_item['weight'] < 1.0:
            rationale_parts.append(f"  • {min_agent} has the lowest weight ({min_weight_item['weight']:.2f}x), giving it")
            rationale_parts.append(f"    LESS influence on the final score. Its score of {min_weight_item['score']:.2f}")
            rationale_parts.append(f"    contributes {min_weight_item['contribution']:.2f} points to the weighted sum.")
        
        # Calculate what score would be with equal weights for comparison
        equal_weight_score = sum(item['score'] for item in weight_breakdown) / len(weight_breakdown)
        weight_effect = blended_score - equal_weight_score
        
        rationale_parts.append(f"\nComparison to Equal Weights:")
        rationale_parts.append(f"  • Equal Weight Score (all 1.0x): {equal_weight_score:.2f}")
        rationale_parts.append(f"  • Actual Weighted Score: {blended_score:.2f}")
        rationale_parts.append(f"  • Weight Effect: {weight_effect:+.2f} points")
        
        if abs(weight_effect) > 0.5:
            if weight_effect > 0:
                rationale_parts.append(f"  • The custom weights INCREASED the final score by emphasizing higher-scoring agents.")
            else:
                rationale_parts.append(f"  • The custom weights DECREASED the final score by emphasizing lower-scoring agents.")
        else:
            rationale_parts.append(f"  • The custom weights had minimal impact on the final score.")
        
        # ========== SECTION 5: CLIENT SUITABILITY ==========
        rationale_parts.append(f"\n🎯 CLIENT SUITABILITY ASSESSMENT:")
        rationale_parts.append("=" * 80)
        
        client_score = client_result.get('score', 50)
        client_eligible = client_result.get('eligible', False)
        client_rationale = client_result.get('rationale', 'No client assessment available')
        
        rationale_parts.append(f"Client Fit Score: {client_score:.2f}/100")
        rationale_parts.append(f"IPS Eligibility: {'✅ ELIGIBLE' if client_eligible else '❌ NOT ELIGIBLE'}")
        rationale_parts.append(f"\nClient Assessment:")
        rationale_parts.append(f"{client_rationale}")
        
        # ========== SECTION 6: FINAL RECOMMENDATION ==========
        rationale_parts.append(f"\n📋 FINAL RECOMMENDATION:")
        rationale_parts.append("=" * 80)
        
        final_score = client_result.get('score', blended_score)
        
        # Determine recommendation
        if not client_eligible:
            recommendation = "AVOID"
            recommendation_rationale = f"Despite a blended analysis score of {blended_score:.2f}, this stock does not meet client investment criteria and should be avoided."
        elif final_score >= 80:
            recommendation = "STRONG BUY"
            recommendation_rationale = f"Excellent score of {final_score:.2f} with strong fundamentals and positive outlook. Highly recommended for client portfolio."
        elif final_score >= 70:
            recommendation = "BUY"
            recommendation_rationale = f"Strong score of {final_score:.2f} indicating good investment potential. Recommended for client portfolio."
        elif final_score >= 60:
            recommendation = "HOLD"
            recommendation_rationale = f"Moderate score of {final_score:.2f}. Suitable for holding if already owned, but not a priority for new purchases."
        elif final_score >= 40:
            recommendation = "WEAK HOLD"
            recommendation_rationale = f"Below-average score of {final_score:.2f}. Consider for reduction or exit."
        else:
            recommendation = "SELL"
            recommendation_rationale = f"Low score of {final_score:.2f} indicates significant concerns. Not recommended for client portfolio."
        
        rationale_parts.append(f"Recommendation: {recommendation}")
        rationale_parts.append(f"Final Score: {final_score:.2f}/100")
        rationale_parts.append(f"Blended Agent Score: {blended_score:.2f}/100")
        rationale_parts.append(f"\nRationale:")
        rationale_parts.append(f"{recommendation_rationale}")
        
        # ========== SECTION 7: KEY INSIGHTS SUMMARY ==========
        rationale_parts.append(f"\n💡 KEY INSIGHTS:")
        rationale_parts.append("=" * 80)
        
        # Extract top positive and negative factors
        positive_factors = []
        negative_factors = []
        
        for item in weight_breakdown:
            if item['score'] >= 70:
                agent_name = item['agent'].replace('_agent', '').replace('_', ' ').title()
                positive_factors.append(f"• {agent_name} scores high ({item['score']:.1f}), indicating strength in this area")
            elif item['score'] < 50:
                agent_name = item['agent'].replace('_agent', '').replace('_', ' ').title()
                negative_factors.append(f"• {agent_name} scores low ({item['score']:.1f}), indicating concerns in this area")
        
        if positive_factors:
            rationale_parts.append("\nStrengths:")
            rationale_parts.extend(positive_factors)
        
        if negative_factors:
            rationale_parts.append("\nConcerns:")
            rationale_parts.extend(negative_factors)
        
        # Market position
        if price and week_52_low and week_52_high:
            position = ((price - week_52_low) / (week_52_high - week_52_low)) * 100
            if position > 80:
                rationale_parts.append(f"• Stock is near 52-week high ({position:.1f}%), may face resistance")
            elif position < 20:
                rationale_parts.append(f"• Stock is near 52-week low ({position:.1f}%), potential value opportunity")
        
        rationale_parts.append("=" * 80)
        
        return '\n'.join(rationale_parts)
    
    def _gather_data(
        self,
        ticker: str,
        analysis_date: str,
        existing_portfolio: Dict = None
    ) -> Dict[str, Any]:
        """Gather all necessary data for analysis using parallel API calls for speed."""
        # Calculate date range (1 year lookback) - ensure no future dates
        end_date = analysis_date
        start_date = pd.to_datetime(analysis_date) - pd.DateOffset(years=1)
        start_date = start_date.strftime('%Y-%m-%d')
        
        # Ensure we don't use future dates that break Yahoo Finance
        today = datetime.now().date()
        if pd.to_datetime(end_date).date() > today:
            end_date = today.strftime('%Y-%m-%d')
            start_date = (pd.to_datetime(today) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        
        data = {
            'ticker': ticker,
            'analysis_date': analysis_date,
        }
        
        # Add sub-progress function for data gathering
        def update_sub_progress(message):
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'analysis_progress'):
                    progress = st.session_state.analysis_progress
                    if progress.get('status_text'):
                        progress['status_text'].text(f"🔍 {message}")
            except:
                pass
        
        benchmark = self.ips_config.get('universe', {}).get('benchmark', '^GSPC')
        
        # PARALLEL DATA GATHERING - Run all 3 API calls simultaneously
        update_sub_progress(f"Launching parallel data retrieval for {ticker}...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all 3 tasks in parallel
            futures = {}
            
            # Task 1: Get fundamentals (Perplexity API - slowest, ~36s)
            if hasattr(self.data_provider, 'get_fundamentals_enhanced'):
                update_sub_progress(f"Task 1/3: Starting Perplexity API call for fundamentals...")
                futures['fundamentals'] = executor.submit(
                    self.data_provider.get_fundamentals_enhanced, ticker
                )
            else:
                futures['fundamentals'] = executor.submit(
                    self.data_provider.get_fundamentals, ticker
                )
            
            # Task 2: Get price history (Yahoo/Polygon API - medium, ~3-5s)
            # We'll decide which method after fundamentals, but submit the full fetch now
            update_sub_progress(f"Task 2/3: Starting price history API call...")
            if hasattr(self.data_provider, 'get_price_history_enhanced'):
                futures['price_history'] = executor.submit(
                    self.data_provider.get_price_history_enhanced,
                    ticker, start_date, end_date
                )
            else:
                futures['price_history'] = executor.submit(
                    self.data_provider.get_price_history,
                    ticker, start_date, end_date
                )
            
            # Task 3: Generate benchmark data (synthetic - fast, <1s)
            update_sub_progress(f"Task 3/3: Generating benchmark correlation data...")
            futures['benchmark'] = executor.submit(
                self._create_benchmark_data, benchmark, start_date, end_date
            )
            
            # Collect results as they complete
            update_sub_progress(f"Waiting for parallel tasks to complete...")
            
            # Get fundamentals result
            try:
                data['fundamentals'] = futures['fundamentals'].result()
                update_sub_progress(f"✅ Fundamentals retrieved")
                
                # Show specific extracted values
                if data['fundamentals']:
                    price = data['fundamentals'].get('price', 'N/A')
                    pe_ratio = data['fundamentals'].get('pe_ratio', 'N/A')
                    eps = data['fundamentals'].get('eps', 'N/A')
                    week_52_low = data['fundamentals'].get('week_52_low', 'N/A')
                    week_52_high = data['fundamentals'].get('week_52_high', 'N/A')
                    beta = data['fundamentals'].get('beta', 'N/A')
                    
                    update_sub_progress(f"Extracted: Price ${price}, P/E {pe_ratio}, EPS ${eps}")
                    
                    # CRITICAL DEBUG: Check what we actually got
                    logger.info(f"🔍 ORCHESTRATOR RECEIVED FUNDAMENTALS FOR {ticker}:")
                    logger.info(f"   → price: {data['fundamentals'].get('price')} (type: {type(data['fundamentals'].get('price')).__name__})")
                    logger.info(f"   → pe_ratio: {data['fundamentals'].get('pe_ratio')} (type: {type(data['fundamentals'].get('pe_ratio')).__name__})")
                    logger.info(f"   → beta: {data['fundamentals'].get('beta')} (type: {type(data['fundamentals'].get('beta')).__name__})")
                    logger.info(f"   → data_sources: {data['fundamentals'].get('data_sources')}")
            except Exception as e:
                logger.error(f"Failed to get fundamentals for {ticker}: {e}")
                data['fundamentals'] = {}
            
            # Get price history result - check if we need it or use synthetic
            try:
                if data.get('fundamentals', {}).get('source') == 'comprehensive_enhanced':
                    # Cancel the price history fetch if not needed
                    futures['price_history'].cancel()
                    update_sub_progress(f"Using comprehensive data - generating synthetic price history")
                    data['price_history'] = self._extract_price_history_from_fundamentals(data['fundamentals'])
                else:
                    # Use the fetched price history
                    data['price_history'] = futures['price_history'].result()
                    update_sub_progress(f"✅ Price history retrieved")
                    
                    if data['price_history'] is not None and not data['price_history'].empty:
                        latest_price = data['price_history']['Close'].iloc[-1] if 'Close' in data['price_history'].columns else 'N/A'
                        days_of_data = len(data['price_history'])
                        update_sub_progress(f"Downloaded {days_of_data} trading days, latest ${latest_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to get price history for {ticker}: {e}")
                # Fallback to synthetic
                if data.get('fundamentals'):
                    data['price_history'] = self._extract_price_history_from_fundamentals(data['fundamentals'])
                else:
                    data['price_history'] = pd.DataFrame()
            
            # Get benchmark result
            try:
                data['benchmark_history'] = futures['benchmark'].result()
                update_sub_progress(f"✅ Benchmark data generated")
                
                if data['benchmark_history'] is not None and not data['benchmark_history'].empty:
                    benchmark_return = ((data['benchmark_history']['Close'].iloc[-1] / data['benchmark_history']['Close'].iloc[0]) - 1) * 100
                    correlation_days = len(data['benchmark_history'])
                    update_sub_progress(f"{benchmark} {correlation_days} days, {benchmark_return:.1f}% return for beta calc")
            except Exception as e:
                logger.error(f"Failed to get benchmark data: {e}")
                data['benchmark_history'] = pd.DataFrame()
        
        # Add existing portfolio for risk analysis
        data['existing_portfolio'] = existing_portfolio or []
        
        update_sub_progress(f"All data gathered for {ticker}")
        return data
    
    def _extract_price_history_from_fundamentals(self, fundamentals: Dict) -> pd.DataFrame:
        """Extract/create price history from fundamentals data for synthetic analysis."""
        # This is a simplified synthetic price history for analysis
        # In a real implementation, you'd extract this from the comprehensive data
        current_price = fundamentals.get('price', 100)
        week_52_high = fundamentals.get('week_52_high', current_price * 1.2)
        week_52_low = fundamentals.get('week_52_low', current_price * 0.8)
        
        # Create synthetic daily data for the past year
        dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
        
        # Generate synthetic price movement between 52-week range
        np.random.seed(42)  # For reproducible results
        price_range = np.linspace(week_52_low, week_52_high, 252)
        noise = np.random.normal(0, current_price * 0.02, 252)  # 2% daily volatility
        synthetic_prices = price_range + noise
        
        # Ensure current price is the last price
        synthetic_prices[-1] = current_price
        
        # Create the DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Close': synthetic_prices,
            'High': synthetic_prices * 1.01,
            'Low': synthetic_prices * 0.99,
            'Volume': np.random.randint(1000000, 5000000, 252)
        }).set_index('Date')
        
        # Add Returns column that the risk agent expects
        df['Returns'] = df['Close'].pct_change()
        
        return df
    
    def _create_benchmark_data(self, benchmark_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create synthetic benchmark data to avoid API issues."""
        # Create simple synthetic S&P 500 data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic benchmark returns with realistic characteristics
        np.random.seed(42)  # Reproducible
        daily_returns = np.random.normal(0.0005, 0.01, len(dates))  # ~0.05% daily return, 1% volatility
        
        # Start at a reasonable level (e.g., 4000 for S&P 500)
        start_price = 4000
        prices = [start_price]
        
        for return_rate in daily_returns[1:]:
            prices.append(prices[-1] * (1 + return_rate))
        
        # Create the DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'High': [p * 1.005 for p in prices],
            'Low': [p * 0.995 for p in prices],
            'Volume': np.random.randint(3000000000, 5000000000, len(dates))
        }).set_index('Date')
        
        # Add Returns column that the risk agent expects
        df['Returns'] = df['Close'].pct_change()
        
        return df
    
    def recommend_portfolio(
        self,
        challenge_context: str = None,
        tickers: List[str] = None,
        num_positions: int = 5,
        analysis_date: str = None,
        universe_size: int = 5000
    ) -> Dict[str, Any]:
        """
        Generate portfolio recommendations using AI-powered ticker selection.
        
        Args:
            challenge_context: Description of investment challenge/goal
            tickers: Optional manual list of tickers (bypasses AI selection)
            num_positions: Target number of positions
            analysis_date: Date for analysis
            universe_size: Size of stock universe to consider (default: 5000 for broad coverage)
        
        Returns:
            Complete portfolio recommendation with analysis
        """
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"🎯 Starting Portfolio Recommendation - {num_positions} positions")
        
        # If no challenge context provided, use default
        if challenge_context is None:
            challenge_context = """
            Generate an optimal diversified portfolio that maximizes risk-adjusted returns 
            while adhering to the client's Investment Policy Statement constraints.
            Focus on high-quality companies with strong fundamentals and growth potential.
            """
        
        # Build client profile for AI selection
        client_profile = {
            'name': 'Portfolio Client',
            'ips_data': self.ips_config,
            'profile_text': challenge_context
        }
        
        # Stage 1: Get tickers (either AI-selected or manual)
        if tickers is None:
            if self.ai_selector is None:
                logger.error("❌ AI Portfolio Selector not available")
                raise ValueError("AI Portfolio Selector not initialized. Check OpenAI and Perplexity API keys.")
            
            logger.info(f"🤖 Running AI-powered ticker selection (universe: {universe_size})...")
            selection_result = self.ai_selector.select_portfolio_tickers(
                challenge_context=challenge_context,
                client_profile=client_profile,
                universe_size=universe_size
            )
            
            selected_tickers = selection_result['final_tickers']
            ticker_rationales = selection_result['ticker_rationales']
            all_candidates = selection_result['all_candidates']
            selection_log = selection_result['session_log']
            
            logger.info(f"✅ AI selected {len(selected_tickers)} tickers: {', '.join(selected_tickers)}")
        else:
            # Manual ticker list provided
            selected_tickers = tickers[:num_positions]
            ticker_rationales = {t: "Manually selected ticker" for t in selected_tickers}
            all_candidates = selected_tickers
            selection_log = {'manual_selection': True, 'tickers': selected_tickers}
            logger.info(f"📝 Using manual ticker list: {', '.join(selected_tickers)}")
        
        # Stage 2: Run full analysis on each ticker
        logger.info(f"📊 Running comprehensive analysis on {len(selected_tickers)} tickers...")
        
        portfolio_analyses = []
        for i, ticker in enumerate(selected_tickers, 1):
            logger.info(f"   → Analyzing {i}/{len(selected_tickers)}: {ticker}")
            
            try:
                analysis = self.analyze_single_stock(
                    ticker=ticker,
                    analysis_date=analysis_date,
                    existing_portfolio=portfolio_analyses
                )
                
                # Add AI rationale if available
                if ticker in ticker_rationales:
                    analysis['ai_rationale'] = ticker_rationales[ticker]
                
                portfolio_analyses.append(analysis)
                logger.info(f"   ✅ {ticker}: Score {analysis['final_score']:.1f}, Eligible: {analysis['eligible']}")
                
            except Exception as e:
                logger.error(f"   ❌ Analysis failed for {ticker}: {e}")
                continue
        
        # Stage 3: Filter and construct portfolio
        logger.info("📋 Constructing portfolio from analyzed stocks...")
        
        # Filter eligible stocks
        eligible_stocks = [a for a in portfolio_analyses if a.get('eligible', False)]
        
        if not eligible_stocks:
            logger.warning("⚠️ No eligible stocks found!")
            # Include all analyzed stocks anyway
            eligible_stocks = portfolio_analyses
        
        # Sort by final score
        eligible_stocks.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Take top N positions
        portfolio_stocks = eligible_stocks[:num_positions]
        
        # Stage 4: Calculate position sizes (equal weight for now)
        equal_weight = 100.0 / len(portfolio_stocks) if portfolio_stocks else 0
        
        portfolio = []
        for stock in portfolio_stocks:
            portfolio.append({
                'ticker': stock['ticker'],
                'name': stock['fundamentals'].get('name', stock['ticker']),
                'sector': stock['fundamentals'].get('sector', 'Unknown'),
                'final_score': stock['final_score'],
                'blended_score': stock['blended_score'],
                'eligible': stock['eligible'],
                'target_weight_pct': equal_weight,
                'rationale': stock.get('ai_rationale', 'See comprehensive analysis'),
                'recommendation': stock['recommendation'],
                'analysis': stock
            })
        
        # Stage 5: Calculate portfolio summary
        total_weight = sum(p['target_weight_pct'] for p in portfolio)
        avg_score = sum(p['final_score'] for p in portfolio) / len(portfolio) if portfolio else 0
        
        # Sector allocation
        sector_exposure = {}
        for p in portfolio:
            sector = p['sector']
            weight = p['target_weight_pct']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        summary = {
            'num_positions': len(portfolio),
            'total_weight_pct': total_weight,
            'avg_score': avg_score,
            'sector_exposure': sector_exposure,
            'challenge_context': challenge_context,
            'selection_method': 'AI-powered' if tickers is None else 'Manual'
        }
        
        logger.info(f"✅ Portfolio constructed: {len(portfolio)} positions, Avg Score: {avg_score:.1f}")
        
        return {
            'portfolio': portfolio,
            'summary': summary,
            'analysis_date': analysis_date,
            'all_candidates': all_candidates if tickers is None else selected_tickers,
            'selection_log': selection_log,
            'eligible_count': len(eligible_stocks),
            'total_analyzed': len(portfolio_analyses),
            'all_analyses': portfolio_analyses  # Include ALL analyzed stocks for QA archive
        }