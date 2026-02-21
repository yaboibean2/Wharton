"""
Risk/Correlation Agent
Monitors volatility, beta, correlations, and diversification.
Ensures portfolio stays within risk limits and properly diversified.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import logging
import os
from agents.base_agent import BaseAgent
from utils.comprehensive_verification import ComprehensiveDataVerifier

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    Risk management and diversification agent.
    Scores stocks based on risk metrics and portfolio fit.
    """
    
    def __init__(self, config: Dict[str, Any], openai_client=None):
        super().__init__("RiskAgent", config, openai_client)
        self.risk_config = config.get('risk_agent', {})
        self.lookback_days = self.risk_config.get('lookback_days', 252)
        self.correlation_threshold = self.risk_config.get('correlation_threshold', 0.7)
        
        # Initialize comprehensive data verifier
        perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        if perplexity_api_key:
            self.verifier = ComprehensiveDataVerifier(perplexity_api_key)
        else:
            logger.warning("PERPLEXITY_API_KEY not found - comprehensive verification disabled")
            self.verifier = None
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stock from risk perspective.
        
        Returns score based on:
        - Volatility (lower is better for risk-adjusted returns)
        - Beta (closer to 1.0 is neutral)
        - Max drawdown (shallower is better)
        - Diversification benefit
        - Market cap and ETF considerations (big stocks/ETFs = lower risk)
        """
        fundamentals = data.get('fundamentals', {})
        price_history = data.get('price_history', pd.DataFrame())
        benchmark_history = data.get('benchmark_history', pd.DataFrame())
        existing_portfolio = data.get('existing_portfolio', [])
        
        scores = {}
        details = {}
        
        # Check if this is a big stock or ETF (inherently lower risk)
        is_low_risk_asset = self._is_low_risk_asset(ticker, fundamentals)
        details['is_low_risk_asset'] = is_low_risk_asset
        
        # 1. Volatility Analysis - always produce a numeric volatility_pct
        if not price_history.empty and 'Returns' in price_history.columns:
            returns = price_history['Returns'].dropna()

            if len(returns) >= 5:
                # Annualized volatility
                volatility = returns.std() * np.sqrt(252) * 100

                # VERY CRITICAL volatility scoring - penalize anything above 18%
                # Excellent: <12%, Acceptable: 12-18%, Concerning: 18-25%, High Risk: 25-35%, Extreme: >35%
                scores['volatility_score'] = self._score_volatility(volatility)
                details['volatility_pct'] = round(volatility, 2)
                details['volatility_data_quality'] = 'high' if len(returns) >= 100 else 'moderate' if len(returns) >= 20 else 'limited'
            else:
                # Fallback: estimate volatility from beta if available
                beta_val = fundamentals.get('beta')
                if beta_val and isinstance(beta_val, (int, float)):
                    estimated_vol = abs(beta_val) * 18.0  # Market vol ~18%, scale by beta
                    scores['volatility_score'] = self._score_volatility(estimated_vol)
                    details['volatility_pct'] = round(estimated_vol, 2)
                    details['volatility_data_quality'] = 'estimated_from_beta'
                    logger.info(f"Estimated volatility for {ticker} from beta ({beta_val}): {estimated_vol:.1f}%")
                else:
                    scores['volatility_score'] = 50
                    details['volatility_pct'] = 20.0  # Market average as default
                    details['volatility_data_quality'] = 'default_estimate'
        else:
            # No price history at all - estimate from beta or use market average
            beta_val = fundamentals.get('beta')
            if beta_val and isinstance(beta_val, (int, float)):
                estimated_vol = abs(beta_val) * 18.0
                scores['volatility_score'] = self._score_volatility(estimated_vol)
                details['volatility_pct'] = round(estimated_vol, 2)
                details['volatility_data_quality'] = 'estimated_from_beta'
            else:
                scores['volatility_score'] = 50
                details['volatility_pct'] = 20.0  # Market average as default
                details['volatility_data_quality'] = 'default_estimate'
        
        # 2. Beta Analysis - VERY CRITICAL assessment
        beta = fundamentals.get('beta', 1.0)
        if beta:
            # EXTREMELY critical beta scoring - penalize any deviation heavily
            if beta < 0.2:
                beta_score = 40  # Extremely low beta indicates serious issues
            elif beta <= 0.7:
                beta_score = 70 + (0.7 - beta) * 40     # 70-90 for defensive (reduced max)
            elif beta <= 1.1:
                beta_score = 50 + (1.1 - beta) * 50     # 50-70 for market-like (reduced)
            elif beta <= 1.3:
                beta_score = 25 + (1.3 - beta) * 125    # 25-50 for moderate high (harsh)
            elif beta <= 1.8:
                beta_score = 5 + (1.8 - beta) * 40      # 5-25 for high beta (very harsh)
            else:
                beta_score = max(0, 5 - (beta - 1.8) * 2.5)  # 0-5 for extreme beta
            scores['beta_score'] = beta_score
            details['beta'] = round(beta, 2)
        else:
            scores['beta_score'] = 50
            details['beta'] = None
        
        # 3. Max Drawdown
        if not price_history.empty and len(price_history) > 20:
            cumulative = (1 + price_history['Returns'].fillna(0)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            # Shallower drawdown = higher score
            # Typical range: 0% to -50%
            dd_score = self._normalize_score(max_drawdown, -50, 0)
            scores['drawdown_score'] = dd_score
            details['max_drawdown_pct'] = round(max_drawdown, 2)
            
            # Flag if drawdown is severe
            if max_drawdown < -30:
                details['drawdown_warning'] = True
        else:
            scores['drawdown_score'] = 50
            details['max_drawdown_pct'] = None
        
        # 4. Diversification Benefit
        # Check correlation with existing portfolio holdings
        if existing_portfolio and not price_history.empty:
            diversification_score = self._calculate_diversification_benefit(
                ticker, price_history, existing_portfolio, data
            )
            scores['diversification_score'] = diversification_score
        else:
            scores['diversification_score'] = 70  # Neutral benefit
        
        # Apply low-risk asset bonus
        if is_low_risk_asset:
            # Boost all scores for big stocks and ETFs
            risk_boost = 15  # 15 point bonus
            for score_key in scores:
                scores[score_key] = min(100, scores[score_key] + risk_boost)
            details['risk_boost_applied'] = risk_boost
        
        # Weighted composite score
        weights = {
            'volatility_score': 0.3,
            'beta_score': 0.25,
            'drawdown_score': 0.25,
            'diversification_score': 0.2
        }
        
        composite_score = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        
        # Additional boost for low-risk assets at composite level
        if is_low_risk_asset:
            composite_score = min(100, composite_score + 10)  # Additional 10 point boost
        
        # Generate detailed scoring explanation
        scoring_explanation = self._generate_scoring_explanation(ticker, scores, details, composite_score)
        details['scoring_explanation'] = scoring_explanation

        # Fetch domain-specific supporting articles
        articles = self._fetch_supporting_articles(
            ticker, "stock risk analysis volatility beta downside risk assessment"
        )
        details['supporting_articles'] = articles

        # Generate rationale
        rationale = self._generate_rationale(ticker, details, scores, composite_score)
        rationale += self._format_article_references(articles)
        
        # Verify key data points for accuracy - DISABLED to prevent timeouts
        # verification_results = self._verify_risk_data(ticker, details, fundamentals)
        # details['verification_results'] = verification_results
        details['verification_results'] = {"status": "disabled", "reason": "Disabled to prevent connection timeouts"}
        logger.info(f"Verification skipped for {ticker} to prevent timeouts")
        
        return {
            'score': round(composite_score, 2),
            'rationale': rationale,
            'details': details,
            'component_scores': scores
        }
    
    def _score_volatility(self, volatility: float) -> float:
        """Score volatility on 0-100 scale. Lower volatility = higher score."""
        if volatility < 12:
            return 80 + (12 - volatility) * 1.5  # 80-100 for very low vol
        elif volatility < 18:
            return 55 + (18 - volatility) * 4.2  # 55-80 for acceptable vol
        elif volatility < 25:
            return 25 + (25 - volatility) * 4.3  # 25-55 for concerning vol
        elif volatility < 35:
            return 5 + (35 - volatility) * 2.0   # 5-25 for high risk vol
        else:
            return max(0, 5 - (volatility - 35) * 0.2)  # 0-5 for extreme vol

    def _is_low_risk_asset(self, ticker: str, fundamentals: Dict) -> bool:
        """
        Determine if this is a low-risk asset (big stocks or ETFs).
        Big stocks: MSFT, NVDA, AAPL, GOOGL, AMD, and other high market cap stocks.
        ETFs: Any ticker containing ETF indicators.
        """
        # Define big stock tickers (add more as needed)
        big_stocks = {
            'MSFT', 'NVDA', 'AAPL', 'GOOGL', 'GOOG', 'AMD', 'AMZN', 'TSLA', 
            'META', 'NFLX', 'CRM', 'ADBE', 'INTC', 'CSCO', 'ORCL', 'IBM',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.A', 'BRK.B',
            'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'TMO', 'DHR', 'ABT',
            'KO', 'PEP', 'WMT', 'HD', 'CVX', 'XOM', 'V', 'MA'
        }
        
        # Check if it's a known big stock
        if ticker.upper() in big_stocks:
            return True
        
        # Check if it's an ETF (common patterns)
        etf_indicators = ['ETF', 'FUND', 'INDEX', 'SPDR', 'ISHARES', 'VANGUARD']
        ticker_upper = ticker.upper()
        
        # Check ticker patterns for ETFs
        if any(indicator in ticker_upper for indicator in etf_indicators):
            return True
        
        # Check common ETF ticker patterns
        if (ticker_upper.startswith(('SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG')) or
            ticker_upper.endswith(('Y', 'F')) and len(ticker) == 3):
            return True
        
        # Check market cap if available (>$100B is considered big)
        market_cap = fundamentals.get('market_cap') or fundamentals.get('marketCap')
        if market_cap and market_cap > 100_000_000_000:  # $100B
            return True
        
        return False

    def _calculate_diversification_benefit(
        self,
        ticker: str,
        price_history: pd.DataFrame,
        existing_portfolio: List[str],
        all_data: Dict
    ) -> float:
        """
        Calculate diversification benefit of adding this stock.
        Lower correlation with portfolio = higher score.
        """
        if ticker in existing_portfolio:
            return 50  # Already in portfolio
        
        # Calculate average correlation with existing holdings
        correlations = []
        
        for existing_ticker in existing_portfolio[:10]:  # Check up to 10 holdings
            existing_data = all_data.get(existing_ticker, {})
            existing_history = existing_data.get('price_history', pd.DataFrame())
            
            if not existing_history.empty:
                # Align dates and calculate correlation
                merged = pd.merge(
                    price_history[['Returns']],
                    existing_history[['Returns']],
                    left_index=True,
                    right_index=True,
                    suffixes=('', '_existing')
                )
                
                if len(merged) > 20:
                    corr = merged['Returns'].corr(merged['Returns_existing'])
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if correlations:
            avg_correlation = np.mean(correlations)
            # Lower correlation = higher diversification benefit
            # Invert: correlation of 0 = 100 score, correlation of 1 = 0 score
            diversification_score = (1 - abs(avg_correlation)) * 100
            return diversification_score
        else:
            return 70  # Default: moderate benefit
    
    def _generate_scoring_explanation(self, ticker: str, scores: Dict, details: Dict, final_score: float) -> str:
        """Generate detailed explanation of why this specific score was assigned."""
        
        explanation = f"**Risk Score Breakdown: {final_score:.1f}/100**\n\n"
        
        # Component score explanations
        vol_score = scores.get('volatility_score', 50)
        beta_score = scores.get('beta_score', 50)
        dd_score = scores.get('drawdown_score', 50)
        div_score = scores.get('diversification_score', 50)
        
        volatility = details.get('volatility_pct')
        beta = details.get('beta')
        max_dd = details.get('max_drawdown_pct')
        is_low_risk = details.get('is_low_risk_asset', False)
        risk_boost = details.get('risk_boost_applied', 0)
        
        explanation += f"**Component Scores:**\n"
        explanation += f"• Volatility: {vol_score:.1f}/100 - "
        if volatility is not None:
            if volatility < 12:
                explanation += f"Low {volatility:.1f}% volatility but still carries market risk and potential losses\n"
            elif volatility < 18:
                explanation += f"Moderate {volatility:.1f}% volatility - expect periodic significant price swings\n"
            elif volatility < 25:
                explanation += f"Concerning {volatility:.1f}% volatility - substantial risk requires strict position limits\n"
            elif volatility < 35:
                explanation += f"High {volatility:.1f}% volatility - potential for severe capital losses during downturns\n"
            else:
                explanation += f"Dangerous {volatility:.1f}% volatility - unsuitable for most portfolios, extreme speculation\n"
        else:
            explanation += "No volatility data available - critical risk metric missing, assume high risk\n"
        
        explanation += f"• Beta Risk: {beta_score:.1f}/100 - "
        if beta is not None:
            if abs(beta - 1.0) < 0.1:
                explanation += f"Market-correlated {beta:.2f} beta - will decline significantly in bear markets\n"
            elif beta > 1.3:
                explanation += f"Extremely dangerous {beta:.2f} beta - amplifies all market losses, unsuitable for most investors\n"
            elif beta > 1.1:
                explanation += f"High {beta:.2f} beta - expect losses exceeding market during downturns\n"
            elif beta < 0.2:
                explanation += f"Suspiciously low {beta:.2f} beta likely indicates data problems or illiquidity issues\n"
            elif beta < 0.7:
                explanation += f"Low {beta:.2f} beta provides limited downside protection but restricts growth potential\n"
            else:
                explanation += f"Moderate {beta:.2f} beta - still vulnerable to market-wide systematic risk\n"
        else:
            explanation += "No beta data available - assume high systematic risk without proper measurement\n"
        
        explanation += f"• Drawdown: {dd_score:.1f}/100 - "
        if max_dd is not None:
            if max_dd > -5:
                explanation += f"Low {max_dd:.1f}% max drawdown but may indicate limited price history\n"
            elif max_dd > -15:
                explanation += f"Moderate {max_dd:.1f}% max drawdown shows some downside protection\n"
            elif max_dd > -25:
                explanation += f"Concerning {max_dd:.1f}% max drawdown indicates material downside risk\n"
            elif max_dd > -40:
                explanation += f"Severe {max_dd:.1f}% max drawdown shows significant capital destruction risk\n"
            else:
                explanation += f"Catastrophic {max_dd:.1f}% max drawdown indicates extreme speculation - major red flag\n"
        else:
            explanation += "No drawdown data available - cannot assess downside protection without price history\n"
        
        explanation += f"• Diversification: {div_score:.1f}/100 - Portfolio diversification benefit assessment\n"
        
        if is_low_risk:
            explanation += f"\n**Large-Cap/ETF Bonus: +{risk_boost} points**\n"
            explanation += f"Applied institutional stability bonus for large market cap or ETF status\n"
        
        explanation += f"\n**CRITICAL RISK WARNING:**\n"
        if final_score >= 80:
            explanation += "Lower risk but still subject to potential losses of 20-40% during market stress - no guarantees.\n"
        elif final_score >= 65:
            explanation += "Moderate-high risk - expect potential losses of 30-50% during market downturns.\n"
        elif final_score >= 50:
            explanation += "High risk - potential for losses exceeding 50% during adverse conditions - use extreme caution.\n"
        elif final_score >= 35:
            explanation += "Very high risk - potential for catastrophic losses of 60-80% - unsuitable for most investors.\n"
        elif final_score >= 20:
            explanation += "Extreme risk - potential for near-total capital loss - speculative investment only.\n"
        else:
            explanation += "MAXIMUM RISK - high probability of substantial or total capital loss - avoid investment.\n"
        
        explanation += f"\n**To improve score:**\n"
        improvements = []
        if vol_score < 70 and volatility is not None and volatility > 25:
            improvements.append(f"Reduce volatility below 25% (currently {volatility:.1f}%)")
        if beta_score < 70 and beta is not None and abs(beta - 1.0) > 0.3:
            improvements.append(f"Beta closer to 1.0 (currently {beta:.2f})")
        if dd_score < 70 and max_dd is not None and max_dd < -20:
            improvements.append(f"Improve downside protection (max drawdown currently {max_dd:.1f}%)")
        
        if improvements:
            for imp in improvements:
                explanation += f"• {imp}\n"
        else:
            explanation += "Score is already strong based on available risk metrics\n"
        
        return explanation

    def _generate_rationale(self, ticker: str, details: Dict, scores: Dict, actual_score: float) -> str:
        """Generate detailed rationale using OpenAI with comprehensive context."""
        system_prompt = """You are a senior risk management analyst at a premier institutional investment firm.
You specialize in quantitative risk assessment, portfolio risk management, and downside protection strategies.
Your analysis should be:
1. Quantitatively rigorous, focusing on specific risk metrics and their implications
2. Context-aware, considering market conditions and portfolio construction principles
3. Forward-looking, discussing potential downside scenarios and risk management
4. Specific about what drives the risk profile and how it affects investment suitability
5. Around 120-180 words with clear, actionable risk insights

CRITICAL: You MUST cite specific numerical values from the data provided (e.g., "With annualized volatility of 24.3%..." or "The beta of 1.21 indicates...").
Reference the exact metrics and scores given to you. Explain HOW each metric contributed to the final score.
State which data sources informed your analysis (e.g., price history, fundamental data, beta coefficient).

ACCURACY RULES — ZERO TOLERANCE FOR ERRORS:
- ONLY use the exact numerical values provided in the user prompt below. NEVER invent, round differently, or hallucinate statistics.
- If a metric is N/A, say so — do NOT substitute a made-up value.
- Before writing each number, mentally verify it matches the data provided verbatim.
- Do NOT claim a volatility, beta, or drawdown figure that is not explicitly in the data below."""
        
        risk_boost = details.get('risk_boost_applied', 0)
        is_low_risk = details.get('is_low_risk_asset', False)
        volatility = details.get('volatility_pct')
        beta = details.get('beta')
        max_drawdown = details.get('max_drawdown_pct')
        
        # Get component scores for context
        vol_score = scores.get('volatility_score', 50)
        beta_score = scores.get('beta_score', 50)
        dd_score = scores.get('drawdown_score', 50)
        div_score = scores.get('diversification_score', 50)
        
        # Safe formatting for None values
        vol_text = f"{volatility:.1f}%" if volatility is not None else "N/A"
        beta_text = f"{beta:.2f}" if beta is not None else "N/A"
        dd_text = f"{max_drawdown:.1f}%" if max_drawdown is not None else "N/A"
        
        user_prompt = f"""
RISK ASSESSMENT REQUEST: {ticker}
FINAL RISK SCORE: {actual_score:.1f}/100

DETAILED RISK METRICS:
• Volatility Analysis: {vol_text} annualized → Score: {vol_score:.0f}/100
• Beta Coefficient: {beta_text} (market correlation) → Score: {beta_score:.0f}/100  
• Maximum Drawdown: {dd_text} (worst decline) → Score: {dd_score:.0f}/100
• Diversification Benefit: Score: {div_score:.0f}/100
{'• Large-Cap Risk Reduction: +' + str(risk_boost) + ' point institutional stability bonus' if risk_boost > 0 else ''}

SCORING CONTEXT:
- Scores above 80 = Lower risk with institutional quality characteristics
- Scores 60-80 = Moderate risk suitable for balanced portfolios
- Scores 40-60 = Higher risk requiring careful position sizing
- Scores 20-40 = High risk suitable only for aggressive allocations
- Scores below 20 = Extreme risk, speculative positions only

ANALYSIS REQUEST:
As a risk management expert, provide a comprehensive analysis explaining why {ticker} earned a {actual_score:.1f}/100 risk score.
Address:
1. What are the primary risk drivers and most concerning metrics?
2. How do volatility, beta, and drawdown patterns interact to create the overall risk profile?
3. What downside scenarios should investors be prepared for?
4. How does this risk level affect appropriate position sizing and portfolio allocation?
5. What risk management considerations are most critical for this investment?

Focus on actionable insights for portfolio risk management and downside protection."""
        
        try:
            rationale = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=250
            )
            return rationale.strip()
        except Exception as e:
            logger.warning(f"Failed to generate rationale: {e}")
            
            # Simple, direct fallback explanations using actual score
            vol = details.get('volatility_pct')
            beta = details.get('beta')
            risk_boost = details.get('risk_boost_applied', 0)
            
            if risk_boost > 0:
                return f"Score {actual_score:.0f}/100: Large-cap gets +{risk_boost} point risk boost"
            elif vol is not None and vol < 15:
                return f"Score {actual_score:.0f}/100: Low {vol:.0f}% volatility = lower risk"
            elif vol is not None and vol > 35:
                return f"Score {actual_score:.0f}/100: High {vol:.0f}% volatility = higher risk"
            elif beta is not None and beta > 1.3:
                return f"Score {actual_score:.0f}/100: High {beta:.1f} beta = more volatile than market"
            elif beta is not None and beta < 0.7:
                return f"Score {actual_score:.0f}/100: Low {beta:.1f} beta = less volatile than market"
            else:
                return f"Score {actual_score:.0f}/100: Standard risk assessment for market conditions"
    
    def _verify_risk_data(self, ticker: str, details: Dict, fundamentals: Dict) -> Dict:
        """
        Verify key risk data points using comprehensive Perplexity verification.
        
        Args:
            ticker: Stock ticker
            details: Risk analysis details
            fundamentals: Company fundamentals
            
        Returns:
            Comprehensive verification results for all key metrics
        """
        try:
            # Collect all risk-related claims to verify
            claims_to_verify = []
            
            # Beta verification
            beta = details.get('beta') or fundamentals.get('beta')
            if beta is not None:
                claims_to_verify.append({
                    'claim': f"{ticker} has a beta of {beta:.2f}",
                    'value': beta,
                    'metric': 'beta',
                    'company': ticker
                })
            
            # Volatility verification
            volatility = details.get('volatility_pct')
            if volatility is not None:
                claims_to_verify.append({
                    'claim': f"{ticker} has annualized volatility of {volatility:.1f}%",
                    'value': f"{volatility:.1f}%",
                    'metric': 'volatility',
                    'company': ticker
                })
            
            # Market cap verification
            market_cap = fundamentals.get('market_cap')
            if market_cap:
                market_cap_b = market_cap / 1e9
                claims_to_verify.append({
                    'claim': f"{ticker} has a market capitalization of ${market_cap_b:.1f} billion",
                    'value': market_cap,
                    'metric': 'market cap',
                    'company': ticker
                })
            
            # P/E ratio verification (if making P/E claims)
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio is not None:
                claims_to_verify.append({
                    'claim': f"{ticker} has a P/E ratio of {pe_ratio:.2f}",
                    'value': pe_ratio,
                    'metric': 'P/E ratio',
                    'company': ticker
                })
            
            # Run comprehensive verification - prepare analysis data structure
            analysis_data = {
                'fundamentals': fundamentals,
                'details': details
            }
            
            if claims_to_verify and self.verifier:
                verification_results = self.verifier.verify_comprehensive_data(ticker, analysis_data)
                logger.info(f"Verified risk metrics for {ticker}")
                return verification_results
            else:
                reason = "No verifiable claims found" if claims_to_verify else "Comprehensive verification disabled"
                return {"verification_available": False, "reason": reason}
            
        except Exception as e:
            logger.warning(f"Comprehensive risk data verification failed for {ticker}: {e}")
            return {"verification_available": False, "error": str(e)}
