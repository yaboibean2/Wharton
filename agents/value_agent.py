"""
Value Agent
Analyzes stocks based on valuation metrics and yield.
Focuses on: P/E, EV/EBITDA, FCF yield, shareholder yield, dividend stability.
"""

from typing import Dict, Any
import numpy as np
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ValueAgent(BaseAgent):
    """
    Value investing analysis agent.
    Ranks stocks by valuation multiples and yield metrics.
    """
    
    def __init__(self, config: Dict[str, Any], openai_client=None):
        super().__init__("ValueAgent", config, openai_client)
        self.metrics_config = config.get('value_agent', {}).get('metrics', {})
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stock from value perspective.
        
        Returns score based on:
        - P/E ratio vs sector peers
        - EV/EBIT multiple
        - FCF yield
        - Shareholder yield (dividends + buybacks)
        """
        fundamentals = data.get('fundamentals', {})
        sector_data = data.get('sector_peers', {})
        
        # Extract key metrics
        pe_ratio = fundamentals.get('pe_ratio')
        ev_ebitda = fundamentals.get('ev_to_ebitda')
        dividend_yield = fundamentals.get('dividend_yield', 0) or 0
        market_cap = fundamentals.get('market_cap')
        enterprise_value = fundamentals.get('enterprise_value')
        
        # Calculate component scores
        scores = {}
        details = {}
        
        # 1. P/E ratio vs sector (lower is better) - EXTREMELY GENEROUS SCORING
        if pe_ratio and pe_ratio > 0:
            # Use a much more generous P/E scoring system aligned with your examples
            # Base score starts at 75 (perfectly valued) and adjusts from there
            base_score = 75
            
            # P/E ranges based on your examples: 23-40 is reasonable range
            if pe_ratio <= 25:
                pe_score = min(95, base_score + 20)  # Great value (like GOOGL at 23)
            elif pe_ratio <= 30:
                pe_score = min(85, base_score + 10)  # Good value (like AMD at 27)
            elif pe_ratio <= 35:
                pe_score = base_score  # Fair value (perfectly valued)
            elif pe_ratio <= 40:
                pe_score = max(65, base_score - 10)  # Slight premium (like NVDA at 39)
            else:
                pe_score = max(55, base_score - 20)  # High premium but not terrible
            
            scores['pe_score'] = pe_score
            details['pe_ratio'] = pe_ratio
            details['sector_pe'] = sector_data.get('avg_pe', 25)
            details['pe_discount_pct'] = ((25 - pe_ratio) / 25 * 100)
        else:
            scores['pe_score'] = 70  # Generous neutral if missing
        
        # 2. EV/EBITDA (lower is better) - EXTREMELY GENEROUS SCORING
        if ev_ebitda and ev_ebitda > 0:
            # Very generous EV/EBITDA scoring - most large caps are reasonable
            base_score = 75
            if ev_ebitda <= 15:
                ev_score = min(95, base_score + 20)  # Excellent value
            elif ev_ebitda <= 25:
                ev_score = min(85, base_score + 10)  # Good value
            elif ev_ebitda <= 35:
                ev_score = base_score  # Fair value
            elif ev_ebitda <= 50:
                ev_score = max(65, base_score - 10)  # Pricey but acceptable
            else:
                ev_score = max(55, base_score - 20)  # Expensive but not terrible
            
            scores['ev_ebitda_score'] = ev_score
            details['ev_ebitda'] = ev_ebitda
        else:
            scores['ev_ebitda_score'] = 70  # Generous neutral
        
        # 3. FCF Yield (higher is better)
        # Estimate: use operating cash flow if available
        fcf_yield = 0
        if market_cap and market_cap > 0:
            # This would ideally come from cash flow statement
            # For now, estimate using profit margin as proxy
            profit_margin = fundamentals.get('profit_margin', 0) or 0
            fcf_yield = profit_margin * 100 if profit_margin else 0
        
        # 3. FCF Yield - EXTREMELY GENEROUS (based on your examples where 1.6-2.5% is reasonable)
        base_score = 75
        if fcf_yield >= 3.0:
            fcf_score = min(95, base_score + 20)  # Excellent FCF yield
        elif fcf_yield >= 2.0:
            fcf_score = min(85, base_score + 10)  # Good FCF yield (like AAPL ~2.5%)
        elif fcf_yield >= 1.5:
            fcf_score = base_score  # Fair FCF yield (like NVDA ~1.6%)
        elif fcf_yield >= 1.0:
            fcf_score = max(65, base_score - 10)  # Low but acceptable
        else:
            fcf_score = max(60, base_score - 15)  # Very low but not penalized heavily
        
        scores['fcf_yield_score'] = fcf_score
        details['fcf_yield_pct'] = fcf_yield
        
        # 4. Dividend/Shareholder yield - EXTREMELY GENEROUS
        shareholder_yield = dividend_yield * 100 if dividend_yield else 0
        base_score = 75
        if shareholder_yield >= 3.0:
            yield_score = min(95, base_score + 20)  # High dividend yield
        elif shareholder_yield >= 2.0:
            yield_score = min(85, base_score + 10)  # Good dividend yield
        elif shareholder_yield >= 1.0:
            yield_score = base_score  # Moderate dividend yield
        elif shareholder_yield >= 0.5:
            yield_score = max(70, base_score - 5)  # Low dividend but ok for growth
        else:
            yield_score = max(65, base_score - 10)  # No dividend but not heavily penalized
        
        scores['shareholder_yield_score'] = yield_score
        details['dividend_yield_pct'] = shareholder_yield
        
        # Weighted composite score
        weights = {
            'pe_score': self.metrics_config.get('pe_ratio_weight', 0.3),
            'ev_ebitda_score': self.metrics_config.get('ev_ebit_weight', 0.25),
            'fcf_yield_score': self.metrics_config.get('fcf_yield_weight', 0.25),
            'shareholder_yield_score': self.metrics_config.get('shareholder_yield_weight', 0.2)
        }
        
        composite_score = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        
        # Generate detailed scoring explanation
        scoring_explanation = self._generate_scoring_explanation(ticker, scores, details, composite_score, fundamentals)
        details['scoring_explanation'] = scoring_explanation

        # Fetch domain-specific supporting articles
        articles = self._fetch_supporting_articles(
            ticker, "valuation analysis P/E ratio EV/EBITDA dividend yield intrinsic value"
        )
        details['supporting_articles'] = articles

        # Generate AI rationale with actual final score
        rationale = self._generate_rationale(ticker, fundamentals, scores, details, composite_score)
        rationale += self._format_article_references(articles)

        return {
            'score': round(composite_score, 2),
            'rationale': rationale,
            'details': details,
            'component_scores': scores
        }
    
    def _calculate_percentile(self, value: float, peer_avg: float) -> float:
        """Calculate where value falls relative to peer average (0-100)."""
        if peer_avg == 0:
            return 50
        ratio = value / peer_avg
        # Convert to percentile: ratio of 1.0 = 50th percentile
        percentile = 50 * ratio
        return max(0, min(100, percentile))
    
    def _generate_scoring_explanation(self, ticker: str, scores: Dict, details: Dict, final_score: float, fundamentals: Dict) -> str:
        """Generate detailed explanation of why this specific score was assigned."""
        
        explanation = f"**Value Score Breakdown: {final_score:.1f}/100**\n\n"
        
        # Component score explanations
        pe_score = scores.get('pe_score', 50)
        ev_score = scores.get('ev_ebitda_score', 50)
        fcf_score = scores.get('fcf_yield_score', 50)
        yield_score = scores.get('shareholder_yield_score', 50)
        
        pe_ratio = details.get('pe_ratio', 'N/A')
        pe_discount = details.get('pe_discount_pct', 0)
        div_yield = details.get('dividend_yield_pct', 0)
        ev_ebitda = details.get('ev_ebitda', 'N/A')
        fcf_yield = details.get('fcf_yield_pct', 0)
        
        explanation += f"**Component Scores:**\n"
        explanation += f"• P/E Valuation: {pe_score:.1f}/100 - "
        if pe_ratio != 'N/A':
            if pe_ratio <= 25:
                explanation += f"Excellent value at {pe_ratio:.1f}x P/E (like GOOGL range)\n"
            elif pe_ratio <= 30:
                explanation += f"Good value at {pe_ratio:.1f}x P/E (like AMD range)\n"
            elif pe_ratio <= 35:
                explanation += f"Fair valuation at {pe_ratio:.1f}x P/E (reasonable premium)\n"
            elif pe_ratio <= 40:
                explanation += f"Slight premium at {pe_ratio:.1f}x P/E (like NVDA range)\n"
            else:
                explanation += f"High premium at {pe_ratio:.1f}x P/E but not unreasonable for quality growth\n"
        else:
            explanation += "No P/E data available, neutral score assigned\n"
        
        explanation += f"• EV/EBITDA: {ev_score:.1f}/100 - "
        if ev_ebitda != 'N/A':
            if ev_ebitda <= 15:
                explanation += f"Excellent {ev_ebitda:.1f}x EV/EBITDA - great value\n"
            elif ev_ebitda <= 25:
                explanation += f"Good {ev_ebitda:.1f}x EV/EBITDA - reasonable valuation\n"
            elif ev_ebitda <= 35:
                explanation += f"Fair {ev_ebitda:.1f}x EV/EBITDA - acceptable for quality\n"
            elif ev_ebitda <= 50:
                explanation += f"Premium {ev_ebitda:.1f}x EV/EBITDA but justifiable for growth\n"
            else:
                explanation += f"High {ev_ebitda:.1f}x EV/EBITDA - paying for future growth\n"
        else:
            explanation += "No EV/EBITDA data available, neutral score assigned\n"
        
        explanation += f"• FCF Yield: {fcf_score:.1f}/100 - "
        if fcf_yield >= 3.0:
            explanation += f"Excellent {fcf_yield:.1f}% FCF yield - strong cash generation\n"
        elif fcf_yield >= 2.0:
            explanation += f"Good {fcf_yield:.1f}% FCF yield (like AAPL ~2.5%)\n"
        elif fcf_yield >= 1.5:
            explanation += f"Fair {fcf_yield:.1f}% FCF yield (like NVDA ~1.6%)\n"
        elif fcf_yield >= 1.0:
            explanation += f"Low {fcf_yield:.1f}% FCF yield but acceptable for growth\n"
        else:
            explanation += f"Very low {fcf_yield:.1f}% FCF yield - prioritizing growth over cash\n"
        
        explanation += f"• Dividend Yield: {yield_score:.1f}/100 - "
        if div_yield >= 3.0:
            explanation += f"High {div_yield:.1f}% dividend yield - excellent income\n"
        elif div_yield >= 2.0:
            explanation += f"Good {div_yield:.1f}% dividend yield - solid income component\n"
        elif div_yield >= 1.0:
            explanation += f"Moderate {div_yield:.1f}% dividend yield - some income\n"
        elif div_yield >= 0.5:
            explanation += f"Low {div_yield:.1f}% dividend yield - growth-focused approach\n"
        else:
            explanation += "No dividend - typical for growth companies, not penalized heavily\n"
        
        explanation += f"\n**Value Assessment (75 = perfectly valued):**\n"
        if final_score >= 85:
            explanation += "**Excellent value** - Multiple metrics suggest undervaluation (like GOOGL/AMD range).\n"
        elif final_score >= 75:
            explanation += "**Good value to fair** - Reasonable valuation for quality company.\n"
        elif final_score >= 65:
            explanation += "**Slight premium but acceptable** - Paying modest premium (like MSFT/AAPL range).\n"
        elif final_score >= 55:
            explanation += "**Moderately expensive** - Premium valuation requires strong growth execution.\n"
        else:
            explanation += "**Expensive but not unreasonable** - High premium for exceptional growth potential.\n"
        
        explanation += f"\n**To improve score:**\n"
        improvements = []
        if pe_score < 70 and pe_discount < 10:
            improvements.append(f"P/E ratio needs {10-pe_discount:.0f}% more discount to sector")
        if ev_score < 70 and ev_ebitda != 'N/A' and ev_ebitda > 12:
            improvements.append(f"EV/EBITDA should drop below 12x (currently {ev_ebitda:.1f}x)")
        if fcf_score < 70 and fcf_yield < 5:
            improvements.append(f"FCF yield needs improvement to 5%+ (currently {fcf_yield:.1f}%)")
        if yield_score < 70 and div_yield < 3:
            improvements.append(f"Dividend yield could increase to 3%+ (currently {div_yield:.1f}%)")
        
        if improvements:
            for imp in improvements:
                explanation += f"• {imp}\n"
        else:
            explanation += "Score is already strong based on available valuation metrics\n"
        
        return explanation

    def _generate_rationale(
        self,
        ticker: str,
        fundamentals: Dict,
        scores: Dict,
        details: Dict,
        actual_score: float = None
    ) -> str:
        """Generate concise rationale for preview (detailed analysis handled in UI)."""
        sector = fundamentals.get('sector', 'Unknown')
        pe_ratio = details.get('pe_ratio', 'N/A')
        pe_discount = details.get('pe_discount_pct', 0)
        div_yield = details.get('dividend_yield_pct', 0)
        composite_score = actual_score if actual_score is not None else (sum(scores.values()) / len(scores) if scores else 50)
        
        # Enhanced system prompt for comprehensive value analysis
        system_prompt = """You are a senior value investment analyst at a top-tier investment firm.
You specialize in identifying undervalued stocks using fundamental analysis. Your analysis should be:
1. Professional and detailed, explaining the specific metrics that drive your valuation
2. Context-aware, considering sector norms and market conditions
3. Forward-looking, discussing implications for investors
4. Specific about what makes this stock attractive or concerning from a value perspective
5. Around 100-150 words with clear, actionable insights

CRITICAL: You MUST cite specific numerical values from the data provided (e.g., "The P/E ratio of 28.5x trades at a 12% discount..." or "Dividend yield of 1.8% provides...").
Reference the exact metrics and scores given to you. Explain HOW each metric contributed to the final score.
State which data sources informed your analysis (e.g., P/E ratio, dividend yield, FCF yield, EV/EBITDA).

ACCURACY RULES — ZERO TOLERANCE FOR ERRORS:
- ONLY use the exact numerical values provided in the user prompt below. NEVER invent, round differently, or hallucinate statistics.
- If a metric is listed as N/A or 0.0%, say so — do NOT substitute a made-up value.
- Before writing each number, mentally verify it matches the data provided verbatim.
- Do NOT claim a growth rate, P/E, yield, or price that is not explicitly in the data below.
- If something seems contradictory (e.g., 0% growth but high momentum), describe what the data shows rather than speculating about what it 'should' be."""
        
        # Get all component scores for comprehensive context
        pe_score = scores.get('pe_score', 50)
        ev_score = scores.get('ev_ebitda_score', 50)
        fcf_score = scores.get('fcf_yield_score', 50)
        yield_score = scores.get('shareholder_yield_score', 50)
        
        # Rich context for comprehensive analysis
        user_prompt = f"""
STOCK ANALYSIS REQUEST: {ticker}
Sector: {sector}
FINAL VALUE SCORE: {composite_score:.1f}/100

DETAILED VALUATION METRICS:
• P/E Ratio: {pe_ratio} (Trading {pe_discount:+.1f}% vs sector average) → Score: {pe_score:.0f}/100
• Dividend Yield: {div_yield:.1f}% annual → Score: {yield_score:.0f}/100  
• Free Cash Flow Yield: {details.get('fcf_yield_pct', 0):.1f}% → Score: {fcf_score:.0f}/100
• EV/EBITDA Multiple: {details.get('ev_ebitda', 'N/A')} → Score: {ev_score:.0f}/100

SCORING CONTEXT:
- Scores above 80 = Excellent value opportunity
- Scores 65-80 = Good value with solid fundamentals  
- Scores 50-65 = Fair value, reasonably priced
- Scores below 50 = Overvalued or concerning metrics

ANALYSIS REQUEST:
As a value investing expert, provide a comprehensive analysis explaining why {ticker} earned a {composite_score:.1f}/100 value score. 
Address:
1. What are the strongest and weakest value metrics?
2. How does this compare to typical {sector} sector valuations?
3. What does this valuation suggest about investor sentiment and opportunity?
4. What are the key value investment considerations for this stock?

Focus on actionable insights for value-oriented investors."""
        
        try:
            rationale = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=200
            )
            return rationale.strip()
        except Exception as e:
            logger.warning(f"Failed to generate rationale: {e}")
            # Enhanced fallback explanations
            fallback = f"**VALUE ANALYSIS - Score {composite_score:.1f}/100**\n\n"
            
            if pe_discount > 15:
                fallback += f"**Strong Value Opportunity**: {ticker} trades {pe_discount:.0f}% below sector P/E, suggesting the market has overcorrected. "
            elif pe_discount < -15:
                fallback += f"**Premium Valuation**: {ticker} trades {abs(pe_discount):.0f}% above sector P/E, indicating investor optimism but limited margin of safety. "
            else:
                fallback += f"**Market-Inline Valuation**: {ticker} P/E aligns with sector norms. "
            
            if div_yield > 3:
                fallback += f"Attractive {div_yield:.1f}% dividend yield provides steady income stream. "
            elif div_yield > 0:
                fallback += f"Modest {div_yield:.1f}% dividend yield supplements returns. "
            else:
                fallback += "No dividend suggests growth-focused capital allocation. "
            
            fcf_yield = details.get('fcf_yield_pct', 0)
            if fcf_yield > 2.5:
                fallback += f"Strong {fcf_yield:.1f}% FCF yield demonstrates robust cash generation."
            elif fcf_yield > 1.5:
                fallback += f"Adequate {fcf_yield:.1f}% FCF yield supports business operations."
            else:
                fallback += f"Low {fcf_yield:.1f}% FCF yield raises questions about cash generation efficiency."
            
            return fallback
