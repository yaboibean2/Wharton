"""
Growth/Momentum Agent
Captures earnings trajectory and price momentum.
Focuses on: EPS growth, sales growth, price momentum, proximity to 52-week high.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class GrowthMomentumAgent(BaseAgent):
    """
    Growth and momentum analysis agent.
    Identifies stocks with strong earnings growth and positive price momentum.
    """
    
    def __init__(self, config: Dict[str, Any], openai_client=None):
        super().__init__("GrowthMomentumAgent", config, openai_client)
        self.metrics_config = config.get('growth_momentum_agent', {}).get('metrics', {})
        self.thresholds = config.get('growth_momentum_agent', {}).get('thresholds', {})
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stock from growth/momentum perspective.
        
        Returns score based on:
        - EPS growth (1y, 3y)
        - Sales/revenue growth
        - Price momentum (3mo, 6mo, 12mo)
        - Distance from 52-week high
        """
        fundamentals = data.get('fundamentals', {})
        price_history = data.get('price_history', pd.DataFrame())
        
        scores = {}
        details = {}
        
        # 1. EPS Growth - HEAVILY REWARD UPSIDE POTENTIAL
        earnings_growth = fundamentals.get('earnings_growth')
        if earnings_growth:
            earnings_growth_pct = earnings_growth * 100
            # AGGRESSIVE scoring to capture upside potential
            # Any positive growth is good, high growth is exceptional
            if earnings_growth_pct >= 30:
                scores['eps_growth_score'] = 95  # Exceptional growth = massive upside
            elif earnings_growth_pct >= 20:
                scores['eps_growth_score'] = 85  # Strong growth = great upside
            elif earnings_growth_pct >= 15:
                scores['eps_growth_score'] = 75  # Good growth = solid upside
            elif earnings_growth_pct >= 10:
                scores['eps_growth_score'] = 70  # Decent growth = upside potential
            elif earnings_growth_pct >= 5:
                scores['eps_growth_score'] = 65  # Modest growth = some upside
            elif earnings_growth_pct >= 0:
                scores['eps_growth_score'] = 55  # Flat = limited upside
            else:
                scores['eps_growth_score'] = 40  # Negative = no upside
            details['earnings_growth_pct'] = earnings_growth_pct
        else:
            scores['eps_growth_score'] = 60  # Neutral when missing (benefit of doubt)
            details['earnings_growth_pct'] = None
        
        # 2. Revenue Growth - REWARD TOP-LINE EXPANSION (upside driver)
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth:
            revenue_growth_pct = revenue_growth * 100
            # Revenue growth = market share gains = upside potential
            if revenue_growth_pct >= 25:
                scores['revenue_growth_score'] = 95  # Explosive growth
            elif revenue_growth_pct >= 15:
                scores['revenue_growth_score'] = 85  # Strong growth
            elif revenue_growth_pct >= 10:
                scores['revenue_growth_score'] = 75  # Good growth
            elif revenue_growth_pct >= 5:
                scores['revenue_growth_score'] = 65  # Steady growth
            elif revenue_growth_pct >= 0:
                scores['revenue_growth_score'] = 55  # Flat
            else:
                scores['revenue_growth_score'] = 40  # Declining
            details['revenue_growth_pct'] = revenue_growth_pct
        else:
            scores['revenue_growth_score'] = 60  # Neutral when missing
            details['revenue_growth_pct'] = None
        
        # 3. Price Momentum - POSITIVE MOMENTUM = UPSIDE CONTINUATION
        if not price_history.empty and len(price_history) > 0:
            current_price = price_history['Close'].iloc[-1]
            
            # 3-month momentum - REWARD upward trends heavily
            if len(price_history) >= 63:  # ~3 months trading days
                price_3m_ago = price_history['Close'].iloc[-63]
                momentum_3m = ((current_price / price_3m_ago) - 1) * 100
                # Positive momentum = upside continuation potential
                if momentum_3m >= 20:
                    scores['momentum_3m_score'] = 90  # Strong upward trend
                elif momentum_3m >= 10:
                    scores['momentum_3m_score'] = 80  # Good momentum
                elif momentum_3m >= 5:
                    scores['momentum_3m_score'] = 70  # Positive trend
                elif momentum_3m >= 0:
                    scores['momentum_3m_score'] = 60  # Stable
                elif momentum_3m >= -10:
                    scores['momentum_3m_score'] = 50  # Slight pullback (buying opportunity?)
                else:
                    scores['momentum_3m_score'] = 40  # Downtrend
                details['momentum_3m_pct'] = momentum_3m
            else:
                scores['momentum_3m_score'] = 60
                details['momentum_3m_pct'] = None
            
            # 6-month momentum - MEDIUM-TERM upside trajectory
            if len(price_history) >= 126:
                price_6m_ago = price_history['Close'].iloc[-126]
                momentum_6m = ((current_price / price_6m_ago) - 1) * 100
                if momentum_6m >= 30:
                    scores['momentum_6m_score'] = 90  # Exceptional run
                elif momentum_6m >= 15:
                    scores['momentum_6m_score'] = 80  # Strong trend
                elif momentum_6m >= 5:
                    scores['momentum_6m_score'] = 70  # Positive
                elif momentum_6m >= 0:
                    scores['momentum_6m_score'] = 60  # Stable
                elif momentum_6m >= -15:
                    scores['momentum_6m_score'] = 50  # Pullback
                else:
                    scores['momentum_6m_score'] = 40  # Downtrend
                details['momentum_6m_pct'] = momentum_6m
            else:
                scores['momentum_6m_score'] = 60
                details['momentum_6m_pct'] = None
            
            # 12-month momentum - LONG-TERM upside validation
            if len(price_history) >= 252:
                price_12m_ago = price_history['Close'].iloc[-252]
                momentum_12m = ((current_price / price_12m_ago) - 1) * 100
                if momentum_12m >= 50:
                    scores['momentum_12m_score'] = 95  # Multi-bagger potential
                elif momentum_12m >= 30:
                    scores['momentum_12m_score'] = 85  # Strong year
                elif momentum_12m >= 15:
                    scores['momentum_12m_score'] = 75  # Good year
                elif momentum_12m >= 5:
                    scores['momentum_12m_score'] = 65  # Positive
                elif momentum_12m >= 0:
                    scores['momentum_12m_score'] = 55  # Flat
                elif momentum_12m >= -20:
                    scores['momentum_12m_score'] = 45  # Pullback (value entry?)
                else:
                    scores['momentum_12m_score'] = 35  # Poor performance
                details['momentum_12m_pct'] = momentum_12m
            else:
                scores['momentum_12m_score'] = 60
                details['momentum_12m_pct'] = None
        else:
            scores['momentum_3m_score'] = 50
            scores['momentum_6m_score'] = 50
            scores['momentum_12m_score'] = 50
        
        # 4. Proximity to 52-week high - BALANCED VIEW (breakouts vs pullbacks)
        pct_from_high = fundamentals.get('pct_from_52w_high')
        if pct_from_high is not None:
            # Two perspectives on upside:
            # - Near highs = momentum continuation (breakout potential)
            # - Off highs = room to recover (mean reversion upside)
            if pct_from_high >= -5:
                scores['high_proximity_score'] = 90  # At/near highs = breakout potential
            elif pct_from_high >= -15:
                scores['high_proximity_score'] = 80  # Slight pullback = healthy consolidation
            elif pct_from_high >= -25:
                scores['high_proximity_score'] = 75  # Good pullback = recovery upside
            elif pct_from_high >= -35:
                scores['high_proximity_score'] = 65  # Deeper pullback = substantial upside if recovers
            else:
                scores['high_proximity_score'] = 55  # Far from highs = uncertain upside
            details['pct_from_52w_high'] = pct_from_high
        else:
            scores['high_proximity_score'] = 60
            details['pct_from_52w_high'] = None
        
        # Weighted composite score - HEAVILY WEIGHT GROWTH METRICS
        # Growth and momentum are THE KEY drivers of upside potential
        weights = {
            'eps_growth_score': self.metrics_config.get('eps_growth_1y_weight', 0.30),  # Increased - KEY upside driver
            'revenue_growth_score': self.metrics_config.get('sales_growth_weight', 0.25),  # Increased - validates growth
            'momentum_3m_score': self.metrics_config.get('price_momentum_3m_weight', 0.15),  # Recent momentum
            'momentum_6m_score': self.metrics_config.get('price_momentum_6m_weight', 0.15),  # Medium-term trend
            'momentum_12m_score': self.metrics_config.get('price_momentum_12m_weight', 0.05),  # Reduced - less relevant
            'high_proximity_score': 0.10  # Reduced - less important than actual growth
        }
        
        composite_score = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        
        # Generate detailed scoring explanation
        scoring_explanation = self._generate_scoring_explanation(ticker, scores, details, composite_score)
        details['scoring_explanation'] = scoring_explanation

        # Fetch domain-specific supporting articles
        articles = self._fetch_supporting_articles(
            ticker, "earnings growth revenue momentum analyst upgrade price target"
        )
        details['supporting_articles'] = articles

        # Generate AI rationale with actual final score
        rationale = self._generate_rationale(ticker, details, composite_score)
        rationale += self._format_article_references(articles)

        return {
            'score': round(composite_score, 2),
            'rationale': rationale,
            'details': details,
            'component_scores': scores
        }
    
    def _generate_scoring_explanation(self, ticker: str, scores: Dict, details: Dict, final_score: float) -> str:
        """Generate detailed explanation of why this specific score was assigned."""
        
        explanation = f"**Growth & Momentum Score Breakdown: {final_score:.1f}/100**\n\n"
        
        # Component score explanations
        eps_score = scores.get('eps_growth_score', 50)
        rev_score = scores.get('revenue_growth_score', 50)
        mom_3m = scores.get('momentum_3m_score', 50)
        mom_6m = scores.get('momentum_6m_score', 50)
        mom_12m = scores.get('momentum_12m_score', 50)
        high_prox = scores.get('high_proximity_score', 50)
        
        earnings_growth = details.get('earnings_growth_pct', 0) or 0
        revenue_growth = details.get('revenue_growth_pct', 0) or 0
        momentum_3m = details.get('momentum_3m_pct', 0) or 0
        momentum_6m = details.get('momentum_6m_pct', 0) or 0
        momentum_12m = details.get('momentum_12m_pct', 0) or 0
        from_52w_high = details.get('pct_from_52w_high', 0) or 0
        
        explanation += f"**Component Scores:**\n"
        explanation += f"• Earnings Growth: {eps_score:.1f}/100 - "
        if earnings_growth > 25:
            explanation += f"Exceptional {earnings_growth:.1f}% earnings growth well above 20% threshold\n"
        elif earnings_growth > 15:
            explanation += f"Strong {earnings_growth:.1f}% earnings growth within excellent range\n"
        elif earnings_growth > 5:
            explanation += f"Moderate {earnings_growth:.1f}% earnings growth shows positive trend\n"
        elif earnings_growth > 0:
            explanation += f"Weak {earnings_growth:.1f}% earnings growth below expectations\n"
        else:
            explanation += f"Declining {abs(earnings_growth):.1f}% earnings growth indicates contraction\n"
        
        explanation += f"• Revenue Growth: {rev_score:.1f}/100 - "
        if revenue_growth > 20:
            explanation += f"Excellent {revenue_growth:.1f}% revenue growth shows strong business expansion\n"
        elif revenue_growth > 10:
            explanation += f"Good {revenue_growth:.1f}% revenue growth within solid range\n"
        elif revenue_growth > 0:
            explanation += f"Modest {revenue_growth:.1f}% revenue growth indicates slow expansion\n"
        else:
            explanation += f"Declining {abs(revenue_growth):.1f}% revenue shows business contraction\n"
        
        explanation += f"• 3-Month Momentum: {mom_3m:.1f}/100 - "
        if momentum_3m > 15:
            explanation += f"Strong {momentum_3m:+.1f}% short-term momentum indicates bullish sentiment\n"
        elif momentum_3m > 5:
            explanation += f"Positive {momentum_3m:+.1f}% momentum shows upward trend\n"
        elif momentum_3m > -5:
            explanation += f"Neutral {momentum_3m:+.1f}% momentum indicates sideways action\n"
        else:
            explanation += f"Negative {momentum_3m:+.1f}% momentum shows bearish pressure\n"
        
        explanation += f"• 6-Month Momentum: {mom_6m:.1f}/100 - "
        if momentum_6m > 25:
            explanation += f"Excellent {momentum_6m:+.1f}% medium-term momentum shows strong trend\n"
        elif momentum_6m > 10:
            explanation += f"Good {momentum_6m:+.1f}% momentum indicates positive trajectory\n"
        elif momentum_6m > -10:
            explanation += f"Mixed {momentum_6m:+.1f}% momentum shows uncertainty\n"
        else:
            explanation += f"Poor {momentum_6m:+.1f}% momentum indicates bearish trend\n"
        
        explanation += f"• 12-Month Momentum: {mom_12m:.1f}/100 - "
        if momentum_12m > 30:
            explanation += f"Outstanding {momentum_12m:+.1f}% long-term momentum shows sustained growth\n"
        elif momentum_12m > 15:
            explanation += f"Strong {momentum_12m:+.1f}% annual momentum demonstrates consistent performance\n"
        elif momentum_12m > 0:
            explanation += f"Positive {momentum_12m:+.1f}% annual return but below market expectations\n"
        else:
            explanation += f"Negative {momentum_12m:+.1f}% annual return shows underperformance\n"
        
        explanation += f"• 52-Week High Proximity: {high_prox:.1f}/100 - "
        if from_52w_high > -5:
            explanation += f"Excellent - trading within {abs(from_52w_high):.1f}% of 52-week highs\n"
        elif from_52w_high > -15:
            explanation += f"Good - {abs(from_52w_high):.1f}% below highs with recovery potential\n"
        elif from_52w_high > -30:
            explanation += f"Moderate - {abs(from_52w_high):.1f}% below highs shows significant pullback\n"
        else:
            explanation += f"Poor - {abs(from_52w_high):.1f}% below highs indicates major decline\n"
        
        explanation += f"\n**Why this score?**\n"
        if final_score >= 80:
            explanation += "Exceptional growth profile with strong momentum across multiple timeframes.\n"
        elif final_score >= 70:
            explanation += "Strong growth and momentum indicators supporting continued outperformance.\n"
        elif final_score >= 50:
            explanation += "Mixed growth signals with moderate momentum requiring careful analysis.\n"
        elif final_score >= 30:
            explanation += "Weak growth profile with concerning momentum trends.\n"
        else:
            explanation += "Poor growth and momentum characteristics indicating potential underperformance.\n"
        
        explanation += f"\n**To improve score:**\n"
        improvements = []
        if eps_score < 70 and earnings_growth < 15:
            improvements.append(f"Earnings growth needs acceleration to 15%+ (currently {earnings_growth:.1f}%)")
        if rev_score < 70 and revenue_growth < 10:
            improvements.append(f"Revenue growth should reach 10%+ (currently {revenue_growth:.1f}%)")
        if mom_6m < 70 and momentum_6m < 10:
            improvements.append(f"6-month momentum needs improvement to 10%+ (currently {momentum_6m:+.1f}%)")
        if high_prox < 70 and from_52w_high < -15:
            improvements.append(f"Stock needs to recover closer to 52-week highs (currently {abs(from_52w_high):.1f}% below)")
        
        if improvements:
            for imp in improvements:
                explanation += f"• {imp}\n"
        else:
            explanation += "Score is already strong based on available growth and momentum metrics\n"
        
        return explanation

    def _generate_rationale(self, ticker: str, details: Dict, actual_score: float = None) -> str:
        """Generate enhanced rationale for growth and momentum analysis."""
        
        system_prompt = """You are a senior growth and momentum analyst at a leading growth-focused investment firm.
You specialize in identifying companies with accelerating fundamentals and positive price momentum.
Your analysis should be:
1. Growth-focused and momentum-aware, explaining how earnings, revenue, and price trends interact
2. Forward-looking, discussing implications for continued growth and momentum
3. Market-context aware, considering how growth rates compare to sector peers and market conditions
4. Specific about what drives sustainable growth momentum vs temporary price movements
5. Around 120-180 words with clear, actionable insights about growth sustainability

CRITICAL: You MUST cite specific numerical values from the data provided (e.g., "Earnings growth of +18.5% combined with 3-month momentum of +12.3%..." or "Revenue growth of 22% validates...").
Reference the exact metrics and scores given to you. Explain HOW each metric contributed to the final score.
State which data sources informed your analysis (e.g., earnings growth rate, revenue growth, price momentum, 52-week proximity).

ACCURACY RULES — ZERO TOLERANCE FOR ERRORS:
- ONLY use the exact numerical values provided in the user prompt below. NEVER invent, round differently, or hallucinate statistics.
- If a metric shows +0.0% or N/A, acknowledge that directly — do NOT describe it as "stagnant" or claim the data is unavailable when a number IS provided.
- Before writing each number, mentally verify it matches the data provided verbatim.
- If earnings growth is +0.0%, do NOT say "lack of fundamental growth" — say earnings growth is flat at +0.0%.
- If 12-month momentum is +168.6%, cite that exact figure — do NOT alter it.
- If something seems contradictory (e.g., 0% earnings growth but strong momentum), describe both facts accurately without fabricating an explanation."""
        
        earnings_growth = details.get('earnings_growth_pct', 0) or 0
        revenue_growth = details.get('revenue_growth_pct', 0) or 0
        momentum_3m = details.get('momentum_3m_pct', 0) or 0
        momentum_6m = details.get('momentum_6m_pct', 0) or 0
        momentum_12m = details.get('momentum_12m_pct', 0) or 0
        from_52w_high = details.get('pct_from_52w_high', 0) or 0
        
        # Get component scores for comprehensive analysis
        eps_score = details.get('eps_growth_score', 50)
        rev_score = details.get('revenue_growth_score', 50)  
        mom_3m_score = details.get('momentum_3m_score', 50)
        mom_6m_score = details.get('momentum_6m_score', 50)
        mom_12m_score = details.get('momentum_12m_score', 50)
        high_prox_score = details.get('high_proximity_score', 50)
        
        # Use actual score if provided, otherwise calculate composite score for context
        composite_score = actual_score if actual_score is not None else sum([eps_score, rev_score, mom_3m_score, mom_6m_score, mom_12m_score, high_prox_score]) / 6
        
        user_prompt = f"""
GROWTH & MOMENTUM ANALYSIS REQUEST: {ticker}
FINAL GROWTH/MOMENTUM SCORE: {composite_score:.1f}/100

DETAILED GROWTH METRICS:
• Earnings Growth Rate: {earnings_growth:+.1f}% → Score: {eps_score:.0f}/100
• Revenue Growth Rate: {revenue_growth:+.1f}% → Score: {rev_score:.0f}/100
• 3-Month Price Momentum: {momentum_3m:+.1f}% → Score: {mom_3m_score:.0f}/100
• 6-Month Price Momentum: {momentum_6m:+.1f}% → Score: {mom_6m_score:.0f}/100
• 12-Month Price Momentum: {momentum_12m:+.1f}% → Score: {mom_12m_score:.0f}/100
• Distance from 52-Week High: {from_52w_high:.1f}% → Score: {high_prox_score:.0f}/100

SCORING CONTEXT:
- Scores above 80 = Exceptional growth with strong momentum acceleration
- Scores 60-80 = Solid growth with positive momentum trends
- Scores 40-60 = Moderate growth with mixed momentum signals
- Scores 20-40 = Weak growth with concerning momentum deterioration
- Scores below 20 = Poor growth fundamentals with negative momentum

ANALYSIS REQUEST:
As a growth and momentum expert, provide a comprehensive analysis explaining why {ticker} earned a {composite_score:.1f}/100 growth/momentum score.
Address:
1. How do fundamental growth rates (earnings/revenue) align with price momentum?
2. What does the momentum pattern across different timeframes suggest about sustainability?
3. How does proximity to 52-week highs/lows affect the momentum outlook?
4. What are the key drivers of growth acceleration or deceleration?
5. How does this growth/momentum profile affect investment timing and expectations?

Focus on actionable insights about growth sustainability and momentum continuation patterns."""
        
        try:
            rationale = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=250
            )
            return rationale.strip()
        except Exception as e:
            logger.warning(f"Failed to generate rationale: {e}")
            
            # Simple, direct fallback explanations
            composite_score = sum([eps_score, rev_score, mom_6m_score, high_prox_score]) / 4
            
            if earnings_growth < 0:
                return f"Score {composite_score:.0f}/100: Declining {abs(earnings_growth):.0f}% earnings growth"
            elif momentum_6m > 20:
                return f"Score {composite_score:.0f}/100: Strong {momentum_6m:.0f}% price momentum"
            elif momentum_6m < -20:
                return f"Score {composite_score:.0f}/100: Weak {momentum_6m:.0f}% price momentum"
            elif earnings_growth > 15:
                return f"Score {composite_score:.0f}/100: Strong {earnings_growth:.0f}% earnings growth"
            elif from_52w_high > -5:
                return f"Score {composite_score:.0f}/100: Trading near 52-week highs"
            else:
                return f"Score {composite_score:.0f}/100: Mixed growth and momentum signals"
