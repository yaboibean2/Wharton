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
        
        # 1. EPS Growth
        earnings_growth = fundamentals.get('earnings_growth')
        if earnings_growth is not None:
            earnings_growth_pct = earnings_growth * 100
            if earnings_growth_pct >= 30:
                scores['eps_growth_score'] = 90   # Exceptional
            elif earnings_growth_pct >= 20:
                scores['eps_growth_score'] = 75   # Strong
            elif earnings_growth_pct >= 10:
                scores['eps_growth_score'] = 60   # Solid
            elif earnings_growth_pct >= 5:
                scores['eps_growth_score'] = 50   # Moderate
            elif earnings_growth_pct >= 0:
                scores['eps_growth_score'] = 40   # Flat
            elif earnings_growth_pct >= -10:
                scores['eps_growth_score'] = 25   # Declining
            else:
                scores['eps_growth_score'] = 10   # Severe decline
            details['earnings_growth_pct'] = earnings_growth_pct
        else:
            scores['eps_growth_score'] = 50  # Neutral when missing
            details['earnings_growth_pct'] = None
        
        # 2. Revenue Growth
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth is not None:
            revenue_growth_pct = revenue_growth * 100
            if revenue_growth_pct >= 25:
                scores['revenue_growth_score'] = 90   # Explosive
            elif revenue_growth_pct >= 15:
                scores['revenue_growth_score'] = 75   # Strong
            elif revenue_growth_pct >= 10:
                scores['revenue_growth_score'] = 60   # Solid
            elif revenue_growth_pct >= 5:
                scores['revenue_growth_score'] = 50   # Moderate
            elif revenue_growth_pct >= 0:
                scores['revenue_growth_score'] = 40   # Flat
            elif revenue_growth_pct >= -10:
                scores['revenue_growth_score'] = 25   # Declining
            else:
                scores['revenue_growth_score'] = 10   # Severe decline
            details['revenue_growth_pct'] = revenue_growth_pct
        else:
            scores['revenue_growth_score'] = 50  # Neutral when missing
            details['revenue_growth_pct'] = None
        
        # 3. Price Momentum
        if not price_history.empty and len(price_history) > 0:
            current_price = price_history['Close'].iloc[-1]

            # 3-month momentum
            if len(price_history) >= 63:
                price_3m_ago = price_history['Close'].iloc[-63]
                momentum_3m = ((current_price / price_3m_ago) - 1) * 100
                if momentum_3m >= 20:
                    scores['momentum_3m_score'] = 85
                elif momentum_3m >= 10:
                    scores['momentum_3m_score'] = 70
                elif momentum_3m >= 5:
                    scores['momentum_3m_score'] = 60
                elif momentum_3m >= 0:
                    scores['momentum_3m_score'] = 50
                elif momentum_3m >= -10:
                    scores['momentum_3m_score'] = 35
                else:
                    scores['momentum_3m_score'] = 20
                details['momentum_3m_pct'] = momentum_3m
            else:
                scores['momentum_3m_score'] = 50
                details['momentum_3m_pct'] = None

            # 6-month momentum
            if len(price_history) >= 126:
                price_6m_ago = price_history['Close'].iloc[-126]
                momentum_6m = ((current_price / price_6m_ago) - 1) * 100
                if momentum_6m >= 30:
                    scores['momentum_6m_score'] = 85
                elif momentum_6m >= 15:
                    scores['momentum_6m_score'] = 70
                elif momentum_6m >= 5:
                    scores['momentum_6m_score'] = 55
                elif momentum_6m >= 0:
                    scores['momentum_6m_score'] = 45
                elif momentum_6m >= -15:
                    scores['momentum_6m_score'] = 30
                else:
                    scores['momentum_6m_score'] = 15
                details['momentum_6m_pct'] = momentum_6m
            else:
                scores['momentum_6m_score'] = 50
                details['momentum_6m_pct'] = None

            # 12-month momentum
            if len(price_history) >= 252:
                price_12m_ago = price_history['Close'].iloc[-252]
                momentum_12m = ((current_price / price_12m_ago) - 1) * 100
                if momentum_12m >= 50:
                    scores['momentum_12m_score'] = 90
                elif momentum_12m >= 30:
                    scores['momentum_12m_score'] = 75
                elif momentum_12m >= 15:
                    scores['momentum_12m_score'] = 60
                elif momentum_12m >= 5:
                    scores['momentum_12m_score'] = 50
                elif momentum_12m >= 0:
                    scores['momentum_12m_score'] = 40
                elif momentum_12m >= -20:
                    scores['momentum_12m_score'] = 25
                else:
                    scores['momentum_12m_score'] = 10
                details['momentum_12m_pct'] = momentum_12m
            else:
                scores['momentum_12m_score'] = 50
                details['momentum_12m_pct'] = None
        else:
            scores['momentum_3m_score'] = 50
            scores['momentum_6m_score'] = 50
            scores['momentum_12m_score'] = 50
        
        # 4. Proximity to 52-week high
        pct_from_high = fundamentals.get('pct_from_52w_high')
        if pct_from_high is not None:
            if pct_from_high >= -5:
                scores['high_proximity_score'] = 80   # Near highs, strong momentum
            elif pct_from_high >= -15:
                scores['high_proximity_score'] = 65   # Healthy consolidation
            elif pct_from_high >= -25:
                scores['high_proximity_score'] = 50   # Moderate pullback
            elif pct_from_high >= -35:
                scores['high_proximity_score'] = 35   # Significant decline
            else:
                scores['high_proximity_score'] = 20   # Deep drawdown
            details['pct_from_52w_high'] = pct_from_high
        else:
            scores['high_proximity_score'] = 50
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
        rationale = self._generate_rationale(ticker, details, composite_score, scores)
        rationale += self._format_article_references(articles)

        # data_quality: fraction of components backed by real data (not defaulted to 50)
        real_data_components = sum(1 for v in scores.values() if v != 50)
        data_quality = real_data_components / max(len(scores), 1)
        details['data_quality'] = round(data_quality, 2)

        return {
            'score': round(composite_score, 2),
            'rationale': rationale,
            'details': details,
            'component_scores': scores,
            'data_quality': round(data_quality, 2),
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

    def _generate_rationale(self, ticker: str, details: Dict, actual_score: float = None, component_scores: Dict = None) -> str:
        """Generate enhanced rationale for growth and momentum analysis."""
        
        system_prompt = """You are a growth and momentum analyst. Write a concise 80-120 word analytical paragraph.

RULES:
- Use the DATA below as your source. Quote numbers exactly (e.g. "+12.1%", "-9.1%").
- If a value is marked DATA NOT AVAILABLE, briefly note it was unavailable — do not guess a value.
- Synthesize the data into a coherent narrative — do NOT just list the metrics back.
- Explain what the overall growth and momentum picture looks like for this stock.
- Highlight the strongest and weakest areas.
- Do NOT invent numbers or data not provided.
- Write in flowing prose, not bullet points."""
        
        # Use component_scores dict (passed from analyze) for accurate score values
        cs = component_scores or {}
        eps_score = cs.get('eps_growth_score', 50)
        rev_score = cs.get('revenue_growth_score', 50)
        mom_3m_score = cs.get('momentum_3m_score', 50)
        mom_6m_score = cs.get('momentum_6m_score', 50)
        mom_12m_score = cs.get('momentum_12m_score', 50)
        high_prox_score = cs.get('high_proximity_score', 50)
        
        composite_score = actual_score if actual_score is not None else sum([eps_score, rev_score, mom_3m_score, mom_6m_score, mom_12m_score, high_prox_score]) / 6
        
        # Explicit data-availability flags
        raw_eg = details.get('earnings_growth_pct')
        raw_rg = details.get('revenue_growth_pct')
        raw_m3 = details.get('momentum_3m_pct')
        raw_m6 = details.get('momentum_6m_pct')
        raw_m12 = details.get('momentum_12m_pct')
        raw_52h = details.get('pct_from_52w_high')
        
        def _fmt(val, suffix='%'):
            if val is None:
                return 'DATA NOT AVAILABLE'
            return f"{val:+.1f}{suffix}"
        
        user_prompt = f"""DATA for {ticker} — Score: {composite_score:.1f}/100

• Earnings Growth: {_fmt(raw_eg)} (score {eps_score:.0f}/100)
• Revenue Growth: {_fmt(raw_rg)} (score {rev_score:.0f}/100)
• 3-Month Momentum: {_fmt(raw_m3)} (score {mom_3m_score:.0f}/100)
• 6-Month Momentum: {_fmt(raw_m6)} (score {mom_6m_score:.0f}/100)
• 12-Month Momentum: {_fmt(raw_m12)} (score {mom_12m_score:.0f}/100)
• Distance from 52-Week High: {_fmt(raw_52h)} (score {high_prox_score:.0f}/100)

Summarize these facts."""
        
        try:
            rationale = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=200
            )
            return rationale.strip()
        except Exception as e:
            logger.warning(f"Failed to generate rationale: {e}")
            
            earnings_growth = raw_eg or 0
            momentum_6m = raw_m6 or 0
            from_52w_high = raw_52h or 0
            
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
