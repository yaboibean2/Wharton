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
        st = config.get('scoring_thresholds', {})
        self._pe_thresholds = st.get('pe_thresholds', [12, 18, 25, 35, 50])
        self._pe_scores = st.get('pe_scores', [90, 75, 60, 45, 30, 15])
        self._ev_thresholds = st.get('ev_ebitda_thresholds', [8, 14, 22, 35, 50])
        self._ev_scores = st.get('ev_ebitda_scores', [90, 70, 55, 40, 25, 10])
        self._fcf_thresholds = st.get('fcf_yield_thresholds', [5.0, 3.0, 2.0, 1.0])
        self._fcf_scores = st.get('fcf_scores', [90, 70, 55, 40])
    
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
        
        # 1. P/E ratio vs sector (lower is better)
        if pe_ratio and pe_ratio > 0:
            pe_score = self._score_with_thresholds(pe_ratio, self._pe_thresholds, self._pe_scores)
            scores['pe_score'] = pe_score
            details['pe_ratio'] = pe_ratio
            details['sector_pe'] = sector_data.get('avg_pe', 25)
            details['pe_discount_pct'] = ((25 - pe_ratio) / 25 * 100)
        else:
            scores['pe_score'] = 50  # Neutral if missing

        # 2. EV/EBITDA (lower is better)
        if ev_ebitda and ev_ebitda > 0:
            ev_score = self._score_with_thresholds(ev_ebitda, self._ev_thresholds, self._ev_scores)
            scores['ev_ebitda_score'] = ev_score
            details['ev_ebitda'] = ev_ebitda
        else:
            scores['ev_ebitda_score'] = 50  # Neutral
        
        # 3. FCF Yield (higher is better)
        fcf_yield = fundamentals.get('fcf_yield') or 0

        if not fcf_yield:
            scores['fcf_yield_score'] = 50  # Neutral — data unavailable, not zero
            details['fcf_yield_pct'] = None
            details['fcf_data'] = 'unavailable'
        else:
            fcf_score = self._score_with_thresholds(
                fcf_yield, self._fcf_thresholds, self._fcf_scores, higher_is_better=True
            )
            scores['fcf_yield_score'] = fcf_score
            details['fcf_yield_pct'] = fcf_yield
        
        # 4. Dividend/Shareholder yield
        shareholder_yield = dividend_yield * 100 if dividend_yield else 0
        if shareholder_yield >= 4.0:
            yield_score = 85   # High income
        elif shareholder_yield >= 2.5:
            yield_score = 70   # Good income
        elif shareholder_yield >= 1.0:
            yield_score = 55   # Moderate
        elif shareholder_yield >= 0.5:
            yield_score = 45   # Low
        else:
            yield_score = 35   # No dividend (growth company)
        
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

        # data_quality: fraction of components backed by real data (not defaulted to 50)
        real_data_components = sum(
            1 for k, v in scores.items()
            if v != 50 or k == 'shareholder_yield_score'  # dividend=0 is real data
        )
        data_quality = real_data_components / max(len(scores), 1)
        details['data_quality'] = round(data_quality, 2)

        return {
            'score': round(composite_score, 2),
            'rationale': rationale,
            'details': details,
            'component_scores': scores,
            'data_quality': round(data_quality, 2),
        }
    
    def _score_with_thresholds(
        self,
        value: float,
        thresholds: list,
        scores: list,
        higher_is_better: bool = False,
    ) -> float:
        """
        Return a score from `scores` based on where `value` falls relative to `thresholds`.
        For lower-is-better metrics (P/E, EV/EBITDA): lower value → higher score.
        For higher-is-better metrics (FCF yield): higher value → higher score.
        `scores` must have len(thresholds) + 1 entries.
        """
        if higher_is_better:
            for threshold, score in zip(thresholds, scores):
                if value >= threshold:
                    return score
            return scores[-1]
        else:
            for threshold, score in zip(thresholds, scores):
                if value <= threshold:
                    return score
            return scores[-1]

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
        
        explanation += f"\n**Value Assessment (50 = fairly valued):**\n"
        if final_score >= 75:
            explanation += "**Strong value** - Multiple metrics suggest undervaluation.\n"
        elif final_score >= 60:
            explanation += "**Good value** - Reasonably priced with some margin of safety.\n"
        elif final_score >= 45:
            explanation += "**Fair value** - Trading near intrinsic value.\n"
        elif final_score >= 30:
            explanation += "**Expensive** - Premium valuation requires strong growth execution.\n"
        else:
            explanation += "**Very expensive** - Significant premium to fundamentals.\n"
        
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
        
        system_prompt = """You are a value analyst. Summarize the data below in 80-120 words.

RULES:
- ONLY state facts from the DATA section. Never invent numbers.
- Quote every number EXACTLY as given (e.g. "28.5x", "1.8%").
- If a value is N/A or DATA NOT AVAILABLE, say so — do not guess.
- Do NOT add predictions, opinions, or context beyond what the data shows.
- Do NOT use phrases like "suggests", "indicates", "implies", or "reflects".
- Structure: start with the score, then cover each metric with its exact value."""
        
        # Get all component scores for comprehensive context
        pe_score = scores.get('pe_score', 50)
        ev_score = scores.get('ev_ebitda_score', 50)
        fcf_score = scores.get('fcf_yield_score', 50)
        yield_score = scores.get('shareholder_yield_score', 50)
        
        def _vfmt(val, suffix=''):
            if val is None or val == 'N/A':
                return 'DATA NOT AVAILABLE'
            return f"{val}{suffix}"
        
        fcf_raw = details.get('fcf_yield_pct')
        ev_raw = details.get('ev_ebitda')
        
        user_prompt = f"""DATA for {ticker} ({sector}) — Value Score: {composite_score:.1f}/100

• P/E Ratio: {_vfmt(pe_ratio, 'x')} ({pe_discount:+.1f}% vs sector avg) (score {pe_score:.0f}/100)
• Dividend Yield: {div_yield:.1f}% (score {yield_score:.0f}/100)
• FCF Yield: {_vfmt(f'{fcf_raw:.1f}' if fcf_raw else None, '%')} (score {fcf_score:.0f}/100)
• EV/EBITDA: {_vfmt(ev_raw)} (score {ev_score:.0f}/100)

Summarize these facts."""
        
        try:
            rationale = self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=180
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
