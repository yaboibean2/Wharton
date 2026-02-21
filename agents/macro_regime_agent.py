"""
Macro/Regime Agent
Analyzes macroeconomic environment and sets sector tilts.
Monitors: yield curve, inflation, PMI, unemployment, credit spreads.
"""

from typing import Dict, Any, List
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MacroRegimeAgent(BaseAgent):
    """
    Macroeconomic regime classification agent.
    Identifies current economic regime and applies sector tilts.
    """
    
    def __init__(self, config: Dict[str, Any], openai_client=None):
        super().__init__("MacroRegimeAgent", config, openai_client)
        self.sector_tilts = config.get('macro_regime_agent', {}).get('sector_tilts', {})
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze stock based on macro environment.
        
        Returns score adjustment based on:
        - Current economic regime
        - Sector positioning for that regime  
        - Market cap positioning
        - Stock-specific macro characteristics
        """
        fundamentals = data.get('fundamentals', {})
        macro_data = data.get('macro_indicators', {})
        
        sector = fundamentals.get('sector', 'Unknown')
        market_cap = fundamentals.get('market_cap', 0)
        beta = fundamentals.get('beta', 1.0)
        
        # Classify regime based on macro indicators
        regime = self._classify_regime(macro_data)
        
        # Start with base score of 50 (neutral)
        final_score = 50
        adjustments = []
        
        # 1. Sector tilt for current regime (-15 to +15)
        sector_adjustment = self._get_sector_tilt(sector, regime)
        final_score += sector_adjustment
        if sector_adjustment != 0:
            adjustments.append(f"Sector fit: {sector_adjustment:+.0f}")
        
        # 2. Market cap positioning score (-10 to +10)
        market_cap_adjustment = self._score_market_cap_positioning(market_cap, regime)
        final_score += market_cap_adjustment
        if market_cap_adjustment != 0:
            adjustments.append(f"Market cap: {market_cap_adjustment:+.0f}")
        
        # 3. Beta/cyclicality score (-10 to +10)
        beta_adjustment = self._score_cyclicality(beta, regime)
        final_score += beta_adjustment
        if beta_adjustment != 0:
            adjustments.append(f"Cyclicality: {beta_adjustment:+.0f}")
        
        # 4. Macro health overlay (-15 to +15)
        macro_health_score = self._calculate_macro_health(macro_data)
        health_adjustment = (macro_health_score - 50) * 0.3  # Scale to ±15
        final_score += health_adjustment
        if abs(health_adjustment) > 1:
            adjustments.append(f"Macro health: {health_adjustment:+.0f}")
        
        # Clamp to 0-100 range
        final_score = max(0, min(100, final_score))

        # Fetch domain-specific supporting articles
        articles = self._fetch_supporting_articles(
            ticker, f"{sector} sector macroeconomic outlook interest rate impact economic regime"
        )

        # Generate rationale with actual final score
        rationale = self._generate_rationale(ticker, sector, regime, sector_adjustment, macro_data, final_score)
        rationale += self._format_article_references(articles)

        details = {
            'regime': regime,
            'sector_adjustment': sector_adjustment,
            'market_cap_adjustment': market_cap_adjustment,
            'beta_adjustment': beta_adjustment,
            'adjustments_breakdown': ', '.join(adjustments) if adjustments else 'No significant adjustments',
            'market_cap': market_cap,
            'beta': beta,
            'yield_curve_slope': macro_data.get('yield_curve_slope'),
            'inflation_yoy': macro_data.get('inflation_yoy'),
            'unemployment': macro_data.get('unemployment_rate'),
            'supporting_articles': articles
        }
        
        return {
            'score': round(final_score, 2),
            'rationale': rationale,
            'details': details,
            'regime': regime
        }
    
    def _classify_regime(self, macro_data: Dict) -> str:
        """
        Classify economic regime based on macro indicators.
        
        Regimes:
        - expansion: positive growth, steepening curve, low unemployment
        - recession: negative growth, inverted curve, rising unemployment
        - high_inflation: high inflation, flat/inverted curve
        - disinflation: falling inflation, normal curve
        """
        yield_curve = macro_data.get('yield_curve_slope', 0.5)
        inflation = macro_data.get('inflation_yoy', 3.0)
        unemployment = macro_data.get('unemployment_rate', 4.0)
        
        # Recession signals
        if yield_curve < 0:  # Inverted yield curve
            return 'recession'
        
        # High inflation regime
        if inflation > 5.0:
            return 'high_inflation'
        
        # Disinflation (falling inflation)
        if inflation < 2.0 and yield_curve > 0.5:
            return 'disinflation'
        
        # Default: expansion
        if unemployment < 5.0 and yield_curve > 0:
            return 'expansion'
        
        return 'expansion'  # Default
    
    def _get_sector_tilt(self, sector: str, regime: str) -> float:
        """
        Get score adjustment based on sector appropriateness for regime.
        
        Returns:
            Adjustment in range [-15, +15]
        """
        regime_tilts = self.sector_tilts.get(regime, {})
        
        overweight_sectors = regime_tilts.get('overweight', [])
        underweight_sectors = regime_tilts.get('underweight', [])
        
        if sector in overweight_sectors:
            return 15  # Positive adjustment
        elif sector in underweight_sectors:
            return -15  # Negative adjustment
        else:
            return 0  # Neutral
    
    def _score_market_cap_positioning(self, market_cap: float, regime: str) -> float:
        """
        Score based on market cap size appropriateness for regime.
        
        Returns:
            Adjustment in range [-10, +10]
        """
        if not market_cap or market_cap == 0:
            return 0
        
        # Large cap: $10B+, Mid cap: $2B-$10B, Small cap: <$2B
        if regime in ['recession', 'high_inflation']:
            # Prefer large caps in tough times (flight to quality)
            if market_cap >= 10e9:
                return 10
            elif market_cap >= 2e9:
                return 0
            else:
                return -10
        elif regime in ['expansion', 'disinflation']:
            # Growth-friendly environment favors small/mid caps
            if market_cap >= 200e9:  # Mega caps
                return -5  # Mature, slower growth
            elif market_cap >= 10e9:  # Large caps
                return 0
            elif market_cap >= 2e9:  # Mid caps
                return 10  # Sweet spot for growth
            else:  # Small caps
                return 8  # High growth potential
        
        return 0
    
    def _score_cyclicality(self, beta: float, regime: str) -> float:
        """
        Score based on beta/cyclicality fit for regime.
        
        Returns:
            Adjustment in range [-10, +10]
        """
        if not beta:
            return 0
        
        if regime in ['expansion', 'disinflation']:
            # Growth environments favor cyclical stocks (high beta)
            if beta > 1.3:
                return 10
            elif beta > 1.1:
                return 5
            elif beta < 0.8:
                return -5  # Too defensive
        elif regime in ['recession', 'high_inflation']:
            # Risk-off environments favor defensive stocks (low beta)
            if beta < 0.8:
                return 10
            elif beta < 1.0:
                return 5
            elif beta > 1.3:
                return -10  # Too risky
        
        return 0
    
    def _calculate_macro_health(self, macro_data: Dict) -> float:
        """
        Calculate overall macro health score (0-100).
        Uses current economic indicators with reasonable assumptions.
        """
        scores = []
        
        # Get macro indicators - use realistic current values if not provided
        # As of Oct 2025: yield curve positive, inflation moderating, unemployment low
        yield_curve = macro_data.get('yield_curve_slope')
        inflation = macro_data.get('inflation_yoy')
        unemployment = macro_data.get('unemployment_rate')
        
        # If no data provided, use neutral baseline (no bonus/penalty)
        if yield_curve is None and inflation is None and unemployment is None:
            # Return a neutral baseline score that allows sector differentiation
            return 50
        
        # Yield curve score (positive slope is healthy)
        if yield_curve is not None:
            if yield_curve > 1.0:
                scores.append(80)
            elif yield_curve > 0:
                scores.append(60)
            elif yield_curve > -0.5:
                scores.append(40)
            else:
                scores.append(20)
        
        # Inflation score (2-4% is ideal)
        if inflation is not None:
            if 2.0 <= inflation <= 4.0:
                scores.append(80)
            elif inflation < 2.0 or (4.0 < inflation <= 5.0):
                scores.append(60)
            else:
                scores.append(40)
        
        # Unemployment score (< 5% is healthy)
        if unemployment is not None:
            if unemployment < 4.5:
                scores.append(80)
            elif unemployment < 6.0:
                scores.append(60)
            else:
                scores.append(40)
        
        return sum(scores) / len(scores) if scores else 50

    def _generate_scoring_explanation(self, ticker: str, sector: str, regime: str, macro_health_score: float, sector_adjustment: float, final_score: float, macro_data: Dict) -> str:
        """Generate detailed explanation of why this specific score was assigned."""
        
        explanation = f"**Macro Regime Score Breakdown: {final_score:.1f}/100**\n\n"
        
        yield_curve = macro_data.get('yield_curve_slope', 0.5)
        inflation = macro_data.get('inflation_yoy', 3.0)
        unemployment = macro_data.get('unemployment_rate', 4.0)
        
        explanation += f"**Current Economic Regime: {regime.replace('_', ' ').title()}**\n\n"
        
        explanation += f"**Component Scores:**\n"
        explanation += f"• Base Macro Health: {macro_health_score:.1f}/100 - "
        
        if macro_health_score >= 75:
            explanation += "Excellent macroeconomic conditions with supportive indicators\n"
        elif macro_health_score >= 60:
            explanation += "Good macro environment with mostly positive signals\n"
        elif macro_health_score >= 40:
            explanation += "Mixed macro conditions with both positive and negative factors\n"
        else:
            explanation += "Challenging macro environment with concerning indicators\n"
        
        explanation += f"• Sector Adjustment: {sector_adjustment:+.1f} points - "
        if sector_adjustment > 10:
            explanation += f"{sector} sector is favored in {regime.replace('_', ' ')} regime\n"
        elif sector_adjustment < -10:
            explanation += f"{sector} sector faces headwinds in {regime.replace('_', ' ')} regime\n"
        else:
            explanation += f"{sector} sector has neutral positioning in current regime\n"
        
        explanation += f"\n**Macro Indicators Analysis:**\n"
        explanation += f"• Yield Curve Slope: {yield_curve:.2f} - "
        if yield_curve > 1.0:
            explanation += "Healthy steep curve supports economic expansion\n"
        elif yield_curve > 0:
            explanation += "Positive but flattening curve shows moderate growth\n"
        elif yield_curve > -0.5:
            explanation += "Flat curve indicates economic uncertainty\n"
        else:
            explanation += "Inverted curve signals potential recession risk\n"
        
        explanation += f"• Inflation Rate: {inflation:.1f}% - "
        if inflation < 2.0:
            explanation += "Below-target inflation allows for accommodative policy\n"
        elif inflation < 4.0:
            explanation += "Moderate inflation within acceptable range\n"
        elif inflation < 6.0:
            explanation += "Elevated inflation creates policy tightening pressure\n"
        else:
            explanation += "High inflation forces aggressive monetary tightening\n"
        
        explanation += f"• Unemployment Rate: {unemployment:.1f}% - "
        if unemployment < 4.0:
            explanation += "Very low unemployment indicates full employment\n"
        elif unemployment < 5.5:
            explanation += "Low unemployment supports consumer spending\n"
        elif unemployment < 7.0:
            explanation += "Moderate unemployment shows economic softness\n"
        else:
            explanation += "High unemployment indicates economic stress\n"
        
        explanation += f"\n**Why this score?**\n"
        if final_score >= 80:
            explanation += f"Excellent macro environment with {regime.replace('_', ' ')} regime strongly favoring {sector} sector.\n"
        elif final_score >= 70:
            explanation += f"Good macro conditions with {regime.replace('_', ' ')} regime providing support for {sector} sector.\n"
        elif final_score >= 50:
            explanation += f"Mixed macro environment with {regime.replace('_', ' ')} regime having neutral impact on {sector} sector.\n"
        elif final_score >= 30:
            explanation += f"Challenging macro conditions with {regime.replace('_', ' ')} regime creating headwinds for {sector} sector.\n"
        else:
            explanation += f"Poor macro environment with {regime.replace('_', ' ')} regime significantly pressuring {sector} sector.\n"
        
        explanation += f"\n**To improve score:**\n"
        improvements = []
        if yield_curve < 0:
            improvements.append("Yield curve needs to steepen (currently inverted)")
        if inflation > 5.0:
            improvements.append(f"Inflation needs to moderate below 4% (currently {inflation:.1f}%)")
        if unemployment > 6.0:
            improvements.append(f"Unemployment needs to decline below 5.5% (currently {unemployment:.1f}%)")
        if sector_adjustment < 0:
            improvements.append(f"Regime change that favors {sector} sector would boost score")
        
        if improvements:
            for imp in improvements:
                explanation += f"• {imp}\n"
        else:
            explanation += "Score is already strong based on current macro regime and sector positioning\n"
        
        return explanation

    def _generate_rationale(
        self,
        ticker: str,
        sector: str,
        regime: str,
        sector_adjustment: float,
        macro_data: Dict,
        actual_score: float = None
    ) -> str:
        """Generate one-line rationale using OpenAI."""
        system_prompt = """You are a senior macroeconomic strategist at a global investment management firm.
You specialize in analyzing how macroeconomic cycles, monetary policy, and regime changes affect sector performance.
Your analysis should be:
1. Macro-focused and regime-aware, explaining how economic conditions drive sector rotations
2. Policy-conscious, considering Fed policy, yield curves, and inflation impacts
3. Forward-looking, discussing implications for sector performance in current macro environment
4. Specific about the transmission mechanisms between macro conditions and sector fundamentals
5. Around 100-150 words with clear, actionable macro-investment insights

CRITICAL: You MUST cite specific numerical values from the data provided (e.g., "With yield curve slope at 0.50 and inflation at 3.2%..." or "The sector adjustment of +15 points reflects...").
Reference the exact metrics and scores given to you. Explain HOW each metric contributed to the final score.
State which data sources informed your analysis (e.g., yield curve, inflation rate, unemployment, regime classification).

ACCURACY RULES — ZERO TOLERANCE FOR ERRORS:
- ONLY use the exact numerical values provided in the user prompt below. NEVER invent, round differently, or hallucinate statistics.
- If a metric says 'Data unavailable', say so — do NOT substitute a made-up value.
- Before writing each number, mentally verify it matches the data provided verbatim.
- Do NOT claim a yield curve slope, inflation rate, or interest rate that is not explicitly in the data below."""
        
        base_score = 50  # Neutral baseline
        final_score = actual_score if actual_score is not None else (base_score + sector_adjustment)
        
        user_prompt = f"""
MACROECONOMIC ANALYSIS REQUEST: {ticker}
Sector: {sector}
FINAL MACRO SCORE: {final_score:.1f}/100

CURRENT ECONOMIC REGIME: {regime.replace('_', ' ').title()}
SECTOR ADJUSTMENT: {sector_adjustment:+.1f} points from neutral baseline

MACROECONOMIC INDICATORS:
• Yield Curve Slope: {macro_data.get('yield_curve_slope', 'Data unavailable')}
• Year-over-Year Inflation: {macro_data.get('inflation_yoy', 'Data unavailable')}%
• Interest Rate Environment: {macro_data.get('fed_funds_rate', 'Data unavailable')}
• Economic Growth Phase: {regime.replace('_', ' ').title()}

SCORING CONTEXT:
- Scores above 70 = Sector strongly favored in current macro environment
- Scores 55-70 = Sector moderately benefits from macro conditions  
- Scores 45-55 = Sector neutral to current macro environment
- Scores 30-45 = Sector faces macro headwinds but manageable
- Scores below 30 = Sector significantly challenged by macro conditions

ANALYSIS REQUEST:
As a macroeconomic strategist, provide a comprehensive analysis explaining why {sector} sector earned a {final_score:.1f}/100 macro score in the current {regime.replace('_', ' ')} regime environment.
Address:
1. How does the current economic regime specifically impact {sector} sector fundamentals?
2. What are the key transmission mechanisms (interest rates, inflation, growth) affecting this sector?
3. How do current monetary policy and yield curve conditions influence sector performance?
4. What macro risks and opportunities should investors consider for {sector} exposure?
5. How does this macro environment affect the sector's relative attractiveness vs alternatives?

Focus on actionable insights about macro-driven sector allocation and timing."""
        
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
            
            # Enhanced fallback explanations with macro context
            if sector_adjustment > 5:
                return f"{sector} strongly benefits in {regime.replace('_', ' ')} regime (+{sector_adjustment:.1f}). This economic environment typically creates favorable conditions for {sector.lower()} sector fundamentals through supportive monetary policy, yield curve positioning, and economic growth dynamics that enhance sector-specific performance drivers."
            elif sector_adjustment < -5:
                return f"{sector} faces significant macro headwinds in {regime.replace('_', ' ')} regime ({sector_adjustment:.1f}). Current economic conditions create challenging dynamics for {sector.lower()} sector performance through interest rate pressures, yield curve impacts, and macro regime characteristics that typically constrain sector fundamentals and relative returns."
            else:
                return f"{sector} maintains neutral positioning in {regime.replace('_', ' ')} regime ({sector_adjustment:+.1f}). While current macroeconomic conditions don't strongly favor or penalize {sector.lower()} exposure, investors should monitor key macro indicators including Fed policy changes, yield curve evolution, and regime transitions that could shift sector dynamics."