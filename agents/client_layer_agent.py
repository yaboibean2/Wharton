"""
Client Layer Agent
Validates compliance with Investment Policy Statement (IPS).
Filters out violations and adjusts scores to respect client constraints.
"""

from typing import Dict, Any, List, Tuple
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ClientLayerAgent(BaseAgent):
    """
    Client mandate compliance agent.
    Enforces IPS constraints and adjusts recommendations accordingly.
    This is the final validation layer before recommendations.
    """
    
    def __init__(self, config: Dict[str, Any], ips_config: Dict[str, Any], openai_client=None):
        super().__init__("ClientLayerAgent", config, openai_client)
        self.ips = ips_config
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate stock against IPS and calculate independent compliance score.
        
        Now uses a true 0-100 scoring system like other agents:
        - 100: Perfect IPS fit with all preferences matched
        - 80-99: Excellent fit, passes all constraints
        - 60-79: Good fit, minor preference mismatches
        - 40-59: Acceptable, some constraint concerns
        - 20-39: Poor fit, significant concerns
        - 0-19: Very poor fit or violations
        
        Returns:
            - eligible: bool (whether stock passes all constraints)
            - score: float (independent 0-100 IPS compliance score)
            - final_score: float (blended score adjusted for IPS)
            - violations: list (any IPS violations detected)
            - constraints: dict (relevant caps and limits)
        """
        fundamentals = data.get('fundamentals', {})
        current_score = data.get('blended_score', 50)
        portfolio = data.get('portfolio', {})
        
        # Run all compliance checks and build independent score
        eligible = True
        violations = []
        constraints = {}
        
        # Start with base score of 100 (perfect compliance)
        compliance_score = 100
        score_deductions = []
        
        # 1. Check exclusions (CRITICAL - automatic fail)
        excluded, reason = self._check_exclusions(ticker, fundamentals)
        if excluded:
            eligible = False
            violations.append(reason)
            compliance_score = 0  # Hard fail
            score_deductions.append(f"Exclusion violation: -100")
        
        # Only continue scoring if not excluded
        if eligible:
            # 2. Check universe constraints (CRITICAL - automatic fail)
            universe_ok, reason = self._check_universe_constraints(fundamentals)
            if not universe_ok:
                eligible = False
                violations.append(reason)
                compliance_score = max(0, compliance_score - 50)  # Major penalty
                score_deductions.append(f"Universe constraint: -50")
            
            # 3. Check position limits (affects score but not eligibility)
            position_cap = self._calculate_position_cap(ticker, fundamentals, portfolio)
            constraints['max_position_pct'] = position_cap
            if position_cap < 5:
                compliance_score -= 10
                score_deductions.append(f"Tight position limit: -10")
            
            # 4. Check sector limits (affects score but not eligibility)
            sector = fundamentals.get('sector', 'Unknown')
            sector_cap = self._calculate_sector_cap(sector, portfolio)
            constraints['max_sector_pct'] = sector_cap
            if sector_cap < 10:
                compliance_score -= 10
                score_deductions.append(f"Limited sector capacity: -10")
            
            # 5. ESG preferences alignment
            esg_score = self._score_esg_alignment(fundamentals)
            compliance_score += esg_score
            if esg_score != 0:
                score_deductions.append(f"ESG alignment: {esg_score:+.0f}")
            
            # 6. Risk tolerance alignment
            risk_score = self._score_risk_alignment(fundamentals)
            compliance_score += risk_score
            if risk_score != 0:
                score_deductions.append(f"Risk tolerance: {risk_score:+.0f}")
            
            # 7. Beta band compliance
            beta_ok, beta_score = self._score_beta_compliance(fundamentals)
            compliance_score += beta_score
            if beta_score != 0:
                score_deductions.append(f"Beta compliance: {beta_score:+.0f}")
        
        # Clamp compliance score to 0-100 range
        compliance_score = max(0, min(100, compliance_score))
        
        # Calculate final adjusted score (blend with other agents)
        # If ineligible, heavily penalize the final score
        if not eligible:
            final_score = min(current_score * 0.3, 30)  # Cap at 30 for ineligible stocks
        else:
            # Weight client compliance 20% and blended score 80%
            final_score = (compliance_score * 0.2) + (current_score * 0.8)
        
        final_score = max(0, min(100, final_score))
        
        # Generate rationale
        rationale = self._generate_rationale(
            ticker, eligible, violations, score_deductions, constraints, compliance_score
        )
        
        return {
            'eligible': eligible,
            'score': round(final_score, 2),  # Final adjusted score
            'compliance_score': round(compliance_score, 2),  # Independent 0-100 IPS score
            'original_score': current_score,
            'rationale': rationale,
            'violations': violations,
            'score_deductions': score_deductions,
            'constraints': constraints,
            'details': {
                'compliance_breakdown': score_deductions,
                'eligible': eligible,
                'position_cap': constraints.get('max_position_pct'),
                'sector_cap': constraints.get('max_sector_pct')
            }
        }
    
    def _check_exclusions(self, ticker: str, fundamentals: Dict) -> Tuple[bool, str]:
        """Check if stock is excluded by IPS."""
        exclusions = self.ips.get('exclusions', {})
        
        # Check ticker exclusions
        excluded_tickers = exclusions.get('tickers', [])
        if ticker in excluded_tickers:
            return True, f"Ticker {ticker} explicitly excluded"
        
        # Check sector exclusions
        excluded_sectors = exclusions.get('sectors', [])
        sector = fundamentals.get('sector', '')
        if sector in excluded_sectors:
            return True, f"Sector {sector} excluded by IPS"
        
        # ESG screens (simplified - would need more data in practice)
        esg_screens = exclusions.get('esg_screens', [])
        # This would require ESG data integration
        
        return False, ""
    
    def _check_universe_constraints(self, fundamentals: Dict) -> Tuple[bool, str]:
        """Check universe constraints (price, volume, geography)."""
        universe = self.ips.get('universe', {})
        
        # Min price check
        min_price = universe.get('min_price', 3)
        current_price = fundamentals.get('current_price')
        if current_price and current_price < min_price:
            return False, f"Price ${current_price:.2f} below min ${min_price}"
        
        # Min volume check
        min_volume = universe.get('min_avg_daily_volume', 2000000)
        avg_volume = fundamentals.get('avg_volume', 0)
        if avg_volume > 0:
            # Convert share volume to dollar volume (approx)
            dollar_volume = avg_volume * (current_price or 1)
            if dollar_volume < min_volume:
                return False, f"Avg daily volume ${dollar_volume:,.0f} below min ${min_volume:,.0f}"
        
        return True, ""
    
    def _calculate_position_cap(
        self,
        ticker: str,
        fundamentals: Dict,
        portfolio: Dict
    ) -> float:
        """Calculate maximum position size for this stock."""
        limits = self.ips.get('position_limits', {})
        max_position = limits.get('max_position_pct', 8)
        
        # Could adjust based on liquidity, volatility, etc.
        # For now, use the static limit
        return max_position
    
    def _calculate_sector_cap(self, sector: str, portfolio: Dict) -> float:
        """Calculate remaining sector capacity."""
        limits = self.ips.get('position_limits', {})
        max_sector = limits.get('max_sector_pct', 30)
        
        # Calculate current sector exposure
        current_sector_pct = portfolio.get('sector_exposures', {}).get(sector, 0)
        remaining = max_sector - current_sector_pct
        
        return max(0, remaining)
    
    def _score_esg_alignment(self, fundamentals: Dict) -> float:
        """Score ESG alignment on -10 to +10 scale."""
        # This would require ESG data
        # Placeholder for now - returns 0 (neutral)
        return 0
    
    def _score_risk_alignment(self, fundamentals: Dict) -> float:
        """Score risk tolerance alignment on -20 to +10 scale."""
        client = self.ips.get('client', {})
        risk_tolerance = client.get('risk_tolerance', 'moderate')
        
        beta = fundamentals.get('beta', 1.0)
        volatility = fundamentals.get('volatility_pct', 20)
        
        if risk_tolerance == 'low':
            # Prefer lower beta, lower vol
            if beta and beta < 0.8:
                return 10  # Excellent defensive fit
            elif beta and beta < 1.0:
                return 5   # Good defensive fit
            elif beta and beta > 1.3:
                return -20  # Very poor fit for conservative client
            elif beta and beta > 1.2:
                return -10  # Poor fit for conservative client
        
        elif risk_tolerance == 'high':
            # More tolerance for volatility
            if beta and beta > 1.3:
                return 10  # Excellent aggressive fit
            elif beta and beta > 1.2:
                return 5   # Good aggressive fit
            elif beta and beta < 0.8:
                return -5  # Too conservative for aggressive client
        
        else:  # moderate
            # Prefer beta near 1.0
            if beta and 0.9 <= beta <= 1.1:
                return 5  # Perfect moderate fit
            elif beta and (beta < 0.7 or beta > 1.4):
                return -10  # Poor fit for moderate client
        
        return 0
    
    def _score_beta_compliance(self, fundamentals: Dict) -> Tuple[bool, float]:
        """Score beta band compliance on -30 to 0 scale."""
        constraints = self.ips.get('portfolio_constraints', {})
        beta_min = constraints.get('beta_min', 0.7)
        beta_max = constraints.get('beta_max', 1.1)
        
        beta = fundamentals.get('beta')
        if not beta:
            return True, 0  # Can't check without beta
        
        if beta < beta_min:
            # Too defensive for portfolio
            deviation = beta_min - beta
            penalty = min(30, deviation * 20)  # Max -30 points
            return False, -penalty
        elif beta > beta_max:
            # Too aggressive for portfolio
            deviation = beta - beta_max
            penalty = min(30, deviation * 20)  # Max -30 points
            return False, -penalty
        
        return True, 0  # Within band, no penalty
    
    def _generate_rationale(
        self,
        ticker: str,
        eligible: bool,
        violations: List[str],
        score_deductions: List[str],
        constraints: Dict,
        compliance_score: float
    ) -> str:
        """Generate compliance rationale."""
        if not eligible:
            # Return first violation with score
            return f"IPS VIOLATION ({compliance_score:.0f}/100): {violations[0]}"
        
        # Categorize based on compliance score
        if compliance_score >= 90:
            status = "Excellent IPS fit"
        elif compliance_score >= 75:
            status = "Good IPS compliance"
        elif compliance_score >= 60:
            status = "Acceptable compliance"
        elif compliance_score >= 40:
            status = "Marginal compliance"
        else:
            status = "Poor IPS fit"
        
        # Add key factors
        if score_deductions:
            factors = ', '.join(score_deductions[:2])  # Top 2 factors
            return f"{status} ({compliance_score:.0f}/100): {factors}"
        
        # Check if tight constraints
        max_pos = constraints.get('max_position_pct', 8)
        if max_pos < 5:
            return f"{status} ({compliance_score:.0f}/100), limited to {max_pos:.1f}% position"
        
        return f"{status} ({compliance_score:.0f}/100), no restrictions"
