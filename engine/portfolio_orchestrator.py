"""
Portfolio Orchestrator
Coordinates all agents and blends scores to generate final recommendations.
Handles position sizing and portfolio construction.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import json
import os
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median as _median, quantiles as _quantiles

from agents.value_agent import ValueAgent
from agents.growth_momentum_agent import GrowthMomentumAgent
from agents.macro_regime_agent import MacroRegimeAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from data.enhanced_data_provider import EnhancedDataProvider
from engine.ai_portfolio_selector import AIPortfolioSelector

logger = logging.getLogger(__name__)


def _load_learned_phase_durations() -> dict:
    """Load phase durations from data/step_times.json.

    Returns dict with keys for legacy phases ('data_gather', 'agents', 'blend')
    plus per-step keys ('fundamentals', 'price_history', 'benchmark',
    'value_agent', 'growth_momentum_agent', 'macro_regime_agent',
    'risk_agent', 'sentiment_agent') and 'total' / 'avg_total'.
    Falls back to conservative defaults if the file is missing or empty.
    """
    defaults = {
        'data_gather': 45.0, 'agents': 25.0, 'blend': 1.0,
        'total': 70.0, 'avg_total': 70.0,
        # Per-step defaults (seconds)
        'fundamentals': 40.0, 'price_history': 5.0, 'benchmark': 0.5,
        'value_agent': 12.0, 'growth_momentum_agent': 12.0,
        'macro_regime_agent': 12.0, 'risk_agent': 12.0,
        'sentiment_agent': 15.0,
    }
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'step_times.json')
        if not os.path.exists(path):
            return defaults
        with open(path, 'r') as f:
            raw = json.load(f)
        st = raw.get('step_times', {})
        if not st:
            return defaults

        def _med(key):
            vals = st.get(key, [])
            return _median(vals) if vals else None

        def _avg(key):
            vals = st.get(key, [])
            return sum(vals) / len(vals) if vals else None

        def _p75(key):
            """75th percentile — conservative estimate for variable durations."""
            vals = st.get(key, [])
            if not vals:
                return None
            if len(vals) < 4:
                return max(vals)
            qs = _quantiles(sorted(vals), n=4)
            return qs[2]  # 75th percentile

        # Legacy phase keys
        dg = _med('1')
        ag = _med('2')
        bl = _med('3')
        tot = _med('total')
        avg_tot = _avg('total')

        # agents_wall: 75th percentile of actual parallel wall time
        # This is the bottleneck duration and is highly variable,
        # so 75th percentile avoids systematic underestimation.
        aw_p75 = _p75('agents_wall')

        result = {
            'data_gather': round(dg, 1) if dg is not None else defaults['data_gather'],
            'agents':      round(ag, 1) if ag is not None else defaults['agents'],
            'blend':       round(bl, 1) if bl is not None else defaults['blend'],
            'total':       round(tot, 1) if tot is not None else defaults['total'],
            'avg_total':   round(avg_tot, 1) if avg_tot is not None else defaults['avg_total'],
            'agents_wall_p75': round(aw_p75, 1) if aw_p75 is not None else defaults['agents'],
        }

        # Per-step keys
        for step_key in ('fundamentals', 'price_history', 'benchmark',
                         'value_agent', 'growth_momentum_agent',
                         'macro_regime_agent', 'risk_agent', 'sentiment_agent'):
            med_val = _med(step_key)
            result[step_key] = round(med_val, 2) if med_val is not None else defaults.get(step_key, 10.0)

        return result
    except Exception:
        return defaults


class PortfolioOrchestrator:
    """
    Main orchestration engine for the multi-agent investment system.
    Coordinates data gathering, agent analysis, score blending, and portfolio construction.
    """

    # Learned phase durations loaded once at import time
    _learned_phases: dict = _load_learned_phase_durations()
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        ips_config: Dict[str, Any],
        enhanced_data_provider: EnhancedDataProvider,
        openai_client=None,
        gemini_api_key=None
    ):
        self.model_config = model_config
        self.ips_config = ips_config
        self.data_provider = enhanced_data_provider
        self.openai_client = openai_client
        self.gemini_api_key = gemini_api_key
        logger.info("Using Enhanced Data Provider with premium fallbacks")
        
        # Initialize agents with their dependencies
        self.agents = {}
        
        # Core analysis agents
        self.agents['value_agent'] = ValueAgent(model_config, openai_client)
        self.agents['growth_momentum_agent'] = GrowthMomentumAgent(model_config, openai_client)
        self.agents['macro_regime_agent'] = MacroRegimeAgent(model_config, openai_client)
        self.agents['risk_agent'] = RiskAgent(model_config, openai_client)
        self.agents['sentiment_agent'] = SentimentAgent(model_config, openai_client)

        # Initialize AI Portfolio Selector
        if openai_client and gemini_api_key:
            self.ai_selector = AIPortfolioSelector(
                openai_client, gemini_api_key, ips_config, model_config
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
            'growth_momentum_agent': agent_weights_config.get('growth_momentum', 0.40),  # DOUBLED - Upside potential is KEY
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
        agent_weights: Dict[str, float] = None,
        progress_callback=None,
        regime_modulation: bool = False,
        regime_sensitivity: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Analyze a single stock using all agents.

        Args:
            progress_callback: Optional callable(progress_pct: float, message: str)
                that receives progress updates. The progress_pct is 0-100 and
                message includes the ETA suffix.

        Returns complete analysis with scores, rationale, and recommendations.
        """
        logger.info(f"Starting comprehensive analysis for {ticker} as of {analysis_date}")

        # Phase durations loaded from data/step_times.json (learned from timing runs)
        lp = PortfolioOrchestrator._learned_phases
        # Phase 1: Data gathering   0-42%  (~{lp['data_gather']}s)
        # Phase 2: Agent analysis   42-98% (~{lp['agents']}s — agents run in PARALLEL)
        # Phase 3: Blend/finalize   98-100% (~{lp['blend']}s)
        PHASE_AGENTS = (42, 98)         # 42% to 98%

        analysis_start_time = time.time()

        # Per-step timing tracker — records start/end for every individual step
        _step_timings = {}

        # Reset ETA tracking for this new analysis
        # (prevents stale _eta_previous from previous runs causing counting-up bug)
        _eta_state = {'previous': None}

        def update_progress(message, progress_pct):
            """Update progress via the direct callback passed from the caller."""
            if not progress_callback:
                return

            try:
                # Append ETA to the message
                time_display = ""
                if progress_pct >= 3:
                    elapsed = time.time() - analysis_start_time
                    time_display = PortfolioOrchestrator._estimate_time_remaining(
                        elapsed, progress_pct, _eta_state
                    )

                progress_callback(progress_pct, f"{message}{time_display}")

            except Exception as e:
                logger.error(f"Progress update failed: {e}")
        
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
        
        # 1. Gather all data (Phase 1: 0-42%)
        update_progress(f"Fetching data for {ticker} from multiple sources...", 3)

        data_gather_start = time.time()
        try:
            _data_cb_count = [0]

            def data_progress_cb(msg):
                """Callback for granular data gathering updates.
                Also advances the progress bar from 3% to 40%.
                """
                _data_cb_count[0] += 1
                # 4 callbacks expected: 1 initial + 3 task completions
                # Map to 3% -> 40% range
                pct = int(3 + (_data_cb_count[0] / 4) * 37)
                pct = min(pct, 40)
                update_progress(msg, pct)

            data = self._gather_data(ticker, analysis_date, existing_portfolio,
                                     progress_callback=data_progress_cb,
                                     step_timings=_step_timings)
        except Exception as e:
            logger.error(f"Error gathering data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ticker': ticker,
                'error': f'Data gathering failed: {str(e)}',
                'fundamentals': {},
                'price_history': {},
                'agent_results': {},
                'agent_scores': {},
                'agent_rationales': {},
                'blended_score': 0,
                'final_score': 0,
                'eligible': False
            }

        data_gather_elapsed = time.time() - data_gather_start

        update_progress(f"Data gathered for {ticker} in {data_gather_elapsed:.0f}s", 42)
        
        if not data or not data.get('fundamentals'):
            logger.warning(f"No fundamental data for {ticker}")
            update_progress(f"No fundamental data found for {ticker}", 10)
            return {
                'ticker': ticker, 
                'error': 'No data available',
                'fundamentals': {},
                'price_history': {},
                'agent_results': {},
                'agent_scores': {},
                'agent_rationales': {},
                'blended_score': 0,
                'final_score': 0,
                'eligible': False
            }
        
        # ── Data quality gate: reject synthetic / no-data results ──
        fundamentals = data.get('fundamentals', {})
        _fund_sources = fundamentals.get('data_sources', [])
        _is_synthetic = isinstance(_fund_sources, list) and 'synthetic' in _fund_sources
        _fund_price   = fundamentals.get('price')
        _has_price    = isinstance(_fund_price, (int, float)) and _fund_price > 0
        if _is_synthetic or not _has_price:
            logger.warning(
                f"No real market data for {ticker} "
                f"(synthetic={_is_synthetic}, price={_fund_price})"
            )
            return {
                'ticker': ticker,
                'error': 'ticker_not_found',
                'fundamentals': {},
                'price_history': {},
                'agent_results': {},
                'agent_scores': {},
                'agent_rationales': {},
                'blended_score': 0,
                'final_score': 0,
                'eligible': False,
            }

        # Show specific extracted values
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
        
        update_progress(f"Data ready: ${price} price, {pe_ratio} P/E, {market_cap_str} mkt cap", 42)

        # 2. Phase 2: Run agents IN PARALLEL (42-98%)
        # For ETFs, skip Value and Growth agents (P/E, EPS growth are meaningless)
        is_etf = data.get('fundamentals', {}).get('is_etf', False)
        _etf_skip = {'value_agent', 'growth_momentum_agent'} if is_etf else set()

        agents_to_run = {
            name: agent for name, agent in self.agents.items()
            if name not in _etf_skip
        }

        if _etf_skip:
            logger.info(f"ETF detected ({ticker}) — skipping agents: {_etf_skip}")

        agent_results = {}
        total_agents = len(agents_to_run)

        agent_labels_map = {
            'value_agent': 'Value',
            'growth_momentum_agent': 'Growth',
            'macro_regime_agent': 'Macro Regime',
            'risk_agent': 'Risk',
            'sentiment_agent': 'Sentiment'
        }

        update_progress(f"Running {total_agents} agents in parallel...", 43)

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_agent = {}
            _agent_start_times = {}
            for i, (agent_name, agent) in enumerate(agents_to_run.items()):
                # Small stagger between agent launches to avoid API burst
                if i > 0:
                    time.sleep(0.2)
                _agent_start_times[agent_name] = time.time()
                future = executor.submit(agent.analyze, ticker, data)
                future_to_agent[future] = agent_name

            completed_count = 0
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                agent_elapsed = time.time() - _agent_start_times[agent_name]
                completed_count += 1
                agent_label = agent_labels_map.get(agent_name, agent_name.replace('_agent', '').title())

                try:
                    result = future.result()
                    agent_results[agent_name] = result

                    score = result.get('score')
                    if score is None:
                        score = 50
                        result['score'] = 50
                    logger.info(f"{agent_name}: {score:.1f} - {result['rationale']}")

                    if 'sentiment' in agent_name.lower():
                        num_articles = result.get('details', {}).get('num_articles', 0)
                        completion_msg = f"{agent_label} Agent: {num_articles} articles analyzed, score {score:.0f}/100"
                    else:
                        completion_msg = f"{agent_label} Agent complete: {score:.0f}/100"
                except Exception as e:
                    logger.error(f"Error in {agent_name} for {ticker}: {e}")
                    agent_results[agent_name] = {
                        'score': 50,
                        'rationale': f'Analysis failed: {str(e)}',
                        'details': {}
                    }
                    completion_msg = f"{agent_label} Agent: analysis failed"

                # Record per-agent timing
                _step_timings[agent_name] = round(agent_elapsed, 3)

                # Progress from 42% to 98% based on completion count
                pct = PHASE_AGENTS[0] + (completed_count / total_agents) * (PHASE_AGENTS[1] - PHASE_AGENTS[0])
                update_progress(completion_msg, int(pct))
        
        # 3. Phase 3: Blend scores and finalize (98-100%)
        blend_start = time.time()
        update_progress(f"Blending agent scores with configured weights...", 98)
        blended_score = self._blend_scores(
            agent_results,
            regime_modulation=regime_modulation,
            regime_sensitivity=regime_sensitivity,
        )

        recommendation = self._generate_recommendation(blended_score)
        update_progress(f"Analysis complete: {blended_score:.1f}/100 - {recommendation}", 99)
        final_score = blended_score

        _step_timings['blend'] = round(time.time() - blend_start, 3)

        total_time = time.time() - analysis_start_time
        _step_timings['total'] = round(total_time, 3)
        _step_timings['data_gather'] = round(data_gather_elapsed, 3)
        # Agents wall-clock (from end of data gather to end of last agent)
        if _step_timings.get('value_agent') is not None:
            # Max of all agent durations (they run in parallel, so wall-clock = max)
            agent_keys = [k for k in _step_timings if k.endswith('_agent')]
            _step_timings['agents_wall'] = round(max(_step_timings[k] for k in agent_keys), 3) if agent_keys else 0

        update_progress(f"Analysis complete: {final_score:.1f}/100 ({total_time:.0f}s total)", 100)

        # Restore original weights if they were changed
        if agent_weights:
            self.agent_weights = original_weights

        # Extract agent scores and rationales for backward compatibility
        agent_scores = {agent_name: (result.get('score') or 50) for agent_name, result in agent_results.items()}
        agent_rationales = {agent_name: result.get('rationale', 'Analysis not available') for agent_name, result in agent_results.items()}

        return {
            'ticker': ticker,
            'analysis_date': analysis_date,
            'agent_results': agent_results,
            'agent_scores': agent_scores,
            'agent_rationales': agent_rationales,
            'blended_score': blended_score,
            'final_score': final_score,
            'eligible': True,
            'recommendation': self._generate_recommendation(final_score),
            'rationale': self._generate_comprehensive_rationale_simple(ticker, agent_results, final_score, data),
            'fundamentals': data.get('fundamentals', {}),
            'price_history': data.get('price_history', {}),
            'step_timings': _step_timings,
            'detected_regime': agent_results.get('macro_regime_agent', {}).get('regime', 'unknown') if regime_modulation else None,
            'regime_adjusted_weights': getattr(self, '_last_regime_adjusted_weights', None),
        }
    
    def analyze_stock(
        self,
        ticker: str,
        analysis_date: str = None,
        existing_portfolio: List[Dict] = None,
        agent_weights: Dict[str, float] = None,
        progress_callback=None,
        regime_modulation: bool = False,
        regime_sensitivity: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Alias for analyze_single_stock method for backward compatibility.

        Args:
            ticker: Stock ticker symbol
            analysis_date: Date for analysis (defaults to today)
            existing_portfolio: Existing portfolio for context
            agent_weights: Custom agent weights for this analysis
            progress_callback: Optional callable(progress_pct, message) for progress updates
            regime_modulation: Whether to dynamically shift weights based on macro regime
            regime_sensitivity: How aggressively to shift (conservative/moderate/aggressive)

        Returns:
            Complete analysis results
        """
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y-%m-%d')

        return self.analyze_single_stock(
            ticker=ticker,
            analysis_date=analysis_date,
            existing_portfolio=existing_portfolio,
            agent_weights=agent_weights,
            progress_callback=progress_callback,
            regime_modulation=regime_modulation,
            regime_sensitivity=regime_sensitivity,
        )
    
    @staticmethod
    def _estimate_time_remaining(elapsed: float, progress_pct: float, eta_state=None) -> str:
        """
        Phase-aware time remaining estimation using learned durations.

        Durations are loaded from data/step_times.json (populated by
        test_pipeline_timing.py) so estimates stay calibrated to the
        real environment.  Falls back to conservative defaults when
        no timing data is available.

        Args:
            elapsed: seconds since analysis started
            progress_pct: current progress 0-100
            eta_state: dict with 'previous' key for EMA smoothing
        """
        if progress_pct < 3 or elapsed < 1.0:
            return ""

        lp = PortfolioOrchestrator._learned_phases
        # Phase definitions: (start_pct, end_pct, learned_duration_seconds)
        phases = [
            (0,  42, lp['data_gather']),   # Data gathering
            (42, 98, lp['agents']),         # Agent analysis (parallel)
            (98, 100, lp['blend']),         # Finalization
        ]

        # Find current phase and calculate remaining time
        remaining = 0.0
        found_current = False

        for start_pct, end_pct, duration in phases:
            if not found_current:
                if progress_pct < end_pct:
                    # We're in this phase
                    found_current = True
                    phase_progress = (progress_pct - start_pct) / (end_pct - start_pct) if end_pct > start_pct else 1.0
                    phase_progress = max(0.0, min(1.0, phase_progress))
                    remaining += duration * (1.0 - phase_progress)
                # else: we've passed this phase, skip it
            else:
                # Future phase - add full duration
                remaining += duration

        if not found_current:
            return ""

        # EMA smoothing to prevent display jumps
        if eta_state is not None:
            prev = eta_state.get('previous')
            if prev is not None and prev > 0:
                alpha = 0.4
                # Cap increases to 15% of previous to avoid upward jumps
                if remaining > prev * 1.15:
                    remaining = prev * 1.15
                remaining = alpha * remaining + (1 - alpha) * prev
            eta_state['previous'] = remaining

        # Ensure non-negative
        if remaining <= 0:
            return ""

        # Format
        secs = max(1, int(remaining))
        if secs < 60:
            return f" ~{secs}s"
        else:
            mins = secs // 60
            s = secs % 60
            return f" ~{mins}m {s}s"

    def _blend_scores(self, agent_results: Dict[str, Dict],
                      regime_modulation: bool = False,
                      regime_sensitivity: str = "moderate") -> float:
        """
        Blend agent scores using configured weights.

        When regime_modulation is True, weights are dynamically shifted based
        on the macro regime detected by the macro_regime_agent.  The upside
        multiplier is skipped in that mode because the theory-based weights
        already encode factor-exposure philosophy.
        """
        # Determine effective weights
        effective_weights = dict(self.agent_weights)

        if regime_modulation:
            regime = (agent_results.get('macro_regime_agent') or {}).get('regime', 'expansion')
            effective_weights = self._apply_regime_modulation(
                effective_weights, regime, regime_sensitivity
            )
            # Stash for the caller to include in the result dict
            self._last_regime_adjusted_weights = {
                k: round(v, 4) for k, v in effective_weights.items()
            }
        else:
            self._last_regime_adjusted_weights = None

        # Calculate base weighted score
        # Exclude agents that flagged data_unavailable (e.g. sentiment with
        # no news) — their weight is automatically redistributed because
        # we normalise by total_weight.
        total_score = 0
        total_weight = 0

        for agent_name, weight in effective_weights.items():
            if agent_name in agent_results:
                result = agent_results[agent_name]
                if result.get('data_unavailable'):
                    logger.info(
                        f"Excluding {agent_name} from blend — data unavailable, "
                        f"redistributing {weight:.0%} weight to remaining agents"
                    )
                    continue
                score = result.get('score') or 50
                total_score += score * weight
                total_weight += weight

        base_score = total_score / total_weight if total_weight > 0 else 50

        # ---- When regime modulation is active, return pure weighted average ----
        if regime_modulation:
            logger.info(f"THEORY-BASED BLEND (regime modulation ON): {base_score:.1f}")
            return base_score

        # ========== UPSIDE POTENTIAL MULTIPLIER (non-theory-based presets only) ==========
        # Tighter bounds now that agent scores are properly centered around 50

        upside_multiplier = 1.0  # Start at neutral
        upside_factors = []

        # Factor 1: Growth/Momentum score
        if 'growth_momentum_agent' in agent_results:
            growth_score = agent_results['growth_momentum_agent'].get('score', 50)
            if growth_score >= 75:
                upside_multiplier += 0.08
                upside_factors.append(f"Strong growth ({growth_score:.0f}/100) \u2192 +8% boost")
            elif growth_score >= 60:
                upside_multiplier += 0.04
                upside_factors.append(f"Good growth ({growth_score:.0f}/100) \u2192 +4% boost")
            elif growth_score < 30:
                upside_multiplier -= 0.06
                upside_factors.append(f"Weak growth ({growth_score:.0f}/100) \u2192 -6% penalty")

        # Factor 2: Sentiment
        if 'sentiment_agent' in agent_results:
            sentiment_score = agent_results['sentiment_agent'].get('score') or 50
            if sentiment_score >= 70:
                upside_multiplier += 0.05
                upside_factors.append(f"Positive sentiment ({sentiment_score:.0f}/100) \u2192 +5% boost")

        # Factor 3: Value (good value = more runway)
        if 'value_agent' in agent_results:
            value_score = agent_results['value_agent'].get('score', 50)
            if value_score >= 70:
                upside_multiplier += 0.04
                upside_factors.append(f"Attractive valuation ({value_score:.0f}/100) \u2192 +4% boost")

        # Factor 4: Penalize extreme risk
        if 'risk_agent' in agent_results:
            risk_score = agent_results['risk_agent'].get('score', 50)
            if risk_score < 25:
                upside_multiplier -= 0.08
                upside_factors.append(f"Extreme risk ({risk_score:.0f}/100) \u2192 -8% penalty")

        # Tighter cap: +/-15% max swing
        upside_multiplier = min(upside_multiplier, 1.15)
        upside_multiplier = max(upside_multiplier, 0.85)

        final_score = base_score * upside_multiplier

        # Log the upside calculation for transparency
        if upside_factors:
            logger.info(f"UPSIDE MULTIPLIER APPLIED: {upside_multiplier:.2f}x")
            logger.info(f"   Base Score: {base_score:.1f}")
            logger.info(f"   Upside Factors:")
            for factor in upside_factors:
                logger.info(f"     - {factor}")
            logger.info(f"   Final Score: {final_score:.1f} (boosted by {((upside_multiplier - 1) * 100):.0f}%)")

        return final_score

    # --- Regime-based weight modulation (Theory Based preset) ---
    # Shift table grounded in regime-switching research (Ang & Bekaert, 2002)
    _REGIME_SHIFTS = {
        'expansion': {
            'value_agent': -0.05,
            'growth_momentum_agent': +0.10,
            'macro_regime_agent': 0.0,
            'risk_agent': -0.05,
            'sentiment_agent': +0.05,
        },
        'recession': {
            'value_agent': +0.10,
            'growth_momentum_agent': -0.15,
            'macro_regime_agent': 0.0,
            'risk_agent': +0.10,
            'sentiment_agent': -0.05,
        },
        'high_inflation': {
            'value_agent': +0.10,
            'growth_momentum_agent': -0.10,
            'macro_regime_agent': 0.0,
            'risk_agent': +0.05,
            'sentiment_agent': -0.05,
        },
        'disinflation': {
            'value_agent': -0.05,
            'growth_momentum_agent': +0.10,
            'macro_regime_agent': 0.0,
            'risk_agent': -0.05,
            'sentiment_agent': +0.05,
        },
    }

    _SENSITIVITY_MULTIPLIERS = {
        'conservative': 0.5,
        'moderate': 1.0,
        'aggressive': 1.5,
    }

    def _apply_regime_modulation(
        self,
        base_weights: Dict[str, float],
        regime: str,
        sensitivity: str = "moderate",
    ) -> Dict[str, float]:
        """
        Shift agent weights based on the detected macro regime.

        Applies the shift table scaled by sensitivity, clamps each weight
        to >= 0.02, and renormalizes so all weights sum to 1.0.
        """
        multiplier = self._SENSITIVITY_MULTIPLIERS.get(sensitivity, 1.0)
        shifts = self._REGIME_SHIFTS.get(regime, {})

        adjusted = {}
        for agent, base_w in base_weights.items():
            shift = shifts.get(agent, 0.0) * multiplier
            adjusted[agent] = max(base_w + shift, 0.02)

        # Clamp-then-normalize may push values below the floor, so iterate
        for _ in range(3):
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total for k, v in adjusted.items()}
            if all(v >= 0.019 for v in adjusted.values()):
                break
            adjusted = {k: max(v, 0.02) for k, v in adjusted.items()}

        logger.info(
            f"REGIME MODULATION: regime={regime}, sensitivity={sensitivity} "
            f"| base={{{', '.join(f'{k}: {v:.2f}' for k, v in base_weights.items())}}} "
            f"| adjusted={{{', '.join(f'{k}: {v:.3f}' for k, v in adjusted.items())}}}"
        )

        return adjusted

    def _generate_recommendation(self, score: float) -> str:
        """Generate investment recommendation based on score."""
        if score >= 80:
            return "STRONG BUY"
        elif score >= 70:
            return "BUY"
        elif score >= 60:
            return "HOLD"
        elif score >= 40:
            return "WEAK HOLD"
        else:
            return "SELL"
    
    def _generate_comprehensive_rationale_simple(self, ticker: str, agent_results: Dict, final_score: float, data: Dict) -> str:
        """Generate comprehensive investment rationale."""
        fundamentals = data.get('fundamentals', {})
        rationale_parts = []

        rationale_parts.append("=" * 80)
        rationale_parts.append(f"COMPREHENSIVE INVESTMENT ANALYSIS: {ticker}")
        rationale_parts.append("=" * 80)

        company_name = fundamentals.get('name', ticker)
        sector = fundamentals.get('sector', 'Unknown')
        rationale_parts.append(f"\nCOMPANY OVERVIEW:")
        rationale_parts.append(f"Company: {company_name}")
        rationale_parts.append(f"Sector: {sector}")

        rationale_parts.append(f"\nKEY FINANCIAL METRICS:")
        price = fundamentals.get('price')
        if price:
            rationale_parts.append(f"Current Price: ${price:.2f}")
        market_cap = fundamentals.get('market_cap')
        if market_cap:
            if market_cap >= 1e12:
                rationale_parts.append(f"Market Cap: ${market_cap/1e12:.2f}T")
            elif market_cap >= 1e9:
                rationale_parts.append(f"Market Cap: ${market_cap/1e9:.2f}B")
            else:
                rationale_parts.append(f"Market Cap: ${market_cap/1e6:.2f}M")
        pe_ratio = fundamentals.get('pe_ratio')
        if pe_ratio:
            rationale_parts.append(f"P/E Ratio: {pe_ratio:.2f}")
        beta = fundamentals.get('beta')
        if beta:
            rationale_parts.append(f"Beta: {beta:.2f}")
        dividend_yield = fundamentals.get('dividend_yield')
        if dividend_yield:
            rationale_parts.append(f"Dividend Yield: {dividend_yield*100:.2f}%")

        rationale_parts.append(f"\nMULTI-AGENT ANALYSIS:")
        rationale_parts.append("=" * 80)
        agent_order = ['value_agent', 'growth_momentum_agent', 'macro_regime_agent', 'risk_agent', 'sentiment_agent']
        agent_labels = {
            'value_agent': 'VALUE ANALYSIS',
            'growth_momentum_agent': 'GROWTH ANALYSIS',
            'macro_regime_agent': 'MACROECONOMIC ANALYSIS',
            'risk_agent': 'RISK ASSESSMENT',
            'sentiment_agent': 'MARKET SENTIMENT ANALYSIS'
        }
        for agent_name in agent_order:
            if agent_name in agent_results:
                result = agent_results[agent_name]
                score = result.get('score') or 50
                rationale = result.get('rationale', 'Analysis not available')
                rationale_parts.append(f"\n{agent_labels.get(agent_name, agent_name.upper())}:")
                rationale_parts.append(f"Score: {score:.2f}/100")
                rationale_parts.append(f"{rationale}")
                rationale_parts.append("-" * 80)

        rationale_parts.append(f"\nFINAL RECOMMENDATION:")
        rationale_parts.append(f"Final Score: {final_score:.2f}/100")
        rationale_parts.append("=" * 80)

        return '\n'.join(rationale_parts)

    def _check_ips_eligibility(self, ticker: str, fundamentals: dict, blended_score: float) -> bool:
        """Check if a stock meets basic IPS eligibility constraints."""
        ips = self.ips_config

        # Check minimum price
        price = fundamentals.get('price', 0)
        min_price = ips.get('universe', {}).get('min_price', 1.0)
        if price and price < min_price:
            return False

        # Check minimum market cap
        market_cap = fundamentals.get('market_cap', 0)
        min_market_cap = ips.get('universe', {}).get('min_market_cap', 0)
        if market_cap and market_cap < min_market_cap:
            return False

        # Check excluded sectors
        sector = fundamentals.get('sector', '')
        excluded_sectors = ips.get('exclusions', {}).get('sectors', [])
        if sector and excluded_sectors:
            if sector.lower() in [s.lower() for s in excluded_sectors]:
                return False

        # Check beta range
        beta = fundamentals.get('beta')
        if beta:
            beta_min = ips.get('portfolio_constraints', {}).get('beta_min', 0)
            beta_max = ips.get('portfolio_constraints', {}).get('beta_max', 999)
            if beta < beta_min or beta > beta_max:
                return False

        return True
    
    def _gather_data(
        self,
        ticker: str,
        analysis_date: str,
        existing_portfolio: Dict = None,
        progress_callback=None,
        step_timings=None
    ) -> Dict[str, Any]:
        """Gather all necessary data for analysis using parallel API calls for speed."""
        # Calculate date range (1 year lookback) - ensure no future dates
        end_date = analysis_date
        start_date = pd.to_datetime(analysis_date) - pd.DateOffset(years=1)
        start_date = start_date.strftime('%Y-%m-%d')

        # Ensure we don't use future dates that break API calls
        today = datetime.now().date()
        if pd.to_datetime(end_date).date() > today:
            end_date = today.strftime('%Y-%m-%d')
            start_date = (pd.to_datetime(today) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

        data = {
            'ticker': ticker,
            'analysis_date': analysis_date,
        }

        benchmark = self.ips_config.get('universe', {}).get('benchmark', '^GSPC')

        if progress_callback:
            progress_callback(f"Querying Polygon, Alpha Vantage, Perplexity for {ticker} fundamentals...")

        # PARALLEL DATA GATHERING - Run all 3 API calls simultaneously
        _data_start_times = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            # Task 1: Get fundamentals (API calls - slowest, ~30-40s)
            _data_start_times['fundamentals'] = time.time()
            if hasattr(self.data_provider, 'get_fundamentals_enhanced'):
                futures['fundamentals'] = executor.submit(
                    self.data_provider.get_fundamentals_enhanced, ticker
                )
            else:
                futures['fundamentals'] = executor.submit(
                    self.data_provider.get_fundamentals, ticker
                )

            # Task 2: Get price history (Polygon/Alpha Vantage API - medium, ~3-5s)
            _data_start_times['price_history'] = time.time()
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
            _data_start_times['benchmark'] = time.time()
            futures['benchmark'] = executor.submit(
                self._create_benchmark_data, benchmark, start_date, end_date
            )

            task_labels = {
                'fundamentals': 'Fundamentals (P/E, EPS, market cap, financials)',
                'price_history': 'Price history (1 year daily prices)',
                'benchmark': 'Benchmark data (S&P 500 comparison)'
            }
            completed_tasks = []

            # Collect results using as_completed for real parallel processing
            future_to_name = {v: k for k, v in futures.items()}
            for future in as_completed(futures.values()):
                name = future_to_name[future]
                task_elapsed = time.time() - _data_start_times[name]
                try:
                    result = future.result()
                    if name == 'fundamentals':
                        data['fundamentals'] = result
                        if result:
                            logger.info(f"ORCHESTRATOR RECEIVED FUNDAMENTALS FOR {ticker}:")
                            logger.info(f"   price: {result.get('price')} pe_ratio: {result.get('pe_ratio')} beta: {result.get('beta')}")
                            logger.info(f"   data_sources: {result.get('data_sources')}")
                    elif name == 'price_history':
                        data['_price_history_raw'] = result
                    elif name == 'benchmark':
                        data['benchmark_history'] = result
                    logger.info(f"Data task '{name}' completed for {ticker} in {task_elapsed:.1f}s")
                    completed_tasks.append(name)

                    # Record per-data-task timing
                    if step_timings is not None:
                        step_timings[name] = round(task_elapsed, 3)

                    if progress_callback:
                        remaining = [task_labels[n] for n in futures if n not in completed_tasks]
                        if remaining:
                            progress_callback(f"Received {task_labels[name]} ({task_elapsed:.0f}s). Waiting: {remaining[0]}...")
                        else:
                            progress_callback(f"All data received for {ticker}. Processing...")
                except Exception as e:
                    logger.error(f"Failed to get {name} for {ticker}: {e}")
                    completed_tasks.append(name)
                    if step_timings is not None:
                        step_timings[name] = round(task_elapsed, 3)
                    if name == 'fundamentals':
                        data['fundamentals'] = {}
                    elif name == 'price_history':
                        data['_price_history_raw'] = None
                    elif name == 'benchmark':
                        data['benchmark_history'] = pd.DataFrame()

        # Process price history (may depend on fundamentals result)
        if data.get('fundamentals', {}).get('source') == 'comprehensive_enhanced':
            data['price_history'] = self._extract_price_history_from_fundamentals(data['fundamentals'])
        elif data.get('_price_history_raw') is not None:
            data['price_history'] = data['_price_history_raw']
        elif data.get('fundamentals'):
            data['price_history'] = self._extract_price_history_from_fundamentals(data['fundamentals'])
        else:
            data['price_history'] = pd.DataFrame()
        data.pop('_price_history_raw', None)

        if 'benchmark_history' not in data:
            data['benchmark_history'] = pd.DataFrame()

        # Add existing portfolio for risk analysis
        data['existing_portfolio'] = existing_portfolio or []

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
        
        logger.info(f"Starting Portfolio Recommendation - {num_positions} positions")
        
        # If no challenge context provided, use default
        if challenge_context is None:
            challenge_context = """
            Generate an optimal diversified portfolio that maximizes risk-adjusted returns 
            while adhering to the Investment Policy Statement constraints.
            Focus on high-quality companies with strong fundamentals and growth potential.
            """
        
        # Build client profile for AI selection
        client_profile = {
            'name': 'Portfolio User',
            'ips_data': self.ips_config,
            'profile_text': challenge_context
        }
        
        # Stage 1: Get tickers (either AI-selected or manual)
        if tickers is None:
            if self.ai_selector is None:
                logger.error("AI Portfolio Selector not available")
                raise ValueError("AI Portfolio Selector not initialized. Check OpenAI and Gemini API keys.")
            
            logger.info(f"Running AI-powered ticker selection (universe: {universe_size})...")
            selection_result = self.ai_selector.select_portfolio_tickers(
                challenge_context=challenge_context,
                client_profile=client_profile,
                universe_size=universe_size
            )
            
            selected_tickers = selection_result['final_tickers']
            ticker_rationales = selection_result['ticker_rationales']
            all_candidates = selection_result['all_candidates']
            selection_log = selection_result['session_log']
            
            logger.info(f"AI selected {len(selected_tickers)} tickers: {', '.join(selected_tickers)}")
        else:
            # Manual ticker list provided
            selected_tickers = tickers[:num_positions]
            ticker_rationales = {t: "Manually selected ticker" for t in selected_tickers}
            all_candidates = selected_tickers
            selection_log = {'manual_selection': True, 'tickers': selected_tickers}
            logger.info(f"Using manual ticker list: {', '.join(selected_tickers)}")
        
        # Stage 2: Run full analysis on each ticker
        logger.info(f"Running comprehensive analysis on {len(selected_tickers)} tickers...")
        
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
                logger.info(f"   {ticker}: Score {analysis['final_score']:.1f}")
                
            except Exception as e:
                logger.error(f"   Analysis failed for {ticker}: {e}")
                continue
        
        # Stage 3: Filter and construct portfolio
        logger.info("Constructing portfolio from analyzed stocks...")
        
        # Sort by final score
        portfolio_analyses.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        # Take top N positions
        portfolio_stocks = portfolio_analyses[:num_positions]
        
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
                'eligible': True,
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
        
        logger.info(f"Portfolio constructed: {len(portfolio)} positions, Avg Score: {avg_score:.1f}")
        
        return {
            'portfolio': portfolio,
            'summary': summary,
            'analysis_date': analysis_date,
            'all_candidates': all_candidates if tickers is None else selected_tickers,
            'selection_log': selection_log,
            'eligible_count': len(portfolio_analyses),
            'total_analyzed': len(portfolio_analyses),
            'all_analyses': portfolio_analyses  # Include ALL analyzed stocks for QA archive
        }