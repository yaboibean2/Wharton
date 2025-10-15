"""
Learning/QA Agent
Tracks performance of past recommendations and updates model parameters.
Implements bounded learning to improve model over time.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LearningAgent(BaseAgent):
    """
    Quality assurance and learning agent.
    Evaluates past recommendations and makes bounded parameter adjustments.
    """
    
    def __init__(self, config: Dict[str, Any], openai_client=None):
        super().__init__("LearningAgent", config, openai_client)
        self.learning_config = config.get('learning_agent', {})
        self.enabled = self.learning_config.get('enabled', True)
        self.evaluation_windows = self.learning_config.get('evaluation_windows', [7, 28, 84])
        self.min_observations = self.learning_config.get('min_observations', 10)
        self.max_adjustment = self.learning_config.get('max_weight_adjustment', 0.1)
        
        # Storage for tracking
        self.history_dir = Path("data/history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "decision_history.jsonl"
    
    def log_recommendation(
        self,
        ticker: str,
        recommendation_date: str,
        agent_scores: Dict[str, float],
        final_score: float,
        target_weight: float,
        fundamentals: Dict[str, Any]
    ):
        """Log a recommendation for future evaluation."""
        if not self.enabled:
            return
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'recommendation_date': recommendation_date,
            'agent_scores': agent_scores,
            'final_score': final_score,
            'target_weight': target_weight,
            'price_at_rec': fundamentals.get('current_price'),
            'sector': fundamentals.get('sector')
        }
        
        # Append to history
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Logged recommendation: {ticker} @ {final_score:.1f}")
    
    def evaluate_performance(
        self,
        data_provider,
        evaluation_date: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate performance of past recommendations.
        
        Returns:
            - Performance metrics (Sharpe, hit rate, IR)
            - Calibration report (agent accuracy)
            - Recommended parameter updates
        """
        if not self.enabled:
            return {'enabled': False}
        
        if not self.history_file.exists():
            logger.info("No recommendation history to evaluate")
            return {'observations': 0}
        
        # Load history
        history = self._load_history()
        
        if len(history) < self.min_observations:
            logger.info(f"Insufficient history: {len(history)} < {self.min_observations}")
            return {'observations': len(history), 'status': 'insufficient_data'}
        
        # Calculate realized returns for each window
        performance = {}
        for window_days in self.evaluation_windows:
            window_perf = self._evaluate_window(history, window_days, data_provider)
            performance[f'{window_days}d'] = window_perf
        
        # Analyze agent calibration
        calibration = self._analyze_agent_calibration(history, data_provider)
        
        # Generate parameter update recommendations
        updates = self._recommend_updates(calibration, performance)
        
        # Generate AI summary
        summary = self._generate_performance_summary(performance, calibration, updates)
        
        report = {
            'evaluation_date': evaluation_date or datetime.now().strftime('%Y-%m-%d'),
            'num_observations': len(history),
            'performance': performance,
            'calibration': calibration,
            'recommended_updates': updates,
            'summary': summary
        }
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _load_history(self) -> List[Dict]:
        """Load recommendation history from file."""
        history = []
        with open(self.history_file, 'r') as f:
            for line in f:
                history.append(json.loads(line))
        return history
    
    def _evaluate_window(
        self,
        history: List[Dict],
        window_days: int,
        data_provider
    ) -> Dict[str, float]:
        """Evaluate performance over a specific time window."""
        returns = []
        scores = []
        
        cutoff_date = datetime.now() - timedelta(days=window_days * 2)  # Look back far enough
        
        for entry in history:
            rec_date = datetime.fromisoformat(entry['timestamp'])
            
            # Skip if too recent (haven't had time to realize returns)
            if rec_date > cutoff_date:
                continue
            
            ticker = entry['ticker']
            price_at_rec = entry.get('price_at_rec')
            final_score = entry['final_score']
            
            if not price_at_rec:
                continue
            
            # Get price N days later
            try:
                end_date = (rec_date + timedelta(days=window_days)).strftime('%Y-%m-%d')
                start_date = rec_date.strftime('%Y-%m-%d')
                
                price_history = data_provider.get_price_history(
                    ticker, start_date, end_date, cache_hours=24
                )
                
                if not price_history.empty and len(price_history) > 0:
                    final_price = price_history['Close'].iloc[-1]
                    realized_return = (final_price / price_at_rec - 1) * 100
                    
                    returns.append(realized_return)
                    scores.append(final_score)
                    
            except Exception as e:
                logger.warning(f"Failed to get return for {ticker}: {e}")
                continue
        
        if not returns:
            return {}
        
        # Calculate metrics
        returns_arr = np.array(returns)
        scores_arr = np.array(scores)
        
        metrics = {
            'avg_return': float(np.mean(returns_arr)),
            'sharpe_ratio': float(np.mean(returns_arr) / np.std(returns_arr)) if np.std(returns_arr) > 0 else 0,
            'hit_rate': float(np.sum(returns_arr > 0) / len(returns_arr)),
            'max_drawdown': float(np.min(returns_arr)),
            'num_trades': len(returns),
            'score_return_correlation': float(np.corrcoef(scores_arr, returns_arr)[0, 1]) if len(returns) > 2 else 0
        }
        
        return metrics
    
    def _analyze_agent_calibration(
        self,
        history: List[Dict],
        data_provider
    ) -> Dict[str, Dict]:
        """Analyze which agents are best calibrated (scores correlate with outcomes)."""
        agent_names = ['value', 'growth_momentum', 'macro_regime', 'risk', 'sentiment']
        
        calibration = {}
        
        for agent in agent_names:
            agent_scores = []
            returns = []
            
            for entry in history[-50:]:  # Use recent 50 observations
                if agent not in entry['agent_scores']:
                    continue
                
                score = entry['agent_scores'][agent]
                ticker = entry['ticker']
                price_at_rec = entry.get('price_at_rec')
                rec_date = datetime.fromisoformat(entry['timestamp'])
                
                if not price_at_rec:
                    continue
                
                # Get 28-day forward return
                try:
                    end_date = (rec_date + timedelta(days=28)).strftime('%Y-%m-%d')
                    start_date = rec_date.strftime('%Y-%m-%d')
                    
                    price_history = data_provider.get_price_history(
                        ticker, start_date, end_date, cache_hours=24
                    )
                    
                    if not price_history.empty:
                        final_price = price_history['Close'].iloc[-1]
                        ret = (final_price / price_at_rec - 1) * 100
                        
                        agent_scores.append(score)
                        returns.append(ret)
                except:
                    continue
            
            if len(returns) > 5:
                correlation = np.corrcoef(agent_scores, returns)[0, 1]
                calibration[agent] = {
                    'correlation': float(correlation),
                    'observations': len(returns),
                    'status': 'well_calibrated' if abs(correlation) > 0.2 else 'needs_tuning'
                }
            else:
                calibration[agent] = {
                    'correlation': 0,
                    'observations': len(returns),
                    'status': 'insufficient_data'
                }
        
        return calibration
    
    def _recommend_updates(
        self,
        calibration: Dict[str, Dict],
        performance: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Recommend bounded parameter updates based on performance."""
        updates = {
            'agent_weights': {},
            'rationale': []
        }
        
        # Analyze agent performance
        for agent, cal_data in calibration.items():
            if cal_data['status'] == 'insufficient_data':
                continue
            
            correlation = cal_data['correlation']
            current_weight = 1.0  # Would load from config
            
            # Bounded adjustment based on correlation
            if correlation > 0.3:
                # Agent is predictive - increase weight slightly
                adjustment = min(self.max_adjustment, correlation * 0.2)
                updates['agent_weights'][agent] = adjustment
                updates['rationale'].append(
                    f"{agent}: +{adjustment:.2f} (strong {correlation:.2f} correlation)"
                )
            elif correlation < -0.1:
                # Agent is counter-predictive - decrease weight
                adjustment = -min(self.max_adjustment, abs(correlation) * 0.2)
                updates['agent_weights'][agent] = adjustment
                updates['rationale'].append(
                    f"{agent}: {adjustment:.2f} (negative {correlation:.2f} correlation)"
                )
        
        return updates
    
    def _generate_performance_summary(
        self,
        performance: Dict,
        calibration: Dict,
        updates: Dict
    ) -> str:
        """Generate one-line summary of QA findings."""
        if not performance:
            return "Insufficient data for performance evaluation"
        
        # Get best performing window
        best_window = max(
            performance.items(),
            key=lambda x: x[1].get('sharpe_ratio', 0) if x[1] else 0
        )
        
        window_name, metrics = best_window
        sharpe = metrics.get('sharpe_ratio', 0)
        hit_rate = metrics.get('hit_rate', 0) * 100
        
        # Identify best/worst agents
        agent_corrs = {k: v['correlation'] for k, v in calibration.items() if v.get('correlation')}
        if agent_corrs:
            best_agent = max(agent_corrs.items(), key=lambda x: x[1])
            
            summary = f"{window_name} Sharpe: {sharpe:.2f}, Hit Rate: {hit_rate:.0f}%. "
            summary += f"Best agent: {best_agent[0]} (corr: {best_agent[1]:.2f})"
            
            if updates['agent_weights']:
                summary += f". Recommended {len(updates['agent_weights'])} weight updates"
            
            return summary
        
        return f"{window_name} Sharpe: {sharpe:.2f}, Hit Rate: {hit_rate:.0f}%"
    
    def _save_report(self, report: Dict):
        """Save evaluation report to file."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.history_dir / f"qa_report_{date_str}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"QA report saved to {report_file}")
    
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        The Learning Agent doesn't analyze individual stocks - it analyzes system performance.
        This method is implemented to satisfy the abstract base class requirement.
        """
        return {
            'score': 50.0,  # Neutral score - this agent doesn't score individual stocks
            'rationale': 'Learning Agent focuses on system performance evaluation, not individual stock analysis',
            'details': {
                'agent_type': 'learning',
                'function': 'quality_assurance',
                'note': 'Use evaluate_system_performance() for actual learning functionality'
            }
        }
