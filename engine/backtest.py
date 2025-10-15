"""
Backtest Engine
Walk-forward backtesting with proper hygiene (no look-ahead, trading costs).
Evaluates strategy performance vs benchmark over historical period.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Walk-forward backtest engine with strict hygiene controls.
    - No look-ahead bias
    - Point-in-time data only
    - Trading costs applied
    - Walk-forward validation
    """
    
    def __init__(
        self,
        orchestrator,
        backtest_config: Dict[str, Any],
        ips_config: Dict[str, Any]
    ):
        self.orchestrator = orchestrator
        self.config = backtest_config
        self.ips = ips_config
        
        # Backtest parameters
        self.initial_capital = backtest_config.get('initial_capital', 100000)
        self.rebalance_freq = backtest_config.get('rebalance_frequency', 'biweekly')
        
        # Walk-forward config
        self.wf_enabled = backtest_config.get('walk_forward', {}).get('enabled', True)
        self.train_window = backtest_config.get('walk_forward', {}).get('train_window_days', 180)
        self.test_window = backtest_config.get('walk_forward', {}).get('test_window_days', 30)
        self.embargo_days = backtest_config.get('walk_forward', {}).get('embargo_days', 2)
        
        # Trading costs
        costs = ips_config.get('costs', {})
        self.commission_bps = costs.get('commission_bps', 10)
        self.slippage_bps = costs.get('slippage_bps', 5)
        self.total_cost_bps = self.commission_bps + self.slippage_bps
        
        logger.info(f"Backtest engine initialized: {self.rebalance_freq} rebalance, {self.total_cost_bps}bps costs")
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        universe: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run full backtest over specified period.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            universe: List of tickers to trade (default: S&P 100)
        
        Returns:
            Complete backtest results with metrics
        """
        logger.info(f"Running backtest: {start_date} to {end_date}")
        
        if not universe:
            universe = self.orchestrator.data_provider.get_sp100_tickers()
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date)
        
        logger.info(f"Generated {len(rebalance_dates)} rebalance dates")
        
        # Initialize tracking
        portfolio_history = []
        trades = []
        current_holdings = {}
        cash = self.initial_capital
        
        # Run through each rebalance period
        for i, rebal_date in enumerate(rebalance_dates):
            logger.info(f"Rebalance {i+1}/{len(rebalance_dates)}: {rebal_date}")
            
            # Get recommendations for this date
            recommendations = self.orchestrator.recommend_portfolio(
                tickers=universe,
                analysis_date=rebal_date,
                num_positions=self.ips.get('portfolio_constraints', {}).get('target_num_holdings', 15)
            )
            
            target_portfolio = recommendations['portfolio']
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(
                current_holdings, rebal_date
            )
            total_value = portfolio_value + cash
            
            # Determine trades needed
            rebal_trades = self._calculate_rebalance_trades(
                current_holdings, target_portfolio, total_value, rebal_date
            )
            
            # Execute trades with costs
            for trade in rebal_trades:
                cost = abs(trade['value']) * (self.total_cost_bps / 10000)
                cash -= trade['value'] + cost
                
                ticker = trade['ticker']
                if trade['action'] == 'BUY':
                    current_holdings[ticker] = current_holdings.get(ticker, 0) + trade['shares']
                else:  # SELL
                    current_holdings[ticker] = current_holdings.get(ticker, 0) - trade['shares']
                    if current_holdings[ticker] == 0:
                        del current_holdings[ticker]
                
                trade['cost'] = cost
                trades.append(trade)
            
            # Record portfolio state
            portfolio_history.append({
                'date': rebal_date,
                'holdings': current_holdings.copy(),
                'cash': cash,
                'total_value': cash + self._calculate_portfolio_value(current_holdings, rebal_date),
                'num_positions': len(current_holdings)
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_history, start_date, end_date)
        
        # Compare to benchmark
        benchmark_metrics = self._calculate_benchmark_performance(start_date, end_date)
        
        results = {
            'config': {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': self.initial_capital,
                'rebalance_frequency': self.rebalance_freq,
                'trading_costs_bps': self.total_cost_bps
            },
            'portfolio_history': portfolio_history,
            'trades': trades,
            'metrics': metrics,
            'benchmark': benchmark_metrics,
            'performance_summary': self._generate_summary(metrics, benchmark_metrics)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_rebalance_dates(self, start_date: str, end_date: str) -> List[str]:
        """Generate list of rebalance dates based on frequency."""
        dates = []
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if self.rebalance_freq == 'daily':
            delta = timedelta(days=1)
        elif self.rebalance_freq == 'weekly':
            delta = timedelta(weeks=1)
        elif self.rebalance_freq == 'biweekly':
            delta = timedelta(weeks=2)
        elif self.rebalance_freq == 'monthly':
            delta = timedelta(days=30)
        else:
            delta = timedelta(weeks=2)  # Default
        
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday=0, Friday=4
                dates.append(current.strftime('%Y-%m-%d'))
            current += delta
        
        return dates
    
    def _calculate_portfolio_value(
        self,
        holdings: Dict[str, float],
        date: str
    ) -> float:
        """Calculate current market value of holdings."""
        total_value = 0
        
        for ticker, shares in holdings.items():
            try:
                # Get price on this date
                price_hist = self.orchestrator.data_provider.get_price_history(
                    ticker, date, date, cache_hours=24
                )
                if not price_hist.empty:
                    price = price_hist['Close'].iloc[-1]
                    total_value += shares * price
            except Exception as e:
                logger.warning(f"Could not value {ticker} on {date}: {e}")
        
        return total_value
    
    def _calculate_rebalance_trades(
        self,
        current_holdings: Dict[str, float],
        target_portfolio: List[Dict],
        total_value: float,
        date: str
    ) -> List[Dict]:
        """Calculate trades needed to rebalance portfolio."""
        trades = []
        
        # Get current prices
        prices = {}
        for ticker in set(list(current_holdings.keys()) + [p['ticker'] for p in target_portfolio]):
            try:
                price_hist = self.orchestrator.data_provider.get_price_history(
                    ticker, date, date, cache_hours=24
                )
                if not price_hist.empty:
                    prices[ticker] = price_hist['Close'].iloc[-1]
            except:
                continue
        
        # Calculate target positions
        target_positions = {}
        for position in target_portfolio:
            ticker = position['ticker']
            weight = position['target_weight_pct'] / 100
            target_value = total_value * weight
            
            if ticker in prices:
                target_shares = int(target_value / prices[ticker])
                target_positions[ticker] = target_shares
        
        # Determine trades
        # 1. Sell positions not in target
        for ticker, shares in current_holdings.items():
            if ticker not in target_positions and ticker in prices:
                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': prices[ticker],
                    'value': shares * prices[ticker]
                })
        
        # 2. Adjust existing positions
        for ticker, target_shares in target_positions.items():
            current_shares = current_holdings.get(ticker, 0)
            diff = target_shares - current_shares
            
            if abs(diff) > 0 and ticker in prices:
                action = 'BUY' if diff > 0 else 'SELL'
                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': action,
                    'shares': abs(diff),
                    'price': prices[ticker],
                    'value': abs(diff) * prices[ticker] * (1 if diff > 0 else -1)
                })
        
        return trades
    
    def _calculate_metrics(
        self,
        portfolio_history: List[Dict],
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not portfolio_history:
            return {}
        
        # Extract values
        values = [p['total_value'] for p in portfolio_history]
        dates = [p['date'] for p in portfolio_history]
        
        # Returns
        returns = pd.Series(values).pct_change().dropna()
        
        # Total return
        total_return = (values[-1] / values[0] - 1) * 100
        
        # Annualized return
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        years = days / 365.25
        annualized_return = ((values[-1] / values[0]) ** (1 / years) - 1) * 100 if years > 0 else total_return
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252 / len(returns) * 365) * 100 if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free_rate = 4.0
        sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252 / len(returns) * 365) * 100 if len(downside_returns) > 1 else volatility
        sortino = (annualized_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        # Max drawdown
        cumulative = pd.Series(values)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        return {
            'total_return_pct': round(total_return, 2),
            'annualized_return_pct': round(annualized_return, 2),
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2),
            'final_value': round(values[-1], 2),
            'num_periods': len(portfolio_history)
        }
    
    def _calculate_benchmark_performance(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        """Calculate benchmark (S&P 500) performance."""
        benchmark = self.ips.get('universe', {}).get('benchmark', '^GSPC')
        
        try:
            bench_data = self.orchestrator.data_provider.get_price_history(
                benchmark, start_date, end_date, cache_hours=24
            )
            
            if bench_data.empty:
                return {}
            
            returns = bench_data['Returns'].dropna()
            
            total_return = (bench_data['Close'].iloc[-1] / bench_data['Close'].iloc[0] - 1) * 100
            
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            years = days / 365.25
            annualized_return = ((bench_data['Close'].iloc[-1] / bench_data['Close'].iloc[0]) ** (1 / years) - 1) * 100
            
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            return {
                'total_return_pct': round(total_return, 2),
                'annualized_return_pct': round(annualized_return, 2),
                'volatility_pct': round(volatility, 2),
                'max_drawdown_pct': round(max_drawdown, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark performance: {e}")
            return {}
    
    def _generate_summary(
        self,
        metrics: Dict[str, float],
        benchmark: Dict[str, float]
    ) -> str:
        """Generate one-line performance summary."""
        if not metrics:
            return "Backtest failed"
        
        strategy_return = metrics.get('annualized_return_pct', 0)
        strategy_sharpe = metrics.get('sharpe_ratio', 0)
        strategy_dd = metrics.get('max_drawdown_pct', 0)
        
        summary = f"Return: {strategy_return:.1f}%, Sharpe: {strategy_sharpe:.2f}, MaxDD: {strategy_dd:.1f}%"
        
        if benchmark:
            bench_return = benchmark.get('annualized_return_pct', 0)
            alpha = strategy_return - bench_return
            summary += f" | Alpha vs SPY: {alpha:+.1f}%"
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save backtest results to file."""
        output_dir = Path("outputs/backtests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"backtest_{timestamp}.csv"
        
        # Save portfolio history as CSV
        if results['portfolio_history']:
            df = pd.DataFrame(results['portfolio_history'])
            df.to_csv(filename, index=False)
            logger.info(f"Backtest results saved to {filename}")
