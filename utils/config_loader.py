"""
Configuration Loader
Loads and validates configuration files (IPS, model, universe).
Provides safe defaults if configs are empty or missing.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Centralized configuration management with safe defaults."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._ips = None
        self._model = None
        self._universe = None
    
    def load_ips(self) -> Dict[str, Any]:
        """Load Investment Policy Statement with safe defaults."""
        if self._ips is not None:
            return self._ips
        
        ips_path = self.config_dir / "ips.yaml"
        try:
            with open(ips_path, 'r') as f:
                ips = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"IPS file not found at {ips_path}, using defaults")
            ips = {}
        
        # Apply safe defaults
        self._ips = self._apply_ips_defaults(ips)
        return self._ips
    
    def _apply_ips_defaults(self, ips: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safe default values for missing IPS fields."""
        defaults = {
            "client": {
                "name": "Default Client",
                "risk_tolerance": "moderate",
                "time_horizon_years": 5,
                "cash_buffer_pct": 5
            },
            "universe": {
                "geography": ["US"],
                "currency": "USD",
                "benchmark": "^GSPC",
                "min_price": 3,
                "min_avg_daily_volume": 2000000
            },
            "exclusions": {
                "sectors": [],
                "tickers": [],
                "esg_screens": []
            },
            "position_limits": {
                "max_position_pct": 8,
                "max_sector_pct": 30,
                "max_industry_pct": 20
            },
            "portfolio_constraints": {
                "target_num_holdings": 15,
                "min_holdings": 10,
                "max_holdings": 25,
                "beta_min": 0.7,
                "beta_max": 1.1
            },
            "rebalancing": {
                "frequency": "biweekly",
                "min_trade_size_pct": 1
            },
            "costs": {
                "commission_bps": 10,
                "slippage_bps": 5
            }
        }
        
        # Deep merge defaults with loaded config
        return self._deep_merge(defaults, ips)
    
    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration (agent weights, parameters)."""
        if self._model is not None:
            return self._model
        
        model_path = self.config_dir / "model.yaml"
        try:
            with open(model_path, 'r') as f:
                self._model = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error(f"Model config not found at {model_path}")
            raise
        
        return self._model
    
    def load_universe_config(self) -> Dict[str, Any]:
        """Load universe configuration."""
        if self._universe is not None:
            return self._universe
        
        universe_path = self.config_dir / "universe.yaml"
        try:
            with open(universe_path, 'r') as f:
                self._universe = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Universe config not found at {universe_path}, using SP100 default")
            self._universe = {"universe_type": "SP100", "custom_tickers": []}
        
        return self._universe
    
    def save_ips(self, ips: Dict[str, Any]) -> None:
        """Save updated IPS configuration."""
        ips_path = self.config_dir / "ips.yaml"
        with open(ips_path, 'w') as f:
            yaml.dump(ips, f, default_flow_style=False, sort_keys=False)
        self._ips = ips
        logger.info(f"IPS saved to {ips_path}")
    
    def update_model_weights(self, new_weights: Dict[str, float]) -> None:
        """Update agent weights in model config."""
        model = self.load_model_config()
        model['agent_weights'].update(new_weights)
        
        model_path = self.config_dir / "model.yaml"
        with open(model_path, 'w') as f:
            yaml.dump(model, f, default_flow_style=False, sort_keys=False)
        self._model = model
        logger.info(f"Model weights updated: {new_weights}")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif value is not None and value != "" and value != []:
                result[key] = value
        return result


# Global singleton instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get or create global ConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader
