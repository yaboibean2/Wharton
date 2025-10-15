"""
Logging Configuration
Centralized logging for audit trail and disclosure requirements.
All AI/tool usage is logged with timestamps for Works Cited.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class DisclosureLogger:
    """
    Special logger for tracking AI/tool usage for competition disclosure.
    Logs: timestamp, tool used, purpose, input summary, output summary.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create disclosure log file with date
        date_str = datetime.now().strftime("%Y%m%d")
        self.disclosure_file = self.log_dir / f"ai_disclosure_{date_str}.jsonl"
    
    def log_ai_usage(
        self,
        tool: str,
        purpose: str,
        prompt_summary: str,
        output_summary: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None
    ):
        """Log AI/tool usage for disclosure."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool,
            "purpose": purpose,
            "prompt_summary": prompt_summary,
            "output_summary": output_summary,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd
        }
        
        # Append to JSONL file
        with open(self.disclosure_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_disclosure_summary(self) -> dict:
        """Generate summary for Works Cited section."""
        if not self.disclosure_file.exists():
            return {"total_calls": 0, "total_tokens": 0, "total_cost_usd": 0}
        
        total_calls = 0
        total_tokens = 0
        total_cost = 0.0
        tools_used = set()
        
        with open(self.disclosure_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                total_calls += 1
                tools_used.add(entry['tool'])
                if entry.get('tokens_used'):
                    total_tokens += entry['tokens_used']
                if entry.get('cost_usd'):
                    total_cost += entry['cost_usd']
        
        return {
            "total_calls": total_calls,
            "tools_used": list(tools_used),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 2),
            "log_file": str(self.disclosure_file)
        }


def setup_logging(log_level: str = "INFO") -> DisclosureLogger:
    """
    Setup application logging with both file and console handlers.
    Returns DisclosureLogger for AI usage tracking.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above)
    date_str = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(log_dir / f"app_{date_str}.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Create and return disclosure logger
    disclosure_logger = DisclosureLogger(log_dir=log_dir)
    
    return disclosure_logger


# Global disclosure logger instance
_disclosure_logger: Optional[DisclosureLogger] = None

def get_disclosure_logger() -> DisclosureLogger:
    """Get or create global DisclosureLogger instance."""
    global _disclosure_logger
    if _disclosure_logger is None:
        _disclosure_logger = DisclosureLogger()
    return _disclosure_logger
