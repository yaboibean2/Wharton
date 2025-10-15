"""
Base Agent Class
Abstract base class for all investment analysis agents.
Defines common interface and OpenAI integration.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from openai import OpenAI
import logging
from utils.logger import get_disclosure_logger

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Each agent analyzes stocks from a specific perspective.
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        openai_client: Optional[OpenAI] = None
    ):
        self.name = name
        self.config = config
        self.disclosure_logger = get_disclosure_logger()
        
        # Initialize OpenAI client
        if openai_client:
            self.openai = openai_client
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            self.openai = OpenAI(api_key=api_key)
        
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a stock and return a score + rationale.
        
        Args:
            ticker: Stock ticker symbol
            data: Dictionary containing all relevant data for analysis
        
        Returns:
            {
                'score': float (0-100),
                'rationale': str (one-line explanation),
                'details': dict (supporting metrics)
            }
        """
        pass
    
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        """
        Call OpenAI API with disclosure logging.
        
        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        
        Returns:
            Response text
        """
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            # Estimate cost (gpt-4o-mini: $0.15/$0.60 per 1M tokens)
            cost = (response.usage.prompt_tokens * 0.15 + response.usage.completion_tokens * 0.60) / 1_000_000
            
            # Log for disclosure
            self.disclosure_logger.log_ai_usage(
                tool=f"OpenAI-{self.model}",
                purpose=f"{self.name} analysis",
                prompt_summary=user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt,
                output_summary=result[:100] + "..." if len(result) > 100 else result,
                tokens_used=tokens,
                cost_usd=cost
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error in {self.name}: {e}")
            raise
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-100 scale."""
        if max_val == min_val:
            return 50.0
        normalized = ((value - min_val) / (max_val - min_val)) * 100
        return max(0.0, min(100.0, normalized))
    
    def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:
        """Safely get value from dict with default."""
        return data.get(key, default)
