from typing import Dict
import logging
import sys
from threading import Lock

# Configure logger for real-time output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add stream handler for real-time console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class TokenCounter:
    """
    Thread-safe token counter for tracking OpenAI API usage
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenCounter, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the counter"""
        self.counts = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # Log initialization outside lock since logging is thread-safe
        logger.info("Initialized token counter")
        
    def update(self, usage: Dict) -> None:
        """
        Update token counts with usage from an API response
        
        Args:
            usage: Usage dictionary from OpenAI API response
        """
        # First update counts under lock
        with self._lock:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            self.counts["prompt_tokens"] += prompt_tokens
            self.counts["completion_tokens"] += completion_tokens
            self.counts["total_tokens"] += total_tokens
            
            # Get current totals while still under lock
            current_totals = self.counts.copy()
        
        # Log outside the lock since logging is already thread-safe
        logger.info(
            f"Token usage - Total: {current_totals['total_tokens']:,} "
            f"(Prompt: {current_totals['prompt_tokens']:,}, "
            f"Completion: {current_totals['completion_tokens']:,})"
        )
        # Force flush the output
        sys.stdout.flush()
            
    def get_counts(self) -> Dict:
        """
        Get current token counts
        
        Returns:
            Dictionary with current token counts
        """
        with self._lock:
            return self.counts.copy() 