from typing import Dict, List, Any, Optional
import time
import logging
import sys
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.token_counter import TokenCounter

# Configure logger for real-time output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add stream handler for real-time console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class OpenAIClient:
    def __init__(self, model: str, max_threads: int = 8, rate_limit_delay: float = 0.001):
        """
        Initialize the OpenAI client
        
        Args:
            model: The model to use for completions
            max_threads: Maximum number of parallel threads
            rate_limit_delay: Delay between API calls to avoid rate limits
        """
        self.model = model
        self.max_threads = max_threads
        self.rate_limit_delay = rate_limit_delay
        self.token_counter = TokenCounter()
        logger.info(f"Initialized OpenAI client with model={model}, max_threads={max_threads}")
        sys.stdout.flush()
        
    def complete(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7,response_format=None) -> str:
        """
        Get a completion from OpenAI API
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            
        Returns:
            The model's response text
        """
        logger.debug(f"Sending prompt to {self.model}: {prompt[:50]}...")
        sys.stdout.flush()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            if response_format:
                logger.info(f"Using response format: {response_format}")
                response = openai.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format
                )
            else:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
            
            # Update token counter with usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            self.token_counter.update(usage)
            
            if response_format:
                result = response.choices[0].message.parsed
            else:
                result = response.choices[0].message.content
            # logger.debug(f"Received response: {result[:50]}...")
            sys.stdout.flush()
            return result
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            sys.stdout.flush()
            raise
        
    def process_single_prompt(self, idx: int, prompt: str, system_prompt: Optional[str], temperature: float,response_format=None) -> tuple:
        """
        Process a single prompt with rate limiting
        
        Args:
            idx: Index of the prompt in the batch
            prompt: The prompt to process
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            
        Returns:
            Tuple of (idx, result)
        """
        if idx > 0:  # Rate limiting
            time.sleep(self.rate_limit_delay)
            
        try:
            result = self.complete(prompt, system_prompt, temperature,response_format)
            return idx, result
        except Exception as e:
            logger.error(f"Error processing prompt {idx}: {e}")
            raise
            
    def batch_complete(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        response_format=None
    ) -> Dict[int, str]:
        """
        Process multiple prompts in parallel with rate limiting
        
        Args:   
            prompts: List of prompts to process
            system_prompt: Optional system prompt for all prompts
            temperature: Temperature for generation
            
        Returns:
            Dictionary mapping indices to results
        """
        logger.info(f"Processing batch of {len(prompts)} prompts with model {self.model}")
        sys.stdout.flush()
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(
                    self.process_single_prompt, 
                    i, prompt, system_prompt, temperature,response_format
                )
                for i, prompt in enumerate(prompts)
            ]
            
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                    logger.debug(f"Completed prompt {idx}")
                    sys.stdout.flush()
                except Exception as e:
                    logger.error(f"Failed to complete prompt: {e}")
                    sys.stdout.flush()
                    raise
                
        logger.info(f"Completed batch processing {len(prompts)} prompts")
        sys.stdout.flush()
        return results
        
    def get_token_usage(self) -> Dict:
        """
        Get current token usage statistics
        
        Returns:
            Dictionary with token usage counts
        """
        return self.token_counter.get_counts() 