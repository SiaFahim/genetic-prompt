"""
LLM interface for evaluating prompts on GSM8K problems.
Supports OpenAI GPT models with caching and error handling.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

import openai
from openai import AsyncOpenAI

from src.utils.config import get_config
from src.utils.answer_extraction import extract_answer_from_response

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for interacting with language models."""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """
        Initialize LLM interface.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        
        # Model configuration
        self.model_config = self.config.get_model_config()
        self.model_name = self.model_config.get('name', 'gpt-4o')
        self.temperature = self.model_config.get('temperature', 0)
        self.max_tokens = self.model_config.get('max_tokens', 200)
        self.timeout = self.model_config.get('timeout', 30)
        
        # API configuration
        api_key = self.config.get_api_key()
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Caching
        self.cache_enabled = self.config.get('evaluation.cache_enabled', True)
        self.cache_dir = Path(self.config.get('paths.cache_dir', './data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'llm_responses.json'
        
        # Load existing cache
        self._cache = self._load_cache()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_tokens_used': 0,
            'total_cost_usd': 0.0
        }
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load response cache from disk."""
        if not self.cache_enabled or not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached responses")
            return cache
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self) -> None:
        """Save response cache to disk."""
        if not self.cache_enabled:
            return
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved {len(self._cache)} cached responses")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, prompt: str, problem: str) -> str:
        """
        Generate cache key for a prompt-problem pair.
        
        Args:
            prompt: The prompt text
            problem: The problem text
            
        Returns:
            Cache key string
        """
        # Include model configuration in cache key
        key_data = {
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'prompt': prompt,
            'problem': problem
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    async def evaluate_prompt(self, prompt: str, problem: str, 
                            ground_truth: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate a prompt on a single problem.
        
        Args:
            prompt: The prompt to evaluate
            problem: The problem text
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary with evaluation results
        """
        self.stats['total_requests'] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, problem)
        
        if self.cache_enabled and cache_key in self._cache:
            self.stats['cache_hits'] += 1
            cached_result = self._cache[cache_key].copy()
            
            # Add ground truth comparison if provided
            if ground_truth is not None:
                extracted_answer = cached_result.get('extracted_answer')
                if extracted_answer is not None:
                    cached_result['correct'] = abs(extracted_answer - ground_truth) < 0.001
                    cached_result['ground_truth'] = ground_truth
            
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        # Construct full prompt
        full_prompt = f"{prompt}\n\nProblem: {problem}\n\nSolution:"
        
        try:
            # Make API call
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            end_time = time.time()
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Extract answer
            extracted_answer = extract_answer_from_response(response_text)
            
            # Calculate cost (approximate)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # GPT-4o pricing (as of 2024)
            cost_per_input_token = 0.005 / 1000  # $0.005 per 1K input tokens
            cost_per_output_token = 0.015 / 1000  # $0.015 per 1K output tokens
            
            estimated_cost = (prompt_tokens * cost_per_input_token + 
                            completion_tokens * cost_per_output_token)
            
            # Update statistics
            self.stats['total_tokens_used'] += total_tokens
            self.stats['total_cost_usd'] += estimated_cost
            
            # Prepare result
            result = {
                'response_text': response_text,
                'extracted_answer': extracted_answer,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'estimated_cost_usd': estimated_cost,
                'response_time_seconds': end_time - start_time,
                'model_used': self.model_name,
                'timestamp': time.time()
            }
            
            # Add ground truth comparison if provided
            if ground_truth is not None:
                if extracted_answer is not None:
                    result['correct'] = abs(extracted_answer - ground_truth) < 0.001
                    result['ground_truth'] = ground_truth
                else:
                    result['correct'] = False
                    result['ground_truth'] = ground_truth
            
            # Cache the result (without ground truth to make it reusable)
            if self.cache_enabled:
                cache_result = result.copy()
                cache_result.pop('ground_truth', None)
                cache_result.pop('correct', None)
                self._cache[cache_key] = cache_result
                
                # Save cache periodically
                if len(self._cache) % 100 == 0:
                    self._save_cache()
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error evaluating prompt: {e}")
            
            # Return error result
            return {
                'error': str(e),
                'response_text': None,
                'extracted_answer': None,
                'correct': False if ground_truth is not None else None,
                'ground_truth': ground_truth,
                'timestamp': time.time()
            }
    
    async def evaluate_batch(self, prompt_problem_pairs: List[Tuple[str, str, Optional[float]]],
                           max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompt-problem pairs concurrently.
        
        Args:
            prompt_problem_pairs: List of (prompt, problem, ground_truth) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(prompt: str, problem: str, 
                                        ground_truth: Optional[float]) -> Dict[str, Any]:
            async with semaphore:
                return await self.evaluate_prompt(prompt, problem, ground_truth)
        
        # Create tasks
        tasks = [
            evaluate_with_semaphore(prompt, problem, ground_truth)
            for prompt, problem, ground_truth in prompt_problem_pairs
        ]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch evaluation {i}: {result}")
                prompt, problem, ground_truth = prompt_problem_pairs[i]
                processed_results.append({
                    'error': str(result),
                    'response_text': None,
                    'extracted_answer': None,
                    'correct': False if ground_truth is not None else None,
                    'ground_truth': ground_truth,
                    'timestamp': time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
            stats['error_rate'] = stats['errors'] / stats['total_requests']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        stats['cache_size'] = len(self._cache)
        stats['model_name'] = self.model_name
        
        return stats
    
    def save_cache(self) -> None:
        """Manually save cache to disk."""
        self._save_cache()
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")
    
    def __del__(self):
        """Destructor to save cache on exit."""
        if hasattr(self, '_cache') and self.cache_enabled:
            self._save_cache()
