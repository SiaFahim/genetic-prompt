"""
Asynchronous LLM Interface for batch processing and concurrent API calls.

This module provides an async version of the LLM interface that supports:
- Batch processing with configurable batch sizes
- Concurrent API calls within batches
- Advanced rate limiting and error handling
- Integration with existing caching system
"""

import asyncio
import aiohttp
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI

from ..utils.config import config
from ..utils.answer_extraction import extract_answer_from_response


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 20
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 3500  # OpenAI TPM limit
    retry_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    timeout: int = 30


@dataclass
class BatchResult:
    """Result of a batch evaluation."""
    results: List[Dict[str, Any]]
    successful_requests: int
    failed_requests: int
    total_time: float
    cache_hits: int
    api_calls_made: int


class AsyncLLMInterface:
    """Asynchronous LLM interface with batch processing capabilities."""
    
    def __init__(self, 
                 model: str = None, 
                 temperature: float = None,
                 max_tokens: int = None,
                 batch_config: Optional[BatchConfig] = None):
        """
        Initialize async LLM interface.
        
        Args:
            model: Model name (defaults to config)
            temperature: Temperature for generation (defaults to config)
            max_tokens: Max tokens for generation (defaults to config)
            batch_config: Batch processing configuration
        """
        self.model = model or config.default_model
        self.temperature = temperature if temperature is not None else config.temperature
        self.max_tokens = max_tokens or config.max_tokens
        self.batch_config = batch_config or BatchConfig()
        
        # Initialize async OpenAI client
        api_key = config.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0
        self.cache_hits = 0

        # Rate limiting performance tracking
        self.total_wait_time = 0.0
        self.rate_limit_hits = 0
        self.start_time = time.time()
        
        # Rate limiting
        self.request_times = []
        self.rate_limit_lock = asyncio.Lock()
        
        # Simple in-memory cache (thread-safe)
        self.response_cache = {}
        self.cache_enabled = config.cache_enabled
        self.cache_lock = asyncio.Lock()
    
    def _create_cache_key(self, prompt: str, question: str) -> str:
        """Create a cache key for the prompt-question pair."""
        combined = f"{prompt}|||{question}|||{self.model}|||{self.temperature}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        async with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times 
                                if current_time - t < 60]
            
            # Check if we're at the rate limit
            if len(self.request_times) >= self.batch_config.rate_limit_per_minute:
                # Calculate minimal wait time (not full minute)
                oldest_request = min(self.request_times)
                time_since_oldest = current_time - oldest_request
                wait_time = max(0, 60 - time_since_oldest + 1)  # Add 1 second buffer
                if wait_time > 0:
                    self.rate_limit_hits += 1
                    self.total_wait_time += wait_time
                    print(f"Rate limit reached, waiting {wait_time:.1f}s (optimized)")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(current_time)
    
    async def _make_api_request_async(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Make async API request with retry logic and rate limiting.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response text or None if failed
        """
        for attempt in range(self.batch_config.retry_attempts):
            try:
                # Rate limiting
                await self._check_rate_limit()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.batch_config.timeout
                )
                
                # Update statistics
                self.total_requests += 1
                self.successful_requests += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens_used += response.usage.total_tokens
                
                return response.choices[0].message.content
                
            except openai.RateLimitError:
                wait_time = min(
                    self.batch_config.base_delay * (2 ** attempt),
                    self.batch_config.max_delay
                )
                print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.batch_config.retry_attempts}")
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError:
                wait_time = min(
                    self.batch_config.base_delay * 2 * (2 ** attempt),
                    self.batch_config.max_delay
                )
                print(f"API timeout, waiting {wait_time}s before retry {attempt + 1}/{self.batch_config.retry_attempts}")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                print(f"API request failed (attempt {attempt + 1}/{self.batch_config.retry_attempts}): {e}")
                if attempt < self.batch_config.retry_attempts - 1:
                    await asyncio.sleep(self.batch_config.base_delay)
        
        self.total_requests += 1
        self.failed_requests += 1
        return None
    
    async def _evaluate_single_problem_async(self, prompt: str, question: str, ground_truth: float) -> Dict[str, Any]:
        """
        Evaluate a single problem asynchronously.
        
        Args:
            prompt: The prompt to evaluate
            question: The GSM8K question
            ground_truth: The correct answer
            
        Returns:
            Evaluation result dictionary
        """
        # Check cache first
        cache_key = self._create_cache_key(prompt, question)
        
        async with self.cache_lock:
            if self.cache_enabled and cache_key in self.response_cache:
                self.cache_hits += 1
                cached_response = self.response_cache[cache_key]
                predicted_answer = cached_response['answer']
                response = cached_response['response']
            else:
                # Construct the full prompt
                full_prompt = f"{prompt}\n\nProblem: {question}\n\nSolution:"
                
                messages = [
                    {"role": "user", "content": full_prompt}
                ]
                
                # Make API request
                response = await self._make_api_request_async(messages)
                
                if response is None:
                    predicted_answer = None
                else:
                    # Extract numerical answer
                    predicted_answer = extract_answer_from_response(response)
                    
                    # Cache the result
                    if self.cache_enabled:
                        self.response_cache[cache_key] = {
                            'answer': predicted_answer,
                            'response': response
                        }
        
        # Calculate correctness
        is_correct = False
        if predicted_answer is not None and ground_truth is not None:
            is_correct = abs(predicted_answer - ground_truth) < 0.001
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'response': response or "",
            'response_length': len(response) if response else 0,
            'cache_hit': cache_key in self.response_cache if self.cache_enabled else False
        }

    async def batch_evaluate_async(self,
                                  prompt: str,
                                  problems: List[Dict[str, Any]],
                                  progress_callback: Optional[Callable] = None) -> BatchResult:
        """
        Evaluate a prompt on multiple problems using batch processing.

        Args:
            prompt: The prompt to evaluate
            problems: List of problem dictionaries with 'question' and 'final_answer'
            progress_callback: Optional callback function for progress updates

        Returns:
            BatchResult with evaluation results and statistics
        """
        start_time = time.time()
        all_results = []
        total_cache_hits = 0
        total_api_calls = 0

        # Split problems into batches
        batches = [problems[i:i + self.batch_config.batch_size]
                  for i in range(0, len(problems), self.batch_config.batch_size)]

        print(f"Processing {len(problems)} problems in {len(batches)} batches of size {self.batch_config.batch_size}")

        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()

            # Create semaphore to limit concurrent requests within batch
            semaphore = asyncio.Semaphore(self.batch_config.max_concurrent_requests)

            async def evaluate_with_semaphore(problem, problem_idx):
                async with semaphore:
                    result = await self._evaluate_single_problem_async(
                        prompt,
                        problem['question'],
                        problem['final_answer']
                    )
                    result['problem_id'] = problem.get('id', f'problem_{batch_idx}_{problem_idx}')
                    return result

            # Process batch concurrently
            tasks = [
                evaluate_with_semaphore(problem, idx)
                for idx, problem in enumerate(batch)
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            successful_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Error processing problem in batch {batch_idx}, item {i}: {result}")
                    # Create a failed result
                    problem = batch[i]
                    failed_result = {
                        'problem_id': problem.get('id', f'problem_{batch_idx}_{i}'),
                        'question': problem['question'],
                        'ground_truth': problem['final_answer'],
                        'predicted_answer': None,
                        'is_correct': False,
                        'response': "",
                        'response_length': 0,
                        'cache_hit': False,
                        'error': str(result)
                    }
                    successful_results.append(failed_result)
                else:
                    successful_results.append(result)
                    if result.get('cache_hit', False):
                        total_cache_hits += 1
                    else:
                        total_api_calls += 1

            all_results.extend(successful_results)

            batch_time = time.time() - batch_start_time
            print(f"Batch {batch_idx + 1}/{len(batches)} completed in {batch_time:.2f}s "
                  f"({len(successful_results)} problems)")

            # Progress callback
            if progress_callback:
                progress_callback(
                    batch_idx + 1,
                    len(batches),
                    {
                        'batch_results': successful_results,
                        'batch_time': batch_time,
                        'problems_processed': len(all_results)
                    }
                )

        total_time = time.time() - start_time

        return BatchResult(
            results=all_results,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            total_time=total_time,
            cache_hits=total_cache_hits,
            api_calls_made=total_api_calls
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        success_rate = (self.successful_requests / self.total_requests
                       if self.total_requests > 0 else 0.0)

        cache_hit_rate = (self.cache_hits / (self.total_requests + self.cache_hits)
                         if (self.total_requests + self.cache_hits) > 0 else 0.0)

        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        throughput = self.total_requests / elapsed_time if elapsed_time > 0 else 0.0
        avg_wait_time = self.total_wait_time / self.rate_limit_hits if self.rate_limit_hits > 0 else 0.0

        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'total_tokens_used': self.total_tokens_used,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'throughput_requests_per_second': throughput,
            'rate_limiting': {
                'total_wait_time': self.total_wait_time,
                'rate_limit_hits': self.rate_limit_hits,
                'average_wait_time': avg_wait_time
            },
            'batch_config': {
                'batch_size': self.batch_config.batch_size,
                'max_concurrent_requests': self.batch_config.max_concurrent_requests,
                'rate_limit_per_minute': self.batch_config.rate_limit_per_minute
            }
        }
