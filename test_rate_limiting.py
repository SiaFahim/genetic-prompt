#!/usr/bin/env python3
"""
Test script to verify rate limiting improvements in AsyncLLMInterface.
"""

import sys
import os
import asyncio
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.async_llm_interface import AsyncLLMInterface, BatchConfig
from src.utils.config import config
from src.embeddings.vocabulary import vocabulary

async def test_rate_limiting_performance():
    """Test that rate limiting improvements work correctly."""
    print("🧪 Testing Rate Limiting Performance")
    print("=" * 50)
    
    # Initialize vocabulary first
    print("Initializing vocabulary...")
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
        print(f"✅ Vocabulary loaded: {len(vocabulary.token_to_id)} tokens")
    else:
        print("📚 Creating basic vocabulary...")
        vocabulary._create_basic_vocabulary()
        print(f"✅ Basic vocabulary created: {len(vocabulary.token_to_id)} tokens")
    
    # Create conservative batch config for testing
    batch_config = BatchConfig(
        batch_size=5,  # Small batch for quick test
        max_concurrent_requests=3,  # Limited concurrency
        rate_limit_per_minute=10,  # Very low limit to trigger rate limiting
        retry_attempts=2,
        base_delay=0.5,
        max_delay=5.0,
        timeout=10
    )
    
    # Create async LLM interface
    try:
        async_llm = AsyncLLMInterface(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=50,  # Short responses for speed
            batch_config=batch_config
        )
        print(f"✅ AsyncLLMInterface created with rate limit: {batch_config.rate_limit_per_minute}/min")
    except Exception as e:
        print(f"❌ Failed to create AsyncLLMInterface: {e}")
        return False
    
    # Create test problems
    test_problems = [
        {
            'id': f'test_{i}',
            'question': f'What is {i} + {i}?',
            'final_answer': i + i
        }
        for i in range(1, 16)  # 15 problems to trigger rate limiting
    ]
    
    print(f"Testing with {len(test_problems)} problems...")
    print(f"Expected to trigger rate limiting with {batch_config.rate_limit_per_minute}/min limit")
    
    # Test batch evaluation
    start_time = time.time()
    try:
        result = await async_llm.batch_evaluate_async(
            prompt="Solve this math problem step by step:",
            problems=test_problems
        )
        
        total_time = time.time() - start_time
        
        # Get statistics
        stats = async_llm.get_statistics()
        
        print(f"\n📊 Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Problems processed: {len(result.results)}")
        print(f"   Throughput: {len(result.results)/total_time:.2f} problems/second")
        print(f"   API calls made: {result.api_calls_made}")
        print(f"   Cache hits: {result.cache_hits}")
        
        print(f"\n🔄 Rate Limiting Stats:")
        rate_limiting = stats.get('rate_limiting', {})
        print(f"   Rate limit hits: {rate_limiting.get('rate_limit_hits', 0)}")
        print(f"   Total wait time: {rate_limiting.get('total_wait_time', 0):.2f}s")
        print(f"   Average wait time: {rate_limiting.get('average_wait_time', 0):.2f}s")
        
        # Verify improvements
        success = True
        if rate_limiting.get('rate_limit_hits', 0) > 0:
            avg_wait = rate_limiting.get('average_wait_time', 0)
            if avg_wait > 30:  # Should be much less than old 60s waits
                print(f"⚠️  Average wait time ({avg_wait:.2f}s) is still high")
                success = False
            else:
                print(f"✅ Rate limiting optimized: avg wait {avg_wait:.2f}s (vs old 60s)")
        
        throughput = len(result.results) / total_time
        if throughput < 0.5:  # Should be much better than 0.3 problems/second
            print(f"⚠️  Throughput ({throughput:.2f}/s) is still low")
            success = False
        else:
            print(f"✅ Throughput improved: {throughput:.2f} problems/second")
        
        return success
        
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_rate_limiting_performance())
        if success:
            print("\n🎉 Rate limiting improvements are working!")
            sys.exit(0)
        else:
            print("\n💥 Rate limiting improvements need more work!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
