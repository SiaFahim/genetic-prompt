"""
Seed prompt validation system for quality and diversity analysis.
"""

import re
import statistics
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass
from collections import Counter

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.seeds.prompt_categories import PromptCategory, PromptCategoryManager
    from src.seeds.base_seeds import BaseSeedCollection, SeedPrompt
    from src.embeddings.vocabulary import vocabulary
    from src.utils.config import config
else:
    from .prompt_categories import PromptCategory, PromptCategoryManager
    from .base_seeds import BaseSeedCollection, SeedPrompt
    from ..embeddings.vocabulary import vocabulary
    from ..utils.config import config


@dataclass
class ValidationMetrics:
    """Metrics for seed prompt validation."""
    length_stats: Dict[str, float]
    diversity_score: float
    category_balance: float
    uniqueness_score: float
    quality_indicators: Dict[str, float]
    readability_score: float
    coverage_score: float
    overall_score: float


class SeedValidator:
    """Validates seed prompt collections for quality and diversity."""
    
    def __init__(self):
        """Initialize seed validator."""
        self.category_manager = PromptCategoryManager()
        
        # Quality indicators (positive patterns)
        self.quality_patterns = {
            'step_indicators': [
                r'\bstep\b', r'\bfirst\b', r'\bnext\b', r'\bthen\b', r'\bfinally\b'
            ],
            'reasoning_words': [
                r'\bbecause\b', r'\bsince\b', r'\btherefore\b', r'\bso\b', r'\bthus\b'
            ],
            'problem_solving': [
                r'\bsolve\b', r'\bfind\b', r'\bcalculate\b', r'\bdetermine\b', r'\bidentify\b'
            ],
            'organization': [
                r'\borganize\b', r'\bstructure\b', r'\bsystematic\b', r'\bmethodical\b'
            ],
            'verification': [
                r'\bcheck\b', r'\bverify\b', r'\bestimate\b', r'\breasonable\b'
            ]
        }
        
        # Negative patterns (to avoid)
        self.negative_patterns = [
            r'\bum\b', r'\buh\b', r'\berr\b',  # Hesitation
            r'\bmaybe\b', r'\bperhaps\b', r'\bmight\b',  # Uncertainty
            r'\bI think\b', r'\bI guess\b',  # Weak confidence
            r'\bkinda\b', r'\bsorta\b'  # Informal language
        ]
    
    def validate_collection(self, seeds: List[SeedPrompt]) -> ValidationMetrics:
        """
        Validate a collection of seed prompts.
        
        Args:
            seeds: List of seed prompts to validate
            
        Returns:
            ValidationMetrics with comprehensive analysis
        """
        # Extract texts for analysis
        texts = [seed.text for seed in seeds]
        
        # Calculate individual metrics
        length_stats = self._calculate_length_statistics(texts)
        diversity_score = self._calculate_diversity_score(texts)
        category_balance = self._calculate_category_balance(seeds)
        uniqueness_score = self._calculate_uniqueness_score(texts)
        quality_indicators = self._calculate_quality_indicators(texts)
        readability_score = self._calculate_readability_score(texts)
        coverage_score = self._calculate_coverage_score(seeds)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            diversity_score, category_balance, uniqueness_score,
            quality_indicators, readability_score, coverage_score
        )
        
        return ValidationMetrics(
            length_stats=length_stats,
            diversity_score=diversity_score,
            category_balance=category_balance,
            uniqueness_score=uniqueness_score,
            quality_indicators=quality_indicators,
            readability_score=readability_score,
            coverage_score=coverage_score,
            overall_score=overall_score
        )
    
    def _calculate_length_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate length statistics for prompt texts."""
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'char_mean': statistics.mean(lengths),
            'char_std': statistics.stdev(lengths) if len(lengths) > 1 else 0,
            'char_min': min(lengths),
            'char_max': max(lengths),
            'word_mean': statistics.mean(word_counts),
            'word_std': statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
            'word_min': min(word_counts),
            'word_max': max(word_counts)
        }
    
    def _calculate_diversity_score(self, texts: List[str]) -> float:
        """Calculate lexical diversity score."""
        # Combine all texts
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        # Calculate type-token ratio (unique words / total words)
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        diversity = unique_words / total_words
        
        # Normalize to 0-1 scale (higher is better)
        return min(1.0, diversity * 2)  # Scale up since TTR is typically low
    
    def _calculate_category_balance(self, seeds: List[SeedPrompt]) -> float:
        """Calculate how balanced the category distribution is."""
        # Get target distribution
        target_dist = self.category_manager.get_category_distribution_target()
        
        # Get actual distribution
        actual_dist = Counter(seed.category for seed in seeds)
        
        # Calculate balance score
        total_deviation = 0
        total_target = sum(target_dist.values())
        
        for category, target_count in target_dist.items():
            actual_count = actual_dist.get(category, 0)
            deviation = abs(actual_count - target_count)
            total_deviation += deviation
        
        # Convert to 0-1 score (1 = perfect balance)
        max_possible_deviation = total_target
        balance_score = 1.0 - (total_deviation / max_possible_deviation)
        
        return max(0.0, balance_score)
    
    def _calculate_uniqueness_score(self, texts: List[str]) -> float:
        """Calculate how unique the prompts are from each other."""
        if len(texts) < 2:
            return 1.0
        
        # Calculate pairwise similarity
        similarities = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self._calculate_text_similarity(texts[i], texts[j])
                similarities.append(similarity)
        
        # Uniqueness is inverse of average similarity
        avg_similarity = statistics.mean(similarities)
        uniqueness = 1.0 - avg_similarity
        
        return max(0.0, uniqueness)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_quality_indicators(self, texts: List[str]) -> Dict[str, float]:
        """Calculate quality indicator scores."""
        quality_scores = {}
        
        # Positive indicators
        for indicator_type, patterns in self.quality_patterns.items():
            total_matches = 0
            for text in texts:
                text_lower = text.lower()
                for pattern in patterns:
                    matches = len(re.findall(pattern, text_lower))
                    total_matches += matches
            
            # Normalize by number of texts
            quality_scores[indicator_type] = total_matches / len(texts)
        
        # Negative indicators (penalty)
        negative_matches = 0
        for text in texts:
            text_lower = text.lower()
            for pattern in self.negative_patterns:
                matches = len(re.findall(pattern, text_lower))
                negative_matches += matches
        
        quality_scores['negative_penalty'] = negative_matches / len(texts)
        
        # Overall quality score
        positive_score = sum(score for key, score in quality_scores.items() 
                           if key != 'negative_penalty')
        penalty = quality_scores['negative_penalty']
        
        quality_scores['overall_quality'] = max(0.0, positive_score - penalty * 2)
        
        return quality_scores
    
    def _calculate_readability_score(self, texts: List[str]) -> float:
        """Calculate readability score based on sentence structure."""
        readability_scores = []
        
        for text in texts:
            # Simple readability metrics
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                readability_scores.append(0.0)
                continue
            
            # Average sentence length (words)
            total_words = sum(len(sentence.split()) for sentence in sentences)
            avg_sentence_length = total_words / len(sentences)
            
            # Readability score (optimal sentence length is 10-20 words)
            if 10 <= avg_sentence_length <= 20:
                readability = 1.0
            elif avg_sentence_length < 10:
                readability = avg_sentence_length / 10
            else:
                readability = max(0.1, 20 / avg_sentence_length)
            
            readability_scores.append(readability)
        
        return statistics.mean(readability_scores)
    
    def _calculate_coverage_score(self, seeds: List[SeedPrompt]) -> float:
        """Calculate how well the seeds cover different problem-solving aspects."""
        # Check coverage of key problem-solving elements
        coverage_elements = {
            'step_by_step': ['step', 'first', 'next', 'then'],
            'understanding': ['understand', 'identify', 'know', 'find'],
            'calculation': ['calculate', 'solve', 'compute', 'determine'],
            'verification': ['check', 'verify', 'reasonable', 'estimate'],
            'organization': ['organize', 'structure', 'systematic', 'method']
        }
        
        covered_elements = set()
        all_text = ' '.join(seed.text.lower() for seed in seeds)
        
        for element_type, keywords in coverage_elements.items():
            for keyword in keywords:
                if keyword in all_text:
                    covered_elements.add(element_type)
                    break
        
        coverage_score = len(covered_elements) / len(coverage_elements)
        return coverage_score
    
    def _calculate_overall_score(self, diversity: float, balance: float, 
                               uniqueness: float, quality: Dict[str, float],
                               readability: float, coverage: float) -> float:
        """Calculate overall validation score."""
        # Weighted combination of all metrics
        weights = {
            'diversity': 0.15,
            'balance': 0.20,
            'uniqueness': 0.15,
            'quality': 0.25,
            'readability': 0.10,
            'coverage': 0.15
        }
        
        quality_score = quality.get('overall_quality', 0.0)
        
        overall = (
            diversity * weights['diversity'] +
            balance * weights['balance'] +
            uniqueness * weights['uniqueness'] +
            quality_score * weights['quality'] +
            readability * weights['readability'] +
            coverage * weights['coverage']
        )
        
        return min(1.0, max(0.0, overall))
    
    def generate_validation_report(self, metrics: ValidationMetrics) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("🔍 SEED PROMPT VALIDATION REPORT")
        report.append("=" * 50)
        
        # Overall score
        score_emoji = "🟢" if metrics.overall_score >= 0.8 else "🟡" if metrics.overall_score >= 0.6 else "🔴"
        report.append(f"{score_emoji} Overall Score: {metrics.overall_score:.3f}")
        report.append("")
        
        # Length statistics
        report.append("📏 Length Statistics:")
        report.append(f"   Characters: {metrics.length_stats['char_mean']:.1f} ± {metrics.length_stats['char_std']:.1f}")
        report.append(f"   Words: {metrics.length_stats['word_mean']:.1f} ± {metrics.length_stats['word_std']:.1f}")
        report.append(f"   Range: {metrics.length_stats['word_min']}-{metrics.length_stats['word_max']} words")
        report.append("")
        
        # Individual scores
        report.append("📊 Individual Metrics:")
        report.append(f"   Diversity Score: {metrics.diversity_score:.3f}")
        report.append(f"   Category Balance: {metrics.category_balance:.3f}")
        report.append(f"   Uniqueness Score: {metrics.uniqueness_score:.3f}")
        report.append(f"   Readability Score: {metrics.readability_score:.3f}")
        report.append(f"   Coverage Score: {metrics.coverage_score:.3f}")
        report.append("")
        
        # Quality indicators
        report.append("✨ Quality Indicators:")
        for indicator, score in metrics.quality_indicators.items():
            if indicator != 'overall_quality':
                report.append(f"   {indicator.replace('_', ' ').title()}: {score:.2f}")
        report.append("")
        
        # Recommendations
        report.append("💡 Recommendations:")
        if metrics.diversity_score < 0.6:
            report.append("   - Increase lexical diversity with more varied vocabulary")
        if metrics.category_balance < 0.8:
            report.append("   - Rebalance category distribution to match targets")
        if metrics.uniqueness_score < 0.7:
            report.append("   - Reduce similarity between prompts for better uniqueness")
        if metrics.quality_indicators.get('overall_quality', 0) < 0.5:
            report.append("   - Add more quality indicators (step words, reasoning terms)")
        if metrics.readability_score < 0.7:
            report.append("   - Improve readability with better sentence structure")
        if metrics.coverage_score < 0.8:
            report.append("   - Ensure coverage of all problem-solving aspects")
        
        if metrics.overall_score >= 0.8:
            report.append("   ✅ Seed collection meets quality standards!")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test seed validation system
    print("Testing seed validation system...")
    
    # Load vocabulary for any needed operations
    vocab_file = config.get_data_dir() / "embeddings" / "vocabulary.pkl"
    if vocab_file.exists():
        vocabulary.load_vocabulary(vocab_file)
    else:
        vocabulary._create_basic_vocabulary()
    
    # Create validator
    validator = SeedValidator()
    
    # Load base seed collection
    collection = BaseSeedCollection()
    seeds = collection.get_all_seeds()
    
    print(f"✅ Loaded {len(seeds)} seeds for validation")
    
    # Validate collection
    metrics = validator.validate_collection(seeds)
    
    print(f"✅ Validation completed")
    print(f"   Overall Score: {metrics.overall_score:.3f}")
    print(f"   Diversity: {metrics.diversity_score:.3f}")
    print(f"   Balance: {metrics.category_balance:.3f}")
    print(f"   Uniqueness: {metrics.uniqueness_score:.3f}")
    
    # Generate report
    report = validator.generate_validation_report(metrics)
    print("\n" + report)
    
    print("\n🎯 Seed validation system tests completed successfully!")
