"""
Prompt categories and strategies for GSM8K mathematical reasoning.
"""

from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass


class PromptCategory(Enum):
    """Categories of mathematical problem-solving strategies."""
    STEP_BY_STEP = "step_by_step"
    VISUAL_REASONING = "visual_reasoning"
    ALGEBRAIC_APPROACH = "algebraic_approach"
    LOGICAL_BREAKDOWN = "logical_breakdown"
    PATTERN_RECOGNITION = "pattern_recognition"
    ESTIMATION_CHECKING = "estimation_checking"
    WORD_PROBLEM_PARSING = "word_problem_parsing"
    MULTIPLE_METHODS = "multiple_methods"
    CONCEPTUAL_UNDERSTANDING = "conceptual_understanding"
    SYSTEMATIC_ORGANIZATION = "systematic_organization"


@dataclass
class CategoryDefinition:
    """Definition of a prompt category."""
    name: str
    description: str
    key_strategies: List[str]
    example_phrases: List[str]
    target_problems: List[str]
    expected_benefits: List[str]


class PromptCategoryManager:
    """Manages prompt categories and their definitions."""
    
    def __init__(self):
        """Initialize category definitions."""
        self.categories = self._define_categories()
    
    def _define_categories(self) -> Dict[PromptCategory, CategoryDefinition]:
        """Define all prompt categories with detailed specifications."""
        
        categories = {}
        
        # 1. Step-by-Step Reasoning
        categories[PromptCategory.STEP_BY_STEP] = CategoryDefinition(
            name="Step-by-Step Reasoning",
            description="Encourages systematic, sequential problem solving with clear steps",
            key_strategies=[
                "Break down complex problems into smaller parts",
                "Show work explicitly at each stage",
                "Use numbered or bulleted steps",
                "Verify each step before proceeding"
            ],
            example_phrases=[
                "Let's solve this step by step",
                "First, let's identify what we know",
                "Step 1:", "Next, we need to",
                "Breaking this down:"
            ],
            target_problems=[
                "Multi-step arithmetic problems",
                "Word problems with multiple operations",
                "Problems requiring sequential calculations"
            ],
            expected_benefits=[
                "Reduces calculation errors",
                "Makes reasoning transparent",
                "Easier to debug incorrect solutions"
            ]
        )
        
        # 2. Visual Reasoning
        categories[PromptCategory.VISUAL_REASONING] = CategoryDefinition(
            name="Visual Reasoning",
            description="Encourages mental visualization and diagrammatic thinking",
            key_strategies=[
                "Visualize the problem scenario",
                "Draw mental pictures or diagrams",
                "Use spatial reasoning",
                "Think about physical representations"
            ],
            example_phrases=[
                "Let's visualize this problem",
                "Imagine we have", "Picture this scenario",
                "If we draw this out", "Visualizing the situation"
            ],
            target_problems=[
                "Geometry problems",
                "Problems involving arrangements",
                "Spatial reasoning tasks",
                "Problems with physical objects"
            ],
            expected_benefits=[
                "Better understanding of problem context",
                "Intuitive problem solving",
                "Reduced abstract thinking burden"
            ]
        )
        
        # 3. Algebraic Approach
        categories[PromptCategory.ALGEBRAIC_APPROACH] = CategoryDefinition(
            name="Algebraic Approach",
            description="Uses variables, equations, and algebraic manipulation",
            key_strategies=[
                "Define variables for unknown quantities",
                "Set up equations based on relationships",
                "Use algebraic manipulation",
                "Solve systematically"
            ],
            example_phrases=[
                "Let's define variables", "Let x represent",
                "Setting up an equation", "We can write this as",
                "Solving algebraically"
            ],
            target_problems=[
                "Problems with unknown quantities",
                "Ratio and proportion problems",
                "Systems of relationships",
                "Complex numerical relationships"
            ],
            expected_benefits=[
                "Handles complex relationships",
                "Systematic solution approach",
                "Generalizable method"
            ]
        )
        
        # 4. Logical Breakdown
        categories[PromptCategory.LOGICAL_BREAKDOWN] = CategoryDefinition(
            name="Logical Breakdown",
            description="Emphasizes logical reasoning and cause-effect relationships",
            key_strategies=[
                "Identify logical relationships",
                "Use if-then reasoning",
                "Consider cause and effect",
                "Apply logical principles"
            ],
            example_phrases=[
                "Let's think logically about this",
                "If this is true, then", "The logical approach is",
                "Reasoning through this", "This means that"
            ],
            target_problems=[
                "Logic puzzles",
                "Conditional problems",
                "Problems with constraints",
                "Deductive reasoning tasks"
            ],
            expected_benefits=[
                "Clear reasoning chains",
                "Handles complex conditions",
                "Systematic logical thinking"
            ]
        )
        
        # 5. Pattern Recognition
        categories[PromptCategory.PATTERN_RECOGNITION] = CategoryDefinition(
            name="Pattern Recognition",
            description="Identifies patterns, sequences, and recurring structures",
            key_strategies=[
                "Look for patterns in numbers",
                "Identify recurring structures",
                "Use pattern-based shortcuts",
                "Recognize familiar problem types"
            ],
            example_phrases=[
                "I notice a pattern here", "This follows the pattern",
                "Looking for patterns", "This is similar to",
                "The pattern suggests"
            ],
            target_problems=[
                "Sequence problems",
                "Repetitive calculations",
                "Problems with regular structures",
                "Series and progressions"
            ],
            expected_benefits=[
                "Faster problem solving",
                "Recognition of problem types",
                "Efficient solution methods"
            ]
        )
        
        # 6. Estimation and Checking
        categories[PromptCategory.ESTIMATION_CHECKING] = CategoryDefinition(
            name="Estimation and Checking",
            description="Uses estimation for validation and reasonableness checks",
            key_strategies=[
                "Estimate before calculating",
                "Check if answers are reasonable",
                "Use approximation for validation",
                "Verify with different methods"
            ],
            example_phrases=[
                "Let's estimate first", "Does this seem reasonable?",
                "Checking our answer", "This should be approximately",
                "Let me verify this"
            ],
            target_problems=[
                "Large number calculations",
                "Real-world context problems",
                "Problems where errors are costly",
                "Complex multi-step problems"
            ],
            expected_benefits=[
                "Catches calculation errors",
                "Builds number sense",
                "Increases confidence in answers"
            ]
        )
        
        # 7. Word Problem Parsing
        categories[PromptCategory.WORD_PROBLEM_PARSING] = CategoryDefinition(
            name="Word Problem Parsing",
            description="Focuses on understanding and extracting information from text",
            key_strategies=[
                "Identify key information",
                "Separate given from unknown",
                "Understand the question being asked",
                "Translate words to mathematical operations"
            ],
            example_phrases=[
                "Let's identify what we know", "The problem tells us",
                "We need to find", "The key information is",
                "Translating to math"
            ],
            target_problems=[
                "Complex word problems",
                "Problems with extraneous information",
                "Multi-part questions",
                "Context-heavy problems"
            ],
            expected_benefits=[
                "Better problem comprehension",
                "Avoids missing key information",
                "Clearer problem setup"
            ]
        )
        
        # 8. Multiple Methods
        categories[PromptCategory.MULTIPLE_METHODS] = CategoryDefinition(
            name="Multiple Methods",
            description="Considers and compares different solution approaches",
            key_strategies=[
                "Try multiple solution methods",
                "Compare different approaches",
                "Use the most efficient method",
                "Cross-validate with different techniques"
            ],
            example_phrases=[
                "There are several ways to solve this",
                "Another approach would be", "Alternatively",
                "Let's try a different method", "Comparing methods"
            ],
            target_problems=[
                "Problems with multiple solution paths",
                "Complex problems requiring validation",
                "Problems where efficiency matters",
                "Educational contexts"
            ],
            expected_benefits=[
                "Increased solution confidence",
                "Learning multiple techniques",
                "Better problem understanding"
            ]
        )
        
        # 9. Conceptual Understanding
        categories[PromptCategory.CONCEPTUAL_UNDERSTANDING] = CategoryDefinition(
            name="Conceptual Understanding",
            description="Emphasizes understanding underlying mathematical concepts",
            key_strategies=[
                "Explain the underlying concepts",
                "Connect to mathematical principles",
                "Understand why methods work",
                "Build conceptual foundations"
            ],
            example_phrases=[
                "The concept here is", "This relates to",
                "Understanding why", "The principle behind this",
                "Conceptually, this means"
            ],
            target_problems=[
                "Problems requiring deep understanding",
                "Conceptual application problems",
                "Problems connecting multiple concepts",
                "Educational scenarios"
            ],
            expected_benefits=[
                "Deeper mathematical understanding",
                "Better transfer to new problems",
                "More robust problem solving"
            ]
        )
        
        # 10. Systematic Organization
        categories[PromptCategory.SYSTEMATIC_ORGANIZATION] = CategoryDefinition(
            name="Systematic Organization",
            description="Organizes information and solution process systematically",
            key_strategies=[
                "Organize information clearly",
                "Use systematic notation",
                "Structure the solution process",
                "Maintain clear organization throughout"
            ],
            example_phrases=[
                "Let's organize this systematically",
                "Structuring our approach", "Organizing the information",
                "Systematically working through", "Clear organization"
            ],
            target_problems=[
                "Complex multi-part problems",
                "Problems with lots of information",
                "Problems requiring careful tracking",
                "Long solution processes"
            ],
            expected_benefits=[
                "Reduces errors from disorganization",
                "Clearer solution presentation",
                "Better problem management"
            ]
        )
        
        return categories
    
    def get_category(self, category: PromptCategory) -> CategoryDefinition:
        """Get definition for a specific category."""
        return self.categories[category]
    
    def get_all_categories(self) -> Dict[PromptCategory, CategoryDefinition]:
        """Get all category definitions."""
        return self.categories.copy()
    
    def get_category_names(self) -> List[str]:
        """Get list of all category names."""
        return [cat.name for cat in self.categories.values()]
    
    def get_strategies_for_category(self, category: PromptCategory) -> List[str]:
        """Get key strategies for a specific category."""
        return self.categories[category].key_strategies
    
    def get_example_phrases(self, category: PromptCategory) -> List[str]:
        """Get example phrases for a specific category."""
        return self.categories[category].example_phrases
    
    def analyze_prompt_categories(self, prompt_text: str) -> Dict[PromptCategory, float]:
        """
        Analyze which categories a prompt text aligns with.
        
        Args:
            prompt_text: Text to analyze
            
        Returns:
            Dictionary mapping categories to alignment scores (0-1)
        """
        prompt_lower = prompt_text.lower()
        category_scores = {}
        
        for category, definition in self.categories.items():
            score = 0.0
            phrase_count = 0
            
            # Check for example phrases
            for phrase in definition.example_phrases:
                if phrase.lower() in prompt_lower:
                    score += 1.0
                    phrase_count += 1
            
            # Normalize by number of phrases checked
            if len(definition.example_phrases) > 0:
                category_scores[category] = min(1.0, score / len(definition.example_phrases))
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def get_category_distribution_target(self) -> Dict[PromptCategory, int]:
        """Get target distribution of prompts across categories for 50 seeds."""
        # Aim for balanced distribution with slight emphasis on core strategies
        return {
            PromptCategory.STEP_BY_STEP: 8,  # Most fundamental
            PromptCategory.WORD_PROBLEM_PARSING: 6,  # Essential for GSM8K
            PromptCategory.LOGICAL_BREAKDOWN: 5,
            PromptCategory.ALGEBRAIC_APPROACH: 5,
            PromptCategory.ESTIMATION_CHECKING: 5,
            PromptCategory.VISUAL_REASONING: 4,
            PromptCategory.SYSTEMATIC_ORGANIZATION: 4,
            PromptCategory.PATTERN_RECOGNITION: 4,
            PromptCategory.MULTIPLE_METHODS: 4,
            PromptCategory.CONCEPTUAL_UNDERSTANDING: 5
        }


if __name__ == "__main__":
    # Test category system
    print("Testing prompt category system...")

    # Create seeds directory
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

    manager = PromptCategoryManager()
    
    # Test category retrieval
    step_by_step = manager.get_category(PromptCategory.STEP_BY_STEP)
    print(f"✅ Step-by-step category: {step_by_step.name}")
    print(f"   Strategies: {len(step_by_step.key_strategies)}")
    print(f"   Example phrases: {len(step_by_step.example_phrases)}")
    
    # Test all categories
    all_categories = manager.get_all_categories()
    print(f"✅ Total categories defined: {len(all_categories)}")
    
    # Test category analysis
    test_prompt = "Let's solve this step by step. First, let's identify what we know."
    scores = manager.analyze_prompt_categories(test_prompt)
    
    # Find top categories
    top_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print("✅ Category analysis for test prompt:")
    for category, score in top_categories:
        if score > 0:
            print(f"   {category.value}: {score:.2f}")
    
    # Test distribution target
    target_dist = manager.get_category_distribution_target()
    total_target = sum(target_dist.values())
    print(f"✅ Target distribution: {total_target} total prompts")
    
    # Verify all categories are covered
    all_category_enums = set(PromptCategory)
    defined_categories = set(all_categories.keys())
    missing = all_category_enums - defined_categories
    
    if not missing:
        print("✅ All categories properly defined")
    else:
        print(f"❌ Missing categories: {missing}")
    
    print("\n🎯 Prompt category system tests completed successfully!")
