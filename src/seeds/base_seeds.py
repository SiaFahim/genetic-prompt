"""
Base seed prompts for GSM8K mathematical reasoning evolution.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.seeds.prompt_categories import PromptCategory, PromptCategoryManager
else:
    from .prompt_categories import PromptCategory, PromptCategoryManager


@dataclass
class SeedPrompt:
    """Represents a seed prompt with metadata."""
    id: str
    text: str
    category: PromptCategory
    description: str
    expected_strength: str
    variations: List[str]


class BaseSeedCollection:
    """Collection of 50 diverse base seed prompts for GSM8K evolution."""
    
    def __init__(self):
        """Initialize the base seed collection."""
        self.seeds = self._create_base_seeds()
        self.category_manager = PromptCategoryManager()
    
    def _create_base_seeds(self) -> List[SeedPrompt]:
        """Create the complete collection of 50 base seed prompts."""
        seeds = []
        
        # Step-by-Step Reasoning (8 prompts)
        seeds.extend([
            SeedPrompt(
                id="step_001",
                text="Let's solve this step by step.",
                category=PromptCategory.STEP_BY_STEP,
                description="Classic step-by-step approach",
                expected_strength="Clear sequential reasoning",
                variations=["Let's work through this step by step", "I'll solve this systematically"]
            ),
            SeedPrompt(
                id="step_002", 
                text="First, let me identify what we know and what we need to find.",
                category=PromptCategory.STEP_BY_STEP,
                description="Information gathering first",
                expected_strength="Organized problem setup",
                variations=["Let me start by listing what we know", "First, I'll identify the given information"]
            ),
            SeedPrompt(
                id="step_003",
                text="I'll break this problem down into smaller, manageable parts.",
                category=PromptCategory.STEP_BY_STEP,
                description="Problem decomposition",
                expected_strength="Handles complex problems",
                variations=["Let me break this into smaller pieces", "I'll divide this into manageable steps"]
            ),
            SeedPrompt(
                id="step_004",
                text="Let me work through this systematically, checking each step.",
                category=PromptCategory.STEP_BY_STEP,
                description="Systematic with verification",
                expected_strength="Error reduction through checking",
                variations=["I'll solve this methodically", "Working systematically through each part"]
            ),
            SeedPrompt(
                id="step_005",
                text="Step 1: Understand the problem. Step 2: Plan the solution. Step 3: Execute.",
                category=PromptCategory.STEP_BY_STEP,
                description="Structured problem-solving framework",
                expected_strength="Comprehensive approach",
                variations=["First understand, then plan, then solve", "Understand → Plan → Execute"]
            ),
            SeedPrompt(
                id="step_006",
                text="Let me solve this carefully, one step at a time.",
                category=PromptCategory.STEP_BY_STEP,
                description="Careful sequential approach",
                expected_strength="Accuracy through deliberation",
                variations=["I'll work through this carefully", "Taking this one step at a time"]
            ),
            SeedPrompt(
                id="step_007",
                text="I'll organize my solution into clear, numbered steps.",
                category=PromptCategory.STEP_BY_STEP,
                description="Numbered step organization",
                expected_strength="Clear solution structure",
                variations=["Let me number each step clearly", "I'll use numbered steps for clarity"]
            ),
            SeedPrompt(
                id="step_008",
                text="Breaking this down: what do we have, what do we need, how do we get there?",
                category=PromptCategory.STEP_BY_STEP,
                description="Three-part breakdown",
                expected_strength="Logical problem analysis",
                variations=["Have → Need → How", "Given → Required → Method"]
            )
        ])
        
        # Word Problem Parsing (6 prompts)
        seeds.extend([
            SeedPrompt(
                id="parse_001",
                text="Let me carefully read and understand what this problem is asking.",
                category=PromptCategory.WORD_PROBLEM_PARSING,
                description="Careful reading emphasis",
                expected_strength="Better comprehension",
                variations=["I'll read this carefully first", "Let me understand what's being asked"]
            ),
            SeedPrompt(
                id="parse_002",
                text="I'll identify the key information and ignore any distractors.",
                category=PromptCategory.WORD_PROBLEM_PARSING,
                description="Information filtering",
                expected_strength="Handles complex word problems",
                variations=["Finding the important information", "Separating key facts from distractors"]
            ),
            SeedPrompt(
                id="parse_003",
                text="Let me translate this word problem into mathematical operations.",
                category=PromptCategory.WORD_PROBLEM_PARSING,
                description="Word-to-math translation",
                expected_strength="Clear mathematical setup",
                variations=["Converting words to math", "Translating to mathematical language"]
            ),
            SeedPrompt(
                id="parse_004",
                text="What is the problem really asking? Let me rephrase it clearly.",
                category=PromptCategory.WORD_PROBLEM_PARSING,
                description="Question clarification",
                expected_strength="Avoids misunderstanding",
                variations=["What exactly are we solving for?", "Let me clarify the question"]
            ),
            SeedPrompt(
                id="parse_005",
                text="I'll extract the numbers and relationships from this word problem.",
                category=PromptCategory.WORD_PROBLEM_PARSING,
                description="Data extraction focus",
                expected_strength="Systematic information gathering",
                variations=["Finding all the numbers and connections", "Extracting the mathematical relationships"]
            ),
            SeedPrompt(
                id="parse_006",
                text="Let me identify what we know, what we don't know, and what we need to find.",
                category=PromptCategory.WORD_PROBLEM_PARSING,
                description="Three-category information sorting",
                expected_strength="Complete problem analysis",
                variations=["Known → Unknown → Target", "Given → Missing → Goal"]
            )
        ])
        
        # Logical Breakdown (5 prompts)
        seeds.extend([
            SeedPrompt(
                id="logic_001",
                text="Let me think about this logically and reason through each part.",
                category=PromptCategory.LOGICAL_BREAKDOWN,
                description="Logical reasoning emphasis",
                expected_strength="Clear reasoning chains",
                variations=["I'll approach this logically", "Using logical reasoning here"]
            ),
            SeedPrompt(
                id="logic_002",
                text="If this is true, then what follows? Let me trace the logical connections.",
                category=PromptCategory.LOGICAL_BREAKDOWN,
                description="Conditional reasoning",
                expected_strength="Handles complex conditions",
                variations=["Following the logical chain", "If-then reasoning"]
            ),
            SeedPrompt(
                id="logic_003",
                text="I'll analyze the cause-and-effect relationships in this problem.",
                category=PromptCategory.LOGICAL_BREAKDOWN,
                description="Causal analysis",
                expected_strength="Understanding problem dynamics",
                variations=["Looking at cause and effect", "Analyzing the relationships"]
            ),
            SeedPrompt(
                id="logic_004",
                text="Let me reason through this systematically using logical principles.",
                category=PromptCategory.LOGICAL_BREAKDOWN,
                description="Principled reasoning",
                expected_strength="Systematic logical approach",
                variations=["Applying logical principles", "Using systematic reasoning"]
            ),
            SeedPrompt(
                id="logic_005",
                text="What are the logical constraints and how do they guide the solution?",
                category=PromptCategory.LOGICAL_BREAKDOWN,
                description="Constraint-based reasoning",
                expected_strength="Handles complex constraints",
                variations=["Considering the constraints", "What limits our options?"]
            )
        ])
        
        # Algebraic Approach (5 prompts)
        seeds.extend([
            SeedPrompt(
                id="algebra_001",
                text="Let me define variables and set up equations for this problem.",
                category=PromptCategory.ALGEBRAIC_APPROACH,
                description="Variable definition approach",
                expected_strength="Handles unknown quantities",
                variations=["I'll use variables here", "Setting up with algebra"]
            ),
            SeedPrompt(
                id="algebra_002",
                text="I can solve this algebraically by representing the relationships mathematically.",
                category=PromptCategory.ALGEBRAIC_APPROACH,
                description="Mathematical representation",
                expected_strength="Complex relationship handling",
                variations=["Using algebraic methods", "Mathematical representation works here"]
            ),
            SeedPrompt(
                id="algebra_003",
                text="Let x represent the unknown quantity, then I can write an equation.",
                category=PromptCategory.ALGEBRAIC_APPROACH,
                description="Classic variable introduction",
                expected_strength="Systematic unknown handling",
                variations=["Let x be the unknown", "Using x for the missing value"]
            ),
            SeedPrompt(
                id="algebra_004",
                text="I'll express the relationships as equations and solve systematically.",
                category=PromptCategory.ALGEBRAIC_APPROACH,
                description="Relationship-based equations",
                expected_strength="Multiple relationship handling",
                variations=["Converting relationships to equations", "Systematic equation solving"]
            ),
            SeedPrompt(
                id="algebra_005",
                text="This problem involves ratios and proportions - I'll use algebraic methods.",
                category=PromptCategory.ALGEBRAIC_APPROACH,
                description="Ratio/proportion focus",
                expected_strength="Proportion problem expertise",
                variations=["Using algebra for ratios", "Proportion problems need algebra"]
            )
        ])
        
        # Estimation and Checking (5 prompts)
        seeds.extend([
            SeedPrompt(
                id="estimate_001",
                text="Let me estimate the answer first, then calculate precisely.",
                category=PromptCategory.ESTIMATION_CHECKING,
                description="Estimate-first approach",
                expected_strength="Error detection through estimation",
                variations=["I'll estimate before calculating", "Starting with a rough estimate"]
            ),
            SeedPrompt(
                id="estimate_002",
                text="Does this answer seem reasonable? Let me check by estimating.",
                category=PromptCategory.ESTIMATION_CHECKING,
                description="Reasonableness checking",
                expected_strength="Answer validation",
                variations=["Is this answer reasonable?", "Let me verify this makes sense"]
            ),
            SeedPrompt(
                id="estimate_003",
                text="I'll solve this and then verify my answer using a different method.",
                category=PromptCategory.ESTIMATION_CHECKING,
                description="Multiple method verification",
                expected_strength="High confidence solutions",
                variations=["Solving then double-checking", "Verification with different approach"]
            ),
            SeedPrompt(
                id="estimate_004",
                text="Let me use approximation to check if my detailed calculation is correct.",
                category=PromptCategory.ESTIMATION_CHECKING,
                description="Approximation validation",
                expected_strength="Calculation error detection",
                variations=["Checking with approximation", "Using rough math to verify"]
            ),
            SeedPrompt(
                id="estimate_005",
                text="I'll work backwards from my answer to verify it's correct.",
                category=PromptCategory.ESTIMATION_CHECKING,
                description="Backward verification",
                expected_strength="Strong answer validation",
                variations=["Working backwards to check", "Reverse verification"]
            )
        ])

        # Visual Reasoning (4 prompts)
        seeds.extend([
            SeedPrompt(
                id="visual_001",
                text="Let me visualize this problem to better understand it.",
                category=PromptCategory.VISUAL_REASONING,
                description="Visualization emphasis",
                expected_strength="Better spatial understanding",
                variations=["I'll picture this problem", "Visualizing helps here"]
            ),
            SeedPrompt(
                id="visual_002",
                text="I can imagine this scenario and work through it step by step.",
                category=PromptCategory.VISUAL_REASONING,
                description="Scenario imagination",
                expected_strength="Contextual problem solving",
                variations=["Imagining the situation", "Picturing the scenario"]
            ),
            SeedPrompt(
                id="visual_003",
                text="If I draw this out mentally, I can see the relationships clearly.",
                category=PromptCategory.VISUAL_REASONING,
                description="Mental diagram approach",
                expected_strength="Relationship visualization",
                variations=["Drawing this mentally", "Mental picture helps"]
            ),
            SeedPrompt(
                id="visual_004",
                text="Let me think about this problem in terms of physical objects and arrangements.",
                category=PromptCategory.VISUAL_REASONING,
                description="Physical object thinking",
                expected_strength="Concrete representation",
                variations=["Thinking of physical objects", "Real-world visualization"]
            )
        ])

        # Systematic Organization (4 prompts)
        seeds.extend([
            SeedPrompt(
                id="organize_001",
                text="I'll organize all the information systematically before solving.",
                category=PromptCategory.SYSTEMATIC_ORGANIZATION,
                description="Information organization first",
                expected_strength="Reduced confusion",
                variations=["Organizing information first", "Systematic information layout"]
            ),
            SeedPrompt(
                id="organize_002",
                text="Let me structure my approach clearly and work through it methodically.",
                category=PromptCategory.SYSTEMATIC_ORGANIZATION,
                description="Structured methodology",
                expected_strength="Clear solution process",
                variations=["Structuring my approach", "Methodical organization"]
            ),
            SeedPrompt(
                id="organize_003",
                text="I'll keep track of each piece of information and how it connects.",
                category=PromptCategory.SYSTEMATIC_ORGANIZATION,
                description="Information tracking",
                expected_strength="Complex problem management",
                variations=["Tracking all information", "Managing the connections"]
            ),
            SeedPrompt(
                id="organize_004",
                text="Using clear notation and organization to avoid confusion.",
                category=PromptCategory.SYSTEMATIC_ORGANIZATION,
                description="Clear notation emphasis",
                expected_strength="Error prevention",
                variations=["Clear notation prevents errors", "Organized approach"]
            )
        ])

        # Pattern Recognition (4 prompts)
        seeds.extend([
            SeedPrompt(
                id="pattern_001",
                text="I notice a pattern here that can help solve this more efficiently.",
                category=PromptCategory.PATTERN_RECOGNITION,
                description="Pattern identification",
                expected_strength="Efficient solutions",
                variations=["There's a pattern here", "I see a helpful pattern"]
            ),
            SeedPrompt(
                id="pattern_002",
                text="This problem is similar to others I've seen - I can use that pattern.",
                category=PromptCategory.PATTERN_RECOGNITION,
                description="Problem type recognition",
                expected_strength="Faster problem solving",
                variations=["This follows a familiar pattern", "I recognize this type"]
            ),
            SeedPrompt(
                id="pattern_003",
                text="Looking for patterns in the numbers to find a shortcut.",
                category=PromptCategory.PATTERN_RECOGNITION,
                description="Numerical pattern focus",
                expected_strength="Computational efficiency",
                variations=["Number patterns help", "Finding numerical shortcuts"]
            ),
            SeedPrompt(
                id="pattern_004",
                text="The structure of this problem suggests a particular approach.",
                category=PromptCategory.PATTERN_RECOGNITION,
                description="Structural pattern recognition",
                expected_strength="Appropriate method selection",
                variations=["Problem structure guides approach", "Structure suggests method"]
            )
        ])

        # Multiple Methods (4 prompts)
        seeds.extend([
            SeedPrompt(
                id="multiple_001",
                text="I can solve this in several ways - let me choose the most efficient.",
                category=PromptCategory.MULTIPLE_METHODS,
                description="Method selection",
                expected_strength="Optimal approach choice",
                variations=["Multiple approaches possible", "Choosing the best method"]
            ),
            SeedPrompt(
                id="multiple_002",
                text="Let me try a different approach to double-check my answer.",
                category=PromptCategory.MULTIPLE_METHODS,
                description="Alternative verification",
                expected_strength="Answer confidence",
                variations=["Trying another way", "Different approach for verification"]
            ),
            SeedPrompt(
                id="multiple_003",
                text="I'll compare two different solution methods to see which works better.",
                category=PromptCategory.MULTIPLE_METHODS,
                description="Method comparison",
                expected_strength="Best practice identification",
                variations=["Comparing methods", "Which approach works better?"]
            ),
            SeedPrompt(
                id="multiple_004",
                text="There are multiple valid approaches here - I'll use the clearest one.",
                category=PromptCategory.MULTIPLE_METHODS,
                description="Clarity-based selection",
                expected_strength="Clear communication",
                variations=["Choosing the clearest method", "Multiple valid approaches"]
            )
        ])

        # Conceptual Understanding (5 prompts)
        seeds.extend([
            SeedPrompt(
                id="concept_001",
                text="Let me understand the underlying mathematical concept first.",
                category=PromptCategory.CONCEPTUAL_UNDERSTANDING,
                description="Concept-first approach",
                expected_strength="Deep understanding",
                variations=["Understanding the concept first", "What's the underlying principle?"]
            ),
            SeedPrompt(
                id="concept_002",
                text="This problem involves the concept of [relevant concept] - let me apply it.",
                category=PromptCategory.CONCEPTUAL_UNDERSTANDING,
                description="Concept application",
                expected_strength="Principled problem solving",
                variations=["Applying the relevant concept", "This concept applies here"]
            ),
            SeedPrompt(
                id="concept_003",
                text="Why does this method work? Understanding the reasoning behind it.",
                category=PromptCategory.CONCEPTUAL_UNDERSTANDING,
                description="Method understanding",
                expected_strength="Robust knowledge",
                variations=["Why does this work?", "Understanding the reasoning"]
            ),
            SeedPrompt(
                id="concept_004",
                text="I'll connect this to the fundamental mathematical principles involved.",
                category=PromptCategory.CONCEPTUAL_UNDERSTANDING,
                description="Principle connection",
                expected_strength="Transferable knowledge",
                variations=["Connecting to principles", "What principles apply?"]
            ),
            SeedPrompt(
                id="concept_005",
                text="This problem helps illustrate an important mathematical concept.",
                category=PromptCategory.CONCEPTUAL_UNDERSTANDING,
                description="Concept illustration",
                expected_strength="Educational value",
                variations=["This illustrates a key concept", "Good example of the concept"]
            )
        ])

        return seeds
    
    def get_all_seeds(self) -> List[SeedPrompt]:
        """Get all seed prompts."""
        return self.seeds.copy()
    
    def get_seeds_by_category(self, category: PromptCategory) -> List[SeedPrompt]:
        """Get seeds for a specific category."""
        return [seed for seed in self.seeds if seed.category == category]
    
    def get_seed_texts(self) -> List[str]:
        """Get just the text of all seeds."""
        return [seed.text for seed in self.seeds]
    
    def get_category_distribution(self) -> Dict[PromptCategory, int]:
        """Get actual distribution of seeds across categories."""
        distribution = {}
        for seed in self.seeds:
            distribution[seed.category] = distribution.get(seed.category, 0) + 1
        return distribution
    
    def validate_collection(self) -> Dict[str, Any]:
        """Validate the seed collection for completeness and balance."""
        validation_results = {
            'total_seeds': len(self.seeds),
            'unique_ids': len(set(seed.id for seed in self.seeds)),
            'category_distribution': self.get_category_distribution(),
            'issues': []
        }
        
        # Check for duplicate IDs
        if validation_results['unique_ids'] != validation_results['total_seeds']:
            validation_results['issues'].append("Duplicate seed IDs found")
        
        # Check target distribution
        target_dist = self.category_manager.get_category_distribution_target()
        actual_dist = validation_results['category_distribution']
        
        for category, target_count in target_dist.items():
            actual_count = actual_dist.get(category, 0)
            if actual_count != target_count:
                validation_results['issues'].append(
                    f"{category.value}: expected {target_count}, got {actual_count}"
                )
        
        # Check for empty texts
        empty_texts = [seed.id for seed in self.seeds if not seed.text.strip()]
        if empty_texts:
            validation_results['issues'].append(f"Empty texts: {empty_texts}")
        
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        
        return validation_results


if __name__ == "__main__":
    # Test base seed collection
    print("Testing base seed collection...")
    
    # Create collection
    collection = BaseSeedCollection()
    
    # Basic tests
    all_seeds = collection.get_all_seeds()
    print(f"✅ Total seeds created: {len(all_seeds)}")
    
    # Test category distribution
    distribution = collection.get_category_distribution()
    print("✅ Category distribution:")
    for category, count in distribution.items():
        print(f"   {category.value}: {count}")
    
    # Test validation
    validation = collection.validate_collection()
    print(f"✅ Collection validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    
    if validation['issues']:
        print("   Issues found:")
        for issue in validation['issues']:
            print(f"   - {issue}")
    
    # Test category retrieval
    step_seeds = collection.get_seeds_by_category(PromptCategory.STEP_BY_STEP)
    print(f"✅ Step-by-step seeds: {len(step_seeds)}")
    
    # Show sample seeds
    print("✅ Sample seeds:")
    for i, seed in enumerate(all_seeds[:3]):
        print(f"   {seed.id}: {seed.text}")
    
    print(f"\n🎯 Base seed collection tests completed!")
    print(f"   Status: {'✅ READY' if validation['is_valid'] else '❌ NEEDS FIXES'}")
