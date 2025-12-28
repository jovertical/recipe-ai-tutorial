# Problem 3: Rule-Based Substitution
#
# Implement a rule-based substitution system. This serves as a baseline and
# can handle cases where we have expert knowledge.
#
# Example:
#   rules = SubstitutionRules()
#   rules.add_rule("butter", "margarine", ratio="1:1", conditions=["baking"])
#   result = rules.substitute("butter", context="baking")
#   # Returns: [("margarine", 1.0, "1:1")]
#
# ML Relevance: Rule-based systems provide interpretable baselines, handle
# known edge cases precisely, can be combined with ML models, and are useful
# when training data is limited. Understanding rules helps design better
# features for ML.
#
# Your Task:
#   1. Implement SubstitutionRule dataclass
#   2. Implement SubstitutionRuleEngine with add/match/apply
#   3. Implement rule conflict resolution
#   4. Implement rule chaining for multi-step substitutions

from typing import List, Dict, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import re


class RulePriority(Enum):
    """Priority levels for rule matching."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    OVERRIDE = 4  # Always takes precedence


@dataclass
class SubstitutionRule:
    """A single substitution rule."""
    original: str
    substitute: str
    ratio: str = "1:1"
    contexts: List[str] = field(default_factory=list)
    dietary_tags: List[str] = field(default_factory=list)
    priority: RulePriority = RulePriority.MEDIUM
    conditions: Optional[Callable] = None
    notes: str = ""
    
    def matches(self, ingredient: str, context: str = None) -> bool:
        """
        Check if rule matches the given ingredient and context.
        
        Hints:
        1. Check if ingredient matches (exact or pattern)
        2. If contexts specified, check context matches
        3. If conditions callable, evaluate it
        """
        # Your solution here
        pass
    
    def parse_ratio(self) -> Tuple[float, float]:
        """
        Parse ratio string to numeric values.
        
        Returns tuple of (substitute_amount, original_amount).
        Example: "0.75:1" -> (0.75, 1.0)
        """
        # Your solution here
        pass


def create_rule(
    original: str,
    substitute: str,
    ratio: str = "1:1",
    contexts: List[str] = None,
    dietary_tags: List[str] = None,
    priority: RulePriority = RulePriority.MEDIUM,
    notes: str = ""
) -> SubstitutionRule:
    """Factory function to create a substitution rule."""
    # Your solution here
    pass


class SubstitutionRuleEngine:
    """Engine for managing and applying substitution rules."""
    
    def __init__(self):
        """Initialize empty rule engine."""
        # Store rules in a structure that allows efficient lookup
        # Your solution here
        pass
    
    def add_rule(self, rule: SubstitutionRule):
        """Add a rule to the engine."""
        # Your solution here
        pass
    
    def add_rules(self, rules: List[SubstitutionRule]):
        """Add multiple rules."""
        # Your solution here
        pass
    
    def remove_rule(self, original: str, substitute: str) -> bool:
        """Remove a specific rule. Returns True if rule was removed."""
        # Your solution here
        pass
    
    def find_rules(
        self,
        ingredient: str,
        context: str = None,
        dietary_tags: List[str] = None
    ) -> List[SubstitutionRule]:
        """
        Find all rules that match the given criteria.
        
        Hints:
        1. Find all rules where original matches ingredient
        2. Filter by context if provided
        3. Filter by dietary_tags if provided
        4. Sort by priority (highest first)
        """
        # Your solution here
        pass
    
    def get_substitutes(
        self,
        ingredient: str,
        context: str = None,
        dietary_tags: List[str] = None,
        top_k: int = 5
    ) -> List[Tuple[str, str, RulePriority]]:
        """Get top substitutes for an ingredient as (substitute, ratio, priority) tuples."""
        # Your solution here
        pass
    
    def apply_substitution(
        self,
        ingredient: str,
        amount: float,
        unit: str,
        context: str = None,
        dietary_tags: List[str] = None
    ) -> Optional[Tuple[str, float, str]]:
        """
        Apply best substitution rule with amount conversion.
        Returns tuple of (substitute, new_amount, unit) or None.
        """
        # Your solution here
        pass
    
    def get_rule_count(self) -> int:
        """Return total number of rules."""
        # Your solution here
        pass
    
    def get_covered_ingredients(self) -> Set[str]:
        """Return set of ingredients with rules."""
        # Your solution here
        pass


def resolve_conflicts(rules: List[SubstitutionRule]) -> SubstitutionRule:
    """
    Resolve conflicts when multiple rules match.
    
    Resolution order:
    1. Highest priority wins
    2. If tied, most specific context wins
    3. If tied, first added wins
    """
    # Your solution here
    pass


class RuleChain:
    """
    Chain of substitution rules for multi-step substitutions.
    Example: cream -> coconut_cream -> coconut_milk
    """
    
    def __init__(self, engine: SubstitutionRuleEngine):
        """Initialize with rule engine."""
        # Your solution here
        pass
    
    def find_chain(
        self,
        ingredient: str,
        max_depth: int = 3,
        available_ingredients: Set[str] = None
    ) -> List[List[str]]:
        """
        Find substitution chains.
        Returns list of substitution chains (each chain is list of ingredients).
        """
        # Your solution here
        pass
    
    def shortest_path(
        self,
        original: str,
        target: str
    ) -> Optional[List[str]]:
        """Find shortest substitution path between two ingredients."""
        # Your solution here
        pass


def load_default_rules() -> SubstitutionRuleEngine:
    """Load a default set of common substitution rules."""
    engine = SubstitutionRuleEngine()
    
    # Fat substitutions
    engine.add_rules([
        create_rule("butter", "margarine", "1:1", ["baking", "spreading"], 
                   ["vegan", "dairy_free"]),
        create_rule("butter", "coconut_oil", "1:1", ["baking"],
                   ["vegan", "dairy_free"], notes="Use refined for neutral flavor"),
        create_rule("butter", "applesauce", "0.5:1", ["baking"],
                   ["vegan", "dairy_free", "low_fat"], notes="Reduces fat, adds moisture"),
        create_rule("butter", "greek_yogurt", "0.5:1", ["baking"],
                   ["vegetarian", "low_fat"]),
        create_rule("butter", "avocado", "1:1", ["spreading", "baking"],
                   ["vegan", "dairy_free"]),
    ])
    
    # Egg substitutions
    engine.add_rules([
        create_rule("egg", "flax_egg", "1:1", ["baking"],
                   ["vegan"], notes="1 tbsp flax + 3 tbsp water per egg"),
        create_rule("egg", "chia_egg", "1:1", ["baking"],
                   ["vegan"], notes="1 tbsp chia + 3 tbsp water per egg"),
        create_rule("egg", "applesauce", "0.25:1", ["baking"],
                   ["vegan"], notes="1/4 cup per egg, adds sweetness"),
        create_rule("egg", "banana", "0.5:1", ["baking"],
                   ["vegan"], notes="1/2 mashed banana per egg"),
        create_rule("egg", "aquafaba", "3:1", ["baking", "meringue"],
                   ["vegan"], notes="3 tbsp per egg, whips like egg whites"),
    ])
    
    # Milk substitutions
    engine.add_rules([
        create_rule("milk", "oat_milk", "1:1", [],
                   ["vegan", "dairy_free", "nut_free"]),
        create_rule("milk", "almond_milk", "1:1", [],
                   ["vegan", "dairy_free"]),
        create_rule("milk", "soy_milk", "1:1", [],
                   ["vegan", "dairy_free", "nut_free"]),
        create_rule("milk", "coconut_milk", "1:1", [],
                   ["vegan", "dairy_free"], notes="May add coconut flavor"),
    ])
    
    # Sweetener substitutions
    engine.add_rules([
        create_rule("sugar", "honey", "0.75:1", ["baking", "sweetening"],
                   ["vegetarian"], notes="Reduce liquid slightly"),
        create_rule("sugar", "maple_syrup", "0.75:1", ["baking", "sweetening"],
                   ["vegan"], notes="Reduce liquid slightly"),
        create_rule("sugar", "coconut_sugar", "1:1", ["baking"],
                   ["vegan"], notes="Lower glycemic index"),
        create_rule("sugar", "stevia", "0.004:1", ["sweetening"],
                   ["vegan", "low_calorie"], notes="Very concentrated"),
    ])
    
    # Flour substitutions
    engine.add_rules([
        create_rule("flour", "almond_flour", "1:1", ["baking"],
                   ["gluten_free"], notes="May need more eggs for binding"),
        create_rule("flour", "oat_flour", "1:1", ["baking"],
                   ["gluten_free"], notes="Make from rolled oats in blender"),
        create_rule("flour", "coconut_flour", "0.25:1", ["baking"],
                   ["gluten_free"], notes="Very absorbent, needs more liquid"),
    ])
    
    return engine


def validate_rules(engine: SubstitutionRuleEngine) -> Dict[str, List[str]]:
    """
    Validate rules for consistency and completeness.
    
    Returns dictionary of issues found:
    - missing_reverse: A->B exists but not B->A
    - conflicting_ratios: Same pair has different ratios
    - circular: Circular substitution chains
    """
    # Your solution here
    pass


def export_rules(engine: SubstitutionRuleEngine, format: str = "json") -> str:
    """Export rules to string format ("json" or "yaml")."""
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing SubstitutionRule...")
    
    # Test 1: Basic rule creation
    rule = create_rule("butter", "margarine", "1:1", ["baking"], ["vegan"])
    assert rule.original == "butter", "Test 1a failed"
    assert rule.substitute == "margarine", "Test 1b failed"
    assert "baking" in rule.contexts, "Test 1c failed"
    print("  ✓ Rule creation works")
    
    # Test 2: Rule matching
    assert rule.matches("butter", "baking"), "Test 2a failed"
    assert not rule.matches("butter", "frying"), "Test 2b failed: context mismatch"
    assert not rule.matches("oil", "baking"), "Test 2c failed: ingredient mismatch"
    print("  ✓ Rule matching works")
    
    # Test 3: Parse ratio
    rule = create_rule("sugar", "honey", "0.75:1")
    sub_amt, orig_amt = rule.parse_ratio()
    assert sub_amt == 0.75, f"Test 3a failed: {sub_amt}"
    assert orig_amt == 1.0, f"Test 3b failed: {orig_amt}"
    print("  ✓ Ratio parsing works")
    
    print("\nTesting SubstitutionRuleEngine...")
    
    # Test 4: Add rules
    engine = SubstitutionRuleEngine()
    engine.add_rule(create_rule("butter", "margarine", "1:1", ["baking"]))
    engine.add_rule(create_rule("butter", "coconut_oil", "1:1", ["baking"]))
    engine.add_rule(create_rule("butter", "oil", "0.75:1", ["frying"]))
    assert engine.get_rule_count() == 3, f"Test 4 failed: {engine.get_rule_count()}"
    print(f"  ✓ Added {engine.get_rule_count()} rules")
    
    # Test 5: Find rules by ingredient
    rules = engine.find_rules("butter")
    assert len(rules) == 3, f"Test 5 failed: {len(rules)}"
    print(f"  ✓ Found {len(rules)} rules for butter")
    
    # Test 6: Find rules by context
    rules = engine.find_rules("butter", context="baking")
    assert len(rules) == 2, f"Test 6 failed: {len(rules)}"
    print(f"  ✓ Found {len(rules)} baking rules for butter")
    
    # Test 7: Get substitutes
    subs = engine.get_substitutes("butter", context="baking")
    assert len(subs) == 2, f"Test 7a failed: {len(subs)}"
    sub_names = [s[0] for s in subs]
    assert "margarine" in sub_names, "Test 7b failed"
    print(f"  ✓ Got {len(subs)} substitutes for butter (baking)")
    
    # Test 8: Apply substitution
    result = engine.apply_substitution("butter", 1.0, "cup", context="baking")
    assert result is not None, "Test 8a failed"
    sub, amount, unit = result
    assert sub in ["margarine", "coconut_oil"], f"Test 8b failed: {sub}"
    assert amount == 1.0, f"Test 8c failed: {amount}"
    print(f"  ✓ Applied: 1 cup butter -> {amount} {unit} {sub}")
    
    # Test 9: Covered ingredients
    covered = engine.get_covered_ingredients()
    assert "butter" in covered, "Test 9 failed"
    print(f"  ✓ {len(covered)} ingredients covered")
    
    print("\nTesting load_default_rules...")
    
    # Test 10: Load defaults
    engine = load_default_rules()
    assert engine.get_rule_count() > 10, f"Test 10 failed: {engine.get_rule_count()}"
    print(f"  ✓ Loaded {engine.get_rule_count()} default rules")
    
    # Test 11: Vegan butter substitute
    subs = engine.get_substitutes("butter", dietary_tags=["vegan"])
    assert len(subs) > 0, "Test 11a failed"
    print(f"  ✓ Found {len(subs)} vegan butter substitutes")
    
    # Test 12: Egg substitutes
    subs = engine.get_substitutes("egg", context="baking", dietary_tags=["vegan"])
    assert len(subs) > 0, "Test 12 failed"
    print(f"  ✓ Found {len(subs)} vegan egg substitutes for baking")
    
    print("\nTesting resolve_conflicts...")
    
    # Test 13: Priority resolution
    low_priority = create_rule("x", "y", priority=RulePriority.LOW)
    high_priority = create_rule("x", "z", priority=RulePriority.HIGH)
    winner = resolve_conflicts([low_priority, high_priority])
    assert winner.substitute == "z", "Test 13 failed: high priority should win"
    print("  ✓ High priority wins")
    
    print("\nTesting RuleChain...")
    
    # Test 14: Chain initialization
    engine = SubstitutionRuleEngine()
    engine.add_rule(create_rule("cream", "coconut_cream"))
    engine.add_rule(create_rule("coconut_cream", "coconut_milk"))
    chain = RuleChain(engine)
    print("  ✓ RuleChain initialized")
    
    # Test 15: Find chain
    chains = chain.find_chain("cream", max_depth=3)
    assert len(chains) > 0, "Test 15a failed"
    print(f"  ✓ Found {len(chains)} substitution chain(s)")
    
    # Test 16: Shortest path
    path = chain.shortest_path("cream", "coconut_milk")
    assert path is not None, "Test 16a failed"
    assert path[0] == "cream", "Test 16b failed"
    assert path[-1] == "coconut_milk", "Test 16c failed"
    print(f"  ✓ Shortest path: {' -> '.join(path)}")
    
    print("\nTesting remove_rule...")
    
    # Test 17: Remove rule
    engine = SubstitutionRuleEngine()
    engine.add_rule(create_rule("a", "b"))
    engine.add_rule(create_rule("a", "c"))
    removed = engine.remove_rule("a", "b")
    assert removed, "Test 17a failed"
    assert engine.get_rule_count() == 1, f"Test 17b failed: {engine.get_rule_count()}"
    print("  ✓ Rule removal works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Rules provide explicit, interpretable substitutions")
    print("2. Context and dietary filters narrow options")
    print("3. Priority resolves conflicts")
    print("4. Chains enable multi-step substitutions")
    print("\nNext: Embedding-based retrieval for learned substitutions!")
