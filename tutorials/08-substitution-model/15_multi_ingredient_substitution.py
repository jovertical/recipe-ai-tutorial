# Problem 15: Multi-Ingredient Substitution
#
# Handle substituting multiple ingredients simultaneously while maintaining
# recipe coherence.
#
# Example:
#   substitutor = MultiSubstitutor(model)
#   subs = substitutor.substitute_all(["butter", "milk", "eggs"], dietary="vegan")
#   # Returns coordinated substitutions that work together
#
# ML Relevance: Multi-item optimization requires considering interactions
# between substitutions, not just individual replacements.
#
# Your Task:
#   1. Implement joint substitution scoring
#   2. Implement ingredient interaction modeling
#   3. Implement beam search for best combinations
#   4. Handle substitution conflicts


from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from itertools import combinations, product
import random


@dataclass
class SubstitutionPlan:
    """A plan for substituting one or more ingredients."""
    original: List[str]
    substitutes: List[str]
    score: float
    functions_covered: Set[str] = field(default_factory=set)
    functions_missing: Set[str] = field(default_factory=set)
    dietary_satisfied: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    
    def __repr__(self):
        orig = ", ".join(self.original)
        subs = ", ".join(self.substitutes)
        return f"[{orig}] -> [{subs}] (score={self.score:.2f})"


class OneToManySubstitutor:
    """
    Handle cases where one ingredient requires multiple substitutes.
    
    Example: egg -> flax_egg (binding) + baking_powder (leavening)
    """
    
    def __init__(
        self,
        ingredient_functions: Dict[str, Set[str]],
        embeddings: Dict[str, np.ndarray] = None
    ):
        """
        Initialize substitutor.
        
        Args:
            ingredient_functions: Ingredient -> set of functions
            embeddings: Optional ingredient embeddings
        """
        self.functions = ingredient_functions
        self.embeddings = embeddings or {}
    
    def substitute(
        self,
        ingredient: str,
        available_substitutes: List[str],
        max_substitutes: int = 3
    ) -> List[SubstitutionPlan]:
        """
        Find substitute combinations that cover all functions.
        
        Args:
            ingredient: Ingredient to replace
            available_substitutes: Pool of possible substitutes
            max_substitutes: Maximum ingredients in combination
            
        Returns:
            List of substitution plans sorted by score
        """
        # Your solution here
        pass
    
    def find_function_coverage(
        self,
        target_functions: Set[str],
        candidate_combo: List[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Find which functions are covered and missing."""
        # Your solution here
        pass
    
    def score_combination(
        self,
        original: str,
        substitutes: List[str]
    ) -> float:
        """Score a substitution combination."""
        # Your solution here
        pass


class ManyToOneSubstitutor:
    """
    Simplify by replacing multiple ingredients with one.
    
    Example: butter + egg -> avocado (in some contexts)
    """
    
    def __init__(
        self,
        ingredient_functions: Dict[str, Set[str]],
        embeddings: Dict[str, np.ndarray] = None
    ):
        self.functions = ingredient_functions
        self.embeddings = embeddings or {}
    
    def substitute(
        self,
        ingredients: List[str],
        available_substitutes: List[str],
        min_function_coverage: float = 0.8
    ) -> List[SubstitutionPlan]:
        """
        Find single substitutes that cover multiple ingredients.
        
        Args:
            ingredients: Ingredients to replace
            available_substitutes: Pool of possible substitutes
            min_function_coverage: Minimum fraction of functions covered
            
        Returns:
            List of substitution plans
        """
        # Your solution here
        pass


class CoherentMultiSubstitutor:
    """
    Find coherent substitutions for multiple ingredients.
    
    Ensures substitutions work well together.
    """
    
    def __init__(
        self,
        ingredient_functions: Dict[str, Set[str]],
        compatibility_scores: Dict[Tuple[str, str], float] = None,
        embeddings: Dict[str, np.ndarray] = None
    ):
        self.functions = ingredient_functions
        self.compatibility = compatibility_scores or {}
        self.embeddings = embeddings or {}
    
    def substitute_all(
        self,
        ingredients: List[str],
        substitute_options: Dict[str, List[str]],
        top_k: int = 5
    ) -> List[Dict[str, str]]:
        """
        Find coherent substitutions for all ingredients.
        
        Args:
            ingredients: List of ingredients to substitute
            substitute_options: Ingredient -> list of options
            top_k: Number of combinations to return
            
        Returns:
            List of substitution mappings {original: substitute}
        """
        # Your solution here
        pass
    
    def score_coherence(
        self,
        substitution_map: Dict[str, str]
    ) -> float:
        """Score how well substitutes work together."""
        # Your solution here
        pass
    
    def check_conflicts(
        self,
        substitution_map: Dict[str, str]
    ) -> List[str]:
        """Check for conflicts between substitutes."""
        # Your solution here
        pass


class RecipeTransformer:
    """
    Transform entire recipes to meet dietary requirements.
    """
    
    def __init__(
        self,
        dietary_db: Dict[str, Dict],
        substitution_rules: Dict[str, Dict[str, List[str]]],
        embeddings: Dict[str, np.ndarray] = None
    ):
        """
        Initialize transformer.
        
        Args:
            dietary_db: Dietary info for each ingredient
            substitution_rules: Diet -> ingredient -> substitutes
            embeddings: Ingredient embeddings
        """
        self.dietary_db = dietary_db
        self.rules = substitution_rules
        self.embeddings = embeddings or {}
    
    def veganize(
        self,
        ingredients: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Transform recipe to be vegan.
        
        Returns:
            Tuple of (new_ingredients, substitution_map)
        """
        # Your solution here
        pass
    
    def make_gluten_free(
        self,
        ingredients: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Transform recipe to be gluten-free."""
        # Your solution here
        pass
    
    def transform(
        self,
        ingredients: List[str],
        target_diet: str
    ) -> Dict[str, any]:
        """
        General diet transformation.
        
        Args:
            ingredients: Recipe ingredients
            target_diet: "vegan", "vegetarian", "gluten_free", etc.
            
        Returns:
            Transformation result with new ingredients and details
        """
        # Your solution here
        pass
    
    def identify_problematic_ingredients(
        self,
        ingredients: List[str],
        target_diet: str
    ) -> List[str]:
        """Identify ingredients that need substitution."""
        # Your solution here
        pass


class SubstitutionOptimizer:
    """
    Optimize substitution combinations using various strategies.
    """
    
    def __init__(
        self,
        scorer,  # Function that scores a substitution plan
        constraints: List[callable] = None
    ):
        self.scorer = scorer
        self.constraints = constraints or []
    
    def greedy_optimize(
        self,
        ingredients: List[str],
        candidates: Dict[str, List[str]],
        max_iterations: int = 100
    ) -> Dict[str, str]:
        """Greedy optimization of substitutions."""
        # Your solution here
        pass
    
    def beam_search(
        self,
        ingredients: List[str],
        candidates: Dict[str, List[str]],
        beam_width: int = 5
    ) -> List[Dict[str, str]]:
        """Beam search for top substitution combinations."""
        # Your solution here
        pass
    
    def constraint_satisfaction(
        self,
        ingredients: List[str],
        candidates: Dict[str, List[str]],
        hard_constraints: List[callable],
        soft_constraints: List[Tuple[callable, float]]  # (constraint, weight)
    ) -> Optional[Dict[str, str]]:
        """Find substitution satisfying constraints."""
        # Your solution here
        pass


def generate_substitution_combinations(
    ingredients: List[str],
    candidates: Dict[str, List[str]],
    max_combinations: int = 1000
) -> List[Dict[str, str]]:
    """Generate possible substitution combinations."""
    # Your solution here
    pass


def evaluate_recipe_transformation(
    original_ingredients: List[str],
    transformed_ingredients: List[str],
    substitution_map: Dict[str, str]
) -> Dict[str, float]:
    """
    Evaluate quality of a recipe transformation.
    
    Returns metrics like similarity, function coverage, etc.
    """
    # Your solution here
    pass


# Sample data for testing
SAMPLE_FUNCTIONS = {
    "butter": {"fat", "moisture", "flavor", "tenderizing"},
    "egg": {"binding", "leavening", "moisture", "emulsifying"},
    "milk": {"moisture", "fat", "protein"},
    "flour": {"structure", "thickening"},
    "sugar": {"sweetening", "moisture", "browning"},
    "margarine": {"fat", "moisture", "tenderizing"},
    "coconut_oil": {"fat", "moisture"},
    "flax_egg": {"binding", "moisture"},
    "baking_powder": {"leavening"},
    "aquafaba": {"emulsifying", "leavening"},
    "oat_milk": {"moisture"},
    "almond_milk": {"moisture"},
    "applesauce": {"moisture", "sweetening", "binding"},
    "banana": {"binding", "moisture", "sweetening"},
    "avocado": {"fat", "moisture", "binding"},
}

SAMPLE_DIETARY = {
    "butter": {"vegan": False, "vegetarian": True, "dairy_free": False},
    "egg": {"vegan": False, "vegetarian": True, "dairy_free": True},
    "milk": {"vegan": False, "vegetarian": True, "dairy_free": False},
    "margarine": {"vegan": True, "vegetarian": True, "dairy_free": True},
    "coconut_oil": {"vegan": True, "vegetarian": True, "dairy_free": True},
    "flax_egg": {"vegan": True, "vegetarian": True, "dairy_free": True},
    "oat_milk": {"vegan": True, "vegetarian": True, "dairy_free": True},
}


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    # Create mock embeddings
    embeddings = {name: np.random.randn(64).astype(np.float32) 
                  for name in SAMPLE_FUNCTIONS.keys()}
    
    print("Testing OneToManySubstitutor...")
    
    one_to_many = OneToManySubstitutor(SAMPLE_FUNCTIONS, embeddings)
    
    # Test 1: Substitute egg (needs binding + leavening)
    candidates = ["flax_egg", "baking_powder", "aquafaba", "applesauce", "banana"]
    plans = one_to_many.substitute("egg", candidates, max_substitutes=2)
    
    assert len(plans) > 0, "Test 1a failed"
    # Check that some plan covers binding and leavening
    found_complete = False
    for plan in plans:
        covered = plan.functions_covered
        if "binding" in covered and "leavening" in covered:
            found_complete = True
            break
    print(f"  ✓ Found {len(plans)} substitution plans for egg")
    print(f"    Best: {plans[0] if plans else 'None'}")
    
    # Test 2: Function coverage
    target = {"binding", "leavening", "moisture"}
    covered, missing = one_to_many.find_function_coverage(
        target, ["flax_egg", "baking_powder"]
    )
    assert "binding" in covered, "Test 2a failed"
    assert "leavening" in covered, "Test 2b failed"
    print(f"  ✓ Function coverage: {covered}, missing: {missing}")
    
    print("\nTesting ManyToOneSubstitutor...")
    
    many_to_one = ManyToOneSubstitutor(SAMPLE_FUNCTIONS, embeddings)
    
    # Test 3: Replace butter + egg with single ingredient
    plans = many_to_one.substitute(
        ["butter", "egg"],
        ["avocado", "coconut_oil", "applesauce"],
        min_function_coverage=0.5
    )
    assert len(plans) >= 0, "Test 3 failed"  # May not find perfect match
    print(f"  ✓ Found {len(plans)} simplification options")
    if plans:
        print(f"    Best: {plans[0]}")
    
    print("\nTesting CoherentMultiSubstitutor...")
    
    coherent = CoherentMultiSubstitutor(SAMPLE_FUNCTIONS, {}, embeddings)
    
    # Test 4: Substitute multiple ingredients coherently
    sub_options = {
        "butter": ["margarine", "coconut_oil"],
        "milk": ["oat_milk", "almond_milk"],
        "egg": ["flax_egg", "applesauce"]
    }
    combinations = coherent.substitute_all(
        ["butter", "milk", "egg"],
        sub_options,
        top_k=3
    )
    assert len(combinations) > 0, "Test 4a failed"
    assert all(len(c) == 3 for c in combinations), "Test 4b failed"
    print(f"  ✓ Found {len(combinations)} coherent combinations")
    print(f"    Best: {combinations[0]}")
    
    # Test 5: Coherence scoring
    test_combo = {"butter": "margarine", "milk": "oat_milk", "egg": "flax_egg"}
    score = coherent.score_coherence(test_combo)
    assert 0 <= score <= 1, f"Test 5 failed: {score}"
    print(f"  ✓ Coherence score: {score:.2f}")
    
    print("\nTesting RecipeTransformer...")
    
    substitution_rules = {
        "vegan": {
            "butter": ["margarine", "coconut_oil"],
            "egg": ["flax_egg", "aquafaba"],
            "milk": ["oat_milk", "almond_milk"],
        }
    }
    
    transformer = RecipeTransformer(SAMPLE_DIETARY, substitution_rules, embeddings)
    
    # Test 6: Veganize recipe
    recipe = ["flour", "butter", "sugar", "egg", "milk"]
    new_recipe, sub_map = transformer.veganize(recipe)
    
    assert "flour" in new_recipe, "Test 6a failed"  # Should keep flour
    assert "sugar" in new_recipe, "Test 6b failed"  # Should keep sugar
    assert "butter" not in new_recipe or sub_map.get("butter"), "Test 6c failed"
    print(f"  ✓ Veganized: {new_recipe}")
    print(f"    Substitutions: {sub_map}")
    
    # Test 7: Identify problematic ingredients
    problematic = transformer.identify_problematic_ingredients(recipe, "vegan")
    assert "butter" in problematic, "Test 7a failed"
    assert "egg" in problematic, "Test 7b failed"
    assert "flour" not in problematic, "Test 7c failed"
    print(f"  ✓ Problematic for vegan: {problematic}")
    
    # Test 8: General transform
    result = transformer.transform(recipe, "vegan")
    assert "new_ingredients" in result, "Test 8a failed"
    assert "substitution_map" in result, "Test 8b failed"
    print(f"  ✓ Transform result: {result}")
    
    print("\nTesting SubstitutionOptimizer...")
    
    def dummy_scorer(plan):
        return random.random()
    
    optimizer = SubstitutionOptimizer(dummy_scorer)
    
    # Test 9: Greedy optimization
    candidates = {
        "butter": ["margarine", "coconut_oil"],
        "egg": ["flax_egg", "applesauce"],
    }
    result = optimizer.greedy_optimize(["butter", "egg"], candidates)
    assert len(result) == 2, f"Test 9 failed: {result}"
    print(f"  ✓ Greedy result: {result}")
    
    # Test 10: Beam search
    results = optimizer.beam_search(["butter", "egg"], candidates, beam_width=3)
    assert len(results) <= 3, f"Test 10 failed: {len(results)}"
    print(f"  ✓ Beam search found {len(results)} solutions")
    
    print("\nTesting generate_substitution_combinations...")
    
    # Test 11: Generate combinations
    combos = generate_substitution_combinations(
        ["butter", "egg"],
        {"butter": ["a", "b"], "egg": ["c", "d"]},
        max_combinations=10
    )
    assert len(combos) == 4, f"Test 11 failed: {len(combos)}"  # 2 * 2 = 4
    print(f"  ✓ Generated {len(combos)} combinations")
    
    print("\nTesting evaluate_recipe_transformation...")
    
    # Test 12: Evaluate transformation
    metrics = evaluate_recipe_transformation(
        ["butter", "egg", "flour"],
        ["margarine", "flax_egg", "flour"],
        {"butter": "margarine", "egg": "flax_egg"}
    )
    assert "similarity" in metrics or len(metrics) > 0, "Test 12 failed"
    print(f"  ✓ Transformation metrics: {metrics}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. One ingredient may need multiple substitutes")
    print("2. Multiple ingredients may simplify to one")
    print("3. Substitutions should be coherent together")
    print("4. Recipe transformation requires global optimization")
    print("\nNext: Generating substitution explanations!")
