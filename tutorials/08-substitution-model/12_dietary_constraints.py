# Problem 12: Dietary Constraint Handling
#
# Build a substitution system that respects dietary constraints like vegan,
# gluten-free, nut-free, etc.
#
# Example:
#   substitutor = DietarySubstitutor(embeddings, constraints)
#   subs = substitutor.find("butter", dietary=["vegan", "dairy_free"])
#   # Returns: ["coconut_oil", "margarine", "avocado"]
#
# ML Relevance: Constraint satisfaction combines ML predictions with hard
# rules - a common pattern in production recommendation systems.
#
# Your Task:
#   1. Implement dietary tag filtering
#   2. Implement constraint-aware scoring
#   3. Implement multi-constraint satisfaction
#   4. Handle constraint conflicts gracefully


from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DietaryRestriction(Enum):
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    SOY_FREE = "soy_free"
    EGG_FREE = "egg_free"
    KOSHER = "kosher"
    HALAL = "halal"
    LOW_SODIUM = "low_sodium"
    LOW_FAT = "low_fat"
    LOW_SUGAR = "low_sugar"
    KETO = "keto"
    PALEO = "paleo"


class Allergen(Enum):
    MILK = "milk"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH = "shellfish"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    WHEAT = "wheat"
    SOY = "soy"
    SESAME = "sesame"


@dataclass
class IngredientDietaryProfile:
    """Dietary profile of an ingredient."""
    name: str
    restrictions_satisfied: Set[DietaryRestriction] = field(default_factory=set)
    allergens: Set[Allergen] = field(default_factory=set)
    is_animal_product: bool = False
    nutritional_info: Dict[str, float] = field(default_factory=dict)


# Sample dietary database
DIETARY_DB = {
    "butter": IngredientDietaryProfile(
        name="butter",
        restrictions_satisfied={DietaryRestriction.VEGETARIAN, DietaryRestriction.GLUTEN_FREE},
        allergens={Allergen.MILK},
        is_animal_product=True,
        nutritional_info={"fat": 81, "sodium": 11, "sugar": 0.1}
    ),
    "margarine": IngredientDietaryProfile(
        name="margarine",
        restrictions_satisfied={DietaryRestriction.VEGAN, DietaryRestriction.VEGETARIAN, 
                               DietaryRestriction.DAIRY_FREE, DietaryRestriction.GLUTEN_FREE},
        allergens=set(),
        is_animal_product=False,
        nutritional_info={"fat": 80, "sodium": 943, "sugar": 0}
    ),
    "coconut_oil": IngredientDietaryProfile(
        name="coconut_oil",
        restrictions_satisfied={DietaryRestriction.VEGAN, DietaryRestriction.VEGETARIAN,
                               DietaryRestriction.DAIRY_FREE, DietaryRestriction.GLUTEN_FREE,
                               DietaryRestriction.NUT_FREE, DietaryRestriction.SOY_FREE,
                               DietaryRestriction.KETO, DietaryRestriction.PALEO},
        allergens=set(),
        is_animal_product=False,
        nutritional_info={"fat": 100, "sodium": 0, "sugar": 0}
    ),
    "almond_milk": IngredientDietaryProfile(
        name="almond_milk",
        restrictions_satisfied={DietaryRestriction.VEGAN, DietaryRestriction.VEGETARIAN,
                               DietaryRestriction.DAIRY_FREE, DietaryRestriction.GLUTEN_FREE,
                               DietaryRestriction.SOY_FREE, DietaryRestriction.LOW_FAT},
        allergens={Allergen.TREE_NUTS},
        is_animal_product=False,
        nutritional_info={"fat": 2.5, "sodium": 150, "sugar": 0}
    ),
    "oat_milk": IngredientDietaryProfile(
        name="oat_milk",
        restrictions_satisfied={DietaryRestriction.VEGAN, DietaryRestriction.VEGETARIAN,
                               DietaryRestriction.DAIRY_FREE, DietaryRestriction.NUT_FREE,
                               DietaryRestriction.SOY_FREE},
        allergens=set(),  # Note: may contain gluten from oat processing
        is_animal_product=False,
        nutritional_info={"fat": 5, "sodium": 100, "sugar": 7}
    ),
    "egg": IngredientDietaryProfile(
        name="egg",
        restrictions_satisfied={DietaryRestriction.VEGETARIAN, DietaryRestriction.GLUTEN_FREE,
                               DietaryRestriction.DAIRY_FREE, DietaryRestriction.NUT_FREE},
        allergens={Allergen.EGGS},
        is_animal_product=True,
        nutritional_info={"fat": 11, "sodium": 124, "sugar": 1}
    ),
    "flax_egg": IngredientDietaryProfile(
        name="flax_egg",
        restrictions_satisfied={DietaryRestriction.VEGAN, DietaryRestriction.VEGETARIAN,
                               DietaryRestriction.GLUTEN_FREE, DietaryRestriction.DAIRY_FREE,
                               DietaryRestriction.NUT_FREE, DietaryRestriction.SOY_FREE,
                               DietaryRestriction.EGG_FREE},
        allergens=set(),
        is_animal_product=False,
        nutritional_info={"fat": 3, "sodium": 2, "sugar": 0}
    ),
}


class ConstraintChecker:
    """Check if ingredients satisfy dietary constraints."""
    
    def __init__(self, dietary_db: Dict[str, IngredientDietaryProfile] = None):
        self.db = dietary_db or DIETARY_DB
    
    def satisfies_restriction(
        self,
        ingredient: str,
        restriction: DietaryRestriction
    ) -> bool:
        """Check if ingredient satisfies a single restriction."""
        # Your solution here
        pass
    
    def satisfies_all_restrictions(
        self,
        ingredient: str,
        restrictions: List[DietaryRestriction]
    ) -> bool:
        """Check if ingredient satisfies all restrictions."""
        # Your solution here
        pass
    
    def get_allergens(self, ingredient: str) -> Set[Allergen]:
        """Get allergens present in ingredient."""
        # Your solution here
        pass
    
    def is_safe_for_allergies(
        self,
        ingredient: str,
        allergens_to_avoid: List[Allergen]
    ) -> bool:
        """Check if ingredient is safe for someone with given allergies."""
        # Your solution here
        pass
    
    def get_violations(
        self,
        ingredient: str,
        restrictions: List[DietaryRestriction],
        allergens_to_avoid: List[Allergen] = None
    ) -> Dict[str, List]:
        """Get all constraint violations for an ingredient."""
        # Your solution here
        pass


class ConstraintAwareRetriever:
    """Retrieve substitutes with constraint awareness."""
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        constraint_checker: ConstraintChecker
    ):
        self.embeddings = embeddings
        self.checker = constraint_checker
    
    def find_substitutes(
        self,
        ingredient: str,
        restrictions: List[DietaryRestriction] = None,
        allergens_to_avoid: List[Allergen] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find substitutes respecting constraints.
        
        Returns:
            List of (substitute, score, constraint_info) tuples
        """
        # Your solution here
        pass
    
    def filter_candidates(
        self,
        candidates: List[str],
        restrictions: List[DietaryRestriction] = None,
        allergens_to_avoid: List[Allergen] = None
    ) -> List[str]:
        """Filter candidates by constraints."""
        # Your solution here
        pass
    
    def rank_by_constraint_satisfaction(
        self,
        ingredient: str,
        candidates: List[str],
        soft_restrictions: List[DietaryRestriction] = None
    ) -> List[Tuple[str, float]]:
        """Rank candidates by how well they satisfy soft constraints."""
        # Your solution here
        pass


class DietaryConstraintEncoder:
    """Encode dietary constraints as vectors."""
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.restriction_embeddings = {}
        self.allergen_embeddings = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings for restrictions and allergens."""
        # Your solution here
        pass
    
    def encode_restrictions(
        self,
        restrictions: List[DietaryRestriction]
    ) -> np.ndarray:
        """Encode a set of restrictions as a vector."""
        # Your solution here
        pass
    
    def encode_ingredient(
        self,
        profile: IngredientDietaryProfile
    ) -> np.ndarray:
        """Encode ingredient's dietary profile."""
        # Your solution here
        pass
    
    def compute_compatibility(
        self,
        ingredient_encoding: np.ndarray,
        constraint_encoding: np.ndarray
    ) -> float:
        """Compute compatibility between ingredient and constraints."""
        # Your solution here
        pass


class AllergySafetyChecker:
    """
    Advanced allergen detection and cross-contamination checking.
    """
    
    def __init__(self):
        # Cross-contamination risks
        self.contamination_risks = {
            Allergen.WHEAT: ["oats", "barley", "rye"],
            Allergen.TREE_NUTS: ["coconut"],  # Sometimes classified together
            Allergen.PEANUTS: ["lupine"],
        }
    
    def check_safety(
        self,
        ingredient: str,
        allergies: List[Allergen],
        include_cross_contamination: bool = True
    ) -> Dict[str, any]:
        """
        Comprehensive allergy safety check.
        
        Returns:
            Dictionary with safety status and details
        """
        # Your solution here
        pass
    
    def find_safe_alternatives(
        self,
        ingredient: str,
        allergies: List[Allergen],
        candidates: List[str]
    ) -> List[str]:
        """Find alternatives that are safe for all listed allergies."""
        # Your solution here
        pass
    
    def get_allergen_warnings(
        self,
        ingredients: List[str]
    ) -> Dict[str, Set[Allergen]]:
        """Get allergen warnings for a list of ingredients."""
        # Your solution here
        pass


def create_dietary_filter(
    restrictions: List[DietaryRestriction] = None,
    allergens_to_avoid: List[Allergen] = None
) -> callable:
    """Create a filter function for dietary constraints."""
    # Your solution here
    pass


def score_nutritional_match(
    original: str,
    substitute: str,
    preferences: Dict[str, str] = None
) -> float:
    """
    Score how well substitute matches original nutritionally.
    
    Args:
        original: Original ingredient
        substitute: Proposed substitute
        preferences: Nutritional preferences ("low_fat", "low_sodium", etc.)
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    # Create mock embeddings
    embeddings = {name: np.random.randn(64).astype(np.float32) for name in DIETARY_DB.keys()}
    for name in embeddings:
        embeddings[name] /= np.linalg.norm(embeddings[name])
    
    print("Testing ConstraintChecker...")
    
    checker = ConstraintChecker()
    
    # Test 1: Single restriction check
    assert checker.satisfies_restriction("butter", DietaryRestriction.VEGETARIAN), "Test 1a failed"
    assert not checker.satisfies_restriction("butter", DietaryRestriction.VEGAN), "Test 1b failed"
    print("  ✓ Single restriction check works")
    
    # Test 2: Multiple restrictions
    assert checker.satisfies_all_restrictions("coconut_oil", 
        [DietaryRestriction.VEGAN, DietaryRestriction.GLUTEN_FREE]), "Test 2a failed"
    assert not checker.satisfies_all_restrictions("butter",
        [DietaryRestriction.VEGAN, DietaryRestriction.DAIRY_FREE]), "Test 2b failed"
    print("  ✓ Multiple restriction check works")
    
    # Test 3: Allergen check
    allergens = checker.get_allergens("butter")
    assert Allergen.MILK in allergens, "Test 3a failed"
    
    allergens = checker.get_allergens("coconut_oil")
    assert len(allergens) == 0, "Test 3b failed"
    print("  ✓ Allergen detection works")
    
    # Test 4: Allergy safety
    assert not checker.is_safe_for_allergies("butter", [Allergen.MILK]), "Test 4a failed"
    assert checker.is_safe_for_allergies("coconut_oil", [Allergen.MILK]), "Test 4b failed"
    assert not checker.is_safe_for_allergies("almond_milk", [Allergen.TREE_NUTS]), "Test 4c failed"
    print("  ✓ Allergy safety check works")
    
    # Test 5: Get violations
    violations = checker.get_violations(
        "butter",
        [DietaryRestriction.VEGAN, DietaryRestriction.DAIRY_FREE],
        [Allergen.MILK]
    )
    assert "restriction_violations" in violations, "Test 5a failed"
    assert "allergen_violations" in violations, "Test 5b failed"
    print(f"  ✓ Violations: {violations}")
    
    print("\nTesting ConstraintAwareRetriever...")
    
    retriever = ConstraintAwareRetriever(embeddings, checker)
    
    # Test 6: Find vegan butter substitutes
    subs = retriever.find_substitutes(
        "butter",
        restrictions=[DietaryRestriction.VEGAN, DietaryRestriction.DAIRY_FREE],
        top_k=3
    )
    for name, score, info in subs:
        assert checker.satisfies_all_restrictions(name, 
            [DietaryRestriction.VEGAN, DietaryRestriction.DAIRY_FREE]), f"Test 6 failed: {name}"
    print(f"  ✓ Vegan butter substitutes: {[s[0] for s in subs]}")
    
    # Test 7: Filter with allergens
    safe_subs = retriever.find_substitutes(
        "egg",
        restrictions=[DietaryRestriction.VEGAN],
        allergens_to_avoid=[Allergen.TREE_NUTS],
        top_k=3
    )
    for name, _, _ in safe_subs:
        assert checker.is_safe_for_allergies(name, [Allergen.TREE_NUTS]), f"Test 7 failed: {name}"
    print(f"  ✓ Nut-free vegan egg substitutes: {[s[0] for s in safe_subs]}")
    
    print("\nTesting DietaryConstraintEncoder...")
    
    encoder = DietaryConstraintEncoder(embedding_dim=32)
    
    # Test 8: Encode restrictions
    vec = encoder.encode_restrictions([DietaryRestriction.VEGAN, DietaryRestriction.GLUTEN_FREE])
    assert vec.shape == (32,), f"Test 8 failed: {vec.shape}"
    print(f"  ✓ Constraint encoding shape: {vec.shape}")
    
    # Test 9: Encode ingredient
    profile = DIETARY_DB["coconut_oil"]
    ing_vec = encoder.encode_ingredient(profile)
    assert ing_vec.shape[0] > 0, "Test 9 failed"
    print(f"  ✓ Ingredient encoding works")
    
    # Test 10: Compatibility score
    constraint_vec = encoder.encode_restrictions([DietaryRestriction.VEGAN])
    vegan_ing = encoder.encode_ingredient(DIETARY_DB["coconut_oil"])
    nonvegan_ing = encoder.encode_ingredient(DIETARY_DB["butter"])
    
    compat_vegan = encoder.compute_compatibility(vegan_ing, constraint_vec)
    compat_nonvegan = encoder.compute_compatibility(nonvegan_ing, constraint_vec)
    assert compat_vegan > compat_nonvegan, "Test 10 failed"
    print(f"  ✓ Compatibility: coconut_oil={compat_vegan:.2f}, butter={compat_nonvegan:.2f}")
    
    print("\nTesting AllergySafetyChecker...")
    
    safety_checker = AllergySafetyChecker()
    
    # Test 11: Safety check
    result = safety_checker.check_safety("almond_milk", [Allergen.TREE_NUTS])
    assert not result.get("safe", True), "Test 11 failed"
    print(f"  ✓ Almond milk safety for nut allergy: {result}")
    
    # Test 12: Find safe alternatives
    safe = safety_checker.find_safe_alternatives(
        "almond_milk",
        [Allergen.TREE_NUTS],
        list(DIETARY_DB.keys())
    )
    assert "almond_milk" not in safe, "Test 12a failed"
    print(f"  ✓ Nut-safe alternatives: {safe}")
    
    # Test 13: Allergen warnings
    warnings = safety_checker.get_allergen_warnings(["butter", "almond_milk", "egg"])
    assert Allergen.MILK in warnings.get("butter", set()), "Test 13 failed"
    print(f"  ✓ Allergen warnings: {warnings}")
    
    print("\nTesting utility functions...")
    
    # Test 14: Create dietary filter
    filter_fn = create_dietary_filter(
        restrictions=[DietaryRestriction.VEGAN],
        allergens_to_avoid=[Allergen.TREE_NUTS]
    )
    assert filter_fn("coconut_oil"), "Test 14a failed"
    assert not filter_fn("butter"), "Test 14b failed"
    assert not filter_fn("almond_milk"), "Test 14c failed"
    print("  ✓ Dietary filter function works")
    
    # Test 15: Nutritional match
    score = score_nutritional_match("butter", "coconut_oil", {"low_sodium": "prefer"})
    assert 0 <= score <= 1, f"Test 15 failed: {score}"
    print(f"  ✓ Nutritional match score: {score:.2f}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Hard constraints (allergies) must never be violated")
    print("2. Soft constraints (preferences) can be traded off")
    print("3. Allergen detection prevents dangerous substitutions")
    print("4. Constraint encoding enables learned filtering")
    print("\nNext: Flavor profile matching!")
