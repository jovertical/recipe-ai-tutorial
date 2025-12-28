# Problem 1: Introduction to Ingredient Substitution
#
# Explore the fundamentals of ingredient substitution - understanding what
# makes two ingredients interchangeable and the factors that determine
# substitution quality.
#
# Example:
#   # Good substitution
#   substitute("butter", context="baking") -> ["margarine", "coconut oil"]
#
#   # Context matters!
#   substitute("butter", context="spreading") -> ["margarine", "cream cheese"]
#
# ML Relevance: Ingredient substitution is a core task for Recipe AI:
# - Dietary restrictions (allergies, vegan, kosher)
# - Ingredient availability (seasonal, regional)
# - Health optimization (lower fat, less sodium)
# - Cost reduction (budget cooking)
#
# This module builds on embeddings to create practical substitution systems.

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class IngredientCategory(Enum):
    """Primary ingredient categories."""
    PROTEIN = "protein"
    FAT = "fat"
    CARBOHYDRATE = "carbohydrate"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"
    DAIRY = "dairy"
    SPICE = "spice"
    LIQUID = "liquid"
    SWEETENER = "sweetener"
    LEAVENING = "leavening"
    OTHER = "other"


class DietaryTag(Enum):
    """Dietary restriction tags."""
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    KOSHER = "kosher"
    HALAL = "halal"
    LOW_SODIUM = "low_sodium"
    LOW_FAT = "low_fat"


@dataclass
class IngredientProfile:
    """Complete profile of an ingredient."""
    name: str
    category: IngredientCategory
    dietary_tags: List[DietaryTag]
    flavor_profile: Dict[str, float]  # sweet, salty, sour, bitter, umami, spicy
    texture: str  # liquid, solid, creamy, crunchy, etc.
    function: List[str]  # binding, leavening, flavoring, etc.

    def __repr__(self):
        return f"IngredientProfile({self.name})"


# Sample ingredient database
INGREDIENT_DB = {
    "butter": IngredientProfile(
        name="butter",
        category=IngredientCategory.FAT,
        dietary_tags=[DietaryTag.VEGETARIAN],
        flavor_profile={"sweet": 0.1, "salty": 0.2, "sour": 0.0, "bitter": 0.0, "umami": 0.1, "spicy": 0.0},
        texture="solid",
        function=["fat", "flavor", "moisture", "tenderizing"]
    ),
    "margarine": IngredientProfile(
        name="margarine",
        category=IngredientCategory.FAT,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.0, "salty": 0.2, "sour": 0.0, "bitter": 0.0, "umami": 0.0, "spicy": 0.0},
        texture="solid",
        function=["fat", "moisture", "tenderizing"]
    ),
    "coconut_oil": IngredientProfile(
        name="coconut_oil",
        category=IngredientCategory.FAT,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.2, "salty": 0.0, "sour": 0.0, "bitter": 0.0, "umami": 0.0, "spicy": 0.0},
        texture="solid",
        function=["fat", "moisture"]
    ),
    "olive_oil": IngredientProfile(
        name="olive_oil",
        category=IngredientCategory.FAT,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.0, "salty": 0.0, "sour": 0.0, "bitter": 0.1, "umami": 0.0, "spicy": 0.0},
        texture="liquid",
        function=["fat", "flavor", "moisture"]
    ),
    "egg": IngredientProfile(
        name="egg",
        category=IngredientCategory.PROTEIN,
        dietary_tags=[DietaryTag.VEGETARIAN, DietaryTag.GLUTEN_FREE],
        flavor_profile={"sweet": 0.0, "salty": 0.1, "sour": 0.0, "bitter": 0.0, "umami": 0.2, "spicy": 0.0},
        texture="liquid",
        function=["binding", "leavening", "moisture", "protein"]
    ),
    "flax_egg": IngredientProfile(
        name="flax_egg",
        category=IngredientCategory.OTHER,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.GLUTEN_FREE, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.0, "salty": 0.0, "sour": 0.0, "bitter": 0.1, "umami": 0.0, "spicy": 0.0},
        texture="gel",
        function=["binding", "moisture"]
    ),
    "applesauce": IngredientProfile(
        name="applesauce",
        category=IngredientCategory.FRUIT,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.GLUTEN_FREE, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.6, "salty": 0.0, "sour": 0.2, "bitter": 0.0, "umami": 0.0, "spicy": 0.0},
        texture="puree",
        function=["binding", "moisture", "sweetening"]
    ),
    "milk": IngredientProfile(
        name="milk",
        category=IngredientCategory.DAIRY,
        dietary_tags=[DietaryTag.VEGETARIAN, DietaryTag.GLUTEN_FREE],
        flavor_profile={"sweet": 0.2, "salty": 0.0, "sour": 0.0, "bitter": 0.0, "umami": 0.1, "spicy": 0.0},
        texture="liquid",
        function=["liquid", "moisture", "protein", "fat"]
    ),
    "oat_milk": IngredientProfile(
        name="oat_milk",
        category=IngredientCategory.LIQUID,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.DAIRY_FREE, DietaryTag.NUT_FREE],
        flavor_profile={"sweet": 0.3, "salty": 0.0, "sour": 0.0, "bitter": 0.0, "umami": 0.0, "spicy": 0.0},
        texture="liquid",
        function=["liquid", "moisture"]
    ),
    "almond_milk": IngredientProfile(
        name="almond_milk",
        category=IngredientCategory.LIQUID,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.DAIRY_FREE, DietaryTag.GLUTEN_FREE],
        flavor_profile={"sweet": 0.2, "salty": 0.0, "sour": 0.0, "bitter": 0.1, "umami": 0.0, "spicy": 0.0},
        texture="liquid",
        function=["liquid", "moisture"]
    ),
    "sugar": IngredientProfile(
        name="sugar",
        category=IngredientCategory.SWEETENER,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.GLUTEN_FREE, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 1.0, "salty": 0.0, "sour": 0.0, "bitter": 0.0, "umami": 0.0, "spicy": 0.0},
        texture="granular",
        function=["sweetening", "texture", "browning"]
    ),
    "honey": IngredientProfile(
        name="honey",
        category=IngredientCategory.SWEETENER,
        dietary_tags=[DietaryTag.VEGETARIAN, DietaryTag.GLUTEN_FREE, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.9, "salty": 0.0, "sour": 0.1, "bitter": 0.0, "umami": 0.0, "spicy": 0.0},
        texture="liquid",
        function=["sweetening", "moisture", "flavor"]
    ),
    "maple_syrup": IngredientProfile(
        name="maple_syrup",
        category=IngredientCategory.SWEETENER,
        dietary_tags=[DietaryTag.VEGAN, DietaryTag.GLUTEN_FREE, DietaryTag.DAIRY_FREE],
        flavor_profile={"sweet": 0.85, "salty": 0.0, "sour": 0.0, "bitter": 0.05, "umami": 0.0, "spicy": 0.0},
        texture="liquid",
        function=["sweetening", "moisture", "flavor"]
    ),
}


def categorize_ingredient(ingredient_name: str) -> Optional[IngredientProfile]:
    """Look up or infer the profile of an ingredient."""
    # Your solution here
    pass


def flavor_similarity(profile1: IngredientProfile, profile2: IngredientProfile) -> float:
    """Calculate flavor profile similarity between two ingredients."""
    # Your solution here
    pass


def function_overlap(profile1: IngredientProfile, profile2: IngredientProfile) -> float:
    """Calculate functional overlap between two ingredients (Jaccard similarity)."""
    # Your solution here
    pass


def texture_compatibility(profile1: IngredientProfile, profile2: IngredientProfile) -> float:
    """Score texture compatibility for substitution."""
    # Your solution here
    pass


def basic_substitution_score(
    original: str,
    substitute: str,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate overall substitution quality score.

    Args:
        original: Original ingredient name
        substitute: Proposed substitute name
        weights: Optional weights for different factors
            - category: Same category importance (default 0.3)
            - flavor: Flavor similarity importance (default 0.25)
            - function: Function overlap importance (default 0.3)
            - texture: Texture compatibility importance (default 0.15)

    Returns:
        Substitution score from 0 to 1
    """
    if weights is None:
        weights = {
            "category": 0.3,
            "flavor": 0.25,
            "function": 0.3,
            "texture": 0.15
        }
    # Your solution here
    pass


def find_substitutes(
    ingredient: str,
    dietary_requirements: List[DietaryTag] = None,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find the best substitutes for an ingredient."""
    # Your solution here
    pass


def check_dietary_compatibility(
    profile: IngredientProfile,
    requirements: List[DietaryTag]
) -> bool:
    """Check if ingredient meets dietary requirements."""
    # Your solution here
    pass


def explain_substitution(original: str, substitute: str) -> Dict[str, any]:
    """Generate explanation for why a substitution works (or doesn't)."""
    # Your solution here
    pass


def substitution_ratio(original: str, substitute: str) -> str:
    """Suggest substitution ratio."""
    # Your solution here
    pass


@dataclass
class SubstitutionResult:
    """Complete substitution recommendation."""
    original: str
    substitute: str
    score: float
    ratio: str
    explanation: Dict[str, any]

    def __repr__(self):
        return f"Substitute {self.original} -> {self.substitute} (score: {self.score:.2f})"


def get_substitution(
    ingredient: str,
    dietary_requirements: List[DietaryTag] = None,
    context: str = None
) -> Optional[SubstitutionResult]:
    """Get the best substitution with full details."""
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    print("Testing categorize_ingredient...")

    # Test 1: Known ingredient
    profile = categorize_ingredient("butter")
    assert profile is not None, "Test 1a failed: butter should be found"
    assert profile.category == IngredientCategory.FAT, "Test 1b failed: butter is a fat"
    assert DietaryTag.VEGETARIAN in profile.dietary_tags, "Test 1c failed"
    print("  ✓ Known ingredient lookup works")

    # Test 2: Unknown ingredient
    profile = categorize_ingredient("unicorn_tears")
    assert profile is None, "Test 2 failed: unknown should return None"
    print("  ✓ Unknown ingredient returns None")

    print("\nTesting flavor_similarity...")

    # Test 3: Similar flavors
    butter = INGREDIENT_DB["butter"]
    margarine = INGREDIENT_DB["margarine"]
    sim = flavor_similarity(butter, margarine)
    assert 0 <= sim <= 1, f"Test 3a failed: {sim}"
    assert sim > 0.7, f"Test 3b failed: butter/margarine should be similar, got {sim}"
    print(f"  ✓ Butter-margarine flavor similarity: {sim:.2f}")

    # Test 4: Different flavors
    sugar = INGREDIENT_DB["sugar"]
    sim = flavor_similarity(butter, sugar)
    assert sim < 0.5, f"Test 4 failed: butter/sugar should be different, got {sim}"
    print(f"  ✓ Butter-sugar flavor similarity: {sim:.2f}")

    print("\nTesting function_overlap...")

    # Test 5: Overlapping functions
    overlap = function_overlap(butter, margarine)
    assert 0 <= overlap <= 1, f"Test 5a failed: {overlap}"
    assert overlap > 0.5, f"Test 5b failed: should have good overlap, got {overlap}"
    print(f"  ✓ Butter-margarine function overlap: {overlap:.2f}")

    # Test 6: Different functions
    egg = INGREDIENT_DB["egg"]
    overlap = function_overlap(butter, egg)
    assert overlap < 0.5, f"Test 6 failed: butter/egg functions should differ, got {overlap}"
    print(f"  ✓ Butter-egg function overlap: {overlap:.2f}")

    print("\nTesting texture_compatibility...")

    # Test 7: Same texture
    compat = texture_compatibility(butter, margarine)
    assert compat == 1.0, f"Test 7 failed: same texture should be 1.0, got {compat}"
    print(f"  ✓ Same texture compatibility: {compat:.2f}")

    # Test 8: Different texture
    olive_oil = INGREDIENT_DB["olive_oil"]
    compat = texture_compatibility(butter, olive_oil)
    assert 0 < compat < 1.0, f"Test 8 failed: different texture, got {compat}"
    print(f"  ✓ Different texture compatibility: {compat:.2f}")

    print("\nTesting basic_substitution_score...")

    # Test 9: Good substitution
    score = basic_substitution_score("butter", "margarine")
    assert 0.7 <= score <= 1.0, f"Test 9 failed: butter/margarine should score high, got {score}"
    print(f"  ✓ Butter -> Margarine score: {score:.2f}")

    # Test 10: Poor substitution
    score = basic_substitution_score("butter", "sugar")
    assert score < 0.5, f"Test 10 failed: butter/sugar should score low, got {score}"
    print(f"  ✓ Butter -> Sugar score: {score:.2f}")

    print("\nTesting check_dietary_compatibility...")

    # Test 11: Compatible
    profile = INGREDIENT_DB["margarine"]
    compat = check_dietary_compatibility(profile, [DietaryTag.VEGAN])
    assert compat is True, "Test 11 failed: margarine is vegan"
    print("  ✓ Margarine is vegan-compatible")

    # Test 12: Incompatible
    profile = INGREDIENT_DB["butter"]
    compat = check_dietary_compatibility(profile, [DietaryTag.VEGAN])
    assert compat is False, "Test 12 failed: butter is not vegan"
    print("  ✓ Butter is not vegan-compatible")

    print("\nTesting find_substitutes...")

    # Test 13: Find vegan substitutes for butter
    subs = find_substitutes("butter", dietary_requirements=[DietaryTag.VEGAN], top_k=3)
    assert len(subs) > 0, "Test 13a failed: should find substitutes"
    assert len(subs) <= 3, f"Test 13b failed: should return at most 3, got {len(subs)}"
    assert all(isinstance(s, tuple) and len(s) == 2 for s in subs), "Test 13c failed"
    # Verify all are vegan
    for name, score in subs:
        profile = INGREDIENT_DB[name]
        assert DietaryTag.VEGAN in profile.dietary_tags, f"Test 13d failed: {name} not vegan"
    print(f"  ✓ Found {len(subs)} vegan butter substitutes")
    for name, score in subs:
        print(f"    - {name}: {score:.2f}")

    # Test 14: Find substitutes without restrictions
    subs = find_substitutes("milk", top_k=3)
    assert len(subs) > 0, "Test 14 failed"
    print(f"  ✓ Found {len(subs)} milk substitutes")

    print("\nTesting explain_substitution...")

    # Test 15: Explanation
    explanation = explain_substitution("butter", "coconut_oil")
    assert "score" in explanation, "Test 15a failed"
    assert "category_match" in explanation, "Test 15b failed"
    assert "warnings" in explanation, "Test 15c failed"
    assert "tips" in explanation, "Test 15d failed"
    print("  ✓ Explanation generated")
    print(f"    Score: {explanation['score']:.2f}")
    if explanation.get("warnings"):
        print(f"    Warnings: {explanation['warnings']}")

    print("\nTesting substitution_ratio...")

    # Test 16: Ratio
    ratio = substitution_ratio("butter", "margarine")
    assert isinstance(ratio, str), "Test 16a failed"
    assert len(ratio) > 0, "Test 16b failed"
    print(f"  ✓ Butter -> Margarine ratio: {ratio}")

    print("\nTesting get_substitution...")

    # Test 17: Full substitution result
    result = get_substitution("butter", [DietaryTag.VEGAN], "baking")
    assert result is not None, "Test 17a failed"
    assert isinstance(result, SubstitutionResult), "Test 17b failed"
    assert result.original == "butter", "Test 17c failed"
    assert result.score > 0, "Test 17d failed"
    print(f"  ✓ Best substitution: {result}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)

    print("\nKey concepts learned:")
    print("1. Ingredients have multiple attributes (category, flavor, function, texture)")
    print("2. Good substitutions match on multiple dimensions")
    print("3. Dietary requirements filter available options")
    print("4. Context affects substitution quality")
