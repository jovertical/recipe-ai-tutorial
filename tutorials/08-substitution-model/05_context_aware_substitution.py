# Problem 5: Context-Aware Substitution
#
# Build a context-aware substitution system. The same ingredient may have
# different substitutes depending on the recipe context.
#
# Example:
#   # Same ingredient, different contexts
#   substitute("butter", context="baking cookies") -> margarine
#   substitute("butter", context="frying eggs") -> olive oil
#   substitute("butter", context="spreading on toast") -> avocado
#
# ML Relevance: Context-aware models consider the recipe type and cooking
# method, account for other ingredients in the recipe, handle ingredient
# function (binding, flavoring, etc.), and produce more relevant substitutions.
# This introduces the concept of conditional substitution.
#
# Your Task:
#   1. Implement context encoding (recipe type, cooking method)
#   2. Implement ContextualSubstitutor with context embeddings
#   3. Implement attention-based context weighting
#   4. Implement recipe-aware substitution

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class RecipeType(Enum):
    """Types of recipes."""
    BAKING = "baking"
    FRYING = "frying"
    ROASTING = "roasting"
    GRILLING = "grilling"
    SAUTEING = "sauteing"
    BOILING = "boiling"
    STEAMING = "steaming"
    RAW = "raw"
    MIXING = "mixing"
    UNKNOWN = "unknown"


class IngredientRole(Enum):
    """Role an ingredient plays in a recipe."""
    FAT = "fat"
    BINDING = "binding"
    LEAVENING = "leavening"
    SWEETENING = "sweetening"
    FLAVORING = "flavoring"
    MOISTURE = "moisture"
    STRUCTURE = "structure"
    PROTEIN = "protein"
    THICKENING = "thickening"
    ACIDIC = "acidic"


@dataclass
class RecipeContext:
    """Context information for a recipe."""
    recipe_type: RecipeType = RecipeType.UNKNOWN
    other_ingredients: List[str] = field(default_factory=list)
    cuisine: str = ""
    dietary_requirements: List[str] = field(default_factory=list)
    cooking_method: str = ""
    temperature: str = ""  # "high", "medium", "low", "none"
    
    def to_vector(self, encoder: 'ContextEncoder') -> np.ndarray:
        """Convert context to vector representation."""
        return encoder.encode(self)


class ContextEncoder:
    """Encode recipe context into vector representations."""
    
    def __init__(self, embedding_dim: int = 32):
        """
        Initialize context encoder.
        
        Initialize embeddings for:
        - Recipe types
        - Cuisines
        - Cooking methods
        - Temperature levels
        """
        self.embedding_dim = embedding_dim
        # Your solution here
        pass
    
    def encode(self, context: RecipeContext) -> np.ndarray:
        """
        Encode a recipe context into a vector.
        
        Hints:
        1. Get embeddings for each context component
        2. Combine (concatenate, add, or average)
        3. Project to embedding_dim if needed
        """
        # Your solution here
        pass
    
    def encode_recipe_type(self, recipe_type: RecipeType) -> np.ndarray:
        """Encode recipe type to vector."""
        # Your solution here
        pass
    
    def encode_ingredients(self, ingredients: List[str]) -> np.ndarray:
        """Encode list of co-occurring ingredients."""
        # Your solution here
        pass
    
    def encode_cuisine(self, cuisine: str) -> np.ndarray:
        """Encode cuisine type."""
        # Your solution here
        pass


def infer_ingredient_role(
    ingredient: str,
    context: RecipeContext
) -> List[IngredientRole]:
    """
    Infer what role an ingredient plays in a recipe context.
    
    Hints:
    1. Start with ingredient's default roles
    2. Adjust based on recipe type
    3. Consider other ingredients that might fill roles
    
    Example:
      context = RecipeContext(recipe_type=RecipeType.BAKING)
      infer_ingredient_role("butter", context)
      # Returns: [IngredientRole.FAT, IngredientRole.MOISTURE, IngredientRole.FLAVORING]
    """
    # Your solution here
    pass


class ContextualSubstitutor:
    """Find substitutes based on recipe context."""
    
    def __init__(
        self,
        ingredient_embeddings: Dict[str, np.ndarray],
        context_encoder: ContextEncoder,
        role_mappings: Dict[str, List[IngredientRole]] = None
    ):
        """Initialize contextual substitutor."""
        # Your solution here
        pass
    
    def find_substitutes(
        self,
        ingredient: str,
        context: RecipeContext,
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Find context-appropriate substitutes.
        Returns list of (substitute, score, reason) tuples.
        
        Hints:
        1. Encode context
        2. Infer ingredient's role in this context
        3. Find candidates that can fill same role
        4. Score by embedding similarity + role match
        """
        # Your solution here
        pass
    
    def compute_contextual_similarity(
        self,
        ingredient1: str,
        ingredient2: str,
        context: RecipeContext
    ) -> float:
        """Compute context-dependent similarity."""
        # Your solution here
        pass
    
    def get_role_substitutes(
        self,
        role: IngredientRole,
        dietary_requirements: List[str] = None
    ) -> List[str]:
        """Get all ingredients that can fill a specific role."""
        # Your solution here
        pass


class AttentionSubstitutor:
    """Use attention mechanism to weight context elements."""
    
    def __init__(
        self,
        ingredient_embeddings: Dict[str, np.ndarray],
        embedding_dim: int = 64,
        attention_dim: int = 32
    ):
        """Initialize attention-based substitutor with attention weights."""
        # Your solution here
        pass
    
    def compute_attention(
        self,
        query_ingredient: str,
        context_ingredients: List[str]
    ) -> np.ndarray:
        """
        Compute attention weights over context ingredients.
        Returns attention weights that sum to 1.
        
        Hints:
        1. Get embeddings for query and context
        2. Compute attention scores (e.g., dot product)
        3. Apply softmax to get weights
        """
        # Your solution here
        pass
    
    def context_weighted_embedding(
        self,
        ingredient: str,
        context_ingredients: List[str]
    ) -> np.ndarray:
        """Create context-weighted query embedding."""
        # Your solution here
        pass
    
    def find_substitutes(
        self,
        ingredient: str,
        context_ingredients: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find substitutes using attention-weighted context."""
        # Your solution here
        pass


class RecipeAwareSubstitutor:
    """Full recipe-aware substitution considering all aspects."""
    
    def __init__(
        self,
        ingredient_embeddings: Dict[str, np.ndarray],
        context_encoder: ContextEncoder
    ):
        """Initialize recipe-aware substitutor."""
        # Your solution here
        pass
    
    def substitute_in_recipe(
        self,
        recipe_ingredients: List[str],
        target_ingredient: str,
        recipe_context: RecipeContext,
        constraints: Dict[str, any] = None
    ) -> List[Dict[str, any]]:
        """
        Find the best substitute considering the full recipe.
        
        Returns list of substitution options with details like:
        {
            "substitute": "coconut_oil",
            "score": 0.88,
            "roles_filled": ["fat", "moisture"],
            "adjustments": "Use solid coconut oil at room temperature",
            "impact": "Slight coconut flavor, good structure"
        }
        """
        # Your solution here
        pass
    
    def analyze_substitution_impact(
        self,
        original: str,
        substitute: str,
        recipe_ingredients: List[str],
        recipe_context: RecipeContext
    ) -> Dict[str, any]:
        """Analyze how a substitution will impact the recipe."""
        # Your solution here
        pass


def detect_recipe_type(
    ingredients: List[str],
    instructions: str = None
) -> RecipeType:
    """
    Detect recipe type from ingredients and/or instructions.
    
    Example:
      detect_recipe_type(["flour", "sugar", "butter", "eggs"])
      # Returns: RecipeType.BAKING
    """
    # Your solution here
    pass


def create_context_from_recipe(
    ingredients: List[str],
    title: str = "",
    instructions: str = "",
    cuisine: str = ""
) -> RecipeContext:
    """Create a RecipeContext from recipe information."""
    # Your solution here
    pass


# Default role mappings
DEFAULT_ROLE_MAPPINGS = {
    "butter": [IngredientRole.FAT, IngredientRole.MOISTURE, IngredientRole.FLAVORING],
    "oil": [IngredientRole.FAT, IngredientRole.MOISTURE],
    "egg": [IngredientRole.BINDING, IngredientRole.LEAVENING, IngredientRole.MOISTURE],
    "flour": [IngredientRole.STRUCTURE, IngredientRole.THICKENING],
    "sugar": [IngredientRole.SWEETENING, IngredientRole.MOISTURE],
    "milk": [IngredientRole.MOISTURE, IngredientRole.FAT],
    "baking_powder": [IngredientRole.LEAVENING],
    "salt": [IngredientRole.FLAVORING],
    "vanilla": [IngredientRole.FLAVORING],
    "lemon_juice": [IngredientRole.ACIDIC, IngredientRole.FLAVORING],
    "cornstarch": [IngredientRole.THICKENING],
}


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    # Create mock embeddings
    ingredients = [
        "butter", "margarine", "coconut_oil", "olive_oil",
        "flour", "almond_flour", "oat_flour",
        "sugar", "honey", "maple_syrup",
        "egg", "flax_egg", "applesauce",
        "milk", "oat_milk", "almond_milk",
        "garlic", "onion", "ginger"
    ]
    
    embeddings = {}
    for ing in ingredients:
        embeddings[ing] = np.random.randn(64).astype(np.float32)
        embeddings[ing] /= np.linalg.norm(embeddings[ing])
    
    # Make related ingredients similar
    for related in [("butter", "margarine"), ("butter", "coconut_oil"),
                    ("milk", "oat_milk"), ("egg", "flax_egg")]:
        a, b = related
        embeddings[b] = embeddings[a] + np.random.randn(64) * 0.1
        embeddings[b] /= np.linalg.norm(embeddings[b])
    
    print("Testing ContextEncoder...")
    
    # Test 1: Initialize encoder
    encoder = ContextEncoder(embedding_dim=32)
    print("  ✓ ContextEncoder initialized")
    
    # Test 2: Encode recipe type
    baking_vec = encoder.encode_recipe_type(RecipeType.BAKING)
    assert baking_vec.shape[0] > 0, "Test 2 failed"
    print(f"  ✓ Recipe type encoding shape: {baking_vec.shape}")
    
    # Test 3: Encode full context
    context = RecipeContext(
        recipe_type=RecipeType.BAKING,
        other_ingredients=["flour", "sugar", "eggs"],
        cuisine="american"
    )
    context_vec = encoder.encode(context)
    assert context_vec.shape == (32,), f"Test 3 failed: {context_vec.shape}"
    print(f"  ✓ Full context encoding shape: {context_vec.shape}")
    
    print("\nTesting infer_ingredient_role...")
    
    # Test 4: Role inference in baking
    context = RecipeContext(recipe_type=RecipeType.BAKING)
    roles = infer_ingredient_role("butter", context)
    assert len(roles) > 0, "Test 4a failed"
    assert IngredientRole.FAT in roles, "Test 4b failed: butter should be fat"
    print(f"  ✓ Butter roles in baking: {[r.value for r in roles]}")
    
    # Test 5: Role inference in frying
    context = RecipeContext(recipe_type=RecipeType.FRYING)
    roles = infer_ingredient_role("butter", context)
    assert IngredientRole.FAT in roles, "Test 5 failed"
    print(f"  ✓ Butter roles in frying: {[r.value for r in roles]}")
    
    print("\nTesting ContextualSubstitutor...")
    
    # Test 6: Initialize substitutor
    substitutor = ContextualSubstitutor(
        ingredient_embeddings=embeddings,
        context_encoder=encoder,
        role_mappings=DEFAULT_ROLE_MAPPINGS
    )
    print("  ✓ ContextualSubstitutor initialized")
    
    # Test 7: Find substitutes with context
    context = RecipeContext(
        recipe_type=RecipeType.BAKING,
        other_ingredients=["flour", "sugar"]
    )
    results = substitutor.find_substitutes("butter", context, top_k=3)
    assert len(results) > 0, "Test 7a failed"
    assert all(len(r) == 3 for r in results), "Test 7b failed: should be (sub, score, reason)"
    print(f"  ✓ Substitutes for butter (baking): {[(r[0], f'{r[1]:.2f}') for r in results]}")
    
    # Test 8: Different context, different results
    context_frying = RecipeContext(recipe_type=RecipeType.FRYING)
    results_frying = substitutor.find_substitutes("butter", context_frying, top_k=3)
    print(f"  ✓ Substitutes for butter (frying): {[(r[0], f'{r[1]:.2f}') for r in results_frying]}")
    
    # Test 9: Contextual similarity
    context = RecipeContext(recipe_type=RecipeType.BAKING)
    sim = substitutor.compute_contextual_similarity("butter", "margarine", context)
    assert 0 <= sim <= 1, f"Test 9 failed: {sim}"
    print(f"  ✓ Contextual similarity (butter-margarine, baking): {sim:.3f}")
    
    print("\nTesting AttentionSubstitutor...")
    
    # Test 10: Initialize attention substitutor
    attention_sub = AttentionSubstitutor(embeddings)
    print("  ✓ AttentionSubstitutor initialized")
    
    # Test 11: Compute attention
    context_ings = ["flour", "sugar", "milk"]
    weights = attention_sub.compute_attention("butter", context_ings)
    assert len(weights) == len(context_ings), f"Test 11a failed: {len(weights)}"
    assert abs(sum(weights) - 1.0) < 0.01, f"Test 11b failed: weights should sum to 1"
    print(f"  ✓ Attention weights: {dict(zip(context_ings, [f'{w:.2f}' for w in weights]))}")
    
    # Test 12: Context-weighted embedding
    weighted_emb = attention_sub.context_weighted_embedding("butter", context_ings)
    assert weighted_emb.shape == (64,), f"Test 12 failed: {weighted_emb.shape}"
    print(f"  ✓ Context-weighted embedding shape: {weighted_emb.shape}")
    
    # Test 13: Attention-based substitutes
    results = attention_sub.find_substitutes("butter", context_ings, top_k=3)
    assert len(results) > 0, "Test 13 failed"
    print(f"  ✓ Attention-based substitutes: {[(r[0], f'{r[1]:.2f}') for r in results]}")
    
    print("\nTesting RecipeAwareSubstitutor...")
    
    # Test 14: Initialize recipe-aware substitutor
    recipe_sub = RecipeAwareSubstitutor(embeddings, encoder)
    print("  ✓ RecipeAwareSubstitutor initialized")
    
    # Test 15: Substitute in full recipe
    recipe_ings = ["flour", "butter", "sugar", "eggs", "vanilla"]
    context = RecipeContext(recipe_type=RecipeType.BAKING)
    options = recipe_sub.substitute_in_recipe(
        recipe_ings, "butter", context,
        constraints={"dietary": ["vegan"]}
    )
    assert len(options) > 0, "Test 15a failed"
    assert "substitute" in options[0], "Test 15b failed"
    print(f"  ✓ Recipe-aware options: {[o.get('substitute') for o in options[:3]]}")
    
    # Test 16: Analyze impact
    impact = recipe_sub.analyze_substitution_impact(
        "butter", "coconut_oil", recipe_ings, context
    )
    assert isinstance(impact, dict), "Test 16 failed"
    print(f"  ✓ Impact analysis keys: {list(impact.keys())}")
    
    print("\nTesting utility functions...")
    
    # Test 17: Detect recipe type
    baking_ings = ["flour", "sugar", "butter", "eggs", "baking_powder"]
    detected = detect_recipe_type(baking_ings)
    assert detected == RecipeType.BAKING, f"Test 17 failed: {detected}"
    print(f"  ✓ Detected recipe type: {detected.value}")
    
    # Test 18: Create context from recipe
    context = create_context_from_recipe(
        ingredients=baking_ings,
        title="Chocolate Chip Cookies",
        cuisine="american"
    )
    assert context.recipe_type == RecipeType.BAKING, "Test 18a failed"
    assert len(context.other_ingredients) > 0, "Test 18b failed"
    print(f"  ✓ Created context: {context.recipe_type.value}, {len(context.other_ingredients)} ingredients")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Recipe context affects which substitutes are appropriate")
    print("2. Ingredient roles vary by recipe type")
    print("3. Attention weights relevant context ingredients")
    print("4. Full recipe analysis provides best substitutions")
    print("\nNext: Build a binary substitution classifier!")
