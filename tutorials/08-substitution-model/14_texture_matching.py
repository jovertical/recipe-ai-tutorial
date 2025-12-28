# Problem 14: Texture Matching
#
# Build a system that considers texture properties when suggesting substitutes
# to maintain recipe mouthfeel.
#
# Example:
#   matcher = TextureMatcher(texture_embeddings)
#   subs = matcher.find_similar_texture("heavy_cream", cooking_method="whipping")
#   # Returns: ["coconut_cream", "cashew_cream"]
#
# ML Relevance: Texture is context-dependent (cream for whipping vs pouring)
# requiring conditional predictions based on usage.
#
# Your Task:
#   1. Implement texture property representation
#   2. Implement context-dependent texture matching
#   3. Implement cooking method awareness
#   4. Combine texture with other substitution factors


from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Texture(Enum):
    LIQUID = "liquid"
    SOLID = "solid"
    SEMI_SOLID = "semi_solid"
    POWDER = "powder"
    GRANULAR = "granular"
    CREAMY = "creamy"
    CHUNKY = "chunky"
    FLAKY = "flaky"
    CHEWY = "chewy"
    CRISPY = "crispy"
    GEL = "gel"


class CulinaryFunction(Enum):
    BINDING = "binding"
    LEAVENING = "leavening"
    MOISTURE = "moisture"
    FAT = "fat"
    SWEETENING = "sweetening"
    THICKENING = "thickening"
    EMULSIFYING = "emulsifying"
    FLAVORING = "flavoring"
    COLORING = "coloring"
    TENDERIZING = "tenderizing"
    PRESERVING = "preserving"
    ACIDIFYING = "acidifying"


class CookingMethod(Enum):
    BAKING = "baking"
    FRYING = "frying"
    ROASTING = "roasting"
    BOILING = "boiling"
    STEAMING = "steaming"
    GRILLING = "grilling"
    SAUTEING = "sauteing"
    RAW = "raw"
    MIXING = "mixing"


@dataclass
class TextureFunctionProfile:
    """Texture and function profile of an ingredient."""
    name: str
    primary_texture: Texture
    secondary_textures: List[Texture] = field(default_factory=list)
    functions: Set[CulinaryFunction] = field(default_factory=set)
    compatible_methods: Set[CookingMethod] = field(default_factory=set)
    temperature_sensitive: bool = False
    melting_point: Optional[float] = None  # Celsius
    
    def get_all_textures(self) -> Set[Texture]:
        """Get all textures including primary."""
        return {self.primary_texture} | set(self.secondary_textures)


# Sample texture/function database
TEXTURE_DB = {
    "butter": TextureFunctionProfile(
        name="butter",
        primary_texture=Texture.SOLID,
        secondary_textures=[Texture.CREAMY],
        functions={CulinaryFunction.FAT, CulinaryFunction.MOISTURE, 
                   CulinaryFunction.FLAVORING, CulinaryFunction.TENDERIZING},
        compatible_methods={CookingMethod.BAKING, CookingMethod.FRYING, 
                           CookingMethod.SAUTEING, CookingMethod.RAW},
        temperature_sensitive=True,
        melting_point=32
    ),
    "coconut_oil": TextureFunctionProfile(
        name="coconut_oil",
        primary_texture=Texture.SOLID,
        secondary_textures=[],
        functions={CulinaryFunction.FAT, CulinaryFunction.MOISTURE},
        compatible_methods={CookingMethod.BAKING, CookingMethod.FRYING, 
                           CookingMethod.SAUTEING},
        temperature_sensitive=True,
        melting_point=24
    ),
    "olive_oil": TextureFunctionProfile(
        name="olive_oil",
        primary_texture=Texture.LIQUID,
        functions={CulinaryFunction.FAT, CulinaryFunction.MOISTURE, 
                   CulinaryFunction.FLAVORING},
        compatible_methods={CookingMethod.FRYING, CookingMethod.SAUTEING, 
                           CookingMethod.RAW, CookingMethod.ROASTING},
        temperature_sensitive=False
    ),
    "egg": TextureFunctionProfile(
        name="egg",
        primary_texture=Texture.LIQUID,
        secondary_textures=[Texture.GEL],
        functions={CulinaryFunction.BINDING, CulinaryFunction.LEAVENING, 
                   CulinaryFunction.MOISTURE, CulinaryFunction.EMULSIFYING},
        compatible_methods={CookingMethod.BAKING, CookingMethod.FRYING, 
                           CookingMethod.BOILING, CookingMethod.MIXING},
        temperature_sensitive=True
    ),
    "flax_egg": TextureFunctionProfile(
        name="flax_egg",
        primary_texture=Texture.GEL,
        functions={CulinaryFunction.BINDING, CulinaryFunction.MOISTURE},
        compatible_methods={CookingMethod.BAKING, CookingMethod.MIXING},
        temperature_sensitive=False
    ),
    "flour": TextureFunctionProfile(
        name="flour",
        primary_texture=Texture.POWDER,
        functions={CulinaryFunction.THICKENING, CulinaryFunction.BINDING},
        compatible_methods={CookingMethod.BAKING, CookingMethod.FRYING, 
                           CookingMethod.MIXING},
        temperature_sensitive=False
    ),
    "cornstarch": TextureFunctionProfile(
        name="cornstarch",
        primary_texture=Texture.POWDER,
        functions={CulinaryFunction.THICKENING},
        compatible_methods={CookingMethod.BAKING, CookingMethod.BOILING, 
                           CookingMethod.MIXING},
        temperature_sensitive=False
    ),
}


def texture_similarity(
    texture1: Texture,
    texture2: Texture
) -> float:
    """
    Compute similarity between two textures.
    
    Returns:
        Similarity score (0 to 1)
    """
    # Texture similarity matrix (simplified)
    similarity_groups = [
        {Texture.LIQUID, Texture.GEL},
        {Texture.SOLID, Texture.SEMI_SOLID},
        {Texture.POWDER, Texture.GRANULAR},
        {Texture.CREAMY, Texture.SEMI_SOLID},
        {Texture.CHUNKY, Texture.SOLID},
    ]
    
    # Your solution here
    pass


def function_overlap(
    functions1: Set[CulinaryFunction],
    functions2: Set[CulinaryFunction]
) -> float:
    """
    Compute Jaccard similarity of function sets.
    
    Returns:
        Overlap score (0 to 1)
    """
    # Your solution here
    pass


def method_compatibility(
    methods1: Set[CookingMethod],
    methods2: Set[CookingMethod],
    required_method: CookingMethod = None
) -> float:
    """
    Compute cooking method compatibility.
    
    Args:
        methods1: First ingredient's compatible methods
        methods2: Second ingredient's compatible methods
        required_method: Specific method that must be supported
    """
    # Your solution here
    pass


class TextureMatcher:
    """Match ingredients by texture."""
    
    def __init__(self, texture_db: Dict[str, TextureFunctionProfile] = None):
        self.db = texture_db or TEXTURE_DB
    
    def find_similar_texture(
        self,
        ingredient: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find ingredients with similar texture."""
        # Your solution here
        pass
    
    def matches_texture(
        self,
        ingredient: str,
        required_texture: Texture
    ) -> bool:
        """Check if ingredient has required texture."""
        # Your solution here
        pass
    
    def filter_by_texture(
        self,
        candidates: List[str],
        required_textures: List[Texture]
    ) -> List[str]:
        """Filter candidates by texture requirements."""
        # Your solution here
        pass


class FunctionMatcher:
    """Match ingredients by culinary function."""
    
    def __init__(self, texture_db: Dict[str, TextureFunctionProfile] = None):
        self.db = texture_db or TEXTURE_DB
    
    def find_functional_substitutes(
        self,
        ingredient: str,
        required_functions: Set[CulinaryFunction] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float, Set[CulinaryFunction]]]:
        """
        Find substitutes that can fulfill the same functions.
        
        Returns:
            List of (ingredient, score, matched_functions)
        """
        # Your solution here
        pass
    
    def get_missing_functions(
        self,
        original: str,
        substitute: str
    ) -> Set[CulinaryFunction]:
        """Get functions that substitute cannot provide."""
        # Your solution here
        pass
    
    def suggest_complementary_ingredients(
        self,
        substitute: str,
        missing_functions: Set[CulinaryFunction]
    ) -> Dict[CulinaryFunction, List[str]]:
        """Suggest ingredients to fill missing functions."""
        # Your solution here
        pass


class CookingMethodMatcher:
    """Match ingredients by cooking method compatibility."""
    
    def __init__(self, texture_db: Dict[str, TextureFunctionProfile] = None):
        self.db = texture_db or TEXTURE_DB
    
    def filter_by_method(
        self,
        candidates: List[str],
        required_method: CookingMethod
    ) -> List[str]:
        """Filter candidates by cooking method."""
        # Your solution here
        pass
    
    def get_incompatible_methods(
        self,
        ingredient: str
    ) -> Set[CookingMethod]:
        """Get cooking methods not compatible with ingredient."""
        # Your solution here
        pass


class TextureFunctionScorer:
    """Combined texture and function scoring."""
    
    def __init__(
        self,
        texture_db: Dict[str, TextureFunctionProfile] = None,
        texture_weight: float = 0.3,
        function_weight: float = 0.5,
        method_weight: float = 0.2
    ):
        self.db = texture_db or TEXTURE_DB
        self.texture_weight = texture_weight
        self.function_weight = function_weight
        self.method_weight = method_weight
    
    def score(
        self,
        original: str,
        substitute: str,
        cooking_method: CookingMethod = None
    ) -> float:
        """
        Score substitution based on texture and function.
        
        Returns:
            Combined score (0 to 1)
        """
        # Your solution here
        pass
    
    def detailed_score(
        self,
        original: str,
        substitute: str,
        cooking_method: CookingMethod = None
    ) -> Dict[str, float]:
        """Get detailed score breakdown."""
        # Your solution here
        pass
    
    def find_best_substitute(
        self,
        ingredient: str,
        candidates: List[str],
        cooking_method: CookingMethod = None,
        required_functions: Set[CulinaryFunction] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Find best substitute considering all factors."""
        # Your solution here
        pass


def encode_texture_function(
    profile: TextureFunctionProfile,
    embedding_dim: int = 32
) -> np.ndarray:
    """
    Encode texture and function profile as vector.
    """
    # Your solution here
    pass


def analyze_substitution_feasibility(
    original: str,
    substitute: str,
    cooking_method: CookingMethod,
    texture_db: Dict[str, TextureFunctionProfile] = None
) -> Dict[str, any]:
    """
    Analyze if substitution is feasible for given cooking method.
    
    Returns:
        Analysis with feasibility score and warnings
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing texture_similarity...")
    
    # Test 1: Same texture
    sim = texture_similarity(Texture.SOLID, Texture.SOLID)
    assert sim == 1.0, f"Test 1 failed: {sim}"
    print("  ✓ Same texture similarity = 1.0")
    
    # Test 2: Similar textures
    sim = texture_similarity(Texture.SOLID, Texture.SEMI_SOLID)
    assert 0.5 < sim < 1.0, f"Test 2 failed: {sim}"
    print(f"  ✓ Solid-SemiSolid similarity: {sim:.2f}")
    
    # Test 3: Different textures
    sim = texture_similarity(Texture.LIQUID, Texture.POWDER)
    assert sim < 0.5, f"Test 3 failed: {sim}"
    print(f"  ✓ Liquid-Powder similarity: {sim:.2f}")
    
    print("\nTesting function_overlap...")
    
    # Test 4: Function overlap
    funcs1 = {CulinaryFunction.BINDING, CulinaryFunction.MOISTURE}
    funcs2 = {CulinaryFunction.BINDING, CulinaryFunction.LEAVENING}
    overlap = function_overlap(funcs1, funcs2)
    expected = 1 / 3  # 1 common / 3 total unique
    assert abs(overlap - expected) < 0.01, f"Test 4 failed: {overlap}"
    print(f"  ✓ Function overlap: {overlap:.2f}")
    
    print("\nTesting method_compatibility...")
    
    # Test 5: Method compatibility
    methods1 = {CookingMethod.BAKING, CookingMethod.FRYING}
    methods2 = {CookingMethod.BAKING, CookingMethod.BOILING}
    compat = method_compatibility(methods1, methods2)
    print(f"  ✓ Method compatibility: {compat:.2f}")
    
    # Test 6: With required method
    compat = method_compatibility(methods1, methods2, CookingMethod.BAKING)
    assert compat > 0, "Test 6 failed"
    print(f"  ✓ With required BAKING: {compat:.2f}")
    
    print("\nTesting TextureMatcher...")
    
    matcher = TextureMatcher()
    
    # Test 7: Find similar texture
    similar = matcher.find_similar_texture("butter", top_k=3)
    assert len(similar) <= 3, f"Test 7 failed: {len(similar)}"
    print(f"  ✓ Similar texture to butter: {[s[0] for s in similar]}")
    
    # Test 8: Matches texture
    assert matcher.matches_texture("butter", Texture.SOLID), "Test 8a failed"
    assert not matcher.matches_texture("olive_oil", Texture.SOLID), "Test 8b failed"
    print("  ✓ Texture matching works")
    
    print("\nTesting FunctionMatcher...")
    
    func_matcher = FunctionMatcher()
    
    # Test 9: Find functional substitutes for egg
    subs = func_matcher.find_functional_substitutes("egg", top_k=3)
    assert len(subs) > 0, "Test 9 failed"
    print(f"  ✓ Functional substitutes for egg: {[(s[0], list(s[2])) for s in subs]}")
    
    # Test 10: Missing functions
    missing = func_matcher.get_missing_functions("egg", "flax_egg")
    assert CulinaryFunction.LEAVENING in missing, "Test 10 failed"
    print(f"  ✓ Flax_egg missing: {[f.value for f in missing]}")
    
    # Test 11: Complementary ingredients
    suggestions = func_matcher.suggest_complementary_ingredients(
        "flax_egg", {CulinaryFunction.LEAVENING}
    )
    print(f"  ✓ Leavening suggestions: {suggestions}")
    
    print("\nTesting CookingMethodMatcher...")
    
    method_matcher = CookingMethodMatcher()
    
    # Test 12: Filter by method
    candidates = list(TEXTURE_DB.keys())
    filtered = method_matcher.filter_by_method(candidates, CookingMethod.BAKING)
    assert "butter" in filtered, "Test 12a failed"
    print(f"  ✓ Baking-compatible: {filtered}")
    
    print("\nTesting TextureFunctionScorer...")
    
    scorer = TextureFunctionScorer()
    
    # Test 13: Score substitution
    score = scorer.score("butter", "coconut_oil", CookingMethod.BAKING)
    assert 0 <= score <= 1, f"Test 13 failed: {score}"
    print(f"  ✓ Butter->CoconutOil score (baking): {score:.2f}")
    
    # Test 14: Detailed score
    details = scorer.detailed_score("egg", "flax_egg", CookingMethod.BAKING)
    assert "texture" in details, "Test 14 failed"
    print(f"  ✓ Egg->FlaxEgg details: {details}")
    
    # Test 15: Find best substitute
    best = scorer.find_best_substitute(
        "butter",
        ["coconut_oil", "olive_oil", "margarine"],
        cooking_method=CookingMethod.BAKING
    )
    assert len(best) > 0, "Test 15 failed"
    print(f"  ✓ Best butter substitutes: {[(b[0], f'{b[1]:.2f}') for b in best]}")
    
    print("\nTesting encode_texture_function...")
    
    # Test 16: Encoding
    profile = TEXTURE_DB["butter"]
    encoding = encode_texture_function(profile, embedding_dim=32)
    assert encoding.shape == (32,), f"Test 16 failed: {encoding.shape}"
    print(f"  ✓ Encoding shape: {encoding.shape}")
    
    print("\nTesting analyze_substitution_feasibility...")
    
    # Test 17: Feasibility analysis
    analysis = analyze_substitution_feasibility(
        "butter", "olive_oil", CookingMethod.BAKING
    )
    assert "feasibility_score" in analysis, "Test 17a failed"
    assert "warnings" in analysis, "Test 17b failed"
    print(f"  ✓ Feasibility: {analysis['feasibility_score']:.2f}")
    print(f"    Warnings: {analysis.get('warnings', [])}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Texture affects cooking behavior")
    print("2. Functions determine substitutability")
    print("3. Cooking method constrains options")
    print("4. Missing functions may need additional ingredients")
    print("\nNext: Multi-ingredient substitution!")
