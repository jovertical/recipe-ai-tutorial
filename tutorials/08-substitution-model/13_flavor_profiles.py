# Problem 13: Flavor Profile Matching
#
# Build a system that considers flavor profiles when suggesting substitutes
# to maintain recipe taste balance.
#
# Example:
#   matcher = FlavorMatcher(flavor_embeddings)
#   subs = matcher.find_similar_flavor("lemon_juice", recipe_context)
#   # Returns: ["lime_juice", "vinegar", "citric_acid"]
#
# ML Relevance: Multi-dimensional similarity (flavor, texture, function)
# requires learning composite embeddings or multi-task models.
#
# Your Task:
#   1. Implement flavor profile representation
#   2. Implement flavor-aware similarity scoring
#   3. Implement recipe flavor balance analysis
#   4. Combine flavor matching with substitutability


from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FlavorProfile:
    """Flavor profile of an ingredient."""
    name: str
    sweet: float = 0.0
    salty: float = 0.0
    sour: float = 0.0
    bitter: float = 0.0
    umami: float = 0.0
    spicy: float = 0.0
    # Additional flavor notes
    aromatic: float = 0.0
    fatty: float = 0.0
    earthy: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector."""
        return np.array([
            self.sweet, self.salty, self.sour, self.bitter,
            self.umami, self.spicy, self.aromatic, self.fatty, self.earthy
        ])
    
    @classmethod
    def from_vector(cls, name: str, vec: np.ndarray) -> 'FlavorProfile':
        """Create from numpy vector."""
        return cls(
            name=name,
            sweet=vec[0], salty=vec[1], sour=vec[2], bitter=vec[3],
            umami=vec[4], spicy=vec[5], aromatic=vec[6], fatty=vec[7], earthy=vec[8]
        )


# Sample flavor profiles
FLAVOR_DB = {
    "butter": FlavorProfile("butter", sweet=0.1, salty=0.2, umami=0.1, fatty=0.9),
    "margarine": FlavorProfile("margarine", salty=0.2, fatty=0.8),
    "olive_oil": FlavorProfile("olive_oil", bitter=0.1, fatty=0.9, aromatic=0.3),
    "coconut_oil": FlavorProfile("coconut_oil", sweet=0.2, fatty=0.9, aromatic=0.4),
    "garlic": FlavorProfile("garlic", spicy=0.3, umami=0.4, aromatic=0.8),
    "onion": FlavorProfile("onion", sweet=0.3, umami=0.3, aromatic=0.5),
    "ginger": FlavorProfile("ginger", spicy=0.5, aromatic=0.7, sweet=0.1),
    "lemon": FlavorProfile("lemon", sour=0.9, sweet=0.1, aromatic=0.4),
    "lime": FlavorProfile("lime", sour=0.85, sweet=0.05, aromatic=0.3),
    "honey": FlavorProfile("honey", sweet=0.95, aromatic=0.3),
    "maple_syrup": FlavorProfile("maple_syrup", sweet=0.9, aromatic=0.4, earthy=0.1),
    "sugar": FlavorProfile("sugar", sweet=1.0),
    "soy_sauce": FlavorProfile("soy_sauce", salty=0.8, umami=0.9),
    "fish_sauce": FlavorProfile("fish_sauce", salty=0.7, umami=0.95),
    "miso": FlavorProfile("miso", salty=0.7, umami=0.9, earthy=0.3),
}


def flavor_similarity(
    profile1: FlavorProfile,
    profile2: FlavorProfile,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute flavor similarity between two profiles.
    
    Args:
        profile1: First flavor profile
        profile2: Second flavor profile
        weights: Optional importance weights for each dimension
        
    Returns:
        Similarity score (0 to 1)
    """
    # Your solution here
    pass


def find_flavor_substitutes(
    ingredient: str,
    candidates: List[str],
    min_similarity: float = 0.5,
    weights: Dict[str, float] = None
) -> List[Tuple[str, float]]:
    """
    Find substitutes with similar flavor profiles.
    
    Args:
        ingredient: Original ingredient
        candidates: Possible substitutes
        min_similarity: Minimum similarity threshold
        weights: Importance weights for flavor dimensions
        
    Returns:
        List of (candidate, similarity) sorted by similarity
    """
    # Your solution here
    pass


def flavor_distance(
    profile1: FlavorProfile,
    profile2: FlavorProfile,
    metric: str = "euclidean"
) -> float:
    """
    Compute flavor distance.
    
    Args:
        profile1: First profile
        profile2: Second profile
        metric: "euclidean", "manhattan", or "chebyshev"
    """
    # Your solution here
    pass


class FlavorMatcher:
    """Match ingredients by flavor profile."""
    
    def __init__(self, flavor_db: Dict[str, FlavorProfile] = None):
        self.db = flavor_db or FLAVOR_DB
    
    def get_profile(self, ingredient: str) -> Optional[FlavorProfile]:
        """Get flavor profile for ingredient."""
        return self.db.get(ingredient)
    
    def find_similar(
        self,
        ingredient: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """Find ingredients with similar flavor."""
        # Your solution here
        pass
    
    def find_by_flavor(
        self,
        target_profile: Dict[str, float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find ingredients matching a target flavor profile."""
        # Your solution here
        pass
    
    def get_dominant_flavors(
        self,
        ingredient: str,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Get dominant flavor characteristics."""
        # Your solution here
        pass


class FlavorComplementFinder:
    """Find ingredients that complement each other."""
    
    def __init__(self, flavor_db: Dict[str, FlavorProfile] = None):
        self.db = flavor_db or FLAVOR_DB
        # Flavor pairing rules
        self.complements = {
            "sweet": ["salty", "sour", "bitter"],
            "salty": ["sweet", "sour"],
            "sour": ["sweet", "fatty"],
            "bitter": ["sweet", "fatty"],
            "umami": ["sour", "sweet"],
            "spicy": ["sweet", "fatty", "sour"],
        }
    
    def find_complements(
        self,
        ingredient: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Find complementary ingredients.
        
        Returns:
            List of (ingredient, score, reason) tuples
        """
        # Your solution here
        pass
    
    def compute_complement_score(
        self,
        profile1: FlavorProfile,
        profile2: FlavorProfile
    ) -> float:
        """Compute how well two profiles complement each other."""
        # Your solution here
        pass


class FlavorBalancer:
    """Balance flavors in a recipe."""
    
    def __init__(self, flavor_db: Dict[str, FlavorProfile] = None):
        self.db = flavor_db or FLAVOR_DB
    
    def analyze_recipe(
        self,
        ingredients: List[str]
    ) -> Dict[str, float]:
        """
        Analyze overall flavor profile of a recipe.
        
        Returns combined flavor intensities.
        """
        # Your solution here
        pass
    
    def suggest_balancing_ingredient(
        self,
        current_ingredients: List[str],
        target_balance: Dict[str, float] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Suggest ingredients to balance the recipe.
        
        Returns:
            List of (ingredient, score, reason)
        """
        # Your solution here
        pass
    
    def is_balanced(
        self,
        ingredients: List[str],
        tolerance: float = 0.3
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if recipe flavors are balanced."""
        # Your solution here
        pass


class FlavorAwareSubstitutor:
    """Substitution with flavor matching."""
    
    def __init__(
        self,
        flavor_db: Dict[str, FlavorProfile] = None,
        embeddings: Dict[str, np.ndarray] = None
    ):
        self.flavor_db = flavor_db or FLAVOR_DB
        self.embeddings = embeddings or {}
        self.matcher = FlavorMatcher(self.flavor_db)
    
    def substitute(
        self,
        ingredient: str,
        recipe_context: List[str] = None,
        flavor_weight: float = 0.5,
        embedding_weight: float = 0.5,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find substitutes considering both flavor and embedding similarity.
        
        Args:
            ingredient: Ingredient to substitute
            recipe_context: Other ingredients in recipe
            flavor_weight: Weight for flavor similarity
            embedding_weight: Weight for embedding similarity
            top_k: Number of results
            
        Returns:
            List of (substitute, score, details)
        """
        # Your solution here
        pass
    
    def score_flavor_impact(
        self,
        original: str,
        substitute: str,
        recipe_ingredients: List[str]
    ) -> Dict[str, any]:
        """
        Score the flavor impact of a substitution on the recipe.
        """
        # Your solution here
        pass


def blend_profiles(
    profiles: List[FlavorProfile],
    weights: List[float] = None
) -> FlavorProfile:
    """
    Blend multiple flavor profiles.
    
    Args:
        profiles: List of profiles to blend
        weights: Optional weights for each profile
        
    Returns:
        Blended flavor profile
    """
    # Your solution here
    pass


def visualize_flavor_profile(profile: FlavorProfile) -> str:
    """Create text visualization of flavor profile."""
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing FlavorProfile...")
    
    # Test 1: Profile creation
    butter = FLAVOR_DB["butter"]
    assert butter.fatty == 0.9, "Test 1a failed"
    assert butter.sweet == 0.1, "Test 1b failed"
    print(f"  ✓ Butter profile: fatty={butter.fatty}, sweet={butter.sweet}")
    
    # Test 2: To vector
    vec = butter.to_vector()
    assert len(vec) == 9, f"Test 2 failed: {len(vec)}"
    print(f"  ✓ Profile vector length: {len(vec)}")
    
    # Test 3: From vector
    reconstructed = FlavorProfile.from_vector("butter", vec)
    assert reconstructed.fatty == butter.fatty, "Test 3 failed"
    print("  ✓ Profile reconstruction works")
    
    print("\nTesting flavor_similarity...")
    
    # Test 4: Similar profiles
    butter = FLAVOR_DB["butter"]
    margarine = FLAVOR_DB["margarine"]
    sim = flavor_similarity(butter, margarine)
    assert 0 <= sim <= 1, f"Test 4a failed: {sim}"
    print(f"  ✓ Butter-margarine similarity: {sim:.3f}")
    
    # Test 5: Different profiles
    garlic = FLAVOR_DB["garlic"]
    sim_diff = flavor_similarity(butter, garlic)
    assert sim_diff < sim, f"Test 5 failed: butter-garlic should be less similar"
    print(f"  ✓ Butter-garlic similarity: {sim_diff:.3f}")
    
    print("\nTesting find_flavor_substitutes...")
    
    # Test 6: Find substitutes
    candidates = list(FLAVOR_DB.keys())
    subs = find_flavor_substitutes("butter", candidates, min_similarity=0.3)
    assert len(subs) > 0, "Test 6a failed"
    assert "margarine" in [s[0] for s in subs], "Test 6b failed"
    print(f"  ✓ Flavor substitutes for butter: {[s[0] for s in subs[:3]]}")
    
    print("\nTesting FlavorMatcher...")
    
    matcher = FlavorMatcher()
    
    # Test 7: Find similar
    similar = matcher.find_similar("butter", top_k=3)
    assert len(similar) <= 3, f"Test 7 failed: {len(similar)}"
    print(f"  ✓ Similar to butter: {[s[0] for s in similar]}")
    
    # Test 8: Find by flavor
    target = {"sweet": 0.8, "aromatic": 0.5}
    matches = matcher.find_by_flavor(target, top_k=3)
    assert len(matches) > 0, "Test 8 failed"
    print(f"  ✓ Sweet+aromatic matches: {[m[0] for m in matches]}")
    
    # Test 9: Dominant flavors
    dominant = matcher.get_dominant_flavors("garlic", threshold=0.3)
    assert len(dominant) > 0, "Test 9a failed"
    assert any(f[0] == "aromatic" for f in dominant), "Test 9b failed"
    print(f"  ✓ Garlic dominant flavors: {dominant}")
    
    print("\nTesting FlavorComplementFinder...")
    
    finder = FlavorComplementFinder()
    
    # Test 10: Find complements
    complements = finder.find_complements("lemon", top_k=3)
    assert len(complements) > 0, "Test 10 failed"
    print(f"  ✓ Lemon complements: {[(c[0], c[2]) for c in complements]}")
    
    # Test 11: Complement score
    lemon = FLAVOR_DB["lemon"]
    honey = FLAVOR_DB["honey"]
    score = finder.compute_complement_score(lemon, honey)
    assert score > 0, f"Test 11 failed: {score}"
    print(f"  ✓ Lemon-honey complement score: {score:.3f}")
    
    print("\nTesting FlavorBalancer...")
    
    balancer = FlavorBalancer()
    
    # Test 12: Analyze recipe
    recipe_ings = ["butter", "garlic", "lemon"]
    analysis = balancer.analyze_recipe(recipe_ings)
    assert "fatty" in analysis, "Test 12a failed"
    assert "sour" in analysis, "Test 12b failed"
    print(f"  ✓ Recipe analysis: {analysis}")
    
    # Test 13: Suggest balancing
    suggestions = balancer.suggest_balancing_ingredient(recipe_ings)
    assert len(suggestions) > 0, "Test 13 failed"
    print(f"  ✓ Balancing suggestions: {[(s[0], s[2]) for s in suggestions[:2]]}")
    
    # Test 14: Check balance
    is_bal, profile = balancer.is_balanced(recipe_ings)
    print(f"  ✓ Recipe balanced: {is_bal}")
    
    print("\nTesting FlavorAwareSubstitutor...")
    
    # Create mock embeddings
    embeddings = {name: np.random.randn(64).astype(np.float32) for name in FLAVOR_DB.keys()}
    substitutor = FlavorAwareSubstitutor(FLAVOR_DB, embeddings)
    
    # Test 15: Substitute with context
    subs = substitutor.substitute(
        "butter",
        recipe_context=["flour", "sugar"],
        flavor_weight=0.6,
        embedding_weight=0.4,
        top_k=3
    )
    assert len(subs) > 0, "Test 15 failed"
    print(f"  ✓ Context-aware substitutes: {[s[0] for s in subs]}")
    
    # Test 16: Score flavor impact
    impact = substitutor.score_flavor_impact("butter", "olive_oil", ["garlic", "pasta"])
    assert "flavor_change" in impact, "Test 16 failed"
    print(f"  ✓ Flavor impact: {impact}")
    
    print("\nTesting blend_profiles...")
    
    # Test 17: Blend profiles
    profiles = [FLAVOR_DB["butter"], FLAVOR_DB["garlic"]]
    blended = blend_profiles(profiles)
    assert blended is not None, "Test 17 failed"
    print(f"  ✓ Blended fatty: {blended.fatty:.2f}, aromatic: {blended.aromatic:.2f}")
    
    print("\nTesting visualize_flavor_profile...")
    
    # Test 18: Visualization
    viz = visualize_flavor_profile(FLAVOR_DB["garlic"])
    assert len(viz) > 0, "Test 18 failed"
    print(f"  ✓ Garlic flavor visualization:\n{viz}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Flavor profiles are multi-dimensional")
    print("2. Substitutes should match flavor characteristics")
    print("3. Complements balance opposing flavors")
    print("4. Recipe context affects substitution choices")
    print("\nNext: Texture and function matching!")
