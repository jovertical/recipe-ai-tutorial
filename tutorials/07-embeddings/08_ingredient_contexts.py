# Problem 8: Ingredient Contexts
#
# Extract contextual information about ingredients from recipe text. The
# context (surrounding words) tells us how ingredients are used, which
# helps create better embeddings.
#
# You'll implement:
# 1. extract_ingredient_contexts() - get surrounding words
# 2. context_window() - fixed-size context extraction
# 3. weighted_context() - closer words matter more
# 4. Build context-based ingredient representations
#
# Example:
#   Recipe: "Melt the butter in a pan, then add minced garlic"
#   Context for "butter": ["melt", "pan", "add"]
#   Context for "garlic": ["minced", "add", "butter"]
#   We learn: butter is melted, garlic is minced
#
# Constraints:
#   - Context should exclude the ingredient itself
#   - Weighted context uses 1/distance for weights
#   - Handle edge cases at start/end of text
#
# ML Relevance: Context is the key insight behind word embeddings. "You shall
# know a word by the company it keeps" (Firth, 1957). Ingredients with similar
# contexts have similar uses. Cooking actions are especially informative.

from typing import List, Dict, Tuple, Set
import numpy as np
from collections import Counter, defaultdict
import re


def tokenize_recipe(recipe_text: str) -> List[str]:
    """Tokenize recipe text into lowercase words."""
    # Your solution here
    pass


def extract_ingredient_contexts(recipe_text: str, ingredients: Set[str], window_size: int = 3) -> Dict[str, List[str]]:
    """Extract context words around each ingredient mention."""
    # Your solution here
    pass


def context_window(tokens: List[str], center_position: int, window_size: int) -> List[str]:
    """Get context words around a position (excluding center)."""
    # Your solution here
    pass


def weighted_context(tokens: List[str], center_position: int, max_window: int = 5) -> List[Tuple[str, float]]:
    """Get context words with weight = 1/distance."""
    # Your solution here
    pass


def build_context_vocabulary(recipes: List[str], ingredients: Set[str], window_size: int = 3, min_count: int = 2) -> Tuple[Dict[str, int], Counter]:
    """Build vocabulary of context words and count frequencies."""
    # Your solution here
    pass


def build_ingredient_context_matrix(recipes: List[str], ingredients: Set[str], context_vocab: Dict[str, int], window_size: int = 3, use_weights: bool = False) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build matrix[ingredient, context_word] = association count."""
    # Your solution here
    pass


def context_similarity(ing1: str, ing2: str, context_matrix: np.ndarray, ingredient_vocab: Dict[str, int]) -> float:
    """Compute cosine similarity of context distributions."""
    # Your solution here
    pass


def find_ingredients_by_context(context_words: List[str], context_matrix: np.ndarray, ingredient_vocab: Dict[str, int], context_vocab: Dict[str, int], top_k: int = 5) -> List[Tuple[str, float]]:
    """Find ingredients that match given context words."""
    # Your solution here
    pass


def extract_cooking_actions(recipe_text: str, action_words: Set[str] = None) -> List[str]:
    """Extract cooking action verbs from recipe text."""
    if action_words is None:
        action_words = {
            "chop", "dice", "mince", "slice", "cut", "peel",
            "mix", "stir", "whisk", "beat", "fold", "combine",
            "cook", "bake", "roast", "fry", "sauté", "grill",
            "boil", "simmer", "steam", "poach", "braise",
            "add", "pour", "sprinkle", "drizzle", "season",
            "melt", "heat", "cool", "chill", "freeze"
        }
    # Your solution here
    pass


def ingredient_action_associations(recipes: List[str], ingredients: Set[str]) -> Dict[str, Counter]:
    """Find which cooking actions are associated with each ingredient."""
    # Your solution here
    pass


def context_based_embeddings(context_matrix: np.ndarray, embedding_dim: int = 50) -> np.ndarray:
    """Create ingredient embeddings from context matrix using SVD."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing tokenize_recipe...")
    
    # Test 1: Basic tokenization
    text = "Melt the butter in a pan!"
    tokens = tokenize_recipe(text)
    assert tokens == ["melt", "the", "butter", "in", "a", "pan"], f"Test 1 failed: {tokens}"
    print("  ✓ Tokenization works")
    
    print("\nTesting context_window...")
    
    # Test 2: Basic window
    tokens = ["a", "b", "c", "d", "e"]
    context = context_window(tokens, 2, 2)
    assert set(context) == {"a", "b", "d", "e"}, f"Test 2a failed: {context}"
    assert "c" not in context, "Test 2b: center word should be excluded"
    print("  ✓ Context window works")
    
    # Test 3: Edge cases
    context = context_window(tokens, 0, 2)
    assert "a" not in context, "Test 3a: center excluded"
    context = context_window(tokens, 4, 2)
    assert "e" not in context, "Test 3b: center excluded"
    print("  ✓ Edge cases handled")
    
    print("\nTesting weighted_context...")
    
    # Test 4: Weighted context
    tokens = ["a", "b", "c", "d", "e"]
    weighted = weighted_context(tokens, 2, max_window=2)
    words = [w for w, _ in weighted]
    weights = [wt for _, wt in weighted]
    assert "b" in words and "d" in words, f"Test 4a failed: {words}"
    for w, wt in weighted:
        if w in ["b", "d"]:
            assert wt == 1.0, f"Test 4b failed: {w} has weight {wt}"
    print("  ✓ Weighted context works")
    
    print("\nTesting extract_ingredient_contexts...")
    
    # Test 5: Extract contexts
    text = "melt the butter in a pan then add the garlic"
    ingredients = {"butter", "garlic"}
    contexts = extract_ingredient_contexts(text, ingredients, window_size=2)
    assert "butter" in contexts, "Test 5a failed"
    assert "garlic" in contexts, "Test 5b failed"
    butter_context = contexts["butter"]
    assert "melt" in butter_context or "the" in butter_context, f"Test 5c failed: {butter_context}"
    print("  ✓ Context extraction works")
    
    print("\nTesting build_context_vocabulary...")
    
    # Test 6: Context vocabulary
    recipes = [
        "melt the butter in a pan",
        "add minced garlic to the butter",
        "sauté onions in butter until soft"
    ]
    ingredients = {"butter", "garlic", "onions"}
    vocab, counts = build_context_vocabulary(recipes, ingredients, window_size=2, min_count=1)
    assert len(vocab) > 0, "Test 6a failed"
    assert len(counts) > 0, "Test 6b failed"
    print("  ✓ Context vocabulary built")
    
    print("\nTesting build_ingredient_context_matrix...")
    
    # Test 7: Context matrix
    matrix, ing_vocab = build_ingredient_context_matrix(recipes, ingredients, vocab, window_size=2)
    assert matrix.shape[0] == len(ing_vocab), "Test 7a failed"
    assert matrix.shape[1] == len(vocab), "Test 7b failed"
    print("  ✓ Context matrix built")
    
    print("\nTesting context_similarity...")
    
    # Test 8: Context similarity
    sim = context_similarity("butter", "butter", matrix, ing_vocab)
    assert abs(sim - 1.0) < 0.01, f"Test 8a failed: self-similarity should be 1, got {sim}"
    print("  ✓ Context similarity works")
    
    print("\nTesting find_ingredients_by_context...")
    
    # Test 9: Find by context
    results = find_ingredients_by_context(["melt"], matrix, ing_vocab, vocab, top_k=2)
    assert len(results) <= 2, f"Test 9a failed: {len(results)}"
    assert all(isinstance(r, tuple) for r in results), "Test 9b failed"
    print("  ✓ Find by context works")
    
    print("\nTesting extract_cooking_actions...")
    
    # Test 10: Extract actions
    text = "Sauté the onions, then add garlic and stir well"
    actions = extract_cooking_actions(text)
    expected = {"sauté", "add", "stir"}
    assert set(actions) == expected or set(actions).issubset(expected), f"Test 10 failed: {actions}"
    print("  ✓ Cooking actions extracted")
    
    print("\nTesting ingredient_action_associations...")
    
    # Test 11: Action associations
    recipes = [
        "mince the garlic and sauté briefly",
        "add crushed garlic to the pan",
        "stir in minced garlic"
    ]
    assoc = ingredient_action_associations(recipes, {"garlic"})
    assert "garlic" in assoc, "Test 11a failed"
    assert isinstance(assoc["garlic"], Counter), "Test 11b failed"
    print("  ✓ Action associations computed")
    
    print("\nTesting context_based_embeddings...")
    
    # Test 12: Create embeddings
    embeddings = context_based_embeddings(matrix, embedding_dim=10)
    assert embeddings.shape[0] == len(ing_vocab), f"Test 12a failed: {embeddings.shape}"
    assert embeddings.shape[1] == 10, f"Test 12b failed: {embeddings.shape}"
    print("  ✓ Context embeddings created")
    
    # Test 13: Full pipeline
    print("\nTesting full pipeline...")
    recipes = [
        "chop the onions finely and sauté in butter",
        "mince the garlic and add to the melted butter",
        "sauté diced onions until golden brown",
        "cook minced garlic in olive oil",
        "stir in chopped onions and crushed garlic"
    ]
    ingredients = {"onions", "garlic", "butter", "olive oil"}
    
    vocab, counts = build_context_vocabulary(recipes, ingredients, window_size=3, min_count=1)
    matrix, ing_vocab = build_ingredient_context_matrix(recipes, ingredients, vocab, window_size=3)
    embeddings = context_based_embeddings(matrix, embedding_dim=5)
    
    print(f"  Ingredients: {list(ing_vocab.keys())}")
    print("  ✓ Full pipeline works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
