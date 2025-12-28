# Problem 11: Sentence Transformers
#
# Learn to use transformer-based embeddings. These models produce contextual
# embeddings that capture richer meaning than static word vectors.
#
# You'll implement:
# 1. MockSentenceEncoder - simulate transformer behavior
# 2. encode_ingredients() - batch ingredient encoding
# 3. encode_recipe() - full recipe encoding with different methods
# 4. RecipeSearchEngine - semantic search for recipes
#
# Example:
#   encoder = MockSentenceEncoder()
#   emb1 = encoder.encode("Add a pinch of salt")  # Contextual
#   recipe_emb = encode_recipe(title, ingredients, instructions, encoder)
#   results = semantic_search("chocolate dessert", recipes, encoder)
#
# Constraints:
#   - Encode returns normalized embeddings
#   - Support concat, average, and weighted recipe encoding
#   - Search uses cosine similarity
#
# ML Relevance: Transformer embeddings are state-of-the-art. They're contextual
# (word meaning depends on surrounding words), pretrained on massive data,
# fine-tunable for specific tasks, and support long sequences.

from typing import List, Dict, Tuple, Union
import numpy as np


class MockSentenceEncoder:
    """Mock sentence encoder that simulates transformer behavior."""
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize mock encoder."""
        self.embedding_dim = embedding_dim
        self._cache = {}
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        """Encode texts into embeddings. Returns (n, dim) or (dim,)."""
        # Your solution here
        pass
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for efficiency."""
        # Your solution here
        pass


def encode_ingredients(ingredients: List[str], encoder: MockSentenceEncoder, add_context: bool = True) -> Dict[str, np.ndarray]:
    """Encode ingredients, optionally with 'ingredient: ' prefix."""
    # Your solution here
    pass


def encode_recipe(title: str, ingredients: List[str], instructions: str, encoder: MockSentenceEncoder, method: str = "concat") -> np.ndarray:
    """Encode recipe using 'concat', 'average', or 'weighted' method."""
    # Your solution here
    pass


def semantic_search(query: str, corpus: List[str], encoder: MockSentenceEncoder, top_k: int = 5) -> List[Tuple[str, float, int]]:
    """Search corpus semantically. Returns (text, similarity, index) tuples."""
    # Your solution here
    pass


def find_similar_recipes(query_recipe: Dict[str, any], recipe_corpus: List[Dict[str, any]], encoder: MockSentenceEncoder, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """Find recipes similar to a query recipe."""
    # Your solution here
    pass


def cluster_by_embeddings(texts: List[str], encoder: MockSentenceEncoder, n_clusters: int = 3) -> Dict[int, List[str]]:
    """Cluster texts using k-means on embeddings."""
    # Your solution here
    pass


def compute_text_similarity(text1: str, text2: str, encoder: MockSentenceEncoder) -> float:
    """Compute cosine similarity between two texts."""
    # Your solution here
    pass


def batch_similarity(queries: List[str], candidates: List[str], encoder: MockSentenceEncoder) -> np.ndarray:
    """Compute pairwise similarities (len(queries), len(candidates))."""
    # Your solution here
    pass


def calculate_fertility(texts: List[str], encoder: MockSentenceEncoder) -> Dict[str, float]:
    """Calculate embedding diversity: avg_pairwise_distance, variance, isotropy."""
    # Your solution here
    pass


class RecipeSearchEngine:
    """Semantic search engine for recipes."""
    
    def __init__(self, encoder: MockSentenceEncoder = None):
        """Initialize search engine."""
        # Your solution here
        pass
    
    def index_recipes(self, recipes: List[Dict[str, any]]) -> None:
        """Index recipes for search."""
        # Your solution here
        pass
    
    def search(self, query: str, top_k: int = 5, filter_by: Dict[str, any] = None) -> List[Tuple[Dict, float]]:
        """Search for recipes matching query."""
        # Your solution here
        pass
    
    def find_similar(self, recipe_idx: int, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find recipes similar to an indexed recipe."""
        # Your solution here
        pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing MockSentenceEncoder...")
    
    encoder = MockSentenceEncoder(embedding_dim=384)
    assert encoder.embedding_dim == 384, "Test 1 failed"
    print("  ✓ Encoder initialized")
    
    emb = encoder.encode("hello world")
    assert emb.shape == (384,), f"Test 2a failed: {emb.shape}"
    assert abs(np.linalg.norm(emb) - 1.0) < 0.01, "Test 2b: should be normalized"
    print("  ✓ Single text encoding works")
    
    embs = encoder.encode(["hello", "world", "test"])
    assert embs.shape == (3, 384), f"Test 3 failed: {embs.shape}"
    print("  ✓ Batch encoding works")
    
    emb1 = encoder.encode("test sentence")
    emb2 = encoder.encode("test sentence")
    assert np.allclose(emb1, emb2), "Test 4 failed: should be deterministic"
    print("  ✓ Encoding is deterministic")
    
    emb1 = encoder.encode("apples")
    emb2 = encoder.encode("oranges")
    assert not np.allclose(emb1, emb2), "Test 5 failed"
    print("  ✓ Different texts get different embeddings")
    
    print("\nTesting encode_ingredients...")
    
    ingredients = ["butter", "flour", "sugar"]
    embeddings = encode_ingredients(ingredients, encoder)
    assert len(embeddings) == 3, "Test 6a failed"
    assert "butter" in embeddings, "Test 6b failed"
    assert embeddings["butter"].shape == (384,), "Test 6c failed"
    print("  ✓ Ingredient encoding works")
    
    print("\nTesting encode_recipe...")
    
    emb = encode_recipe(
        title="Chocolate Cake",
        ingredients=["flour", "sugar", "cocoa", "eggs"],
        instructions="Mix all ingredients and bake at 350F for 30 minutes.",
        encoder=encoder,
        method="concat"
    )
    assert emb.shape == (384,), f"Test 7 failed: {emb.shape}"
    print("  ✓ Recipe encoding works")
    
    emb_avg = encode_recipe("Cake", ["flour"], "Bake it.", encoder, method="average")
    emb_wt = encode_recipe("Cake", ["flour"], "Bake it.", encoder, method="weighted")
    assert emb_avg.shape == (384,), "Test 8a failed"
    assert emb_wt.shape == (384,), "Test 8b failed"
    print("  ✓ Different encoding methods work")
    
    print("\nTesting semantic_search...")
    
    corpus = [
        "How to make chocolate cake",
        "Grilled chicken recipe",
        "Vegetable soup instructions",
        "Baking a birthday cake"
    ]
    results = semantic_search("cake recipe", corpus, encoder, top_k=2)
    assert len(results) == 2, f"Test 9a failed: {len(results)}"
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results), "Test 9b failed"
    print("  ✓ Semantic search works")
    
    print("\nTesting find_similar_recipes...")
    
    recipes = [
        {"title": "Chocolate Cake", "ingredients": ["flour", "cocoa"], "instructions": "Bake"},
        {"title": "Vanilla Cake", "ingredients": ["flour", "vanilla"], "instructions": "Bake"},
        {"title": "Chicken Soup", "ingredients": ["chicken", "broth"], "instructions": "Simmer"}
    ]
    similar = find_similar_recipes(recipes[0], recipes[1:], encoder, top_k=1)
    assert len(similar) == 1, f"Test 10 failed: {len(similar)}"
    print("  ✓ Find similar recipes works")
    
    print("\nTesting cluster_by_embeddings...")
    
    texts = ["cake", "cookies", "chicken", "beef", "soup", "stew"]
    clusters = cluster_by_embeddings(texts, encoder, n_clusters=2)
    assert len(clusters) == 2, f"Test 11a failed: {len(clusters)}"
    total = sum(len(v) for v in clusters.values())
    assert total == len(texts), f"Test 11b failed: {total}"
    print("  ✓ Clustering works")
    
    print("\nTesting compute_text_similarity...")
    
    sim = compute_text_similarity("hello world", "hello world", encoder)
    assert abs(sim - 1.0) < 0.01, f"Test 12a failed: {sim}"
    sim = compute_text_similarity("cats", "dogs", encoder)
    assert -1 <= sim <= 1, f"Test 12b failed: {sim}"
    print("  ✓ Text similarity works")
    
    print("\nTesting batch_similarity...")
    
    queries = ["cake", "soup"]
    candidates = ["chocolate cake", "chicken soup", "grilled fish"]
    sims = batch_similarity(queries, candidates, encoder)
    assert sims.shape == (2, 3), f"Test 13 failed: {sims.shape}"
    print("  ✓ Batch similarity works")
    
    print("\nTesting calculate_fertility...")
    
    texts = ["apple", "banana", "car", "house", "computer"]
    fertility = calculate_fertility(texts, encoder)
    assert "avg_pairwise_distance" in fertility, "Test 14a failed"
    assert fertility["avg_pairwise_distance"] >= 0, "Test 14b failed"
    print("  ✓ Fertility calculation works")
    
    print("\nTesting RecipeSearchEngine...")
    
    engine = RecipeSearchEngine(encoder)
    recipes = [
        {"title": "Chocolate Cake", "ingredients": ["flour", "cocoa"], "instructions": "Mix and bake"},
        {"title": "Chicken Soup", "ingredients": ["chicken", "vegetables"], "instructions": "Simmer for hours"},
        {"title": "Pasta Primavera", "ingredients": ["pasta", "vegetables"], "instructions": "Cook pasta and toss"}
    ]
    engine.index_recipes(recipes)
    
    results = engine.search("dessert with chocolate", top_k=2)
    assert len(results) <= 2, f"Test 15a failed: {len(results)}"
    print("  ✓ Recipe search engine works")
    
    similar = engine.find_similar(0, top_k=1)
    assert len(similar) <= 1, f"Test 16 failed: {len(similar)}"
    print("  ✓ Find similar indexed recipes works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
