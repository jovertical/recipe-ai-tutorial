# Problem 4: Embedding-Based Retrieval
#
# Use ingredient embeddings from Part 7 to find substitutes based on semantic
# similarity. This leverages learned representations to discover substitutions
# beyond explicit rules.
#
# Example:
#   retriever = EmbeddingRetriever(embeddings)
#   substitutes = retriever.find_similar("butter", top_k=5)
#   # Returns: [("margarine", 0.92), ("shortening", 0.85), ("lard", 0.78), ...]
#
# ML Relevance: Embedding-based retrieval generalizes beyond explicit rules,
# discovers novel substitutions, scales to large ingredient sets, and forms
# the basis for more complex models. This bridges Part 7 (embeddings) and
# Part 8 (substitution).
#
# Your Task:
#   1. Implement EmbeddingRetriever for similarity search
#   2. Implement approximate nearest neighbor search
#   3. Implement filtered retrieval (by category, dietary tags)
#   4. Implement hybrid retrieval (rules + embeddings)

from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import heapq


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


@dataclass
class RetrievalResult:
    """Result from embedding retrieval."""
    ingredient: str
    score: float
    embedding: np.ndarray = None
    metadata: Dict = None
    
    def __repr__(self):
        return f"{self.ingredient} (score={self.score:.3f})"


class EmbeddingRetriever:
    """Retrieve similar ingredients using embeddings."""
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        metadata: Dict[str, Dict] = None,
        similarity_fn: str = "cosine"
    ):
        """
        Initialize retriever.
        
        Hints:
        1. Store embeddings and metadata
        2. Pre-compute normalized embeddings for cosine similarity
        3. Build ingredient list for indexing
        """
        # Your solution here
        pass
    
    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[RetrievalResult]:
        """Find most similar ingredients to query."""
        # Your solution here
        pass
    
    def find_similar_to_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Find ingredients similar to a query vector. Useful for analogy-based queries."""
        # Your solution here
        pass
    
    def find_in_range(
        self,
        query: str,
        min_similarity: float = 0.5,
        max_similarity: float = 1.0
    ) -> List[RetrievalResult]:
        """Find ingredients within similarity range. Useful for "moderately similar" alternatives."""
        # Your solution here
        pass
    
    def batch_find_similar(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> Dict[str, List[RetrievalResult]]:
        """Find similar ingredients for multiple queries efficiently."""
        # Your solution here
        pass
    
    def get_embedding(self, ingredient: str) -> Optional[np.ndarray]:
        """Get embedding for an ingredient."""
        # Your solution here
        pass
    
    def compute_similarity(self, ing1: str, ing2: str) -> float:
        """Compute similarity between two ingredients."""
        # Your solution here
        pass
    
    def get_all_ingredients(self) -> List[str]:
        """Get list of all indexed ingredients."""
        # Your solution here
        pass


class FilteredRetriever:
    """Retriever with filtering capabilities."""
    
    def __init__(
        self,
        base_retriever: EmbeddingRetriever,
        categories: Dict[str, str] = None,
        dietary_tags: Dict[str, List[str]] = None
    ):
        """Initialize filtered retriever."""
        # Your solution here
        pass
    
    def find_similar_in_category(
        self,
        query: str,
        category: str = None,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Find similar ingredients in same or specified category."""
        # Your solution here
        pass
    
    def find_with_dietary_tags(
        self,
        query: str,
        required_tags: List[str],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Find similar ingredients with required dietary tags."""
        # Your solution here
        pass
    
    def find_excluding(
        self,
        query: str,
        exclude_ingredients: Set[str],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Find similar ingredients excluding specified ones. Useful when certain ingredients are unavailable."""
        # Your solution here
        pass


class ApproximateRetriever:
    """
    Approximate nearest neighbor retrieval for large ingredient sets.
    Uses locality-sensitive hashing (LSH) for fast retrieval.
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        num_tables: int = 10,
        num_projections: int = 8
    ):
        """
        Initialize approximate retriever.
        
        Hints:
        1. Generate random projection vectors
        2. Hash each embedding into buckets
        3. Build inverted index
        """
        # Your solution here
        pass
    
    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        """Compute LSH hash for a vector."""
        # Your solution here
        pass
    
    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        num_candidates: int = 100
    ) -> List[RetrievalResult]:
        """
        Find similar ingredients using approximate search.
        
        Hints:
        1. Hash query into each table
        2. Collect candidates from matching buckets
        3. Re-rank candidates by exact similarity
        4. Return top_k
        """
        # Your solution here
        pass
    
    def find_candidates(self, query: str) -> Set[str]:
        """Get candidate ingredients from hash buckets."""
        # Your solution here
        pass


class HybridRetriever:
    """Combine rule-based and embedding-based retrieval."""
    
    def __init__(
        self,
        embedding_retriever: EmbeddingRetriever,
        rule_substitutes: Dict[str, List[str]],
        rule_weight: float = 0.6,
        embedding_weight: float = 0.4
    ):
        """Initialize hybrid retriever."""
        # Your solution here
        pass
    
    def find_substitutes(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Find substitutes using hybrid approach.
        
        Combines:
        - Rule-based: Known substitutes get high base score
        - Embedding-based: Similar ingredients by embedding
        
        Hints:
        1. Get rule-based substitutes (if any)
        2. Get embedding-based results
        3. Merge scores: final = rule_weight * rule_score + embedding_weight * emb_score
        4. Sort and return top_k
        """
        # Your solution here
        pass
    
    def explain_ranking(
        self,
        query: str,
        result: str
    ) -> Dict[str, float]:
        """Explain why a result was ranked highly."""
        # Your solution here
        pass


def build_index(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """Build matrix index for efficient batch operations."""
    # Your solution here
    pass


def batch_cosine_similarity(
    query: np.ndarray,
    index: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between query and all index vectors."""
    # Your solution here
    pass


def analogy_query(
    a: str,
    b: str,
    c: str,
    retriever: EmbeddingRetriever,
    top_k: int = 5
) -> List[RetrievalResult]:
    """
    Find D such that A:B :: C:D using vector arithmetic: D = B - A + C
    
    Example:
      # butter:margarine :: cream:?
      analogy_query("butter", "margarine", "cream", retriever)
      # Returns: [coconut_cream, cashew_cream, ...]
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    # Create mock embeddings with semantic structure
    categories = {
        "butter": "fat", "margarine": "fat", "coconut_oil": "fat", 
        "olive_oil": "fat", "lard": "fat",
        "milk": "dairy", "cream": "dairy", "yogurt": "dairy",
        "oat_milk": "milk_alt", "almond_milk": "milk_alt", "soy_milk": "milk_alt",
        "flour": "grain", "oat_flour": "grain", "almond_flour": "grain",
        "sugar": "sweetener", "honey": "sweetener", "maple_syrup": "sweetener",
        "egg": "protein", "tofu": "protein", "tempeh": "protein",
        "garlic": "aromatic", "onion": "aromatic", "ginger": "aromatic",
    }
    
    dietary_tags = {
        "butter": ["vegetarian"],
        "margarine": ["vegan", "dairy_free"],
        "coconut_oil": ["vegan", "dairy_free"],
        "olive_oil": ["vegan", "dairy_free"],
        "oat_milk": ["vegan", "dairy_free", "nut_free"],
        "almond_milk": ["vegan", "dairy_free"],
        "soy_milk": ["vegan", "dairy_free", "nut_free"],
        "honey": ["vegetarian"],
        "maple_syrup": ["vegan"],
        "tofu": ["vegan"],
        "tempeh": ["vegan"],
    }
    
    # Create embeddings where same-category items are similar
    embeddings = {}
    category_vectors = {}
    for cat in set(categories.values()):
        category_vectors[cat] = np.random.randn(64).astype(np.float32)
        category_vectors[cat] /= np.linalg.norm(category_vectors[cat])
    
    for ing, cat in categories.items():
        emb = category_vectors[cat] + np.random.randn(64) * 0.2
        emb = emb / np.linalg.norm(emb)
        embeddings[ing] = emb.astype(np.float32)
    
    print("Testing EmbeddingRetriever...")
    
    # Test 1: Initialize retriever
    retriever = EmbeddingRetriever(embeddings)
    assert retriever.get_all_ingredients() is not None, "Test 1 failed"
    print(f"  ✓ Retriever initialized with {len(retriever.get_all_ingredients())} ingredients")
    
    # Test 2: Find similar
    results = retriever.find_similar("butter", top_k=5)
    assert len(results) == 5, f"Test 2a failed: {len(results)}"
    assert all(isinstance(r, RetrievalResult) for r in results), "Test 2b failed"
    assert results[0].ingredient != "butter", "Test 2c failed: should exclude self"
    print(f"  ✓ Top 5 similar to butter: {[r.ingredient for r in results]}")
    
    # Test 3: Similar ingredients should be from same category
    top_result = results[0]
    assert categories[top_result.ingredient] == "fat", f"Test 3 failed: {top_result.ingredient}"
    print(f"  ✓ Most similar is from same category: {top_result.ingredient}")
    
    # Test 4: Get embedding
    emb = retriever.get_embedding("butter")
    assert emb is not None, "Test 4a failed"
    assert emb.shape == (64,), f"Test 4b failed: {emb.shape}"
    print("  ✓ Get embedding works")
    
    # Test 5: Compute similarity
    sim = retriever.compute_similarity("butter", "margarine")
    assert 0 < sim <= 1, f"Test 5a failed: {sim}"
    sim_diff = retriever.compute_similarity("butter", "garlic")
    assert sim > sim_diff, f"Test 5b failed: butter-margarine should be more similar"
    print(f"  ✓ Similarity: butter-margarine={sim:.3f}, butter-garlic={sim_diff:.3f}")
    
    # Test 6: Find in range
    results = retriever.find_in_range("butter", min_similarity=0.5, max_similarity=0.9)
    for r in results:
        assert 0.5 <= r.score <= 0.9, f"Test 6 failed: {r.score}"
    print(f"  ✓ Found {len(results)} ingredients in similarity range [0.5, 0.9]")
    
    # Test 7: Batch find
    batch_results = retriever.batch_find_similar(["butter", "milk", "sugar"], top_k=3)
    assert len(batch_results) == 3, "Test 7a failed"
    assert "butter" in batch_results, "Test 7b failed"
    print("  ✓ Batch retrieval works")
    
    print("\nTesting FilteredRetriever...")
    
    # Test 8: Initialize filtered retriever
    filtered = FilteredRetriever(retriever, categories, dietary_tags)
    print("  ✓ FilteredRetriever initialized")
    
    # Test 9: Find in category
    results = filtered.find_similar_in_category("butter", "fat", top_k=5)
    for r in results:
        assert categories.get(r.ingredient) == "fat", f"Test 9 failed: {r.ingredient}"
    print(f"  ✓ All results in 'fat' category: {[r.ingredient for r in results]}")
    
    # Test 10: Find with dietary tags
    results = filtered.find_with_dietary_tags("butter", ["vegan", "dairy_free"], top_k=3)
    for r in results:
        tags = dietary_tags.get(r.ingredient, [])
        assert "vegan" in tags, f"Test 10 failed: {r.ingredient} not vegan"
    print(f"  ✓ Vegan substitutes for butter: {[r.ingredient for r in results]}")
    
    # Test 11: Find excluding
    results = filtered.find_excluding("butter", {"margarine", "coconut_oil"}, top_k=3)
    for r in results:
        assert r.ingredient not in {"margarine", "coconut_oil"}, f"Test 11 failed"
    print(f"  ✓ Substitutes excluding margarine/coconut_oil: {[r.ingredient for r in results]}")
    
    print("\nTesting ApproximateRetriever...")
    
    # Test 12: Initialize approximate retriever
    approx = ApproximateRetriever(embeddings, num_tables=5, num_projections=4)
    print("  ✓ ApproximateRetriever initialized")
    
    # Test 13: Find candidates
    candidates = approx.find_candidates("butter")
    assert len(candidates) > 0, "Test 13 failed"
    print(f"  ✓ Found {len(candidates)} candidates via LSH")
    
    # Test 14: Approximate search
    results = approx.find_similar("butter", top_k=5)
    assert len(results) <= 5, f"Test 14 failed: {len(results)}"
    print(f"  ✓ Approximate top 5: {[r.ingredient for r in results]}")
    
    print("\nTesting HybridRetriever...")
    
    # Test 15: Initialize hybrid retriever
    rule_subs = {
        "butter": ["margarine", "coconut_oil"],
        "milk": ["oat_milk", "soy_milk"],
        "egg": ["tofu"],
    }
    hybrid = HybridRetriever(retriever, rule_subs, rule_weight=0.6, embedding_weight=0.4)
    print("  ✓ HybridRetriever initialized")
    
    # Test 16: Hybrid search
    results = hybrid.find_substitutes("butter", top_k=5)
    assert len(results) > 0, "Test 16a failed"
    # Rule-based substitutes should rank highly
    top_names = [r.ingredient for r in results[:3]]
    assert "margarine" in top_names or "coconut_oil" in top_names, f"Test 16b failed: {top_names}"
    print(f"  ✓ Hybrid results for butter: {[r.ingredient for r in results]}")
    
    # Test 17: Explain ranking
    explanation = hybrid.explain_ranking("butter", "margarine")
    assert "rule_score" in explanation or "embedding_score" in explanation, "Test 17 failed"
    print(f"  ✓ Ranking explanation: {explanation}")
    
    print("\nTesting utility functions...")
    
    # Test 18: Build index
    index = build_index(embeddings)
    assert index.shape == (len(embeddings), 64), f"Test 18 failed: {index.shape}"
    print(f"  ✓ Built index: {index.shape}")
    
    # Test 19: Batch similarity
    query = embeddings["butter"]
    scores = batch_cosine_similarity(query, index)
    assert scores.shape == (len(embeddings),), f"Test 19 failed: {scores.shape}"
    print("  ✓ Batch similarity computation works")
    
    # Test 20: Analogy query
    results = analogy_query("butter", "margarine", "milk", retriever, top_k=3)
    assert len(results) > 0, "Test 20 failed"
    print(f"  ✓ Analogy butter:margarine :: milk:? -> {[r.ingredient for r in results]}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nRetrieval Summary:")
    print(f"  Ingredients indexed: {len(embeddings)}")
    print(f"  Embedding dimension: 64")
    print(f"  Categories: {len(set(categories.values()))}")
    
    print("\nKey concepts learned:")
    print("1. Embeddings enable similarity-based retrieval")
    print("2. Filtering narrows results by constraints")
    print("3. LSH enables fast approximate search")
    print("4. Hybrid combines rules and learned representations")
    print("\nNext: Context-aware substitution!")
