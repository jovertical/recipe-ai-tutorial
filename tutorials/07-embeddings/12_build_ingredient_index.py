# Problem 12: Build Ingredient Index
#
# Build a searchable index for ingredient embeddings. This enables fast
# similarity search at scale using techniques like approximate nearest neighbors.
#
# You'll implement:
# 1. IngredientIndex class - basic exact search
# 2. LSHIndex class - approximate search with locality-sensitive hashing
# 3. HierarchicalIndex - category-aware search
# 4. Evaluation and benchmarking functions
#
# Example:
#   index = IngredientIndex(embeddings)
#   results = index.search("butter", k=5)
#   → [("margarine", 0.92), ("oil", 0.85), ...]
#   batch_results = index.batch_search(["butter", "flour"], k=3)
#
# Constraints:
#   - Support add/remove operations
#   - Normalize embeddings for cosine similarity
#   - LSH should be faster but approximate
#
# ML Relevance: Efficient similarity search is crucial for real-time ingredient
# substitution, recipe recommendation, search and retrieval, and production
# ML systems at scale.

from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from collections import defaultdict


class IngredientIndex:
    """Index for fast ingredient similarity search."""
    
    def __init__(self, embeddings: Dict[str, np.ndarray] = None, normalize: bool = True):
        """Initialize index, optionally with initial embeddings."""
        # Your solution here
        pass
    
    def add(self, name: str, embedding: np.ndarray) -> None:
        """Add an ingredient to the index."""
        # Your solution here
        pass
    
    def add_batch(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Add multiple ingredients to the index."""
        # Your solution here
        pass
    
    def remove(self, name: str) -> bool:
        """Remove an ingredient. Returns True if found."""
        # Your solution here
        pass
    
    def search(self, query: str, k: int = 5, exclude_self: bool = True) -> List[Tuple[str, float]]:
        """Find k most similar ingredients."""
        # Your solution here
        pass
    
    def search_by_vector(self, vector: np.ndarray, k: int = 5, exclude: Set[str] = None) -> List[Tuple[str, float]]:
        """Search by embedding vector."""
        # Your solution here
        pass
    
    def batch_search(self, queries: List[str], k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Search for multiple queries at once."""
        # Your solution here
        pass
    
    def __len__(self) -> int:
        """Return number of indexed ingredients."""
        return len(self.name_to_idx)
    
    def __contains__(self, name: str) -> bool:
        """Check if ingredient is indexed."""
        return name in self.name_to_idx


def build_faiss_index(embeddings: Dict[str, np.ndarray], index_type: str = "flat") -> Tuple[any, Dict[int, str]]:
    """Build a FAISS-style index (mock). Types: 'flat', 'ivf', 'hnsw'."""
    # Your solution here
    pass


class LSHIndex:
    """Locality-Sensitive Hashing for approximate nearest neighbors."""
    
    def __init__(self, embedding_dim: int, num_tables: int = 10, num_bits: int = 8):
        """Initialize with random projection vectors for each table."""
        # Your solution here
        pass
    
    def _hash(self, vector: np.ndarray, table_idx: int) -> int:
        """Compute hash for a vector in a specific table."""
        # Your solution here
        pass
    
    def add(self, name: str, vector: np.ndarray) -> None:
        """Add item to index."""
        # Your solution here
        pass
    
    def search(self, query: np.ndarray, k: int = 5, num_candidates: int = 50) -> List[Tuple[str, float]]:
        """Approximate nearest neighbor search."""
        # Your solution here
        pass


class HierarchicalIndex:
    """Hierarchical index using ingredient categories."""
    
    def __init__(self, embeddings: Dict[str, np.ndarray], categories: Dict[str, str]):
        """Initialize with per-category indices."""
        # Your solution here
        pass
    
    def search_in_category(self, query: str, category: str = None, k: int = 5) -> List[Tuple[str, float]]:
        """Search within a specific category."""
        # Your solution here
        pass
    
    def search_cross_category(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """Search across all categories. Returns (name, sim, category)."""
        # Your solution here
        pass


def evaluate_index_quality(index: IngredientIndex, test_queries: List[str], ground_truth: Dict[str, List[str]], k: int = 5) -> Dict[str, float]:
    """Evaluate search quality with precision@k and recall@k."""
    # Your solution here
    pass


def benchmark_index(index: IngredientIndex, num_queries: int = 100, k: int = 10) -> Dict[str, float]:
    """Benchmark search speed."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    ingredients = [
        "butter", "margarine", "oil", "olive_oil",
        "flour", "sugar", "salt",
        "garlic", "onion", "ginger",
        "chicken", "beef", "pork", "tofu"
    ]
    
    embeddings = {}
    for ing in ingredients:
        seed = sum(ord(c) for c in ing)
        np.random.seed(seed)
        embeddings[ing] = np.random.randn(64).astype(np.float32)
        embeddings[ing] /= np.linalg.norm(embeddings[ing])
    
    embeddings["margarine"] = embeddings["butter"] + np.random.randn(64) * 0.1
    embeddings["margarine"] /= np.linalg.norm(embeddings["margarine"])
    embeddings["olive_oil"] = embeddings["oil"] + np.random.randn(64) * 0.1
    embeddings["olive_oil"] /= np.linalg.norm(embeddings["olive_oil"])
    
    print("Testing IngredientIndex...")
    
    index = IngredientIndex(embeddings)
    assert len(index) == len(ingredients), f"Test 1 failed: {len(index)}"
    print(f"  ✓ Index created with {len(index)} ingredients")
    
    assert "butter" in index, "Test 2a failed"
    assert "xyz" not in index, "Test 2b failed"
    print("  ✓ Contains check works")
    
    results = index.search("butter", k=3)
    assert len(results) == 3, f"Test 3a failed: {len(results)}"
    assert results[0][0] == "margarine", f"Test 3b failed: {results[0]}"
    assert all(isinstance(r, tuple) for r in results), "Test 3c failed"
    print(f"  ✓ Search works: butter -> {[r[0] for r in results]}")
    
    results_no_self = index.search("butter", k=3, exclude_self=True)
    assert "butter" not in [r[0] for r in results_no_self], "Test 4 failed"
    print("  ✓ Exclude self works")
    
    index.add("ghee", embeddings["butter"] + np.random.randn(64) * 0.05)
    assert "ghee" in index, "Test 5a failed"
    assert len(index) == len(ingredients) + 1, "Test 5b failed"
    print("  ✓ Add ingredient works")
    
    removed = index.remove("ghee")
    assert removed, "Test 6a failed"
    assert "ghee" not in index, "Test 6b failed"
    print("  ✓ Remove ingredient works")
    
    query_vec = embeddings["butter"]
    results = index.search_by_vector(query_vec, k=3)
    assert len(results) == 3, f"Test 7 failed: {len(results)}"
    print("  ✓ Search by vector works")
    
    batch_results = index.batch_search(["butter", "garlic"], k=2)
    assert len(batch_results) == 2, "Test 8a failed"
    assert "butter" in batch_results, "Test 8b failed"
    assert "garlic" in batch_results, "Test 8c failed"
    print("  ✓ Batch search works")
    
    print("\nTesting LSHIndex...")
    
    lsh = LSHIndex(embedding_dim=64, num_tables=5, num_bits=4)
    for name, vec in embeddings.items():
        lsh.add(name, vec)
    print("  ✓ LSH index built")
    
    results = lsh.search(embeddings["butter"], k=3)
    assert len(results) <= 3, f"Test 10a failed: {len(results)}"
    print(f"  ✓ LSH search works: {[r[0] for r in results]}")
    
    print("\nTesting HierarchicalIndex...")
    
    categories = {
        "butter": "fat", "margarine": "fat", "oil": "fat", "olive_oil": "fat",
        "flour": "grain", "sugar": "sweetener", "salt": "seasoning",
        "garlic": "aromatic", "onion": "aromatic", "ginger": "aromatic",
        "chicken": "protein", "beef": "protein", "pork": "protein", "tofu": "protein"
    }
    
    hier_index = HierarchicalIndex(embeddings, categories)
    print("  ✓ Hierarchical index built")
    
    results = hier_index.search_in_category("butter", category="fat", k=2)
    assert len(results) <= 2, f"Test 12 failed: {len(results)}"
    print(f"  ✓ Category search: butter (fat) -> {[r[0] for r in results]}")
    
    results = hier_index.search_cross_category("butter", k=3)
    assert len(results) <= 3, f"Test 13 failed: {len(results)}"
    print("  ✓ Cross-category search works")
    
    print("\nTesting evaluate_index_quality...")
    
    ground_truth = {
        "butter": ["margarine", "oil"],
        "garlic": ["onion", "ginger"]
    }
    metrics = evaluate_index_quality(index, ["butter", "garlic"], ground_truth, k=5)
    assert "precision" in metrics or "recall" in metrics, "Test 14 failed"
    print("  ✓ Index evaluation works")
    
    print("\nTesting benchmark_index...")
    
    stats = benchmark_index(index, num_queries=10, k=5)
    assert "avg_time" in stats or "total_time" in stats, "Test 15 failed"
    print("  ✓ Benchmarking works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
