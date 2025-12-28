# Problem 9: Train Ingredient Embeddings
#
# Build a complete pipeline to train ingredient embeddings specifically
# optimized for recipe data. This combines techniques from previous exercises.
#
# You'll implement:
# 1. IngredientDataset class - generates training pairs from recipes
# 2. IngredientEmbeddingModel - neural network with negative sampling
# 3. IngredientEmbeddingTrainer - complete training pipeline
# 4. Evaluation and visualization functions
#
# Example:
#   trainer = IngredientEmbeddingTrainer(embedding_dim=64)
#   trainer.fit(recipes)
#   trainer.most_similar("butter") → [("margarine", 0.89), ("oil", 0.82), ...]
#
# Constraints:
#   - Use negative sampling for efficient training
#   - Filter vocabulary by minimum count
#   - Compute frequencies for weighted negative sampling
#
# ML Relevance: Domain-specific embeddings often outperform general ones.
# Training on recipes captures ingredient relationships specific to cooking,
# learns from real ingredient co-occurrence, and enables substitution,
# recommendation, and search applications.

from typing import List, Dict, Tuple, Iterator
import numpy as np
from collections import Counter
import random


class IngredientDataset:
    """Dataset for ingredient embedding training."""
    
    def __init__(self, recipes: List[List[str]], window_size: int = 2, min_count: int = 2):
        """Build vocabulary and prepare training pairs."""
        # Your solution here
        pass
    
    def __len__(self) -> int:
        """Return total number of training pairs."""
        # Your solution here
        pass
    
    def __iter__(self) -> Iterator[Tuple[int, int]]:
        """Iterate over (center_idx, context_idx) pairs."""
        # Your solution here
        pass
    
    def get_negative_samples(self, positive_idx: int, num_samples: int) -> List[int]:
        """Sample negatives excluding positive_idx."""
        # Your solution here
        pass
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.word_to_idx)


class IngredientEmbeddingModel:
    """Neural network for learning ingredient embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, learning_rate: float = 0.025):
        """Initialize embedding matrices."""
        # Your solution here
        pass
    
    def forward(self, center_idx: int, context_idx: int) -> float:
        """Compute dot product score for (center, context) pair."""
        # Your solution here
        pass
    
    def train_step(self, center_idx: int, context_idx: int, negative_indices: List[int]) -> float:
        """Perform one training step with negative sampling. Returns loss."""
        # Your solution here
        pass
    
    def get_embedding(self, idx: int) -> np.ndarray:
        """Get embedding for an index."""
        return self.embeddings[idx].copy()
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings."""
        return self.embeddings.copy()


class IngredientEmbeddingTrainer:
    """Complete trainer for ingredient embeddings."""
    
    def __init__(self, embedding_dim: int = 64, window_size: int = 2, min_count: int = 2, num_negatives: int = 5, learning_rate: float = 0.025, epochs: int = 5):
        """Initialize trainer with hyperparameters."""
        # Your solution here
        pass
    
    def fit(self, recipes: List[List[str]], verbose: bool = True) -> List[float]:
        """Train embeddings on recipes. Returns list of average losses per epoch."""
        # Your solution here
        pass
    
    def get_embedding(self, ingredient: str) -> np.ndarray:
        """Get embedding for an ingredient."""
        # Your solution here
        pass
    
    def most_similar(self, ingredient: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar ingredients."""
        # Your solution here
        pass
    
    def save(self, path: str) -> None:
        """Save model to file."""
        # Your solution here
        pass
    
    @classmethod
    def load(cls, path: str) -> 'IngredientEmbeddingTrainer':
        """Load model from file."""
        # Your solution here
        pass


def evaluate_embeddings(embeddings: Dict[str, np.ndarray], test_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
    """Evaluate embedding quality using similarity pairs."""
    # Your solution here
    pass


def compute_intrinsic_quality(embeddings: Dict[str, np.ndarray], categories: Dict[str, str]) -> Dict[str, float]:
    """Compute quality metrics: same/different category similarity, separation ratio."""
    # Your solution here
    pass


def visualize_ingredient_space(embeddings: Dict[str, np.ndarray], categories: Dict[str, str] = None, n_components: int = 2) -> Dict[str, Tuple[float, float]]:
    """Get 2D coordinates for visualization using PCA."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    recipes = [
        ["flour", "sugar", "butter", "eggs", "vanilla"],
        ["flour", "yeast", "salt", "water", "olive_oil"],
        ["butter", "garlic", "herbs", "olive_oil"],
        ["flour", "butter", "sugar", "chocolate"],
        ["garlic", "olive_oil", "tomato", "basil"],
        ["flour", "eggs", "milk", "butter"],
        ["chicken", "garlic", "herbs", "olive_oil"],
        ["pasta", "tomato", "basil", "garlic", "olive_oil"],
        ["beef", "onion", "garlic", "herbs"],
        ["flour", "sugar", "eggs", "butter", "milk"],
        ["rice", "soy_sauce", "garlic", "ginger"],
        ["tofu", "soy_sauce", "ginger", "garlic"],
    ]
    
    print("Testing IngredientDataset...")
    
    dataset = IngredientDataset(recipes, window_size=2, min_count=1)
    assert dataset.vocab_size > 0, "Test 1a failed"
    assert len(dataset) > 0, "Test 1b failed"
    print(f"  ✓ Dataset created with {dataset.vocab_size} ingredients")
    
    pairs = list(dataset)
    assert len(pairs) > 0, "Test 2a failed"
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs[:10]), "Test 2b failed"
    print(f"  ✓ Generated {len(pairs)} training pairs")
    
    negs = dataset.get_negative_samples(0, 5)
    assert len(negs) == 5, f"Test 3a failed: {len(negs)}"
    assert 0 not in negs, "Test 3b failed: positive should be excluded"
    print("  ✓ Negative sampling works")
    
    print("\nTesting IngredientEmbeddingModel...")
    
    model = IngredientEmbeddingModel(vocab_size=dataset.vocab_size, embedding_dim=32)
    assert model.embeddings.shape == (dataset.vocab_size, 32), "Test 4 failed"
    print("  ✓ Model created")
    
    score = model.forward(0, 1)
    assert isinstance(score, (int, float)), f"Test 5 failed: {type(score)}"
    print("  ✓ Forward pass works")
    
    loss = model.train_step(0, 1, [2, 3, 4])
    assert loss > 0, f"Test 6 failed: {loss}"
    print("  ✓ Training step works")
    
    print("\nTesting IngredientEmbeddingTrainer...")
    
    trainer = IngredientEmbeddingTrainer(embedding_dim=32, window_size=2, min_count=1, num_negatives=3, epochs=2)
    assert trainer is not None, "Test 7 failed"
    print("  ✓ Trainer created")
    
    losses = trainer.fit(recipes, verbose=False)
    assert len(losses) == 2, f"Test 8a failed: {len(losses)}"
    assert all(l > 0 for l in losses), "Test 8b failed"
    print(f"  ✓ Training completed, final loss: {losses[-1]:.4f}")
    
    if "flour" in trainer.dataset.word_to_idx:
        emb = trainer.get_embedding("flour")
        assert emb.shape == (32,), f"Test 9 failed: {emb.shape}"
        print("  ✓ Get embedding works")
    
    if "butter" in trainer.dataset.word_to_idx:
        similar = trainer.most_similar("butter", top_k=5)
        assert len(similar) <= 5, f"Test 10a failed: {len(similar)}"
        assert all(isinstance(s, tuple) for s in similar), "Test 10b failed"
        print(f"  ✓ Most similar to 'butter': {[s[0] for s in similar[:3]]}")
    
    print("\nTesting save and load...")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.npz")
        trainer.save(path)
        assert os.path.exists(path), "Test 11a failed"
        
        loaded = IngredientEmbeddingTrainer.load(path)
        assert loaded is not None, "Test 11b failed"
        assert loaded.dataset.vocab_size == trainer.dataset.vocab_size, "Test 11c failed"
        print("  ✓ Save and load work")
    
    print("\nTesting evaluate_embeddings...")
    
    embeddings = {}
    for word, idx in trainer.dataset.word_to_idx.items():
        embeddings[word] = trainer.model.get_embedding(idx)
    
    test_pairs = []
    words = list(embeddings.keys())[:4]
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            test_pairs.append((w1, w2, 0.5))
    
    if len(test_pairs) > 0:
        metrics = evaluate_embeddings(embeddings, test_pairs)
        assert isinstance(metrics, dict), "Test 12 failed"
        print("  ✓ Evaluation works")
    
    print("\nTesting compute_intrinsic_quality...")
    
    categories = {
        "flour": "grain", "sugar": "sweetener", "butter": "fat",
        "eggs": "protein", "milk": "dairy", "garlic": "aromatic",
        "olive_oil": "fat", "tomato": "vegetable", "basil": "herb"
    }
    categories = {k: v for k, v in categories.items() if k in embeddings}
    
    if len(categories) >= 4:
        quality = compute_intrinsic_quality(embeddings, categories)
        assert isinstance(quality, dict), "Test 13 failed"
        print("  ✓ Intrinsic quality computed")
    
    print("\nTesting visualize_ingredient_space...")
    
    coords = visualize_ingredient_space(embeddings, categories)
    assert len(coords) == len(embeddings), f"Test 14a failed: {len(coords)}"
    for ing, (x, y) in coords.items():
        assert isinstance(x, (int, float)), f"Test 14b failed for {ing}"
        assert isinstance(y, (int, float)), f"Test 14c failed for {ing}"
    print("  ✓ Visualization coordinates generated")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
