# Problem 13: Contrastive Learning
#
# Implement contrastive learning for fine-tuning embeddings. Contrastive
# learning pulls similar items together and pushes dissimilar items apart
# in embedding space.
#
# You'll implement:
# 1. create_positive_pairs() - find similar ingredient pairs
# 2. create_negative_pairs() - find dissimilar pairs as triplets
# 3. ContrastiveLoss and InfoNCELoss classes
# 4. ContrastiveTrainer for training loop
#
# Example:
#   anchor, positive = "butter", "margarine"  # Similar (both fats)
#   negative = "garlic"  # Dissimilar
#   # Training: similarity(butter, margarine) > similarity(butter, garlic)
#
# Constraints:
#   - Positive pairs from co-occurrence or same category
#   - Hard negatives provide stronger training signal
#   - Margin loss: max(0, margin - sim(a,p) + sim(a,n))
#
# ML Relevance: Contrastive learning is powerful for embeddings. It's
# self-supervised (no labels needed), learns from similarity relationships,
# and is the foundation for CLIP, SimCLR, and many modern methods.

from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict
import random


def create_positive_pairs(recipes: List[List[str]], min_cooccurrence: int = 2) -> List[Tuple[str, str]]:
    """Create positive pairs from co-occurring ingredients."""
    # Your solution here
    pass


def create_positive_pairs_from_categories(ingredients: List[str], categories: Dict[str, str], max_pairs_per_category: int = 50) -> List[Tuple[str, str]]:
    """Create positive pairs from same-category ingredients."""
    # Your solution here
    pass


def create_negative_pairs(ingredients: List[str], positive_pairs: Set[Tuple[str, str]], negatives_per_positive: int = 5) -> List[Tuple[str, str, str]]:
    """Create (anchor, positive, negative) triplets."""
    # Your solution here
    pass


def create_hard_negatives(anchor: str, embeddings: Dict[str, np.ndarray], positive_set: Set[str], num_negatives: int = 5) -> List[str]:
    """Find hard negatives: similar to anchor but not in positive set."""
    # Your solution here
    pass


class ContrastiveLoss:
    """Margin-based or NCE contrastive loss."""
    
    def __init__(self, margin: float = 0.5, loss_type: str = "margin"):
        """Initialize with margin and loss_type ('margin' or 'nce')."""
        # Your solution here
        pass
    
    def __call__(self, anchor: np.ndarray, positive: np.ndarray, negatives: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """Compute contrastive loss. Returns (loss, info_dict)."""
        # Your solution here
        pass


class InfoNCELoss:
    """InfoNCE loss (used in CLIP, SimCLR)."""
    
    def __init__(self, temperature: float = 0.07):
        """Initialize with temperature for softmax."""
        # Your solution here
        pass
    
    def __call__(self, anchor: np.ndarray, positive: np.ndarray, negatives: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """Compute InfoNCE loss: -log(exp(sim(a,p)/t) / sum(exp(sim(a,all)/t)))."""
        # Your solution here
        pass


class ContrastiveTrainer:
    """Trainer for contrastive learning of embeddings."""
    
    def __init__(self, embedding_dim: int = 64, learning_rate: float = 0.01, margin: float = 0.5):
        """Initialize trainer."""
        # Your solution here
        pass
    
    def initialize_embeddings(self, vocabulary: List[str], pretrained: Dict[str, np.ndarray] = None) -> None:
        """Initialize embeddings, optionally from pretrained."""
        # Your solution here
        pass
    
    def train_step(self, anchor: str, positive: str, negatives: List[str]) -> float:
        """Perform one training step. Returns loss."""
        # Your solution here
        pass
    
    def train(self, triplets: List[Tuple[str, str, str]], epochs: int = 10) -> List[float]:
        """Train on triplets. Returns loss history."""
        # Your solution here
        pass
    
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Return trained embeddings."""
        # Your solution here
        pass


def evaluate_contrastive_model(embeddings: Dict[str, np.ndarray], test_triplets: List[Tuple[str, str, str]]) -> Dict[str, float]:
    """Evaluate: accuracy = % where sim(a,p) > sim(a,n)."""
    # Your solution here
    pass


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    recipes = [
        ["flour", "sugar", "butter", "eggs"],
        ["flour", "sugar", "butter", "vanilla"],
        ["flour", "yeast", "salt", "water"],
        ["garlic", "olive_oil", "tomato", "basil"],
        ["garlic", "ginger", "soy_sauce", "chicken"],
        ["beef", "onion", "garlic", "herbs"],
        ["pasta", "tomato", "basil", "garlic"],
        ["flour", "butter", "sugar", "chocolate"],
    ]
    
    categories = {
        "flour": "grain", "sugar": "sweetener", "butter": "fat",
        "eggs": "protein", "vanilla": "flavoring", "yeast": "leavening",
        "salt": "seasoning", "water": "liquid", "garlic": "aromatic",
        "olive_oil": "fat", "tomato": "vegetable", "basil": "herb",
        "ginger": "aromatic", "soy_sauce": "sauce", "chicken": "protein",
        "beef": "protein", "onion": "aromatic", "herbs": "herb",
        "pasta": "grain", "chocolate": "sweetener"
    }
    
    print("Testing create_positive_pairs...")
    
    pairs = create_positive_pairs(recipes, min_cooccurrence=2)
    assert len(pairs) > 0, "Test 1a failed: no pairs created"
    has_flour_sugar = any(set(p) == {"flour", "sugar"} for p in pairs)
    assert has_flour_sugar, f"Test 1b failed: flour-sugar not found"
    print(f"  ✓ Created {len(pairs)} positive pairs")
    
    all_ingredients = list(categories.keys())
    cat_pairs = create_positive_pairs_from_categories(all_ingredients, categories, max_pairs_per_category=10)
    assert len(cat_pairs) > 0, "Test 2a failed"
    has_fat_pair = any(categories.get(p[0]) == "fat" and categories.get(p[1]) == "fat" for p in cat_pairs)
    assert has_fat_pair, "Test 2b failed: no fat pairs"
    print(f"  ✓ Created {len(cat_pairs)} category-based pairs")
    
    print("\nTesting create_negative_pairs...")
    
    pos_set = {(p[0], p[1]) for p in pairs}
    pos_set.update({(p[1], p[0]) for p in pairs})
    triplets = create_negative_pairs(all_ingredients, pos_set, negatives_per_positive=2)
    assert len(triplets) > 0, "Test 3a failed"
    assert all(len(t) == 3 for t in triplets), "Test 3b failed"
    print(f"  ✓ Created {len(triplets)} triplets")
    
    print("\nTesting create_hard_negatives...")
    
    embeddings = {}
    for ing in all_ingredients:
        embeddings[ing] = np.random.randn(64).astype(np.float32)
        embeddings[ing] /= np.linalg.norm(embeddings[ing])
    
    positive_set = {"butter", "olive_oil"}
    hard_negs = create_hard_negatives("butter", embeddings, positive_set, num_negatives=3)
    assert len(hard_negs) == 3, f"Test 4a failed: {len(hard_negs)}"
    assert "butter" not in hard_negs, "Test 4b failed"
    assert "olive_oil" not in hard_negs, "Test 4c failed"
    print("  ✓ Hard negatives created")
    
    print("\nTesting ContrastiveLoss...")
    
    loss_fn = ContrastiveLoss(margin=0.5, loss_type="margin")
    anchor = np.random.randn(64)
    positive = anchor + np.random.randn(64) * 0.1
    negatives = [np.random.randn(64) for _ in range(3)]
    
    loss, info = loss_fn(anchor, positive, negatives)
    assert isinstance(loss, (int, float)), "Test 5a failed"
    assert loss >= 0, f"Test 5b failed: loss should be >= 0, got {loss}"
    print(f"  ✓ Margin loss: {loss:.4f}")
    
    anchor = np.array([1, 0, 0], dtype=np.float32)
    positive = np.array([1, 0, 0], dtype=np.float32)
    negative = np.array([0, 1, 0], dtype=np.float32)
    
    loss1, _ = loss_fn(anchor, positive, [negative])
    positive2 = np.array([0, 1, 0], dtype=np.float32)
    loss2, _ = loss_fn(anchor, positive2, [negative])
    assert loss1 < loss2, f"Test 6 failed: {loss1} vs {loss2}"
    print("  ✓ Loss is lower for more similar positive")
    
    print("\nTesting InfoNCELoss...")
    
    nce_loss = InfoNCELoss(temperature=0.1)
    anchor = np.random.randn(64)
    positive = anchor + np.random.randn(64) * 0.1
    negatives = [np.random.randn(64) for _ in range(5)]
    
    loss, info = nce_loss(anchor, positive, negatives)
    assert loss >= 0, f"Test 7 failed: InfoNCE loss should be >= 0, got {loss}"
    print(f"  ✓ InfoNCE loss: {loss:.4f}")
    
    print("\nTesting ContrastiveTrainer...")
    
    trainer = ContrastiveTrainer(embedding_dim=32, learning_rate=0.01)
    trainer.initialize_embeddings(all_ingredients)
    print("  ✓ Trainer initialized")
    
    loss = trainer.train_step("butter", "olive_oil", ["garlic", "flour"])
    assert isinstance(loss, (int, float)), "Test 9 failed"
    print(f"  ✓ Training step loss: {loss:.4f}")
    
    triplets = [
        ("butter", "olive_oil", "garlic"),
        ("flour", "sugar", "chicken"),
        ("garlic", "onion", "flour"),
    ] * 10
    
    losses = trainer.train(triplets, epochs=3)
    assert len(losses) == 3, f"Test 10a failed: {len(losses)}"
    assert losses[-1] <= losses[0] + 0.5, f"Test 10b: loss should decrease or stay similar"
    print(f"  ✓ Training completed: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    trained_embeddings = trainer.get_embeddings()
    assert len(trained_embeddings) == len(all_ingredients), "Test 11a failed"
    assert all(v.shape == (32,) for v in trained_embeddings.values()), "Test 11b failed"
    print("  ✓ Embeddings retrieved")
    
    print("\nTesting evaluate_contrastive_model...")
    
    test_triplets = [
        ("butter", "olive_oil", "garlic"),
        ("flour", "sugar", "beef"),
    ]
    metrics = evaluate_contrastive_model(trained_embeddings, test_triplets)
    assert "accuracy" in metrics, "Test 12 failed"
    print(f"  ✓ Evaluation accuracy: {metrics['accuracy']:.2%}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
