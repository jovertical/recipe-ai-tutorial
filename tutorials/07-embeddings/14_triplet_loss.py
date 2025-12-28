# Problem 14: Triplet Loss
#
# Implement triplet loss, a popular objective for learning embeddings.
# Triplet loss directly optimizes relative distances between anchor,
# positive, and negative examples.
#
# You'll implement:
# 1. create_triplets() - generate training triplets from categories
# 2. TripletLoss class - margin-based triplet loss
# 3. hard_negative_mining() - find challenging negatives
# 4. TripletTrainer with different mining strategies
#
# Example:
#   triplet = ("butter", "margarine", "garlic")
#   # Objective: distance(butter, margarine) + margin < distance(butter, garlic)
#
# Constraints:
#   - Positive from same category, negative from different
#   - Support random, hard, and semi-hard mining
#   - Loss = max(0, d(a,p) - d(a,n) + margin)
#
# ML Relevance: Triplet loss is widely used in face recognition (FaceNet),
# image retrieval, and metric learning. It's often better than contrastive
# loss for ranking tasks.

from typing import List, Dict, Tuple, Set
import numpy as np
import random


def create_triplets(ingredients: List[str], categories: Dict[str, str], num_triplets: int = 1000) -> List[Tuple[str, str, str]]:
    """Create triplets: positive from same category, negative from different."""
    # Your solution here
    pass


def create_triplets_from_similarity(embeddings: Dict[str, np.ndarray], positive_threshold: float = 0.7, negative_threshold: float = 0.3, num_triplets: int = 1000) -> List[Tuple[str, str, str]]:
    """Create triplets based on embedding similarity thresholds."""
    # Your solution here
    pass


class TripletLoss:
    """Triplet loss: max(0, d(a,p) - d(a,n) + margin)."""
    
    def __init__(self, margin: float = 0.3, distance: str = "cosine"):
        """Initialize with margin and distance type ('cosine' or 'euclidean')."""
        # Your solution here
        pass
    
    def compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        # Your solution here
        pass
    
    def __call__(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute triplet loss. Returns (loss, info_dict with distances)."""
        # Your solution here
        pass


class BatchHardTripletLoss:
    """Batch-hard: mine hardest positive and negative within batch."""
    
    def __init__(self, margin: float = 0.3):
        """Initialize batch-hard loss."""
        # Your solution here
        pass
    
    def __call__(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute batch-hard triplet loss on (batch_size, dim) embeddings."""
        # Your solution here
        pass


def hard_negative_mining(anchor: str, positive: str, embeddings: Dict[str, np.ndarray], categories: Dict[str, str], num_negatives: int = 5) -> List[str]:
    """Find hard negatives: different category but close to anchor."""
    # Your solution here
    pass


def semi_hard_negative_mining(anchor: str, positive: str, embeddings: Dict[str, np.ndarray], categories: Dict[str, str], margin: float = 0.3) -> List[str]:
    """Find semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin."""
    # Your solution here
    pass


class TripletTrainer:
    """Trainer using triplet loss with mining strategies."""
    
    def __init__(self, embedding_dim: int = 64, margin: float = 0.3, learning_rate: float = 0.01, mining_strategy: str = "random"):
        """Initialize with mining_strategy: 'random', 'hard', or 'semi-hard'."""
        # Your solution here
        pass
    
    def initialize_embeddings(self, vocabulary: List[str]) -> None:
        """Initialize random embeddings."""
        # Your solution here
        pass
    
    def mine_triplets(self, categories: Dict[str, str], batch_size: int = 32) -> List[Tuple[str, str, str]]:
        """Mine triplets according to strategy."""
        # Your solution here
        pass
    
    def train_step(self, anchor: str, positive: str, negative: str) -> float:
        """Perform one training step. Returns loss."""
        # Your solution here
        pass
    
    def train(self, categories: Dict[str, str], epochs: int = 10, triplets_per_epoch: int = 100) -> List[float]:
        """Train the model. Returns loss history."""
        # Your solution here
        pass
    
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Return trained embeddings."""
        # Your solution here
        pass


def evaluate_triplet_accuracy(embeddings: Dict[str, np.ndarray], test_triplets: List[Tuple[str, str, str]]) -> float:
    """Accuracy = fraction where d(a,p) < d(a,n)."""
    # Your solution here
    pass


def analyze_embedding_space(embeddings: Dict[str, np.ndarray], categories: Dict[str, str]) -> Dict[str, float]:
    """Analyze: intra_class_distance, inter_class_distance, separation_ratio."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    categories = {
        "butter": "fat", "oil": "fat", "margarine": "fat", "lard": "fat",
        "flour": "grain", "rice": "grain", "pasta": "grain",
        "garlic": "aromatic", "onion": "aromatic", "ginger": "aromatic",
        "chicken": "protein", "beef": "protein", "pork": "protein",
        "sugar": "sweetener", "honey": "sweetener"
    }
    ingredients = list(categories.keys())
    
    print("Testing create_triplets...")
    
    triplets = create_triplets(ingredients, categories, num_triplets=50)
    assert len(triplets) == 50, f"Test 1a failed: {len(triplets)}"
    
    for a, p, n in triplets[:5]:
        assert a in categories, f"Test 1b failed: {a}"
        assert p in categories, f"Test 1c failed: {p}"
        assert n in categories, f"Test 1d failed: {n}"
        assert categories[a] == categories[p], f"Test 1e failed: {a}, {p}"
        assert categories[a] != categories[n], f"Test 1f failed: {a}, {n}"
    print(f"  ✓ Created {len(triplets)} valid triplets")
    
    print("\nTesting TripletLoss...")
    
    loss_fn = TripletLoss(margin=0.3)
    
    anchor = np.array([1.0, 0.0, 0.0])
    positive = np.array([0.99, 0.01, 0.0])
    negative = np.array([0.0, 1.0, 0.0])
    
    loss, info = loss_fn(anchor, positive, negative)
    assert loss == 0, f"Test 2a failed: easy triplet should have 0 loss, got {loss}"
    print("  ✓ Easy triplet has 0 loss")
    
    negative_hard = np.array([0.95, 0.05, 0.0])
    loss, info = loss_fn(anchor, positive, negative_hard)
    assert loss > 0, f"Test 2b failed: hard triplet should have positive loss"
    print(f"  ✓ Hard triplet loss: {loss:.4f}")
    
    d_pos = loss_fn.compute_distance(anchor, positive)
    d_neg = loss_fn.compute_distance(anchor, negative)
    assert d_pos < d_neg, f"Test 3 failed: {d_pos} vs {d_neg}"
    print("  ✓ Distance computation correct")
    
    print("\nTesting BatchHardTripletLoss...")
    
    batch_loss = BatchHardTripletLoss(margin=0.3)
    embeddings_batch = np.random.randn(8, 32).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    loss, info = batch_loss(embeddings_batch, labels)
    assert isinstance(loss, (int, float)), f"Test 4a failed: {type(loss)}"
    assert loss >= 0, f"Test 4b failed: {loss}"
    print(f"  ✓ Batch-hard loss: {loss:.4f}")
    
    print("\nTesting hard_negative_mining...")
    
    embeddings = {}
    for ing in ingredients:
        seed = sum(ord(c) for c in ing)
        np.random.seed(seed)
        embeddings[ing] = np.random.randn(64).astype(np.float32)
        embeddings[ing] /= np.linalg.norm(embeddings[ing])
    
    hard_negs = hard_negative_mining("butter", "margarine", embeddings, categories, num_negatives=3)
    assert len(hard_negs) == 3, f"Test 5a failed: {len(hard_negs)}"
    assert all(categories[n] != categories["butter"] for n in hard_negs), "Test 5b failed"
    print(f"  ✓ Hard negatives: {hard_negs}")
    
    print("\nTesting semi_hard_negative_mining...")
    
    semi_hard = semi_hard_negative_mining("butter", "margarine", embeddings, categories, margin=0.3)
    assert isinstance(semi_hard, list), "Test 6 failed"
    print(f"  ✓ Semi-hard negatives: {len(semi_hard)} found")
    
    print("\nTesting TripletTrainer...")
    
    trainer = TripletTrainer(embedding_dim=32, margin=0.3, learning_rate=0.01, mining_strategy="random")
    trainer.initialize_embeddings(ingredients)
    print("  ✓ Trainer initialized")
    
    mined = trainer.mine_triplets(categories, batch_size=10)
    assert len(mined) == 10, f"Test 8 failed: {len(mined)}"
    print("  ✓ Triplet mining works")
    
    a, p, n = mined[0]
    loss = trainer.train_step(a, p, n)
    assert isinstance(loss, (int, float)), "Test 9 failed"
    print(f"  ✓ Training step loss: {loss:.4f}")
    
    losses = trainer.train(categories, epochs=3, triplets_per_epoch=50)
    assert len(losses) == 3, f"Test 10 failed: {len(losses)}"
    print(f"  ✓ Training: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    trained_emb = trainer.get_embeddings()
    assert len(trained_emb) == len(ingredients), "Test 11 failed"
    print("  ✓ Retrieved trained embeddings")
    
    print("\nTesting evaluate_triplet_accuracy...")
    
    test_triplets = create_triplets(ingredients, categories, num_triplets=20)
    accuracy = evaluate_triplet_accuracy(trained_emb, test_triplets)
    assert 0 <= accuracy <= 1, f"Test 12 failed: {accuracy}"
    print(f"  ✓ Triplet accuracy: {accuracy:.2%}")
    
    print("\nTesting analyze_embedding_space...")
    
    analysis = analyze_embedding_space(trained_emb, categories)
    assert "intra_class_distance" in analysis, "Test 13a failed"
    assert "inter_class_distance" in analysis, "Test 13b failed"
    assert "separation_ratio" in analysis, "Test 13c failed"
    print(f"  ✓ Separation ratio: {analysis['separation_ratio']:.2f}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
