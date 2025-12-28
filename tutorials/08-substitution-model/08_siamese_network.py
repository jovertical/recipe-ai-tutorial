# Problem 8: Siamese Network for Substitution
#
# Build a Siamese network that learns to embed ingredients such that
# substitutable pairs are close together.
#
# Example:
#   siamese = SiameseNetwork(input_dim=300, embedding_dim=64)
#   loss = siamese.contrastive_loss(butter, margarine, label=1)  # Similar
#   loss = siamese.contrastive_loss(butter, salt, label=0)       # Different
#
# ML Relevance: Siamese networks share weights between twin networks, learn
# metric spaces for similarity, work well with limited labeled data, and
# enable one-shot learning.
#
# Your Task:
#   1. Implement SiameseNetwork with shared encoder
#   2. Implement contrastive loss function
#   3. Implement triplet loss variant
#   4. Implement training and embedding extraction


from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import random


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8)


@dataclass
class SiameseConfig:
    """Configuration for Siamese network."""
    input_dim: int = 300  # Input feature dimension
    embedding_dim: int = 64  # Output embedding dimension
    hidden_dims: List[int] = None  # Hidden layer dimensions
    dropout_rate: float = 0.2
    margin: float = 1.0  # Margin for contrastive loss
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class SiameseEncoder:
    """
    Encoder network that maps inputs to embeddings.
    
    This is the shared network in a Siamese architecture.
    """
    
    def __init__(self, config: SiameseConfig):
        """
        Initialize encoder.
        
        Args:
            config: Network configuration
        """
        self.config = config
        # Your solution here
        # Build layers: input -> hidden1 -> hidden2 -> ... -> embedding
        pass
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        # Your solution here
        pass
    
    def forward(
        self,
        x: np.ndarray,
        training: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input vector
            training: Whether in training mode
            
        Returns:
            Tuple of (embedding, cache for backprop)
        """
        # Your solution here
        pass
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to embedding (inference mode).
        
        Args:
            x: Input vector
            
        Returns:
            Embedding vector
        """
        # Your solution here
        pass
    
    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Encode batch of inputs.
        
        Args:
            X: Input batch (n x input_dim)
            
        Returns:
            Embeddings (n x embedding_dim)
        """
        # Your solution here
        pass


class SiameseNetwork:
    """
    Siamese network for learning substitution embeddings.
    
    Uses a shared encoder to embed both ingredients,
    then measures similarity in embedding space.
    """
    
    def __init__(self, config: SiameseConfig = None):
        """
        Initialize Siamese network.
        
        Args:
            config: Network configuration
        """
        self.config = config or SiameseConfig()
        self.encoder = SiameseEncoder(self.config)
    
    def forward(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass for a pair of inputs.
        
        Args:
            x1: First ingredient features
            x2: Second ingredient features
            training: Whether in training mode
            
        Returns:
            Tuple of (embedding1, embedding2, cache)
        """
        # Your solution here
        pass
    
    def compute_distance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        distance_type: str = "euclidean"
    ) -> float:
        """
        Compute distance between two inputs in embedding space.
        
        Args:
            x1: First input
            x2: Second input
            distance_type: "euclidean" or "cosine"
            
        Returns:
            Distance value
        """
        # Your solution here
        pass
    
    def compute_similarity(
        self,
        x1: np.ndarray,
        x2: np.ndarray
    ) -> float:
        """
        Compute similarity between two inputs.
        
        Args:
            x1: First input
            x2: Second input
            
        Returns:
            Similarity score (higher = more similar)
        """
        # Your solution here
        pass
    
    def contrastive_loss(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        label: int,
        margin: float = None
    ) -> Tuple[float, Dict]:
        """
        Compute contrastive loss.
        
        Loss = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
        
        Where Y=0 for similar pairs, Y=1 for dissimilar pairs.
        (Note: Some implementations swap Y meanings)
        
        Args:
            x1: First input
            x2: Second input
            label: 1 if similar (substitutes), 0 if dissimilar
            margin: Margin for dissimilar pairs
            
        Returns:
            Tuple of (loss, cache for backprop)
        """
        if margin is None:
            margin = self.config.margin
        
        # Your solution here
        # Hints:
        # 1. Compute embeddings
        # 2. Compute Euclidean distance
        # 3. If similar (label=1): minimize distance
        # 4. If dissimilar (label=0): push apart up to margin
        pass
    
    def triplet_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float = None
    ) -> Tuple[float, Dict]:
        """
        Compute triplet loss.
        
        Loss = max(0, D(anchor, positive) - D(anchor, negative) + margin)
        
        Args:
            anchor: Anchor ingredient
            positive: Similar ingredient (substitute)
            negative: Dissimilar ingredient (not substitute)
            margin: Minimum difference between distances
            
        Returns:
            Tuple of (loss, cache)
        """
        if margin is None:
            margin = self.config.margin
        
        # Your solution here
        pass
    
    def backward_contrastive(
        self,
        cache: Dict,
        label: int
    ) -> Dict[str, np.ndarray]:
        """
        Backward pass for contrastive loss.
        
        Args:
            cache: Cache from forward pass
            label: Similarity label
            
        Returns:
            Gradients dictionary
        """
        # Your solution here
        pass
    
    def train_step(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, int]],
        loss_type: str = "contrastive"
    ) -> float:
        """
        Perform one training step.
        
        Args:
            batch: List of (x1, x2, label) tuples
            loss_type: "contrastive" or "triplet"
            
        Returns:
            Batch loss
        """
        # Your solution here
        pass
    
    def get_embedding(self, x: np.ndarray) -> np.ndarray:
        """Get embedding for a single input."""
        return self.encoder.encode(x)
    
    def get_embeddings(self, inputs: List[np.ndarray]) -> np.ndarray:
        """Get embeddings for multiple inputs."""
        return self.encoder.encode_batch(np.array(inputs))


def train_siamese(
    network: SiameseNetwork,
    train_pairs: List[Tuple[np.ndarray, np.ndarray, int]],
    val_pairs: List[Tuple[np.ndarray, np.ndarray, int]],
    epochs: int = 50,
    batch_size: int = 32,
    loss_type: str = "contrastive"
) -> Dict[str, List[float]]:
    """
    Train the Siamese network.
    
    Args:
        network: Siamese network to train
        train_pairs: Training pairs (x1, x2, label)
        val_pairs: Validation pairs
        epochs: Number of epochs
        batch_size: Batch size
        loss_type: Loss function type
        
    Returns:
        Training history
    """
    # Your solution here
    pass


def create_pairs_from_labels(
    features: Dict[str, np.ndarray],
    substitutes: Dict[str, List[str]],
    num_negative_per_positive: int = 1
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Create training pairs from substitution labels.
    
    Args:
        features: Ingredient name -> feature vector
        substitutes: Ingredient -> list of substitutes
        num_negative_per_positive: Negative pairs per positive
        
    Returns:
        List of (x1, x2, label) tuples
    """
    # Your solution here
    pass


def mine_hard_negatives(
    network: SiameseNetwork,
    anchor_features: np.ndarray,
    candidate_features: List[np.ndarray],
    positive_indices: List[int],
    num_negatives: int = 5
) -> List[int]:
    """
    Mine hard negatives (close but not substitutes).
    
    Args:
        network: Trained Siamese network
        anchor_features: Anchor ingredient features
        candidate_features: All candidate features
        positive_indices: Indices of true substitutes
        num_negatives: Number of hard negatives to return
        
    Returns:
        Indices of hard negatives
    """
    # Your solution here
    pass


class OnlineTripletMiner:
    """
    Online triplet mining during training.
    """
    
    def __init__(self, strategy: str = "hard"):
        """
        Initialize miner.
        
        Args:
            strategy: "hard", "semi-hard", or "random"
        """
        self.strategy = strategy
    
    def mine(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplets from a batch.
        
        Args:
            embeddings: Batch embeddings (n x dim)
            labels: Batch labels (n,)
            
        Returns:
            List of (anchor_idx, positive_idx, negative_idx)
        """
        # Your solution here
        pass
    
    def _get_hard_triplets(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Get hardest triplets (hardest positive, hardest negative)."""
        # Your solution here
        pass
    
    def _get_semihard_triplets(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Get semi-hard triplets (negatives closer than positives + margin)."""
        # Your solution here
        pass


def evaluate_siamese(
    network: SiameseNetwork,
    test_pairs: List[Tuple[np.ndarray, np.ndarray, int]],
    threshold: float = None
) -> Dict[str, float]:
    """
    Evaluate Siamese network.
    
    Args:
        network: Trained network
        test_pairs: Test pairs
        threshold: Distance threshold for classification
        
    Returns:
        Evaluation metrics
    """
    # Your solution here
    pass


def visualize_embeddings(
    network: SiameseNetwork,
    features: Dict[str, np.ndarray],
    categories: Dict[str, str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for visualization.
    
    Args:
        network: Trained network
        features: Ingredient features
        categories: Optional category labels
        
    Returns:
        Dictionary with embeddings and metadata
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    input_dim = 100
    embedding_dim = 32
    
    # Create mock features
    def create_features():
        return np.random.randn(input_dim).astype(np.float32)
    
    # Create similar and dissimilar pairs
    positive_pairs = []
    for _ in range(50):
        base = create_features()
        similar = base + np.random.randn(input_dim) * 0.1
        positive_pairs.append((base, similar, 1))
    
    negative_pairs = []
    for _ in range(50):
        x1 = create_features()
        x2 = create_features()
        negative_pairs.append((x1, x2, 0))
    
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    train_pairs = all_pairs[:80]
    val_pairs = all_pairs[80:]
    
    print("Testing SiameseConfig...")
    
    # Test 1: Config initialization
    config = SiameseConfig(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=[64, 48]
    )
    assert config.input_dim == input_dim, "Test 1a failed"
    assert config.embedding_dim == embedding_dim, "Test 1b failed"
    print(f"  ✓ Config: input={config.input_dim}, embed={config.embedding_dim}")
    
    print("\nTesting SiameseEncoder...")
    
    # Test 2: Encoder initialization
    encoder = SiameseEncoder(config)
    print("  ✓ Encoder initialized")
    
    # Test 3: Encode single
    x = create_features()
    embedding = encoder.encode(x)
    assert embedding.shape == (embedding_dim,), f"Test 3 failed: {embedding.shape}"
    print(f"  ✓ Embedding shape: {embedding.shape}")
    
    # Test 4: Encode batch
    batch = np.array([create_features() for _ in range(10)])
    embeddings = encoder.encode_batch(batch)
    assert embeddings.shape == (10, embedding_dim), f"Test 4 failed: {embeddings.shape}"
    print(f"  ✓ Batch embedding shape: {embeddings.shape}")
    
    print("\nTesting SiameseNetwork...")
    
    # Test 5: Network initialization
    network = SiameseNetwork(config)
    print("  ✓ Network initialized")
    
    # Test 6: Forward pass
    x1, x2, label = train_pairs[0]
    emb1, emb2, cache = network.forward(x1, x2, training=True)
    assert emb1.shape == (embedding_dim,), f"Test 6a failed: {emb1.shape}"
    assert emb2.shape == (embedding_dim,), f"Test 6b failed: {emb2.shape}"
    print(f"  ✓ Forward pass: emb shapes {emb1.shape}, {emb2.shape}")
    
    # Test 7: Compute distance
    dist = network.compute_distance(x1, x2)
    assert dist >= 0, f"Test 7 failed: distance should be non-negative"
    print(f"  ✓ Distance: {dist:.4f}")
    
    # Test 8: Compute similarity
    sim = network.compute_similarity(x1, x2)
    print(f"  ✓ Similarity: {sim:.4f}")
    
    print("\nTesting contrastive loss...")
    
    # Test 9: Contrastive loss for similar pair
    loss_similar, _ = network.contrastive_loss(x1, x2, label=1)
    assert loss_similar >= 0, f"Test 9a failed: {loss_similar}"
    
    # Loss for dissimilar pair
    x1_neg, x2_neg, _ = negative_pairs[0]
    loss_dissimilar, _ = network.contrastive_loss(x1_neg, x2_neg, label=0)
    assert loss_dissimilar >= 0, f"Test 9b failed: {loss_dissimilar}"
    print(f"  ✓ Contrastive loss: similar={loss_similar:.4f}, dissimilar={loss_dissimilar:.4f}")
    
    print("\nTesting triplet loss...")
    
    # Test 10: Triplet loss
    anchor = create_features()
    positive = anchor + np.random.randn(input_dim) * 0.1
    negative = create_features()
    
    loss, _ = network.triplet_loss(anchor, positive, negative)
    assert loss >= 0, f"Test 10 failed: {loss}"
    print(f"  ✓ Triplet loss: {loss:.4f}")
    
    print("\nTesting training...")
    
    # Test 11: Train step
    batch = train_pairs[:16]
    loss = network.train_step(batch, loss_type="contrastive")
    assert loss >= 0, f"Test 11 failed: {loss}"
    print(f"  ✓ Train step loss: {loss:.4f}")
    
    # Test 12: Full training
    history = train_siamese(
        network, train_pairs, val_pairs,
        epochs=10, batch_size=16, loss_type="contrastive"
    )
    assert "train_loss" in history, "Test 12a failed"
    assert len(history["train_loss"]) > 0, "Test 12b failed"
    print(f"  ✓ Training complete: {len(history['train_loss'])} epochs")
    print(f"    Final loss: {history['train_loss'][-1]:.4f}")
    
    print("\nTesting pair creation...")
    
    # Test 13: Create pairs from labels
    features = {f"ing_{i}": create_features() for i in range(10)}
    substitutes = {
        "ing_0": ["ing_1", "ing_2"],
        "ing_3": ["ing_4"],
    }
    pairs = create_pairs_from_labels(features, substitutes)
    assert len(pairs) > 0, "Test 13a failed"
    assert all(len(p) == 3 for p in pairs), "Test 13b failed"
    pos_count = sum(1 for p in pairs if p[2] == 1)
    neg_count = sum(1 for p in pairs if p[2] == 0)
    print(f"  ✓ Created {len(pairs)} pairs: {pos_count} positive, {neg_count} negative")
    
    print("\nTesting hard negative mining...")
    
    # Test 14: Mine hard negatives
    anchor = features["ing_0"]
    candidates = list(features.values())
    positive_indices = [1, 2]  # ing_1 and ing_2 are substitutes
    hard_negatives = mine_hard_negatives(
        network, anchor, candidates, positive_indices, num_negatives=3
    )
    assert len(hard_negatives) <= 3, f"Test 14 failed: {len(hard_negatives)}"
    print(f"  ✓ Mined {len(hard_negatives)} hard negatives")
    
    print("\nTesting OnlineTripletMiner...")
    
    # Test 15: Triplet mining
    miner = OnlineTripletMiner(strategy="hard")
    batch_embeddings = np.random.randn(20, embedding_dim).astype(np.float32)
    batch_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    triplets = miner.mine(batch_embeddings, batch_labels)
    assert len(triplets) > 0, "Test 15a failed"
    assert all(len(t) == 3 for t in triplets), "Test 15b failed"
    print(f"  ✓ Mined {len(triplets)} triplets")
    
    print("\nTesting evaluation...")
    
    # Test 16: Evaluate network
    metrics = evaluate_siamese(network, val_pairs)
    assert "accuracy" in metrics, "Test 16a failed"
    assert 0 <= metrics["accuracy"] <= 1, f"Test 16b failed: {metrics['accuracy']}"
    print(f"  ✓ Evaluation metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}")
    
    print("\nTesting embedding visualization...")
    
    # Test 17: Extract embeddings
    viz_data = visualize_embeddings(network, features)
    assert "embeddings" in viz_data, "Test 17 failed"
    print(f"  ✓ Extracted embeddings for {len(features)} ingredients")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Siamese networks use shared encoders")
    print("2. Contrastive loss pulls similar pairs together")
    print("3. Triplet loss enforces relative distances")
    print("4. Hard negative mining improves training")
    print("5. Learned embeddings capture substitutability")
    print("\nNext: Cross-encoder for fine-grained scoring!")
