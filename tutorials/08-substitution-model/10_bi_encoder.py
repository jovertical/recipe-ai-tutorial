# Problem 10: Bi-Encoder for Efficient Retrieval
#
# Build a bi-encoder that encodes ingredients separately for fast similarity
# search at scale.
#
# Example:
#   bi_encoder = BiEncoder(embedding_dim=64)
#   emb_butter = bi_encoder.encode("butter")
#   emb_margarine = bi_encoder.encode("margarine")
#   similarity = cosine_similarity(emb_butter, emb_margarine)
#
# ML Relevance: Bi-encoders enable pre-computed embeddings for fast retrieval,
# approximate nearest neighbor search, and scale to millions of ingredients.
#
# Your Task:
#   1. Implement BiEncoder with separate ingredient encoder
#   2. Implement index building for fast retrieval
#   3. Implement training with in-batch negatives
#   4. Implement retrieval pipeline with optional re-ranking


from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
import random
from collections import defaultdict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def batch_cosine_similarity(query: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all index vectors."""
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    index_norm = index / (np.linalg.norm(index, axis=1, keepdims=True) + 1e-8)
    return np.dot(index_norm, query_norm)


@dataclass  
class BiEncoderConfig:
    """Configuration for bi-encoder."""
    input_dim: int = 100  # Input feature dimension
    embedding_dim: int = 64  # Output embedding dimension
    hidden_dims: List[int] = None
    normalize: bool = True  # L2 normalize embeddings
    temperature: float = 0.05  # Temperature for softmax
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class IngredientEncoder:
    """
    Encoder network for ingredients.
    
    Transforms ingredient features to dense embeddings.
    """
    
    def __init__(self, config: BiEncoderConfig):
        """Initialize encoder."""
        self.config = config
        # Your solution here
        pass
    
    def encode(self, x: np.ndarray, normalize: bool = None) -> np.ndarray:
        """
        Encode a single ingredient.
        
        Args:
            x: Input features
            normalize: Whether to L2 normalize (uses config default if None)
            
        Returns:
            Embedding vector
        """
        # Your solution here
        pass
    
    def encode_batch(self, X: np.ndarray, normalize: bool = None) -> np.ndarray:
        """
        Encode batch of ingredients.
        
        Args:
            X: Input batch (n x input_dim)
            normalize: Whether to L2 normalize
            
        Returns:
            Embeddings (n x embedding_dim)
        """
        # Your solution here
        pass


class BiEncoder:
    """
    Bi-encoder for ingredient pair scoring.
    
    Encodes ingredients separately, then computes similarity.
    """
    
    def __init__(self, config: BiEncoderConfig = None):
        """
        Initialize bi-encoder.
        
        Args:
            config: Model configuration
        """
        self.config = config or BiEncoderConfig()
        self.encoder = IngredientEncoder(self.config)
        self.index = None
        self.index_names = None
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a single ingredient."""
        return self.encoder.encode(x)
    
    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """Encode batch of ingredients."""
        return self.encoder.encode_batch(X)
    
    def compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score
        """
        # Your solution here
        pass
    
    def score_pair(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> float:
        """
        Score a pair of ingredients.
        
        Args:
            features1: First ingredient features
            features2: Second ingredient features
            
        Returns:
            Substitutability score
        """
        # Your solution here
        pass
    
    def score_batch(
        self,
        queries: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """
        Score all query-candidate pairs.
        
        Args:
            queries: Query embeddings (m x dim)
            candidates: Candidate embeddings (n x dim)
            
        Returns:
            Scores matrix (m x n)
        """
        # Your solution here
        pass
    
    def in_batch_negatives_loss(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss using in-batch negatives.
        
        All other samples in the batch are used as negatives.
        
        Args:
            embeddings: Batch embeddings (n x dim)
            labels: Labels indicating positive pairs
            
        Returns:
            Tuple of (loss, gradients)
        """
        # Your solution here
        # Hints:
        # 1. Compute all pairwise similarities
        # 2. Apply temperature scaling
        # 3. Treat diagonal as positives, off-diagonal as negatives
        # 4. Compute InfoNCE/contrastive loss
        pass
    
    def contrastive_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negatives: np.ndarray
    ) -> float:
        """
        Compute contrastive loss.
        
        Args:
            anchor: Anchor embedding
            positive: Positive embedding
            negatives: Negative embeddings (n x dim)
            
        Returns:
            Loss value
        """
        # Your solution here
        pass
    
    def train_step(
        self,
        batch_features: np.ndarray,
        positive_pairs: List[Tuple[int, int]],
        learning_rate: float = 0.001
    ) -> float:
        """
        Perform one training step.
        
        Args:
            batch_features: Features for all items in batch
            positive_pairs: Indices of positive pairs
            learning_rate: Learning rate
            
        Returns:
            Batch loss
        """
        # Your solution here
        pass


class EmbeddingIndex:
    """
    Index for fast nearest neighbor retrieval.
    """
    
    def __init__(self):
        """Initialize empty index."""
        self.embeddings = None
        self.names = []
        self.metadata = {}
    
    def build(
        self,
        embeddings: Dict[str, np.ndarray],
        metadata: Dict[str, Dict] = None
    ):
        """
        Build index from embeddings.
        
        Args:
            embeddings: Name -> embedding dictionary
            metadata: Optional metadata for each item
        """
        # Your solution here
        pass
    
    def add(self, name: str, embedding: np.ndarray, meta: Dict = None):
        """Add single item to index."""
        # Your solution here
        pass
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        exclude: Set[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query embedding
            top_k: Number of results
            exclude: Items to exclude
            
        Returns:
            List of (name, score) tuples
        """
        # Your solution here
        pass
    
    def batch_search(
        self,
        queries: np.ndarray,
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for efficiency.
        
        Args:
            queries: Query embeddings (m x dim)
            top_k: Results per query
            
        Returns:
            List of result lists
        """
        # Your solution here
        pass
    
    def filter_search(
        self,
        query: np.ndarray,
        filter_fn: callable,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search with filtering.
        
        Args:
            query: Query embedding
            filter_fn: Function that takes name, returns True to include
            top_k: Number of results
            
        Returns:
            Filtered results
        """
        # Your solution here
        pass


class RetrievalPipeline:
    """
    Full retrieval pipeline with bi-encoder and optional re-ranking.
    """
    
    def __init__(
        self,
        bi_encoder: BiEncoder,
        index: EmbeddingIndex,
        reranker = None  # Optional cross-encoder
    ):
        """
        Initialize pipeline.
        
        Args:
            bi_encoder: Bi-encoder for initial retrieval
            index: Embedding index
            reranker: Optional cross-encoder for re-ranking
        """
        self.bi_encoder = bi_encoder
        self.index = index
        self.reranker = reranker
    
    def retrieve(
        self,
        query_features: np.ndarray,
        top_k: int = 10,
        rerank_top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top substitutes.
        
        Args:
            query_features: Query ingredient features
            top_k: Final number of results
            rerank_top_k: Number to re-rank (if reranker available)
            
        Returns:
            List of (name, score) tuples
        """
        # Your solution here
        pass
    
    def retrieve_with_filters(
        self,
        query_features: np.ndarray,
        dietary_filter: List[str] = None,
        category_filter: str = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve with filtering.
        
        Args:
            query_features: Query features
            dietary_filter: Required dietary tags
            category_filter: Required category
            top_k: Number of results
            
        Returns:
            Filtered results
        """
        # Your solution here
        pass


def train_bi_encoder(
    bi_encoder: BiEncoder,
    train_data: Dict[str, np.ndarray],
    positive_pairs: List[Tuple[str, str]],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict[str, List[float]]:
    """
    Train bi-encoder with in-batch negatives.
    
    Args:
        bi_encoder: Bi-encoder to train
        train_data: Ingredient name -> features
        positive_pairs: List of positive (substitute) pairs
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Training history
    """
    # Your solution here
    pass


def evaluate_retrieval(
    pipeline: RetrievalPipeline,
    test_queries: Dict[str, np.ndarray],
    ground_truth: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.
    
    Args:
        pipeline: Retrieval pipeline
        test_queries: Query name -> features
        ground_truth: Query name -> list of correct answers
        k_values: Values of k for recall@k
        
    Returns:
        Evaluation metrics
    """
    # Your solution here
    pass


def compare_with_cross_encoder(
    bi_encoder_results: List[Tuple[str, float]],
    cross_encoder_scores: Dict[str, float],
    ground_truth: List[str]
) -> Dict[str, float]:
    """
    Compare bi-encoder retrieval with cross-encoder re-ranking.
    
    Args:
        bi_encoder_results: Bi-encoder ranked results
        cross_encoder_scores: Cross-encoder scores for same pairs
        ground_truth: Correct answers
        
    Returns:
        Comparison metrics
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    input_dim = 100
    embedding_dim = 32
    
    # Create mock ingredient features
    def create_features():
        return np.random.randn(input_dim).astype(np.float32)
    
    ingredients = {
        "butter": create_features(),
        "margarine": create_features(),
        "coconut_oil": create_features(),
        "olive_oil": create_features(),
        "milk": create_features(),
        "oat_milk": create_features(),
        "almond_milk": create_features(),
        "egg": create_features(),
        "flax_egg": create_features(),
        "sugar": create_features(),
        "honey": create_features(),
        "salt": create_features(),
        "flour": create_features(),
    }
    
    # Make related items have similar features
    for base, related in [("butter", "margarine"), ("butter", "coconut_oil"),
                          ("milk", "oat_milk"), ("milk", "almond_milk")]:
        ingredients[related] = ingredients[base] + np.random.randn(input_dim) * 0.1
    
    config = BiEncoderConfig(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=[64, 48],
        normalize=True
    )
    
    print("Testing IngredientEncoder...")
    
    # Test 1: Encoder initialization
    encoder = IngredientEncoder(config)
    print("  ✓ Encoder initialized")
    
    # Test 2: Encode single
    emb = encoder.encode(ingredients["butter"])
    assert emb.shape == (embedding_dim,), f"Test 2a failed: {emb.shape}"
    assert abs(np.linalg.norm(emb) - 1.0) < 0.01, "Test 2b failed: not normalized"
    print(f"  ✓ Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
    
    # Test 3: Encode batch
    batch = np.array([ingredients[n] for n in ["butter", "milk", "egg"]])
    embs = encoder.encode_batch(batch)
    assert embs.shape == (3, embedding_dim), f"Test 3 failed: {embs.shape}"
    print(f"  ✓ Batch embedding shape: {embs.shape}")
    
    print("\nTesting BiEncoder...")
    
    # Test 4: BiEncoder initialization
    bi_encoder = BiEncoder(config)
    print("  ✓ BiEncoder initialized")
    
    # Test 5: Compute similarity
    emb_butter = bi_encoder.encode(ingredients["butter"])
    emb_margarine = bi_encoder.encode(ingredients["margarine"])
    emb_salt = bi_encoder.encode(ingredients["salt"])
    
    sim_similar = bi_encoder.compute_similarity(emb_butter, emb_margarine)
    sim_different = bi_encoder.compute_similarity(emb_butter, emb_salt)
    print(f"  ✓ Similarity: butter-margarine={sim_similar:.4f}, butter-salt={sim_different:.4f}")
    
    # Test 6: Score pair
    score = bi_encoder.score_pair(ingredients["butter"], ingredients["margarine"])
    assert -1 <= score <= 1, f"Test 6 failed: {score}"
    print(f"  ✓ Score pair: {score:.4f}")
    
    # Test 7: Score batch
    queries = np.array([ingredients["butter"], ingredients["milk"]])
    candidates = np.array([ingredients["margarine"], ingredients["oat_milk"], ingredients["salt"]])
    scores = bi_encoder.score_batch(queries, candidates)
    assert scores.shape == (2, 3), f"Test 7 failed: {scores.shape}"
    print(f"  ✓ Score batch shape: {scores.shape}")
    
    print("\nTesting loss functions...")
    
    # Test 8: Contrastive loss
    negatives = np.array([ingredients[n] for n in ["salt", "flour", "sugar"]])
    neg_embs = bi_encoder.encode_batch(negatives)
    loss = bi_encoder.contrastive_loss(emb_butter, emb_margarine, neg_embs)
    assert loss >= 0, f"Test 8 failed: {loss}"
    print(f"  ✓ Contrastive loss: {loss:.4f}")
    
    print("\nTesting EmbeddingIndex...")
    
    # Test 9: Build index
    index = EmbeddingIndex()
    embeddings_dict = {name: bi_encoder.encode(feat) for name, feat in ingredients.items()}
    index.build(embeddings_dict)
    print(f"  ✓ Index built with {len(index.names)} items")
    
    # Test 10: Search
    results = index.search(emb_butter, top_k=5, exclude={"butter"})
    assert len(results) == 5, f"Test 10a failed: {len(results)}"
    assert "butter" not in [r[0] for r in results], "Test 10b failed: should exclude self"
    print(f"  ✓ Top 5 for butter: {[(r[0], f'{r[1]:.2f}') for r in results]}")
    
    # Test 11: Batch search
    query_embs = np.array([emb_butter, bi_encoder.encode(ingredients["milk"])])
    batch_results = index.batch_search(query_embs, top_k=3)
    assert len(batch_results) == 2, f"Test 11 failed: {len(batch_results)}"
    print(f"  ✓ Batch search: {len(batch_results)} query results")
    
    # Test 12: Filter search
    def dietary_filter(name):
        vegan = {"margarine", "coconut_oil", "olive_oil", "oat_milk", "almond_milk"}
        return name in vegan
    
    filtered = index.filter_search(emb_butter, dietary_filter, top_k=3)
    for name, _ in filtered:
        assert dietary_filter(name), f"Test 12 failed: {name} doesn't pass filter"
    print(f"  ✓ Filtered search: {[r[0] for r in filtered]}")
    
    print("\nTesting RetrievalPipeline...")
    
    # Test 13: Pipeline initialization
    pipeline = RetrievalPipeline(bi_encoder, index)
    print("  ✓ Pipeline initialized")
    
    # Test 14: Retrieve
    results = pipeline.retrieve(ingredients["butter"], top_k=5)
    assert len(results) <= 5, f"Test 14 failed: {len(results)}"
    print(f"  ✓ Retrieve: {[(r[0], f'{r[1]:.2f}') for r in results]}")
    
    print("\nTesting training...")
    
    # Test 15: Create positive pairs
    positive_pairs = [
        ("butter", "margarine"), ("butter", "coconut_oil"),
        ("milk", "oat_milk"), ("milk", "almond_milk"),
        ("egg", "flax_egg"), ("sugar", "honey")
    ]
    
    # Test 16: Train
    history = train_bi_encoder(
        bi_encoder, ingredients, positive_pairs,
        epochs=10, batch_size=8
    )
    assert "loss" in history, "Test 16 failed"
    print(f"  ✓ Training complete: {len(history['loss'])} epochs")
    
    print("\nTesting evaluation...")
    
    # Test 17: Evaluate retrieval
    ground_truth = {
        "butter": ["margarine", "coconut_oil"],
        "milk": ["oat_milk", "almond_milk"]
    }
    test_queries = {k: ingredients[k] for k in ground_truth.keys()}
    
    # Rebuild index after training
    embeddings_dict = {name: bi_encoder.encode(feat) for name, feat in ingredients.items()}
    index.build(embeddings_dict)
    pipeline = RetrievalPipeline(bi_encoder, index)
    
    metrics = evaluate_retrieval(pipeline, test_queries, ground_truth, k_values=[1, 3, 5])
    assert "recall@1" in metrics, "Test 17 failed"
    print(f"  ✓ Evaluation metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Bi-encoders encode items independently")
    print("2. Pre-computed embeddings enable fast retrieval")
    print("3. In-batch negatives provide efficient training")
    print("4. Index structures enable nearest neighbor search")
    print("5. Filtering supports constraint-based retrieval")
    print("\nNext: Hard negative mining for better training!")
