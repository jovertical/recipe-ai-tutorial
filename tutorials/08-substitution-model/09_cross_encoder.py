# Problem 9: Cross-Encoder Architecture
#
# Build a cross-encoder that jointly processes ingredient pairs for more
# accurate substitution scoring.
#
# Example:
#   cross_encoder = CrossEncoder(vocab_size=1000, embedding_dim=64)
#   score = cross_encoder.score("butter and margarine for baking")
#   # Returns: 0.95 (high substitutability)
#
# ML Relevance: Cross-encoders use full attention between inputs (more
# accurate) while bi-encoders use separate embeddings (faster). Cross-encoders
# are used for re-ranking after retrieval.
#
# Your Task:
#   1. Implement CrossEncoder with self-attention
#   2. Implement pair encoding with special tokens
#   3. Implement re-ranking pipeline
#   4. Compare with bi-encoder performance


from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import random


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder."""
    vocab_size: int = 1000
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_attention_heads: int = 4
    num_layers: int = 2
    max_length: int = 32
    dropout_rate: float = 0.1


class SelfAttention:
    """
    Self-attention layer for cross-encoder.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        """
        Initialize self-attention.
        
        Args:
            embedding_dim: Input/output dimension
            num_heads: Number of attention heads
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Your solution here
        # Initialize Q, K, V projection weights
        pass
    
    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor (seq_len x embedding_dim)
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Your solution here
        # 1. Project to Q, K, V
        # 2. Split into heads
        # 3. Compute attention scores
        # 4. Apply mask if provided
        # 5. Compute weighted sum
        # 6. Concatenate heads and project
        pass
    
    def compute_attention_weights(
        self,
        query: np.ndarray,
        key: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """Compute attention weights from Q and K."""
        # Your solution here
        pass


class TransformerLayer:
    """
    Single transformer layer with self-attention and FFN.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int
    ):
        """Initialize transformer layer."""
        self.attention = SelfAttention(embedding_dim, num_heads)
        
        # Your solution here
        # Initialize FFN weights and layer norms
        pass
    
    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input (seq_len x embedding_dim)
            mask: Attention mask
            
        Returns:
            Output tensor
        """
        # Your solution here
        # 1. Self-attention with residual
        # 2. Layer norm
        # 3. FFN with residual
        # 4. Layer norm
        pass


class CrossEncoder:
    """
    Cross-encoder for ingredient pair scoring.
    
    Processes both ingredients jointly with full attention.
    """
    
    def __init__(self, config: CrossEncoderConfig = None):
        """
        Initialize cross-encoder.
        
        Args:
            config: Model configuration
        """
        self.config = config or CrossEncoderConfig()
        
        # Your solution here
        # Initialize:
        # - Token embeddings
        # - Position embeddings
        # - Transformer layers
        # - Classification head
        pass
    
    def encode_pair(
        self,
        ingredient1: str,
        ingredient2: str,
        context: str = None,
        vocab: Dict[str, int] = None
    ) -> np.ndarray:
        """
        Encode an ingredient pair as token sequence.
        
        Format: [CLS] ingredient1 [SEP] ingredient2 [SEP] context
        
        Args:
            ingredient1: First ingredient
            ingredient2: Second ingredient  
            context: Optional recipe context
            vocab: Token to index mapping
            
        Returns:
            Token indices array
        """
        # Your solution here
        pass
    
    def forward(
        self,
        token_ids: np.ndarray,
        attention_mask: np.ndarray = None
    ) -> Tuple[float, np.ndarray]:
        """
        Forward pass through cross-encoder.
        
        Args:
            token_ids: Input token indices
            attention_mask: Attention mask
            
        Returns:
            Tuple of (score, cls_embedding)
        """
        # Your solution here
        # 1. Look up token embeddings
        # 2. Add position embeddings
        # 3. Pass through transformer layers
        # 4. Extract [CLS] representation
        # 5. Pass through classification head
        pass
    
    def score(
        self,
        ingredient1: str,
        ingredient2: str,
        context: str = None
    ) -> float:
        """
        Score a substitution pair.
        
        Args:
            ingredient1: Original ingredient
            ingredient2: Candidate substitute
            context: Recipe context
            
        Returns:
            Substitutability score (0 to 1)
        """
        # Your solution here
        pass
    
    def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        contexts: List[str] = None
    ) -> np.ndarray:
        """
        Score multiple pairs.
        
        Args:
            pairs: List of (ingredient1, ingredient2) tuples
            contexts: Optional contexts for each pair
            
        Returns:
            Scores array
        """
        # Your solution here
        pass
    
    def get_attention_weights(
        self,
        ingredient1: str,
        ingredient2: str
    ) -> Dict[str, np.ndarray]:
        """
        Get attention weights for interpretability.
        
        Args:
            ingredient1: First ingredient
            ingredient2: Second ingredient
            
        Returns:
            Attention weights for each layer
        """
        # Your solution here
        pass


def cross_encoder_loss(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Binary cross-entropy loss for cross-encoder.
    
    Args:
        scores: Predicted scores
        labels: True labels (0 or 1)
        
    Returns:
        Loss value
    """
    # Your solution here
    pass


def train_cross_encoder(
    encoder: CrossEncoder,
    train_pairs: List[Tuple[str, str, int]],
    val_pairs: List[Tuple[str, str, int]],
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.001
) -> Dict[str, List[float]]:
    """
    Train the cross-encoder.
    
    Args:
        encoder: Cross-encoder to train
        train_pairs: Training pairs (ing1, ing2, label)
        val_pairs: Validation pairs
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Training history
    """
    # Your solution here
    pass


class ReRanker:
    """
    Re-rank candidates using cross-encoder.
    
    First-stage retrieval (bi-encoder) -> Second-stage re-ranking (cross-encoder)
    """
    
    def __init__(
        self,
        cross_encoder: CrossEncoder,
        top_k_rerank: int = 20
    ):
        """
        Initialize re-ranker.
        
        Args:
            cross_encoder: Trained cross-encoder
            top_k_rerank: Number of candidates to re-rank
        """
        self.cross_encoder = cross_encoder
        self.top_k_rerank = top_k_rerank
    
    def rerank(
        self,
        query: str,
        candidates: List[str],
        initial_scores: List[float] = None,
        context: str = None
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            query: Query ingredient
            candidates: Candidate substitutes
            initial_scores: Scores from first-stage retrieval
            context: Recipe context
            
        Returns:
            Re-ranked list of (candidate, score)
        """
        # Your solution here
        pass
    
    def two_stage_ranking(
        self,
        query: str,
        all_candidates: List[str],
        bi_encoder_scores: np.ndarray,
        context: str = None,
        final_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Full two-stage ranking pipeline.
        
        Args:
            query: Query ingredient
            all_candidates: All possible candidates
            bi_encoder_scores: Scores from bi-encoder
            context: Recipe context
            final_k: Number of final results
            
        Returns:
            Top-k results after re-ranking
        """
        # Your solution here
        pass


class CrossEncoderEnsemble:
    """
    Ensemble of cross-encoders for robustness.
    """
    
    def __init__(self, encoders: List[CrossEncoder]):
        """Initialize ensemble."""
        self.encoders = encoders
    
    def score(
        self,
        ingredient1: str,
        ingredient2: str,
        context: str = None,
        aggregation: str = "mean"
    ) -> float:
        """
        Ensemble score prediction.
        
        Args:
            ingredient1: First ingredient
            ingredient2: Second ingredient
            context: Recipe context
            aggregation: "mean", "max", or "vote"
            
        Returns:
            Aggregated score
        """
        # Your solution here
        pass


def compare_encoders(
    cross_encoder: CrossEncoder,
    bi_encoder_scores: Dict[Tuple[str, str], float],
    test_pairs: List[Tuple[str, str, int]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare cross-encoder vs bi-encoder performance.
    
    Args:
        cross_encoder: Trained cross-encoder
        bi_encoder_scores: Pre-computed bi-encoder scores
        test_pairs: Test pairs with labels
        
    Returns:
        Comparison metrics for each encoder type
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    # Create sample vocabulary
    vocab = {
        "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3,
        "butter": 4, "margarine": 5, "coconut": 6, "oil": 7,
        "olive": 8, "milk": 9, "oat": 10, "almond": 11,
        "egg": 12, "flax": 13, "baking": 14, "frying": 15,
        "sugar": 16, "honey": 17, "salt": 18, "flour": 19
    }
    
    config = CrossEncoderConfig(
        vocab_size=len(vocab),
        embedding_dim=32,
        hidden_dim=64,
        num_attention_heads=2,
        num_layers=1,
        max_length=16
    )
    
    print("Testing SelfAttention...")
    
    # Test 1: Self-attention initialization
    attention = SelfAttention(embedding_dim=32, num_heads=2)
    print("  ✓ SelfAttention initialized")
    
    # Test 2: Attention forward
    x = np.random.randn(5, 32).astype(np.float32)  # seq_len=5
    output, weights = attention.forward(x)
    assert output.shape == (5, 32), f"Test 2a failed: {output.shape}"
    assert weights.shape[0] == 2, f"Test 2b failed: {weights.shape}"  # num_heads
    print(f"  ✓ Attention output shape: {output.shape}")
    
    print("\nTesting TransformerLayer...")
    
    # Test 3: Transformer layer
    layer = TransformerLayer(embedding_dim=32, hidden_dim=64, num_heads=2)
    output = layer.forward(x)
    assert output.shape == x.shape, f"Test 3 failed: {output.shape}"
    print(f"  ✓ Transformer layer output shape: {output.shape}")
    
    print("\nTesting CrossEncoder...")
    
    # Test 4: Cross-encoder initialization
    encoder = CrossEncoder(config)
    print("  ✓ CrossEncoder initialized")
    
    # Test 5: Encode pair
    tokens = encoder.encode_pair("butter", "margarine", context="baking", vocab=vocab)
    assert len(tokens) > 0, "Test 5a failed"
    assert tokens[0] == vocab["[CLS]"], "Test 5b failed: should start with [CLS]"
    print(f"  ✓ Encoded pair length: {len(tokens)}")
    
    # Test 6: Forward pass
    attention_mask = np.ones(len(tokens))
    score, cls_emb = encoder.forward(tokens, attention_mask)
    assert 0 <= score <= 1, f"Test 6a failed: {score}"
    assert cls_emb.shape == (config.embedding_dim,), f"Test 6b failed: {cls_emb.shape}"
    print(f"  ✓ Score: {score:.4f}, CLS shape: {cls_emb.shape}")
    
    # Test 7: Score method
    score = encoder.score("butter", "margarine", context="baking")
    assert 0 <= score <= 1, f"Test 7 failed: {score}"
    print(f"  ✓ Score (butter, margarine): {score:.4f}")
    
    # Test 8: Score batch
    pairs = [("butter", "margarine"), ("butter", "salt"), ("milk", "oat")]
    scores = encoder.score_batch(pairs)
    assert len(scores) == 3, f"Test 8 failed: {len(scores)}"
    print(f"  ✓ Batch scores: {scores}")
    
    print("\nTesting attention weights...")
    
    # Test 9: Get attention weights
    attn_weights = encoder.get_attention_weights("butter", "margarine")
    assert len(attn_weights) > 0, "Test 9 failed"
    print(f"  ✓ Got attention weights for {len(attn_weights)} layers")
    
    print("\nTesting training...")
    
    # Test 10: Create training data
    train_pairs = [
        ("butter", "margarine", 1), ("butter", "coconut", 1),
        ("butter", "salt", 0), ("butter", "flour", 0),
        ("milk", "oat", 1), ("milk", "almond", 1),
        ("milk", "salt", 0), ("egg", "flax", 1),
    ] * 10
    val_pairs = train_pairs[:8]
    
    # Test 11: Train encoder
    history = train_cross_encoder(
        encoder, train_pairs, val_pairs,
        epochs=5, batch_size=8
    )
    assert "train_loss" in history, "Test 11 failed"
    print(f"  ✓ Training complete: {len(history['train_loss'])} epochs")
    
    print("\nTesting ReRanker...")
    
    # Test 12: Initialize re-ranker
    reranker = ReRanker(encoder, top_k_rerank=5)
    print("  ✓ ReRanker initialized")
    
    # Test 13: Re-rank
    candidates = ["margarine", "coconut", "salt", "flour", "olive"]
    initial_scores = [0.8, 0.7, 0.3, 0.2, 0.6]
    reranked = reranker.rerank("butter", candidates, initial_scores)
    assert len(reranked) == len(candidates), f"Test 13 failed: {len(reranked)}"
    print(f"  ✓ Re-ranked: {[(c, f'{s:.2f}') for c, s in reranked[:3]]}")
    
    print("\nTesting CrossEncoderEnsemble...")
    
    # Test 14: Ensemble
    encoders = [CrossEncoder(config) for _ in range(3)]
    ensemble = CrossEncoderEnsemble(encoders)
    score = ensemble.score("butter", "margarine", aggregation="mean")
    assert 0 <= score <= 1, f"Test 14 failed: {score}"
    print(f"  ✓ Ensemble score: {score:.4f}")
    
    print("\nTesting encoder comparison...")
    
    # Test 15: Compare encoders
    bi_encoder_scores = {
        ("butter", "margarine"): 0.9,
        ("butter", "salt"): 0.2,
        ("milk", "oat"): 0.85,
    }
    test_pairs = [("butter", "margarine", 1), ("butter", "salt", 0), ("milk", "oat", 1)]
    comparison = compare_encoders(encoder, bi_encoder_scores, test_pairs)
    assert "cross_encoder" in comparison, "Test 15a failed"
    assert "bi_encoder" in comparison, "Test 15b failed"
    print(f"  ✓ Comparison: {comparison}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Cross-encoders process pairs jointly with attention")
    print("2. Self-attention captures ingredient interactions")
    print("3. [CLS] token aggregates sequence information")
    print("4. Re-ranking improves retrieval quality")
    print("5. Trade-off: accuracy vs. computational cost")
    print("\nNext: Bi-encoder for efficient retrieval!")
