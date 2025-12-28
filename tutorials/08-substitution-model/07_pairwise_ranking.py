# Problem 7: Pairwise Ranking Model
#
# Build a pairwise ranking model that learns to rank substitutes by quality.
# Instead of binary yes/no, it learns relative ordering.
#
# Example:
#   ranker = PairwiseRanker(embedding_dim=64)
#   ranker.compare("butter", "margarine", "olive_oil")
#   # Returns: True (margarine > olive_oil as butter substitute)
#
# ML Relevance: Pairwise ranking learns relative preferences, is more robust
# to annotation inconsistencies, natural for "which is better?" questions,
# and is the foundation for BPR, RankNet, and LambdaRank.
#
# Your Task:
#   1. Implement PairwiseRanker with margin-based loss
#   2. Implement training with preference pairs
#   3. Implement NDCG and MRR evaluation
#   4. Implement list-wise ranking from pairwise model


from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import random
from collections import defaultdict


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class PreferencePair:
    """A preference pair for training."""
    query: str  # The ingredient to substitute
    preferred: str  # Better substitute
    non_preferred: str  # Worse substitute
    margin: float = 1.0  # How much better preferred is
    
    def __repr__(self):
        return f"{self.query}: {self.preferred} > {self.non_preferred}"


@dataclass
class RankingConfig:
    """Configuration for ranking model training."""
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 50
    margin: float = 1.0  # Margin for hinge loss
    weight_decay: float = 0.0001


class PairwiseRanker:
    """
    Pairwise ranking model for substitution quality.
    
    Learns to score substitutes such that better substitutes
    have higher scores.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128
    ):
        """
        Initialize ranker.
        
        Args:
            embedding_dim: Dimension of ingredient embeddings
            hidden_dim: Hidden layer dimension
        """
        # Input: concat(query_emb, candidate_emb, |diff|) = 3 * embedding_dim
        self.input_dim = 3 * embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Your solution here
        # Initialize scoring network
        pass
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        # Your solution here
        pass
    
    def create_features(
        self,
        query_emb: np.ndarray,
        candidate_emb: np.ndarray
    ) -> np.ndarray:
        """
        Create feature vector for query-candidate pair.
        
        Args:
            query_emb: Query ingredient embedding
            candidate_emb: Candidate substitute embedding
            
        Returns:
            Feature vector
        """
        # Your solution here
        pass
    
    def score(
        self,
        query_emb: np.ndarray,
        candidate_emb: np.ndarray
    ) -> float:
        """
        Compute relevance score for a candidate substitute.
        
        Args:
            query_emb: Query embedding
            candidate_emb: Candidate embedding
            
        Returns:
            Relevance score (higher = better substitute)
        """
        # Your solution here
        pass
    
    def compare(
        self,
        query_emb: np.ndarray,
        candidate_a_emb: np.ndarray,
        candidate_b_emb: np.ndarray
    ) -> bool:
        """
        Compare two candidates for a query.
        
        Args:
            query_emb: Query embedding
            candidate_a_emb: First candidate
            candidate_b_emb: Second candidate
            
        Returns:
            True if candidate_a is better substitute than candidate_b
        """
        # Your solution here
        pass
    
    def rank_candidates(
        self,
        query_emb: np.ndarray,
        candidate_embs: List[np.ndarray],
        candidate_names: List[str] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank all candidates for a query.
        
        Args:
            query_emb: Query embedding
            candidate_embs: List of candidate embeddings
            candidate_names: Optional names for candidates
            
        Returns:
            List of (index, score) sorted by score descending
        """
        # Your solution here
        pass
    
    def forward(
        self,
        features: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward pass through scoring network."""
        # Your solution here
        pass
    
    def compute_pairwise_loss(
        self,
        query_emb: np.ndarray,
        preferred_emb: np.ndarray,
        non_preferred_emb: np.ndarray,
        margin: float = 1.0
    ) -> Tuple[float, Dict]:
        """
        Compute pairwise hinge loss.
        
        Loss = max(0, margin - (score_preferred - score_non_preferred))
        
        Args:
            query_emb: Query embedding
            preferred_emb: Better substitute embedding
            non_preferred_emb: Worse substitute embedding
            margin: Minimum desired score difference
            
        Returns:
            Tuple of (loss, gradients_cache)
        """
        # Your solution here
        pass
    
    def train_step(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        learning_rate: float,
        margin: float = 1.0
    ) -> float:
        """
        Perform one training step.
        
        Args:
            batch: List of (query, preferred, non_preferred) embedding tuples
            learning_rate: Learning rate
            margin: Margin for hinge loss
            
        Returns:
            Batch loss
        """
        # Your solution here
        pass


def train_pairwise_ranker(
    ranker: PairwiseRanker,
    train_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    val_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: RankingConfig
) -> Dict[str, List[float]]:
    """
    Train the pairwise ranker.
    
    Args:
        ranker: Ranker to train
        train_pairs: Training preference pairs
        val_pairs: Validation pairs
        config: Training configuration
        
    Returns:
        Training history
    """
    # Your solution here
    pass


def evaluate_pairwise_accuracy(
    ranker: PairwiseRanker,
    test_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> float:
    """
    Evaluate pairwise accuracy.
    
    What fraction of pairs does the model order correctly?
    
    Args:
        ranker: Trained ranker
        test_pairs: Test preference pairs
        
    Returns:
        Accuracy (0 to 1)
    """
    # Your solution here
    pass


def compute_ndcg(
    relevance_scores: List[float],
    predicted_ranking: List[int],
    k: int = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    
    NDCG measures ranking quality considering graded relevance.
    
    Args:
        relevance_scores: True relevance of each item
        predicted_ranking: Predicted ranking (indices)
        k: Compute NDCG@k (None = all)
        
    Returns:
        NDCG score (0 to 1)
        
    Example:
        >>> relevance = [3, 2, 1, 0]  # Item 0 is most relevant
        >>> predicted = [0, 1, 2, 3]  # Perfect ranking
        >>> compute_ndcg(relevance, predicted)
        1.0
    """
    # Your solution here
    # DCG = sum(relevance[i] / log2(i+2))
    # IDCG = DCG of perfect ranking
    # NDCG = DCG / IDCG
    pass


def compute_mrr(
    queries: List[str],
    true_best: Dict[str, str],
    ranker: PairwiseRanker,
    embeddings: Dict[str, np.ndarray],
    candidates: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR = mean(1 / rank of first correct answer)
    
    Args:
        queries: Query ingredients
        true_best: Best substitute for each query
        ranker: Trained ranker
        embeddings: Ingredient embeddings
        candidates: All possible substitutes
        
    Returns:
        MRR score
    """
    # Your solution here
    pass


def generate_preference_pairs(
    substitution_scores: Dict[str, Dict[str, float]],
    embeddings: Dict[str, np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate training pairs from substitution scores.
    
    Args:
        substitution_scores: {query: {candidate: score}}
        embeddings: Ingredient embeddings
        
    Returns:
        List of (query_emb, preferred_emb, non_preferred_emb)
    """
    # Your solution here
    pass


class ListwiseRanker:
    """
    Convert pairwise model to listwise ranking.
    
    Uses aggregated pairwise comparisons for full list ranking.
    """
    
    def __init__(self, pairwise_ranker: PairwiseRanker):
        """
        Initialize listwise ranker.
        
        Args:
            pairwise_ranker: Trained pairwise model
        """
        # Your solution here
        pass
    
    def rank(
        self,
        query_emb: np.ndarray,
        candidate_embs: List[np.ndarray],
        method: str = "score"
    ) -> List[int]:
        """
        Rank candidates for a query.
        
        Args:
            query_emb: Query embedding
            candidate_embs: Candidate embeddings
            method: Ranking method
                - "score": Use raw scores
                - "copeland": Use pairwise wins
                - "borda": Use Borda count
                
        Returns:
            Indices in ranked order (best first)
        """
        # Your solution here
        pass
    
    def copeland_ranking(
        self,
        query_emb: np.ndarray,
        candidate_embs: List[np.ndarray]
    ) -> List[int]:
        """
        Rank using Copeland's method (pairwise wins - losses).
        
        More robust than raw scores.
        """
        # Your solution here
        pass


class BPRModel:
    """
    Bayesian Personalized Ranking model.
    
    Optimizes the posterior probability of correct pairwise ordering.
    """
    
    def __init__(self, embedding_dim: int = 64):
        """Initialize BPR model."""
        # Your solution here
        pass
    
    def compute_bpr_loss(
        self,
        query_emb: np.ndarray,
        positive_emb: np.ndarray,
        negative_emb: np.ndarray
    ) -> float:
        """
        Compute BPR loss.
        
        Loss = -log(sigmoid(score_pos - score_neg))
        """
        # Your solution here
        pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    embedding_dim = 64
    
    # Create mock embeddings with structure
    def create_embedding():
        emb = np.random.randn(embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)
    
    # Create ingredient embeddings
    ingredients = ["butter", "margarine", "coconut_oil", "olive_oil", "lard",
                   "milk", "oat_milk", "almond_milk", "cream"]
    embeddings = {ing: create_embedding() for ing in ingredients}
    
    # Make related items similar
    for base, related in [("butter", "margarine"), ("butter", "coconut_oil"),
                          ("milk", "oat_milk"), ("milk", "almond_milk")]:
        embeddings[related] = embeddings[base] + np.random.randn(embedding_dim) * 0.1
        embeddings[related] /= np.linalg.norm(embeddings[related])
    
    # Create preference pairs (butter: margarine > olive_oil > lard)
    train_pairs = []
    for _ in range(50):
        train_pairs.append((embeddings["butter"], embeddings["margarine"], embeddings["olive_oil"]))
        train_pairs.append((embeddings["butter"], embeddings["coconut_oil"], embeddings["lard"]))
        train_pairs.append((embeddings["milk"], embeddings["oat_milk"], embeddings["cream"]))
    
    val_pairs = train_pairs[:10]
    
    print("Testing PairwiseRanker...")
    
    # Test 1: Initialize ranker
    ranker = PairwiseRanker(embedding_dim=embedding_dim, hidden_dim=128)
    print("  ✓ Ranker initialized")
    
    # Test 2: Create features
    features = ranker.create_features(embeddings["butter"], embeddings["margarine"])
    assert features.shape == (3 * embedding_dim,), f"Test 2 failed: {features.shape}"
    print(f"  ✓ Feature shape: {features.shape}")
    
    # Test 3: Score
    score = ranker.score(embeddings["butter"], embeddings["margarine"])
    assert isinstance(score, (float, np.floating)), f"Test 3 failed: {type(score)}"
    print(f"  ✓ Score: {score:.4f}")
    
    # Test 4: Compare
    result = ranker.compare(
        embeddings["butter"],
        embeddings["margarine"],
        embeddings["olive_oil"]
    )
    assert isinstance(result, bool), f"Test 4 failed: {type(result)}"
    print(f"  ✓ Compare result: {result}")
    
    # Test 5: Rank candidates
    candidate_embs = [embeddings[ing] for ing in ["margarine", "olive_oil", "lard"]]
    ranking = ranker.rank_candidates(embeddings["butter"], candidate_embs)
    assert len(ranking) == 3, f"Test 5a failed: {len(ranking)}"
    assert all(isinstance(r[1], (float, np.floating)) for r in ranking), "Test 5b failed"
    print(f"  ✓ Ranking: {ranking}")
    
    print("\nTesting pairwise loss...")
    
    # Test 6: Compute loss
    loss, cache = ranker.compute_pairwise_loss(
        embeddings["butter"],
        embeddings["margarine"],
        embeddings["olive_oil"],
        margin=1.0
    )
    assert loss >= 0, f"Test 6 failed: loss should be non-negative, got {loss}"
    print(f"  ✓ Pairwise loss: {loss:.4f}")
    
    print("\nTesting training...")
    
    # Test 7: Train ranker
    config = RankingConfig(learning_rate=0.01, epochs=20, batch_size=16)
    history = train_pairwise_ranker(ranker, train_pairs, val_pairs, config)
    assert "train_loss" in history, "Test 7 failed"
    print(f"  ✓ Training complete: {len(history['train_loss'])} epochs")
    print(f"    Final loss: {history['train_loss'][-1]:.4f}")
    
    # Test 8: Pairwise accuracy
    accuracy = evaluate_pairwise_accuracy(ranker, val_pairs)
    assert 0 <= accuracy <= 1, f"Test 8 failed: {accuracy}"
    print(f"  ✓ Pairwise accuracy: {accuracy:.2%}")
    
    print("\nTesting NDCG...")
    
    # Test 9: NDCG computation
    relevance = [3, 2, 1, 0]
    perfect_ranking = [0, 1, 2, 3]
    ndcg = compute_ndcg(relevance, perfect_ranking)
    assert abs(ndcg - 1.0) < 0.01, f"Test 9a failed: perfect ranking should have NDCG=1, got {ndcg}"
    
    # Imperfect ranking
    bad_ranking = [3, 2, 1, 0]  # Worst first
    ndcg_bad = compute_ndcg(relevance, bad_ranking)
    assert ndcg_bad < ndcg, f"Test 9b failed: bad ranking should have lower NDCG"
    print(f"  ✓ NDCG: perfect={ndcg:.4f}, reversed={ndcg_bad:.4f}")
    
    print("\nTesting MRR...")
    
    # Test 10: MRR computation
    true_best = {"butter": "margarine", "milk": "oat_milk"}
    mrr = compute_mrr(
        queries=["butter", "milk"],
        true_best=true_best,
        ranker=ranker,
        embeddings=embeddings,
        candidates=["margarine", "olive_oil", "oat_milk", "cream"]
    )
    assert 0 <= mrr <= 1, f"Test 10 failed: {mrr}"
    print(f"  ✓ MRR: {mrr:.4f}")
    
    print("\nTesting ListwiseRanker...")
    
    # Test 11: Initialize listwise ranker
    listwise = ListwiseRanker(ranker)
    print("  ✓ ListwiseRanker initialized")
    
    # Test 12: Score-based ranking
    ranking = listwise.rank(
        embeddings["butter"],
        [embeddings[ing] for ing in ["margarine", "olive_oil", "lard"]],
        method="score"
    )
    assert len(ranking) == 3, f"Test 12 failed: {len(ranking)}"
    print(f"  ✓ Score ranking: {ranking}")
    
    # Test 13: Copeland ranking
    ranking = listwise.copeland_ranking(
        embeddings["butter"],
        [embeddings[ing] for ing in ["margarine", "olive_oil", "lard"]]
    )
    assert len(ranking) == 3, f"Test 13 failed: {len(ranking)}"
    print(f"  ✓ Copeland ranking: {ranking}")
    
    print("\nTesting generate_preference_pairs...")
    
    # Test 14: Generate pairs from scores
    substitution_scores = {
        "butter": {"margarine": 0.9, "coconut_oil": 0.8, "olive_oil": 0.6, "lard": 0.4}
    }
    pairs = generate_preference_pairs(substitution_scores, embeddings)
    assert len(pairs) > 0, "Test 14a failed"
    assert all(len(p) == 3 for p in pairs), "Test 14b failed"
    print(f"  ✓ Generated {len(pairs)} preference pairs")
    
    print("\nTesting BPRModel...")
    
    # Test 15: BPR loss
    bpr = BPRModel(embedding_dim=embedding_dim)
    loss = bpr.compute_bpr_loss(
        embeddings["butter"],
        embeddings["margarine"],
        embeddings["lard"]
    )
    assert loss >= 0, f"Test 15 failed: {loss}"
    print(f"  ✓ BPR loss: {loss:.4f}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Pairwise ranking learns relative preferences")
    print("2. Margin-based loss ensures score separation")
    print("3. NDCG evaluates ranking quality with graded relevance")
    print("4. MRR focuses on first correct result")
    print("5. Copeland aggregates pairwise comparisons")
    print("\nNext: Siamese networks for learning embeddings!")
