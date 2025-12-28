"""
Exercise 04: Word2Vec Skip-gram

In this exercise, you'll implement the Skip-gram model from Word2Vec.
Skip-gram learns word embeddings by predicting context words from a center word.

Example:
    Sentence: "I love cooking delicious recipes"
    With window_size=2, for center word "cooking":
    
    Predict: "I" from "cooking"
    Predict: "love" from "cooking"  
    Predict: "delicious" from "cooking"
    Predict: "recipes" from "cooking"

ML Relevance:
    Word2Vec revolutionized NLP:
    - First widely successful word embeddings
    - Key insight: similar contexts → similar meanings
    - "You shall know a word by the company it keeps"
    - Skip-gram works better for rare words than CBOW

Your Task:
    1. Implement create_skipgram_pairs() - generate training pairs
    2. Implement SkipGramModel class - the neural network
    3. Implement negative_sampling() - efficient training trick
    4. Understand the training objective

Run:
    python tutorials/07-embeddings/04_word2vec_skipgram.py
"""

from typing import List, Dict, Tuple, Set
import numpy as np
from collections import Counter


def create_skipgram_pairs(
    tokens: List[str],
    window_size: int = 2
) -> List[Tuple[str, str]]:
    """
    Create (center, context) pairs for Skip-gram training.
    
    For each word, pair it with words within window_size on each side.
    
    Args:
        tokens: List of tokens in a document
        window_size: Number of words to look on each side
        
    Returns:
        List of (center_word, context_word) tuples
        
    Example:
        >>> tokens = ["I", "love", "cooking"]
        >>> pairs = create_skipgram_pairs(tokens, window_size=1)
        >>> pairs
        [("I", "love"), ("love", "I"), ("love", "cooking"), ("cooking", "love")]
    """
    # Your solution here
    # Hints:
    # 1. For each position i in tokens
    # 2. Look at positions i-window to i+window (excluding i)
    # 3. Create pair (tokens[i], tokens[j]) for valid j
    pass


def build_vocabulary(
    tokens: List[str],
    min_count: int = 1
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from tokens.
    
    Args:
        tokens: List of all tokens
        min_count: Minimum frequency to include word
        
    Returns:
        Tuple of (word_to_idx, idx_to_word)
        
    Example:
        >>> tokens = ["a", "b", "a", "c", "a"]
        >>> w2i, i2w = build_vocabulary(tokens, min_count=2)
        >>> w2i
        {"a": 0}  # Only "a" appears >= 2 times
    """
    # Your solution here
    pass


def compute_word_frequencies(
    tokens: List[str],
    vocabulary: Dict[str, int]
) -> np.ndarray:
    """
    Compute frequency of each word in vocabulary.
    
    Args:
        tokens: List of all tokens
        vocabulary: Word to index mapping
        
    Returns:
        Array of frequencies (vocab_size,), normalized to sum to 1
        
    Example:
        >>> tokens = ["a", "b", "a", "a", "b"]
        >>> vocab = {"a": 0, "b": 1}
        >>> freqs = compute_word_frequencies(tokens, vocab)
        >>> freqs
        array([0.6, 0.4])  # 3/5, 2/5
    """
    # Your solution here
    pass


class SkipGramModel:
    """
    Skip-gram model for learning word embeddings.
    
    The model has two embedding matrices:
    - W_in: embeddings for center words (what we want to learn)
    - W_out: embeddings for context words (auxiliary)
    
    Prediction: P(context | center) = softmax(W_out @ W_in[center])
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        learning_rate: float = 0.01
    ):
        """
        Initialize Skip-gram model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            learning_rate: Learning rate for SGD
        """
        # Your solution here
        # Hints:
        # 1. Initialize W_in and W_out with small random values
        # 2. Store learning_rate
        pass
    
    def forward(
        self,
        center_idx: int,
        context_idx: int
    ) -> float:
        """
        Compute score for a (center, context) pair.
        
        score = dot(W_in[center], W_out[context])
        
        Args:
            center_idx: Index of center word
            context_idx: Index of context word
            
        Returns:
            Unnormalized score (logit)
        """
        # Your solution here
        pass
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def train_pair_negative_sampling(
        self,
        center_idx: int,
        context_idx: int,
        negative_indices: List[int]
    ) -> float:
        """
        Train on one (center, context) pair with negative sampling.
        
        The objective:
        - Maximize score for true (center, context) pair
        - Minimize score for (center, negative) pairs
        
        Args:
            center_idx: Index of center word
            context_idx: Index of true context word
            negative_indices: Indices of negative samples
            
        Returns:
            Loss value for this pair
        """
        # Your solution here
        # Hints (using sigmoid and binary cross-entropy):
        # 1. Positive: want sigmoid(score) close to 1
        #    loss_pos = -log(sigmoid(score_positive))
        # 2. Negatives: want sigmoid(score) close to 0
        #    loss_neg = sum(-log(1 - sigmoid(score_negative)))
        # 3. Compute gradients and update W_in, W_out
        pass
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get the learned embedding for a word."""
        return self.W_in[word_idx].copy()
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings (W_in matrix)."""
        return self.W_in.copy()


def negative_sampling(
    positive_idx: int,
    vocab_size: int,
    num_negatives: int,
    word_frequencies: np.ndarray,
    power: float = 0.75
) -> List[int]:
    """
    Sample negative examples for training.
    
    Negative sampling makes training efficient:
    - Instead of updating all vocab words
    - Only update a few random "negative" words
    
    Words are sampled proportional to frequency^power.
    power=0.75 reduces the dominance of frequent words.
    
    Args:
        positive_idx: Index of the positive (true) word
        vocab_size: Size of vocabulary
        num_negatives: Number of negative samples
        word_frequencies: Frequency of each word
        power: Power to raise frequencies to
        
    Returns:
        List of negative word indices (not including positive_idx)
        
    Example:
        >>> freqs = np.array([0.5, 0.3, 0.2])
        >>> negs = negative_sampling(0, 3, 2, freqs)
        >>> 0 not in negs  # Positive index excluded
        True
    """
    # Your solution here
    # Hints:
    # 1. Compute sampling probabilities: freq^power / sum(freq^power)
    # 2. Set probability of positive_idx to 0
    # 3. Renormalize
    # 4. Sample num_negatives indices
    pass


def train_skipgram(
    pairs: List[Tuple[int, int]],
    model: SkipGramModel,
    word_frequencies: np.ndarray,
    epochs: int = 1,
    num_negatives: int = 5
) -> List[float]:
    """
    Train Skip-gram model on (center, context) pairs.
    
    Args:
        pairs: List of (center_idx, context_idx) tuples
        model: SkipGramModel to train
        word_frequencies: Word frequency distribution
        epochs: Number of training epochs
        num_negatives: Number of negative samples per pair
        
    Returns:
        List of average loss per epoch
    """
    # Your solution here
    pass


def subsampling_probability(
    word_freq: float,
    threshold: float = 1e-5
) -> float:
    """
    Calculate probability of keeping a word (subsampling).
    
    Frequent words like "the", "a" are downsampled to:
    1. Speed up training
    2. Improve quality (rare words get more updates)
    
    P(keep) = sqrt(threshold / freq) + threshold / freq
    (clamped to [0, 1])
    
    Args:
        word_freq: Frequency of the word
        threshold: Subsampling threshold (typically 1e-5 to 1e-3)
        
    Returns:
        Probability of keeping the word
        
    Example:
        >>> subsampling_probability(0.01, 1e-5)  # Very frequent
        0.032  # Low probability of keeping
        >>> subsampling_probability(0.0001, 1e-5)  # Rare
        1.0  # Always keep
    """
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing create_skipgram_pairs...")
    
    # Test 1: Basic pairs
    tokens = ["I", "love", "cooking", "food"]
    pairs = create_skipgram_pairs(tokens, window_size=1)
    expected = [
        ("I", "love"),
        ("love", "I"), ("love", "cooking"),
        ("cooking", "love"), ("cooking", "food"),
        ("food", "cooking")
    ]
    assert len(pairs) == len(expected), f"Test 1a failed: {len(pairs)} vs {len(expected)}"
    print("  ✓ Correct number of pairs")
    
    # Test 2: Window size 2
    pairs = create_skipgram_pairs(tokens, window_size=2)
    # "love" should pair with "I", "cooking", "food"
    love_pairs = [(c, ctx) for c, ctx in pairs if c == "love"]
    assert len(love_pairs) == 3, f"Test 2 failed: {love_pairs}"
    print("  ✓ Window size 2 works")
    
    print("\nTesting build_vocabulary...")
    
    # Test 3: Basic vocabulary
    tokens = ["a", "b", "a", "c", "a", "b"]
    w2i, i2w = build_vocabulary(tokens, min_count=1)
    assert len(w2i) == 3, f"Test 3a failed: {len(w2i)}"
    assert i2w[w2i["a"]] == "a", "Test 3b failed"
    print("  ✓ Basic vocabulary built")
    
    # Test 4: Min count filtering
    w2i, i2w = build_vocabulary(tokens, min_count=2)
    assert "c" not in w2i, "Test 4 failed: 'c' should be filtered"
    print("  ✓ Min count filtering works")
    
    print("\nTesting compute_word_frequencies...")
    
    # Test 5: Frequency computation
    tokens = ["a", "a", "a", "b", "b"]
    vocab = {"a": 0, "b": 1}
    freqs = compute_word_frequencies(tokens, vocab)
    assert abs(freqs[0] - 0.6) < 0.01, f"Test 5a failed: {freqs[0]}"
    assert abs(freqs[1] - 0.4) < 0.01, f"Test 5b failed: {freqs[1]}"
    assert abs(freqs.sum() - 1.0) < 0.01, "Test 5c failed: should sum to 1"
    print("  ✓ Word frequencies computed")
    
    print("\nTesting SkipGramModel...")
    
    # Test 6: Model initialization
    model = SkipGramModel(vocab_size=100, embedding_dim=50)
    assert model.W_in.shape == (100, 50), f"Test 6a failed: {model.W_in.shape}"
    assert model.W_out.shape == (100, 50), f"Test 6b failed: {model.W_out.shape}"
    print("  ✓ Model initialized")
    
    # Test 7: Forward pass
    score = model.forward(0, 1)
    assert isinstance(score, (int, float)), f"Test 7 failed: {type(score)}"
    print("  ✓ Forward pass works")
    
    # Test 8: Training step
    loss = model.train_pair_negative_sampling(
        center_idx=0,
        context_idx=1,
        negative_indices=[2, 3, 4]
    )
    assert loss > 0, f"Test 8 failed: loss should be positive, got {loss}"
    print("  ✓ Training step works")
    
    # Test 9: Get embedding
    emb = model.get_embedding(0)
    assert emb.shape == (50,), f"Test 9 failed: {emb.shape}"
    print("  ✓ Get embedding works")
    
    print("\nTesting negative_sampling...")
    
    # Test 10: Negative samples don't include positive
    freqs = np.array([0.5, 0.3, 0.15, 0.05])
    for _ in range(10):
        negs = negative_sampling(0, 4, 2, freqs)
        assert 0 not in negs, f"Test 10 failed: {negs}"
    print("  ✓ Positive index excluded")
    
    # Test 11: Correct number of negatives
    negs = negative_sampling(0, 4, 3, freqs)
    assert len(negs) == 3, f"Test 11 failed: {len(negs)}"
    print("  ✓ Correct number of negatives")
    
    print("\nTesting train_skipgram...")
    
    # Test 12: Training loop
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1)]
    model = SkipGramModel(vocab_size=10, embedding_dim=32)
    freqs = np.ones(10) / 10
    losses = train_skipgram(pairs, model, freqs, epochs=2, num_negatives=3)
    assert len(losses) == 2, f"Test 12a failed: {len(losses)}"
    assert all(l > 0 for l in losses), "Test 12b failed: losses should be positive"
    print("  ✓ Training loop works")
    
    print("\nTesting subsampling_probability...")
    
    # Test 13: Frequent words have low keep probability
    p_freq = subsampling_probability(0.01, threshold=1e-5)
    p_rare = subsampling_probability(0.0001, threshold=1e-5)
    assert p_freq < p_rare, f"Test 13a failed: {p_freq} vs {p_rare}"
    assert 0 <= p_freq <= 1, f"Test 13b failed: {p_freq}"
    assert p_rare == 1.0 or p_rare > 0.9, f"Test 13c failed: rare should be kept {p_rare}"
    print("  ✓ Subsampling probabilities correct")
    
    print("\nTesting end-to-end training...")
    
    # Test 14: Full pipeline
    text = "I love cooking delicious food I love eating tasty recipes cooking is fun"
    tokens = text.lower().split()
    
    w2i, i2w = build_vocabulary(tokens, min_count=1)
    freqs = compute_word_frequencies(tokens, w2i)
    
    pairs = create_skipgram_pairs(tokens, window_size=2)
    indexed_pairs = [(w2i[c], w2i[ctx]) for c, ctx in pairs if c in w2i and ctx in w2i]
    
    model = SkipGramModel(vocab_size=len(w2i), embedding_dim=16)
    losses = train_skipgram(indexed_pairs, model, freqs, epochs=5, num_negatives=3)
    
    assert losses[-1] < losses[0] or losses[-1] < 2, f"Test 14 failed: loss should decrease {losses}"
    print("  ✓ End-to-end training works")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    print("\nKey takeaways:")
    print("- Skip-gram predicts context from center word")
    print("- Negative sampling makes training efficient")
    print("- Similar words get similar embeddings")
    print("- Subsampling helps with frequent words")
