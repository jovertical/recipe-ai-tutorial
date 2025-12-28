# Problem 5: Train Word2Vec on Recipe Text
#
# Build a complete Word2Vec training pipeline for recipe text. This connects
# the concepts from the previous exercises into a working system.
#
# You'll implement:
# 1. prepare_corpus() - clean and tokenize recipe text
# 2. build_training_data() - create skipgram pairs with batching
# 3. Word2VecTrainer class - complete training pipeline
# 4. Save and load functions for model persistence
#
# Example:
#   corpus = ["mix flour and sugar", "add butter and eggs", "bake until golden"]
#   trainer = Word2VecTrainer(embedding_dim=100)
#   trainer.train(corpus, epochs=5)
#   trainer.most_similar("flour") → ["sugar", "butter", "eggs", ...]
#
# Constraints:
#   - Use negative sampling for efficient training
#   - Apply subsampling to reduce frequent word dominance
#   - Linear learning rate decay across training
#
# ML Relevance: Training embeddings on domain-specific text (recipes) captures
# ingredient relationships (flour goes with sugar), cooking action patterns
# (mix, stir, fold are similar), and domain vocabulary (sauté, braise, julienne).

from typing import List, Dict, Tuple, Iterator
import numpy as np
import re
from collections import Counter
import json


def prepare_corpus(recipes: List[str], min_word_length: int = 2, lowercase: bool = True) -> List[List[str]]:
    """Clean and tokenize recipe text."""
    # Your solution here
    pass


def build_vocabulary_with_counts(corpus: List[List[str]], min_count: int = 5, max_vocab_size: int = None) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """Build vocabulary with word counts, filtering by min_count."""
    # Your solution here
    pass


def create_training_batches(corpus: List[List[str]], vocabulary: Dict[str, int], window_size: int = 5, batch_size: int = 256) -> Iterator[List[Tuple[int, int]]]:
    """Create batches of (center, context) pairs for training."""
    # Your solution here
    pass


class Word2VecTrainer:
    """Complete Word2Vec training pipeline."""
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 5, min_count: int = 5, num_negatives: int = 5, learning_rate: float = 0.025, min_learning_rate: float = 0.0001):
        """Initialize trainer with hyperparameters."""
        # Your solution here
        pass
    
    def build_vocab(self, corpus: List[List[str]]) -> None:
        """Build vocabulary from corpus."""
        # Your solution here
        pass
    
    def train(self, corpus: List[List[str]], epochs: int = 5, report_every: int = 10000) -> List[float]:
        """Train Word2Vec. Returns list of average loss per epoch."""
        # Your solution here
        pass
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word."""
        # Your solution here
        pass
    
    def most_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words."""
        # Your solution here
        pass
    
    def save(self, path: str) -> None:
        """Save model to file."""
        # Your solution here
        pass
    
    @classmethod
    def load(cls, path: str) -> 'Word2VecTrainer':
        """Load model from file."""
        # Your solution here
        pass


def compute_word_frequencies_for_sampling(word_counts: Counter, vocabulary: Dict[str, int], power: float = 0.75) -> np.ndarray:
    """Compute sampling frequencies for negative sampling (freq^power)."""
    # Your solution here
    pass


def subsample_corpus(corpus: List[List[str]], word_counts: Counter, threshold: float = 1e-5) -> List[List[str]]:
    """Subsample frequent words from corpus."""
    # Your solution here
    pass


def analogy(word1: str, word2: str, word3: str, embeddings: np.ndarray, vocabulary: Dict[str, int], idx_to_word: Dict[int, str], top_k: int = 5) -> List[Tuple[str, float]]:
    """Solve word analogy: word1 is to word2 as word3 is to ?"""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    
    print("Testing prepare_corpus...")
    
    # Test 1: Basic preprocessing
    recipes = [
        "Mix 2 cups of flour!",
        "Add the butter and eggs.",
        "Bake at 350°F for 30 minutes."
    ]
    corpus = prepare_corpus(recipes)
    assert len(corpus) == 3, f"Test 1a failed: {len(corpus)}"
    assert all(isinstance(doc, list) for doc in corpus), "Test 1b failed"
    all_words = [w for doc in corpus for w in doc]
    assert all(w.isalpha() for w in all_words), f"Test 1c failed: {all_words}"
    print("  ✓ Corpus preparation works")
    
    # Test 2: Lowercase and filtering
    corpus = prepare_corpus(["HELLO World a"], min_word_length=2)
    assert "hello" in corpus[0], "Test 2a failed: should lowercase"
    assert "a" not in corpus[0], "Test 2b failed: short words filtered"
    print("  ✓ Lowercase and filtering work")
    
    print("\nTesting build_vocabulary_with_counts...")
    
    # Test 3: Vocabulary building
    corpus = [["a", "b"], ["a", "c"], ["a", "b"], ["a", "d"]]
    w2i, i2w, counts = build_vocabulary_with_counts(corpus, min_count=2)
    assert "a" in w2i, "Test 3a failed: 'a' should be in vocab"
    assert "b" in w2i, "Test 3b failed: 'b' should be in vocab"
    assert "c" not in w2i, "Test 3c failed: 'c' has count 1"
    assert counts["a"] == 4, f"Test 3d failed: {counts['a']}"
    print("  ✓ Vocabulary with counts works")
    
    # Test 4: Max vocab size
    corpus = [["a"] * 10, ["b"] * 5, ["c"] * 3, ["d"] * 1]
    w2i, _, counts = build_vocabulary_with_counts(corpus, min_count=1, max_vocab_size=2)
    assert len(w2i) == 2, f"Test 4 failed: {len(w2i)}"
    assert "a" in w2i and "b" in w2i, "Test 4b: should keep most frequent"
    print("  ✓ Max vocab size works")
    
    print("\nTesting create_training_batches...")
    
    # Test 5: Batch creation
    corpus = [["hello", "world", "test"], ["foo", "bar"]]
    vocab = {"hello": 0, "world": 1, "test": 2, "foo": 3, "bar": 4}
    batches = list(create_training_batches(corpus, vocab, window_size=1, batch_size=3))
    assert len(batches) > 0, "Test 5a failed: no batches"
    assert all(len(b) <= 3 for b in batches), "Test 5b failed: batch too large"
    print("  ✓ Batch creation works")
    
    print("\nTesting Word2VecTrainer...")
    
    # Test 6: Trainer initialization
    trainer = Word2VecTrainer(embedding_dim=32, window_size=2, min_count=1, num_negatives=3)
    assert trainer is not None, "Test 6 failed"
    print("  ✓ Trainer initialized")
    
    # Test 7: Build vocabulary
    corpus = [
        ["mix", "flour", "sugar"],
        ["add", "butter", "eggs"],
        ["mix", "butter", "flour"],
        ["bake", "until", "golden"]
    ]
    trainer.build_vocab(corpus)
    assert hasattr(trainer, 'vocabulary'), "Test 7a failed"
    assert len(trainer.vocabulary) > 0, "Test 7b failed"
    print("  ✓ Vocabulary built")
    
    # Test 8: Training
    losses = trainer.train(corpus, epochs=2, report_every=100)
    assert len(losses) == 2, f"Test 8a failed: {len(losses)}"
    assert all(l > 0 for l in losses), "Test 8b failed: losses should be positive"
    print("  ✓ Training works")
    
    # Test 9: Get embedding
    if "mix" in trainer.vocabulary:
        emb = trainer.get_embedding("mix")
        assert emb.shape == (32,), f"Test 9 failed: {emb.shape}"
        print("  ✓ Get embedding works")
    
    # Test 10: Most similar
    if "flour" in trainer.vocabulary:
        similar = trainer.most_similar("flour", top_k=3)
        assert len(similar) <= 3, f"Test 10a failed: {len(similar)}"
        assert all(isinstance(s, tuple) and len(s) == 2 for s in similar), "Test 10b failed"
        print("  ✓ Most similar works")
    
    print("\nTesting compute_word_frequencies_for_sampling...")
    
    # Test 11: Sampling frequencies
    word_counts = Counter({"a": 100, "b": 50, "c": 10})
    vocab = {"a": 0, "b": 1, "c": 2}
    freqs = compute_word_frequencies_for_sampling(word_counts, vocab, power=0.75)
    assert len(freqs) == 3, f"Test 11a failed: {len(freqs)}"
    assert abs(freqs.sum() - 1.0) < 0.01, f"Test 11b failed: {freqs.sum()}"
    assert freqs[0] < 100/160, "Test 11c failed: power should reduce dominance"
    print("  ✓ Sampling frequencies computed")
    
    print("\nTesting subsample_corpus...")
    
    # Test 12: Subsampling
    np.random.seed(42)
    corpus = [["the", "the", "cat", "sat"], ["the", "dog", "ran"]]
    word_counts = Counter(["the"] * 100 + ["cat", "dog", "sat", "ran"])
    subsampled = subsample_corpus(corpus, word_counts, threshold=0.01)
    assert len(subsampled) == 2, "Test 12a failed"
    assert all(isinstance(doc, list) for doc in subsampled), "Test 12b failed"
    print("  ✓ Subsampling works")
    
    print("\nTesting analogy...")
    
    # Test 13: Analogy
    embeddings = np.array([
        [1, 0, 0],
        [0.9, 0.1, 0],
        [0.5, 0, 0.5],
        [0.4, 0.1, 0.5],
    ])
    vocab = {"king": 0, "queen": 1, "man": 2, "woman": 3}
    i2w = {v: k for k, v in vocab.items()}
    results = analogy("man", "king", "woman", embeddings, vocab, i2w, top_k=2)
    assert len(results) <= 2, f"Test 13a failed: {len(results)}"
    assert all(isinstance(r, tuple) for r in results), "Test 13b failed"
    print("  ✓ Analogy function works")
    
    print("\nTesting save and load...")
    
    # Test 14: Save and load
    import tempfile
    import os
    
    trainer = Word2VecTrainer(embedding_dim=16, min_count=1)
    corpus = [["test", "save", "load"], ["save", "model", "test"]]
    trainer.train(corpus, epochs=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.json")
        trainer.save(path)
        assert os.path.exists(path), "Test 14a failed: file not saved"
        
        loaded = Word2VecTrainer.load(path)
        assert loaded is not None, "Test 15a failed"
        assert len(loaded.vocabulary) == len(trainer.vocabulary), "Test 15b failed"
        print("  ✓ Save and load work")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
