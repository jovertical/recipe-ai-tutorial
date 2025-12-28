# Problem 3: Byte Pair Encoding (BPE) Algorithm
#
# Implement the BPE training algorithm from scratch.
# BPE iteratively merges the most frequent pair of tokens.
#
# How BPE works:
# 1. Start with characters as initial tokens
# 2. Count frequency of all adjacent token pairs
# 3. Merge the most frequent pair into a new token
# 4. Repeat until desired vocabulary size is reached
#
# Example:
#   corpus = ["low", "lower", "newest", "widest"]
#   # Initial tokens: ['l', 'o', 'w', '</w>'], ['l', 'o', 'w', 'e', 'r', '</w>'], ...
#   # After merging 'e' + 's' -> 'es':
#   # ['l', 'o', 'w', '</w>'], ['l', 'o', 'w', 'e', 'r', '</w>'], ['n', 'ew', 'es', 't', '</w>'], ...
#
# Constraints:
#   - Use </w> to mark word boundaries (end of word)
#   - Track merge operations in order (needed for encoding later)
#   - Handle ties by taking the lexicographically first pair
#
# ML Relevance: This is EXACTLY how GPT, Llama, and most modern tokenizers work.
# Understanding BPE helps you debug tokenization issues and choose vocab sizes.

from collections import Counter


def tokenize_word(word: str) -> list[str]:
    """
    Convert a word into initial character tokens with </w> at the end.

    Example: "hello" -> ["h", "e", "l", "l", "o", "</w>"]
    """
    # Your solution here
    pass


def get_pair_frequencies(tokenized_corpus: list[list[str]]) -> Counter:
    """
    Count frequencies of adjacent token pairs across the corpus.

    Args:
        tokenized_corpus: List of tokenized words, e.g. [["h", "e", "l", "l", "o", "</w>"], ...]

    Returns:
        Counter of (token1, token2) pairs and their frequencies

    Example:
        [["l", "o", "w", "</w>"], ["l", "o", "w", "e", "r", "</w>"]]
        -> Counter({("l", "o"): 2, ("o", "w"): 2, ("w", "</w>"): 1, ("w", "e"): 1, ...})
    """
    # Your solution here
    pass


def merge_pair(tokenized_corpus: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    """
    Merge all occurrences of a pair into a single token.

    Args:
        tokenized_corpus: List of tokenized words
        pair: The pair to merge, e.g. ("e", "s")

    Returns:
        New corpus with the pair merged

    Example:
        corpus = [["n", "e", "w", "e", "s", "t", "</w>"]]
        merge_pair(corpus, ("e", "s"))
        -> [["n", "e", "w", "es", "t", "</w>"]]
    """
    # Your solution here
    pass


def train_bpe(words: list[str], num_merges: int) -> tuple[list[list[str]], list[tuple[str, str]]]:
    """
    Train BPE by performing num_merges merge operations.

    Args:
        words: List of words to train on
        num_merges: Number of merge operations to perform

    Returns:
        Tuple of:
        - Final tokenized corpus
        - List of merge operations in order [(pair1), (pair2), ...]

    Example:
        words = ["low", "lower", "newest", "widest"]
        corpus, merges = train_bpe(words, num_merges=10)
        # merges might be: [("e", "s"), ("es", "t"), ("l", "o"), ...]
    """
    # Your solution here
    pass


def get_vocabulary(tokenized_corpus: list[list[str]]) -> set[str]:
    """
    Extract the vocabulary (unique tokens) from the tokenized corpus.
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: tokenize_word
    tokens = tokenize_word("hello")
    assert tokens == ["h", "e", "l", "l", "o", "</w>"], f"Test 1 failed: got {tokens}"

    # Test 2: tokenize_word with short word
    tokens = tokenize_word("a")
    assert tokens == ["a", "</w>"], f"Test 2 failed: got {tokens}"

    # Test 3: get_pair_frequencies
    corpus = [["l", "o", "w", "</w>"], ["l", "o", "w", "</w>"]]
    freqs = get_pair_frequencies(corpus)
    assert freqs[("l", "o")] == 2, f"Test 3a failed: expected 2, got {freqs[('l', 'o')]}"
    assert freqs[("o", "w")] == 2, f"Test 3b failed: expected 2, got {freqs[('o', 'w')]}"

    # Test 4: merge_pair basic
    corpus = [["l", "o", "w", "</w>"]]
    merged = merge_pair(corpus, ("l", "o"))
    assert merged == [["lo", "w", "</w>"]], f"Test 4 failed: got {merged}"

    # Test 5: merge_pair with multiple occurrences
    corpus = [["a", "b", "a", "b", "c", "</w>"]]
    merged = merge_pair(corpus, ("a", "b"))
    assert merged == [["ab", "ab", "c", "</w>"]], f"Test 5 failed: got {merged}"

    # Test 6: merge_pair across multiple words
    corpus = [["l", "o", "</w>"], ["l", "o", "w", "</w>"]]
    merged = merge_pair(corpus, ("l", "o"))
    assert merged == [["lo", "</w>"], ["lo", "w", "</w>"]], f"Test 6 failed: got {merged}"

    # Test 7: train_bpe produces correct number of merges
    words = ["low", "lower", "newest", "widest"]
    corpus, merges = train_bpe(words, num_merges=5)
    assert len(merges) == 5, f"Test 7 failed: expected 5 merges, got {len(merges)}"

    # Test 8: train_bpe merges reduce tokens
    words = ["aaa", "aaa", "aaa"]  # Should merge 'a' + 'a' first
    initial_corpus = [tokenize_word(w) for w in words]
    initial_tokens = sum(len(w) for w in initial_corpus)

    corpus, merges = train_bpe(words, num_merges=2)
    final_tokens = sum(len(w) for w in corpus)

    assert final_tokens < initial_tokens, "Test 8 failed: merges should reduce token count"

    # Test 9: get_vocabulary
    corpus = [["lo", "w", "</w>"], ["lo", "w", "er", "</w>"]]
    vocab = get_vocabulary(corpus)
    assert vocab == {"lo", "w", "</w>", "er"}, f"Test 9 failed: got {vocab}"

    # Test 10: BPE on recipe-like words
    words = ["baking", "baked", "bake", "baker"]
    corpus, merges = train_bpe(words, num_merges=8)
    vocab = get_vocabulary(corpus)
    # "bak" should likely be merged as common prefix
    assert any("bak" in token for token in vocab), "Test 10 failed: expected 'bak' substring in vocab"

    print("All tests passed!")
