# Problem 5: Subword Intuition
#
# Understand WHY subword tokenization works by analyzing its properties.
# Compare character, word, and subword approaches empirically.
#
# Key insights to discover:
# 1. Subwords handle rare/unknown words gracefully
# 2. Subwords balance vocabulary size vs sequence length
# 3. Morphologically related words share tokens
#
# Example:
#   Word tokenizer: "unhappiness" -> ["<UNK>"] (if rare)
#   Char tokenizer: "unhappiness" -> ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]
#   BPE tokenizer:  "unhappiness" -> ["un", "happiness"] (shares "un" with "unlikely")
#
# ML Relevance: This builds intuition for why every modern LLM uses subword
# tokenization. Understanding this helps you choose vocabulary sizes and debug
# tokenization issues.

from collections import Counter


def analyze_morphology(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """
    Show how BPE breaks down a word, revealing morphological structure.

    Apply merges to see the final tokenization.
    Returns the list of subword tokens.

    Example:
        word = "unhappiness"
        merges learned from corpus with "un", "happy", "ness" patterns
        -> ["un", "happi", "ness", "</w>"] (approximately)
    """
    # Your solution here
    # Start with characters, apply merges in order
    pass


def compare_tokenizations(
    word: str,
    char_tokens: list[str],
    word_tokens: list[str],
    bpe_tokens: list[str]
) -> dict:
    """
    Compare the three tokenization approaches for a single word.

    Returns a dict with:
    - 'char_len': number of character tokens
    - 'word_len': number of word tokens (1 or 0 if OOV)
    - 'bpe_len': number of BPE tokens
    - 'char_is_oov': False (chars never OOV)
    - 'word_is_oov': True if word not in vocabulary
    - 'bpe_is_oov': False (BPE falls back to chars)
    """
    # Your solution here
    pass


def compression_ratio(original_text: str, tokens: list[str]) -> float:
    """
    Calculate how efficiently the tokenization compresses text.

    Compression ratio = number of characters / number of tokens

    Higher ratio = more compression = fewer tokens for same text

    Example:
        text = "hello" (5 chars)
        tokens = ["hel", "lo"] (2 tokens)
        ratio = 5 / 2 = 2.5
    """
    # Your solution here
    pass


def vocabulary_coverage(vocab: set[str], test_words: list[str]) -> dict:
    """
    Analyze how well a vocabulary covers test words.

    Returns:
    - 'coverage': fraction of test words fully in vocab
    - 'oov_words': list of out-of-vocabulary words
    - 'oov_rate': fraction of words that are OOV
    """
    # Your solution here
    pass


def shared_subwords(words: list[str], tokenizer_func) -> dict:
    """
    Find subwords that are shared across multiple words.

    This shows how BPE captures morphological patterns.

    Args:
        words: List of words to analyze
        tokenizer_func: Function that takes a word and returns tokens

    Returns:
        Dict mapping shared subwords to the words containing them

    Example:
        words = ["unhappy", "unlikely", "unable"]
        -> {"un": ["unhappy", "unlikely", "unable"], ...}
    """
    # Your solution here
    pass


def analyze_rare_word_handling(
    rare_word: str,
    char_tokenizer,
    word_tokenizer,
    bpe_tokenizer
) -> dict:
    """
    Show how each tokenizer handles a rare/unseen word.

    Returns dict with:
    - 'char': character tokenization (always works)
    - 'word': word tokenization (likely <UNK>)
    - 'bpe': BPE tokenization (falls back gracefully)
    - 'char_len': length of char tokenization
    - 'bpe_len': length of BPE tokenization
    - 'info_preserved': whether meaning can be recovered
    """
    # Your solution here
    pass


def sequence_length_analysis(texts: list[str], tokenizer_func) -> dict:
    """
    Analyze sequence lengths produced by a tokenizer.

    Returns:
    - 'mean_length': average number of tokens per text
    - 'max_length': maximum sequence length
    - 'min_length': minimum sequence length
    - 'total_tokens': total tokens across all texts
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # Test 1: analyze_morphology
    merges = [("u", "n"), ("h", "a"), ("ha", "p"), ("hap", "p"), ("happ", "y")]
    tokens = analyze_morphology("unhappy", merges)
    assert isinstance(tokens, list), "Test 1a failed: should return list"
    assert len(tokens) < len("unhappy") + 1, "Test 1b failed: should reduce token count"

    # Test 2: compare_tokenizations
    result = compare_tokenizations(
        word="hello",
        char_tokens=["h", "e", "l", "l", "o"],
        word_tokens=["hello"],
        bpe_tokens=["hel", "lo"]
    )
    assert result["char_len"] == 5, f"Test 2a failed: got {result['char_len']}"
    assert result["word_len"] == 1, f"Test 2b failed: got {result['word_len']}"
    assert result["bpe_len"] == 2, f"Test 2c failed: got {result['bpe_len']}"

    # Test 3: compare_tokenizations with OOV
    result = compare_tokenizations(
        word="xyz",
        char_tokens=["x", "y", "z"],
        word_tokens=["<UNK>"],
        bpe_tokens=["x", "y", "z"]
    )
    assert result["word_is_oov"] == True, "Test 3 failed: word should be OOV"

    # Test 4: compression_ratio
    ratio = compression_ratio("hello", ["hel", "lo"])
    assert ratio == 2.5, f"Test 4 failed: expected 2.5, got {ratio}"

    # Test 5: compression_ratio with single token
    ratio = compression_ratio("hello", ["hello"])
    assert ratio == 5.0, f"Test 5 failed: expected 5.0, got {ratio}"

    # Test 6: vocabulary_coverage
    vocab = {"hello", "world", "the"}
    test_words = ["hello", "world", "universe", "the", "galaxy"]
    coverage = vocabulary_coverage(vocab, test_words)
    assert coverage["coverage"] == 0.6, f"Test 6a failed: expected 0.6, got {coverage['coverage']}"
    assert "universe" in coverage["oov_words"], "Test 6b failed: 'universe' should be OOV"

    # Test 7: shared_subwords
    def mock_tokenizer(word):
        if word.startswith("un"):
            return ["un", word[2:]]
        return [word]

    shared = shared_subwords(["unhappy", "unlikely", "happy"], mock_tokenizer)
    assert "un" in shared, "Test 7a failed: 'un' should be shared"
    assert len(shared["un"]) == 2, f"Test 7b failed: 'un' should appear in 2 words"

    # Test 8: sequence_length_analysis
    def simple_tokenizer(text):
        return text.split()

    texts = ["hello world", "a b c d", "one"]
    analysis = sequence_length_analysis(texts, simple_tokenizer)
    assert analysis["mean_length"] == (2 + 4 + 1) / 3, f"Test 8a failed: got {analysis['mean_length']}"
    assert analysis["max_length"] == 4, f"Test 8b failed: got {analysis['max_length']}"
    assert analysis["min_length"] == 1, f"Test 8c failed: got {analysis['min_length']}"

    # Test 9: Real-world intuition test
    # BPE should compress common patterns more than rare ones
    common_text = "the the the"
    rare_text = "xyz xyz xyz"

    def common_tokenizer(text):
        return ["the"] * text.count("the") if "the" in text else list(text.replace(" ", ""))

    common_ratio = compression_ratio(common_text.replace(" ", ""), common_tokenizer(common_text))
    rare_ratio = compression_ratio(rare_text.replace(" ", ""), list(rare_text.replace(" ", "")))

    assert common_ratio > rare_ratio, "Test 9 failed: common words should compress better"

    print("All tests passed!")
