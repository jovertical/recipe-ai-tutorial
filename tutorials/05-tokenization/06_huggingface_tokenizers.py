# Problem 6: Hugging Face Tokenizers
#
# Now that you understand tokenization from scratch, let's use production tokenizers.
# Compare how different pretrained tokenizers handle the same text.
#
# We'll explore:
# 1. GPT-2 tokenizer (BPE)
# 2. BERT tokenizer (WordPiece)
# 3. Llama tokenizer (SentencePiece BPE)
#
# Key observations:
# - Different tokenizers split text differently
# - Vocabulary size varies (GPT-2: 50k, BERT: 30k, Llama: 32k)
# - Some handle spaces/punctuation differently
#
# Prerequisites:
#   pip install transformers
#
# ML Relevance: Choosing the right tokenizer affects model performance.
# Understanding tokenizer behavior helps you debug issues like:
# - Why does my prompt get truncated?
# - Why does the model make spelling errors?
# - Why are some languages tokenized poorly?

from transformers import AutoTokenizer


def load_tokenizers() -> dict:
    """
    Load three different tokenizers from Hugging Face.

    Returns dict with keys: 'gpt2', 'bert', 'llama'

    Note: For Llama, we'll use a compatible open tokenizer if the official
    one requires authentication.
    """
    # Your solution here
    # Use AutoTokenizer.from_pretrained() for each
    # For llama, try "meta-llama/Llama-2-7b-hf" or fall back to another
    pass


def analyze_tokenization(text: str, tokenizer) -> dict:
    """
    Analyze how a tokenizer handles the given text.

    Returns:
    - 'tokens': list of token strings
    - 'token_ids': list of token IDs
    - 'num_tokens': number of tokens
    - 'vocab_size': size of tokenizer's vocabulary
    - 'has_special_tokens': whether special tokens were added
    """
    # Your solution here
    # Use tokenizer.encode() and tokenizer.convert_ids_to_tokens()
    pass


def calculate_fertility(text: str, tokenizer) -> float:
    """
    Calculate fertility: average number of tokens per word.

    Fertility = num_tokens / num_words

    Lower fertility = more efficient tokenization
    English typically has fertility ~1.3-1.5 for good tokenizers

    Example:
        text = "Hello world"
        tokens = ["Hello", "world"]
        fertility = 2 / 2 = 1.0
    """
    # Your solution here
    pass


def compare_tokenizers(text: str, tokenizers: dict) -> dict:
    """
    Compare how different tokenizers handle the same text.

    Args:
        text: Text to tokenize
        tokenizers: Dict of name -> tokenizer

    Returns dict with comparison:
    - 'text': original text
    - 'results': dict of tokenizer_name -> analysis dict
    - 'most_efficient': name of tokenizer with fewest tokens
    - 'least_efficient': name of tokenizer with most tokens
    """
    # Your solution here
    pass


def tokenize_recipe(recipe_text: str, tokenizer) -> dict:
    """
    Analyze tokenization of recipe-specific text.

    Look for:
    - How ingredient quantities are tokenized ("1/2 cup")
    - How cooking terms are split ("sauté", "350°F")
    - How ingredient names are handled

    Returns:
    - 'tokens': full token list
    - 'num_tokens': count
    - 'quantity_tokens': tokens that look like quantities
    - 'interesting_splits': unusual tokenizations worth noting
    """
    # Your solution here
    pass


def find_tokenization_differences(text: str, tokenizers: dict) -> list:
    """
    Find specific words/phrases that are tokenized differently.

    Returns list of dicts, each containing:
    - 'position': character position in text
    - 'substring': the differently-tokenized part
    - 'tokenizations': dict of tokenizer_name -> tokens for that part
    """
    # Your solution here
    pass


def special_token_analysis(tokenizer) -> dict:
    """
    Analyze special tokens in a tokenizer.

    Returns:
    - 'bos_token': beginning of sequence token (or None)
    - 'eos_token': end of sequence token (or None)
    - 'pad_token': padding token (or None)
    - 'unk_token': unknown token (or None)
    - 'additional_special_tokens': any extra special tokens
    """
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    # We'll use GPT-2 for testing since it's always available
    try:
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Could not load GPT-2 tokenizer: {e}")
        print("Please run: pip install transformers")
        exit(1)

    # Test 1: analyze_tokenization basic
    analysis = analyze_tokenization("Hello world", gpt2_tokenizer)
    assert "tokens" in analysis, "Test 1a failed: should have 'tokens' key"
    assert "num_tokens" in analysis, "Test 1b failed: should have 'num_tokens' key"
    assert analysis["num_tokens"] >= 2, "Test 1c failed: should have at least 2 tokens"

    # Test 2: calculate_fertility
    fertility = calculate_fertility("The quick brown fox", gpt2_tokenizer)
    assert 1.0 <= fertility <= 3.0, f"Test 2 failed: fertility {fertility} seems unreasonable"

    # Test 3: Recipe tokenization
    recipe = "Preheat oven to 350°F. Mix 1/2 cup flour with 2 eggs."
    result = tokenize_recipe(recipe, gpt2_tokenizer)
    assert "tokens" in result, "Test 3a failed: should have 'tokens'"
    assert result["num_tokens"] > 0, "Test 3b failed: should have tokens"

    # Test 4: Special token analysis
    special = special_token_analysis(gpt2_tokenizer)
    assert "eos_token" in special, "Test 4a failed: should analyze eos_token"
    assert "unk_token" in special, "Test 4b failed: should analyze unk_token"

    # Test 5: Compare tokenizers (just GPT-2 with itself for basic test)
    tokenizers = {"gpt2": gpt2_tokenizer}
    comparison = compare_tokenizers("Hello world", tokenizers)
    assert "results" in comparison, "Test 5a failed: should have 'results'"
    assert "gpt2" in comparison["results"], "Test 5b failed: should have gpt2 results"

    # Test 6: Fertility for different text types
    english_fertility = calculate_fertility("The cat sat on the mat", gpt2_tokenizer)
    code_fertility = calculate_fertility("def hello_world():", gpt2_tokenizer)
    # Code often has higher fertility due to underscores, etc.
    assert isinstance(english_fertility, float), "Test 6a failed: should return float"
    assert isinstance(code_fertility, float), "Test 6b failed: should return float"

    # Test 7: Empty text handling
    analysis = analyze_tokenization("", gpt2_tokenizer)
    assert analysis["num_tokens"] == 0 or analysis["num_tokens"] == 1, \
        "Test 7 failed: empty text should have 0 or 1 tokens"

    # Test 8: Unicode handling
    analysis = analyze_tokenization("Café résumé naïve", gpt2_tokenizer)
    assert analysis["num_tokens"] > 0, "Test 8 failed: should handle unicode"

    # Test 9: Long text
    long_text = "word " * 100
    analysis = analyze_tokenization(long_text, gpt2_tokenizer)
    assert analysis["num_tokens"] >= 100, "Test 9 failed: should tokenize long text"

    # Test 10: Ingredient tokenization patterns
    ingredients = "2 cups all-purpose flour, sifted"
    result = tokenize_recipe(ingredients, gpt2_tokenizer)
    assert result["num_tokens"] > 0, "Test 10 failed: should tokenize ingredients"

    print("All tests passed!")
    print("\nBonus: Try loading multiple tokenizers and comparing them!")
    print("tokenizers = load_tokenizers()")
    print("compare_tokenizers('Your recipe text here', tokenizers)")
