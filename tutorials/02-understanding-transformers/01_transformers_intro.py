import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 02: Understanding Transformers and LLMs

    In this tutorial, we'll explore:
    1. What are Transformers?
    2. The Attention mechanism
    3. How LLMs generate text
    4. Hands-on: Loading and using models with Hugging Face

    ## Why This Matters for Recipe AI

    Our Recipe AI will use transformer-based models for:
    - **Recipe generation**: Given ingredients, generate a recipe
    - **Ingredient embeddings**: Understanding semantic similarity between ingredients
    - **Substitution suggestions**: "What can I use instead of butter?"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. What are Transformers?

    Transformers are a neural network architecture introduced in the 2017 paper **"Attention Is All You Need"**.

    Before Transformers, we had:
    - **RNNs/LSTMs**: Process text sequentially (slow, forget long-range dependencies)

    Transformers solved this by:
    - Processing all tokens **in parallel**
    - Using **attention** to relate any word to any other word directly

    ### Key Components

    ```
    Input: "The cat sat on the mat"
             ↓
       [Tokenization]
             ↓
       [Embeddings] → Convert tokens to vectors
             ↓
       [Positional Encoding] → Add position information
             ↓
       [Transformer Blocks] × N
          - Self-Attention
          - Feed-Forward Network
             ↓
       [Output]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. The Attention Mechanism

    Attention answers: **"When processing this word, which other words should I focus on?"**

    ### Example

    Sentence: *"The **cat** sat on the mat because **it** was tired."*

    When processing "it", attention helps the model understand that "it" refers to "cat".

    ### Self-Attention Formula

    ```
    Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
    ```

    Where:
    - **Q (Query)**: What am I looking for?
    - **K (Key)**: What do I contain?
    - **V (Value)**: What information do I provide?

    Don't worry if this is confusing - let's see it in action!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Setup

    First, let's install and import what we need.
    """)
    return


@app.cell
def _():
    # Install dependencies (run once)
    # # (use marimo's built-in package management features instead) !pip install transformers torch
    return


@app.cell
def _():
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    import numpy as np

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return AutoModel, AutoModelForCausalLM, AutoTokenizer, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Tokenization: Text → Numbers

    Models don't understand text - they work with numbers. Tokenization converts text into token IDs.

    Remember the `SimpleTokenizer` we built? Real tokenizers are more sophisticated but the concept is the same.
    """)
    return


@app.cell
def _(AutoTokenizer):
    # Load a real tokenizer (GPT-2)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    _text = 'Preheat the oven to 350 degrees.'
    tokens = tokenizer.tokenize(_text)
    token_ids = tokenizer.encode(_text)
    # Tokenize
    print('Original text:', _text)
    print('Tokens:', tokens)
    print('Token IDs:', token_ids)
    print(f'\nVocab size: {tokenizer.vocab_size:,}')
    return (tokenizer,)


@app.cell
def _(tokenizer):
    # Let's see how different words get tokenized
    examples = ['butter', 'margarine', 'butterscotch', 'The quick brown fox', 'ingredients: flour, sugar, eggs']
    for example in examples:
        tokens_1 = tokenizer.tokenize(example)
        print(f'{example!r:40} → {tokens_1}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observations

    - Common words are single tokens ("butter")
    - Rare/long words get split into subwords ("butterscotch" → "butter" + "scotch")
    - Spaces are often part of tokens (notice the "Ġ" prefix means "preceded by space")

    This is called **Byte-Pair Encoding (BPE)** - a balance between word-level and character-level tokenization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Embeddings: Numbers → Vectors

    Token IDs are just indices. We need to convert them to dense vectors (embeddings) that capture meaning.

    This is exactly like the `EmbeddingTable` we built, but with pre-trained weights!
    """)
    return


@app.cell
def _(AutoModel):
    # Load GPT-2 model
    model = AutoModel.from_pretrained("gpt2")

    # Get the embedding layer
    embedding_layer = model.wte  # wte = word token embeddings

    print(f"Embedding shape: {embedding_layer.weight.shape}")
    print(f"  → {embedding_layer.weight.shape[0]:,} tokens")
    print(f"  → {embedding_layer.weight.shape[1]} dimensions per token")
    return embedding_layer, model


@app.cell
def _(embedding_layer, tokenizer):
    # Get embeddings for some food-related words
    words = ['butter', 'margarine', 'oil', 'flour', 'computer']
    embeddings = {}
    for _word in words:
        token_id = tokenizer.encode(_word)[0]
        embedding = embedding_layer.weight[token_id].detach().numpy()  # Get first token ID
        embeddings[_word] = embedding
        print(f'{_word:12} → token_id={token_id:5}, embedding shape={embedding.shape}')
    return embeddings, words


@app.cell
def _(embeddings, np, words):
    # Let's compute cosine similarity between these words!
    # Using the cosine_similarity function concept we learned
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print('Cosine Similarities:')
    print('-' * 40)
    for _word in words:
        if _word != 'butter':
            sim = cosine_similarity(embeddings['butter'], embeddings[_word])
    # Compare butter to everything
            print(f'butter ↔ {_word:12}: {sim:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What do you notice?

    - Similar ingredients (butter, margarine, oil) should have higher similarity
    - Unrelated words (computer) should have lower similarity

    This is the foundation of how our Recipe AI will find ingredient substitutions!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Text Generation: How LLMs Work

    LLMs generate text **one token at a time**:

    ```
    Input: "To make pancakes, first"
                                  ↓ predict next token
                                "mix"
                                  ↓ add to input, predict again
           "To make pancakes, first mix"
                                       ↓
                                     "the"
           ... and so on
    ```

    This is called **autoregressive generation**.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, tokenizer, torch):
    # Load GPT-2 for text generation
    generator_model = AutoModelForCausalLM.from_pretrained('gpt2')
    generator_model.eval()  # Set to evaluation mode

    def generate_text(prompt, max_new_tokens=50):
        """Generate text continuation from a prompt."""
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = generator_model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generate_text, generator_model


@app.cell
def _(generate_text):
    # Try some recipe-related prompts!
    prompts = ['To make chocolate chip cookies, you will need:', 'The secret to a perfect omelette is', 'Ingredients for pasta carbonara:']
    for _prompt in prompts:
        print(f'Prompt: {_prompt}')
        print(f'Generated: {generate_text(_prompt)}')
        print('-' * 60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observations

    GPT-2 is a general model, not fine-tuned for recipes. The outputs might be:
    - Sometimes relevant
    - Sometimes nonsensical
    - Not formatted like real recipes

    **This is exactly why we'll fine-tune our own model in later tutorials!**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. Understanding Generation Parameters

    Let's explore how different parameters affect generation.
    """)
    return


@app.cell
def _(generator_model, tokenizer, torch):
    def generate_with_params(prompt, temperature=1.0, top_p=1.0, max_tokens=30):
        """Generate with specific parameters."""
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = generator_model.generate(inputs.input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    _prompt = 'The best way to cook an egg is'
    print('Temperature comparison:')
    print('=' * 60)
    for temp in [0.3, 0.7, 1.0, 1.5]:
        result = generate_with_params(_prompt, temperature=temp)
        print(f'\nTemp={temp}: {result}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temperature

    - **Low (0.1-0.5)**: More deterministic, repetitive, "safe" outputs
    - **Medium (0.7-0.9)**: Balanced creativity and coherence
    - **High (1.0+)**: More random, creative, sometimes nonsensical

    For recipes, we typically want **medium temperature** for creativity with accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. Visualizing Attention (Bonus)

    Let's see what the model "pays attention to" when processing text.
    """)
    return


@app.cell
def _(model, tokenizer, torch):
    _text = 'The chef added salt to the soup'
    inputs = tokenizer(_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1][0, 0].numpy()
    tokens_2 = tokenizer.tokenize(_text)
    print(f'Attention shape: {attention.shape}')
    print(f'Tokens: {tokens_2}')
    return attention, tokens_2


@app.cell
def _(attention, tokens_2):
    # Simple text visualization of attention
    # (For proper visualization, use matplotlib or BertViz library)
    print("\nAttention from 'soup' to other tokens:")
    print('-' * 40)
    soup_idx = tokens_2.index('Ġsoup') if 'Ġsoup' in tokens_2 else -1
    if soup_idx != -1:
        for (i, (token, weight)) in enumerate(zip(tokens_2, attention[soup_idx])):
            bar = '█' * int(weight * 20)
            print(f'{token:12} {weight:.3f} {bar}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    In this tutorial, we learned:

    1. **Transformers** process all tokens in parallel using attention
    2. **Tokenization** converts text to numbers (like our SimpleTokenizer)
    3. **Embeddings** convert token IDs to dense vectors (like our EmbeddingTable)
    4. **LLMs generate text** one token at a time (autoregressive)
    5. **Temperature** controls randomness in generation
    6. **Attention** helps the model focus on relevant parts of the input

    ## What's Next?

    In **Tutorial 03**, we'll collect recipe datasets that we'll use to fine-tune our own model!

    ---

    ## Exercises

    1. Try different prompts and temperatures - what works best for recipe generation?
    2. Compare similarity between different ingredient pairs
    3. Can you find ingredients that are similar according to embeddings but not in real cooking?
    """)
    return


@app.cell
def _():
    # Exercise space - try your own experiments!
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
