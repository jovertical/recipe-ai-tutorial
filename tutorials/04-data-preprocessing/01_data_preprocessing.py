import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 04: Data Preprocessing and Exploration

    In this tutorial, we'll clean and prepare our recipe data for model training.

    ## What We'll Cover

    1. Loading and inspecting raw data
    2. Text cleaning and normalization
    3. Ingredient parsing and standardization
    4. Creating training formats for LLMs
    5. Train/validation/test splits
    6. Saving processed datasets

    ## Why This Matters

    Raw data is messy. Before training, we need to:
    - Remove noise and inconsistencies
    - Standardize formats
    - Create the right structure for our models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Setup
    """)
    return


@app.cell
def _():
    import json
    import re
    from pathlib import Path
    from collections import Counter

    import pandas as pd
    import numpy as np

    # Paths
    DATA_DIR = Path("../../data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Libraries loaded!")
    print(f"Raw data: {RAW_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")
    return DATA_DIR, RAW_DIR, PROCESSED_DIR, Path, json, re, pd, np, Counter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Loading Raw Data

    Let's load our sample recipe data and see what we're working with.
    """)
    return


@app.cell
def _(RAW_DIR, json):
    # Load RecipeNLG-style data
    recipenlg_file = RAW_DIR / "recipenlg" / "sample_recipes.json"

    if recipenlg_file.exists():
        with open(recipenlg_file) as f:
            recipes_raw = json.load(f)
        print(f"Loaded {len(recipes_raw)} recipes")
    else:
        print("Sample data not found. Run Tutorial 03 first to create it.")
        recipes_raw = []

    # Show first recipe
    if recipes_raw:
        print("\nFirst recipe:")
        print(json.dumps(recipes_raw[0], indent=2))
    return recipes_raw, recipenlg_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Text Cleaning Functions

    Let's build reusable functions to clean our text data.
    """)
    return


@app.cell
def _(re):
    def clean_text(text):
        """Basic text cleaning."""
        if not text:
            return ""

        # Convert to string if needed
        text = str(text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


    def normalize_fractions(text):
        """Convert unicode fractions to text."""
        fraction_map = {
            '½': '1/2',
            '⅓': '1/3',
            '⅔': '2/3',
            '¼': '1/4',
            '¾': '3/4',
            '⅛': '1/8',
            '⅜': '3/8',
            '⅝': '5/8',
            '⅞': '7/8',
        }
        for unicode_frac, text_frac in fraction_map.items():
            text = text.replace(unicode_frac, text_frac)
        return text


    def normalize_units(text):
        """Standardize measurement units."""
        unit_map = {
            r'\btbsp\.?\b': 'tablespoon',
            r'\btbs\.?\b': 'tablespoon',
            r'\btsp\.?\b': 'teaspoon',
            r'\boz\.?\b': 'ounce',
            r'\blb\.?\b': 'pound',
            r'\blbs\.?\b': 'pounds',
            r'\bc\.?\b': 'cup',
            r'\bpt\.?\b': 'pint',
            r'\bqt\.?\b': 'quart',
            r'\bgal\.?\b': 'gallon',
        }
        for pattern, replacement in unit_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


    # Test the functions
    test_cases = [
        "2½ cups   flour",
        "1 tbsp. olive oil",
        "3 lbs chicken breast",
        "  extra   whitespace  here  "
    ]

    print("Text Cleaning Examples:")
    print("-" * 50)
    for test in test_cases:
        cleaned = clean_text(normalize_fractions(normalize_units(test)))
        print(f"  '{test}'")
        print(f"  → '{cleaned}'")
        print()
    return clean_text, normalize_fractions, normalize_units, test_cases


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Ingredient Parsing

    Ingredients often follow a pattern: **quantity + unit + ingredient + preparation**

    Example: "2 cups flour, sifted" → quantity=2, unit=cups, ingredient=flour, prep=sifted
    """)
    return


@app.cell
def _(re):
    def parse_ingredient(ingredient_text):
        """
        Parse an ingredient string into components.

        Returns dict with: quantity, unit, ingredient, preparation
        """
        text = ingredient_text.strip()

        # Common patterns
        # Pattern: "2 cups flour" or "1/2 tsp salt" or "3 large eggs"
        pattern = r'^([\d\s/.-]+)?\s*([a-zA-Z]+)?\s+(.+)$'

        # Try to extract quantity
        quantity_pattern = r'^([\d]+(?:[/.\s-][\d]+)?)\s*'
        quantity_match = re.match(quantity_pattern, text)

        quantity = None
        if quantity_match:
            quantity = quantity_match.group(1).strip()
            text = text[quantity_match.end():].strip()

        # Common units
        units = [
            'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp',
            'teaspoon', 'teaspoons', 'tsp', 'ounce', 'ounces', 'oz',
            'pound', 'pounds', 'lb', 'lbs', 'gram', 'grams', 'g',
            'kilogram', 'kg', 'ml', 'milliliter', 'liter', 'l',
            'pinch', 'dash', 'bunch', 'clove', 'cloves',
            'slice', 'slices', 'piece', 'pieces',
            'can', 'cans', 'package', 'packages', 'bag', 'bags',
            'small', 'medium', 'large'
        ]

        unit = None
        for u in units:
            if text.lower().startswith(u + ' ') or text.lower().startswith(u + 's '):
                unit = u
                text = text[len(u):].strip()
                if text.startswith('s '):
                    text = text[2:]
                break

        # Check for preparation instructions (after comma)
        preparation = None
        if ',' in text:
            parts = text.split(',', 1)
            text = parts[0].strip()
            preparation = parts[1].strip()

        return {
            'original': ingredient_text,
            'quantity': quantity,
            'unit': unit,
            'ingredient': text,
            'preparation': preparation
        }


    # Test ingredient parsing
    test_ingredients = [
        "2 cups all-purpose flour",
        "1/2 teaspoon salt",
        "3 large eggs",
        "1 pound chicken breast, diced",
        "Fresh parsley, chopped",
        "Salt and pepper to taste"
    ]

    print("Ingredient Parsing Examples:")
    print("-" * 60)
    for ing in test_ingredients:
        parsed = parse_ingredient(ing)
        print(f"Input: {ing}")
        print(f"  Quantity: {parsed['quantity']}")
        print(f"  Unit: {parsed['unit']}")
        print(f"  Ingredient: {parsed['ingredient']}")
        print(f"  Preparation: {parsed['preparation']}")
        print()
    return parse_ingredient, test_ingredients


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Creating Training Formats

    For fine-tuning LLMs, we need to format our data as text prompts. Common formats:

    ### Format 1: Instruction-Response
    ```
    ### Instruction: Generate a recipe for chocolate chip cookies
    ### Response: [recipe text]
    ```

    ### Format 2: Chat Format
    ```
    User: How do I make chocolate chip cookies?
    Assistant: Here's a recipe for chocolate chip cookies...
    ```

    ### Format 3: Structured Recipe
    ```
    Recipe: Chocolate Chip Cookies
    Ingredients:
    - 2 cups flour
    - 1 cup sugar
    ...
    Instructions:
    1. Preheat oven...
    ```
    """)
    return


@app.cell
def _():
    def format_recipe_instruction(recipe):
        """Format recipe as instruction-response pair."""
        # Create the instruction
        instruction = f"Generate a recipe for {recipe['title']}"

        # Create the response
        ingredients_text = "\n".join(f"- {ing}" for ing in recipe['ingredients'])
        directions_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe['directions']))

        response = f"""Recipe: {recipe['title']}

Ingredients:
{ingredients_text}

Instructions:
{directions_text}"""

        return {
            "instruction": instruction,
            "response": response,
            "full_text": f"### Instruction: {instruction}\n\n### Response: {response}"
        }


    def format_recipe_chat(recipe):
        """Format recipe as chat conversation."""
        user_msg = f"How do I make {recipe['title'].lower()}?"

        ingredients_text = ", ".join(recipe['ingredients'][:5])
        if len(recipe['ingredients']) > 5:
            ingredients_text += f", and {len(recipe['ingredients']) - 5} more ingredients"

        directions_summary = recipe['directions'][0] if recipe['directions'] else "Follow the recipe steps."

        assistant_msg = f"""Here's how to make {recipe['title']}!

You'll need: {ingredients_text}.

Start by: {directions_summary}

Would you like the complete recipe with all ingredients and steps?"""

        return {
            "user": user_msg,
            "assistant": assistant_msg,
            "conversation": f"User: {user_msg}\n\nAssistant: {assistant_msg}"
        }


    def format_recipe_structured(recipe):
        """Format recipe in a structured format."""
        ingredients_text = "\n".join(f"- {ing}" for ing in recipe['ingredients'])
        directions_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe['directions']))

        text = f"""<recipe>
<title>{recipe['title']}</title>
<ingredients>
{ingredients_text}
</ingredients>
<instructions>
{directions_text}
</instructions>
</recipe>"""

        return {"structured_text": text}

    return format_recipe_instruction, format_recipe_chat, format_recipe_structured


@app.cell
def _(recipes_raw, format_recipe_instruction, format_recipe_chat, format_recipe_structured):
    # Demo the formats with first recipe
    if recipes_raw:
        sample = recipes_raw[0]

        print("=" * 60)
        print("FORMAT 1: Instruction-Response")
        print("=" * 60)
        formatted = format_recipe_instruction(sample)
        print(formatted['full_text'][:500] + "...")

        print("\n" + "=" * 60)
        print("FORMAT 2: Chat")
        print("=" * 60)
        chat = format_recipe_chat(sample)
        print(chat['conversation'])

        print("\n" + "=" * 60)
        print("FORMAT 3: Structured")
        print("=" * 60)
        structured = format_recipe_structured(sample)
        print(structured['structured_text'][:500] + "...")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Processing the Full Dataset

    Now let's process all recipes and create our training dataset.
    """)
    return


@app.cell
def _(recipes_raw, clean_text, normalize_fractions, format_recipe_instruction):
    def process_recipe(recipe):
        """Process a single recipe."""
        # Clean text fields
        cleaned = {
            'title': clean_text(recipe['title']),
            'ingredients': [
                clean_text(normalize_fractions(ing))
                for ing in recipe['ingredients']
            ],
            'directions': [
                clean_text(step)
                for step in recipe['directions']
            ],
            'ner': recipe.get('NER', [])
        }

        # Add formatted versions
        formatted = format_recipe_instruction(cleaned)
        cleaned['instruction'] = formatted['instruction']
        cleaned['response'] = formatted['response']
        cleaned['full_text'] = formatted['full_text']

        return cleaned


    # Process all recipes
    processed_recipes = [process_recipe(r) for r in recipes_raw]

    print(f"Processed {len(processed_recipes)} recipes")
    if processed_recipes:
        print("\nSample processed recipe:")
        print(f"  Title: {processed_recipes[0]['title']}")
        print(f"  Ingredients: {len(processed_recipes[0]['ingredients'])}")
        print(f"  Steps: {len(processed_recipes[0]['directions'])}")
    return process_recipe, processed_recipes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. Train/Validation/Test Split

    For model training, we need to split our data:
    - **Training set** (80%): Used to train the model
    - **Validation set** (10%): Used to tune hyperparameters
    - **Test set** (10%): Final evaluation, never seen during training
    """)
    return


@app.cell
def _(np, processed_recipes):
    def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
        """Split dataset into train/val/test."""
        np.random.seed(seed)

        # Shuffle indices
        n = len(data)
        indices = np.random.permutation(n)

        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Split
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        return train_data, val_data, test_data


    # Split the data
    train_recipes, val_recipes, test_recipes = split_dataset(processed_recipes)

    print("Dataset Split:")
    print(f"  Training:   {len(train_recipes)} recipes ({len(train_recipes)/len(processed_recipes)*100:.1f}%)")
    print(f"  Validation: {len(val_recipes)} recipes ({len(val_recipes)/len(processed_recipes)*100:.1f}%)")
    print(f"  Test:       {len(test_recipes)} recipes ({len(test_recipes)/len(processed_recipes)*100:.1f}%)")
    return split_dataset, train_recipes, val_recipes, test_recipes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. Saving Processed Data

    Let's save our processed data in formats ready for training.
    """)
    return


@app.cell
def _(PROCESSED_DIR, json, train_recipes, val_recipes, test_recipes):
    # Save as JSON
    def save_jsonl(data, filepath):
        """Save as JSON Lines format (one JSON object per line)."""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


    # Save splits
    save_jsonl(train_recipes, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val_recipes, PROCESSED_DIR / "val.jsonl")
    save_jsonl(test_recipes, PROCESSED_DIR / "test.jsonl")

    print("Saved processed datasets:")
    print(f"  {PROCESSED_DIR / 'train.jsonl'}")
    print(f"  {PROCESSED_DIR / 'val.jsonl'}")
    print(f"  {PROCESSED_DIR / 'test.jsonl'}")

    # Also save the full text versions for easy LLM training
    def save_text_file(data, filepath):
        """Save just the full_text field for training."""
        with open(filepath, 'w') as f:
            for item in data:
                if 'full_text' in item:
                    f.write(item['full_text'] + '\n\n---\n\n')


    save_text_file(train_recipes, PROCESSED_DIR / "train_text.txt")

    print(f"\nAlso saved: {PROCESSED_DIR / 'train_text.txt'}")
    return save_jsonl, save_text_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9. Data Statistics and Exploration

    Let's compute some statistics about our processed dataset.
    """)
    return


@app.cell
def _(processed_recipes, Counter, np):
    # Compute statistics
    print("Dataset Statistics")
    print("=" * 50)

    # Basic counts
    total_recipes = len(processed_recipes)
    total_ingredients = sum(len(r['ingredients']) for r in processed_recipes)
    total_steps = sum(len(r['directions']) for r in processed_recipes)

    print(f"\nTotal recipes: {total_recipes}")
    print(f"Total ingredients: {total_ingredients}")
    print(f"Total steps: {total_steps}")
    print(f"Avg ingredients per recipe: {total_ingredients/total_recipes:.1f}")
    print(f"Avg steps per recipe: {total_steps/total_recipes:.1f}")

    # Text length statistics
    text_lengths = [len(r['full_text']) for r in processed_recipes]
    print(f"\nText length (characters):")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    print(f"  Mean: {np.mean(text_lengths):.0f}")
    print(f"  Median: {np.median(text_lengths):.0f}")

    # Most common ingredients (from NER)
    all_ner = []
    for r in processed_recipes:
        all_ner.extend(r['ner'])

    ner_counts = Counter(all_ner)
    print(f"\nMost common ingredients:")
    for ing, count in ner_counts.most_common(10):
        print(f"  {ing}: {count}")
    return total_recipes, total_ingredients, total_steps, text_lengths, all_ner, ner_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    In this tutorial, we:

    1. **Loaded raw recipe data** from our sample dataset
    2. **Built text cleaning functions** for normalization
    3. **Created ingredient parsers** to extract structured info
    4. **Formatted data for LLM training** (instruction, chat, structured)
    5. **Split data** into train/val/test sets
    6. **Saved processed data** in JSONL format

    ## Files Created

    ```
    data/processed/
    ├── train.jsonl      # Training data
    ├── val.jsonl        # Validation data
    ├── test.jsonl       # Test data
    └── train_text.txt   # Plain text for training
    ```

    ## What's Next?

    In **Part 2: Training Custom Models**, we'll:
    - Deep dive into tokenization
    - Fine-tune models with LoRA/QLoRA
    - Train custom ingredient embeddings

    ---

    ## Exercises

    1. Add more cleaning functions (handle special characters, fix encoding issues)
    2. Create additional training formats (alpaca format, sharegpt format)
    3. Implement deduplication based on ingredient similarity
    4. Calculate token counts for each recipe (for context length planning)
    """)
    return


@app.cell
def _():
    # Exercise space
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
