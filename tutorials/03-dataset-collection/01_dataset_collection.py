import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 03: Dataset Collection for Recipe AI

    In this tutorial, we'll collect and explore recipe datasets that we'll use to fine-tune our models.

    ## What We'll Cover

    1. Overview of available recipe datasets
    2. Downloading RecipeNLG dataset
    3. Exploring the Food.com dataset from Kaggle
    4. Understanding data formats and structure
    5. Initial data quality assessment

    ## Why This Matters

    The quality of your training data directly determines the quality of your model. We need:
    - **Diverse recipes**: Different cuisines, cooking methods, skill levels
    - **Structured data**: Ingredients, instructions, metadata
    - **Clean data**: Minimal errors, consistent formatting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Available Recipe Datasets

    There are several high-quality recipe datasets available:

    | Dataset | Size | Features | Source |
    |---------|------|----------|--------|
    | **RecipeNLG** | 2.2M recipes | Title, ingredients, instructions | Academic (from multiple sites) |
    | **Food.com** | 230K recipes | Full metadata, reviews, nutrition | Kaggle |
    | **Recipe1M+** | 1M recipes | Images + text | Academic |
    | **Epicurious** | 20K recipes | Ratings, nutrition | Kaggle |

    For our Recipe AI, we'll primarily use:
    - **RecipeNLG** for large-scale training
    - **Food.com** for rich metadata and user interactions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Setup

    First, let's set up our environment and create the data directory.
    """)
    return


@app.cell
def _():
    import os
    import json
    from pathlib import Path

    # Create data directories
    DATA_DIR = Path("../../data")
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"

    for dir_path in [RAW_DIR, PROCESSED_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Raw data: {RAW_DIR.absolute()}")
    print(f"Processed data: {PROCESSED_DIR.absolute()}")
    return DATA_DIR, RAW_DIR, PROCESSED_DIR, Path, os, json


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from collections import Counter

    print("Libraries loaded successfully!")
    return pd, np, Counter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. RecipeNLG Dataset

    RecipeNLG is a large-scale dataset of 2.2 million recipes, perfect for training language models.

    ### Download Instructions

    The dataset is available at: https://recipenlg.cs.put.poznan.pl/

    1. Visit the website and accept the terms
    2. Download `full_dataset.csv` (approximately 2GB)
    3. Place it in `data/raw/recipenlg/`

    For this tutorial, we'll create a sample dataset to demonstrate the structure.
    """)
    return


@app.cell
def _(RAW_DIR, Path, json):
    # Create sample RecipeNLG-style data for demonstration
    recipenlg_dir = RAW_DIR / "recipenlg"
    recipenlg_dir.mkdir(exist_ok=True)

    # Sample recipes in RecipeNLG format
    sample_recipes = [
        {
            "title": "Classic Chocolate Chip Cookies",
            "ingredients": [
                "2 1/4 cups all-purpose flour",
                "1 tsp baking soda",
                "1 tsp salt",
                "1 cup butter, softened",
                "3/4 cup granulated sugar",
                "3/4 cup packed brown sugar",
                "2 large eggs",
                "2 tsp vanilla extract",
                "2 cups chocolate chips"
            ],
            "directions": [
                "Preheat oven to 375°F.",
                "Combine flour, baking soda and salt in small bowl.",
                "Beat butter, granulated sugar, brown sugar and vanilla extract in large mixer bowl until creamy.",
                "Add eggs, one at a time, beating well after each addition.",
                "Gradually beat in flour mixture.",
                "Stir in chocolate chips.",
                "Drop rounded tablespoon of dough onto ungreased baking sheets.",
                "Bake for 9 to 11 minutes or until golden brown.",
                "Cool on baking sheets for 2 minutes.",
                "Remove to wire racks to cool completely."
            ],
            "NER": ["flour", "baking soda", "salt", "butter", "sugar", "brown sugar", "eggs", "vanilla", "chocolate chips"]
        },
        {
            "title": "Simple Tomato Pasta",
            "ingredients": [
                "1 lb spaghetti",
                "2 tbsp olive oil",
                "4 cloves garlic, minced",
                "1 can (28 oz) crushed tomatoes",
                "1 tsp dried basil",
                "1 tsp dried oregano",
                "Salt and pepper to taste",
                "1/4 cup fresh basil, chopped",
                "Parmesan cheese for serving"
            ],
            "directions": [
                "Cook pasta according to package directions. Drain and set aside.",
                "Heat olive oil in a large skillet over medium heat.",
                "Add garlic and sauté for 1 minute until fragrant.",
                "Add crushed tomatoes, dried basil, and oregano.",
                "Simmer for 15 minutes, stirring occasionally.",
                "Season with salt and pepper.",
                "Toss pasta with sauce.",
                "Top with fresh basil and Parmesan cheese."
            ],
            "NER": ["spaghetti", "olive oil", "garlic", "tomatoes", "basil", "oregano", "salt", "pepper", "parmesan"]
        },
        {
            "title": "Fluffy Pancakes",
            "ingredients": [
                "1 1/2 cups all-purpose flour",
                "3 1/2 tsp baking powder",
                "1 tbsp sugar",
                "1/4 tsp salt",
                "1 1/4 cups milk",
                "1 egg",
                "3 tbsp melted butter"
            ],
            "directions": [
                "In a large bowl, sift together flour, baking powder, sugar, and salt.",
                "Make a well in the center and pour in milk, egg, and melted butter.",
                "Mix until smooth.",
                "Heat a lightly oiled griddle over medium-high heat.",
                "Pour batter onto griddle, using approximately 1/4 cup for each pancake.",
                "Brown on both sides and serve hot."
            ],
            "NER": ["flour", "baking powder", "sugar", "salt", "milk", "egg", "butter"]
        },
        {
            "title": "Garlic Butter Shrimp",
            "ingredients": [
                "1 lb large shrimp, peeled and deveined",
                "4 tbsp butter",
                "6 cloves garlic, minced",
                "1/4 cup white wine",
                "2 tbsp lemon juice",
                "1/4 tsp red pepper flakes",
                "Salt and pepper to taste",
                "2 tbsp fresh parsley, chopped"
            ],
            "directions": [
                "Pat shrimp dry with paper towels. Season with salt and pepper.",
                "Melt butter in a large skillet over medium-high heat.",
                "Add shrimp in a single layer and cook 1-2 minutes per side until pink.",
                "Remove shrimp and set aside.",
                "Add garlic to the skillet and cook for 30 seconds.",
                "Pour in wine and lemon juice, scraping up any browned bits.",
                "Add red pepper flakes and return shrimp to pan.",
                "Toss to coat and garnish with parsley."
            ],
            "NER": ["shrimp", "butter", "garlic", "white wine", "lemon juice", "red pepper flakes", "salt", "pepper", "parsley"]
        },
        {
            "title": "Classic Caesar Salad",
            "ingredients": [
                "1 large head romaine lettuce",
                "1/2 cup Caesar dressing",
                "1/2 cup croutons",
                "1/4 cup Parmesan cheese, shaved",
                "Freshly ground black pepper"
            ],
            "directions": [
                "Wash and dry romaine lettuce. Tear into bite-sized pieces.",
                "Place lettuce in a large bowl.",
                "Add Caesar dressing and toss to coat evenly.",
                "Top with croutons and Parmesan shavings.",
                "Season with black pepper and serve immediately."
            ],
            "NER": ["romaine lettuce", "caesar dressing", "croutons", "parmesan", "black pepper"]
        }
    ]

    # Save sample data
    sample_file = recipenlg_dir / "sample_recipes.json"
    with open(sample_file, "w") as f:
        json.dump(sample_recipes, f, indent=2)

    print(f"Created sample dataset with {len(sample_recipes)} recipes")
    print(f"Saved to: {sample_file}")
    return sample_recipes, recipenlg_dir, sample_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Exploring Recipe Data Structure

    Let's analyze the structure of our recipe data.
    """)
    return


@app.cell
def _(pd, sample_recipes):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(sample_recipes)

    # Display basic info
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample recipe:")
    df.head(1)
    return (df,)


@app.cell
def _(df):
    # Analyze recipe components
    print("Recipe Analysis")
    print("=" * 50)

    # Ingredient counts
    ingredient_counts = df['ingredients'].apply(len)
    print(f"\nIngredients per recipe:")
    print(f"  Min: {ingredient_counts.min()}")
    print(f"  Max: {ingredient_counts.max()}")
    print(f"  Mean: {ingredient_counts.mean():.1f}")

    # Direction counts
    direction_counts = df['directions'].apply(len)
    print(f"\nSteps per recipe:")
    print(f"  Min: {direction_counts.min()}")
    print(f"  Max: {direction_counts.max()}")
    print(f"  Mean: {direction_counts.mean():.1f}")

    # Title lengths
    title_lengths = df['title'].apply(len)
    print(f"\nTitle length (characters):")
    print(f"  Min: {title_lengths.min()}")
    print(f"  Max: {title_lengths.max()}")
    print(f"  Mean: {title_lengths.mean():.1f}")
    return ingredient_counts, direction_counts, title_lengths


@app.cell
def _(Counter, sample_recipes):
    # Analyze ingredient frequency
    all_ingredients = []
    for recipe in sample_recipes:
        all_ingredients.extend(recipe['NER'])

    ingredient_freq = Counter(all_ingredients)
    print("Most Common Ingredients:")
    print("-" * 30)
    for ingredient, count in ingredient_freq.most_common(10):
        print(f"  {ingredient}: {count}")
    return all_ingredients, ingredient_freq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Food.com Dataset

    The Food.com dataset from Kaggle provides rich metadata including:
    - User reviews and ratings
    - Nutritional information
    - Cooking time and difficulty
    - Tags and categories

    ### Download Instructions

    1. Create a Kaggle account at https://kaggle.com
    2. Go to https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
    3. Download the dataset
    4. Extract to `data/raw/foodcom/`

    The dataset includes:
    - `RAW_recipes.csv` - Recipe details
    - `RAW_interactions.csv` - User reviews and ratings
    """)
    return


@app.cell
def _(RAW_DIR, pd):
    # Create sample Food.com-style data
    foodcom_dir = RAW_DIR / "foodcom"
    foodcom_dir.mkdir(exist_ok=True)

    # Sample Food.com recipe format
    foodcom_recipes = pd.DataFrame([
        {
            "id": 1,
            "name": "Classic Chocolate Chip Cookies",
            "minutes": 30,
            "contributor_id": 1001,
            "submitted": "2020-01-15",
            "tags": "dessert, cookies, baking, chocolate",
            "nutrition": "[250.0, 12.0, 8.0, 30.0, 3.0, 150.0, 2.0]",  # [cal, fat, sat_fat, sugar, protein, sodium, carbs]
            "n_steps": 10,
            "steps": "Preheat oven to 375°F. | Combine dry ingredients. | Cream butter and sugars. | Add eggs and vanilla. | Mix in dry ingredients. | Fold in chocolate chips. | Drop dough on baking sheet. | Bake 9-11 minutes. | Cool on pan. | Transfer to rack.",
            "description": "The classic cookie everyone loves",
            "ingredients": "flour, baking soda, salt, butter, sugar, brown sugar, eggs, vanilla, chocolate chips",
            "n_ingredients": 9
        },
        {
            "id": 2,
            "name": "Quick Tomato Pasta",
            "minutes": 25,
            "contributor_id": 1002,
            "submitted": "2020-02-20",
            "tags": "dinner, pasta, italian, quick",
            "nutrition": "[380.0, 10.0, 2.0, 8.0, 12.0, 450.0, 58.0]",
            "n_steps": 8,
            "steps": "Cook pasta. | Heat oil. | Sauté garlic. | Add tomatoes and herbs. | Simmer 15 minutes. | Season to taste. | Toss with pasta. | Serve with cheese.",
            "description": "A simple weeknight dinner",
            "ingredients": "spaghetti, olive oil, garlic, crushed tomatoes, basil, oregano, salt, pepper, parmesan",
            "n_ingredients": 9
        },
        {
            "id": 3,
            "name": "Fluffy Buttermilk Pancakes",
            "minutes": 20,
            "contributor_id": 1003,
            "submitted": "2020-03-10",
            "tags": "breakfast, pancakes, brunch",
            "nutrition": "[320.0, 8.0, 3.0, 12.0, 8.0, 380.0, 52.0]",
            "n_steps": 6,
            "steps": "Mix dry ingredients. | Combine wet ingredients. | Fold together until just combined. | Heat griddle. | Cook pancakes until bubbly. | Flip and finish cooking.",
            "description": "Light and fluffy breakfast pancakes",
            "ingredients": "flour, baking powder, sugar, salt, buttermilk, egg, butter",
            "n_ingredients": 7
        }
    ])

    # Save sample data
    foodcom_recipes.to_csv(foodcom_dir / "sample_recipes.csv", index=False)
    print(f"Created sample Food.com dataset")
    print(f"\nColumns: {foodcom_recipes.columns.tolist()}")
    foodcom_recipes
    return foodcom_dir, foodcom_recipes


@app.cell
def _(pd, RAW_DIR):
    # Create sample interactions data
    interactions = pd.DataFrame([
        {"user_id": 2001, "recipe_id": 1, "date": "2020-02-01", "rating": 5, "review": "Best cookies ever! My family loved them."},
        {"user_id": 2002, "recipe_id": 1, "date": "2020-02-15", "rating": 4, "review": "Great recipe, slightly too sweet for my taste."},
        {"user_id": 2003, "recipe_id": 1, "date": "2020-03-01", "rating": 5, "review": "Perfect! I added walnuts."},
        {"user_id": 2001, "recipe_id": 2, "date": "2020-02-10", "rating": 4, "review": "Quick and easy weeknight dinner."},
        {"user_id": 2004, "recipe_id": 2, "date": "2020-02-20", "rating": 5, "review": "Simple but delicious!"},
        {"user_id": 2002, "recipe_id": 3, "date": "2020-03-15", "rating": 5, "review": "Fluffiest pancakes I've ever made!"},
    ])

    interactions.to_csv(RAW_DIR / "foodcom" / "sample_interactions.csv", index=False)
    print("Sample interactions data:")
    interactions
    return (interactions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. Data Quality Checklist

    Before using any dataset for training, we should check for:

    ### Content Quality
    - [ ] Missing values in critical fields
    - [ ] Duplicate recipes
    - [ ] Empty ingredients or instructions
    - [ ] Corrupted or malformed text

    ### Format Consistency
    - [ ] Consistent ingredient formatting
    - [ ] Reasonable step counts
    - [ ] Valid cooking times
    - [ ] Proper encoding (UTF-8)

    ### Data Integrity
    - [ ] Matching IDs across tables
    - [ ] Valid date formats
    - [ ] Reasonable numerical ranges

    Let's implement some of these checks.
    """)
    return


@app.cell
def _(df):
    # Data quality checks
    print("Data Quality Report")
    print("=" * 50)

    # Check for missing values
    print("\n1. Missing Values:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        print(f"   {col}: {null_count}")

    # Check for empty lists
    print("\n2. Empty Lists:")
    empty_ingredients = (df['ingredients'].apply(len) == 0).sum()
    empty_directions = (df['directions'].apply(len) == 0).sum()
    print(f"   Empty ingredients: {empty_ingredients}")
    print(f"   Empty directions: {empty_directions}")

    # Check for duplicates
    print("\n3. Duplicates:")
    duplicate_titles = df['title'].duplicated().sum()
    print(f"   Duplicate titles: {duplicate_titles}")

    # Check title quality
    print("\n4. Title Quality:")
    short_titles = (df['title'].str.len() < 5).sum()
    long_titles = (df['title'].str.len() > 100).sum()
    print(f"   Titles < 5 chars: {short_titles}")
    print(f"   Titles > 100 chars: {long_titles}")

    print("\n" + "=" * 50)
    print("Quality check complete!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. Creating a Download Script

    For the full datasets, we'll create helper scripts.
    """)
    return


@app.cell
def _(DATA_DIR):
    # Create a download helper script
    download_script = '''#!/usr/bin/env python3
"""
Dataset Download Helper for Recipe AI

This script helps download the required datasets.
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download_recipenlg():
    """Instructions for RecipeNLG dataset."""
    print("=" * 60)
    print("RecipeNLG Dataset")
    print("=" * 60)
    print("""
    1. Visit: https://recipenlg.cs.put.poznan.pl/
    2. Accept the terms and conditions
    3. Download 'full_dataset.csv'
    4. Place it in: data/raw/recipenlg/full_dataset.csv

    Dataset size: ~2GB
    Records: 2.2 million recipes
    """)

def download_foodcom():
    """Instructions for Food.com dataset."""
    print("=" * 60)
    print("Food.com Dataset (Kaggle)")
    print("=" * 60)
    print("""
    1. Create a Kaggle account: https://kaggle.com
    2. Install kaggle CLI: pip install kaggle
    3. Set up API credentials (~/.kaggle/kaggle.json)
    4. Run: kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
    5. Extract to: data/raw/foodcom/

    Or download manually from:
    https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

    Files:
    - RAW_recipes.csv (~60MB)
    - RAW_interactions.csv (~300MB)
    """)

if __name__ == "__main__":
    print("Recipe AI Dataset Download Helper")
    print()
    download_recipenlg()
    print()
    download_foodcom()
'''

    script_path = DATA_DIR / "download_datasets.py"
    with open(script_path, "w") as f:
        f.write(download_script)

    print(f"Created download helper: {script_path}")
    print("\nRun it with: python data/download_datasets.py")
    return download_script, script_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    In this tutorial, we:

    1. **Explored available datasets** - RecipeNLG (2.2M recipes) and Food.com (230K recipes)
    2. **Created sample data** - Demonstrated the structure and format
    3. **Analyzed recipe structure** - Ingredients, steps, metadata
    4. **Performed quality checks** - Missing values, duplicates, validation
    5. **Created download helpers** - Scripts to assist with full dataset download

    ## Dataset Comparison

    | Feature | RecipeNLG | Food.com |
    |---------|-----------|----------|
    | Size | 2.2M recipes | 230K recipes |
    | Ingredients | Structured list | Comma-separated |
    | Instructions | Step-by-step | Pipe-separated |
    | Reviews | No | Yes |
    | Nutrition | No | Yes |
    | Best for | Large-scale training | Rich metadata |

    ## What's Next?

    In **Tutorial 04**, we'll preprocess these datasets:
    - Clean and normalize ingredients
    - Parse cooking times and servings
    - Create training/validation splits
    - Format data for model training

    ---

    ## Exercises

    1. Download the full RecipeNLG dataset and explore its statistics
    2. Find the most common ingredients across all recipes
    3. What's the distribution of recipe lengths (by step count)?
    4. How would you detect and handle duplicate recipes?
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
