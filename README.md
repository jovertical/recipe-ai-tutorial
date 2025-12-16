# Recipe AI: From Zero to Production

A hands-on tutorial series for learning AI/ML engineering by building a real product.

## The Goal

Build **Recipe AI** - an API platform for intelligent recipe generation, ingredient substitution, and nutritional analysis. This is not a toy project; it is a production-ready platform that will power real applications.

## What You Will Learn

- Data collection and preprocessing for LLMs
- Fine-tuning open-source models (Llama, Mistral)
- Training custom embedding models
- Model evaluation and iteration
- Serving models in production
- Building APIs around ML models
- Deploying and scaling AI applications

## Tutorial Structure

### Part 1: Foundations
- [ ] 00 - Python Fundamentals
- [x] 01 - Environment Setup (Python, CUDA, Hugging Face)
- [ ] 02 - Understanding Transformers and LLMs
- [ ] 03 - Dataset Collection (RecipeNLG, Food.com)
- [ ] 04 - Data Preprocessing and Exploration

### Part 2: Training Custom Models
- [ ] 05 - Tokenization Deep Dive
- [ ] 06 - Fine-tuning with LoRA/QLoRA
- [ ] 07 - Training Ingredient Embeddings
- [ ] 08 - Building a Substitution Model
- [ ] 09 - Evaluation and Iteration

### Part 3: Serving and Infrastructure
- [ ] 10 - Model Serving with vLLM/Ollama
- [ ] 11 - Vector Database Setup (pgvector/Qdrant)
- [ ] 12 - Building the FastAPI Backend
- [ ] 13 - Authentication and Rate Limiting

### Part 4: The Platform
- [ ] 14 - API Design and Documentation
- [ ] 15 - Dashboard (API keys, usage tracking)
- [ ] 16 - Deployment and Scaling
- [ ] 17 - Integrating with Real world apps

## Prerequisites

- Python basics
- Some familiarity with APIs
- Curiosity about how AI actually works

## Hardware Requirements

- **For learning**: Google Colab Pro or Kaggle notebooks (free)
- **For serious training**: RunPod/Lambda Labs GPU rental
- **Local option**: NVIDIA GPU with 8GB+ VRAM

## Project Structure

recipe-ai-tutorial/
  tutorials/           # Step-by-step guides
  notebooks/           # Jupyter notebooks for experimentation
  src/                 # Reusable code we build along the way
  data/                # Datasets (gitignored, with download scripts)
  models/              # Trained models (gitignored)
  platform/            # The final Recipe AI platform

## Lets Begin

Start with Tutorial 01: Environment Setup in tutorials/01-environment-setup/

---

This tutorial series is being built as a learning journey. Each part is written as we go.
