# Tutorial 01: Environment Setup

Before we can train models, we need to set up our development environment. This tutorial covers everything you need to get started.

## What We Are Setting Up

1. **Python environment** - Isolated environment for our project
2. **Core ML libraries** - PyTorch, Transformers, etc.
3. **Hugging Face account** - Access to models and datasets
4. **GPU access** - Where we will actually train (cloud options)

---

## Step 1: Python Environment

We will use Python 3.10+ with a virtual environment.

### Check your Python version

python3 --version

You need Python 3.10 or higher. If you do not have it:
- macOS: brew install python@3.11
- Ubuntu: sudo apt install python3.11

### Create a virtual environment

cd ~/Documents/Code/recipe-ai-tutorial
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate

Your terminal should now show (venv) at the start.

---

## Step 2: Install Core Libraries

Create a requirements.txt with our initial dependencies:

See requirements.txt in this folder.

Install them:

pip install -r tutorials/01-environment-setup/requirements.txt

This will take a few minutes. Here is what each package does:

| Package | Purpose |
|---------|---------|
| torch | The deep learning framework (PyTorch) |
| transformers | Hugging Face library for working with LLMs |
| datasets | Hugging Face library for loading datasets |
| accelerate | Distributed training and mixed precision |
| peft | Parameter-Efficient Fine-Tuning (LoRA, etc.) |
| bitsandbytes | 8-bit quantization for memory efficiency |
| jupyter | For running notebooks |
| pandas | Data manipulation |
| numpy | Numerical computing |

---

## Step 3: Hugging Face Account

Hugging Face is like GitHub for ML models. We need an account to:
- Download pre-trained models
- Access datasets
- Push our fine-tuned models

### Create an account

1. Go to https://huggingface.co/join
2. Sign up (free)
3. Verify your email

### Get an access token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "recipe-ai" 
4. Select "Write" access
5. Copy the token

### Login from terminal

huggingface-cli login

Paste your token when prompted.

---

## Step 4: Verify Installation

Run the verification script:

python tutorials/01-environment-setup/verify.py

You should see all checks passing.

---

## Step 5: GPU Access (For Later)

Training on CPU is possible but painfully slow. Here are your options:

### Free Options
- **Google Colab** - Free tier has limited GPU time
- **Kaggle Notebooks** - 30 hours/week of free GPU

### Paid Options (When Ready to Train)
- **Google Colab Pro** - 0/month, better GPUs
- **RunPod** - Pay per hour, great GPUs
- **Lambda Labs** - Similar to RunPod
- **Vast.ai** - Cheapest, community GPUs

For now, we will use CPU for learning concepts. We will set up cloud GPU access when we start actual training in Tutorial 06.

---

## What is Next?

In Tutorial 02, we will learn how Transformers and LLMs actually work - the theory you need before training.

---

## Troubleshooting

### "torch not found" or CUDA errors
Make sure you activated the venv: source venv/bin/activate

### Slow installation
PyTorch is large (~2GB). Be patient.

### M1/M2 Mac users
PyTorch works on Apple Silicon. You will use "mps" instead of "cuda" for GPU acceleration.

### bitsandbytes fails on Mac
This is expected - bitsandbytes requires CUDA. Skip it for now; we will use it on cloud GPUs.
