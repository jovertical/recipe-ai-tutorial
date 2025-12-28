# Problem 15: Fine-tune Encoder
#
# Learn to fine-tune a pretrained encoder for ingredient embeddings.
# This combines transfer learning with domain adaptation.
#
# You'll implement:
# 1. freeze_layers() / unfreeze_layers() - control trainable layers
# 2. ProjectionHead class - learnable projection layer
# 3. EncoderFineTuner - complete fine-tuning pipeline
# 4. gradual_unfreezing() - progressive training strategy
#
# Example:
#   encoder = MockPretrainedEncoder()
#   fine_tuner = EncoderFineTuner(encoder, projection_dim=64)
#   fine_tuner.train(triplets)
#   # Result: domain-adapted embeddings
#
# Constraints:
#   - Frozen layers don't update during training
#   - Projection head adapts pretrained features
#   - Support gradual unfreezing strategy
#
# ML Relevance: Fine-tuning pretrained models leverages general language
# understanding, adapts to specific domains (recipes), is usually better
# than training from scratch, and is more data-efficient.

from typing import List, Dict, Tuple, Optional
import numpy as np
import random


class MockPretrainedEncoder:
    """Mock pretrained encoder (simulates a transformer encoder)."""
    
    def __init__(self, hidden_dim: int = 384, num_layers: int = 6):
        """Initialize with mock layer weights."""
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        np.random.seed(42)
        self.layers = []
        for i in range(num_layers):
            self.layers.append({
                'weight': np.random.randn(hidden_dim, hidden_dim) * 0.02,
                'frozen': False
            })
        self._vocab_embeddings = {}
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to normalized embedding."""
        if text not in self._vocab_embeddings:
            seed = sum(ord(c) for c in text) % 10000
            np.random.seed(seed)
            emb = np.random.randn(self.hidden_dim).astype(np.float32)
            for layer in self.layers:
                if not layer['frozen']:
                    emb = np.tanh(emb @ layer['weight'])
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            self._vocab_embeddings[text] = emb
        return self._vocab_embeddings[text].copy()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return np.stack([self.encode(t) for t in texts])


def freeze_layers(encoder: MockPretrainedEncoder, num_layers_to_freeze: int = None, freeze_all: bool = False) -> None:
    """Freeze encoder layers (frozen layers don't update)."""
    # Your solution here
    pass


def unfreeze_layers(encoder: MockPretrainedEncoder, num_layers_to_unfreeze: int = None, unfreeze_all: bool = False) -> None:
    """Unfreeze encoder layers."""
    # Your solution here
    pass


class ProjectionHead:
    """Projection head for fine-tuning (MLP or linear)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64, use_hidden: bool = True):
        """Initialize projection weights."""
        # Your solution here
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Handles (dim,) or (batch, dim) input."""
        # Your solution here
        pass
    
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass and weight update. Returns gradient to encoder."""
        # Your solution here
        pass


class EncoderFineTuner:
    """Fine-tuner for pretrained encoder."""
    
    def __init__(self, encoder: MockPretrainedEncoder, projection_dim: int = 64, learning_rate: float = 1e-4, freeze_encoder: bool = True):
        """Initialize fine-tuner with encoder and projection head."""
        # Your solution here
        pass
    
    def encode(self, text: str) -> np.ndarray:
        """Encode through encoder + projection head."""
        # Your solution here
        pass
    
    def compute_triplet_loss(self, anchor: str, positive: str, negative: str, margin: float = 0.3) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute triplet loss and gradients."""
        # Your solution here
        pass
    
    def train_step(self, anchor: str, positive: str, negative: str) -> float:
        """Perform one training step. Returns loss."""
        # Your solution here
        pass
    
    def train(self, triplets: List[Tuple[str, str, str]], epochs: int = 5, batch_size: int = 32) -> List[float]:
        """Train on triplets. Returns loss history."""
        # Your solution here
        pass
    
    def get_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple texts."""
        # Your solution here
        pass


def gradual_unfreezing(fine_tuner: EncoderFineTuner, triplets: List[Tuple[str, str, str]], epochs_per_stage: int = 2) -> List[float]:
    """Train with gradual unfreezing: frozen -> unfreeze top -> unfreeze more."""
    # Your solution here
    pass


def compare_frozen_vs_finetuned(encoder: MockPretrainedEncoder, triplets: List[Tuple[str, str, str]], test_triplets: List[Tuple[str, str, str]]) -> Dict[str, float]:
    """Compare frozen encoder vs fine-tuned accuracy."""
    # Your solution here
    pass


def learning_rate_schedule(base_lr: float, step: int, total_steps: int, warmup_steps: int = 100, schedule: str = "linear") -> float:
    """Compute LR with warmup and decay: 'linear', 'cosine', or 'constant'."""
    # Your solution here
    pass


def evaluate_fine_tuned(fine_tuner: EncoderFineTuner, test_triplets: List[Tuple[str, str, str]]) -> Dict[str, float]:
    """Evaluate fine-tuned model on triplets."""
    # Your solution here
    pass


# ----- Tests (do not modify) -----
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    categories = {
        "butter": "fat", "oil": "fat", "margarine": "fat",
        "flour": "grain", "rice": "grain",
        "garlic": "aromatic", "onion": "aromatic",
        "chicken": "protein", "beef": "protein"
    }
    ingredients = list(categories.keys())
    
    triplets = []
    for _ in range(100):
        cat = random.choice(list(set(categories.values())))
        same_cat = [i for i, c in categories.items() if c == cat]
        diff_cat = [i for i, c in categories.items() if c != cat]
        if len(same_cat) >= 2:
            a, p = random.sample(same_cat, 2)
            n = random.choice(diff_cat)
            triplets.append((a, p, n))
    
    print("Testing MockPretrainedEncoder...")
    
    encoder = MockPretrainedEncoder(hidden_dim=64, num_layers=4)
    assert encoder.hidden_dim == 64, "Test 1a failed"
    assert encoder.num_layers == 4, "Test 1b failed"
    print("  ✓ Encoder initialized")
    
    emb = encoder.encode("butter")
    assert emb.shape == (64,), f"Test 2a failed: {emb.shape}"
    assert abs(np.linalg.norm(emb) - 1.0) < 0.01, "Test 2b: should be normalized"
    print("  ✓ Encoding works")
    
    embs = encoder.encode_batch(["butter", "flour"])
    assert embs.shape == (2, 64), f"Test 3 failed: {embs.shape}"
    print("  ✓ Batch encoding works")
    
    print("\nTesting freeze_layers...")
    
    freeze_layers(encoder, num_layers_to_freeze=2)
    assert encoder.layers[0]['frozen'], "Test 4a failed"
    assert encoder.layers[1]['frozen'], "Test 4b failed"
    assert not encoder.layers[2]['frozen'], "Test 4c failed"
    print("  ✓ Partial freezing works")
    
    freeze_layers(encoder, freeze_all=True)
    assert all(l['frozen'] for l in encoder.layers), "Test 5 failed"
    print("  ✓ Freeze all works")
    
    print("\nTesting unfreeze_layers...")
    
    unfreeze_layers(encoder, num_layers_to_unfreeze=2)
    assert not encoder.layers[-1]['frozen'], "Test 6a failed"
    assert not encoder.layers[-2]['frozen'], "Test 6b failed"
    assert encoder.layers[0]['frozen'], "Test 6c failed"
    print("  ✓ Partial unfreezing works")
    
    print("\nTesting ProjectionHead...")
    
    proj = ProjectionHead(input_dim=64, hidden_dim=128, output_dim=32)
    x = np.random.randn(64)
    out = proj.forward(x)
    assert out.shape == (32,), f"Test 7a failed: {out.shape}"
    print("  ✓ Projection head forward works")
    
    x_batch = np.random.randn(5, 64)
    out_batch = proj.forward(x_batch)
    assert out_batch.shape == (5, 32), f"Test 8 failed: {out_batch.shape}"
    print("  ✓ Batch projection works")
    
    print("\nTesting EncoderFineTuner...")
    
    encoder = MockPretrainedEncoder(hidden_dim=64, num_layers=4)
    fine_tuner = EncoderFineTuner(encoder, projection_dim=32, freeze_encoder=True)
    print("  ✓ Fine-tuner initialized")
    
    emb = fine_tuner.encode("butter")
    assert emb.shape == (32,), f"Test 10 failed: {emb.shape}"
    print("  ✓ Fine-tuner encoding works")
    
    loss = fine_tuner.train_step("butter", "margarine", "garlic")
    assert isinstance(loss, (int, float)), f"Test 11 failed: {type(loss)}"
    print(f"  ✓ Training step loss: {loss:.4f}")
    
    losses = fine_tuner.train(triplets[:30], epochs=3)
    assert len(losses) == 3, f"Test 12a failed: {len(losses)}"
    print(f"  ✓ Training: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    embeddings = fine_tuner.get_embeddings(ingredients)
    assert len(embeddings) == len(ingredients), "Test 13 failed"
    print("  ✓ Get embeddings works")
    
    print("\nTesting gradual_unfreezing...")
    
    encoder = MockPretrainedEncoder(hidden_dim=64, num_layers=4)
    fine_tuner = EncoderFineTuner(encoder, projection_dim=32)
    losses = gradual_unfreezing(fine_tuner, triplets[:20], epochs_per_stage=1)
    assert len(losses) > 0, "Test 14 failed"
    print(f"  ✓ Gradual unfreezing: {len(losses)} total epochs")
    
    print("\nTesting compare_frozen_vs_finetuned...")
    
    encoder = MockPretrainedEncoder(hidden_dim=64, num_layers=4)
    train_triplets = triplets[:50]
    test_triplets = triplets[50:60]
    
    comparison = compare_frozen_vs_finetuned(encoder, train_triplets, test_triplets)
    assert "frozen_accuracy" in comparison, "Test 15a failed"
    assert "finetuned_accuracy" in comparison, "Test 15b failed"
    print(f"  ✓ Frozen: {comparison['frozen_accuracy']:.2%}, Fine-tuned: {comparison['finetuned_accuracy']:.2%}")
    
    print("\nTesting learning_rate_schedule...")
    
    lr = learning_rate_schedule(1e-4, step=0, total_steps=1000, warmup_steps=100)
    assert lr < 1e-4, "Test 16a: should be in warmup"
    
    lr = learning_rate_schedule(1e-4, step=100, total_steps=1000, warmup_steps=100)
    assert abs(lr - 1e-4) < 1e-8, f"Test 16b: should be at base LR, got {lr}"
    
    lr = learning_rate_schedule(1e-4, step=1000, total_steps=1000, warmup_steps=100, schedule="linear")
    assert lr < 1e-5, f"Test 16c: should be near 0, got {lr}"
    print("  ✓ LR schedule works")
    
    print("\nTesting evaluate_fine_tuned...")
    
    metrics = evaluate_fine_tuned(fine_tuner, test_triplets)
    assert "accuracy" in metrics, "Test 17 failed"
    print(f"  ✓ Evaluation accuracy: {metrics['accuracy']:.2%}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
