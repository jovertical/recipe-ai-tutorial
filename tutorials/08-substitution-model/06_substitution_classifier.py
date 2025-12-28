# Problem 6: Binary Substitution Classifier
#
# Build a neural network classifier that predicts whether two ingredients can
# substitute for each other.
#
# Example:
#   classifier = SubstitutionClassifier(embedding_dim=64, hidden_dim=128)
#   prob = classifier.predict("butter", "margarine")
#   # Returns: 0.92 (high probability of valid substitution)
#   prob = classifier.predict("butter", "salt")
#   # Returns: 0.03 (low probability)
#
# ML Relevance: Binary classification is a fundamental ML task with clear
# decision boundary (substitute or not), probability output for ranking,
# trainable on labeled pairs, and foundation for more complex models.
# This introduces neural network training for substitution.
#
# Your Task:
#   1. Implement SubstitutionClassifier with feedforward layers
#   2. Implement training loop with BCE loss
#   3. Implement evaluation metrics (accuracy, precision, recall, F1)
#   4. Implement threshold tuning for optimal classification

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import random


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Compute binary cross-entropy loss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    validation_split: float = 0.1


class SubstitutionClassifier:
    """
    Neural network classifier for ingredient substitution.
    
    Architecture:
        Input: [embedding_a, embedding_b, |a-b|, a*b]
        -> Hidden Layer 1 (ReLU)
        -> Hidden Layer 2 (ReLU)
        -> Output (Sigmoid)
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2
    ):
        """
        Initialize classifier.
        
        Input: concat(emb_a, emb_b, |a-b|, a*b) = 4 * embedding_dim
        Initialize weights using Xavier/He initialization.
        Layers: input -> hidden1 -> hidden2 -> output
        """
        self.input_dim = 4 * embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        # Your solution here
        pass
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # Your solution here
        pass
    
    def create_pair_features(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray
    ) -> np.ndarray:
        """
        Create feature vector from two embeddings.
        Combine: [emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b]
        """
        # Your solution here
        pass
    
    def forward(
        self,
        features: np.ndarray,
        training: bool = False
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward pass through the network. Returns (prediction, cache for backprop)."""
        # Your solution here
        pass
    
    def predict(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray
    ) -> float:
        """Predict substitution probability (0 to 1)."""
        # Your solution here
        pass
    
    def predict_batch(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """Predict for batch of pairs."""
        # Your solution here
        pass
    
    def backward(
        self,
        y_true: float,
        y_pred: float,
        cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients."""
        # Your solution here
        pass
    
    def update_weights(
        self,
        gradients: Dict[str, np.ndarray],
        learning_rate: float
    ):
        """Update weights using gradients."""
        # Your solution here
        pass
    
    def train_step(
        self,
        batch_features: np.ndarray,
        batch_labels: np.ndarray,
        learning_rate: float
    ) -> float:
        """Perform one training step on a batch. Returns batch loss."""
        # Your solution here
        pass
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all network weights."""
        # Your solution here
        pass
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set network weights."""
        # Your solution here
        pass


def train_classifier(
    classifier: SubstitutionClassifier,
    train_data: List[Tuple[np.ndarray, np.ndarray, float]],
    val_data: List[Tuple[np.ndarray, np.ndarray, float]],
    config: TrainingConfig
) -> Dict[str, List[float]]:
    """
    Train the classifier.
    
    Hints:
    1. Create feature vectors from embeddings
    2. Shuffle and batch data
    3. For each epoch, iterate batches and update weights
    4. Evaluate on validation set
    5. Implement early stopping
    """
    # Your solution here
    pass


def evaluate_classifier(
    classifier: SubstitutionClassifier,
    test_data: List[Tuple[np.ndarray, np.ndarray, float]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate classifier performance.
    
    Returns dictionary with metrics: accuracy, precision, recall, f1, auc
    """
    # Your solution here
    pass


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, int]:
    """Compute confusion matrix values. Returns dict with TP, TN, FP, FN counts."""
    # Your solution here
    pass


def find_optimal_threshold(
    classifier: SubstitutionClassifier,
    val_data: List[Tuple[np.ndarray, np.ndarray, float]],
    metric: str = "f1"
) -> float:
    """Find optimal classification threshold for given metric ("f1", "precision", "recall")."""
    # Your solution here
    pass


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve. Returns tuple of (fpr, tpr, thresholds)."""
    # Your solution here
    pass


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute area under ROC curve using trapezoidal rule."""
    # Your solution here
    pass


class EnsembleClassifier:
    """Ensemble of substitution classifiers."""
    
    def __init__(self, classifiers: List[SubstitutionClassifier]):
        """Initialize ensemble."""
        # Your solution here
        pass
    
    def predict(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        aggregation: str = "mean"
    ) -> float:
        """Ensemble prediction with aggregation: "mean", "max", or "vote"."""
        # Your solution here
        pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    embedding_dim = 64
    
    # Create mock embeddings
    def create_embedding():
        emb = np.random.randn(embedding_dim).astype(np.float32)
        return emb / np.linalg.norm(emb)
    
    # Create similar pairs (positive examples)
    positive_pairs = []
    for _ in range(100):
        base = create_embedding()
        similar = base + np.random.randn(embedding_dim) * 0.1
        similar = similar / np.linalg.norm(similar)
        positive_pairs.append((base, similar, 1.0))
    
    # Create dissimilar pairs (negative examples)
    negative_pairs = []
    for _ in range(100):
        emb_a = create_embedding()
        emb_b = create_embedding()
        negative_pairs.append((emb_a, emb_b, 0.0))
    
    all_data = positive_pairs + negative_pairs
    random.shuffle(all_data)
    
    # Split data
    train_data = all_data[:160]
    val_data = all_data[160:180]
    test_data = all_data[180:]
    
    print("Testing SubstitutionClassifier...")
    
    # Test 1: Initialize classifier
    classifier = SubstitutionClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=128
    )
    print("  ✓ Classifier initialized")
    
    # Test 2: Create pair features
    emb_a, emb_b, _ = train_data[0]
    features = classifier.create_pair_features(emb_a, emb_b)
    assert features.shape == (4 * embedding_dim,), f"Test 2 failed: {features.shape}"
    print(f"  ✓ Feature vector shape: {features.shape}")
    
    # Test 3: Forward pass
    pred, cache = classifier.forward(features, training=True)
    assert 0 <= pred <= 1, f"Test 3a failed: {pred}"
    assert len(cache) > 0, "Test 3b failed: cache empty"
    print(f"  ✓ Forward pass: prediction = {pred:.4f}")
    
    # Test 4: Predict
    pred = classifier.predict(emb_a, emb_b)
    assert 0 <= pred <= 1, f"Test 4 failed: {pred}"
    print(f"  ✓ Predict: {pred:.4f}")
    
    # Test 5: Batch predict
    batch_a = np.array([d[0] for d in train_data[:10]])
    batch_b = np.array([d[1] for d in train_data[:10]])
    preds = classifier.predict_batch(batch_a, batch_b)
    assert preds.shape == (10,), f"Test 5 failed: {preds.shape}"
    print(f"  ✓ Batch predict shape: {preds.shape}")
    
    print("\nTesting training...")
    
    # Test 6: Training config
    config = TrainingConfig(
        learning_rate=0.01,
        batch_size=16,
        epochs=20,
        early_stopping_patience=5
    )
    print(f"  ✓ Config: lr={config.learning_rate}, epochs={config.epochs}")
    
    # Test 7: Train classifier
    history = train_classifier(classifier, train_data, val_data, config)
    assert "train_loss" in history, "Test 7a failed"
    assert len(history["train_loss"]) > 0, "Test 7b failed"
    print(f"  ✓ Training complete: {len(history['train_loss'])} epochs")
    print(f"    Final train loss: {history['train_loss'][-1]:.4f}")
    if "val_loss" in history:
        print(f"    Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Test 8: Evaluate
    metrics = evaluate_classifier(classifier, test_data)
    assert "accuracy" in metrics, "Test 8a failed"
    assert "f1" in metrics, "Test 8b failed"
    assert 0 <= metrics["accuracy"] <= 1, f"Test 8c failed: {metrics['accuracy']}"
    print(f"  ✓ Test metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}")
    
    print("\nTesting threshold tuning...")
    
    # Test 9: Find optimal threshold
    optimal_threshold = find_optimal_threshold(classifier, val_data, metric="f1")
    assert 0 < optimal_threshold < 1, f"Test 9 failed: {optimal_threshold}"
    print(f"  ✓ Optimal threshold: {optimal_threshold:.3f}")
    
    # Test 10: Evaluate with optimal threshold
    metrics_optimal = evaluate_classifier(classifier, test_data, threshold=optimal_threshold)
    print(f"  ✓ Metrics with optimal threshold:")
    print(f"    Accuracy: {metrics_optimal['accuracy']:.4f}")
    print(f"    F1: {metrics_optimal['f1']:.4f}")
    
    print("\nTesting confusion matrix...")
    
    # Test 11: Confusion matrix
    y_true = np.array([d[2] for d in test_data])
    y_pred = np.array([classifier.predict(d[0], d[1]) for d in test_data])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert "TP" in cm and "TN" in cm, "Test 11 failed"
    print(f"  ✓ Confusion matrix: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")
    
    print("\nTesting ROC curve...")
    
    # Test 12: ROC curve
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_pred)
    assert len(fpr) == len(tpr), "Test 12a failed"
    assert fpr[0] == 0 and fpr[-1] == 1, "Test 12b failed"
    print(f"  ✓ ROC curve: {len(fpr)} points")
    
    # Test 13: AUC
    auc = compute_auc(fpr, tpr)
    assert 0 <= auc <= 1, f"Test 13 failed: {auc}"
    print(f"  ✓ AUC: {auc:.4f}")
    
    print("\nTesting EnsembleClassifier...")
    
    # Test 14: Create ensemble
    classifiers = [
        SubstitutionClassifier(embedding_dim, hidden_dim=64),
        SubstitutionClassifier(embedding_dim, hidden_dim=128),
        SubstitutionClassifier(embedding_dim, hidden_dim=64)
    ]
    ensemble = EnsembleClassifier(classifiers)
    print("  ✓ Ensemble created with 3 classifiers")
    
    # Test 15: Ensemble predict
    pred = ensemble.predict(emb_a, emb_b, aggregation="mean")
    assert 0 <= pred <= 1, f"Test 15 failed: {pred}"
    print(f"  ✓ Ensemble prediction: {pred:.4f}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Neural network architecture for pair classification")
    print("2. Feature engineering from embedding pairs")
    print("3. Training loop with backpropagation")
    print("4. Threshold tuning for optimal decisions")
    print("5. Ensemble methods for robustness")
    print("\nNext: Pairwise ranking for better ordering!")
