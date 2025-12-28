# Problem 17: Confidence Scoring
#
# Estimate prediction confidence to know when to trust substitution suggestions.
#
# Example:
#   scorer = ConfidenceScorer(model)
#   score, confidence = scorer.predict("butter", "margarine")
#   # Returns: (0.92, 0.85) - high score, high confidence
#
# ML Relevance: Confidence estimation enables fallback strategies and helps
# users understand prediction reliability.
#
# Your Task:
#   1. Implement softmax-based confidence
#   2. Implement ensemble disagreement
#   3. Implement uncertainty quantification
#   4. Calibrate confidence scores


from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import random


@dataclass
class PredictionWithConfidence:
    """Prediction with confidence information."""
    original: str
    substitute: str
    score: float
    confidence: float
    uncertainty_type: str  # "low", "medium", "high"
    calibrated_score: float = None
    ensemble_agreement: float = None
    is_out_of_distribution: bool = False
    explanation: str = ""


class ConfidenceScorer:
    """
    Score confidence of substitution predictions.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        calibration_data: List[Tuple[float, int]] = None
    ):
        """
        Initialize scorer.
        
        Args:
            temperature: Temperature for softmax calibration
            calibration_data: List of (predicted_prob, actual_label) for calibration
        """
        self.temperature = temperature
        self.calibration_data = calibration_data or []
        self.calibration_map = None
    
    def compute_confidence(
        self,
        logits: np.ndarray,
        method: str = "softmax"
    ) -> float:
        """
        Compute confidence from model outputs.
        
        Args:
            logits: Raw model outputs
            method: "softmax", "entropy", or "margin"
            
        Returns:
            Confidence score (0 to 1)
        """
        # Your solution here
        pass
    
    def softmax_confidence(self, logits: np.ndarray) -> float:
        """Confidence from softmax probability."""
        # Your solution here
        pass
    
    def entropy_confidence(self, logits: np.ndarray) -> float:
        """Confidence from prediction entropy (low entropy = high confidence)."""
        # Your solution here
        pass
    
    def margin_confidence(self, logits: np.ndarray) -> float:
        """Confidence from margin between top two predictions."""
        # Your solution here
        pass
    
    def calibrate_confidence(
        self,
        raw_confidence: float
    ) -> float:
        """Apply calibration to raw confidence."""
        # Your solution here
        pass
    
    def fit_calibration(
        self,
        predictions: List[float],
        labels: List[int],
        method: str = "isotonic"
    ):
        """
        Fit calibration mapping.
        
        Args:
            predictions: Model predictions
            labels: True labels
            method: "isotonic", "platt", or "temperature"
        """
        # Your solution here
        pass


class UncertaintyEstimator:
    """
    Estimate uncertainty in predictions.
    """
    
    def __init__(self):
        pass
    
    def epistemic_uncertainty(
        self,
        predictions: List[float]
    ) -> float:
        """
        Estimate epistemic uncertainty (model uncertainty).
        
        Higher when model has limited knowledge about this case.
        
        Args:
            predictions: Multiple predictions (e.g., from dropout sampling)
            
        Returns:
            Uncertainty score
        """
        # Your solution here
        pass
    
    def aleatoric_uncertainty(
        self,
        mean_prediction: float,
        variance_prediction: float
    ) -> float:
        """
        Estimate aleatoric uncertainty (data uncertainty).
        
        Higher when the data itself is inherently uncertain.
        """
        # Your solution here
        pass
    
    def total_uncertainty(
        self,
        predictions: List[float],
        variance: float = None
    ) -> Dict[str, float]:
        """
        Compute total uncertainty breakdown.
        
        Returns:
            Dictionary with epistemic, aleatoric, and total uncertainty
        """
        # Your solution here
        pass
    
    def classify_uncertainty(
        self,
        uncertainty: float,
        thresholds: Tuple[float, float] = (0.1, 0.3)
    ) -> str:
        """Classify uncertainty as low/medium/high."""
        # Your solution here
        pass


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    """
    
    def __init__(
        self,
        model,  # Model with dropout
        num_samples: int = 20
    ):
        """
        Initialize MC Dropout predictor.
        
        Args:
            model: Model with dropout layers
            num_samples: Number of forward passes
        """
        self.model = model
        self.num_samples = num_samples
    
    def predict_with_uncertainty(
        self,
        x: np.ndarray
    ) -> Dict[str, float]:
        """
        Make prediction with uncertainty estimate.
        
        Returns:
            Dictionary with mean, std, and uncertainty
        """
        # Your solution here
        pass
    
    def sample_predictions(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """Get multiple predictions with dropout enabled."""
        # Your solution here
        pass


class EnsembleConfidence:
    """
    Confidence from ensemble disagreement.
    """
    
    def __init__(self, models: List = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of models in ensemble
        """
        self.models = models or []
    
    def predict_with_confidence(
        self,
        x: np.ndarray
    ) -> Dict[str, float]:
        """
        Make prediction with ensemble confidence.
        
        Returns:
            Dictionary with prediction, confidence, and agreement
        """
        # Your solution here
        pass
    
    def compute_agreement(
        self,
        predictions: List[float],
        threshold: float = 0.5
    ) -> float:
        """
        Compute ensemble agreement.
        
        Returns:
            Agreement score (0 to 1)
        """
        # Your solution here
        pass
    
    def get_prediction_stats(
        self,
        predictions: List[float]
    ) -> Dict[str, float]:
        """Get statistics about ensemble predictions."""
        # Your solution here
        pass


class OutOfDistributionDetector:
    """
    Detect out-of-distribution inputs.
    """
    
    def __init__(
        self,
        training_embeddings: np.ndarray = None,
        threshold_percentile: float = 95
    ):
        """
        Initialize detector.
        
        Args:
            training_embeddings: Embeddings from training data
            threshold_percentile: Percentile for OOD threshold
        """
        self.training_embeddings = training_embeddings
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        
        if training_embeddings is not None:
            self._compute_threshold()
    
    def _compute_threshold(self):
        """Compute OOD threshold from training data."""
        # Your solution here
        pass
    
    def is_ood(
        self,
        embedding: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if input is out of distribution.
        
        Returns:
            Tuple of (is_ood, ood_score)
        """
        # Your solution here
        pass
    
    def compute_ood_score(
        self,
        embedding: np.ndarray,
        method: str = "mahalanobis"
    ) -> float:
        """
        Compute OOD score.
        
        Args:
            embedding: Input embedding
            method: "mahalanobis", "knn", or "reconstruction"
        """
        # Your solution here
        pass
    
    def fit(self, training_embeddings: np.ndarray):
        """Fit detector on training data."""
        self.training_embeddings = training_embeddings
        self._compute_threshold()


class CalibrationMetrics:
    """Compute calibration metrics."""
    
    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures how well confidence aligns with accuracy.
        """
        # Your solution here
        pass
    
    @staticmethod
    def reliability_diagram(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagram.
        
        Returns:
            Dictionary with bin_centers, accuracies, confidences
        """
        # Your solution here
        pass
    
    @staticmethod
    def brier_score(
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Brier score (lower is better)."""
        # Your solution here
        pass


def temperature_scaling(
    logits: np.ndarray,
    temperature: float
) -> np.ndarray:
    """Apply temperature scaling to logits."""
    # Your solution here
    pass


def platt_scaling(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters.
    
    Returns:
        Tuple of (A, B) for sigmoid(A * x + B)
    """
    # Your solution here
    pass


def create_confidence_report(
    prediction: PredictionWithConfidence
) -> str:
    """Create human-readable confidence report."""
    # Your solution here
    pass


# ----- Tests (do not modify below this line) -----

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    print("Testing ConfidenceScorer...")
    
    scorer = ConfidenceScorer(temperature=1.0)
    
    # Test 1: Softmax confidence
    logits = np.array([2.0, 0.5, -1.0])
    conf = scorer.softmax_confidence(logits)
    assert 0 <= conf <= 1, f"Test 1 failed: {conf}"
    print(f"  ✓ Softmax confidence: {conf:.3f}")
    
    # Test 2: Entropy confidence
    conf_entropy = scorer.entropy_confidence(logits)
    assert 0 <= conf_entropy <= 1, f"Test 2 failed: {conf_entropy}"
    print(f"  ✓ Entropy confidence: {conf_entropy:.3f}")
    
    # Test 3: Margin confidence
    conf_margin = scorer.margin_confidence(logits)
    assert 0 <= conf_margin <= 1, f"Test 3 failed: {conf_margin}"
    print(f"  ✓ Margin confidence: {conf_margin:.3f}")
    
    # Test 4: High confidence for certain prediction
    certain_logits = np.array([10.0, 0.0, 0.0])
    conf_certain = scorer.compute_confidence(certain_logits)
    assert conf_certain > 0.9, f"Test 4 failed: {conf_certain}"
    print(f"  ✓ Certain prediction confidence: {conf_certain:.3f}")
    
    print("\nTesting UncertaintyEstimator...")
    
    estimator = UncertaintyEstimator()
    
    # Test 5: Epistemic uncertainty
    # Consistent predictions = low uncertainty
    consistent = [0.8, 0.82, 0.79, 0.81, 0.80]
    epistemic = estimator.epistemic_uncertainty(consistent)
    
    # Inconsistent predictions = high uncertainty
    inconsistent = [0.9, 0.3, 0.7, 0.4, 0.8]
    epistemic_high = estimator.epistemic_uncertainty(inconsistent)
    
    assert epistemic < epistemic_high, f"Test 5 failed: {epistemic} vs {epistemic_high}"
    print(f"  ✓ Epistemic uncertainty: low={epistemic:.3f}, high={epistemic_high:.3f}")
    
    # Test 6: Total uncertainty
    total = estimator.total_uncertainty(inconsistent)
    assert "epistemic" in total, "Test 6a failed"
    assert "total" in total, "Test 6b failed"
    print(f"  ✓ Total uncertainty: {total}")
    
    # Test 7: Classify uncertainty
    level = estimator.classify_uncertainty(0.05)
    assert level == "low", f"Test 7a failed: {level}"
    level = estimator.classify_uncertainty(0.5)
    assert level == "high", f"Test 7b failed: {level}"
    print(f"  ✓ Uncertainty classification works")
    
    print("\nTesting EnsembleConfidence...")
    
    ensemble = EnsembleConfidence()
    
    # Test 8: Compute agreement
    # High agreement
    predictions_agree = [0.85, 0.87, 0.84, 0.86]
    agreement = ensemble.compute_agreement(predictions_agree)
    assert agreement > 0.8, f"Test 8a failed: {agreement}"
    
    # Low agreement
    predictions_disagree = [0.9, 0.2, 0.6, 0.4]
    agreement_low = ensemble.compute_agreement(predictions_disagree)
    assert agreement > agreement_low, f"Test 8b failed"
    print(f"  ✓ Agreement: high={agreement:.3f}, low={agreement_low:.3f}")
    
    # Test 9: Prediction stats
    stats = ensemble.get_prediction_stats(predictions_agree)
    assert "mean" in stats, "Test 9a failed"
    assert "std" in stats, "Test 9b failed"
    print(f"  ✓ Ensemble stats: {stats}")
    
    print("\nTesting OutOfDistributionDetector...")
    
    # Create training embeddings
    training_embs = np.random.randn(100, 64).astype(np.float32)
    detector = OutOfDistributionDetector(training_embs, threshold_percentile=95)
    
    # Test 10: In-distribution sample
    in_dist = np.random.randn(64).astype(np.float32)
    is_ood, score = detector.is_ood(in_dist)
    print(f"  ✓ In-dist sample: is_ood={is_ood}, score={score:.3f}")
    
    # Test 11: Out-of-distribution sample
    out_dist = np.random.randn(64).astype(np.float32) * 10  # Far from training
    is_ood_out, score_out = detector.is_ood(out_dist)
    assert score_out > score, f"Test 11 failed: OOD should have higher score"
    print(f"  ✓ Out-dist sample: is_ood={is_ood_out}, score={score_out:.3f}")
    
    print("\nTesting CalibrationMetrics...")
    
    # Create sample predictions and labels
    predictions = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    labels = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    
    # Test 12: ECE
    ece = CalibrationMetrics.expected_calibration_error(predictions, labels)
    assert 0 <= ece <= 1, f"Test 12 failed: {ece}"
    print(f"  ✓ ECE: {ece:.4f}")
    
    # Test 13: Reliability diagram
    diagram = CalibrationMetrics.reliability_diagram(predictions, labels)
    assert "bin_centers" in diagram, "Test 13a failed"
    assert "accuracies" in diagram, "Test 13b failed"
    print(f"  ✓ Reliability diagram: {len(diagram['bin_centers'])} bins")
    
    # Test 14: Brier score
    brier = CalibrationMetrics.brier_score(predictions, labels)
    assert 0 <= brier <= 1, f"Test 14 failed: {brier}"
    print(f"  ✓ Brier score: {brier:.4f}")
    
    print("\nTesting temperature scaling...")
    
    # Test 15: Temperature scaling
    logits = np.array([2.0, 1.0, 0.0])
    scaled = temperature_scaling(logits, temperature=2.0)
    assert len(scaled) == len(logits), "Test 15a failed"
    # Higher temperature should produce more uniform distribution
    print(f"  ✓ Temperature scaled logits: {scaled}")
    
    print("\nTesting PredictionWithConfidence...")
    
    # Test 16: Create prediction
    pred = PredictionWithConfidence(
        original="butter",
        substitute="margarine",
        score=0.92,
        confidence=0.88,
        uncertainty_type="low",
        calibrated_score=0.90,
        ensemble_agreement=0.95,
        is_out_of_distribution=False,
        explanation="High confidence prediction"
    )
    print(f"  ✓ Prediction: {pred.original} -> {pred.substitute}")
    print(f"    Score: {pred.score:.2f}, Confidence: {pred.confidence:.2f}")
    
    print("\nTesting confidence report...")
    
    # Test 17: Create report
    report = create_confidence_report(pred)
    assert len(report) > 0, "Test 17 failed"
    print(f"  ✓ Report generated ({len(report)} chars)")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
    
    print("\nKey concepts learned:")
    print("1. Confidence scores should be calibrated")
    print("2. Epistemic uncertainty reflects model knowledge")
    print("3. Ensemble disagreement indicates uncertainty")
    print("4. OOD detection warns about unfamiliar inputs")
    print("\nNext: Building the substitution API!")
