# Problem 17: Model Monitoring and Drift Detection
#
# Implement monitoring for deployed models to detect degradation.
#
# Example:
#   monitor = ModelMonitor(baseline_metrics)
#   alert = monitor.check(current_metrics)
#   # Returns: {"drift_detected": True, "metric": "accuracy", "severity": "high"}
#
# ML Relevance: Monitoring catches:
# - Data drift (input distribution changes)
# - Concept drift (relationship changes)
# - Model degradation over time
# - Data quality issues

from typing import List, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass
from collections import deque


@dataclass
class Alert:
    """Monitoring alert."""
    metric: str
    severity: str  # "low", "medium", "high"
    current_value: float
    threshold: float
    message: str


def ks_test(sample1: np.ndarray, sample2: np.ndarray) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov test for distribution comparison.

    Returns:
        {"statistic": x, "p_value": y}
    """
    # Your solution here
    pass


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index for drift detection.

    PSI < 0.1: no significant change
    PSI 0.1-0.25: moderate change
    PSI > 0.25: significant change
    """
    # Your solution here
    pass


def detect_outliers(
    values: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in metric values.

    Methods: "iqr", "zscore", "mad"

    Returns:
        Boolean mask of outliers
    """
    # Your solution here
    pass


class ModelMonitor:
    """Production model monitoring."""

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        thresholds: Dict[str, float] = None,
        window_size: int = 100
    ):
        """
        Initialize monitor.

        Args:
            baseline_metrics: Expected metric values
            thresholds: Alert thresholds per metric
            window_size: Rolling window for comparison
        """
        # Your solution here
        pass

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: any,
        ground_truth: any = None
    ):
        """Log a prediction for monitoring."""
        # Your solution here
        pass

    def check_metrics(self) -> List[Alert]:
        """
        Check current metrics against baseline.

        Returns:
            List of alerts if any thresholds exceeded
        """
        # Your solution here
        pass

    def check_data_drift(
        self,
        current_features: np.ndarray,
        baseline_features: np.ndarray
    ) -> Dict[str, float]:
        """
        Check for data drift using PSI.

        Returns:
            {feature_name: psi_value}
        """
        # Your solution here
        pass

    def get_health_status(self) -> Dict[str, any]:
        """
        Overall model health summary.

        Returns:
            {"status": "healthy/degraded/critical", "alerts": [...], "metrics": {...}}
        """
        # Your solution here
        pass


class DriftDetector:
    """Statistical drift detection."""

    def __init__(self, sensitivity: float = 0.05):
        """Initialize detector."""
        self.sensitivity = sensitivity

    def fit(self, baseline_data: np.ndarray):
        """Fit on baseline data."""
        # Your solution here
        pass

    def detect(self, current_data: np.ndarray) -> Dict[str, any]:
        """
        Detect drift from baseline.

        Returns:
            {"drift_detected": bool, "p_value": x, "drift_score": y}
        """
        # Your solution here
        pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    print("Testing model monitoring...")

    # Test 1: KS test - same distribution
    sample1 = np.random.normal(0, 1, 100)
    sample2 = np.random.normal(0, 1, 100)
    result = ks_test(sample1, sample2)
    assert result["p_value"] > 0.05, "Test 1a failed: same dist should not reject"

    # Different distribution
    sample2 = np.random.normal(1, 1, 100)  # Shifted mean
    result = ks_test(sample1, sample2)
    assert result["p_value"] < 0.05, "Test 1b failed: different dist should reject"
    print(f"  ✓ KS test detects distribution shift")

    # Test 2: PSI
    expected = np.random.normal(0, 1, 1000)
    actual_same = np.random.normal(0, 1, 1000)
    actual_drift = np.random.normal(0.5, 1.2, 1000)

    psi_same = psi(expected, actual_same)
    psi_drift = psi(expected, actual_drift)
    assert psi_same < psi_drift, "Test 2 failed: drift should have higher PSI"
    print(f"  ✓ PSI: no drift={psi_same:.4f}, with drift={psi_drift:.4f}")

    # Test 3: ModelMonitor
    baseline = {"accuracy": 0.90, "latency_p50": 50}
    monitor = ModelMonitor(baseline, thresholds={"accuracy": 0.05})
    print(f"  ✓ ModelMonitor initialized")

    # Test 4: Outlier detection
    values = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
    outliers = detect_outliers(values)
    assert outliers[-1] == True, "Test 4 failed: should detect 100 as outlier"
    print(f"  ✓ Outlier detection: {outliers}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
