# Problem 22: Complete Evaluation Pipeline
#
# Build an end-to-end evaluation pipeline combining all components.
#
# Example:
#   pipeline = EvaluationPipeline(config)
#   report = pipeline.run(model, test_data)
#   # Returns comprehensive evaluation report
#
# ML Relevance: A complete pipeline:
# - Standardizes evaluation across models
# - Ensures nothing is forgotten
# - Generates stakeholder-ready reports
# - Enables automated CI/CD evaluation

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1"])
    compute_confidence_intervals: bool = True
    n_bootstrap: int = 1000
    run_slice_analysis: bool = True
    run_error_analysis: bool = True
    run_fairness_analysis: bool = False
    generate_visualizations: bool = True
    output_format: str = "markdown"


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    model_name: str
    timestamp: datetime
    overall_metrics: Dict[str, float]
    confidence_intervals: Dict[str, tuple] = None
    slice_metrics: Dict[str, Dict[str, float]] = None
    error_analysis: Dict[str, Any] = None
    comparisons: Dict[str, Any] = None
    recommendations: List[str] = None


class EvaluationPipeline:
    """End-to-end evaluation pipeline."""

    def __init__(self, config: EvaluationConfig = None):
        """Initialize pipeline."""
        self.config = config or EvaluationConfig()
        # Your solution here
        pass

    def run(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        slices: Dict[str, np.ndarray] = None
    ) -> EvaluationReport:
        """
        Run complete evaluation.

        Args:
            model: Model to evaluate
            X_test, y_test: Test data
            model_name: Name for report
            slices: Optional data slices to analyze

        Returns:
            Complete evaluation report
        """
        # Your solution here
        pass

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Compare multiple models."""
        # Your solution here
        pass

    def generate_report(
        self,
        report: EvaluationReport,
        format: str = "markdown"
    ) -> str:
        """
        Generate formatted report.

        Formats: "markdown", "html", "json"
        """
        # Your solution here
        pass

    def save_report(self, report: EvaluationReport, path: str):
        """Save report to file."""
        # Your solution here
        pass


class ContinuousEvaluation:
    """Continuous evaluation for production."""

    def __init__(
        self,
        pipeline: EvaluationPipeline,
        baseline_report: EvaluationReport
    ):
        """Initialize continuous evaluation."""
        # Your solution here
        pass

    def evaluate_update(
        self,
        new_model: Any,
        test_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate a model update.

        Returns:
            {"approved": bool, "regressions": [...], "improvements": [...]}
        """
        # Your solution here
        pass

    def gate_decision(
        self,
        evaluation_result: Dict[str, Any],
        policy: str = "no_regression"
    ) -> bool:
        """
        Make go/no-go decision.

        Policies: "no_regression", "net_positive", "threshold"
        """
        # Your solution here
        pass


def quick_evaluate(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Quick evaluation with default metrics."""
    # Your solution here
    pass


# ----- Tests -----

if __name__ == "__main__":
    np.random.seed(42)

    X_test = np.random.randn(100, 10)
    y_test = (X_test[:, 0] > 0).astype(int)

    model = lambda X: (X[:, 0] > 0).astype(int)

    print("Testing evaluation pipeline...")

    # Test 1: Basic pipeline
    config = EvaluationConfig(metrics=["accuracy"])
    pipeline = EvaluationPipeline(config)
    print(f"  ✓ Pipeline initialized")

    # Test 2: Run evaluation
    report = pipeline.run(model, X_test, y_test, model_name="test_model")
    assert report.overall_metrics is not None, "Test 2 failed"
    assert "accuracy" in report.overall_metrics, "Test 2b failed"
    print(f"  ✓ Evaluation complete: {report.overall_metrics}")

    # Test 3: Generate report
    markdown = pipeline.generate_report(report, format="markdown")
    assert len(markdown) > 0, "Test 3 failed"
    print(f"  ✓ Report generated: {len(markdown)} chars")

    # Test 4: Compare models
    models = {
        "model_a": model,
        "model_b": lambda X: (X[:, 0] > 0.1).astype(int),  # Slightly different
    }
    comparison = pipeline.compare_models(models, X_test, y_test)
    assert len(comparison) > 0, "Test 4 failed"
    print(f"  ✓ Model comparison: {list(comparison.keys())}")

    # Test 5: Quick evaluate
    quick = quick_evaluate(model, X_test, y_test)
    assert "accuracy" in quick, "Test 5 failed"
    print(f"  ✓ Quick eval: {quick}")

    # Test 6: Continuous evaluation
    continuous = ContinuousEvaluation(pipeline, report)
    update_result = continuous.evaluate_update(model, {"X": X_test, "y": y_test})
    assert "approved" in update_result, "Test 6 failed"
    print(f"  ✓ Continuous eval: approved={update_result['approved']}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)

    print("\nKey concepts learned in Part 09:")
    print("1. Comprehensive metrics beyond accuracy")
    print("2. Statistical testing for significance")
    print("3. Domain-specific evaluation")
    print("4. Online evaluation and A/B testing")
    print("5. Monitoring and drift detection")
    print("6. Experiment tracking")
    print("7. Systematic iteration")
    print("8. Regression testing")
    print("\nYou're ready for Part 10: Model Serving!")
