# Problem 18: Experiment Tracking
#
# Build experiment tracking for reproducible ML development.
#
# Example:
#   tracker = ExperimentTracker("substitution_model")
#   with tracker.run("baseline_v1") as run:
#       run.log_params({"lr": 0.01, "epochs": 10})
#       run.log_metric("accuracy", 0.85)
#       run.log_artifact("model.pt")
#
# ML Relevance: Experiment tracking:
# - Enables reproducibility
# - Tracks what worked and what didn't
# - Facilitates collaboration
# - Required for serious ML development

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Run:
    """Single experiment run."""
    run_id: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    start_time: datetime = None
    end_time: datetime = None
    status: str = "running"


class ExperimentTracker:
    """Track ML experiments."""

    def __init__(self, project_name: str, storage_path: str = "./experiments"):
        """
        Initialize tracker.

        Args:
            project_name: Name of the project
            storage_path: Where to store experiment data
        """
        # Your solution here
        pass

    def start_run(self, name: str, tags: Dict[str, str] = None) -> Run:
        """Start a new run."""
        # Your solution here
        pass

    def log_params(self, run_id: str, params: Dict[str, Any]):
        """Log hyperparameters."""
        # Your solution here
        pass

    def log_metric(self, run_id: str, name: str, value: float, step: int = None):
        """Log a metric value."""
        # Your solution here
        pass

    def log_artifact(self, run_id: str, artifact_path: str):
        """Log an artifact (file)."""
        # Your solution here
        pass

    def end_run(self, run_id: str, status: str = "completed"):
        """End a run."""
        # Your solution here
        pass

    def get_run(self, run_id: str) -> Run:
        """Get run by ID."""
        # Your solution here
        pass

    def list_runs(self, filter_tags: Dict[str, str] = None) -> List[Run]:
        """List all runs, optionally filtered by tags."""
        # Your solution here
        pass

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple runs.

        Returns:
            {run_id: {"params": {...}, "final_metrics": {...}}}
        """
        # Your solution here
        pass

    def best_run(self, metric: str, mode: str = "max") -> Run:
        """Find best run by metric."""
        # Your solution here
        pass


def format_run_summary(run: Run) -> str:
    """Format run for display."""
    # Your solution here
    pass


def export_runs_to_csv(runs: List[Run], output_path: str):
    """Export runs to CSV for analysis."""
    # Your solution here
    pass


# ----- Tests -----

if __name__ == "__main__":
    print("Testing experiment tracking...")

    # Test 1: Initialize tracker
    tracker = ExperimentTracker("test_project", storage_path="/tmp/experiments")
    print(f"  ✓ Tracker initialized")

    # Test 2: Start and log run
    run = tracker.start_run("baseline", tags={"model": "v1"})
    assert run.status == "running", "Test 2a failed"

    tracker.log_params(run.run_id, {"lr": 0.01, "batch_size": 32})
    tracker.log_metric(run.run_id, "loss", 0.5, step=1)
    tracker.log_metric(run.run_id, "loss", 0.3, step=2)
    tracker.log_metric(run.run_id, "accuracy", 0.85)
    tracker.end_run(run.run_id)

    stored_run = tracker.get_run(run.run_id)
    assert stored_run.status == "completed", "Test 2b failed"
    assert stored_run.params["lr"] == 0.01, "Test 2c failed"
    print(f"  ✓ Run logged: {len(stored_run.metrics)} metrics")

    # Test 3: Multiple runs
    for i, lr in enumerate([0.001, 0.01, 0.1]):
        r = tracker.start_run(f"exp_{i}")
        tracker.log_params(r.run_id, {"lr": lr})
        tracker.log_metric(r.run_id, "accuracy", 0.7 + lr)  # Fake correlation
        tracker.end_run(r.run_id)

    runs = tracker.list_runs()
    assert len(runs) >= 4, f"Test 3 failed: {len(runs)}"
    print(f"  ✓ {len(runs)} runs tracked")

    # Test 4: Best run
    best = tracker.best_run("accuracy", mode="max")
    assert best is not None, "Test 4 failed"
    print(f"  ✓ Best run: {best.name} with accuracy {best.metrics.get('accuracy', [None])[-1]}")

    # Test 5: Compare runs
    run_ids = [r.run_id for r in runs[:2]]
    comparison = tracker.compare_runs(run_ids)
    assert len(comparison) == 2, "Test 5 failed"
    print(f"  ✓ Comparison: {list(comparison.keys())}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
