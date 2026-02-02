# -*- coding: utf-8 -*-
"""
Tests for Spend Classification Evaluation Metrics.

Tests accuracy calculation, per-category metrics, confusion matrix,
error analysis, and performance comparison (LLM vs rules).

Target: 300+ lines, 15 tests
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple


# Mock evaluation module
class SpendClassificationEvaluator:
    """Evaluator for spend classification performance."""

    def __init__(self, categories: int = 15):
        self.categories = categories
        self.predictions = []
        self.actuals = []
        self.methods = []  # Track which method was used (LLM vs rules)

    def add_prediction(self, predicted: int, actual: int, method: str = "llm"):
        """Add a single prediction."""
        if not (1 <= predicted <= self.categories):
            raise ValueError(f"Predicted category must be between 1 and {self.categories}")

        if not (1 <= actual <= self.categories):
            raise ValueError(f"Actual category must be between 1 and {self.categories}")

        self.predictions.append(predicted)
        self.actuals.append(actual)
        self.methods.append(method)

    def add_predictions(self, predictions: List[int], actuals: List[int],
                       methods: List[str] = None):
        """Add multiple predictions."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        if methods is None:
            methods = ["llm"] * len(predictions)

        for pred, act, method in zip(predictions, actuals, methods):
            self.add_prediction(pred, act, method)

    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.predictions:
            raise ValueError("No predictions to evaluate")

        correct = sum(p == a for p, a in zip(self.predictions, self.actuals))
        return correct / len(self.predictions)

    def calculate_per_category_metrics(self) -> Dict[int, Dict]:
        """Calculate precision, recall, F1 for each category."""
        if not self.predictions:
            raise ValueError("No predictions to evaluate")

        preds = np.array(self.predictions)
        acts = np.array(self.actuals)

        metrics = {}

        for category in range(1, self.categories + 1):
            tp = np.sum((preds == category) & (acts == category))
            fp = np.sum((preds == category) & (acts != category))
            fn = np.sum((preds != category) & (acts == category))
            tn = np.sum((preds != category) & (acts != category))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[category] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(np.sum(acts == category))
            }

        return metrics

    def calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix."""
        if not self.predictions:
            raise ValueError("No predictions to evaluate")

        matrix = np.zeros((self.categories, self.categories), dtype=int)

        for pred, act in zip(self.predictions, self.actuals):
            matrix[act - 1][pred - 1] += 1

        return matrix

    def analyze_errors(self) -> Dict:
        """Analyze classification errors."""
        if not self.predictions:
            raise ValueError("No predictions to evaluate")

        errors = []

        for i, (pred, act) in enumerate(zip(self.predictions, self.actuals)):
            if pred != act:
                errors.append({
                    "index": i,
                    "predicted": pred,
                    "actual": act,
                    "method": self.methods[i] if i < len(self.methods) else "unknown"
                })

        # Calculate error rate by method
        llm_errors = sum(1 for e in errors if e["method"] == "llm")
        rule_errors = sum(1 for e in errors if e["method"] == "rules")

        llm_total = sum(1 for m in self.methods if m == "llm")
        rule_total = sum(1 for m in self.methods if m == "rules")

        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(self.predictions),
            "errors_by_method": {
                "llm": {
                    "count": llm_errors,
                    "rate": llm_errors / llm_total if llm_total > 0 else 0.0
                },
                "rules": {
                    "count": rule_errors,
                    "rate": rule_errors / rule_total if rule_total > 0 else 0.0
                }
            },
            "error_details": errors[:10]  # First 10 errors
        }

    def compare_methods(self) -> Dict:
        """Compare LLM vs rules performance."""
        if not self.predictions:
            raise ValueError("No predictions to evaluate")

        llm_correct = 0
        llm_total = 0
        rule_correct = 0
        rule_total = 0

        for pred, act, method in zip(self.predictions, self.actuals, self.methods):
            if method == "llm":
                llm_total += 1
                if pred == act:
                    llm_correct += 1
            elif method == "rules":
                rule_total += 1
                if pred == act:
                    rule_correct += 1

        return {
            "llm": {
                "accuracy": llm_correct / llm_total if llm_total > 0 else 0.0,
                "total": llm_total,
                "correct": llm_correct
            },
            "rules": {
                "accuracy": rule_correct / rule_total if rule_total > 0 else 0.0,
                "total": rule_total,
                "correct": rule_correct
            }
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        return {
            "overall_accuracy": self.calculate_accuracy(),
            "per_category_metrics": self.calculate_per_category_metrics(),
            "confusion_matrix": self.calculate_confusion_matrix().tolist(),
            "error_analysis": self.analyze_errors(),
            "method_comparison": self.compare_methods(),
            "n_samples": len(self.predictions)
        }

    def reset(self):
        """Reset evaluator."""
        self.predictions = []
        self.actuals = []
        self.methods = []


# ============================================================================
# TEST SUITE
# ============================================================================

class TestSpendClassificationEvaluator:
    """Test suite for spend classification evaluation."""

    def test_add_single_prediction(self):
        """Test adding a single prediction."""
        evaluator = SpendClassificationEvaluator()
        evaluator.add_prediction(6, 6, "llm")

        assert len(evaluator.predictions) == 1
        assert evaluator.predictions[0] == 6
        assert evaluator.actuals[0] == 6

    def test_add_prediction_with_invalid_category_raises_error(self):
        """Test that invalid category raises error."""
        evaluator = SpendClassificationEvaluator()

        with pytest.raises(ValueError, match="Predicted category must be between"):
            evaluator.add_prediction(16, 1)  # Max is 15

        with pytest.raises(ValueError, match="Predicted category must be between"):
            evaluator.add_prediction(0, 1)  # Min is 1

    def test_add_multiple_predictions(self):
        """Test adding multiple predictions."""
        evaluator = SpendClassificationEvaluator()

        preds = [6, 4, 5, 1]
        acts = [6, 4, 5, 2]
        methods = ["llm", "rules", "llm", "llm"]

        evaluator.add_predictions(preds, acts, methods)

        assert len(evaluator.predictions) == 4

    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        evaluator = SpendClassificationEvaluator()

        # 3 correct out of 5 = 0.6
        evaluator.add_predictions(
            [6, 4, 5, 1, 2],
            [6, 4, 5, 2, 3]
        )

        accuracy = evaluator.calculate_accuracy()
        assert pytest.approx(accuracy, 0.01) == 0.6

    def test_calculate_per_category_metrics(self):
        """Test per-category metrics calculation."""
        evaluator = SpendClassificationEvaluator()

        # Category 6 (Business Travel): 2 correct, 1 wrong
        evaluator.add_predictions(
            [6, 6, 6, 4],
            [6, 6, 4, 4]
        )

        metrics = evaluator.calculate_per_category_metrics()

        # Category 6 metrics
        assert 6 in metrics
        assert "precision" in metrics[6]
        assert "recall" in metrics[6]
        assert "f1_score" in metrics[6]

    def test_calculate_confusion_matrix(self):
        """Test confusion matrix calculation."""
        evaluator = SpendClassificationEvaluator()

        evaluator.add_predictions(
            [6, 6, 4, 4, 5],
            [6, 4, 4, 6, 5]
        )

        cm = evaluator.calculate_confusion_matrix()

        assert cm.shape == (15, 15)
        assert cm[5, 5] == 1  # Category 6 predicted as 6 (correct)
        assert cm[5, 3] == 1  # Category 6 predicted as 4 (error)

    def test_analyze_errors(self):
        """Test error analysis."""
        evaluator = SpendClassificationEvaluator()

        evaluator.add_predictions(
            [6, 6, 4, 1],
            [6, 4, 4, 2],
            ["llm", "llm", "rules", "llm"]
        )

        error_analysis = evaluator.analyze_errors()

        assert "total_errors" in error_analysis
        assert error_analysis["total_errors"] == 2  # 2 errors out of 4
        assert "error_rate" in error_analysis

    def test_compare_methods_llm_vs_rules(self):
        """Test comparison of LLM vs rules performance."""
        evaluator = SpendClassificationEvaluator()

        # LLM: 2 correct out of 3
        # Rules: 1 correct out of 2
        evaluator.add_predictions(
            [6, 4, 5, 1, 2],
            [6, 4, 4, 1, 3],
            ["llm", "llm", "llm", "rules", "rules"]
        )

        comparison = evaluator.compare_methods()

        assert "llm" in comparison
        assert "rules" in comparison
        assert comparison["llm"]["total"] == 3
        assert comparison["rules"]["total"] == 2

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        evaluator = SpendClassificationEvaluator()

        evaluator.add_predictions(
            [6, 6, 4, 5, 1],
            [6, 4, 4, 5, 1],
            ["llm", "llm", "rules", "llm", "rules"]
        )

        report = evaluator.generate_report()

        assert "overall_accuracy" in report
        assert "per_category_metrics" in report
        assert "confusion_matrix" in report
        assert "error_analysis" in report
        assert "method_comparison" in report
        assert "n_samples" in report

    def test_reset_evaluator(self):
        """Test resetting evaluator state."""
        evaluator = SpendClassificationEvaluator()

        evaluator.add_predictions([6, 4], [6, 4])
        assert len(evaluator.predictions) == 2

        evaluator.reset()

        assert len(evaluator.predictions) == 0
        assert len(evaluator.actuals) == 0
        assert len(evaluator.methods) == 0

    def test_evaluate_with_no_data_raises_error(self):
        """Test that evaluation without data raises error."""
        evaluator = SpendClassificationEvaluator()

        with pytest.raises(ValueError, match="No predictions to evaluate"):
            evaluator.calculate_accuracy()

    def test_perfect_accuracy(self):
        """Test with perfect accuracy."""
        evaluator = SpendClassificationEvaluator()

        evaluator.add_predictions(
            [6, 4, 5, 1, 2],
            [6, 4, 5, 1, 2]
        )

        accuracy = evaluator.calculate_accuracy()
        assert accuracy == 1.0

    def test_zero_accuracy(self):
        """Test with zero accuracy."""
        evaluator = SpendClassificationEvaluator()

        evaluator.add_predictions(
            [6, 4, 5, 1, 2],
            [1, 2, 3, 4, 5]
        )

        accuracy = evaluator.calculate_accuracy()
        assert accuracy == 0.0

    def test_category_with_no_predictions(self):
        """Test metrics for category with no predictions."""
        evaluator = SpendClassificationEvaluator()

        # Only predict categories 1, 4, 6
        evaluator.add_predictions(
            [1, 4, 6],
            [1, 4, 6]
        )

        metrics = evaluator.calculate_per_category_metrics()

        # Category 2 should have zero support
        assert metrics[2]["support"] == 0
