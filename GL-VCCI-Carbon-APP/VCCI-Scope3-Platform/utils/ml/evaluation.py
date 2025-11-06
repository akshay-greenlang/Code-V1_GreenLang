# GL-VCCI ML Module - Evaluation
# Spend Classification ML System - Evaluation Framework

"""
Evaluation Framework
===================

Classification evaluation framework for measuring accuracy and performance.

Metrics:
-------
- Accuracy: Overall classification accuracy
- Precision: Per-category precision
- Recall: Per-category recall
- F1-score: Per-category F1-score
- Confusion Matrix: 15x15 category confusion matrix
- Error Analysis: Common misclassifications

Features:
--------
- Comprehensive metrics calculation
- Confusion matrix visualization
- Per-category performance analysis
- Error analysis and reporting
- Comparison between LLM and rules performance
- Export results to CSV/JSON

Usage:
------
```python
from utils.ml.evaluation import ModelEvaluator
from utils.ml.spend_classification import SpendClassifier
from utils.ml.training_data import TrainingDataLoader

# Load test data
loader = TrainingDataLoader()
test_dataset = loader.load_csv("data/test/spend_labels.csv")

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate classifier
classifier = SpendClassifier(config)
results = await evaluator.evaluate_classifier(classifier, test_dataset)

# Print metrics
print(f"Overall Accuracy: {results.overall_accuracy:.2%}")
print(f"Macro F1-score: {results.macro_f1:.2f}")

# Get confusion matrix
confusion = results.confusion_matrix
print(confusion)

# Export results
evaluator.export_results(results, "evaluation_results.json")
```
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import MLConfig, Scope3Category
from .spend_classification import ClassificationResult, SpendClassifier
from .training_data import TrainingDataset, TrainingExample

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Models
# ============================================================================

class CategoryMetrics(BaseModel):
    """
    Per-category evaluation metrics.

    Attributes:
        category: Scope 3 category
        category_name: Human-readable category name
        precision: Precision score (0.0-1.0)
        recall: Recall score (0.0-1.0)
        f1_score: F1-score (0.0-1.0)
        support: Number of true samples
        correct: Number of correct predictions
        incorrect: Number of incorrect predictions
    """
    category: str = Field(description="Scope 3 category")
    category_name: str = Field(description="Human-readable category name")
    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(ge=0.0, le=1.0, description="F1-score")
    support: int = Field(ge=0, description="Number of true samples")
    correct: int = Field(ge=0, description="Correct predictions")
    incorrect: int = Field(ge=0, description="Incorrect predictions")


class EvaluationResults(BaseModel):
    """
    Complete evaluation results.

    Attributes:
        overall_accuracy: Overall accuracy (0.0-1.0)
        macro_precision: Macro-averaged precision
        macro_recall: Macro-averaged recall
        macro_f1: Macro-averaged F1-score
        weighted_precision: Weighted precision
        weighted_recall: Weighted recall
        weighted_f1: Weighted F1-score
        category_metrics: Per-category metrics
        confusion_matrix: Confusion matrix (list of lists)
        total_samples: Total test samples
        correct_predictions: Total correct predictions
        incorrect_predictions: Total incorrect predictions
        evaluation_time_seconds: Evaluation time
        metadata: Additional metadata
    """
    overall_accuracy: float = Field(ge=0.0, le=1.0, description="Overall accuracy")
    macro_precision: float = Field(ge=0.0, le=1.0, description="Macro precision")
    macro_recall: float = Field(ge=0.0, le=1.0, description="Macro recall")
    macro_f1: float = Field(ge=0.0, le=1.0, description="Macro F1-score")
    weighted_precision: float = Field(ge=0.0, le=1.0, description="Weighted precision")
    weighted_recall: float = Field(ge=0.0, le=1.0, description="Weighted recall")
    weighted_f1: float = Field(ge=0.0, le=1.0, description="Weighted F1-score")
    category_metrics: List[CategoryMetrics] = Field(description="Per-category metrics")
    confusion_matrix: List[List[int]] = Field(description="Confusion matrix")
    total_samples: int = Field(ge=0, description="Total test samples")
    correct_predictions: int = Field(ge=0, description="Correct predictions")
    incorrect_predictions: int = Field(ge=0, description="Incorrect predictions")
    evaluation_time_seconds: float = Field(description="Evaluation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ErrorAnalysis(BaseModel):
    """
    Error analysis report.

    Attributes:
        total_errors: Total number of errors
        error_rate: Error rate (0.0-1.0)
        common_errors: Most common misclassifications [(true, pred, count), ...]
        category_errors: Errors per category {category: count}
        low_confidence_errors: Errors with low confidence
    """
    total_errors: int = Field(ge=0, description="Total errors")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate")
    common_errors: List[Tuple[str, str, int]] = Field(
        description="Common misclassifications [(true, pred, count), ...]"
    )
    category_errors: Dict[str, int] = Field(description="Errors per category")
    low_confidence_errors: int = Field(description="Errors with low confidence")


# ============================================================================
# Model Evaluator
# ============================================================================

class ModelEvaluator:
    """
    Model evaluation framework.

    Evaluates classification performance using standard metrics and provides
    detailed error analysis.
    """

    def __init__(self, config: Optional[MLConfig] = None):
        """
        Initialize model evaluator.

        Args:
            config: ML configuration (uses default if not provided)
        """
        if config is None:
            from .config import load_config
            config = load_config()

        self.config = config
        logger.info("Initialized model evaluator")

    async def evaluate_classifier(
        self,
        classifier: SpendClassifier,
        test_dataset: TrainingDataset,
        batch_size: int = 10
    ) -> EvaluationResults:
        """
        Evaluate classifier on test dataset.

        Args:
            classifier: Spend classifier to evaluate
            test_dataset: Test dataset
            batch_size: Batch size for evaluation

        Returns:
            Evaluation results
        """
        start_time = datetime.utcnow()

        logger.info(f"Evaluating classifier on {len(test_dataset)} test samples")

        # Get predictions
        y_true = []
        y_pred = []
        predictions = []

        # Process in batches
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset.examples[i:i + batch_size]

            # Classify batch
            items = [
                {
                    "description": ex.description,
                    "amount": ex.amount,
                    "supplier": ex.supplier
                }
                for ex in batch
            ]

            batch_results = await classifier.classify_batch(items, batch_size=batch_size)

            # Collect results
            for example, result in zip(batch, batch_results):
                y_true.append(example.category)
                y_pred.append(result.category)
                predictions.append((example, result))

        # Calculate metrics
        results = self._calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            predictions=predictions
        )

        # Add metadata
        end_time = datetime.utcnow()
        results.evaluation_time_seconds = (end_time - start_time).total_seconds()
        results.metadata.update({
            "test_dataset_name": test_dataset.name,
            "test_dataset_size": len(test_dataset),
            "classifier_config": {
                "use_llm_primary": self.config.classification.use_llm_primary,
                "use_rules_fallback": self.config.classification.use_rules_fallback,
                "confidence_threshold": self.config.classification.confidence_threshold
            },
            "evaluation_timestamp": end_time.isoformat()
        })

        logger.info(
            f"Evaluation complete: accuracy={results.overall_accuracy:.2%}, "
            f"macro_f1={results.macro_f1:.2f}, time={results.evaluation_time_seconds:.1f}s"
        )

        return results

    def _calculate_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        predictions: List[Tuple[TrainingExample, ClassificationResult]]
    ) -> EvaluationResults:
        """Calculate evaluation metrics."""
        # Get all categories
        categories = Scope3Category.get_all_categories()

        # Overall metrics
        overall_accuracy = accuracy_score(y_true, y_pred)

        # Macro metrics (unweighted average)
        macro_precision = precision_score(
            y_true, y_pred, labels=categories, average="macro", zero_division=0
        )
        macro_recall = recall_score(
            y_true, y_pred, labels=categories, average="macro", zero_division=0
        )
        macro_f1 = f1_score(
            y_true, y_pred, labels=categories, average="macro", zero_division=0
        )

        # Weighted metrics (weighted by support)
        weighted_precision = precision_score(
            y_true, y_pred, labels=categories, average="weighted", zero_division=0
        )
        weighted_recall = recall_score(
            y_true, y_pred, labels=categories, average="weighted", zero_division=0
        )
        weighted_f1 = f1_score(
            y_true, y_pred, labels=categories, average="weighted", zero_division=0
        )

        # Per-category metrics
        category_metrics = self._calculate_category_metrics(y_true, y_pred, categories)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=categories)

        # Counts
        total_samples = len(y_true)
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        incorrect_predictions = total_samples - correct_predictions

        return EvaluationResults(
            overall_accuracy=overall_accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_precision=weighted_precision,
            weighted_recall=weighted_recall,
            weighted_f1=weighted_f1,
            category_metrics=category_metrics,
            confusion_matrix=conf_matrix.tolist(),
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            incorrect_predictions=incorrect_predictions,
            evaluation_time_seconds=0.0  # Will be set by caller
        )

    def _calculate_category_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        categories: List[str]
    ) -> List[CategoryMetrics]:
        """Calculate per-category metrics."""
        # Get classification report
        report = classification_report(
            y_true,
            y_pred,
            labels=categories,
            output_dict=True,
            zero_division=0
        )

        category_metrics = []
        for category in categories:
            if category in report:
                cat_report = report[category]

                # Count correct/incorrect
                correct = sum(
                    1 for t, p in zip(y_true, y_pred)
                    if t == category and p == category
                )
                support = int(cat_report["support"])
                incorrect = support - correct

                metrics = CategoryMetrics(
                    category=category,
                    category_name=Scope3Category.get_category_name(category),
                    precision=cat_report["precision"],
                    recall=cat_report["recall"],
                    f1_score=cat_report["f1-score"],
                    support=support,
                    correct=correct,
                    incorrect=incorrect
                )
                category_metrics.append(metrics)

        return category_metrics

    def analyze_errors(
        self,
        y_true: List[str],
        y_pred: List[str],
        predictions: List[Tuple[TrainingExample, ClassificationResult]],
        top_n: int = 10
    ) -> ErrorAnalysis:
        """
        Analyze classification errors.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            predictions: List of (example, result) tuples
            top_n: Number of top errors to return

        Returns:
            Error analysis
        """
        # Count errors
        total_errors = sum(1 for t, p in zip(y_true, y_pred) if t != p)
        error_rate = total_errors / len(y_true) if y_true else 0.0

        # Common misclassifications
        error_pairs = {}
        for t, p in zip(y_true, y_pred):
            if t != p:
                pair = (t, p)
                error_pairs[pair] = error_pairs.get(pair, 0) + 1

        common_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        common_errors = [(t, p, count) for (t, p), count in common_errors]

        # Errors per category
        category_errors = {}
        for t, p in zip(y_true, y_pred):
            if t != p:
                category_errors[t] = category_errors.get(t, 0) + 1

        # Low confidence errors
        low_confidence_errors = sum(
            1 for (example, result), t, p in zip(predictions, y_true, y_pred)
            if t != p and result.confidence < 0.5
        )

        return ErrorAnalysis(
            total_errors=total_errors,
            error_rate=error_rate,
            common_errors=common_errors,
            category_errors=category_errors,
            low_confidence_errors=low_confidence_errors
        )

    def compare_methods(
        self,
        y_true: List[str],
        llm_predictions: List[str],
        rules_predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Compare LLM vs rules performance.

        Args:
            y_true: True labels
            llm_predictions: LLM predictions
            rules_predictions: Rules predictions

        Returns:
            Comparison results
        """
        llm_accuracy = accuracy_score(y_true, llm_predictions)
        rules_accuracy = accuracy_score(y_true, rules_predictions)

        llm_f1 = f1_score(
            y_true,
            llm_predictions,
            labels=Scope3Category.get_all_categories(),
            average="macro",
            zero_division=0
        )

        rules_f1 = f1_score(
            y_true,
            rules_predictions,
            labels=Scope3Category.get_all_categories(),
            average="macro",
            zero_division=0
        )

        # Agreement
        agreement = sum(
            1 for l, r in zip(llm_predictions, rules_predictions) if l == r
        ) / len(y_true)

        return {
            "llm_accuracy": llm_accuracy,
            "rules_accuracy": rules_accuracy,
            "llm_f1_macro": llm_f1,
            "rules_f1_macro": rules_f1,
            "accuracy_difference": llm_accuracy - rules_accuracy,
            "f1_difference": llm_f1 - rules_f1,
            "llm_rules_agreement": agreement,
            "better_method": "llm" if llm_accuracy > rules_accuracy else "rules"
        }

    def export_results(
        self,
        results: EvaluationResults,
        output_path: str,
        format: str = "json"
    ):
        """
        Export evaluation results.

        Args:
            results: Evaluation results
            output_path: Output file path
            format: Output format (json, csv)
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results.model_dump(), f, indent=2, default=str)
            logger.info(f"Exported results to JSON: {output_path}")

        elif format == "csv":
            # Export category metrics to CSV
            df = pd.DataFrame([
                {
                    "category": m.category,
                    "category_name": m.category_name,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "support": m.support,
                    "correct": m.correct,
                    "incorrect": m.incorrect
                }
                for m in results.category_metrics
            ])
            df.to_csv(output_path, index=False)
            logger.info(f"Exported results to CSV: {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def print_summary(self, results: EvaluationResults):
        """
        Print evaluation summary.

        Args:
            results: Evaluation results
        """
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {results.overall_accuracy:.2%}")
        print(f"  Macro Precision:    {results.macro_precision:.2%}")
        print(f"  Macro Recall:       {results.macro_recall:.2%}")
        print(f"  Macro F1-score:     {results.macro_f1:.4f}")
        print(f"  Weighted F1-score:  {results.weighted_f1:.4f}")

        print(f"\nSample Counts:")
        print(f"  Total samples:      {results.total_samples}")
        print(f"  Correct:            {results.correct_predictions} ({results.overall_accuracy:.1%})")
        print(f"  Incorrect:          {results.incorrect_predictions}")

        print(f"\nTop 5 Categories by F1-score:")
        sorted_metrics = sorted(
            results.category_metrics,
            key=lambda m: m.f1_score,
            reverse=True
        )[:5]

        for m in sorted_metrics:
            print(
                f"  {m.category_name[:40]:40s}: "
                f"F1={m.f1_score:.3f}, P={m.precision:.3f}, R={m.recall:.3f}, n={m.support}"
            )

        print(f"\nEvaluation Time: {results.evaluation_time_seconds:.1f}s")
        print("=" * 80 + "\n")
