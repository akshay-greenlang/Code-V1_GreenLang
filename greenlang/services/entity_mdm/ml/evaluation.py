# -*- coding: utf-8 -*-
"""
Model evaluation framework for entity resolution.

This module implements comprehensive evaluation including metrics,
confusion matrices, ROC curves, and performance reporting.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import json
import numpy as np
from sklearn.metrics import (
from greenlang.determinism import DeterministicClock
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from entity_mdm.ml.config import MLConfig
from entity_mdm.ml.matching_model import MatchingModel, EntityPair
from entity_mdm.ml.exceptions import EvaluationException

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.

    This class provides:
    - Standard classification metrics
    - Confusion matrix analysis
    - ROC and PR curve generation
    - Performance reporting (HTML/JSON)
    """

    def __init__(
        self,
        matching_model: MatchingModel,
        config: Optional[MLConfig] = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            matching_model: Trained matching model to evaluate
            config: ML configuration object
        """
        self.matching_model = matching_model
        self.config = config or MLConfig()

        # Evaluation results
        self._results: Dict[str, Any] = {}

        logger.info("Initialized ModelEvaluator")

    def evaluate(
        self,
        test_pairs: List[EntityPair],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            test_pairs: Test entity pairs with ground truth labels
            threshold: Classification threshold for binary predictions

        Returns:
            Dictionary with evaluation metrics

        Raises:
            EvaluationException: If evaluation fails
        """
        try:
            logger.info(f"Evaluating model on {len(test_pairs)} test pairs")

            # Get predictions
            pairs = [(p.entity1_text, p.entity2_text) for p in test_pairs]
            predictions = self.matching_model.predict_batch(pairs)

            # Extract labels and scores
            y_true = np.array([p.label for p in test_pairs])
            y_pred_proba = np.array([conf if pred == 1 else (1 - conf)
                                     for pred, conf in predictions])
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)

            # Store results
            self._results = {
                "evaluation_date": DeterministicClock.utcnow().isoformat(),
                "num_test_pairs": len(test_pairs),
                "threshold": threshold,
                "metrics": metrics,
            }

            logger.info(
                f"Evaluation complete - "
                f"Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}, "
                f"F1: {metrics['f1']:.3f}"
            )

            return self._results

        except Exception as e:
            raise EvaluationException(
                message=f"Evaluation failed: {e}",
                details={"num_test_pairs": len(test_pairs)},
            )

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities

        Returns:
            Dictionary with metrics
        """
        try:
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            metrics.update(
                {
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                }
            )

            # ROC AUC
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                metrics["roc_auc"] = float(auc(fpr, tpr))

                # PR AUC
                precision_curve, recall_curve, _ = precision_recall_curve(
                    y_true, y_pred_proba
                )
                metrics["pr_auc"] = float(auc(recall_curve, precision_curve))
            else:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

            return metrics

        except Exception as e:
            raise EvaluationException(
                metric="all",
                message=f"Failed to calculate metrics: {e}",
            )

    def generate_confusion_matrix(
        self,
        test_pairs: List[EntityPair],
        output_path: Optional[Path] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate and optionally save confusion matrix.

        Args:
            test_pairs: Test entity pairs
            output_path: Path to save confusion matrix plot (optional)
            threshold: Classification threshold

        Returns:
            Confusion matrix as numpy array

        Raises:
            EvaluationException: If generation fails
        """
        try:
            # Get predictions
            pairs = [(p.entity1_text, p.entity2_text) for p in test_pairs]
            predictions = self.matching_model.predict_batch(pairs)

            y_true = np.array([p.label for p in test_pairs])
            y_pred_proba = np.array([conf if pred == 1 else (1 - conf)
                                     for pred, conf in predictions])
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Plot if output path provided
            if output_path:
                self._plot_confusion_matrix(cm, output_path)

            return cm

        except Exception as e:
            raise EvaluationException(
                metric="confusion_matrix",
                message=f"Failed to generate confusion matrix: {e}",
            )

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            output_path: Output file path
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Labels
        classes = ["No Match", "Match"]
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                )

        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix to {output_path}")

    def generate_roc_curve(
        self,
        test_pairs: List[EntityPair],
        output_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate ROC curve.

        Args:
            test_pairs: Test entity pairs
            output_path: Path to save ROC curve plot

        Returns:
            Tuple of (fpr, tpr, auc_score)

        Raises:
            EvaluationException: If generation fails
        """
        try:
            # Get predictions
            pairs = [(p.entity1_text, p.entity2_text) for p in test_pairs]
            predictions = self.matching_model.predict_batch(pairs)

            y_true = np.array([p.label for p in test_pairs])
            y_pred_proba = np.array([conf if pred == 1 else (1 - conf)
                                     for pred, conf in predictions])

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.3f})",
            )
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic (ROC) Curve")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved ROC curve to {output_path}")

            return fpr, tpr, roc_auc

        except Exception as e:
            raise EvaluationException(
                metric="roc_curve",
                message=f"Failed to generate ROC curve: {e}",
            )

    def generate_precision_recall_curve(
        self,
        test_pairs: List[EntityPair],
        output_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate precision-recall curve.

        Args:
            test_pairs: Test entity pairs
            output_path: Path to save PR curve plot

        Returns:
            Tuple of (precision, recall, auc_score)

        Raises:
            EvaluationException: If generation fails
        """
        try:
            # Get predictions
            pairs = [(p.entity1_text, p.entity2_text) for p in test_pairs]
            predictions = self.matching_model.predict_batch(pairs)

            y_true = np.array([p.label for p in test_pairs])
            y_pred_proba = np.array([conf if pred == 1 else (1 - conf)
                                     for pred, conf in predictions])

            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                recall,
                precision,
                color="darkgreen",
                lw=2,
                label=f"PR curve (AUC = {pr_auc:.3f})",
            )
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend(loc="lower left")
            ax.grid(alpha=0.3)

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved PR curve to {output_path}")

            return precision, recall, pr_auc

        except Exception as e:
            raise EvaluationException(
                metric="precision_recall_curve",
                message=f"Failed to generate PR curve: {e}",
            )

    def save_report(
        self,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """
        Save evaluation report.

        Args:
            output_path: Output file path
            format: Report format ('json' or 'html')

        Raises:
            EvaluationException: If saving fails
        """
        try:
            if not self._results:
                raise ValueError("No evaluation results available. Run evaluate() first.")

            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(self._results, f, indent=2)
            elif format == "html":
                html = self._generate_html_report()
                with open(output_path, "w") as f:
                    f.write(html)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved evaluation report to {output_path}")

        except Exception as e:
            raise EvaluationException(
                message=f"Failed to save report: {e}",
                details={"output_path": str(output_path), "format": format},
            )

    def _generate_html_report(self) -> str:
        """
        Generate HTML evaluation report.

        Returns:
            HTML string
        """
        metrics = self._results.get("metrics", {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    min-width: 150px;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #7f8c8d;
                    text-transform: uppercase;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .bad {{ color: #e74c3c; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <h1>Entity Resolution Model Evaluation Report</h1>

            <h2>Overview</h2>
            <p><strong>Evaluation Date:</strong> {self._results.get('evaluation_date', 'N/A')}</p>
            <p><strong>Test Pairs:</strong> {self._results.get('num_test_pairs', 0)}</p>
            <p><strong>Classification Threshold:</strong> {self._results.get('threshold', 0.5)}</p>

            <h2>Performance Metrics</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value {'good' if metrics.get('accuracy', 0) >= 0.90 else 'warning' if metrics.get('accuracy', 0) >= 0.80 else 'bad'}">
                        {metrics.get('accuracy', 0):.3f}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value {'good' if metrics.get('precision', 0) >= 0.95 else 'warning' if metrics.get('precision', 0) >= 0.85 else 'bad'}">
                        {metrics.get('precision', 0):.3f}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value {'good' if metrics.get('recall', 0) >= 0.90 else 'warning' if metrics.get('recall', 0) >= 0.80 else 'bad'}">
                        {metrics.get('recall', 0):.3f}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value {'good' if metrics.get('f1', 0) >= 0.90 else 'warning' if metrics.get('f1', 0) >= 0.80 else 'bad'}">
                        {metrics.get('f1', 0):.3f}
                    </div>
                </div>
            </div>

            <h2>Confusion Matrix</h2>
            <table style="width: 400px;">
                <tr>
                    <th></th>
                    <th>Predicted: No Match</th>
                    <th>Predicted: Match</th>
                </tr>
                <tr>
                    <th>Actual: No Match</th>
                    <td>{metrics.get('true_negatives', 0)}</td>
                    <td>{metrics.get('false_positives', 0)}</td>
                </tr>
                <tr>
                    <th>Actual: Match</th>
                    <td>{metrics.get('false_negatives', 0)}</td>
                    <td>{metrics.get('true_positives', 0)}</td>
                </tr>
            </table>

            <h2>Additional Metrics</h2>
            <table style="width: 400px;">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Specificity</td>
                    <td>{metrics.get('specificity', 0):.3f}</td>
                </tr>
                <tr>
                    <td>ROC AUC</td>
                    <td>{metrics.get('roc_auc', 0):.3f}</td>
                </tr>
                <tr>
                    <td>PR AUC</td>
                    <td>{metrics.get('pr_auc', 0):.3f}</td>
                </tr>
            </table>

            <h2>Target Performance</h2>
            <p>The model targets ≥95% auto-match rate at 95% precision.</p>
            <p><strong>Current Precision:</strong> {metrics.get('precision', 0):.1%}
               {'✓ Target met' if metrics.get('precision', 0) >= 0.95 else '✗ Below target'}</p>
        </body>
        </html>
        """

        return html

    def get_results(self) -> Dict[str, Any]:
        """
        Get evaluation results.

        Returns:
            Dictionary with evaluation results
        """
        return self._results.copy()
