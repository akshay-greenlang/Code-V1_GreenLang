"""
Training pipeline for entity resolution models.

This module implements the complete training workflow including
data loading, preprocessing, model training, and evaluation.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from entity_mdm.ml.config import MLConfig, TrainingConfig
from entity_mdm.ml.matching_model import MatchingModel, EntityPair
from entity_mdm.ml.evaluation import ModelEvaluator
from entity_mdm.ml.exceptions import TrainingException

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for entity matching models.

    This class handles:
    - Loading labeled training data
    - Train/validation splitting
    - Model training with checkpointing
    - Hyperparameter tuning
    - Training metrics logging
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        matching_model: Optional[MatchingModel] = None,
    ) -> None:
        """
        Initialize training pipeline.

        Args:
            config: ML configuration object
            matching_model: Matching model to train. If None, creates new one.
        """
        self.config = config or MLConfig()
        self.training_config: TrainingConfig = self.config.training
        self.matching_model = matching_model or MatchingModel(self.config)

        # Training history
        self._history: Dict[str, Any] = {}

        logger.info("Initialized TrainingPipeline")

    def load_labeled_data(
        self,
        data_path: Path,
        format: str = "csv",
    ) -> List[EntityPair]:
        """
        Load labeled entity pairs from file.

        Args:
            data_path: Path to labeled data file
            format: File format ('csv', 'json', 'parquet')

        Returns:
            List of EntityPair objects

        Raises:
            TrainingException: If data loading fails

        Expected CSV format:
            entity1_id, entity1_name, entity1_text, entity2_id, entity2_name, entity2_text, label
        """
        try:
            logger.info(f"Loading labeled data from {data_path}")

            if format == "csv":
                df = pd.read_csv(data_path)
            elif format == "json":
                df = pd.read_json(data_path)
            elif format == "parquet":
                df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Validate required columns
            required_cols = ["entity1_text", "entity2_text", "label"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Convert to EntityPair objects
            pairs = []
            for _, row in df.iterrows():
                metadata = {}
                if "entity1_id" in df.columns:
                    metadata["entity1_id"] = row["entity1_id"]
                if "entity2_id" in df.columns:
                    metadata["entity2_id"] = row["entity2_id"]

                pair = EntityPair(
                    entity1_text=str(row["entity1_text"]),
                    entity2_text=str(row["entity2_text"]),
                    label=int(row["label"]),
                    metadata=metadata,
                )
                pairs.append(pair)

            logger.info(f"Loaded {len(pairs)} labeled pairs")

            # Log class distribution
            labels = [p.label for p in pairs]
            num_positive = sum(labels)
            num_negative = len(labels) - num_positive
            logger.info(
                f"Class distribution - Matches: {num_positive}, "
                f"Non-matches: {num_negative} "
                f"(ratio: {num_positive / len(labels):.2%})"
            )

            return pairs

        except Exception as e:
            raise TrainingException(
                message=f"Failed to load labeled data: {e}",
                details={"data_path": str(data_path), "format": format},
            )

    def split_data(
        self,
        pairs: List[EntityPair],
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple[List[EntityPair], List[EntityPair], Optional[List[EntityPair]]]:
        """
        Split data into train/validation/test sets.

        Args:
            pairs: List of entity pairs
            val_split: Validation set fraction (uses config default if None)
            test_split: Test set fraction (optional)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_pairs, val_pairs, test_pairs)
            test_pairs is None if test_split not specified

        Raises:
            TrainingException: If splitting fails
        """
        try:
            val_split = val_split or self.training_config.validation_split

            if test_split:
                # Three-way split
                train_val_pairs, test_pairs = train_test_split(
                    pairs,
                    test_size=test_split,
                    random_state=random_state,
                    stratify=[p.label for p in pairs],
                )

                train_pairs, val_pairs = train_test_split(
                    train_val_pairs,
                    test_size=val_split / (1 - test_split),
                    random_state=random_state,
                    stratify=[p.label for p in train_val_pairs],
                )
            else:
                # Two-way split
                train_pairs, val_pairs = train_test_split(
                    pairs,
                    test_size=val_split,
                    random_state=random_state,
                    stratify=[p.label for p in pairs],
                )
                test_pairs = None

            logger.info(
                f"Split data - Train: {len(train_pairs)}, "
                f"Val: {len(val_pairs)}"
                + (f", Test: {len(test_pairs)}" if test_pairs else "")
            )

            return train_pairs, val_pairs, test_pairs

        except Exception as e:
            raise TrainingException(
                message=f"Failed to split data: {e}",
                details={
                    "num_pairs": len(pairs),
                    "val_split": val_split,
                    "test_split": test_split,
                },
            )

    def train(
        self,
        train_pairs: List[EntityPair],
        val_pairs: Optional[List[EntityPair]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_checkpoints: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the matching model.

        Args:
            train_pairs: Training entity pairs
            val_pairs: Validation entity pairs (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Training history dictionary

        Raises:
            TrainingException: If training fails
        """
        try:
            logger.info(
                f"Starting training with {len(train_pairs)} training pairs"
            )

            # Create checkpoint directory
            if save_checkpoints:
                self.training_config.checkpoint_dir.mkdir(
                    parents=True, exist_ok=True
                )

            # Train model
            history = self.matching_model.train(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                epochs=epochs,
                batch_size=batch_size,
            )

            # Store history
            self._history = {
                "training_date": datetime.utcnow().isoformat(),
                "num_train_pairs": len(train_pairs),
                "num_val_pairs": len(val_pairs) if val_pairs else 0,
                "epochs": len(history["train_loss"]),
                "history": history,
            }

            # Save final model
            if save_checkpoints:
                final_path = self.training_config.checkpoint_dir / "final_model.pt"
                self.matching_model.save(final_path)

            logger.info("Training completed successfully")

            return history

        except Exception as e:
            raise TrainingException(
                message=f"Training failed: {e}",
                details={
                    "num_train_pairs": len(train_pairs),
                    "num_val_pairs": len(val_pairs) if val_pairs else 0,
                },
            )

    def train_from_file(
        self,
        data_path: Path,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
        data_format: str = "csv",
        run_evaluation: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete training workflow from file.

        Args:
            data_path: Path to labeled data file
            val_split: Validation set fraction
            test_split: Test set fraction (for final evaluation)
            data_format: Input file format
            run_evaluation: Whether to run evaluation on test set

        Returns:
            Dictionary with training results and evaluation metrics

        Raises:
            TrainingException: If training fails
        """
        try:
            # Load data
            pairs = self.load_labeled_data(data_path, format=data_format)

            # Split data
            train_pairs, val_pairs, test_pairs = self.split_data(
                pairs,
                val_split=val_split,
                test_split=test_split,
            )

            # Train model
            history = self.train(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
            )

            results = {
                "training_history": history,
                "data_path": str(data_path),
                "num_train": len(train_pairs),
                "num_val": len(val_pairs),
                "num_test": len(test_pairs) if test_pairs else 0,
            }

            # Evaluate on test set if available
            if run_evaluation and test_pairs:
                logger.info("Running evaluation on test set")
                evaluator = ModelEvaluator(
                    self.matching_model,
                    self.config,
                )
                eval_results = evaluator.evaluate(test_pairs)
                results["evaluation"] = eval_results

                logger.info(
                    f"Test metrics - "
                    f"Precision: {eval_results['precision']:.3f}, "
                    f"Recall: {eval_results['recall']:.3f}, "
                    f"F1: {eval_results['f1']:.3f}"
                )

            # Save results
            results_path = self.training_config.checkpoint_dir / "training_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved training results to {results_path}")

            return results

        except Exception as e:
            raise TrainingException(
                message=f"Training workflow failed: {e}",
                details={"data_path": str(data_path)},
            )

    def tune_hyperparameters(
        self,
        train_pairs: List[EntityPair],
        val_pairs: List[EntityPair],
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.

        Args:
            train_pairs: Training pairs
            val_pairs: Validation pairs
            param_grid: Dictionary of parameters to tune
                Example: {
                    'learning_rate': [1e-5, 2e-5, 5e-5],
                    'batch_size': [16, 32],
                }

        Returns:
            Dictionary with best parameters and results

        Raises:
            TrainingException: If tuning fails
        """
        try:
            logger.info("Starting hyperparameter tuning")

            # Generate all parameter combinations
            import itertools

            keys = param_grid.keys()
            values = param_grid.values()
            combinations = [
                dict(zip(keys, v)) for v in itertools.product(*values)
            ]

            logger.info(f"Testing {len(combinations)} parameter combinations")

            best_score = 0.0
            best_params = {}
            results = []

            for i, params in enumerate(combinations):
                logger.info(
                    f"Combination {i + 1}/{len(combinations)}: {params}"
                )

                # Update config with current parameters
                # This is simplified - in practice, would need proper config updating
                epochs = params.get("epochs", self.training_config.epochs)
                batch_size = params.get("batch_size", self.config.model.batch_size)

                # Create new model for each combination
                model = MatchingModel(self.config)

                # Train
                history = model.train(
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    epochs=epochs,
                    batch_size=batch_size,
                )

                # Get validation score (use final validation accuracy)
                val_acc = history["val_acc"][-1]

                results.append(
                    {
                        "params": params,
                        "val_accuracy": val_acc,
                        "train_accuracy": history["train_acc"][-1],
                    }
                )

                if val_acc > best_score:
                    best_score = val_acc
                    best_params = params
                    # Save best model
                    best_path = (
                        self.training_config.checkpoint_dir / "best_tuned_model.pt"
                    )
                    model.save(best_path)

                logger.info(f"Validation accuracy: {val_acc:.4f}")

            logger.info(
                f"Tuning complete. Best params: {best_params} "
                f"(val_acc: {best_score:.4f})"
            )

            return {
                "best_params": best_params,
                "best_score": best_score,
                "all_results": results,
            }

        except Exception as e:
            raise TrainingException(
                message=f"Hyperparameter tuning failed: {e}",
                details={"param_grid": param_grid},
            )

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history.

        Returns:
            Training history dictionary
        """
        return self._history.copy()

    def save_training_report(
        self,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """
        Save training report to file.

        Args:
            output_path: Output file path
            format: Report format ('json' or 'html')

        Raises:
            TrainingException: If saving fails
        """
        try:
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(self._history, f, indent=2)
            elif format == "html":
                # Generate HTML report
                html = self._generate_html_report()
                with open(output_path, "w") as f:
                    f.write(html)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved training report to {output_path}")

        except Exception as e:
            raise TrainingException(
                message=f"Failed to save training report: {e}",
                details={"output_path": str(output_path), "format": format},
            )

    def _generate_html_report(self) -> str:
        """
        Generate HTML training report.

        Returns:
            HTML string
        """
        if not self._history:
            return "<html><body><p>No training history available</p></body></html>"

        history = self._history.get("history", {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .metric {{ font-weight: bold; color: #27ae60; }}
            </style>
        </head>
        <body>
            <h1>Entity Resolution Model Training Report</h1>

            <h2>Training Configuration</h2>
            <p><strong>Training Date:</strong> {self._history.get('training_date', 'N/A')}</p>
            <p><strong>Training Pairs:</strong> {self._history.get('num_train_pairs', 0)}</p>
            <p><strong>Validation Pairs:</strong> {self._history.get('num_val_pairs', 0)}</p>
            <p><strong>Epochs Completed:</strong> {self._history.get('epochs', 0)}</p>

            <h2>Training Metrics</h2>
            <table>
                <tr>
                    <th>Epoch</th>
                    <th>Train Loss</th>
                    <th>Train Accuracy</th>
                    <th>Val Loss</th>
                    <th>Val Accuracy</th>
                </tr>
        """

        for i in range(len(history.get("train_loss", []))):
            html += f"""
                <tr>
                    <td>{i + 1}</td>
                    <td>{history['train_loss'][i]:.4f}</td>
                    <td>{history['train_acc'][i]:.4f}</td>
                    <td>{history.get('val_loss', ['N/A'] * (i + 1))[i] if i < len(history.get('val_loss', [])) else 'N/A'}</td>
                    <td>{history.get('val_acc', ['N/A'] * (i + 1))[i] if i < len(history.get('val_acc', [])) else 'N/A'}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Final Metrics</h2>
        """

        if "train_acc" in history and history["train_acc"]:
            html += f'<p class="metric">Final Training Accuracy: {history["train_acc"][-1]:.4f}</p>'
        if "val_acc" in history and history["val_acc"]:
            html += f'<p class="metric">Final Validation Accuracy: {history["val_acc"][-1]:.4f}</p>'

        html += """
        </body>
        </html>
        """

        return html
