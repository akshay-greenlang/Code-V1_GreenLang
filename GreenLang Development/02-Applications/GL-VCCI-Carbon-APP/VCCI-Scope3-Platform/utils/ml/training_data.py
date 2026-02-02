# -*- coding: utf-8 -*-
# GL-VCCI ML Module - Training Data
# Spend Classification ML System - Training Data Management

"""
Training Data Management
=========================

Training data loading, preprocessing, and validation for Scope 3 classification.

Features:
--------
- CSV/JSON/Excel data loading
- Label validation (15 Scope 3 categories)
- Data cleaning and preprocessing
- Train/validation/test splitting
- Data augmentation for imbalanced categories
- Statistics and reporting

Data Format:
-----------
Training data should have the following columns:
- description: Procurement description (required)
- category: Scope 3 category label (required)
- amount: Spend amount (optional)
- supplier: Supplier name (optional)
- confidence: Label confidence 0-1 (optional, default: 1.0)

Example CSV:
```csv
description,category,amount,supplier,confidence
"Office furniture purchase",category_1_purchased_goods_services,5000.00,IKEA,1.0
"Flight to NYC",category_6_business_travel,450.00,Delta,1.0
"Electricity bill",category_3_fuel_energy_related,1200.00,ConEd,1.0
```

Usage:
------
```python
from utils.ml.training_data import TrainingDataLoader, TrainingDataset
from greenlang.determinism import deterministic_random

# Load training data
loader = TrainingDataLoader()
dataset = loader.load_csv("data/training/spend_labels.csv")

# Validate data
loader.validate_dataset(dataset)

# Split data
train, val, test = loader.split_dataset(dataset, train=0.7, val=0.15, test=0.15)

# Get statistics
stats = loader.get_statistics(dataset)
print(f"Total samples: {stats['total_samples']}")
print(f"Category distribution: {stats['category_distribution']}")

# Augment imbalanced categories
augmented = loader.augment_dataset(dataset, min_samples_per_category=100)
```
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .config import Scope3Category
from .exceptions import (
    InsufficientDataException,
    InvalidLabelException,
    TrainingDataException,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class TrainingExample(BaseModel):
    """
    Single training example.

    Attributes:
        description: Procurement description
        category: Scope 3 category label
        amount: Spend amount (optional)
        supplier: Supplier name (optional)
        confidence: Label confidence (0.0-1.0)
        metadata: Additional metadata
    """
    description: str = Field(description="Procurement description")
    category: str = Field(description="Scope 3 category label")
    amount: Optional[float] = Field(default=None, ge=0, description="Spend amount")
    supplier: Optional[str] = Field(default=None, description="Supplier name")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Label confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category is valid Scope 3 category."""
        valid_categories = Scope3Category.get_all_categories()
        if v not in valid_categories:
            raise InvalidLabelException(
                message=f"Invalid category: {v}",
                label=v,
                valid_labels=valid_categories
            )
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is not empty."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()


class TrainingDataset(BaseModel):
    """
    Training dataset.

    Attributes:
        examples: List of training examples
        name: Dataset name
        version: Dataset version
        metadata: Dataset metadata
    """
    examples: List[TrainingExample] = Field(description="Training examples")
    name: str = Field(default="training_dataset", description="Dataset name")
    version: str = Field(default="1.0", description="Dataset version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)

    def get_category_distribution(self) -> Dict[str, int]:
        """Get category distribution."""
        distribution = {}
        for example in self.examples:
            category = example.category
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    def filter_by_category(self, category: str) -> "TrainingDataset":
        """Filter dataset by category."""
        filtered_examples = [
            example for example in self.examples
            if example.category == category
        ]
        return TrainingDataset(
            examples=filtered_examples,
            name=f"{self.name}_filtered_{category}",
            version=self.version,
            metadata={**self.metadata, "filtered_category": category}
        )


# ============================================================================
# Training Data Loader
# ============================================================================

class TrainingDataLoader:
    """
    Training data loader and preprocessor.

    Loads, validates, and preprocesses training data for Scope 3 classification.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize training data loader.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)

        logger.info(f"Initialized training data loader (seed={random_seed})")

    def load_csv(
        self,
        file_path: str,
        description_col: str = "description",
        category_col: str = "category",
        amount_col: Optional[str] = "amount",
        supplier_col: Optional[str] = "supplier",
        confidence_col: Optional[str] = "confidence"
    ) -> TrainingDataset:
        """
        Load training data from CSV file.

        Args:
            file_path: Path to CSV file
            description_col: Description column name
            category_col: Category column name
            amount_col: Amount column name (optional)
            supplier_col: Supplier column name (optional)
            confidence_col: Confidence column name (optional)

        Returns:
            Training dataset

        Raises:
            TrainingDataException: If loading fails
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV: {file_path} ({len(df)} rows)")

            return self._dataframe_to_dataset(
                df,
                description_col=description_col,
                category_col=category_col,
                amount_col=amount_col,
                supplier_col=supplier_col,
                confidence_col=confidence_col,
                dataset_name=Path(file_path).stem
            )

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}", exc_info=True)
            raise TrainingDataException(
                message=f"Failed to load training data from CSV: {str(e)}",
                details={"file_path": file_path},
                original_error=e
            )

    def load_json(self, file_path: str) -> TrainingDataset:
        """
        Load training data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Training dataset

        Raises:
            TrainingDataException: If loading fails
        """
        try:
            df = pd.read_json(file_path)
            logger.info(f"Loaded JSON: {file_path} ({len(df)} rows)")

            return self._dataframe_to_dataset(
                df,
                dataset_name=Path(file_path).stem
            )

        except Exception as e:
            logger.error(f"Failed to load JSON: {e}", exc_info=True)
            raise TrainingDataException(
                message=f"Failed to load training data from JSON: {str(e)}",
                details={"file_path": file_path},
                original_error=e
            )

    def load_excel(
        self,
        file_path: str,
        sheet_name: str = "training_data"
    ) -> TrainingDataset:
        """
        Load training data from Excel file.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name

        Returns:
            Training dataset

        Raises:
            TrainingDataException: If loading fails
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Loaded Excel: {file_path}, sheet={sheet_name} ({len(df)} rows)")

            return self._dataframe_to_dataset(
                df,
                dataset_name=f"{Path(file_path).stem}_{sheet_name}"
            )

        except Exception as e:
            logger.error(f"Failed to load Excel: {e}", exc_info=True)
            raise TrainingDataException(
                message=f"Failed to load training data from Excel: {str(e)}",
                details={"file_path": file_path, "sheet_name": sheet_name},
                original_error=e
            )

    def _dataframe_to_dataset(
        self,
        df: pd.DataFrame,
        description_col: str = "description",
        category_col: str = "category",
        amount_col: Optional[str] = "amount",
        supplier_col: Optional[str] = "supplier",
        confidence_col: Optional[str] = "confidence",
        dataset_name: str = "training_dataset"
    ) -> TrainingDataset:
        """Convert pandas DataFrame to TrainingDataset."""
        examples = []

        for idx, row in df.iterrows():
            try:
                example = TrainingExample(
                    description=row[description_col],
                    category=row[category_col],
                    amount=row.get(amount_col) if amount_col and amount_col in df.columns else None,
                    supplier=row.get(supplier_col) if supplier_col and supplier_col in df.columns else None,
                    confidence=row.get(confidence_col, 1.0) if confidence_col and confidence_col in df.columns else 1.0,
                    metadata={"row_index": idx}
                )
                examples.append(example)
            except Exception as e:
                logger.warning(f"Skipping invalid row {idx}: {e}")

        logger.info(f"Converted {len(examples)} examples from DataFrame")

        return TrainingDataset(
            examples=examples,
            name=dataset_name,
            metadata={
                "source": "dataframe",
                "total_rows": len(df),
                "valid_rows": len(examples)
            }
        )

    def validate_dataset(
        self,
        dataset: TrainingDataset,
        min_samples: int = 10,
        min_samples_per_category: int = 3
    ) -> Dict[str, Any]:
        """
        Validate training dataset.

        Args:
            dataset: Training dataset
            min_samples: Minimum total samples
            min_samples_per_category: Minimum samples per category

        Returns:
            Validation report

        Raises:
            InsufficientDataException: If validation fails
        """
        # Check minimum samples
        if len(dataset) < min_samples:
            raise InsufficientDataException(
                message=f"Insufficient training data: {len(dataset)} samples",
                data_count=len(dataset),
                required_count=min_samples
            )

        # Check category distribution
        distribution = dataset.get_category_distribution()

        # Check minimum per category
        insufficient_categories = []
        for category in Scope3Category.get_all_categories():
            count = distribution.get(category, 0)
            if count < min_samples_per_category:
                insufficient_categories.append((category, count))

        validation_report = {
            "valid": len(insufficient_categories) == 0,
            "total_samples": len(dataset),
            "category_distribution": distribution,
            "insufficient_categories": insufficient_categories,
            "min_samples_required": min_samples,
            "min_samples_per_category_required": min_samples_per_category
        }

        if insufficient_categories:
            logger.warning(
                f"Insufficient samples for {len(insufficient_categories)} categories: "
                f"{insufficient_categories}"
            )

        logger.info(f"Dataset validation: {validation_report['valid']}")
        return validation_report

    def split_dataset(
        self,
        dataset: TrainingDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True
    ) -> Tuple[TrainingDataset, TrainingDataset, TrainingDataset]:
        """
        Split dataset into train/validation/test sets.

        Args:
            dataset: Training dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Stratify by category

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Split ratios must sum to 1.0")

        if stratify:
            # Stratified split by category
            train_examples = []
            val_examples = []
            test_examples = []

            for category in Scope3Category.get_all_categories():
                category_examples = [
                    ex for ex in dataset.examples if ex.category == category
                ]

                if not category_examples:
                    continue

                # Shuffle
                deterministic_random().shuffle(category_examples)

                # Split
                n = len(category_examples)
                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)

                train_examples.extend(category_examples[:train_end])
                val_examples.extend(category_examples[train_end:val_end])
                test_examples.extend(category_examples[val_end:])

        else:
            # Random split
            examples = dataset.examples.copy()
            deterministic_random().shuffle(examples)

            n = len(examples)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_examples = examples[:train_end]
            val_examples = examples[train_end:val_end]
            test_examples = examples[val_end:]

        # Create datasets
        train_dataset = TrainingDataset(
            examples=train_examples,
            name=f"{dataset.name}_train",
            version=dataset.version,
            metadata={**dataset.metadata, "split": "train"}
        )

        val_dataset = TrainingDataset(
            examples=val_examples,
            name=f"{dataset.name}_val",
            version=dataset.version,
            metadata={**dataset.metadata, "split": "validation"}
        )

        test_dataset = TrainingDataset(
            examples=test_examples,
            name=f"{dataset.name}_test",
            version=dataset.version,
            metadata={**dataset.metadata, "split": "test"}
        )

        logger.info(
            f"Dataset split: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def augment_dataset(
        self,
        dataset: TrainingDataset,
        min_samples_per_category: int = 100,
        augmentation_methods: Optional[List[str]] = None
    ) -> TrainingDataset:
        """
        Augment dataset for imbalanced categories.

        Args:
            dataset: Training dataset
            min_samples_per_category: Minimum samples per category
            augmentation_methods: Augmentation methods (e.g., ["synonym", "paraphrase"])

        Returns:
            Augmented dataset
        """
        if augmentation_methods is None:
            augmentation_methods = ["duplicate"]  # Simple duplication

        augmented_examples = dataset.examples.copy()
        distribution = dataset.get_category_distribution()

        for category in Scope3Category.get_all_categories():
            current_count = distribution.get(category, 0)

            if current_count < min_samples_per_category and current_count > 0:
                # Get category examples
                category_examples = [
                    ex for ex in dataset.examples if ex.category == category
                ]

                # Calculate how many to add
                needed = min_samples_per_category - current_count

                # Simple duplication (can be extended with more sophisticated methods)
                for _ in range(needed):
                    original = deterministic_random().choice(category_examples)

                    # Create augmented example
                    augmented = TrainingExample(
                        description=original.description,
                        category=original.category,
                        amount=original.amount,
                        supplier=original.supplier,
                        confidence=original.confidence * 0.9,  # Lower confidence for augmented
                        metadata={**original.metadata, "augmented": True}
                    )
                    augmented_examples.append(augmented)

        logger.info(
            f"Augmented dataset: {len(dataset)} -> {len(augmented_examples)} examples"
        )

        return TrainingDataset(
            examples=augmented_examples,
            name=f"{dataset.name}_augmented",
            version=dataset.version,
            metadata={**dataset.metadata, "augmented": True}
        )

    def get_statistics(self, dataset: TrainingDataset) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Args:
            dataset: Training dataset

        Returns:
            Statistics dictionary
        """
        distribution = dataset.get_category_distribution()

        # Calculate statistics
        desc_lengths = [len(ex.description) for ex in dataset.examples]
        amounts = [ex.amount for ex in dataset.examples if ex.amount is not None]

        stats = {
            "total_samples": len(dataset),
            "category_distribution": distribution,
            "categories_with_data": len([c for c in distribution.values() if c > 0]),
            "avg_description_length": sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0,
            "min_description_length": min(desc_lengths) if desc_lengths else 0,
            "max_description_length": max(desc_lengths) if desc_lengths else 0,
            "samples_with_amount": len(amounts),
            "avg_amount": sum(amounts) / len(amounts) if amounts else 0,
            "samples_with_supplier": sum(1 for ex in dataset.examples if ex.supplier),
        }

        return stats
