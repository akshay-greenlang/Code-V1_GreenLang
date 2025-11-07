"""
BERT-based pairwise matching model for entity resolution.

This module implements fine-tuning and inference for BERT-based
pairwise entity matching with high precision.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np

from entity_mdm.ml.config import MLConfig, ModelConfig, TrainingConfig
from entity_mdm.ml.exceptions import MatchingException, ModelNotTrainedException

logger = logging.getLogger(__name__)


class EntityPair:
    """Data model for entity pairs."""

    def __init__(
        self,
        entity1_text: str,
        entity2_text: str,
        label: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize entity pair.

        Args:
            entity1_text: Text representation of first entity
            entity2_text: Text representation of second entity
            label: Ground truth label (1=match, 0=no match)
            metadata: Additional metadata
        """
        self.entity1_text = entity1_text
        self.entity2_text = entity2_text
        self.label = label
        self.metadata = metadata or {}


class EntityPairDataset(Dataset):
    """PyTorch dataset for entity pairs."""

    def __init__(
        self,
        pairs: List[EntityPair],
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ) -> None:
        """
        Initialize dataset.

        Args:
            pairs: List of entity pairs
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized inputs and label
        """
        pair = self.pairs[idx]

        # Tokenize pair of sequences
        encoding = self.tokenizer(
            pair.entity1_text,
            pair.entity2_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if pair.label is not None:
            item["label"] = torch.tensor(pair.label, dtype=torch.long)

        return item


class BertMatchingModel(nn.Module):
    """BERT-based binary classification model for entity matching."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the model.

        Args:
            bert_model_name: Pretrained BERT model name
            dropout: Dropout probability
        """
        super().__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Logits for binary classification
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Classify
        logits = self.classifier(pooled_output)
        return logits


class MatchingModel:
    """
    High-level interface for BERT-based entity matching.

    This class handles:
    - Model initialization and loading
    - Training with validation
    - Inference for pairwise matching
    - Model persistence
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the matching model.

        Args:
            config: ML configuration object
            model_path: Path to saved model checkpoint
        """
        self.config = config or MLConfig()
        self.model_config: ModelConfig = self.config.model
        self.training_config: TrainingConfig = self.config.training

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_config.bert_model,
            cache_dir=str(self.model_config.model_cache_dir),
        )

        # Initialize model
        self.model: Optional[BertMatchingModel] = None
        self.device = self._get_device()
        self.is_trained = False

        # Load model if path provided
        if model_path:
            self.load(model_path)
        else:
            self._init_model()

        logger.info(
            f"Initialized MatchingModel with model={self.model_config.bert_model}, "
            f"device={self.device}"
        )

    def _get_device(self) -> torch.device:
        """Get computation device."""
        if self.model_config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.model_config.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _init_model(self) -> None:
        """Initialize a new model."""
        self.model = BertMatchingModel(
            bert_model_name=self.model_config.bert_model,
        )
        self.model.to(self.device)
        self.is_trained = False

    def train(
        self,
        train_pairs: List[EntityPair],
        val_pairs: Optional[List[EntityPair]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_pairs: Training entity pairs
            val_pairs: Validation entity pairs
            epochs: Number of training epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)

        Returns:
            Dictionary with training history (loss, accuracy per epoch)

        Raises:
            MatchingException: If training fails
        """
        try:
            epochs = epochs or self.training_config.epochs
            batch_size = batch_size or self.model_config.batch_size

            # Create datasets
            train_dataset = EntityPairDataset(
                train_pairs,
                self.tokenizer,
                self.model_config.max_sequence_length,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            val_loader = None
            if val_pairs:
                val_dataset = EntityPairDataset(
                    val_pairs,
                    self.tokenizer,
                    self.model_config.max_sequence_length,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                )

            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )

            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.training_config.warmup_steps,
                num_training_steps=total_steps,
            )

            # Loss function
            criterion = nn.CrossEntropyLoss()

            # Training loop
            history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
            }

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                progress_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                )

                for batch in progress_bar:
                    # Move to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    # Forward pass
                    optimizer.zero_grad()
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()

                    # Track metrics
                    train_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "loss": loss.item(),
                            "acc": train_correct / train_total,
                        }
                    )

                # Calculate epoch metrics
                epoch_train_loss = train_loss / len(train_loader)
                epoch_train_acc = train_correct / train_total
                history["train_loss"].append(epoch_train_loss)
                history["train_acc"].append(epoch_train_acc)

                # Validation phase
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader, criterion)
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(val_acc)

                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        if self.training_config.save_best_only:
                            self.save(self.training_config.checkpoint_dir / "best_model.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= self.training_config.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                            break
                else:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
                    )

            self.is_trained = True
            return history

        except Exception as e:
            raise MatchingException(
                message=f"Training failed: {e}",
                details={"epochs": epochs, "batch_size": batch_size},
            )

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Run validation.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        return val_loss / len(val_loader), val_correct / val_total

    def predict(
        self,
        entity1_text: str,
        entity2_text: str,
    ) -> Tuple[int, float]:
        """
        Predict if two entities match.

        Args:
            entity1_text: First entity text
            entity2_text: Second entity text

        Returns:
            Tuple of (prediction, confidence)
            - prediction: 1 if match, 0 if no match
            - confidence: Confidence score (0.0 to 1.0)

        Raises:
            ModelNotTrainedException: If model hasn't been trained
            MatchingException: If prediction fails
        """
        if not self.is_trained:
            raise ModelNotTrainedException("MatchingModel")

        try:
            self.model.eval()

            # Tokenize
            encoding = self.tokenizer(
                entity1_text,
                entity2_text,
                add_special_tokens=True,
                max_length=self.model_config.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()

            return prediction, confidence

        except Exception as e:
            raise MatchingException(
                message=f"Prediction failed: {e}",
            )

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Predict for multiple entity pairs.

        Args:
            pairs: List of (entity1_text, entity2_text) tuples
            batch_size: Batch size for inference

        Returns:
            List of (prediction, confidence) tuples

        Raises:
            ModelNotTrainedException: If model hasn't been trained
            MatchingException: If prediction fails
        """
        if not self.is_trained:
            raise ModelNotTrainedException("MatchingModel")

        try:
            batch_size = batch_size or self.model_config.batch_size
            entity_pairs = [
                EntityPair(e1, e2) for e1, e2 in pairs
            ]

            dataset = EntityPairDataset(
                entity_pairs,
                self.tokenizer,
                self.model_config.max_sequence_length,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            results = []
            self.model.eval()

            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    logits = self.model(input_ids, attention_mask)
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)

                    for i in range(len(predictions)):
                        pred = predictions[i].item()
                        conf = probabilities[i][pred].item()
                        results.append((pred, conf))

            return results

        except Exception as e:
            raise MatchingException(
                message=f"Batch prediction failed: {e}",
                details={"num_pairs": len(pairs)},
            )

    def save(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint

        Raises:
            MatchingException: If save fails
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "is_trained": self.is_trained,
                "config": {
                    "bert_model": self.model_config.bert_model,
                    "max_sequence_length": self.model_config.max_sequence_length,
                },
            }

            torch.save(checkpoint, path)
            logger.info(f"Saved model checkpoint to {path}")

        except Exception as e:
            raise MatchingException(
                message=f"Failed to save model: {e}",
                details={"path": str(path)},
            )

    def load(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Raises:
            MatchingException: If load fails
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Initialize model if needed
            if self.model is None:
                self._init_model()

            # Load state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.is_trained = checkpoint["is_trained"]

            logger.info(f"Loaded model checkpoint from {path}")

        except Exception as e:
            raise MatchingException(
                message=f"Failed to load model: {e}",
                details={"path": str(path)},
            )
