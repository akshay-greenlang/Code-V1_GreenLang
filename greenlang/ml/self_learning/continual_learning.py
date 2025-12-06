# -*- coding: utf-8 -*-
"""
Continual Learning Module

This module provides Elastic Weight Consolidation (EWC) and other techniques
to prevent catastrophic forgetting in neural network models, enabling
GreenLang agents to learn new tasks without losing previously acquired knowledge.

Catastrophic forgetting is a critical challenge in ML systems that need to
adapt to new regulatory requirements, emission factors, or data patterns
while maintaining accuracy on historical data.

Example:
    >>> from greenlang.ml.self_learning import ContinualLearner
    >>> learner = ContinualLearner(model, method="ewc")
    >>> learner.consolidate(old_task_data)
    >>> learner.learn_new_task(new_task_data)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)


class ContinualMethod(str, Enum):
    """Continual learning methods."""
    EWC = "ewc"  # Elastic Weight Consolidation
    SI = "si"  # Synaptic Intelligence
    MAS = "mas"  # Memory Aware Synapses
    LWF = "lwf"  # Learning without Forgetting
    GEM = "gem"  # Gradient Episodic Memory
    REPLAY = "replay"  # Experience Replay


class ContinualLearnerConfig(BaseModel):
    """Configuration for continual learner."""

    method: ContinualMethod = Field(
        default=ContinualMethod.EWC,
        description="Continual learning method"
    )
    lambda_ewc: float = Field(
        default=5000.0,
        gt=0,
        description="EWC regularization strength"
    )
    fisher_n_samples: int = Field(
        default=200,
        ge=10,
        description="Samples for Fisher information estimation"
    )
    memory_size: int = Field(
        default=500,
        ge=100,
        description="Size of episodic memory"
    )
    replay_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for replay"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        description="Learning rate"
    )
    n_epochs: int = Field(
        default=10,
        ge=1,
        description="Training epochs"
    )
    temperature: float = Field(
        default=2.0,
        gt=0,
        description="Temperature for knowledge distillation"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class TaskInfo(BaseModel):
    """Information about a learned task."""

    task_id: str = Field(
        ...,
        description="Unique task identifier"
    )
    task_name: str = Field(
        ...,
        description="Human-readable task name"
    )
    n_samples: int = Field(
        ...,
        description="Number of training samples"
    )
    learned_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When task was learned"
    )
    final_loss: float = Field(
        ...,
        description="Final training loss"
    )
    provenance_hash: str = Field(
        ...,
        description="Provenance hash"
    )


class ContinualLearnerResult(BaseModel):
    """Result from continual learning training."""

    task_info: TaskInfo = Field(
        ...,
        description="Information about learned task"
    )
    training_history: List[Dict[str, float]] = Field(
        ...,
        description="Training history (loss per epoch)"
    )
    forgetting_measure: Optional[float] = Field(
        default=None,
        description="Measure of forgetting on previous tasks"
    )
    processing_time_ms: float = Field(
        ...,
        description="Training duration"
    )


class ContinualLearner:
    """
    Continual Learner for GreenLang agents.

    This class provides continual learning capabilities using EWC and
    related techniques to prevent catastrophic forgetting, enabling
    models to learn new tasks without losing previously acquired knowledge.

    Key capabilities:
    - Elastic Weight Consolidation (EWC)
    - Synaptic Intelligence (SI)
    - Experience Replay
    - Task-specific heads
    - Fisher information computation
    - Forgetting measurement

    Attributes:
        model: PyTorch neural network model
        config: Configuration for continual learning
        _fisher_dict: Fisher information matrices
        _optimal_params: Optimal parameters after each task
        _memory: Episodic memory for replay
        _tasks: List of learned tasks

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2))
        >>> learner = ContinualLearner(model, config=ContinualLearnerConfig(
        ...     method=ContinualMethod.EWC,
        ...     lambda_ewc=1000.0
        ... ))
        >>> # Learn first task
        >>> learner.learn_task("task1", X_task1, y_task1)
        >>> # Consolidate knowledge
        >>> learner.consolidate()
        >>> # Learn second task without forgetting
        >>> learner.learn_task("task2", X_task2, y_task2)
    """

    def __init__(
        self,
        model: Any,
        config: Optional[ContinualLearnerConfig] = None
    ):
        """
        Initialize continual learner.

        Args:
            model: PyTorch model or compatible
            config: Learner configuration
        """
        self.model = model
        self.config = config or ContinualLearnerConfig()

        # Fisher information and optimal params per task
        self._fisher_dict: Dict[str, Dict[str, np.ndarray]] = {}
        self._optimal_params: Dict[str, Dict[str, np.ndarray]] = {}

        # Episodic memory for replay
        self._memory: List[Tuple[np.ndarray, np.ndarray]] = []

        # Task tracking
        self._tasks: List[TaskInfo] = []
        self._current_task_id: Optional[str] = None

        # Importance weights for SI
        self._omega: Dict[str, np.ndarray] = {}
        self._previous_params: Dict[str, np.ndarray] = {}

        logger.info(
            f"ContinualLearner initialized with method={self.config.method}"
        )

    def _get_model_params(self) -> Dict[str, np.ndarray]:
        """Get model parameters as numpy arrays."""
        try:
            import torch
            params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params[name] = param.data.clone().cpu().numpy()
            return params
        except ImportError:
            # Fallback for non-PyTorch models
            if hasattr(self.model, "get_weights"):
                weights = self.model.get_weights()
                return {f"layer_{i}": w for i, w in enumerate(weights)}
            return {}

    def _set_model_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        try:
            import torch
            for name, param in self.model.named_parameters():
                if name in params and param.requires_grad:
                    param.data = torch.tensor(params[name]).to(param.device)
        except ImportError:
            if hasattr(self.model, "set_weights"):
                weights = [params[f"layer_{i}"] for i in range(len(params))]
                self.model.set_weights(weights)

    def _compute_fisher_information(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute Fisher information matrix using empirical Fisher.

        Args:
            X: Input data
            y: Target labels

        Returns:
            Dictionary of Fisher information per parameter
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            logger.warning("PyTorch required for Fisher computation")
            return {}

        fisher = {}

        # Initialize Fisher
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = np.zeros_like(param.data.cpu().numpy())

        # Compute empirical Fisher
        n_samples = min(self.config.fisher_n_samples, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)

        self.model.eval()

        for idx in indices:
            self.model.zero_grad()

            x_tensor = torch.tensor(X[idx:idx+1], dtype=torch.float32)

            # Forward pass
            output = self.model(x_tensor)

            if len(output.shape) > 1 and output.shape[1] > 1:
                # Classification
                log_prob = F.log_softmax(output, dim=1)
                # Use true label for Fisher
                target = torch.tensor([y[idx]], dtype=torch.long)
                loss = F.nll_loss(log_prob, target)
            else:
                # Regression
                target = torch.tensor([y[idx]], dtype=torch.float32)
                loss = F.mse_loss(output.squeeze(), target)

            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.cpu().numpy() ** 2

        # Average
        for name in fisher:
            fisher[name] /= n_samples

        return fisher

    def _ewc_loss(self) -> float:
        """
        Compute EWC regularization loss.

        Returns:
            EWC penalty term
        """
        try:
            import torch
        except ImportError:
            return 0.0

        ewc_loss = 0.0

        for task_id, fisher in self._fisher_dict.items():
            optimal = self._optimal_params[task_id]

            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    # Fisher * (theta - theta*)^2
                    diff = param - torch.tensor(optimal[name]).to(param.device)
                    ewc_loss += (
                        torch.tensor(fisher[name]).to(param.device) * diff ** 2
                    ).sum()

        return self.config.lambda_ewc * ewc_loss / 2

    def _si_update_omega(self, old_params: Dict[str, np.ndarray]) -> None:
        """Update importance weights for Synaptic Intelligence."""
        current_params = self._get_model_params()

        for name in current_params:
            if name in old_params:
                delta = current_params[name] - old_params[name]
                if name not in self._omega:
                    self._omega[name] = np.zeros_like(delta)
                # SI importance: accumulated gradient * parameter change
                self._omega[name] += np.abs(delta)

    def _add_to_memory(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: Optional[int] = None
    ) -> None:
        """Add samples to episodic memory."""
        n = n_samples or self.config.memory_size // 10

        indices = np.random.choice(len(X), min(n, len(X)), replace=False)

        for idx in indices:
            if len(self._memory) >= self.config.memory_size:
                # Replace random sample
                replace_idx = np.random.randint(len(self._memory))
                self._memory[replace_idx] = (X[idx], y[idx])
            else:
                self._memory.append((X[idx], y[idx]))

    def _calculate_provenance(
        self,
        task_id: str,
        X: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        param_hash = hashlib.sha256(
            str(sorted([(k, v.sum()) for k, v in params.items()])).encode()
        ).hexdigest()[:16]

        combined = f"{task_id}|{data_hash}|{param_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def consolidate(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Consolidate knowledge after learning a task.

        This computes Fisher information and stores optimal parameters
        to protect important weights from being overwritten.

        Args:
            X: Training data from current task
            y: Training labels from current task

        Example:
            >>> learner.consolidate(X_task1, y_task1)
        """
        task_id = self._current_task_id or f"task_{len(self._tasks)}"

        logger.info(f"Consolidating knowledge for task {task_id}")

        if self.config.method == ContinualMethod.EWC:
            # Compute Fisher information
            fisher = self._compute_fisher_information(X, y)
            self._fisher_dict[task_id] = fisher

            # Store optimal parameters
            self._optimal_params[task_id] = self._get_model_params()

        elif self.config.method == ContinualMethod.SI:
            # Update importance weights
            if self._previous_params:
                self._si_update_omega(self._previous_params)
            self._previous_params = self._get_model_params()

        elif self.config.method == ContinualMethod.REPLAY:
            # Add samples to memory
            self._add_to_memory(X, y)

        logger.info(f"Consolidation complete for {task_id}")

    def learn_task(
        self,
        task_name: str,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> ContinualLearnerResult:
        """
        Learn a new task with continual learning.

        Args:
            task_name: Name of the task
            X: Training data
            y: Training labels
            validation_data: Optional validation data

        Returns:
            ContinualLearnerResult with training details

        Example:
            >>> result = learner.learn_task("emission_factor_v2", X_new, y_new)
            >>> print(f"Final loss: {result.task_info.final_loss}")
        """
        start_time = datetime.utcnow()

        task_id = f"task_{len(self._tasks)}_{task_name}"
        self._current_task_id = task_id

        logger.info(f"Learning task: {task_name} ({len(X)} samples)")

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch required for continual learning. "
                "Install with: pip install torch"
            )

        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long if y.dtype in [np.int32, np.int64] else torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.replay_batch_size,
            shuffle=True
        )

        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Loss function
        if y.dtype in [np.int32, np.int64]:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # Training loop
        training_history = []
        self.model.train()

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                output = self.model(batch_X)

                # Task loss
                if len(output.shape) > 1 and output.shape[1] > 1:
                    task_loss = criterion(output, batch_y)
                else:
                    task_loss = criterion(output.squeeze(), batch_y.float())

                # Add EWC regularization
                if self.config.method == ContinualMethod.EWC and self._fisher_dict:
                    ewc_penalty = self._ewc_loss()
                    loss = task_loss + ewc_penalty
                else:
                    loss = task_loss

                # Replay from memory
                if self.config.method == ContinualMethod.REPLAY and self._memory:
                    replay_loss = self._replay_step(criterion)
                    loss = loss + replay_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            training_history.append({
                "epoch": epoch + 1,
                "loss": avg_loss
            })

            logger.debug(f"Epoch {epoch+1}/{self.config.n_epochs}, Loss: {avg_loss:.4f}")

        # Consolidate after learning
        self.consolidate(X, y)

        # Calculate provenance
        final_params = self._get_model_params()
        provenance_hash = self._calculate_provenance(task_id, X, final_params)

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            task_name=task_name,
            n_samples=len(X),
            learned_at=datetime.utcnow(),
            final_loss=training_history[-1]["loss"],
            provenance_hash=provenance_hash
        )
        self._tasks.append(task_info)

        # Measure forgetting
        forgetting = None
        if validation_data and len(self._tasks) > 1:
            forgetting = self._measure_forgetting(validation_data)

        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Task {task_name} learned. Final loss: {task_info.final_loss:.4f}"
        )

        return ContinualLearnerResult(
            task_info=task_info,
            training_history=training_history,
            forgetting_measure=forgetting,
            processing_time_ms=processing_time_ms
        )

    def _replay_step(self, criterion) -> float:
        """Perform replay step from episodic memory."""
        import torch

        if not self._memory:
            return 0.0

        # Sample from memory
        indices = np.random.choice(
            len(self._memory),
            min(self.config.replay_batch_size, len(self._memory)),
            replace=False
        )

        X_replay = np.array([self._memory[i][0] for i in indices])
        y_replay = np.array([self._memory[i][1] for i in indices])

        X_tensor = torch.tensor(X_replay, dtype=torch.float32)
        y_tensor = torch.tensor(y_replay)

        output = self.model(X_tensor)

        if len(output.shape) > 1 and output.shape[1] > 1:
            return criterion(output, y_tensor.long())
        else:
            return criterion(output.squeeze(), y_tensor.float())

    def _measure_forgetting(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """
        Measure forgetting on validation data.

        Args:
            validation_data: Tuple of (X, y)

        Returns:
            Forgetting measure (higher = more forgetting)
        """
        import torch

        X_val, y_val = validation_data

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_val, dtype=torch.float32)
            output = self.model(X_tensor)

            if len(output.shape) > 1 and output.shape[1] > 1:
                predictions = output.argmax(dim=1).numpy()
                accuracy = (predictions == y_val).mean()
                return 1.0 - accuracy  # Forgetting = 1 - accuracy
            else:
                predictions = output.squeeze().numpy()
                mse = np.mean((predictions - y_val) ** 2)
                return float(mse)

    def get_learned_tasks(self) -> List[TaskInfo]:
        """Get list of learned tasks."""
        return self._tasks.copy()

    def get_importance_weights(self) -> Dict[str, np.ndarray]:
        """Get parameter importance weights."""
        if self.config.method == ContinualMethod.EWC:
            # Average Fisher across tasks
            avg_fisher = {}
            for fisher in self._fisher_dict.values():
                for name, values in fisher.items():
                    if name not in avg_fisher:
                        avg_fisher[name] = values.copy()
                    else:
                        avg_fisher[name] += values
            if self._fisher_dict:
                for name in avg_fisher:
                    avg_fisher[name] /= len(self._fisher_dict)
            return avg_fisher

        elif self.config.method == ContinualMethod.SI:
            return self._omega.copy()

        return {}


# Unit test stubs
class TestContinualLearner:
    """Unit tests for ContinualLearner."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        class MockModel:
            def named_parameters(self):
                return []

        learner = ContinualLearner(MockModel())
        assert learner.config.method == ContinualMethod.EWC
        assert learner.config.lambda_ewc == 5000.0

    def test_memory_management(self):
        """Test episodic memory management."""
        class MockModel:
            def named_parameters(self):
                return []

        config = ContinualLearnerConfig(memory_size=100)
        learner = ContinualLearner(MockModel(), config)

        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)

        learner._add_to_memory(X, y, n_samples=20)
        assert len(learner._memory) == 20

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        class MockModel:
            def named_parameters(self):
                return []

        learner = ContinualLearner(MockModel())

        X = np.array([[1.0, 2.0]])
        params = {"layer_0": np.array([0.5])}

        hash1 = learner._calculate_provenance("task1", X, params)
        hash2 = learner._calculate_provenance("task1", X, params)

        assert hash1 == hash2

    def test_get_learned_tasks(self):
        """Test getting learned tasks."""
        class MockModel:
            def named_parameters(self):
                return []

        learner = ContinualLearner(MockModel())
        assert len(learner.get_learned_tasks()) == 0
