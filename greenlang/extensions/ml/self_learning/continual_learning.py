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
    EWC_ONLINE = "ewc_online"  # Online EWC (running Fisher)
    SI = "si"  # Synaptic Intelligence
    MAS = "mas"  # Memory Aware Synapses
    LWF = "lwf"  # Learning without Forgetting
    GEM = "gem"  # Gradient Episodic Memory
    AGEM = "agem"  # Averaged GEM (more efficient)
    REPLAY = "replay"  # Experience Replay
    PNN = "pnn"  # Progressive Neural Networks


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
    # Online EWC parameters
    ewc_online_gamma: float = Field(
        default=0.95,
        ge=0,
        le=1.0,
        description="Decay factor for online EWC Fisher information"
    )
    # GEM/A-GEM parameters
    gem_memory_strength: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Strength of gradient projection for GEM"
    )
    gem_n_memories: int = Field(
        default=256,
        ge=10,
        description="Number of memories per task for GEM"
    )
    # MAS parameters
    mas_lambda: float = Field(
        default=1.0,
        gt=0,
        description="MAS regularization strength"
    )
    # Progressive Neural Networks
    pnn_lateral_connections: bool = Field(
        default=True,
        description="Enable lateral connections in PNN"
    )
    pnn_hidden_dim: int = Field(
        default=64,
        ge=16,
        description="Hidden dimension for PNN columns"
    )
    # Forgetting metrics
    track_forgetting: bool = Field(
        default=True,
        description="Track detailed forgetting metrics"
    )
    forgetting_eval_frequency: int = Field(
        default=1,
        ge=1,
        description="Evaluate forgetting every N tasks"
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


class ForgettingMetrics(BaseModel):
    """Detailed metrics for tracking catastrophic forgetting."""

    backward_transfer: float = Field(
        default=0.0,
        description="Backward transfer: impact on previous tasks (-1 to 1)"
    )
    forward_transfer: float = Field(
        default=0.0,
        description="Forward transfer: benefit to future tasks (-1 to 1)"
    )
    average_forgetting: float = Field(
        default=0.0,
        description="Average forgetting across all tasks"
    )
    max_forgetting: float = Field(
        default=0.0,
        description="Maximum forgetting on any single task"
    )
    retention_score: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Overall knowledge retention score"
    )
    plasticity_score: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="Model's ability to learn new tasks"
    )
    stability_plasticity_ratio: float = Field(
        default=1.0,
        gt=0,
        description="Balance between stability and plasticity"
    )
    task_performances: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance on each task after training"
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
    forgetting_metrics: Optional[ForgettingMetrics] = Field(
        default=None,
        description="Detailed forgetting metrics"
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

        # Online EWC: running Fisher information
        self._online_fisher: Dict[str, np.ndarray] = {}
        self._online_optimal_params: Dict[str, np.ndarray] = {}

        # Episodic memory for replay
        self._memory: List[Tuple[np.ndarray, np.ndarray]] = []

        # GEM/A-GEM: gradient episodic memory
        self._gem_memories: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._gem_grads: Dict[str, Dict[str, np.ndarray]] = {}

        # MAS: Memory Aware Synapses importance
        self._mas_importance: Dict[str, np.ndarray] = {}

        # Task tracking
        self._tasks: List[TaskInfo] = []
        self._current_task_id: Optional[str] = None

        # Importance weights for SI
        self._omega: Dict[str, np.ndarray] = {}
        self._previous_params: Dict[str, np.ndarray] = {}

        # Forgetting tracking
        self._task_val_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._initial_performances: Dict[str, float] = {}
        self._current_performances: Dict[str, float] = {}

        # PNN columns
        self._pnn_columns: List[Any] = []

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

        elif self.config.method == ContinualMethod.EWC_ONLINE:
            return self._online_fisher.copy()

        elif self.config.method == ContinualMethod.SI:
            return self._omega.copy()

        elif self.config.method == ContinualMethod.MAS:
            return self._mas_importance.copy()

        return {}

    # =========================================================================
    # Online EWC Implementation
    # =========================================================================

    def _update_online_ewc(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Update running Fisher information for Online EWC.

        Online EWC maintains a single Fisher information matrix that
        is updated with exponential moving average after each task.

        Args:
            X: Training data
            y: Training labels
        """
        # Compute new Fisher
        new_fisher = self._compute_fisher_information(X, y)

        if not self._online_fisher:
            # First task - initialize
            self._online_fisher = new_fisher
            self._online_optimal_params = self._get_model_params()
        else:
            # Update with EMA
            gamma = self.config.ewc_online_gamma
            for name in new_fisher:
                if name in self._online_fisher:
                    self._online_fisher[name] = (
                        gamma * self._online_fisher[name] +
                        (1 - gamma) * new_fisher[name]
                    )
                else:
                    self._online_fisher[name] = new_fisher[name]

            # Update optimal params (use current)
            self._online_optimal_params = self._get_model_params()

        logger.info("Online EWC Fisher updated")

    def _online_ewc_loss(self) -> float:
        """Compute Online EWC regularization loss."""
        try:
            import torch
        except ImportError:
            return 0.0

        if not self._online_fisher:
            return 0.0

        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self._online_fisher and param.requires_grad:
                optimal = self._online_optimal_params.get(name)
                if optimal is not None:
                    diff = param - torch.tensor(optimal).to(param.device)
                    fisher = torch.tensor(self._online_fisher[name]).to(param.device)
                    ewc_loss += (fisher * diff ** 2).sum()

        return self.config.lambda_ewc * ewc_loss / 2

    # =========================================================================
    # Gradient Episodic Memory (GEM) Implementation
    # =========================================================================

    def _store_gem_memory(
        self,
        task_id: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Store memory samples for GEM.

        Args:
            task_id: Task identifier
            X: Training data
            y: Training labels
        """
        n_samples = min(self.config.gem_n_memories, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        self._gem_memories[task_id] = (X[indices], y[indices])

        logger.debug(f"Stored {n_samples} memories for GEM task {task_id}")

    def _compute_gem_gradients(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute reference gradients for all previous tasks.

        Returns:
            Dictionary of gradients per task per parameter
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return {}

        grads = {}

        for task_id, (X_mem, y_mem) in self._gem_memories.items():
            self.model.zero_grad()

            X_tensor = torch.tensor(X_mem, dtype=torch.float32)
            y_tensor = torch.tensor(y_mem)

            output = self.model(X_tensor)

            if len(output.shape) > 1 and output.shape[1] > 1:
                loss = F.cross_entropy(output, y_tensor.long())
            else:
                loss = F.mse_loss(output.squeeze(), y_tensor.float())

            loss.backward()

            grads[task_id] = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grads[task_id][name] = param.grad.data.cpu().numpy().copy()

        return grads

    def _project_gem_gradient(
        self,
        current_grads: Dict[str, np.ndarray],
        ref_grads: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Project gradient to satisfy GEM constraints.

        Uses quadratic programming to find the closest gradient
        that does not increase loss on any previous task.

        Args:
            current_grads: Current task gradients
            ref_grads: Reference gradients from previous tasks

        Returns:
            Projected gradients
        """
        if not ref_grads:
            return current_grads

        # Flatten gradients
        def flatten(grad_dict):
            return np.concatenate([g.flatten() for g in grad_dict.values()])

        def unflatten(flat_grad, template):
            result = {}
            offset = 0
            for name, g in template.items():
                size = g.size
                result[name] = flat_grad[offset:offset + size].reshape(g.shape)
                offset += size
            return result

        g = flatten(current_grads)

        # Check if any constraint is violated
        violations = []
        for task_id, ref_g in ref_grads.items():
            ref_flat = flatten(ref_g)
            dot_product = np.dot(g, ref_flat)
            if dot_product < 0:  # Constraint violated
                violations.append((task_id, ref_flat))

        if not violations:
            return current_grads

        # Project onto feasible region using simplified projection
        # (Full QP would require additional dependencies)
        for task_id, ref_flat in violations:
            dot_product = np.dot(g, ref_flat)
            if dot_product < 0:
                # Project g onto the half-space
                g = g - (dot_product / (np.dot(ref_flat, ref_flat) + 1e-8)) * ref_flat
                g *= self.config.gem_memory_strength

        return unflatten(g, current_grads)

    # =========================================================================
    # Averaged GEM (A-GEM) Implementation
    # =========================================================================

    def _agem_update(
        self,
        current_loss: float,
        criterion: Any
    ) -> None:
        """
        Apply A-GEM gradient projection.

        A-GEM is a more efficient version of GEM that uses a single
        reference gradient computed from averaged episodic memory.

        Args:
            current_loss: Current task loss
            criterion: Loss criterion
        """
        try:
            import torch
        except ImportError:
            return

        if not self._gem_memories:
            return

        # Sample uniformly from all episodic memories
        all_X, all_y = [], []
        for X_mem, y_mem in self._gem_memories.values():
            all_X.append(X_mem)
            all_y.append(y_mem)

        X_ref = np.vstack(all_X)
        y_ref = np.concatenate(all_y)

        # Subsample
        n_samples = min(self.config.replay_batch_size, len(X_ref))
        indices = np.random.choice(len(X_ref), n_samples, replace=False)
        X_ref, y_ref = X_ref[indices], y_ref[indices]

        # Compute reference gradient
        self.model.zero_grad()
        X_tensor = torch.tensor(X_ref, dtype=torch.float32)
        y_tensor = torch.tensor(y_ref)

        output = self.model(X_tensor)
        if len(output.shape) > 1 and output.shape[1] > 1:
            ref_loss = criterion(output, y_tensor.long())
        else:
            ref_loss = criterion(output.squeeze(), y_tensor.float())

        ref_loss.backward()

        # Store reference gradients
        ref_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                ref_grads[name] = param.grad.data.clone()

        # Check for violation
        current_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                current_grads[name] = param.grad.data.clone()

        # Compute dot product
        dot_product = sum(
            (current_grads[n] * ref_grads[n]).sum()
            for n in current_grads if n in ref_grads
        )

        if dot_product < 0:
            # Project
            ref_norm_sq = sum((g ** 2).sum() for g in ref_grads.values())
            if ref_norm_sq > 1e-8:
                proj_factor = dot_product / ref_norm_sq
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in ref_grads:
                        param.grad.data -= proj_factor * ref_grads[name]

    # =========================================================================
    # Memory Aware Synapses (MAS) Implementation
    # =========================================================================

    def _compute_mas_importance(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute parameter importance using MAS.

        MAS computes importance based on the sensitivity of the
        network output to parameter changes, using unlabeled data.

        Args:
            X: Unlabeled input data

        Returns:
            Importance weights per parameter
        """
        try:
            import torch
        except ImportError:
            return {}

        importance = {}

        # Initialize importance
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = np.zeros_like(param.data.cpu().numpy())

        self.model.eval()
        n_samples = min(self.config.fisher_n_samples, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)

        for idx in indices:
            self.model.zero_grad()

            x_tensor = torch.tensor(X[idx:idx+1], dtype=torch.float32)
            output = self.model(x_tensor)

            # L2 norm of output as importance signal
            output_norm = output.pow(2).mean()
            output_norm.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    importance[name] += np.abs(param.grad.data.cpu().numpy())

        # Normalize
        for name in importance:
            importance[name] /= n_samples

        return importance

    def _update_mas_importance(self, X: np.ndarray) -> None:
        """Update MAS importance weights."""
        new_importance = self._compute_mas_importance(X)

        for name, imp in new_importance.items():
            if name in self._mas_importance:
                self._mas_importance[name] += imp
            else:
                self._mas_importance[name] = imp

        logger.info("MAS importance updated")

    def _mas_loss(self) -> float:
        """Compute MAS regularization loss."""
        try:
            import torch
        except ImportError:
            return 0.0

        if not self._mas_importance or not self._optimal_params:
            return 0.0

        # Use the most recent optimal params
        if not self._optimal_params:
            return 0.0

        # Get latest task's optimal params
        latest_task = list(self._optimal_params.keys())[-1]
        optimal = self._optimal_params[latest_task]

        mas_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self._mas_importance and param.requires_grad:
                if name in optimal:
                    diff = param - torch.tensor(optimal[name]).to(param.device)
                    imp = torch.tensor(self._mas_importance[name]).to(param.device)
                    mas_loss += (imp * diff ** 2).sum()

        return self.config.mas_lambda * mas_loss / 2

    # =========================================================================
    # Forgetting Metrics Tracking
    # =========================================================================

    def _evaluate_task_performance(
        self,
        task_id: str
    ) -> Optional[float]:
        """Evaluate current performance on a specific task."""
        if task_id not in self._task_val_data:
            return None

        X_val, y_val = self._task_val_data[task_id]
        return 1.0 - self._measure_forgetting((X_val, y_val))

    def _compute_forgetting_metrics(self) -> ForgettingMetrics:
        """
        Compute comprehensive forgetting metrics.

        Returns:
            ForgettingMetrics with detailed analysis
        """
        if len(self._tasks) == 0:
            return ForgettingMetrics()

        # Evaluate all tasks
        task_performances = {}
        forgetting_values = []

        for task_info in self._tasks:
            task_id = task_info.task_id
            current_perf = self._evaluate_task_performance(task_id)

            if current_perf is not None:
                task_performances[task_id] = current_perf
                self._current_performances[task_id] = current_perf

                # Compare with initial performance
                initial_perf = self._initial_performances.get(task_id, current_perf)
                forgetting = max(0, initial_perf - current_perf)
                forgetting_values.append(forgetting)

        # Compute metrics
        avg_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0
        max_forgetting = np.max(forgetting_values) if forgetting_values else 0.0

        # Retention score (1 - average forgetting)
        retention_score = max(0, 1.0 - avg_forgetting)

        # Plasticity score (ability to learn new tasks)
        # Based on how well the latest task was learned
        latest_perf = list(task_performances.values())[-1] if task_performances else 1.0
        plasticity_score = min(1.0, max(0, latest_perf))

        # Stability-plasticity ratio
        sp_ratio = retention_score / (plasticity_score + 1e-8)

        # Backward transfer
        backward_transfer = -avg_forgetting  # Negative forgetting = positive transfer

        return ForgettingMetrics(
            backward_transfer=backward_transfer,
            forward_transfer=0.0,  # Would need zero-shot evaluation
            average_forgetting=avg_forgetting,
            max_forgetting=max_forgetting,
            retention_score=retention_score,
            plasticity_score=plasticity_score,
            stability_plasticity_ratio=sp_ratio,
            task_performances=task_performances
        )

    def store_validation_data(
        self,
        task_id: str,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Store validation data for forgetting measurement.

        Args:
            task_id: Task identifier
            X_val: Validation features
            y_val: Validation labels
        """
        self._task_val_data[task_id] = (X_val.copy(), y_val.copy())

        # Record initial performance
        perf = self._evaluate_task_performance(task_id)
        if perf is not None:
            self._initial_performances[task_id] = perf

    def get_forgetting_metrics(self) -> ForgettingMetrics:
        """Get current forgetting metrics."""
        return self._compute_forgetting_metrics()

    # =========================================================================
    # Progressive Neural Networks (PNN)
    # =========================================================================

    def _create_pnn_column(
        self,
        task_id: str,
        input_dim: int,
        output_dim: int
    ) -> Any:
        """
        Create a new column for Progressive Neural Network.

        Args:
            task_id: Task identifier
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            New neural network column
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return None

        hidden_dim = self.config.pnn_hidden_dim
        n_prev_columns = len(self._pnn_columns)

        class PNNColumn(nn.Module):
            """Single column in Progressive Neural Network."""

            def __init__(self, input_dim, hidden_dim, output_dim, n_prev):
                super().__init__()

                # Main layers
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)

                # Lateral connections from previous columns
                self.lateral1 = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim) for _ in range(n_prev)
                ])
                self.lateral2 = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim) for _ in range(n_prev)
                ])

                self.relu = nn.ReLU()
                self.n_prev = n_prev

            def forward(self, x, prev_hiddens=None):
                h1 = self.relu(self.fc1(x))

                # Add lateral connections
                if prev_hiddens and len(prev_hiddens) > 0:
                    for i, (lateral, prev_h1) in enumerate(zip(self.lateral1, prev_hiddens)):
                        if i < len(prev_hiddens) and prev_hiddens[i] is not None:
                            h1 = h1 + self.relu(lateral(prev_hiddens[i][0]))

                h2 = self.relu(self.fc2(h1))

                if prev_hiddens and len(prev_hiddens) > 0:
                    for i, (lateral, prev_h2) in enumerate(zip(self.lateral2, prev_hiddens)):
                        if i < len(prev_hiddens) and prev_hiddens[i] is not None:
                            h2 = h2 + self.relu(lateral(prev_hiddens[i][1]))

                out = self.fc3(h2)
                return out, (h1, h2)

        column = PNNColumn(input_dim, hidden_dim, output_dim, n_prev_columns)

        # Freeze previous columns
        for prev_col in self._pnn_columns:
            for param in prev_col.parameters():
                param.requires_grad = False

        self._pnn_columns.append(column)
        logger.info(f"Created PNN column {len(self._pnn_columns)} for task {task_id}")

        return column

    def forward_pnn(self, x: Any) -> Any:
        """
        Forward pass through all PNN columns.

        Args:
            x: Input tensor

        Returns:
            Output from latest column
        """
        if not self._pnn_columns:
            return self.model(x)

        prev_hiddens = []
        for column in self._pnn_columns[:-1]:
            _, hiddens = column(x, prev_hiddens)
            prev_hiddens.append(hiddens)

        # Output from latest column
        output, _ = self._pnn_columns[-1](x, prev_hiddens)
        return output


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
