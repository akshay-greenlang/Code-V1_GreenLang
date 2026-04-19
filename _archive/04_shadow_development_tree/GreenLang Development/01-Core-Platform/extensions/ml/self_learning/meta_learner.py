# -*- coding: utf-8 -*-
"""
Meta-Learner Module

This module provides Model-Agnostic Meta-Learning (MAML) implementation
for GreenLang agents, enabling few-shot adaptation to new tasks, regions,
or regulatory frameworks.

Meta-learning enables models to "learn how to learn", allowing rapid
adaptation to new emission factors, regulations, or data distributions
with minimal training samples.

Example:
    >>> from greenlang.ml.self_learning import MetaLearner
    >>> learner = MetaLearner(model, n_inner_steps=5)
    >>> adapted_model = learner.adapt(support_set, n_shots=5)
    >>> predictions = adapted_model.predict(query_set)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
import copy

logger = logging.getLogger(__name__)


class MetaLearningMethod(str, Enum):
    """Meta-learning methods."""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # Simpler first-order approximation
    FOMAML = "fomaml"  # First-Order MAML
    ANIL = "anil"  # Almost No Inner Loop
    PROTONET = "protonet"  # Prototypical Networks


class MetaLearnerConfig(BaseModel):
    """Configuration for meta-learner."""

    method: MetaLearningMethod = Field(
        default=MetaLearningMethod.MAML,
        description="Meta-learning method"
    )
    n_inner_steps: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Inner loop gradient steps"
    )
    inner_lr: float = Field(
        default=0.01,
        gt=0,
        description="Inner loop learning rate"
    )
    outer_lr: float = Field(
        default=0.001,
        gt=0,
        description="Outer loop (meta) learning rate"
    )
    n_tasks_per_batch: int = Field(
        default=4,
        ge=1,
        description="Number of tasks per meta-batch"
    )
    n_shots: int = Field(
        default=5,
        ge=1,
        description="Default number of support examples"
    )
    n_queries: int = Field(
        default=15,
        ge=1,
        description="Number of query examples"
    )
    first_order: bool = Field(
        default=False,
        description="Use first-order approximation (faster)"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class TaskData(BaseModel):
    """Data for a single task."""

    task_id: str = Field(
        ...,
        description="Task identifier"
    )
    support_X: List[List[float]] = Field(
        ...,
        description="Support set features"
    )
    support_y: List[Union[int, float]] = Field(
        ...,
        description="Support set labels"
    )
    query_X: List[List[float]] = Field(
        ...,
        description="Query set features"
    )
    query_y: List[Union[int, float]] = Field(
        ...,
        description="Query set labels"
    )


class AdaptationResult(BaseModel):
    """Result from task adaptation."""

    task_id: str = Field(
        ...,
        description="Task identifier"
    )
    support_loss: float = Field(
        ...,
        description="Loss on support set"
    )
    query_loss: float = Field(
        ...,
        description="Loss on query set after adaptation"
    )
    query_accuracy: Optional[float] = Field(
        default=None,
        description="Accuracy on query set (classification)"
    )
    n_inner_steps: int = Field(
        ...,
        description="Inner steps performed"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing duration"
    )


class MetaTrainingResult(BaseModel):
    """Result from meta-training."""

    n_iterations: int = Field(
        ...,
        description="Number of meta-training iterations"
    )
    final_meta_loss: float = Field(
        ...,
        description="Final meta-loss"
    )
    training_history: List[Dict[str, float]] = Field(
        ...,
        description="Training history"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total training time"
    )


class MetaLearner:
    """
    Meta-Learner for GreenLang agents.

    This class implements Model-Agnostic Meta-Learning (MAML) and related
    algorithms, enabling few-shot adaptation to new tasks like:
    - New regulatory frameworks
    - New geographic regions
    - New emission factor updates
    - New industry sectors

    Key capabilities:
    - MAML for few-shot learning
    - REPTILE for simpler meta-learning
    - Task-agnostic adaptation
    - Provenance tracking

    Attributes:
        model: Base neural network model
        config: Meta-learning configuration
        _meta_optimizer: Optimizer for outer loop
        _task_history: History of adapted tasks

    Example:
        >>> model = create_emission_model()
        >>> meta_learner = MetaLearner(model, config=MetaLearnerConfig(
        ...     method=MetaLearningMethod.MAML,
        ...     n_inner_steps=5,
        ...     n_shots=10
        ... ))
        >>> # Meta-train on multiple tasks
        >>> meta_learner.meta_train(task_distribution, n_iterations=1000)
        >>> # Adapt to new task
        >>> result = meta_learner.adapt(new_task_support_data)
    """

    def __init__(
        self,
        model: Any,
        config: Optional[MetaLearnerConfig] = None
    ):
        """
        Initialize meta-learner.

        Args:
            model: PyTorch model to meta-learn
            config: Meta-learning configuration
        """
        self.model = model
        self.config = config or MetaLearnerConfig()
        self._meta_optimizer = None
        self._task_history: List[AdaptationResult] = []
        self._initialized = False

        np.random.seed(self.config.random_state)

        logger.info(
            f"MetaLearner initialized with method={self.config.method}"
        )

    def _initialize_optimizer(self) -> None:
        """Initialize meta-optimizer."""
        try:
            import torch.optim as optim
            self._meta_optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.outer_lr
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "PyTorch required. Install with: pip install torch"
            )

    def _clone_model(self) -> Any:
        """Create a copy of the model for inner loop."""
        return copy.deepcopy(self.model)

    def _get_model_params(self) -> Dict[str, np.ndarray]:
        """Get model parameters as dict."""
        try:
            import torch
            return {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }
        except ImportError:
            return {}

    def _set_model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        try:
            import torch
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data = params[name].clone()
        except ImportError:
            pass

    def _calculate_provenance(
        self,
        task_id: str,
        support_X: np.ndarray,
        query_loss: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data_hash = hashlib.sha256(support_X.tobytes()).hexdigest()[:16]
        combined = f"{task_id}|{data_hash}|{query_loss}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _inner_loop(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray,
        n_steps: Optional[int] = None
    ) -> Tuple[Any, List[float]]:
        """
        Perform inner loop adaptation on support set.

        Args:
            support_X: Support set features
            support_y: Support set labels
            n_steps: Number of gradient steps

        Returns:
            Tuple of (adapted_model, losses)
        """
        import torch
        import torch.nn.functional as F

        n_steps = n_steps or self.config.n_inner_steps

        # Clone model for adaptation
        adapted_model = self._clone_model()

        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )

        X_tensor = torch.tensor(support_X, dtype=torch.float32)
        y_tensor = torch.tensor(support_y)

        losses = []

        for step in range(n_steps):
            inner_optimizer.zero_grad()

            output = adapted_model(X_tensor)

            # Determine loss function
            if len(output.shape) > 1 and output.shape[1] > 1:
                loss = F.cross_entropy(output, y_tensor.long())
            else:
                loss = F.mse_loss(output.squeeze(), y_tensor.float())

            loss.backward()
            inner_optimizer.step()

            losses.append(loss.item())

        return adapted_model, losses

    def _compute_meta_gradient(
        self,
        task: TaskData
    ) -> Tuple[Dict[str, Any], float, float]:
        """
        Compute meta-gradient for a single task.

        Args:
            task: Task data with support and query sets

        Returns:
            Tuple of (gradients, support_loss, query_loss)
        """
        import torch
        import torch.nn.functional as F

        support_X = np.array(task.support_X)
        support_y = np.array(task.support_y)
        query_X = np.array(task.query_X)
        query_y = np.array(task.query_y)

        # Inner loop adaptation
        adapted_model, support_losses = self._inner_loop(support_X, support_y)

        # Compute loss on query set
        X_query = torch.tensor(query_X, dtype=torch.float32)
        y_query = torch.tensor(query_y)

        output = adapted_model(X_query)

        if len(output.shape) > 1 and output.shape[1] > 1:
            query_loss = F.cross_entropy(output, y_query.long())
        else:
            query_loss = F.mse_loss(output.squeeze(), y_query.float())

        # Compute gradients w.r.t. original model parameters
        if not self.config.first_order:
            # Full MAML: backprop through adaptation
            query_loss.backward()

            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
        else:
            # FOMAML: First-order approximation
            gradients = {}
            for (name, orig_param), (_, adapted_param) in zip(
                self.model.named_parameters(),
                adapted_model.named_parameters()
            ):
                gradients[name] = (orig_param.data - adapted_param.data) / self.config.inner_lr

        return gradients, support_losses[-1], query_loss.item()

    def adapt(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray,
        task_id: Optional[str] = None,
        n_steps: Optional[int] = None
    ) -> AdaptationResult:
        """
        Adapt the model to a new task using the support set.

        This performs inner loop adaptation without updating the
        meta-parameters, useful for deployment/inference.

        Args:
            support_X: Support set features
            support_y: Support set labels
            task_id: Optional task identifier
            n_steps: Override number of inner steps

        Returns:
            AdaptationResult with adapted model performance

        Example:
            >>> # New regulatory framework with few examples
            >>> support_X = np.array([[...], [...], [...]])  # 5 examples
            >>> support_y = np.array([...])
            >>> result = meta_learner.adapt(support_X, support_y, task_id="eu_cbam")
            >>> print(f"Adapted with query loss: {result.query_loss}")
        """
        import torch

        start_time = datetime.utcnow()
        task_id = task_id or f"task_{len(self._task_history)}"

        logger.info(f"Adapting to task {task_id} with {len(support_X)} samples")

        # Perform inner loop
        adapted_model, losses = self._inner_loop(
            support_X, support_y, n_steps
        )

        # Store adapted model as current
        for (name, param), (_, adapted_param) in zip(
            self.model.named_parameters(),
            adapted_model.named_parameters()
        ):
            param.data = adapted_param.data.clone()

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            task_id, support_X, losses[-1]
        )

        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        result = AdaptationResult(
            task_id=task_id,
            support_loss=losses[-1],
            query_loss=losses[-1],  # Use support loss as proxy
            query_accuracy=None,
            n_inner_steps=len(losses),
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms
        )

        self._task_history.append(result)

        logger.info(
            f"Adaptation complete. Final loss: {losses[-1]:.4f}"
        )

        return result

    def meta_train(
        self,
        tasks: List[TaskData],
        n_iterations: int = 1000,
        eval_interval: int = 100
    ) -> MetaTrainingResult:
        """
        Perform meta-training on a distribution of tasks.

        Args:
            tasks: List of training tasks
            n_iterations: Number of meta-training iterations
            eval_interval: Interval for logging

        Returns:
            MetaTrainingResult with training history

        Example:
            >>> tasks = [TaskData(...), TaskData(...), ...]
            >>> result = meta_learner.meta_train(tasks, n_iterations=5000)
            >>> print(f"Final meta-loss: {result.final_meta_loss}")
        """
        import torch

        start_time = datetime.utcnow()

        if not self._initialized:
            self._initialize_optimizer()

        training_history = []

        logger.info(
            f"Starting meta-training with {len(tasks)} tasks, "
            f"{n_iterations} iterations"
        )

        for iteration in range(n_iterations):
            self._meta_optimizer.zero_grad()

            # Sample tasks for this iteration
            task_indices = np.random.choice(
                len(tasks),
                min(self.config.n_tasks_per_batch, len(tasks)),
                replace=False
            )

            meta_loss = 0.0
            for idx in task_indices:
                task = tasks[idx]
                gradients, support_loss, query_loss = self._compute_meta_gradient(task)

                # Accumulate gradients
                for name, param in self.model.named_parameters():
                    if name in gradients:
                        if param.grad is None:
                            param.grad = gradients[name]
                        else:
                            param.grad += gradients[name]

                meta_loss += query_loss

            # Average gradients
            meta_loss /= len(task_indices)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= len(task_indices)

            # Meta-update
            self._meta_optimizer.step()

            training_history.append({
                "iteration": iteration + 1,
                "meta_loss": meta_loss
            })

            if (iteration + 1) % eval_interval == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations}, "
                    f"Meta-loss: {meta_loss:.4f}"
                )

        # Calculate provenance
        param_sum = sum(
            p.data.sum().item() for p in self.model.parameters()
        )
        provenance_hash = hashlib.sha256(
            f"{n_iterations}|{meta_loss}|{param_sum}".encode()
        ).hexdigest()

        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Meta-training complete. Final loss: {meta_loss:.4f}, "
            f"Time: {processing_time_ms:.0f}ms"
        )

        return MetaTrainingResult(
            n_iterations=n_iterations,
            final_meta_loss=meta_loss,
            training_history=training_history,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms
        )

    def reptile_train(
        self,
        tasks: List[TaskData],
        n_iterations: int = 1000,
        epsilon: float = 0.1
    ) -> MetaTrainingResult:
        """
        Perform REPTILE meta-training (simpler than MAML).

        Args:
            tasks: List of training tasks
            n_iterations: Number of iterations
            epsilon: Step size towards task-adapted parameters

        Returns:
            MetaTrainingResult
        """
        import torch

        start_time = datetime.utcnow()
        training_history = []

        logger.info(f"Starting REPTILE training with {len(tasks)} tasks")

        for iteration in range(n_iterations):
            # Sample a task
            task = tasks[np.random.randint(len(tasks))]

            support_X = np.array(task.support_X)
            support_y = np.array(task.support_y)

            # Store original params
            original_params = self._get_model_params()

            # Adapt on support set
            adapted_model, losses = self._inner_loop(support_X, support_y)

            # REPTILE update: move towards adapted parameters
            for (name, param), (_, adapted_param) in zip(
                self.model.named_parameters(),
                adapted_model.named_parameters()
            ):
                param.data = (
                    param.data + epsilon * (adapted_param.data - param.data)
                )

            training_history.append({
                "iteration": iteration + 1,
                "loss": losses[-1]
            })

            if (iteration + 1) % 100 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{n_iterations}, "
                    f"Loss: {losses[-1]:.4f}"
                )

        provenance_hash = hashlib.sha256(
            f"reptile|{n_iterations}|{losses[-1]}".encode()
        ).hexdigest()

        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        return MetaTrainingResult(
            n_iterations=n_iterations,
            final_meta_loss=losses[-1],
            training_history=training_history,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms
        )

    def get_adaptation_history(self) -> List[AdaptationResult]:
        """Get history of task adaptations."""
        return self._task_history.copy()

    def save_meta_state(self) -> bytes:
        """Save meta-learned state for persistence."""
        import pickle
        return pickle.dumps({
            "model_state": self._get_model_params(),
            "config": self.config.dict(),
            "history": [r.dict() for r in self._task_history]
        })

    def load_meta_state(self, data: bytes) -> None:
        """Load meta-learned state."""
        import pickle
        loaded = pickle.loads(data)
        self._set_model_params(loaded["model_state"])
        self.config = MetaLearnerConfig(**loaded["config"])


# Unit test stubs
class TestMetaLearner:
    """Unit tests for MetaLearner."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        class MockModel:
            def named_parameters(self):
                return []
            def parameters(self):
                return []

        learner = MetaLearner(MockModel())
        assert learner.config.method == MetaLearningMethod.MAML
        assert learner.config.n_inner_steps == 5

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        class MockModel:
            def named_parameters(self):
                return []
            def parameters(self):
                return []

        config = MetaLearnerConfig(
            method=MetaLearningMethod.REPTILE,
            n_inner_steps=10
        )
        learner = MetaLearner(MockModel(), config)
        assert learner.config.method == MetaLearningMethod.REPTILE

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        class MockModel:
            def named_parameters(self):
                return []
            def parameters(self):
                return []

        learner = MetaLearner(MockModel())

        X = np.array([[1.0, 2.0]])
        hash1 = learner._calculate_provenance("task1", X, 0.5)
        hash2 = learner._calculate_provenance("task1", X, 0.5)

        assert hash1 == hash2

    def test_task_data_validation(self):
        """Test TaskData validation."""
        task = TaskData(
            task_id="test",
            support_X=[[1.0, 2.0], [3.0, 4.0]],
            support_y=[0, 1],
            query_X=[[5.0, 6.0]],
            query_y=[0]
        )
        assert task.task_id == "test"
        assert len(task.support_X) == 2
