# -*- coding: utf-8 -*-
"""
Learning Rate Schedulers Module

This module provides advanced learning rate scheduling strategies for
GreenLang Process Heat agents, including cyclic learning rates, cosine
annealing, and drift-aware schedulers optimized for continual learning.

Proper learning rate scheduling is critical for achieving convergence
while preventing oscillation and maintaining model plasticity for
ongoing adaptation to new data distributions.

Example:
    >>> from greenlang.ml.self_learning import CyclicLRScheduler, OneCycleScheduler
    >>> scheduler = CyclicLRScheduler(optimizer, base_lr=0.001, max_lr=0.01)
    >>> for epoch in range(100):
    ...     train_one_epoch(model, optimizer)
    ...     scheduler.step()
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import math

logger = logging.getLogger(__name__)


class SchedulerMode(str, Enum):
    """Learning rate scheduler modes."""
    TRIANGULAR = "triangular"  # Linear increase/decrease
    TRIANGULAR2 = "triangular2"  # Halved amplitude each cycle
    EXP_RANGE = "exp_range"  # Exponential decay
    COSINE = "cosine"  # Cosine annealing
    COSINE_WARM = "cosine_warm"  # Cosine with warm restarts
    ONE_CYCLE = "one_cycle"  # One-cycle policy
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class SchedulerConfig(BaseModel):
    """Base configuration for learning rate schedulers."""

    base_lr: float = Field(
        default=0.001,
        gt=0,
        description="Base (minimum) learning rate"
    )
    max_lr: float = Field(
        default=0.01,
        gt=0,
        description="Maximum learning rate"
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of warmup steps"
    )
    min_lr: float = Field(
        default=1e-7,
        gt=0,
        description="Absolute minimum learning rate"
    )
    enable_logging: bool = Field(
        default=True,
        description="Log learning rate changes"
    )

    @validator('max_lr')
    def max_lr_greater_than_base(cls, v, values):
        """Validate max_lr > base_lr."""
        if 'base_lr' in values and v <= values['base_lr']:
            raise ValueError('max_lr must be greater than base_lr')
        return v


class CyclicLRConfig(SchedulerConfig):
    """Configuration for Cyclic Learning Rate scheduler."""

    mode: SchedulerMode = Field(
        default=SchedulerMode.TRIANGULAR,
        description="Cycling mode"
    )
    step_size_up: int = Field(
        default=2000,
        ge=1,
        description="Steps in increasing half of cycle"
    )
    step_size_down: Optional[int] = Field(
        default=None,
        description="Steps in decreasing half (default: same as up)"
    )
    gamma: float = Field(
        default=0.99994,
        gt=0,
        le=1.0,
        description="Decay factor for exp_range mode"
    )
    scale_fn: Optional[str] = Field(
        default=None,
        description="Custom scaling function name"
    )


class CosineAnnealingConfig(SchedulerConfig):
    """Configuration for Cosine Annealing scheduler."""

    T_max: int = Field(
        default=100,
        ge=1,
        description="Maximum number of iterations for cosine cycle"
    )
    T_mult: float = Field(
        default=2.0,
        ge=1.0,
        description="Factor to increase T_max after each restart"
    )
    eta_min: float = Field(
        default=1e-6,
        ge=0,
        description="Minimum learning rate"
    )
    warm_restarts: bool = Field(
        default=True,
        description="Enable warm restarts"
    )


class OneCycleConfig(SchedulerConfig):
    """Configuration for One-Cycle Policy scheduler."""

    total_steps: int = Field(
        default=10000,
        ge=1,
        description="Total number of training steps"
    )
    pct_start: float = Field(
        default=0.3,
        gt=0,
        lt=1.0,
        description="Percentage of cycle spent increasing LR"
    )
    anneal_strategy: str = Field(
        default="cos",
        description="Annealing strategy: 'cos' or 'linear'"
    )
    div_factor: float = Field(
        default=25.0,
        gt=0,
        description="Initial LR = max_lr / div_factor"
    )
    final_div_factor: float = Field(
        default=1e4,
        gt=0,
        description="Final LR = initial_lr / final_div_factor"
    )


class ReduceOnPlateauConfig(SchedulerConfig):
    """Configuration for Reduce on Plateau scheduler with drift awareness."""

    mode: str = Field(
        default="min",
        description="'min' for loss, 'max' for metrics like accuracy"
    )
    factor: float = Field(
        default=0.1,
        gt=0,
        lt=1.0,
        description="Factor to reduce LR by"
    )
    patience: int = Field(
        default=10,
        ge=1,
        description="Epochs to wait before reduction"
    )
    threshold: float = Field(
        default=1e-4,
        gt=0,
        description="Threshold for measuring improvement"
    )
    cooldown: int = Field(
        default=0,
        ge=0,
        description="Epochs to wait after LR reduction"
    )
    drift_sensitivity: float = Field(
        default=0.1,
        gt=0,
        description="How much to boost LR when drift detected"
    )
    drift_reset_patience: bool = Field(
        default=True,
        description="Reset patience counter on drift detection"
    )


class SchedulerState(BaseModel):
    """State of a learning rate scheduler."""

    current_lr: float = Field(
        ...,
        description="Current learning rate"
    )
    step_count: int = Field(
        default=0,
        description="Total steps taken"
    )
    epoch_count: int = Field(
        default=0,
        description="Total epochs completed"
    )
    cycle_count: int = Field(
        default=0,
        description="Number of cycles completed"
    )
    warmup_complete: bool = Field(
        default=False,
        description="Whether warmup is complete"
    )
    last_reset: Optional[int] = Field(
        default=None,
        description="Step at last reset/restart"
    )
    lr_history: List[float] = Field(
        default_factory=list,
        description="History of learning rates"
    )


class BaseLRScheduler(ABC):
    """
    Abstract base class for learning rate schedulers.

    This provides a common interface for all GreenLang learning rate
    schedulers, ensuring consistent API and state management.

    Subclasses must implement:
    - get_lr(): Calculate current learning rate
    - step(): Advance scheduler by one step
    """

    def __init__(self, optimizer: Any, config: SchedulerConfig):
        """
        Initialize base scheduler.

        Args:
            optimizer: PyTorch optimizer or compatible
            config: Scheduler configuration
        """
        self.optimizer = optimizer
        self.config = config
        self.state = SchedulerState(current_lr=config.base_lr)
        self._lr_history: List[float] = []

        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def get_lr(self) -> float:
        """
        Calculate current learning rate.

        Returns:
            Current learning rate value
        """
        pass

    @abstractmethod
    def step(self, metrics: Optional[float] = None) -> float:
        """
        Advance scheduler by one step.

        Args:
            metrics: Optional metric value (for plateau schedulers)

        Returns:
            New learning rate
        """
        pass

    def _set_optimizer_lr(self, lr: float) -> None:
        """Set learning rate in optimizer."""
        try:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        except (AttributeError, TypeError):
            # Non-PyTorch optimizer
            pass

    def _apply_warmup(self, lr: float) -> float:
        """Apply warmup if in warmup phase."""
        if self.config.warmup_steps > 0 and self.state.step_count < self.config.warmup_steps:
            warmup_factor = self.state.step_count / self.config.warmup_steps
            return self.config.min_lr + warmup_factor * (lr - self.config.min_lr)
        return lr

    def get_state(self) -> SchedulerState:
        """Get current scheduler state."""
        self.state.lr_history = self._lr_history[-100:]  # Keep last 100
        return self.state

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.state = SchedulerState(current_lr=self.config.base_lr)
        self._lr_history.clear()
        logger.info(f"{self.__class__.__name__} reset")

    def get_lr_history(self, limit: Optional[int] = None) -> List[float]:
        """Get learning rate history."""
        if limit:
            return self._lr_history[-limit:]
        return self._lr_history.copy()


class CyclicLRScheduler(BaseLRScheduler):
    """
    Cyclic Learning Rate Scheduler.

    Implements cyclical learning rates as described in
    "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017).

    The learning rate cycles between base_lr and max_lr, which can help
    escape saddle points and local minima, leading to better convergence.

    Modes:
    - triangular: Linear increase then decrease
    - triangular2: Amplitude halved each cycle
    - exp_range: Exponential decay within cycle

    Example:
        >>> scheduler = CyclicLRScheduler(
        ...     optimizer,
        ...     config=CyclicLRConfig(
        ...         base_lr=0.001,
        ...         max_lr=0.01,
        ...         step_size_up=2000,
        ...         mode=SchedulerMode.TRIANGULAR
        ...     )
        ... )
        >>> for step in range(10000):
        ...     loss = train_step(model, optimizer)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Any,
        config: Optional[CyclicLRConfig] = None,
        base_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        step_size_up: Optional[int] = None
    ):
        """
        Initialize Cyclic LR scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Full configuration
            base_lr: Override base learning rate
            max_lr: Override maximum learning rate
            step_size_up: Override step size
        """
        config = config or CyclicLRConfig()

        if base_lr is not None:
            config.base_lr = base_lr
        if max_lr is not None:
            config.max_lr = max_lr
        if step_size_up is not None:
            config.step_size_up = step_size_up

        super().__init__(optimizer, config)
        self.config: CyclicLRConfig = config

        # Calculate step size down
        self._step_size_down = config.step_size_down or config.step_size_up
        self._total_cycle_size = config.step_size_up + self._step_size_down

    def _get_scale_fn(self) -> Callable[[float], float]:
        """Get scaling function based on mode."""
        if self.config.mode == SchedulerMode.TRIANGULAR:
            return lambda x: 1.0
        elif self.config.mode == SchedulerMode.TRIANGULAR2:
            return lambda x: 1.0 / (2.0 ** (self.state.cycle_count))
        elif self.config.mode == SchedulerMode.EXP_RANGE:
            return lambda x: self.config.gamma ** x
        else:
            return lambda x: 1.0

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        cycle = math.floor(1 + self.state.step_count / self._total_cycle_size)
        x = 1 + self.state.step_count / self._total_cycle_size - cycle

        if x <= self.config.step_size_up / self._total_cycle_size:
            # Increasing phase
            scale = x / (self.config.step_size_up / self._total_cycle_size)
        else:
            # Decreasing phase
            scale = (x - self.config.step_size_up / self._total_cycle_size) / (
                self._step_size_down / self._total_cycle_size
            )
            scale = 1 - scale

        scale_fn = self._get_scale_fn()
        amplitude = (self.config.max_lr - self.config.base_lr) * scale_fn(self.state.step_count)

        lr = self.config.base_lr + amplitude * scale
        return max(lr, self.config.min_lr)

    def step(self, metrics: Optional[float] = None) -> float:
        """
        Advance scheduler by one step.

        Args:
            metrics: Ignored for cyclic scheduler

        Returns:
            New learning rate
        """
        # Update cycle count
        new_cycle = math.floor(1 + (self.state.step_count + 1) / self._total_cycle_size)
        if new_cycle > self.state.cycle_count:
            self.state.cycle_count = new_cycle
            if self.config.enable_logging:
                logger.info(f"CyclicLR: Starting cycle {new_cycle}")

        self.state.step_count += 1

        # Apply warmup if needed
        lr = self.get_lr()
        lr = self._apply_warmup(lr)

        self.state.current_lr = lr
        self._lr_history.append(lr)
        self._set_optimizer_lr(lr)

        return lr


class CosineAnnealingScheduler(BaseLRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Warm Restarts.

    Implements SGDR: Stochastic Gradient Descent with Warm Restarts
    (Loshchilov & Hutter, 2017).

    The learning rate follows a cosine curve, optionally restarting
    at the maximum value. This can help escape local minima and
    improves exploration of the loss landscape.

    Example:
        >>> scheduler = CosineAnnealingScheduler(
        ...     optimizer,
        ...     config=CosineAnnealingConfig(
        ...         base_lr=0.001,
        ...         max_lr=0.01,
        ...         T_max=100,
        ...         warm_restarts=True,
        ...         T_mult=2.0
        ...     )
        ... )
        >>> for epoch in range(500):
        ...     train_one_epoch(model)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Any,
        config: Optional[CosineAnnealingConfig] = None,
        T_max: Optional[int] = None,
        eta_min: Optional[float] = None
    ):
        """
        Initialize Cosine Annealing scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Full configuration
            T_max: Override cycle length
            eta_min: Override minimum learning rate
        """
        config = config or CosineAnnealingConfig()

        if T_max is not None:
            config.T_max = T_max
        if eta_min is not None:
            config.eta_min = eta_min

        super().__init__(optimizer, config)
        self.config: CosineAnnealingConfig = config

        self._current_T_max = config.T_max
        self._T_cur = 0

    def get_lr(self) -> float:
        """Calculate current learning rate using cosine annealing."""
        if self._T_cur >= self._current_T_max:
            return self.config.eta_min

        # Cosine annealing formula
        lr = self.config.eta_min + (self.config.max_lr - self.config.eta_min) * (
            1 + math.cos(math.pi * self._T_cur / self._current_T_max)
        ) / 2

        return max(lr, self.config.min_lr)

    def step(self, metrics: Optional[float] = None) -> float:
        """
        Advance scheduler by one step (typically per epoch).

        Args:
            metrics: Ignored for cosine scheduler

        Returns:
            New learning rate
        """
        self._T_cur += 1
        self.state.step_count += 1

        # Check for restart
        if self.config.warm_restarts and self._T_cur >= self._current_T_max:
            self._T_cur = 0
            self._current_T_max = int(self._current_T_max * self.config.T_mult)
            self.state.cycle_count += 1
            self.state.last_reset = self.state.step_count

            if self.config.enable_logging:
                logger.info(
                    f"CosineAnnealing: Warm restart, new T_max={self._current_T_max}"
                )

        # Apply warmup if needed
        lr = self.get_lr()
        lr = self._apply_warmup(lr)

        self.state.current_lr = lr
        self._lr_history.append(lr)
        self._set_optimizer_lr(lr)

        return lr


class OneCycleScheduler(BaseLRScheduler):
    """
    One-Cycle Learning Rate Policy Scheduler.

    Implements the 1cycle policy from "Super-Convergence: Very Fast
    Training of Neural Networks Using Large Learning Rates" (Smith, 2018).

    This policy uses a single cycle of increasing then decreasing LR,
    with an optional final annealing phase. It often achieves faster
    convergence with better final performance.

    Example:
        >>> scheduler = OneCycleScheduler(
        ...     optimizer,
        ...     config=OneCycleConfig(
        ...         max_lr=0.01,
        ...         total_steps=10000,
        ...         pct_start=0.3,
        ...         anneal_strategy="cos"
        ...     )
        ... )
        >>> for step in range(10000):
        ...     loss = train_step(model, optimizer)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Any,
        config: Optional[OneCycleConfig] = None,
        max_lr: Optional[float] = None,
        total_steps: Optional[int] = None
    ):
        """
        Initialize One-Cycle scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Full configuration
            max_lr: Override maximum learning rate
            total_steps: Override total training steps
        """
        config = config or OneCycleConfig()

        if max_lr is not None:
            config.max_lr = max_lr
        if total_steps is not None:
            config.total_steps = total_steps

        # Calculate initial LR
        config.base_lr = config.max_lr / config.div_factor

        super().__init__(optimizer, config)
        self.config: OneCycleConfig = config

        # Calculate phase boundaries
        self._step_up_end = int(config.total_steps * config.pct_start)
        self._step_down_end = config.total_steps

        # Final LR
        self._final_lr = config.base_lr / config.final_div_factor

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        step = self.state.step_count

        if step <= self._step_up_end:
            # Phase 1: Increase from initial to max
            pct = step / self._step_up_end

            if self.config.anneal_strategy == "cos":
                lr = self.config.base_lr + (self.config.max_lr - self.config.base_lr) * (
                    1 - math.cos(math.pi * pct)
                ) / 2
            else:
                lr = self.config.base_lr + (self.config.max_lr - self.config.base_lr) * pct

        else:
            # Phase 2: Decrease from max to final
            pct = (step - self._step_up_end) / (self._step_down_end - self._step_up_end)
            pct = min(pct, 1.0)

            if self.config.anneal_strategy == "cos":
                lr = self._final_lr + (self.config.max_lr - self._final_lr) * (
                    1 + math.cos(math.pi * pct)
                ) / 2
            else:
                lr = self.config.max_lr - (self.config.max_lr - self._final_lr) * pct

        return max(lr, self.config.min_lr)

    def step(self, metrics: Optional[float] = None) -> float:
        """
        Advance scheduler by one step.

        Args:
            metrics: Ignored for one-cycle scheduler

        Returns:
            New learning rate
        """
        self.state.step_count += 1

        lr = self.get_lr()
        self.state.current_lr = lr
        self._lr_history.append(lr)
        self._set_optimizer_lr(lr)

        # Log phase transitions
        if self.config.enable_logging:
            if self.state.step_count == self._step_up_end:
                logger.info("OneCycle: Transitioning to annealing phase")

        return lr

    def is_complete(self) -> bool:
        """Check if one-cycle is complete."""
        return self.state.step_count >= self.config.total_steps


class ReduceOnPlateauScheduler(BaseLRScheduler):
    """
    Reduce Learning Rate on Plateau with Drift Awareness.

    Reduces learning rate when a metric has stopped improving,
    with special handling for concept drift scenarios common
    in continual learning for Process Heat applications.

    When drift is detected, the scheduler can:
    - Boost learning rate to adapt faster
    - Reset patience counter
    - Enter a more aggressive learning phase

    Example:
        >>> scheduler = ReduceOnPlateauScheduler(
        ...     optimizer,
        ...     config=ReduceOnPlateauConfig(
        ...         base_lr=0.01,
        ...         factor=0.1,
        ...         patience=10,
        ...         drift_sensitivity=0.1
        ...     )
        ... )
        >>> for epoch in range(100):
        ...     train_loss = train_one_epoch(model)
        ...     val_loss = validate(model)
        ...     scheduler.step(val_loss, drift_detected=check_drift())
    """

    def __init__(
        self,
        optimizer: Any,
        config: Optional[ReduceOnPlateauConfig] = None,
        factor: Optional[float] = None,
        patience: Optional[int] = None
    ):
        """
        Initialize Reduce on Plateau scheduler.

        Args:
            optimizer: PyTorch optimizer
            config: Full configuration
            factor: Override reduction factor
            patience: Override patience
        """
        config = config or ReduceOnPlateauConfig()

        if factor is not None:
            config.factor = factor
        if patience is not None:
            config.patience = patience

        super().__init__(optimizer, config)
        self.config: ReduceOnPlateauConfig = config

        self._best_metric: Optional[float] = None
        self._bad_epochs: int = 0
        self._cooldown_counter: int = 0
        self._reduction_count: int = 0
        self._drift_detected: bool = False

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.config.mode == "min":
            return current < best - self.config.threshold
        else:
            return current > best + self.config.threshold

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.state.current_lr

    def step(
        self,
        metrics: Optional[float] = None,
        drift_detected: bool = False
    ) -> float:
        """
        Advance scheduler based on metric value.

        Args:
            metrics: Validation metric (loss or accuracy)
            drift_detected: Whether concept drift was detected

        Returns:
            New learning rate
        """
        self.state.step_count += 1
        self._drift_detected = drift_detected

        if metrics is None:
            return self.state.current_lr

        # Handle drift detection
        if drift_detected:
            self._handle_drift()
            return self.state.current_lr

        # Check cooldown
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self.state.current_lr

        # Update best metric
        if self._best_metric is None:
            self._best_metric = metrics
            return self.state.current_lr

        # Check for improvement
        if self._is_better(metrics, self._best_metric):
            self._best_metric = metrics
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1

        # Reduce LR if patience exceeded
        if self._bad_epochs >= self.config.patience:
            self._reduce_lr()

        self._lr_history.append(self.state.current_lr)
        return self.state.current_lr

    def _reduce_lr(self) -> None:
        """Reduce learning rate."""
        old_lr = self.state.current_lr
        new_lr = old_lr * self.config.factor

        # Enforce minimum
        new_lr = max(new_lr, self.config.min_lr)

        if new_lr < old_lr:
            self.state.current_lr = new_lr
            self._set_optimizer_lr(new_lr)
            self._reduction_count += 1
            self._bad_epochs = 0
            self._cooldown_counter = self.config.cooldown

            if self.config.enable_logging:
                logger.info(
                    f"ReduceOnPlateau: Reducing LR from {old_lr:.6f} to {new_lr:.6f}"
                )

    def _handle_drift(self) -> None:
        """Handle drift detection - boost LR for faster adaptation."""
        old_lr = self.state.current_lr

        # Boost LR to adapt faster (but not above max)
        boost_factor = 1.0 + self.config.drift_sensitivity
        new_lr = min(old_lr * boost_factor, self.config.max_lr)

        if new_lr > old_lr:
            self.state.current_lr = new_lr
            self._set_optimizer_lr(new_lr)

            if self.config.enable_logging:
                logger.info(
                    f"ReduceOnPlateau: Drift detected, boosting LR from "
                    f"{old_lr:.6f} to {new_lr:.6f}"
                )

        # Optionally reset patience
        if self.config.drift_reset_patience:
            self._bad_epochs = 0
            self._best_metric = None

    def get_reduction_count(self) -> int:
        """Get number of LR reductions."""
        return self._reduction_count


class WarmupScheduler(BaseLRScheduler):
    """
    Warmup wrapper for any learning rate scheduler.

    Adds a linear warmup phase to any base scheduler, useful for
    stabilizing training in the initial steps.

    Example:
        >>> base_scheduler = CosineAnnealingScheduler(optimizer)
        >>> scheduler = WarmupScheduler(
        ...     optimizer,
        ...     base_scheduler,
        ...     warmup_steps=1000
        ... )
    """

    def __init__(
        self,
        optimizer: Any,
        base_scheduler: BaseLRScheduler,
        warmup_steps: int = 1000,
        warmup_start_lr: float = 1e-7
    ):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            base_scheduler: Scheduler to wrap
            warmup_steps: Number of warmup steps
            warmup_start_lr: Starting learning rate for warmup
        """
        config = SchedulerConfig(
            base_lr=warmup_start_lr,
            max_lr=base_scheduler.config.max_lr,
            warmup_steps=warmup_steps
        )
        super().__init__(optimizer, config)

        self.base_scheduler = base_scheduler
        self._warmup_start_lr = warmup_start_lr
        self._target_lr = base_scheduler.config.base_lr

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.state.step_count < self.config.warmup_steps:
            # Linear warmup
            progress = self.state.step_count / self.config.warmup_steps
            return self._warmup_start_lr + progress * (self._target_lr - self._warmup_start_lr)
        else:
            return self.base_scheduler.get_lr()

    def step(self, metrics: Optional[float] = None) -> float:
        """Advance scheduler by one step."""
        self.state.step_count += 1

        if self.state.step_count <= self.config.warmup_steps:
            lr = self.get_lr()
            self.state.warmup_complete = False

            if self.state.step_count == self.config.warmup_steps:
                self.state.warmup_complete = True
                if self.config.enable_logging:
                    logger.info(f"Warmup complete at step {self.state.step_count}")
        else:
            lr = self.base_scheduler.step(metrics)

        self.state.current_lr = lr
        self._lr_history.append(lr)
        self._set_optimizer_lr(lr)

        return lr


# Factory functions
def create_cyclic_scheduler(
    optimizer: Any,
    base_lr: float = 0.001,
    max_lr: float = 0.01,
    step_size: int = 2000
) -> CyclicLRScheduler:
    """Create a cyclic learning rate scheduler."""
    config = CyclicLRConfig(
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size,
        mode=SchedulerMode.TRIANGULAR
    )
    return CyclicLRScheduler(optimizer, config)


def create_cosine_annealing_scheduler(
    optimizer: Any,
    T_max: int = 100,
    eta_min: float = 1e-6,
    warm_restarts: bool = True
) -> CosineAnnealingScheduler:
    """Create a cosine annealing scheduler with warm restarts."""
    config = CosineAnnealingConfig(
        T_max=T_max,
        eta_min=eta_min,
        warm_restarts=warm_restarts
    )
    return CosineAnnealingScheduler(optimizer, config)


def create_one_cycle_scheduler(
    optimizer: Any,
    max_lr: float = 0.01,
    total_steps: int = 10000,
    pct_start: float = 0.3
) -> OneCycleScheduler:
    """Create a one-cycle policy scheduler."""
    config = OneCycleConfig(
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start
    )
    return OneCycleScheduler(optimizer, config)


def create_reduce_on_plateau_scheduler(
    optimizer: Any,
    factor: float = 0.1,
    patience: int = 10,
    drift_aware: bool = True
) -> ReduceOnPlateauScheduler:
    """Create a reduce on plateau scheduler with optional drift awareness."""
    config = ReduceOnPlateauConfig(
        factor=factor,
        patience=patience,
        drift_sensitivity=0.1 if drift_aware else 0.0
    )
    return ReduceOnPlateauScheduler(optimizer, config)


# Unit test stubs
class TestLRSchedulers:
    """Unit tests for learning rate schedulers."""

    def test_cyclic_lr_triangular(self):
        """Test cyclic LR with triangular mode."""
        scheduler = CyclicLRScheduler(
            optimizer=None,
            config=CyclicLRConfig(
                base_lr=0.001,
                max_lr=0.01,
                step_size_up=100
            )
        )

        lrs = []
        for _ in range(200):
            lrs.append(scheduler.step())

        # Should oscillate between base and max
        assert min(lrs) >= 0.001
        assert max(lrs) <= 0.01

    def test_cosine_annealing(self):
        """Test cosine annealing scheduler."""
        scheduler = CosineAnnealingScheduler(
            optimizer=None,
            config=CosineAnnealingConfig(
                max_lr=0.01,
                T_max=50,
                eta_min=1e-6
            )
        )

        lrs = []
        for _ in range(50):
            lrs.append(scheduler.step())

        # Should decrease following cosine
        assert lrs[0] < lrs[-1] or lrs[-1] <= 1e-5

    def test_one_cycle(self):
        """Test one-cycle scheduler."""
        scheduler = OneCycleScheduler(
            optimizer=None,
            config=OneCycleConfig(
                max_lr=0.01,
                total_steps=100,
                pct_start=0.3
            )
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.step())

        # Should increase then decrease
        assert lrs[30] > lrs[0]  # Increased
        assert lrs[-1] < lrs[30]  # Decreased

    def test_reduce_on_plateau(self):
        """Test reduce on plateau scheduler."""
        config = ReduceOnPlateauConfig(
            base_lr=0.01,
            factor=0.1,
            patience=3
        )
        scheduler = ReduceOnPlateauScheduler(optimizer=None, config=config)

        # Simulate plateau
        for _ in range(5):
            scheduler.step(1.0)  # Same metric, no improvement

        # Should have reduced LR
        assert scheduler.state.current_lr < 0.01

    def test_warmup(self):
        """Test warmup scheduler."""
        base = CosineAnnealingScheduler(optimizer=None)
        scheduler = WarmupScheduler(
            optimizer=None,
            base_scheduler=base,
            warmup_steps=10
        )

        lrs = []
        for _ in range(15):
            lrs.append(scheduler.step())

        # Should increase during warmup
        assert lrs[9] > lrs[0]
        assert scheduler.state.warmup_complete

    def test_scheduler_state(self):
        """Test scheduler state tracking."""
        scheduler = CyclicLRScheduler(optimizer=None)

        for _ in range(10):
            scheduler.step()

        state = scheduler.get_state()
        assert state.step_count == 10
        assert len(state.lr_history) > 0
