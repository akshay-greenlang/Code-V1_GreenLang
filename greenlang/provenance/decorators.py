# -*- coding: utf-8 -*-
"""
GreenLang Provenance - Decorators Module
Decorators for automatic provenance tracking.
"""

from functools import wraps
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timezone
import logging

from .records import ProvenanceContext, ProvenanceRecord
from .hashing import hash_data

logger = logging.getLogger(__name__)


# ============================================================================
# @traced DECORATOR
# ============================================================================

def traced(
    record_id: Optional[str] = None,
    save_path: Optional[str] = None,
    track_inputs: bool = True,
    track_outputs: bool = True
):
    """
    Decorator to automatically track provenance for functions.

    Wraps a function to automatically capture:
    - Execution environment
    - Input arguments
    - Output results
    - Execution time
    - Any exceptions

    Args:
        record_id: Optional custom record ID
        save_path: Optional path to save provenance record
        track_inputs: Whether to track input arguments
        track_outputs: Whether to track output results

    Example:
        >>> @traced(save_path="provenance.json")
        ... def process_data(input_file, config):
        ...     # Process data
        ...     return results

        >>> # Provenance automatically recorded when function is called
        >>> results = process_data("data.csv", config={...})
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create provenance context
            ctx = ProvenanceContext(
                name=func.__name__,
                record_id=record_id
            )

            # Record start time
            start_time = datetime.now(timezone.utc)

            # Track inputs if enabled
            if track_inputs:
                ctx.metadata["inputs"] = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Track outputs if enabled
                if track_outputs:
                    ctx.metadata["outputs"] = str(result)

                ctx.metadata["status"] = "success"

                return result

            except Exception as e:
                # Record error
                ctx.metadata["status"] = "failed"
                ctx.metadata["error"] = str(e)
                ctx.metadata["error_type"] = type(e).__name__

                # Re-raise
                raise

            finally:
                # Record end time and duration
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                # Add execution metadata
                ctx.record_agent_execution(
                    agent_name=func.__name__,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration
                )

                # Finalize and save
                provenance = ctx.finalize(output_path=save_path)

                logger.info(f"Provenance tracked for {func.__name__}: {provenance.record_id}")

        return wrapper

    return decorator


# ============================================================================
# @track_provenance DECORATOR (For Methods)
# ============================================================================

def track_provenance(
    context_attr: str = "_provenance_context",
    save_on_completion: bool = False
):
    """
    Decorator to track provenance for class methods.

    Designed to work with classes that have a provenance context attribute.
    Automatically records method execution in the context.

    Args:
        context_attr: Name of the context attribute on the class
        save_on_completion: Whether to save provenance when method completes

    Example:
        >>> class DataPipeline:
        ...     def __init__(self):
        ...         self._provenance_context = ProvenanceContext("pipeline")
        ...
        ...     @track_provenance()
        ...     def load_data(self, path):
        ...         # Load data
        ...         return data
        ...
        ...     @track_provenance()
        ...     def transform_data(self, data):
        ...         # Transform data
        ...         return transformed
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get context from instance
            if not hasattr(self, context_attr):
                # Create context if it doesn't exist
                setattr(self, context_attr, ProvenanceContext(name=self.__class__.__name__))

            ctx = getattr(self, context_attr)

            # Record start time
            start_time = datetime.now(timezone.utc)

            try:
                # Execute method
                result = func(self, *args, **kwargs)

                # Record execution
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                ctx.record_agent_execution(
                    agent_name=func.__name__,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    metadata={
                        "class": self.__class__.__name__,
                        "method": func.__name__
                    }
                )

                # Save if requested
                if save_on_completion:
                    ctx.finalize(output_path=f"{ctx.record_id}_provenance.json")

                return result

            except Exception as e:
                # Record error but don't save yet
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                ctx.record_agent_execution(
                    agent_name=func.__name__,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    metadata={
                        "class": self.__class__.__name__,
                        "method": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )

                # Re-raise
                raise

        return wrapper

    return decorator


# ============================================================================
# CONTEXT MANAGER FOR PROVENANCE TRACKING
# ============================================================================

class provenance_tracker:
    """
    Context manager for provenance tracking.

    Provides a convenient way to track provenance for a block of code.

    Example:
        >>> with provenance_tracker("my_operation") as ctx:
        ...     # Do work
        ...     data = load_data("input.csv")
        ...     ctx.record_input("input.csv", {"rows": len(data)})
        ...
        ...     # Process
        ...     result = process(data)
        ...     ctx.record_output("output.csv", {"rows": len(result)})
        ...
        >>> # Provenance automatically saved
    """

    def __init__(
        self,
        name: str,
        record_id: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Initialize provenance tracker.

        Args:
            name: Operation name
            record_id: Optional record ID
            save_path: Optional path to save provenance
        """
        self.name = name
        self.record_id = record_id
        self.save_path = save_path
        self.context: Optional[ProvenanceContext] = None
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> ProvenanceContext:
        """Enter context and start tracking."""
        self.context = ProvenanceContext(name=self.name, record_id=self.record_id)
        self.start_time = datetime.now(timezone.utc)

        logger.debug(f"Started provenance tracking: {self.context.record_id}")

        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and save provenance."""
        if self.context and self.start_time:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            # Record overall execution
            status = "success" if exc_type is None else "failed"
            self.context.metadata["status"] = status
            self.context.metadata["duration_seconds"] = duration

            if exc_type:
                self.context.metadata["error"] = str(exc_val)
                self.context.metadata["error_type"] = exc_type.__name__

            # Save provenance
            provenance = self.context.finalize(output_path=self.save_path)

            logger.info(f"Provenance saved: {provenance.record_id} (status: {status})")

        # Don't suppress exceptions
        return False


# ============================================================================
# HELPER DECORATORS
# ============================================================================

def record_inputs(ctx_attr: str = "_provenance_context"):
    """
    Decorator to automatically record function inputs in provenance context.

    Args:
        ctx_attr: Name of context attribute

    Example:
        >>> class Pipeline:
        ...     def __init__(self):
        ...         self._provenance_context = ProvenanceContext("pipeline")
        ...
        ...     @record_inputs()
        ...     def process(self, input_file):
        ...         # Automatically records input_file
        ...         pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get context
            if hasattr(self, ctx_attr):
                ctx = getattr(self, ctx_attr)

                # Record inputs
                if args:
                    ctx.metadata[f"{func.__name__}_inputs_args"] = [str(arg) for arg in args]
                if kwargs:
                    ctx.metadata[f"{func.__name__}_inputs_kwargs"] = {
                        k: str(v) for k, v in kwargs.items()
                    }

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def record_outputs(ctx_attr: str = "_provenance_context"):
    """
    Decorator to automatically record function outputs in provenance context.

    Args:
        ctx_attr: Name of context attribute

    Example:
        >>> @record_outputs()
        ... def compute(self, x):
        ...     return x * 2  # Automatically records output
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Get context and record output
            if hasattr(self, ctx_attr):
                ctx = getattr(self, ctx_attr)
                ctx.metadata[f"{func.__name__}_output"] = str(result)

            return result

        return wrapper

    return decorator
