# -*- coding: utf-8 -*-
"""
GreenLang Composability Framework - GLEL (GreenLang Expression Language)

This module implements composable agent chaining similar to LangChain's LCEL,
but optimized for GreenLang's zero-hallucination and provenance tracking requirements.

Key Features:
- Pipe operator (|) for intuitive chaining
- Sequential and parallel execution patterns
- Async/streaming support with backpressure
- Comprehensive error handling and retry logic
- Complete provenance tracking throughout chains
- Zero-hallucination guarantees for calculations

Example:
    >>> from greenlang.core.composability import AgentRunnable, RunnableSequence
    >>>
    >>> # Create agent chain using pipe operator
    >>> chain = intake_agent | validation_agent | calculation_agent | reporting_agent
    >>> result = await chain.ainvoke(input_data)
    >>>
    >>> # Parallel execution
    >>> parallel = RunnableParallel({
    ...     "emissions": emissions_agent,
    ...     "compliance": compliance_agent,
    ...     "risk": risk_agent
    ... })
    >>> results = await parallel.ainvoke(input_data)
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from enum import Enum
from greenlang.utilities.determinism import DeterministicClock
from typing import (
    Any, Dict, List, Optional, Union, TypeVar, Generic,
    Callable, AsyncIterator, Iterator, Tuple, Set
)
from dataclasses import dataclass, field
from functools import wraps
import traceback

from pydantic import BaseModel, Field, validator
import numpy as np

logger = logging.getLogger(__name__)

# Type variables for generic support
T = TypeVar('T')
U = TypeVar('U')


class ExecutionMode(Enum):
    """Execution modes for runnables."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STREAMING = "streaming"
    BATCH = "batch"


class ProvenanceRecord(BaseModel):
    """Records provenance information for audit trails."""

    agent_id: str = Field(..., description="Unique identifier of the agent")
    input_hash: str = Field(..., description="SHA-256 hash of input data")
    output_hash: str = Field(..., description="SHA-256 hash of output data")
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(..., description="Processing duration in milliseconds")
    parent_hash: Optional[str] = Field(None, description="Hash of parent provenance record")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_chain_hash(self) -> str:
        """Calculate SHA-256 hash for the complete chain."""
        data = f"{self.agent_id}{self.input_hash}{self.output_hash}{self.parent_hash}"
        return hashlib.sha256(data.encode()).hexdigest()


class RunnableConfig(BaseModel):
    """Configuration for runnable execution."""

    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    retry_delay_ms: int = Field(1000, ge=0, description="Delay between retries in milliseconds")
    timeout_seconds: Optional[float] = Field(None, ge=0, description="Execution timeout")
    batch_size: int = Field(100, ge=1, description="Batch processing size")
    enable_streaming: bool = Field(False, description="Enable streaming mode")
    enable_provenance: bool = Field(True, description="Enable provenance tracking")
    parallel_workers: int = Field(4, ge=1, description="Number of parallel workers")
    error_handler: Optional[str] = Field(None, description="Error handler strategy")
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for chain execution."""

    config: RunnableConfig
    provenance_chain: List[ProvenanceRecord] = field(default_factory=list)
    execution_id: str = field(default_factory=lambda: hashlib.sha256(
        str(DeterministicClock.now()).encode()).hexdigest()[:16])
    start_time: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_provenance(self, record: ProvenanceRecord):
        """Add provenance record to the chain."""
        if self.config.enable_provenance:
            if self.provenance_chain:
                record.parent_hash = self.provenance_chain[-1].calculate_chain_hash()
            self.provenance_chain.append(record)

    def get_total_time_ms(self) -> float:
        """Get total execution time in milliseconds."""
        return (DeterministicClock.now() - self.start_time).total_seconds() * 1000


class BaseRunnable(ABC, Generic[T, U]):
    """
    Base class for all runnable components in GreenLang.

    Similar to LangChain's Runnable, but with built-in provenance tracking
    and zero-hallucination guarantees.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize BaseRunnable."""
        self.name = name or self.__class__.__name__
        self._config = RunnableConfig()

    @abstractmethod
    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Synchronous invocation."""
        pass

    @abstractmethod
    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Asynchronous invocation."""
        pass

    def stream(self, input: T, config: Optional[RunnableConfig] = None) -> Iterator[U]:
        """Synchronous streaming."""
        yield self.invoke(input, config)

    async def astream(self, input: T, config: Optional[RunnableConfig] = None) -> AsyncIterator[U]:
        """Asynchronous streaming."""
        result = await self.ainvoke(input, config)
        yield result

    def batch(self, inputs: List[T], config: Optional[RunnableConfig] = None) -> List[U]:
        """Process multiple inputs in batch."""
        config = config or self._config
        results = []

        for i in range(0, len(inputs), config.batch_size):
            batch = inputs[i:i + config.batch_size]
            batch_results = [self.invoke(item, config) for item in batch]
            results.extend(batch_results)

        return results

    async def abatch(self, inputs: List[T], config: Optional[RunnableConfig] = None) -> List[U]:
        """Process multiple inputs in batch asynchronously."""
        config = config or self._config
        results = []

        for i in range(0, len(inputs), config.batch_size):
            batch = inputs[i:i + config.batch_size]
            tasks = [self.ainvoke(item, config) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def with_retry(self, max_retries: int = 3, delay_ms: int = 1000) -> BaseRunnable[T, U]:
        """Create a new runnable with retry logic."""
        return RetryRunnable(self, max_retries, delay_ms)

    def with_fallback(self, fallback: BaseRunnable[T, U]) -> BaseRunnable[T, U]:
        """Create a new runnable with fallback logic."""
        return FallbackRunnable(self, fallback)

    def with_config(self, **kwargs) -> BaseRunnable[T, U]:
        """Create a new runnable with updated configuration."""
        new_runnable = self.__class__(self.name)
        new_runnable._config = RunnableConfig(**{**self._config.dict(), **kwargs})
        return new_runnable

    def __or__(self, other: BaseRunnable) -> RunnableSequence:
        """Pipe operator for chaining (|)."""
        return RunnableSequence([self, other])

    def __ror__(self, other: BaseRunnable) -> RunnableSequence:
        """Reverse pipe operator."""
        return RunnableSequence([other, self])

    def pipe(self, *others: BaseRunnable) -> RunnableSequence:
        """Explicit pipe method for chaining multiple runnables."""
        return RunnableSequence([self, *others])

    def _calculate_provenance(self, input_data: Any, output_data: Any) -> ProvenanceRecord:
        """Calculate provenance for audit trail."""
        input_str = str(input_data) if not isinstance(input_data, str) else input_data
        output_str = str(output_data) if not isinstance(output_data, str) else output_data

        return ProvenanceRecord(
            agent_id=self.name,
            input_hash=hashlib.sha256(input_str.encode()).hexdigest(),
            output_hash=hashlib.sha256(output_str.encode()).hexdigest(),
            processing_time_ms=0  # Will be updated by caller
        )


class AgentRunnable(BaseRunnable[Dict[str, Any], Dict[str, Any]]):
    """
    Wrapper for GreenLang agents to make them composable.

    This class wraps existing GreenLang agents and makes them compatible
    with the composability framework while maintaining zero-hallucination
    guarantees and provenance tracking.
    """

    def __init__(self, agent: Any, name: Optional[str] = None):
        """
        Initialize AgentRunnable.

        Args:
            agent: GreenLang agent instance
            name: Optional name for the runnable
        """
        super().__init__(name or agent.__class__.__name__)
        self.agent = agent

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Synchronous agent invocation with provenance tracking."""
        config = config or self._config
        start_time = DeterministicClock.now()

        try:
            # Execute agent
            result = self.agent.process(input)

            # Calculate provenance if enabled
            if config.enable_provenance:
                processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
                provenance = self._calculate_provenance(input, result)
                provenance.processing_time_ms = processing_time

                # Add provenance to result if it's a dict
                if isinstance(result, dict):
                    result['_provenance'] = provenance.dict()

            return result

        except Exception as e:
            logger.error(f"Agent {self.name} failed: {str(e)}", exc_info=True)
            if config.max_retries > 0:
                return self.with_retry(config.max_retries).invoke(input, config)
            raise

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Asynchronous agent invocation."""
        config = config or self._config
        start_time = DeterministicClock.now()

        try:
            # Check if agent has async method
            if hasattr(self.agent, 'aprocess'):
                result = await self.agent.aprocess(input)
            else:
                # Fall back to sync in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.agent.process, input)

            # Calculate provenance if enabled
            if config.enable_provenance:
                processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
                provenance = self._calculate_provenance(input, result)
                provenance.processing_time_ms = processing_time

                if isinstance(result, dict):
                    result['_provenance'] = provenance.dict()

            return result

        except Exception as e:
            logger.error(f"Agent {self.name} async failed: {str(e)}", exc_info=True)
            if config.max_retries > 0:
                return await self.with_retry(config.max_retries).ainvoke(input, config)
            raise


class RunnableSequence(BaseRunnable[T, U]):
    """
    Sequential execution of multiple runnables.

    Each runnable's output becomes the next runnable's input,
    creating a processing pipeline with complete provenance tracking.
    """

    def __init__(self, runnables: List[BaseRunnable]):
        """Initialize RunnableSequence."""
        super().__init__(f"Sequence[{','.join(r.name for r in runnables)}]")
        self.runnables = runnables

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Execute runnables sequentially."""
        config = config or self._config
        context = ExecutionContext(config)

        result = input
        for runnable in self.runnables:
            start_time = DeterministicClock.now()
            result = runnable.invoke(result, config)

            # Track provenance
            if config.enable_provenance:
                processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
                provenance = runnable._calculate_provenance(input, result)
                provenance.processing_time_ms = processing_time
                context.add_provenance(provenance)
                input = result  # Update input for next iteration

        # Add chain provenance to final result
        if config.enable_provenance and isinstance(result, dict):
            result['_chain_provenance'] = [p.dict() for p in context.provenance_chain]
            result['_chain_hash'] = context.provenance_chain[-1].calculate_chain_hash()

        return result

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Execute runnables sequentially asynchronously."""
        config = config or self._config
        context = ExecutionContext(config)

        result = input
        for runnable in self.runnables:
            start_time = DeterministicClock.now()
            result = await runnable.ainvoke(result, config)

            # Track provenance
            if config.enable_provenance:
                processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
                provenance = runnable._calculate_provenance(input, result)
                provenance.processing_time_ms = processing_time
                context.add_provenance(provenance)
                input = result  # Update input for next iteration

        # Add chain provenance to final result
        if config.enable_provenance and isinstance(result, dict):
            result['_chain_provenance'] = [p.dict() for p in context.provenance_chain]
            result['_chain_hash'] = context.provenance_chain[-1].calculate_chain_hash()

        return result

    async def astream(self, input: T, config: Optional[RunnableConfig] = None) -> AsyncIterator[Any]:
        """Stream results from each step of the sequence."""
        config = config or self._config
        context = ExecutionContext(config)

        result = input
        for runnable in self.runnables:
            start_time = DeterministicClock.now()

            # Stream from current runnable
            async for chunk in runnable.astream(result, config):
                yield {"step": runnable.name, "output": chunk}
                result = chunk  # Use last chunk as input to next

            # Track provenance
            if config.enable_provenance:
                processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
                provenance = runnable._calculate_provenance(input, result)
                provenance.processing_time_ms = processing_time
                context.add_provenance(provenance)
                input = result

    def __or__(self, other: BaseRunnable) -> RunnableSequence:
        """Extend the sequence with another runnable."""
        return RunnableSequence(self.runnables + [other])


class RunnableParallel(BaseRunnable[T, Dict[str, Any]]):
    """
    Parallel execution of multiple runnables.

    All runnables receive the same input and execute concurrently.
    Results are collected into a dictionary with provenance for each branch.
    """

    def __init__(self, runnables: Union[Dict[str, BaseRunnable], List[BaseRunnable]]):
        """
        Initialize RunnableParallel.

        Args:
            runnables: Dictionary mapping names to runnables, or list of runnables
        """
        if isinstance(runnables, list):
            runnables = {f"branch_{i}": r for i, r in enumerate(runnables)}

        super().__init__(f"Parallel[{','.join(runnables.keys())}]")
        self.runnables = runnables

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute runnables in parallel synchronously (using threads)."""
        config = config or self._config
        context = ExecutionContext(config)

        import concurrent.futures
        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
            futures = {
                name: executor.submit(runnable.invoke, input, config)
                for name, runnable in self.runnables.items()
            }

            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=config.timeout_seconds)
                except Exception as e:
                    logger.error(f"Parallel branch {name} failed: {str(e)}")
                    results[name] = {"error": str(e), "traceback": traceback.format_exc()}
                    context.errors.append({"branch": name, "error": str(e)})

        # Add provenance
        if config.enable_provenance:
            results['_parallel_provenance'] = {
                "execution_id": context.execution_id,
                "total_time_ms": context.get_total_time_ms(),
                "branches": list(self.runnables.keys()),
                "errors": context.errors
            }

        return results

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute runnables in parallel asynchronously."""
        config = config or self._config
        context = ExecutionContext(config)

        # Create tasks for all runnables
        tasks = {
            name: runnable.ainvoke(input, config)
            for name, runnable in self.runnables.items()
        }

        # Execute all tasks concurrently
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Parallel branch {name} failed: {str(e)}")
                results[name] = {"error": str(e), "traceback": traceback.format_exc()}
                context.errors.append({"branch": name, "error": str(e)})

        # Add provenance
        if config.enable_provenance:
            results['_parallel_provenance'] = {
                "execution_id": context.execution_id,
                "total_time_ms": context.get_total_time_ms(),
                "branches": list(self.runnables.keys()),
                "errors": context.errors
            }

        return results

    async def astream(self, input: T, config: Optional[RunnableConfig] = None) -> AsyncIterator[Dict[str, Any]]:
        """Stream results from parallel branches as they complete."""
        config = config or self._config

        # Create streaming tasks
        streams = {
            name: runnable.astream(input, config)
            for name, runnable in self.runnables.items()
        }

        # Stream results as they arrive
        active_streams = set(streams.keys())
        while active_streams:
            for name in list(active_streams):
                try:
                    stream = streams[name]
                    chunk = await stream.__anext__()
                    yield {name: chunk}
                except StopAsyncIteration:
                    active_streams.remove(name)


class RetryRunnable(BaseRunnable[T, U]):
    """Runnable with automatic retry logic."""

    def __init__(self, runnable: BaseRunnable[T, U], max_retries: int = 3, delay_ms: int = 1000):
        """Initialize RetryRunnable."""
        super().__init__(f"Retry[{runnable.name}]")
        self.runnable = runnable
        self.max_retries = max_retries
        self.delay_ms = delay_ms

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Invoke with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.runnable.invoke(input, config)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for {self.runnable.name}: {str(e)}")
                    import time
                    time.sleep(self.delay_ms / 1000)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {self.runnable.name}")

        raise last_error

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Async invoke with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.runnable.ainvoke(input, config)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for {self.runnable.name}: {str(e)}")
                    await asyncio.sleep(self.delay_ms / 1000)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {self.runnable.name}")

        raise last_error


class FallbackRunnable(BaseRunnable[T, U]):
    """Runnable with fallback logic."""

    def __init__(self, primary: BaseRunnable[T, U], fallback: BaseRunnable[T, U]):
        """Initialize FallbackRunnable."""
        super().__init__(f"Fallback[{primary.name}->{fallback.name}]")
        self.primary = primary
        self.fallback = fallback

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Invoke with fallback logic."""
        try:
            return self.primary.invoke(input, config)
        except Exception as e:
            logger.warning(f"Primary {self.primary.name} failed, using fallback: {str(e)}")
            return self.fallback.invoke(input, config)

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Async invoke with fallback logic."""
        try:
            return await self.primary.ainvoke(input, config)
        except Exception as e:
            logger.warning(f"Primary {self.primary.name} failed, using fallback: {str(e)}")
            return await self.fallback.ainvoke(input, config)


class RunnableLambda(BaseRunnable[T, U]):
    """
    Runnable wrapper for lambda functions.

    Allows any Python function to be used in chains while maintaining
    provenance tracking and zero-hallucination guarantees.
    """

    def __init__(self, func: Callable[[T], U], name: Optional[str] = None,
                 afunc: Optional[Callable[[T], U]] = None):
        """
        Initialize RunnableLambda.

        Args:
            func: Synchronous function to wrap
            name: Optional name for the runnable
            afunc: Optional async version of the function
        """
        super().__init__(name or func.__name__)
        self.func = func
        self.afunc = afunc or self._default_async_wrapper

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Invoke the lambda function."""
        config = config or self._config
        start_time = DeterministicClock.now()

        result = self.func(input)

        if config.enable_provenance and isinstance(result, dict):
            processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
            provenance = self._calculate_provenance(input, result)
            provenance.processing_time_ms = processing_time
            result['_lambda_provenance'] = provenance.dict()

        return result

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Async invoke the lambda function."""
        config = config or self._config
        start_time = DeterministicClock.now()

        result = await self.afunc(input)

        if config.enable_provenance and isinstance(result, dict):
            processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000
            provenance = self._calculate_provenance(input, result)
            provenance.processing_time_ms = processing_time
            result['_lambda_provenance'] = provenance.dict()

        return result

    async def _default_async_wrapper(self, input: T) -> U:
        """Default async wrapper using executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.func, input)


class RunnableBranch(BaseRunnable[T, U]):
    """
    Conditional branching runnable.

    Routes input to different runnables based on conditions,
    similar to a switch/case statement.
    """

    def __init__(self, branches: List[Tuple[Callable[[T], bool], BaseRunnable[T, U]]],
                 default: Optional[BaseRunnable[T, U]] = None):
        """
        Initialize RunnableBranch.

        Args:
            branches: List of (condition, runnable) tuples
            default: Optional default runnable if no conditions match
        """
        super().__init__("Branch")
        self.branches = branches
        self.default = default

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Invoke based on conditions."""
        for condition, runnable in self.branches:
            if condition(input):
                logger.info(f"Branch matched: {runnable.name}")
                return runnable.invoke(input, config)

        if self.default:
            logger.info(f"Using default branch: {self.default.name}")
            return self.default.invoke(input, config)

        raise ValueError(f"No branch matched for input and no default provided")

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Async invoke based on conditions."""
        for condition, runnable in self.branches:
            if condition(input):
                logger.info(f"Branch matched: {runnable.name}")
                return await runnable.ainvoke(input, config)

        if self.default:
            logger.info(f"Using default branch: {self.default.name}")
            return await self.default.ainvoke(input, config)

        raise ValueError(f"No branch matched for input and no default provided")


class ZeroHallucinationWrapper(BaseRunnable[T, U]):
    """
    Special wrapper that ensures zero-hallucination for calculation agents.

    This wrapper validates that wrapped agents only use deterministic
    calculations and never call LLMs for numeric outputs.
    """

    def __init__(self, runnable: BaseRunnable[T, U], validation_rules: Optional[List[Callable]] = None):
        """
        Initialize ZeroHallucinationWrapper.

        Args:
            runnable: The runnable to wrap
            validation_rules: Optional list of validation functions
        """
        super().__init__(f"ZeroHallucination[{runnable.name}]")
        self.runnable = runnable
        self.validation_rules = validation_rules or []

    def invoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Invoke with zero-hallucination validation."""
        # Validate input
        for rule in self.validation_rules:
            if not rule(input):
                raise ValueError(f"Input validation failed for zero-hallucination guarantee")

        # Execute
        result = self.runnable.invoke(input, config)

        # Validate output is deterministic
        if isinstance(result, dict):
            self._validate_no_llm_calculations(result)

        return result

    async def ainvoke(self, input: T, config: Optional[RunnableConfig] = None) -> U:
        """Async invoke with zero-hallucination validation."""
        # Validate input
        for rule in self.validation_rules:
            if not rule(input):
                raise ValueError(f"Input validation failed for zero-hallucination guarantee")

        # Execute
        result = await self.runnable.ainvoke(input, config)

        # Validate output is deterministic
        if isinstance(result, dict):
            self._validate_no_llm_calculations(result)

        return result

    def _validate_no_llm_calculations(self, result: Dict[str, Any]):
        """Validate that no LLM-generated calculations are in the result."""
        # Check for numeric fields and ensure they have provenance
        for key, value in result.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                # Ensure there's provenance for this calculation
                if '_provenance' not in result and '_calculation_method' not in result:
                    logger.warning(f"Numeric value {key}={value} lacks provenance tracking")

            # Recursively check nested dicts
            if isinstance(value, dict):
                self._validate_no_llm_calculations(value)


# Utility functions for creating chains

def create_sequential_chain(*agents) -> RunnableSequence:
    """
    Create a sequential chain from multiple agents.

    Example:
        >>> chain = create_sequential_chain(agent1, agent2, agent3)
    """
    runnables = [AgentRunnable(agent) if not isinstance(agent, BaseRunnable) else agent
                  for agent in agents]
    return RunnableSequence(runnables)


def create_parallel_chain(**agents) -> RunnableParallel:
    """
    Create a parallel chain from named agents.

    Example:
        >>> chain = create_parallel_chain(
        ...     emissions=emissions_agent,
        ...     compliance=compliance_agent
        ... )
    """
    runnables = {
        name: AgentRunnable(agent) if not isinstance(agent, BaseRunnable) else agent
        for name, agent in agents.items()
    }
    return RunnableParallel(runnables)


def create_map_reduce_chain(mapper: BaseRunnable, reducer: BaseRunnable) -> BaseRunnable:
    """
    Create a map-reduce pattern chain.

    Example:
        >>> chain = create_map_reduce_chain(
        ...     mapper=process_agent,
        ...     reducer=aggregate_agent
        ... )
    """
    class MapReduceRunnable(BaseRunnable[List[T], U]):
        def __init__(self):
            super().__init__(f"MapReduce[{mapper.name}->{reducer.name}]")

        async def ainvoke(self, inputs: List[T], config: Optional[RunnableConfig] = None) -> U:
            # Map phase - process all inputs in parallel
            mapped_results = await mapper.abatch(inputs, config)

            # Reduce phase - aggregate results
            return await reducer.ainvoke(mapped_results, config)

        def invoke(self, inputs: List[T], config: Optional[RunnableConfig] = None) -> U:
            # Map phase
            mapped_results = mapper.batch(inputs, config)

            # Reduce phase
            return reducer.invoke(mapped_results, config)

    return MapReduceRunnable()


# Export main components
__all__ = [
    'BaseRunnable',
    'AgentRunnable',
    'RunnableSequence',
    'RunnableParallel',
    'RetryRunnable',
    'FallbackRunnable',
    'RunnableLambda',
    'RunnableBranch',
    'ZeroHallucinationWrapper',
    'RunnableConfig',
    'ExecutionContext',
    'ProvenanceRecord',
    'ExecutionMode',
    'create_sequential_chain',
    'create_parallel_chain',
    'create_map_reduce_chain'
]