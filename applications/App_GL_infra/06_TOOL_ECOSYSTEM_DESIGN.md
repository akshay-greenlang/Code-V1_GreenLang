# GREENLANG TOOL ECOSYSTEM DESIGN

**Advanced Processing, Orchestration, and Testing Infrastructure**

**Version:** 1.0
**Date:** 2025-10-16
**Status:** Technical Specification
**Audience:** Engineers, Architects

---

## 🎯 OVERVIEW

This document specifies the advanced tool ecosystem for GreenLang framework, covering batch processing, pipeline orchestration, computation caching, reporting utilities, and testing infrastructure. These tools represent the next layer of framework capabilities that will further increase code reusability and developer productivity.

### **Tool Modules Covered**

| Module | Purpose | LOC Saved | Priority |
|--------|---------|-----------|----------|
| **Batch Processing** | Parallel execution with progress tracking | 300 lines | ⭐⭐⭐⭐ |
| **Pipeline Orchestration** | Multi-agent workflow management | 200 lines | ⭐⭐⭐⭐ |
| **Computation Cache** | Deterministic calculation caching | 200 lines | ⭐⭐⭐ |
| **Reporting Utilities** | Multi-dimensional aggregation | 300 lines | ⭐⭐⭐ |
| **Testing Framework** | Test fixtures & assertions | 400 lines | ⭐⭐⭐ |
| **TOTAL** | | **1,400 lines** | |

---

## 📦 MODULE 4: BATCH PROCESSING

### **Purpose**

Provide high-performance batch processing infrastructure for:
- Parallel execution of items
- Progress tracking with callbacks
- Statistics collection
- Error handling and recovery
- Memory-efficient streaming

### **Module Structure**

```
greenlang/processing/
├── __init__.py
├── batch.py              # BatchProcessor main class
├── stats.py              # StatsTracker
├── progress.py           # ProgressTracker
└── parallel.py           # Parallel execution utilities
```

### **Core API**

#### **1. BatchProcessor Class**

```python
from typing import List, Dict, Any, Callable, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

@dataclass
class BatchResult:
    """Result of batch processing."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[Dict[str, Any]] = None
    duration: float = 0.0
    items_per_second: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict:
        return {
            'total': self.total,
            'successful': self.successful,
            'failed': self.failed,
            'skipped': self.skipped,
            'error_count': len(self.errors),
            'duration': self.duration,
            'items_per_second': self.items_per_second
        }


class BatchProcessor:
    """
    High-performance batch processing with parallelization.

    Features:
    - Parallel execution (multi-threading)
    - Progress tracking with callbacks
    - Automatic error handling
    - Statistics collection
    - Memory-efficient streaming
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        fail_fast: bool = False
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_workers: Maximum parallel workers (None = CPU count)
            fail_fast: If True, stop on first error
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.fail_fast = fail_fast

    def process(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Any],
        validate_fn: Optional[Callable[[Any], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        error_callback: Optional[Callable[[Any, Exception], None]] = None
    ) -> BatchResult:
        """
        Process items in batches with validation and error handling.

        Args:
            items: List of items to process
            process_fn: Function to process each item
            validate_fn: Optional validation function
            progress_callback: Optional callback(current, total)
            error_callback: Optional callback(item, error)

        Returns:
            BatchResult with statistics
        """
        result = BatchResult(total=len(items))
        start_time = time.time()

        # Process in batches
        for batch_start in range(0, len(items), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(items))
            batch = items[batch_start:batch_end]

            # Process batch
            batch_result = self._process_batch(
                batch=batch,
                process_fn=process_fn,
                validate_fn=validate_fn,
                error_callback=error_callback
            )

            # Update results
            result.successful += batch_result.successful
            result.failed += batch_result.failed
            result.skipped += batch_result.skipped
            result.errors.extend(batch_result.errors)

            # Progress callback
            if progress_callback:
                progress_callback(batch_end, len(items))

            # Fail fast
            if self.fail_fast and batch_result.failed > 0:
                break

        # Calculate statistics
        result.duration = time.time() - start_time
        if result.duration > 0:
            result.items_per_second = result.total / result.duration

        return result

    def _process_batch(
        self,
        batch: List[Any],
        process_fn: Callable,
        validate_fn: Optional[Callable],
        error_callback: Optional[Callable]
    ) -> BatchResult:
        """Process single batch."""
        result = BatchResult()

        for item in batch:
            try:
                # Validate
                if validate_fn and not validate_fn(item):
                    result.skipped += 1
                    continue

                # Process
                process_fn(item)
                result.successful += 1

            except Exception as e:
                result.failed += 1
                result.errors.append({
                    'item': item,
                    'error': str(e),
                    'type': type(e).__name__
                })

                if error_callback:
                    error_callback(item, e)

                if self.fail_fast:
                    break

        return result

    def process_parallel(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Any],
        validate_fn: Optional[Callable[[Any], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        error_callback: Optional[Callable[[Any, Exception], None]] = None
    ) -> BatchResult:
        """
        Process items in parallel using thread pool.

        Args:
            items: List of items to process
            process_fn: Function to process each item (must be thread-safe)
            validate_fn: Optional validation function
            progress_callback: Optional callback(current, total)
            error_callback: Optional callback(item, error)

        Returns:
            BatchResult with statistics
        """
        result = BatchResult(total=len(items))
        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_item, item, process_fn, validate_fn): item
                for item in items
            }

            # Process completed tasks
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                completed += 1

                try:
                    status, error = future.result()

                    if status == 'success':
                        result.successful += 1
                    elif status == 'skipped':
                        result.skipped += 1
                    elif status == 'failed':
                        result.failed += 1
                        result.errors.append({
                            'item': item,
                            'error': str(error),
                            'type': type(error).__name__ if error else 'Unknown'
                        })

                        if error_callback:
                            error_callback(item, error)

                except Exception as e:
                    result.failed += 1
                    result.errors.append({
                        'item': item,
                        'error': str(e),
                        'type': type(e).__name__
                    })

                # Progress callback
                if progress_callback:
                    progress_callback(completed, len(items))

                # Fail fast
                if self.fail_fast and result.failed > 0:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        # Calculate statistics
        result.duration = time.time() - start_time
        if result.duration > 0:
            result.items_per_second = result.total / result.duration

        return result

    def _process_item(
        self,
        item: Any,
        process_fn: Callable,
        validate_fn: Optional[Callable]
    ) -> tuple[str, Optional[Exception]]:
        """Process single item (thread-safe)."""
        try:
            # Validate
            if validate_fn and not validate_fn(item):
                return 'skipped', None

            # Process
            process_fn(item)
            return 'success', None

        except Exception as e:
            return 'failed', e

    def process_streaming(
        self,
        item_iterator: Iterator[Any],
        process_fn: Callable[[Any], Any],
        validate_fn: Optional[Callable[[Any], bool]] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Iterator[tuple[Any, str, Optional[Exception]]]:
        """
        Process items in streaming fashion (memory-efficient).

        Args:
            item_iterator: Iterator yielding items
            process_fn: Function to process each item
            validate_fn: Optional validation function
            progress_callback: Optional callback(processed_count)

        Yields:
            Tuples of (item, status, error)
        """
        processed = 0

        for item in item_iterator:
            try:
                # Validate
                if validate_fn and not validate_fn(item):
                    yield (item, 'skipped', None)
                    continue

                # Process
                process_fn(item)
                yield (item, 'success', None)

            except Exception as e:
                yield (item, 'failed', e)

                if self.fail_fast:
                    break

            processed += 1
            if progress_callback:
                progress_callback(processed)
```

#### **2. StatsTracker Class**

```python
from typing import Dict, Any
from collections import defaultdict
import time

class StatsTracker:
    """
    Track processing statistics.

    Features:
    - Count tracking (success, failure, etc.)
    - Timing tracking
    - Memory tracking
    - Custom metrics
    """

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}

    def increment(self, counter: str, amount: int = 1):
        """Increment counter."""
        self.counters[counter] += amount

    def record(self, metric: str, value: Any):
        """Record metric value."""
        self.metrics[metric] = value

    def start_timer(self, name: str):
        """Start timer."""
        self.start_times[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop timer and return elapsed time."""
        if name not in self.start_times:
            return 0.0

        elapsed = time.time() - self.start_times[name]
        self.timers[name] = elapsed
        del self.start_times[name]
        return elapsed

    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        return {
            'counters': dict(self.counters),
            'timers': dict(self.timers),
            'metrics': dict(self.metrics)
        }

    def reset(self):
        """Reset all statistics."""
        self.counters.clear()
        self.timers.clear()
        self.start_times.clear()
        self.metrics.clear()
```

### **Usage Examples**

#### **Example 1: Simple Batch Processing**

```python
from greenlang.processing import BatchProcessor

# Initialize processor
processor = BatchProcessor(batch_size=100)

# Process items
items = [{'id': i, 'value': i * 2} for i in range(1000)]

def process_item(item):
    # Processing logic
    item['processed'] = True
    return item

def validate_item(item):
    # Validation logic
    return item['value'] > 0

def progress(current, total):
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

# Process
result = processor.process(
    items=items,
    process_fn=process_item,
    validate_fn=validate_item,
    progress_callback=progress
)

print(f"Processed: {result.successful}")
print(f"Failed: {result.failed}")
print(f"Speed: {result.items_per_second:.2f} items/sec")
```

#### **Example 2: Parallel Processing**

```python
from greenlang.processing import BatchProcessor
import time

# Initialize with parallelization
processor = BatchProcessor(
    batch_size=100,
    max_workers=4,  # 4 parallel workers
    fail_fast=False
)

# Process items in parallel
items = range(10000)

def process_item(item):
    # Simulate CPU-intensive work
    time.sleep(0.01)
    return item * 2

result = processor.process_parallel(
    items=items,
    process_fn=process_item,
    progress_callback=lambda c, t: print(f"{c}/{t}")
)

print(f"Duration: {result.duration:.2f}s")
print(f"Throughput: {result.items_per_second:.2f} items/sec")
```

#### **Example 3: Streaming Processing (Memory-Efficient)**

```python
from greenlang.processing import BatchProcessor

processor = BatchProcessor()

# Stream from large file
def read_large_file(file_path):
    """Generator for memory-efficient reading."""
    with open(file_path) as f:
        for line in f:
            yield json.loads(line)

# Process stream
success_count = 0
for item, status, error in processor.process_streaming(
    item_iterator=read_large_file('large_file.jsonl'),
    process_fn=lambda item: process_item(item)
):
    if status == 'success':
        success_count += 1

print(f"Processed: {success_count} items")
```

### **Migration Example**

#### **Before (Custom Code - 200 lines)**

```python
class CustomBatchProcessor:
    def __init__(self):
        self.stats = {'success': 0, 'failed': 0}

    def process_batch(self, items):
        # 100 lines of batch logic
        results = []
        for item in items:
            try:
                result = self.process_item(item)
                results.append(result)
                self.stats['success'] += 1
            except Exception as e:
                self.stats['failed'] += 1
                # ... error handling
        return results

    # ... 100 more lines for parallel, progress, etc.
```

#### **After (Framework - 50 lines)**

```python
from greenlang.processing import BatchProcessor

processor = BatchProcessor(batch_size=100, max_workers=4)

result = processor.process_parallel(
    items=items,
    process_fn=lambda item: self.process_item(item),
    progress_callback=lambda c, t: print(f"{c}/{t}")
)

# Stats automatically tracked
print(f"Success: {result.successful}, Failed: {result.failed}")
```

**LOC Reduction: 200 → 50 lines (75% reduction)**

---

## 📦 MODULE 5: PIPELINE ORCHESTRATION

### **Purpose**

Provide multi-agent pipeline orchestration for:
- Sequential agent execution
- Parallel agent execution
- Data flow management
- Dependency resolution
- Error handling and recovery

### **Module Structure**

```
greenlang/pipelines/
├── __init__.py
├── orchestrator.py       # Pipeline orchestrator
├── registry.py           # Agent registry
├── graph.py              # Dependency graph
└── execution.py          # Execution engine
```

### **Core API**

#### **1. Pipeline Class**

```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PipelineStep:
    """Single step in pipeline."""
    agent_id: str
    inputs: Dict[str, Any]
    outputs: List[str]
    depends_on: List[str] = None

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    status: str  # 'success', 'failed', 'partial'
    steps_completed: int
    steps_failed: int
    outputs: Dict[str, Any]
    errors: List[Dict[str, Any]]
    duration: float


class Pipeline:
    """
    Multi-agent pipeline orchestrator.

    Features:
    - Sequential and parallel execution
    - Automatic dependency resolution
    - Data flow management
    - Error handling and recovery
    - Pipeline visualization
    """

    def __init__(self, pipeline_id: str, description: str = ""):
        """
        Initialize pipeline.

        Args:
            pipeline_id: Unique pipeline identifier
            description: Human-readable description
        """
        self.pipeline_id = pipeline_id
        self.description = description
        self.steps: List[PipelineStep] = []
        self.agent_registry: Dict[str, Any] = {}

    def add_step(
        self,
        agent_id: str,
        inputs: Dict[str, Any],
        outputs: List[str],
        depends_on: Optional[List[str]] = None
    ):
        """
        Add step to pipeline.

        Args:
            agent_id: Agent identifier
            inputs: Input parameters {param: value or $output_ref}
            outputs: List of output names to capture
            depends_on: List of step agent_ids this depends on
        """
        step = PipelineStep(
            agent_id=agent_id,
            inputs=inputs,
            outputs=outputs,
            depends_on=depends_on or []
        )
        self.steps.append(step)

    def register_agent(self, agent_id: str, agent_instance: Any):
        """Register agent instance."""
        self.agent_registry[agent_id] = agent_instance

    def execute(
        self,
        parallel: bool = False,
        fail_fast: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> PipelineResult:
        """
        Execute pipeline.

        Args:
            parallel: Execute independent steps in parallel
            fail_fast: Stop on first error
            progress_callback: Optional callback(step, total)

        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()

        result = PipelineResult(
            status='success',
            steps_completed=0,
            steps_failed=0,
            outputs={},
            errors=[],
            duration=0.0
        )

        # Resolve execution order
        execution_order = self._resolve_dependencies()

        # Execute steps
        for step_idx, step in enumerate(execution_order):
            try:
                # Resolve inputs (replace $output_refs with actual values)
                resolved_inputs = self._resolve_inputs(step, result.outputs)

                # Get agent
                agent = self.agent_registry.get(step.agent_id)
                if not agent:
                    raise ValueError(f"Agent not registered: {step.agent_id}")

                # Execute agent
                step_outputs = agent.execute(**resolved_inputs)

                # Capture outputs
                for output_name in step.outputs:
                    if output_name in step_outputs:
                        result.outputs[f"{step.agent_id}.{output_name}"] = step_outputs[output_name]

                result.steps_completed += 1

                # Progress callback
                if progress_callback:
                    progress_callback(step_idx + 1, len(execution_order))

            except Exception as e:
                result.steps_failed += 1
                result.errors.append({
                    'agent_id': step.agent_id,
                    'error': str(e),
                    'type': type(e).__name__
                })

                if fail_fast:
                    result.status = 'failed'
                    break

        # Calculate duration
        result.duration = time.time() - start_time

        # Set final status
        if result.steps_failed == 0:
            result.status = 'success'
        elif result.steps_completed > 0:
            result.status = 'partial'
        else:
            result.status = 'failed'

        return result

    def _resolve_dependencies(self) -> List[PipelineStep]:
        """Resolve step dependencies and return execution order."""
        # Simple topological sort
        # (In production, use a proper dependency graph library)

        executed = set()
        order = []

        while len(order) < len(self.steps):
            for step in self.steps:
                if step.agent_id in executed:
                    continue

                # Check if all dependencies are satisfied
                if all(dep in executed for dep in step.depends_on):
                    order.append(step)
                    executed.add(step.agent_id)

        return order

    def _resolve_inputs(
        self,
        step: PipelineStep,
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve input references to actual values."""
        resolved = {}

        for param, value in step.inputs.items():
            if isinstance(value, str) and value.startswith('$'):
                # Output reference: $agent_id.output_name
                output_ref = value[1:]  # Remove $
                if output_ref in outputs:
                    resolved[param] = outputs[output_ref]
                else:
                    raise ValueError(f"Output not found: {output_ref}")
            else:
                # Literal value
                resolved[param] = value

        return resolved

    def save(self, file_path: Path):
        """Save pipeline definition to file."""
        definition = {
            'pipeline_id': self.pipeline_id,
            'description': self.description,
            'steps': [
                {
                    'agent_id': step.agent_id,
                    'inputs': step.inputs,
                    'outputs': step.outputs,
                    'depends_on': step.depends_on
                }
                for step in self.steps
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(definition, f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> 'Pipeline':
        """Load pipeline definition from file."""
        with open(file_path) as f:
            definition = json.load(f)

        pipeline = cls(
            pipeline_id=definition['pipeline_id'],
            description=definition.get('description', '')
        )

        for step_def in definition['steps']:
            pipeline.add_step(
                agent_id=step_def['agent_id'],
                inputs=step_def['inputs'],
                outputs=step_def['outputs'],
                depends_on=step_def.get('depends_on')
            )

        return pipeline
```

### **Usage Examples**

#### **Example 1: Simple Sequential Pipeline**

```python
from greenlang.pipelines import Pipeline

# Create pipeline
pipeline = Pipeline(
    pipeline_id='cbam-import-pipeline',
    description='Import and process CBAM goods'
)

# Add steps
pipeline.add_step(
    agent_id='importer',
    inputs={'file_path': 'input.csv'},
    outputs=['imported_goods']
)

pipeline.add_step(
    agent_id='validator',
    inputs={'goods': '$importer.imported_goods'},  # Reference previous output
    outputs=['validated_goods'],
    depends_on=['importer']
)

pipeline.add_step(
    agent_id='calculator',
    inputs={'goods': '$validator.validated_goods'},
    outputs=['calculated_goods'],
    depends_on=['validator']
)

# Register agents
pipeline.register_agent('importer', importer_instance)
pipeline.register_agent('validator', validator_instance)
pipeline.register_agent('calculator', calculator_instance)

# Execute
result = pipeline.execute(progress_callback=lambda s, t: print(f"{s}/{t}"))

print(f"Status: {result.status}")
print(f"Completed: {result.steps_completed}")
print(f"Duration: {result.duration:.2f}s")
```

#### **Example 2: Save/Load Pipeline Definition**

```python
from greenlang.pipelines import Pipeline
from pathlib import Path

# Save pipeline
pipeline.save(Path('pipelines/cbam_import.json'))

# Load pipeline
loaded_pipeline = Pipeline.load(Path('pipelines/cbam_import.json'))

# Register agents and execute
loaded_pipeline.register_agent('importer', importer_instance)
# ... register other agents
result = loaded_pipeline.execute()
```

### **Migration Example**

**LOC Reduction: 200 → 80 lines (60% reduction)**

---

## 📦 MODULE 6: COMPUTATION CACHE

### **Purpose**

Provide deterministic calculation caching for:
- Avoiding redundant calculations
- Performance optimization
- Cache invalidation strategies
- Persistent caching

### **Core API**

```python
from typing import Callable, Any, Optional
from functools import wraps
import hashlib
import json
import pickle
from pathlib import Path

class ComputationCache:
    """
    Deterministic computation caching.

    Features:
    - Function result caching
    - Automatic cache key generation
    - TTL support
    - Persistent storage
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize computation cache.

        Args:
            cache_dir: Directory for persistent cache (None = memory only)
        """
        self.memory_cache: Dict[str, Any] = {}
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Serialize arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)

        # Hash for compact key
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Try persistent cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                    # Store in memory for faster access
                    self.memory_cache[key] = value
                    return value

        return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Store in memory
        self.memory_cache[key] = value

        # Store in persistent cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


# Global cache instance
_global_cache = ComputationCache()


def cached(cache: Optional[ComputationCache] = None):
    """
    Decorator to cache function results.

    Usage:
        @cached()
        def expensive_calculation(x, y):
            return x ** y
    """
    def decorator(func: Callable) -> Callable:
        nonlocal cache
        if cache is None:
            cache = _global_cache

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._generate_key(func.__name__, args, kwargs)

            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result)

            return result

        return wrapper
    return decorator
```

### **Usage Example**

```python
from greenlang.compute import cached, ComputationCache

# Use global cache
@cached()
def calculate_cbam_emissions(quantity, emission_factor):
    """Expensive calculation - results are cached."""
    return quantity * emission_factor

# Use custom cache
cache = ComputationCache(cache_dir=Path('cache'))

@cached(cache=cache)
def calculate_indirect_emissions(good_data):
    # Complex calculation
    return result
```

**LOC Reduction: 150 → 50 lines (67% reduction)**

---

## 📦 MODULE 7: REPORTING UTILITIES

### **Purpose**

Provide multi-dimensional reporting and aggregation for:
- Group-by aggregations
- Pivot tables
- Report formatting (Excel, JSON, HTML)
- Template system

### **Core API**

```python
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

class ReportBuilder:
    """
    Multi-dimensional report builder.

    Features:
    - Group-by aggregations
    - Pivot tables
    - Multiple output formats
    - Template support
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize report builder.

        Args:
            data: Source DataFrame
        """
        self.data = data
        self.aggregations: List[Dict] = []

    def add_aggregation(
        self,
        group_by: List[str],
        metrics: Dict[str, str],
        name: str
    ):
        """
        Add aggregation.

        Args:
            group_by: Columns to group by
            metrics: {column: agg_func} (e.g., {'quantity': 'sum'})
            name: Aggregation name
        """
        self.aggregations.append({
            'name': name,
            'group_by': group_by,
            'metrics': metrics
        })

    def build(self) -> Dict[str, pd.DataFrame]:
        """Build all aggregations."""
        results = {}

        for agg in self.aggregations:
            grouped = self.data.groupby(agg['group_by']).agg(agg['metrics'])
            results[agg['name']] = grouped.reset_index()

        return results

    def to_excel(self, file_path: Path):
        """Export to Excel with multiple sheets."""
        results = self.build()

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for name, df in results.items():
                df.to_excel(writer, sheet_name=name, index=False)

    def to_json(self, file_path: Path):
        """Export to JSON."""
        results = self.build()

        output = {
            name: df.to_dict('records')
            for name, df in results.items()
        }

        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)
```

### **Usage Example**

```python
from greenlang.reporting import ReportBuilder
import pandas as pd

# Load data
df = pd.read_csv('cbam_goods.csv')

# Create report
report = ReportBuilder(df)

# Add aggregations
report.add_aggregation(
    group_by=['country_of_origin'],
    metrics={'quantity': 'sum', 'emissions': 'sum'},
    name='by_country'
)

report.add_aggregation(
    group_by=['cn_code'],
    metrics={'quantity': 'sum', 'emissions': 'mean'},
    name='by_product'
)

# Export
report.to_excel(Path('report.xlsx'))
```

**LOC Reduction: 300 → 150 lines (50% reduction)**

---

## 📦 MODULE 8: TESTING FRAMEWORK

### **Purpose**

Provide comprehensive testing infrastructure for:
- Standard test fixtures
- Domain-specific assertions
- Mock utilities
- Test data generators

### **Core API**

```python
from typing import Any, Dict, List
import pytest
from pathlib import Path

class AgentTestCase:
    """Base class for agent testing."""

    @pytest.fixture
    def sample_data(self) -> Dict:
        """Provide sample test data."""
        return {
            'cn_code': '7208.10',
            'quantity': 1000,
            'emissions': 2.5
        }

    @pytest.fixture
    def agent_config(self) -> Dict:
        """Provide agent configuration."""
        return {
            'batch_size': 100,
            'validate': True
        }

    def assert_valid_output(self, output: Dict):
        """Assert output has required fields."""
        required_fields = ['status', 'processed_count']
        for field in required_fields:
            assert field in output, f"Missing field: {field}"

    def assert_no_errors(self, result: Dict):
        """Assert no errors in result."""
        assert result.get('errors', []) == [], f"Errors found: {result['errors']}"
```

**LOC Reduction: 600 → 200 lines (67% reduction)**

---

## 📊 SUMMARY

### **Complete Tool Ecosystem LOC Savings**

| Module | Custom LOC | Framework LOC | Custom After | Reduction |
|--------|-----------|---------------|--------------|-----------|
| **Batch Processing** | 200 | 300 framework | 50 | 75% |
| **Pipeline Orchestration** | 200 | 200 framework | 80 | 60% |
| **Computation Cache** | 150 | 200 framework | 50 | 67% |
| **Reporting Utilities** | 300 | 300 framework | 150 | 50% |
| **Testing Framework** | 600 | 400 framework | 200 | 67% |
| **TOTAL** | **1,450** | **1,400** | **530** | **63%** |

**Combined with Utilities (Modules 1-3):** 3,955 LOC saved (67% average reduction)

---

## 🎯 NEXT STEPS

1. **Review**: Validate specifications with team
2. **Prototype**: Build Tier 2 modules (Batch, Pipelines, Cache)
3. **Integrate**: Combine with Tier 1 modules
4. **Test**: Full integration testing
5. **Document**: API documentation and examples
6. **Release**: Framework v0.2 with complete tool ecosystem

---

**Status:** ✅ **Ready for Engineering Review**

**Next Document:** Reference Implementation Guide

---

*"Great tools disappear into the workflow - you forget they're even there."*
