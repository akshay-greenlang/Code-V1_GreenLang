"""
Concurrent Operations Edge Case Tests for GL-004 BURNMASTER

Tests system behavior under concurrent operation scenarios:
- Race conditions in optimization
- Deadlock detection and prevention
- Thread safety of calculations
- Concurrent data access patterns
- Lock contention scenarios
- Async operation ordering

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
import random
import copy

# Import system modules
import sys
sys.path.insert(0, 'C:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-004_Burnmaster')

from combustion.stoichiometry import (
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_air_percent,
    compute_excess_o2,
    infer_lambda_from_o2,
)
from combustion.fuel_properties import (
    FuelType, FuelComposition,
    compute_fuel_properties, get_fuel_properties,
)
from combustion.thermodynamics import (
    compute_stack_loss,
    compute_efficiency_indirect,
)
from safety.safety_envelope import SafetyEnvelope, Setpoint, EnvelopeStatus
from calculators.stability_calculator import (
    FlameStabilityCalculator, StabilityLevel, RiskLevel,
)


# ============================================================================
# THREAD-SAFE DATA STRUCTURES
# ============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for testing concurrent increments."""

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class ThreadSafeQueue:
    """Thread-safe queue wrapper for testing."""

    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()

    def put(self, item: Any, timeout: float = None):
        self._queue.put(item, timeout=timeout)

    def get(self, timeout: float = None) -> Any:
        return self._queue.get(timeout=timeout)

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()


@dataclass
class SharedState:
    """Shared state for concurrent access testing."""
    o2_setpoint: float = 3.0
    lambda_value: float = 1.15
    firing_rate: float = 80.0
    stability_index: float = 0.85
    update_count: int = 0
    last_update_time: datetime = field(default_factory=datetime.utcnow)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.update_count += 1
            self.last_update_time = datetime.utcnow()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'o2_setpoint': self.o2_setpoint,
                'lambda_value': self.lambda_value,
                'firing_rate': self.firing_rate,
                'stability_index': self.stability_index,
                'update_count': self.update_count
            }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def stability_calculator():
    """Create FlameStabilityCalculator instance."""
    return FlameStabilityCalculator(precision=4)


@pytest.fixture
def safety_envelope():
    """Create configured SafetyEnvelope instance."""
    envelope = SafetyEnvelope(unit_id="BLR-TEST")
    envelope.define_envelope("BLR-TEST", {
        "o2_min": 1.5,
        "o2_max": 8.0,
        "co_max": 200,
        "nox_max": 100,
        "draft_min": -0.5,
        "draft_max": -0.01,
        "flame_signal_min": 30.0,
        "steam_temp_max": 550.0,
        "steam_pressure_max": 150.0,
        "firing_rate_min": 10.0,
        "firing_rate_max": 100.0,
    })
    return envelope


@pytest.fixture
def shared_state():
    """Create shared state for concurrent testing."""
    return SharedState()


@pytest.fixture
def thread_pool():
    """Create thread pool for concurrent testing."""
    pool = ThreadPoolExecutor(max_workers=10)
    yield pool
    pool.shutdown(wait=True)


# ============================================================================
# RACE CONDITION TESTS
# ============================================================================

class TestRaceConditions:
    """Test suite for race condition scenarios."""

    def test_concurrent_lambda_calculations(self, thread_pool):
        """Test concurrent lambda calculations don't interfere."""
        results = []

        def calculate_lambda(stoich_af: float, actual_af: float) -> float:
            # Simulate some processing time
            time.sleep(random.uniform(0.001, 0.005))
            return compute_lambda(actual_af, stoich_af)

        # Submit many concurrent calculations
        futures = []
        expected = []
        for i in range(100):
            stoich = 17.2
            actual = 17.2 * (1.0 + i * 0.01)  # Lambda from 1.0 to 2.0
            expected_lambda = actual / stoich
            expected.append(expected_lambda)
            futures.append(thread_pool.submit(calculate_lambda, stoich, actual))

        # Collect results
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)

        # Verify all calculations are correct (no interference)
        for i, (result, exp) in enumerate(zip(results, expected)):
            assert abs(result - exp) < 0.001, f"Calculation {i} incorrect: {result} vs {exp}"

    def test_concurrent_setpoint_validation(self, safety_envelope, thread_pool):
        """Test concurrent setpoint validations don't interfere."""
        validations = []

        def validate_setpoint(o2_value: float) -> Tuple[float, bool]:
            setpoint = Setpoint(
                parameter_name="o2",
                value=o2_value,
                unit="%"
            )
            result = safety_envelope.validate_within_envelope(setpoint)
            return o2_value, result.is_valid

        # Submit concurrent validations
        futures = []
        for i in range(100):
            o2_value = 0.5 + i * 0.1  # Range from 0.5 to 10.5
            futures.append(thread_pool.submit(validate_setpoint, o2_value))

        # Collect results
        for future in as_completed(futures):
            o2_value, is_valid = future.result()
            validations.append((o2_value, is_valid))

        # Verify correct validation (1.5 <= o2 <= 8.0 is valid)
        for o2_value, is_valid in validations:
            expected_valid = 1.5 <= o2_value <= 8.0
            assert is_valid == expected_valid, f"O2={o2_value}: expected {expected_valid}, got {is_valid}"

    def test_concurrent_stability_calculations(self, stability_calculator, thread_pool):
        """Test concurrent stability calculations are deterministic."""
        # Generate test signals
        signals = [
            np.array([100.0 + np.random.normal(0, 5) for _ in range(50)])
            for _ in range(20)
        ]

        def calculate_stability(signal, o2_variance):
            return stability_calculator.compute_stability_index(signal, o2_variance)

        # Run each signal multiple times concurrently
        results_by_signal = {i: [] for i in range(len(signals))}

        futures = []
        for run in range(5):  # 5 runs per signal
            for i, signal in enumerate(signals):
                futures.append((i, thread_pool.submit(calculate_stability, signal.copy(), 0.1)))

        for signal_idx, future in futures:
            result = future.result()
            results_by_signal[signal_idx].append(result.provenance_hash)

        # Verify same signal produces same hash (deterministic)
        for signal_idx, hashes in results_by_signal.items():
            assert len(set(hashes)) == 1, f"Signal {signal_idx} produced different hashes"

    def test_shared_state_race_condition(self, shared_state, thread_pool):
        """Test shared state is properly protected from race conditions."""
        num_updates = 1000

        def update_state(i: int):
            shared_state.update(
                o2_setpoint=3.0 + (i % 10) * 0.1,
                update_count=shared_state.update_count + 1
            )

        # Submit many concurrent updates
        futures = [thread_pool.submit(update_state, i) for i in range(num_updates)]

        # Wait for all to complete
        for future in as_completed(futures):
            future.result()

        # The update_count should reflect all updates
        # Note: update_count is incremented in update(), so should be 1000
        # But due to the += 1 in the update call being racy, the internal count should still be correct
        assert shared_state.update_count == num_updates


# ============================================================================
# DEADLOCK DETECTION TESTS
# ============================================================================

class TestDeadlockPrevention:
    """Test suite for deadlock prevention."""

    def test_no_deadlock_on_nested_envelope_operations(self, safety_envelope):
        """Test that nested envelope operations don't deadlock."""
        timeout = 5.0  # 5 second timeout

        def nested_operation():
            # Outer validation
            setpoint1 = Setpoint(parameter_name="o2", value=3.0, unit="%")
            result1 = safety_envelope.validate_within_envelope(setpoint1)

            # Inner validation (nested)
            setpoint2 = Setpoint(parameter_name="co", value=50.0, unit="ppm")
            result2 = safety_envelope.validate_within_envelope(setpoint2)

            return result1.is_valid, result2.is_valid

        # Run in thread with timeout
        result_queue = queue.Queue()

        def run_with_timeout():
            try:
                result = nested_operation()
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))

        thread = threading.Thread(target=run_with_timeout)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            pytest.fail("Deadlock detected - operation did not complete within timeout")

        status, result = result_queue.get()
        assert status == "success", f"Operation failed: {result}"

    def test_no_deadlock_on_concurrent_shrink_expand(self, safety_envelope):
        """Test no deadlock when shrink and expand operations overlap."""
        timeout = 5.0
        operations_complete = threading.Event()

        def shrink_operation():
            for _ in range(10):
                try:
                    safety_envelope.shrink_envelope(0.95, "Test shrink")
                except Exception:
                    pass
                time.sleep(0.01)

        def expand_operation():
            for _ in range(10):
                try:
                    safety_envelope.expand_envelope(1.05, "APPROVED-TEST-USER")
                except Exception:
                    pass
                time.sleep(0.01)

        # Start both operations
        shrink_thread = threading.Thread(target=shrink_operation)
        expand_thread = threading.Thread(target=expand_operation)

        shrink_thread.start()
        expand_thread.start()

        # Wait for completion with timeout
        shrink_thread.join(timeout=timeout / 2)
        expand_thread.join(timeout=timeout / 2)

        if shrink_thread.is_alive() or expand_thread.is_alive():
            pytest.fail("Deadlock detected in concurrent shrink/expand")

    def test_lock_ordering_prevents_deadlock(self):
        """Test that consistent lock ordering prevents deadlock."""
        lock_a = threading.Lock()
        lock_b = threading.Lock()
        lock_c = threading.Lock()

        deadlock_detected = threading.Event()
        operations_completed = []

        def operation_abc():
            # Always acquire in order: A -> B -> C
            with lock_a:
                time.sleep(0.01)
                with lock_b:
                    time.sleep(0.01)
                    with lock_c:
                        operations_completed.append("abc")

        def operation_partial():
            # Also respects order: A -> B
            with lock_a:
                time.sleep(0.01)
                with lock_b:
                    operations_completed.append("ab")

        threads = [
            threading.Thread(target=operation_abc),
            threading.Thread(target=operation_partial),
            threading.Thread(target=operation_abc),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        # All operations should complete
        for t in threads:
            assert not t.is_alive(), "Thread didn't complete - possible deadlock"

        assert len(operations_completed) == 3


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Test suite for thread safety of calculations."""

    def test_calculator_thread_safety(self, thread_pool):
        """Test that FlameStabilityCalculator is thread-safe."""
        calculator = FlameStabilityCalculator(precision=4)

        results = []
        errors = []

        def calculate(i: int):
            try:
                signal = np.array([100.0 + i * 0.1] * 50)
                result = calculator.compute_stability_index(signal, 0.1)
                return i, float(result.stability_index)
            except Exception as e:
                return i, str(e)

        futures = [thread_pool.submit(calculate, i) for i in range(100)]

        for future in as_completed(futures):
            idx, result = future.result()
            if isinstance(result, str):
                errors.append((idx, result))
            else:
                results.append((idx, result))

        assert len(errors) == 0, f"Errors in thread-safe calculation: {errors}"
        assert len(results) == 100

    def test_stoichiometry_thread_safety(self, thread_pool):
        """Test stoichiometry calculations are thread-safe."""
        results = []

        def calculate_stoich(ch4_pct: float):
            composition = {"CH4": ch4_pct, "N2": 100.0 - ch4_pct}
            return ch4_pct, compute_stoichiometric_air(composition)

        futures = [thread_pool.submit(calculate_stoich, 50.0 + i) for i in range(50)]

        for future in as_completed(futures):
            ch4, stoich_air = future.result()
            results.append((ch4, stoich_air))
            # Stoich air should increase with CH4 content
            assert stoich_air > 0

        # Verify ordering relationship (more CH4 = more air required)
        sorted_results = sorted(results, key=lambda x: x[0])
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i][1] < sorted_results[i + 1][1]

    def test_concurrent_property_calculation(self, thread_pool):
        """Test concurrent fuel property calculations."""
        fuel_types = list(FuelType)[:5]  # First 5 fuel types

        results = {}

        def get_properties(fuel_type: FuelType):
            try:
                props = get_fuel_properties(fuel_type)
                return fuel_type.value, props.hhv
            except Exception as e:
                return fuel_type.value, str(e)

        futures = []
        for _ in range(10):  # 10 iterations each
            for ft in fuel_types:
                futures.append(thread_pool.submit(get_properties, ft))

        for future in as_completed(futures):
            fuel_name, result = future.result()
            if fuel_name not in results:
                results[fuel_name] = []
            results[fuel_name].append(result)

        # All results for same fuel type should be identical
        for fuel_name, values in results.items():
            if all(isinstance(v, float) for v in values):
                assert len(set(values)) == 1, f"{fuel_name} has inconsistent HHV values"


# ============================================================================
# CONCURRENT DATA ACCESS TESTS
# ============================================================================

class TestConcurrentDataAccess:
    """Test suite for concurrent data access patterns."""

    def test_producer_consumer_pattern(self, thread_pool):
        """Test producer-consumer pattern with calculations."""
        data_queue = queue.Queue(maxsize=100)
        results_queue = queue.Queue()
        stop_event = threading.Event()

        produced_count = 0
        consumed_count = 0

        def producer(count: int):
            nonlocal produced_count
            for i in range(count):
                signal = np.array([100.0 + np.random.normal(0, 3) for _ in range(20)])
                data_queue.put(signal)
                produced_count += 1
            stop_event.set()

        def consumer(calculator):
            nonlocal consumed_count
            while not (stop_event.is_set() and data_queue.empty()):
                try:
                    signal = data_queue.get(timeout=0.1)
                    result = calculator.compute_stability_index(signal, 0.1)
                    results_queue.put(result.stability_index)
                    consumed_count += 1
                except queue.Empty:
                    continue

        calculator = FlameStabilityCalculator(precision=4)

        producer_thread = threading.Thread(target=producer, args=(50,))
        consumer_threads = [threading.Thread(target=consumer, args=(calculator,)) for _ in range(3)]

        producer_thread.start()
        for ct in consumer_threads:
            ct.start()

        producer_thread.join(timeout=10)
        for ct in consumer_threads:
            ct.join(timeout=10)

        assert produced_count == consumed_count == results_queue.qsize()

    def test_reader_writer_pattern(self, shared_state, thread_pool):
        """Test reader-writer pattern with shared state."""
        read_results = []
        write_results = []

        def reader(iterations: int):
            for _ in range(iterations):
                snapshot = shared_state.get_snapshot()
                read_results.append(snapshot)
                time.sleep(0.001)

        def writer(iterations: int):
            for i in range(iterations):
                shared_state.update(
                    o2_setpoint=3.0 + (i % 10) * 0.1,
                    firing_rate=80.0 + (i % 20)
                )
                write_results.append(i)
                time.sleep(0.002)

        # Start multiple readers and writers
        futures = []
        for _ in range(5):
            futures.append(thread_pool.submit(reader, 20))
        for _ in range(2):
            futures.append(thread_pool.submit(writer, 20))

        for future in as_completed(futures):
            future.result()

        # Verify all reads got valid data
        for snapshot in read_results:
            assert 'o2_setpoint' in snapshot
            assert 'firing_rate' in snapshot

    def test_concurrent_envelope_validation(self, safety_envelope, thread_pool):
        """Test concurrent envelope validation requests."""
        validation_results = []

        parameters = ['o2', 'co', 'firing_rate', 'nox']
        values = {
            'o2': [1.0, 2.0, 3.0, 5.0, 7.0, 9.0],
            'co': [10.0, 50.0, 100.0, 150.0, 250.0],
            'firing_rate': [5.0, 20.0, 50.0, 80.0, 100.0, 110.0],
            'nox': [10.0, 40.0, 80.0, 120.0],
        }

        def validate(param: str, value: float):
            setpoint = Setpoint(parameter_name=param, value=value, unit="")
            result = safety_envelope.validate_within_envelope(setpoint)
            return param, value, result.is_valid

        futures = []
        for param in parameters:
            for val in values.get(param, []):
                futures.append(thread_pool.submit(validate, param, val))

        for future in as_completed(futures):
            param, value, is_valid = future.result()
            validation_results.append((param, value, is_valid))

        # Verify consistent validation rules
        for param, value, is_valid in validation_results:
            if param == 'o2':
                expected = 1.5 <= value <= 8.0
            elif param == 'co':
                expected = 0 <= value <= 200
            elif param == 'firing_rate':
                expected = 10.0 <= value <= 100.0
            elif param == 'nox':
                expected = 0 <= value <= 100
            else:
                continue

            assert is_valid == expected, f"{param}={value}: expected {expected}, got {is_valid}"


# ============================================================================
# LOCK CONTENTION TESTS
# ============================================================================

class TestLockContention:
    """Test suite for lock contention scenarios."""

    def test_high_contention_counter(self, thread_pool):
        """Test thread-safe counter under high contention."""
        counter = ThreadSafeCounter()
        num_increments = 10000
        num_threads = 10

        def increment_many(count: int):
            for _ in range(count):
                counter.increment()

        futures = [
            thread_pool.submit(increment_many, num_increments // num_threads)
            for _ in range(num_threads)
        ]

        for future in as_completed(futures):
            future.result()

        assert counter.value == num_increments

    def test_contention_on_shared_state(self, thread_pool):
        """Test shared state under high contention."""
        state = SharedState()
        num_updates = 1000
        num_threads = 20

        def update_many(thread_id: int, count: int):
            for i in range(count):
                state.update(
                    o2_setpoint=3.0 + (thread_id + i) % 10 * 0.1
                )

        futures = [
            thread_pool.submit(update_many, t, num_updates // num_threads)
            for t in range(num_threads)
        ]

        for future in as_completed(futures):
            future.result()

        assert state.update_count == num_updates

    def test_mixed_read_write_contention(self, shared_state, thread_pool):
        """Test mixed read/write operations under contention."""
        read_count = 0
        write_count = 0
        lock = threading.Lock()

        def reader():
            nonlocal read_count
            for _ in range(100):
                shared_state.get_snapshot()
                with lock:
                    read_count += 1

        def writer():
            nonlocal write_count
            for _ in range(50):
                shared_state.update(o2_setpoint=random.uniform(2.0, 5.0))
                with lock:
                    write_count += 1

        # 10 readers, 5 writers
        futures = []
        for _ in range(10):
            futures.append(thread_pool.submit(reader))
        for _ in range(5):
            futures.append(thread_pool.submit(writer))

        for future in as_completed(futures):
            future.result()

        assert read_count == 1000  # 10 * 100
        assert write_count == 250  # 5 * 50


# ============================================================================
# ASYNC OPERATION ORDERING TESTS
# ============================================================================

class TestAsyncOperationOrdering:
    """Test suite for async operation ordering."""

    @pytest.mark.asyncio
    async def test_sequential_async_calculations(self):
        """Test sequential async calculations maintain order."""
        results = []

        async def calculate(i: int):
            await asyncio.sleep(random.uniform(0.001, 0.005))
            return i

        for i in range(20):
            result = await calculate(i)
            results.append(result)

        assert results == list(range(20))

    @pytest.mark.asyncio
    async def test_concurrent_async_calculations(self):
        """Test concurrent async calculations complete correctly."""
        async def calculate_stability(signal_id: int):
            await asyncio.sleep(random.uniform(0.001, 0.01))
            calculator = FlameStabilityCalculator(precision=4)
            signal = np.array([100.0] * 50)
            result = calculator.compute_stability_index(signal, 0.1)
            return signal_id, float(result.stability_index)

        tasks = [calculate_stability(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 50

        # All should have valid stability indices
        for signal_id, stability in results:
            assert 0 <= stability <= 1

    @pytest.mark.asyncio
    async def test_async_with_timeout(self):
        """Test async operations with timeout handling."""
        async def slow_calculation(delay: float):
            await asyncio.sleep(delay)
            return "completed"

        # Should complete
        result = await asyncio.wait_for(slow_calculation(0.01), timeout=1.0)
        assert result == "completed"

        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_calculation(2.0), timeout=0.1)

    @pytest.mark.asyncio
    async def test_async_queue_processing(self):
        """Test async queue processing maintains FIFO order."""
        async_queue = asyncio.Queue()
        results = []

        async def producer():
            for i in range(20):
                await async_queue.put(i)
                await asyncio.sleep(0.001)

        async def consumer():
            for _ in range(20):
                item = await async_queue.get()
                results.append(item)
                async_queue.task_done()

        await asyncio.gather(producer(), consumer())

        # Results should be in FIFO order
        assert results == list(range(20))

    @pytest.mark.asyncio
    async def test_async_semaphore_limiting(self):
        """Test semaphore limits concurrent async operations."""
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()

        async def limited_operation(i: int):
            nonlocal concurrent_count, max_concurrent_observed

            async with semaphore:
                async with lock:
                    concurrent_count += 1
                    max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

                await asyncio.sleep(0.01)

                async with lock:
                    concurrent_count -= 1

            return i

        tasks = [limited_operation(i) for i in range(20)]
        await asyncio.gather(*tasks)

        assert max_concurrent_observed <= max_concurrent


# ============================================================================
# ATOMICITY TESTS
# ============================================================================

class TestAtomicity:
    """Test suite for operation atomicity."""

    def test_atomic_update_operation(self, shared_state):
        """Test that state updates are atomic."""
        original_snapshot = shared_state.get_snapshot()

        # Update should be atomic
        shared_state.update(
            o2_setpoint=5.0,
            lambda_value=1.2,
            firing_rate=90.0
        )

        new_snapshot = shared_state.get_snapshot()

        # Either all changed or none (atomic)
        changes = [
            new_snapshot['o2_setpoint'] != original_snapshot['o2_setpoint'],
            new_snapshot['lambda_value'] != original_snapshot['lambda_value'],
            new_snapshot['firing_rate'] != original_snapshot['firing_rate'],
        ]

        # All should have changed together
        assert all(changes) or not any(changes)

    def test_atomic_envelope_shrink(self, safety_envelope):
        """Test that envelope shrink is atomic."""
        # Get original limits
        original_limits = copy.deepcopy(safety_envelope.limits)

        # Shrink should be atomic
        safety_envelope.shrink_envelope(0.9, "Atomic test")

        new_limits = safety_envelope.limits

        # All limit changes should be applied together
        # (If any changed, all relevant ones should have changed)
        assert new_limits.o2_max < original_limits.o2_max
        assert new_limits.o2_min > original_limits.o2_min

    def test_no_partial_calculation_results(self, stability_calculator):
        """Test that calculation results are complete."""
        signal = np.array([100.0] * 50)
        result = stability_calculator.compute_stability_index(signal, 0.1)

        # Result should be complete
        assert result.stability_index is not None
        assert result.stability_level is not None
        assert result.provenance_hash is not None
        assert result.flame_signal_stats is not None
        assert 'mean' in result.flame_signal_stats
        assert 'std' in result.flame_signal_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
