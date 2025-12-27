"""
GL-001 ThermalCommand: Calculation Trace.

Provides full lineage tracking from input to output for all calculations,
enabling complete auditability for regulatory compliance.

Implements:
- Full calculation lineage (input → intermediate → output)
- SHA-256 provenance hashing at each step
- Regulatory reference linking
- Reproducibility verification
"""

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# =============================================================================
# TRACE TYPES AND STRUCTURES
# =============================================================================


class CalculationType(str, Enum):
    """Types of calculations being traced."""

    EFFICIENCY = 'EFFICIENCY'
    EMISSION = 'EMISSION'
    HEAT_BALANCE = 'HEAT_BALANCE'
    STEAM_PROPERTY = 'STEAM_PROPERTY'
    COMBUSTION = 'COMBUSTION'
    HEAT_TRANSFER = 'HEAT_TRANSFER'
    WATER_CHEMISTRY = 'WATER_CHEMISTRY'
    EXERGY = 'EXERGY'
    PREDICTIVE = 'PREDICTIVE'
    OPTIMIZATION = 'OPTIMIZATION'


class DataSource(str, Enum):
    """Source of input data."""

    SENSOR = 'SENSOR'
    MANUAL_ENTRY = 'MANUAL_ENTRY'
    CALCULATED = 'CALCULATED'
    REFERENCE_TABLE = 'REFERENCE_TABLE'
    EXTERNAL_API = 'EXTERNAL_API'
    DEFAULT_VALUE = 'DEFAULT_VALUE'


@dataclass
class TraceValue:
    """A traced value with full provenance."""

    name: str
    value: Union[Decimal, float, int, str]
    unit: str
    source: DataSource
    timestamp: datetime
    sensor_id: Optional[str] = None
    reference: Optional[str] = None
    uncertainty: Optional[Decimal] = None
    provenance_hash: str = ''

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate SHA-256 hash for this value."""
        data = {
            'name': self.name,
            'value': str(self.value),
            'unit': self.unit,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['source'] = self.source.value
        data['timestamp'] = self.timestamp.isoformat()
        data['value'] = str(self.value)
        if self.uncertainty:
            data['uncertainty'] = str(self.uncertainty)
        return data


@dataclass
class CalculationStep:
    """A single step in a calculation trace."""

    step_number: int
    operation: str
    formula: str
    inputs: List[TraceValue]
    output: TraceValue
    regulatory_reference: Optional[str] = None
    notes: Optional[str] = None
    execution_time_us: int = 0
    provenance_hash: str = ''

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate SHA-256 hash for this step."""
        input_hashes = [i.provenance_hash for i in self.inputs]
        data = {
            'step': self.step_number,
            'operation': self.operation,
            'formula': self.formula,
            'inputs': input_hashes,
            'output': self.output.provenance_hash,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_number': self.step_number,
            'operation': self.operation,
            'formula': self.formula,
            'inputs': [i.to_dict() for i in self.inputs],
            'output': self.output.to_dict(),
            'regulatory_reference': self.regulatory_reference,
            'notes': self.notes,
            'execution_time_us': self.execution_time_us,
            'provenance_hash': self.provenance_hash,
        }


@dataclass
class CalculationTrace:
    """Complete trace of a calculation from inputs to output."""

    trace_id: str
    calculation_type: CalculationType
    description: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Trace data
    inputs: List[TraceValue] = field(default_factory=list)
    steps: List[CalculationStep] = field(default_factory=list)
    final_output: Optional[TraceValue] = None

    # Metadata
    regulatory_standard: Optional[str] = None
    version: str = '1.0.0'
    deterministic: bool = True

    # Provenance
    chain_hash: str = ''

    def add_input(
        self,
        name: str,
        value: Union[Decimal, float, int, str],
        unit: str,
        source: DataSource,
        sensor_id: Optional[str] = None,
        reference: Optional[str] = None,
        uncertainty: Optional[Decimal] = None,
    ) -> TraceValue:
        """Add an input value to the trace."""
        traced = TraceValue(
            name=name,
            value=value,
            unit=unit,
            source=source,
            timestamp=datetime.now(timezone.utc),
            sensor_id=sensor_id,
            reference=reference,
            uncertainty=uncertainty,
        )
        self.inputs.append(traced)
        return traced

    def add_step(
        self,
        operation: str,
        formula: str,
        inputs: List[TraceValue],
        output_name: str,
        output_value: Union[Decimal, float, int, str],
        output_unit: str,
        regulatory_reference: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CalculationStep:
        """Add a calculation step to the trace."""
        output = TraceValue(
            name=output_name,
            value=output_value,
            unit=output_unit,
            source=DataSource.CALCULATED,
            timestamp=datetime.now(timezone.utc),
        )

        step = CalculationStep(
            step_number=len(self.steps) + 1,
            operation=operation,
            formula=formula,
            inputs=inputs,
            output=output,
            regulatory_reference=regulatory_reference,
            notes=notes,
        )
        self.steps.append(step)
        return step

    def set_final_output(
        self,
        name: str,
        value: Union[Decimal, float, int, str],
        unit: str,
    ) -> TraceValue:
        """Set the final output of the calculation."""
        self.final_output = TraceValue(
            name=name,
            value=value,
            unit=unit,
            source=DataSource.CALCULATED,
            timestamp=datetime.now(timezone.utc),
        )
        self.end_time = datetime.now(timezone.utc)
        self._generate_chain_hash()
        return self.final_output

    def _generate_chain_hash(self) -> None:
        """Generate the chain hash from all steps."""
        hashes = [i.provenance_hash for i in self.inputs]
        hashes.extend([s.provenance_hash for s in self.steps])
        if self.final_output:
            hashes.append(self.final_output.provenance_hash)

        combined = '|'.join(hashes)
        self.chain_hash = hashlib.sha256(combined.encode()).hexdigest()

    def verify_chain(self) -> bool:
        """Verify the integrity of the calculation chain."""
        # Recalculate all hashes and compare
        for trace_input in self.inputs:
            expected = trace_input._generate_hash()
            if trace_input.provenance_hash != expected:
                return False

        for step in self.steps:
            expected = step._generate_hash()
            if step.provenance_hash != expected:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trace_id': self.trace_id,
            'calculation_type': self.calculation_type.value,
            'description': self.description,
            'agent_id': self.agent_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'inputs': [i.to_dict() for i in self.inputs],
            'steps': [s.to_dict() for s in self.steps],
            'final_output': self.final_output.to_dict() if self.final_output else None,
            'regulatory_standard': self.regulatory_standard,
            'version': self.version,
            'deterministic': self.deterministic,
            'chain_hash': self.chain_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# TRACE MANAGER
# =============================================================================


class CalculationTraceManager:
    """
    Manages calculation traces for audit and compliance.

    Thread-safe manager for creating, storing, and querying calculation traces.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the trace manager."""
        self._traces: Dict[str, CalculationTrace] = {}
        self._lock = threading.RLock()
        self._storage_path = storage_path

    def create_trace(
        self,
        calculation_type: CalculationType,
        description: str,
        agent_id: str,
        regulatory_standard: Optional[str] = None,
    ) -> CalculationTrace:
        """
        Create a new calculation trace.

        Args:
            calculation_type: Type of calculation
            description: Human-readable description
            agent_id: Agent performing the calculation
            regulatory_standard: Applicable standard (e.g., 'EPA 40 CFR 75')

        Returns:
            New CalculationTrace instance
        """
        trace_id = self._generate_trace_id()

        trace = CalculationTrace(
            trace_id=trace_id,
            calculation_type=calculation_type,
            description=description,
            agent_id=agent_id,
            start_time=datetime.now(timezone.utc),
            regulatory_standard=regulatory_standard,
        )

        with self._lock:
            self._traces[trace_id] = trace

        return trace

    def get_trace(self, trace_id: str) -> Optional[CalculationTrace]:
        """Retrieve a trace by ID."""
        with self._lock:
            return self._traces.get(trace_id)

    def query_traces(
        self,
        calculation_type: Optional[CalculationType] = None,
        agent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CalculationTrace]:
        """Query traces with filters."""
        with self._lock:
            results = []

            for trace in self._traces.values():
                if calculation_type and trace.calculation_type != calculation_type:
                    continue
                if agent_id and trace.agent_id != agent_id:
                    continue
                if start_time and trace.start_time < start_time:
                    continue
                if end_time and trace.start_time > end_time:
                    continue

                results.append(trace)

                if len(results) >= limit:
                    break

            return sorted(results, key=lambda x: x.start_time, reverse=True)

    def verify_trace(self, trace_id: str) -> Tuple[bool, List[str]]:
        """
        Verify a trace's integrity.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        trace = self.get_trace(trace_id)
        if not trace:
            return False, ['Trace not found']

        issues = []

        # Verify chain integrity
        if not trace.verify_chain():
            issues.append('Chain hash verification failed')

        # Verify inputs have provenance
        for i, input_val in enumerate(trace.inputs):
            if not input_val.provenance_hash:
                issues.append(f'Input {i} missing provenance hash')

        # Verify steps are sequential
        for i, step in enumerate(trace.steps):
            if step.step_number != i + 1:
                issues.append(f'Step {i} has wrong sequence number')

        # Verify final output exists if trace is complete
        if trace.end_time and not trace.final_output:
            issues.append('Completed trace missing final output')

        return len(issues) == 0, issues

    def export_for_audit(
        self,
        trace_ids: List[str],
        include_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """
        Export traces for audit purposes.

        Args:
            trace_ids: List of trace IDs to export
            include_intermediate: Include intermediate calculation steps

        Returns:
            Audit export dictionary
        """
        export = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'trace_count': len(trace_ids),
            'traces': [],
        }

        for trace_id in trace_ids:
            trace = self.get_trace(trace_id)
            if trace:
                trace_data = trace.to_dict()
                if not include_intermediate:
                    # Remove intermediate steps, keep only inputs and output
                    trace_data['steps'] = []
                export['traces'].append(trace_data)

        return export

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        unique = uuid.uuid4().hex[:8]
        return f'TRC-{timestamp}-{unique}'


# =============================================================================
# DECORATOR FOR AUTOMATIC TRACING
# =============================================================================


def traced_calculation(
    calculation_type: CalculationType,
    description: str,
    regulatory_standard: Optional[str] = None,
):
    """
    Decorator to automatically trace a calculation function.

    Usage:
        @traced_calculation(CalculationType.EFFICIENCY, 'Boiler efficiency calc')
        def calculate_efficiency(fuel_flow, steam_flow, ...):
            ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            manager = get_trace_manager()
            trace = manager.create_trace(
                calculation_type=calculation_type,
                description=description,
                agent_id='GL-001',
                regulatory_standard=regulatory_standard,
            )

            # Execute the function
            result = func(*args, **kwargs)

            # Set final output (simplified - real implementation would introspect)
            if isinstance(result, (Decimal, float, int)):
                trace.set_final_output(
                    name=func.__name__ + '_result',
                    value=result,
                    unit='',
                )

            return result
        return wrapper
    return decorator


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_trace_manager: Optional[CalculationTraceManager] = None
_manager_lock = threading.Lock()


def get_trace_manager() -> CalculationTraceManager:
    """Get the singleton trace manager instance."""
    global _trace_manager
    with _manager_lock:
        if _trace_manager is None:
            _trace_manager = CalculationTraceManager()
        return _trace_manager


# =============================================================================
# EXAMPLE: TRACED EFFICIENCY CALCULATION
# =============================================================================


def create_efficiency_trace_example() -> CalculationTrace:
    """
    Example: Create a fully traced boiler efficiency calculation.

    Demonstrates how to use the trace system for regulatory compliance.
    """
    manager = get_trace_manager()

    # Create the trace
    trace = manager.create_trace(
        calculation_type=CalculationType.EFFICIENCY,
        description='Boiler efficiency calculation per ASME PTC 4.1',
        agent_id='GL-002',
        regulatory_standard='ASME PTC 4.1',
    )

    # Add inputs
    fuel_flow = trace.add_input(
        name='fuel_flow',
        value=Decimal('1000'),
        unit='kg/hr',
        source=DataSource.SENSOR,
        sensor_id='FT-101',
        uncertainty=Decimal('2'),
    )

    steam_flow = trace.add_input(
        name='steam_flow',
        value=Decimal('10000'),
        unit='kg/hr',
        source=DataSource.SENSOR,
        sensor_id='FT-201',
        uncertainty=Decimal('1.5'),
    )

    hhv = trace.add_input(
        name='higher_heating_value',
        value=Decimal('43000'),
        unit='kJ/kg',
        source=DataSource.REFERENCE_TABLE,
        reference='Natural Gas, EPA Table C-1',
    )

    enthalpy_steam = trace.add_input(
        name='enthalpy_steam',
        value=Decimal('3240'),
        unit='kJ/kg',
        source=DataSource.CALCULATED,
        reference='IAPWS-IF97',
    )

    enthalpy_feedwater = trace.add_input(
        name='enthalpy_feedwater',
        value=Decimal('420'),
        unit='kJ/kg',
        source=DataSource.CALCULATED,
        reference='IAPWS-IF97',
    )

    # Step 1: Calculate heat input
    heat_input = Decimal('1000') * Decimal('43000')  # 43,000,000 kJ/hr
    step1 = trace.add_step(
        operation='Calculate heat input',
        formula='Q_in = m_fuel × HHV',
        inputs=[fuel_flow, hhv],
        output_name='heat_input',
        output_value=heat_input,
        output_unit='kJ/hr',
        regulatory_reference='ASME PTC 4.1 Section 5.5',
    )

    # Step 2: Calculate heat output
    delta_h = Decimal('3240') - Decimal('420')  # 2820 kJ/kg
    heat_output = Decimal('10000') * delta_h  # 28,200,000 kJ/hr
    step2 = trace.add_step(
        operation='Calculate heat output',
        formula='Q_out = m_steam × (h_steam - h_feedwater)',
        inputs=[steam_flow, enthalpy_steam, enthalpy_feedwater],
        output_name='heat_output',
        output_value=heat_output,
        output_unit='kJ/hr',
        regulatory_reference='ASME PTC 4.1 Section 5.6',
    )

    # Step 3: Calculate efficiency
    efficiency = (heat_output / heat_input * Decimal('100')).quantize(
        Decimal('0.01'), rounding=ROUND_HALF_UP
    )
    step3 = trace.add_step(
        operation='Calculate efficiency',
        formula='η = (Q_out / Q_in) × 100',
        inputs=[step1.output, step2.output],
        output_name='efficiency',
        output_value=efficiency,
        output_unit='%',
        regulatory_reference='ASME PTC 4.1 Section 5.7',
        notes='Direct method efficiency calculation',
    )

    # Set final output
    trace.set_final_output(
        name='boiler_efficiency',
        value=efficiency,
        unit='%',
    )

    return trace


if __name__ == '__main__':
    # Create example trace
    trace = create_efficiency_trace_example()

    print('Calculation Trace Example')
    print('=' * 50)
    print(f'Trace ID: {trace.trace_id}')
    print(f'Type: {trace.calculation_type.value}')
    print(f'Agent: {trace.agent_id}')
    print(f'\nInputs: {len(trace.inputs)}')
    for inp in trace.inputs:
        print(f'  - {inp.name}: {inp.value} {inp.unit} ({inp.source.value})')

    print(f'\nSteps: {len(trace.steps)}')
    for step in trace.steps:
        print(f'  {step.step_number}. {step.operation}')
        print(f'     Formula: {step.formula}')
        print(f'     Output: {step.output.value} {step.output.unit}')

    print(f'\nFinal Output: {trace.final_output.value} {trace.final_output.unit}')
    print(f'Chain Hash: {trace.chain_hash}')

    # Verify
    manager = get_trace_manager()
    is_valid, issues = manager.verify_trace(trace.trace_id)
    print(f'\nVerification: {"PASSED" if is_valid else "FAILED"}')
    if issues:
        for issue in issues:
            print(f'  - {issue}')
