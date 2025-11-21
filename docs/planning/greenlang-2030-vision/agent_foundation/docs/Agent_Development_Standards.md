# Agent Development Standards

## Executive Summary

This document defines the development standards and best practices for building GreenLang AI agents. These standards ensure consistency, maintainability, and excellence across all agent implementations.

---

## Table of Contents

1. [Code Style and Conventions](#code-style-and-conventions)
2. [Agent Design Patterns](#agent-design-patterns)
3. [Best Practices for Tool Implementation](#best-practices-for-tool-implementation)
4. [Error Handling Patterns](#error-handling-patterns)
5. [Logging Standards](#logging-standards)
6. [Documentation Requirements](#documentation-requirements)
7. [Reference Implementation](#reference-implementation)

---

## Code Style and Conventions

### Python Standards

All GreenLang agents follow PEP 8 with additional conventions:

```python
# File: agent_style_example.py

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Constants in UPPER_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30

# Enums for type safety
class AgentStatus(Enum):
    """Agent operational status."""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"

# Use dataclasses for data structures
@dataclass
class AgentConfig:
    """Configuration for agent initialization."""
    name: str
    version: str
    max_workers: int = 4
    timeout: int = DEFAULT_TIMEOUT
    retry_attempts: int = MAX_RETRY_ATTEMPTS

class BaseAgent:
    """
    Base class for all GreenLang agents.

    Attributes:
        config: Agent configuration
        logger: Structured logger instance

    Methods:
        process: Main processing method
        validate_input: Input validation
        handle_error: Error handling
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = self._setup_logger()
        self._status = AgentStatus.IDLE

    def _setup_logger(self) -> logging.Logger:
        """Initialize structured logging."""
        logger = logging.getLogger(f"greenlang.{self.config.name}")
        logger.setLevel(logging.INFO)
        return logger

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.

        Args:
            input_data: Input dictionary with required fields

        Returns:
            Processed results dictionary

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing encounters errors
        """
        self._status = AgentStatus.PROCESSING

        try:
            # Validate input
            validated_data = await self.validate_input(input_data)

            # Process data
            result = await self._execute_processing(validated_data)

            # Post-process results
            final_result = await self._post_process(result)

            self._status = AgentStatus.COMPLETED
            return final_result

        except Exception as e:
            self._status = AgentStatus.ERROR
            return await self.handle_error(e, input_data)
```

### Naming Conventions

```python
# Classes: PascalCase
class DataProcessor:
    pass

# Functions and methods: snake_case
def calculate_emissions(data: Dict) -> float:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 1000

# Private methods: prefix with underscore
def _internal_helper() -> None:
    pass

# Type hints: Always use them
def process_data(
    input_data: List[Dict[str, Any]],
    config: Optional[Dict] = None
) -> Tuple[List[Dict], Dict[str, float]]:
    pass
```

---

## Agent Design Patterns

### 1. Single Responsibility Pattern

Each agent should have one clear purpose:

```python
# GOOD: Single responsibility
class EmissionsCalculatorAgent:
    """Calculates emissions from activity data."""

    async def calculate(self, activity_data: Dict) -> Dict:
        """Calculate emissions for given activity."""
        pass

# BAD: Multiple responsibilities
class SuperAgent:
    """Does everything."""

    async def calculate_and_report_and_validate(self):
        # Too many responsibilities
        pass
```

### 2. Composition Pattern

Complex agents compose simpler agents:

```python
from typing import List

class OrchestrationAgent:
    """Orchestrates multiple specialized agents."""

    def __init__(self):
        self.validators = ValidatorAgent()
        self.calculator = CalculatorAgent()
        self.reporter = ReporterAgent()

    async def process_pipeline(self, data: Dict) -> Dict:
        """Execute processing pipeline."""
        # Step 1: Validate
        validated = await self.validators.validate(data)

        # Step 2: Calculate
        results = await self.calculator.calculate(validated)

        # Step 3: Report
        report = await self.reporter.generate(results)

        return report
```

### 3. Strategy Pattern

For algorithms that vary:

```python
from abc import ABC, abstractmethod

class CalculationStrategy(ABC):
    """Abstract strategy for calculations."""

    @abstractmethod
    async def calculate(self, data: Dict) -> float:
        """Calculate result based on strategy."""
        pass

class Scope1Strategy(CalculationStrategy):
    """Scope 1 emissions calculation."""

    async def calculate(self, data: Dict) -> float:
        return data['fuel_consumption'] * data['emission_factor']

class Scope2Strategy(CalculationStrategy):
    """Scope 2 emissions calculation."""

    async def calculate(self, data: Dict) -> float:
        return data['electricity_kwh'] * data['grid_factor']

class EmissionsAgent:
    """Agent using strategy pattern."""

    def __init__(self, strategy: CalculationStrategy):
        self.strategy = strategy

    async def process(self, data: Dict) -> Dict:
        result = await self.strategy.calculate(data)
        return {'emissions': result, 'unit': 'tCO2e'}
```

### 4. Chain of Responsibility Pattern

For sequential processing with bail-out:

```python
class ProcessingStep(ABC):
    """Abstract processing step."""

    def __init__(self):
        self.next_step: Optional[ProcessingStep] = None

    def set_next(self, step: 'ProcessingStep') -> 'ProcessingStep':
        self.next_step = step
        return step

    async def handle(self, request: Dict) -> Dict:
        result = await self._process(request)

        if self.next_step and result.get('continue', True):
            return await self.next_step.handle(result)
        return result

    @abstractmethod
    async def _process(self, request: Dict) -> Dict:
        pass

# Usage
validator = ValidationStep()
enricher = EnrichmentStep()
calculator = CalculationStep()

validator.set_next(enricher).set_next(calculator)
result = await validator.handle(input_data)
```

---

## Best Practices for Tool Implementation

### Tool Interface Standards

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Tool(Protocol):
    """Standard tool interface."""

    name: str
    description: str

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with parameters."""
        ...

    def validate_params(self, **kwargs) -> bool:
        """Validate tool parameters."""
        ...

class DatabaseTool:
    """Database interaction tool."""

    def __init__(self, connection_string: str):
        self.name = "database"
        self.description = "Interact with database"
        self.conn = self._connect(connection_string)

    async def execute(self, query: str, params: Dict = None) -> Dict[str, Any]:
        """Execute database query."""
        try:
            # Validate SQL injection risks
            if not self._is_safe_query(query):
                raise SecurityError("Unsafe query detected")

            # Execute with timeout
            async with asyncio.timeout(30):
                result = await self.conn.execute(query, params)

            return {
                'success': True,
                'data': result.fetchall(),
                'rows_affected': result.rowcount
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
```

### Tool Registration Pattern

```python
class ToolRegistry:
    """Central tool registry."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if not isinstance(tool, Tool):
            raise TypeError("Tool must implement Tool protocol")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self._tools.keys())

# Usage
registry = ToolRegistry()
registry.register(DatabaseTool(conn_string))
registry.register(CalculatorTool())
registry.register(ValidatorTool())

# Agent using tools
class ToolUsingAgent:
    def __init__(self, registry: ToolRegistry):
        self.tools = registry

    async def use_tool(self, tool_name: str, **params) -> Dict:
        tool = self.tools.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        return await tool.execute(**params)
```

---

## Error Handling Patterns

### Exception Hierarchy

```python
class GreenLangError(Exception):
    """Base exception for all GreenLang errors."""
    pass

class ValidationError(GreenLangError):
    """Input validation errors."""
    pass

class ProcessingError(GreenLangError):
    """Processing errors."""
    pass

class ConfigurationError(GreenLangError):
    """Configuration errors."""
    pass

class RetryableError(GreenLangError):
    """Errors that can be retried."""
    pass
```

### Error Handling Strategies

```python
from functools import wraps
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (RetryableError,)
):
    """Decorator for retry logic."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            attempt = 1
            current_delay = delay

            while attempt <= max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise

                    logging.warning(
                        f"Attempt {attempt} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

        return wrapper
    return decorator

class ResilientAgent:
    """Agent with resilient error handling."""

    @retry_on_error(max_attempts=3, delay=1.0)
    async def fetch_data(self, source: str) -> Dict:
        """Fetch data with retry."""
        response = await self.client.get(source)
        if response.status >= 500:
            raise RetryableError(f"Server error: {response.status}")
        return response.json()

    async def process_with_fallback(self, data: Dict) -> Dict:
        """Process with fallback strategy."""
        try:
            # Try primary processing
            return await self.primary_processor.process(data)
        except ProcessingError as e:
            self.logger.warning(f"Primary processing failed: {e}")

            # Try fallback
            try:
                return await self.fallback_processor.process(data)
            except Exception as e2:
                self.logger.error(f"Fallback also failed: {e2}")

                # Return degraded result
                return {
                    'status': 'degraded',
                    'partial_result': self._extract_partial(data),
                    'error': str(e)
                }
```

### Circuit Breaker Pattern

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >
            timedelta(seconds=self.recovery_timeout)
        )

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'closed'

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
```

---

## Logging Standards

### Structured Logging

```python
import structlog
from typing import Any

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

class LoggingAgent:
    """Agent with structured logging."""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.logger = self.logger.bind(
            agent_name=name,
            agent_version="1.0.0"
        )

    async def process(self, request_id: str, data: Dict) -> Dict:
        """Process with detailed logging."""
        # Create request-scoped logger
        log = self.logger.bind(request_id=request_id)

        log.info(
            "processing_started",
            data_size=len(data),
            data_keys=list(data.keys())
        )

        try:
            # Validation
            log.debug("validation_started")
            validated = await self._validate(data)
            log.debug(
                "validation_completed",
                validation_errors=validated.get('errors', [])
            )

            # Processing
            log.debug("calculation_started")
            result = await self._calculate(validated)
            log.debug(
                "calculation_completed",
                result_summary=self._summarize(result)
            )

            log.info(
                "processing_completed",
                duration_ms=self._get_duration(),
                success=True
            )

            return result

        except Exception as e:
            log.error(
                "processing_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=self._get_duration(),
                success=False,
                exc_info=True
            )
            raise
```

### Log Levels and Usage

```python
# Log level guidelines
class LoggingGuidelines:
    """
    DEBUG: Detailed diagnostic information
    - Input/output of internal functions
    - State transitions
    - Configuration values

    INFO: Important business events
    - Request received/completed
    - Major state changes
    - Performance metrics

    WARNING: Potentially harmful situations
    - Deprecated feature usage
    - Retry attempts
    - Fallback to degraded mode

    ERROR: Error events but application continues
    - Handled exceptions
    - Failed operations with fallback

    CRITICAL: Critical problems requiring immediate attention
    - Unhandled exceptions
    - System resource exhaustion
    - Data corruption detected
    """

    @staticmethod
    def example_usage():
        logger = structlog.get_logger()

        # DEBUG
        logger.debug("cache_check", key="user_123", found=True)

        # INFO
        logger.info("report_generated",
                   report_id="rep_456",
                   format="PDF",
                   size_mb=2.5)

        # WARNING
        logger.warning("rate_limit_approaching",
                      current=95,
                      limit=100,
                      reset_in_seconds=60)

        # ERROR
        logger.error("external_api_failed",
                    service="emissions_db",
                    status_code=503,
                    retry_count=3)

        # CRITICAL
        logger.critical("database_connection_lost",
                       connection_pool_size=0,
                       pending_queries=150)
```

---

## Documentation Requirements

### Code Documentation Standards

```python
"""
Module: carbon_calculator.py
Purpose: Calculate carbon emissions from activity data
Author: GreenLang Team
Created: 2025-01-01
Modified: 2025-01-15
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

class CarbonCalculator:
    """
    Calculate carbon emissions using various methodologies.

    This class implements multiple calculation methodologies including:
    - IPCC 2006 Guidelines
    - GHG Protocol
    - ISO 14064 standards

    Attributes:
        emission_factors: Dictionary of emission factors by activity type
        global_warming_potentials: GWP values for different gases

    Example:
        >>> calculator = CarbonCalculator()
        >>> result = calculator.calculate_scope1(
        ...     fuel_type="natural_gas",
        ...     consumption=1000,
        ...     unit="m3"
        ... )
        >>> print(f"Emissions: {result['tCO2e']} tCO2e")

    Note:
        All calculations are performed using IPCC AR6 GWP values
        unless specified otherwise.
    """

    def calculate_scope1(
        self,
        fuel_type: str,
        consumption: float,
        unit: str,
        custom_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Scope 1 emissions from direct fuel combustion.

        Args:
            fuel_type: Type of fuel (e.g., "natural_gas", "diesel", "coal")
            consumption: Amount of fuel consumed
            unit: Unit of consumption (e.g., "m3", "L", "kg")
            custom_factor: Optional custom emission factor (kgCO2e/unit)

        Returns:
            Dictionary containing:
                - tCO2e: Total emissions in metric tons CO2 equivalent
                - kgCO2e: Total emissions in kilograms CO2 equivalent
                - calculation_method: Method used for calculation
                - emission_factor: Applied emission factor

        Raises:
            ValueError: If fuel_type is not recognized and no custom_factor provided
            TypeError: If consumption is not numeric

        Example:
            >>> result = calculator.calculate_scope1(
            ...     fuel_type="diesel",
            ...     consumption=1000,
            ...     unit="L"
            ... )

        Note:
            Default emission factors are from IPCC 2006 Guidelines.
            For region-specific factors, use custom_factor parameter.
        """
        # Implementation here
        pass
```

### API Documentation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="GreenLang Carbon Calculator API",
    description="Calculate carbon emissions from activity data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class CalculationRequest(BaseModel):
    """Request model for emission calculation."""

    fuel_type: str = Field(
        ...,
        description="Type of fuel (natural_gas, diesel, coal, etc.)",
        example="natural_gas"
    )
    consumption: float = Field(
        ...,
        description="Amount of fuel consumed",
        gt=0,
        example=1000
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (m3, L, kg)",
        example="m3"
    )
    custom_factor: Optional[float] = Field(
        None,
        description="Custom emission factor (kgCO2e/unit)",
        gt=0
    )

class CalculationResponse(BaseModel):
    """Response model for emission calculation."""

    tCO2e: float = Field(
        ...,
        description="Total emissions in metric tons CO2 equivalent"
    )
    kgCO2e: float = Field(
        ...,
        description="Total emissions in kilograms CO2 equivalent"
    )
    calculation_method: str = Field(
        ...,
        description="Methodology used for calculation"
    )
    emission_factor: float = Field(
        ...,
        description="Applied emission factor"
    )

@app.post(
    "/calculate/scope1",
    response_model=CalculationResponse,
    summary="Calculate Scope 1 Emissions",
    description="Calculate direct emissions from fuel combustion",
    response_description="Calculated emission values",
    tags=["Calculations"]
)
async def calculate_scope1_emissions(
    request: CalculationRequest
) -> CalculationResponse:
    """
    Calculate Scope 1 emissions from direct fuel combustion.

    This endpoint calculates direct GHG emissions from sources that are
    owned or controlled by the reporting organization.

    **Supported Fuel Types:**
    - natural_gas
    - diesel
    - gasoline
    - coal
    - propane

    **Calculation Methodology:**

    Uses IPCC 2006 Guidelines with AR6 GWP values.

    Formula: `Emissions = Consumption × Emission Factor`
    """
    # Implementation
    pass
```

### README Template

```markdown
# Agent Name

## Overview

Brief description of what this agent does and its primary use case.

## Features

- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Installation

\`\`\`bash
pip install greenlang-agent-name
\`\`\`

## Quick Start

\`\`\`python
from greenlang.agents import AgentName

agent = AgentName(config={
    'api_key': 'your-key',
    'region': 'us-east-1'
})

result = await agent.process({
    'input': 'data'
})

print(result)
\`\`\`

## Configuration

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| api_key | str | Yes | - | API key for authentication |
| region | str | No | us-east-1 | AWS region |
| timeout | int | No | 30 | Timeout in seconds |

## API Reference

### Methods

#### process(input_data)

Process input data and return results.

**Parameters:**
- `input_data` (dict): Input data dictionary

**Returns:**
- `dict`: Processing results

**Example:**
\`\`\`python
result = await agent.process({
    'field1': 'value1',
    'field2': 'value2'
})
\`\`\`

## Architecture

\`\`\`mermaid
graph TD
    A[Input] --> B[Validation]
    B --> C[Processing]
    C --> D[Output]

    B --> E[Error Handler]
    C --> E
    E --> F[Retry Logic]
    F --> C
\`\`\`

## Testing

\`\`\`bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_agent.py::test_process

# Run with coverage
pytest --cov=greenlang.agents tests/
\`\`\`

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 1000 req/s | Single instance |
| Latency P50 | 10ms | - |
| Latency P99 | 50ms | - |
| Memory Usage | 100MB | Base consumption |

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

MIT License - see [LICENSE](../LICENSE)
```

---

## Reference Implementation

### Complete Agent Example

```python
"""
Reference implementation of a GreenLang agent following all standards.
"""

import asyncio
import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    requests_processed: int = 0
    requests_failed: int = 0
    total_processing_time: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

@dataclass
class ProcessingResult:
    """Result of agent processing."""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ReferenceAgent:
    """
    Reference implementation demonstrating all standards.

    This agent implements:
    - Proper error handling
    - Structured logging
    - Metrics collection
    - Retry logic
    - Circuit breaker
    - Input validation
    - Result caching
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.logger = structlog.get_logger(name)
        self.metrics = AgentMetrics()
        self._cache: Dict[str, ProcessingResult] = {}
        self._circuit_breaker = CircuitBreaker()

    async def process(
        self,
        input_data: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Main processing method following all standards.

        Args:
            input_data: Input data to process
            request_id: Optional request ID for tracking

        Returns:
            ProcessingResult with success status and data
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = self._generate_request_id(input_data)

        # Create request-scoped logger
        log = self.logger.bind(request_id=request_id)

        # Start timing
        start_time = asyncio.get_event_loop().time()

        try:
            log.info("processing_started", input_keys=list(input_data.keys()))

            # Check cache
            cache_key = self._get_cache_key(input_data)
            if cache_key in self._cache:
                log.debug("cache_hit", cache_key=cache_key)
                return self._cache[cache_key]

            # Validate input
            validation_result = await self._validate_input(input_data, log)
            if not validation_result['valid']:
                return ProcessingResult(
                    success=False,
                    data={},
                    errors=validation_result['errors']
                )

            # Process with circuit breaker
            result = await self._circuit_breaker.call(
                self._execute_processing,
                input_data,
                log
            )

            # Cache result
            self._cache[cache_key] = result

            # Update metrics
            self.metrics.requests_processed += 1
            self.metrics.total_processing_time += (
                asyncio.get_event_loop().time() - start_time
            )

            log.info(
                "processing_completed",
                duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                success=True
            )

            return result

        except Exception as e:
            # Update metrics
            self.metrics.requests_failed += 1
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()

            log.error(
                "processing_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                exc_info=True
            )

            return ProcessingResult(
                success=False,
                data={},
                errors=[f"Processing failed: {str(e)}"]
            )

    async def _validate_input(
        self,
        input_data: Dict[str, Any],
        log: structlog.BoundLogger
    ) -> Dict[str, Any]:
        """Validate input data."""
        log.debug("validation_started")

        errors = []
        warnings = []

        # Check required fields
        required_fields = self.config.get('required_fields', [])
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")

        # Validate data types
        type_requirements = self.config.get('type_requirements', {})
        for field, expected_type in type_requirements.items():
            if field in input_data:
                if not isinstance(input_data[field], expected_type):
                    errors.append(
                        f"Field {field} must be of type {expected_type.__name__}"
                    )

        # Check data ranges
        range_requirements = self.config.get('range_requirements', {})
        for field, (min_val, max_val) in range_requirements.items():
            if field in input_data:
                value = input_data[field]
                if value < min_val or value > max_val:
                    errors.append(
                        f"Field {field} must be between {min_val} and {max_val}"
                    )

        log.debug(
            "validation_completed",
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    async def _execute_processing(
        self,
        input_data: Dict[str, Any],
        log: structlog.BoundLogger
    ) -> ProcessingResult:
        """Execute actual processing logic."""
        log.debug("execution_started")

        # Simulate processing
        await asyncio.sleep(0.1)

        # Process data (example implementation)
        processed_data = {
            'input_hash': self._generate_hash(input_data),
            'processed_at': datetime.now().isoformat(),
            'processor': self.name,
            'result': sum(
                v for v in input_data.values()
                if isinstance(v, (int, float))
            )
        }

        log.debug("execution_completed", result_keys=list(processed_data.keys()))

        return ProcessingResult(
            success=True,
            data=processed_data,
            metadata={
                'processor_version': '1.0.0',
                'processing_time_ms': 100
            }
        )

    def _generate_request_id(self, input_data: Dict) -> str:
        """Generate unique request ID."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}_{json.dumps(input_data, sort_keys=True)}".encode()
        ).hexdigest()[:8]

    def _get_cache_key(self, input_data: Dict) -> str:
        """Generate cache key from input data."""
        return hashlib.md5(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

    def _generate_hash(self, data: Dict) -> str:
        """Generate hash of data."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            'requests_processed': self.metrics.requests_processed,
            'requests_failed': self.metrics.requests_failed,
            'success_rate': (
                self.metrics.requests_processed /
                max(1, self.metrics.requests_processed + self.metrics.requests_failed)
            ),
            'average_processing_time_ms': (
                self.metrics.total_processing_time * 1000 /
                max(1, self.metrics.requests_processed)
            ),
            'last_error': self.metrics.last_error,
            'last_error_time': (
                self.metrics.last_error_time.isoformat()
                if self.metrics.last_error_time else None
            ),
            'cache_size': len(self._cache)
        }


# Example usage
async def main():
    """Example usage of reference agent."""

    # Configure agent
    config = {
        'required_fields': ['value1', 'value2'],
        'type_requirements': {
            'value1': (int, float),
            'value2': (int, float)
        },
        'range_requirements': {
            'value1': (0, 1000),
            'value2': (0, 1000)
        }
    }

    # Create agent
    agent = ReferenceAgent("example_agent", config)

    # Process data
    result = await agent.process({
        'value1': 100,
        'value2': 200,
        'metadata': {'source': 'test'}
    })

    print(f"Success: {result.success}")
    print(f"Data: {result.data}")

    # Get metrics
    metrics = await agent.get_metrics()
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

These development standards ensure:

1. **Consistency** - All agents follow the same patterns
2. **Maintainability** - Code is easy to understand and modify
3. **Reliability** - Proper error handling and resilience
4. **Observability** - Comprehensive logging and metrics
5. **Documentation** - Clear documentation at all levels
6. **Quality** - High standards for code quality

All GreenLang agents must adhere to these standards to ensure world-class quality and maintainability.