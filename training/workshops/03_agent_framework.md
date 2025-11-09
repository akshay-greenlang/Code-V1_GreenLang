# Workshop 3: Building Agents with GreenLang Framework

**Duration:** 3 hours
**Level:** Intermediate
**Prerequisites:** Workshops 1-2 completed

---

## Workshop Overview

Master the Agent framework - the foundation of all GreenLang applications. Learn to build production-ready agents with proper lifecycle management, error handling, and batch processing.

### Learning Objectives

- Understand Agent base class architecture
- Implement agent lifecycle (setup, execute, teardown)
- Use agent templates (Calculator, DataIntake, etc.)
- Build batch processing agents
- Handle errors and retries
- Monitor agent performance
- Deploy agents to production

---

## Part 1: Agent Architecture (30 minutes)

### Why Agents?

**The Problem:**
Every application needs consistent patterns for:
- Initialization and cleanup
- Error handling and retries
- Logging and monitoring
- State management
- Testing and deployment

**The Solution:**
Agent base class that enforces best practices.

### Agent Lifecycle

```python
from GL_COMMONS.infrastructure.agents import Agent

class MyAgent(Agent):
    """All agents follow this pattern."""

    def setup(self):
        """
        Called once at startup.
        Initialize resources, connections, infrastructure.
        """
        pass

    def execute(self):
        """
        Called for each execution.
        Main business logic goes here.
        Returns: dict with results
        """
        pass

    def teardown(self):
        """
        Called at shutdown.
        Cleanup resources, close connections.
        """
        pass
```

### Execution Flow

```
1. Agent instantiation
   └─> __init__()

2. Setup phase
   └─> setup()
       ├─ Initialize infrastructure
       ├─ Load configurations
       ├─ Establish connections
       └─ Prepare resources

3. Execution phase (can run multiple times)
   └─> execute()
       ├─ Validate input
       ├─ Execute business logic
       ├─ Handle errors
       └─ Return results

4. Teardown phase
   └─> teardown()
       ├─ Close connections
       ├─ Cleanup resources
       └─ Flush logs
```

---

## Part 2: Building Your First Agent (40 minutes)

### Example: Emission Calculator Agent

```python
# emission_calculator_agent.py
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.validation import ValidationFramework
import logging

logger = logging.getLogger(__name__)

class EmissionCalculatorAgent(Agent):
    """Calculates carbon emissions for various activities."""

    def __init__(self):
        super().__init__(
            name="emission_calculator",
            version="1.0.0",
            description="Calculates carbon emissions"
        )

        # Agent-specific attributes
        self.llm_session = None
        self.cache = None
        self.validator = None

    def setup(self):
        """Initialize infrastructure components."""
        logger.info("Setting up EmissionCalculatorAgent")

        # LLM for calculation logic
        self.llm_session = ChatSession(
            provider="openai",
            model="gpt-4",
            system_message="""You are a carbon emission calculation expert.
            Calculate emissions based on activity data.
            Return results as JSON: {"co2_kg": float, "methodology": str}"""
        )

        # Cache for repeated calculations
        self.cache = CacheManager()

        # Validator for input data
        self.validator = ValidationFramework()

        # Load emission factors from database
        self._load_emission_factors()

        logger.info("Setup complete")

    def execute(self):
        """Calculate emissions for given activity."""

        # Get input data
        activity_type = self.input_data.get("activity_type")
        amount = self.input_data.get("amount")
        unit = self.input_data.get("unit")

        # Validate input
        self._validate_input(activity_type, amount, unit)

        # Check cache
        cache_key = f"emission:{activity_type}:{amount}:{unit}"
        cached_result = self.cache.get(cache_key)

        if cached_result:
            logger.info("Cache hit for emission calculation")
            return cached_result

        # Calculate emission
        result = self._calculate_emission(activity_type, amount, unit)

        # Cache result
        self.cache.set(cache_key, result, ttl=3600)

        # Update metrics
        self.metrics["calculations_performed"] += 1
        self.metrics["total_emissions_calculated"] += result["co2_kg"]

        return result

    def teardown(self):
        """Cleanup resources."""
        if self.llm_session:
            self.llm_session.close()

        logger.info("Teardown complete")

    def _validate_input(self, activity_type, amount, unit):
        """Validate input parameters."""
        schema = {
            "activity_type": {"type": "string", "required": True},
            "amount": {"type": "number", "min": 0, "required": True},
            "unit": {"type": "string", "required": True}
        }

        data = {
            "activity_type": activity_type,
            "amount": amount,
            "unit": unit
        }

        self.validator.validate(data, schema)

    def _calculate_emission(self, activity_type, amount, unit):
        """Calculate emission using LLM and emission factors."""

        # Get emission factor
        emission_factor = self.emission_factors.get(activity_type)

        if emission_factor:
            # Use direct calculation
            co2_kg = amount * emission_factor
            methodology = "Emission Factor Database"

        else:
            # Use LLM for complex calculations
            prompt = f"""Calculate CO2 emissions for:
            Activity: {activity_type}
            Amount: {amount} {unit}

            Return JSON with co2_kg and methodology."""

            response = self.llm_session.send_message(prompt)
            result = json.loads(response)
            co2_kg = result["co2_kg"]
            methodology = result["methodology"]

        return {
            "activity_type": activity_type,
            "amount": amount,
            "unit": unit,
            "co2_kg": co2_kg,
            "methodology": methodology,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _load_emission_factors(self):
        """Load emission factors from database."""
        # In real implementation, load from database
        self.emission_factors = {
            "electricity_kwh": 0.5,  # kg CO2 per kWh
            "flight_km": 0.115,      # kg CO2 per km
            "car_km": 0.17           # kg CO2 per km
        }
```

### Using the Agent

```python
# test_emission_agent.py
from emission_calculator_agent import EmissionCalculatorAgent

# Create agent
agent = EmissionCalculatorAgent()

# Setup (called once)
agent.setup()

# Execute with different inputs
inputs = [
    {"activity_type": "electricity_kwh", "amount": 100, "unit": "kWh"},
    {"activity_type": "flight_km", "amount": 500, "unit": "km"},
    {"activity_type": "car_km", "amount": 50, "unit": "km"}
]

for input_data in inputs:
    result = agent.execute_with_input(input_data)
    print(f"{input_data['activity_type']}: {result['co2_kg']} kg CO2")

# Teardown (called once)
agent.teardown()
```

---

## Part 3: Agent Templates (45 minutes)

### CalculatorAgent Template

Pre-built template for calculation-heavy agents.

```python
from GL_COMMONS.infrastructure.agents.templates import CalculatorAgent

class MyCalculator(CalculatorAgent):
    """Inherit from CalculatorAgent for automatic features."""

    def calculate(self, input_data: dict) -> dict:
        """
        Implement your calculation logic.

        Automatic features:
        - Input validation
        - Result caching
        - Error handling
        - Performance tracking
        """

        # Your calculation logic
        result = self._perform_calculation(input_data)

        return result

# Usage
calc = MyCalculator()
calc.setup()
result = calc.execute_with_input({"value": 100})
```

### DataIntakeAgent Template

For agents that ingest and process data.

```python
from GL_COMMONS.infrastructure.agents.templates import DataIntakeAgent

class CSVIntakeAgent(DataIntakeAgent):
    """Ingest CSV files into database."""

    def validate_source(self, source_path: str) -> bool:
        """Validate data source."""
        return source_path.endswith('.csv') and os.path.exists(source_path)

    def extract_data(self, source_path: str) -> list:
        """Extract data from source."""
        import csv
        with open(source_path, 'r') as f:
            return list(csv.DictReader(f))

    def transform_data(self, raw_data: list) -> list:
        """Transform data to target schema."""
        return [self._transform_row(row) for row in raw_data]

    def load_data(self, transformed_data: list) -> dict:
        """Load data to target system."""
        self.db.bulk_insert("emissions", transformed_data)
        return {"records_loaded": len(transformed_data)}

    def _transform_row(self, row: dict) -> dict:
        """Transform individual row."""
        return {
            "company": row["Company"],
            "year": int(row["Year"]),
            "emissions": float(row["CO2_tons"])
        }

# Usage
agent = CSVIntakeAgent()
agent.setup()
result = agent.execute_with_input({
    "source_path": "data/emissions.csv"
})
print(f"Loaded {result['records_loaded']} records")
```

### ReportingAgent Template

For agents that generate reports.

```python
from GL_COMMONS.infrastructure.agents.templates import ReportingAgent

class EmissionsReportAgent(ReportingAgent):
    """Generate emissions reports."""

    def gather_data(self, parameters: dict) -> dict:
        """Gather data for report."""
        company = parameters["company"]
        year = parameters["year"]

        return self.db.query(
            "SELECT * FROM emissions WHERE company=? AND year=?",
            [company, year]
        )

    def analyze_data(self, data: dict) -> dict:
        """Analyze data."""
        total = sum(row["emissions"] for row in data)
        avg = total / len(data) if data else 0

        return {
            "total_emissions": total,
            "average": avg,
            "record_count": len(data)
        }

    def generate_report(self, analysis: dict) -> str:
        """Generate report document."""
        return f"""
        Emissions Report
        ================

        Total Emissions: {analysis['total_emissions']:,.0f} tons CO2
        Average: {analysis['average']:,.0f} tons CO2
        Records: {analysis['record_count']}
        """

    def distribute_report(self, report: str, parameters: dict):
        """Distribute report to stakeholders."""
        # Send via email, save to file, etc.
        self.email_service.send(
            to=parameters["recipients"],
            subject="Emissions Report",
            body=report
        )
```

---

## Part 4: Batch Processing (40 minutes)

### BatchProcessor

Process large datasets efficiently.

```python
from GL_COMMONS.infrastructure.agents import BatchProcessor

class EmissionBatchProcessor(BatchProcessor):
    """Process emissions in batches."""

    def __init__(self):
        super().__init__(
            batch_size=100,        # Process 100 items at a time
            max_workers=4,         # Use 4 parallel workers
            retry_failed=True      # Retry failed items
        )

    def process_item(self, item: dict) -> dict:
        """Process single item."""
        # Calculate emission for one activity
        return self._calculate_emission(item)

    def process_batch(self, batch: list) -> list:
        """Process batch of items (optional optimization)."""
        # Override for batch-specific optimizations
        # e.g., bulk database queries

        # Get all emission factors at once
        activity_types = [item["activity_type"] for item in batch]
        factors = self._get_emission_factors_bulk(activity_types)

        # Calculate all emissions
        results = []
        for item in batch:
            factor = factors[item["activity_type"]]
            result = item["amount"] * factor
            results.append({"item": item, "emission": result})

        return results

# Usage
processor = EmissionBatchProcessor()
processor.setup()

# Process 10,000 items
items = load_emission_data()  # 10,000 records

results = processor.execute_with_input({
    "items": items
})

print(f"Processed: {results['total_processed']}")
print(f"Failed: {results['total_failed']}")
print(f"Duration: {results['duration_seconds']}s")
```

### Parallel Processing

```python
from GL_COMMONS.infrastructure.agents import ParallelAgent

class ParallelEmissionAgent(ParallelAgent):
    """Process emissions in parallel."""

    def __init__(self):
        super().__init__(
            num_workers=8,              # 8 parallel workers
            worker_type="process"       # "process" or "thread"
        )

    def execute_worker(self, worker_id: int, items: list) -> list:
        """Execute in worker."""
        results = []
        for item in items:
            result = self._calculate_emission(item)
            results.append(result)
        return results

# Usage
agent = ParallelEmissionAgent()
agent.setup()

# Automatically distributes across 8 workers
result = agent.execute_with_input({
    "items": large_dataset
})
```

---

## Part 5: Error Handling & Retries (30 minutes)

### Automatic Retry

```python
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.exceptions import RetryableError

class RobustAgent(Agent):
    """Agent with automatic retry."""

    def __init__(self):
        super().__init__(
            name="robust_agent",
            max_retries=3,              # Retry up to 3 times
            retry_delay=5,              # Wait 5 seconds
            exponential_backoff=True    # 5s, 10s, 20s
        )

    def execute(self):
        """Execute with automatic retry."""
        try:
            return self._perform_operation()

        except RetryableError as e:
            # Automatically retried by base class
            raise

        except Exception as e:
            # Non-retryable errors fail immediately
            logger.error(f"Non-retryable error: {e}")
            raise
```

### Error Recovery

```python
class RecoverableAgent(Agent):
    """Agent with error recovery."""

    def execute(self):
        """Execute with recovery."""
        try:
            return self._primary_method()

        except PrimaryMethodError:
            logger.warning("Primary method failed, using fallback")
            return self._fallback_method()

    def _primary_method(self):
        """Primary execution path."""
        # Try the optimal method
        pass

    def _fallback_method(self):
        """Fallback execution path."""
        # Less optimal but more reliable
        pass
```

### Circuit Breaker

```python
from GL_COMMONS.infrastructure.agents import CircuitBreakerAgent

class ExternalAPIAgent(CircuitBreakerAgent):
    """Agent with circuit breaker for external API."""

    def __init__(self):
        super().__init__(
            failure_threshold=5,     # Open after 5 failures
            recovery_timeout=60,     # Try again after 60s
            expected_exception=APIError
        )

    def execute(self):
        """Call external API with circuit breaker."""
        # If circuit is open, fails fast without calling API
        response = self._call_external_api()
        return response
```

---

## Part 6: Hands-On Lab - Build a Complete Agent (60 minutes)

### Lab: Data Intake Agent for CSRD Reporting

**Requirements:**
1. Ingest CSV files with emission data
2. Validate data against schema
3. Calculate additional metrics
4. Store in database
5. Generate summary report
6. Handle errors and retries

### Step 1: Agent Structure

```python
# csrd_intake_agent.py
from GL_COMMONS.infrastructure.agents.templates import DataIntakeAgent
from GL_COMMONS.infrastructure.validation import ValidationFramework
from GL_COMMONS.infrastructure.database import DatabaseManager
from GL_COMMONS.infrastructure.cache import CacheManager
import csv
import logging

logger = logging.getLogger(__name__)

class CSRDIntakeAgent(DataIntakeAgent):
    """Ingest CSRD emission data from CSV files."""

    def __init__(self):
        super().__init__(
            name="csrd_intake_agent",
            version="1.0.0",
            description="Ingest CSRD emission data"
        )

        self.validator = None
        self.db = None
        self.cache = None

    def setup(self):
        """Initialize infrastructure."""
        logger.info("Setting up CSRDIntakeAgent")

        self.validator = ValidationFramework()
        self.db = DatabaseManager()
        self.cache = CacheManager()

        # Define validation schema
        self.schema = {
            "company_name": {"type": "string", "required": True},
            "year": {"type": "integer", "min": 2020, "max": 2030},
            "scope_1": {"type": "number", "min": 0},
            "scope_2": {"type": "number", "min": 0},
            "scope_3": {"type": "number", "min": 0},
            "unit": {"type": "string", "enum": ["tons_co2", "kg_co2"]}
        }

        logger.info("Setup complete")

    # TODO: Implement validate_source
    # TODO: Implement extract_data
    # TODO: Implement transform_data
    # TODO: Implement load_data
```

### Step 2: Implement Methods

```python
def validate_source(self, source_path: str) -> bool:
    """Validate CSV file exists and is readable."""
    if not source_path.endswith('.csv'):
        raise ValueError("Source must be a CSV file")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"File not found: {source_path}")

    # Check file is not empty
    if os.path.getsize(source_path) == 0:
        raise ValueError("File is empty")

    return True

def extract_data(self, source_path: str) -> list:
    """Extract data from CSV."""
    logger.info(f"Extracting data from {source_path}")

    data = []
    with open(source_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    logger.info(f"Extracted {len(data)} records")
    return data

def transform_data(self, raw_data: list) -> list:
    """Transform and validate data."""
    logger.info("Transforming data")

    transformed = []
    errors = []

    for i, row in enumerate(raw_data):
        try:
            # Transform row
            transformed_row = self._transform_row(row)

            # Validate
            self.validator.validate(transformed_row, self.schema)

            transformed.append(transformed_row)

        except Exception as e:
            errors.append({
                "row": i + 1,
                "data": row,
                "error": str(e)
            })

    # Log errors
    if errors:
        logger.warning(f"{len(errors)} rows failed validation")
        for error in errors[:10]:  # Log first 10
            logger.warning(f"Row {error['row']}: {error['error']}")

    logger.info(f"Transformed {len(transformed)} valid records")

    return transformed

def load_data(self, transformed_data: list) -> dict:
    """Load data to database."""
    logger.info("Loading data to database")

    # Bulk insert
    self.db.bulk_insert("csrd_emissions", transformed_data)

    # Generate summary
    total_scope1 = sum(row["scope_1"] for row in transformed_data)
    total_scope2 = sum(row["scope_2"] for row in transformed_data)
    total_scope3 = sum(row["scope_3"] for row in transformed_data)

    summary = {
        "records_loaded": len(transformed_data),
        "total_scope_1": total_scope1,
        "total_scope_2": total_scope2,
        "total_scope_3": total_scope3,
        "total_emissions": total_scope1 + total_scope2 + total_scope3
    }

    logger.info(f"Loaded {summary['records_loaded']} records")

    return summary

def _transform_row(self, row: dict) -> dict:
    """Transform individual row."""
    return {
        "company_name": row["Company"].strip(),
        "year": int(row["Year"]),
        "scope_1": float(row["Scope 1"]),
        "scope_2": float(row["Scope 2"]),
        "scope_3": float(row["Scope 3"]),
        "unit": "tons_co2",
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Step 3: Test the Agent

```python
# test_csrd_intake_agent.py
from csrd_intake_agent import CSRDIntakeAgent
import csv

# Create test data
test_data = [
    {"Company": "Tesla", "Year": "2023", "Scope 1": "100", "Scope 2": "200", "Scope 3": "300"},
    {"Company": "Apple", "Year": "2023", "Scope 1": "150", "Scope 2": "250", "Scope 3": "350"},
    {"Company": "Microsoft", "Year": "2023", "Scope 1": "120", "Scope 2": "220", "Scope 3": "320"}
]

# Write test CSV
with open('test_emissions.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=test_data[0].keys())
    writer.writeheader()
    writer.writerows(test_data)

# Run agent
agent = CSRDIntakeAgent()
agent.setup()

result = agent.execute_with_input({
    "source_path": "test_emissions.csv"
})

print("Results:")
print(f"Records loaded: {result['records_loaded']}")
print(f"Total Scope 1: {result['total_scope_1']:,.0f} tons")
print(f"Total Scope 2: {result['total_scope_2']:,.0f} tons")
print(f"Total Scope 3: {result['total_scope_3']:,.0f} tons")
print(f"Total Emissions: {result['total_emissions']:,.0f} tons")

agent.teardown()
```

---

## Part 7: Best Practices (20 minutes)

### 1. Single Responsibility

```python
# Bad: Agent does too much
class SuperAgent(Agent):
    def execute(self):
        # Fetches data
        # Processes data
        # Sends emails
        # Updates database
        # Generates reports
        pass

# Good: Focused agent
class DataProcessorAgent(Agent):
    def execute(self):
        # Only processes data
        pass
```

### 2. Dependency Injection

```python
# Good: Dependencies injected
class MyAgent(Agent):
    def __init__(self, db_manager, cache_manager):
        super().__init__()
        self.db = db_manager
        self.cache = cache_manager

# Easier to test with mocks
agent = MyAgent(
    db_manager=MockDatabase(),
    cache_manager=MockCache()
)
```

### 3. Idempotency

```python
# Good: Idempotent execution
class IdempotentAgent(Agent):
    def execute(self):
        """Can be run multiple times safely."""

        # Check if already processed
        if self._is_already_processed():
            return {"status": "already_processed"}

        # Process
        result = self._process()

        # Mark as processed
        self._mark_processed()

        return result
```

### 4. Logging

```python
# Good: Comprehensive logging
class WellLoggedAgent(Agent):
    def execute(self):
        logger.info("Starting execution", extra={
            "agent": self.name,
            "version": self.version,
            "input_size": len(self.input_data)
        })

        try:
            result = self._process()
            logger.info("Execution successful", extra=result)
            return result

        except Exception as e:
            logger.error("Execution failed", extra={
                "error": str(e),
                "input": self.input_data
            })
            raise
```

---

## Workshop Wrap-Up

### What You Learned

✓ Agent base class architecture
✓ Lifecycle management (setup, execute, teardown)
✓ Agent templates (Calculator, DataIntake, Reporting)
✓ Batch and parallel processing
✓ Error handling and retries
✓ Built a complete data intake agent

### Key Takeaways

1. **Always inherit from Agent** - Never create standalone classes
2. **Use templates** - Don't reinvent common patterns
3. **Follow lifecycle** - Setup, execute, teardown
4. **Handle errors gracefully** - Use retries and fallbacks
5. **Log everything** - Future you will thank you

### Homework

Build a complete agent system:
1. Data intake agent
2. Calculation agent
3. Reporting agent
4. Chain them together
5. Add error handling
6. Deploy to production

---

**Workshop Complete! Ready for Workshop 4: Data & Caching**
