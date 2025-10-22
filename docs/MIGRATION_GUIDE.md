# GreenLang Migration Guide

**Transform Your Legacy Climate Code into Production-Ready Framework Applications**

This guide will help you migrate from custom-built carbon calculations and climate analytics to the GreenLang framework, reducing complexity, improving maintainability, and unlocking enterprise-grade features.

---

## Table of Contents

1. [Why Migrate to GreenLang?](#why-migrate-to-greenlang)
2. [Migration Checklist](#migration-checklist)
3. [Before and After Comparisons](#before-and-after-comparisons)
4. [Step-by-Step Migration Process](#step-by-step-migration-process)
5. [Common Migration Patterns](#common-migration-patterns)
6. [Testing Migrated Code](#testing-migrated-code)
7. [Rollback Strategy](#rollback-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Pitfalls to Avoid](#pitfalls-to-avoid)

---

## Why Migrate to GreenLang?

### Benefits of Framework Adoption

**Code Reduction: 60-80% Less Code**
- Replace 500+ lines of custom calculation logic with 50 lines using GreenLang agents
- Eliminate boilerplate for validation, error handling, and provenance tracking
- Built-in best practices reduce technical debt

**Enterprise Features Out-of-the-Box**
- Automatic provenance tracking for regulatory compliance
- Deterministic calculations with reproducible seeds
- Built-in caching for performance optimization
- Comprehensive error handling and logging
- Industry-standard emission factors (IPCC, EPA, DEFRA)

**Maintainability and Future-Proofing**
- Framework updates automatically improve your code
- Community-driven emission factor updates
- No need to track regulatory changes manually
- Clear separation of concerns with agent-based architecture

**Scalability**
- Batch processing for large datasets
- Pipeline orchestration for complex workflows
- Cloud-native deployment support
- Multi-tenant architecture support

### When to Migrate

✅ **Good Candidates for Migration:**
- Custom carbon calculators with hardcoded emission factors
- Spreadsheet-based calculations moving to code
- Legacy Python scripts with minimal testing
- Monolithic applications needing modularization
- Systems requiring audit trails and provenance

❌ **Consider Gradual Adoption:**
- Highly specialized industry calculations not yet supported
- Real-time systems with <10ms latency requirements
- Applications with complex legacy integrations
- Teams with limited Python expertise

---

## Migration Checklist

### Pre-Migration Assessment

- [ ] **Inventory existing calculations**: Document all emission calculation logic
- [ ] **Identify data sources**: List all input data formats and locations
- [ ] **Map emission factors**: Compare your factors with GreenLang's standards
- [ ] **Review dependencies**: Identify external libraries and APIs
- [ ] **Assess test coverage**: Document existing tests for validation
- [ ] **Plan deployment**: Determine migration strategy (big bang vs incremental)

### Infrastructure Setup

- [ ] **Install GreenLang**: `pip install greenlang-cli[full]==0.3.0`
- [ ] **Setup development environment**: Virtual environment with all dependencies
- [ ] **Configure version control**: Create migration branch
- [ ] **Setup CI/CD**: Automated testing for migrated code
- [ ] **Provision staging environment**: Test migrated code safely

### Migration Execution

- [ ] **Create agent implementations**: Port logic to GreenLang agents
- [ ] **Migrate data models**: Convert to Pydantic models or use framework types
- [ ] **Implement validation**: Leverage framework validation utilities
- [ ] **Add provenance tracking**: Decorate agents with @traced
- [ ] **Migrate tests**: Port existing tests to pytest framework
- [ ] **Setup integration tests**: Verify end-to-end functionality

### Post-Migration Validation

- [ ] **Run parallel testing**: Compare outputs with legacy system
- [ ] **Performance benchmarking**: Measure improvement/regression
- [ ] **Security audit**: Review for hardcoded credentials
- [ ] **Documentation update**: Update team documentation
- [ ] **Training completion**: Team trained on framework
- [ ] **Rollback plan prepared**: Documented rollback procedure

---

## Before and After Comparisons

### Example 1: Basic Carbon Calculation

#### Before: Custom Implementation (250+ lines)

```python
# legacy_calculator.py - Simplified excerpt
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class CarbonCalculator:
    """Legacy carbon calculator with hardcoded factors"""

    def __init__(self):
        # Hardcoded emission factors (needs manual updates!)
        self.emission_factors = {
            'electricity_us': 0.417,  # kg CO2e/kWh
            'natural_gas': 0.184,     # kg CO2e/kWh
            'diesel': 2.68,           # kg CO2e/liter
            'coal': 2.86,             # kg CO2e/kg
        }
        self.logger = logging.getLogger(__name__)

    def validate_input(self, data):
        """Manual validation logic"""
        if not isinstance(data, dict):
            raise ValueError("Input must be dictionary")
        if 'fuel_type' not in data:
            raise ValueError("Missing fuel_type")
        if 'consumption' not in data:
            raise ValueError("Missing consumption")
        if data['consumption'] < 0:
            raise ValueError("Consumption cannot be negative")
        return True

    def get_emission_factor(self, fuel_type, location='US'):
        """Retrieve emission factor with location logic"""
        key = f"{fuel_type}_{location.lower()}"
        if key not in self.emission_factors:
            key = fuel_type  # Fallback to generic
        if key not in self.emission_factors:
            raise ValueError(f"Unknown fuel type: {fuel_type}")
        return self.emission_factors[key]

    def calculate_emissions(self, fuel_type, consumption, unit='kWh', location='US'):
        """Calculate emissions with manual error handling"""
        try:
            # Validate inputs
            data = {
                'fuel_type': fuel_type,
                'consumption': consumption,
                'unit': unit
            }
            self.validate_input(data)

            # Unit conversion (manual logic)
            if unit == 'MWh':
                consumption = consumption * 1000
            elif unit == 'therms':
                consumption = consumption * 29.3  # Convert to kWh

            # Get emission factor
            factor = self.get_emission_factor(fuel_type, location)

            # Calculate
            emissions_kg = consumption * factor
            emissions_tons = emissions_kg / 1000

            # Log for audit (manual logging)
            self.logger.info(f"Calculated {emissions_tons:.2f} tCO2e for {consumption} {unit} of {fuel_type}")

            # Return results
            return {
                'success': True,
                'emissions_kg': emissions_kg,
                'emissions_tons': emissions_tons,
                'fuel_type': fuel_type,
                'consumption': consumption,
                'unit': unit,
                'emission_factor': factor,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Calculation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def calculate_building_emissions(self, building_data):
        """Calculate total building emissions"""
        try:
            total_emissions = 0
            results = []

            # Process each fuel type
            for fuel_data in building_data.get('fuels', []):
                result = self.calculate_emissions(
                    fuel_type=fuel_data['type'],
                    consumption=fuel_data['consumption'],
                    unit=fuel_data.get('unit', 'kWh'),
                    location=building_data.get('location', 'US')
                )

                if result['success']:
                    total_emissions += result['emissions_tons']
                    results.append(result)
                else:
                    # Error handling
                    return {
                        'success': False,
                        'error': f"Failed to calculate {fuel_data['type']}: {result['error']}"
                    }

            # Calculate intensity
            building_area = building_data.get('area_sqft', 1)
            intensity = (total_emissions * 1000) / building_area  # kgCO2e/sqft

            return {
                'success': True,
                'total_emissions_tons': total_emissions,
                'intensity_per_sqft': intensity,
                'breakdown': results,
                'building_name': building_data.get('name', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Building calculation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Usage (lots of manual setup)
calculator = CarbonCalculator()
result = calculator.calculate_building_emissions({
    'name': 'Office Building',
    'area_sqft': 10000,
    'location': 'US',
    'fuels': [
        {'type': 'electricity', 'consumption': 50000, 'unit': 'kWh'},
        {'type': 'natural_gas', 'consumption': 1000, 'unit': 'therms'}
    ]
})

if result['success']:
    print(f"Total emissions: {result['total_emissions_tons']:.2f} tCO2e")
else:
    print(f"Error: {result['error']}")
```

#### After: GreenLang Framework (50 lines)

```python
# greenlang_calculator.py
from greenlang.sdk import Agent, Result
from greenlang.provenance.decorators import traced
from pydantic import BaseModel, Field
from typing import List

class FuelInput(BaseModel):
    """Validated fuel input model"""
    fuel_type: str = Field(..., description="Type of fuel")
    consumption: float = Field(..., gt=0, description="Consumption amount")
    unit: str = Field(default="kWh", description="Unit of measurement")

class BuildingInput(BaseModel):
    """Validated building input model"""
    name: str
    area_sqft: float = Field(..., gt=0)
    location: str = "US"
    fuels: List[FuelInput]

class BuildingEmissionsAgent(Agent[BuildingInput, dict]):
    """GreenLang agent with automatic validation and provenance"""

    def __init__(self):
        super().__init__(
            metadata={
                "id": "building_emissions",
                "name": "Building Emissions Calculator",
                "version": "1.0.0"
            }
        )

    def validate(self, input_data: BuildingInput) -> bool:
        """Pydantic handles validation automatically"""
        return True  # Already validated by Pydantic

    @traced(save_path="provenance/building_calc.json", track_inputs=True, track_outputs=True)
    def process(self, input_data: BuildingInput) -> dict:
        """Calculate building emissions - framework handles everything else"""
        from greenlang.emissions import EmissionFactorService

        # Service provides up-to-date emission factors automatically
        ef_service = EmissionFactorService(region=input_data.location)

        total_emissions = 0
        breakdown = []

        for fuel in input_data.fuels:
            # Automatic unit conversion and emission calculation
            emissions = ef_service.calculate_emissions(
                fuel_type=fuel.fuel_type,
                consumption=fuel.consumption,
                unit=fuel.unit
            )

            total_emissions += emissions.tons
            breakdown.append(emissions.to_dict())

        # Calculate intensity
        intensity = (total_emissions * 1000) / input_data.area_sqft

        return {
            "total_emissions_tons": total_emissions,
            "intensity_per_sqft": intensity,
            "breakdown": breakdown,
            "building_name": input_data.name
        }

# Usage (simple and clean)
agent = BuildingEmissionsAgent()

input_data = BuildingInput(
    name="Office Building",
    area_sqft=10000,
    location="US",
    fuels=[
        FuelInput(fuel_type="electricity", consumption=50000, unit="kWh"),
        FuelInput(fuel_type="natural_gas", consumption=1000, unit="therms")
    ]
)

result = agent.run(input_data)

if result.success:
    print(f"Total emissions: {result.data['total_emissions_tons']:.2f} tCO2e")
    print(f"Provenance automatically saved to: provenance/building_calc.json")
else:
    print(f"Error: {result.error}")
```

**Comparison:**
- **Lines of code**: 250+ → 50 (80% reduction)
- **Validation**: Manual → Automatic (Pydantic)
- **Emission factors**: Hardcoded → Auto-updated
- **Error handling**: Manual → Framework-managed
- **Provenance**: None → Automatic with @traced
- **Testing**: Custom → Framework test utilities
- **Maintenance**: High → Low (framework updates)

---

### Example 2: Data Processing Pipeline

#### Before: Custom Pipeline (400+ lines)

```python
# legacy_pipeline.py
import csv
import json
from datetime import datetime
import logging

class DataPipeline:
    """Legacy pipeline with manual orchestration"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.errors = []

    def load_csv(self, filepath):
        """Manual CSV loading"""
        try:
            data = []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            self.logger.info(f"Loaded {len(data)} rows from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            raise

    def validate_row(self, row):
        """Manual validation"""
        required_fields = ['building_id', 'electricity_kwh', 'gas_therms']
        for field in required_fields:
            if field not in row or not row[field]:
                return False, f"Missing {field}"

        try:
            float(row['electricity_kwh'])
            float(row['gas_therms'])
        except ValueError:
            return False, "Invalid numeric value"

        return True, None

    def process_row(self, row):
        """Process single row"""
        try:
            # Validate
            valid, error = self.validate_row(row)
            if not valid:
                self.errors.append({'row': row, 'error': error})
                return None

            # Calculate emissions
            electricity_kwh = float(row['electricity_kwh'])
            gas_therms = float(row['gas_therms'])

            # Hardcoded factors (needs updates!)
            electricity_factor = 0.417  # kgCO2e/kWh
            gas_factor = 5.3  # kgCO2e/therm

            electricity_emissions = electricity_kwh * electricity_factor
            gas_emissions = gas_therms * gas_factor
            total_emissions = (electricity_emissions + gas_emissions) / 1000  # tons

            result = {
                'building_id': row['building_id'],
                'electricity_emissions_kg': electricity_emissions,
                'gas_emissions_kg': gas_emissions,
                'total_emissions_tons': total_emissions,
                'processed_at': datetime.now().isoformat()
            }

            self.results.append(result)
            return result

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            self.errors.append({'row': row, 'error': str(e)})
            return None

    def process_batch(self, data):
        """Process all rows"""
        self.logger.info(f"Processing {len(data)} rows...")

        for idx, row in enumerate(data):
            if idx % 100 == 0:
                self.logger.info(f"Processed {idx}/{len(data)} rows")

            self.process_row(row)

        self.logger.info(f"Completed. Success: {len(self.results)}, Errors: {len(self.errors)}")

    def save_results(self, output_path):
        """Save results to JSON"""
        with open(output_path, 'w') as f:
            json.dump({
                'results': self.results,
                'errors': self.errors,
                'summary': {
                    'total_processed': len(self.results) + len(self.errors),
                    'successful': len(self.results),
                    'failed': len(self.errors)
                }
            }, f, indent=2)
        self.logger.info(f"Results saved to {output_path}")

# Usage (manual orchestration)
pipeline = DataPipeline()
data = pipeline.load_csv('buildings.csv')
pipeline.process_batch(data)
pipeline.save_results('results.json')
```

#### After: GreenLang Framework (80 lines)

```python
# greenlang_pipeline.py
from greenlang.sdk import Agent, Pipeline, Context
from greenlang.provenance.decorators import traced
from pydantic import BaseModel, Field, validator
from typing import List
from pathlib import Path

class BuildingRow(BaseModel):
    """Validated data model"""
    building_id: str
    electricity_kwh: float = Field(..., gt=0)
    gas_therms: float = Field(..., gt=0)

    @validator('electricity_kwh', 'gas_therms')
    def check_realistic_values(cls, v):
        if v > 1000000:  # Sanity check
            raise ValueError("Unrealistic consumption value")
        return v

class BuildingDataProcessor(Agent[BuildingRow, dict]):
    """Framework agent with automatic validation"""

    def validate(self, input_data: BuildingRow) -> bool:
        return True  # Pydantic handles it

    @traced(track_inputs=True, track_outputs=True)
    def process(self, input_data: BuildingRow) -> dict:
        """Process with automatic provenance"""
        from greenlang.emissions import EmissionFactorService

        ef_service = EmissionFactorService()

        # Framework handles emission factor lookups
        elec_emissions = ef_service.calculate_emissions(
            fuel_type="electricity",
            consumption=input_data.electricity_kwh,
            unit="kWh"
        )

        gas_emissions = ef_service.calculate_emissions(
            fuel_type="natural_gas",
            consumption=input_data.gas_therms,
            unit="therms"
        )

        return {
            'building_id': input_data.building_id,
            'electricity_emissions_kg': elec_emissions.kg,
            'gas_emissions_kg': gas_emissions.kg,
            'total_emissions_tons': elec_emissions.tons + gas_emissions.tons
        }

class BuildingPipeline(Pipeline):
    """Framework pipeline with automatic orchestration"""

    def __init__(self):
        super().__init__()
        self.processor = BuildingDataProcessor()

    def execute(self, input_data: dict) -> dict:
        """Execute pipeline with framework utilities"""
        from greenlang.utils import CSVLoader, BatchProcessor, ResultWriter

        ctx = Context(inputs=input_data, artifacts_dir=Path("output"))

        # Load data (framework handles errors)
        loader = CSVLoader()
        buildings_data = loader.load(input_data['csv_path'], model=BuildingRow)

        # Process batch (framework handles parallelization)
        batch_processor = BatchProcessor(self.processor, batch_size=100)
        results = batch_processor.process_batch(buildings_data)

        # Save results (framework handles formatting)
        writer = ResultWriter(ctx)
        writer.save_results(results, format=['json', 'csv'])

        # Return summary with automatic provenance
        return ctx.to_result()

# Usage (framework handles everything)
pipeline = BuildingPipeline()
result = pipeline.execute({'csv_path': 'buildings.csv'})

if result.success:
    print(f"Processed {result.metadata['total']} buildings")
    print(f"Results saved to: output/")
    print(f"Provenance tracked automatically")
```

**Comparison:**
- **Lines of code**: 400+ → 80 (80% reduction)
- **CSV loading**: Manual → Framework utility
- **Validation**: Manual → Pydantic models
- **Error handling**: Manual tracking → Automatic
- **Batch processing**: Manual loop → BatchProcessor
- **Provenance**: None → Automatic with @traced
- **Parallelization**: None → Framework-managed
- **Output formats**: Manual JSON → Multiple formats (JSON, CSV, Excel)

---

## Step-by-Step Migration Process

### Step 1: Analyze Existing Code

**Inventory Your Calculations**
```bash
# Find all calculation logic
grep -r "emission.*factor" your_project/
grep -r "calculate.*carbon" your_project/
grep -r "CO2.*conversion" your_project/

# Identify hardcoded values
grep -r "= 0\.[0-9]" your_project/*.py | grep -i "factor\|emission\|carbon"
```

**Document Data Flow**
```
Input Data → Validation → Calculation → Results → Output
     ↓            ↓            ↓           ↓        ↓
  Files?    Checks?    Factors?    Format?  Storage?
```

### Step 2: Setup GreenLang Development Environment

```bash
# Create migration environment
python -m venv greenlang-migration
source greenlang-migration/bin/activate

# Install GreenLang with all extras
pip install greenlang-cli[full]==0.3.0

# Verify installation
gl version
gl doctor

# Create project structure
mkdir -p src/agents src/models src/pipelines tests/
```

### Step 3: Create Data Models (Pydantic)

**Convert dictionaries to Pydantic models:**

```python
# OLD: Dictionary with manual validation
def validate_building(data):
    assert 'name' in data
    assert 'area' in data and data['area'] > 0
    return True

# NEW: Pydantic model with automatic validation
from pydantic import BaseModel, Field

class Building(BaseModel):
    name: str
    area_sqft: float = Field(..., gt=0, description="Building area in square feet")
    year_built: int = Field(default=2000, ge=1800, le=2030)
    location: str = "US"
```

### Step 4: Migrate Calculation Logic to Agents

**Pattern: Extract calculation logic into Agent.process()**

```python
# OLD: Function-based calculation
def calculate_emissions(fuel_type, consumption):
    factor = get_factor(fuel_type)
    return consumption * factor

# NEW: Agent-based calculation
from greenlang.sdk import Agent, Result
from greenlang.provenance.decorators import traced

class EmissionsAgent(Agent[FuelInput, dict]):
    def validate(self, input_data: FuelInput) -> bool:
        return True  # Pydantic handles it

    @traced(save_path="provenance/emissions.json")
    def process(self, input_data: FuelInput) -> dict:
        from greenlang.emissions import EmissionFactorService

        ef_service = EmissionFactorService()
        result = ef_service.calculate_emissions(
            fuel_type=input_data.fuel_type,
            consumption=input_data.consumption,
            unit=input_data.unit
        )

        return result.to_dict()
```

### Step 5: Add Provenance Tracking

**Add @traced decorator to all agents:**

```python
from greenlang.provenance.decorators import traced

@traced(
    record_id="building_emissions_001",
    save_path="provenance/emissions.json",
    track_inputs=True,
    track_outputs=True
)
def process(self, input_data: BuildingInput) -> dict:
    # Your processing logic
    result = calculate_something(input_data)
    return result
```

### Step 6: Migrate Tests

**Convert to pytest with framework utilities:**

```python
# OLD: unittest-based tests
import unittest

class TestCalculator(unittest.TestCase):
    def test_emission_calculation(self):
        calc = Calculator()
        result = calc.calculate(fuel='electricity', consumption=100)
        self.assertAlmostEqual(result, 41.7, places=1)

# NEW: pytest with framework fixtures
import pytest
from greenlang.testing import AgentTestCase

class TestEmissionsAgent(AgentTestCase):
    @pytest.fixture
    def agent(self):
        return EmissionsAgent()

    def test_emission_calculation(self, agent):
        """Test with automatic validation and mocking"""
        input_data = FuelInput(fuel_type="electricity", consumption=100, unit="kWh")
        result = agent.run(input_data)

        assert result.success
        assert result.data['emissions_kg'] == pytest.approx(41.7, rel=0.01)
        assert result.metadata['agent'] == 'emissionsagent'
```

### Step 7: Run Parallel Testing

**Compare outputs between legacy and new systems:**

```python
# parallel_test.py
from legacy_calculator import LegacyCalculator
from greenlang_calculator import EmissionsAgent, FuelInput

legacy = LegacyCalculator()
agent = EmissionsAgent()

test_cases = [
    {'fuel_type': 'electricity', 'consumption': 100, 'unit': 'kWh'},
    {'fuel_type': 'natural_gas', 'consumption': 50, 'unit': 'therms'},
    # ... more test cases
]

for test_case in test_cases:
    # Legacy result
    legacy_result = legacy.calculate_emissions(**test_case)

    # GreenLang result
    input_data = FuelInput(**test_case)
    gl_result = agent.run(input_data)

    # Compare
    if legacy_result['emissions_tons'] != gl_result.data['emissions_tons']:
        print(f"MISMATCH: {test_case}")
        print(f"  Legacy: {legacy_result['emissions_tons']}")
        print(f"  GreenLang: {gl_result.data['emissions_tons']}")
```

### Step 8: Deploy Incrementally

**Use feature flags for gradual rollout:**

```python
# Feature flag approach
USE_GREENLANG = os.getenv('USE_GREENLANG', 'false').lower() == 'true'

if USE_GREENLANG:
    from greenlang_calculator import EmissionsAgent
    calculator = EmissionsAgent()
else:
    from legacy_calculator import LegacyCalculator
    calculator = LegacyCalculator()

# Use calculator regardless of implementation
result = calculator.run(input_data)
```

---

## Common Migration Patterns

### Pattern 1: Hardcoded Factors → EmissionFactorService

```python
# BEFORE
FACTORS = {
    'electricity_us': 0.417,
    'gas_us': 0.184
}
emissions = consumption * FACTORS[fuel_type]

# AFTER
from greenlang.emissions import EmissionFactorService
ef_service = EmissionFactorService(region="US")
emissions = ef_service.calculate_emissions(fuel_type, consumption, unit)
```

### Pattern 2: Manual Validation → Pydantic Models

```python
# BEFORE
if not data.get('name'):
    raise ValueError("Name required")
if data.get('area', 0) <= 0:
    raise ValueError("Area must be positive")

# AFTER
from pydantic import BaseModel, Field

class Building(BaseModel):
    name: str = Field(..., min_length=1)
    area_sqft: float = Field(..., gt=0)
```

### Pattern 3: Manual Logging → Provenance Tracking

```python
# BEFORE
logger.info(f"Processing {building_id}")
logger.info(f"Input: {input_data}")
result = process(input_data)
logger.info(f"Output: {result}")

# AFTER
from greenlang.provenance.decorators import traced

@traced(save_path="provenance/process.json", track_inputs=True, track_outputs=True)
def process(input_data):
    result = do_work(input_data)
    return result  # Automatically logged with provenance
```

### Pattern 4: CSV Loops → BatchProcessor

```python
# BEFORE
results = []
with open('data.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        result = process_row(row)
        results.append(result)

# AFTER
from greenlang.utils import CSVLoader, BatchProcessor

loader = CSVLoader()
data = loader.load('data.csv', model=InputModel)

processor = BatchProcessor(agent, batch_size=100, parallel=True)
results = processor.process_batch(data)
```

### Pattern 5: Custom Pipelines → GreenLang Pipelines

```python
# BEFORE
data = load_data()
validated = validate_data(data)
calculated = calculate_emissions(validated)
report = generate_report(calculated)

# AFTER
from greenlang.sdk import Pipeline

class EmissionsPipeline(Pipeline):
    def execute(self, input_data):
        ctx = Context(inputs=input_data)

        # Framework handles orchestration
        self.add_agent(DataLoader())
        self.add_agent(DataValidator())
        self.add_agent(EmissionsCalculator())
        self.add_agent(ReportGenerator())

        return self.run(ctx)
```

---

## Testing Migrated Code

### Unit Testing with Framework Utilities

```python
# tests/test_emissions_agent.py
import pytest
from greenlang.testing import AgentTestCase, mock_emission_factors
from src.agents.emissions import EmissionsAgent
from src.models.inputs import FuelInput

class TestEmissionsAgent(AgentTestCase):
    """Test emissions agent with framework utilities"""

    @pytest.fixture
    def agent(self):
        return EmissionsAgent()

    def test_valid_input(self, agent):
        """Test with valid input"""
        input_data = FuelInput(
            fuel_type="electricity",
            consumption=100,
            unit="kWh"
        )

        result = agent.run(input_data)

        assert result.success
        assert result.data['emissions_kg'] > 0
        assert 'provenance' in result.metadata

    def test_invalid_input(self, agent):
        """Test validation catches errors"""
        with pytest.raises(ValidationError):
            FuelInput(
                fuel_type="electricity",
                consumption=-100,  # Invalid: negative
                unit="kWh"
            )

    @mock_emission_factors({'electricity': 0.5})
    def test_mocked_factors(self, agent):
        """Test with mocked emission factors"""
        input_data = FuelInput(fuel_type="electricity", consumption=100, unit="kWh")
        result = agent.run(input_data)

        assert result.data['emissions_kg'] == 50.0  # 100 * 0.5
```

### Integration Testing

```python
# tests/test_pipeline_integration.py
import pytest
from greenlang.testing import PipelineTestCase
from src.pipelines.emissions_pipeline import EmissionsPipeline

class TestEmissionsPipeline(PipelineTestCase):
    """Integration tests for complete pipeline"""

    @pytest.fixture
    def pipeline(self):
        return EmissionsPipeline()

    def test_end_to_end(self, pipeline, tmp_path):
        """Test complete pipeline execution"""
        # Create test input file
        test_data = tmp_path / "test_data.csv"
        test_data.write_text("building_id,electricity_kwh\n1,1000\n2,2000\n")

        # Execute pipeline
        result = pipeline.execute({'csv_path': str(test_data)})

        assert result.success
        assert result.data['total_buildings'] == 2
        assert result.data['total_emissions_tons'] > 0

        # Verify provenance was tracked
        assert (tmp_path / "provenance").exists()
```

### Comparison Testing (Legacy vs Framework)

```python
# tests/test_migration_parity.py
import pytest
from legacy.calculator import LegacyCalculator
from src.agents.emissions import EmissionsAgent
from src.models.inputs import FuelInput

@pytest.mark.parametrize("fuel_type,consumption,unit", [
    ("electricity", 1000, "kWh"),
    ("natural_gas", 100, "therms"),
    ("diesel", 500, "liters"),
])
def test_legacy_parity(fuel_type, consumption, unit):
    """Verify GreenLang matches legacy calculations"""

    # Legacy calculation
    legacy = LegacyCalculator()
    legacy_result = legacy.calculate_emissions(fuel_type, consumption, unit)

    # GreenLang calculation
    agent = EmissionsAgent()
    input_data = FuelInput(fuel_type=fuel_type, consumption=consumption, unit=unit)
    gl_result = agent.run(input_data)

    # Compare (allow 1% tolerance for rounding)
    assert gl_result.success
    assert gl_result.data['emissions_tons'] == pytest.approx(
        legacy_result['emissions_tons'],
        rel=0.01
    )
```

---

## Rollback Strategy

### Preparation

**1. Version Control**
```bash
# Create migration branch
git checkout -b migration-to-greenlang
git commit -am "Start GreenLang migration"

# Tag pre-migration state
git tag pre-greenlang-migration
```

**2. Feature Flags**
```python
# config.py
FEATURE_FLAGS = {
    'use_greenlang': os.getenv('USE_GREENLANG', 'false').lower() == 'true',
    'greenlang_percentage': int(os.getenv('GREENLANG_PERCENTAGE', '0'))
}

# Gradual rollout
import random

def should_use_greenlang():
    if not FEATURE_FLAGS['use_greenlang']:
        return False
    return random.randint(1, 100) <= FEATURE_FLAGS['greenlang_percentage']
```

### Rollback Execution

**Quick Rollback (Production Issue)**
```bash
# 1. Disable feature flag
export USE_GREENLANG=false
systemctl restart your-app

# 2. Revert deployment
kubectl rollout undo deployment/your-app

# 3. Restore code
git revert HEAD~1  # Or use specific commit
git push origin main
```

**Gradual Rollback (Quality Issues)**
```bash
# Reduce traffic gradually
export GREENLANG_PERCENTAGE=50  # 50% traffic
# Monitor for issues

export GREENLANG_PERCENTAGE=25  # 25% traffic
# Monitor for issues

export GREENLANG_PERCENTAGE=0   # Full rollback
```

### Post-Rollback Analysis

```python
# analysis/rollback_analysis.py
"""
Analyze why rollback was needed
"""
import json

def analyze_rollback():
    """Generate rollback analysis report"""

    # Compare error rates
    legacy_errors = count_errors(system='legacy', period='last_7_days')
    greenlang_errors = count_errors(system='greenlang', period='migration_period')

    # Performance comparison
    legacy_latency = measure_latency(system='legacy')
    greenlang_latency = measure_latency(system='greenlang')

    # Data quality issues
    parity_issues = compare_calculations(legacy_vs_greenlang)

    report = {
        'rollback_reason': 'TODO: Document',
        'error_comparison': {
            'legacy': legacy_errors,
            'greenlang': greenlang_errors
        },
        'performance': {
            'legacy_p95_ms': legacy_latency,
            'greenlang_p95_ms': greenlang_latency
        },
        'parity_issues': parity_issues,
        'recommendations': []
    }

    return report
```

---

## Performance Considerations

### Expected Performance Changes

**Improvements:**
- **Caching**: 30-50% faster for repeated calculations
- **Batch Processing**: 2-5x faster for large datasets (parallelization)
- **Database Queries**: Optimized emission factor lookups

**Potential Regressions:**
- **First Run**: Framework initialization overhead (one-time cost)
- **Pydantic Validation**: ~1-2ms per object (acceptable trade-off)
- **Provenance Tracking**: ~5-10ms per operation (can be disabled)

### Optimization Strategies

```python
# 1. Enable caching for production
from greenlang.sdk import GreenLangClient

client = GreenLangClient()
client.enable_cache(
    cache_duration=3600,  # 1 hour
    cache_backend='redis'  # or 'memory'
)

# 2. Batch processing for large datasets
from greenlang.utils import BatchProcessor

processor = BatchProcessor(
    agent=your_agent,
    batch_size=500,
    parallel=True,
    num_workers=8
)

# 3. Disable provenance in performance-critical paths
@traced(track_inputs=False, track_outputs=False)  # Minimal tracking
def high_frequency_calculation(data):
    return calculate(data)

# 4. Use async agents for I/O-bound operations
from greenlang.sdk import AsyncAgent

class AsyncEmissionsAgent(AsyncAgent):
    async def process(self, input_data):
        # Async processing
        return await async_calculation(input_data)
```

---

## Pitfalls to Avoid

### Common Mistakes

**❌ Mistake 1: Not validating migration with real data**
```python
# DON'T just test with synthetic data
test_data = {'fuel_type': 'electricity', 'consumption': 100}

# DO test with production data samples
production_sample = load_recent_production_data(days=7)
for data in production_sample:
    verify_migration_parity(data)
```

**❌ Mistake 2: Ignoring emission factor differences**
```python
# Legacy might use outdated factors
legacy_factor = 0.417  # From 2015

# GreenLang uses current factors
from greenlang.emissions import EmissionFactorService
ef_service = EmissionFactorService()
current_factor = ef_service.get_factor('electricity', 'US')  # Updated monthly

# SOLUTION: Document and communicate differences
if abs(legacy_factor - current_factor) > 0.05:
    log_warning(f"Factor changed: {legacy_factor} → {current_factor}")
```

**❌ Mistake 3: Not planning for gradual rollout**
```python
# DON'T flip the switch all at once
USE_GREENLANG = True  # All traffic immediately

# DO use gradual rollout with monitoring
GREENLANG_PERCENTAGE = 10  # Start with 10%
# Monitor for 48 hours, then increase to 25%, 50%, 100%
```

**❌ Mistake 4: Skipping provenance configuration**
```python
# DON'T leave provenance with default settings in production
@traced()  # Uses defaults (may not be production-ready)

# DO configure appropriately for your environment
@traced(
    save_path=f"provenance/{environment}/{date}/",
    track_inputs=should_track_inputs(environment),
    track_outputs=should_track_outputs(environment)
)
```

**❌ Mistake 5: Not preparing rollback plan**
```python
# DON'T migrate without a way back
git push origin main  # No rollback plan

# DO prepare rollback strategy BEFORE migration
# - Feature flags in place
# - Monitoring dashboards configured
# - Rollback procedure documented
# - Team trained on rollback process
```

---

## Success Metrics

### Track These KPIs

**Code Quality:**
- Lines of code reduced (target: 60-80%)
- Test coverage increased (target: >80%)
- Cyclomatic complexity reduced

**Performance:**
- API latency (target: <2x current)
- Throughput (target: 2-5x improvement for batch)
- Error rate (target: <current rate)

**Maintainability:**
- Time to add new fuel types (should decrease significantly)
- Time to update emission factors (should be automatic)
- Onboarding time for new developers (should decrease)

**Compliance:**
- Provenance tracking coverage (target: 100% of calculations)
- Audit trail completeness
- Regulatory compliance improvement

---

## Next Steps After Migration

1. **Optimize Performance**: Profile and optimize hot paths
2. **Enhance Testing**: Increase test coverage to >90%
3. **Add ML Features**: Leverage GreenLang's ML capabilities
4. **Integrate Advanced Features**: RAG, real-time monitoring, optimization agents
5. **Contribute Back**: Share improvements with community

---

## Getting Help

**Community Support:**
- Discord: [discord.gg/greenlang](https://discord.gg/greenlang)
- GitHub Discussions: [github.com/greenlang/greenlang/discussions](https://github.com/greenlang/greenlang/discussions)

**Professional Services:**
- Migration consulting: migrations@greenlang.io
- Training programs: training@greenlang.io
- Enterprise support: enterprise@greenlang.io

---

**Ready to migrate? Start with a small pilot project, validate results, then scale! 🚀**

*GreenLang v0.3.0 - The Climate Intelligence Platform*
