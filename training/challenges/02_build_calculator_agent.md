# Challenge 2: Build a Calculator Agent

**Difficulty:** Intermediate
**Time:** 60 minutes
**Goal:** Build a production-ready calculator agent from scratch

---

## Requirements

Build an agent that:
1. Calculates carbon emissions for various activities
2. Uses CalculatorAgent template
3. Implements caching
4. Includes validation
5. Tracks metrics
6. Passes all tests

---

## Specification

### Inputs
```python
{
    "activity_type": "electricity" | "flight" | "car" | "custom",
    "amount": float,
    "unit": string
}
```

### Outputs
```python
{
    "activity_type": string,
    "amount": float,
    "unit": string,
    "co2_kg": float,
    "methodology": string,
    "cached": boolean,
    "calculation_time_ms": float
}
```

### Emission Factors
- Electricity: 0.5 kg CO2/kWh
- Flight: 0.115 kg CO2/km
- Car: 0.17 kg CO2/km
- Custom: Use LLM

---

## Starter Code

```python
# emission_calculator.py
from GL_COMMONS.infrastructure.agents.templates import CalculatorAgent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.validation import ValidationFramework

class EmissionCalculatorAgent(CalculatorAgent):
    """Calculate carbon emissions."""

    def setup(self):
        # TODO: Initialize infrastructure
        pass

    def calculate(self, input_data: dict) -> dict:
        # TODO: Implement calculation logic
        pass

    def teardown(self):
        # TODO: Cleanup
        pass
```

---

## Tests

```python
# test_emission_calculator.py
import pytest
from emission_calculator import EmissionCalculatorAgent

def test_electricity_calculation():
    agent = EmissionCalculatorAgent()
    agent.setup()

    result = agent.execute_with_input({
        "activity_type": "electricity",
        "amount": 100,
        "unit": "kWh"
    })

    assert result["co2_kg"] == 50.0  # 100 * 0.5
    assert result["methodology"] == "Emission Factor Database"

def test_caching():
    agent = EmissionCalculatorAgent()
    agent.setup()

    # First call
    result1 = agent.execute_with_input({
        "activity_type": "electricity",
        "amount": 100,
        "unit": "kWh"
    })
    assert result1["cached"] == False

    # Second call - should be cached
    result2 = agent.execute_with_input({
        "activity_type": "electricity",
        "amount": 100,
        "unit": "kWh"
    })
    assert result2["cached"] == True

def test_validation():
    agent = EmissionCalculatorAgent()
    agent.setup()

    # Invalid input
    with pytest.raises(ValidationError):
        agent.execute_with_input({
            "activity_type": "electricity",
            "amount": -100,  # Negative!
            "unit": "kWh"
        })

def test_llm_fallback():
    agent = EmissionCalculatorAgent()
    agent.setup()

    # Unknown activity - should use LLM
    result = agent.execute_with_input({
        "activity_type": "custom",
        "amount": 50,
        "unit": "widgets"
    })

    assert "co2_kg" in result
    assert result["methodology"] == "LLM Calculation"

# Run: pytest test_emission_calculator.py -v
```

---

## Grading Rubric

| Criteria | Points |
|----------|--------|
| Uses CalculatorAgent template | 20 |
| Implements caching correctly | 20 |
| Validates input | 15 |
| All tests pass | 25 |
| Tracks metrics | 10 |
| Proper error handling | 10 |
| **Total** | **100** |

---

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
from GL_COMMONS.infrastructure.agents.templates import CalculatorAgent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.validation import ValidationFramework
from GL_COMMONS.infrastructure.telemetry import TelemetryManager
import time

class EmissionCalculatorAgent(CalculatorAgent):
    def __init__(self):
        super().__init__(
            name="emission_calculator",
            version="1.0.0"
        )

    def setup(self):
        super().setup()

        self.llm = ChatSession(
            provider="openai",
            model="gpt-4",
            system_message="Calculate CO2 emissions. Return only the kg value."
        )

        self.cache = CacheManager()
        self.validator = ValidationFramework()
        self.telemetry = TelemetryManager(service_name="emission_calculator")

        self.emission_factors = {
            "electricity": 0.5,
            "flight": 0.115,
            "car": 0.17
        }

    def calculate(self, input_data: dict) -> dict:
        start_time = time.time()

        # Validate
        schema = {
            "activity_type": {"type": "string", "required": True},
            "amount": {"type": "number", "min": 0, "required": True},
            "unit": {"type": "string", "required": True}
        }
        self.validator.validate(input_data, schema)

        # Check cache
        cache_key = f"emission:{input_data['activity_type']}:{input_data['amount']}"
        cached = self.cache.get(cache_key)

        if cached:
            self.telemetry.increment("cache_hits")
            return {**cached, "cached": True}

        # Calculate
        activity_type = input_data["activity_type"]
        amount = input_data["amount"]

        if activity_type in self.emission_factors:
            co2_kg = amount * self.emission_factors[activity_type]
            methodology = "Emission Factor Database"
        else:
            # Use LLM for custom
            response = self.llm.send_message(
                f"CO2 emissions for {amount} {input_data['unit']} of {activity_type}?"
            )
            co2_kg = float(response)
            methodology = "LLM Calculation"

        result = {
            "activity_type": activity_type,
            "amount": amount,
            "unit": input_data["unit"],
            "co2_kg": co2_kg,
            "methodology": methodology,
            "cached": False,
            "calculation_time_ms": (time.time() - start_time) * 1000
        }

        # Cache
        self.cache.set(cache_key, result, ttl=3600)

        # Metrics
        self.telemetry.increment("calculations_total")
        self.telemetry.histogram("emission_value", co2_kg)

        return result
```

</details>

---

## Bonus Challenges

1. **Batch Processing** (+10 points)
   - Process multiple activities at once
   - Use batch optimization

2. **Cost Optimization** (+10 points)
   - Use cheaper model for simple calculations
   - Track and report LLM costs

3. **Advanced Caching** (+10 points)
   - Implement semantic caching
   - Pre-warm cache with common queries
