"""
GreenLang Framework - Calculator Template

Specialized template for deterministic calculation agents.
"""

from typing import Dict, List, Any
from .base_agent import BaseAgentTemplate, AgentConfig, AgentType


class CalculatorTemplate(BaseAgentTemplate):
    """
    Template for calculator-type agents.

    Extends base template with:
    - Specialized calculator modules
    - Mathematical formula documentation
    - Golden master testing patterns
    - Numerical stability checks
    """

    # Additional directories for calculators
    CALCULATOR_DIRS = {
        "calculators/core": "Core calculation algorithms",
        "calculators/thermal": "Thermal calculations",
        "calculators/energy": "Energy balance calculations",
        "tests/golden": "Golden master tests for reproducibility",
    }

    def __init__(self, config: AgentConfig):
        """Initialize calculator template."""
        if config.agent_type != AgentType.CALCULATOR:
            config.agent_type = AgentType.CALCULATOR
        super().__init__(config)

    def get_directory_structure(self) -> Dict[str, str]:
        """Get calculator-specific directory structure."""
        structure = super().get_directory_structure()
        structure.update(self.CALCULATOR_DIRS)
        return structure

    def generate_calculator_base(self) -> str:
        """Generate base calculator class."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Base Calculator

Deterministic calculation base class with provenance tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, TypeVar
import hashlib
import json

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


@dataclass
class CalculationResult(Generic[OutputT]):
    """Standard result wrapper with provenance."""
    result: OutputT
    computation_hash: str
    inputs_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0
    calculator_name: str = ""
    calculator_version: str = ""
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)


class BaseCalculator(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for deterministic calculators.

    Requirements:
    1. Deterministic: same inputs always produce same outputs
    2. Traceable: SHA-256 hashes for all computations
    3. Validated: inputs validated before calculation
    4. Documented: all formulas with references
    """

    NAME: str = "BaseCalculator"
    VERSION: str = "1.0.0"

    def __init__(self, agent_id: str = "{self.config.agent_id}"):
        """Initialize calculator."""
        self.agent_id = agent_id
        self._history: List[Dict[str, Any]] = []

    def calculate(self, inputs: InputT) -> CalculationResult[OutputT]:
        """
        Perform calculation with provenance tracking.

        Args:
            inputs: Calculation inputs

        Returns:
            CalculationResult with result and provenance
        """
        start_time = datetime.now(timezone.utc)

        # Validate inputs
        errors = self._validate_inputs(inputs)
        if errors:
            return CalculationResult(
                result=None,
                computation_hash="",
                inputs_hash=self._compute_hash(inputs),
                timestamp=start_time,
                calculator_name=self.NAME,
                calculator_version=self.VERSION,
                is_valid=False,
                warnings=errors,
            )

        # Perform calculation
        result = self._calculate(inputs)

        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Compute hashes
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(result)
        computation_hash = self._compute_hash({{
            "inputs": inputs_hash,
            "outputs": outputs_hash,
            "calculator": self.NAME,
            "version": self.VERSION,
        }})

        return CalculationResult(
            result=result,
            computation_hash=computation_hash,
            inputs_hash=inputs_hash,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            is_valid=True,
        )

    @abstractmethod
    def _validate_inputs(self, inputs: InputT) -> List[str]:
        """Validate inputs, return list of errors."""
        pass

    @abstractmethod
    def _calculate(self, inputs: InputT) -> OutputT:
        """Perform the deterministic calculation."""
        pass

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash."""
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        elif hasattr(data, '__dict__'):
            data = data.__dict__
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_formula_documentation(self) -> Dict[str, Any]:
        """
        Get documentation of formulas used.

        Override in subclasses to document specific formulas.
        """
        return {{
            "calculator": self.NAME,
            "version": self.VERSION,
            "formulas": [],
            "references": [],
            "assumptions": [],
        }}
'''

    def generate_golden_test_template(self) -> str:
        """Generate golden master test template."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Golden Master Tests

Golden master tests ensure deterministic reproducibility.
Each test case has a known input-output pair that must remain stable.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

# Path to golden test data
GOLDEN_DATA_DIR = Path(__file__).parent / "golden_data"


def load_golden_case(name: str) -> Dict[str, Any]:
    """Load a golden test case."""
    path = GOLDEN_DATA_DIR / f"{{name}}.json"
    with open(path) as f:
        return json.load(f)


def save_golden_case(name: str, data: Dict[str, Any]) -> None:
    """Save a golden test case (for initial creation)."""
    GOLDEN_DATA_DIR.mkdir(exist_ok=True)
    path = GOLDEN_DATA_DIR / f"{{name}}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


class TestGoldenMaster:
    """
    Golden master tests for deterministic verification.

    These tests verify that calculations produce identical outputs
    for known inputs. Any change in output indicates a potential
    regression or intentional formula change.
    """

    # Example:
    # def test_basic_calculation_golden(self):
    #     """Golden test for basic calculation."""
    #     golden = load_golden_case("basic_calculation")
    #
    #     calculator = MyCalculator()
    #     result = calculator.calculate(golden["inputs"])
    #
    #     assert result.computation_hash == golden["expected_hash"]
    #     assert result.result == golden["expected_output"]

    def test_golden_data_exists(self):
        """Verify golden data directory exists."""
        # This is a placeholder - replace with actual tests
        assert True, "Add golden master tests for this agent"


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_inputs_same_hash(self):
        """Same inputs must produce same hash."""
        # calculator = MyCalculator()
        #
        # inputs = MyInputs(...)
        # result1 = calculator.calculate(inputs)
        # result2 = calculator.calculate(inputs)
        #
        # assert result1.computation_hash == result2.computation_hash
        # assert result1.inputs_hash == result2.inputs_hash
        pass

    def test_hash_changes_with_input(self):
        """Different inputs must produce different hashes."""
        pass
'''

    def generate_formula_docs_template(self) -> str:
        """Generate formula documentation template."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Formula Documentation

Mathematical formulas with references and validation ranges.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class FormulaCategory(Enum):
    """Categories of formulas."""
    THERMODYNAMIC = "thermodynamic"
    ENERGY_BALANCE = "energy_balance"
    HEAT_TRANSFER = "heat_transfer"
    FLUID_FLOW = "fluid_flow"
    COMBUSTION = "combustion"
    EMISSIONS = "emissions"


@dataclass
class FormulaReference:
    """Reference for a formula."""
    source: str
    section: Optional[str] = None
    equation_number: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None


@dataclass
class FormulaDefinition:
    """Definition of a mathematical formula."""
    name: str
    category: FormulaCategory
    equation_latex: str
    description: str
    variables: Dict[str, str]  # variable_name: description with units
    references: List[FormulaReference]
    valid_range: Optional[Dict[str, tuple]] = None  # variable: (min, max)
    assumptions: List[str] = None
    implementation_notes: Optional[str] = None


# Example:
#
# HEAT_TRANSFER_FORMULA = FormulaDefinition(
#     name="Heat Transfer Rate",
#     category=FormulaCategory.HEAT_TRANSFER,
#     equation_latex=r"Q = U \\cdot A \\cdot \\Delta T_{lm}",
#     description="Heat transfer rate through a heat exchanger",
#     variables={{
#         "Q": "Heat transfer rate [W]",
#         "U": "Overall heat transfer coefficient [W/(m²·K)]",
#         "A": "Heat transfer area [m²]",
#         "ΔT_lm": "Log mean temperature difference [K]",
#     }},
#     references=[
#         FormulaReference(
#             source="Incropera & DeWitt",
#             section="Chapter 11",
#             year=2007,
#         ),
#     ],
#     valid_range={{
#         "Q": (0, 1e9),
#         "U": (10, 10000),
#         "A": (0.01, 10000),
#     }},
#     assumptions=[
#         "Steady-state conditions",
#         "Constant fluid properties",
#         "No phase change",
#     ],
# )


class FormulaRegistry:
    """Registry of all formulas used in this agent."""

    def __init__(self):
        self._formulas: Dict[str, FormulaDefinition] = {{}}

    def register(self, formula: FormulaDefinition) -> None:
        """Register a formula."""
        self._formulas[formula.name] = formula

    def get(self, name: str) -> Optional[FormulaDefinition]:
        """Get a formula by name."""
        return self._formulas.get(name)

    def get_by_category(self, category: FormulaCategory) -> List[FormulaDefinition]:
        """Get all formulas in a category."""
        return [f for f in self._formulas.values() if f.category == category]

    def export_documentation(self) -> Dict[str, Any]:
        """Export all formulas as documentation."""
        return {{
            "agent_id": "{self.config.agent_id}",
            "agent_name": "{self.config.name}",
            "formula_count": len(self._formulas),
            "formulas": [
                {{
                    "name": f.name,
                    "category": f.category.value,
                    "equation": f.equation_latex,
                    "description": f.description,
                    "variables": f.variables,
                    "references": [
                        {{
                            "source": r.source,
                            "section": r.section,
                            "year": r.year,
                        }}
                        for r in f.references
                    ],
                }}
                for f in self._formulas.values()
            ],
        }}


# Global formula registry
FORMULA_REGISTRY = FormulaRegistry()
'''

    def get_all_templates(self) -> Dict[str, str]:
        """Get all calculator template contents."""
        templates = super().get_all_templates()
        templates.update({
            "calculators/base.py": self.generate_calculator_base(),
            "tests/golden/test_golden_master.py": self.generate_golden_test_template(),
            "docs/formulas.py": self.generate_formula_docs_template(),
        })
        return templates
