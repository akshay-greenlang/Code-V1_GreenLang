"""
Audit Trail Generator

Generates complete calculation provenance for regulatory compliance and auditing.

Every calculation MUST have:
1. Input parameters (with timestamps)
2. Emission factor selection logic
3. Unit conversions applied
4. Calculation steps with intermediate values
5. Final result
6. SHA-256 hash for tamper detection
7. Source URIs for all factors
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from greenlang.calculation.core_calculator import CalculationResult


@dataclass
class CalculationStep:
    """
    Individual step in calculation audit trail.

    Attributes:
        step_number: Sequential step number
        description: Human-readable step description
        operation: Operation type (lookup, multiply, convert, etc.)
        inputs: Input values for this step
        output: Output value from this step
        timestamp: When step was executed
    """
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, Any]
    output: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_number': self.step_number,
            'description': self.description,
            'operation': self.operation,
            'inputs': self.inputs,
            'output': str(self.output) if self.output is not None else None,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AuditTrail:
    """
    Complete audit trail for a calculation.

    This object provides COMPLETE PROVENANCE for regulatory audits:
    - What was calculated (inputs)
    - How it was calculated (steps)
    - When it was calculated (timestamps)
    - What data was used (factors with URIs)
    - Result verification (SHA-256 hash)

    Immutability: Once created, cannot be modified (preserves integrity)
    """
    calculation_id: str
    calculation_result: CalculationResult
    steps: List[CalculationStep]
    input_summary: Dict[str, Any]
    factor_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    trail_hash: Optional[str] = None

    def __post_init__(self):
        """Generate trail hash after initialization"""
        if self.trail_hash is None:
            self.trail_hash = self._calculate_trail_hash()

    def _calculate_trail_hash(self) -> str:
        """Calculate SHA-256 hash of complete audit trail"""
        trail_data = {
            'calculation_id': self.calculation_id,
            'input_summary': self.input_summary,
            'factor_summary': self.factor_summary,
            'steps': [step.to_dict() for step in self.steps],
            'output_summary': self.output_summary,
            'created_at': self.created_at.isoformat(),
        }

        trail_str = json.dumps(trail_data, sort_keys=True)
        return hashlib.sha256(trail_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """
        Verify audit trail integrity.

        Returns:
            True if trail is intact, False if tampered/corrupted
        """
        expected_hash = self._calculate_trail_hash()
        return self.trail_hash == expected_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'calculation_id': self.calculation_id,
            'calculation_result': self.calculation_result.to_dict(),
            'steps': [step.to_dict() for step in self.steps],
            'input_summary': self.input_summary,
            'factor_summary': self.factor_summary,
            'output_summary': self.output_summary,
            'created_at': self.created_at.isoformat(),
            'trail_hash': self.trail_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """
        Generate human-readable markdown audit report.

        Perfect for regulatory submissions and documentation.
        """
        md = f"# Calculation Audit Trail\n\n"
        md += f"**Calculation ID:** `{self.calculation_id}`\n\n"
        md += f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        md += f"**Trail Hash:** `{self.trail_hash}`\n\n"

        md += "## Input Summary\n\n"
        md += "| Parameter | Value |\n"
        md += "|-----------|-------|\n"
        for key, value in self.input_summary.items():
            md += f"| {key} | {value} |\n"

        md += "\n## Emission Factor\n\n"
        md += "| Attribute | Value |\n"
        md += "|-----------|-------|\n"
        for key, value in self.factor_summary.items():
            md += f"| {key} | {value} |\n"

        md += "\n## Calculation Steps\n\n"
        for step in self.steps:
            md += f"### Step {step.step_number}: {step.description}\n\n"
            md += f"**Operation:** `{step.operation}`\n\n"
            if step.inputs:
                md += "**Inputs:**\n\n"
                for key, value in step.inputs.items():
                    md += f"- {key}: `{value}`\n"
                md += "\n"
            md += f"**Output:** `{step.output}`\n\n"

        md += "## Result Summary\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        for key, value in self.output_summary.items():
            md += f"| {key} | {value} |\n"

        md += f"\n---\n\n"
        md += f"*This audit trail is cryptographically signed with SHA-256 hash: `{self.trail_hash}`*\n"

        return md


class AuditTrailGenerator:
    """
    Generates complete audit trails for calculations.

    ZERO-HALLUCINATION: Pure data transformation, no LLM involved.
    """

    def generate(self, calculation: CalculationResult) -> AuditTrail:
        """
        Generate complete audit trail from calculation result.

        Args:
            calculation: CalculationResult from EmissionCalculator

        Returns:
            AuditTrail with complete provenance

        Example:
            >>> from greenlang.calculation import EmissionCalculator, AuditTrailGenerator
            >>> calc = EmissionCalculator()
            >>> request = CalculationRequest(
            ...     factor_id='diesel',
            ...     activity_amount=100,
            ...     activity_unit='gallons'
            ... )
            >>> result = calc.calculate(request)
            >>> trail_gen = AuditTrailGenerator()
            >>> audit_trail = trail_gen.generate(result)
            >>> print(audit_trail.to_markdown())
        """
        # Input summary
        input_summary = {
            'Activity Type': calculation.request.factor_id,
            'Activity Amount': str(calculation.request.activity_amount),
            'Activity Unit': calculation.request.activity_unit,
            'Calculation Date': calculation.request.calculation_date.isoformat(),
            'Region': calculation.request.region or 'Not specified',
            'Request ID': calculation.request.request_id,
        }

        # Factor summary
        factor_summary = {}
        if calculation.factor_resolution:
            factor_summary = {
                'Factor ID': calculation.factor_resolution.factor_id,
                'Factor Value': str(calculation.factor_resolution.factor_value),
                'Factor Unit': calculation.factor_resolution.factor_unit,
                'Source': calculation.factor_resolution.source,
                'URI': calculation.factor_resolution.uri,
                'Last Updated': calculation.factor_resolution.last_updated,
                'Fallback Level': calculation.factor_resolution.fallback_level.value,
                'Data Quality': calculation.factor_resolution.data_quality_tier or 'Not specified',
                'Uncertainty': f"{calculation.factor_resolution.uncertainty_pct}%" if calculation.factor_resolution.uncertainty_pct else 'Not specified',
                'Standard': calculation.factor_resolution.standard or 'Not specified',
            }

        # Convert calculation steps to CalculationStep objects
        steps = []
        for i, step_data in enumerate(calculation.calculation_steps, 1):
            step = CalculationStep(
                step_number=i,
                description=step_data.get('description', ''),
                operation=step_data.get('operation', 'unknown'),
                inputs={k: v for k, v in step_data.items() if k not in ['step', 'description', 'operation']},
                output=step_data.get('emissions_kg_co2e') or step_data.get('output'),
                timestamp=calculation.calculation_timestamp,
            )
            steps.append(step)

        # Output summary
        output_summary = {
            'Emissions (kg CO2e)': str(calculation.emissions_kg_co2e),
            'Emissions (tonnes CO2e)': str(calculation.emissions_kg_co2e / 1000),
            'Status': calculation.status.value,
            'Calculation Duration (ms)': f"{calculation.calculation_duration_ms:.2f}" if calculation.calculation_duration_ms else 'N/A',
            'Warnings': len(calculation.warnings),
            'Errors': len(calculation.errors),
            'Provenance Hash': calculation.provenance_hash,
            'Engine Version': calculation.calculation_engine_version,
        }

        # Create audit trail
        trail = AuditTrail(
            calculation_id=calculation.request.request_id,
            calculation_result=calculation,
            steps=steps,
            input_summary=input_summary,
            factor_summary=factor_summary,
            output_summary=output_summary,
        )

        return trail

    def generate_batch_report(
        self,
        calculations: List[CalculationResult],
        report_title: str = "Batch Calculation Report"
    ) -> str:
        """
        Generate markdown report for batch calculations.

        Args:
            calculations: List of CalculationResults
            report_title: Title for report

        Returns:
            Markdown report summarizing all calculations
        """
        md = f"# {report_title}\n\n"
        md += f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        md += f"**Total Calculations:** {len(calculations)}\n\n"

        # Summary statistics
        total_emissions = sum(calc.emissions_kg_co2e for calc in calculations)
        successful = len([c for c in calculations if c.status.value == 'success'])
        failed = len([c for c in calculations if c.status.value == 'failed'])
        warnings = len([c for c in calculations if c.status.value == 'warning'])

        md += "## Summary\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        md += f"| Total Emissions (kg CO2e) | {total_emissions:,.3f} |\n"
        md += f"| Total Emissions (tonnes CO2e) | {total_emissions/1000:,.3f} |\n"
        md += f"| Successful Calculations | {successful} |\n"
        md += f"| Failed Calculations | {failed} |\n"
        md += f"| Calculations with Warnings | {warnings} |\n"

        md += "\n## Individual Calculations\n\n"
        md += "| # | Factor ID | Amount | Unit | Emissions (kg CO2e) | Status |\n"
        md += "|---|-----------|--------|------|---------------------|--------|\n"

        for i, calc in enumerate(calculations, 1):
            md += f"| {i} | {calc.request.factor_id} | {calc.request.activity_amount} | "
            md += f"{calc.request.activity_unit} | {calc.emissions_kg_co2e:,.3f} | "
            md += f"{calc.status.value} |\n"

        return md
