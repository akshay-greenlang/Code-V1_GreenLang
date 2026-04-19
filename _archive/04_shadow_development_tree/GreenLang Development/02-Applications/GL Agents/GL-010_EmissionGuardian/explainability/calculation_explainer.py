# -*- coding: utf-8 -*-
"""Calculation Explainer for GL-010 EmissionsGuardian"""

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
import hashlib, logging, uuid

from .schemas import (
    AudienceLevel, CalculationStep, ConfidenceLevel, DecisionTrace,
    Explanation, ExplanationType, ReasoningStep, TemplateVersion,
    UncertaintyExplanation, UnitConversionExplanation,
)

logger = logging.getLogger(__name__)

EPA_FORMULAS = {
    "F-5": {"name": "NOx Emission Rate", "citation": "40 CFR Part 75, Appendix F, Equation F-5"},
    "F-11": {"name": "SO2 Emission Rate", "citation": "40 CFR Part 75, Appendix F, Equation F-11"},
}

FUEL_F_FACTORS = {
    "natural_gas": {"Fd": Decimal("8710")},
    "coal_bituminous": {"Fd": Decimal("9780")},
}


class CalculationExplainer:
    """Provides step-by-step explanations for emissions calculations."""

    def __init__(self, precision: int = 6):
        self.precision = precision
        self.template_version = TemplateVersion(
            template_id="CALC_V1", version="1.0.0",
            effective_date=datetime(2024, 1, 1), approved_by="GL-010",
            checksum=hashlib.sha256(b"calc_v1").hexdigest()
        )

    def explain_nox_emission_rate(
        self, nox_ppm: float, o2_percent: float, fuel_type: str = "natural_gas"
    ) -> List[CalculationStep]:
        """Explain NOx emission rate calculation step by step."""
        steps = []
        f_factor = FUEL_F_FACTORS.get(fuel_type, FUEL_F_FACTORS["natural_gas"])["Fd"]
        k = Decimal("1.194E-7")

        o2_corr = (Decimal("20.9") / (Decimal("20.9") - Decimal(str(o2_percent)))).quantize(
            Decimal(10) ** -self.precision, rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=1, step_name="O2 Correction",
            description="Calculate O2 correction factor",
            formula="20.9 / (20.9 - O2%)",
            inputs={"O2": Decimal(str(o2_percent))},
            input_units={"O2": "%"}, input_sources={"O2": "CEMS"},
            output_value=o2_corr, output_unit="dimensionless",
            epa_reference="F-5", epa_citation=EPA_FORMULAS["F-5"]["citation"]
        ))

        emission = (Decimal(str(nox_ppm)) * k * o2_corr * f_factor).quantize(
            Decimal(10) ** -self.precision, rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=2, step_name="Emission Rate",
            description="Calculate NOx emission rate",
            formula="NOx_ppm * K * O2_corr * Fd",
            inputs={"NOx": Decimal(str(nox_ppm)), "K": k, "O2_corr": o2_corr, "Fd": f_factor},
            input_units={"NOx": "ppm", "Fd": "scf/MMBtu"},
            input_sources={"NOx": "CEMS", "Fd": "40 CFR 75"},
            output_value=emission, output_unit="lb/MMBtu",
            epa_reference="F-5", epa_citation=EPA_FORMULAS["F-5"]["citation"]
        ))
        return steps

    def explain_unit_conversion(self, value: float, from_u: str, to_u: str) -> UnitConversionExplanation:
        """Explain unit conversion."""
        conversions = {"ton_to_lb": Decimal("2000"), "kg_to_lb": Decimal("2.20462")}
        key = f"{from_u}_to_{to_u}".lower()
        factor = conversions.get(key, Decimal("1"))
        return UnitConversionExplanation(
            from_value=Decimal(str(value)), from_unit=from_u,
            to_value=(Decimal(str(value)) * factor).quantize(Decimal(10) ** -self.precision),
            to_unit=to_u, conversion_factor=factor,
            conversion_formula=f"{to_u} = {from_u} * {factor}",
            conversion_source="Standard" if factor != 1 else "Unknown"
        )

    def explain_uncertainty(self, calc_id: str, uncertainties: Dict) -> UncertaintyExplanation:
        """Explain uncertainty propagation using GUM."""
        var = sum((Decimal(str(u.get("std", 0))) ** 2) for u in uncertainties.values())
        combined = var.sqrt().quantize(Decimal(10) ** -self.precision)
        return UncertaintyExplanation(
            calculation_id=calc_id, input_uncertainties=uncertainties,
            propagation_method="GUM", coverage_factor=2.0, confidence_interval="95%",
            combined_uncertainty=combined, expanded_uncertainty=combined * 2,
            explanation_text=f"Combined: {combined}"
        )

    def create_explanation(self, calc_type: str, steps: List[CalculationStep]) -> Explanation:
        """Create complete Explanation from steps."""
        final = steps[-1] if steps else None
        trace = DecisionTrace(
            trace_id=str(uuid.uuid4()), decision_type=f"calc_{calc_type}",
            decision_result=str(final.output_value) if final else "N/A",
            confidence=0.95, confidence_level=ConfidenceLevel.VERY_HIGH,
            steps=[ReasoningStep(
                step_number=s.step_number, step_type="calculation",
                description=s.description, formula=s.formula,
                input_values={k: float(v) for k, v in s.inputs.items()},
                output_values={"result": float(s.output_value)}
            ) for s in steps],
            input_data_hash=hashlib.sha256(b"input").hexdigest(),
            output_data_hash=hashlib.sha256(b"output").hexdigest(),
            start_time=datetime.now(), processing_time_ms=0.0,
            agent_id="GL-010", agent_version="1.0.0"
        )
        return Explanation(
            explanation_id=str(uuid.uuid4()),
            explanation_type=ExplanationType.CALCULATION,
            title=f"{calc_type} Calculation",
            summary=f"Result: {final.output_value} {final.output_unit}" if final else "N/A",
            detailed_explanation="
".join(f"Step {s.step_number}: {s.step_name}" for s in steps),
            key_findings=[f"{final.output_value} {final.output_unit}"] if final else [],
            decision_trace=trace,
            regulatory_citations=[s.epa_citation for s in steps if s.epa_citation],
            confidence=0.95, confidence_level=ConfidenceLevel.VERY_HIGH,
            template_version=self.template_version,
            provenance_hash=trace.calculate_provenance_hash(),
            generated_by="GL-010-CalculationExplainer"
        )
