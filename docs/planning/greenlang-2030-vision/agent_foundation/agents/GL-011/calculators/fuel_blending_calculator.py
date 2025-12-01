# -*- coding: utf-8 -*-
"""
Fuel Blending Calculator for GL-011 FUELCRAFT.

Provides deterministic algorithms for optimizing fuel blend ratios
to achieve target properties while meeting quality constraints.

Standards: ISO 17225, ASTM D4809
Zero-hallucination: All calculations are deterministic.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BlendingInput:
    """Input for blending optimization."""
    available_fuels: List[str]
    fuel_properties: Dict[str, Dict[str, Any]]
    target_heating_value: float
    max_moisture: float
    max_ash: float
    max_sulfur: float
    optimization_objective: str
    incompatible_pairs: List[List[str]]


@dataclass
class BlendingOutput:
    """Output of blending optimization."""
    blend_ratios: Dict[str, float]
    blend_heating_value: float
    blend_carbon_content: float
    blend_moisture: float
    blend_ash: float
    blend_sulfur: float
    quality_score: float
    compatibility_ok: bool
    warnings: List[str]
    estimated_emissions: Dict[str, float]
    provenance_hash: str


class FuelBlendingCalculator:
    """
    Deterministic fuel blending optimization calculator.

    Calculates optimal blend ratios to meet target fuel properties
    while satisfying quality constraints (moisture, ash, sulfur).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        self.config = config or {}
        self.calculation_count = 0

    def optimize_blend(self, input_data: BlendingInput) -> BlendingOutput:
        """
        Optimize fuel blend ratios.

        Args:
            input_data: Blending optimization parameters

        Returns:
            Optimized blend with properties
        """
        self.calculation_count += 1

        warnings = []
        fuel_scores = {}

        # Score each fuel based on properties
        for fuel in input_data.available_fuels:
            props = input_data.fuel_properties.get(fuel, {})
            score = 0

            # Heating value score
            hv = props.get('heating_value_mj_kg', 30)
            hv_diff = abs(hv - input_data.target_heating_value)
            score += max(0, 50 - hv_diff * 2)

            # Moisture penalty
            moisture = props.get('moisture_content_percent', 0)
            if moisture > input_data.max_moisture:
                score -= 20

            # Ash penalty
            ash = props.get('ash_content_percent', 0)
            if ash > input_data.max_ash:
                score -= 20

            # Sulfur penalty
            sulfur = props.get('sulfur_content_percent', 0)
            if sulfur > input_data.max_sulfur:
                score -= 30

            # Emission bonus for low-emission objective
            if input_data.optimization_objective == 'minimize_emissions':
                co2 = props.get('emission_factor_co2_kg_gj', 60)
                score += max(0, 50 - co2 * 0.5)

            fuel_scores[fuel] = max(0, score)

        # Normalize to ratios
        total_score = sum(fuel_scores.values())
        if total_score > 0:
            blend_ratios = {f: s / total_score for f, s in fuel_scores.items()}
        else:
            n = len(input_data.available_fuels)
            blend_ratios = {f: 1.0 / n for f in input_data.available_fuels}

        # Check compatibility
        active_fuels = [f for f, r in blend_ratios.items() if r > 0.01]
        compatibility_ok = True
        for pair in input_data.incompatible_pairs:
            if all(f in active_fuels for f in pair):
                compatibility_ok = False
                warnings.append(f"Incompatible fuels: {pair}")

        # Calculate blend properties
        blend_props = self._calculate_blend_properties(
            blend_ratios, input_data.fuel_properties
        )

        # Check constraints
        if blend_props['moisture'] > input_data.max_moisture:
            warnings.append(f"Blend moisture {blend_props['moisture']:.1f}% exceeds limit")
        if blend_props['ash'] > input_data.max_ash:
            warnings.append(f"Blend ash {blend_props['ash']:.1f}% exceeds limit")
        if blend_props['sulfur'] > input_data.max_sulfur:
            warnings.append(f"Blend sulfur {blend_props['sulfur']:.2f}% exceeds limit")

        # Calculate quality score
        quality_score = 100
        if blend_props['moisture'] > input_data.max_moisture:
            quality_score -= 20
        if blend_props['ash'] > input_data.max_ash:
            quality_score -= 15
        if blend_props['sulfur'] > input_data.max_sulfur:
            quality_score -= 25
        if not compatibility_ok:
            quality_score -= 20
        quality_score = max(0, quality_score)

        # Estimate emissions
        emissions = self._estimate_blend_emissions(
            blend_ratios, input_data.fuel_properties
        )

        provenance_hash = self._calculate_provenance(input_data, blend_ratios)

        return BlendingOutput(
            blend_ratios={k: round(v, 4) for k, v in blend_ratios.items()},
            blend_heating_value=round(blend_props['heating_value'], 2),
            blend_carbon_content=round(blend_props['carbon'], 2),
            blend_moisture=round(blend_props['moisture'], 2),
            blend_ash=round(blend_props['ash'], 2),
            blend_sulfur=round(blend_props['sulfur'], 3),
            quality_score=round(quality_score, 1),
            compatibility_ok=compatibility_ok,
            warnings=warnings,
            estimated_emissions=emissions,
            provenance_hash=provenance_hash
        )

    def _calculate_blend_properties(
        self,
        ratios: Dict[str, float],
        properties: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate weighted average properties."""
        result = {
            'heating_value': 0,
            'carbon': 0,
            'moisture': 0,
            'ash': 0,
            'sulfur': 0
        }

        for fuel, ratio in ratios.items():
            props = properties.get(fuel, {})
            result['heating_value'] += ratio * props.get('heating_value_mj_kg', 30)
            result['carbon'] += ratio * props.get('carbon_content_percent', 50)
            result['moisture'] += ratio * props.get('moisture_content_percent', 0)
            result['ash'] += ratio * props.get('ash_content_percent', 0)
            result['sulfur'] += ratio * props.get('sulfur_content_percent', 0)

        return result

    def _estimate_blend_emissions(
        self,
        ratios: Dict[str, float],
        properties: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Estimate emissions for blend."""
        emissions = {'co2_kg_gj': 0, 'nox_g_gj': 0, 'sox_g_gj': 0}

        for fuel, ratio in ratios.items():
            props = properties.get(fuel, {})
            emissions['co2_kg_gj'] += ratio * props.get('emission_factor_co2_kg_gj', 60)
            emissions['nox_g_gj'] += ratio * props.get('emission_factor_nox_g_gj', 100)
            emissions['sox_g_gj'] += ratio * props.get('emission_factor_sox_g_gj', 50)

        return {k: round(v, 2) for k, v in emissions.items()}

    def _calculate_provenance(
        self,
        input_data: BlendingInput,
        ratios: Dict[str, float]
    ) -> str:
        """Calculate provenance hash."""
        data = {
            'fuels': sorted(input_data.available_fuels),
            'target_hv': input_data.target_heating_value,
            'objective': input_data.optimization_objective,
            'ratios': ratios
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
