# -*- coding: utf-8 -*-
"""
Steam Balance Explainer for GL-003 UnifiedSteam
===============================================

Provides natural language explanations for steam system imbalances,
optimization recommendations, and causal analysis.

Features:
    - Mass balance explanation
    - Energy balance explanation
    - Imbalance root cause analysis
    - Optimization recommendations with rationale
    - Sankey diagram data export

Author: GL-ExplainabilityEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import hashlib
import json


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ImbalanceType(Enum):
    """Types of steam system imbalances."""
    MASS_SURPLUS = "mass_surplus"
    MASS_DEFICIT = "mass_deficit"
    ENERGY_SURPLUS = "energy_surplus"
    ENERGY_DEFICIT = "energy_deficit"
    PRESSURE_MISMATCH = "pressure_mismatch"
    QUALITY_DEGRADATION = "quality_degradation"
    TRAP_LOSSES = "trap_losses"
    CONDENSATE_LOSSES = "condensate_losses"


class SeverityLevel(Enum):
    """Severity levels for imbalances."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


THRESHOLD_CONFIG = {
    "mass_imbalance_percent": 2.0,  # > 2% triggers warning
    "energy_imbalance_percent": 3.0,  # > 3% triggers warning
    "critical_multiplier": 3.0,  # 3x warning threshold = critical
    "pressure_drop_kpa": 50.0,  # > 50 kPa unexpected drop
    "quality_degradation_percent": 5.0,  # > 5% moisture increase
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SteamHeader:
    """Represents a steam header in the system."""
    name: str
    pressure_kPa: float
    temperature_C: float
    mass_flow_kg_s: float
    enthalpy_kJ_kg: float
    quality: float = 1.0  # 1.0 = saturated vapor, <1.0 = wet steam

    @property
    def energy_flow_kW(self) -> float:
        """Calculate energy flow rate."""
        return self.mass_flow_kg_s * self.enthalpy_kJ_kg


@dataclass
class SteamBalance:
    """Steam system balance calculation results."""
    total_generation_kg_s: float
    total_consumption_kg_s: float
    total_losses_kg_s: float
    mass_imbalance_kg_s: float
    mass_imbalance_percent: float

    total_energy_in_kW: float
    total_energy_out_kW: float
    energy_imbalance_kW: float
    energy_imbalance_percent: float

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImbalanceExplanation:
    """Natural language explanation of an imbalance."""
    imbalance_type: ImbalanceType
    severity: SeverityLevel
    summary: str
    detailed_explanation: str
    probable_causes: List[str]
    recommended_actions: List[str]
    affected_equipment: List[str]
    estimated_loss_cost_per_hour: float = 0.0
    confidence_percent: float = 0.0
    provenance_hash: str = ""


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with explanation."""
    title: str
    description: str
    rationale: str
    expected_savings_kW: float
    expected_savings_percent: float
    implementation_difficulty: str  # "easy", "medium", "hard"
    payback_period_months: float
    affected_headers: List[str]
    prerequisites: List[str]


# =============================================================================
# STEAM BALANCE EXPLAINER
# =============================================================================

class SteamBalanceExplainer:
    """
    Explains steam system balances and imbalances in natural language.

    Provides:
        - Root cause analysis for imbalances
        - Natural language explanations
        - Optimization recommendations
        - Sankey diagram data for visualization
    """

    def __init__(self, steam_cost_per_kg: float = 0.03):
        """
        Initialize explainer.

        Args:
            steam_cost_per_kg: Cost of steam in $/kg (default $0.03/kg)
        """
        self.steam_cost_per_kg = steam_cost_per_kg
        self.thresholds = THRESHOLD_CONFIG.copy()

    def explain_balance(
        self,
        balance: SteamBalance,
        headers: List[SteamHeader]
    ) -> List[ImbalanceExplanation]:
        """
        Generate explanations for steam balance issues.

        Args:
            balance: Calculated steam balance
            headers: List of steam headers in system

        Returns:
            List of explanations for any detected issues
        """
        explanations = []

        # Check mass balance
        if abs(balance.mass_imbalance_percent) > self.thresholds["mass_imbalance_percent"]:
            explanation = self._explain_mass_imbalance(balance, headers)
            explanations.append(explanation)

        # Check energy balance
        if abs(balance.energy_imbalance_percent) > self.thresholds["energy_imbalance_percent"]:
            explanation = self._explain_energy_imbalance(balance, headers)
            explanations.append(explanation)

        # Check for quality degradation
        wet_headers = [h for h in headers if h.quality < 0.95]
        if wet_headers:
            explanation = self._explain_quality_issues(wet_headers)
            explanations.append(explanation)

        return explanations

    def _explain_mass_imbalance(
        self,
        balance: SteamBalance,
        headers: List[SteamHeader]
    ) -> ImbalanceExplanation:
        """Explain mass balance discrepancy."""
        is_surplus = balance.mass_imbalance_kg_s > 0
        imbalance_type = ImbalanceType.MASS_SURPLUS if is_surplus else ImbalanceType.MASS_DEFICIT

        # Determine severity
        threshold = self.thresholds["mass_imbalance_percent"]
        critical_threshold = threshold * self.thresholds["critical_multiplier"]

        if abs(balance.mass_imbalance_percent) > critical_threshold:
            severity = SeverityLevel.CRITICAL
        else:
            severity = SeverityLevel.WARNING

        # Generate explanation
        if is_surplus:
            summary = (
                f"Steam mass surplus of {balance.mass_imbalance_kg_s:.2f} kg/s "
                f"({balance.mass_imbalance_percent:.1f}%) detected"
            )
            detailed = (
                f"The steam system is generating {balance.mass_imbalance_kg_s:.2f} kg/s more steam "
                f"than is being consumed or accounted for. This represents a "
                f"{balance.mass_imbalance_percent:.1f}% imbalance relative to total generation. "
                f"Unaccounted steam surplus typically indicates metering errors, unmetered loads, "
                f"or venting/relief valve activation."
            )
            causes = [
                "Steam flowmeter reading high (calibration drift)",
                "Unmetered process loads not included in balance",
                "Relief valve lifting and venting steam",
                "Steam trap blowing through (failed open)",
                "Bypass valves partially open",
            ]
            actions = [
                "Verify flowmeter calibration on generation side",
                "Survey for unmetered steam consumers",
                "Check relief valve set points and operation",
                "Conduct steam trap survey",
                "Verify all bypass valves are fully closed",
            ]
        else:
            summary = (
                f"Steam mass deficit of {abs(balance.mass_imbalance_kg_s):.2f} kg/s "
                f"({abs(balance.mass_imbalance_percent):.1f}%) detected"
            )
            detailed = (
                f"The steam system is consuming {abs(balance.mass_imbalance_kg_s):.2f} kg/s more steam "
                f"than is being generated or accounted for. This represents a "
                f"{abs(balance.mass_imbalance_percent):.1f}% imbalance. Steam deficit typically indicates "
                f"metering errors on generation side, leaks, or unaccounted condensate losses."
            )
            causes = [
                "Steam flowmeter reading low on generation side",
                "Significant steam leaks in distribution system",
                "Condensate not being returned (dumped to drain)",
                "Flash steam losses at condensate receivers",
                "Header pressure drop causing additional condensation",
            ]
            actions = [
                "Verify flowmeter calibration on generation side",
                "Conduct visual and ultrasonic leak survey",
                "Check condensate return rates and flash recovery",
                "Verify condensate tank vents are not excessive",
                "Review insulation condition on headers",
            ]

        # Calculate cost impact
        loss_cost = abs(balance.mass_imbalance_kg_s) * self.steam_cost_per_kg * 3600

        # Generate provenance hash
        provenance_data = f"{balance.timestamp}:{balance.mass_imbalance_kg_s}:{imbalance_type.value}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()[:16]

        return ImbalanceExplanation(
            imbalance_type=imbalance_type,
            severity=severity,
            summary=summary,
            detailed_explanation=detailed,
            probable_causes=causes,
            recommended_actions=actions,
            affected_equipment=[h.name for h in headers],
            estimated_loss_cost_per_hour=loss_cost,
            confidence_percent=85.0,
            provenance_hash=provenance_hash,
        )

    def _explain_energy_imbalance(
        self,
        balance: SteamBalance,
        headers: List[SteamHeader]
    ) -> ImbalanceExplanation:
        """Explain energy balance discrepancy."""
        is_surplus = balance.energy_imbalance_kW > 0
        imbalance_type = ImbalanceType.ENERGY_SURPLUS if is_surplus else ImbalanceType.ENERGY_DEFICIT

        threshold = self.thresholds["energy_imbalance_percent"]
        critical_threshold = threshold * self.thresholds["critical_multiplier"]

        if abs(balance.energy_imbalance_percent) > critical_threshold:
            severity = SeverityLevel.CRITICAL
        else:
            severity = SeverityLevel.WARNING

        if is_surplus:
            summary = (
                f"Energy surplus of {balance.energy_imbalance_kW:.0f} kW "
                f"({balance.energy_imbalance_percent:.1f}%) detected"
            )
            detailed = (
                f"The steam system shows {balance.energy_imbalance_kW:.0f} kW more energy input "
                f"than energy output. This indicates either measurement errors, unaccounted "
                f"thermal losses, or energy being stored in the system."
            )
            causes = [
                "Boiler efficiency higher than assumed",
                "Heat recovery not accounted for",
                "System thermal mass accumulating heat",
                "Enthalpy calculations using incorrect steam properties",
            ]
            actions = [
                "Verify boiler efficiency test results",
                "Check all heat recovery equipment operation",
                "Wait for steady-state conditions before recalculating",
                "Verify steam property calculations match actual conditions",
            ]
        else:
            summary = (
                f"Energy deficit of {abs(balance.energy_imbalance_kW):.0f} kW "
                f"({abs(balance.energy_imbalance_percent):.1f}%) detected"
            )
            detailed = (
                f"The steam system shows {abs(balance.energy_imbalance_kW):.0f} kW less energy output "
                f"than energy input. This represents heat losses through radiation, convection, "
                f"uninsulated surfaces, or steam leaks."
            )
            causes = [
                "Significant heat losses from uninsulated piping",
                "Steam leaks (also causes mass deficit)",
                "Condensate dumped at high temperature",
                "Boiler blowdown losses not accounted for",
                "Heat losses from steam traps discharging to atmosphere",
            ]
            actions = [
                "Conduct thermographic survey for insulation gaps",
                "Quantify and reduce steam leak losses",
                "Maximize condensate return temperature",
                "Review blowdown rates and recovery options",
                "Survey steam trap discharge conditions",
            ]

        provenance_data = f"{balance.timestamp}:{balance.energy_imbalance_kW}:{imbalance_type.value}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()[:16]

        return ImbalanceExplanation(
            imbalance_type=imbalance_type,
            severity=severity,
            summary=summary,
            detailed_explanation=detailed,
            probable_causes=causes,
            recommended_actions=actions,
            affected_equipment=[h.name for h in headers],
            estimated_loss_cost_per_hour=abs(balance.energy_imbalance_kW) * 0.05,
            confidence_percent=80.0,
            provenance_hash=provenance_hash,
        )

    def _explain_quality_issues(
        self,
        wet_headers: List[SteamHeader]
    ) -> ImbalanceExplanation:
        """Explain steam quality degradation."""
        worst_header = min(wet_headers, key=lambda h: h.quality)
        moisture_percent = (1 - worst_header.quality) * 100

        if moisture_percent > 10:
            severity = SeverityLevel.CRITICAL
        else:
            severity = SeverityLevel.WARNING

        summary = (
            f"Steam quality degradation: {moisture_percent:.1f}% moisture "
            f"at {worst_header.name}"
        )
        detailed = (
            f"Steam header '{worst_header.name}' shows {moisture_percent:.1f}% moisture content, "
            f"indicating wet steam delivery. Wet steam reduces heat transfer efficiency, "
            f"causes water hammer risk, accelerates erosion in piping and valves, and "
            f"reduces the effective energy content of the steam."
        )
        causes = [
            "Insufficient steam separator capacity",
            "Boiler carryover due to high drum level",
            "Excessive pressure drops causing flashing",
            "Inadequate header drainage",
            "Failed or stuck-closed steam traps on header drip legs",
        ]
        actions = [
            "Check and clean steam separators",
            "Verify boiler drum level control",
            "Review header pressure profile for excessive drops",
            "Inspect and repair header drip leg traps",
            "Consider adding additional separation equipment",
        ]

        return ImbalanceExplanation(
            imbalance_type=ImbalanceType.QUALITY_DEGRADATION,
            severity=severity,
            summary=summary,
            detailed_explanation=detailed,
            probable_causes=causes,
            recommended_actions=actions,
            affected_equipment=[h.name for h in wet_headers],
            estimated_loss_cost_per_hour=moisture_percent * 10,
            confidence_percent=90.0,
            provenance_hash=hashlib.sha256(
                f"{worst_header.name}:{worst_header.quality}".encode()
            ).hexdigest()[:16],
        )

    def generate_optimization_recommendations(
        self,
        balance: SteamBalance,
        headers: List[SteamHeader]
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on current system state.

        Args:
            balance: Current steam balance
            headers: List of steam headers

        Returns:
            List of prioritized optimization recommendations
        """
        recommendations = []

        # Check for condensate recovery opportunities
        if balance.total_losses_kg_s > 0.1:
            recommendations.append(OptimizationRecommendation(
                title="Improve Condensate Recovery",
                description=(
                    "Increase condensate return rate to reduce makeup water "
                    "and energy requirements."
                ),
                rationale=(
                    f"Current condensate losses of {balance.total_losses_kg_s:.2f} kg/s "
                    f"represent significant energy waste. Condensate at high temperature "
                    f"contains substantial sensible heat that can reduce boiler fuel consumption."
                ),
                expected_savings_kW=balance.total_losses_kg_s * 400,
                expected_savings_percent=5.0,
                implementation_difficulty="medium",
                payback_period_months=12,
                affected_headers=[h.name for h in headers],
                prerequisites=[
                    "Identify condensate loss locations",
                    "Assess condensate return piping condition",
                    "Verify condensate receiver capacity",
                ],
            ))

        # Check for pressure optimization
        high_pressure_headers = [h for h in headers if h.pressure_kPa > 1000]
        if high_pressure_headers:
            recommendations.append(OptimizationRecommendation(
                title="Optimize Steam Pressure Levels",
                description=(
                    "Review steam pressure requirements and potentially "
                    "reduce header pressures where possible."
                ),
                rationale=(
                    "Higher pressure steam requires more energy to generate. "
                    "Matching steam pressure to actual process requirements "
                    "reduces boiler fuel consumption and improves safety."
                ),
                expected_savings_kW=len(high_pressure_headers) * 50,
                expected_savings_percent=2.0,
                implementation_difficulty="hard",
                payback_period_months=6,
                affected_headers=[h.name for h in high_pressure_headers],
                prerequisites=[
                    "Survey actual pressure requirements for each consumer",
                    "Verify control valve sizing at lower pressures",
                    "Assess impact on steam quality",
                ],
            ))

        return recommendations

    def export_sankey_data(
        self,
        headers: List[SteamHeader],
        balance: SteamBalance
    ) -> Dict[str, Any]:
        """
        Export data for Sankey diagram visualization.

        Args:
            headers: List of steam headers
            balance: Steam balance

        Returns:
            Dictionary with nodes and links for Sankey diagram
        """
        nodes = []
        links = []

        # Add generation node
        nodes.append({
            "id": "generation",
            "name": "Steam Generation",
            "value": balance.total_generation_kg_s,
        })

        # Add header nodes
        for i, header in enumerate(headers):
            nodes.append({
                "id": f"header_{i}",
                "name": header.name,
                "value": header.mass_flow_kg_s,
            })

        # Add consumption node
        nodes.append({
            "id": "consumption",
            "name": "Process Consumption",
            "value": balance.total_consumption_kg_s,
        })

        # Add losses node if significant
        if balance.total_losses_kg_s > 0.01:
            nodes.append({
                "id": "losses",
                "name": "System Losses",
                "value": balance.total_losses_kg_s,
            })

        # Create links
        for i, header in enumerate(headers):
            links.append({
                "source": "generation",
                "target": f"header_{i}",
                "value": header.mass_flow_kg_s,
            })
            links.append({
                "source": f"header_{i}",
                "target": "consumption",
                "value": header.mass_flow_kg_s * 0.9,
            })
            if balance.total_losses_kg_s > 0.01:
                links.append({
                    "source": f"header_{i}",
                    "target": "losses",
                    "value": header.mass_flow_kg_s * 0.1,
                })

        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "timestamp": balance.timestamp.isoformat(),
                "total_flow_kg_s": balance.total_generation_kg_s,
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def explain_steam_balance(
    generation_kg_s: float,
    consumption_kg_s: float,
    losses_kg_s: float = 0.0,
    headers: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to explain a steam balance.

    Args:
        generation_kg_s: Total steam generation rate
        consumption_kg_s: Total steam consumption rate
        losses_kg_s: Known system losses
        headers: Optional list of header dictionaries

    Returns:
        Dictionary with explanations and recommendations
    """
    # Create balance object
    imbalance = generation_kg_s - consumption_kg_s - losses_kg_s
    total = generation_kg_s if generation_kg_s > 0 else 1.0

    balance = SteamBalance(
        total_generation_kg_s=generation_kg_s,
        total_consumption_kg_s=consumption_kg_s,
        total_losses_kg_s=losses_kg_s,
        mass_imbalance_kg_s=imbalance,
        mass_imbalance_percent=(imbalance / total) * 100,
        total_energy_in_kW=generation_kg_s * 2700,
        total_energy_out_kW=consumption_kg_s * 2700,
        energy_imbalance_kW=imbalance * 2700,
        energy_imbalance_percent=(imbalance / total) * 100,
    )

    # Create header objects
    header_objects = []
    if headers:
        for h in headers:
            header_objects.append(SteamHeader(
                name=h.get("name", "Unknown"),
                pressure_kPa=h.get("pressure_kPa", 1000),
                temperature_C=h.get("temperature_C", 180),
                mass_flow_kg_s=h.get("mass_flow_kg_s", 1.0),
                enthalpy_kJ_kg=h.get("enthalpy_kJ_kg", 2750),
                quality=h.get("quality", 1.0),
            ))

    # Generate explanations
    explainer = SteamBalanceExplainer()
    explanations = explainer.explain_balance(balance, header_objects)
    recommendations = explainer.generate_optimization_recommendations(balance, header_objects)

    return {
        "balance": {
            "generation_kg_s": balance.total_generation_kg_s,
            "consumption_kg_s": balance.total_consumption_kg_s,
            "losses_kg_s": balance.total_losses_kg_s,
            "imbalance_kg_s": balance.mass_imbalance_kg_s,
            "imbalance_percent": balance.mass_imbalance_percent,
        },
        "explanations": [
            {
                "type": e.imbalance_type.value,
                "severity": e.severity.value,
                "summary": e.summary,
                "details": e.detailed_explanation,
                "causes": e.probable_causes,
                "actions": e.recommended_actions,
                "cost_per_hour": e.estimated_loss_cost_per_hour,
            }
            for e in explanations
        ],
        "recommendations": [
            {
                "title": r.title,
                "description": r.description,
                "rationale": r.rationale,
                "savings_kW": r.expected_savings_kW,
                "payback_months": r.payback_period_months,
            }
            for r in recommendations
        ],
    }


if __name__ == "__main__":
    # Example usage
    result = explain_steam_balance(
        generation_kg_s=100.0,
        consumption_kg_s=92.0,
        losses_kg_s=3.0,
        headers=[
            {"name": "HP Header", "pressure_kPa": 4000, "mass_flow_kg_s": 60},
            {"name": "LP Header", "pressure_kPa": 400, "mass_flow_kg_s": 40},
        ],
    )
    print(json.dumps(result, indent=2))
