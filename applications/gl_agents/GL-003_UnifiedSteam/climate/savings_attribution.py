"""
Savings Attribution for GL-003 UNIFIEDSTEAM

Provides causal savings attribution linking operational interventions
to measured energy and emissions reductions. Integrates with the
causal inference layer for root cause analysis.

Author: GL-003 Climate Intelligence Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of interventions that can be attributed."""
    DESUPERHEATER_OPTIMIZATION = "desuperheater_optimization"
    TRAP_REPAIR = "trap_repair"
    TRAP_REPLACEMENT = "trap_replacement"
    CONDENSATE_RECOVERY = "condensate_recovery"
    HEADER_PRESSURE_OPTIMIZATION = "header_pressure_optimization"
    INSULATION_REPAIR = "insulation_repair"
    BLOWDOWN_OPTIMIZATION = "blowdown_optimization"
    BOILER_EFFICIENCY = "boiler_efficiency"
    SETPOINT_ADJUSTMENT = "setpoint_adjustment"
    MAINTENANCE_ACTION = "maintenance_action"
    OPERATIONAL_CHANGE = "operational_change"


class AttributionConfidence(Enum):
    """Confidence level in attribution."""
    HIGH = "high"        # Strong causal evidence, low uncertainty
    MEDIUM = "medium"    # Moderate causal evidence
    LOW = "low"          # Weak causal evidence, high uncertainty
    UNCERTAIN = "uncertain"  # Cannot confidently attribute


@dataclass
class CausalSavingsLink:
    """
    Link between an intervention and observed savings.

    Establishes causal relationship with confidence and evidence.
    """
    intervention_id: str
    intervention_type: InterventionType
    intervention_date: datetime
    intervention_description: str

    # Observed savings
    steam_savings_kg: Decimal
    energy_savings_gj: Decimal
    fuel_savings_gj: Decimal
    co2e_savings_kg: Decimal

    # Attribution
    attribution_confidence: AttributionConfidence
    attribution_pct: Decimal  # Percentage of savings attributed to this intervention

    # Evidence
    causal_evidence: List[str]
    supporting_signals: List[str]
    confounding_factors: List[str]

    # Uncertainty
    uncertainty_pct: Decimal
    lower_bound_co2e: Decimal
    upper_bound_co2e: Decimal

    # Provenance
    analysis_method: str
    calculation_hash: str

    def get_attributed_savings(self) -> Dict[str, Decimal]:
        """Get savings attributed to this intervention."""
        factor = self.attribution_pct / Decimal("100")
        return {
            "steam_kg": self.steam_savings_kg * factor,
            "energy_gj": self.energy_savings_gj * factor,
            "fuel_gj": self.fuel_savings_gj * factor,
            "co2e_kg": self.co2e_savings_kg * factor,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intervention_id": self.intervention_id,
            "type": self.intervention_type.value,
            "date": self.intervention_date.isoformat(),
            "description": self.intervention_description,
            "savings": {
                "steam_kg": str(self.steam_savings_kg),
                "energy_gj": str(self.energy_savings_gj),
                "fuel_gj": str(self.fuel_savings_gj),
                "co2e_kg": str(self.co2e_savings_kg),
            },
            "attribution": {
                "confidence": self.attribution_confidence.value,
                "percentage": str(self.attribution_pct),
                "attributed_co2e_kg": str(
                    self.co2e_savings_kg * self.attribution_pct / Decimal("100")
                ),
            },
            "uncertainty": {
                "pct": str(self.uncertainty_pct),
                "lower_co2e_kg": str(self.lower_bound_co2e),
                "upper_co2e_kg": str(self.upper_bound_co2e),
            },
            "evidence": {
                "causal": self.causal_evidence,
                "signals": self.supporting_signals,
                "confounders": self.confounding_factors,
            },
            "analysis_method": self.analysis_method,
            "calculation_hash": self.calculation_hash,
        }


@dataclass
class InterventionImpact:
    """
    Measured impact of an intervention over time.

    Tracks pre/post performance with statistical analysis.
    """
    intervention_id: str
    intervention_type: InterventionType
    intervention_date: datetime

    # Pre-intervention metrics (baseline)
    pre_period_start: datetime
    pre_period_end: datetime
    pre_steam_rate_kg_hr: Decimal
    pre_energy_rate_gj_hr: Decimal
    pre_efficiency_pct: Decimal

    # Post-intervention metrics
    post_period_start: datetime
    post_period_end: datetime
    post_steam_rate_kg_hr: Decimal
    post_energy_rate_gj_hr: Decimal
    post_efficiency_pct: Decimal

    # Calculated impact
    steam_rate_change_pct: Decimal
    energy_rate_change_pct: Decimal
    efficiency_change_pct: Decimal

    # Statistical significance
    p_value: Decimal
    is_significant: bool
    sample_size_pre: int
    sample_size_post: int

    # Annualized savings estimate
    annualized_steam_savings_kg: Decimal
    annualized_energy_savings_gj: Decimal
    annualized_co2e_savings_kg: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intervention": {
                "id": self.intervention_id,
                "type": self.intervention_type.value,
                "date": self.intervention_date.isoformat(),
            },
            "pre_intervention": {
                "period": f"{self.pre_period_start.date()} to {self.pre_period_end.date()}",
                "steam_rate_kg_hr": str(self.pre_steam_rate_kg_hr),
                "energy_rate_gj_hr": str(self.pre_energy_rate_gj_hr),
                "efficiency_pct": str(self.pre_efficiency_pct),
                "sample_size": self.sample_size_pre,
            },
            "post_intervention": {
                "period": f"{self.post_period_start.date()} to {self.post_period_end.date()}",
                "steam_rate_kg_hr": str(self.post_steam_rate_kg_hr),
                "energy_rate_gj_hr": str(self.post_energy_rate_gj_hr),
                "efficiency_pct": str(self.post_efficiency_pct),
                "sample_size": self.sample_size_post,
            },
            "impact": {
                "steam_rate_change_pct": str(self.steam_rate_change_pct),
                "energy_rate_change_pct": str(self.energy_rate_change_pct),
                "efficiency_change_pct": str(self.efficiency_change_pct),
            },
            "statistical": {
                "p_value": str(self.p_value),
                "is_significant": self.is_significant,
            },
            "annualized_savings": {
                "steam_kg": str(self.annualized_steam_savings_kg),
                "energy_gj": str(self.annualized_energy_savings_gj),
                "co2e_kg": str(self.annualized_co2e_savings_kg),
            },
        }


@dataclass
class AttributedSavings:
    """
    Complete attributed savings report.

    Summarizes all savings with attribution to interventions.
    """
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Total observed savings
    total_steam_savings_kg: Decimal
    total_energy_savings_gj: Decimal
    total_fuel_savings_gj: Decimal
    total_co2e_savings_kg: Decimal

    # Attributed savings
    attributed_savings: List[CausalSavingsLink]
    total_attributed_co2e_kg: Decimal
    unattributed_co2e_kg: Decimal
    attribution_coverage_pct: Decimal

    # Summary by intervention type
    savings_by_type: Dict[InterventionType, Decimal]

    # Uncertainty
    total_uncertainty_pct: Decimal
    confidence_weighted_co2e: Decimal

    # Audit
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "total_savings": {
                "steam_kg": str(self.total_steam_savings_kg),
                "energy_gj": str(self.total_energy_savings_gj),
                "fuel_gj": str(self.total_fuel_savings_gj),
                "co2e_kg": str(self.total_co2e_savings_kg),
            },
            "attribution": {
                "total_attributed_co2e_kg": str(self.total_attributed_co2e_kg),
                "unattributed_co2e_kg": str(self.unattributed_co2e_kg),
                "coverage_pct": str(self.attribution_coverage_pct),
            },
            "by_intervention_type": {
                it.value: str(v) for it, v in self.savings_by_type.items()
            },
            "attributed_savings_details": [
                s.to_dict() for s in self.attributed_savings
            ],
            "uncertainty_pct": str(self.total_uncertainty_pct),
            "audit_trail": self.audit_trail,
        }


class SavingsAttributor:
    """
    Savings attribution engine for GL-003 UNIFIEDSTEAM.

    Links operational interventions to observed energy and emissions
    savings using causal inference methods.
    """

    # Default attribution percentages by confidence level
    DEFAULT_ATTRIBUTION_PCT = {
        AttributionConfidence.HIGH: Decimal("90"),
        AttributionConfidence.MEDIUM: Decimal("70"),
        AttributionConfidence.LOW: Decimal("40"),
        AttributionConfidence.UNCERTAIN: Decimal("20"),
    }

    def __init__(
        self,
        emission_factor_kg_per_gj: Decimal = Decimal("56.1"),  # Natural gas
        significance_threshold: Decimal = Decimal("0.05"),
    ):
        """
        Initialize savings attributor.

        Args:
            emission_factor_kg_per_gj: Default emission factor for fuel
            significance_threshold: P-value threshold for significance
        """
        self.emission_factor = emission_factor_kg_per_gj
        self.significance_threshold = significance_threshold
        self._interventions: Dict[str, Dict[str, Any]] = {}
        self._impacts: List[InterventionImpact] = []
        self._attributions: List[CausalSavingsLink] = []
        self._audit_log: List[Dict[str, Any]] = []

    def register_intervention(
        self,
        intervention_id: str,
        intervention_type: InterventionType,
        intervention_date: datetime,
        description: str,
        asset_id: str,
        expected_savings_pct: Optional[Decimal] = None,
    ):
        """
        Register a new intervention for tracking.

        Args:
            intervention_id: Unique identifier
            intervention_type: Type of intervention
            intervention_date: Date/time of intervention
            description: Description of what was done
            asset_id: ID of affected asset
            expected_savings_pct: Expected savings percentage
        """
        self._interventions[intervention_id] = {
            "type": intervention_type,
            "date": intervention_date,
            "description": description,
            "asset_id": asset_id,
            "expected_savings_pct": expected_savings_pct,
            "registered_at": datetime.now(timezone.utc),
        }

        self._log_action("register_intervention", {
            "intervention_id": intervention_id,
            "type": intervention_type.value,
        })

    def calculate_intervention_impact(
        self,
        intervention_id: str,
        pre_data: List[Dict[str, Decimal]],
        post_data: List[Dict[str, Decimal]],
        operating_hours_per_year: Decimal = Decimal("8760"),
    ) -> InterventionImpact:
        """
        Calculate the impact of an intervention.

        Args:
            intervention_id: ID of registered intervention
            pre_data: List of pre-intervention measurements
            post_data: List of post-intervention measurements
            operating_hours_per_year: Hours of operation per year

        Returns:
            InterventionImpact with calculated metrics
        """
        if intervention_id not in self._interventions:
            raise KeyError(f"Intervention not found: {intervention_id}")

        intervention = self._interventions[intervention_id]

        # Calculate pre-intervention averages
        pre_steam = [d["steam_rate_kg_hr"] for d in pre_data if "steam_rate_kg_hr" in d]
        pre_energy = [d["energy_rate_gj_hr"] for d in pre_data if "energy_rate_gj_hr" in d]

        pre_steam_avg = sum(pre_steam) / len(pre_steam) if pre_steam else Decimal("0")
        pre_energy_avg = sum(pre_energy) / len(pre_energy) if pre_energy else Decimal("0")

        # Calculate post-intervention averages
        post_steam = [d["steam_rate_kg_hr"] for d in post_data if "steam_rate_kg_hr" in d]
        post_energy = [d["energy_rate_gj_hr"] for d in post_data if "energy_rate_gj_hr" in d]

        post_steam_avg = sum(post_steam) / len(post_steam) if post_steam else Decimal("0")
        post_energy_avg = sum(post_energy) / len(post_energy) if post_energy else Decimal("0")

        # Calculate changes
        steam_change = Decimal("0")
        if pre_steam_avg > 0:
            steam_change = (
                (post_steam_avg - pre_steam_avg) / pre_steam_avg
            ) * Decimal("100")

        energy_change = Decimal("0")
        if pre_energy_avg > 0:
            energy_change = (
                (post_energy_avg - pre_energy_avg) / pre_energy_avg
            ) * Decimal("100")

        # Estimate efficiency change (simplified)
        efficiency_change = -energy_change  # Negative energy = positive efficiency

        # Simple t-test approximation for significance
        # In production, use scipy.stats.ttest_ind
        p_value = self._estimate_p_value(pre_energy, post_energy)
        is_significant = p_value < self.significance_threshold

        # Calculate annualized savings
        steam_rate_reduction = max(Decimal("0"), pre_steam_avg - post_steam_avg)
        energy_rate_reduction = max(Decimal("0"), pre_energy_avg - post_energy_avg)

        annual_steam = steam_rate_reduction * operating_hours_per_year
        annual_energy = energy_rate_reduction * operating_hours_per_year
        annual_co2e = annual_energy * self.emission_factor

        impact = InterventionImpact(
            intervention_id=intervention_id,
            intervention_type=intervention["type"],
            intervention_date=intervention["date"],
            pre_period_start=pre_data[0].get("timestamp", intervention["date"]) if pre_data else intervention["date"],
            pre_period_end=pre_data[-1].get("timestamp", intervention["date"]) if pre_data else intervention["date"],
            pre_steam_rate_kg_hr=pre_steam_avg.quantize(Decimal("0.1")),
            pre_energy_rate_gj_hr=pre_energy_avg.quantize(Decimal("0.001")),
            pre_efficiency_pct=Decimal("82"),  # Placeholder
            post_period_start=post_data[0].get("timestamp", intervention["date"]) if post_data else intervention["date"],
            post_period_end=post_data[-1].get("timestamp", intervention["date"]) if post_data else intervention["date"],
            post_steam_rate_kg_hr=post_steam_avg.quantize(Decimal("0.1")),
            post_energy_rate_gj_hr=post_energy_avg.quantize(Decimal("0.001")),
            post_efficiency_pct=Decimal("82") + efficiency_change,
            steam_rate_change_pct=steam_change.quantize(Decimal("0.1")),
            energy_rate_change_pct=energy_change.quantize(Decimal("0.1")),
            efficiency_change_pct=efficiency_change.quantize(Decimal("0.1")),
            p_value=p_value.quantize(Decimal("0.001")),
            is_significant=is_significant,
            sample_size_pre=len(pre_data),
            sample_size_post=len(post_data),
            annualized_steam_savings_kg=annual_steam.quantize(Decimal("1")),
            annualized_energy_savings_gj=annual_energy.quantize(Decimal("0.1")),
            annualized_co2e_savings_kg=annual_co2e.quantize(Decimal("1")),
        )

        self._impacts.append(impact)
        self._log_action("calculate_impact", {
            "intervention_id": intervention_id,
            "is_significant": is_significant,
            "annual_co2e_savings_kg": str(annual_co2e),
        })

        return impact

    def create_attribution(
        self,
        intervention_id: str,
        observed_savings: Dict[str, Decimal],
        causal_evidence: List[str],
        confidence: AttributionConfidence = AttributionConfidence.MEDIUM,
        attribution_pct: Optional[Decimal] = None,
    ) -> CausalSavingsLink:
        """
        Create a causal attribution link.

        Args:
            intervention_id: ID of intervention
            observed_savings: Dictionary of observed savings
            causal_evidence: List of causal evidence items
            confidence: Attribution confidence level
            attribution_pct: Override attribution percentage

        Returns:
            CausalSavingsLink
        """
        if intervention_id not in self._interventions:
            raise KeyError(f"Intervention not found: {intervention_id}")

        intervention = self._interventions[intervention_id]

        # Use default or provided attribution percentage
        attr_pct = attribution_pct or self.DEFAULT_ATTRIBUTION_PCT[confidence]

        # Extract savings
        steam_savings = observed_savings.get("steam_kg", Decimal("0"))
        energy_savings = observed_savings.get("energy_gj", Decimal("0"))
        fuel_savings = observed_savings.get("fuel_gj", energy_savings)
        co2e_savings = fuel_savings * self.emission_factor

        # Calculate uncertainty based on confidence
        uncertainty_map = {
            AttributionConfidence.HIGH: Decimal("10"),
            AttributionConfidence.MEDIUM: Decimal("25"),
            AttributionConfidence.LOW: Decimal("50"),
            AttributionConfidence.UNCERTAIN: Decimal("75"),
        }
        uncertainty = uncertainty_map[confidence]

        margin = co2e_savings * (uncertainty / Decimal("100")) * Decimal("1.96")

        calc_hash = self._compute_hash(
            intervention_id,
            str(co2e_savings),
            confidence.value,
        )

        link = CausalSavingsLink(
            intervention_id=intervention_id,
            intervention_type=intervention["type"],
            intervention_date=intervention["date"],
            intervention_description=intervention["description"],
            steam_savings_kg=steam_savings.quantize(Decimal("0.1")),
            energy_savings_gj=energy_savings.quantize(Decimal("0.001")),
            fuel_savings_gj=fuel_savings.quantize(Decimal("0.001")),
            co2e_savings_kg=co2e_savings.quantize(Decimal("0.1")),
            attribution_confidence=confidence,
            attribution_pct=attr_pct,
            causal_evidence=causal_evidence,
            supporting_signals=[],
            confounding_factors=[],
            uncertainty_pct=uncertainty,
            lower_bound_co2e=(co2e_savings - margin).quantize(Decimal("0.1")),
            upper_bound_co2e=(co2e_savings + margin).quantize(Decimal("0.1")),
            analysis_method="Causal attribution with pre/post analysis",
            calculation_hash=calc_hash,
        )

        self._attributions.append(link)
        self._log_action("create_attribution", {
            "intervention_id": intervention_id,
            "confidence": confidence.value,
            "co2e_savings_kg": str(co2e_savings),
        })

        return link

    def generate_attribution_report(
        self,
        period_start: datetime,
        period_end: datetime,
        total_observed_savings: Dict[str, Decimal],
    ) -> AttributedSavings:
        """
        Generate complete attribution report.

        Args:
            period_start: Start of period
            period_end: End of period
            total_observed_savings: Total observed savings

        Returns:
            AttributedSavings report
        """
        import uuid

        # Filter attributions for period
        period_attrs = [
            a for a in self._attributions
            if period_start <= a.intervention_date <= period_end
        ]

        # Calculate totals
        total_steam = total_observed_savings.get("steam_kg", Decimal("0"))
        total_energy = total_observed_savings.get("energy_gj", Decimal("0"))
        total_fuel = total_observed_savings.get("fuel_gj", total_energy)
        total_co2e = total_fuel * self.emission_factor

        # Calculate attributed totals
        attributed_co2e = Decimal("0")
        savings_by_type: Dict[InterventionType, Decimal] = {}

        for attr in period_attrs:
            attributed = attr.get_attributed_savings()
            attributed_co2e += attributed["co2e_kg"]

            int_type = attr.intervention_type
            savings_by_type[int_type] = savings_by_type.get(
                int_type, Decimal("0")
            ) + attributed["co2e_kg"]

        unattributed = max(Decimal("0"), total_co2e - attributed_co2e)
        coverage = (
            (attributed_co2e / total_co2e) * Decimal("100")
            if total_co2e > 0 else Decimal("0")
        )

        # Calculate confidence-weighted total
        weighted_total = Decimal("0")
        for attr in period_attrs:
            conf_weight = {
                AttributionConfidence.HIGH: Decimal("1.0"),
                AttributionConfidence.MEDIUM: Decimal("0.7"),
                AttributionConfidence.LOW: Decimal("0.4"),
                AttributionConfidence.UNCERTAIN: Decimal("0.2"),
            }[attr.attribution_confidence]
            weighted_total += attr.get_attributed_savings()["co2e_kg"] * conf_weight

        # Calculate overall uncertainty
        if period_attrs:
            avg_unc = sum(a.uncertainty_pct for a in period_attrs) / len(period_attrs)
        else:
            avg_unc = Decimal("50")

        report_id = f"ATTR-{uuid.uuid4().hex[:8].upper()}"

        report = AttributedSavings(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc),
            period_start=period_start,
            period_end=period_end,
            total_steam_savings_kg=total_steam.quantize(Decimal("0.1")),
            total_energy_savings_gj=total_energy.quantize(Decimal("0.001")),
            total_fuel_savings_gj=total_fuel.quantize(Decimal("0.001")),
            total_co2e_savings_kg=total_co2e.quantize(Decimal("0.1")),
            attributed_savings=period_attrs,
            total_attributed_co2e_kg=attributed_co2e.quantize(Decimal("0.1")),
            unattributed_co2e_kg=unattributed.quantize(Decimal("0.1")),
            attribution_coverage_pct=coverage.quantize(Decimal("0.1")),
            savings_by_type=savings_by_type,
            total_uncertainty_pct=avg_unc.quantize(Decimal("0.1")),
            confidence_weighted_co2e=weighted_total.quantize(Decimal("0.1")),
            audit_trail=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "generate_report",
                "report_id": report_id,
            }],
        )

        self._log_action("generate_attribution_report", {
            "report_id": report_id,
            "total_co2e_kg": str(total_co2e),
            "attributed_co2e_kg": str(attributed_co2e),
        })

        return report

    def _estimate_p_value(
        self,
        pre_values: List[Decimal],
        post_values: List[Decimal],
    ) -> Decimal:
        """
        Estimate p-value for difference in means.

        Simple approximation - in production use scipy.stats.
        """
        if len(pre_values) < 2 or len(post_values) < 2:
            return Decimal("1.0")  # Not enough data

        pre_mean = sum(pre_values) / len(pre_values)
        post_mean = sum(post_values) / len(post_values)

        # Simplified: larger difference = smaller p-value
        if pre_mean == 0:
            return Decimal("0.5")

        pct_diff = abs(post_mean - pre_mean) / pre_mean

        # Map to approximate p-value
        if pct_diff > Decimal("0.10"):
            return Decimal("0.01")
        elif pct_diff > Decimal("0.05"):
            return Decimal("0.05")
        elif pct_diff > Decimal("0.02"):
            return Decimal("0.10")
        else:
            return Decimal("0.50")

    def _compute_hash(self, *args) -> str:
        """Compute deterministic hash."""
        data = "|".join(str(a) for a in args)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log."""
        return self._audit_log.copy()
