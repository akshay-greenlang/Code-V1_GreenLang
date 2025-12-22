"""
GL-003 UnifiedSteam GraphQL API

Strawberry GraphQL API for client-facing aggregation.
Provides queries, mutations, and subscriptions for steam system optimization.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID, uuid4

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info
from strawberry.permission import BasePermission

from .api_auth import (
    Permission,
    SteamSystemUser,
    get_current_user,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GraphQL Enums (mirrors Pydantic enums)
# =============================================================================

@strawberry.enum
class SteamPhaseEnum(Enum):
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    TWO_PHASE = "two_phase"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"
    SUPERCRITICAL = "supercritical"


@strawberry.enum
class SteamRegionEnum(Enum):
    REGION_1 = "region_1"
    REGION_2 = "region_2"
    REGION_3 = "region_3"
    REGION_4 = "region_4"
    REGION_5 = "region_5"


@strawberry.enum
class TrapConditionEnum(Enum):
    GOOD = "good"
    LEAKING = "leaking"
    BLOCKED = "blocked"
    BLOW_THROUGH = "blow_through"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@strawberry.enum
class TrapTypeEnum(Enum):
    THERMOSTATIC = "thermostatic"
    THERMODYNAMIC = "thermodynamic"
    MECHANICAL = "mechanical"
    INVERTED_BUCKET = "inverted_bucket"
    FLOAT_THERMOSTATIC = "float_thermostatic"


@strawberry.enum
class OptimizationTypeEnum(Enum):
    DESUPERHEATER = "desuperheater"
    CONDENSATE_RECOVERY = "condensate_recovery"
    NETWORK = "network"
    PRESSURE_REDUCTION = "pressure_reduction"
    HEAT_RECOVERY = "heat_recovery"


@strawberry.enum
class RecommendationPriorityEnum(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@strawberry.enum
class RecommendationStatusEnum(Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    EXPIRED = "expired"


@strawberry.enum
class AlarmSeverityEnum(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@strawberry.enum
class KPICategoryEnum(Enum):
    ENERGY = "energy"
    EFFICIENCY = "efficiency"
    COST = "cost"
    EMISSIONS = "emissions"
    RELIABILITY = "reliability"
    SAFETY = "safety"


# =============================================================================
# GraphQL Types
# =============================================================================

@strawberry.type
class SteamState:
    """Complete thermodynamic state of steam."""
    pressure_kpa: float
    temperature_c: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    density_kg_m3: float
    quality: Optional[float] = None
    phase: SteamPhaseEnum = SteamPhaseEnum.SUPERHEATED_VAPOR
    region: SteamRegionEnum = SteamRegionEnum.REGION_2
    cp_kj_kg_k: Optional[float] = None
    cv_kj_kg_k: Optional[float] = None
    speed_of_sound_m_s: Optional[float] = None


@strawberry.type
class SteamProperties:
    """Steam properties query response."""
    request_id: strawberry.ID
    success: bool
    steam_state: Optional[SteamState] = None
    computation_time_ms: float = 0.0
    error_message: Optional[str] = None


@strawberry.type
class EnthalpyBalance:
    """Enthalpy balance calculation result."""
    equipment_id: str
    equipment_name: str
    total_inlet_enthalpy_kw: float
    total_outlet_enthalpy_kw: float
    enthalpy_imbalance_kw: float
    enthalpy_imbalance_percent: float
    balance_closed: bool
    data_quality_score: float
    timestamp: datetime


@strawberry.type
class TrapStatus:
    """Steam trap status."""
    trap_id: strawberry.ID
    trap_name: str
    trap_type: TrapTypeEnum
    condition: TrapConditionEnum
    condition_confidence: float
    location: str
    inlet_pressure_kpa: Optional[float] = None
    outlet_pressure_kpa: Optional[float] = None
    inlet_temperature_c: Optional[float] = None
    outlet_temperature_c: Optional[float] = None
    estimated_steam_loss_kg_h: float = 0.0
    estimated_energy_loss_kw: float = 0.0
    estimated_annual_cost_loss_usd: float = 0.0
    last_inspection_date: Optional[datetime] = None


@strawberry.type
class TrapDiagnostics:
    """Trap diagnostics with failure prediction."""
    trap_id: strawberry.ID
    status: TrapStatus
    failure_probability_30d: float
    failure_probability_90d: float
    predicted_failure_mode: TrapConditionEnum
    risk_score: float
    risk_factors: List[str]
    recommended_action: str
    priority: RecommendationPriorityEnum
    model_confidence: float
    anomalies_detected: List[str]


@strawberry.type
class FeatureContribution:
    """Feature contribution for explainability."""
    feature_name: str
    feature_value: str  # Stringified for GraphQL
    contribution_score: float
    contribution_direction: str
    explanation: Optional[str] = None


@strawberry.type
class Recommendation:
    """Optimization recommendation."""
    recommendation_id: strawberry.ID
    recommendation_type: OptimizationTypeEnum
    priority: RecommendationPriorityEnum
    status: RecommendationStatusEnum
    title: str
    description: str
    rationale: str
    estimated_energy_savings_kw: Optional[float] = None
    estimated_cost_savings_usd_year: Optional[float] = None
    estimated_emissions_reduction_kg_co2_year: Optional[float] = None
    estimated_payback_months: Optional[float] = None
    affected_equipment: List[str]
    confidence_score: float
    created_at: datetime
    valid_until: Optional[datetime] = None


@strawberry.type
class Explainability:
    """Explainability payload for recommendations."""
    recommendation_id: strawberry.ID
    shap_contributions: List[FeatureContribution]
    lime_contributions: List[FeatureContribution]
    plain_english_explanation: str
    technical_explanation: Optional[str] = None
    key_drivers: List[str]
    counterfactual_changes: Optional[List[str]] = None


@strawberry.type
class CausalFactor:
    """Causal factor from root cause analysis."""
    factor_id: strawberry.ID
    factor_name: str
    factor_description: str
    causal_strength: float
    confidence: float
    is_root_cause: bool
    is_contributing_factor: bool
    supporting_evidence: List[str]
    related_variables: List[str]


@strawberry.type
class CausalAnalysis:
    """Root cause analysis result."""
    analysis_id: strawberry.ID
    target_event: str
    event_timestamp: datetime
    root_causes: List[CausalFactor]
    contributing_factors: List[CausalFactor]
    causal_chain: List[str]
    executive_summary: str
    recommended_actions: List[str]
    model_confidence: float
    counterfactual_scenarios: Optional[List[str]] = None


@strawberry.type
class KPIValue:
    """Single KPI measurement."""
    kpi_id: strawberry.ID
    kpi_name: str
    category: KPICategoryEnum
    current_value: float
    target_value: Optional[float] = None
    unit: str
    trend: Optional[str] = None
    trend_percent: Optional[float] = None
    is_on_target: Optional[bool] = None
    measurement_timestamp: datetime


@strawberry.type
class KPIDashboard:
    """KPI dashboard aggregation."""
    period_start: datetime
    period_end: datetime
    energy_kpis: List[KPIValue]
    efficiency_kpis: List[KPIValue]
    cost_kpis: List[KPIValue]
    emissions_kpis: List[KPIValue]
    reliability_kpis: List[KPIValue]
    overall_performance_score: float
    kpis_on_target: int
    kpis_off_target: int


@strawberry.type
class EnergyMetrics:
    """Energy consumption metrics."""
    total_steam_consumption_kg_h: float
    total_steam_generation_kg_h: float
    total_energy_consumption_mw: float
    boiler_efficiency_percent: Optional[float] = None
    system_efficiency_percent: Optional[float] = None
    condensate_recovery_percent: float
    energy_intensity_mj_per_unit: Optional[float] = None


@strawberry.type
class EmissionsMetrics:
    """Emissions metrics."""
    total_co2_emissions_kg_h: float
    total_nox_emissions_kg_h: float = 0.0
    total_sox_emissions_kg_h: float = 0.0
    carbon_intensity_kg_co2_per_mwh: float
    avoided_emissions_kg_co2_h: Optional[float] = None


@strawberry.type
class ClimateImpact:
    """Climate and energy impact summary."""
    period_start: datetime
    period_end: datetime
    energy_metrics: EnergyMetrics
    emissions_metrics: EmissionsMetrics
    annual_emissions_target_tonnes_co2: Optional[float] = None
    ytd_emissions_tonnes_co2: Optional[float] = None
    on_track_for_target: Optional[bool] = None
    reporting_standard: str = "GHG Protocol"


@strawberry.type
class AlarmEvent:
    """Real-time alarm event."""
    alarm_id: strawberry.ID
    alarm_code: str
    name: str
    description: str
    severity: AlarmSeverityEnum
    source_equipment: str
    triggered_at: datetime
    measured_value: Optional[float] = None
    threshold_value: Optional[float] = None
    unit: Optional[str] = None
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None


@strawberry.type
class SteamStateUpdate:
    """Real-time steam state update for subscriptions."""
    equipment_id: str
    equipment_name: str
    steam_state: SteamState
    timestamp: datetime
    data_quality: float


@strawberry.type
class RecommendationAlert:
    """New recommendation alert for subscriptions."""
    recommendation: Recommendation
    is_new: bool = True
    notification_timestamp: datetime


# =============================================================================
# GraphQL Input Types
# =============================================================================

@strawberry.input
class SteamPropertiesInput:
    """Input for steam properties computation."""
    pressure_kpa: Optional[float] = None
    temperature_c: Optional[float] = None
    specific_enthalpy_kj_kg: Optional[float] = None
    specific_entropy_kj_kg_k: Optional[float] = None
    quality: Optional[float] = None
    include_transport_properties: bool = False


@strawberry.input
class TrapDiagnosticsInput:
    """Input for trap diagnostics."""
    trap_id: str
    inlet_pressure_kpa: Optional[float] = None
    outlet_pressure_kpa: Optional[float] = None
    inlet_temperature_c: Optional[float] = None
    outlet_temperature_c: Optional[float] = None
    include_prediction: bool = True


@strawberry.input
class OptimizationInput:
    """Input for requesting optimization."""
    optimization_type: OptimizationTypeEnum
    equipment_ids: List[str]
    target_metric: Optional[str] = None
    constraints: Optional[str] = None  # JSON string
    horizon_hours: int = 24


@strawberry.input
class RecommendationAckInput:
    """Input for acknowledging a recommendation."""
    recommendation_id: strawberry.ID
    acknowledgement_note: Optional[str] = None


@strawberry.input
class ConfigUpdateInput:
    """Input for configuration updates."""
    config_key: str
    config_value: str
    effective_from: Optional[datetime] = None


@strawberry.input
class TimeRangeInput:
    """Time range filter input."""
    start_time: datetime
    end_time: datetime


@strawberry.input
class RCAInput:
    """Input for root cause analysis."""
    target_event: str
    event_timestamp: datetime
    affected_equipment: List[str]
    lookback_hours: int = 24
    include_counterfactuals: bool = True


# =============================================================================
# Authorization Permissions
# =============================================================================

class IsAuthenticated(BasePermission):
    """Check if user is authenticated."""
    message = "User is not authenticated"

    async def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        request = info.context.get("request")
        if not request:
            return False
        try:
            await get_current_user(request)
            return True
        except Exception:
            return False


class HasPermission(BasePermission):
    """Check if user has specific permission."""
    message = "User lacks required permission"

    def __init__(self, permission: Permission):
        self.permission = permission

    async def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        request = info.context.get("request")
        if not request:
            return False
        try:
            user = await get_current_user(request)
            return user.has_permission(self.permission)
        except Exception:
            return False


# =============================================================================
# Data Store (Mock implementation)
# =============================================================================

class GraphQLDataStore:
    """
    Mock data store for GraphQL resolvers.
    In production, this would interface with actual databases and services.
    """

    def __init__(self):
        self._recommendations: Dict[str, Recommendation] = {}
        self._traps: Dict[str, TrapStatus] = {}
        self._kpis: Dict[str, KPIValue] = {}
        self._alarms: Dict[str, AlarmEvent] = {}

        # Event queues for subscriptions
        self._state_update_queue: asyncio.Queue = asyncio.Queue()
        self._recommendation_queue: asyncio.Queue = asyncio.Queue()
        self._alarm_queue: asyncio.Queue = asyncio.Queue()

        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data."""
        now = datetime.utcnow()

        # Sample recommendations
        rec_id = str(uuid4())
        self._recommendations[rec_id] = Recommendation(
            recommendation_id=strawberry.ID(rec_id),
            recommendation_type=OptimizationTypeEnum.CONDENSATE_RECOVERY,
            priority=RecommendationPriorityEnum.HIGH,
            status=RecommendationStatusEnum.PENDING,
            title="Increase condensate recovery to 85%",
            description="Current recovery rate of 65% leaves significant savings on table.",
            rationale="Higher recovery reduces makeup water and fuel costs.",
            estimated_energy_savings_kw=150.0,
            estimated_cost_savings_usd_year=75000.0,
            estimated_emissions_reduction_kg_co2_year=50000.0,
            affected_equipment=["condensate_system_1"],
            confidence_score=0.92,
            created_at=now,
        )

        # Sample traps
        for i in range(5):
            trap_id = f"trap_{i+1:03d}"
            self._traps[trap_id] = TrapStatus(
                trap_id=strawberry.ID(trap_id),
                trap_name=f"Steam Trap {i+1}",
                trap_type=TrapTypeEnum.THERMODYNAMIC,
                condition=TrapConditionEnum.GOOD if i < 4 else TrapConditionEnum.LEAKING,
                condition_confidence=0.9,
                location=f"Building A, Level {i+1}",
                inlet_pressure_kpa=800.0,
                outlet_pressure_kpa=101.325,
                inlet_temperature_c=175.0,
                outlet_temperature_c=100.0,
                estimated_steam_loss_kg_h=0.0 if i < 4 else 15.0,
                estimated_energy_loss_kw=0.0 if i < 4 else 37.5,
            )

        # Sample KPIs
        kpi_data = [
            ("kpi_001", "Steam Consumption", KPICategoryEnum.ENERGY, 15000, 14000, "kg/h"),
            ("kpi_002", "Boiler Efficiency", KPICategoryEnum.EFFICIENCY, 88.5, 90.0, "%"),
            ("kpi_003", "Condensate Recovery", KPICategoryEnum.EFFICIENCY, 72.0, 85.0, "%"),
            ("kpi_004", "CO2 Emissions", KPICategoryEnum.EMISSIONS, 2500, 2200, "kg/h"),
        ]

        for kpi_id, name, category, value, target, unit in kpi_data:
            self._kpis[kpi_id] = KPIValue(
                kpi_id=strawberry.ID(kpi_id),
                kpi_name=name,
                category=category,
                current_value=value,
                target_value=target,
                unit=unit,
                trend="stable",
                is_on_target=value <= target if target else None,
                measurement_timestamp=now,
            )

    async def get_steam_properties(self, input: SteamPropertiesInput) -> SteamProperties:
        """Compute steam properties."""
        P = input.pressure_kpa or 1000.0
        T = input.temperature_c or 200.0

        steam_state = SteamState(
            pressure_kpa=P,
            temperature_c=T,
            specific_enthalpy_kj_kg=2827.9,
            specific_entropy_kj_kg_k=6.694,
            specific_volume_m3_kg=0.206,
            density_kg_m3=4.85,
            phase=SteamPhaseEnum.SUPERHEATED_VAPOR,
            region=SteamRegionEnum.REGION_2,
        )

        return SteamProperties(
            request_id=strawberry.ID(str(uuid4())),
            success=True,
            steam_state=steam_state,
            computation_time_ms=5.0,
        )

    async def get_enthalpy_balance(
        self,
        equipment_id: str,
        time_range: Optional[TimeRangeInput] = None,
    ) -> EnthalpyBalance:
        """Get enthalpy balance for equipment."""
        return EnthalpyBalance(
            equipment_id=equipment_id,
            equipment_name=f"Equipment {equipment_id}",
            total_inlet_enthalpy_kw=5000.0,
            total_outlet_enthalpy_kw=4950.0,
            enthalpy_imbalance_kw=50.0,
            enthalpy_imbalance_percent=1.0,
            balance_closed=True,
            data_quality_score=95.0,
            timestamp=datetime.utcnow(),
        )

    async def get_recommendations(
        self,
        status: Optional[RecommendationStatusEnum] = None,
        priority: Optional[RecommendationPriorityEnum] = None,
        limit: int = 20,
    ) -> List[Recommendation]:
        """Get recommendations with optional filtering."""
        recs = list(self._recommendations.values())

        if status:
            recs = [r for r in recs if r.status == status]
        if priority:
            recs = [r for r in recs if r.priority == priority]

        return recs[:limit]

    async def get_recommendation(self, recommendation_id: str) -> Optional[Recommendation]:
        """Get a specific recommendation."""
        return self._recommendations.get(recommendation_id)

    async def get_trap_diagnostics(self, trap_id: str) -> Optional[TrapDiagnostics]:
        """Get diagnostics for a trap."""
        trap = self._traps.get(trap_id)
        if not trap:
            return None

        return TrapDiagnostics(
            trap_id=trap.trap_id,
            status=trap,
            failure_probability_30d=0.05 if trap.condition == TrapConditionEnum.GOOD else 0.4,
            failure_probability_90d=0.12 if trap.condition == TrapConditionEnum.GOOD else 0.7,
            predicted_failure_mode=TrapConditionEnum.LEAKING,
            risk_score=15.0 if trap.condition == TrapConditionEnum.GOOD else 65.0,
            risk_factors=["Age > 3 years"] if trap.condition == TrapConditionEnum.GOOD else ["Current leak detected"],
            recommended_action="Continue monitoring" if trap.condition == TrapConditionEnum.GOOD else "Replace immediately",
            priority=RecommendationPriorityEnum.LOW if trap.condition == TrapConditionEnum.GOOD else RecommendationPriorityEnum.HIGH,
            model_confidence=0.88,
            anomalies_detected=[] if trap.condition == TrapConditionEnum.GOOD else ["Steam leak detected"],
        )

    async def get_kpi_dashboard(self, time_range: Optional[TimeRangeInput] = None) -> KPIDashboard:
        """Get KPI dashboard."""
        now = datetime.utcnow()
        kpis = list(self._kpis.values())

        return KPIDashboard(
            period_start=time_range.start_time if time_range else now - timedelta(hours=24),
            period_end=time_range.end_time if time_range else now,
            energy_kpis=[k for k in kpis if k.category == KPICategoryEnum.ENERGY],
            efficiency_kpis=[k for k in kpis if k.category == KPICategoryEnum.EFFICIENCY],
            cost_kpis=[k for k in kpis if k.category == KPICategoryEnum.COST],
            emissions_kpis=[k for k in kpis if k.category == KPICategoryEnum.EMISSIONS],
            reliability_kpis=[k for k in kpis if k.category == KPICategoryEnum.RELIABILITY],
            overall_performance_score=78.5,
            kpis_on_target=1,
            kpis_off_target=3,
        )

    async def get_climate_impact(self, time_range: Optional[TimeRangeInput] = None) -> ClimateImpact:
        """Get climate impact metrics."""
        now = datetime.utcnow()

        return ClimateImpact(
            period_start=time_range.start_time if time_range else now - timedelta(days=30),
            period_end=time_range.end_time if time_range else now,
            energy_metrics=EnergyMetrics(
                total_steam_consumption_kg_h=15000,
                total_steam_generation_kg_h=15500,
                total_energy_consumption_mw=12.5,
                boiler_efficiency_percent=88.5,
                system_efficiency_percent=82.0,
                condensate_recovery_percent=72.0,
            ),
            emissions_metrics=EmissionsMetrics(
                total_co2_emissions_kg_h=2500,
                carbon_intensity_kg_co2_per_mwh=200,
                avoided_emissions_kg_co2_h=150,
            ),
            annual_emissions_target_tonnes_co2=20000,
            ytd_emissions_tonnes_co2=18000,
            on_track_for_target=True,
        )

    async def get_explainability(self, recommendation_id: str) -> Optional[Explainability]:
        """Get explainability for a recommendation."""
        rec = self._recommendations.get(recommendation_id)
        if not rec:
            return None

        return Explainability(
            recommendation_id=rec.recommendation_id,
            shap_contributions=[
                FeatureContribution(
                    feature_name="condensate_recovery_rate",
                    feature_value="0.65",
                    contribution_score=0.35,
                    contribution_direction="positive",
                    explanation="Low recovery rate is primary driver",
                ),
            ],
            lime_contributions=[],
            plain_english_explanation=(
                "This recommendation is driven by the current low condensate recovery rate. "
                "Increasing recovery would significantly reduce water and energy costs."
            ),
            key_drivers=["Low condensate recovery (65%)", "High makeup water cost"],
        )

    async def get_causal_analysis(
        self,
        target_event: str,
        event_timestamp: datetime,
        lookback_hours: int = 24,
    ) -> CausalAnalysis:
        """Perform root cause analysis."""
        return CausalAnalysis(
            analysis_id=strawberry.ID(str(uuid4())),
            target_event=target_event,
            event_timestamp=event_timestamp,
            root_causes=[
                CausalFactor(
                    factor_id=strawberry.ID(str(uuid4())),
                    factor_name="Steam trap failure",
                    factor_description="Upstream steam trap failed in open position",
                    causal_strength=0.85,
                    confidence=0.82,
                    is_root_cause=True,
                    is_contributing_factor=False,
                    supporting_evidence=["Temperature spike detected 15 min before"],
                    related_variables=["trap_temperature", "steam_loss"],
                ),
            ],
            contributing_factors=[
                CausalFactor(
                    factor_id=strawberry.ID(str(uuid4())),
                    factor_name="High system load",
                    factor_description="System at 95% capacity",
                    causal_strength=0.45,
                    confidence=0.75,
                    is_root_cause=False,
                    is_contributing_factor=True,
                    supporting_evidence=["Steam demand 15% above normal"],
                    related_variables=["steam_demand"],
                ),
            ],
            causal_chain=["trap_failure", "steam_loss", "pressure_drop"],
            executive_summary="Event caused by steam trap failure with high load contributing to severity.",
            recommended_actions=["Replace failed trap", "Inspect adjacent traps"],
            model_confidence=0.82,
        )

    async def acknowledge_recommendation(
        self,
        recommendation_id: str,
        note: Optional[str] = None,
    ) -> Recommendation:
        """Acknowledge a recommendation."""
        rec = self._recommendations.get(recommendation_id)
        if rec:
            # Create updated recommendation
            self._recommendations[recommendation_id] = Recommendation(
                recommendation_id=rec.recommendation_id,
                recommendation_type=rec.recommendation_type,
                priority=rec.priority,
                status=RecommendationStatusEnum.ACKNOWLEDGED,
                title=rec.title,
                description=rec.description,
                rationale=rec.rationale,
                estimated_energy_savings_kw=rec.estimated_energy_savings_kw,
                estimated_cost_savings_usd_year=rec.estimated_cost_savings_usd_year,
                estimated_emissions_reduction_kg_co2_year=rec.estimated_emissions_reduction_kg_co2_year,
                affected_equipment=rec.affected_equipment,
                confidence_score=rec.confidence_score,
                created_at=rec.created_at,
            )
            return self._recommendations[recommendation_id]
        return None

    async def request_optimization(self, input: OptimizationInput) -> Recommendation:
        """Request a new optimization."""
        rec_id = str(uuid4())
        rec = Recommendation(
            recommendation_id=strawberry.ID(rec_id),
            recommendation_type=input.optimization_type,
            priority=RecommendationPriorityEnum.MEDIUM,
            status=RecommendationStatusEnum.PENDING,
            title=f"Optimization request for {input.optimization_type.value}",
            description=f"Optimization requested for {', '.join(input.equipment_ids)}",
            rationale="User-initiated optimization request",
            affected_equipment=input.equipment_ids,
            confidence_score=0.0,
            created_at=datetime.utcnow(),
        )
        self._recommendations[rec_id] = rec

        # Notify subscribers
        await self._recommendation_queue.put(RecommendationAlert(
            recommendation=rec,
            is_new=True,
            notification_timestamp=datetime.utcnow(),
        ))

        return rec


# Global data store instance
data_store = GraphQLDataStore()


# =============================================================================
# GraphQL Query
# =============================================================================

@strawberry.type
class Query:
    """GraphQL Query root."""

    @strawberry.field(description="Get steam state and properties")
    async def steam_state(
        self,
        info: Info,
        equipment_id: str,
    ) -> Optional[SteamState]:
        """Get current steam state for equipment."""
        return SteamState(
            pressure_kpa=1000.0,
            temperature_c=200.0,
            specific_enthalpy_kj_kg=2827.9,
            specific_entropy_kj_kg_k=6.694,
            specific_volume_m3_kg=0.206,
            density_kg_m3=4.85,
            phase=SteamPhaseEnum.SUPERHEATED_VAPOR,
            region=SteamRegionEnum.REGION_2,
        )

    @strawberry.field(description="Compute steam properties from inputs")
    async def steam_properties(
        self,
        info: Info,
        input: SteamPropertiesInput,
    ) -> SteamProperties:
        """Compute steam properties from given inputs."""
        return await data_store.get_steam_properties(input)

    @strawberry.field(description="Get enthalpy balance for equipment")
    async def enthalpy_balance(
        self,
        info: Info,
        equipment_id: str,
        time_range: Optional[TimeRangeInput] = None,
    ) -> EnthalpyBalance:
        """Get enthalpy balance calculations."""
        return await data_store.get_enthalpy_balance(equipment_id, time_range)

    @strawberry.field(description="Get optimization recommendations")
    async def recommendations(
        self,
        info: Info,
        status: Optional[RecommendationStatusEnum] = None,
        priority: Optional[RecommendationPriorityEnum] = None,
        limit: int = 20,
    ) -> List[Recommendation]:
        """Get list of optimization recommendations."""
        return await data_store.get_recommendations(status, priority, limit)

    @strawberry.field(description="Get trap diagnostics")
    async def trap_diagnostics(
        self,
        info: Info,
        trap_id: str,
    ) -> Optional[TrapDiagnostics]:
        """Get diagnostics for a specific steam trap."""
        return await data_store.get_trap_diagnostics(trap_id)

    @strawberry.field(description="Get KPI dashboard")
    async def kpi_dashboard(
        self,
        info: Info,
        time_range: Optional[TimeRangeInput] = None,
    ) -> KPIDashboard:
        """Get KPI dashboard with all metrics."""
        return await data_store.get_kpi_dashboard(time_range)

    @strawberry.field(description="Get climate and energy impact metrics")
    async def climate_impact(
        self,
        info: Info,
        time_range: Optional[TimeRangeInput] = None,
    ) -> ClimateImpact:
        """Get climate impact summary."""
        return await data_store.get_climate_impact(time_range)

    @strawberry.field(description="Get explainability for a recommendation")
    async def explainability(
        self,
        info: Info,
        recommendation_id: strawberry.ID,
    ) -> Optional[Explainability]:
        """Get SHAP/LIME explainability for a recommendation."""
        return await data_store.get_explainability(str(recommendation_id))

    @strawberry.field(description="Perform causal analysis")
    async def causal_analysis(
        self,
        info: Info,
        input: RCAInput,
    ) -> CausalAnalysis:
        """Perform root cause analysis for an event."""
        return await data_store.get_causal_analysis(
            input.target_event,
            input.event_timestamp,
            input.lookback_hours,
        )


# =============================================================================
# GraphQL Mutation
# =============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation root."""

    @strawberry.mutation(description="Request optimization analysis")
    async def request_optimization(
        self,
        info: Info,
        input: OptimizationInput,
    ) -> Recommendation:
        """Request a new optimization analysis."""
        return await data_store.request_optimization(input)

    @strawberry.mutation(description="Acknowledge a recommendation")
    async def acknowledge_recommendation(
        self,
        info: Info,
        input: RecommendationAckInput,
    ) -> Optional[Recommendation]:
        """Acknowledge a recommendation."""
        return await data_store.acknowledge_recommendation(
            str(input.recommendation_id),
            input.acknowledgement_note,
        )

    @strawberry.mutation(description="Update configuration")
    async def update_configuration(
        self,
        info: Info,
        input: ConfigUpdateInput,
    ) -> bool:
        """Update system configuration."""
        logger.info(f"Configuration update: {input.config_key} = {input.config_value}")
        return True


# =============================================================================
# GraphQL Subscription
# =============================================================================

@strawberry.type
class Subscription:
    """GraphQL Subscription root."""

    @strawberry.subscription(description="Subscribe to steam state updates")
    async def steam_state_updates(
        self,
        info: Info,
        equipment_ids: Optional[List[str]] = None,
        interval_seconds: int = 5,
    ) -> AsyncGenerator[SteamStateUpdate, None]:
        """Stream real-time steam state updates."""
        while True:
            await asyncio.sleep(interval_seconds)

            # Generate mock update
            import random
            equipment_id = equipment_ids[0] if equipment_ids else "equipment_001"

            yield SteamStateUpdate(
                equipment_id=equipment_id,
                equipment_name=f"Equipment {equipment_id}",
                steam_state=SteamState(
                    pressure_kpa=1000.0 + random.uniform(-10, 10),
                    temperature_c=200.0 + random.uniform(-2, 2),
                    specific_enthalpy_kj_kg=2827.9 + random.uniform(-5, 5),
                    specific_entropy_kj_kg_k=6.694,
                    specific_volume_m3_kg=0.206,
                    density_kg_m3=4.85,
                    phase=SteamPhaseEnum.SUPERHEATED_VAPOR,
                    region=SteamRegionEnum.REGION_2,
                ),
                timestamp=datetime.utcnow(),
                data_quality=0.98,
            )

    @strawberry.subscription(description="Subscribe to recommendation alerts")
    async def recommendation_alerts(
        self,
        info: Info,
        priority_filter: Optional[List[RecommendationPriorityEnum]] = None,
    ) -> AsyncGenerator[RecommendationAlert, None]:
        """Stream new recommendation alerts."""
        while True:
            try:
                alert = await asyncio.wait_for(
                    data_store._recommendation_queue.get(),
                    timeout=60.0,
                )

                # Apply filter
                if priority_filter and alert.recommendation.priority not in priority_filter:
                    continue

                yield alert

            except asyncio.TimeoutError:
                # Send heartbeat
                continue

    @strawberry.subscription(description="Subscribe to alarm notifications")
    async def alarm_notifications(
        self,
        info: Info,
        severity_filter: Optional[List[AlarmSeverityEnum]] = None,
    ) -> AsyncGenerator[AlarmEvent, None]:
        """Stream alarm notifications."""
        while True:
            try:
                alarm = await asyncio.wait_for(
                    data_store._alarm_queue.get(),
                    timeout=60.0,
                )

                # Apply filter
                if severity_filter and alarm.severity not in severity_filter:
                    continue

                yield alarm

            except asyncio.TimeoutError:
                # Send heartbeat - generate mock alarm occasionally
                import random
                if random.random() < 0.1:  # 10% chance
                    yield AlarmEvent(
                        alarm_id=strawberry.ID(str(uuid4())),
                        alarm_code="ALM_001",
                        name="High Temperature",
                        description="Temperature exceeded threshold",
                        severity=AlarmSeverityEnum.MEDIUM,
                        source_equipment="equipment_001",
                        triggered_at=datetime.utcnow(),
                        measured_value=205.0,
                        threshold_value=200.0,
                        unit="C",
                    )


# =============================================================================
# GraphQL Schema and Router
# =============================================================================

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)


async def get_context(request):
    """Build GraphQL context with request."""
    return {"request": request}


graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    graphiql=True,
)
