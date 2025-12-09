"""
BurnerMaintenancePredictorAgent - Predictive maintenance for industrial burners

This module implements the BurnerMaintenancePredictorAgent (GL-021 BURNERSENTRY)
for predictive maintenance of industrial burners using Weibull reliability
analysis, flame quality assessment, and deterministic health scoring.

The agent follows GreenLang's zero-hallucination principle by using only
deterministic calculations from reliability engineering standards - no ML/LLM
in the calculation path.

Example:
    >>> config = AgentConfig(agent_id="GL-021")
    >>> agent = BurnerMaintenancePredictorAgent(config)
    >>> result = agent.run(input_data)
    >>> assert result.validation_status == "PASS"
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any
import hashlib
import logging

from pydantic import BaseModel, Field, validator, root_validator

from .calculators.weibull import (
    weibull_reliability,
    weibull_failure_rate,
    weibull_mean_life,
    remaining_useful_life,
    calculate_failure_probability,
)
from .calculators.flame_quality import (
    calculate_flame_quality_score,
    calculate_combustion_efficiency,
    detect_flame_anomalies,
)
from .calculators.health_score import (
    calculate_overall_health,
    calculate_degradation_rate,
    determine_maintenance_priority,
)
from .calculators.maintenance import (
    generate_maintenance_recommendations,
    calculate_next_maintenance_date,
    should_replace_burner,
)

logger = logging.getLogger(__name__)


class FuelType(str, Enum):
    """Supported fuel types for combustion analysis."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"


class MaintenancePriority(str, Enum):
    """Maintenance priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class WeibullParameters(BaseModel):
    """Weibull distribution parameters for failure modeling."""

    beta: float = Field(
        ...,
        gt=0,
        description="Shape parameter (beta > 1 indicates wear-out failures)"
    )
    eta: float = Field(
        ...,
        gt=0,
        description="Scale parameter (characteristic life in hours)"
    )

    @validator('beta')
    def validate_beta_range(cls, v: float) -> float:
        """Validate beta is in reasonable range for burner components."""
        if v < 0.5 or v > 10:
            logger.warning(f"Beta value {v} is outside typical range [0.5, 10]")
        return v

    @validator('eta')
    def validate_eta_range(cls, v: float) -> float:
        """Validate eta is in reasonable range for industrial equipment."""
        if v < 1000 or v > 200000:
            logger.warning(f"Eta value {v} hours is outside typical range")
        return v


class BurnerComponentHealth(BaseModel):
    """Health metrics for individual burner components."""

    component_name: str = Field(..., description="Name of the component")
    health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Health score 0-100"
    )
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    cycles_count: int = Field(
        default=0,
        ge=0,
        description="Number of on/off cycles"
    )
    last_inspection_date: Optional[date] = Field(
        default=None,
        description="Date of last inspection"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class FlameQualityMetrics(BaseModel):
    """Flame quality measurement data."""

    flame_temperature_c: float = Field(
        ...,
        ge=0,
        le=3000,
        description="Flame temperature in Celsius"
    )
    stability_index: float = Field(
        ...,
        ge=0,
        le=1,
        description="Flame stability index 0-1 (1 = perfectly stable)"
    )
    o2_percent: float = Field(
        ...,
        ge=0,
        le=21,
        description="Oxygen percentage in flue gas"
    )
    co_ppm: float = Field(
        ...,
        ge=0,
        description="Carbon monoxide in parts per million"
    )
    nox_ppm: float = Field(
        ...,
        ge=0,
        description="NOx in parts per million"
    )

    @root_validator(skip_on_failure=True)
    def validate_combustion_chemistry(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate combustion measurements are physically consistent."""
        o2 = values.get('o2_percent', 0)
        co = values.get('co_ppm', 0)

        # High CO with low O2 indicates incomplete combustion
        if o2 < 1 and co > 1000:
            logger.warning("Very low O2 with high CO indicates severe incomplete combustion")

        return values


class MaintenanceRecommendation(BaseModel):
    """Individual maintenance recommendation."""

    action: str = Field(..., description="Recommended maintenance action")
    priority: MaintenancePriority = Field(..., description="Priority level")
    due_date: Optional[date] = Field(default=None, description="Recommended due date")
    estimated_hours: float = Field(
        default=0,
        ge=0,
        description="Estimated labor hours"
    )
    component: Optional[str] = Field(
        default=None,
        description="Affected component"
    )
    reason: str = Field(..., description="Reason for recommendation")


class BurnerInput(BaseModel):
    """Input data model for BurnerMaintenancePredictorAgent."""

    burner_id: str = Field(..., min_length=1, description="Unique burner identifier")
    burner_model: str = Field(..., description="Burner model/type")
    fuel_type: FuelType = Field(..., description="Type of fuel used")

    # Operating data
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        ...,
        gt=0,
        description="Design life in hours"
    )
    cycles_count: int = Field(
        default=0,
        ge=0,
        description="Total on/off cycles"
    )
    operating_hours_per_day: float = Field(
        default=24,
        gt=0,
        le=24,
        description="Average operating hours per day"
    )

    # Weibull parameters (from historical data or manufacturer)
    weibull_params: WeibullParameters = Field(
        ...,
        description="Weibull distribution parameters"
    )

    # Current flame quality measurements
    flame_metrics: FlameQualityMetrics = Field(
        ...,
        description="Current flame quality measurements"
    )

    # Component health data
    component_health: List[BurnerComponentHealth] = Field(
        default_factory=list,
        description="Health data for individual components"
    )

    # Historical health scores for degradation analysis
    health_history: List[float] = Field(
        default_factory=list,
        description="Historical health scores (most recent last)"
    )
    health_history_interval_hours: float = Field(
        default=1000,
        gt=0,
        description="Time interval between health history readings"
    )

    # Installation and maintenance history
    installation_date: date = Field(..., description="Burner installation date")
    last_maintenance_date: Optional[date] = Field(
        default=None,
        description="Date of last maintenance"
    )

    # Cost factors for replacement decision
    repair_cost_ratio: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Repair cost as ratio of replacement cost"
    )

    @validator('operating_hours')
    def validate_operating_hours(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate operating hours against installation date."""
        installation = values.get('installation_date')
        if installation:
            days_since_install = (date.today() - installation).days
            max_possible_hours = days_since_install * 24
            if v > max_possible_hours * 1.1:  # 10% tolerance
                logger.warning(
                    f"Operating hours {v} exceeds maximum possible {max_possible_hours}"
                )
        return v

    @validator('health_history')
    def validate_health_history(cls, v: List[float]) -> List[float]:
        """Validate health history values are in range."""
        for score in v:
            if score < 0 or score > 100:
                raise ValueError(f"Health score {score} must be between 0 and 100")
        return v


class BurnerOutput(BaseModel):
    """Output data model for BurnerMaintenancePredictorAgent."""

    burner_id: str = Field(..., description="Burner identifier from input")
    assessment_timestamp: datetime = Field(
        ...,
        description="Timestamp of assessment"
    )

    # Reliability metrics
    reliability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current reliability R(t)"
    )
    failure_rate: float = Field(
        ...,
        ge=0,
        description="Current failure rate h(t) per hour"
    )
    mttf_hours: float = Field(
        ...,
        ge=0,
        description="Mean time to failure in hours"
    )
    remaining_useful_life_hours: float = Field(
        ...,
        ge=0,
        description="Remaining useful life in hours"
    )
    failure_probability_30d: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of failure in next 30 days"
    )

    # Quality and health metrics
    flame_quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Flame quality score 0-100"
    )
    combustion_efficiency: float = Field(
        ...,
        ge=0,
        le=100,
        description="Combustion efficiency percentage"
    )
    overall_health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall health score 0-100"
    )
    degradation_rate: float = Field(
        ...,
        description="Health degradation rate per 1000 hours"
    )

    # Anomaly detection
    flame_anomalies: List[str] = Field(
        default_factory=list,
        description="Detected flame anomalies"
    )

    # Maintenance recommendations
    maintenance_priority: MaintenancePriority = Field(
        ...,
        description="Overall maintenance priority"
    )
    recommendations: List[MaintenanceRecommendation] = Field(
        ...,
        description="List of maintenance recommendations"
    )
    next_maintenance_date: Optional[date] = Field(
        default=None,
        description="Recommended next maintenance date"
    )
    should_replace: bool = Field(
        ...,
        description="Recommendation to replace vs repair"
    )
    replacement_reason: Optional[str] = Field(
        default=None,
        description="Reason for replacement recommendation"
    )

    # Provenance and audit
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing duration in milliseconds"
    )
    validation_status: str = Field(
        ...,
        pattern="^(PASS|FAIL)$",
        description="PASS or FAIL"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages if any"
    )


class AgentConfig(BaseModel):
    """Configuration for BurnerMaintenancePredictorAgent."""

    agent_id: str = Field(default="GL-021", description="Agent identifier")
    agent_name: str = Field(default="BURNERSENTRY", description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")
    reliability_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Reliability threshold for RUL calculation"
    )
    critical_health_threshold: float = Field(
        default=30,
        ge=0,
        le=100,
        description="Health score below which maintenance is critical"
    )
    high_health_threshold: float = Field(
        default=50,
        ge=0,
        le=100,
        description="Health score below which maintenance is high priority"
    )


class BurnerMaintenancePredictorAgent:
    """
    BurnerMaintenancePredictorAgent implementation (GL-021 BURNERSENTRY).

    This agent performs predictive maintenance analysis for industrial burners
    using Weibull reliability analysis, flame quality assessment, and
    deterministic health scoring. It follows zero-hallucination principles
    by using only physics-based and statistical formulas.

    Attributes:
        config: Agent configuration
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
        version: Agent version string

    Example:
        >>> config = AgentConfig()
        >>> agent = BurnerMaintenancePredictorAgent(config)
        >>> input_data = BurnerInput(
        ...     burner_id="BRN-001",
        ...     burner_model="ACME-5000",
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     operating_hours=15000,
        ...     design_life_hours=50000,
        ...     weibull_params=WeibullParameters(beta=2.5, eta=40000),
        ...     flame_metrics=FlameQualityMetrics(
        ...         flame_temperature_c=1200,
        ...         stability_index=0.95,
        ...         o2_percent=3.5,
        ...         co_ppm=50,
        ...         nox_ppm=80
        ...     ),
        ...     installation_date=date(2020, 1, 15)
        ... )
        >>> result = agent.run(input_data)
        >>> assert result.validation_status == "PASS"
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize BurnerMaintenancePredictorAgent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self.agent_id = self.config.agent_id
        self.agent_name = self.config.agent_name
        self.version = self.config.version

        logger.info(
            f"Initialized {self.agent_name} agent v{self.version} (ID: {self.agent_id})"
        )

    def run(self, input_data: BurnerInput) -> BurnerOutput:
        """
        Execute burner maintenance prediction analysis.

        This is the main entry point for the agent. It performs:
        1. Weibull reliability analysis
        2. Flame quality assessment
        3. Overall health calculation
        4. Maintenance recommendation generation
        5. Replacement decision analysis

        Args:
            input_data: Validated burner input data

        Returns:
            BurnerOutput with complete analysis results and provenance

        Raises:
            ValueError: If input validation fails
            RuntimeError: If calculation fails
        """
        start_time = datetime.now()
        validation_errors: List[str] = []

        logger.info(f"Starting analysis for burner {input_data.burner_id}")

        try:
            # Step 1: Calculate Weibull reliability metrics
            reliability_metrics = self._calculate_reliability_metrics(
                input_data.operating_hours,
                input_data.weibull_params,
                input_data.operating_hours_per_day
            )
            logger.debug(f"Reliability: {reliability_metrics['reliability']:.4f}")

            # Step 2: Calculate flame quality metrics
            flame_quality = self._calculate_flame_metrics(
                input_data.flame_metrics,
                input_data.fuel_type
            )
            logger.debug(f"Flame quality score: {flame_quality['score']:.1f}")

            # Step 3: Detect flame anomalies
            anomalies = detect_flame_anomalies(
                input_data.flame_metrics.stability_index,
                input_data.flame_metrics.co_ppm,
                input_data.flame_metrics.flame_temperature_c
            )
            if anomalies:
                logger.warning(f"Flame anomalies detected: {anomalies}")

            # Step 4: Calculate overall health score
            health_metrics = self._calculate_health_metrics(
                input_data,
                flame_quality['score']
            )
            logger.debug(f"Overall health: {health_metrics['health_score']:.1f}")

            # Step 5: Determine maintenance priority
            priority = determine_maintenance_priority(
                health_metrics['health_score'],
                reliability_metrics['rul_hours'],
                reliability_metrics['failure_prob_30d']
            )
            logger.debug(f"Maintenance priority: {priority}")

            # Step 6: Generate maintenance recommendations
            recommendations = self._generate_recommendations(
                input_data,
                health_metrics,
                flame_quality,
                reliability_metrics,
                anomalies
            )

            # Step 7: Calculate next maintenance date
            next_maint_date = calculate_next_maintenance_date(
                date.today(),
                reliability_metrics['rul_hours'],
                input_data.operating_hours_per_day
            )

            # Step 8: Replacement decision
            replacement_decision = self._evaluate_replacement(
                input_data,
                health_metrics['health_score'],
                reliability_metrics['rul_hours']
            )

            # Step 9: Calculate provenance hash
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            provenance_hash = self._calculate_provenance_hash(
                input_data,
                reliability_metrics,
                flame_quality,
                health_metrics
            )

            # Step 10: Validate output
            validation_status = "PASS"
            if health_metrics['health_score'] < 0 or health_metrics['health_score'] > 100:
                validation_errors.append("Health score out of range")
                validation_status = "FAIL"

            # Build output
            output = BurnerOutput(
                burner_id=input_data.burner_id,
                assessment_timestamp=datetime.now(),
                reliability=reliability_metrics['reliability'],
                failure_rate=reliability_metrics['failure_rate'],
                mttf_hours=reliability_metrics['mttf'],
                remaining_useful_life_hours=reliability_metrics['rul_hours'],
                failure_probability_30d=reliability_metrics['failure_prob_30d'],
                flame_quality_score=flame_quality['score'],
                combustion_efficiency=flame_quality['efficiency'],
                overall_health_score=health_metrics['health_score'],
                degradation_rate=health_metrics['degradation_rate'],
                flame_anomalies=anomalies,
                maintenance_priority=MaintenancePriority(priority),
                recommendations=recommendations,
                next_maintenance_date=next_maint_date,
                should_replace=replacement_decision['should_replace'],
                replacement_reason=replacement_decision.get('reason'),
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
                validation_status=validation_status,
                validation_errors=validation_errors
            )

            logger.info(
                f"Completed analysis for {input_data.burner_id} in {processing_time_ms:.1f}ms"
            )

            return output

        except Exception as e:
            logger.error(f"Analysis failed for {input_data.burner_id}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Burner analysis failed: {str(e)}") from e

    def _calculate_reliability_metrics(
        self,
        operating_hours: float,
        weibull_params: WeibullParameters,
        hours_per_day: float
    ) -> Dict[str, float]:
        """
        Calculate Weibull-based reliability metrics.

        Uses deterministic Weibull distribution formulas:
        - R(t) = exp(-(t/eta)^beta)
        - h(t) = (beta/eta) * (t/eta)^(beta-1)
        - MTTF = eta * Gamma(1 + 1/beta)

        Args:
            operating_hours: Current operating hours
            weibull_params: Weibull distribution parameters
            hours_per_day: Average operating hours per day

        Returns:
            Dictionary with reliability, failure_rate, mttf, rul_hours, failure_prob_30d
        """
        beta = weibull_params.beta
        eta = weibull_params.eta

        # Current reliability
        reliability = weibull_reliability(operating_hours, beta, eta)

        # Current failure rate (hazard function)
        failure_rate = weibull_failure_rate(operating_hours, beta, eta)

        # Mean time to failure
        mttf = weibull_mean_life(beta, eta)

        # Remaining useful life to reach reliability threshold
        rul_hours = remaining_useful_life(
            operating_hours,
            beta,
            eta,
            self.config.reliability_threshold
        )

        # Probability of failure in next 30 days
        hours_in_30d = 30 * hours_per_day
        failure_prob_30d = calculate_failure_probability(
            operating_hours,
            operating_hours + hours_in_30d,
            beta,
            eta
        )

        return {
            'reliability': reliability,
            'failure_rate': failure_rate,
            'mttf': mttf,
            'rul_hours': rul_hours,
            'failure_prob_30d': failure_prob_30d
        }

    def _calculate_flame_metrics(
        self,
        flame: FlameQualityMetrics,
        fuel_type: FuelType
    ) -> Dict[str, float]:
        """
        Calculate flame quality and combustion efficiency.

        Uses physics-based combustion analysis without ML/LLM.

        Args:
            flame: Flame quality measurements
            fuel_type: Type of fuel being burned

        Returns:
            Dictionary with score and efficiency
        """
        # Calculate flame quality score (0-100)
        score = calculate_flame_quality_score(
            flame.flame_temperature_c,
            flame.stability_index,
            flame.o2_percent,
            flame.co_ppm,
            flame.nox_ppm
        )

        # Calculate combustion efficiency
        efficiency = calculate_combustion_efficiency(
            flame.o2_percent,
            flame.co_ppm,
            fuel_type.value
        )

        return {
            'score': score,
            'efficiency': efficiency
        }

    def _calculate_health_metrics(
        self,
        input_data: BurnerInput,
        flame_quality_score: float
    ) -> Dict[str, float]:
        """
        Calculate overall health score and degradation rate.

        Uses weighted scoring based on operating parameters.

        Args:
            input_data: Complete input data
            flame_quality_score: Calculated flame quality score

        Returns:
            Dictionary with health_score and degradation_rate
        """
        # Calculate age factor (0-1, 1 = new)
        age_years = (date.today() - input_data.installation_date).days / 365.25
        age_factor = max(0, 1 - (age_years / 20))  # Assume 20-year nominal life

        # Calculate cycles factor (0-1, 1 = low cycles)
        # Typical burner tolerates ~50000 cycles
        cycles_factor = max(0, 1 - (input_data.cycles_count / 50000))

        # Calculate overall health
        health_score = calculate_overall_health(
            input_data.operating_hours,
            input_data.design_life_hours,
            flame_quality_score,
            cycles_factor,
            age_factor
        )

        # Calculate degradation rate from history
        degradation_rate = 0.0
        if len(input_data.health_history) >= 2:
            degradation_rate = calculate_degradation_rate(
                input_data.health_history,
                input_data.health_history_interval_hours
            )

        return {
            'health_score': health_score,
            'degradation_rate': degradation_rate,
            'age_factor': age_factor,
            'cycles_factor': cycles_factor
        }

    def _generate_recommendations(
        self,
        input_data: BurnerInput,
        health_metrics: Dict[str, float],
        flame_quality: Dict[str, float],
        reliability_metrics: Dict[str, float],
        anomalies: List[str]
    ) -> List[MaintenanceRecommendation]:
        """
        Generate maintenance recommendations based on analysis.

        Args:
            input_data: Original input data
            health_metrics: Calculated health metrics
            flame_quality: Flame quality analysis results
            reliability_metrics: Reliability analysis results
            anomalies: Detected flame anomalies

        Returns:
            List of prioritized maintenance recommendations
        """
        # Build component health dict for recommendation generator
        component_health = {}
        for comp in input_data.component_health:
            component_health[comp.component_name] = {
                'health_score': comp.health_score,
                'operating_hours': comp.operating_hours,
                'cycles': comp.cycles_count
            }

        # Build operating conditions dict
        operating_conditions = {
            'operating_hours': input_data.operating_hours,
            'design_life': input_data.design_life_hours,
            'reliability': reliability_metrics['reliability'],
            'rul_hours': reliability_metrics['rul_hours'],
            'failure_prob_30d': reliability_metrics['failure_prob_30d'],
            'anomalies': anomalies
        }

        # Generate recommendations
        raw_recommendations = generate_maintenance_recommendations(
            component_health,
            flame_quality,
            operating_conditions
        )

        # Convert to MaintenanceRecommendation objects
        recommendations = []
        for rec in raw_recommendations:
            recommendations.append(MaintenanceRecommendation(
                action=rec['action'],
                priority=MaintenancePriority(rec['priority']),
                due_date=rec.get('due_date'),
                estimated_hours=rec.get('estimated_hours', 1.0),
                component=rec.get('component'),
                reason=rec['reason']
            ))

        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'NONE': 4}
        recommendations.sort(key=lambda r: priority_order.get(r.priority.value, 5))

        return recommendations

    def _evaluate_replacement(
        self,
        input_data: BurnerInput,
        health_score: float,
        rul_hours: float
    ) -> Dict[str, Any]:
        """
        Evaluate whether burner should be replaced vs repaired.

        Args:
            input_data: Original input data
            health_score: Current health score
            rul_hours: Remaining useful life in hours

        Returns:
            Dictionary with should_replace and reason
        """
        age_years = (date.today() - input_data.installation_date).days / 365.25

        should_replace, reason = should_replace_burner(
            age_years,
            health_score,
            rul_hours,
            input_data.repair_cost_ratio
        )

        return {
            'should_replace': should_replace,
            'reason': reason
        }

    def _calculate_provenance_hash(
        self,
        input_data: BurnerInput,
        reliability_metrics: Dict[str, float],
        flame_quality: Dict[str, float],
        health_metrics: Dict[str, float]
    ) -> str:
        """
        Calculate SHA-256 hash for complete audit trail.

        This hash provides cryptographic proof of the input data
        and calculated results for regulatory compliance.

        Args:
            input_data: Original input data
            reliability_metrics: Reliability calculation results
            flame_quality: Flame quality results
            health_metrics: Health calculation results

        Returns:
            SHA-256 hash as hexadecimal string
        """
        provenance_data = {
            'input': input_data.json(),
            'reliability': str(reliability_metrics),
            'flame_quality': str(flame_quality),
            'health': str(health_metrics),
            'agent_id': self.agent_id,
            'version': self.version,
            'timestamp': datetime.now().isoformat()
        }

        provenance_str = str(provenance_data)
        return hashlib.sha256(provenance_str.encode('utf-8')).hexdigest()
