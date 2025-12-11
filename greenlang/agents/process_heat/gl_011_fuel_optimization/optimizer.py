"""
GL-011 FUELCRAFT - Fuel Optimization Agent

This module implements the main FuelOptimizationAgent class that orchestrates
all fuel optimization components including pricing, blending, switching,
inventory, and cost optimization.

The agent inherits from BaseProcessHeatAgent and provides comprehensive
fuel management for industrial process heat applications.

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization import (
    ...     FuelOptimizationAgent,
    ...     FuelOptimizationConfig,
    ...     FuelOptimizationInput,
    ... )
    >>>
    >>> config = FuelOptimizationConfig(
    ...     facility_id="PLANT-001",
    ...     primary_fuel=FuelType.NATURAL_GAS,
    ... )
    >>> agent = FuelOptimizationAgent(config)
    >>> result = agent.process(input_data)
    >>> print(f"Recommended fuel: {result.optimization_result.optimal_fuel}")
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import logging

from pydantic import BaseModel

from greenlang.agents.process_heat.shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig,
    AgentCapability,
    SafetyLevel,
    ProcessingError,
    ValidationError,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    FuelOptimizationConfig,
    FuelPricingConfig,
    BlendingConfig,
    SwitchingConfig,
    InventoryConfig,
    CostOptimizationConfig,
    FuelType,
    OptimizationMode,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelOptimizationInput,
    FuelOptimizationOutput,
    FuelProperties,
    FuelPrice,
    OptimizationResult,
    BlendRecommendation,
    SwitchingRecommendation,
    InventoryStatus,
    CostAnalysis,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    HeatingValueCalculator,
    HeatingValueInput,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import (
    FuelPricingService,
    PriceQuote,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_blending import (
    FuelBlendingOptimizer,
    BlendInput,
    BlendOutput,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_switching import (
    FuelSwitchingController,
    SwitchingInput,
    SwitchingOutput,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.inventory import (
    InventoryManager,
    TankConfig,
    TankStatus,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.cost_optimization import (
    CostOptimizer,
    TotalCostInput,
    TotalCostOutput,
)
from greenlang.agents.intelligence_mixin import IntelligenceMixin, IntelligenceConfig
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT FUEL PROPERTIES
# =============================================================================

DEFAULT_FUEL_PROPERTIES = {
    "natural_gas": FuelProperties(
        fuel_type="natural_gas",
        fuel_name="Natural Gas",
        hhv_btu_scf=1020.0,
        lhv_btu_scf=918.0,
        specific_gravity=0.60,
        wobbe_index=1317.0,
        co2_kg_mmbtu=53.06,
        methane_pct=95.0,
        ethane_pct=2.5,
        propane_pct=0.5,
        nitrogen_pct=1.5,
        co2_pct=0.5,
    ),
    "no2_fuel_oil": FuelProperties(
        fuel_type="no2_fuel_oil",
        fuel_name="#2 Fuel Oil",
        hhv_btu_lb=19580.0,
        lhv_btu_lb=18410.0,
        density_lb_gal=7.21,
        co2_kg_mmbtu=73.16,
    ),
    "no6_fuel_oil": FuelProperties(
        fuel_type="no6_fuel_oil",
        fuel_name="#6 Fuel Oil",
        hhv_btu_lb=18300.0,
        lhv_btu_lb=17250.0,
        density_lb_gal=8.10,
        co2_kg_mmbtu=75.10,
    ),
    "lpg_propane": FuelProperties(
        fuel_type="lpg_propane",
        fuel_name="Propane",
        hhv_btu_scf=2516.0,
        lhv_btu_scf=2315.0,
        specific_gravity=1.52,
        wobbe_index=2040.0,
        co2_kg_mmbtu=62.87,
    ),
    "biogas": FuelProperties(
        fuel_type="biogas",
        fuel_name="Biogas",
        hhv_btu_scf=600.0,
        lhv_btu_scf=540.0,
        specific_gravity=0.90,
        wobbe_index=632.0,
        co2_kg_mmbtu=0.0,
        methane_pct=60.0,
        co2_pct=38.0,
        nitrogen_pct=1.5,
    ),
    "hydrogen": FuelProperties(
        fuel_type="hydrogen",
        fuel_name="Hydrogen",
        hhv_btu_scf=324.0,
        lhv_btu_scf=274.0,
        specific_gravity=0.07,
        wobbe_index=1226.0,
        co2_kg_mmbtu=0.0,
        hydrogen_pct=100.0,
    ),
}


# =============================================================================
# FUEL OPTIMIZATION AGENT
# =============================================================================

class FuelOptimizationAgent(IntelligenceMixin, BaseProcessHeatAgent[FuelOptimizationInput, FuelOptimizationOutput]):
    """
    GL-011 FUELCRAFT Fuel Optimization Agent.

    This agent provides comprehensive fuel optimization for industrial
    process heat applications including:
    - Real-time fuel price integration
    - Heating value calculations (HHV, LHV, Wobbe Index)
    - Multi-fuel blending optimization
    - Automated fuel switching
    - Inventory management
    - Total cost of ownership optimization

    All calculations are deterministic (zero-hallucination) with complete
    provenance tracking for regulatory compliance.

    Intelligence Level: STANDARD
    Regulatory Context: ISO 50001, GHG Protocol

    Attributes:
        fuel_config: Fuel optimization configuration
        pricing_service: Real-time fuel pricing service
        hv_calculator: Heating value calculator
        blending_optimizer: Multi-fuel blending optimizer
        switching_controller: Fuel switching controller
        inventory_manager: Fuel inventory manager
        cost_optimizer: Total cost optimizer

    Example:
        >>> config = FuelOptimizationConfig(
        ...     facility_id="PLANT-001",
        ...     primary_fuel=FuelType.NATURAL_GAS,
        ...     available_fuels=[FuelType.NATURAL_GAS, FuelType.NO2_FUEL_OIL],
        ... )
        >>> agent = FuelOptimizationAgent(config)
        >>> result = agent.process(input_data)
    """

    def __init__(self, fuel_config: FuelOptimizationConfig) -> None:
        """
        Initialize the Fuel Optimization Agent.

        Args:
            fuel_config: Fuel optimization configuration
        """
        # Create base agent config
        agent_config = AgentConfig(
            agent_type="GL-011",
            name=f"FUELCRAFT-{fuel_config.facility_id}",
            version=fuel_config.agent_version,
            capabilities={
                AgentCapability.REAL_TIME_MONITORING,
                AgentCapability.OPTIMIZATION,
                AgentCapability.PREDICTIVE_ANALYTICS,
            },
        )

        # Initialize base agent
        super().__init__(
            config=agent_config,
            safety_level=SafetyLevel(fuel_config.safety_level),
        )

        self.fuel_config = fuel_config

        # Initialize components
        self._init_components()

        # Initialize intelligence with STANDARD level configuration
        self._init_intelligence(IntelligenceConfig(
            enabled=True,
            model="auto",
            max_budget_per_call_usd=0.10,
            enable_explanations=True,
            enable_recommendations=True,
            enable_anomaly_detection=False,
            domain_context="fuel optimization and energy management",
            regulatory_context="ISO 50001, GHG Protocol",
        ))

        logger.info(
            f"FuelOptimizationAgent initialized for {fuel_config.facility_id}"
        )

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return IntelligenceLevel.STANDARD

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=False,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    def _init_components(self) -> None:
        """Initialize all sub-components."""
        # Heating value calculator
        self.hv_calculator = HeatingValueCalculator()

        # Pricing service
        self.pricing_service = FuelPricingService(
            config=self.fuel_config.pricing,
            carbon_price_usd_ton=self.fuel_config.cost_optimization.carbon_price_usd_ton,
        )

        # Blending optimizer
        self.blending_optimizer = FuelBlendingOptimizer(
            config=self.fuel_config.blending,
            heating_value_calculator=self.hv_calculator,
        )

        # Switching controller
        self.switching_controller = FuelSwitchingController(
            config=self.fuel_config.switching,
        )

        # Inventory manager
        self.inventory_manager = InventoryManager(
            config=self.fuel_config.inventory,
        )

        # Cost optimizer
        self.cost_optimizer = CostOptimizer(
            config=self.fuel_config.cost_optimization,
        )

        # Initialize fuel properties
        self._fuel_properties: Dict[str, FuelProperties] = dict(DEFAULT_FUEL_PROPERTIES)

    def process(
        self,
        input_data: FuelOptimizationInput,
    ) -> FuelOptimizationOutput:
        """
        Process fuel optimization request.

        This is the main entry point for fuel optimization. It:
        1. Validates input data
        2. Fetches current fuel prices
        3. Evaluates blending options
        4. Evaluates switching options
        5. Optimizes total cost
        6. Returns comprehensive recommendations

        Args:
            input_data: Fuel optimization input

        Returns:
            FuelOptimizationOutput with recommendations

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        start_time = datetime.now(timezone.utc)

        logger.info(
            f"Processing fuel optimization for {input_data.facility_id}"
        )

        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValidationError("Input validation failed")

            # Enter safety context
            with self.safety_guard():
                # Step 1: Get fuel prices
                fuel_prices = self._get_fuel_prices(input_data)

                # Step 2: Get/update fuel properties
                fuel_properties = self._get_fuel_properties(input_data)

                # Step 3: Evaluate blending options
                blend_recommendation = self._evaluate_blending(
                    input_data,
                    fuel_prices,
                    fuel_properties,
                )

                # Step 4: Evaluate switching options
                switching_recommendation = self._evaluate_switching(
                    input_data,
                    fuel_prices,
                )

                # Step 5: Optimize total cost
                cost_analysis = self._optimize_cost(
                    input_data,
                    fuel_prices,
                    fuel_properties,
                )

                # Step 6: Create optimization result
                optimization_result = self._create_optimization_result(
                    input_data,
                    fuel_prices,
                    blend_recommendation,
                    switching_recommendation,
                    cost_analysis,
                )

                # Step 7: Check inventory alerts
                inventory_alerts = self._check_inventory(input_data)

                # Calculate processing time
                processing_time = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000

                # Calculate provenance hash
                provenance_hash = self.calculate_provenance_hash(
                    input_data,
                    optimization_result,
                )

                # Calculate input hash
                input_hash = self._hash_object(input_data)

                # Create output
                output = FuelOptimizationOutput(
                    facility_id=input_data.facility_id,
                    request_id=input_data.request_id,
                    status="success",
                    processing_time_ms=round(processing_time, 2),
                    optimization_result=optimization_result,
                    fuel_prices_used={
                        fuel: self._price_quote_to_schema(quote)
                        for fuel, quote in fuel_prices.items()
                    },
                    inventory_alerts=inventory_alerts,
                    delivery_recommendations=self._get_delivery_recommendations(),
                    kpis=self._calculate_kpis(optimization_result, input_data),
                    provenance_hash=provenance_hash,
                    input_hash=input_hash,
                    metadata={
                        "agent_version": self.fuel_config.agent_version,
                        "optimization_mode": self.fuel_config.cost_optimization.mode.value,
                    },
                )

                # Validate output
                if not self.validate_output(output):
                    raise ValidationError("Output validation failed")

                # Generate LLM explanation for fuel optimization results
                explanation = self.generate_explanation(
                    input_data=input_data.dict() if hasattr(input_data, 'dict') else {"facility_id": input_data.facility_id},
                    output_data={
                        "optimization_status": optimization_result.optimization_status,
                        "current_fuel_cost_usd_hr": optimization_result.current_fuel_cost_usd_hr,
                        "recommended_fuel_cost_usd_hr": optimization_result.recommended_fuel_cost_usd_hr,
                        "potential_savings_usd_hr": optimization_result.potential_savings_usd_hr,
                        "potential_savings_usd_year": optimization_result.potential_savings_usd_year,
                        "co2_reduction_kg_hr": optimization_result.co2_reduction_kg_hr,
                    },
                    calculation_steps=[
                        f"Analyzed {len(fuel_prices)} fuel options",
                        f"Evaluated blending opportunities" if blend_recommendation else "Blending not applicable",
                        f"Evaluated switching opportunities" if switching_recommendation else "Switching not applicable",
                        f"Calculated total cost of ownership",
                        f"Determined potential savings: ${optimization_result.potential_savings_usd_year:.0f}/year",
                    ]
                )

                # Generate recommendations for optimization actions
                recommendations = self.generate_recommendations(
                    analysis={
                        "current_fuel": input_data.current_fuel,
                        "current_cost_usd_hr": optimization_result.current_fuel_cost_usd_hr,
                        "potential_savings_usd_year": optimization_result.potential_savings_usd_year,
                        "blend_recommendation": blend_recommendation.dict() if blend_recommendation else None,
                        "switching_recommendation": switching_recommendation.dict() if switching_recommendation else None,
                        "co2_reduction_kg_hr": optimization_result.co2_reduction_kg_hr,
                    },
                    max_recommendations=5,
                    focus_areas=["cost reduction", "emissions reduction", "fuel flexibility"]
                )

                logger.info(
                    f"Fuel optimization completed in {processing_time:.1f}ms"
                )
                logger.debug(f"Generated explanation and {len(recommendations)} recommendations")

                return output

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Fuel optimization failed: {e}", exc_info=True)
            raise ProcessingError(f"Fuel optimization failed: {str(e)}") from e

    def validate_input(self, input_data: FuelOptimizationInput) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input to validate

        Returns:
            True if valid
        """
        # Check required fields
        if not input_data.facility_id:
            logger.error("Missing facility_id")
            return False

        if not input_data.current_fuel:
            logger.error("Missing current_fuel")
            return False

        if input_data.current_heat_input_mmbtu_hr <= 0:
            logger.error("Invalid heat input")
            return False

        return True

    def validate_output(self, output_data: FuelOptimizationOutput) -> bool:
        """
        Validate output data.

        Args:
            output_data: Output to validate

        Returns:
            True if valid
        """
        # Check optimization result exists
        if output_data.optimization_result is None:
            logger.error("Missing optimization result")
            return False

        # Check provenance hash
        if not output_data.provenance_hash:
            logger.error("Missing provenance hash")
            return False

        return True

    def _get_fuel_prices(
        self,
        input_data: FuelOptimizationInput,
    ) -> Dict[str, PriceQuote]:
        """Get current fuel prices."""
        # Use provided prices if available
        if input_data.fuel_prices:
            return {
                fuel: self._schema_to_price_quote(price)
                for fuel, price in input_data.fuel_prices.items()
            }

        # Fetch from pricing service
        available_fuels = [f.value for f in self.fuel_config.available_fuels]
        available_fuels.append(input_data.current_fuel)
        available_fuels = list(set(available_fuels))

        prices = {}
        for fuel in available_fuels:
            try:
                prices[fuel] = self.pricing_service.get_current_price(fuel)
            except Exception as e:
                logger.warning(f"Could not get price for {fuel}: {e}")

        return prices

    def _get_fuel_properties(
        self,
        input_data: FuelOptimizationInput,
    ) -> Dict[str, FuelProperties]:
        """Get fuel properties."""
        # Use provided properties if available
        if input_data.fuel_properties:
            return input_data.fuel_properties

        # Use stored properties
        return self._fuel_properties

    def _evaluate_blending(
        self,
        input_data: FuelOptimizationInput,
        fuel_prices: Dict[str, PriceQuote],
        fuel_properties: Dict[str, FuelProperties],
    ) -> Optional[BlendRecommendation]:
        """Evaluate fuel blending options."""
        if not self.fuel_config.blending.enabled:
            return None

        if len(fuel_prices) < 2:
            return None

        try:
            # Create blend input
            blend_input = BlendInput(
                available_fuels=list(fuel_prices.keys()),
                fuel_properties=fuel_properties,
                fuel_prices={
                    fuel: self._price_quote_to_schema(quote)
                    for fuel, quote in fuel_prices.items()
                },
                current_blend=input_data.current_blend,
                required_heat_input_mmbtu_hr=input_data.current_heat_input_mmbtu_hr,
                max_co2_kg_hr=input_data.max_emissions_kg_hr,
            )

            # Optimize blend
            blend_output = self.blending_optimizer.optimize_blend(blend_input)

            # Create recommendation
            return self.blending_optimizer.create_blend_recommendation(
                blend_output,
                input_data.current_blend,
            )

        except Exception as e:
            logger.warning(f"Blending evaluation failed: {e}")
            return None

    def _evaluate_switching(
        self,
        input_data: FuelOptimizationInput,
        fuel_prices: Dict[str, PriceQuote],
    ) -> Optional[SwitchingRecommendation]:
        """Evaluate fuel switching options."""
        if self.fuel_config.switching.mode.value == "disabled":
            return None

        try:
            # Get current price
            current_price = fuel_prices.get(input_data.current_fuel)
            if not current_price:
                return None

            # Create switching input
            switching_input = SwitchingInput(
                current_fuel=input_data.current_fuel,
                current_cost_usd_mmbtu=current_price.total_price,
                current_heat_input_mmbtu_hr=input_data.current_heat_input_mmbtu_hr,
                available_fuels=list(fuel_prices.keys()),
                fuel_prices={
                    fuel: self._price_quote_to_schema(quote)
                    for fuel, quote in fuel_prices.items()
                },
                equipment_id=input_data.facility_id,
                current_load_pct=input_data.current_load_pct,
                switches_today=self.switching_controller.switches_today,
            )

            # Evaluate switch
            switching_output = self.switching_controller.evaluate_switch(switching_input)

            if switching_output.recommended:
                return SwitchingRecommendation(
                    recommended=True,
                    current_fuel=switching_output.current_fuel,
                    recommended_fuel=switching_output.recommended_fuel,
                    trigger_reason=switching_output.trigger_reason,
                    current_cost_usd_hr=switching_output.current_cost_usd_hr,
                    recommended_cost_usd_hr=switching_output.recommended_cost_usd_hr,
                    savings_usd_hr=switching_output.savings_usd_hr,
                    payback_hours=switching_output.payback_hours,
                    transition_time_minutes=switching_output.transition_time_minutes,
                    transition_cost_usd=0.0,
                    safety_checks_passed=switching_output.safety_checks_passed,
                    safety_warnings=switching_output.safety_warnings,
                    requires_purge=switching_output.requires_purge,
                    operator_approval_required=switching_output.requires_operator_approval,
                    valid_until=switching_output.approval_timeout,
                )

            return None

        except Exception as e:
            logger.warning(f"Switching evaluation failed: {e}")
            return None

    def _optimize_cost(
        self,
        input_data: FuelOptimizationInput,
        fuel_prices: Dict[str, PriceQuote],
        fuel_properties: Dict[str, FuelProperties],
    ) -> Optional[CostAnalysis]:
        """Optimize total cost of ownership."""
        if not self.fuel_config.cost_optimization.enabled:
            return None

        try:
            # Create cost input
            cost_input = TotalCostInput(
                fuel_options=list(fuel_prices.keys()),
                fuel_prices={
                    fuel: self._price_quote_to_schema(quote)
                    for fuel, quote in fuel_prices.items()
                },
                fuel_properties=fuel_properties,
                heat_demand_mmbtu_hr=input_data.current_heat_input_mmbtu_hr,
                current_fuel=input_data.current_fuel,
                carbon_price_usd_ton=self.fuel_config.cost_optimization.carbon_price_usd_ton,
            )

            # Optimize
            cost_output = self.cost_optimizer.optimize(cost_input)

            # Create analysis
            return self.cost_optimizer.create_cost_analysis(cost_output, cost_input)

        except Exception as e:
            logger.warning(f"Cost optimization failed: {e}")
            return None

    def _check_inventory(
        self,
        input_data: FuelOptimizationInput,
    ) -> List[Dict[str, Any]]:
        """Check inventory alerts."""
        if not self.fuel_config.inventory.enabled:
            return []

        alerts = self.inventory_manager.get_active_alerts()
        return [
            {
                "alert_id": a.alert_id,
                "tank_id": a.tank_id,
                "alert_type": a.alert_type.value,
                "level": a.level.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts
        ]

    def _get_delivery_recommendations(self) -> List[Dict[str, Any]]:
        """Get delivery recommendations."""
        if not self.fuel_config.inventory.enabled:
            return []

        pending = self.inventory_manager.get_pending_deliveries()
        return [
            {
                "delivery_id": d.delivery_id,
                "tank_id": d.tank_id,
                "fuel_type": d.fuel_type,
                "scheduled_date": d.scheduled_date.isoformat(),
                "quantity_gal": d.quantity_gal,
                "status": d.status.value,
            }
            for d in pending
        ]

    def _create_optimization_result(
        self,
        input_data: FuelOptimizationInput,
        fuel_prices: Dict[str, PriceQuote],
        blend_recommendation: Optional[BlendRecommendation],
        switching_recommendation: Optional[SwitchingRecommendation],
        cost_analysis: Optional[CostAnalysis],
    ) -> OptimizationResult:
        """Create comprehensive optimization result."""
        # Get current fuel cost
        current_price = fuel_prices.get(input_data.current_fuel)
        current_cost_hr = (
            current_price.total_price * input_data.current_heat_input_mmbtu_hr
            if current_price else 0.0
        )

        # Determine recommended fuel and cost
        if switching_recommendation and switching_recommendation.recommended:
            recommended_fuel = switching_recommendation.recommended_fuel
            recommended_cost_hr = switching_recommendation.recommended_cost_usd_hr
        elif blend_recommendation:
            recommended_fuel = blend_recommendation.primary_fuel
            recommended_cost_hr = (
                blend_recommendation.blended_cost_usd_mmbtu *
                input_data.current_heat_input_mmbtu_hr
            )
        else:
            recommended_fuel = input_data.current_fuel
            recommended_cost_hr = current_cost_hr

        # Calculate savings
        savings_hr = current_cost_hr - recommended_cost_hr
        savings_year = savings_hr * 8760 * 0.9  # 90% availability

        # Calculate emissions
        current_co2 = self._get_emission_factor(input_data.current_fuel)
        recommended_co2 = self._get_emission_factor(recommended_fuel)
        current_co2_hr = current_co2 * input_data.current_heat_input_mmbtu_hr
        recommended_co2_hr = recommended_co2 * input_data.current_heat_input_mmbtu_hr
        co2_reduction = current_co2_hr - recommended_co2_hr

        return OptimizationResult(
            optimization_status="success",
            optimization_mode=self.fuel_config.cost_optimization.mode.value,
            blend_recommendation=blend_recommendation,
            switching_recommendation=switching_recommendation,
            cost_analysis=cost_analysis,
            recommended_fuel_cost_usd_hr=round(recommended_cost_hr, 2),
            current_fuel_cost_usd_hr=round(current_cost_hr, 2),
            potential_savings_usd_hr=round(savings_hr, 2),
            potential_savings_usd_year=round(savings_year, 0),
            current_co2_kg_hr=round(current_co2_hr, 2),
            recommended_co2_kg_hr=round(recommended_co2_hr, 2),
            co2_reduction_kg_hr=round(co2_reduction, 2),
            confidence_score=0.95,
        )

    def _calculate_kpis(
        self,
        result: OptimizationResult,
        input_data: FuelOptimizationInput,
    ) -> Dict[str, float]:
        """Calculate key performance indicators."""
        return {
            "fuel_cost_usd_hr": result.current_fuel_cost_usd_hr,
            "potential_savings_usd_hr": result.potential_savings_usd_hr,
            "potential_savings_pct": (
                result.potential_savings_usd_hr / result.current_fuel_cost_usd_hr * 100
                if result.current_fuel_cost_usd_hr > 0 else 0.0
            ),
            "co2_emissions_kg_hr": result.current_co2_kg_hr,
            "co2_reduction_potential_kg_hr": result.co2_reduction_kg_hr,
            "heat_input_mmbtu_hr": input_data.current_heat_input_mmbtu_hr,
            "load_pct": input_data.current_load_pct,
        }

    def _get_emission_factor(self, fuel_type: str) -> float:
        """Get CO2 emission factor for fuel type."""
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")
        props = self._fuel_properties.get(fuel_key)
        if props and props.co2_kg_mmbtu is not None:
            return props.co2_kg_mmbtu
        return 53.06  # Default to natural gas

    def _price_quote_to_schema(self, quote: PriceQuote) -> FuelPrice:
        """Convert PriceQuote to FuelPrice schema."""
        return FuelPrice(
            fuel_type=quote.fuel_type,
            price=quote.total_price,
            unit=quote.unit,
            source=quote.source,
            timestamp=quote.timestamp,
            effective_until=quote.valid_until,
            commodity_price=quote.commodity_price,
            transport_cost=quote.transport_cost,
            basis_differential=quote.basis_differential,
            taxes=quote.taxes,
            confidence=quote.confidence,
        )

    def _schema_to_price_quote(self, price: FuelPrice) -> PriceQuote:
        """Convert FuelPrice schema to PriceQuote."""
        return PriceQuote(
            fuel_type=price.fuel_type,
            commodity_price=price.commodity_price,
            basis_differential=price.basis_differential,
            transport_cost=price.transport_cost,
            taxes=price.taxes,
            carbon_cost=0.0,
            total_price=price.price,
            unit=price.unit,
            source=price.source,
            timestamp=price.timestamp,
            valid_until=price.effective_until or datetime.now(timezone.utc),
            confidence=price.confidence,
        )

    def _hash_object(self, obj: Any) -> str:
        """Hash an object for provenance tracking."""
        import json

        if hasattr(obj, "json"):
            data_str = obj.json()
        elif hasattr(obj, "dict"):
            data_str = json.dumps(obj.dict(), sort_keys=True, default=str)
        else:
            data_str = json.dumps(obj, sort_keys=True, default=str)

        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
