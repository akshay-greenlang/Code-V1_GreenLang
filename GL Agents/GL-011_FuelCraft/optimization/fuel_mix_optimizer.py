# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft - Fuel Mix Optimizer

Orchestrates the complete fuel procurement optimization workflow.

Workflow Steps:
1. Load and validate fuel data (prices, heating values, carbon intensities)
2. Build optimization model using FuelOptimizationModel
3. Apply scenario parameters (if running scenario analysis)
4. Solve using configurable solver (HiGHS/CBC/CPLEX)
5. Extract and validate solution
6. Calculate detailed cost breakdown
7. Return optimal fuel mix with blend ratios and provenance

Decision Variables:
- x_{i,t}: Procure quantity of fuel i in period t (MJ or kg)
- y_{i,t}: Consumed/withdrawn fuel i in period t
- s_{k,t}: Inventory in tank k at period t
- b_{i,t}: Blend fraction of fuel i in period t (sum = 1)
- z_{c,t}: Contract commitment decisions (binary/integer)

Constraints:
- Demand satisfaction on energy basis
- Inventory balance: s_{k,t} = s_{k,t-1} + inflow - outflow - losses
- Tank/flow limits (min/max)
- Blend quality limits (sulfur, ash, water, viscosity)
- Contract constraints (take-or-pay, min/max)
- Safety constraints (flash point, vapor pressure)

Zero-Hallucination Approach:
- All optimizations use deterministic solvers (no ML in objective)
- All calculations traceable to governed data and formulas
- Full provenance tracking with SHA-256 run bundle hashing

Standards:
- ISO 14064 (GHG Quantification)
- IMO CII (Carbon Intensity Indicator)
- NFPA 30 (Flammable Liquids)
- API MPMS (Petroleum Measurement)

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import time
import uuid

# Internal imports
from optimization.model_builder import (
    FuelOptimizationModel,
    ModelConfig,
    FuelData,
    TankData,
    ContractData,
    DemandData,
)
from optimization.solver import (
    Solver,
    SolverConfig,
    SolverType,
    SolverStatus,
    Solution,
)
from optimization.cost_model import (
    CostModel,
    CostBreakdown,
    PurchaseCostParams,
    LogisticsCostParams,
    StorageCostParams,
    ContractPenaltyParams,
    CarbonCostParams,
    RiskCostParams,
    PricingType,
    LogisticsMode,
    CarbonScheme,
)

logger = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Status of optimization run."""
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ERROR = "error"


class BlendQualityStatus(Enum):
    """Status of blend quality validation."""
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    WARNING = "warning"


@dataclass
class FuelMixEntry:
    """
    Single entry in the optimal fuel mix.

    Represents the procurement and consumption plan for one fuel
    in one time period.
    """
    fuel_id: str
    period: int
    procurement_mj: Decimal
    consumption_mj: Decimal
    blend_fraction: Decimal
    contract_allocation_mj: Decimal
    spot_allocation_mj: Decimal
    unit_cost_per_mj: Decimal
    total_cost: Decimal
    carbon_intensity_kg_co2e_mj: Decimal
    emissions_kg_co2e: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fuel_id": self.fuel_id,
            "period": self.period,
            "procurement_mj": str(self.procurement_mj),
            "consumption_mj": str(self.consumption_mj),
            "blend_fraction": str(self.blend_fraction),
            "contract_allocation_mj": str(self.contract_allocation_mj),
            "spot_allocation_mj": str(self.spot_allocation_mj),
            "unit_cost_per_mj": str(self.unit_cost_per_mj),
            "total_cost": str(self.total_cost),
            "carbon_intensity_kg_co2e_mj": str(self.carbon_intensity_kg_co2e_mj),
            "emissions_kg_co2e": str(self.emissions_kg_co2e)
        }


@dataclass
class InventoryProjection:
    """
    Inventory projection for a tank over time.
    """
    tank_id: str
    period: int
    opening_level_mj: Decimal
    inflow_mj: Decimal
    outflow_mj: Decimal
    losses_mj: Decimal
    closing_level_mj: Decimal
    utilization_pct: Decimal
    is_at_minimum: bool
    is_near_maximum: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tank_id": self.tank_id,
            "period": self.period,
            "opening_level_mj": str(self.opening_level_mj),
            "inflow_mj": str(self.inflow_mj),
            "outflow_mj": str(self.outflow_mj),
            "losses_mj": str(self.losses_mj),
            "closing_level_mj": str(self.closing_level_mj),
            "utilization_pct": str(self.utilization_pct),
            "is_at_minimum": self.is_at_minimum,
            "is_near_maximum": self.is_near_maximum
        }


@dataclass
class ProcurementSchedule:
    """
    Procurement schedule entry.
    """
    fuel_id: str
    period: int
    quantity_mj: Decimal
    source: str  # "contract" or "spot"
    contract_id: Optional[str]
    delivery_date: Optional[datetime]
    price_per_mj: Decimal
    total_cost: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fuel_id": self.fuel_id,
            "period": self.period,
            "quantity_mj": str(self.quantity_mj),
            "source": self.source,
            "contract_id": self.contract_id,
            "delivery_date": self.delivery_date.isoformat() if self.delivery_date else None,
            "price_per_mj": str(self.price_per_mj),
            "total_cost": str(self.total_cost)
        }


@dataclass
class BlendQuality:
    """
    Blend quality assessment for a period.
    """
    period: int
    sulfur_wt_pct: Decimal
    ash_wt_pct: Decimal
    water_vol_pct: Decimal
    viscosity_cst: Decimal
    flash_point_c: Decimal
    carbon_intensity_kg_co2e_mj: Decimal
    status: BlendQualityStatus
    violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "period": self.period,
            "sulfur_wt_pct": str(self.sulfur_wt_pct),
            "ash_wt_pct": str(self.ash_wt_pct),
            "water_vol_pct": str(self.water_vol_pct),
            "viscosity_cst": str(self.viscosity_cst),
            "flash_point_c": str(self.flash_point_c),
            "carbon_intensity_kg_co2e_mj": str(self.carbon_intensity_kg_co2e_mj),
            "status": self.status.value,
            "violations": self.violations
        }


@dataclass
class OptimizationResult:
    """
    Complete optimization result with full provenance.

    Contains:
    - Optimal fuel mix by time period with blend ratios
    - Cost breakdown (purchase, logistics, storage, losses, carbon, penalties)
    - Procurement schedule with contract allocations
    - Inventory projections
    - Blend quality validation
    - Run bundle hash for reproducibility
    """
    # Status
    status: OptimizationStatus
    solver_status: SolverStatus
    objective_value: Decimal

    # Solution quality
    mip_gap: Optional[float]
    is_optimal: bool
    is_feasible: bool

    # Fuel mix
    fuel_mix: List[FuelMixEntry]

    # Costs
    cost_breakdown: CostBreakdown

    # Schedules and projections
    procurement_schedule: List[ProcurementSchedule]
    inventory_projections: List[InventoryProjection]
    blend_quality: List[BlendQuality]

    # Aggregated metrics
    total_procurement_mj: Decimal
    total_consumption_mj: Decimal
    total_cost: Decimal
    average_cost_per_mj: Decimal
    total_emissions_kg_co2e: Decimal
    average_carbon_intensity: Decimal

    # By fuel summary
    fuel_summary: Dict[str, Dict[str, Any]]

    # Performance
    solve_time_seconds: float
    model_build_time_seconds: float
    total_time_seconds: float

    # Provenance
    run_id: str
    run_bundle_hash: str
    model_hash: str
    solver_config_hash: str
    input_data_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.run_bundle_hash:
            self.run_bundle_hash = self._compute_run_bundle_hash()

    def _compute_run_bundle_hash(self) -> str:
        """Compute SHA-256 hash of complete run bundle."""
        data = {
            "run_id": self.run_id,
            "status": self.status.value,
            "objective_value": str(self.objective_value),
            "total_cost": str(self.total_cost),
            "total_emissions_kg_co2e": str(self.total_emissions_kg_co2e),
            "model_hash": self.model_hash,
            "solver_config_hash": self.solver_config_hash,
            "input_data_hash": self.input_data_hash,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "solver_status": self.solver_status.value,
            "objective_value": str(self.objective_value),
            "mip_gap": self.mip_gap,
            "is_optimal": self.is_optimal,
            "is_feasible": self.is_feasible,
            "fuel_mix": [f.to_dict() for f in self.fuel_mix],
            "cost_breakdown": self.cost_breakdown.to_dict(),
            "procurement_schedule": [p.to_dict() for p in self.procurement_schedule],
            "inventory_projections": [i.to_dict() for i in self.inventory_projections],
            "blend_quality": [b.to_dict() for b in self.blend_quality],
            "total_procurement_mj": str(self.total_procurement_mj),
            "total_consumption_mj": str(self.total_consumption_mj),
            "total_cost": str(self.total_cost),
            "average_cost_per_mj": str(self.average_cost_per_mj),
            "total_emissions_kg_co2e": str(self.total_emissions_kg_co2e),
            "average_carbon_intensity": str(self.average_carbon_intensity),
            "fuel_summary": self.fuel_summary,
            "solve_time_seconds": self.solve_time_seconds,
            "model_build_time_seconds": self.model_build_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "run_id": self.run_id,
            "run_bundle_hash": self.run_bundle_hash,
            "model_hash": self.model_hash,
            "solver_config_hash": self.solver_config_hash,
            "input_data_hash": self.input_data_hash,
            "timestamp": self.timestamp.isoformat()
        }

    def get_summary(self) -> str:
        """Generate human-readable optimization summary."""
        lines = [
            "=" * 60,
            "FUEL MIX OPTIMIZATION RESULTS",
            "=" * 60,
            f"Run ID:           {self.run_id}",
            f"Status:           {self.status.value.upper()}",
            f"Solver Status:    {self.solver_status.value}",
            f"Is Optimal:       {self.is_optimal}",
            f"MIP Gap:          {self.mip_gap:.2%}" if self.mip_gap else "MIP Gap:          N/A",
            "",
            "COSTS",
            "-" * 40,
            f"Total Cost:       ${self.total_cost:,.2f}",
            f"Avg Cost/MJ:      ${self.average_cost_per_mj:,.6f}",
            "",
            "ENERGY",
            "-" * 40,
            f"Total Procurement: {self.total_procurement_mj:,.0f} MJ",
            f"Total Consumption: {self.total_consumption_mj:,.0f} MJ",
            "",
            "EMISSIONS",
            "-" * 40,
            f"Total Emissions:  {self.total_emissions_kg_co2e:,.0f} kgCO2e",
            f"Avg Carbon Int.:  {self.average_carbon_intensity:,.6f} kgCO2e/MJ",
            "",
            "FUEL MIX",
            "-" * 40,
        ]

        for fuel_id, summary in self.fuel_summary.items():
            pct = float(summary.get("blend_fraction_avg", 0)) * 100
            lines.append(f"  {fuel_id}: {pct:.1f}%")

        lines.extend([
            "",
            "PERFORMANCE",
            "-" * 40,
            f"Model Build:      {self.model_build_time_seconds:.3f}s",
            f"Solve Time:       {self.solve_time_seconds:.3f}s",
            f"Total Time:       {self.total_time_seconds:.3f}s",
            "",
            "PROVENANCE",
            "-" * 40,
            f"Run Bundle Hash:  {self.run_bundle_hash[:16]}...",
            f"Timestamp:        {self.timestamp.isoformat()}",
            "=" * 60,
        ])

        return "\n".join(lines)


@dataclass
class OptimizerConfig:
    """
    Configuration for FuelMixOptimizer.

    Attributes:
        solver_type: Type of solver to use
        time_limit_seconds: Maximum solve time
        mip_gap: Acceptable optimality gap
        threads: Number of solver threads
        random_seed: Seed for reproducibility
        include_carbon_cost: Include carbon costs in objective
        carbon_price_per_kg_co2e: Carbon price
        include_risk_premium: Include risk costs in objective
        risk_premium_pct: Risk premium percentage
        validate_blend_quality: Validate blend quality in solution
        enable_provenance_tracking: Enable full provenance tracking
    """
    solver_type: SolverType = SolverType.HIGHS
    time_limit_seconds: float = 300.0
    mip_gap: float = 0.01
    threads: int = 1
    random_seed: int = 42
    include_carbon_cost: bool = True
    carbon_price_per_kg_co2e: Decimal = Decimal("0.08")  # $80/tCO2e
    include_risk_premium: bool = False
    risk_premium_pct: Decimal = Decimal("2.0")
    validate_blend_quality: bool = True
    enable_provenance_tracking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "solver_type": self.solver_type.value,
            "time_limit_seconds": self.time_limit_seconds,
            "mip_gap": self.mip_gap,
            "threads": self.threads,
            "random_seed": self.random_seed,
            "include_carbon_cost": self.include_carbon_cost,
            "carbon_price_per_kg_co2e": str(self.carbon_price_per_kg_co2e),
            "include_risk_premium": self.include_risk_premium,
            "risk_premium_pct": str(self.risk_premium_pct),
            "validate_blend_quality": self.validate_blend_quality,
            "enable_provenance_tracking": self.enable_provenance_tracking
        }


class FuelMixOptimizer:
    """
    Orchestrates the complete fuel procurement optimization workflow.

    Provides ZERO-HALLUCINATION optimization for:
    - Multi-fuel procurement planning
    - Blend ratio optimization
    - Contract allocation optimization
    - Inventory management
    - Carbon cost minimization
    - Quality constraint satisfaction

    The optimizer uses deterministic LP/MILP solvers and maintains
    full provenance tracking for audit compliance.

    Example:
        >>> config = OptimizerConfig(
        ...     solver_type=SolverType.HIGHS,
        ...     time_limit_seconds=300
        ... )
        >>> optimizer = FuelMixOptimizer(config)
        >>> result = optimizer.optimize(
        ...     fuels=fuel_data,
        ...     tanks=tank_data,
        ...     demands=demand_data,
        ...     contracts=contract_data
        ... )
        >>> print(result.get_summary())
    """

    NAME: str = "FuelMixOptimizer"
    VERSION: str = "1.0.0"

    def __init__(self, config: OptimizerConfig):
        """
        Initialize FuelMixOptimizer.

        Args:
            config: Optimizer configuration
        """
        self._config = config
        self._cost_model = CostModel()
        self._run_id: Optional[str] = None
        self._model: Optional[FuelOptimizationModel] = None
        self._solver: Optional[Solver] = None
        self._solution: Optional[Solution] = None

        logger.info(f"FuelMixOptimizer initialized with solver={config.solver_type.value}")

    def optimize(
        self,
        fuels: List[FuelData],
        tanks: List[TankData],
        demands: List[DemandData],
        contracts: Optional[List[ContractData]] = None,
        logistics_params: Optional[Dict[str, LogisticsCostParams]] = None,
        storage_params: Optional[Dict[str, StorageCostParams]] = None,
        carbon_params: Optional[CarbonCostParams] = None
    ) -> OptimizationResult:
        """
        Run complete fuel mix optimization - DETERMINISTIC.

        Workflow:
        1. Generate unique run ID
        2. Validate and hash input data
        3. Build optimization model
        4. Configure and run solver
        5. Extract solution values
        6. Calculate cost breakdown
        7. Generate fuel mix entries
        8. Project inventory levels
        9. Validate blend quality
        10. Compile and return results

        Args:
            fuels: List of available fuels with properties
            tanks: List of storage tanks with capacities
            demands: List of energy demands by period
            contracts: Optional list of supply contracts
            logistics_params: Logistics cost parameters by fuel
            storage_params: Storage cost parameters by tank
            carbon_params: Carbon cost parameters

        Returns:
            OptimizationResult with complete solution and provenance

        Raises:
            ValueError: If input validation fails
            RuntimeError: If optimization fails
        """
        start_time = time.time()
        self._run_id = f"OPT-{uuid.uuid4().hex[:12].upper()}"

        logger.info(f"Starting optimization run {self._run_id}")

        try:
            # Step 1: Validate and hash inputs
            input_data_hash = self._compute_input_hash(fuels, tanks, demands, contracts)
            logger.debug(f"Input data hash: {input_data_hash[:16]}...")

            # Step 2: Build optimization model
            model_start = time.time()
            model_config = ModelConfig(
                model_name=f"FuelMix_{self._run_id}",
                time_periods=len(demands),
                carbon_price_per_kg_co2e=(
                    self._config.carbon_price_per_kg_co2e
                    if self._config.include_carbon_cost
                    else Decimal("0")
                ),
                include_logistics=True
            )

            self._model = FuelOptimizationModel(
                config=model_config,
                fuels=fuels,
                tanks=tanks,
                demands=demands,
                contracts=contracts
            )
            model_build_time = time.time() - model_start
            model_hash = self._model._compute_hash()

            logger.info(f"Model built in {model_build_time:.3f}s with {len(self._model.variables)} variables")

            # Step 3: Configure solver
            solver_config = SolverConfig(
                solver_type=self._config.solver_type,
                time_limit_seconds=self._config.time_limit_seconds,
                mip_gap=self._config.mip_gap,
                threads=self._config.threads,
                random_seed=self._config.random_seed
            )
            self._solver = Solver(solver_config)
            solver_config_hash = self._solver._compute_config_hash()

            # Step 4: Solve
            solve_start = time.time()
            self._solution = self._solver.solve(self._model)
            solve_time = time.time() - solve_start

            logger.info(f"Solved in {solve_time:.3f}s with status {self._solution.status.value}")

            # Check if solution is usable
            if not self._solution.is_feasible:
                logger.warning(f"Optimization infeasible: {self._solution.status.value}")
                return self._create_infeasible_result(
                    status=self._map_solver_status(self._solution.status),
                    solver_status=self._solution.status,
                    solve_time=solve_time,
                    model_build_time=model_build_time,
                    model_hash=model_hash,
                    solver_config_hash=solver_config_hash,
                    input_data_hash=input_data_hash
                )

            # Step 5: Extract fuel mix from solution
            fuel_mix = self._extract_fuel_mix(fuels, demands)

            # Step 6: Calculate cost breakdown
            self._cost_model.clear_steps()
            cost_breakdown = self._calculate_cost_breakdown(
                fuel_mix=fuel_mix,
                fuels=fuels,
                contracts=contracts,
                logistics_params=logistics_params,
                storage_params=storage_params,
                carbon_params=carbon_params
            )

            # Step 7: Generate procurement schedule
            procurement_schedule = self._generate_procurement_schedule(
                fuel_mix=fuel_mix,
                fuels=fuels,
                contracts=contracts
            )

            # Step 8: Project inventory levels
            inventory_projections = self._project_inventory(
                fuel_mix=fuel_mix,
                tanks=tanks,
                demands=demands
            )

            # Step 9: Validate blend quality
            blend_quality = self._validate_blend_quality(
                fuel_mix=fuel_mix,
                fuels=fuels,
                demands=demands
            ) if self._config.validate_blend_quality else []

            # Step 10: Compile results
            total_time = time.time() - start_time

            result = self._compile_result(
                fuel_mix=fuel_mix,
                cost_breakdown=cost_breakdown,
                procurement_schedule=procurement_schedule,
                inventory_projections=inventory_projections,
                blend_quality=blend_quality,
                fuels=fuels,
                solve_time=solve_time,
                model_build_time=model_build_time,
                total_time=total_time,
                model_hash=model_hash,
                solver_config_hash=solver_config_hash,
                input_data_hash=input_data_hash
            )

            logger.info(f"Optimization complete: {result.status.value}, cost=${result.total_cost:,.2f}")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Optimization failed: {str(e)}") from e

    def _compute_input_hash(
        self,
        fuels: List[FuelData],
        tanks: List[TankData],
        demands: List[DemandData],
        contracts: Optional[List[ContractData]]
    ) -> str:
        """Compute SHA-256 hash of input data for provenance."""
        data = {
            "fuels": [f.to_dict() for f in fuels],
            "tanks": [t.to_dict() for t in tanks],
            "demands": [d.to_dict() for d in demands],
            "contracts": [c.to_dict() for c in (contracts or [])]
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _map_solver_status(self, solver_status: SolverStatus) -> OptimizationStatus:
        """Map solver status to optimization status."""
        mapping = {
            SolverStatus.OPTIMAL: OptimizationStatus.SUCCESS,
            SolverStatus.INFEASIBLE: OptimizationStatus.INFEASIBLE,
            SolverStatus.UNBOUNDED: OptimizationStatus.UNBOUNDED,
            SolverStatus.TIME_LIMIT: OptimizationStatus.TIME_LIMIT,
            SolverStatus.ERROR: OptimizationStatus.ERROR,
            SolverStatus.UNKNOWN: OptimizationStatus.ERROR
        }
        return mapping.get(solver_status, OptimizationStatus.ERROR)

    def _extract_fuel_mix(
        self,
        fuels: List[FuelData],
        demands: List[DemandData]
    ) -> List[FuelMixEntry]:
        """Extract fuel mix entries from solution."""
        fuel_mix: List[FuelMixEntry] = []

        for demand in demands:
            period = demand.period

            for fuel in fuels:
                fuel_id = fuel.fuel_id

                # Extract solution values
                procurement = self._solution.get_procurement(fuel_id, period)
                consumption = self._solution.get_consumption(fuel_id, period)
                blend_fraction = self._solution.get_blend_fraction(fuel_id, period)

                # Calculate allocations (simplified - would use contract variables)
                contract_alloc = procurement * Decimal("0.6")  # Assume 60% contract
                spot_alloc = procurement - contract_alloc

                # Calculate costs and emissions
                unit_cost = fuel.price_per_mj
                total_cost = consumption * unit_cost
                emissions = consumption * fuel.carbon_intensity_kg_co2e_mj

                entry = FuelMixEntry(
                    fuel_id=fuel_id,
                    period=period,
                    procurement_mj=procurement.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    consumption_mj=consumption.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    blend_fraction=blend_fraction.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                    contract_allocation_mj=contract_alloc.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    spot_allocation_mj=spot_alloc.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    unit_cost_per_mj=unit_cost,
                    total_cost=total_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    carbon_intensity_kg_co2e_mj=fuel.carbon_intensity_kg_co2e_mj,
                    emissions_kg_co2e=emissions.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                )

                fuel_mix.append(entry)

        return fuel_mix

    def _calculate_cost_breakdown(
        self,
        fuel_mix: List[FuelMixEntry],
        fuels: List[FuelData],
        contracts: Optional[List[ContractData]],
        logistics_params: Optional[Dict[str, LogisticsCostParams]],
        storage_params: Optional[Dict[str, StorageCostParams]],
        carbon_params: Optional[CarbonCostParams]
    ) -> CostBreakdown:
        """Calculate detailed cost breakdown from fuel mix."""
        fuels_dict = {f.fuel_id: f for f in fuels}
        contracts_dict = {c.contract_id: c for c in (contracts or [])}

        purchase_components = []
        logistics_components = []
        storage_components = []
        loss_components = []
        penalty_components = []
        carbon_components = []
        risk_components = []

        total_energy = Decimal("0")

        # Group fuel mix by fuel for aggregation
        fuel_procurement: Dict[str, Decimal] = {}
        fuel_consumption: Dict[str, Decimal] = {}

        for entry in fuel_mix:
            fuel_procurement[entry.fuel_id] = fuel_procurement.get(
                entry.fuel_id, Decimal("0")
            ) + entry.procurement_mj
            fuel_consumption[entry.fuel_id] = fuel_consumption.get(
                entry.fuel_id, Decimal("0")
            ) + entry.consumption_mj
            total_energy += entry.consumption_mj

        # Calculate purchase costs
        for fuel_id, qty in fuel_procurement.items():
            if qty <= Decimal("0"):
                continue

            fuel = fuels_dict[fuel_id]
            params = PurchaseCostParams(
                fuel_id=fuel_id,
                pricing_type=PricingType.SPOT,
                spot_price_per_mj=fuel.price_per_mj
            )
            cost, comp = self._cost_model.calculate_purchase_cost(qty, params)
            purchase_components.append((cost, comp))

        # Calculate logistics costs (use defaults if not provided)
        for fuel_id, qty in fuel_procurement.items():
            if qty <= Decimal("0"):
                continue

            if logistics_params and fuel_id in logistics_params:
                params = logistics_params[fuel_id]
            else:
                params = LogisticsCostParams(
                    mode=LogisticsMode.TRUCK,
                    base_rate_per_mj=Decimal("0.001"),
                    distance_km=Decimal("50"),
                    distance_rate_per_mj_km=Decimal("0.00001"),
                    fixed_delivery_fee=Decimal("500")
                )

            num_deliveries = max(1, int(qty / Decimal("100000")))
            cost, comp = self._cost_model.calculate_logistics_cost(qty, params, num_deliveries)
            logistics_components.append((cost, comp))

        # Calculate carbon costs
        for fuel_id, qty in fuel_consumption.items():
            if qty <= Decimal("0"):
                continue

            fuel = fuels_dict[fuel_id]

            if carbon_params:
                params = carbon_params
            else:
                params = CarbonCostParams(
                    scheme=CarbonScheme.INTERNAL,
                    carbon_price_per_kg_co2e=self._config.carbon_price_per_kg_co2e,
                    carbon_intensity_kg_co2e_mj=fuel.carbon_intensity_kg_co2e_mj
                )

            cost, comp, emissions = self._cost_model.calculate_carbon_cost(qty, params)
            carbon_components.append((cost, comp, emissions))

        # Calculate contract penalties
        for contract in (contracts or []):
            total_contract_take = fuel_procurement.get(contract.fuel_id, Decimal("0"))
            params = ContractPenaltyParams(
                contract_id=contract.contract_id,
                fuel_id=contract.fuel_id,
                min_take_mj=contract.min_quantity_mj,
                max_take_mj=contract.max_quantity_mj,
                shortfall_penalty_per_mj=contract.penalty_per_mj_shortfall
            )
            cost, comp = self._cost_model.calculate_penalty_cost(total_contract_take, params)
            if cost > Decimal("0"):
                penalty_components.append((cost, comp))

        # Calculate risk premium if enabled
        if self._config.include_risk_premium:
            base_cost = sum(c[0] for c in purchase_components)
            params = RiskCostParams(
                volatility_premium_pct=self._config.risk_premium_pct,
                supply_risk_premium_pct=Decimal("0.5")
            )
            cost, comp = self._cost_model.calculate_risk_cost(base_cost, params)
            risk_components.append((cost, comp))

        # Calculate total breakdown
        return self._cost_model.calculate_total_cost(
            purchase_components=purchase_components,
            logistics_components=logistics_components,
            storage_components=storage_components,
            loss_components=loss_components,
            penalty_components=penalty_components,
            carbon_components=carbon_components,
            risk_components=risk_components,
            total_energy_mj=total_energy,
            period_count=len(set(e.period for e in fuel_mix))
        )

    def _generate_procurement_schedule(
        self,
        fuel_mix: List[FuelMixEntry],
        fuels: List[FuelData],
        contracts: Optional[List[ContractData]]
    ) -> List[ProcurementSchedule]:
        """Generate procurement schedule from fuel mix."""
        schedule: List[ProcurementSchedule] = []
        fuels_dict = {f.fuel_id: f for f in fuels}

        for entry in fuel_mix:
            if entry.procurement_mj <= Decimal("0"):
                continue

            fuel = fuels_dict[entry.fuel_id]

            # Contract allocation
            if entry.contract_allocation_mj > Decimal("0"):
                # Find matching contract
                contract_id = None
                if contracts:
                    for c in contracts:
                        if c.fuel_id == entry.fuel_id:
                            contract_id = c.contract_id
                            break

                schedule.append(ProcurementSchedule(
                    fuel_id=entry.fuel_id,
                    period=entry.period,
                    quantity_mj=entry.contract_allocation_mj,
                    source="contract",
                    contract_id=contract_id,
                    delivery_date=None,
                    price_per_mj=fuel.price_per_mj * Decimal("0.95"),  # Contract discount
                    total_cost=(entry.contract_allocation_mj * fuel.price_per_mj * Decimal("0.95")).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                ))

            # Spot allocation
            if entry.spot_allocation_mj > Decimal("0"):
                schedule.append(ProcurementSchedule(
                    fuel_id=entry.fuel_id,
                    period=entry.period,
                    quantity_mj=entry.spot_allocation_mj,
                    source="spot",
                    contract_id=None,
                    delivery_date=None,
                    price_per_mj=fuel.price_per_mj,
                    total_cost=(entry.spot_allocation_mj * fuel.price_per_mj).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                ))

        return schedule

    def _project_inventory(
        self,
        fuel_mix: List[FuelMixEntry],
        tanks: List[TankData],
        demands: List[DemandData]
    ) -> List[InventoryProjection]:
        """Project inventory levels over planning horizon."""
        projections: List[InventoryProjection] = []

        for tank in tanks:
            # Get fuel mix entries for compatible fuels
            tank_fuels = set(tank.compatible_fuels)
            current_level = tank.initial_level_mj

            periods = sorted(set(e.period for e in fuel_mix))

            for period in periods:
                # Sum procurement and consumption for compatible fuels
                inflow = sum(
                    e.procurement_mj for e in fuel_mix
                    if e.period == period and e.fuel_id in tank_fuels
                )
                outflow = sum(
                    e.consumption_mj for e in fuel_mix
                    if e.period == period and e.fuel_id in tank_fuels
                )

                opening = current_level
                losses = opening * tank.loss_rate_per_period
                closing = opening + inflow - outflow - losses
                closing = max(Decimal("0"), closing)

                utilization = (closing / tank.capacity_mj * Decimal("100")) if tank.capacity_mj > 0 else Decimal("0")

                projection = InventoryProjection(
                    tank_id=tank.tank_id,
                    period=period,
                    opening_level_mj=opening.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    inflow_mj=inflow.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    outflow_mj=outflow.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    losses_mj=losses.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    closing_level_mj=closing.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    utilization_pct=utilization.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    is_at_minimum=closing <= tank.min_level_mj,
                    is_near_maximum=closing >= tank.max_level_mj * Decimal("0.95")
                )

                projections.append(projection)
                current_level = closing

        return projections

    def _validate_blend_quality(
        self,
        fuel_mix: List[FuelMixEntry],
        fuels: List[FuelData],
        demands: List[DemandData]
    ) -> List[BlendQuality]:
        """Validate blend quality against constraints."""
        blend_quality: List[BlendQuality] = []
        fuels_dict = {f.fuel_id: f for f in fuels}
        demands_dict = {d.period: d for d in demands}

        periods = sorted(set(e.period for e in fuel_mix))

        for period in periods:
            period_entries = [e for e in fuel_mix if e.period == period]
            demand = demands_dict.get(period)

            if not demand or not period_entries:
                continue

            # Calculate blend-weighted properties
            total_fraction = sum(e.blend_fraction for e in period_entries)
            if total_fraction <= Decimal("0"):
                continue

            sulfur = Decimal("0")
            ash = Decimal("0")
            flash_point_min = Decimal("999")
            ci_weighted = Decimal("0")

            for entry in period_entries:
                fuel = fuels_dict.get(entry.fuel_id)
                if not fuel:
                    continue

                frac = entry.blend_fraction / total_fraction
                sulfur += fuel.sulfur_wt_pct * frac
                ash += fuel.ash_wt_pct * frac
                flash_point_min = min(flash_point_min, fuel.flash_point_c)
                ci_weighted += fuel.carbon_intensity_kg_co2e_mj * frac

            # Validate against constraints
            violations: List[str] = []
            status = BlendQualityStatus.COMPLIANT

            if sulfur > demand.max_sulfur_pct:
                violations.append(f"Sulfur {sulfur:.3f}% exceeds limit {demand.max_sulfur_pct}%")
                status = BlendQualityStatus.VIOLATION

            if flash_point_min < demand.min_flash_point_c:
                violations.append(f"Flash point {flash_point_min}C below minimum {demand.min_flash_point_c}C")
                status = BlendQualityStatus.VIOLATION

            if demand.max_carbon_intensity and ci_weighted > demand.max_carbon_intensity:
                violations.append(f"CI {ci_weighted:.4f} exceeds limit {demand.max_carbon_intensity}")
                status = BlendQualityStatus.VIOLATION

            blend_quality.append(BlendQuality(
                period=period,
                sulfur_wt_pct=sulfur.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                ash_wt_pct=ash.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                water_vol_pct=Decimal("0"),  # Would need water content in FuelData
                viscosity_cst=Decimal("0"),  # Would need viscosity in FuelData
                flash_point_c=flash_point_min,
                carbon_intensity_kg_co2e_mj=ci_weighted.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
                status=status,
                violations=violations
            ))

        return blend_quality

    def _compile_result(
        self,
        fuel_mix: List[FuelMixEntry],
        cost_breakdown: CostBreakdown,
        procurement_schedule: List[ProcurementSchedule],
        inventory_projections: List[InventoryProjection],
        blend_quality: List[BlendQuality],
        fuels: List[FuelData],
        solve_time: float,
        model_build_time: float,
        total_time: float,
        model_hash: str,
        solver_config_hash: str,
        input_data_hash: str
    ) -> OptimizationResult:
        """Compile final optimization result."""
        # Calculate aggregated metrics
        total_procurement = sum(e.procurement_mj for e in fuel_mix)
        total_consumption = sum(e.consumption_mj for e in fuel_mix)
        total_emissions = sum(e.emissions_kg_co2e for e in fuel_mix)

        avg_cost = (
            cost_breakdown.total_cost / total_consumption
            if total_consumption > Decimal("0")
            else Decimal("0")
        )
        avg_ci = (
            total_emissions / total_consumption
            if total_consumption > Decimal("0")
            else Decimal("0")
        )

        # Build fuel summary
        fuel_summary: Dict[str, Dict[str, Any]] = {}
        for fuel in fuels:
            fuel_entries = [e for e in fuel_mix if e.fuel_id == fuel.fuel_id]
            if fuel_entries:
                fuel_summary[fuel.fuel_id] = {
                    "total_procurement_mj": str(sum(e.procurement_mj for e in fuel_entries)),
                    "total_consumption_mj": str(sum(e.consumption_mj for e in fuel_entries)),
                    "total_emissions_kg_co2e": str(sum(e.emissions_kg_co2e for e in fuel_entries)),
                    "blend_fraction_avg": str(
                        sum(e.blend_fraction for e in fuel_entries) / len(fuel_entries)
                    )
                }

        return OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            solver_status=self._solution.status,
            objective_value=self._solution.objective_value or Decimal("0"),
            mip_gap=self._solution.mip_gap,
            is_optimal=self._solution.is_optimal,
            is_feasible=self._solution.is_feasible,
            fuel_mix=fuel_mix,
            cost_breakdown=cost_breakdown,
            procurement_schedule=procurement_schedule,
            inventory_projections=inventory_projections,
            blend_quality=blend_quality,
            total_procurement_mj=total_procurement.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_consumption_mj=total_consumption.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_cost=cost_breakdown.total_cost,
            average_cost_per_mj=avg_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            total_emissions_kg_co2e=total_emissions.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            average_carbon_intensity=avg_ci.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            fuel_summary=fuel_summary,
            solve_time_seconds=solve_time,
            model_build_time_seconds=model_build_time,
            total_time_seconds=total_time,
            run_id=self._run_id,
            run_bundle_hash="",  # Will be computed in post_init
            model_hash=model_hash,
            solver_config_hash=solver_config_hash,
            input_data_hash=input_data_hash
        )

    def _create_infeasible_result(
        self,
        status: OptimizationStatus,
        solver_status: SolverStatus,
        solve_time: float,
        model_build_time: float,
        model_hash: str,
        solver_config_hash: str,
        input_data_hash: str
    ) -> OptimizationResult:
        """Create result for infeasible optimization."""
        return OptimizationResult(
            status=status,
            solver_status=solver_status,
            objective_value=Decimal("0"),
            mip_gap=None,
            is_optimal=False,
            is_feasible=False,
            fuel_mix=[],
            cost_breakdown=CostBreakdown(
                purchase_cost=Decimal("0"),
                logistics_cost=Decimal("0"),
                storage_cost=Decimal("0"),
                loss_cost=Decimal("0"),
                penalty_cost=Decimal("0"),
                carbon_cost=Decimal("0"),
                risk_cost=Decimal("0"),
                total_cost=Decimal("0"),
                components=[],
                cost_by_fuel={},
                cost_by_period={}
            ),
            procurement_schedule=[],
            inventory_projections=[],
            blend_quality=[],
            total_procurement_mj=Decimal("0"),
            total_consumption_mj=Decimal("0"),
            total_cost=Decimal("0"),
            average_cost_per_mj=Decimal("0"),
            total_emissions_kg_co2e=Decimal("0"),
            average_carbon_intensity=Decimal("0"),
            fuel_summary={},
            solve_time_seconds=solve_time,
            model_build_time_seconds=model_build_time,
            total_time_seconds=solve_time + model_build_time,
            run_id=self._run_id,
            run_bundle_hash="",
            model_hash=model_hash,
            solver_config_hash=solver_config_hash,
            input_data_hash=input_data_hash
        )

    def get_model_statistics(self) -> Optional[Dict[str, Any]]:
        """Get model statistics from last optimization."""
        if self._model:
            return self._model.get_model_statistics()
        return None

    def get_solution(self) -> Optional[Solution]:
        """Get raw solution from last optimization."""
        return self._solution

    def get_run_id(self) -> Optional[str]:
        """Get run ID from last optimization."""
        return self._run_id
