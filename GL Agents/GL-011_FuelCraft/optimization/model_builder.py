"""
GL-011 FuelCraft - Optimization Model Builder

LP/MILP model construction for fuel procurement optimization.

Decision Variables:
- x_{i,t}: Procure quantity of fuel i in period t (MJ)
- y_{i,t}: Consumed/withdrawn fuel i in period t
- s_{k,t}: Inventory in tank k at period t
- b_{i,t}: Blend fraction of fuel i in period t
- z_{c,t}: Contract commitment decisions (binary)

Objective:
Minimize: Purchase cost + Logistics cost + Penalties + Carbon cost

Constraints:
1. Demand satisfaction
2. Inventory balance
3. Tank/flow limits
4. Blend quality limits
5. Contract constraints
6. Safety constraints
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json


class VariableType(Enum):
    """Type of optimization variable."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class ConstraintSense(Enum):
    """Constraint sense."""
    LE = "<="  # Less than or equal
    GE = ">="  # Greater than or equal
    EQ = "=="  # Equal


class ObjectiveType(Enum):
    """Objective function type."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class OptimizationVariable:
    """
    Single optimization variable definition.
    """
    name: str
    indices: Tuple[str, ...]  # e.g., ("fuel_i", "period_t")
    var_type: VariableType
    lower_bound: Decimal
    upper_bound: Decimal
    description: str = ""

    def get_full_name(self, index_values: Tuple[Any, ...]) -> str:
        """Get fully indexed variable name."""
        index_str = "_".join(str(v) for v in index_values)
        return f"{self.name}_{index_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "indices": self.indices,
            "var_type": self.var_type.value,
            "lower_bound": str(self.lower_bound),
            "upper_bound": str(self.upper_bound),
            "description": self.description
        }


@dataclass
class Constraint:
    """
    Single constraint definition.
    """
    name: str
    expression: str  # Linear expression as string
    sense: ConstraintSense
    rhs: Decimal  # Right-hand side
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "sense": self.sense.value,
            "rhs": str(self.rhs),
            "description": self.description
        }


@dataclass
class FuelData:
    """
    Fuel data for optimization.
    """
    fuel_id: str
    fuel_type: str
    lhv_mj_kg: Decimal
    density_kg_m3: Decimal
    price_per_mj: Decimal
    carbon_intensity_kg_co2e_mj: Decimal
    # Quality properties
    sulfur_wt_pct: Decimal
    ash_wt_pct: Decimal
    viscosity_cst: Decimal
    flash_point_c: Decimal
    # Availability
    min_order_mj: Decimal
    max_order_mj: Decimal
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fuel_id": self.fuel_id,
            "fuel_type": self.fuel_type,
            "lhv_mj_kg": str(self.lhv_mj_kg),
            "density_kg_m3": str(self.density_kg_m3),
            "price_per_mj": str(self.price_per_mj),
            "carbon_intensity_kg_co2e_mj": str(self.carbon_intensity_kg_co2e_mj),
            "sulfur_wt_pct": str(self.sulfur_wt_pct),
            "ash_wt_pct": str(self.ash_wt_pct),
            "viscosity_cst": str(self.viscosity_cst),
            "flash_point_c": str(self.flash_point_c),
            "min_order_mj": str(self.min_order_mj),
            "max_order_mj": str(self.max_order_mj),
            "available": self.available
        }


@dataclass
class TankData:
    """
    Tank data for optimization.
    """
    tank_id: str
    capacity_mj: Decimal
    min_level_mj: Decimal
    max_level_mj: Decimal
    initial_level_mj: Decimal
    loss_rate_per_period: Decimal
    compatible_fuels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tank_id": self.tank_id,
            "capacity_mj": str(self.capacity_mj),
            "min_level_mj": str(self.min_level_mj),
            "max_level_mj": str(self.max_level_mj),
            "initial_level_mj": str(self.initial_level_mj),
            "loss_rate_per_period": str(self.loss_rate_per_period),
            "compatible_fuels": self.compatible_fuels
        }


@dataclass
class ContractData:
    """
    Contract data for take-or-pay constraints.
    """
    contract_id: str
    fuel_id: str
    min_quantity_mj: Decimal  # Take-or-pay minimum
    max_quantity_mj: Decimal  # Contract cap
    price_per_mj: Decimal
    penalty_per_mj_shortfall: Decimal
    start_period: int
    end_period: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "fuel_id": self.fuel_id,
            "min_quantity_mj": str(self.min_quantity_mj),
            "max_quantity_mj": str(self.max_quantity_mj),
            "price_per_mj": str(self.price_per_mj),
            "penalty_per_mj_shortfall": str(self.penalty_per_mj_shortfall),
            "start_period": self.start_period,
            "end_period": self.end_period
        }


@dataclass
class DemandData:
    """
    Energy demand for each period.
    """
    period: int
    demand_mj: Decimal
    max_sulfur_pct: Decimal
    max_carbon_intensity: Optional[Decimal] = None
    min_flash_point_c: Decimal = Decimal("60.0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "demand_mj": str(self.demand_mj),
            "max_sulfur_pct": str(self.max_sulfur_pct),
            "max_carbon_intensity": str(self.max_carbon_intensity) if self.max_carbon_intensity else None,
            "min_flash_point_c": str(self.min_flash_point_c)
        }


@dataclass
class ModelConfig:
    """
    Optimization model configuration.
    """
    model_name: str = "FuelProcurement"
    objective_type: ObjectiveType = ObjectiveType.MINIMIZE
    time_periods: int = 12
    carbon_price_per_kg_co2e: Decimal = Decimal("0.0")
    include_logistics: bool = True
    logistics_cost_per_mj: Decimal = Decimal("0.001")
    big_m: Decimal = Decimal("1e9")  # For indicator constraints
    tolerance: Decimal = Decimal("1e-6")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "objective_type": self.objective_type.value,
            "time_periods": self.time_periods,
            "carbon_price_per_kg_co2e": str(self.carbon_price_per_kg_co2e),
            "include_logistics": self.include_logistics,
            "logistics_cost_per_mj": str(self.logistics_cost_per_mj),
            "big_m": str(self.big_m),
            "tolerance": str(self.tolerance)
        }


class FuelOptimizationModel:
    """
    LP/MILP model builder for fuel procurement optimization.

    Builds a deterministic optimization model with:
    - Indexed decision variables
    - Linear constraints
    - Linear/quadratic objective

    The model can be exported to MPS format for solver input.
    """

    NAME: str = "FuelOptimizationModel"
    VERSION: str = "1.0.0"

    def __init__(
        self,
        config: ModelConfig,
        fuels: List[FuelData],
        tanks: List[TankData],
        demands: List[DemandData],
        contracts: Optional[List[ContractData]] = None
    ):
        """
        Initialize model builder.

        Args:
            config: Model configuration
            fuels: Available fuel data
            tanks: Storage tank data
            demands: Period demands
            contracts: Optional contract data
        """
        self.config = config
        self.fuels = {f.fuel_id: f for f in fuels}
        self.tanks = {t.tank_id: t for t in tanks}
        self.demands = {d.period: d for d in demands}
        self.contracts = {c.contract_id: c for c in (contracts or [])}

        # Model components
        self.variables: Dict[str, OptimizationVariable] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.objective_terms: List[Tuple[str, Decimal]] = []

        # Index sets
        self.fuel_ids = list(self.fuels.keys())
        self.tank_ids = list(self.tanks.keys())
        self.periods = list(range(1, config.time_periods + 1))
        self.contract_ids = list(self.contracts.keys())

        # Build model
        self._build_model()

    def _build_model(self) -> None:
        """Build complete optimization model."""
        self._create_variables()
        self._create_demand_constraints()
        self._create_inventory_constraints()
        self._create_blend_constraints()
        self._create_contract_constraints()
        self._create_safety_constraints()
        self._build_objective()

    def _create_variables(self) -> None:
        """Create all decision variables."""
        # x_{i,t}: Procurement quantity of fuel i in period t (MJ)
        for fuel_id in self.fuel_ids:
            fuel = self.fuels[fuel_id]
            for t in self.periods:
                var_name = f"x_{fuel_id}_{t}"
                self.variables[var_name] = OptimizationVariable(
                    name="x",
                    indices=(fuel_id, str(t)),
                    var_type=VariableType.CONTINUOUS,
                    lower_bound=Decimal("0"),
                    upper_bound=fuel.max_order_mj,
                    description=f"Procure {fuel_id} in period {t} (MJ)"
                )

        # y_{i,t}: Consumption/withdrawal of fuel i in period t (MJ)
        for fuel_id in self.fuel_ids:
            for t in self.periods:
                var_name = f"y_{fuel_id}_{t}"
                self.variables[var_name] = OptimizationVariable(
                    name="y",
                    indices=(fuel_id, str(t)),
                    var_type=VariableType.CONTINUOUS,
                    lower_bound=Decimal("0"),
                    upper_bound=self.config.big_m,
                    description=f"Consume {fuel_id} in period {t} (MJ)"
                )

        # s_{k,t}: Inventory in tank k at end of period t (MJ)
        for tank_id in self.tank_ids:
            tank = self.tanks[tank_id]
            for t in self.periods:
                var_name = f"s_{tank_id}_{t}"
                self.variables[var_name] = OptimizationVariable(
                    name="s",
                    indices=(tank_id, str(t)),
                    var_type=VariableType.CONTINUOUS,
                    lower_bound=tank.min_level_mj,
                    upper_bound=tank.max_level_mj,
                    description=f"Inventory in {tank_id} at period {t} (MJ)"
                )

        # b_{i,t}: Blend fraction of fuel i in period t
        for fuel_id in self.fuel_ids:
            for t in self.periods:
                var_name = f"b_{fuel_id}_{t}"
                self.variables[var_name] = OptimizationVariable(
                    name="b",
                    indices=(fuel_id, str(t)),
                    var_type=VariableType.CONTINUOUS,
                    lower_bound=Decimal("0"),
                    upper_bound=Decimal("1"),
                    description=f"Blend fraction of {fuel_id} in period {t}"
                )

        # z_{c,t}: Contract commitment (binary)
        for contract_id in self.contract_ids:
            contract = self.contracts[contract_id]
            for t in range(contract.start_period, contract.end_period + 1):
                var_name = f"z_{contract_id}_{t}"
                self.variables[var_name] = OptimizationVariable(
                    name="z",
                    indices=(contract_id, str(t)),
                    var_type=VariableType.BINARY,
                    lower_bound=Decimal("0"),
                    upper_bound=Decimal("1"),
                    description=f"Contract {contract_id} active in period {t}"
                )

    def _create_demand_constraints(self) -> None:
        """Create demand satisfaction constraints."""
        for t in self.periods:
            if t not in self.demands:
                continue

            demand = self.demands[t]

            # Sum of consumption = demand
            # sum_i(y_{i,t}) = D_t
            terms = [f"y_{fuel_id}_{t}" for fuel_id in self.fuel_ids]
            expr = " + ".join(terms)

            constraint_name = f"demand_satisfaction_{t}"
            self.constraints[constraint_name] = Constraint(
                name=constraint_name,
                expression=expr,
                sense=ConstraintSense.EQ,
                rhs=demand.demand_mj,
                description=f"Meet energy demand in period {t}"
            )

    def _create_inventory_constraints(self) -> None:
        """Create inventory balance constraints."""
        for tank_id in self.tank_ids:
            tank = self.tanks[tank_id]

            for t in self.periods:
                # s_{k,t} = s_{k,t-1} + inflow - outflow - losses
                # For simplicity, assume one fuel per tank or aggregated

                if t == 1:
                    # Initial inventory
                    prev_inventory = tank.initial_level_mj
                else:
                    prev_var = f"s_{tank_id}_{t-1}"
                    prev_inventory = prev_var

                # Loss factor
                loss_factor = Decimal("1") - tank.loss_rate_per_period

                # Inflow = sum of procurement for compatible fuels
                inflow_terms = []
                outflow_terms = []
                for fuel_id in tank.compatible_fuels:
                    if fuel_id in self.fuel_ids:
                        inflow_terms.append(f"x_{fuel_id}_{t}")
                        outflow_terms.append(f"y_{fuel_id}_{t}")

                # Build constraint
                # s_{k,t} = prev * loss_factor + sum(x) - sum(y)
                constraint_name = f"inventory_balance_{tank_id}_{t}"

                if t == 1:
                    lhs = f"s_{tank_id}_{t}"
                    rhs_value = prev_inventory * loss_factor
                    if inflow_terms:
                        lhs = f"s_{tank_id}_{t} - " + " - ".join(inflow_terms)
                        if outflow_terms:
                            lhs += " + " + " + ".join(outflow_terms)
                else:
                    # s_t - loss_factor*s_{t-1} - sum(x) + sum(y) = 0
                    lhs_parts = [f"s_{tank_id}_{t}", f"-{loss_factor}*s_{tank_id}_{t-1}"]
                    if inflow_terms:
                        lhs_parts.extend([f"-{term}" for term in inflow_terms])
                    if outflow_terms:
                        lhs_parts.extend([f"+{term}" for term in outflow_terms])
                    lhs = " ".join(lhs_parts)
                    rhs_value = Decimal("0")

                self.constraints[constraint_name] = Constraint(
                    name=constraint_name,
                    expression=lhs,
                    sense=ConstraintSense.EQ,
                    rhs=rhs_value,
                    description=f"Inventory balance for {tank_id} in period {t}"
                )

    def _create_blend_constraints(self) -> None:
        """Create blend fraction and quality constraints."""
        for t in self.periods:
            # Blend fractions sum to 1
            # sum_i(b_{i,t}) = 1
            terms = [f"b_{fuel_id}_{t}" for fuel_id in self.fuel_ids]
            expr = " + ".join(terms)

            constraint_name = f"blend_sum_{t}"
            self.constraints[constraint_name] = Constraint(
                name=constraint_name,
                expression=expr,
                sense=ConstraintSense.EQ,
                rhs=Decimal("1"),
                description=f"Blend fractions sum to 1 in period {t}"
            )

            # Link blend fraction to consumption
            # b_{i,t} * D_t = y_{i,t}  (for each fuel)
            # Linearized: y_{i,t} - D_t * b_{i,t} = 0
            if t in self.demands:
                demand = self.demands[t]
                for fuel_id in self.fuel_ids:
                    constraint_name = f"blend_link_{fuel_id}_{t}"
                    # y - D*b = 0
                    expr = f"y_{fuel_id}_{t} - {demand.demand_mj}*b_{fuel_id}_{t}"

                    self.constraints[constraint_name] = Constraint(
                        name=constraint_name,
                        expression=expr,
                        sense=ConstraintSense.EQ,
                        rhs=Decimal("0"),
                        description=f"Link blend fraction to consumption for {fuel_id}"
                    )

            # Sulfur constraint: sum_i(b_{i,t} * S_i) <= S_max
            if t in self.demands:
                demand = self.demands[t]
                sulfur_terms = []
                for fuel_id in self.fuel_ids:
                    fuel = self.fuels[fuel_id]
                    sulfur_terms.append(f"{fuel.sulfur_wt_pct}*b_{fuel_id}_{t}")

                expr = " + ".join(sulfur_terms)
                constraint_name = f"sulfur_limit_{t}"

                self.constraints[constraint_name] = Constraint(
                    name=constraint_name,
                    expression=expr,
                    sense=ConstraintSense.LE,
                    rhs=demand.max_sulfur_pct,
                    description=f"Blend sulfur limit in period {t}"
                )

    def _create_contract_constraints(self) -> None:
        """Create take-or-pay contract constraints."""
        for contract_id, contract in self.contracts.items():
            # Minimum quantity (take-or-pay)
            for t in range(contract.start_period, contract.end_period + 1):
                if t > self.config.time_periods:
                    continue

                # x_{fuel,t} >= min * z_{c,t}
                constraint_name = f"contract_min_{contract_id}_{t}"
                expr = f"x_{contract.fuel_id}_{t} - {contract.min_quantity_mj}*z_{contract_id}_{t}"

                self.constraints[constraint_name] = Constraint(
                    name=constraint_name,
                    expression=expr,
                    sense=ConstraintSense.GE,
                    rhs=Decimal("0"),
                    description=f"Contract minimum for {contract_id} in period {t}"
                )

                # x_{fuel,t} <= max * z_{c,t}
                constraint_name = f"contract_max_{contract_id}_{t}"
                expr = f"x_{contract.fuel_id}_{t} - {contract.max_quantity_mj}*z_{contract_id}_{t}"

                self.constraints[constraint_name] = Constraint(
                    name=constraint_name,
                    expression=expr,
                    sense=ConstraintSense.LE,
                    rhs=Decimal("0"),
                    description=f"Contract maximum for {contract_id} in period {t}"
                )

    def _create_safety_constraints(self) -> None:
        """Create safety constraints (flash point, etc.)."""
        for t in self.periods:
            if t not in self.demands:
                continue

            demand = self.demands[t]

            # Flash point constraint (conservative: minimum of blend components)
            # This is typically handled non-linearly, but we can use indicator constraints
            # For LP, we ensure all components meet minimum
            for fuel_id in self.fuel_ids:
                fuel = self.fuels[fuel_id]
                if fuel.flash_point_c < demand.min_flash_point_c:
                    # If fuel doesn't meet flash point, limit its use
                    # b_{i,t} <= some_limit if flash point too low
                    # For strict safety, we might exclude it entirely
                    pass  # Handled in preprocessing

    def _build_objective(self) -> None:
        """Build objective function."""
        # Minimize: Purchase cost + Logistics + Carbon cost

        for t in self.periods:
            for fuel_id in self.fuel_ids:
                fuel = self.fuels[fuel_id]

                # Purchase cost: price * x_{i,t}
                var_name = f"x_{fuel_id}_{t}"
                self.objective_terms.append((var_name, fuel.price_per_mj))

                # Logistics cost
                if self.config.include_logistics:
                    self.objective_terms.append(
                        (var_name, self.config.logistics_cost_per_mj)
                    )

                # Carbon cost
                if self.config.carbon_price_per_kg_co2e > Decimal("0"):
                    carbon_cost = fuel.carbon_intensity_kg_co2e_mj * \
                                 self.config.carbon_price_per_kg_co2e
                    self.objective_terms.append((var_name, carbon_cost))

        # Contract penalties (for shortfall)
        for contract_id, contract in self.contracts.items():
            for t in range(contract.start_period, contract.end_period + 1):
                if t > self.config.time_periods:
                    continue
                # Add penalty variable if needed
                # This requires slack variables for shortfall

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        num_continuous = sum(
            1 for v in self.variables.values()
            if v.var_type == VariableType.CONTINUOUS
        )
        num_integer = sum(
            1 for v in self.variables.values()
            if v.var_type == VariableType.INTEGER
        )
        num_binary = sum(
            1 for v in self.variables.values()
            if v.var_type == VariableType.BINARY
        )

        return {
            "num_variables": len(self.variables),
            "num_continuous": num_continuous,
            "num_integer": num_integer,
            "num_binary": num_binary,
            "num_constraints": len(self.constraints),
            "num_objective_terms": len(self.objective_terms),
            "num_fuels": len(self.fuel_ids),
            "num_tanks": len(self.tank_ids),
            "num_periods": len(self.periods),
            "num_contracts": len(self.contract_ids)
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export model to dictionary format."""
        return {
            "config": self.config.to_dict(),
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "constraints": {k: v.to_dict() for k, v in self.constraints.items()},
            "objective_terms": [(t[0], str(t[1])) for t in self.objective_terms],
            "statistics": self.get_model_statistics(),
            "provenance_hash": self._compute_hash()
        }

    def _compute_hash(self) -> str:
        """Compute model provenance hash."""
        data = {
            "config": self.config.to_dict(),
            "num_variables": len(self.variables),
            "num_constraints": len(self.constraints),
            "fuels": [f.to_dict() for f in self.fuels.values()],
            "tanks": [t.to_dict() for t in self.tanks.values()],
            "demands": [d.to_dict() for d in self.demands.values()]
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def export_mps(self, filepath: str) -> str:
        """
        Export model to MPS format for solver input.

        Returns the MPS file content as string.
        """
        lines = []

        # NAME section
        lines.append(f"NAME          {self.config.model_name}")

        # ROWS section
        lines.append("ROWS")
        # Objective row
        lines.append(" N  OBJ")
        # Constraint rows
        for name, constraint in self.constraints.items():
            sense_map = {
                ConstraintSense.LE: "L",
                ConstraintSense.GE: "G",
                ConstraintSense.EQ: "E"
            }
            lines.append(f" {sense_map[constraint.sense]}  {name}")

        # COLUMNS section
        lines.append("COLUMNS")
        # Group by variable
        for var_name, var in self.variables.items():
            # Objective coefficient
            obj_coef = Decimal("0")
            for term_name, coef in self.objective_terms:
                if term_name == var_name:
                    obj_coef += coef

            if obj_coef != Decimal("0"):
                lines.append(f"    {var_name}  OBJ  {float(obj_coef)}")

            # Constraint coefficients (simplified - would need expression parsing)
            # This is a placeholder - real implementation would parse expressions

        # RHS section
        lines.append("RHS")
        for name, constraint in self.constraints.items():
            lines.append(f"    RHS  {name}  {float(constraint.rhs)}")

        # BOUNDS section
        lines.append("BOUNDS")
        for var_name, var in self.variables.items():
            if var.var_type == VariableType.BINARY:
                lines.append(f" BV BOUND  {var_name}")
            elif var.var_type == VariableType.INTEGER:
                lines.append(f" LI BOUND  {var_name}  {float(var.lower_bound)}")
                lines.append(f" UI BOUND  {var_name}  {float(var.upper_bound)}")
            else:
                if var.lower_bound > Decimal("0"):
                    lines.append(f" LO BOUND  {var_name}  {float(var.lower_bound)}")
                if var.upper_bound < self.config.big_m:
                    lines.append(f" UP BOUND  {var_name}  {float(var.upper_bound)}")

        # ENDATA
        lines.append("ENDATA")

        mps_content = "\n".join(lines)

        # Write to file if path provided
        if filepath:
            with open(filepath, 'w') as f:
                f.write(mps_content)

        return mps_content
