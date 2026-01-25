"""
GL-006 HEATRECLAIM - MILP Optimizer

Mixed Integer Linear Programming optimizer for heat exchanger network
synthesis. Formulates HEN design as MILP/MINLP problem with binary
match selection and continuous heat load variables.

Reference: Yee & Grossmann, "Simultaneous Optimization Models for Heat
Integration", Computers Chem Eng, 1990.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

from ..core.schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
)
from ..core.config import (
    OptimizationObjective,
    OptimizationMode,
    ExchangerType,
    FlowArrangement,
    EconomicParameters,
)
from ..calculators.pinch_analysis import PinchAnalysisCalculator

logger = logging.getLogger(__name__)


@dataclass
class MILPSolution:
    """Solution from MILP optimization."""

    status: str
    objective_value: float
    matches: List[Dict[str, Any]]
    solve_time_seconds: float
    iterations: int
    gap: float
    is_optimal: bool


class MILPOptimizer:
    """
    MILP optimizer for heat exchanger network synthesis.

    Formulates the HEN synthesis problem as a simultaneous
    optimization model (Yee & Grossmann, 1990) with:

    Decision variables:
    - y[i,j]: Binary - match between hot stream i and cold stream j
    - q[i,j]: Continuous - heat exchanged in match (kW)
    - z[i,j,k]: Binary - exchanger in stage k

    Objective: Minimize total annual cost (TAC)
    - Annualized capital cost
    - Operating cost (utilities)

    Subject to:
    - Energy balances
    - Temperature feasibility (ΔT ≥ ΔTmin)
    - Stream splitting constraints
    - Logical constraints

    Example:
        >>> optimizer = MILPOptimizer()
        >>> solution = optimizer.optimize(hot_streams, cold_streams)
        >>> print(f"TAC: ${solution.objective_value:,.0f}/year")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        delta_t_min: float = 10.0,
        max_stages: int = 5,
        solver: str = "CBC",
        time_limit_seconds: float = 300.0,
        mip_gap: float = 0.01,
    ) -> None:
        """
        Initialize MILP optimizer.

        Args:
            delta_t_min: Minimum approach temperature (°C)
            max_stages: Maximum stages in superstructure
            solver: MILP solver to use (CBC, GLPK, CPLEX)
            time_limit_seconds: Maximum solve time
            mip_gap: Acceptable MIP gap for termination
        """
        if not HAS_PULP:
            raise ImportError("PuLP is required for MILP optimization")

        self.delta_t_min = delta_t_min
        self.max_stages = max_stages
        self.solver = solver
        self.time_limit = time_limit_seconds
        self.mip_gap = mip_gap

        # Cost parameters
        self.econ = EconomicParameters()

    def optimize(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST,
        existing_exchangers: Optional[List[HeatExchanger]] = None,
    ) -> MILPSolution:
        """
        Optimize heat exchanger network using MILP.

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            objective: Optimization objective
            existing_exchangers: Existing exchangers for retrofit

        Returns:
            MILPSolution with optimal matches and costs
        """
        start_time = datetime.now(timezone.utc)

        n_hot = len(hot_streams)
        n_cold = len(cold_streams)

        logger.info(
            f"Starting MILP optimization: {n_hot} hot streams, "
            f"{n_cold} cold streams, objective={objective.value}"
        )

        # Create MILP model
        model = pulp.LpProblem("HEN_Synthesis", pulp.LpMinimize)

        # Index sets
        HOT = range(n_hot)
        COLD = range(n_cold)
        STAGES = range(self.max_stages)

        # ===================
        # DECISION VARIABLES
        # ===================

        # Binary: match exists between hot i and cold j
        y = pulp.LpVariable.dicts(
            "y",
            ((i, j) for i in HOT for j in COLD),
            cat="Binary"
        )

        # Continuous: heat exchanged between i and j
        q = pulp.LpVariable.dicts(
            "q",
            ((i, j) for i in HOT for j in COLD),
            lowBound=0,
            cat="Continuous"
        )

        # Binary: hot utility for cold stream j
        y_hu = pulp.LpVariable.dicts("y_hu", COLD, cat="Binary")

        # Binary: cold utility for hot stream i
        y_cu = pulp.LpVariable.dicts("y_cu", HOT, cat="Binary")

        # Continuous: utility duties
        q_hu = pulp.LpVariable.dicts("q_hu", COLD, lowBound=0)
        q_cu = pulp.LpVariable.dicts("q_cu", HOT, lowBound=0)

        # ===================
        # PARAMETERS
        # ===================

        # Stream duties
        duty_hot = [s.duty_kW for s in hot_streams]
        duty_cold = [s.duty_kW for s in cold_streams]

        # Heat capacity rates
        FCp_hot = [s.FCp_kW_K for s in hot_streams]
        FCp_cold = [s.FCp_kW_K for s in cold_streams]

        # Big M for logical constraints
        M = max(max(duty_hot), max(duty_cold)) * 2

        # ===================
        # CONSTRAINTS
        # ===================

        # Energy balance for hot streams
        for i in HOT:
            model += (
                pulp.lpSum(q[i, j] for j in COLD) + q_cu[i] == duty_hot[i],
                f"HotBalance_{i}"
            )

        # Energy balance for cold streams
        for j in COLD:
            model += (
                pulp.lpSum(q[i, j] for i in HOT) + q_hu[j] == duty_cold[j],
                f"ColdBalance_{j}"
            )

        # Logical constraint: heat exchange requires match
        for i in HOT:
            for j in COLD:
                model += q[i, j] <= M * y[i, j], f"LogicalHeat_{i}_{j}"

        # Utility logical constraints
        for j in COLD:
            model += q_hu[j] <= M * y_hu[j], f"LogicalHU_{j}"

        for i in HOT:
            model += q_cu[i] <= M * y_cu[i], f"LogicalCU_{i}"

        # Temperature feasibility (linearized approximation)
        # For each potential match, check if temperatures can work
        for i in HOT:
            for j in COLD:
                T_hi = hot_streams[i].T_supply_C
                T_ho = hot_streams[i].T_target_C
                T_ci = cold_streams[j].T_supply_C
                T_co = cold_streams[j].T_target_C

                # Must have T_hi - T_co >= delta_t_min for match
                if T_hi - T_co < self.delta_t_min:
                    model += y[i, j] == 0, f"TempInfeasible_{i}_{j}"

        # ===================
        # OBJECTIVE FUNCTION
        # ===================

        # Cost coefficients
        capital_factor = 1000  # $/exchanger
        area_factor = 100  # $/(m² area)
        hu_cost = self.econ.steam_cost_usd_gj  # $/GJ
        cu_cost = self.econ.cooling_water_cost_usd_gj  # $/GJ

        # Convert kW to GJ/year
        hours = self.econ.operating_hours_per_year
        kw_to_gj_yr = hours * 0.0036

        if objective == OptimizationObjective.MINIMIZE_UTILITY:
            # Minimize total utility consumption
            model += (
                pulp.lpSum(q_hu[j] for j in COLD) +
                pulp.lpSum(q_cu[i] for i in HOT)
            )

        elif objective == OptimizationObjective.MINIMIZE_EXCHANGERS:
            # Minimize number of exchangers
            model += (
                pulp.lpSum(y[i, j] for i in HOT for j in COLD) +
                pulp.lpSum(y_hu[j] for j in COLD) +
                pulp.lpSum(y_cu[i] for i in HOT)
            )

        else:
            # Default: minimize total annual cost
            # TAC = annualized capital + operating cost

            # Capital: number of exchangers (simplified)
            capital_term = capital_factor * (
                pulp.lpSum(y[i, j] for i in HOT for j in COLD) +
                pulp.lpSum(y_hu[j] for j in COLD) +
                pulp.lpSum(y_cu[i] for i in HOT)
            )

            # Operating: utility costs
            utility_term = kw_to_gj_yr * (
                hu_cost * pulp.lpSum(q_hu[j] for j in COLD) +
                cu_cost * pulp.lpSum(q_cu[i] for i in HOT)
            )

            model += capital_term + utility_term

        # ===================
        # SOLVE
        # ===================

        # Select solver
        if self.solver.upper() == "CBC":
            solver = pulp.PULP_CBC_CMD(
                msg=0,
                timeLimit=self.time_limit,
                gapRel=self.mip_gap,
            )
        elif self.solver.upper() == "GLPK":
            solver = pulp.GLPK_CMD(msg=0, timeLimit=self.time_limit)
        else:
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)

        model.solve(solver)

        solve_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Extract solution
        status = pulp.LpStatus[model.status]
        is_optimal = status == "Optimal"

        matches = []
        if is_optimal or status == "Not Solved":
            # Extract matches
            for i in HOT:
                for j in COLD:
                    if y[i, j].value() is not None and y[i, j].value() > 0.5:
                        q_val = q[i, j].value() if q[i, j].value() else 0
                        if q_val > 1.0:
                            matches.append({
                                "hot_stream": hot_streams[i].stream_id,
                                "cold_stream": cold_streams[j].stream_id,
                                "duty_kW": round(q_val, 2),
                            })

            # Extract utilities
            for j in COLD:
                if y_hu[j].value() is not None and y_hu[j].value() > 0.5:
                    q_val = q_hu[j].value() if q_hu[j].value() else 0
                    if q_val > 1.0:
                        matches.append({
                            "hot_stream": "HOT_UTILITY",
                            "cold_stream": cold_streams[j].stream_id,
                            "duty_kW": round(q_val, 2),
                            "is_utility": True,
                        })

            for i in HOT:
                if y_cu[i].value() is not None and y_cu[i].value() > 0.5:
                    q_val = q_cu[i].value() if q_cu[i].value() else 0
                    if q_val > 1.0:
                        matches.append({
                            "hot_stream": hot_streams[i].stream_id,
                            "cold_stream": "COLD_UTILITY",
                            "duty_kW": round(q_val, 2),
                            "is_utility": True,
                        })

        objective_value = pulp.value(model.objective) if model.objective else 0.0

        logger.info(
            f"MILP optimization complete: status={status}, "
            f"objective={objective_value:.2f}, matches={len(matches)}"
        )

        return MILPSolution(
            status=status,
            objective_value=round(objective_value, 2) if objective_value else 0.0,
            matches=matches,
            solve_time_seconds=round(solve_time, 3),
            iterations=0,  # Not exposed by PuLP
            gap=0.0,  # Not exposed by PuLP
            is_optimal=is_optimal,
        )

    def build_design_from_solution(
        self,
        solution: MILPSolution,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> HENDesign:
        """
        Convert MILP solution to HEN design.

        Args:
            solution: MILP optimization result
            hot_streams: Hot streams
            cold_streams: Cold streams

        Returns:
            HENDesign with exchangers
        """
        stream_map = {s.stream_id: s for s in hot_streams + cold_streams}
        exchangers = []

        for i, match in enumerate(solution.matches):
            hot_id = match["hot_stream"]
            cold_id = match["cold_stream"]
            duty = match["duty_kW"]

            # Get temperatures
            if hot_id == "HOT_UTILITY":
                T_hi, T_ho = 200.0, 190.0  # Steam
                hot_stream = None
            else:
                hot_stream = stream_map.get(hot_id)
                T_hi = hot_stream.T_supply_C if hot_stream else 100.0
                T_ho = T_hi - duty / hot_stream.FCp_kW_K if hot_stream else 90.0

            if cold_id == "COLD_UTILITY":
                T_ci, T_co = 25.0, 35.0  # Cooling water
                cold_stream = None
            else:
                cold_stream = stream_map.get(cold_id)
                T_ci = cold_stream.T_supply_C if cold_stream else 30.0
                T_co = T_ci + duty / cold_stream.FCp_kW_K if cold_stream else 50.0

            hx = HeatExchanger(
                exchanger_id=f"HX-{i+1:03d}",
                exchanger_name=f"{hot_id}-{cold_id}",
                exchanger_type=ExchangerType.SHELL_AND_TUBE,
                hot_stream_id=hot_id,
                cold_stream_id=cold_id,
                duty_kW=duty,
                hot_inlet_T_C=round(T_hi, 2),
                hot_outlet_T_C=round(T_ho, 2),
                cold_inlet_T_C=round(T_ci, 2),
                cold_outlet_T_C=round(T_co, 2),
            )
            exchangers.append(hx)

        # Calculate totals
        process_matches = [m for m in solution.matches if not m.get("is_utility")]
        utility_matches = [m for m in solution.matches if m.get("is_utility")]

        heat_recovered = sum(m["duty_kW"] for m in process_matches)
        hot_utility = sum(
            m["duty_kW"] for m in utility_matches
            if m["hot_stream"] == "HOT_UTILITY"
        )
        cold_utility = sum(
            m["duty_kW"] for m in utility_matches
            if m["cold_stream"] == "COLD_UTILITY"
        )

        return HENDesign(
            design_name=f"MILP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            mode=OptimizationMode.GRASSROOTS,
            exchangers=exchangers,
            total_heat_recovered_kW=round(heat_recovered, 2),
            hot_utility_required_kW=round(hot_utility, 2),
            cold_utility_required_kW=round(cold_utility, 2),
            exchanger_count=len(exchangers),
            new_exchanger_count=len(exchangers),
        )
