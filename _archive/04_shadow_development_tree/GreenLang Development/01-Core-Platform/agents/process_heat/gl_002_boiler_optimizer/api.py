"""
GL-002 BoilerOptimizer Agent - API Module

REST API endpoints for boiler optimization operations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BoilerAPIResponse(BaseModel):
    """Standard API response."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None


class EfficiencyRequest(BaseModel):
    """Request for efficiency calculation."""

    boiler_id: str
    fuel_type: str = "natural_gas"
    fuel_flow_rate: float
    steam_flow_rate_lb_hr: float
    steam_pressure_psig: float
    steam_temperature_f: Optional[float] = None
    feedwater_temperature_f: float = 200.0
    flue_gas_o2_pct: float
    flue_gas_co_ppm: float = 0.0
    flue_gas_temperature_f: float
    blowdown_rate_pct: float = 2.0


class OptimizationRequest(BaseModel):
    """Request for optimization recommendations."""

    boiler_id: str
    target_efficiency_pct: Optional[float] = None
    optimization_priority: str = "efficiency"  # efficiency, emissions, cost


class BoilerAPIController:
    """
    REST API controller for BoilerOptimizer.

    Provides RESTful endpoints for boiler operations.
    """

    def __init__(self, optimizer_agent: Any) -> None:
        """Initialize API controller."""
        self._agent = optimizer_agent
        self._routes = self._setup_routes()

    def _setup_routes(self) -> Dict[str, Dict[str, callable]]:
        """Setup API routes."""
        return {
            "/api/v1/boilers/{boiler_id}/efficiency": {
                "POST": self.calculate_efficiency,
            },
            "/api/v1/boilers/{boiler_id}/optimize": {
                "POST": self.get_recommendations,
            },
            "/api/v1/boilers/{boiler_id}/status": {
                "GET": self.get_status,
            },
            "/api/v1/boilers/{boiler_id}/kpis": {
                "GET": self.get_kpis,
            },
            "/api/v1/boilers/{boiler_id}/combustion": {
                "POST": self.analyze_combustion,
            },
            "/api/v1/boilers/{boiler_id}/steam": {
                "POST": self.analyze_steam,
            },
            "/api/v1/boilers/{boiler_id}/economizer": {
                "POST": self.analyze_economizer,
            },
        }

    async def calculate_efficiency(
        self,
        boiler_id: str,
        request: EfficiencyRequest,
    ) -> BoilerAPIResponse:
        """POST /api/v1/boilers/{boiler_id}/efficiency"""
        try:
            from greenlang.agents.process_heat.gl_002_boiler_optimizer.schemas import (
                BoilerInput,
            )

            input_data = BoilerInput(
                boiler_id=boiler_id,
                fuel_type=request.fuel_type,
                fuel_flow_rate=request.fuel_flow_rate,
                steam_flow_rate_lb_hr=request.steam_flow_rate_lb_hr,
                steam_pressure_psig=request.steam_pressure_psig,
                steam_temperature_f=request.steam_temperature_f,
                feedwater_temperature_f=request.feedwater_temperature_f,
                feedwater_flow_rate_lb_hr=request.steam_flow_rate_lb_hr * 1.05,
                flue_gas_o2_pct=request.flue_gas_o2_pct,
                flue_gas_co_ppm=request.flue_gas_co_ppm,
                flue_gas_temperature_f=request.flue_gas_temperature_f,
                blowdown_rate_pct=request.blowdown_rate_pct,
                load_pct=75.0,
            )

            result = self._agent.process(input_data)

            return BoilerAPIResponse(
                success=True,
                data={
                    "efficiency": result.efficiency.dict(),
                    "kpis": result.kpis,
                    "provenance_hash": result.provenance_hash,
                }
            )

        except Exception as e:
            logger.error(f"Efficiency calculation failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))

    async def get_recommendations(
        self,
        boiler_id: str,
        request: OptimizationRequest,
    ) -> BoilerAPIResponse:
        """POST /api/v1/boilers/{boiler_id}/optimize"""
        try:
            # Use last processed result for recommendations
            return BoilerAPIResponse(
                success=True,
                data={
                    "message": "Use /efficiency endpoint with full data for recommendations"
                }
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))

    async def get_status(self, boiler_id: str) -> BoilerAPIResponse:
        """GET /api/v1/boilers/{boiler_id}/status"""
        try:
            return BoilerAPIResponse(
                success=True,
                data={
                    "boiler_id": boiler_id,
                    "agent_state": self._agent.state.name,
                    "last_efficiency": self._agent._last_efficiency,
                    "config": {
                        "fuel_type": self._agent.boiler_config.fuel_type.value,
                        "design_capacity": self._agent.boiler_config.design_capacity_mmbtu_hr,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))

    async def get_kpis(self, boiler_id: str) -> BoilerAPIResponse:
        """GET /api/v1/boilers/{boiler_id}/kpis"""
        try:
            # Return trend data
            return BoilerAPIResponse(
                success=True,
                data={
                    "boiler_id": boiler_id,
                    "efficiency_trend": self._agent._efficiency_trend[-20:],
                    "trend_length": len(self._agent._efficiency_trend),
                }
            )

        except Exception as e:
            logger.error(f"KPI retrieval failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))

    async def analyze_combustion(
        self,
        boiler_id: str,
        data: Dict[str, Any],
    ) -> BoilerAPIResponse:
        """POST /api/v1/boilers/{boiler_id}/combustion"""
        try:
            from greenlang.agents.process_heat.gl_002_boiler_optimizer.combustion import (
                CombustionOptimizer,
                CombustionInput,
            )

            optimizer = CombustionOptimizer(
                fuel_type=data.get("fuel_type", "natural_gas"),
                target_o2_pct=self._agent.boiler_config.combustion.o2_setpoint_pct,
            )

            input_data = CombustionInput(**data)
            result = optimizer.optimize(input_data)

            return BoilerAPIResponse(success=True, data=result.dict())

        except Exception as e:
            logger.error(f"Combustion analysis failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))

    async def analyze_steam(
        self,
        boiler_id: str,
        data: Dict[str, Any],
    ) -> BoilerAPIResponse:
        """POST /api/v1/boilers/{boiler_id}/steam"""
        try:
            from greenlang.agents.process_heat.gl_002_boiler_optimizer.steam import (
                SteamSystemAnalyzer,
                SteamInput,
            )

            analyzer = SteamSystemAnalyzer(
                design_pressure_psig=self._agent.boiler_config.steam.design_pressure_psig,
                design_blowdown_pct=self._agent.boiler_config.steam.blowdown_rate_pct,
            )

            input_data = SteamInput(**data)
            result = analyzer.analyze(input_data)

            return BoilerAPIResponse(success=True, data=result.dict())

        except Exception as e:
            logger.error(f"Steam analysis failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))

    async def analyze_economizer(
        self,
        boiler_id: str,
        data: Dict[str, Any],
    ) -> BoilerAPIResponse:
        """POST /api/v1/boilers/{boiler_id}/economizer"""
        try:
            from greenlang.agents.process_heat.gl_002_boiler_optimizer.economizer import (
                EconomizerOptimizer,
                EconomizerInput,
            )

            optimizer = EconomizerOptimizer(
                design_duty_btu_hr=self._agent.boiler_config.economizer.design_duty_btu_hr,
                design_effectiveness=self._agent.boiler_config.economizer.design_effectiveness,
            )

            input_data = EconomizerInput(**data)
            result = optimizer.analyze(input_data)

            return BoilerAPIResponse(success=True, data=result.dict())

        except Exception as e:
            logger.error(f"Economizer analysis failed: {e}")
            return BoilerAPIResponse(success=False, error=str(e))
