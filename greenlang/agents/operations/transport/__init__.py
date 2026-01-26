# -*- coding: utf-8 -*-
"""
GreenLang Operations Transport Sector Agents
=============================================

Transport operations optimization agents for fleet management,
route optimization, and emissions monitoring.

Agents:
    GL-OPS-TRN-001 to GL-OPS-TRN-009
"""

from greenlang.agents.operations.transport.agents import (
    FleetOperationsOptimizerAgent,
    RouteOptimizationAgent,
    DriverBehaviorMonitoringAgent,
    ChargingScheduleOptimizerAgent,
    FuelEfficiencyMonitorAgent,
    MaintenancePredictionAgent,
    LoadOptimizationAgent,
    EmissionsTrackingAgent,
    VehicleDispatchAgent,
)

__all__ = [
    "FleetOperationsOptimizerAgent",
    "RouteOptimizationAgent",
    "DriverBehaviorMonitoringAgent",
    "ChargingScheduleOptimizerAgent",
    "FuelEfficiencyMonitorAgent",
    "MaintenancePredictionAgent",
    "LoadOptimizationAgent",
    "EmissionsTrackingAgent",
    "VehicleDispatchAgent",
]

AGENT_REGISTRY = {
    "GL-OPS-TRN-001": FleetOperationsOptimizerAgent,
    "GL-OPS-TRN-002": RouteOptimizationAgent,
    "GL-OPS-TRN-003": DriverBehaviorMonitoringAgent,
    "GL-OPS-TRN-004": ChargingScheduleOptimizerAgent,
    "GL-OPS-TRN-005": FuelEfficiencyMonitorAgent,
    "GL-OPS-TRN-006": MaintenancePredictionAgent,
    "GL-OPS-TRN-007": LoadOptimizationAgent,
    "GL-OPS-TRN-008": EmissionsTrackingAgent,
    "GL-OPS-TRN-009": VehicleDispatchAgent,
}
