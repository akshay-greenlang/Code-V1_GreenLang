# -*- coding: utf-8 -*-
"""
GreenLang Industrial Operations Optimization Agents
====================================================

Operations agents for industrial energy and emissions optimization:
    - GL-OPS-IND-001: Energy Management Agent
    - GL-OPS-IND-002: Process Optimization Agent
    - GL-OPS-IND-003: Maintenance Optimization Agent
    - GL-OPS-IND-004: Production Scheduling Agent
    - GL-OPS-IND-005: Waste Management Agent

Features:
    - Real-time energy optimization
    - Process efficiency improvements
    - Predictive maintenance for emissions
    - Production scheduling for low-carbon periods

Author: GreenLang Framework Team
Version: 1.0.0
"""

from .sector_agents import (
    IndustrialOperationsBaseAgent,
    OperationsInput,
    OperationsOutput,
    EnergyManagementAgent,
    ProcessOptimizationAgent,
    MaintenanceOptimizationAgent,
    ProductionSchedulingAgent,
    WasteManagementAgent,
)

__all__ = [
    "IndustrialOperationsBaseAgent",
    "OperationsInput",
    "OperationsOutput",
    "EnergyManagementAgent",
    "ProcessOptimizationAgent",
    "MaintenanceOptimizationAgent",
    "ProductionSchedulingAgent",
    "WasteManagementAgent",
]

__version__ = "1.0.0"
