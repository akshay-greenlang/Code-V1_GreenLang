# -*- coding: utf-8 -*-
"""
GreenLang Agent Template Library
================================

Production-ready agent templates for industrial process heat applications.
This module provides 20+ templates across 4 categories:

Categories:
- Efficiency Agents (5): Boiler, Furnace, Heat Exchanger, Steam System, Heat Recovery
- Safety Agents (5): Combustion Safety, SIL Assessment, Interlock Validation, PSV Sizing, HAZOP
- Emissions Agents (5): Emissions Calculator, Excess Air Optimizer, Stack Monitoring, Carbon Footprint, Decarbonization
- Maintenance Agents (5): Thermal Imaging, Steam Trap Survey, Insulation Audit, Predictive Maintenance, Tube Life

All templates follow GreenLang's AgentSpec DSL format and include:
- Input/output schemas with Pydantic validation
- Required formulas with standards citations
- Safety constraints and guardrails
- Zero-hallucination guarantees
- Provenance tracking (SHA-256)

Example:
    >>> from greenlang.templates import TemplateRegistry
    >>> registry = TemplateRegistry()
    >>> template = registry.get_template("boiler_efficiency_agent")
    >>> agent = registry.instantiate_agent(template, config)

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from .template_registry import (
    TemplateRegistry,
    TemplateMetadata,
    TemplateCategory,
    TemplateVersion,
    TemplateVariable,
    TemplateLoader,
    TemplateValidator,
    TemplateComposer,
    VariableInterpolator,
    get_template_registry,
)

from .base_templates import (
    BaseAgentTemplate,
    EfficiencyAgentTemplate,
    SafetyAgentTemplate,
    EmissionsAgentTemplate,
    MaintenanceAgentTemplate,
    AgentInputSchema,
    AgentOutputSchema,
    FormulaReference,
    StandardReference,
    SafetyConstraint,
    ZeroHallucinationConfig,
)

__version__ = "1.0.0"
__author__ = "GreenLang Engineering Team"

__all__ = [
    # Registry
    "TemplateRegistry",
    "TemplateMetadata",
    "TemplateCategory",
    "TemplateVersion",
    "TemplateVariable",
    "TemplateLoader",
    "TemplateValidator",
    "TemplateComposer",
    "VariableInterpolator",
    "get_template_registry",
    # Base Templates
    "BaseAgentTemplate",
    "EfficiencyAgentTemplate",
    "SafetyAgentTemplate",
    "EmissionsAgentTemplate",
    "MaintenanceAgentTemplate",
    "AgentInputSchema",
    "AgentOutputSchema",
    "FormulaReference",
    "StandardReference",
    "SafetyConstraint",
    "ZeroHallucinationConfig",
]

# Template file paths (relative to templates directory)
TEMPLATE_FILES = {
    # Efficiency Agents
    "boiler_efficiency_agent": "efficiency/boiler_efficiency_agent.yaml",
    "furnace_efficiency_agent": "efficiency/furnace_efficiency_agent.yaml",
    "heat_exchanger_agent": "efficiency/heat_exchanger_agent.yaml",
    "steam_system_agent": "efficiency/steam_system_agent.yaml",
    "heat_recovery_agent": "efficiency/heat_recovery_agent.yaml",
    # Safety Agents
    "combustion_safety_agent": "safety/combustion_safety_agent.yaml",
    "sil_assessment_agent": "safety/sil_assessment_agent.yaml",
    "interlock_validation_agent": "safety/interlock_validation_agent.yaml",
    "psv_sizing_agent": "safety/psv_sizing_agent.yaml",
    "hazop_assistant_agent": "safety/hazop_assistant_agent.yaml",
    # Emissions Agents
    "emissions_calculator_agent": "emissions/emissions_calculator_agent.yaml",
    "excess_air_optimizer_agent": "emissions/excess_air_optimizer_agent.yaml",
    "stack_monitoring_agent": "emissions/stack_monitoring_agent.yaml",
    "carbon_footprint_agent": "emissions/carbon_footprint_agent.yaml",
    "decarbonization_agent": "emissions/decarbonization_agent.yaml",
    # Maintenance Agents
    "thermal_imaging_agent": "maintenance/thermal_imaging_agent.yaml",
    "steam_trap_survey_agent": "maintenance/steam_trap_survey_agent.yaml",
    "insulation_audit_agent": "maintenance/insulation_audit_agent.yaml",
    "predictive_maintenance_agent": "maintenance/predictive_maintenance_agent.yaml",
    "tube_life_agent": "maintenance/tube_life_agent.yaml",
}
