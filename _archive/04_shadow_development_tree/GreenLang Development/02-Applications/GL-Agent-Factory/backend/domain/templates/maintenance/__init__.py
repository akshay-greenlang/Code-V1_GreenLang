# =============================================================================
# Maintenance Agent Templates Package
# GreenLang Agent Factory - Process Heat Applications
# =============================================================================
"""
Maintenance agent templates for predictive maintenance, condition monitoring,
and equipment life assessment in process heat applications.

Templates:
- thermal_imaging_agent: IR thermography analysis for hot spot detection
- steam_trap_survey_agent: Steam trap assessment and loss calculation
- insulation_audit_agent: Insulation defect detection and heat loss analysis
- predictive_maintenance_agent: Equipment health monitoring and failure prediction
- tube_life_agent: API 530/579 heater tube remaining life assessment

Standards:
- ISO 13381: Condition monitoring and diagnostics
- ISO 17359: General guidelines on machinery monitoring
- ISO 10816: Vibration evaluation standards
- API 530: Heater tube thickness calculation
- API 579-1/ASME FFS-1: Fitness-for-service
- API 691: Risk-based inspection
- ASTM E1934: Thermographic inspection
- DOE Steam Best Practices
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "GreenLang Engineering Team"

# Template file paths
TEMPLATE_DIR = Path(__file__).parent

MAINTENANCE_TEMPLATES = {
    "thermal_imaging_agent": TEMPLATE_DIR / "thermal_imaging_agent.yaml",
    "steam_trap_survey_agent": TEMPLATE_DIR / "steam_trap_survey_agent.yaml",
    "insulation_audit_agent": TEMPLATE_DIR / "insulation_audit_agent.yaml",
    "predictive_maintenance_agent": TEMPLATE_DIR / "predictive_maintenance_agent.yaml",
    "tube_life_agent": TEMPLATE_DIR / "tube_life_agent.yaml",
}

# Template categories and metadata
TEMPLATE_METADATA = {
    "thermal_imaging_agent": {
        "name": "IR Thermography Analysis Agent",
        "category": "condition_monitoring",
        "standards": ["ASTM E1934", "ISO 18434", "NETA MTS"],
        "equipment_types": ["boiler", "furnace", "heat_exchanger", "piping", "electrical"],
    },
    "steam_trap_survey_agent": {
        "name": "Steam Trap Assessment Agent",
        "category": "energy_loss",
        "standards": ["DOE Steam Best Practices", "ASME"],
        "equipment_types": ["steam_trap"],
    },
    "insulation_audit_agent": {
        "name": "Insulation Defect Detection Agent",
        "category": "energy_efficiency",
        "standards": ["ASTM C1055", "ASTM C680", "3EPlus", "NAIMA"],
        "equipment_types": ["pipe_insulation", "vessel_insulation", "tank_insulation"],
    },
    "predictive_maintenance_agent": {
        "name": "Equipment Health Monitoring Agent",
        "category": "predictive_maintenance",
        "standards": ["ISO 13381", "ISO 17359", "ISO 10816", "API 691"],
        "equipment_types": ["pump", "fan", "compressor", "motor", "turbine", "gearbox"],
    },
    "tube_life_agent": {
        "name": "Heater Tube Remaining Life Agent",
        "category": "fitness_for_service",
        "standards": ["API 530", "API 579-1/ASME FFS-1", "API 571", "API 560"],
        "equipment_types": ["fired_heater_tube", "boiler_tube", "reformer_tube"],
    },
}


def get_template_path(template_id: str) -> Path:
    """Get the file path for a maintenance template.

    Args:
        template_id: Template identifier

    Returns:
        Path to template YAML file

    Raises:
        KeyError: If template_id not found
    """
    if template_id not in MAINTENANCE_TEMPLATES:
        raise KeyError(f"Unknown maintenance template: {template_id}")
    return MAINTENANCE_TEMPLATES[template_id]


def list_templates() -> list:
    """List all available maintenance templates.

    Returns:
        List of template IDs
    """
    return list(MAINTENANCE_TEMPLATES.keys())


def get_template_metadata(template_id: str) -> dict:
    """Get metadata for a maintenance template.

    Args:
        template_id: Template identifier

    Returns:
        Template metadata dictionary

    Raises:
        KeyError: If template_id not found
    """
    if template_id not in TEMPLATE_METADATA:
        raise KeyError(f"Unknown maintenance template: {template_id}")
    return TEMPLATE_METADATA[template_id]


__all__ = [
    "MAINTENANCE_TEMPLATES",
    "TEMPLATE_METADATA",
    "get_template_path",
    "list_templates",
    "get_template_metadata",
]
