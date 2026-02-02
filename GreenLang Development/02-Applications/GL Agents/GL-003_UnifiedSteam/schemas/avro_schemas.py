"""
Avro Schema Definitions for GL-003 UNIFIEDSTEAM

Provides Avro schema definitions for Schema Registry integration.

Author: GL-003 Data Engineering Team
"""

from typing import Dict, Any


def get_raw_signal_avro() -> Dict[str, Any]:
    """
    Get Avro schema for raw signal messages.

    Returns:
        Avro schema dictionary
    """
    return {
        "type": "record",
        "name": "RawSignal",
        "namespace": "com.greenlang.gl003.steam",
        "doc": "Raw signal from OT system",
        "fields": [
            {
                "name": "ts",
                "type": {"type": "long", "logicalType": "timestamp-millis"},
                "doc": "Timestamp in milliseconds since epoch"
            },
            {
                "name": "site",
                "type": "string",
                "doc": "Site identifier"
            },
            {
                "name": "area",
                "type": "string",
                "doc": "Area/unit identifier"
            },
            {
                "name": "asset",
                "type": "string",
                "doc": "Asset identifier"
            },
            {
                "name": "tag",
                "type": "string",
                "doc": "Tag name"
            },
            {
                "name": "value",
                "type": "double",
                "doc": "Signal value"
            },
            {
                "name": "unit",
                "type": "string",
                "doc": "Engineering unit"
            },
            {
                "name": "quality",
                "type": {
                    "type": "record",
                    "name": "SignalQuality",
                    "fields": [
                        {
                            "name": "status",
                            "type": {
                                "type": "enum",
                                "name": "QualityStatus",
                                "symbols": ["GOOD", "UNCERTAIN", "BAD", "STALE", "NOT_CONNECTED"]
                            }
                        },
                        {
                            "name": "flags",
                            "type": {"type": "array", "items": "string"},
                            "default": []
                        }
                    ]
                }
            },
            {
                "name": "sensor",
                "type": ["null", {
                    "type": "record",
                    "name": "SensorMetadata",
                    "fields": [
                        {"name": "sensor_id", "type": "string"},
                        {"name": "sensor_type", "type": "string"},
                        {"name": "accuracy_pct_fs", "type": ["null", "double"], "default": None},
                        {"name": "calibration_date", "type": ["null", "string"], "default": None}
                    ]
                }],
                "default": None
            }
        ]
    }


def get_validated_signal_avro() -> Dict[str, Any]:
    """
    Get Avro schema for validated signal messages.

    Returns:
        Avro schema dictionary
    """
    return {
        "type": "record",
        "name": "ValidatedSignal",
        "namespace": "com.greenlang.gl003.steam",
        "doc": "Validated and normalized signal",
        "fields": [
            {
                "name": "ts",
                "type": {"type": "long", "logicalType": "timestamp-millis"}
            },
            {"name": "site", "type": "string"},
            {"name": "area", "type": "string"},
            {"name": "asset", "type": "string"},
            {"name": "tag", "type": "string"},
            {"name": "value", "type": "double"},
            {"name": "unit", "type": "string"},
            {"name": "original_value", "type": "double"},
            {"name": "original_unit", "type": "string"},
            {
                "name": "status",
                "type": {
                    "type": "enum",
                    "name": "ValidationStatus",
                    "symbols": [
                        "VALID", "RANGE_WARNING", "RANGE_ERROR",
                        "RATE_WARNING", "RATE_ERROR",
                        "CONSISTENCY_ERROR", "QUARANTINED"
                    ]
                }
            },
            {
                "name": "quality_flags",
                "type": {
                    "type": "record",
                    "name": "QualityFlags",
                    "fields": [
                        {"name": "in_range", "type": "boolean", "default": True},
                        {"name": "rate_ok", "type": "boolean", "default": True},
                        {"name": "consistent", "type": "boolean", "default": True},
                        {"name": "flags", "type": {"type": "array", "items": "string"}, "default": []}
                    ]
                }
            },
            {"name": "sensor_accuracy_pct", "type": ["null", "double"], "default": None},
            {"name": "derived_uncertainty", "type": ["null", "double"], "default": None},
            {"name": "validation_hash", "type": ["null", "string"], "default": None}
        ]
    }


def get_feature_avro() -> Dict[str, Any]:
    """
    Get Avro schema for feature messages.

    Returns:
        Avro schema dictionary
    """
    return {
        "type": "record",
        "name": "FeatureSet",
        "namespace": "com.greenlang.gl003.steam",
        "doc": "Engineered features for ML models",
        "fields": [
            {
                "name": "ts",
                "type": {"type": "long", "logicalType": "timestamp-millis"}
            },
            {"name": "site", "type": "string"},
            {"name": "area", "type": "string"},
            {"name": "asset", "type": "string"},
            {"name": "feature_set_id", "type": "string"},
            {"name": "feature_version", "type": "string"},
            {
                "name": "features",
                "type": {"type": "map", "values": "double"},
                "doc": "Feature name to value mapping"
            },
            {
                "name": "feature_quality",
                "type": {"type": "map", "values": "double"},
                "default": {},
                "doc": "Feature name to quality score mapping"
            },
            {"name": "computation_hash", "type": ["null", "string"], "default": None}
        ]
    }


def get_computed_avro() -> Dict[str, Any]:
    """
    Get Avro schema for computed properties messages.

    Returns:
        Avro schema dictionary
    """
    return {
        "type": "record",
        "name": "ComputedProperties",
        "namespace": "com.greenlang.gl003.steam",
        "doc": "Computed thermodynamic properties and KPIs",
        "fields": [
            {
                "name": "ts",
                "type": {"type": "long", "logicalType": "timestamp-millis"}
            },
            {"name": "site", "type": "string"},
            {"name": "area", "type": "string"},
            {"name": "computation_id", "type": "string"},
            {
                "name": "steam_properties",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "SteamProperties",
                        "fields": [
                            {"name": "asset", "type": "string"},
                            {"name": "pressure_kpa", "type": "double"},
                            {"name": "temperature_c", "type": "double"},
                            {"name": "enthalpy_kj_kg", "type": "double"},
                            {"name": "entropy_kj_kg_k", "type": "double"},
                            {"name": "specific_volume_m3_kg", "type": "double"},
                            {"name": "density_kg_m3", "type": "double"},
                            {"name": "saturation_temp_c", "type": "double"},
                            {"name": "superheat_c", "type": "double"},
                            {"name": "steam_quality", "type": ["null", "double"], "default": None},
                            {"name": "if97_region", "type": "int"},
                            {"name": "computation_hash", "type": "string"}
                        ]
                    }
                },
                "default": []
            },
            {
                "name": "enthalpy_balances",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "EnthalpyBalance",
                        "fields": [
                            {"name": "balance_node", "type": "string"},
                            {"name": "mass_in_kg_s", "type": "double"},
                            {"name": "mass_out_kg_s", "type": "double"},
                            {"name": "mass_balance_kg_s", "type": "double"},
                            {"name": "enthalpy_in_kw", "type": "double"},
                            {"name": "enthalpy_out_kw", "type": "double"},
                            {"name": "heat_loss_kw", "type": "double"},
                            {"name": "energy_balance_kw", "type": "double"},
                            {"name": "mass_closure_pct", "type": "double"},
                            {"name": "energy_closure_pct", "type": "double"},
                            {"name": "computation_hash", "type": "string"}
                        ]
                    }
                },
                "default": []
            },
            {
                "name": "kpis",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "KPI",
                        "fields": [
                            {"name": "kpi_type", "type": "string"},
                            {"name": "value", "type": "double"},
                            {"name": "unit", "type": "string"},
                            {"name": "target", "type": ["null", "double"], "default": None},
                            {"name": "vs_target_pct", "type": ["null", "double"], "default": None}
                        ]
                    }
                },
                "default": []
            }
        ]
    }


def get_recommendation_avro() -> Dict[str, Any]:
    """
    Get Avro schema for recommendation messages.

    Returns:
        Avro schema dictionary
    """
    return {
        "type": "record",
        "name": "Recommendation",
        "namespace": "com.greenlang.gl003.steam",
        "doc": "Optimization recommendation",
        "fields": [
            {
                "name": "ts",
                "type": {"type": "long", "logicalType": "timestamp-millis"}
            },
            {"name": "recommendation_id", "type": "string"},
            {"name": "site", "type": "string"},
            {"name": "area", "type": "string"},
            {"name": "asset", "type": "string"},
            {
                "name": "recommendation_type",
                "type": {
                    "type": "enum",
                    "name": "RecommendationType",
                    "symbols": [
                        "desuperheater_setpoint", "trap_inspection", "trap_replacement",
                        "prv_adjustment", "condensate_routing", "insulation_repair",
                        "blowdown_adjustment", "header_pressure", "maintenance_schedule",
                        "operational_change"
                    ]
                }
            },
            {
                "name": "priority",
                "type": {
                    "type": "enum",
                    "name": "Priority",
                    "symbols": ["critical", "high", "medium", "low", "informational"]
                }
            },
            {"name": "action", "type": "string"},
            {"name": "rationale", "type": "string"},
            {
                "name": "impact",
                "type": {
                    "type": "record",
                    "name": "ImpactEstimate",
                    "fields": [
                        {"name": "steam_savings_kg_hr", "type": ["null", "double"], "default": None},
                        {"name": "energy_savings_kw", "type": ["null", "double"], "default": None},
                        {"name": "cost_savings_usd", "type": ["null", "double"], "default": None},
                        {"name": "co2e_reduction_kg", "type": ["null", "double"], "default": None},
                        {"name": "uncertainty_pct", "type": ["null", "double"], "default": None}
                    ]
                }
            },
            {"name": "confidence_score", "type": "double"},
            {"name": "constraints_checked", "type": {"type": "array", "items": "string"}, "default": []},
            {"name": "safety_envelope_ok", "type": "boolean", "default": True},
            {"name": "verification_plan", "type": "string", "default": ""},
            {
                "name": "explainability",
                "type": {
                    "type": "record",
                    "name": "Explainability",
                    "fields": [
                        {"name": "primary_drivers", "type": {"type": "array", "items": "string"}, "default": []},
                        {"name": "supporting_signals", "type": {"type": "array", "items": "string"}, "default": []}
                    ]
                }
            },
            {
                "name": "disposition",
                "type": {
                    "type": "enum",
                    "name": "Disposition",
                    "symbols": ["pending", "accepted", "rejected", "implemented"]
                },
                "default": "pending"
            }
        ]
    }


def get_event_avro() -> Dict[str, Any]:
    """
    Get Avro schema for event messages.

    Returns:
        Avro schema dictionary
    """
    return {
        "type": "record",
        "name": "Event",
        "namespace": "com.greenlang.gl003.steam",
        "doc": "System event",
        "fields": [
            {
                "name": "ts",
                "type": {"type": "long", "logicalType": "timestamp-millis"}
            },
            {"name": "event_id", "type": "string"},
            {"name": "site", "type": "string"},
            {"name": "area", "type": "string"},
            {
                "name": "event_type",
                "type": {
                    "type": "enum",
                    "name": "EventType",
                    "symbols": [
                        "alarm", "maintenance", "setpoint_change",
                        "mode_change", "calibration", "configuration", "system"
                    ]
                }
            },
            {"name": "source", "type": "string"},
            {"name": "description", "type": "string"},
            {
                "name": "details",
                "type": {"type": "map", "values": "string"},
                "default": {}
            }
        ]
    }


# Schema version information
SCHEMA_VERSIONS = {
    "RawSignal": "1.0.0",
    "ValidatedSignal": "1.0.0",
    "FeatureSet": "1.0.0",
    "ComputedProperties": "1.0.0",
    "Recommendation": "1.0.0",
    "Event": "1.0.0",
}


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get all Avro schemas.

    Returns:
        Dictionary mapping schema name to schema definition
    """
    return {
        "RawSignal": get_raw_signal_avro(),
        "ValidatedSignal": get_validated_signal_avro(),
        "FeatureSet": get_feature_avro(),
        "ComputedProperties": get_computed_avro(),
        "Recommendation": get_recommendation_avro(),
        "Event": get_event_avro(),
    }
