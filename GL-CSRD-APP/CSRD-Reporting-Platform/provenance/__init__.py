"""
Provenance Tracking Module
===========================

Complete audit trail and data lineage tracking for CSRD reports.

This module provides enterprise-grade provenance tracking for regulatory compliance.

Core Functions:
- hash_file: SHA-256 file integrity hashing
- hash_data: Hash arbitrary data
- capture_environment: Capture execution environment
- get_dependency_versions: Get package versions

Data Source Tracking:
- create_data_source: Create data source record

Calculation Lineage:
- track_calculation_lineage: Track calculation lineage

Provenance Records:
- create_provenance_record: Create provenance record

Graph Analysis:
- build_lineage_graph: Build dependency graph
- get_calculation_path: Get calculation path for metric

Serialization:
- serialize_provenance: Serialize to JSON
- save_provenance_json: Save to JSON file

Audit Package:
- create_audit_package: Create ZIP audit package
- generate_audit_report: Generate Markdown audit report

Models:
- DataSource: Data source model
- CalculationLineage: Calculation lineage model
- EnvironmentSnapshot: Environment snapshot model
- ProvenanceRecord: Complete provenance record model
"""

from .provenance_utils import (
    # Core functions
    hash_file,
    hash_data,
    capture_environment,
    get_dependency_versions,

    # Data source tracking
    create_data_source,

    # Calculation lineage
    track_calculation_lineage,

    # Provenance records
    create_provenance_record,

    # Graph analysis
    build_lineage_graph,
    get_calculation_path,

    # Serialization
    serialize_provenance,
    save_provenance_json,

    # Audit package
    create_audit_package,
    generate_audit_report,

    # Models
    DataSource,
    CalculationLineage,
    EnvironmentSnapshot,
    ProvenanceRecord,
)

__all__ = [
    # Core functions
    "hash_file",
    "hash_data",
    "capture_environment",
    "get_dependency_versions",

    # Data source tracking
    "create_data_source",

    # Calculation lineage
    "track_calculation_lineage",

    # Provenance records
    "create_provenance_record",

    # Graph analysis
    "build_lineage_graph",
    "get_calculation_path",

    # Serialization
    "serialize_provenance",
    "save_provenance_json",

    # Audit package
    "create_audit_package",
    "generate_audit_report",

    # Models
    "DataSource",
    "CalculationLineage",
    "EnvironmentSnapshot",
    "ProvenanceRecord",
]
