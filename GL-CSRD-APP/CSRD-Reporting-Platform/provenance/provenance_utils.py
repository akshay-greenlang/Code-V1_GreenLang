"""
CSRD/ESRS Digital Reporting Platform - Provenance Tracking Framework
=====================================================================

Enterprise-grade provenance tracking for regulatory compliance and audit trails.

This module provides comprehensive provenance tracking that meets EU CSRD regulatory
requirements for data integrity, traceability, and reproducibility.

Key Features:
- Complete calculation lineage tracking (formula → inputs → outputs)
- SHA-256 hashing for data integrity verification
- Environment snapshot capture (Python, OS, dependencies, LLM models)
- Data source tracking (file paths, sheet names, row/column references)
- Transformation history and dependency graphs
- Audit package generation (ZIP with complete provenance)
- JSON serialization for external auditors
- NetworkX graph support for dependency visualization

Architecture:
- Zero dependencies on agents (agents import this, not vice versa)
- Pydantic models for type safety
- Full type hints throughout
- Comprehensive logging
- CLI interface for testing

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import networkx as nx
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS - TYPE-SAFE DATA STRUCTURES
# ============================================================================

class DataSource(BaseModel):
    """
    Tracks the origin of a single data point.

    This model captures complete information about where a data point came from,
    enabling full traceability back to source documents.

    Attributes:
        source_id: Unique identifier for this data source
        source_type: Type of source (csv, json, excel, database, api, manual)
        file_path: Absolute path to source file (if applicable)
        file_hash: SHA-256 hash of source file
        sheet_name: Excel sheet name (if applicable)
        row_index: Row number in source file
        column_name: Column name or identifier
        cell_reference: Excel cell reference (e.g., "A1")
        table_name: Database table name (if applicable)
        query: SQL query or API endpoint (if applicable)
        timestamp: When this data was sourced
        metadata: Additional source metadata
    """
    source_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: str = Field(..., description="csv|json|excel|database|api|manual|calculated")
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    sheet_name: Optional[str] = None
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    cell_reference: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('source_type')
    def validate_source_type(cls, v):
        """Validate source type is one of the allowed values."""
        allowed = {'csv', 'json', 'excel', 'database', 'api', 'manual', 'calculated'}
        if v not in allowed:
            logger.warning(f"Source type '{v}' not in standard types: {allowed}")
        return v


class CalculationLineage(BaseModel):
    """
    Complete lineage for a single calculation.

    This model captures every detail needed to reproduce a calculation,
    including the formula, all inputs, intermediate steps, and final output.

    Attributes:
        lineage_id: Unique identifier for this calculation
        metric_code: ESRS metric code (e.g., "E1-1")
        metric_name: Human-readable metric name
        formula: Formula string or description
        formula_type: Type of calculation (sum, division, lookup_multiply, etc.)
        input_values: Dictionary of input variable names to values
        input_sources: List of data sources for each input
        intermediate_steps: List of calculation steps (for audit trail)
        output_value: Final calculated value
        output_unit: Unit of measurement
        data_sources: Complete list of all data sources used
        calculation_timestamp: When calculation was performed
        hash: SHA-256 hash of (formula + inputs) for verification
        agent_name: Name of agent that performed calculation
        dependencies: List of metric codes this calculation depends on
        metadata: Additional calculation metadata
    """
    lineage_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_code: str
    metric_name: str
    formula: str
    formula_type: str = Field(default="expression", description="sum|division|percentage|lookup_multiply|expression|direct")
    input_values: Dict[str, Any] = Field(default_factory=dict)
    input_sources: List[DataSource] = Field(default_factory=list)
    intermediate_steps: List[str] = Field(default_factory=list)
    output_value: Union[float, str, bool, int, None]
    output_unit: str
    data_sources: List[DataSource] = Field(default_factory=list)
    calculation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hash: str = ""
    agent_name: str = "CalculatorAgent"
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        """Initialize and auto-compute hash if not provided."""
        super().__init__(**data)
        if not self.hash:
            self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of formula and inputs for reproducibility.

        Returns:
            Hexadecimal hash string
        """
        # Create deterministic string representation
        hash_input = {
            "formula": self.formula,
            "input_values": {k: str(v) for k, v in sorted(self.input_values.items())},
            "formula_type": self.formula_type
        }
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()


class EnvironmentSnapshot(BaseModel):
    """
    Complete snapshot of execution environment.

    Captures all information needed to reproduce the exact execution environment,
    critical for regulatory compliance and reproducibility.

    Attributes:
        snapshot_id: Unique identifier
        timestamp: When snapshot was taken
        python_version: Full Python version string
        python_major: Python major version
        python_minor: Python minor version
        python_micro: Python micro version
        platform: Operating system (Linux, Windows, Darwin)
        platform_release: OS release version
        platform_version: OS version details
        machine: Machine architecture (x86_64, ARM64, etc.)
        processor: Processor type
        hostname: Machine hostname
        user: Username who ran the process
        working_directory: Current working directory
        package_versions: Dictionary of package name to version
        config_hash: SHA-256 hash of configuration file
        llm_models: LLM models used (for MaterialityAgent)
        metadata: Additional environment metadata
    """
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Python environment
    python_version: str
    python_major: int
    python_minor: int
    python_micro: int
    python_implementation: str
    python_compiler: str

    # System environment
    platform: str
    platform_release: str
    platform_version: str
    machine: str
    processor: str
    hostname: str

    # Process information
    user: str
    working_directory: str
    process_id: int

    # Dependencies
    package_versions: Dict[str, str] = Field(default_factory=dict)

    # Configuration
    config_hash: Optional[str] = None

    # LLM models (for MaterialityAgent)
    llm_models: Dict[str, str] = Field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    """
    Complete provenance record for an operation.

    This is the top-level provenance structure that captures all information
    about a single operation or transformation.

    Attributes:
        record_id: Unique record identifier
        timestamp: When this record was created
        agent_name: Name of agent that performed operation
        operation: Type of operation (calculate, validate, transform, etc.)
        inputs: Input data/parameters
        outputs: Output data/results
        metadata: Additional operation metadata
        environment: Environment snapshot
        calculation_lineage: Calculation lineage (if applicable)
        data_sources: Data sources used
        duration_seconds: How long operation took
        status: Operation status (success, warning, error)
        errors: List of errors encountered
        warnings: List of warnings
    """
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: str
    operation: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    environment: Optional[EnvironmentSnapshot] = None
    calculation_lineage: Optional[CalculationLineage] = None
    data_sources: List[DataSource] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    status: str = Field(default="success", description="success|warning|error")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# FILE INTEGRITY - SHA-256 HASHING
# ============================================================================

def hash_file(file_path: Union[str, Path], algorithm: str = "sha256") -> Dict[str, Any]:
    """
    Calculate cryptographic hash of a file for integrity verification.

    Creates a unique fingerprint of the input file that can be used to verify
    that the file hasn't been tampered with. Required for EU CSRD regulatory
    compliance and audit trails.

    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm (sha256, sha512, md5)

    Returns:
        Dictionary with hash details:
        {
            "file_path": str (absolute path),
            "file_name": str,
            "file_size_bytes": int,
            "hash_algorithm": str,
            "hash_value": str (hex),
            "hash_timestamp": str (ISO 8601),
            "human_readable_size": str
        }

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm not supported

    Example:
        >>> hash_info = hash_file("esg_data.csv")
        >>> print(f"SHA256: {hash_info['hash_value']}")
        >>> # Later, verify integrity:
        >>> new_hash = hash_file("esg_data.csv")
        >>> assert new_hash['hash_value'] == hash_info['hash_value']
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select hash algorithm
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Read file in chunks (memory-efficient for large files)
    chunk_size = 65536  # 64 KB chunks

    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    file_size = file_path.stat().st_size

    return {
        "file_path": str(file_path.absolute()),
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "hash_algorithm": algorithm.upper(),
        "hash_value": hash_value,
        "hash_timestamp": datetime.now(timezone.utc).isoformat(),
        "human_readable_size": _format_bytes(file_size)
    }


def hash_data(data: Any) -> str:
    """
    Calculate SHA-256 hash of arbitrary data.

    Converts data to deterministic JSON string and hashes it.
    Useful for hashing configurations, dictionaries, etc.

    Args:
        data: Data to hash (must be JSON-serializable)

    Returns:
        Hexadecimal hash string

    Example:
        >>> config = {"model": "gpt-4o", "temperature": 0.3}
        >>> hash1 = hash_data(config)
        >>> hash2 = hash_data(config)
        >>> assert hash1 == hash2  # Deterministic
    """
    # Convert to deterministic JSON string
    json_string = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_string.encode('utf-8')).hexdigest()


def _format_bytes(bytes_size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


# ============================================================================
# ENVIRONMENT CAPTURE
# ============================================================================

def capture_environment(
    config_path: Optional[Union[str, Path]] = None,
    llm_models: Optional[Dict[str, str]] = None
) -> EnvironmentSnapshot:
    """
    Capture complete execution environment for reproducibility.

    Records all relevant system and Python environment details needed to
    reproduce the exact execution environment. Critical for regulatory
    compliance and debugging.

    Args:
        config_path: Optional path to configuration file (will be hashed)
        llm_models: Optional dictionary of LLM models used
            (e.g., {"materiality": "gpt-4o", "narratives": "claude-3-5-sonnet"})

    Returns:
        EnvironmentSnapshot with complete environment details

    Example:
        >>> env = capture_environment(
        ...     config_path="config/csrd_config.yaml",
        ...     llm_models={"materiality": "gpt-4o"}
        ... )
        >>> print(f"Python: {env.python_major}.{env.python_minor}.{env.python_micro}")
        >>> print(f"OS: {env.platform} {env.platform_release}")
    """
    # Capture Python version
    py_version = sys.version_info

    # Capture system info
    try:
        hostname = platform.node()
    except:
        hostname = "unknown"

    try:
        processor = platform.processor()
    except:
        processor = "unknown"

    # Capture user
    user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"

    # Hash configuration file if provided
    config_hash = None
    if config_path:
        try:
            config_hash = hash_file(config_path)["hash_value"]
        except Exception as e:
            logger.warning(f"Could not hash config file: {e}")

    # Get package versions
    package_versions = get_dependency_versions()

    # Create snapshot
    snapshot = EnvironmentSnapshot(
        python_version=sys.version,
        python_major=py_version.major,
        python_minor=py_version.minor,
        python_micro=py_version.micro,
        python_implementation=platform.python_implementation(),
        python_compiler=platform.python_compiler(),
        platform=platform.system(),
        platform_release=platform.release(),
        platform_version=platform.version(),
        machine=platform.machine(),
        processor=processor,
        hostname=hostname,
        user=user,
        working_directory=os.getcwd(),
        process_id=os.getpid(),
        package_versions=package_versions,
        config_hash=config_hash,
        llm_models=llm_models or {}
    )

    logger.debug(f"Captured environment snapshot: Python {snapshot.python_major}.{snapshot.python_minor}.{snapshot.python_micro} on {snapshot.platform}")

    return snapshot


def get_dependency_versions() -> Dict[str, str]:
    """
    Get versions of all critical dependencies.

    Captures exact versions of all Python packages used in the pipeline.
    Essential for reproducibility and regulatory compliance.

    Returns:
        Dictionary mapping package name to version:
        {
            "pandas": "2.0.3",
            "pydantic": "2.1.1",
            ...
        }

    Example:
        >>> deps = get_dependency_versions()
        >>> print(f"Using pandas {deps.get('pandas', 'unknown')}")
    """
    dependencies = {}

    # Core dependencies for CSRD platform
    critical_packages = [
        "pandas",
        "pydantic",
        "jsonschema",
        "pyyaml",
        "networkx",
        "numpy",
        "openpyxl",
        "jinja2",
        "lxml"
    ]

    for package in critical_packages:
        try:
            # Try importlib.metadata (Python 3.8+)
            try:
                from importlib.metadata import version
                dependencies[package] = version(package)
            except ImportError:
                # Fallback to pkg_resources
                import pkg_resources
                dependencies[package] = pkg_resources.get_distribution(package).version
        except Exception:
            dependencies[package] = "unknown"

    return dependencies


# ============================================================================
# DATA SOURCE TRACKING
# ============================================================================

def create_data_source(
    source_type: str,
    file_path: Optional[str] = None,
    sheet_name: Optional[str] = None,
    row_index: Optional[int] = None,
    column_name: Optional[str] = None,
    **kwargs
) -> DataSource:
    """
    Create a data source record.

    Args:
        source_type: Type of source (csv, json, excel, database, etc.)
        file_path: Path to source file
        sheet_name: Excel sheet name
        row_index: Row number in source
        column_name: Column name
        **kwargs: Additional metadata

    Returns:
        DataSource instance

    Example:
        >>> source = create_data_source(
        ...     source_type="excel",
        ...     file_path="data/esg_data.xlsx",
        ...     sheet_name="Emissions",
        ...     row_index=5,
        ...     column_name="Scope1_tCO2e"
        ... )
    """
    # Hash file if path provided
    file_hash = None
    if file_path:
        try:
            file_hash = hash_file(file_path)["hash_value"]
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")

    return DataSource(
        source_type=source_type,
        file_path=file_path,
        file_hash=file_hash,
        sheet_name=sheet_name,
        row_index=row_index,
        column_name=column_name,
        metadata=kwargs
    )


# ============================================================================
# CALCULATION LINEAGE TRACKING
# ============================================================================

def track_calculation_lineage(
    metric_code: str,
    metric_name: str,
    formula: str,
    input_values: Dict[str, Any],
    output_value: Any,
    output_unit: str,
    formula_type: str = "expression",
    intermediate_steps: Optional[List[str]] = None,
    data_sources: Optional[List[DataSource]] = None,
    dependencies: Optional[List[str]] = None,
    agent_name: str = "CalculatorAgent",
    **metadata
) -> CalculationLineage:
    """
    Create a calculation lineage record.

    Tracks complete lineage of a calculation from inputs to output,
    including formula, intermediate steps, and data sources.

    Args:
        metric_code: ESRS metric code (e.g., "E1-1")
        metric_name: Human-readable metric name
        formula: Formula string
        input_values: Dictionary of input variable names to values
        output_value: Final calculated value
        output_unit: Unit of measurement
        formula_type: Type of calculation (sum, division, etc.)
        intermediate_steps: List of calculation steps
        data_sources: List of data sources for inputs
        dependencies: List of metric codes this depends on
        agent_name: Name of agent performing calculation
        **metadata: Additional metadata

    Returns:
        CalculationLineage instance

    Example:
        >>> lineage = track_calculation_lineage(
        ...     metric_code="E1-1",
        ...     metric_name="Total GHG Emissions",
        ...     formula="Scope1 + Scope2 + Scope3",
        ...     input_values={"Scope1": 1000, "Scope2": 500, "Scope3": 2000},
        ...     output_value=3500,
        ...     output_unit="tCO2e",
        ...     formula_type="sum",
        ...     intermediate_steps=["1000 + 500 + 2000 = 3500"]
        ... )
        >>> print(f"Hash: {lineage.hash}")
    """
    return CalculationLineage(
        metric_code=metric_code,
        metric_name=metric_name,
        formula=formula,
        formula_type=formula_type,
        input_values=input_values,
        intermediate_steps=intermediate_steps or [],
        output_value=output_value,
        output_unit=output_unit,
        data_sources=data_sources or [],
        dependencies=dependencies or [],
        agent_name=agent_name,
        metadata=metadata
    )


# ============================================================================
# PROVENANCE RECORD CREATION
# ============================================================================

def create_provenance_record(
    agent_name: str,
    operation: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    calculation_lineage: Optional[CalculationLineage] = None,
    data_sources: Optional[List[DataSource]] = None,
    environment: Optional[EnvironmentSnapshot] = None,
    duration_seconds: Optional[float] = None,
    status: str = "success",
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    **metadata
) -> ProvenanceRecord:
    """
    Create a provenance record for an operation.

    This is the main function for creating provenance records. It captures
    all information about an operation including inputs, outputs, environment,
    and any calculation lineage.

    Args:
        agent_name: Name of agent performing operation
        operation: Type of operation (calculate, validate, transform, etc.)
        inputs: Input data/parameters
        outputs: Output data/results
        calculation_lineage: Calculation lineage (if applicable)
        data_sources: Data sources used
        environment: Environment snapshot (will auto-capture if not provided)
        duration_seconds: How long operation took
        status: Operation status (success, warning, error)
        errors: List of errors encountered
        warnings: List of warnings
        **metadata: Additional metadata

    Returns:
        ProvenanceRecord instance

    Example:
        >>> record = create_provenance_record(
        ...     agent_name="IntakeAgent",
        ...     operation="validate_data",
        ...     inputs={"file": "esg_data.csv", "records": 100},
        ...     outputs={"valid": 95, "invalid": 5},
        ...     duration_seconds=2.5,
        ...     status="success"
        ... )
    """
    # Auto-capture environment if not provided
    if environment is None:
        environment = capture_environment()

    return ProvenanceRecord(
        agent_name=agent_name,
        operation=operation,
        inputs=inputs or {},
        outputs=outputs or {},
        calculation_lineage=calculation_lineage,
        data_sources=data_sources or [],
        environment=environment,
        duration_seconds=duration_seconds,
        status=status,
        errors=errors or [],
        warnings=warnings or [],
        metadata=metadata
    )


# ============================================================================
# LINEAGE GRAPH CONSTRUCTION
# ============================================================================

def build_lineage_graph(
    calculation_lineages: List[CalculationLineage]
) -> nx.DiGraph:
    """
    Build directed graph of calculation dependencies.

    Creates a NetworkX directed graph showing how calculations depend on
    each other. Useful for visualization and dependency analysis.

    Args:
        calculation_lineages: List of calculation lineage records

    Returns:
        NetworkX DiGraph with:
        - Nodes: metric codes
        - Edges: dependencies (A → B means B depends on A)
        - Node attributes: metric_name, output_value, output_unit, formula
        - Edge attributes: formula_type

    Example:
        >>> lineages = [lineage1, lineage2, lineage3]
        >>> G = build_lineage_graph(lineages)
        >>> print(f"Metrics: {len(G.nodes())}")
        >>> print(f"Dependencies: {len(G.edges())}")
        >>> # Find metrics with no dependencies
        >>> root_metrics = [n for n in G.nodes() if G.in_degree(n) == 0]
    """
    G = nx.DiGraph()

    # Add nodes for each metric
    for lineage in calculation_lineages:
        G.add_node(
            lineage.metric_code,
            metric_name=lineage.metric_name,
            output_value=lineage.output_value,
            output_unit=lineage.output_unit,
            formula=lineage.formula,
            formula_type=lineage.formula_type,
            timestamp=lineage.calculation_timestamp
        )

    # Add edges for dependencies
    for lineage in calculation_lineages:
        for dep in lineage.dependencies:
            # Edge from dependency to dependent metric
            G.add_edge(
                dep,
                lineage.metric_code,
                formula_type=lineage.formula_type
            )

    logger.debug(f"Built lineage graph: {len(G.nodes())} metrics, {len(G.edges())} dependencies")

    return G


def get_calculation_path(
    graph: nx.DiGraph,
    target_metric: str
) -> List[str]:
    """
    Get calculation path for a target metric (all dependencies in order).

    Args:
        graph: Lineage graph from build_lineage_graph()
        target_metric: Target metric code

    Returns:
        List of metric codes in calculation order (dependencies first)

    Example:
        >>> G = build_lineage_graph(lineages)
        >>> path = get_calculation_path(G, "E1-1")
        >>> print(f"Calculation path: {' → '.join(path)}")
    """
    if target_metric not in graph:
        return []

    # Get all ancestors (dependencies)
    ancestors = nx.ancestors(graph, target_metric)

    # Create subgraph with target and all ancestors
    subgraph = graph.subgraph(ancestors | {target_metric})

    # Topological sort gives calculation order
    try:
        return list(nx.topological_sort(subgraph))
    except nx.NetworkXError:
        logger.warning(f"Circular dependency detected for {target_metric}")
        return []


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def serialize_provenance(
    provenance_records: List[ProvenanceRecord]
) -> Dict[str, Any]:
    """
    Serialize provenance records to JSON-compatible dictionary.

    Converts Pydantic models to dictionaries suitable for JSON serialization.
    Includes metadata about the provenance export.

    Args:
        provenance_records: List of provenance records

    Returns:
        Dictionary with:
        {
            "metadata": {...},
            "records": [...],
            "summary": {...}
        }

    Example:
        >>> records = [record1, record2, record3]
        >>> data = serialize_provenance(records)
        >>> with open("provenance.json", "w") as f:
        ...     json.dump(data, f, indent=2)
    """
    # Convert records to dictionaries
    records_data = [record.dict() for record in provenance_records]

    # Compute summary statistics
    total_calculations = sum(
        1 for r in provenance_records
        if r.calculation_lineage is not None
    )

    agents_used = list(set(r.agent_name for r in provenance_records))

    total_errors = sum(len(r.errors) for r in provenance_records)
    total_warnings = sum(len(r.warnings) for r in provenance_records)

    return {
        "metadata": {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(provenance_records),
            "format_version": "1.0.0",
            "platform": "CSRD/ESRS Digital Reporting Platform"
        },
        "summary": {
            "total_calculations": total_calculations,
            "agents_used": agents_used,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "status_distribution": {
                status: sum(1 for r in provenance_records if r.status == status)
                for status in ["success", "warning", "error"]
            }
        },
        "records": records_data
    }


def save_provenance_json(
    provenance_records: List[ProvenanceRecord],
    output_path: Union[str, Path]
) -> None:
    """
    Save provenance records to JSON file.

    Args:
        provenance_records: List of provenance records
        output_path: Output file path

    Example:
        >>> save_provenance_json(records, "output/provenance/provenance.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = serialize_provenance(provenance_records)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved provenance to {output_path} ({len(provenance_records)} records)")


# ============================================================================
# AUDIT PACKAGE CREATION
# ============================================================================

def create_audit_package(
    provenance_records: List[ProvenanceRecord],
    output_path: Union[str, Path],
    include_files: Optional[List[Union[str, Path]]] = None,
    include_config: Optional[Union[str, Path]] = None,
    include_lineage_graph: bool = True
) -> Path:
    """
    Create complete audit package (ZIP) with all provenance data.

    The audit package includes:
    - provenance.json: All provenance records
    - environment.json: Environment snapshot
    - lineage_graph.json: Calculation dependency graph (if applicable)
    - manifest.json: Package manifest
    - Optional: Source files, configuration files

    Args:
        provenance_records: List of provenance records
        output_path: Output ZIP file path
        include_files: Optional list of files to include in package
        include_config: Optional configuration file to include
        include_lineage_graph: Whether to include lineage graph

    Returns:
        Path to created ZIP file

    Example:
        >>> audit_pkg = create_audit_package(
        ...     provenance_records=records,
        ...     output_path="output/audit_package.zip",
        ...     include_config="config/csrd_config.yaml",
        ...     include_files=["data/esg_data.csv"]
        ... )
        >>> print(f"Audit package: {audit_pkg}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating audit package: {output_path}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Add provenance records
        provenance_data = serialize_provenance(provenance_records)
        zf.writestr(
            "provenance.json",
            json.dumps(provenance_data, indent=2, default=str)
        )

        # 2. Add environment snapshot
        if provenance_records and provenance_records[0].environment:
            env_data = provenance_records[0].environment.dict()
            zf.writestr(
                "environment.json",
                json.dumps(env_data, indent=2, default=str)
            )

        # 3. Add lineage graph (if applicable)
        if include_lineage_graph:
            calculation_lineages = [
                r.calculation_lineage for r in provenance_records
                if r.calculation_lineage is not None
            ]

            if calculation_lineages:
                G = build_lineage_graph(calculation_lineages)

                # Convert graph to JSON-serializable format
                graph_data = {
                    "nodes": [
                        {
                            "id": node,
                            **data
                        }
                        for node, data in G.nodes(data=True)
                    ],
                    "edges": [
                        {
                            "source": source,
                            "target": target,
                            **data
                        }
                        for source, target, data in G.edges(data=True)
                    ],
                    "metadata": {
                        "total_nodes": len(G.nodes()),
                        "total_edges": len(G.edges()),
                        "is_acyclic": nx.is_directed_acyclic_graph(G)
                    }
                }

                zf.writestr(
                    "lineage_graph.json",
                    json.dumps(graph_data, indent=2, default=str)
                )

        # 4. Add configuration file
        if include_config:
            config_path = Path(include_config)
            if config_path.exists():
                zf.write(config_path, f"config/{config_path.name}")

        # 5. Add additional files
        if include_files:
            for file_path in include_files:
                file_path = Path(file_path)
                if file_path.exists():
                    zf.write(file_path, f"data/{file_path.name}")

        # 6. Add manifest
        manifest = {
            "package_created": datetime.now(timezone.utc).isoformat(),
            "platform": "CSRD/ESRS Digital Reporting Platform",
            "version": "1.0.0",
            "contents": {
                "provenance_records": len(provenance_records),
                "environment_snapshot": bool(provenance_records and provenance_records[0].environment),
                "lineage_graph": include_lineage_graph,
                "configuration": bool(include_config),
                "data_files": len(include_files) if include_files else 0
            },
            "files": [
                {
                    "name": info.filename,
                    "size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size
                }
                for info in zf.filelist
            ]
        }

        zf.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, default=str)
        )

    file_size = output_path.stat().st_size
    logger.info(f"Created audit package: {output_path} ({_format_bytes(file_size)})")

    return output_path


# ============================================================================
# AUDIT REPORT GENERATION
# ============================================================================

def generate_audit_report(
    provenance_records: List[ProvenanceRecord],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate human-readable audit report from provenance records.

    Creates a Markdown-formatted report suitable for auditors and
    regulatory compliance.

    Args:
        provenance_records: List of provenance records
        output_path: Optional path to save report

    Returns:
        Markdown-formatted audit report string

    Example:
        >>> report = generate_audit_report(records)
        >>> print(report)
        >>> # Or save to file:
        >>> with open("audit_report.md", "w") as f:
        ...     f.write(report)
    """
    lines = []

    # Header
    lines.append("# CSRD/ESRS PROVENANCE AUDIT REPORT")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Platform:** CSRD/ESRS Digital Reporting Platform v1.0.0")
    lines.append(f"**Total Records:** {len(provenance_records)}")
    lines.append("")

    # Environment (from first record)
    if provenance_records and provenance_records[0].environment:
        env = provenance_records[0].environment
        lines.append("## Execution Environment")
        lines.append("")
        lines.append(f"- **Python:** {env.python_major}.{env.python_minor}.{env.python_micro}")
        lines.append(f"- **Platform:** {env.platform} {env.platform_release}")
        lines.append(f"- **Machine:** {env.machine}")
        lines.append(f"- **Hostname:** {env.hostname}")
        lines.append(f"- **User:** {env.user}")
        lines.append(f"- **Working Directory:** {env.working_directory}")
        lines.append("")

        if env.package_versions:
            lines.append("### Dependencies")
            lines.append("")
            for pkg, version in sorted(env.package_versions.items()):
                lines.append(f"- {pkg}: {version}")
            lines.append("")

        if env.llm_models:
            lines.append("### LLM Models Used")
            lines.append("")
            for purpose, model in env.llm_models.items():
                lines.append(f"- {purpose}: {model}")
            lines.append("")

    # Agent Operations
    lines.append("## Agent Operations")
    lines.append("")

    agents = {}
    for record in provenance_records:
        if record.agent_name not in agents:
            agents[record.agent_name] = []
        agents[record.agent_name].append(record)

    for agent_name, records in sorted(agents.items()):
        lines.append(f"### {agent_name}")
        lines.append("")
        lines.append(f"- **Total Operations:** {len(records)}")

        successes = sum(1 for r in records if r.status == "success")
        warnings = sum(1 for r in records if r.status == "warning")
        errors = sum(1 for r in records if r.status == "error")

        lines.append(f"- **Success:** {successes}")
        lines.append(f"- **Warnings:** {warnings}")
        lines.append(f"- **Errors:** {errors}")

        total_duration = sum(r.duration_seconds or 0 for r in records)
        lines.append(f"- **Total Duration:** {total_duration:.2f}s")
        lines.append("")

    # Calculation Lineage
    calculation_records = [r for r in provenance_records if r.calculation_lineage]
    if calculation_records:
        lines.append("## Calculation Lineage")
        lines.append("")
        lines.append(f"**Total Calculations:** {len(calculation_records)}")
        lines.append("")

        # Sample calculations (first 10)
        for record in calculation_records[:10]:
            lineage = record.calculation_lineage
            lines.append(f"### {lineage.metric_code}: {lineage.metric_name}")
            lines.append("")
            lines.append(f"- **Formula:** `{lineage.formula}`")
            lines.append(f"- **Output:** {lineage.output_value} {lineage.output_unit}")
            lines.append(f"- **Hash:** `{lineage.hash[:16]}...`")

            if lineage.intermediate_steps:
                lines.append(f"- **Steps:**")
                for step in lineage.intermediate_steps:
                    lines.append(f"  - {step}")

            lines.append("")

        if len(calculation_records) > 10:
            lines.append(f"*(Showing 10 of {len(calculation_records)} calculations)*")
            lines.append("")

    # Data Quality
    lines.append("## Data Quality Summary")
    lines.append("")

    total_errors = sum(len(r.errors) for r in provenance_records)
    total_warnings = sum(len(r.warnings) for r in provenance_records)

    lines.append(f"- **Total Errors:** {total_errors}")
    lines.append(f"- **Total Warnings:** {total_warnings}")
    lines.append("")

    if total_errors > 0:
        lines.append("### Errors")
        lines.append("")
        for record in provenance_records:
            if record.errors:
                lines.append(f"**{record.agent_name} - {record.operation}:**")
                for error in record.errors:
                    lines.append(f"- {error}")
                lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*This audit report was automatically generated by the CSRD/ESRS Digital Reporting Platform.*")
    lines.append("")

    report = "\n".join(lines)

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved audit report to {output_path}")

    return report


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for provenance utilities."""
    parser = argparse.ArgumentParser(
        description="CSRD/ESRS Provenance Tracking Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hash a file
  python -m provenance.provenance_utils hash-file data/esg_data.csv

  # Capture environment
  python -m provenance.provenance_utils capture-env --config config/csrd_config.yaml

  # Create audit package
  python -m provenance.provenance_utils create-audit-package \\
      --provenance output/provenance.json \\
      --output output/audit_package.zip
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Hash file command
    hash_parser = subparsers.add_parser('hash-file', help='Calculate file hash')
    hash_parser.add_argument('file', help='File to hash')
    hash_parser.add_argument('--algorithm', default='sha256', choices=['sha256', 'sha512', 'md5'])

    # Capture environment command
    env_parser = subparsers.add_parser('capture-env', help='Capture environment snapshot')
    env_parser.add_argument('--config', help='Path to config file to hash')
    env_parser.add_argument('--output', help='Output JSON file')

    # Verify hash command
    verify_parser = subparsers.add_parser('verify-hash', help='Verify file hash')
    verify_parser.add_argument('file', help='File to verify')
    verify_parser.add_argument('expected_hash', help='Expected hash value')

    args = parser.parse_args()

    if args.command == 'hash-file':
        hash_info = hash_file(args.file, args.algorithm)
        print("\n" + "="*80)
        print("FILE HASH")
        print("="*80)
        print(f"File: {hash_info['file_name']}")
        print(f"Size: {hash_info['human_readable_size']}")
        print(f"Algorithm: {hash_info['hash_algorithm']}")
        print(f"Hash: {hash_info['hash_value']}")
        print(f"Timestamp: {hash_info['hash_timestamp']}")
        print("="*80 + "\n")

    elif args.command == 'capture-env':
        env = capture_environment(config_path=args.config)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(env.dict(), f, indent=2, default=str)
            print(f"Environment snapshot saved to {output_path}")
        else:
            print(json.dumps(env.dict(), indent=2, default=str))

    elif args.command == 'verify-hash':
        hash_info = hash_file(args.file)
        if hash_info['hash_value'] == args.expected_hash:
            print(f"✓ Hash verified: {args.file}")
        else:
            print(f"✗ Hash mismatch: {args.file}")
            print(f"  Expected: {args.expected_hash}")
            print(f"  Actual:   {hash_info['hash_value']}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
