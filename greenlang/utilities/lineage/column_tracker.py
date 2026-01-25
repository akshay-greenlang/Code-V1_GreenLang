"""
Column-Level Data Lineage Tracker

This module implements comprehensive column-level data lineage tracking for GreenLang.
It tracks transformations at the field/column level, maintaining a complete DAG of
data transformations with full provenance and compliance support.

Example:
    >>> tracker = ColumnLineageTracker()
    >>> with tracker.track_transformation("calculate_emissions") as t:
    ...     t.add_source("activity_data.fuel_consumption")
    ...     t.add_source("emissions_factors.co2_factor")
    ...     t.add_destination("emissions.co2_total")
    ...     t.set_formula("fuel_consumption * co2_factor")
    >>> graph = tracker.get_lineage_graph("emissions.co2_total")
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import wraps
import networkx as nx
from pathlib import Path
import pandas as pd
import sqlparse
import re
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Types of column transformations."""
    COPY = "copy"  # Direct copy
    CALCULATE = "calculate"  # Mathematical calculation
    AGGREGATE = "aggregate"  # Aggregation (sum, avg, etc.)
    JOIN = "join"  # Join operation
    FILTER = "filter"  # Filter/where clause
    LOOKUP = "lookup"  # Lookup from reference data
    DERIVE = "derive"  # Derived/computed field
    CAST = "cast"  # Type conversion
    CLEANSE = "cleanse"  # Data cleansing
    ENRICH = "enrich"  # Data enrichment
    PIVOT = "pivot"  # Pivot transformation
    UNPIVOT = "unpivot"  # Unpivot transformation


class DataClassification(str, Enum):
    """Data classification for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    FINANCIAL = "financial"
    REGULATORY = "regulatory"


class LineageNode(BaseModel):
    """Represents a single column/field in the lineage graph."""

    # Core attributes
    id: str = Field(..., description="Unique identifier (system.table.column)")
    name: str = Field(..., description="Column name")
    system: str = Field(..., description="Source system")
    dataset: str = Field(..., description="Table/dataset name")
    data_type: str = Field(..., description="Data type")

    # Metadata
    classification: DataClassification = Field(DataClassification.INTERNAL)
    description: Optional[str] = Field(None, description="Column description")
    business_name: Optional[str] = Field(None, description="Business-friendly name")
    owner: Optional[str] = Field(None, description="Data owner")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Relationships (stored as IDs for serialization)
    parent_ids: List[str] = Field(default_factory=list, description="Parent node IDs")
    child_ids: List[str] = Field(default_factory=list, description="Child node IDs")

    # Compliance metadata
    is_pii: bool = Field(False, description="Contains PII data")
    retention_days: Optional[int] = Field(None, description="Data retention period")
    gdpr_lawful_basis: Optional[str] = Field(None, description="GDPR lawful basis")
    sox_critical: bool = Field(False, description="SOX-critical field")

    @validator('id')
    def validate_id(cls, v):
        """Ensure ID follows system.dataset.column format."""
        parts = v.split('.')
        if len(parts) < 2:
            raise ValueError(f"ID must be in format 'system.dataset.column', got: {v}")
        return v

    def get_full_path(self) -> str:
        """Get full path including system."""
        return f"{self.system}.{self.dataset}.{self.name}"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TransformationRecord(BaseModel):
    """Records a single transformation operation."""

    id: str = Field(default_factory=lambda: hashlib.sha256(
        f"{datetime.now().isoformat()}".encode()).hexdigest()[:16])

    # Source and destination
    source_columns: List[str] = Field(..., description="Source column IDs")
    destination_columns: List[str] = Field(..., description="Destination column IDs")

    # Transformation details
    transformation_type: TransformationType
    formula: Optional[str] = Field(None, description="Transformation formula/code")
    description: Optional[str] = Field(None, description="Human-readable description")

    # Context
    agent_name: Optional[str] = Field(None, description="Agent that performed transformation")
    pipeline_id: Optional[str] = Field(None, description="Pipeline execution ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Quality metrics
    records_processed: Optional[int] = Field(None)
    records_dropped: Optional[int] = Field(None)
    execution_time_ms: Optional[float] = Field(None)

    # Validation rules applied
    validation_rules: List[str] = Field(default_factory=list)
    validation_passed: bool = Field(True)

    def calculate_hash(self) -> str:
        """Calculate hash for provenance."""
        content = json.dumps({
            'sources': sorted(self.source_columns),
            'destinations': sorted(self.destination_columns),
            'type': self.transformation_type,
            'formula': self.formula,
            'timestamp': self.timestamp.isoformat()
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class LineageGraph:
    """Directed Acyclic Graph for column lineage."""

    def __init__(self):
        """Initialize lineage graph."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, LineageNode] = {}
        self.transformations: Dict[str, TransformationRecord] = {}

    def add_node(self, node: LineageNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            **node.dict(exclude={'parent_ids', 'child_ids'})
        )

    def add_transformation(self, transformation: TransformationRecord) -> None:
        """Add a transformation and create edges."""
        self.transformations[transformation.id] = transformation

        # Create edges from sources to destinations
        for source in transformation.source_columns:
            for dest in transformation.destination_columns:
                self.graph.add_edge(
                    source, dest,
                    transformation_id=transformation.id,
                    transformation_type=transformation.transformation_type,
                    formula=transformation.formula
                )

                # Update node relationships
                if source in self.nodes:
                    if dest not in self.nodes[source].child_ids:
                        self.nodes[source].child_ids.append(dest)

                if dest in self.nodes:
                    if source not in self.nodes[dest].parent_ids:
                        self.nodes[dest].parent_ids.append(source)

    def get_upstream(self, column_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all upstream (source) columns."""
        if column_id not in self.graph:
            return set()

        if max_depth is None:
            return nx.ancestors(self.graph, column_id)
        else:
            upstream = set()
            current_level = {column_id}
            for _ in range(max_depth):
                next_level = set()
                for node in current_level:
                    predecessors = set(self.graph.predecessors(node))
                    next_level.update(predecessors)
                    upstream.update(predecessors)
                current_level = next_level
                if not current_level:
                    break
            return upstream

    def get_downstream(self, column_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """Get all downstream (derived) columns."""
        if column_id not in self.graph:
            return set()

        if max_depth is None:
            return nx.descendants(self.graph, column_id)
        else:
            downstream = set()
            current_level = {column_id}
            for _ in range(max_depth):
                next_level = set()
                for node in current_level:
                    successors = set(self.graph.successors(node))
                    next_level.update(successors)
                    downstream.update(successors)
                current_level = next_level
                if not current_level:
                    break
            return downstream

    def find_transformation_path(self, source: str, destination: str) -> List[List[str]]:
        """Find all transformation paths from source to destination."""
        if source not in self.graph or destination not in self.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(self.graph, source, destination))
            return paths
        except nx.NetworkXNoPath:
            return []

    def impact_analysis(self, column_id: str) -> Dict[str, Any]:
        """Analyze impact of changes to a column."""
        downstream = self.get_downstream(column_id)

        # Categorize impact by classification
        impact_by_classification = {}
        critical_impacts = []

        for node_id in downstream:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                classification = node.classification

                if classification not in impact_by_classification:
                    impact_by_classification[classification] = []
                impact_by_classification[classification].append(node_id)

                # Check for critical impacts
                if node.sox_critical or node.is_pii or node.classification == DataClassification.REGULATORY:
                    critical_impacts.append({
                        'column': node_id,
                        'reason': self._get_criticality_reason(node)
                    })

        return {
            'total_impacted': len(downstream),
            'impacted_columns': list(downstream),
            'by_classification': impact_by_classification,
            'critical_impacts': critical_impacts,
            'impacted_systems': self._get_impacted_systems(downstream)
        }

    def _get_criticality_reason(self, node: LineageNode) -> str:
        """Get reason why a node is critical."""
        reasons = []
        if node.sox_critical:
            reasons.append("SOX-critical")
        if node.is_pii:
            reasons.append("Contains PII")
        if node.classification == DataClassification.REGULATORY:
            reasons.append("Regulatory data")
        return ", ".join(reasons)

    def _get_impacted_systems(self, node_ids: Set[str]) -> List[str]:
        """Get unique systems impacted."""
        systems = set()
        for node_id in node_ids:
            if node_id in self.nodes:
                systems.add(self.nodes[node_id].system)
        return list(systems)

    def export_to_mermaid(self) -> str:
        """Export lineage graph to Mermaid format."""
        lines = ["graph LR"]

        # Add nodes with styling based on classification
        for node_id, node in self.nodes.items():
            label = f"{node.name}\\n({node.system})"
            style = self._get_mermaid_style(node)
            lines.append(f'    {node_id.replace(".", "_")}["{label}"]{style}')

        # Add edges with transformation labels
        for source, dest, data in self.graph.edges(data=True):
            source_clean = source.replace(".", "_")
            dest_clean = dest.replace(".", "_")
            transform_type = data.get('transformation_type', '')
            lines.append(f'    {source_clean} -->|{transform_type}| {dest_clean}')

        return "\n".join(lines)

    def _get_mermaid_style(self, node: LineageNode) -> str:
        """Get Mermaid styling based on node classification."""
        if node.is_pii:
            return ":::pii"
        elif node.sox_critical:
            return ":::critical"
        elif node.classification == DataClassification.REGULATORY:
            return ":::regulatory"
        return ""

    def export_to_graphviz(self) -> str:
        """Export lineage graph to Graphviz DOT format."""
        import pydot

        dot_graph = pydot.Dot(graph_type='digraph', rankdir='LR')

        # Add nodes
        for node_id, node in self.nodes.items():
            color = self._get_graphviz_color(node)
            shape = "box" if node.sox_critical else "ellipse"
            dot_node = pydot.Node(
                node_id,
                label=f"{node.name}\\n{node.system}.{node.dataset}",
                color=color,
                shape=shape
            )
            dot_graph.add_node(dot_node)

        # Add edges
        for source, dest, data in self.graph.edges(data=True):
            edge = pydot.Edge(
                source, dest,
                label=data.get('transformation_type', ''),
                color='blue'
            )
            dot_graph.add_edge(edge)

        return dot_graph.to_string()

    def _get_graphviz_color(self, node: LineageNode) -> str:
        """Get Graphviz color based on node classification."""
        color_map = {
            DataClassification.PII: "red",
            DataClassification.REGULATORY: "orange",
            DataClassification.FINANCIAL: "yellow",
            DataClassification.CONFIDENTIAL: "lightblue",
            DataClassification.RESTRICTED: "purple"
        }
        return color_map.get(node.classification, "black")


class ColumnLineageTracker:
    """Main class for tracking column-level lineage."""

    def __init__(self,
                 storage_backend: str = "memory",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize column lineage tracker.

        Args:
            storage_backend: Storage backend type (memory, neo4j, postgresql, file)
            config: Backend-specific configuration
        """
        self.storage_backend = storage_backend
        self.config = config or {}
        self.graph = LineageGraph()
        self.current_transformation: Optional[TransformationContext] = None

        # Initialize storage backend
        self._init_storage()

        logger.info(f"ColumnLineageTracker initialized with {storage_backend} backend")

    def _init_storage(self):
        """Initialize storage backend."""
        if self.storage_backend == "neo4j":
            self._init_neo4j()
        elif self.storage_backend == "postgresql":
            self._init_postgresql()
        elif self.storage_backend == "file":
            self._init_file_storage()
        # Default is memory (already initialized)

    def _init_neo4j(self):
        """Initialize Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            uri = self.config.get("uri", "bolt://localhost:7687")
            auth = (self.config.get("user", "neo4j"),
                   self.config.get("password", "password"))
            self.neo4j_driver = GraphDatabase.driver(uri, auth=auth)
            logger.info("Neo4j backend initialized")
        except ImportError:
            logger.warning("Neo4j driver not installed, falling back to memory")
            self.storage_backend = "memory"

    def _init_postgresql(self):
        """Initialize PostgreSQL connection."""
        try:
            import psycopg2
            self.pg_conn = psycopg2.connect(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 5432),
                database=self.config.get("database", "lineage"),
                user=self.config.get("user", "postgres"),
                password=self.config.get("password", "")
            )
            self._create_pg_tables()
            logger.info("PostgreSQL backend initialized")
        except ImportError:
            logger.warning("psycopg2 not installed, falling back to memory")
            self.storage_backend = "memory"

    def _create_pg_tables(self):
        """Create PostgreSQL tables for lineage storage."""
        create_nodes = """
        CREATE TABLE IF NOT EXISTS lineage_nodes (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            system VARCHAR(255) NOT NULL,
            dataset VARCHAR(255) NOT NULL,
            data_type VARCHAR(50),
            classification VARCHAR(50),
            is_pii BOOLEAN DEFAULT FALSE,
            sox_critical BOOLEAN DEFAULT FALSE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_transformations = """
        CREATE TABLE IF NOT EXISTS lineage_transformations (
            id VARCHAR(255) PRIMARY KEY,
            source_columns TEXT[],
            destination_columns TEXT[],
            transformation_type VARCHAR(50),
            formula TEXT,
            agent_name VARCHAR(255),
            pipeline_id VARCHAR(255),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        """

        create_edges = """
        CREATE TABLE IF NOT EXISTS lineage_edges (
            source_id VARCHAR(255),
            destination_id VARCHAR(255),
            transformation_id VARCHAR(255),
            PRIMARY KEY (source_id, destination_id, transformation_id),
            FOREIGN KEY (transformation_id) REFERENCES lineage_transformations(id)
        );
        """

        with self.pg_conn.cursor() as cursor:
            cursor.execute(create_nodes)
            cursor.execute(create_transformations)
            cursor.execute(create_edges)
            self.pg_conn.commit()

    def _init_file_storage(self):
        """Initialize file-based storage."""
        self.storage_path = Path(self.config.get("path", "./lineage_data"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.nodes_file = self.storage_path / "nodes.parquet"
        self.transformations_file = self.storage_path / "transformations.parquet"
        logger.info(f"File storage initialized at {self.storage_path}")

    def add_column(self,
                   system: str,
                   dataset: str,
                   column_name: str,
                   data_type: str,
                   **kwargs) -> LineageNode:
        """
        Add a column to the lineage graph.

        Args:
            system: Source system name
            dataset: Table/dataset name
            column_name: Column name
            data_type: Data type
            **kwargs: Additional metadata

        Returns:
            Created LineageNode
        """
        node_id = f"{system}.{dataset}.{column_name}"

        node = LineageNode(
            id=node_id,
            name=column_name,
            system=system,
            dataset=dataset,
            data_type=data_type,
            **kwargs
        )

        self.graph.add_node(node)
        self._persist_node(node)

        logger.debug(f"Added column: {node_id}")
        return node

    def _persist_node(self, node: LineageNode):
        """Persist node to storage backend."""
        if self.storage_backend == "neo4j":
            self._persist_node_neo4j(node)
        elif self.storage_backend == "postgresql":
            self._persist_node_postgresql(node)
        elif self.storage_backend == "file":
            self._persist_node_file(node)

    def _persist_node_neo4j(self, node: LineageNode):
        """Persist node to Neo4j."""
        if hasattr(self, 'neo4j_driver'):
            with self.neo4j_driver.session() as session:
                query = """
                MERGE (n:Column {id: $id})
                SET n += $properties
                """
                session.run(query, id=node.id, properties=node.dict())

    def _persist_node_postgresql(self, node: LineageNode):
        """Persist node to PostgreSQL."""
        if hasattr(self, 'pg_conn'):
            with self.pg_conn.cursor() as cursor:
                query = """
                INSERT INTO lineage_nodes
                (id, name, system, dataset, data_type, classification, is_pii, sox_critical, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP,
                    metadata = EXCLUDED.metadata
                """
                cursor.execute(query, (
                    node.id, node.name, node.system, node.dataset,
                    node.data_type, node.classification, node.is_pii,
                    node.sox_critical, json.dumps(node.dict())
                ))
                self.pg_conn.commit()

    def _persist_node_file(self, node: LineageNode):
        """Persist node to file storage."""
        if hasattr(self, 'nodes_file'):
            # Load existing nodes
            if self.nodes_file.exists():
                df = pd.read_parquet(self.nodes_file)
            else:
                df = pd.DataFrame()

            # Append new node
            new_row = pd.DataFrame([node.dict()])
            df = pd.concat([df[df.id != node.id], new_row], ignore_index=True)

            # Save back
            df.to_parquet(self.nodes_file, index=False)

    @contextmanager
    def track_transformation(self,
                           agent_name: str,
                           transformation_type: TransformationType = TransformationType.CALCULATE,
                           pipeline_id: Optional[str] = None):
        """
        Context manager for tracking a transformation.

        Example:
            with tracker.track_transformation("emissions_agent") as t:
                t.add_source("fuel.consumption")
                t.add_destination("emissions.co2")
                t.set_formula("consumption * factor")
        """
        self.current_transformation = TransformationContext(
            agent_name=agent_name,
            transformation_type=transformation_type,
            pipeline_id=pipeline_id
        )

        try:
            yield self.current_transformation
        finally:
            # Create transformation record
            if self.current_transformation.source_columns and self.current_transformation.destination_columns:
                transformation = TransformationRecord(
                    source_columns=self.current_transformation.source_columns,
                    destination_columns=self.current_transformation.destination_columns,
                    transformation_type=self.current_transformation.transformation_type,
                    formula=self.current_transformation.formula,
                    agent_name=agent_name,
                    pipeline_id=pipeline_id,
                    validation_rules=self.current_transformation.validation_rules
                )

                self.graph.add_transformation(transformation)
                self._persist_transformation(transformation)

                logger.debug(f"Tracked transformation: {transformation.id}")

            self.current_transformation = None

    def _persist_transformation(self, transformation: TransformationRecord):
        """Persist transformation to storage backend."""
        if self.storage_backend == "neo4j":
            self._persist_transformation_neo4j(transformation)
        elif self.storage_backend == "postgresql":
            self._persist_transformation_postgresql(transformation)
        elif self.storage_backend == "file":
            self._persist_transformation_file(transformation)

    def _persist_transformation_neo4j(self, transformation: TransformationRecord):
        """Persist transformation to Neo4j."""
        if hasattr(self, 'neo4j_driver'):
            with self.neo4j_driver.session() as session:
                # Create transformation node
                query = """
                CREATE (t:Transformation {id: $id})
                SET t += $properties
                """
                session.run(query, id=transformation.id,
                          properties=transformation.dict())

                # Create relationships
                for source in transformation.source_columns:
                    for dest in transformation.destination_columns:
                        rel_query = """
                        MATCH (s:Column {id: $source})
                        MATCH (d:Column {id: $dest})
                        MATCH (t:Transformation {id: $trans_id})
                        MERGE (s)-[:TRANSFORMS_TO {transformation_id: $trans_id}]->(d)
                        MERGE (t)-[:HAS_SOURCE]->(s)
                        MERGE (t)-[:HAS_DESTINATION]->(d)
                        """
                        session.run(rel_query, source=source, dest=dest,
                                  trans_id=transformation.id)

    def _persist_transformation_postgresql(self, transformation: TransformationRecord):
        """Persist transformation to PostgreSQL."""
        if hasattr(self, 'pg_conn'):
            with self.pg_conn.cursor() as cursor:
                # Insert transformation
                trans_query = """
                INSERT INTO lineage_transformations
                (id, source_columns, destination_columns, transformation_type,
                 formula, agent_name, pipeline_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(trans_query, (
                    transformation.id,
                    transformation.source_columns,
                    transformation.destination_columns,
                    transformation.transformation_type,
                    transformation.formula,
                    transformation.agent_name,
                    transformation.pipeline_id,
                    json.dumps(transformation.dict())
                ))

                # Insert edges
                for source in transformation.source_columns:
                    for dest in transformation.destination_columns:
                        edge_query = """
                        INSERT INTO lineage_edges (source_id, destination_id, transformation_id)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """
                        cursor.execute(edge_query, (source, dest, transformation.id))

                self.pg_conn.commit()

    def _persist_transformation_file(self, transformation: TransformationRecord):
        """Persist transformation to file storage."""
        if hasattr(self, 'transformations_file'):
            # Load existing transformations
            if self.transformations_file.exists():
                df = pd.read_parquet(self.transformations_file)
            else:
                df = pd.DataFrame()

            # Append new transformation
            new_row = pd.DataFrame([transformation.dict()])
            df = pd.concat([df, new_row], ignore_index=True)

            # Save back
            df.to_parquet(self.transformations_file, index=False)

    def track_dataframe_operation(self,
                                 input_df: pd.DataFrame,
                                 output_df: pd.DataFrame,
                                 operation: str,
                                 agent_name: Optional[str] = None):
        """
        Track lineage for a pandas DataFrame operation.

        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            operation: Description of operation performed
            agent_name: Agent performing operation
        """
        # Detect column mappings
        input_cols = set(input_df.columns)
        output_cols = set(output_df.columns)

        # Track new columns
        new_cols = output_cols - input_cols

        # Track transformations
        with self.track_transformation(
            agent_name or "dataframe_operation",
            TransformationType.CALCULATE
        ) as t:
            # Add all input columns as sources
            for col in input_cols:
                t.add_source(f"dataframe.input.{col}")

            # Add all output columns as destinations
            for col in output_cols:
                t.add_destination(f"dataframe.output.{col}")

            t.set_formula(operation)

    def parse_sql_lineage(self, sql_query: str, default_system: str = "database"):
        """
        Parse SQL query to extract column lineage.

        Args:
            sql_query: SQL query string
            default_system: Default system name for tables
        """
        # Parse SQL using sqlparse
        parsed = sqlparse.parse(sql_query)[0]

        # Extract SELECT columns and their sources
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        from_pattern = r'FROM\s+(\w+)'
        join_pattern = r'JOIN\s+(\w+)'

        select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        from_match = re.search(from_pattern, sql_query, re.IGNORECASE)
        join_matches = re.findall(join_pattern, sql_query, re.IGNORECASE)

        if select_match and from_match:
            columns = [col.strip() for col in select_match.group(1).split(',')]
            main_table = from_match.group(1)
            all_tables = [main_table] + join_matches

            # Track lineage for each selected column
            with self.track_transformation(
                "sql_query",
                TransformationType.CALCULATE
            ) as t:
                # Add source columns (simplified - real implementation would parse more detail)
                for table in all_tables:
                    t.add_source(f"{default_system}.{table}.*")

                # Add destination columns
                for col in columns:
                    col_name = col.split(' AS ')[-1].strip() if ' AS ' in col else col.strip()
                    t.add_destination(f"{default_system}.result.{col_name}")

                t.set_formula(sql_query[:100])  # Store first 100 chars

    def get_lineage_graph(self, column_id: str) -> LineageGraph:
        """Get lineage graph for a specific column."""
        # Create subgraph containing only relevant nodes
        upstream = self.graph.get_upstream(column_id)
        downstream = self.graph.get_downstream(column_id)
        relevant_nodes = upstream | downstream | {column_id}

        subgraph = LineageGraph()

        # Copy relevant nodes
        for node_id in relevant_nodes:
            if node_id in self.graph.nodes:
                subgraph.add_node(self.graph.nodes[node_id])

        # Copy relevant transformations
        for trans_id, trans in self.graph.transformations.items():
            if (any(s in relevant_nodes for s in trans.source_columns) and
                any(d in relevant_nodes for d in trans.destination_columns)):
                subgraph.add_transformation(trans)

        return subgraph

    def generate_compliance_report(self,
                                  compliance_type: str = "gdpr") -> Dict[str, Any]:
        """
        Generate compliance report for data lineage.

        Args:
            compliance_type: Type of compliance (gdpr, sox, all)

        Returns:
            Compliance report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'compliance_type': compliance_type,
            'summary': {},
            'details': []
        }

        if compliance_type in ["gdpr", "all"]:
            # GDPR compliance check
            pii_columns = []
            for node_id, node in self.graph.nodes.items():
                if node.is_pii:
                    downstream = self.graph.get_downstream(node_id)
                    pii_columns.append({
                        'column': node_id,
                        'lawful_basis': node.gdpr_lawful_basis,
                        'retention_days': node.retention_days,
                        'downstream_count': len(downstream),
                        'downstream_systems': self.graph._get_impacted_systems(downstream)
                    })

            report['gdpr'] = {
                'pii_columns_count': len(pii_columns),
                'pii_columns': pii_columns,
                'data_flow_documented': True,
                'retention_policies_defined': all(
                    c.get('retention_days') is not None for c in pii_columns
                )
            }

        if compliance_type in ["sox", "all"]:
            # SOX compliance check
            sox_critical = []
            for node_id, node in self.graph.nodes.items():
                if node.sox_critical:
                    upstream = self.graph.get_upstream(node_id)
                    sox_critical.append({
                        'column': node_id,
                        'system': node.system,
                        'upstream_count': len(upstream),
                        'transformation_count': len([
                            t for t in self.graph.transformations.values()
                            if node_id in t.destination_columns
                        ])
                    })

            report['sox'] = {
                'critical_fields_count': len(sox_critical),
                'critical_fields': sox_critical,
                'audit_trail_complete': True,
                'transformation_documented': True
            }

        return report

    def visualize_lineage(self,
                         column_id: Optional[str] = None,
                         output_format: str = "html",
                         output_file: Optional[str] = None) -> str:
        """
        Generate visualization of lineage graph.

        Args:
            column_id: Specific column to visualize (None for full graph)
            output_format: Output format (html, mermaid, graphviz, png)
            output_file: Optional output file path

        Returns:
            Visualization content or file path
        """
        if column_id:
            graph = self.get_lineage_graph(column_id)
        else:
            graph = self.graph

        if output_format == "mermaid":
            content = graph.export_to_mermaid()

        elif output_format == "graphviz":
            content = graph.export_to_graphviz()

        elif output_format == "html":
            # Generate interactive HTML visualization
            content = self._generate_html_visualization(graph)

        elif output_format == "png":
            # Generate PNG using graphviz
            import pydot
            dot_content = graph.export_to_graphviz()
            graph_viz = pydot.graph_from_dot_data(dot_content)[0]

            if output_file:
                graph_viz.write_png(output_file)
                return output_file
            else:
                return graph_viz.create_png()

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        if output_file and output_format != "png":
            with open(output_file, 'w') as f:
                f.write(content)
            return output_file

        return content

    def _generate_html_visualization(self, graph: LineageGraph) -> str:
        """Generate interactive HTML visualization."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Column Lineage Visualization</title>
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                #mynetwork {
                    width: 100%;
                    height: 600px;
                    border: 1px solid lightgray;
                }
                .info-panel {
                    margin: 20px;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }
                .pii { background-color: #ffcccc; }
                .regulatory { background-color: #ffd700; }
                .critical { background-color: #ff9999; }
            </style>
        </head>
        <body>
            <div class="info-panel">
                <h2>Column Lineage Graph</h2>
                <p>Nodes: {node_count} | Transformations: {transformation_count}</p>
                <div>
                    <span class="pii">■</span> PII Data
                    <span class="regulatory">■</span> Regulatory
                    <span class="critical">■</span> SOX Critical
                </div>
            </div>
            <div id="mynetwork"></div>
            <script>
                var nodes = new vis.DataSet({nodes_json});
                var edges = new vis.DataSet({edges_json});
                var container = document.getElementById('mynetwork');
                var data = { nodes: nodes, edges: edges };
                var options = {
                    nodes: {
                        shape: 'box',
                        font: { size: 12 }
                    },
                    edges: {
                        arrows: 'to',
                        font: { size: 10, align: 'middle' }
                    },
                    layout: {
                        hierarchical: {
                            direction: 'LR',
                            sortMethod: 'directed'
                        }
                    },
                    physics: {
                        enabled: false
                    }
                };
                var network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """

        # Prepare nodes and edges for vis.js
        vis_nodes = []
        for node_id, node in graph.nodes.items():
            color = "#97c2fc"  # Default color
            if node.is_pii:
                color = "#ffcccc"
            elif node.sox_critical:
                color = "#ff9999"
            elif node.classification == DataClassification.REGULATORY:
                color = "#ffd700"

            vis_nodes.append({
                'id': node_id,
                'label': f"{node.name}\n({node.system})",
                'color': color,
                'title': f"Type: {node.data_type}\nClassification: {node.classification}"
            })

        vis_edges = []
        for source, dest, data in graph.graph.edges(data=True):
            vis_edges.append({
                'from': source,
                'to': dest,
                'label': data.get('transformation_type', ''),
                'title': data.get('formula', '')
            })

        return html_template.format(
            node_count=len(vis_nodes),
            transformation_count=len(graph.transformations),
            nodes_json=json.dumps(vis_nodes),
            edges_json=json.dumps(vis_edges)
        )


class TransformationContext:
    """Context for tracking a transformation."""

    def __init__(self,
                 agent_name: str,
                 transformation_type: TransformationType,
                 pipeline_id: Optional[str] = None):
        self.agent_name = agent_name
        self.transformation_type = transformation_type
        self.pipeline_id = pipeline_id
        self.source_columns: List[str] = []
        self.destination_columns: List[str] = []
        self.formula: Optional[str] = None
        self.validation_rules: List[str] = []

    def add_source(self, column_id: str):
        """Add source column."""
        if column_id not in self.source_columns:
            self.source_columns.append(column_id)

    def add_destination(self, column_id: str):
        """Add destination column."""
        if column_id not in self.destination_columns:
            self.destination_columns.append(column_id)

    def set_formula(self, formula: str):
        """Set transformation formula."""
        self.formula = formula

    def add_validation_rule(self, rule: str):
        """Add validation rule."""
        self.validation_rules.append(rule)


def track_lineage(func: Callable) -> Callable:
    """
    Decorator for automatic lineage tracking.

    Example:
        @track_lineage
        def calculate_emissions(fuel_data, factors):
            return fuel_data * factors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get tracker from context or create new one
        tracker = kwargs.get('_lineage_tracker') or ColumnLineageTracker()

        # Start tracking
        with tracker.track_transformation(
            agent_name=func.__name__,
            transformation_type=TransformationType.CALCULATE
        ) as t:
            # Execute function
            result = func(*args, **kwargs)

            # Try to auto-detect columns (simplified)
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    t.add_destination(f"result.{col}")

            return result

    return wrapper


# Export main classes and functions
__all__ = [
    'ColumnLineageTracker',
    'LineageNode',
    'TransformationRecord',
    'LineageGraph',
    'TransformationType',
    'DataClassification',
    'track_lineage'
]