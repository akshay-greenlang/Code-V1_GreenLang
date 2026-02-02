"""
GL-EUDR-001: SQLAlchemy Database Models

Database models for PostgreSQL persistence of supply chain data.
Uses PostGIS for geospatial operations on origin plots.

Tables:
- supply_chain_nodes: Supply chain entities (producers, processors, traders)
- supply_chain_edges: Relationships between entities
- origin_plots: Production plot geolocation data
- plot_producers: Many-to-many for cooperative shared plots
- supply_chain_snapshots: Immutable audit snapshots
- entity_resolution_candidates: Duplicate detection queue
- supply_chain_gaps: Coverage gap tracking

Note: Requires PostGIS extension for geometry operations.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Enum, ForeignKey, Index,
    Integer, Numeric, String, Text, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

Base = declarative_base()


# =============================================================================
# SUPPLY CHAIN NODES
# =============================================================================

class SupplyChainNodeModel(Base):
    """
    Supply chain entity (producer, processor, trader, importer, aggregator).
    Represents a single deduplicated entity with golden record fields.
    """
    __tablename__ = 'supply_chain_nodes'

    node_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_type = Column(String(50), nullable=False)
    name = Column(String(500), nullable=False)
    country_code = Column(String(2), nullable=False)
    address = Column(JSONB, default={})

    # Identifiers
    tax_id = Column(String(100), index=True)
    duns_number = Column(String(9))
    eori_number = Column(String(17))

    # Certifications and commodities
    certifications = Column(JSONB, default=[])
    commodities = Column(ARRAY(String), nullable=False)

    # Multi-tier support
    tier = Column(Integer)  # tier_min
    tier_max = Column(Integer)
    all_tiers = Column(ARRAY(Integer), default=[])

    # Risk and verification
    risk_score = Column(Numeric(3, 2))
    verification_status = Column(String(50), default='UNVERIFIED')
    disclosure_status = Column(String(50), default='FULL')

    # Operator classification (EUDR deadline handling)
    operator_size = Column(String(10))  # LARGE, SME

    # Golden record tracking
    field_sources = Column(JSONB, default={})

    # Metadata
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    outgoing_edges = relationship(
        'SupplyChainEdgeModel',
        foreign_keys='SupplyChainEdgeModel.source_node_id',
        back_populates='source_node'
    )
    incoming_edges = relationship(
        'SupplyChainEdgeModel',
        foreign_keys='SupplyChainEdgeModel.target_node_id',
        back_populates='target_node'
    )
    plots = relationship('OriginPlotModel', back_populates='producer')

    __table_args__ = (
        CheckConstraint(
            "node_type IN ('PRODUCER', 'PROCESSOR', 'TRADER', 'IMPORTER', 'AGGREGATOR')",
            name='valid_node_type'
        ),
        CheckConstraint(
            "disclosure_status IN ('FULL', 'PARTIAL', 'NONE')",
            name='valid_disclosure'
        ),
        Index('idx_nodes_type', 'node_type'),
        Index('idx_nodes_country', 'country_code'),
        Index('idx_nodes_verification', 'verification_status'),
        Index('idx_nodes_disclosure', 'disclosure_status'),
    )


# =============================================================================
# SUPPLY CHAIN EDGES
# =============================================================================

class SupplyChainEdgeModel(Base):
    """
    Relationship between supply chain nodes with provenance tracking.
    Supports multiple data sources and inference methods.
    """
    __tablename__ = 'supply_chain_edges'

    edge_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    target_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    edge_type = Column(String(50), nullable=False)
    commodity = Column(String(50), nullable=False)

    # Transaction details
    quantity = Column(Numeric(15, 3))
    quantity_unit = Column(String(20))
    transaction_date = Column(Date)
    documents = Column(JSONB, default=[])

    # Verification
    verified = Column(Boolean, default=False)
    confidence_score = Column(Numeric(3, 2), default=0.5)

    # Data provenance (ENHANCED per PRD)
    data_source = Column(String(50), default='SUPPLIER_DECLARED')
    inference_method = Column(String(100))
    inference_evidence = Column(JSONB, default=[])

    # Multi-tier context
    edge_context = Column(JSONB, default={})

    # DDS eligibility
    dds_eligible = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source_node = relationship(
        'SupplyChainNodeModel',
        foreign_keys=[source_node_id],
        back_populates='outgoing_edges'
    )
    target_node = relationship(
        'SupplyChainNodeModel',
        foreign_keys=[target_node_id],
        back_populates='incoming_edges'
    )

    __table_args__ = (
        CheckConstraint(
            "edge_type IN ('SUPPLIES', 'PROCESSES', 'TRADES', 'IMPORTS', 'AGGREGATES')",
            name='valid_edge_type'
        ),
        CheckConstraint(
            "commodity IN ('CATTLE', 'COCOA', 'COFFEE', 'PALM_OIL', 'RUBBER', 'SOY', 'WOOD')",
            name='valid_commodity'
        ),
        CheckConstraint(
            "data_source IN ('SUPPLIER_DECLARED', 'INFERRED_CUSTOMS', 'INFERRED_SHIPPING', "
            "'INFERRED_CERTIFICATION', 'INFERRED_SATELLITE', 'THIRD_PARTY_DATA')",
            name='valid_data_source'
        ),
        Index('idx_edges_source', 'source_node_id'),
        Index('idx_edges_target', 'target_node_id'),
        Index('idx_edges_commodity', 'commodity'),
        Index('idx_edges_data_source', 'data_source'),
        Index('idx_edges_dds_eligible', 'dds_eligible'),
    )


# =============================================================================
# ORIGIN PLOTS
# =============================================================================

class OriginPlotModel(Base):
    """
    Production plot with geolocation (point or polygon).
    Uses PostGIS for spatial operations and validation.
    """
    __tablename__ = 'origin_plots'

    plot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    producer_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    plot_identifier = Column(String(255))
    geometry = Column(Geometry('GEOMETRY', srid=4326), nullable=False)
    area_hectares = Column(Numeric(10, 2))
    commodity = Column(String(50), nullable=False)
    country_code = Column(String(2), nullable=False)

    # Validation and risk
    validation_status = Column(String(50), default='PENDING')
    deforestation_risk_score = Column(Numeric(3, 2))
    last_assessment_date = Column(Date)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    producer = relationship('SupplyChainNodeModel', back_populates='plots')
    plot_producers = relationship('PlotProducerModel', back_populates='plot')

    __table_args__ = (
        CheckConstraint("ST_IsValid(geometry)", name='valid_geometry'),
        Index('idx_plots_geometry', 'geometry', postgresql_using='gist'),
        Index('idx_plots_producer', 'producer_node_id'),
    )


class PlotProducerModel(Base):
    """
    Many-to-many relationship for cooperative/shared plots.
    Tracks ownership shares and tenure types.
    """
    __tablename__ = 'plot_producers'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plot_id = Column(
        UUID(as_uuid=True),
        ForeignKey('origin_plots.plot_id'),
        nullable=False
    )
    producer_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    share_percentage = Column(Numeric(5, 2))
    tenure_type = Column(String(50))  # OWNER, LEASE, COMMUNITY
    valid_from = Column(Date)
    valid_to = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    plot = relationship('OriginPlotModel', back_populates='plot_producers')
    producer = relationship('SupplyChainNodeModel')

    __table_args__ = (
        UniqueConstraint('plot_id', 'producer_node_id', 'valid_from', name='unique_plot_producer'),
    )


# =============================================================================
# SUPPLY CHAIN SNAPSHOTS
# =============================================================================

class SupplyChainSnapshotModel(Base):
    """
    Immutable snapshot of supply chain state for audit and temporal queries.
    """
    __tablename__ = 'supply_chain_snapshots'

    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_date = Column(DateTime, default=datetime.utcnow)
    importer_node_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    commodity = Column(String(50), nullable=False)
    graph_hash = Column(String(64), nullable=False)

    # Counts
    node_count = Column(Integer)
    edge_count = Column(Integer)
    plot_count = Column(Integer)

    # Coverage metrics
    coverage_percentage = Column(Numeric(5, 2))
    mapping_completeness = Column(Numeric(5, 2))
    plot_coverage = Column(Numeric(5, 2))

    # Full snapshot data
    snapshot_data = Column(JSONB, nullable=False)

    # Trigger metadata
    trigger_type = Column(String(50))  # SCHEDULED, DDS_SUBMISSION, COVERAGE_DROP, MANUAL
    created_by = Column(String(100))

    # Relationships
    gaps = relationship('SupplyChainGapModel', back_populates='snapshot')

    __table_args__ = (
        UniqueConstraint('importer_node_id', 'commodity', 'snapshot_date', name='unique_snapshot'),
        Index('idx_snapshots_importer', 'importer_node_id'),
        Index('idx_snapshots_date', 'snapshot_date'),
    )


# =============================================================================
# ENTITY RESOLUTION
# =============================================================================

class EntityResolutionCandidateModel(Base):
    """
    Candidate pairs for entity resolution with matching features.
    """
    __tablename__ = 'entity_resolution_candidates'

    candidate_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_a_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    node_b_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id'),
        nullable=False
    )
    similarity_score = Column(Numeric(3, 2), nullable=False)
    matching_features = Column(JSONB, nullable=False)
    resolution_status = Column(String(50), default='PENDING')
    resolved_by = Column(String(100))
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    node_a = relationship('SupplyChainNodeModel', foreign_keys=[node_a_id])
    node_b = relationship('SupplyChainNodeModel', foreign_keys=[node_b_id])

    __table_args__ = (
        UniqueConstraint('node_a_id', 'node_b_id', name='unique_candidate_pair'),
        Index('idx_er_candidates_status', 'resolution_status'),
    )


# =============================================================================
# SUPPLY CHAIN GAPS
# =============================================================================

class SupplyChainGapModel(Base):
    """
    Coverage gaps identified in supply chain mapping.
    """
    __tablename__ = 'supply_chain_gaps'

    gap_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_snapshots.snapshot_id')
    )
    node_id = Column(
        UUID(as_uuid=True),
        ForeignKey('supply_chain_nodes.node_id')
    )
    gap_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    description = Column(Text)
    remediation_suggestion = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    snapshot = relationship('SupplyChainSnapshotModel', back_populates='gaps')
    node = relationship('SupplyChainNodeModel')

    __table_args__ = (
        CheckConstraint(
            "gap_type IN ('UNVERIFIED_SUPPLIER', 'MISSING_PLOT_DATA', 'MISSING_COORDINATES', "
            "'PARTIAL_DISCLOSURE', 'EXPIRED_CERTIFICATION', 'CYCLE_DETECTED', "
            "'MISSING_TIER_DATA', 'UNVERIFIED_TRANSFORMATION')",
            name='valid_gap_type'
        ),
        CheckConstraint(
            "severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')",
            name='valid_severity'
        ),
        Index('idx_gaps_snapshot', 'snapshot_id'),
        Index('idx_gaps_severity', 'severity'),
    )


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def create_all_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all tables from the database."""
    Base.metadata.drop_all(engine)


# SQL for creating PostGIS extension and indexes (run manually)
POSTGIS_SETUP_SQL = """
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create GIN index for commodities array
CREATE INDEX IF NOT EXISTS idx_nodes_commodity ON supply_chain_nodes USING GIN(commodities);

-- Create spatial index for plots
CREATE INDEX IF NOT EXISTS idx_plots_geometry ON origin_plots USING GIST(geometry);
"""
