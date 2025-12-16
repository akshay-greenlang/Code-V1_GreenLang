# -*- coding: utf-8 -*-
"""
Unit tests for GL-024 Air Preheater Agent Provenance Module

Tests SHA-256 hashing, Merkle tree verification, audit trail generation,
and compliance with ASME PTC 4.3, ISO 27001, SOX, and EPA RATA requirements.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_024_air_preheater.provenance import (
    ProvenanceTracker,
    ProvenanceConfig,
    ProvenanceRecord,
    DataLineage,
    VerificationResult,
    AuditLevel,
    HashAlgorithm,
    CalculationType,
    ComplianceFramework,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Create default provenance configuration."""
    return ProvenanceConfig()


@pytest.fixture
def detailed_config():
    """Create detailed audit level configuration."""
    return ProvenanceConfig(
        audit_level=AuditLevel.DETAILED,
        hash_algorithm=HashAlgorithm.SHA256,
        enable_merkle_tree=True,
    )


@pytest.fixture
def tracker(default_config):
    """Create provenance tracker with default config."""
    return ProvenanceTracker(default_config)


@pytest.fixture
def detailed_tracker(detailed_config):
    """Create provenance tracker with detailed config."""
    return ProvenanceTracker(detailed_config)


@pytest.fixture
def sample_calculation_inputs():
    """Create sample calculation inputs."""
    return {
        "gas_inlet_temp_f": 650.0,
        "gas_outlet_temp_f": 300.0,
        "air_inlet_temp_f": 80.0,
        "air_outlet_temp_f": 550.0,
        "gas_flow_lb_hr": 500000.0,
    }


@pytest.fixture
def sample_calculation_outputs():
    """Create sample calculation outputs."""
    return {
        "effectiveness_pct": 75.0,
        "ntu": 2.5,
        "heat_duty_mmbtu_hr": 45.0,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestProvenanceConfig:
    """Test suite for ProvenanceConfig."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = ProvenanceConfig()
        assert config.audit_level is not None
        assert config.hash_algorithm is not None

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProvenanceConfig(
            audit_level=AuditLevel.DETAILED,
            hash_algorithm=HashAlgorithm.SHA384,
            enable_merkle_tree=True,
        )
        assert config.audit_level == AuditLevel.DETAILED
        assert config.hash_algorithm == HashAlgorithm.SHA384
        assert config.enable_merkle_tree is True


# =============================================================================
# TRACKER INITIALIZATION TESTS
# =============================================================================

class TestProvenanceTrackerInit:
    """Test suite for ProvenanceTracker initialization."""

    @pytest.mark.unit
    def test_tracker_creation_default(self):
        """Test tracker creation with default config."""
        tracker = ProvenanceTracker()
        assert tracker is not None

    @pytest.mark.unit
    def test_tracker_creation_with_config(self, detailed_config):
        """Test tracker creation with custom config."""
        tracker = ProvenanceTracker(detailed_config)
        assert tracker.config == detailed_config


# =============================================================================
# HASH ALGORITHM TESTS
# =============================================================================

class TestHashAlgorithms:
    """Test suite for hash algorithm functionality."""

    @pytest.mark.unit
    def test_sha256_hash(self, tracker):
        """Test SHA-256 hash generation."""
        data = "test_data_for_hashing"
        hash_result = tracker._calculate_hash(data)

        # SHA-256 produces 64 character hex digest
        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)

    @pytest.mark.unit
    def test_hash_determinism(self, tracker):
        """Test hash is deterministic."""
        data = "same_data"
        hash1 = tracker._calculate_hash(data)
        hash2 = tracker._calculate_hash(data)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_different_for_different_data(self, tracker):
        """Test different data produces different hash."""
        hash1 = tracker._calculate_hash("data_1")
        hash2 = tracker._calculate_hash("data_2")

        assert hash1 != hash2


# =============================================================================
# PROVENANCE RECORD TESTS
# =============================================================================

class TestProvenanceRecord:
    """Test suite for ProvenanceRecord creation and validation."""

    @pytest.mark.unit
    def test_create_record(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test provenance record creation."""
        record = detailed_tracker.create_record(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            equipment_tag="APH-001",
            methodology="epsilon-NTU per ASME PTC 4.3",
        )

        assert isinstance(record, ProvenanceRecord)
        assert record.calculation_type == CalculationType.HEAT_TRANSFER
        assert record.equipment_tag == "APH-001"

    @pytest.mark.unit
    def test_record_has_hash(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test record includes content hash."""
        record = detailed_tracker.create_record(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            equipment_tag="APH-001",
        )

        assert record.content_hash is not None
        assert len(record.content_hash) == 64

    @pytest.mark.unit
    def test_record_has_timestamp(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test record includes timestamp."""
        record = detailed_tracker.create_record(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            equipment_tag="APH-001",
        )

        assert record.timestamp is not None
        assert isinstance(record.timestamp, datetime)


# =============================================================================
# SESSION MANAGEMENT TESTS
# =============================================================================

class TestSessionManagement:
    """Test suite for provenance session management."""

    @pytest.mark.unit
    def test_start_session(self, detailed_tracker):
        """Test starting a provenance session."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="full_optimization",
        )

        assert detailed_tracker._session_active is True

    @pytest.mark.unit
    def test_record_calculation_in_session(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test recording calculation within session."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="heat_transfer_analysis",
        )

        detailed_tracker.record_calculation(
            calculation_type="heat_transfer_analysis",
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            methodology="epsilon-NTU per ASME PTC 4.3",
        )

        assert len(detailed_tracker._session_records) > 0

    @pytest.mark.unit
    def test_finalize_session(self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs):
        """Test finalizing a provenance session."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="full_optimization",
        )

        detailed_tracker.record_calculation(
            calculation_type="heat_transfer_analysis",
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            methodology="epsilon-NTU",
        )

        session_hash = detailed_tracker.finalize_session()

        assert session_hash is not None
        assert len(session_hash) == 64


# =============================================================================
# MERKLE TREE TESTS
# =============================================================================

class TestMerkleTree:
    """Test suite for Merkle tree verification."""

    @pytest.mark.unit
    def test_merkle_tree_single_record(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test Merkle tree with single record."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="single_calc",
        )

        detailed_tracker.record_calculation(
            calculation_type="heat_transfer",
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
        )

        root = detailed_tracker._calculate_merkle_root()
        assert root is not None

    @pytest.mark.unit
    def test_merkle_tree_multiple_records(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test Merkle tree with multiple records."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="multi_calc",
        )

        # Add multiple records
        for i in range(5):
            inputs = sample_calculation_inputs.copy()
            inputs["gas_inlet_temp_f"] = 650.0 + i * 10
            detailed_tracker.record_calculation(
                calculation_type=f"calc_{i}",
                inputs=inputs,
                outputs=sample_calculation_outputs,
            )

        root = detailed_tracker._calculate_merkle_root()
        assert root is not None


# =============================================================================
# CHAIN VERIFICATION TESTS
# =============================================================================

class TestChainVerification:
    """Test suite for calculation chain verification."""

    @pytest.mark.unit
    def test_verify_chain_valid(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test verification of valid calculation chain."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="verification_test",
        )

        record = detailed_tracker.create_record(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            equipment_tag="APH-001",
        )

        result = detailed_tracker.verify_chain(record.record_id)

        assert isinstance(result, VerificationResult)
        assert result.is_valid is True

    @pytest.mark.unit
    def test_verification_result_structure(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test verification result structure."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="result_structure_test",
        )

        record = detailed_tracker.create_record(
            calculation_type=CalculationType.LEAKAGE,
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            equipment_tag="APH-001",
        )

        result = detailed_tracker.verify_chain(record.record_id)

        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'verification_timestamp')


# =============================================================================
# AUDIT TRAIL TESTS
# =============================================================================

class TestAuditTrail:
    """Test suite for audit trail export functionality."""

    @pytest.mark.unit
    def test_export_audit_trail(
        self, detailed_tracker, sample_calculation_inputs, sample_calculation_outputs
    ):
        """Test audit trail export."""
        detailed_tracker.start_session(
            equipment_tag="APH-001",
            calculation_type="audit_test",
        )

        detailed_tracker.record_calculation(
            calculation_type="heat_transfer",
            inputs=sample_calculation_inputs,
            outputs=sample_calculation_outputs,
            methodology="ASME PTC 4.3",
        )

        audit_trail = detailed_tracker.export_audit_trail()

        assert audit_trail is not None
        assert hasattr(audit_trail, 'records') or isinstance(audit_trail, dict)


# =============================================================================
# COMPLIANCE FRAMEWORK TESTS
# =============================================================================

class TestComplianceFrameworks:
    """Test suite for compliance framework support."""

    @pytest.mark.unit
    def test_asme_ptc_43_compliance(self, detailed_tracker):
        """Test ASME PTC 4.3 compliance metadata."""
        record = detailed_tracker.create_record(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs={"temp": 650.0},
            outputs={"effectiveness": 0.75},
            equipment_tag="APH-001",
            compliance_frameworks=[ComplianceFramework.ASME_PTC_43],
        )

        assert ComplianceFramework.ASME_PTC_43 in record.compliance_frameworks

    @pytest.mark.unit
    def test_multiple_compliance_frameworks(self, detailed_tracker):
        """Test multiple compliance frameworks."""
        record = detailed_tracker.create_record(
            calculation_type=CalculationType.HEAT_TRANSFER,
            inputs={"temp": 650.0},
            outputs={"effectiveness": 0.75},
            equipment_tag="APH-001",
            compliance_frameworks=[
                ComplianceFramework.ASME_PTC_43,
                ComplianceFramework.ISO_27001,
            ],
        )

        assert len(record.compliance_frameworks) == 2


# =============================================================================
# DATA LINEAGE TESTS
# =============================================================================

class TestDataLineage:
    """Test suite for data lineage tracking."""

    @pytest.mark.unit
    def test_data_lineage_creation(self):
        """Test DataLineage creation."""
        lineage = DataLineage(
            source_system="DCS",
            source_tag="TI-101",
            acquisition_timestamp=datetime.now(timezone.utc),
            transformation_steps=["scaling", "validation"],
        )

        assert lineage.source_system == "DCS"
        assert len(lineage.transformation_steps) == 2


# =============================================================================
# AUDIT LEVEL TESTS
# =============================================================================

class TestAuditLevels:
    """Test suite for audit level functionality."""

    @pytest.mark.unit
    def test_minimal_audit_level(self):
        """Test minimal audit level configuration."""
        config = ProvenanceConfig(audit_level=AuditLevel.MINIMAL)
        tracker = ProvenanceTracker(config)

        assert tracker.config.audit_level == AuditLevel.MINIMAL

    @pytest.mark.unit
    def test_standard_audit_level(self):
        """Test standard audit level configuration."""
        config = ProvenanceConfig(audit_level=AuditLevel.STANDARD)
        tracker = ProvenanceTracker(config)

        assert tracker.config.audit_level == AuditLevel.STANDARD

    @pytest.mark.unit
    def test_detailed_audit_level(self):
        """Test detailed audit level configuration."""
        config = ProvenanceConfig(audit_level=AuditLevel.DETAILED)
        tracker = ProvenanceTracker(config)

        assert tracker.config.audit_level == AuditLevel.DETAILED


# =============================================================================
# HASH ALGORITHM ENUM TESTS
# =============================================================================

class TestHashAlgorithmEnum:
    """Test suite for HashAlgorithm enumeration."""

    @pytest.mark.unit
    def test_hash_algorithm_values(self):
        """Test hash algorithm enumeration values."""
        assert HashAlgorithm.SHA256.value == "sha256"
        assert HashAlgorithm.SHA384.value == "sha384"
        assert HashAlgorithm.SHA512.value == "sha512"


# =============================================================================
# CALCULATION TYPE ENUM TESTS
# =============================================================================

class TestCalculationTypeEnum:
    """Test suite for CalculationType enumeration."""

    @pytest.mark.unit
    def test_calculation_type_values(self):
        """Test calculation type enumeration values."""
        assert CalculationType.HEAT_TRANSFER is not None
        assert CalculationType.LEAKAGE is not None
        assert CalculationType.ACID_DEW_POINT is not None
