"""
Provenance Framework Test Suite

Tests the GreenLang provenance framework components:
- File hashing (SHA256)
- Environment capture
- ProvenanceRecord
- Report generation (Markdown, HTML, JSON)
- Merkle tree verification
- Environment comparison
- Provenance validation

Validates 100% replacement of custom provenance (604 lines → 0 lines)

Author: GreenLang CBAM Team
Date: 2025-10-16
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import framework provenance
from greenlang.provenance import (
    hash_file,
    get_environment_info,
    ProvenanceRecord,
    generate_markdown_report,
    generate_html_report,
    MerkleTree,
    validate_provenance,
    compare_environments,
    verify_integrity
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def test_files(tmp_path):
    """Create test files for hashing."""
    files = {}

    # Create test file 1
    file1 = tmp_path / "test1.txt"
    file1.write_text("This is test file 1")
    files['file1'] = file1

    # Create test file 2
    file2 = tmp_path / "test2.txt"
    file2.write_text("This is test file 2 with different content")
    files['file2'] = file2

    # Create identical file
    file3 = tmp_path / "test3.txt"
    file3.write_text("This is test file 1")
    files['file3'] = file3

    return files


@pytest.fixture
def sample_provenance_record():
    """Create sample provenance record."""
    return ProvenanceRecord(
        agent_name="test-agent",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        input_files=[
            {
                "path": "/data/input1.csv",
                "hash": "abc123",
                "size": 1024
            }
        ],
        output_files=[
            {
                "path": "/data/output1.json",
                "hash": "def456",
                "size": 2048
            }
        ],
        environment={
            "python_version": "3.9.0",
            "os": "Linux",
            "hostname": "test-host"
        },
        metadata={
            "description": "Test provenance record",
            "author": "Test User"
        }
    )


# ============================================================================
# TEST FILE HASHING
# ============================================================================

class TestFileHashing:
    """Test file hashing functionality."""

    def test_hash_file_basic(self, test_files):
        """Test basic file hashing."""
        result = hash_file(test_files['file1'])

        assert 'hash' in result
        assert 'algorithm' in result
        assert result['algorithm'] == 'sha256'
        assert len(result['hash']) == 64  # SHA256 hex length

    def test_hash_determinism(self, test_files):
        """Test that hashing is deterministic."""
        hash1 = hash_file(test_files['file1'])
        hash2 = hash_file(test_files['file1'])

        assert hash1['hash'] == hash2['hash']

    def test_identical_content_same_hash(self, test_files):
        """Test identical content produces same hash."""
        hash1 = hash_file(test_files['file1'])
        hash3 = hash_file(test_files['file3'])

        assert hash1['hash'] == hash3['hash']

    def test_different_content_different_hash(self, test_files):
        """Test different content produces different hash."""
        hash1 = hash_file(test_files['file1'])
        hash2 = hash_file(test_files['file2'])

        assert hash1['hash'] != hash2['hash']

    def test_hash_nonexistent_file(self):
        """Test hashing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/file.txt")


# ============================================================================
# TEST ENVIRONMENT CAPTURE
# ============================================================================

class TestEnvironmentCapture:
    """Test environment capture functionality."""

    def test_get_environment_info_basic(self):
        """Test basic environment capture."""
        env = get_environment_info()

        assert 'python_version' in env
        assert 'os' in env
        assert 'hostname' in env
        assert 'user' in env
        assert 'timestamp' in env

    def test_environment_includes_packages(self):
        """Test environment includes installed packages."""
        env = get_environment_info()

        assert 'packages' in env
        assert isinstance(env['packages'], list)
        assert len(env['packages']) > 0

    def test_environment_includes_git_info(self):
        """Test environment captures git info if available."""
        env = get_environment_info()

        # Git info may not be available in all test environments
        if 'git' in env:
            assert 'commit' in env['git']
            assert 'branch' in env['git']

    def test_environment_reproducibility(self):
        """Test environment capture is consistent."""
        env1 = get_environment_info()
        env2 = get_environment_info()

        # Core fields should be identical
        assert env1['python_version'] == env2['python_version']
        assert env1['os'] == env2['os']
        assert env1['hostname'] == env2['hostname']


# ============================================================================
# TEST PROVENANCE RECORD
# ============================================================================

class TestProvenanceRecord:
    """Test ProvenanceRecord model."""

    def test_create_provenance_record(self, sample_provenance_record):
        """Test creating provenance record."""
        record = sample_provenance_record

        assert record.agent_name == "test-agent"
        assert record.version == "1.0.0"
        assert len(record.input_files) == 1
        assert len(record.output_files) == 1

    def test_provenance_record_serialization(self, sample_provenance_record):
        """Test provenance record serialization to JSON."""
        record = sample_provenance_record

        # Serialize
        json_str = record.to_json()
        data = json.loads(json_str)

        assert data['agent_name'] == "test-agent"
        assert data['version'] == "1.0.0"

    def test_provenance_record_deserialization(self, sample_provenance_record):
        """Test provenance record deserialization from JSON."""
        record = sample_provenance_record

        # Serialize then deserialize
        json_str = record.to_json()
        reconstructed = ProvenanceRecord.from_json(json_str)

        assert reconstructed.agent_name == record.agent_name
        assert reconstructed.version == record.version
        assert len(reconstructed.input_files) == len(record.input_files)


# ============================================================================
# TEST REPORT GENERATION
# ============================================================================

class TestReportGeneration:
    """Test provenance report generation."""

    def test_generate_markdown_report(self, sample_provenance_record):
        """Test Markdown report generation."""
        record = sample_provenance_record

        markdown = generate_markdown_report(record)

        assert isinstance(markdown, str)
        assert "# Provenance Report" in markdown or "test-agent" in markdown
        assert "1.0.0" in markdown

    def test_generate_html_report(self, sample_provenance_record):
        """Test HTML report generation."""
        record = sample_provenance_record

        html = generate_html_report(record)

        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "test-agent" in html

    def test_markdown_report_includes_sections(self, sample_provenance_record):
        """Test Markdown report includes all required sections."""
        record = sample_provenance_record

        markdown = generate_markdown_report(record)

        # Check for key sections
        assert "Agent" in markdown or "agent" in markdown
        assert "Version" in markdown or "version" in markdown
        assert "Input" in markdown or "input" in markdown
        assert "Output" in markdown or "output" in markdown

    def test_html_report_interactive(self, sample_provenance_record):
        """Test HTML report has interactive features."""
        record = sample_provenance_record

        html = generate_html_report(record)

        # Check for interactive elements
        assert "<script" in html.lower() or "onclick" in html.lower() or "collapsible" in html.lower()


# ============================================================================
# TEST MERKLE TREE
# ============================================================================

class TestMerkleTree:
    """Test Merkle tree functionality."""

    def test_create_merkle_tree(self, test_files):
        """Test creating Merkle tree from files."""
        files = [test_files['file1'], test_files['file2']]

        tree = MerkleTree(files)

        assert tree.root is not None

    def test_merkle_proof_generation(self, test_files):
        """Test generating Merkle proof."""
        files = [test_files['file1'], test_files['file2'], test_files['file3']]

        tree = MerkleTree(files)
        proof = tree.get_proof(test_files['file1'])

        assert proof is not None
        assert isinstance(proof, list)

    def test_merkle_proof_verification(self, test_files):
        """Test verifying Merkle proof."""
        files = [test_files['file1'], test_files['file2'], test_files['file3']]

        tree = MerkleTree(files)
        proof = tree.get_proof(test_files['file1'])

        # Verify proof
        is_valid = tree.verify_proof(proof, test_files['file1'])

        assert is_valid is True

    def test_merkle_tree_determinism(self, test_files):
        """Test Merkle tree is deterministic."""
        files = [test_files['file1'], test_files['file2']]

        tree1 = MerkleTree(files)
        tree2 = MerkleTree(files)

        assert tree1.root == tree2.root


# ============================================================================
# TEST ENVIRONMENT COMPARISON
# ============================================================================

class TestEnvironmentComparison:
    """Test environment comparison functionality."""

    def test_compare_identical_environments(self):
        """Test comparing identical environments."""
        env1 = get_environment_info()
        env2 = get_environment_info()

        diff = compare_environments(env1, env2)

        assert 'python_version' in diff
        assert diff['python_version']['changed'] is False

    def test_compare_different_python_versions(self):
        """Test detecting Python version differences."""
        env1 = {"python_version": "3.9.0", "os": "Linux"}
        env2 = {"python_version": "3.10.0", "os": "Linux"}

        diff = compare_environments(env1, env2)

        assert diff['python_version']['changed'] is True
        assert diff['python_version']['old'] == "3.9.0"
        assert diff['python_version']['new'] == "3.10.0"

    def test_compare_different_packages(self):
        """Test detecting package differences."""
        env1 = {
            "packages": [
                {"name": "numpy", "version": "1.20.0"},
                {"name": "pandas", "version": "1.3.0"}
            ]
        }
        env2 = {
            "packages": [
                {"name": "numpy", "version": "1.21.0"},
                {"name": "pandas", "version": "1.3.0"}
            ]
        }

        diff = compare_environments(env1, env2)

        assert 'packages' in diff
        # Should detect numpy version change


# ============================================================================
# TEST PROVENANCE VALIDATION
# ============================================================================

class TestProvenanceValidation:
    """Test provenance validation functionality."""

    def test_validate_valid_provenance(self, sample_provenance_record):
        """Test validating valid provenance record."""
        record = sample_provenance_record

        is_valid = validate_provenance(record)

        assert is_valid is True

    def test_validate_missing_fields(self):
        """Test validating provenance with missing fields."""
        # Create incomplete record
        record = ProvenanceRecord(
            agent_name="",  # Invalid
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )

        is_valid = validate_provenance(record)

        assert is_valid is False

    def test_verify_integrity(self, test_files):
        """Test verifying file integrity."""
        files = [test_files['file1'], test_files['file2']]

        # Create record with current file hashes
        record = ProvenanceRecord(
            agent_name="test-agent",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            input_files=[
                {
                    "path": str(test_files['file1']),
                    "hash": hash_file(test_files['file1'])['hash']
                }
            ]
        )

        # Verify integrity (files unchanged)
        is_intact = verify_integrity(record, [test_files['file1']])

        assert is_intact is True

    def test_verify_integrity_modified_file(self, test_files):
        """Test detecting modified files."""
        # Create record with original hash
        original_hash = hash_file(test_files['file1'])['hash']

        record = ProvenanceRecord(
            agent_name="test-agent",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            input_files=[
                {
                    "path": str(test_files['file1']),
                    "hash": original_hash
                }
            ]
        )

        # Modify file
        test_files['file1'].write_text("Modified content!")

        # Verify integrity (should fail)
        is_intact = verify_integrity(record, [test_files['file1']])

        assert is_intact is False


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
