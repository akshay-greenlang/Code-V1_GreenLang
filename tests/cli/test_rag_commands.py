# -*- coding: utf-8 -*-
"""
Tests for RAG CLI commands.

Tests all RAG commands without requiring Docker or Weaviate to be running.
Uses mocks for external dependencies and validates command behavior.
"""

import subprocess
from pathlib import Path
from datetime import date
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pytest
from typer.testing import CliRunner

from greenlang.cli.rag_commands import app
from greenlang.intelligence.rag.models import (
    IngestionManifest,
    QueryResult,
    Chunk,
    RAGCitation,
    DocMeta,
)


runner = CliRunner()


@pytest.fixture
def mock_compose_file(tmp_path, monkeypatch):
    """Create a mock docker-compose.yml file in the expected location."""
    # Create the docker-compose file in expected location
    docker_dir = tmp_path / "docker" / "weaviate"
    docker_dir.mkdir(parents=True)
    compose_file = docker_dir / "docker-compose.yml"
    compose_file.write_text("version: '3.8'\nservices:\n  weaviate:\n    image: test\n")

    # Create a proper mock Path that resolves correctly
    class MockPath:
        def __init__(self, path_str):
            self._path = tmp_path

        @property
        def parent(self):
            """Return a parent mock that chains to tmp_path."""
            parent_mock = type('Parent', (), {})()
            parent_mock.parent = type('Parent', (), {})()
            parent_mock.parent.parent = tmp_path
            return parent_mock

    # Patch Path to return our mock
    monkeypatch.setattr("greenlang.cli.rag_commands.Path", MockPath)

    return compose_file


@pytest.fixture
def mock_test_file(tmp_path):
    """Create a test file for ingestion."""
    test_file = tmp_path / "test_document.md"
    test_file.write_text("# Test Document\n\nThis is a test document for RAG ingestion.")
    return test_file


class TestRagUp:
    """Tests for 'gl rag up' command."""

    def test_up_without_detach(self, mock_compose_file):
        """Test starting Weaviate without detach flag."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = runner.invoke(app, ["up"])

            assert result.exit_code == 0
            assert "Starting Weaviate container" in result.stdout
            assert "Weaviate stopped" in result.stdout

            # Verify subprocess.run was called with correct command
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "docker-compose"
            assert "-f" in call_args
            assert "up" in call_args
            assert "-d" not in call_args

    def test_up_with_detach(self, mock_compose_file):
        """Test starting Weaviate with detach flag."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = runner.invoke(app, ["up", "--detach"])

            assert result.exit_code == 0
            assert "Starting Weaviate container" in result.stdout
            assert "Weaviate started in background" in result.stdout
            assert "Connect at: http://localhost:8080" in result.stdout

            # Verify subprocess.run was called with -d flag
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "-d" in call_args

    def test_up_with_detach_short_flag(self, mock_compose_file):
        """Test starting Weaviate with -d short flag."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = runner.invoke(app, ["up", "-d"])

            assert result.exit_code == 0
            assert "Weaviate started in background" in result.stdout

    def test_up_docker_compose_not_found(self, mock_compose_file):
        """Test error when docker-compose is not in PATH."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = runner.invoke(app, ["up"])

            assert result.exit_code == 1
            assert "docker-compose not found in PATH" in result.stdout
            assert "Install Docker Compose" in result.stdout

    def test_up_subprocess_error(self, mock_compose_file):
        """Test error when docker-compose command fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "docker-compose")

            result = runner.invoke(app, ["up"])

            assert result.exit_code == 1
            assert "Error starting Weaviate" in result.stdout

    def test_up_compose_file_not_found(self, tmp_path, monkeypatch):
        """Test error when docker-compose.yml is not found."""
        # Create a Path mock that points to non-existent directory
        nonexistent_dir = tmp_path / "nonexistent"

        class MockPath:
            def __init__(self, path_str):
                self._path = nonexistent_dir

            @property
            def parent(self):
                """Return a parent mock that chains to nonexistent dir."""
                parent_mock = type('Parent', (), {})()
                parent_mock.parent = type('Parent', (), {})()
                parent_mock.parent.parent = nonexistent_dir
                return parent_mock

        monkeypatch.setattr("greenlang.cli.rag_commands.Path", MockPath)

        result = runner.invoke(app, ["up"])

        assert result.exit_code == 1
        assert "docker-compose.yml not found" in result.stdout
        assert "Expected location: docker/weaviate/docker-compose.yml" in result.stdout


class TestRagDown:
    """Tests for 'gl rag down' command."""

    def test_down_success(self, mock_compose_file):
        """Test stopping Weaviate successfully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = runner.invoke(app, ["down"])

            assert result.exit_code == 0
            assert "Stopping Weaviate container" in result.stdout
            assert "Weaviate stopped" in result.stdout

            # Verify subprocess.run was called with correct command
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "docker-compose"
            assert "-f" in call_args
            assert "down" in call_args

    def test_down_subprocess_error(self, mock_compose_file):
        """Test error when docker-compose down fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "docker-compose")

            result = runner.invoke(app, ["down"])

            assert result.exit_code == 1
            assert "Error stopping Weaviate" in result.stdout

    def test_down_compose_file_not_found(self, tmp_path, monkeypatch):
        """Test error when docker-compose.yml is not found."""
        # Create a Path mock that points to non-existent directory
        nonexistent_dir = tmp_path / "nonexistent"

        class MockPath:
            def __init__(self, path_str):
                self._path = nonexistent_dir

            @property
            def parent(self):
                """Return a parent mock that chains to nonexistent dir."""
                parent_mock = type('Parent', (), {})()
                parent_mock.parent = type('Parent', (), {})()
                parent_mock.parent.parent = nonexistent_dir
                return parent_mock

        monkeypatch.setattr("greenlang.cli.rag_commands.Path", MockPath)

        result = runner.invoke(app, ["down"])

        assert result.exit_code == 1
        assert "docker-compose.yml not found" in result.stdout


class TestRagIngest:
    """Tests for 'gl rag ingest' command."""

    def test_ingest_file_not_found(self, tmp_path):
        """Test error when file does not exist."""
        result = runner.invoke(app, [
            "ingest",
            "--collection", "test_collection",
            "--file", str(tmp_path / "nonexistent.pdf"),
        ])

        assert result.exit_code == 1
        assert "File not found" in result.stdout

    def test_ingest_missing_required_collection(self, mock_test_file):
        """Test error when collection parameter is missing."""
        result = runner.invoke(app, [
            "ingest",
            "--file", str(mock_test_file),
        ])

        # Typer will exit with code 2 for missing required options
        assert result.exit_code == 2

    def test_ingest_missing_required_file(self):
        """Test error when file parameter is missing."""
        result = runner.invoke(app, [
            "ingest",
            "--collection", "test_collection",
        ])

        # Typer will exit with code 2 for missing required options
        assert result.exit_code == 2


class TestRagQuery:
    """Tests for 'gl rag query' command."""

    def test_query_missing_required_query(self):
        """Test error when query parameter is missing."""
        result = runner.invoke(app, [
            "query-cmd",
            "--collection", "test_collection",
        ])

        # Typer will exit with code 2 for missing required options
        assert result.exit_code == 2


class TestRagStats:
    """Tests for 'gl rag stats' command - basic structure tests."""

    pass  # Stats command can be tested in integration tests


class TestRagHelp:
    """Tests for RAG command help."""

    def test_rag_help(self):
        """Test RAG help command."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "RAG (Retrieval-Augmented Generation) commands" in result.stdout
        assert "up" in result.stdout
        assert "down" in result.stdout
        assert "ingest" in result.stdout
        assert "query-cmd" in result.stdout
        assert "stats" in result.stdout

    def test_up_help(self):
        """Test 'gl rag up --help'."""
        result = runner.invoke(app, ["up", "--help"])

        assert result.exit_code == 0
        assert "Start Weaviate container" in result.stdout
        assert "--detach" in result.stdout
        assert "Run in background" in result.stdout

    def test_down_help(self):
        """Test 'gl rag down --help'."""
        result = runner.invoke(app, ["down", "--help"])

        assert result.exit_code == 0
        assert "Stop Weaviate container" in result.stdout

    def test_ingest_help(self):
        """Test 'gl rag ingest --help'."""
        result = runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "Ingest document into collection" in result.stdout
        assert "--collection" in result.stdout
        assert "--file" in result.stdout
        assert "--title" in result.stdout
        assert "--publisher" in result.stdout
        assert "--version" in result.stdout
        assert "--year" in result.stdout
        assert "--uri" in result.stdout

    def test_query_help(self):
        """Test 'gl rag query-cmd --help'."""
        result = runner.invoke(app, ["query-cmd", "--help"])

        assert result.exit_code == 0
        assert "Query RAG system" in result.stdout
        assert "--query" in result.stdout
        assert "--collection" in result.stdout
        assert "--top-k" in result.stdout
        assert "--fetch-k" in result.stdout
        assert "--lambda" in result.stdout

    def test_stats_help(self):
        """Test 'gl rag stats --help'."""
        result = runner.invoke(app, ["stats", "--help"])

        assert result.exit_code == 0
        assert "Show RAG system statistics" in result.stdout
        assert "--collection" in result.stdout
