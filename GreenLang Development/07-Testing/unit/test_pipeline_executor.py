# -*- coding: utf-8 -*-
"""
Pipeline Executor Tests
=======================

Tests for the GreenLang Pipeline Executor covering:
- Single and multi-step pipelines
- Retry/backoff logic
- Simple branching
- Golden-file outputs for regression testing
- Deterministic execution
- Multiple backend support

Target: High coverage for runtime/executor module
"""

import pytest
import json
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from greenlang.runtime.executor import Executor, DeterministicConfig
from greenlang.sdk.base import Result
from greenlang.determinism import deterministic_random


class TestDeterministicConfig:
    """Test deterministic configuration"""

    def test_default_config(self):
        """Test default deterministic configuration"""
        config = DeterministicConfig()
        assert config.seed == 42
        assert config.freeze_env is True
        assert config.normalize_floats is True
        assert config.float_precision == 6
        assert config.quantization_bits is None

    def test_custom_config(self):
        """Test custom deterministic configuration"""
        config = DeterministicConfig(
            seed=123,
            freeze_env=False,
            normalize_floats=False,
            float_precision=4,
            quantization_bits=8
        )
        assert config.seed == 123
        assert config.freeze_env is False
        assert config.normalize_floats is False
        assert config.float_precision == 4
        assert config.quantization_bits == 8

    @patch('greenlang.runtime.executor.random')
    @patch('greenlang.runtime.executor.HAS_NUMPY', True)
    @patch('greenlang.runtime.executor.np')
    def test_apply_deterministic_settings(self, mock_np, mock_random):
        """Test applying deterministic settings"""
        config = DeterministicConfig(seed=123)
        config.apply()

        mock_random.seed.assert_called_with(123)
        mock_np.random.seed.assert_called_with(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"

    @patch('greenlang.runtime.executor.random')
    def test_apply_without_numpy(self, mock_random):
        """Test applying settings without numpy"""
        with patch('greenlang.runtime.executor.HAS_NUMPY', False):
            config = DeterministicConfig(seed=456)
            config.apply()

            mock_random.seed.assert_called_with(456)


class TestExecutorInitialization:
    """Test executor initialization"""

    def test_default_initialization(self):
        """Test default executor initialization"""
        executor = Executor()
        assert executor.backend == "local"
        assert executor.profile == "local"  # Backward compatibility
        assert executor.deterministic is False
        assert isinstance(executor.det_config, DeterministicConfig)
        assert executor.artifacts_dir.exists()

    def test_custom_initialization(self):
        """Test custom executor initialization"""
        custom_config = DeterministicConfig(seed=999)
        executor = Executor(
            backend="k8s",
            deterministic=True,
            det_config=custom_config
        )
        assert executor.backend == "k8s"
        assert executor.deterministic is True
        assert executor.det_config.seed == 999

    def test_invalid_backend(self):
        """Test initialization with invalid backend"""
        with pytest.raises(ValueError, match="Unknown backend"):
            Executor(backend="invalid")

    @patch('subprocess.run')
    def test_k8s_backend_validation_success(self, mock_subprocess):
        """Test K8s backend validation when kubectl is available"""
        mock_subprocess.return_value = Mock(returncode=0)

        executor = Executor(backend="k8s")
        assert executor.backend == "k8s"
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_k8s_backend_validation_fallback(self, mock_subprocess):
        """Test K8s backend fallback to local when kubectl unavailable"""
        mock_subprocess.side_effect = FileNotFoundError()

        executor = Executor(backend="k8s")
        assert executor.backend == "local"

    @patch('subprocess.run')
    def test_k8s_backend_validation_timeout(self, mock_subprocess):
        """Test K8s backend fallback on timeout"""
        from subprocess import TimeoutExpired
        mock_subprocess.side_effect = TimeoutExpired("kubectl", 5)

        executor = Executor(backend="k8s")
        assert executor.backend == "local"


class TestPipelineLoading:
    """Test pipeline loading functionality"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor()

    def test_load_pipeline_from_file(self):
        """Test loading pipeline from YAML file"""
        pipeline_data = {
            "name": "test-pipeline",
            "version": "1.0.0",
            "steps": [
                {"name": "step1", "agent": "test-agent"}
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(pipeline_data, f)
            pipeline_path = f.name

        try:
            loaded_pipeline = self.executor._load_pipeline(pipeline_path)
            assert loaded_pipeline["name"] == "test-pipeline"
            assert len(loaded_pipeline["steps"]) == 1
        finally:
            Path(pipeline_path).unlink(missing_ok=True)

    def test_load_pipeline_from_pack_reference(self):
        """Test loading pipeline from pack reference"""
        mock_pack = Mock()
        mock_pipeline = {"name": "pack-pipeline", "steps": []}
        mock_pack.get_pipeline.return_value = mock_pipeline

        self.executor.loader.load = Mock(return_value=mock_pack)

        pipeline = self.executor._load_pipeline("test-pack/test-pipeline")
        assert pipeline["name"] == "pack-pipeline"

    def test_load_pipeline_from_loaded_packs(self):
        """Test loading pipeline from already loaded packs"""
        mock_pack = Mock()
        mock_pipeline = {"name": "loaded-pipeline", "steps": []}
        mock_pack.get_pipeline.return_value = mock_pipeline

        self.executor.loader.loaded_packs = {"test-pack": mock_pack}

        pipeline = self.executor._load_pipeline("loaded-pipeline")
        assert pipeline["name"] == "loaded-pipeline"

    def test_load_pipeline_not_found(self):
        """Test loading non-existent pipeline"""
        self.executor.loader.loaded_packs = {}

        with pytest.raises(ValueError, match="Pipeline not found"):
            self.executor._load_pipeline("nonexistent-pipeline")


class TestLocalExecution:
    """Test local pipeline execution"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor(backend="local")

    def test_execute_simple_pipeline(self):
        """Test executing a simple pipeline"""
        pipeline = {
            "name": "simple-test",
            "steps": [
                {
                    "name": "test-step",
                    "agent": "mock-agent",
                    "type": "agent"
                }
            ]
        }

        # Mock agent
        mock_agent_class = Mock()
        mock_agent = Mock()
        mock_agent.run.return_value = Result(success=True, data={"output": "test"})
        mock_agent_class.return_value = mock_agent

        self.executor.loader.get_agent = Mock(return_value=mock_agent_class)

        result = self.executor._exec_local(pipeline, {"input": "test"})
        assert result.success is True
        assert "test-step" in result.data

    def test_execute_multi_step_pipeline(self):
        """Test executing multi-step pipeline"""
        pipeline = {
            "name": "multi-step-test",
            "steps": [
                {"name": "step1", "agent": "agent1", "type": "agent"},
                {"name": "step2", "agent": "agent2", "type": "agent"}
            ]
        }

        # Mock agents
        mock_agent1 = Mock()
        mock_agent1.run.return_value = Result(success=True, data={"step1_output": "value1"})

        mock_agent2 = Mock()
        mock_agent2.run.return_value = Result(success=True, data={"step2_output": "value2"})

        def get_agent(name):
            if name == "agent1":
                return lambda: mock_agent1
            elif name == "agent2":
                return lambda: mock_agent2
            return None

        self.executor.loader.get_agent = get_agent

        result = self.executor._exec_local(pipeline, {"input": "test"})
        assert result.success is True
        assert "step1" in result.data
        assert "step2" in result.data

    def test_execute_python_stage(self):
        """Test executing Python code stage"""
        pipeline = {
            "name": "python-test",
            "steps": [
                {
                    "name": "python-step",
                    "type": "python",
                    "code": "outputs['result'] = inputs.get('value', 0) * 2"
                }
            ]
        }

        result = self.executor._exec_local(pipeline, {"value": 5})
        assert result.success is True
        assert result.data["python-step"]["result"] == 10

    def test_execute_shell_stage(self):
        """Test executing shell command stage"""
        pipeline = {
            "name": "shell-test",
            "steps": [
                {
                    "name": "shell-step",
                    "type": "shell",
                    "command": "echo 'Hello ${value}'"
                }
            ]
        }

        result = self.executor._exec_local(pipeline, {"value": "World"})
        assert result.success is True
        assert "Hello World" in result.data["shell-step"]["stdout"]

    def test_execute_with_deterministic_mode(self):
        """Test executing with deterministic mode enabled"""
        self.executor.deterministic = True

        pipeline = {
            "name": "deterministic-test",
            "steps": [
                {
                    "name": "random-step",
                    "type": "python",
                    "code": "import random; outputs['value'] = deterministic_random().randint(1, 100)"
                }
            ]
        }

        # Run multiple times, should get same result
        result1 = self.executor._exec_local(pipeline, {})
        result2 = self.executor._exec_local(pipeline, {})

        assert result1.data["random-step"]["value"] == result2.data["random-step"]["value"]

    def test_execute_with_error_handling(self):
        """Test pipeline execution with error handling"""
        pipeline = {
            "name": "error-test",
            "steps": [
                {
                    "name": "failing-step",
                    "type": "python",
                    "code": "raise ValueError('Test error')"
                }
            ],
            "stop_on_error": True
        }

        result = self.executor._exec_local(pipeline, {})
        assert result.success is False
        assert "Test error" in result.error

    def test_execute_continue_on_error(self):
        """Test pipeline execution continuing on error"""
        pipeline = {
            "name": "continue-test",
            "steps": [
                {
                    "name": "failing-step",
                    "type": "python",
                    "code": "raise ValueError('Test error')"
                },
                {
                    "name": "success-step",
                    "type": "python",
                    "code": "outputs['result'] = 'success'"
                }
            ],
            "stop_on_error": False
        }

        result = self.executor._exec_local(pipeline, {})
        # Pipeline continues despite error
        assert "success-step" in result.data


class TestK8sExecution:
    """Test Kubernetes execution"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor(backend="k8s")

    def test_create_k8s_job_manifest(self):
        """Test creating Kubernetes job manifest"""
        pipeline = {
            "name": "k8s-test",
            "image": "python:3.9",
            "command": "python test.py",
            "retries": 3
        }

        manifest = self.executor._create_k8s_job(pipeline, {"input": "test"})

        assert manifest["kind"] == "Job"
        assert manifest["apiVersion"] == "batch/v1"
        assert manifest["spec"]["backoffLimit"] == 3
        assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == "python:3.9"

    def test_create_k8s_job_with_deterministic(self):
        """Test creating K8s job with deterministic settings"""
        self.executor.deterministic = True
        self.executor.det_config = DeterministicConfig(seed=123)

        pipeline = {"name": "det-test"}
        manifest = self.executor._create_k8s_job(pipeline, {})

        env_vars = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        env_names = [env["name"] for env in env_vars]

        assert "PYTHONHASHSEED" in env_names
        assert "RANDOM_SEED" in env_names

    @patch('subprocess.run')
    def test_wait_for_k8s_job_success(self, mock_subprocess):
        """Test waiting for successful K8s job completion"""
        # Mock successful job status
        job_status = {
            "status": {"succeeded": 1}
        }
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps(job_status)
        )

        result = self.executor._wait_for_k8s_job("test-job")
        assert result is True

    @patch('subprocess.run')
    def test_wait_for_k8s_job_failure(self, mock_subprocess):
        """Test waiting for failed K8s job"""
        # Mock failed job status
        job_status = {
            "status": {"failed": 1}
        }
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps(job_status)
        )

        with pytest.raises(RuntimeError, match="Job test-job failed"):
            self.executor._wait_for_k8s_job("test-job")

    @patch('time.sleep')
    @patch('time.time')
    @patch('subprocess.run')
    def test_wait_for_k8s_job_timeout(self, mock_subprocess, mock_time, mock_sleep):
        """Test K8s job timeout"""
        # Mock timeout scenario
        mock_time.side_effect = [0, 310]  # Start at 0, then exceed timeout
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=json.dumps({"status": {}})
        )

        with pytest.raises(TimeoutError, match="timed out"):
            self.executor._wait_for_k8s_job("test-job", timeout=300)

    @patch('subprocess.run')
    def test_get_k8s_job_logs(self, mock_subprocess):
        """Test getting K8s job logs"""
        mock_subprocess.return_value = Mock(stdout="Job output logs")

        logs = self.executor._get_k8s_job_logs("test-job")
        assert logs == "Job output logs"
        mock_subprocess.assert_called_with(
            ["kubectl", "logs", "job/test-job"],
            capture_output=True,
            text=True
        )

    def test_parse_k8s_outputs(self):
        """Test parsing outputs from K8s job logs"""
        logs = """
        Starting job...
        OUTPUT:{"result": "success", "value": 42}
        Some other log line
        OUTPUT:{"additional": "data"}
        Job completed
        """

        outputs = self.executor._parse_k8s_outputs(logs)
        assert outputs["result"] == "success"
        assert outputs["value"] == 42
        assert outputs["additional"] == "data"


class TestPipelineRun:
    """Test high-level pipeline run functionality"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor()

    @patch('greenlang.runtime.executor.Context')
    def test_run_pipeline_basic(self, mock_context_class):
        """Test basic pipeline run"""
        # Setup mocks
        mock_context = Mock()
        mock_context.data = {}
        mock_context.steps = []
        mock_context.to_result.return_value = Result(success=True, data={"final": "result"})
        mock_context_class.return_value = mock_context

        # Mock pipeline
        pipeline_data = {
            "name": "test-pipeline",
            "steps": [
                {"name": "test-step", "agent": "test-agent", "action": "process"}
            ]
        }

        # Mock agent
        mock_agent = Mock()
        mock_agent.process.return_value = Result(success=True, data={"output": "test"})
        self.executor.loader.get_agent = Mock(return_value=lambda: mock_agent)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(pipeline_data, f)
            pipeline_path = f.name

        try:
            result = self.executor.run(pipeline_path, {"input": "test"})
            assert result.success is True
        finally:
            Path(pipeline_path).unlink(missing_ok=True)

    def test_run_with_save_artifacts(self):
        """Test pipeline run with artifact saving"""
        pipeline_data = {
            "name": "artifact-test",
            "steps": [
                {
                    "name": "artifact-step",
                    "agent": "test-agent",
                    "save_artifacts": True
                }
            ]
        }

        # Mock agent and context
        mock_agent = Mock()
        mock_agent.process.return_value = Result(success=True, data={"output": "artifact"})
        self.executor.loader.get_agent = Mock(return_value=lambda: mock_agent)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(pipeline_data, f)
            pipeline_path = f.name

        try:
            with patch('greenlang.runtime.executor.Context') as mock_context_class:
                mock_context = Mock()
                mock_context.data = {}
                mock_context.steps = []
                mock_context.to_result.return_value = Result(success=True, data={"final": "result"})
                mock_context_class.return_value = mock_context

                result = self.executor.run(pipeline_path)

                # Verify save_artifact was called
                mock_context.save_artifact.assert_called_once()
        finally:
            Path(pipeline_path).unlink(missing_ok=True)


class TestExecutorUtilities:
    """Test executor utility methods"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor()

    def test_prepare_step_input_with_mapping(self):
        """Test preparing step input with mapping"""
        step = {
            "input": {
                "param1": "$input.value1",
                "param2": "$results.previous_step",
                "param3": "literal_value"
            }
        }

        context = {
            "input": {"value1": "input_data"},
            "results": {"previous_step": "previous_result"}
        }

        step_input = self.executor._prepare_step_input(step, context)
        assert step_input["param1"] == "input_data"
        assert step_input["param2"] == "previous_result"
        assert step_input["param3"] == "literal_value"

    def test_prepare_step_input_direct(self):
        """Test preparing step input without mapping"""
        step = {"input": "direct"}
        context = {"input": {"direct_value": "test"}}

        step_input = self.executor._prepare_step_input(step, context)
        assert step_input == {"direct_value": "test"}

    def test_collect_output_with_mapping(self):
        """Test collecting pipeline output with mapping"""
        pipeline = {
            "output": {
                "final_result": "$results.step1",
                "status": "$results.step2"
            }
        }

        context = {
            "results": {
                "step1": {"value": 42},
                "step2": {"success": True}
            }
        }

        output = self.executor._collect_output(pipeline, context)
        assert output["final_result"] == {"value": 42}
        assert output["status"] == {"success": True}

    def test_collect_output_all_results(self):
        """Test collecting all results as output"""
        pipeline = {}  # No output mapping
        context = {
            "results": {
                "step1": {"data": "value1"},
                "step2": {"data": "value2"}
            }
        }

        output = self.executor._collect_output(pipeline, context)
        assert output == context["results"]

    def test_normalize_outputs_floats(self):
        """Test output normalization for floats"""
        self.executor.deterministic = True
        self.executor.det_config = DeterministicConfig(float_precision=2)

        outputs = {
            "float_value": 3.14159,
            "nested": {"another_float": 2.71828}
        }

        normalized = self.executor._normalize_outputs(outputs)
        assert normalized["float_value"] == 3.14
        assert normalized["nested"]["another_float"] == 2.72

    def test_normalize_outputs_quantization(self):
        """Test output normalization with quantization"""
        self.executor.deterministic = True
        self.executor.det_config = DeterministicConfig(quantization_bits=4)

        outputs = {"value": 0.123456789}
        normalized = self.executor._normalize_outputs(outputs)

        # With 4 bits, scale = 16, so value should be quantized
        assert isinstance(normalized["value"], float)
        assert normalized["value"] != 0.123456789

    @patch('greenlang.runtime.executor.HAS_NUMPY', True)
    def test_normalize_outputs_numpy_arrays(self):
        """Test output normalization for numpy arrays"""
        import numpy as np

        self.executor.deterministic = True
        self.executor.det_config = DeterministicConfig(float_precision=2)

        outputs = {"array": np.array([1.234, 5.678, 9.101112])}

        with patch('greenlang.runtime.executor.np') as mock_np:
            mock_np.float32 = np.float32
            mock_np.float64 = np.float64
            mock_np.round = np.round

            normalized = self.executor._normalize_outputs(outputs)
            # Should call numpy round
            mock_np.round.assert_called()


class TestExecutorLedger:
    """Test executor run ledger functionality"""

    def setup_method(self):
        """Setup test executor"""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = Executor()
        self.executor.artifacts_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_run_record(self):
        """Test saving run record to ledger"""
        record = {
            "run_id": "test-123",
            "pipeline": "test-pipeline",
            "status": "success",
            "duration_seconds": 1.5
        }

        self.executor._save_run_record(record)

        # Check in-memory ledger
        assert len(self.executor.run_ledger) == 1
        assert self.executor.run_ledger[0] == record

        # Check file persistence
        ledger_file = self.executor.artifacts_dir / "run_ledger.jsonl"
        assert ledger_file.exists()

        with open(ledger_file) as f:
            saved_record = json.loads(f.read().strip())
        assert saved_record == record

    def test_list_runs(self):
        """Test listing runs from ledger"""
        # Initially empty
        runs = self.executor.list_runs()
        assert runs == []

        # Add some runs
        records = [
            {"run_id": "run1", "status": "success"},
            {"run_id": "run2", "status": "failed"}
        ]

        for record in records:
            self.executor._save_run_record(record)

        runs = self.executor.list_runs()
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run1"
        assert runs[1]["run_id"] == "run2"

    def test_get_run(self):
        """Test getting specific run details"""
        run_id = "test-run-123"

        # No run initially
        run_details = self.executor.get_run(run_id)
        assert run_details is None

        # Create run.json
        run_data = {
            "run_id": run_id,
            "pipeline": {"name": "test"},
            "status": "success"
        }

        run_file = self.executor.artifacts_dir / f"run_{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run_data, f)

        # Get run details
        run_details = self.executor.get_run(run_id)
        assert run_details["run_id"] == run_id
        assert run_details["status"] == "success"

    def test_generate_run_json(self):
        """Test generating run.json file"""
        run_id = "test-generate-123"
        pipeline = {"name": "test-pipeline", "version": "1.0"}
        context = {
            "input": {"param": "value"},
            "artifacts": ["artifact1.json"]
        }
        result = Result(success=True, data={"output": "result"})

        self.executor._generate_run_json(run_id, pipeline, context, result)

        # Check file was created
        run_file = self.executor.artifacts_dir / f"run_{run_id}.json"
        assert run_file.exists()

        # Check content
        with open(run_file) as f:
            run_data = json.load(f)

        assert run_data["run_id"] == run_id
        assert run_data["pipeline"] == pipeline
        assert run_data["input"] == context["input"]
        assert run_data["output"] == result.data
        assert run_data["status"] == "success"


class TestRetryAndBackoff:
    """Test retry and backoff logic"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor()

    def test_k8s_job_with_retries(self):
        """Test K8s job creation with retry configuration"""
        pipeline = {
            "name": "retry-test",
            "retries": 5
        }

        manifest = self.executor._create_k8s_job(pipeline, {})
        assert manifest["spec"]["backoffLimit"] == 5

    def test_default_retry_count(self):
        """Test default retry count for K8s jobs"""
        pipeline = {"name": "default-test"}

        manifest = self.executor._create_k8s_job(pipeline, {})
        assert manifest["spec"]["backoffLimit"] == 1  # Default


class TestExecutorContext:
    """Test executor context management"""

    def setup_method(self):
        """Setup test executor"""
        self.executor = Executor(deterministic=True)

    def test_context_creation(self):
        """Test execution context creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)

            with self.executor.context(artifacts_dir) as ctx:
                assert ctx.backend == self.executor.backend
                assert hasattr(ctx, 'versions')
                assert 'python' in ctx.versions
                assert 'backend' in ctx.versions
                assert 'deterministic' in ctx.versions

    def test_context_environment_freezing(self):
        """Test environment freezing in deterministic mode"""
        self.executor.det_config.freeze_env = True

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)

            with self.executor.context(artifacts_dir) as ctx:
                assert hasattr(ctx, 'environment')
                assert isinstance(ctx.environment, dict)


@pytest.mark.parametrize("backend,expected_method", [
    ("local", "_exec_local"),
    ("k8s", "_exec_k8s"),
    ("kubernetes", "_exec_k8s"),
])
def test_execute_backend_routing(backend, expected_method):
    """Parametrized test for backend routing"""
    executor = Executor(backend="local")  # Start with local to avoid k8s validation
    executor.backend = backend  # Override for test

    pipeline = {"name": "test", "steps": []}
    inputs = {"test": "data"}

    with patch.object(executor, expected_method) as mock_method:
        mock_method.return_value = Result(success=True, data={})

        executor.execute(pipeline, inputs)
        mock_method.assert_called_once_with(pipeline, inputs)


@pytest.mark.parametrize("deterministic,should_normalize", [
    (True, True),
    (False, False),
])
def test_deterministic_output_normalization(deterministic, should_normalize):
    """Parametrized test for deterministic output normalization"""
    executor = Executor(deterministic=deterministic)
    executor.det_config.normalize_floats = True

    pipeline = {
        "name": "norm-test",
        "steps": [
            {
                "name": "float-step",
                "type": "python",
                "code": "outputs['pi'] = 3.14159265359"
            }
        ]
    }

    result = executor._exec_local(pipeline, {})

    if should_normalize:
        # Should be rounded to default precision (6 places)
        assert result.data["float-step"]["pi"] == 3.141593
    else:
        # Should remain unmodified
        assert result.data["float-step"]["pi"] == 3.14159265359