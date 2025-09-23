"""
Comprehensive tests for greenlang.runtime.executor module
"""

import pytest
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
from datetime import datetime

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang.runtime.executor import (
    DeterministicConfig, Executor
)
from greenlang.sdk.base import Result
from greenlang.sdk.context import Context


class TestDeterministicConfig:
    """Test DeterministicConfig class"""

    def test_deterministic_config_creation(self):
        """Test creating DeterministicConfig with defaults"""
        config = DeterministicConfig()
        assert config.seed == 42
        assert config.freeze_env is True
        assert config.normalize_floats is True
        assert config.float_precision == 6
        assert config.quantization_bits is None

    def test_deterministic_config_custom(self):
        """Test creating DeterministicConfig with custom values"""
        config = DeterministicConfig(
            seed=123,
            freeze_env=False,
            normalize_floats=False,
            float_precision=3,
            quantization_bits=8
        )
        assert config.seed == 123
        assert config.freeze_env is False
        assert config.normalize_floats is False
        assert config.float_precision == 3
        assert config.quantization_bits == 8

    @patch('greenlang.runtime.executor.random')
    @patch('greenlang.runtime.executor.os')
    def test_deterministic_config_apply_basic(self, mock_os, mock_random):
        """Test DeterministicConfig.apply() basic functionality"""
        config = DeterministicConfig(seed=999)
        config.apply()

        mock_random.seed.assert_called_once_with(999)
        assert mock_os.environ.__setitem__.call_count >= 2  # PYTHONHASHSEED and PYTHONDONTWRITEBYTECODE

    @patch('greenlang.runtime.executor.HAS_NUMPY', True)
    @patch('greenlang.runtime.executor.np')
    @patch('greenlang.runtime.executor.random')
    def test_deterministic_config_apply_with_numpy(self, mock_random, mock_np):
        """Test DeterministicConfig.apply() with numpy available"""
        config = DeterministicConfig(seed=456)
        config.apply()

        mock_random.seed.assert_called_once_with(456)
        mock_np.random.seed.assert_called_once_with(456)

    @patch('greenlang.runtime.executor.importlib.import_module')
    @patch('greenlang.runtime.executor.random')
    def test_deterministic_config_apply_with_tensorflow(self, mock_random, mock_import):
        """Test DeterministicConfig.apply() with TensorFlow"""
        mock_tf = MagicMock()
        mock_import.side_effect = lambda name: mock_tf if name == 'tensorflow' else MagicMock()

        config = DeterministicConfig(seed=789)
        config.apply()

        mock_random.seed.assert_called_once_with(789)

    @patch('greenlang.runtime.executor.importlib.import_module')
    @patch('greenlang.runtime.executor.random')
    def test_deterministic_config_apply_with_torch(self, mock_random, mock_import):
        """Test DeterministicConfig.apply() with PyTorch"""
        mock_torch = MagicMock()
        mock_import.side_effect = lambda name: mock_torch if name == 'torch' else MagicMock()

        config = DeterministicConfig(seed=101)
        config.apply()

        mock_random.seed.assert_called_once_with(101)

    @patch('greenlang.runtime.executor.random')
    def test_deterministic_config_apply_import_errors(self, mock_random):
        """Test DeterministicConfig.apply() handles import errors gracefully"""
        config = DeterministicConfig()

        # Should not raise exceptions even if TF/PyTorch not available
        config.apply()
        mock_random.seed.assert_called_once()


class TestExecutor:
    """Test Executor class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_artifacts_dir = Path("/tmp/test_artifacts")

    def test_executor_creation_default(self):
        """Test creating Executor with default settings"""
        executor = Executor()
        assert executor.backend == "local"
        assert executor.profile == "local"  # Backward compatibility
        assert executor.deterministic is False
        assert isinstance(executor.det_config, DeterministicConfig)
        assert executor.run_ledger == []

    def test_executor_creation_custom(self):
        """Test creating Executor with custom settings"""
        det_config = DeterministicConfig(seed=999)
        executor = Executor(
            backend="k8s",
            deterministic=True,
            det_config=det_config
        )
        assert executor.backend == "k8s"
        assert executor.deterministic is True
        assert executor.det_config is det_config

    def test_executor_validate_backend_valid(self):
        """Test backend validation with valid backends"""
        valid_backends = ["local", "k8s", "kubernetes", "cloud"]
        for backend in valid_backends:
            executor = Executor(backend=backend)
            assert executor.backend == backend

    def test_executor_validate_backend_invalid(self):
        """Test backend validation with invalid backend"""
        with pytest.raises(ValueError, match="Unknown backend"):
            Executor(backend="invalid_backend")

    @patch('subprocess.run')
    def test_executor_validate_backend_k8s_available(self, mock_run):
        """Test backend validation when kubectl is available"""
        mock_run.return_value = MagicMock()
        executor = Executor(backend="k8s")
        assert executor.backend == "k8s"
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_executor_validate_backend_k8s_unavailable(self, mock_run):
        """Test backend validation when kubectl is unavailable"""
        mock_run.side_effect = FileNotFoundError()
        executor = Executor(backend="k8s")
        assert executor.backend == "local"  # Should fall back

    @patch('subprocess.run')
    def test_executor_validate_backend_k8s_timeout(self, mock_run):
        """Test backend validation when kubectl times out"""
        mock_run.side_effect = subprocess.TimeoutExpired("kubectl", 5)
        executor = Executor(backend="kubernetes")
        assert executor.backend == "local"  # Should fall back

    def test_executor_context_manager_basic(self):
        """Test Executor context manager basic functionality"""
        executor = Executor()

        with patch('greenlang.runtime.executor.ProvenanceContext') as mock_prov_ctx:
            mock_ctx = MagicMock()
            mock_prov_ctx.return_value = mock_ctx

            with executor.context(self.test_artifacts_dir) as ctx:
                assert ctx is mock_ctx
                assert ctx.backend == "local"

    def test_executor_context_manager_deterministic(self):
        """Test Executor context manager with deterministic execution"""
        det_config = DeterministicConfig(seed=123, freeze_env=True)
        executor = Executor(deterministic=True, det_config=det_config)

        with patch('greenlang.runtime.executor.ProvenanceContext') as mock_prov_ctx, \
             patch.object(det_config, 'apply') as mock_apply:

            mock_ctx = MagicMock()
            mock_prov_ctx.return_value = mock_ctx

            with executor.context(self.test_artifacts_dir) as ctx:
                mock_apply.assert_called_once()
                assert hasattr(ctx, 'environment')
                assert hasattr(ctx, 'versions')

    def test_executor_execute_local_backend(self):
        """Test Executor.execute() with local backend"""
        executor = Executor(backend="local")
        pipeline = {"name": "test_pipeline"}
        inputs = {"test": "data"}

        with patch.object(executor, '_exec_local') as mock_exec:
            mock_exec.return_value = Result(success=True, data={"result": "test"})

            result = executor.execute(pipeline, inputs)
            assert result.success is True
            mock_exec.assert_called_once_with(pipeline, inputs)

    def test_executor_execute_k8s_backend(self):
        """Test Executor.execute() with k8s backend"""
        executor = Executor(backend="k8s")
        pipeline = {"name": "test_pipeline"}
        inputs = {"test": "data"}

        with patch.object(executor, '_exec_k8s') as mock_exec:
            mock_exec.return_value = Result(success=True, data={"result": "test"})

            result = executor.execute(pipeline, inputs)
            assert result.success is True
            mock_exec.assert_called_once_with(pipeline, inputs)

    def test_executor_execute_unknown_backend(self):
        """Test Executor.execute() with unknown backend"""
        # Bypass validation by setting backend directly
        executor = Executor()
        executor.backend = "unknown"

        with pytest.raises(ValueError, match="Unknown backend"):
            executor.execute({}, {})

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_pipeline_from_file(self, mock_yaml, mock_file, mock_exists):
        """Test _load_pipeline from file"""
        executor = Executor()
        mock_exists.return_value = True
        mock_yaml.return_value = {"name": "test_pipeline", "steps": []}

        pipeline = executor._load_pipeline("/path/to/pipeline.yaml")
        assert pipeline["name"] == "test_pipeline"
        mock_exists.assert_called_once()
        mock_yaml.assert_called_once()

    def test_load_pipeline_from_pack_reference(self):
        """Test _load_pipeline from pack reference"""
        executor = Executor()
        mock_pack = MagicMock()
        mock_pack.get_pipeline.return_value = {"name": "pack_pipeline"}
        executor.loader.load.return_value = mock_pack

        with patch('pathlib.Path.exists', return_value=False):
            pipeline = executor._load_pipeline("test_pack/test_pipeline")

        assert pipeline["name"] == "pack_pipeline"
        executor.loader.load.assert_called_once_with("test_pack")

    def test_load_pipeline_not_found(self):
        """Test _load_pipeline when pipeline not found"""
        executor = Executor()

        with patch('pathlib.Path.exists', return_value=False), \
             patch.object(executor.loader, 'load', side_effect=Exception("Pack not found")):

            with pytest.raises(ValueError, match="Pipeline not found"):
                executor._load_pipeline("nonexistent_pipeline")

    def test_run_method_basic(self):
        """Test Executor.run() basic functionality"""
        executor = Executor()
        pipeline_data = {
            "name": "test_pipeline",
            "steps": [
                {
                    "name": "step1",
                    "agent": "test_agent",
                    "action": "process"
                }
            ]
        }

        mock_agent_class = MagicMock()
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.process.return_value = Result(success=True, data={"output": "test"})

        with patch.object(executor, '_load_pipeline', return_value=pipeline_data), \
             patch.object(executor.loader, 'get_agent', return_value=mock_agent_class):

            result = executor.run("test_pipeline", {"input": "data"})

        assert result.success is True

    def test_run_method_with_artifacts_dir(self):
        """Test Executor.run() with custom artifacts directory"""
        executor = Executor()
        pipeline_data = {"name": "test", "steps": []}

        with patch.object(executor, '_load_pipeline', return_value=pipeline_data):
            result = executor.run("test", {}, artifacts_dir=Path("/custom/artifacts"))

        # Should use custom artifacts directory
        assert result is not None

    def test_exec_local_basic(self):
        """Test _exec_local method basic functionality"""
        executor = Executor()
        pipeline = {
            "name": "test_pipeline",
            "steps": [
                {
                    "name": "step1",
                    "type": "agent",
                    "agent": "test_agent"
                }
            ]
        }
        inputs = {"test": "data"}

        mock_agent_class = MagicMock()
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.run.return_value = Result(success=True, data={"result": "success"})

        with patch.object(executor.loader, 'get_agent', return_value=mock_agent_class):
            result = executor._exec_local(pipeline, inputs)

        assert result.success is True
        assert "step1" in result.data
        assert result.metadata["backend"] == "local"

    def test_exec_local_deterministic(self):
        """Test _exec_local with deterministic execution"""
        det_config = DeterministicConfig(seed=42)
        executor = Executor(deterministic=True, det_config=det_config)
        pipeline = {"name": "test", "steps": []}

        with patch.object(det_config, 'apply') as mock_apply:
            result = executor._exec_local(pipeline, {})

        mock_apply.assert_called_once()
        assert result.metadata["deterministic"] is True

    def test_exec_local_python_stage(self):
        """Test _exec_local with Python code stage"""
        executor = Executor()
        pipeline = {
            "name": "test",
            "steps": [
                {
                    "name": "python_step",
                    "type": "python",
                    "code": "outputs['result'] = inputs.get('value', 0) * 2"
                }
            ]
        }
        inputs = {"value": 5}

        result = executor._exec_local(pipeline, inputs)
        assert result.success is True
        assert result.data["python_step"]["result"] == 10

    def test_exec_local_shell_stage(self):
        """Test _exec_local with shell command stage"""
        executor = Executor()
        pipeline = {
            "name": "test",
            "steps": [
                {
                    "name": "shell_step",
                    "type": "shell",
                    "command": "echo 'Hello World'"
                }
            ]
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Hello World\n",
                stderr="",
                returncode=0
            )

            result = executor._exec_local(pipeline, {})

        assert result.success is True
        mock_run.assert_called_once()

    def test_exec_local_error_handling(self):
        """Test _exec_local error handling"""
        executor = Executor()
        pipeline = {
            "name": "test",
            "steps": [
                {
                    "name": "failing_step",
                    "type": "agent",
                    "agent": "nonexistent_agent"
                }
            ]
        }

        with patch.object(executor.loader, 'get_agent', return_value=None):
            result = executor._exec_local(pipeline, {})

        assert result.success is False
        assert "duration" in result.metadata

    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('yaml.dump')
    def test_exec_k8s_basic(self, mock_yaml_dump, mock_run, mock_temp):
        """Test _exec_k8s basic functionality"""
        executor = Executor(backend="k8s")
        pipeline = {"name": "test_pipeline"}
        inputs = {"test": "data"}

        # Mock temporary file
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/manifest.yaml"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        # Mock kubectl commands
        mock_run.side_effect = [
            MagicMock(),  # kubectl apply
            MagicMock(),  # kubectl delete (cleanup)
        ]

        with patch.object(executor, '_wait_for_k8s_job'), \
             patch.object(executor, '_get_k8s_job_logs', return_value="OUTPUT: {}"), \
             patch('pathlib.Path.unlink'):

            result = executor._exec_k8s(pipeline, inputs)

        assert result.success is True
        assert result.metadata["backend"] == "k8s"

    def test_exec_k8s_deterministic(self):
        """Test _exec_k8s with deterministic execution"""
        det_config = DeterministicConfig(seed=123)
        executor = Executor(backend="k8s", deterministic=True, det_config=det_config)

        manifest = executor._create_k8s_job({"name": "test"}, {})

        # Should have deterministic environment variables
        env_vars = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        env_names = [env["name"] for env in env_vars]
        assert "PYTHONHASHSEED" in env_names
        assert "RANDOM_SEED" in env_names

    @patch('time.sleep')
    @patch('subprocess.run')
    @patch('json.loads')
    def test_wait_for_k8s_job_success(self, mock_json, mock_run, mock_sleep):
        """Test _wait_for_k8s_job successful completion"""
        executor = Executor()

        # Mock successful job status
        mock_json.return_value = {"status": {"succeeded": 1}}
        mock_run.return_value = MagicMock(returncode=0, stdout='{"status": {"succeeded": 1}}')

        result = executor._wait_for_k8s_job("test-job")
        assert result is True

    @patch('time.sleep')
    @patch('subprocess.run')
    @patch('json.loads')
    def test_wait_for_k8s_job_failure(self, mock_json, mock_run, mock_sleep):
        """Test _wait_for_k8s_job job failure"""
        executor = Executor()

        # Mock failed job status
        mock_json.return_value = {"status": {"failed": 1}}
        mock_run.return_value = MagicMock(returncode=0, stdout='{"status": {"failed": 1}}')

        with pytest.raises(RuntimeError, match="Job test-job failed"):
            executor._wait_for_k8s_job("test-job")

    @patch('time.sleep')
    @patch('time.time')
    @patch('subprocess.run')
    def test_wait_for_k8s_job_timeout(self, mock_run, mock_time, mock_sleep):
        """Test _wait_for_k8s_job timeout"""
        executor = Executor()

        # Mock time progression to trigger timeout
        mock_time.side_effect = [0, 301]  # Start time, then past timeout
        mock_run.return_value = MagicMock(returncode=0, stdout='{"status": {}}')

        with pytest.raises(TimeoutError, match="timed out"):
            executor._wait_for_k8s_job("test-job", timeout=300)

    @patch('subprocess.run')
    def test_get_k8s_job_logs(self, mock_run):
        """Test _get_k8s_job_logs method"""
        executor = Executor()
        mock_run.return_value = MagicMock(stdout="Job output logs")

        logs = executor._get_k8s_job_logs("test-job")
        assert logs == "Job output logs"
        mock_run.assert_called_once_with(
            ["kubectl", "logs", "job/test-job"],
            capture_output=True,
            text=True
        )

    def test_parse_k8s_outputs(self):
        """Test _parse_k8s_outputs method"""
        executor = Executor()
        logs = """
        Starting job...
        OUTPUT: {"result": "success", "value": 42}
        Some other log line
        OUTPUT: {"additional": "data"}
        Job completed
        """

        outputs = executor._parse_k8s_outputs(logs)
        assert outputs["result"] == "success"
        assert outputs["value"] == 42
        assert outputs["additional"] == "data"

    def test_parse_k8s_outputs_invalid_json(self):
        """Test _parse_k8s_outputs with invalid JSON"""
        executor = Executor()
        logs = """
        OUTPUT: {"valid": "json"}
        OUTPUT: invalid json here
        OUTPUT: {"another": "valid"}
        """

        outputs = executor._parse_k8s_outputs(logs)
        assert outputs["valid"] == "json"
        assert outputs["another"] == "valid"
        # Invalid JSON should be ignored

    @pytest.mark.parametrize("test_input,expected", [
        (1.23456789, 1.234568),  # Default precision 6
        ({"nested": 2.71828}, {"nested": 2.718280}),
        ([1.111, 2.222], [1.111000, 2.222000]),
        ("string", "string"),  # Non-numeric should pass through
        (42, 42),  # Integers should pass through
    ])
    def test_normalize_outputs(self, test_input, expected):
        """Test _normalize_outputs method"""
        det_config = DeterministicConfig(float_precision=6)
        executor = Executor(deterministic=True, det_config=det_config)

        result = executor._normalize_outputs(test_input)
        assert result == expected

    def test_normalize_outputs_quantization(self):
        """Test _normalize_outputs with quantization"""
        det_config = DeterministicConfig(quantization_bits=4)
        executor = Executor(deterministic=True, det_config=det_config)

        # With 4 bits, scale = 16, so 0.1 * 16 = 1.6 -> 2 -> 2/16 = 0.125
        result = executor._normalize_outputs(0.1)
        assert result == 0.125

    @patch('greenlang.runtime.executor.HAS_NUMPY', True)
    @patch('greenlang.runtime.executor.np')
    def test_normalize_outputs_numpy(self, mock_np):
        """Test _normalize_outputs with numpy arrays"""
        det_config = DeterministicConfig(float_precision=3)
        executor = Executor(deterministic=True, det_config=det_config)

        # Mock numpy array
        mock_array = MagicMock()
        mock_array.dtype = mock_np.float64
        mock_np.round.return_value = mock_array

        result = executor._normalize_outputs(mock_array)
        mock_np.round.assert_called_once_with(mock_array, 3)

    def test_run_legacy_method(self):
        """Test run_legacy method"""
        executor = Executor()
        mock_pipeline = {"name": "test", "steps": []}

        with patch.object(executor.loader, 'get_pipeline', return_value=mock_pipeline), \
             patch.object(executor, '_run_local') as mock_run_local, \
             patch.object(executor, '_save_run_record'), \
             patch.object(executor, '_generate_run_json'):

            mock_run_local.return_value = Result(success=True, data={"result": "test"})

            result = executor.run_legacy("test.pipeline", {"input": "data"})

        assert result.success is True
        mock_run_local.assert_called_once()

    def test_run_legacy_pipeline_not_found(self):
        """Test run_legacy with pipeline not found"""
        executor = Executor()

        with patch.object(executor.loader, 'get_pipeline', return_value=None):
            result = executor.run_legacy("nonexistent.pipeline", {})

        assert result.success is False
        assert "Pipeline not found" in result.error

    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_run_record(self, mock_file, mock_mkdir):
        """Test _save_run_record method"""
        executor = Executor()
        record = {
            "run_id": "test-123",
            "pipeline": "test.pipeline",
            "status": "success"
        }

        executor._save_run_record(record)

        assert len(executor.run_ledger) == 1
        assert executor.run_ledger[0] == record
        mock_file.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_generate_run_json(self, mock_json_dump, mock_file):
        """Test _generate_run_json method"""
        executor = Executor()
        pipeline = {"name": "test"}
        context = {"input": {"test": "data"}, "artifacts": []}
        result = Result(success=True, data={"output": "result"})

        executor._generate_run_json("test-123", pipeline, context, result)

        mock_json_dump.assert_called_once()
        run_data = mock_json_dump.call_args[0][0]
        assert run_data["run_id"] == "test-123"
        assert run_data["status"] == "success"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_list_runs_empty(self, mock_file, mock_exists):
        """Test list_runs with no existing ledger"""
        executor = Executor()
        mock_exists.return_value = False

        runs = executor.list_runs()
        assert runs == []

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.loads')
    def test_list_runs_with_data(self, mock_json_loads, mock_file, mock_exists):
        """Test list_runs with existing data"""
        executor = Executor()
        mock_exists.return_value = True
        mock_file.return_value.__iter__ = lambda self: iter([
            '{"run_id": "1", "status": "success"}\n',
            '{"run_id": "2", "status": "failed"}\n'
        ])
        mock_json_loads.side_effect = [
            {"run_id": "1", "status": "success"},
            {"run_id": "2", "status": "failed"}
        ]

        runs = executor.list_runs()
        assert len(runs) == 2
        assert runs[0]["run_id"] == "1"
        assert runs[1]["run_id"] == "2"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_get_run_existing(self, mock_json_load, mock_file, mock_exists):
        """Test get_run with existing run"""
        executor = Executor()
        mock_exists.return_value = True
        mock_json_load.return_value = {"run_id": "test-123", "status": "success"}

        run_data = executor.get_run("test-123")
        assert run_data["run_id"] == "test-123"

    @patch('pathlib.Path.exists')
    def test_get_run_nonexistent(self, mock_exists):
        """Test get_run with non-existent run"""
        executor = Executor()
        mock_exists.return_value = False

        run_data = executor.get_run("nonexistent")
        assert run_data is None

    def test_create_k8s_job_basic(self):
        """Test _create_k8s_job basic manifest creation"""
        executor = Executor()
        pipeline = {"name": "test-pipeline"}
        inputs = {"test": "data"}

        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "abcd1234"
            manifest = executor._create_k8s_job(pipeline, inputs)

        assert manifest["kind"] == "Job"
        assert manifest["apiVersion"] == "batch/v1"
        assert "test-pipeline" in manifest["metadata"]["name"]
        assert manifest["metadata"]["labels"]["app"] == "greenlang"

    def test_prepare_step_input(self):
        """Test _prepare_step_input method"""
        executor = Executor()
        step = {
            "input": {
                "field1": "$input.source_field",
                "field2": "$results.previous_step",
                "field3": "literal_value"
            }
        }
        context = {
            "input": {"source_field": "input_value"},
            "results": {"previous_step": "result_value"}
        }

        step_input = executor._prepare_step_input(step, context)
        assert step_input["field1"] == "input_value"
        assert step_input["field2"] == "result_value"
        assert step_input["field3"] == "literal_value"

    def test_prepare_step_input_no_mapping(self):
        """Test _prepare_step_input with no input mapping"""
        executor = Executor()
        step = {}
        context = {"input": {"test": "data"}}

        step_input = executor._prepare_step_input(step, context)
        assert step_input == {"test": "data"}

    def test_collect_output(self):
        """Test _collect_output method"""
        executor = Executor()
        pipeline = {
            "output": {
                "final_result": "$results.step1",
                "metadata": "$results.step2"
            }
        }
        context = {
            "results": {
                "step1": {"value": 42},
                "step2": {"duration": 1.5}
            }
        }

        output = executor._collect_output(pipeline, context)
        assert output["final_result"] == {"value": 42}
        assert output["metadata"] == {"duration": 1.5}

    def test_collect_output_no_mapping(self):
        """Test _collect_output with no output mapping"""
        executor = Executor()
        pipeline = {}
        context = {"results": {"all": "results"}}

        output = executor._collect_output(pipeline, context)
        assert output == {"all": "results"}

    def test_exec_python_stage_basic(self):
        """Test _exec_python_stage method"""
        executor = Executor()
        stage = {"code": "outputs['doubled'] = inputs.get('value', 0) * 2"}
        context = {"input": {"value": 21}}

        result = executor._exec_python_stage(stage, context)
        assert result["doubled"] == 42

    def test_exec_python_stage_deterministic(self):
        """Test _exec_python_stage with deterministic execution"""
        det_config = DeterministicConfig(seed=999)
        executor = Executor(deterministic=True, det_config=det_config)
        stage = {"code": "outputs['seed'] = __seed__"}
        context = {"input": {}}

        result = executor._exec_python_stage(stage, context)
        assert result["seed"] == 999

    @patch('subprocess.run')
    def test_exec_shell_stage(self, mock_run):
        """Test _exec_shell_stage method"""
        executor = Executor()
        stage = {"command": "echo 'Hello ${name}'"}
        context = {"input": {"name": "World"}}

        mock_run.return_value = MagicMock(
            stdout="Hello World",
            stderr="",
            returncode=0
        )

        result = executor._exec_shell_stage(stage, context)
        assert result["stdout"] == "Hello World"
        assert result["returncode"] == 0

        # Verify command substitution
        mock_run.assert_called_once()
        called_command = mock_run.call_args[0][0]
        assert "Hello World" in called_command