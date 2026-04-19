# -*- coding: utf-8 -*-
"""
Tests for GreenLang Runtime Backends
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path
from datetime import datetime

from greenlang.runtime.backends import (
    Pipeline, PipelineStep, ExecutionContext, ExecutionStatus,
    BackendFactory, LocalBackend
)
from greenlang.runtime.backends.executor import PipelineExecutor, PipelineBuilder


class TestPipelineModels(unittest.TestCase):
    """Test pipeline data models"""
    
    def test_pipeline_step_creation(self):
        """Test creating pipeline step"""
        step = PipelineStep(
            name="test-step",
            command=["python", "-m"],
            args=["greenlang.test"],
            env={"TEST": "value"},
            image="python:3.11"
        )
        
        self.assertEqual(step.name, "test-step")
        self.assertEqual(step.command, ["python", "-m"])
        self.assertEqual(step.args, ["greenlang.test"])
        self.assertEqual(step.env["TEST"], "value")
    
    def test_pipeline_creation(self):
        """Test creating pipeline"""
        steps = [
            PipelineStep(name="step1", command=["echo", "hello"]),
            PipelineStep(name="step2", command=["echo", "world"], depends_on=["step1"])
        ]
        
        pipeline = Pipeline(
            name="test-pipeline",
            steps=steps,
            description="Test pipeline"
        )
        
        self.assertEqual(pipeline.name, "test-pipeline")
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[1].depends_on, ["step1"])
    
    def test_pipeline_serialization(self):
        """Test pipeline serialization"""
        pipeline = Pipeline(
            name="test",
            steps=[PipelineStep(name="step1", command=["echo"])]
        )
        
        data = pipeline.to_dict()
        self.assertEqual(data["name"], "test")
        self.assertIsInstance(data["steps"], list)
        
        # Test deserialization
        pipeline2 = Pipeline.from_dict(data)
        self.assertEqual(pipeline2.name, "test")
        self.assertEqual(len(pipeline2.steps), 1)
    
    def test_execution_context(self):
        """Test execution context"""
        context = ExecutionContext(
            user="testuser",
            project="testproject",
            parameters={"key": "value"}
        )
        
        self.assertEqual(context.user, "testuser")
        self.assertEqual(context.project, "testproject")
        self.assertEqual(context.parameters["key"], "value")
        
        # Test environment conversion
        env = context.to_env()
        self.assertIn("GL_RUN_ID", env)
        self.assertIn("GL_USER", env)
        self.assertEqual(env["GL_USER"], "testuser")


class TestBackendFactory(unittest.TestCase):
    """Test backend factory"""
    
    def test_backend_factory_creation(self):
        """Test creating backends with factory"""
        # Test local backend
        backend = BackendFactory.create("local", {"working_dir": "/tmp"})
        self.assertIsInstance(backend, LocalBackend)
        
        # Test listing backends
        backends = BackendFactory.list_backends()
        self.assertIn("local", backends)
        self.assertIn("docker", backends)
        self.assertIn("kubernetes", backends)
    
    def test_backend_info(self):
        """Test getting backend info"""
        info = BackendFactory.get_backend_info("local")
        self.assertEqual(info["class"], "LocalBackend")
        
        # Test unknown backend
        info = BackendFactory.get_backend_info("unknown")
        self.assertIn("error", info)


class TestLocalBackend(unittest.TestCase):
    """Test local backend"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = LocalBackend({"working_dir": self.temp_dir})
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_single_step_execution(self):
        """Test executing single step pipeline"""
        pipeline = Pipeline(
            name="test",
            steps=[
                PipelineStep(
                    name="echo",
                    command=["echo", "test output"]
                )
            ]
        )
        
        context = ExecutionContext()
        
        # Mock subprocess to avoid actual execution
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.wait.return_value = 0
            mock_process.poll.return_value = 0
            mock_popen.return_value = mock_process
            
            result = self.backend.execute(pipeline, context)
            
            self.assertEqual(result.pipeline_name, "test")
            self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
    
    def test_multi_step_execution(self):
        """Test executing multi-step pipeline"""
        pipeline = Pipeline(
            name="multi-test",
            steps=[
                PipelineStep(name="step1", command=["echo", "1"]),
                PipelineStep(name="step2", command=["echo", "2"], depends_on=["step1"])
            ]
        )
        
        context = ExecutionContext()
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.wait.return_value = 0
            mock_process.poll.return_value = 0
            mock_popen.return_value = mock_process
            
            result = self.backend.execute(pipeline, context)
            
            self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
            # Should have been called twice (once for each step)
            self.assertEqual(mock_popen.call_count, 2)
    
    def test_pipeline_validation(self):
        """Test pipeline validation"""
        # Invalid pipeline - no name
        pipeline = Pipeline(name="", steps=[])
        errors = self.backend.validate_pipeline(pipeline)
        self.assertIn("Pipeline name is required", errors)
        
        # Invalid pipeline - no steps
        pipeline = Pipeline(name="test", steps=[])
        errors = self.backend.validate_pipeline(pipeline)
        self.assertIn("Pipeline must have at least one step", errors)
        
        # Invalid step dependencies
        pipeline = Pipeline(
            name="test",
            steps=[
                PipelineStep(name="step1", command=["echo"]),
                PipelineStep(name="step2", command=["echo"], depends_on=["unknown"])
            ]
        )
        errors = self.backend.validate_pipeline(pipeline)
        self.assertTrue(any("unknown step" in e for e in errors))


class TestPipelineExecutor(unittest.TestCase):
    """Test pipeline executor"""
    
    def test_pipeline_builder(self):
        """Test building pipeline programmatically"""
        pipeline = (PipelineBuilder("test-pipeline")
                   .with_description("Test pipeline")
                   .add_step("step1", ["echo", "hello"])
                   .add_step("step2", ["echo", "world"], depends_on=["step1"])
                   .with_labels({"env": "test"})
                   .build())
        
        self.assertEqual(pipeline.name, "test-pipeline")
        self.assertEqual(pipeline.description, "Test pipeline")
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.labels["env"], "test")
    
    @patch('greenlang.runtime.backends.factory.BackendFactory.create')
    def test_pipeline_executor(self, mock_create):
        """Test pipeline executor"""
        # Mock backend
        mock_backend = MagicMock()
        mock_backend.execute.return_value = MagicMock(
            status=ExecutionStatus.SUCCEEDED,
            run_id="test-run",
            pipeline_name="test"
        )
        mock_create.return_value = mock_backend
        
        # Create executor
        executor = PipelineExecutor(backend_type="local")
        
        # Create pipeline
        pipeline = Pipeline(
            name="test",
            steps=[PipelineStep(name="test", command=["echo"])]
        )
        
        # Execute
        result = executor.execute(pipeline)
        
        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        mock_backend.execute.assert_called_once()
    
    def test_pipeline_loading(self):
        """Test loading pipeline from file"""
        # Create temp pipeline file
        pipeline_data = {
            "name": "test-pipeline",
            "steps": [
                {
                    "name": "step1",
                    "command": ["echo", "test"]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(pipeline_data, f)
            pipeline_file = f.name
        
        try:
            executor = PipelineExecutor()
            pipeline = executor.load_pipeline(pipeline_file)
            
            self.assertEqual(pipeline.name, "test-pipeline")
            self.assertEqual(len(pipeline.steps), 1)
        finally:
            Path(pipeline_file).unlink()
    
    def test_callback_registration(self):
        """Test registering callbacks"""
        executor = PipelineExecutor()
        
        callback_called = []
        
        def on_start_callback(pipeline, context):
            callback_called.append("start")
        
        def on_complete_callback(pipeline, context, result):
            callback_called.append("complete")
        
        executor.register_callback("on_start", on_start_callback)
        executor.register_callback("on_complete", on_complete_callback)
        
        # Verify callbacks are registered
        self.assertIn(on_start_callback, executor.callbacks["on_start"])
        self.assertIn(on_complete_callback, executor.callbacks["on_complete"])


class TestDockerBackend(unittest.TestCase):
    """Test Docker backend (if available)"""
    
    @patch('greenlang.runtime.backends.docker.docker')
    def test_docker_backend_creation(self, mock_docker):
        """Test creating Docker backend"""
        from greenlang.runtime.backends.docker import DockerBackend
        
        mock_docker.from_env.return_value = MagicMock()
        
        backend = DockerBackend({
            "network": "test-network",
            "volumes": {"/host": "/container"}
        })
        
        self.assertEqual(backend.network, "test-network")
        self.assertIn("/host", backend.volumes)


class TestKubernetesBackend(unittest.TestCase):
    """Test Kubernetes backend (if available)"""
    
    @patch('greenlang.runtime.backends.k8s.config')
    @patch('greenlang.runtime.backends.k8s.client')
    def test_k8s_backend_creation(self, mock_client, mock_config):
        """Test creating Kubernetes backend"""
        from greenlang.runtime.backends.k8s import KubernetesBackend
        
        mock_config.load_kube_config.return_value = None
        mock_client.BatchV1Api.return_value = MagicMock()
        mock_client.CoreV1Api.return_value = MagicMock()
        
        backend = KubernetesBackend({
            "namespace": "test-namespace",
            "image": "test-image"
        })
        
        self.assertEqual(backend.namespace, "test-namespace")
        self.assertEqual(backend.default_image, "test-image")


if __name__ == '__main__':
    unittest.main()