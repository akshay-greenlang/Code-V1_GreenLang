# -*- coding: utf-8 -*-
"""
Pipeline execution and management with checkpoint support
"""

import yaml
import json
import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator

# Import checkpointing components
try:
    from greenlang.pipeline.checkpointing import (
        CheckpointManager,
        CheckpointStrategy,
        CheckpointStatus,
        PipelineCheckpoint
    )
    CHECKPOINTING_AVAILABLE = True
except ImportError:
    CHECKPOINTING_AVAILABLE = False
    CheckpointManager = None
    CheckpointStrategy = None

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """Pipeline execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMED = "resumed"
    CHECKPOINTED = "checkpointed"


@dataclass
class StepResult:
    """Result from a pipeline step execution."""
    step_name: str
    agent_name: str
    status: str
    output: Any
    provenance_hash: Optional[str] = None
    execution_time_ms: float = 0
    error_message: Optional[str] = None


class Pipeline(BaseModel):
    """Pipeline runner and manager with checkpoint support"""

    name: str
    version: str = "1.0"
    description: Optional[str] = None
    inputs: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    outputs: Dict[str, Any] = {}

    # Checkpoint configuration
    checkpoint_enabled: bool = Field(default=False, description="Enable checkpointing")
    checkpoint_strategy: str = Field(default="file", description="Checkpoint storage strategy")
    checkpoint_config: Dict[str, Any] = Field(default_factory=dict, description="Checkpoint storage config")
    auto_resume: bool = Field(default=True, description="Automatically resume from checkpoint")
    checkpoint_after_each_step: bool = Field(default=True, description="Create checkpoint after each step")

    # Runtime state (not persisted in YAML)
    _pipeline_id: Optional[str] = None
    _checkpoint_manager: Optional[Any] = None
    _execution_state: PipelineState = PipelineState.PENDING
    _completed_steps: List[str] = []
    _agent_outputs: Dict[str, Any] = {}
    _provenance_hashes: Dict[str, str] = {}
    _resume_context: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **data):
        """Initialize pipeline with checkpoint support."""
        super().__init__(**data)
        self._pipeline_id = self._generate_pipeline_id()
        self._completed_steps = []
        self._agent_outputs = {}
        self._provenance_hashes = {}

        # Initialize checkpoint manager if enabled
        if self.checkpoint_enabled and CHECKPOINTING_AVAILABLE:
            self._initialize_checkpoint_manager()

    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline execution ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(self.name.encode()).hexdigest()[:8]
        return f"{self.name}_{name_hash}_{timestamp}"

    def _initialize_checkpoint_manager(self):
        """Initialize checkpoint manager with configured strategy."""
        if not CHECKPOINTING_AVAILABLE:
            logger.warning("Checkpointing requested but not available. Install checkpoint dependencies.")
            return

        try:
            self._checkpoint_manager = CheckpointManager(
                strategy=self.checkpoint_strategy,
                **self.checkpoint_config
            )
            logger.info(f"Initialized checkpoint manager with {self.checkpoint_strategy} strategy")
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint manager: {str(e)}")
            self.checkpoint_enabled = False

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        """Load pipeline from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save pipeline to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()

    def load_inputs_file(self, path: str) -> None:
        """Load inputs from file"""
        if Path(path).suffix == ".json":
            import json

            with open(path) as f:
                self.inputs.update(json.load(f))
        elif Path(path).suffix in [".yaml", ".yml"]:
            with open(path) as f:
                self.inputs.update(yaml.safe_load(f))

    def validate(self) -> List[str]:
        """Validate pipeline structure"""
        errors = []

        if not self.name:
            errors.append("Pipeline name is required")

        if not self.steps:
            errors.append("Pipeline must have at least one step")

        for i, step in enumerate(self.steps):
            if "name" not in step:
                errors.append(f"Step {i} missing name")
            if "agent" not in step and "pipeline" not in step:
                errors.append(f"Step {i} must specify agent or pipeline")

        return errors

    def execute(self, resume: bool = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute pipeline with checkpoint support.

        Args:
            resume: Resume from checkpoint if available (defaults to auto_resume)
            dry_run: Simulate execution without running agents

        Returns:
            Pipeline execution results
        """
        # Use configured auto_resume if resume not explicitly set
        if resume is None:
            resume = self.auto_resume

        # Validate pipeline
        errors = self.validate()
        if errors:
            raise ValueError(f"Pipeline validation failed: {errors}")

        # Check for resumable checkpoint
        if resume and self.checkpoint_enabled and self._checkpoint_manager:
            self._resume_context = self._checkpoint_manager.resume_pipeline(self._pipeline_id)
            if self._resume_context:
                logger.info(f"Resuming pipeline {self._pipeline_id} from checkpoint")
                return self._execute_with_resume(dry_run)

        # Execute from beginning
        logger.info(f"Starting pipeline {self._pipeline_id} execution")
        return self._execute_pipeline(dry_run)

    def _execute_pipeline(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute pipeline from beginning."""
        self._execution_state = PipelineState.RUNNING
        results = {
            "pipeline_id": self._pipeline_id,
            "name": self.name,
            "version": self.version,
            "started_at": datetime.now().isoformat(),
            "steps_results": [],
            "final_outputs": {},
            "execution_state": self._execution_state.value,
            "checkpoints_created": []
        }

        try:
            for i, step in enumerate(self.steps):
                step_name = step["name"]

                # Check if step already completed (in resume scenarios)
                if step_name in self._completed_steps:
                    logger.info(f"Skipping already completed step: {step_name}")
                    continue

                logger.info(f"Executing step {i+1}/{len(self.steps)}: {step_name}")

                # Execute step
                if dry_run:
                    step_result = self._simulate_step(step, i)
                else:
                    step_result = self._execute_step(step, i)

                results["steps_results"].append(step_result.__dict__)

                # Store outputs
                self._agent_outputs[step_name] = step_result.output
                if step_result.provenance_hash:
                    self._provenance_hashes[step_name] = step_result.provenance_hash

                # Mark step as completed
                self._completed_steps.append(step_name)

                # Create checkpoint if enabled
                if self.checkpoint_enabled and self.checkpoint_after_each_step:
                    checkpoint_id = self._create_checkpoint(step_name, i)
                    if checkpoint_id:
                        results["checkpoints_created"].append(checkpoint_id)

                # Check for step failure
                if step_result.status == "failed":
                    raise RuntimeError(f"Step {step_name} failed: {step_result.error_message}")

            # Pipeline completed successfully
            self._execution_state = PipelineState.COMPLETED
            results["completed_at"] = datetime.now().isoformat()
            results["final_outputs"] = self._agent_outputs
            results["execution_state"] = self._execution_state.value

            # Final checkpoint
            if self.checkpoint_enabled:
                final_checkpoint_id = self._create_checkpoint("pipeline_complete", len(self.steps))
                if final_checkpoint_id:
                    results["checkpoints_created"].append(final_checkpoint_id)

            logger.info(f"Pipeline {self._pipeline_id} completed successfully")

        except Exception as e:
            self._execution_state = PipelineState.FAILED
            results["execution_state"] = self._execution_state.value
            results["error"] = str(e)
            results["failed_at"] = datetime.now().isoformat()

            # Create failure checkpoint
            if self.checkpoint_enabled:
                self._create_checkpoint(f"failed_at_step_{len(self._completed_steps)}",
                                       len(self._completed_steps),
                                       error_message=str(e))

            logger.error(f"Pipeline {self._pipeline_id} failed: {str(e)}")
            raise

        return results

    def _execute_with_resume(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute pipeline with resume from checkpoint."""
        self._execution_state = PipelineState.RESUMED

        # Restore state from checkpoint
        context = self._resume_context
        self._completed_steps = context["completed_stages"]
        self._agent_outputs = context["agent_outputs"]
        self._provenance_hashes = context["provenance_hashes"]

        resume_index = context["resume_from_index"]

        logger.info(f"Resuming from step {resume_index}: {context['resume_from_stage']}")
        logger.info(f"Already completed steps: {self._completed_steps}")

        # Continue execution
        return self._execute_pipeline(dry_run)

    def _execute_step(self, step: Dict[str, Any], index: int) -> StepResult:
        """Execute a single pipeline step."""
        start_time = datetime.now()
        step_name = step["name"]

        try:
            # Get agent or sub-pipeline
            if "agent" in step:
                agent_name = step["agent"]
                agent_config = step.get("config", {})

                # Import and instantiate agent dynamically
                agent = self._load_agent(agent_name, agent_config)

                # Prepare inputs
                step_inputs = self._prepare_step_inputs(step)

                # Execute agent
                agent_output = agent.process(step_inputs)

                # Calculate provenance
                provenance_hash = self._calculate_provenance(step_inputs, agent_output)

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                return StepResult(
                    step_name=step_name,
                    agent_name=agent_name,
                    status="completed",
                    output=agent_output,
                    provenance_hash=provenance_hash,
                    execution_time_ms=execution_time
                )

            elif "pipeline" in step:
                # Execute sub-pipeline
                sub_pipeline_name = step["pipeline"]
                # Implementation for sub-pipeline execution
                raise NotImplementedError("Sub-pipeline execution not yet implemented")

            else:
                raise ValueError(f"Step {step_name} must specify agent or pipeline")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return StepResult(
                step_name=step_name,
                agent_name=step.get("agent", "unknown"),
                status="failed",
                output=None,
                error_message=str(e),
                execution_time_ms=execution_time
            )

    def _simulate_step(self, step: Dict[str, Any], index: int) -> StepResult:
        """Simulate step execution for dry run."""
        return StepResult(
            step_name=step["name"],
            agent_name=step.get("agent", "simulated"),
            status="simulated",
            output={"simulated": True, "step_index": index},
            provenance_hash="simulated_hash_" + hashlib.md5(step["name"].encode()).hexdigest(),
            execution_time_ms=0
        )

    def _load_agent(self, agent_name: str, config: Dict[str, Any]) -> Any:
        """Dynamically load and instantiate an agent."""
        # This is a placeholder - implement actual agent loading logic
        # based on your agent registry or import system
        logger.info(f"Loading agent: {agent_name} with config: {config}")

        # Example implementation:
        # from greenlang.agents import get_agent
        # return get_agent(agent_name, config)

        # For now, return a mock agent
        class MockAgent:
            def process(self, inputs):
                return {"processed": True, "agent": agent_name, "inputs": inputs}

        return MockAgent()

    def _prepare_step_inputs(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for a step."""
        step_inputs = {}

        # Start with pipeline inputs
        step_inputs.update(self.inputs)

        # Add outputs from previous steps if referenced
        if "inputs" in step:
            for input_key, input_ref in step["inputs"].items():
                if isinstance(input_ref, str) and input_ref.startswith("$"):
                    # Reference to previous step output
                    ref_parts = input_ref[1:].split(".")
                    if ref_parts[0] in self._agent_outputs:
                        value = self._agent_outputs[ref_parts[0]]
                        for part in ref_parts[1:]:
                            value = value.get(part) if isinstance(value, dict) else None
                        step_inputs[input_key] = value
                else:
                    step_inputs[input_key] = input_ref

        return step_inputs

    def _calculate_provenance(self, inputs: Any, outputs: Any) -> str:
        """Calculate provenance hash for audit trail."""
        provenance_data = {
            "pipeline_id": self._pipeline_id,
            "inputs": str(inputs),
            "outputs": str(outputs),
            "timestamp": datetime.now().isoformat()
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _create_checkpoint(self, stage_name: str, stage_index: int,
                          error_message: Optional[str] = None) -> Optional[str]:
        """Create a checkpoint for current pipeline state."""
        if not self._checkpoint_manager:
            return None

        try:
            # Determine pending stages
            pending_stages = [step["name"] for step in self.steps[stage_index + 1:]]

            # Create checkpoint
            checkpoint_id = self._checkpoint_manager.create_checkpoint(
                pipeline_id=self._pipeline_id,
                stage_name=stage_name,
                stage_index=stage_index,
                state_data={
                    "inputs": self.inputs,
                    "pipeline_config": self.to_dict()
                },
                completed_stages=self._completed_steps.copy(),
                pending_stages=pending_stages,
                agent_outputs=self._agent_outputs.copy(),
                provenance_hashes=self._provenance_hashes.copy()
            )

            # Update status based on error
            if error_message:
                self._checkpoint_manager.update_checkpoint_status(
                    checkpoint_id,
                    CheckpointStatus.FAILED,
                    error_message
                )
            else:
                self._checkpoint_manager.update_checkpoint_status(
                    checkpoint_id,
                    CheckpointStatus.COMPLETED
                )

            logger.info(f"Created checkpoint {checkpoint_id} at stage {stage_name}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {str(e)}")
            return None

    def pause(self) -> bool:
        """Pause pipeline execution and create checkpoint."""
        if self._execution_state != PipelineState.RUNNING:
            return False

        self._execution_state = PipelineState.PAUSED

        # Create pause checkpoint
        if self.checkpoint_enabled:
            checkpoint_id = self._create_checkpoint(
                f"paused_at_step_{len(self._completed_steps)}",
                len(self._completed_steps)
            )
            logger.info(f"Pipeline paused with checkpoint {checkpoint_id}")

        return True

    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get checkpoint history for this pipeline."""
        if not self._checkpoint_manager:
            return []

        history = self._checkpoint_manager.get_checkpoint_history(self._pipeline_id)
        return [meta.to_dict() for meta in history]

    def visualize_progress(self) -> str:
        """Visualize pipeline execution progress."""
        lines = [f"Pipeline: {self.name} (ID: {self._pipeline_id})"]
        lines.append(f"Status: {self._execution_state.value}")
        lines.append("=" * 60)

        for i, step in enumerate(self.steps):
            step_name = step["name"]
            if step_name in self._completed_steps:
                status_symbol = "✓"
                status = "COMPLETED"
            elif i == len(self._completed_steps):
                status_symbol = "⟳"
                status = "IN PROGRESS"
            else:
                status_symbol = "○"
                status = "PENDING"

            lines.append(f"{status_symbol} Step {i+1}: {step_name} [{status}]")

            if step_name in self._agent_outputs:
                output_preview = str(self._agent_outputs[step_name])[:50]
                lines.append(f"  └─ Output: {output_preview}...")

        if self.checkpoint_enabled and self._checkpoint_manager:
            lines.append("\nCheckpoint Chain:")
            lines.append(self._checkpoint_manager.visualize_checkpoint_chain(self._pipeline_id))

        return "\n".join(lines)

    def cleanup_checkpoints(self) -> int:
        """Clean up checkpoints for this pipeline."""
        if not self._checkpoint_manager:
            return 0

        return self._checkpoint_manager.cleanup_completed_pipelines([self._pipeline_id])

    async def execute_async(self, resume: bool = None, dry_run: bool = False) -> Dict[str, Any]:
        """Async version of pipeline execution."""
        # Run synchronous execution in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, resume, dry_run)
