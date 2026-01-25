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

# Import agent registry for validation
try:
    from greenlang.agents.registry import get_agent_info
    AGENT_REGISTRY_AVAILABLE = True
except ImportError:
    AGENT_REGISTRY_AVAILABLE = False
    get_agent_info = None

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
    """
    Pipeline runner and manager with checkpoint support.

    This class provides a complete pipeline execution framework with built-in
    checkpointing for fault tolerance and resumability. Pipelines can be
    defined in YAML or constructed programmatically.

    Checkpoint Behavior:
        When checkpoint_enabled=True, the pipeline will attempt to initialize
        checkpoint storage. If checkpointing dependencies are not available:

        - With strict_checkpoint=False (default): Pipeline continues execution
          WITHOUT checkpointing. A warning is logged and the reason is stored
          in checkpoint_disabled_reason.

        - With strict_checkpoint=True: A RuntimeError is raised immediately,
          preventing execution without checkpoint capability.

        Always check checkpoint_actually_enabled after initialization to verify
        that checkpointing is operational.

    Attributes:
        name: Pipeline name identifier
        version: Semantic version string (default "1.0")
        description: Human-readable pipeline description
        inputs: Initial input data for pipeline execution
        steps: List of step definitions (each with name, agent, inputs, outputs)
        outputs: Expected output configuration
        checkpoint_enabled: Request checkpointing (may not succeed)
        checkpoint_strategy: Storage strategy ("file", "redis", "s3")
        checkpoint_config: Strategy-specific configuration
        auto_resume: Automatically resume from checkpoint if available
        checkpoint_after_each_step: Create checkpoint after each step completion
        strict_checkpoint: Raise error if checkpointing unavailable
        checkpoint_status_callback: Optional callback for checkpoint status changes

    Example:
        >>> pipeline = Pipeline(
        ...     name="carbon-calc",
        ...     checkpoint_enabled=True,
        ...     strict_checkpoint=True,  # Fail if checkpointing unavailable
        ...     steps=[{"name": "intake", "agent": "IntakeAgent"}]
        ... )
        >>> if not pipeline.checkpoint_actually_enabled:
        ...     print(f"Warning: {pipeline.checkpoint_disabled_reason}")
    """

    name: str
    version: str = "1.0"
    description: Optional[str] = None
    inputs: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    outputs: Dict[str, Any] = {}

    # Checkpoint configuration
    checkpoint_enabled: bool = Field(
        default=False,
        description="Request checkpointing - verify with checkpoint_actually_enabled"
    )
    checkpoint_strategy: str = Field(
        default="file",
        description="Checkpoint storage strategy: file, redis, s3"
    )
    checkpoint_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific checkpoint configuration"
    )
    auto_resume: bool = Field(
        default=True,
        description="Automatically resume from checkpoint if available"
    )
    checkpoint_after_each_step: bool = Field(
        default=True,
        description="Create checkpoint after each step completion"
    )
    strict_checkpoint: bool = Field(
        default=False,
        description="Raise RuntimeError if checkpoint dependencies unavailable"
    )

    # Runtime state (not persisted in YAML)
    _pipeline_id: Optional[str] = None
    _checkpoint_manager: Optional[Any] = None
    _execution_state: PipelineState = PipelineState.PENDING
    _completed_steps: List[str] = []
    _agent_outputs: Dict[str, Any] = {}
    _provenance_hashes: Dict[str, str] = {}
    _resume_context: Optional[Dict[str, Any]] = None
    _checkpoint_actually_enabled: bool = False
    _checkpoint_disabled_reason: Optional[str] = None
    _checkpoint_status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **data):
        """
        Initialize pipeline with checkpoint support.

        Args:
            **data: Pipeline configuration including name, steps, checkpoint settings

        Raises:
            RuntimeError: If strict_checkpoint=True and checkpointing unavailable
        """
        super().__init__(**data)
        self._pipeline_id = self._generate_pipeline_id()
        self._completed_steps = []
        self._agent_outputs = {}
        self._provenance_hashes = {}
        self._checkpoint_actually_enabled = False
        self._checkpoint_disabled_reason = None
        self._checkpoint_status_callback = None

        # Initialize checkpoint manager if enabled
        if self.checkpoint_enabled:
            self._initialize_checkpoint_manager()

    def set_checkpoint_status_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Set a callback for checkpoint status notifications.

        The callback will be invoked when:
        - Checkpointing is disabled (event="checkpoint_disabled")
        - A checkpoint is created (event="checkpoint_created")
        - Checkpoint creation fails (event="checkpoint_failed")
        - Pipeline resumes from checkpoint (event="checkpoint_resumed")

        Args:
            callback: Function accepting (event_name, event_data) parameters

        Example:
            >>> def on_checkpoint_status(event: str, data: dict):
            ...     if event == "checkpoint_disabled":
            ...         send_alert(f"Checkpoint disabled: {data['reason']}")
            >>> pipeline.set_checkpoint_status_callback(on_checkpoint_status)
        """
        self._checkpoint_status_callback = callback

    def _notify_checkpoint_status(self, event: str, data: Dict[str, Any]) -> None:
        """
        Notify callback of checkpoint status change.

        Args:
            event: Event type (checkpoint_disabled, checkpoint_created, etc.)
            data: Event-specific data dictionary
        """
        if self._checkpoint_status_callback:
            try:
                self._checkpoint_status_callback(event, data)
            except Exception as e:
                logger.warning(f"Checkpoint status callback failed: {e}")

    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline execution ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(self.name.encode()).hexdigest()[:8]
        return f"{self.name}_{name_hash}_{timestamp}"

    def _initialize_checkpoint_manager(self) -> None:
        """
        Initialize checkpoint manager with configured strategy.

        This method handles checkpoint initialization with proper error reporting.
        If checkpointing cannot be enabled, the reason is stored in _checkpoint_disabled_reason
        and _checkpoint_actually_enabled remains False.

        The method will:
        1. Check if checkpointing dependencies are available
        2. Attempt to initialize the checkpoint manager
        3. Log prominently if checkpointing is disabled
        4. Notify via callback if registered
        5. Raise RuntimeError if strict_checkpoint is True and initialization fails

        Raises:
            RuntimeError: If strict_checkpoint is True and checkpointing cannot be initialized.

        Note:
            This method is called automatically during __init__ if checkpoint_enabled=True.
            After initialization, always check checkpoint_actually_enabled to verify
            that checkpointing is operational.
        """
        if not CHECKPOINTING_AVAILABLE:
            reason = (
                "Checkpointing requested but dependencies not available. "
                "Install checkpoint dependencies: pip install greenlang[checkpointing]"
            )
            self._checkpoint_disabled_reason = reason
            self._checkpoint_actually_enabled = False

            # Log prominently - this is a significant configuration issue
            logger.warning("=" * 70)
            logger.warning("CHECKPOINT DISABLED: %s", reason)
            logger.warning("Pipeline will execute WITHOUT checkpointing capability.")
            logger.warning("To verify checkpoint status, check pipeline.checkpoint_actually_enabled")
            logger.warning("=" * 70)

            # Notify callback if registered
            self._notify_checkpoint_status("checkpoint_disabled", {
                "reason": reason,
                "pipeline_id": self._pipeline_id,
                "pipeline_name": self.name,
                "strategy_requested": self.checkpoint_strategy,
                "strict_mode": self.strict_checkpoint,
            })

            if self.strict_checkpoint:
                raise RuntimeError(
                    f"Strict checkpoint mode enabled but checkpointing unavailable: {reason}"
                )
            return

        try:
            self._checkpoint_manager = CheckpointManager(
                strategy=self.checkpoint_strategy,
                **self.checkpoint_config
            )
            self._checkpoint_actually_enabled = True
            self._checkpoint_disabled_reason = None
            logger.info(f"Initialized checkpoint manager with {self.checkpoint_strategy} strategy")

            # Notify callback of successful initialization
            self._notify_checkpoint_status("checkpoint_enabled", {
                "pipeline_id": self._pipeline_id,
                "pipeline_name": self.name,
                "strategy": self.checkpoint_strategy,
            })

        except Exception as e:
            reason = f"Failed to initialize checkpoint manager: {str(e)}"
            self._checkpoint_disabled_reason = reason
            self._checkpoint_actually_enabled = False

            # Log prominently - checkpoint initialization failed
            logger.warning("=" * 70)
            logger.warning("CHECKPOINT DISABLED: %s", reason)
            logger.warning("Pipeline will execute WITHOUT checkpointing capability.")
            logger.warning("To verify checkpoint status, check pipeline.checkpoint_actually_enabled")
            logger.warning("=" * 70)

            # Notify callback if registered
            self._notify_checkpoint_status("checkpoint_disabled", {
                "reason": reason,
                "pipeline_id": self._pipeline_id,
                "pipeline_name": self.name,
                "strategy_requested": self.checkpoint_strategy,
                "strict_mode": self.strict_checkpoint,
                "exception": str(e),
            })

            if self.strict_checkpoint:
                raise RuntimeError(
                    f"Strict checkpoint mode enabled but initialization failed: {reason}"
                )

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

    def to_policy_doc(self) -> Dict[str, Any]:
        """
        Return a policy-safe representation for OPA evaluation.

        This method creates a sanitized dictionary suitable for policy
        enforcement. It excludes sensitive runtime state and internal
        implementation details while providing the structural information
        needed for policy decisions.

        Returns:
            Dict[str, Any]: Policy-safe document containing:
                - name: Pipeline name
                - version: Pipeline version
                - steps: List of step metadata (name, agent, input/output keys only)
                - metadata: Pipeline metadata if available

        Note:
            This method deliberately excludes:
            - Runtime state (_agent_outputs, _provenance_hashes, etc.)
            - Checkpoint configuration and manager references
            - Actual input/output values (only keys are exposed)
            - Internal execution state

        Example:
            >>> pipeline = Pipeline(name="test", steps=[...])
            >>> policy_doc = pipeline.to_policy_doc()
            >>> # Use with OPA evaluation
            >>> decision = opa_eval("bundles/run.rego", {"pipeline": policy_doc})
        """
        return {
            "name": self.name,
            "version": getattr(self, "version", "1.0.0"),
            "steps": [
                {
                    "name": step.get("name", ""),
                    "agent": step.get("agent", ""),
                    "inputs": list(step.get("inputs", {}).keys()),
                    "outputs": list(step.get("outputs", {}).keys()),
                }
                for step in self.steps
            ],
            "metadata": getattr(self, "metadata", {}),
        }

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
            elif "agent" in step:
                # Validate agent exists in registry
                agent_name = step["agent"]
                if AGENT_REGISTRY_AVAILABLE and get_agent_info is not None:
                    agent_info = get_agent_info(agent_name)
                    if agent_info is None:
                        errors.append(f"Step {i}: Agent '{agent_name}' not found in registry")

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
        """
        Execute pipeline with resume from checkpoint.

        This method restores pipeline state from a checkpoint and continues
        execution from the last successfully completed step.

        Args:
            dry_run: If True, simulate execution without running agents

        Returns:
            Pipeline execution results dictionary
        """
        self._execution_state = PipelineState.RESUMED

        # Restore state from checkpoint
        context = self._resume_context
        self._completed_steps = context["completed_stages"]
        self._agent_outputs = context["agent_outputs"]
        self._provenance_hashes = context["provenance_hashes"]

        resume_index = context["resume_from_index"]
        resume_stage = context["resume_from_stage"]

        logger.info(f"Resuming from step {resume_index}: {resume_stage}")
        logger.info(f"Already completed steps: {self._completed_steps}")

        # Notify callback of resume
        self._notify_checkpoint_status("checkpoint_resumed", {
            "pipeline_id": self._pipeline_id,
            "pipeline_name": self.name,
            "resume_from_stage": resume_stage,
            "resume_from_index": resume_index,
            "completed_steps": len(self._completed_steps),
            "remaining_steps": len(self.steps) - resume_index,
        })

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
                sub_pipeline_ref = step["pipeline"]
                sub_pipeline_config = step.get("config", {})

                # Load and execute sub-pipeline
                sub_pipeline_output = self._execute_sub_pipeline(
                    sub_pipeline_ref=sub_pipeline_ref,
                    sub_pipeline_config=sub_pipeline_config,
                    step=step,
                    parent_inputs=self._prepare_step_inputs(step)
                )

                # Calculate provenance for sub-pipeline execution
                provenance_hash = self._calculate_provenance(
                    {"sub_pipeline": sub_pipeline_ref, "inputs": self._prepare_step_inputs(step)},
                    sub_pipeline_output
                )

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                return StepResult(
                    step_name=step_name,
                    agent_name=f"pipeline:{sub_pipeline_ref}",
                    status="completed",
                    output=sub_pipeline_output,
                    provenance_hash=provenance_hash,
                    execution_time_ms=execution_time
                )

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
        """
        Dynamically load and instantiate an agent from the registry.

        This method uses GreenLang's AgentRegistry to load agents by name.
        It supports:
        - Named agents from the registry (e.g., "FuelAgent", "CarbonAgent")
        - Alias resolution (e.g., "FuelAgentAI" -> "FuelAgent")
        - File path loading (e.g., "path/to/agent.py:ClassName")
        - Pack-based loading (e.g., "pack_name:AgentName")

        Args:
            agent_name: Name of the agent to load (registry name, alias, or path)
            config: Configuration dictionary to pass to the agent constructor

        Returns:
            Instantiated agent instance ready for processing

        Raises:
            ValueError: If agent cannot be found or instantiated
            ImportError: If agent module cannot be loaded
        """
        logger.info(f"Loading agent: {agent_name} with config: {config}")

        # Strategy 1: Try loading from AgentRegistry (recommended approach)
        try:
            from greenlang.agents.registry import AgentRegistry, create_agent

            # Check if agent exists in registry (get_agent_info already imported at module level)
            if AGENT_REGISTRY_AVAILABLE and get_agent_info is not None:
                agent_info = get_agent_info(agent_name)
                if agent_info:
                    logger.info(f"Found agent '{agent_name}' in registry (v{agent_info.version})")
                    return create_agent(agent_name, config=config if config else None)
        except ImportError as e:
            logger.debug(f"AgentRegistry not available: {e}")
        except ValueError as e:
            logger.debug(f"Agent '{agent_name}' not found in registry: {e}")

        # Strategy 2: Try loading from PackLoader (for pack:agent format)
        if ":" in agent_name and "/" not in agent_name and "\\" not in agent_name:
            try:
                from greenlang.packs.loader import PackLoader

                loader = PackLoader()
                agent_class = loader.get_agent(agent_name)
                if agent_class:
                    logger.info(f"Loaded agent '{agent_name}' from pack")
                    return agent_class(config) if config else agent_class()
            except ImportError as e:
                logger.debug(f"PackLoader not available: {e}")
            except ValueError as e:
                logger.debug(f"Agent '{agent_name}' not found in packs: {e}")

        # Strategy 3: Try dynamic import for module path format (module.path:ClassName)
        if ":" in agent_name or "." in agent_name:
            try:
                import importlib

                if ":" in agent_name:
                    module_path, class_name = agent_name.rsplit(":", 1)
                else:
                    # Assume last part is class name (e.g., greenlang.agents.fuel_agent.FuelAgent)
                    parts = agent_name.rsplit(".", 1)
                    if len(parts) == 2:
                        module_path, class_name = parts
                    else:
                        raise ValueError(f"Invalid agent path format: {agent_name}")

                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                logger.info(f"Loaded agent '{class_name}' from module '{module_path}'")
                return agent_class(config) if config else agent_class()

            except (ImportError, AttributeError) as e:
                logger.debug(f"Dynamic import failed for '{agent_name}': {e}")

        # Strategy 4: Try lazy import from greenlang.agents (for backward compatibility)
        try:
            from greenlang import agents as agents_module

            if hasattr(agents_module, agent_name):
                agent_class = getattr(agents_module, agent_name)
                logger.info(f"Loaded agent '{agent_name}' from greenlang.agents module")
                return agent_class(config) if config else agent_class()
        except (ImportError, AttributeError) as e:
            logger.debug(f"Lazy import from greenlang.agents failed: {e}")

        # If all strategies fail, raise a clear error
        raise ValueError(
            f"Agent '{agent_name}' could not be loaded. Tried:\n"
            f"  1. AgentRegistry lookup\n"
            f"  2. Pack loader (pack:agent format)\n"
            f"  3. Dynamic module import (module.path:ClassName)\n"
            f"  4. greenlang.agents lazy import\n"
            f"Please ensure the agent is registered or the path is correct."
        )

    def _execute_sub_pipeline(
        self,
        sub_pipeline_ref: str,
        sub_pipeline_config: Dict[str, Any],
        step: Dict[str, Any],
        parent_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a sub-pipeline as part of a parent pipeline step.

        This method handles loading and executing nested pipelines, supporting:
        - YAML file paths (e.g., "pipelines/validation.yaml")
        - Pack-based pipelines (e.g., "pack_name:pipeline_name")
        - Inline pipeline definitions (Dict with steps)

        Args:
            sub_pipeline_ref: Reference to the sub-pipeline (path, pack:name, or inline dict)
            sub_pipeline_config: Configuration overrides for the sub-pipeline
            step: The parent step definition containing the pipeline reference
            parent_inputs: Inputs prepared from parent pipeline context

        Returns:
            Dict containing sub-pipeline execution results including:
            - final_outputs: The outputs from the sub-pipeline
            - execution_state: Final state of sub-pipeline execution
            - steps_results: Results from each sub-pipeline step

        Raises:
            ValueError: If sub-pipeline cannot be found or loaded
            RuntimeError: If sub-pipeline execution fails
        """
        logger.info(f"Executing sub-pipeline: {sub_pipeline_ref}")

        sub_pipeline = None

        # Strategy 1: Load from YAML file path
        if isinstance(sub_pipeline_ref, str) and (
            sub_pipeline_ref.endswith(".yaml") or sub_pipeline_ref.endswith(".yml")
        ):
            try:
                # Check if it's an absolute path or relative to current working directory
                pipeline_path = Path(sub_pipeline_ref)
                if not pipeline_path.is_absolute():
                    # Try relative to current working directory
                    if not pipeline_path.exists():
                        # Try relative to parent pipeline location if available
                        pass  # Already checked relative path

                if pipeline_path.exists():
                    sub_pipeline = Pipeline.from_yaml(str(pipeline_path))
                    logger.info(f"Loaded sub-pipeline from file: {pipeline_path}")
                else:
                    raise ValueError(f"Sub-pipeline file not found: {sub_pipeline_ref}")
            except Exception as e:
                logger.debug(f"Failed to load sub-pipeline from file: {e}")

        # Strategy 2: Load from PackLoader (pack:pipeline format)
        if sub_pipeline is None and ":" in sub_pipeline_ref:
            try:
                from greenlang.packs.loader import PackLoader

                pack_name, pipeline_name = sub_pipeline_ref.split(":", 1)
                loader = PackLoader()

                # Load the pack
                loaded_pack = loader.load(pack_name)
                pipeline_data = loaded_pack.get_pipeline(pipeline_name)

                if pipeline_data:
                    sub_pipeline = Pipeline(**pipeline_data)
                    logger.info(f"Loaded sub-pipeline '{pipeline_name}' from pack '{pack_name}'")
                else:
                    raise ValueError(
                        f"Pipeline '{pipeline_name}' not found in pack '{pack_name}'"
                    )
            except ImportError as e:
                logger.debug(f"PackLoader not available: {e}")
            except ValueError as e:
                logger.debug(f"Sub-pipeline not found in pack: {e}")

        # Strategy 3: Inline pipeline definition (Dict)
        if sub_pipeline is None and isinstance(sub_pipeline_ref, dict):
            try:
                sub_pipeline = Pipeline(**sub_pipeline_ref)
                logger.info("Created sub-pipeline from inline definition")
            except Exception as e:
                raise ValueError(f"Invalid inline pipeline definition: {e}")

        # Strategy 4: Try loading from common pipeline directories
        if sub_pipeline is None and isinstance(sub_pipeline_ref, str):
            common_paths = [
                Path("pipelines") / f"{sub_pipeline_ref}.yaml",
                Path("pipelines") / f"{sub_pipeline_ref}.yml",
                Path(".") / f"{sub_pipeline_ref}.yaml",
                Path(".") / f"{sub_pipeline_ref}.yml",
            ]

            for path in common_paths:
                if path.exists():
                    try:
                        sub_pipeline = Pipeline.from_yaml(str(path))
                        logger.info(f"Loaded sub-pipeline from: {path}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load from {path}: {e}")

        # If still no pipeline found, raise error
        if sub_pipeline is None:
            raise ValueError(
                f"Sub-pipeline '{sub_pipeline_ref}' could not be loaded. Tried:\n"
                f"  1. YAML file path\n"
                f"  2. Pack loader (pack:pipeline format)\n"
                f"  3. Inline definition\n"
                f"  4. Common pipeline directories\n"
                f"Please ensure the pipeline exists and is accessible."
            )

        # Apply configuration overrides
        if sub_pipeline_config:
            # Override sub-pipeline settings with provided config
            for key, value in sub_pipeline_config.items():
                if hasattr(sub_pipeline, key):
                    setattr(sub_pipeline, key, value)

        # Pass inputs from parent to sub-pipeline
        # Merge parent inputs with any explicit inputs defined in step
        sub_pipeline.inputs.update(parent_inputs)

        # Handle input mapping if specified in step
        input_mapping = step.get("input_mapping", {})
        for sub_key, parent_ref in input_mapping.items():
            if isinstance(parent_ref, str) and parent_ref.startswith("$"):
                # Reference to parent step output
                ref_parts = parent_ref[1:].split(".")
                if ref_parts[0] in self._agent_outputs:
                    value = self._agent_outputs[ref_parts[0]]
                    for part in ref_parts[1:]:
                        value = value.get(part) if isinstance(value, dict) else None
                    sub_pipeline.inputs[sub_key] = value
            else:
                sub_pipeline.inputs[sub_key] = parent_ref

        # Inherit checkpoint settings from parent (unless explicitly overridden)
        if "checkpoint_enabled" not in sub_pipeline_config:
            sub_pipeline.checkpoint_enabled = self.checkpoint_enabled
        if "checkpoint_strategy" not in sub_pipeline_config:
            sub_pipeline.checkpoint_strategy = self.checkpoint_strategy

        # Execute the sub-pipeline
        try:
            logger.info(f"Starting sub-pipeline execution: {sub_pipeline.name}")
            results = sub_pipeline.execute(resume=False, dry_run=False)

            # Return sub-pipeline results
            return {
                "sub_pipeline_name": sub_pipeline.name,
                "sub_pipeline_id": results.get("pipeline_id"),
                "execution_state": results.get("execution_state"),
                "final_outputs": results.get("final_outputs", {}),
                "steps_results": results.get("steps_results", []),
                "started_at": results.get("started_at"),
                "completed_at": results.get("completed_at"),
            }

        except Exception as e:
            logger.error(f"Sub-pipeline '{sub_pipeline_ref}' execution failed: {e}")
            raise RuntimeError(
                f"Sub-pipeline '{sub_pipeline_ref}' failed: {str(e)}"
            ) from e

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
        """
        Create a checkpoint for current pipeline state.

        This method captures the current execution state including completed steps,
        agent outputs, and provenance hashes for later resumption.

        Args:
            stage_name: Name of the current stage/step
            stage_index: Index of the current stage in the pipeline
            error_message: Optional error message if checkpoint is for a failure

        Returns:
            Checkpoint ID if successful, None otherwise

        Note:
            If checkpoint_manager is not available (checkpointing disabled),
            this method returns None silently. Check checkpoint_actually_enabled
            to verify checkpointing is operational.
        """
        if not self._checkpoint_manager:
            # Log at debug level since this is expected when checkpointing is disabled
            logger.debug(
                f"Checkpoint skipped at stage {stage_name}: "
                f"checkpoint manager not initialized"
            )
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
                # Notify callback of checkpoint with error
                self._notify_checkpoint_status("checkpoint_created", {
                    "checkpoint_id": checkpoint_id,
                    "pipeline_id": self._pipeline_id,
                    "stage_name": stage_name,
                    "stage_index": stage_index,
                    "status": "failed",
                    "error_message": error_message,
                })
            else:
                self._checkpoint_manager.update_checkpoint_status(
                    checkpoint_id,
                    CheckpointStatus.COMPLETED
                )
                # Notify callback of successful checkpoint
                self._notify_checkpoint_status("checkpoint_created", {
                    "checkpoint_id": checkpoint_id,
                    "pipeline_id": self._pipeline_id,
                    "stage_name": stage_name,
                    "stage_index": stage_index,
                    "status": "completed",
                    "completed_steps": len(self._completed_steps),
                    "pending_steps": len(pending_stages),
                })

            logger.info(f"Created checkpoint {checkpoint_id} at stage {stage_name}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {str(e)}")
            # Notify callback of checkpoint failure
            self._notify_checkpoint_status("checkpoint_failed", {
                "pipeline_id": self._pipeline_id,
                "stage_name": stage_name,
                "stage_index": stage_index,
                "error": str(e),
            })
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

    @property
    def checkpoint_actually_enabled(self) -> bool:
        """
        Check if checkpointing is actually enabled and working.

        This property reflects the real state of checkpoint capability, not just
        the user's configuration request. Use this to verify that checkpoints
        will actually be created during pipeline execution.

        Returns:
            bool: True if checkpointing is both requested AND successfully initialized,
                  False if checkpointing was requested but failed to initialize.

        Example:
            >>> pipeline = Pipeline(name="test", checkpoint_enabled=True)
            >>> if pipeline.checkpoint_enabled and not pipeline.checkpoint_actually_enabled:
            ...     print(f"Warning: {pipeline.checkpoint_disabled_reason}")
        """
        return self._checkpoint_actually_enabled

    @property
    def checkpoint_disabled_reason(self) -> Optional[str]:
        """
        Get the reason why checkpointing is disabled (if applicable).

        Returns:
            Optional[str]: The reason checkpointing was disabled, or None if
                          checkpointing is enabled or was never requested.

        Example:
            >>> pipeline = Pipeline(name="test", checkpoint_enabled=True)
            >>> if not pipeline.checkpoint_actually_enabled:
            ...     print(f"Checkpoint disabled: {pipeline.checkpoint_disabled_reason}")
        """
        return self._checkpoint_disabled_reason

    def get_checkpoint_status(self) -> Dict[str, Any]:
        """
        Get comprehensive checkpoint status information.

        Returns a dictionary with complete checkpoint configuration and status
        information, useful for debugging and monitoring.

        Returns:
            Dict[str, Any]: Checkpoint status including:
                - requested: Whether checkpoint was requested in config
                - actually_enabled: Whether checkpoint is actually working
                - disabled_reason: Why checkpoint was disabled (if applicable)
                - strategy: The configured checkpoint strategy
                - strict_mode: Whether strict checkpoint mode is enabled
                - manager_initialized: Whether checkpoint manager was created

        Example:
            >>> pipeline = Pipeline(name="test", checkpoint_enabled=True)
            >>> status = pipeline.get_checkpoint_status()
            >>> print(json.dumps(status, indent=2))
        """
        return {
            "requested": self.checkpoint_enabled,
            "actually_enabled": self._checkpoint_actually_enabled,
            "disabled_reason": self._checkpoint_disabled_reason,
            "strategy": self.checkpoint_strategy,
            "strict_mode": self.strict_checkpoint,
            "manager_initialized": self._checkpoint_manager is not None,
            "auto_resume": self.auto_resume,
            "checkpoint_after_each_step": self.checkpoint_after_each_step,
        }

    async def execute_async(self, resume: bool = None, dry_run: bool = False) -> Dict[str, Any]:
        """Async version of pipeline execution."""
        # Run synchronous execution in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, resume, dry_run)
