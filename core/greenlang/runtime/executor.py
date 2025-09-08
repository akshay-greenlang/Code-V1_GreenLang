"""
Runtime Executor
================

Executes pipelines with different runtime profiles (local, k8s, cloud).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from greenlang.sdk.base import Result, Status
from greenlang.packs.loader import PackLoader


logger = logging.getLogger(__name__)


class Executor:
    """
    Pipeline executor with runtime profiles
    
    Profiles:
    - local: Run on local machine
    - k8s: Run on Kubernetes
    - cloud: Run on cloud functions
    """
    
    def __init__(self, profile: str = "local"):
        """
        Initialize executor
        
        Args:
            profile: Runtime profile (local, k8s, cloud)
        """
        self.profile = profile
        self.loader = PackLoader()
        self.run_ledger = []
        self.artifacts_dir = Path.home() / ".greenlang" / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, pipeline_ref: str, input_data: Dict[str, Any]) -> Result:
        """
        Execute a pipeline
        
        Args:
            pipeline_ref: Pipeline reference (pack.pipeline)
            input_data: Input data for pipeline
        
        Returns:
            Execution result
        """
        run_id = str(uuid4())
        run_start = datetime.now()
        
        logger.info(f"Starting run {run_id} for pipeline {pipeline_ref}")
        
        try:
            # Load pipeline
            pipeline = self.loader.get_pipeline(pipeline_ref)
            if not pipeline:
                return Result(
                    success=False,
                    error=f"Pipeline not found: {pipeline_ref}"
                )
            
            # Create run context
            context = {
                "run_id": run_id,
                "pipeline": pipeline_ref,
                "profile": self.profile,
                "input": input_data,
                "artifacts": [],
                "results": {}
            }
            
            # Execute based on profile
            if self.profile == "local":
                result = self._run_local(pipeline, context)
            elif self.profile == "k8s":
                result = self._run_k8s(pipeline, context)
            elif self.profile == "cloud":
                result = self._run_cloud(pipeline, context)
            else:
                return Result(
                    success=False,
                    error=f"Unknown profile: {self.profile}"
                )
            
            # Record run
            run_end = datetime.now()
            run_record = {
                "run_id": run_id,
                "pipeline": pipeline_ref,
                "profile": self.profile,
                "status": "success" if result.success else "failed",
                "start_time": run_start.isoformat(),
                "end_time": run_end.isoformat(),
                "duration_seconds": (run_end - run_start).total_seconds(),
                "artifacts": context.get("artifacts", [])
            }
            
            self._save_run_record(run_record)
            
            # Generate run.json
            self._generate_run_json(run_id, pipeline, context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return Result(
                success=False,
                error=str(e)
            )
    
    def _run_local(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Result:
        """Execute pipeline locally"""
        steps = pipeline.get("steps", [])
        
        for step in steps:
            agent_ref = step.get("agent")
            if not agent_ref:
                continue
            
            # Load agent
            agent_class = self.loader.get_agent(agent_ref)
            if not agent_class:
                return Result(
                    success=False,
                    error=f"Agent not found: {agent_ref}"
                )
            
            # Instantiate agent
            agent = agent_class()
            
            # Prepare input
            step_input = self._prepare_step_input(step, context)
            
            # Run agent
            result = agent.run(step_input)
            
            # Store result
            step_name = step.get("name", agent_ref)
            context["results"][step_name] = result.data if result.success else result.error
            
            # Handle failure
            if not result.success:
                on_failure = step.get("on_failure", "stop")
                if on_failure == "stop":
                    return Result(
                        success=False,
                        error=f"Step {step_name} failed: {result.error}"
                    )
                elif on_failure == "skip":
                    continue
                # continue on failure
        
        # Collect final output
        output = self._collect_output(pipeline, context)
        
        return Result(
            success=True,
            data=output,
            metadata={
                "run_id": context["run_id"],
                "pipeline": context["pipeline"]
            }
        )
    
    def _run_k8s(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Result:
        """Execute pipeline on Kubernetes"""
        # TODO: Implement K8s execution
        # This would create Jobs/Pods for each step
        return Result(
            success=False,
            error="Kubernetes execution not yet implemented"
        )
    
    def _run_cloud(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Result:
        """Execute pipeline on cloud functions"""
        # TODO: Implement cloud execution
        # This would invoke Lambda/Cloud Functions for each step
        return Result(
            success=False,
            error="Cloud execution not yet implemented"
        )
    
    def _prepare_step_input(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a step"""
        input_mapping = step.get("input", {})
        
        if isinstance(input_mapping, dict):
            # Map from context
            step_input = {}
            for key, path in input_mapping.items():
                # Simple path resolution (could be more sophisticated)
                if path.startswith("$input."):
                    field = path.replace("$input.", "")
                    step_input[key] = context["input"].get(field)
                elif path.startswith("$results."):
                    field = path.replace("$results.", "")
                    step_input[key] = context["results"].get(field)
                else:
                    step_input[key] = path
            return step_input
        else:
            # Use context input directly
            return context["input"]
    
    def _collect_output(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect pipeline output"""
        output_mapping = pipeline.get("output", {})
        
        if output_mapping:
            output = {}
            for key, path in output_mapping.items():
                if path.startswith("$results."):
                    field = path.replace("$results.", "")
                    output[key] = context["results"].get(field)
                else:
                    output[key] = path
            return output
        else:
            # Return all results
            return context["results"]
    
    def _save_run_record(self, record: Dict[str, Any]):
        """Save run record to ledger"""
        self.run_ledger.append(record)
        
        # Persist to file
        ledger_file = self.artifacts_dir / "run_ledger.jsonl"
        with open(ledger_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def _generate_run_json(self, run_id: str, pipeline: Dict[str, Any], 
                          context: Dict[str, Any], result: Result):
        """Generate deterministic run.json for reproducibility"""
        run_json = {
            "run_id": run_id,
            "pipeline": pipeline,
            "input": context["input"],
            "output": result.data if result.success else None,
            "status": "success" if result.success else "failed",
            "error": result.error if not result.success else None,
            "artifacts": context.get("artifacts", []),
            "profile": self.profile,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save run.json
        run_file = self.artifacts_dir / f"run_{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run_json, f, indent=2)
        
        logger.info(f"Generated run.json: {run_file}")
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs from ledger"""
        ledger_file = self.artifacts_dir / "run_ledger.jsonl"
        
        if not ledger_file.exists():
            return []
        
        runs = []
        with open(ledger_file) as f:
            for line in f:
                runs.append(json.loads(line))
        
        return runs
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific run"""
        run_file = self.artifacts_dir / f"run_{run_id}.json"
        
        if run_file.exists():
            with open(run_file) as f:
                return json.load(f)
        
        return None