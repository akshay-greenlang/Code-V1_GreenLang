from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
import json


class WorkflowStep(BaseModel):
    name: str = Field(..., description="Step name")
    agent_id: str = Field(..., description="ID of the agent to execute")
    description: Optional[str] = Field(None, description="Step description")
    input_mapping: Optional[Dict[str, str]] = Field(None, description="Maps step input from context")
    output_key: Optional[str] = Field(None, description="Key to store step output in context")
    condition: Optional[str] = Field(None, description="Condition to evaluate before executing step")
    on_failure: str = Field(default="stop", description="Action on failure: stop, skip, or continue")
    retry_count: int = Field(default=0, description="Number of retries on failure")


class Workflow(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="0.0.1", description="Workflow version")
    steps: List[WorkflowStep] = Field(..., description="List of workflow steps")
    output_mapping: Optional[Dict[str, str]] = Field(None, description="Maps workflow output from context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Workflow":
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_path: str) -> "Workflow":
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def add_step(self, step: WorkflowStep):
        self.steps.append(step)
    
    def remove_step(self, step_name: str):
        self.steps = [s for s in self.steps if s.name != step_name]
    
    def get_step(self, step_name: str) -> Optional[WorkflowStep]:
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def validate_workflow(self) -> List[str]:
        errors = []
        
        if not self.steps:
            errors.append("Workflow has no steps")
        
        step_names = set()
        for step in self.steps:
            if step.name in step_names:
                errors.append(f"Duplicate step name: {step.name}")
            step_names.add(step.name)
        
        return errors


class WorkflowBuilder:
    def __init__(self, name: str, description: str):
        self.workflow = Workflow(name=name, description=description, steps=[])
    
    def add_step(self, name: str, agent_id: str, **kwargs) -> "WorkflowBuilder":
        step = WorkflowStep(name=name, agent_id=agent_id, **kwargs)
        self.workflow.add_step(step)
        return self
    
    def with_output_mapping(self, mapping: Dict[str, str]) -> "WorkflowBuilder":
        self.workflow.output_mapping = mapping
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> "WorkflowBuilder":
        self.workflow.metadata = metadata
        return self
    
    def build(self) -> Workflow:
        errors = self.workflow.validate_workflow()
        if errors:
            raise ValueError(f"Workflow validation failed: {', '.join(errors)}")
        return self.workflow