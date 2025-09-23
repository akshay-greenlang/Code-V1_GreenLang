from typing import Dict, Any, Optional, Callable
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
from greenlang.core.workflow import Workflow, WorkflowStep


class AgentBuilder:
    """Builder for creating custom agents"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execute_func = None
        self.validate_func = None
        self.preprocess_func = None
        self.postprocess_func = None
        self.parameters = {}

    def with_execute(
        self, func: Callable[[Dict[str, Any]], AgentResult]
    ) -> "AgentBuilder":
        self.execute_func = func
        return self

    def with_validation(self, func: Callable[[Dict[str, Any]], bool]) -> "AgentBuilder":
        self.validate_func = func
        return self

    def with_preprocessing(
        self, func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "AgentBuilder":
        self.preprocess_func = func
        return self

    def with_postprocessing(
        self, func: Callable[[AgentResult], AgentResult]
    ) -> "AgentBuilder":
        self.postprocess_func = func
        return self

    def with_parameters(self, **params) -> "AgentBuilder":
        self.parameters.update(params)
        return self

    def build(self) -> BaseAgent:
        if not self.execute_func:
            raise ValueError("Execute function is required")

        class CustomAgent(BaseAgent):
            def __init__(agent_self):
                config = AgentConfig(
                    name=self.name,
                    description=self.description,
                    parameters=self.parameters,
                )
                super().__init__(config)

            def execute(agent_self, input_data: Dict[str, Any]) -> AgentResult:
                return self.execute_func(input_data)

            def validate_input(agent_self, input_data: Dict[str, Any]) -> bool:
                if self.validate_func:
                    return self.validate_func(input_data)
                return super().validate_input(input_data)

            def preprocess(agent_self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                if self.preprocess_func:
                    return self.preprocess_func(input_data)
                return super().preprocess(input_data)

            def postprocess(agent_self, result: AgentResult) -> AgentResult:
                if self.postprocess_func:
                    return self.postprocess_func(result)
                return super().postprocess(result)

        return CustomAgent()


class WorkflowBuilder:
    """Enhanced workflow builder with fluent interface"""

    def __init__(self, name: str, description: str):
        self.workflow = Workflow(name=name, description=description, steps=[])
        self.current_step = None

    def add_step(
        self, name: str, agent_id: str, description: Optional[str] = None
    ) -> "WorkflowBuilder":
        step = WorkflowStep(
            name=name,
            agent_id=agent_id,
            description=description or f"Execute {agent_id}",
        )
        self.workflow.steps.append(step)
        self.current_step = step
        return self

    def with_input_mapping(self, **mapping) -> "WorkflowBuilder":
        if self.current_step:
            self.current_step.input_mapping = mapping
        return self

    def with_condition(self, condition: str) -> "WorkflowBuilder":
        if self.current_step:
            self.current_step.condition = condition
        return self

    def on_failure(self, action: str) -> "WorkflowBuilder":
        if self.current_step and action in ["stop", "skip", "continue"]:
            self.current_step.on_failure = action
        return self

    def with_retry(self, count: int) -> "WorkflowBuilder":
        if self.current_step:
            self.current_step.retry_count = count
        return self

    def with_output_mapping(self, **mapping) -> "WorkflowBuilder":
        self.workflow.output_mapping = mapping
        return self

    def with_metadata(self, **metadata) -> "WorkflowBuilder":
        self.workflow.metadata.update(metadata)
        return self

    def parallel_steps(self, *steps: tuple) -> "WorkflowBuilder":
        for name, agent_id in steps:
            self.add_step(name, agent_id)
        return self

    def sequential_steps(self, *steps: tuple) -> "WorkflowBuilder":
        for i, (name, agent_id) in enumerate(steps):
            self.add_step(name, agent_id)
            if i > 0:
                prev_step = self.workflow.steps[-2].name
                self.with_condition(f"context['results']['{prev_step}']['success']")
        return self

    def build(self) -> Workflow:
        errors = self.workflow.validate_workflow()
        if errors:
            raise ValueError(f"Workflow validation failed: {', '.join(errors)}")
        return self.workflow

    def to_yaml(self, path: str):
        self.workflow.to_yaml(path)

    def to_json(self, path: str):
        self.workflow.to_json(path)
