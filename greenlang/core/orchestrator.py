from typing import Dict, Any, List, Optional
from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.core.workflow import Workflow
import logging
import json

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrates the execution of agent workflows"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict] = []
        self.logger = logger
    
    def register_agent(self, agent_id: str, agent: BaseAgent):
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent: {agent_id}")
    
    def register_workflow(self, workflow_id: str, workflow: Workflow):
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Registered workflow: {workflow_id}")
    
    def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = f"{workflow_id}_{len(self.execution_history)}"
        
        self.logger.info(f"Starting workflow execution: {execution_id}")
        
        context = {
            "input": input_data,
            "results": {},
            "errors": [],
            "workflow_id": workflow_id,
            "execution_id": execution_id
        }
        
        for step in workflow.steps:
            if not self._should_execute_step(step, context):
                self.logger.info(f"Skipping step: {step.name}")
                continue
            
            self.logger.info(f"Executing step: {step.name}")
            
            try:
                step_input = self._prepare_step_input(step, context)
                agent = self.agents.get(step.agent_id)
                
                if not agent:
                    raise ValueError(f"Agent '{step.agent_id}' not found")
                
                result = agent.run(step_input)
                
                context["results"][step.name] = result
                
                if not result.success:
                    context["errors"].append({
                        "step": step.name,
                        "error": result.error
                    })
                    
                    if step.on_failure == "stop":
                        self.logger.error(f"Step failed, stopping workflow: {step.name}")
                        break
                    elif step.on_failure == "skip":
                        self.logger.warning(f"Step failed, continuing: {step.name}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                context["errors"].append({
                    "step": step.name,
                    "error": str(e)
                })
                
                if step.on_failure == "stop":
                    break
        
        execution_record = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "input": input_data,
            "results": context["results"],
            "errors": context["errors"],
            "success": len(context["errors"]) == 0
        }
        
        self.execution_history.append(execution_record)
        
        return self._format_workflow_output(workflow, context)
    
    def _should_execute_step(self, step, context: Dict) -> bool:
        if not step.condition:
            return True
        
        try:
            return eval(step.condition, {"context": context})
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return True
    
    def _prepare_step_input(self, step, context: Dict) -> Dict[str, Any]:
        if step.input_mapping:
            mapped_input = {}
            for key, path in step.input_mapping.items():
                value = self._get_value_from_path(context, path)
                if value is not None:
                    mapped_input[key] = value
            return mapped_input
        else:
            return context["input"]
    
    def _get_value_from_path(self, data: Dict, path: str) -> Any:
        parts = path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _format_workflow_output(self, workflow: Workflow, context: Dict) -> Dict[str, Any]:
        output = {
            "workflow_id": context["workflow_id"],
            "execution_id": context["execution_id"],
            "success": len(context["errors"]) == 0,
            "errors": context["errors"]
        }
        
        if workflow.output_mapping:
            output["data"] = {}
            for key, path in workflow.output_mapping.items():
                value = self._get_value_from_path(context, path)
                if value is not None:
                    output["data"][key] = value
        else:
            output["results"] = context["results"]
        
        return output
    
    def execute_single_agent(self, agent_id: str, input_data: Dict[str, Any]) -> AgentResult:
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")
        
        return agent.run(input_data)
    
    def get_execution_history(self) -> List[Dict]:
        return self.execution_history
    
    def clear_history(self):
        self.execution_history = []
    
    def list_agents(self) -> List[str]:
        return list(self.agents.keys())
    
    def list_workflows(self) -> List[str]:
        return list(self.workflows.keys())
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        
        return {
            "id": agent_id,
            "name": agent.config.name,
            "description": agent.config.description,
            "version": agent.config.version,
            "enabled": agent.config.enabled
        }
    
    def get_workflow_info(self, workflow_id: str) -> Dict[str, Any]:
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "name": step.name,
                    "agent_id": step.agent_id,
                    "description": step.description
                }
                for step in workflow.steps
            ]
        }