from typing import Dict, Any, List, Optional
from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.core.workflow import Workflow
import logging
import json
import ast

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
                
                # Handle both dict and AgentResult returns
                if isinstance(result, dict):
                    # Convert dict to AgentResult-like structure
                    success = result.get("success", False)
                    context["results"][step.name] = result
                else:
                    # Assume it's an AgentResult or has success attribute
                    success = getattr(result, "success", False)
                    # Store the data from the AgentResult, not the object itself
                    if hasattr(result, 'data'):
                        context["results"][step.name] = {"success": success, "data": result.data}
                    else:
                        context["results"][step.name] = result
                
                if not success:
                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else getattr(result, "error", "Unknown error")
                    context["errors"].append({
                        "step": step.name,
                        "error": error_msg
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
            return self._evaluate_condition(step.condition, context)
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False

    def _evaluate_condition(self, expression: str, context: Dict) -> bool:
        """Safely evaluate a boolean expression against the given context."""
        allowed_names = {
            "context": context,
            "input": context.get("input", {}),
            "results": context.get("results", {}),
        }

        def eval_node(node):
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return all(eval_node(v) for v in node.values)
                if isinstance(node.op, ast.Or):
                    return any(eval_node(v) for v in node.values)
                raise ValueError("Unsupported boolean operator")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not eval_node(node.operand)
            if isinstance(node, ast.Compare):
                left = eval_node(node.left)
                for op, comp in zip(node.ops, node.comparators):
                    right = eval_node(comp)
                    if isinstance(op, ast.Eq):
                        ok = left == right
                    elif isinstance(op, ast.NotEq):
                        ok = left != right
                    elif isinstance(op, ast.Gt):
                        ok = left > right
                    elif isinstance(op, ast.GtE):
                        ok = left >= right
                    elif isinstance(op, ast.Lt):
                        ok = left < right
                    elif isinstance(op, ast.LtE):
                        ok = left <= right
                    elif isinstance(op, ast.In):
                        ok = left in right
                    elif isinstance(op, ast.NotIn):
                        ok = left not in right
                    else:
                        raise ValueError("Unsupported comparison operator")
                    if not ok:
                        return False
                    left = right
                return True
            if isinstance(node, ast.Name):
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValueError(f"Name '{node.id}' is not allowed")
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Subscript):
                value = eval_node(node.value)
                index = eval_node(node.slice)
                return value[index]
            if isinstance(node, ast.Attribute):
                value = eval_node(node.value)
                if isinstance(value, dict):
                    return value.get(node.attr)
                return getattr(value, node.attr)
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        tree = ast.parse(expression, mode="eval")
        return bool(eval_node(tree.body))
    
    def _prepare_step_input(self, step, context: Dict) -> Dict[str, Any]:
        if step.input_mapping:
            mapped_input = {}
            for key, path in step.input_mapping.items():
                value = self._get_value_from_path(context, path)
                if value is not None:
                    mapped_input[key] = value
            return mapped_input
        else:
            # Pass the entire context input for now
            # Agents should be able to extract what they need
            return context.get("input", {})
    
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
    
    def execute_single_agent(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")
        
        result = agent.run(input_data)
        
        # Handle both dict and AgentResult types
        if isinstance(result, dict):
            return result
        elif hasattr(result, 'model_dump'):
            # Pydantic model
            return result.model_dump()
        elif hasattr(result, '__dict__'):
            # Object with attributes
            return {
                "success": getattr(result, 'success', False),
                "data": getattr(result, 'data', {}),
                "error": getattr(result, 'error', None),
                "metadata": getattr(result, 'metadata', {})
            }
        else:
            return result
    
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