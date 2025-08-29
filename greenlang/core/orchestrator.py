from typing import Dict, Any, List, Optional
from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.core.workflow import Workflow
import logging
import json
import ast
import operator

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
            
            # Implement retry logic
            max_retries = step.retry_count if step.retry_count > 0 else 0
            attempt = 0
            step_succeeded = False
            last_error = None
            
            while attempt <= max_retries:
                try:
                    if attempt > 0:
                        self.logger.info(f"Retrying step {step.name} (attempt {attempt}/{max_retries})")
                    
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
                    
                    if success:
                        step_succeeded = True
                        break  # Success, exit retry loop
                    else:
                        # Step failed but returned normally
                        last_error = result.get("error", "Unknown error") if isinstance(result, dict) else getattr(result, "error", "Unknown error")
                        
                        if attempt < max_retries:
                            self.logger.warning(f"Step {step.name} failed, will retry. Error: {last_error}")
                            attempt += 1
                            continue
                        else:
                            # No more retries
                            break
                    
                except Exception as e:
                    last_error = str(e)
                    self.logger.error(f"Error in step {step.name}: {last_error}")
                    
                    if attempt < max_retries:
                        self.logger.warning(f"Will retry step {step.name}")
                        attempt += 1
                        continue
                    else:
                        # No more retries, handle as final failure
                        break
            
            # Handle final failure after all retries
            if not step_succeeded:
                if last_error:
                    context["errors"].append({
                        "step": step.name,
                        "error": last_error,
                        "attempts": attempt + 1
                    })
                
                if step.on_failure == "stop":
                    self.logger.error(f"Step failed after {attempt + 1} attempts, stopping workflow: {step.name}")
                    break
                elif step.on_failure == "skip":
                    self.logger.warning(f"Step failed after {attempt + 1} attempts, continuing: {step.name}")
                    continue
        
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
            # Safe expression evaluation using AST
            return self._safe_eval_condition(step.condition, context)
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return True
    
    def _safe_eval_condition(self, condition: str, context: Dict) -> bool:
        """
        Safely evaluate a condition string without using eval().
        Supports basic comparisons and logical operations.
        """
        # Define allowed operators
        ops = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.And: operator.and_,
            ast.Or: operator.or_,
            ast.Not: operator.not_,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
        }
        
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                # Only allow access to 'context' variable
                if node.id == 'context':
                    return context
                raise ValueError(f"Unauthorized variable access: {node.id}")
            elif isinstance(node, ast.Subscript):
                # Handle dictionary/list access like context['results']
                value = _eval(node.value)
                if isinstance(node.slice, ast.Constant):
                    return value[node.slice.value]
                elif isinstance(node.slice, ast.Name):
                    return value[_eval(node.slice)]
                else:
                    raise ValueError("Complex subscript not supported")
            elif isinstance(node, ast.Attribute):
                # Handle attribute access like context.results
                value = _eval(node.value)
                return getattr(value, node.attr, None)
            elif isinstance(node, ast.Compare):
                # Handle comparison operations
                left = _eval(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    if type(op) not in ops:
                        raise ValueError(f"Unsupported operator: {type(op).__name__}")
                    right = _eval(comparator)
                    if not ops[type(op)](left, right):
                        return False
                    left = right
                return True
            elif isinstance(node, ast.BoolOp):
                # Handle and/or operations
                if isinstance(node.op, ast.And):
                    return all(_eval(value) for value in node.values)
                elif isinstance(node.op, ast.Or):
                    return any(_eval(value) for value in node.values)
            elif isinstance(node, ast.UnaryOp):
                # Handle not operation
                if type(node.op) in ops:
                    return ops[type(node.op)](_eval(node.operand))
            elif isinstance(node, ast.Call):
                # Block function calls for security
                raise ValueError("Function calls are not allowed in conditions")
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")
        
        try:
            tree = ast.parse(condition, mode='eval')
            return _eval(tree.body)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid condition expression: {e}")
    
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