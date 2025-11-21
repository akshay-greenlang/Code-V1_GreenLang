# -*- coding: utf-8 -*-
"""
Planning Framework - Comprehensive planning algorithms for GreenLang agents.

This module implements various planning algorithms including:
- Hierarchical planning with top-down task decomposition
- Reactive planning for immediate situation response
- Deliberative planning for long-term strategic goals
- Hybrid planning combining multiple approaches

Example:
    >>> planner = HierarchicalPlanner(config)
    >>> plan = await planner.create_plan(goals, constraints)
    >>> result = await planner.execute_plan(plan)
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
import networkx as nx
import numpy as np
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class PlanType(str, Enum):
    """Types of planning algorithms."""

    HIERARCHICAL = "hierarchical"
    REACTIVE = "reactive"
    DELIBERATIVE = "deliberative"
    HYBRID = "hybrid"


class PlanStatus(str, Enum):
    """Status of a plan or plan step."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ActionType(str, Enum):
    """Types of actions in a plan."""

    ATOMIC = "atomic"          # Single indivisible action
    COMPOSITE = "composite"    # Multiple sub-actions
    CONDITIONAL = "conditional" # Branch based on condition
    PARALLEL = "parallel"      # Concurrent execution
    SEQUENTIAL = "sequential"  # Step-by-step execution
    ITERATIVE = "iterative"   # Repeated execution


@dataclass
class PlanGoal:
    """Representation of a planning goal."""

    goal_id: str
    description: str
    priority: int = 1
    deadline: Optional[datetime] = None
    preconditions: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanAction:
    """Representation of an action in a plan."""

    action_id: str
    name: str
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    duration_estimate_seconds: float = 1.0
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    sub_actions: List['PlanAction'] = field(default_factory=list)


@dataclass
class PlanStep:
    """A step in the execution plan."""

    step_id: str
    action: PlanAction
    status: PlanStatus = PlanStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class Plan(BaseModel):
    """Complete execution plan."""

    plan_id: str = Field(..., description="Unique plan identifier")
    plan_type: PlanType = Field(..., description="Type of planning algorithm used")
    goals: List[PlanGoal] = Field(default_factory=list, description="Goals to achieve")
    steps: List[PlanStep] = Field(default_factory=list, description="Plan steps")
    status: PlanStatus = Field(PlanStatus.PENDING, description="Overall plan status")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    execution_graph: Optional[Dict[str, Any]] = Field(None, description="DAG of execution")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Plan metrics")


class PlanningAlgorithm(ABC):
    """Abstract base class for planning algorithms."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize planning algorithm."""
        self.config = config or {}
        self.planning_horizon = config.get("planning_horizon", 100)
        self.max_depth = config.get("max_depth", 10)
        self.timeout_seconds = config.get("timeout_seconds", 30)

    @abstractmethod
    async def create_plan(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any] = None
    ) -> Plan:
        """Create a plan to achieve goals."""
        pass

    @abstractmethod
    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate plan for consistency and feasibility."""
        pass

    @abstractmethod
    async def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize plan for efficiency."""
        pass


class HierarchicalPlanner(PlanningAlgorithm):
    """Hierarchical task network (HTN) planning."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize hierarchical planner."""
        super().__init__(config)
        self.decomposition_methods = {}
        self.task_library = {}

    async def create_plan(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any] = None
    ) -> Plan:
        """Create hierarchical plan through top-down decomposition."""
        plan_id = self._generate_plan_id()
        plan_steps = []
        execution_graph = nx.DiGraph()

        try:
            # Sort goals by priority
            sorted_goals = sorted(goals, key=lambda g: g.priority)

            for goal in sorted_goals:
                # Decompose high-level goal into subgoals
                subgoals = await self._decompose_goal(goal, initial_state)

                # Create action sequences for subgoals
                for subgoal in subgoals:
                    action_sequence = await self._create_action_sequence(
                        subgoal,
                        available_actions,
                        initial_state
                    )

                    # Add steps to plan
                    for action in action_sequence:
                        step = PlanStep(
                            step_id=self._generate_step_id(),
                            action=action,
                            dependencies=self._get_dependencies(action, plan_steps)
                        )
                        plan_steps.append(step)
                        execution_graph.add_node(
                            step.step_id,
                            action=action.name,
                            type=action.action_type
                        )

                    # Update state with effects
                    initial_state = self._apply_effects(action_sequence, initial_state)

            # Add edges to execution graph
            for step in plan_steps:
                for dep in step.dependencies:
                    execution_graph.add_edge(dep, step.step_id)

            # Create plan
            plan = Plan(
                plan_id=plan_id,
                plan_type=PlanType.HIERARCHICAL,
                goals=goals,
                steps=plan_steps,
                execution_graph=nx.node_link_data(execution_graph)
            )

            # Optimize plan
            plan = await self.optimize_plan(plan)

            return plan

        except Exception as e:
            logger.error(f"Hierarchical planning failed: {str(e)}")
            raise

    async def _decompose_goal(
        self,
        goal: PlanGoal,
        state: Dict[str, Any]
    ) -> List[PlanGoal]:
        """Decompose high-level goal into subgoals."""
        subgoals = []

        # Check if we have a decomposition method for this goal type
        goal_type = self._identify_goal_type(goal)

        if goal_type in self.decomposition_methods:
            # Use predefined decomposition
            decomposition = self.decomposition_methods[goal_type]
            for subgoal_template in decomposition:
                subgoal = PlanGoal(
                    goal_id=f"{goal.goal_id}_{subgoal_template['id']}",
                    description=subgoal_template['description'],
                    priority=goal.priority,
                    preconditions=subgoal_template.get('preconditions', []),
                    success_criteria=subgoal_template.get('criteria', {})
                )
                subgoals.append(subgoal)
        else:
            # Default decomposition based on goal structure
            if goal.preconditions:
                # Create subgoals for each precondition
                for i, precond in enumerate(goal.preconditions):
                    subgoal = PlanGoal(
                        goal_id=f"{goal.goal_id}_pre_{i}",
                        description=f"Achieve: {precond}",
                        priority=goal.priority + 1,
                        success_criteria={"condition": precond}
                    )
                    subgoals.append(subgoal)

            # Add main goal as final subgoal
            subgoals.append(goal)

        return subgoals

    async def _create_action_sequence(
        self,
        goal: PlanGoal,
        available_actions: List[PlanAction],
        state: Dict[str, Any]
    ) -> List[PlanAction]:
        """Create sequence of actions to achieve goal."""
        sequence = []
        current_state = state.copy()

        # Find actions that can contribute to goal
        relevant_actions = self._find_relevant_actions(goal, available_actions)

        # Order actions based on dependencies
        ordered_actions = self._order_actions_by_dependencies(relevant_actions)

        for action in ordered_actions:
            # Check if action's preconditions are satisfied
            if self._preconditions_satisfied(action, current_state):
                sequence.append(action)
                # Apply action effects to state
                current_state = self._apply_action_effects(action, current_state)

                # Check if goal is achieved
                if self._goal_achieved(goal, current_state):
                    break

        return sequence

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate hierarchical plan."""
        errors = []

        # Check for cycles in dependency graph
        if plan.execution_graph:
            graph = nx.node_link_graph(plan.execution_graph)
            if not nx.is_directed_acyclic_graph(graph):
                errors.append("Plan contains circular dependencies")

        # Validate step dependencies
        step_ids = {step.step_id for step in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} has invalid dependency {dep}")

        # Check resource conflicts
        resource_usage = defaultdict(list)
        for step in plan.steps:
            for resource, amount in step.action.resource_requirements.items():
                resource_usage[resource].append((step.step_id, amount))

        # Additional validations...

        return len(errors) == 0, errors

    async def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize hierarchical plan."""
        # Identify parallelizable steps
        parallel_groups = self._identify_parallel_groups(plan)

        # Merge sequential steps that can be batched
        plan = self._merge_batchable_steps(plan)

        # Optimize resource allocation
        plan = self._optimize_resource_allocation(plan)

        return plan

    def _identify_goal_type(self, goal: PlanGoal) -> str:
        """Identify the type of goal for decomposition."""
        # Simple classification based on description
        description_lower = goal.description.lower()

        if "calculate" in description_lower:
            return "calculation"
        elif "analyze" in description_lower:
            return "analysis"
        elif "report" in description_lower:
            return "reporting"
        elif "validate" in description_lower:
            return "validation"
        else:
            return "generic"

    def _find_relevant_actions(
        self,
        goal: PlanGoal,
        available_actions: List[PlanAction]
    ) -> List[PlanAction]:
        """Find actions relevant to achieving goal."""
        relevant = []

        for action in available_actions:
            # Check if action effects contribute to goal
            for effect in action.effects:
                if self._effect_contributes_to_goal(effect, goal):
                    relevant.append(action)
                    break

        return relevant

    def _effect_contributes_to_goal(self, effect: str, goal: PlanGoal) -> bool:
        """Check if an effect contributes to goal achievement."""
        # Simple string matching for now
        for criterion in goal.success_criteria.values():
            if isinstance(criterion, str) and effect in criterion:
                return True
        return False

    def _order_actions_by_dependencies(
        self,
        actions: List[PlanAction]
    ) -> List[PlanAction]:
        """Order actions based on their dependencies."""
        # Build dependency graph
        dep_graph = nx.DiGraph()

        for action in actions:
            dep_graph.add_node(action.action_id, action=action)

        for action in actions:
            for other in actions:
                if action != other:
                    # Check if other's effects satisfy action's preconditions
                    for precond in action.preconditions:
                        if precond in other.effects:
                            dep_graph.add_edge(other.action_id, action.action_id)

        # Topological sort
        try:
            ordered_ids = list(nx.topological_sort(dep_graph))
            ordered_actions = []
            for action_id in ordered_ids:
                action = dep_graph.nodes[action_id]['action']
                ordered_actions.append(action)
            return ordered_actions
        except nx.NetworkXError:
            # Contains cycle, return original order
            return actions

    def _preconditions_satisfied(
        self,
        action: PlanAction,
        state: Dict[str, Any]
    ) -> bool:
        """Check if action preconditions are satisfied in state."""
        for precond in action.preconditions:
            if not self._evaluate_condition(precond, state):
                return False
        return True

    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate a condition against state."""
        # Simple evaluation - in production would use proper logic engine
        return condition in state and state.get(condition, False)

    def _apply_action_effects(
        self,
        action: PlanAction,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply action effects to state."""
        new_state = state.copy()

        for effect in action.effects:
            # Simple effect application
            new_state[effect] = True

        return new_state

    def _apply_effects(
        self,
        actions: List[PlanAction],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply effects of multiple actions to state."""
        new_state = state.copy()

        for action in actions:
            new_state = self._apply_action_effects(action, new_state)

        return new_state

    def _goal_achieved(self, goal: PlanGoal, state: Dict[str, Any]) -> bool:
        """Check if goal is achieved in state."""
        for criterion_name, criterion_value in goal.success_criteria.items():
            if criterion_name not in state:
                return False
            if state[criterion_name] != criterion_value:
                return False
        return True

    def _get_dependencies(
        self,
        action: PlanAction,
        existing_steps: List[PlanStep]
    ) -> Set[str]:
        """Get dependencies for an action based on existing steps."""
        dependencies = set()

        for step in existing_steps:
            # Check if this step produces required preconditions
            for precond in action.preconditions:
                if precond in step.action.effects:
                    dependencies.add(step.step_id)

        return dependencies

    def _identify_parallel_groups(self, plan: Plan) -> List[List[str]]:
        """Identify groups of steps that can execute in parallel."""
        if not plan.execution_graph:
            return []

        graph = nx.node_link_graph(plan.execution_graph)
        parallel_groups = []

        # Find steps with no dependencies on each other
        for node in graph.nodes():
            group = [node]
            for other in graph.nodes():
                if node != other:
                    # Check if there's a path between them
                    if not (nx.has_path(graph, node, other) or
                           nx.has_path(graph, other, node)):
                        group.append(other)

            if len(group) > 1 and group not in parallel_groups:
                parallel_groups.append(group)

        return parallel_groups

    def _merge_batchable_steps(self, plan: Plan) -> Plan:
        """Merge steps that can be batched together."""
        # Group steps by action type
        action_groups = defaultdict(list)
        for step in plan.steps:
            action_groups[step.action.name].append(step)

        # Merge similar consecutive steps
        # Implementation depends on specific action types

        return plan

    def _optimize_resource_allocation(self, plan: Plan) -> Plan:
        """Optimize resource allocation across plan steps."""
        # Calculate total resource requirements
        total_resources = defaultdict(float)
        for step in plan.steps:
            for resource, amount in step.action.resource_requirements.items():
                total_resources[resource] += amount

        # Store in plan metrics
        plan.metrics["total_resources"] = dict(total_resources)

        return plan

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        timestamp = DeterministicClock.now().isoformat()
        return f"plan_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"

    def _generate_step_id(self) -> str:
        """Generate unique step ID."""
        timestamp = DeterministicClock.now().isoformat()
        random_part = hashlib.sha256(f"{timestamp}{id(self)}".encode()).hexdigest()[:6]
        return f"step_{random_part}"


class ReactivePlanner(PlanningAlgorithm):
    """Reactive planning for immediate situation response."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize reactive planner."""
        super().__init__(config)
        self.reaction_rules = {}
        self.situation_assessor = None

    async def create_plan(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any] = None
    ) -> Plan:
        """Create reactive plan based on current situation."""
        plan_id = self._generate_plan_id()

        # Assess current situation
        situation = await self._assess_situation(initial_state)

        # Select immediate actions based on rules
        immediate_actions = self._select_immediate_actions(
            situation,
            goals,
            available_actions
        )

        # Create plan steps
        plan_steps = []
        for action in immediate_actions:
            step = PlanStep(
                step_id=f"reactive_{hashlib.sha256(action.name.encode()).hexdigest()[:6]}",
                action=action,
                dependencies=set()  # Reactive plans have minimal dependencies
            )
            plan_steps.append(step)

        # Create reactive plan
        plan = Plan(
            plan_id=plan_id,
            plan_type=PlanType.REACTIVE,
            goals=goals,
            steps=plan_steps,
            metrics={
                "reaction_time_ms": 0,
                "situation_type": situation.get("type", "unknown")
            }
        )

        return plan

    async def _assess_situation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current situation for reactive planning."""
        situation = {
            "type": "normal",
            "urgency": 0,
            "threats": [],
            "opportunities": []
        }

        # Check for urgent conditions
        if state.get("error_count", 0) > 0:
            situation["type"] = "error_recovery"
            situation["urgency"] = min(state["error_count"] * 2, 10)

        if state.get("deadline_approaching", False):
            situation["type"] = "deadline_pressure"
            situation["urgency"] = 8

        # Identify threats and opportunities
        for key, value in state.items():
            if "risk" in key and value > 0.5:
                situation["threats"].append(key)
            if "opportunity" in key and value > 0.5:
                situation["opportunities"].append(key)

        return situation

    def _select_immediate_actions(
        self,
        situation: Dict[str, Any],
        goals: List[PlanGoal],
        available_actions: List[PlanAction]
    ) -> List[PlanAction]:
        """Select immediate actions based on situation."""
        selected_actions = []

        # Apply reaction rules
        for rule_name, rule in self.reaction_rules.items():
            if self._rule_applies(rule, situation):
                action = self._find_action_by_name(
                    rule["action"],
                    available_actions
                )
                if action:
                    selected_actions.append(action)

        # If no rules apply, select quick actions toward goals
        if not selected_actions:
            for goal in goals[:1]:  # Focus on highest priority goal
                quick_actions = [
                    a for a in available_actions
                    if a.duration_estimate_seconds < 5
                ]
                if quick_actions:
                    selected_actions.append(quick_actions[0])

        return selected_actions

    def _rule_applies(self, rule: Dict[str, Any], situation: Dict[str, Any]) -> bool:
        """Check if a reaction rule applies to situation."""
        if "condition" in rule:
            # Evaluate condition against situation
            condition = rule["condition"]
            if condition["type"] == "urgency_above":
                return situation["urgency"] > condition["threshold"]
            if condition["type"] == "situation_type":
                return situation["type"] == condition["value"]

        return False

    def _find_action_by_name(
        self,
        name: str,
        actions: List[PlanAction]
    ) -> Optional[PlanAction]:
        """Find action by name."""
        for action in actions:
            if action.name == name:
                return action
        return None

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate reactive plan."""
        errors = []

        # Reactive plans should be simple and quick
        if len(plan.steps) > 10:
            errors.append("Reactive plan too complex")

        total_duration = sum(
            step.action.duration_estimate_seconds
            for step in plan.steps
        )
        if total_duration > 30:
            errors.append("Reactive plan takes too long")

        return len(errors) == 0, errors

    async def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize reactive plan for speed."""
        # Sort steps by urgency/priority
        plan.steps.sort(
            key=lambda s: s.action.duration_estimate_seconds
        )

        return plan

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        timestamp = DeterministicClock.now().isoformat()
        return f"reactive_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"


class DeliberativePlanner(PlanningAlgorithm):
    """Deliberative planning for long-term strategic goals."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize deliberative planner."""
        super().__init__(config)
        self.search_algorithm = config.get("search_algorithm", "a_star")
        self.heuristic_function = None

    async def create_plan(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any] = None
    ) -> Plan:
        """Create deliberative plan through search and optimization."""
        plan_id = self._generate_plan_id()

        # Generate multiple plan options
        plan_options = await self._generate_plan_options(
            goals,
            initial_state,
            available_actions,
            constraints
        )

        # Evaluate options using cost-benefit analysis
        best_option = await self._select_best_option(
            plan_options,
            goals,
            constraints
        )

        # Add contingencies
        plan_with_contingencies = await self._add_contingencies(
            best_option,
            available_actions
        )

        return plan_with_contingencies

    async def _generate_plan_options(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any]
    ) -> List[Plan]:
        """Generate multiple plan options."""
        options = []

        # Use different search strategies
        strategies = ["greedy", "optimal", "balanced"]

        for strategy in strategies:
            plan = await self._search_for_plan(
                goals,
                initial_state,
                available_actions,
                strategy,
                constraints
            )
            if plan:
                options.append(plan)

        return options

    async def _search_for_plan(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        strategy: str,
        constraints: Dict[str, Any]
    ) -> Optional[Plan]:
        """Search for a plan using specified strategy."""
        if self.search_algorithm == "a_star":
            return await self._a_star_search(
                goals,
                initial_state,
                available_actions,
                strategy,
                constraints
            )
        # Add other search algorithms as needed
        return None

    async def _a_star_search(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        strategy: str,
        constraints: Dict[str, Any]
    ) -> Optional[Plan]:
        """A* search for optimal plan."""
        # Simplified A* implementation
        # In production, would use proper search with heuristics

        plan_steps = []
        current_state = initial_state.copy()

        for goal in goals:
            # Find path to goal
            path = self._find_path_to_goal(
                current_state,
                goal,
                available_actions,
                strategy
            )

            for action in path:
                step = PlanStep(
                    step_id=f"delib_{hashlib.sha256(action.name.encode()).hexdigest()[:6]}",
                    action=action
                )
                plan_steps.append(step)
                current_state = self._apply_action_effects(action, current_state)

        plan = Plan(
            plan_id=self._generate_plan_id(),
            plan_type=PlanType.DELIBERATIVE,
            goals=goals,
            steps=plan_steps
        )

        return plan

    def _find_path_to_goal(
        self,
        state: Dict[str, Any],
        goal: PlanGoal,
        actions: List[PlanAction],
        strategy: str
    ) -> List[PlanAction]:
        """Find path from state to goal."""
        # Simplified pathfinding
        path = []

        # Select actions based on strategy
        if strategy == "greedy":
            # Choose quickest actions
            sorted_actions = sorted(
                actions,
                key=lambda a: a.duration_estimate_seconds
            )
        elif strategy == "optimal":
            # Choose most effective actions
            sorted_actions = sorted(
                actions,
                key=lambda a: len(a.effects),
                reverse=True
            )
        else:  # balanced
            # Balance speed and effectiveness
            sorted_actions = sorted(
                actions,
                key=lambda a: a.duration_estimate_seconds / max(len(a.effects), 1)
            )

        # Build path
        current_state = state.copy()
        for action in sorted_actions:
            if self._action_contributes_to_goal(action, goal):
                path.append(action)
                current_state = self._apply_action_effects(action, current_state)

                if self._goal_achieved(goal, current_state):
                    break

        return path

    async def _select_best_option(
        self,
        options: List[Plan],
        goals: List[PlanGoal],
        constraints: Dict[str, Any]
    ) -> Plan:
        """Select best plan option using cost-benefit analysis."""
        best_plan = None
        best_score = -float('inf')

        for plan in options:
            score = self._evaluate_plan(plan, goals, constraints)
            if score > best_score:
                best_score = score
                best_plan = plan

        return best_plan or options[0]

    def _evaluate_plan(
        self,
        plan: Plan,
        goals: List[PlanGoal],
        constraints: Dict[str, Any]
    ) -> float:
        """Evaluate plan quality."""
        score = 0.0

        # Time efficiency
        total_time = sum(
            step.action.duration_estimate_seconds
            for step in plan.steps
        )
        score -= total_time * 0.1

        # Resource efficiency
        total_resources = 0
        for step in plan.steps:
            total_resources += sum(step.action.resource_requirements.values())
        score -= total_resources * 0.05

        # Goal achievement
        score += len(goals) * 10

        # Constraint satisfaction
        if constraints:
            violations = self._count_constraint_violations(plan, constraints)
            score -= violations * 5

        return score

    async def _add_contingencies(
        self,
        plan: Plan,
        available_actions: List[PlanAction]
    ) -> Plan:
        """Add contingency actions to plan."""
        # Identify failure points
        failure_points = self._identify_failure_points(plan)

        # Add contingency steps
        for step in plan.steps:
            if step.step_id in failure_points:
                # Find alternative action
                alternative = self._find_alternative_action(
                    step.action,
                    available_actions
                )
                if alternative:
                    # Add as conditional branch
                    contingency_step = PlanStep(
                        step_id=f"{step.step_id}_contingency",
                        action=alternative,
                        dependencies={step.step_id}
                    )
                    plan.steps.append(contingency_step)

        return plan

    def _identify_failure_points(self, plan: Plan) -> Set[str]:
        """Identify potential failure points in plan."""
        failure_points = set()

        for step in plan.steps:
            # High duration indicates complexity
            if step.action.duration_estimate_seconds > 10:
                failure_points.add(step.step_id)

            # Many preconditions indicate fragility
            if len(step.action.preconditions) > 3:
                failure_points.add(step.step_id)

        return failure_points

    def _find_alternative_action(
        self,
        action: PlanAction,
        available_actions: List[PlanAction]
    ) -> Optional[PlanAction]:
        """Find alternative action with similar effects."""
        for alt_action in available_actions:
            if alt_action != action:
                # Check if effects overlap
                effect_overlap = len(
                    set(action.effects) & set(alt_action.effects)
                )
                if effect_overlap > 0:
                    return alt_action
        return None

    def _apply_action_effects(
        self,
        action: PlanAction,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply action effects to state."""
        new_state = state.copy()
        for effect in action.effects:
            new_state[effect] = True
        return new_state

    def _action_contributes_to_goal(
        self,
        action: PlanAction,
        goal: PlanGoal
    ) -> bool:
        """Check if action contributes to goal."""
        for effect in action.effects:
            for criterion in goal.success_criteria.values():
                if isinstance(criterion, str) and effect in criterion:
                    return True
        return False

    def _goal_achieved(
        self,
        goal: PlanGoal,
        state: Dict[str, Any]
    ) -> bool:
        """Check if goal is achieved in state."""
        for criterion_name, criterion_value in goal.success_criteria.items():
            if state.get(criterion_name) != criterion_value:
                return False
        return True

    def _count_constraint_violations(
        self,
        plan: Plan,
        constraints: Dict[str, Any]
    ) -> int:
        """Count constraint violations in plan."""
        violations = 0

        # Check time constraints
        if "max_duration" in constraints:
            total_duration = sum(
                step.action.duration_estimate_seconds
                for step in plan.steps
            )
            if total_duration > constraints["max_duration"]:
                violations += 1

        # Check resource constraints
        if "max_resources" in constraints:
            for resource, max_amount in constraints["max_resources"].items():
                total_usage = sum(
                    step.action.resource_requirements.get(resource, 0)
                    for step in plan.steps
                )
                if total_usage > max_amount:
                    violations += 1

        return violations

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate deliberative plan."""
        errors = []

        # Check plan completeness
        if not plan.steps:
            errors.append("Plan has no steps")

        # Check goal coverage
        # More complex validation...

        return len(errors) == 0, errors

    async def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize deliberative plan."""
        # Resource optimization
        plan = self._optimize_resource_usage(plan)

        # Time optimization
        plan = self._optimize_execution_time(plan)

        return plan

    def _optimize_resource_usage(self, plan: Plan) -> Plan:
        """Optimize resource usage in plan."""
        # Implementation for resource optimization
        return plan

    def _optimize_execution_time(self, plan: Plan) -> Plan:
        """Optimize execution time of plan."""
        # Implementation for time optimization
        return plan

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        timestamp = DeterministicClock.now().isoformat()
        return f"delib_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"


class HybridPlanner(PlanningAlgorithm):
    """Hybrid planner combining reactive, tactical, and strategic layers."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize hybrid planner."""
        super().__init__(config)
        self.reactive_planner = ReactivePlanner(config)
        self.tactical_planner = HierarchicalPlanner(config)
        self.strategic_planner = DeliberativePlanner(config)

    async def create_plan(
        self,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any] = None
    ) -> Plan:
        """Create hybrid plan with multiple layers."""
        # Categorize goals by time horizon
        immediate_goals, short_term_goals, long_term_goals = self._categorize_goals(goals)

        # Create plans for each layer
        reactive_plan = await self.reactive_planner.create_plan(
            immediate_goals,
            initial_state,
            available_actions,
            constraints
        )

        tactical_plan = await self.tactical_planner.create_plan(
            short_term_goals,
            initial_state,
            available_actions,
            constraints
        )

        strategic_plan = await self.strategic_planner.create_plan(
            long_term_goals,
            initial_state,
            available_actions,
            constraints
        )

        # Merge plans
        hybrid_plan = self._merge_plans(
            reactive_plan,
            tactical_plan,
            strategic_plan
        )

        return hybrid_plan

    def _categorize_goals(
        self,
        goals: List[PlanGoal]
    ) -> Tuple[List[PlanGoal], List[PlanGoal], List[PlanGoal]]:
        """Categorize goals by time horizon."""
        immediate = []
        short_term = []
        long_term = []

        now = DeterministicClock.now()

        for goal in goals:
            if goal.deadline:
                time_until = (goal.deadline - now).total_seconds()
                if time_until < 60:  # Less than 1 minute
                    immediate.append(goal)
                elif time_until < 3600:  # Less than 1 hour
                    short_term.append(goal)
                else:
                    long_term.append(goal)
            else:
                # No deadline, use priority
                if goal.priority <= 2:
                    immediate.append(goal)
                elif goal.priority <= 5:
                    short_term.append(goal)
                else:
                    long_term.append(goal)

        return immediate, short_term, long_term

    def _merge_plans(
        self,
        reactive: Plan,
        tactical: Plan,
        strategic: Plan
    ) -> Plan:
        """Merge plans from different layers."""
        merged_steps = []

        # Add reactive steps first (immediate)
        merged_steps.extend(reactive.steps)

        # Add tactical steps
        for step in tactical.steps:
            # Adjust dependencies to come after reactive
            if reactive.steps:
                step.dependencies.add(reactive.steps[-1].step_id)
            merged_steps.append(step)

        # Add strategic steps
        for step in strategic.steps:
            # Adjust dependencies to come after tactical
            if tactical.steps:
                step.dependencies.add(tactical.steps[-1].step_id)
            merged_steps.append(step)

        # Create merged plan
        merged_plan = Plan(
            plan_id=self._generate_plan_id(),
            plan_type=PlanType.HYBRID,
            goals=reactive.goals + tactical.goals + strategic.goals,
            steps=merged_steps,
            metrics={
                "layers": ["reactive", "tactical", "strategic"],
                "reactive_steps": len(reactive.steps),
                "tactical_steps": len(tactical.steps),
                "strategic_steps": len(strategic.steps)
            }
        )

        return merged_plan

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate hybrid plan."""
        errors = []

        # Validate each layer
        # Check layer integration
        # Check for conflicts between layers

        return len(errors) == 0, errors

    async def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize hybrid plan across layers."""
        # Optimize inter-layer transitions
        # Balance resources across layers

        return plan

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        timestamp = DeterministicClock.now().isoformat()
        return f"hybrid_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"


class PlanExecutor:
    """Executor for plan execution with monitoring."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plan executor."""
        self.config = config or {}
        self.execution_queue = asyncio.Queue()
        self.active_plans: Dict[str, Plan] = {}
        self.execution_history = []

    async def execute_plan(
        self,
        plan: Plan,
        action_executor: Callable = None
    ) -> Dict[str, Any]:
        """Execute a plan with monitoring and control."""
        self.active_plans[plan.plan_id] = plan
        results = {}

        try:
            # Update plan status
            plan.status = PlanStatus.EXECUTING

            # Execute steps based on dependencies
            completed_steps = set()

            while len(completed_steps) < len(plan.steps):
                # Find executable steps
                executable = self._find_executable_steps(
                    plan,
                    completed_steps
                )

                if not executable:
                    # Deadlock or all remaining steps failed
                    break

                # Execute steps (potentially in parallel)
                step_results = await self._execute_steps(
                    executable,
                    action_executor
                )

                # Update completed steps
                for step_id, result in step_results.items():
                    if result["status"] == "completed":
                        completed_steps.add(step_id)
                    results[step_id] = result

            # Update plan status
            if len(completed_steps) == len(plan.steps):
                plan.status = PlanStatus.COMPLETED
            else:
                plan.status = PlanStatus.FAILED

        except Exception as e:
            plan.status = PlanStatus.FAILED
            logger.error(f"Plan execution failed: {str(e)}")

        finally:
            del self.active_plans[plan.plan_id]
            self.execution_history.append({
                "plan_id": plan.plan_id,
                "status": plan.status,
                "results": results,
                "timestamp": DeterministicClock.now()
            })

        return results

    def _find_executable_steps(
        self,
        plan: Plan,
        completed: Set[str]
    ) -> List[PlanStep]:
        """Find steps that can be executed now."""
        executable = []

        for step in plan.steps:
            if step.step_id not in completed:
                # Check if all dependencies are completed
                if step.dependencies.issubset(completed):
                    executable.append(step)

        return executable

    async def _execute_steps(
        self,
        steps: List[PlanStep],
        action_executor: Callable
    ) -> Dict[str, Any]:
        """Execute multiple steps."""
        results = {}

        # Execute in parallel if possible
        tasks = []
        for step in steps:
            task = self._execute_single_step(step, action_executor)
            tasks.append(task)

        step_results = await asyncio.gather(*tasks, return_exceptions=True)

        for step, result in zip(steps, step_results):
            if isinstance(result, Exception):
                results[step.step_id] = {
                    "status": "failed",
                    "error": str(result)
                }
            else:
                results[step.step_id] = result

        return results

    async def _execute_single_step(
        self,
        step: PlanStep,
        action_executor: Callable
    ) -> Dict[str, Any]:
        """Execute a single plan step."""
        step.status = PlanStatus.EXECUTING
        step.start_time = DeterministicClock.now()

        try:
            if action_executor:
                result = await action_executor(step.action)
            else:
                # Simulate execution
                await asyncio.sleep(step.action.duration_estimate_seconds)
                result = {"simulated": True}

            step.status = PlanStatus.COMPLETED
            step.result = result
            step.end_time = DeterministicClock.now()

            return {
                "status": "completed",
                "result": result,
                "duration": (step.end_time - step.start_time).total_seconds()
            }

        except Exception as e:
            step.status = PlanStatus.FAILED
            step.error = str(e)
            step.end_time = DeterministicClock.now()

            return {
                "status": "failed",
                "error": str(e),
                "duration": (step.end_time - step.start_time).total_seconds()
            }


class PlanningFramework:
    """Main framework for planning capabilities."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize planning framework."""
        self.config = config or {}

        # Initialize planners
        self.planners = {
            PlanType.HIERARCHICAL: HierarchicalPlanner(config),
            PlanType.REACTIVE: ReactivePlanner(config),
            PlanType.DELIBERATIVE: DeliberativePlanner(config),
            PlanType.HYBRID: HybridPlanner(config)
        }

        self.executor = PlanExecutor(config)
        self.plan_cache = {}

    async def create_plan(
        self,
        plan_type: PlanType,
        goals: List[PlanGoal],
        initial_state: Dict[str, Any],
        available_actions: List[PlanAction],
        constraints: Dict[str, Any] = None
    ) -> Plan:
        """Create a plan using specified planner."""
        if plan_type not in self.planners:
            raise ValueError(f"Unknown plan type: {plan_type}")

        planner = self.planners[plan_type]
        plan = await planner.create_plan(
            goals,
            initial_state,
            available_actions,
            constraints
        )

        # Cache plan
        self.plan_cache[plan.plan_id] = plan

        return plan

    async def execute_plan(
        self,
        plan: Plan,
        action_executor: Callable = None
    ) -> Dict[str, Any]:
        """Execute a plan."""
        return await self.executor.execute_plan(plan, action_executor)

    def validate_plan(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Validate a plan."""
        if plan.plan_type in self.planners:
            return self.planners[plan.plan_type].validate_plan(plan)
        return False, ["Unknown plan type"]

    async def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize a plan."""
        if plan.plan_type in self.planners:
            return await self.planners[plan.plan_type].optimize_plan(plan)
        return plan

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get cached plan by ID."""
        return self.plan_cache.get(plan_id)