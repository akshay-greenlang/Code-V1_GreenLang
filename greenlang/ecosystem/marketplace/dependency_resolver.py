# -*- coding: utf-8 -*-
"""
Dependency Resolution System

Implements dependency graph construction, version conflict resolution,
and transitive dependency management for marketplace agents.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json

from sqlalchemy.orm import Session

from greenlang.marketplace.models import MarketplaceAgent, AgentVersion, AgentDependency
from greenlang.marketplace.versioning import SemanticVersion, VersionConstraint

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving version conflicts"""
    NEWEST_COMPATIBLE = "newest_compatible"
    MOST_STABLE = "most_stable"
    USER_PREFERENCE = "user_preference"


@dataclass
class DependencyNode:
    """Node in dependency graph"""
    agent_id: str
    agent_name: str
    version: Optional[str] = None
    version_constraint: Optional[str] = None
    required_by: Set[str] = field(default_factory=set)
    dependencies: List['DependencyNode'] = field(default_factory=list)
    optional: bool = False


@dataclass
class VersionConflict:
    """Version conflict information"""
    agent_id: str
    agent_name: str
    required_versions: List[Tuple[str, str]]  # [(requester, constraint), ...]
    suggested_resolution: Optional[str] = None


@dataclass
class ResolutionResult:
    """Dependency resolution result"""
    success: bool
    resolved_versions: Dict[str, str] = field(default_factory=dict)  # {agent_id: version}
    conflicts: List[VersionConflict] = field(default_factory=list)
    install_order: List[str] = field(default_factory=list)  # Topologically sorted
    errors: List[str] = field(default_factory=list)


class DependencyGraph:
    """
    Dependency graph builder and analyzer.

    Builds a graph of agent dependencies and provides graph analysis.
    """

    def __init__(self, session: Session):
        self.session = session
        self.graph: Dict[str, DependencyNode] = {}

    def build_from_agent(
        self,
        agent_id: str,
        version: Optional[str] = None
    ) -> DependencyNode:
        """
        Build dependency graph starting from an agent.

        Args:
            agent_id: Root agent UUID
            version: Optional specific version (uses latest if None)

        Returns:
            Root dependency node
        """
        # Get agent
        agent = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.id == agent_id
        ).first()

        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Get version
        if version:
            agent_version = self.session.query(AgentVersion).filter(
                AgentVersion.agent_id == agent_id,
                AgentVersion.version == version
            ).first()
        else:
            # Get latest
            agent_version = self.session.query(AgentVersion).filter(
                AgentVersion.agent_id == agent_id,
                AgentVersion.deprecated == False
            ).order_by(AgentVersion.published_at.desc()).first()

        if not agent_version:
            raise ValueError(f"No version found for agent {agent_id}")

        # Build graph recursively
        root = self._build_node(agent, agent_version)
        self.graph[agent_id] = root

        return root

    def _build_node(
        self,
        agent: MarketplaceAgent,
        version: AgentVersion,
        visited: Optional[Set[str]] = None
    ) -> DependencyNode:
        """Build dependency node recursively"""
        if visited is None:
            visited = set()

        agent_id = str(agent.id)

        # Check for circular dependency
        if agent_id in visited:
            logger.warning(f"Circular dependency detected for agent {agent_id}")
            return DependencyNode(
                agent_id=agent_id,
                agent_name=agent.name,
                version=version.version
            )

        visited.add(agent_id)

        node = DependencyNode(
            agent_id=agent_id,
            agent_name=agent.name,
            version=version.version
        )

        # Get dependencies
        dependencies = self.session.query(AgentDependency).filter(
            AgentDependency.version_id == version.id
        ).all()

        for dep in dependencies:
            # Get dependency agent
            dep_agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == dep.dependency_agent_id
            ).first()

            if not dep_agent:
                continue

            # Find compatible version
            dep_version = self._find_compatible_version(
                str(dep.dependency_agent_id),
                dep.version_constraint
            )

            if dep_version:
                # Recursively build dependency node
                dep_node = self._build_node(dep_agent, dep_version, visited.copy())
                dep_node.version_constraint = dep.version_constraint
                dep_node.required_by.add(agent_id)
                dep_node.optional = dep.optional

                node.dependencies.append(dep_node)
            else:
                logger.warning(
                    f"No compatible version found for {dep_agent.name} "
                    f"with constraint {dep.version_constraint}"
                )

        return node

    def _find_compatible_version(
        self,
        agent_id: str,
        constraint: str
    ) -> Optional[AgentVersion]:
        """Find version matching constraint"""
        constraints = VersionConstraint.parse(constraint)

        versions = self.session.query(AgentVersion).filter(
            AgentVersion.agent_id == agent_id,
            AgentVersion.deprecated == False
        ).order_by(AgentVersion.published_at.desc()).all()

        # Find latest matching version
        for version in versions:
            sem_ver = SemanticVersion.parse(version.version)

            if all(c.matches(sem_ver) for c in constraints):
                return version

        return None

    def get_all_dependencies(self, root: DependencyNode) -> Set[str]:
        """Get all dependencies (transitive) as set of agent IDs"""
        dependencies = set()

        def traverse(node: DependencyNode):
            for dep in node.dependencies:
                dependencies.add(dep.agent_id)
                traverse(dep)

        traverse(root)
        return dependencies

    def topological_sort(self, root: DependencyNode) -> List[str]:
        """
        Get topologically sorted list of agent IDs (install order).

        Dependencies come before dependents.

        Args:
            root: Root dependency node

        Returns:
            List of agent IDs in install order
        """
        # Build adjacency list and in-degree count
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set()

        def build_graph(node: DependencyNode):
            all_nodes.add(node.agent_id)
            for dep in node.dependencies:
                adj_list[dep.agent_id].append(node.agent_id)
                in_degree[node.agent_id] += 1
                build_graph(dep)

        build_graph(root)

        # Initialize in-degree for nodes with no dependencies
        for node_id in all_nodes:
            if node_id not in in_degree:
                in_degree[node_id] = 0

        # Kahn's algorithm
        queue = deque([node_id for node_id in all_nodes if in_degree[node_id] == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(all_nodes):
            logger.error("Cycle detected in dependency graph")
            return []

        return result

    def detect_circular_dependencies(self, root: DependencyNode) -> List[List[str]]:
        """
        Detect circular dependencies.

        Returns:
            List of circular dependency chains
        """
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node: DependencyNode):
            visited.add(node.agent_id)
            rec_stack.append(node.agent_id)

            for dep in node.dependencies:
                if dep.agent_id not in visited:
                    dfs(dep)
                elif dep.agent_id in rec_stack:
                    # Found cycle
                    cycle_start = rec_stack.index(dep.agent_id)
                    cycle = rec_stack[cycle_start:] + [dep.agent_id]
                    cycles.append(cycle)

            rec_stack.pop()

        dfs(root)
        return cycles


class VersionConflictResolver:
    """
    Resolve version conflicts in dependency graph.

    When multiple agents require different versions of the same dependency,
    find a compatible version that satisfies all constraints.
    """

    def __init__(self, session: Session):
        self.session = session

    def resolve_conflicts(
        self,
        conflicts: Dict[str, List[Tuple[str, str]]],
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.NEWEST_COMPATIBLE
    ) -> Dict[str, Optional[str]]:
        """
        Resolve version conflicts.

        Args:
            conflicts: Dict of {agent_id: [(requester, constraint), ...]}
            strategy: Resolution strategy

        Returns:
            Dict of {agent_id: resolved_version}
        """
        resolutions = {}

        for agent_id, requirements in conflicts.items():
            resolved = self._resolve_single_conflict(
                agent_id,
                requirements,
                strategy
            )
            resolutions[agent_id] = resolved

        return resolutions

    def _resolve_single_conflict(
        self,
        agent_id: str,
        requirements: List[Tuple[str, str]],
        strategy: ConflictResolutionStrategy
    ) -> Optional[str]:
        """Resolve conflict for single agent"""
        # Get all available versions
        versions = self.session.query(AgentVersion).filter(
            AgentVersion.agent_id == agent_id,
            AgentVersion.deprecated == False
        ).order_by(AgentVersion.published_at.desc()).all()

        # Parse all constraints
        all_constraints = []
        for _, constraint_str in requirements:
            constraints = VersionConstraint.parse(constraint_str)
            all_constraints.extend(constraints)

        # Find versions satisfying all constraints
        compatible_versions = []
        for version in versions:
            sem_ver = SemanticVersion.parse(version.version)

            if all(c.matches(sem_ver) for c in all_constraints):
                compatible_versions.append(version)

        if not compatible_versions:
            logger.error(
                f"No version of agent {agent_id} satisfies all constraints: "
                f"{[c for _, c in requirements]}"
            )
            return None

        # Apply strategy
        if strategy == ConflictResolutionStrategy.NEWEST_COMPATIBLE:
            # Return latest compatible version
            return compatible_versions[0].version

        elif strategy == ConflictResolutionStrategy.MOST_STABLE:
            # Prefer versions with most downloads
            stable = max(compatible_versions, key=lambda v: v.downloads)
            return stable.version

        else:
            # Default to newest
            return compatible_versions[0].version


class DependencyResolver:
    """
    Main dependency resolver.

    Resolves all dependencies for an agent, detects conflicts,
    and provides installation plan.
    """

    def __init__(self, session: Session):
        self.session = session
        self.graph_builder = DependencyGraph(session)
        self.conflict_resolver = VersionConflictResolver(session)

    def resolve(
        self,
        agent_id: str,
        version: Optional[str] = None,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.NEWEST_COMPATIBLE
    ) -> ResolutionResult:
        """
        Resolve all dependencies for an agent.

        Args:
            agent_id: Agent UUID
            version: Optional specific version
            strategy: Conflict resolution strategy

        Returns:
            Resolution result with install plan
        """
        result = ResolutionResult(success=False)

        try:
            # Build dependency graph
            root = self.graph_builder.build_from_agent(agent_id, version)

            # Check for circular dependencies
            cycles = self.graph_builder.detect_circular_dependencies(root)
            if cycles:
                result.errors.append(
                    f"Circular dependencies detected: {cycles}"
                )
                return result

            # Collect all dependency requirements
            requirements = self._collect_requirements(root)

            # Detect conflicts
            conflicts = self._detect_conflicts(requirements)

            if conflicts:
                # Try to resolve conflicts
                resolutions = self.conflict_resolver.resolve_conflicts(
                    conflicts,
                    strategy
                )

                # Check if all conflicts resolved
                unresolved = [
                    agent_id for agent_id, version in resolutions.items()
                    if version is None
                ]

                if unresolved:
                    result.conflicts = [
                        VersionConflict(
                            agent_id=agent_id,
                            agent_name=self._get_agent_name(agent_id),
                            required_versions=conflicts[agent_id]
                        )
                        for agent_id in unresolved
                    ]
                    result.errors.append(
                        f"Unable to resolve conflicts for: {unresolved}"
                    )
                    return result

                # Apply resolutions
                result.resolved_versions.update(resolutions)

            # Add all dependencies with their versions
            self._collect_all_versions(root, result.resolved_versions)

            # Get installation order
            result.install_order = self.graph_builder.topological_sort(root)

            result.success = True

        except Exception as e:
            logger.error(f"Error resolving dependencies: {e}", exc_info=True)
            result.errors.append(f"Resolution failed: {str(e)}")

        return result

    def _collect_requirements(
        self,
        root: DependencyNode
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Collect all dependency requirements.

        Returns dict of {agent_id: [(requester, constraint), ...]}
        """
        requirements = defaultdict(list)

        def traverse(node: DependencyNode):
            for dep in node.dependencies:
                requirements[dep.agent_id].append(
                    (node.agent_id, dep.version_constraint)
                )
                traverse(dep)

        traverse(root)
        return dict(requirements)

    def _detect_conflicts(
        self,
        requirements: Dict[str, List[Tuple[str, str]]]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Detect version conflicts.

        Returns agents with conflicting requirements.
        """
        conflicts = {}

        for agent_id, reqs in requirements.items():
            if len(reqs) > 1:
                # Multiple requirements for same agent
                # Check if they conflict
                if self._has_conflicts(reqs):
                    conflicts[agent_id] = reqs

        return conflicts

    def _has_conflicts(self, requirements: List[Tuple[str, str]]) -> bool:
        """Check if requirements have conflicts"""
        # Parse all constraints
        all_constraints = []
        for _, constraint_str in requirements:
            constraints = VersionConstraint.parse(constraint_str)
            all_constraints.append(constraints)

        # Check if there's any version that satisfies all
        # For simplicity, we'll consider it a conflict if constraints differ
        constraint_strings = set(c for _, c in requirements)
        return len(constraint_strings) > 1

    def _collect_all_versions(
        self,
        root: DependencyNode,
        versions: Dict[str, str]
    ):
        """Collect all versions from dependency tree"""
        versions[root.agent_id] = root.version

        for dep in root.dependencies:
            if dep.agent_id not in versions:
                self._collect_all_versions(dep, versions)

    def _get_agent_name(self, agent_id: str) -> str:
        """Get agent name from ID"""
        agent = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.id == agent_id
        ).first()
        return agent.name if agent else agent_id

    def generate_lockfile(
        self,
        agent_id: str,
        resolved_versions: Dict[str, str]
    ) -> str:
        """
        Generate lockfile with exact versions.

        Args:
            agent_id: Root agent UUID
            resolved_versions: Resolved dependency versions

        Returns:
            Lockfile content (JSON)
        """
        lockfile = {
            "version": "1.0",
            "agent_id": agent_id,
            "dependencies": {}
        }

        for dep_agent_id, version in resolved_versions.items():
            agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == dep_agent_id
            ).first()

            if agent:
                lockfile["dependencies"][agent.name] = {
                    "id": dep_agent_id,
                    "version": version
                }

        return json.dumps(lockfile, indent=2)
