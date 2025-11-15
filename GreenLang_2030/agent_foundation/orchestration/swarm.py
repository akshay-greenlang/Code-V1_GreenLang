"""
Swarm - Agent swarm implementation for distributed intelligence.

This module implements swarm intelligence patterns enabling 10,000+ agents
to work collectively on large-scale problems with emergent behavior and
distributed decision-making capabilities.

Example:
    >>> swarm = SwarmOrchestrator(message_bus, config)
    >>> await swarm.initialize()
    >>>
    >>> # Deploy swarm for distributed processing
    >>> task = SwarmTask(
    ...     objective="Process 1M carbon calculations",
    ...     data_partitions=1000,
    ...     agents_required=100
    ... )
    >>> result = await swarm.deploy(task)
"""

from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import logging
from datetime import datetime, timezone
import uuid
import random
import math
from dataclasses import dataclass, field
import hashlib
import numpy as np

from prometheus_client import Counter, Histogram, Gauge
import networkx as nx

from .message_bus import MessageBus, Message, MessageType, Priority

logger = logging.getLogger(__name__)

# Metrics
swarm_size_gauge = Gauge('swarm_agents_total', 'Total agents in swarm', ['swarm_id', 'role'])
swarm_task_counter = Counter('swarm_tasks_total', 'Total swarm tasks', ['status'])
swarm_convergence_histogram = Histogram('swarm_convergence_time_ms', 'Time to swarm convergence')
swarm_efficiency_gauge = Gauge('swarm_efficiency', 'Swarm processing efficiency')


class SwarmRole(str, Enum):
    """Roles within the swarm."""
    QUEEN = "QUEEN"           # Central coordinator
    WORKER = "WORKER"         # Task executor
    SCOUT = "SCOUT"           # Explorer/searcher
    SOLDIER = "SOLDIER"       # Defender/validator
    NURSE = "NURSE"           # Healer/recovery
    FORAGER = "FORAGER"       # Resource gatherer


class SwarmBehavior(str, Enum):
    """Swarm behavior patterns."""
    FORAGING = "FORAGING"           # Search and gather
    FLOCKING = "FLOCKING"           # Coordinated movement
    SWARMING = "SWARMING"           # Converge on target
    DISPERSING = "DISPERSING"       # Spread out
    CLUSTERING = "CLUSTERING"       # Form groups
    MIGRATING = "MIGRATING"         # Move to new location


class SwarmAgent(BaseModel):
    """Individual swarm agent."""

    agent_id: str = Field(default_factory=lambda: f"swarm-agent-{uuid.uuid4().hex[:8]}")
    role: SwarmRole = Field(default=SwarmRole.WORKER)
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    fitness: float = Field(default=0.0, ge=0.0, le=1.0)
    energy: float = Field(default=1.0, ge=0.0, le=1.0)
    memory: Dict[str, Any] = Field(default_factory=dict)
    neighbors: List[str] = Field(default_factory=list)
    task_queue: List[str] = Field(default_factory=list)
    state: str = Field(default="IDLE")
    pheromone_trails: Dict[str, float] = Field(default_factory=dict)

    def distance_to(self, other: "SwarmAgent") -> float:
        """Calculate Euclidean distance to another agent."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))

    def update_velocity(
        self,
        target: List[float],
        neighbors: List["SwarmAgent"],
        weights: Dict[str, float]
    ) -> None:
        """Update velocity based on swarm rules."""
        # Separation - avoid crowding neighbors
        separation = [0.0, 0.0, 0.0]
        for neighbor in neighbors:
            if self.distance_to(neighbor) < weights.get("separation_radius", 1.0):
                for i in range(3):
                    separation[i] -= (neighbor.position[i] - self.position[i])

        # Alignment - steer towards average heading of neighbors
        alignment = [0.0, 0.0, 0.0]
        if neighbors:
            for i in range(3):
                alignment[i] = sum(n.velocity[i] for n in neighbors) / len(neighbors)

        # Cohesion - steer towards average position of neighbors
        cohesion = [0.0, 0.0, 0.0]
        if neighbors:
            center = [sum(n.position[i] for n in neighbors) / len(neighbors) for i in range(3)]
            for i in range(3):
                cohesion[i] = center[i] - self.position[i]

        # Attraction to target
        attraction = [0.0, 0.0, 0.0]
        if target:
            for i in range(3):
                attraction[i] = target[i] - self.position[i]

        # Update velocity with weighted sum
        for i in range(3):
            self.velocity[i] = (
                weights.get("inertia", 0.5) * self.velocity[i] +
                weights.get("separation", 0.3) * separation[i] +
                weights.get("alignment", 0.2) * alignment[i] +
                weights.get("cohesion", 0.2) * cohesion[i] +
                weights.get("attraction", 0.3) * attraction[i]
            )

            # Limit velocity
            max_speed = weights.get("max_speed", 1.0)
            speed = math.sqrt(sum(v ** 2 for v in self.velocity))
            if speed > max_speed:
                self.velocity[i] = (self.velocity[i] / speed) * max_speed

    def update_position(self, dt: float = 1.0) -> None:
        """Update position based on velocity."""
        for i in range(3):
            self.position[i] += self.velocity[i] * dt

    def deposit_pheromone(self, pheromone_type: str, strength: float) -> None:
        """Deposit pheromone trail."""
        self.pheromone_trails[pheromone_type] = min(
            self.pheromone_trails.get(pheromone_type, 0.0) + strength,
            1.0
        )

    def evaporate_pheromones(self, rate: float = 0.01) -> None:
        """Evaporate pheromone trails over time."""
        for pheromone_type in list(self.pheromone_trails.keys()):
            self.pheromone_trails[pheromone_type] *= (1 - rate)
            if self.pheromone_trails[pheromone_type] < 0.01:
                del self.pheromone_trails[pheromone_type]


class SwarmTask(BaseModel):
    """Task for swarm execution."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str = Field(..., description="Task objective")
    data_partitions: int = Field(default=1, ge=1, description="Number of data partitions")
    agents_required: int = Field(default=10, ge=1, description="Number of agents needed")
    behavior: SwarmBehavior = Field(default=SwarmBehavior.FORAGING)
    target_position: Optional[List[float]] = Field(None, description="Target position in space")
    timeout_ms: int = Field(default=60000, ge=0)
    convergence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmState(BaseModel):
    """Current swarm state."""

    swarm_id: str = Field(...)
    agents: List[SwarmAgent] = Field(default_factory=list)
    behavior: SwarmBehavior = Field(...)
    center_of_mass: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    spread: float = Field(default=0.0, description="Swarm spread/dispersion")
    convergence: float = Field(default=0.0, ge=0.0, le=1.0)
    fitness: float = Field(default=0.0, ge=0.0, le=1.0)
    energy: float = Field(default=1.0, ge=0.0, le=1.0)
    pheromone_map: Dict[str, Dict[Tuple[int, int, int], float]] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)


@dataclass
class SwarmConfig:
    """Swarm configuration parameters."""
    min_agents: int = 10
    max_agents: int = 1000
    neighbor_radius: float = 5.0
    communication_radius: float = 10.0
    pheromone_evaporation_rate: float = 0.01
    energy_consumption_rate: float = 0.001
    reproduction_threshold: float = 0.8
    death_threshold: float = 0.1
    mutation_rate: float = 0.05

    # Swarm dynamics weights
    separation_weight: float = 0.3
    alignment_weight: float = 0.2
    cohesion_weight: float = 0.2
    attraction_weight: float = 0.3
    inertia_weight: float = 0.5
    max_speed: float = 1.0


class SwarmOrchestrator:
    """
    Swarm orchestrator for distributed intelligence.

    Manages swarm behavior, coordination, and emergent intelligence
    for solving complex distributed problems.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        config: Optional[SwarmConfig] = None
    ):
        """Initialize swarm orchestrator."""
        self.message_bus = message_bus
        self.config = config or SwarmConfig()

        # Swarm management
        self.swarms: Dict[str, SwarmState] = {}
        self.active_tasks: Dict[str, SwarmTask] = {}

        # Agent pool
        self.agent_pool: List[SwarmAgent] = []
        self.available_agents: Set[str] = set()

        # Optimization state
        self.global_best_position: Optional[List[float]] = None
        self.global_best_fitness: float = 0.0

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize swarm orchestrator."""
        logger.info("Initializing SwarmOrchestrator")

        self._running = True

        # Create initial agent pool
        await self._create_agent_pool(self.config.min_agents)

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._swarm_coordinator())
        )
        self._tasks.append(
            asyncio.create_task(self._pheromone_manager())
        )
        self._tasks.append(
            asyncio.create_task(self._evolution_manager())
        )

        logger.info(f"SwarmOrchestrator initialized with {len(self.agent_pool)} agents")

    async def shutdown(self) -> None:
        """Shutdown swarm orchestrator."""
        logger.info("Shutting down SwarmOrchestrator")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("SwarmOrchestrator shutdown complete")

    async def deploy(
        self,
        task: SwarmTask,
        swarm_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy swarm for task execution.

        Args:
            task: Swarm task specification
            swarm_id: Optional swarm ID to reuse

        Returns:
            Task execution result
        """
        swarm_id = swarm_id or f"swarm-{uuid.uuid4().hex[:8]}"
        self.active_tasks[task.task_id] = task

        try:
            logger.info(f"Deploying swarm {swarm_id} for task {task.task_id}")

            # Allocate agents
            agents = await self._allocate_agents(task.agents_required)

            # Initialize swarm state
            swarm = SwarmState(
                swarm_id=swarm_id,
                agents=agents,
                behavior=task.behavior
            )
            self.swarms[swarm_id] = swarm

            # Update metrics
            swarm_size_gauge.labels(swarm_id=swarm_id, role="WORKER").set(len(agents))

            # Execute swarm behavior
            result = await asyncio.wait_for(
                self._execute_swarm_task(swarm, task),
                timeout=task.timeout_ms / 1000
            )

            swarm_task_counter.labels(status="success").inc()
            return result

        except asyncio.TimeoutError:
            logger.error(f"Swarm task {task.task_id} timeout")
            swarm_task_counter.labels(status="timeout").inc()
            raise

        except Exception as e:
            logger.error(f"Swarm task {task.task_id} failed: {e}")
            swarm_task_counter.labels(status="failure").inc()
            raise

        finally:
            # Release agents
            await self._release_agents([a.agent_id for a in swarm.agents])
            del self.swarms[swarm_id]
            del self.active_tasks[task.task_id]

    async def _execute_swarm_task(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> Dict[str, Any]:
        """Execute swarm task based on behavior."""
        start_time = datetime.now(timezone.utc)
        iterations = 0
        max_iterations = 1000

        while iterations < max_iterations:
            iterations += 1

            # Update swarm dynamics
            await self._update_swarm_dynamics(swarm, task)

            # Execute behavior-specific logic
            if task.behavior == SwarmBehavior.FORAGING:
                await self._execute_foraging(swarm, task)
            elif task.behavior == SwarmBehavior.FLOCKING:
                await self._execute_flocking(swarm, task)
            elif task.behavior == SwarmBehavior.CLUSTERING:
                await self._execute_clustering(swarm, task)
            elif task.behavior == SwarmBehavior.SWARMING:
                await self._execute_swarming(swarm, task)
            else:
                await self._execute_generic(swarm, task)

            # Calculate swarm metrics
            swarm.convergence = self._calculate_convergence(swarm)
            swarm.fitness = self._calculate_swarm_fitness(swarm)

            # Check convergence
            if swarm.convergence >= task.convergence_threshold:
                logger.info(f"Swarm converged after {iterations} iterations")
                break

            # Energy consumption
            for agent in swarm.agents:
                agent.energy -= self.config.energy_consumption_rate
                if agent.energy <= 0:
                    agent.state = "EXHAUSTED"

            await asyncio.sleep(0.01)  # Small delay for async processing

        # Calculate final metrics
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        swarm_convergence_histogram.observe(duration_ms)

        efficiency = swarm.fitness / (iterations / max_iterations) if iterations > 0 else 0
        swarm_efficiency_gauge.set(efficiency)

        return {
            "swarm_id": swarm.swarm_id,
            "task_id": task.task_id,
            "convergence": swarm.convergence,
            "fitness": swarm.fitness,
            "iterations": iterations,
            "duration_ms": duration_ms,
            "efficiency": efficiency,
            "center_of_mass": swarm.center_of_mass,
            "spread": swarm.spread,
            "best_position": self.global_best_position,
            "best_fitness": self.global_best_fitness
        }

    async def _update_swarm_dynamics(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> None:
        """Update swarm dynamics (position, velocity, etc.)."""
        # Find neighbors for each agent
        for agent in swarm.agents:
            neighbors = self._find_neighbors(agent, swarm.agents, self.config.neighbor_radius)
            agent.neighbors = [n.agent_id for n in neighbors]

            # Update velocity based on swarm rules
            weights = {
                "separation": self.config.separation_weight,
                "alignment": self.config.alignment_weight,
                "cohesion": self.config.cohesion_weight,
                "attraction": self.config.attraction_weight,
                "inertia": self.config.inertia_weight,
                "max_speed": self.config.max_speed,
                "separation_radius": self.config.neighbor_radius / 2
            }

            target = task.target_position or self.global_best_position or [0, 0, 0]
            agent.update_velocity(target, neighbors, weights)
            agent.update_position()

        # Update swarm center of mass
        if swarm.agents:
            for i in range(3):
                swarm.center_of_mass[i] = sum(a.position[i] for a in swarm.agents) / len(swarm.agents)

        # Calculate spread
        distances = [agent.distance_to(SwarmAgent(position=swarm.center_of_mass)) for agent in swarm.agents]
        swarm.spread = np.std(distances) if distances else 0

    async def _execute_foraging(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> None:
        """Execute foraging behavior - search and gather."""
        for agent in swarm.agents:
            if agent.state == "EXHAUSTED":
                continue

            # Explore or exploit based on fitness
            if agent.fitness < 0.5:
                # Explore - random walk
                for i in range(3):
                    agent.velocity[i] += random.uniform(-0.1, 0.1)
            else:
                # Exploit - follow pheromone trails
                strongest_pheromone = self._find_strongest_pheromone(
                    agent.position,
                    swarm.pheromone_map.get("food", {})
                )
                if strongest_pheromone:
                    for i in range(3):
                        agent.velocity[i] += 0.1 * (strongest_pheromone[i] - agent.position[i])

            # Evaluate fitness at new position
            agent.fitness = await self._evaluate_fitness(agent, task)

            # Update global best
            if agent.fitness > self.global_best_fitness:
                self.global_best_fitness = agent.fitness
                self.global_best_position = agent.position.copy()

            # Deposit pheromone if successful
            if agent.fitness > 0.7:
                agent.deposit_pheromone("food", agent.fitness)

    async def _execute_flocking(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> None:
        """Execute flocking behavior - coordinated movement."""
        # Already handled by swarm dynamics update
        # Additional flocking-specific logic here
        pass

    async def _execute_clustering(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> None:
        """Execute clustering behavior - form groups."""
        # K-means style clustering
        num_clusters = task.metadata.get("num_clusters", 3)

        # Assign agents to clusters based on position
        clusters = {i: [] for i in range(num_clusters)}
        for agent in swarm.agents:
            cluster_id = hash(agent.agent_id) % num_clusters
            clusters[cluster_id].append(agent)

        # Move agents towards cluster centers
        for cluster_id, cluster_agents in clusters.items():
            if cluster_agents:
                center = [
                    sum(a.position[i] for a in cluster_agents) / len(cluster_agents)
                    for i in range(3)
                ]
                for agent in cluster_agents:
                    for i in range(3):
                        agent.velocity[i] += 0.1 * (center[i] - agent.position[i])

    async def _execute_swarming(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> None:
        """Execute swarming behavior - converge on target."""
        if task.target_position:
            for agent in swarm.agents:
                # Strong attraction to target
                for i in range(3):
                    agent.velocity[i] += 0.5 * (task.target_position[i] - agent.position[i])

    async def _execute_generic(
        self,
        swarm: SwarmState,
        task: SwarmTask
    ) -> None:
        """Execute generic swarm behavior."""
        # Particle Swarm Optimization (PSO) style update
        for agent in swarm.agents:
            # Personal best update
            if "personal_best_position" not in agent.memory:
                agent.memory["personal_best_position"] = agent.position.copy()
                agent.memory["personal_best_fitness"] = agent.fitness

            # Evaluate fitness
            agent.fitness = await self._evaluate_fitness(agent, task)

            # Update personal best
            if agent.fitness > agent.memory["personal_best_fitness"]:
                agent.memory["personal_best_fitness"] = agent.fitness
                agent.memory["personal_best_position"] = agent.position.copy()

            # Update global best
            if agent.fitness > self.global_best_fitness:
                self.global_best_fitness = agent.fitness
                self.global_best_position = agent.position.copy()

            # PSO velocity update
            cognitive_weight = 0.5
            social_weight = 0.5

            for i in range(3):
                cognitive = random.random() * cognitive_weight * (
                    agent.memory["personal_best_position"][i] - agent.position[i]
                )
                social = 0
                if self.global_best_position:
                    social = random.random() * social_weight * (
                        self.global_best_position[i] - agent.position[i]
                    )
                agent.velocity[i] = self.config.inertia_weight * agent.velocity[i] + cognitive + social

    async def _evaluate_fitness(
        self,
        agent: SwarmAgent,
        task: SwarmTask
    ) -> float:
        """Evaluate agent fitness for task."""
        # Example fitness function - distance to target
        if task.target_position:
            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(agent.position, task.target_position))
            )
            max_distance = math.sqrt(3 * 100 ** 2)  # Assuming 100x100x100 space
            fitness = 1.0 - (distance / max_distance)
        else:
            # Random fitness for demonstration
            fitness = random.random()

        # Send agent task via message bus for actual computation
        if self.message_bus:
            message = Message(
                sender_id=f"swarm-{task.task_id}",
                recipient_id=agent.agent_id,
                message_type=MessageType.REQUEST,
                priority=Priority.NORMAL,
                payload={
                    "task": "evaluate_fitness",
                    "position": agent.position,
                    "task_data": task.metadata
                }
            )

            response = await self.message_bus.request_response(message, timeout_ms=1000)
            if response:
                fitness = response.payload.get("fitness", fitness)

        return max(0.0, min(1.0, fitness))

    def _find_neighbors(
        self,
        agent: SwarmAgent,
        all_agents: List[SwarmAgent],
        radius: float
    ) -> List[SwarmAgent]:
        """Find neighboring agents within radius."""
        neighbors = []
        for other in all_agents:
            if other.agent_id != agent.agent_id:
                if agent.distance_to(other) <= radius:
                    neighbors.append(other)
        return neighbors

    def _find_strongest_pheromone(
        self,
        position: List[float],
        pheromone_map: Dict[Tuple[int, int, int], float]
    ) -> Optional[List[float]]:
        """Find strongest pheromone signal near position."""
        if not pheromone_map:
            return None

        # Discretize position
        grid_pos = tuple(int(p) for p in position)

        # Check surrounding cells
        strongest = None
        max_strength = 0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_pos = (grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz)
                    strength = pheromone_map.get(check_pos, 0)
                    if strength > max_strength:
                        max_strength = strength
                        strongest = list(check_pos)

        return strongest

    def _calculate_convergence(self, swarm: SwarmState) -> float:
        """Calculate swarm convergence metric."""
        if not swarm.agents:
            return 0.0

        # Convergence based on spread and fitness variance
        fitness_values = [a.fitness for a in swarm.agents]
        fitness_variance = np.var(fitness_values) if len(fitness_values) > 1 else 0

        max_spread = 50.0  # Maximum expected spread
        spread_convergence = 1.0 - min(swarm.spread / max_spread, 1.0)
        fitness_convergence = 1.0 - fitness_variance

        return (spread_convergence + fitness_convergence) / 2

    def _calculate_swarm_fitness(self, swarm: SwarmState) -> float:
        """Calculate overall swarm fitness."""
        if not swarm.agents:
            return 0.0

        # Average fitness of all agents
        return sum(a.fitness for a in swarm.agents) / len(swarm.agents)

    async def _create_agent_pool(self, count: int) -> None:
        """Create initial agent pool."""
        for _ in range(count):
            agent = SwarmAgent(
                role=random.choice(list(SwarmRole)),
                position=[random.uniform(-50, 50) for _ in range(3)],
                velocity=[random.uniform(-1, 1) for _ in range(3)]
            )
            self.agent_pool.append(agent)
            self.available_agents.add(agent.agent_id)

    async def _allocate_agents(self, count: int) -> List[SwarmAgent]:
        """Allocate agents from pool for task."""
        allocated = []

        # Use available agents first
        while len(allocated) < count and self.available_agents:
            agent_id = self.available_agents.pop()
            agent = next((a for a in self.agent_pool if a.agent_id == agent_id), None)
            if agent:
                agent.state = "ACTIVE"
                allocated.append(agent)

        # Create new agents if needed
        if len(allocated) < count:
            needed = count - len(allocated)
            for _ in range(needed):
                agent = SwarmAgent(
                    role=SwarmRole.WORKER,
                    position=[random.uniform(-50, 50) for _ in range(3)]
                )
                agent.state = "ACTIVE"
                self.agent_pool.append(agent)
                allocated.append(agent)

        return allocated

    async def _release_agents(self, agent_ids: List[str]) -> None:
        """Release agents back to pool."""
        for agent_id in agent_ids:
            agent = next((a for a in self.agent_pool if a.agent_id == agent_id), None)
            if agent:
                agent.state = "IDLE"
                agent.energy = 1.0
                agent.task_queue.clear()
                self.available_agents.add(agent_id)

    async def _swarm_coordinator(self) -> None:
        """Coordinate swarm activities."""
        while self._running:
            try:
                # Coordinate active swarms
                for swarm in self.swarms.values():
                    # Send coordination messages
                    for agent in swarm.agents:
                        if agent.neighbors:
                            message = Message(
                                sender_id=f"swarm-coordinator",
                                recipient_id=agent.agent_id,
                                message_type=MessageType.EVENT,
                                payload={
                                    "neighbors": agent.neighbors,
                                    "swarm_fitness": swarm.fitness,
                                    "swarm_convergence": swarm.convergence
                                }
                            )
                            await self.message_bus.publish(message)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Swarm coordinator error: {e}")

    async def _pheromone_manager(self) -> None:
        """Manage pheromone trails and evaporation."""
        while self._running:
            try:
                for swarm in self.swarms.values():
                    # Evaporate pheromones
                    for pheromone_type, pheromone_map in swarm.pheromone_map.items():
                        for position in list(pheromone_map.keys()):
                            pheromone_map[position] *= (1 - self.config.pheromone_evaporation_rate)
                            if pheromone_map[position] < 0.01:
                                del pheromone_map[position]

                    # Agent pheromone evaporation
                    for agent in swarm.agents:
                        agent.evaporate_pheromones(self.config.pheromone_evaporation_rate)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Pheromone manager error: {e}")

    async def _evolution_manager(self) -> None:
        """Manage agent evolution and adaptation."""
        while self._running:
            try:
                for swarm in self.swarms.values():
                    # Reproduction - high fitness agents
                    for agent in swarm.agents[:]:
                        if agent.fitness >= self.config.reproduction_threshold and len(swarm.agents) < self.config.max_agents:
                            # Clone with mutation
                            offspring = SwarmAgent(
                                role=agent.role,
                                position=[p + random.uniform(-1, 1) for p in agent.position],
                                velocity=[v + random.uniform(-0.1, 0.1) for v in agent.velocity]
                            )
                            swarm.agents.append(offspring)

                    # Death - low energy agents
                    swarm.agents = [
                        a for a in swarm.agents
                        if a.energy > self.config.death_threshold
                    ]

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Evolution manager error: {e}")

    async def get_swarm_status(self, swarm_id: str) -> Optional[SwarmState]:
        """Get swarm status."""
        return self.swarms.get(swarm_id)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get swarm orchestrator metrics."""
        return {
            "total_agents": len(self.agent_pool),
            "available_agents": len(self.available_agents),
            "active_swarms": len(self.swarms),
            "active_tasks": len(self.active_tasks),
            "global_best_fitness": self.global_best_fitness,
            "global_best_position": self.global_best_position
        }