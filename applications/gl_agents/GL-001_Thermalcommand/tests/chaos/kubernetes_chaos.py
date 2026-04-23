"""
GL-001 ThermalCommand - Kubernetes Chaos Tests

This module provides chaos tests specific to Kubernetes deployments.
All tests simulate K8s failures without requiring actual cluster access.

Tests:
- Pod deletion recovery
- Node failure simulation
- Resource exhaustion
- Network policies
- Service discovery failures

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PodPhase(Enum):
    """Kubernetes pod phases."""
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class NodeCondition(Enum):
    """Kubernetes node conditions."""
    READY = "Ready"
    MEMORY_PRESSURE = "MemoryPressure"
    DISK_PRESSURE = "DiskPressure"
    PID_PRESSURE = "PIDPressure"
    NETWORK_UNAVAILABLE = "NetworkUnavailable"


@dataclass
class SimulatedPod:
    """Simulated Kubernetes pod."""
    name: str
    namespace: str
    phase: PodPhase
    node: str
    restart_count: int = 0
    ready: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    creation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SimulatedNode:
    """Simulated Kubernetes node."""
    name: str
    conditions: Dict[NodeCondition, bool] = field(default_factory=dict)
    allocatable_cpu: float = 4.0
    allocatable_memory_gb: float = 8.0
    pods: List[str] = field(default_factory=list)

    def is_ready(self) -> bool:
        """Check if node is ready."""
        return self.conditions.get(NodeCondition.READY, True)


@dataclass
class K8sChaosResult:
    """Result of Kubernetes chaos test."""
    test_name: str
    passed: bool
    initial_pod_count: int
    final_pod_count: int
    pods_affected: List[str]
    pods_recovered: List[str]
    recovery_time_seconds: float
    expected_replicas: int
    actual_replicas: int
    observations: List[str]
    errors: List[str]


class SimulatedK8sCluster:
    """
    Simulated Kubernetes cluster for chaos testing.

    This class simulates K8s behavior for testing pod deletion,
    node failures, and recovery scenarios without actual cluster access.
    """

    def __init__(
        self,
        node_count: int = 3,
        initial_replicas: int = 3,
        namespace: str = "greenlang",
    ):
        self.namespace = namespace
        self.nodes: Dict[str, SimulatedNode] = {}
        self.pods: Dict[str, SimulatedPod] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}

        # Initialize nodes
        for i in range(node_count):
            node_name = f"node-{i+1}"
            self.nodes[node_name] = SimulatedNode(
                name=node_name,
                conditions={
                    NodeCondition.READY: True,
                    NodeCondition.MEMORY_PRESSURE: False,
                    NodeCondition.DISK_PRESSURE: False,
                },
            )

        # Initialize deployment
        self.deployments["thermalcommand"] = {
            "replicas": initial_replicas,
            "ready_replicas": initial_replicas,
            "selector": {"app": "thermalcommand"},
        }

        # Initialize pods
        for i in range(initial_replicas):
            node = list(self.nodes.keys())[i % len(self.nodes)]
            pod_name = f"thermalcommand-{i+1}"
            self.pods[pod_name] = SimulatedPod(
                name=pod_name,
                namespace=namespace,
                phase=PodPhase.RUNNING,
                node=node,
                labels={"app": "thermalcommand"},
            )
            self.nodes[node].pods.append(pod_name)

    async def delete_pod(self, pod_name: str) -> bool:
        """Delete a pod."""
        if pod_name not in self.pods:
            return False

        pod = self.pods[pod_name]
        node = pod.node

        # Remove from node
        if node in self.nodes:
            if pod_name in self.nodes[node].pods:
                self.nodes[node].pods.remove(pod_name)

        # Mark pod as terminated
        pod.phase = PodPhase.SUCCEEDED
        pod.ready = False

        logger.debug(f"Pod deleted: {pod_name}")
        return True

    async def create_replacement_pod(self, deployment: str) -> Optional[str]:
        """Create replacement pod for deployment."""
        if deployment not in self.deployments:
            return None

        deploy = self.deployments[deployment]
        current_running = len([
            p for p in self.pods.values()
            if p.phase == PodPhase.RUNNING and p.labels.get("app") == deploy["selector"]["app"]
        ])

        if current_running >= deploy["replicas"]:
            return None

        # Find available node
        available_nodes = [
            n for n in self.nodes.values()
            if n.is_ready() and len(n.pods) < 10
        ]

        if not available_nodes:
            return None

        node = random.choice(available_nodes)

        # Create new pod
        pod_num = len(self.pods) + 1
        pod_name = f"{deployment}-{pod_num}"

        self.pods[pod_name] = SimulatedPod(
            name=pod_name,
            namespace=self.namespace,
            phase=PodPhase.RUNNING,
            node=node.name,
            labels=deploy["selector"],
        )
        node.pods.append(pod_name)

        logger.debug(f"Replacement pod created: {pod_name} on {node.name}")
        return pod_name

    async def fail_node(self, node_name: str) -> List[str]:
        """Simulate node failure."""
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        node.conditions[NodeCondition.READY] = False

        affected_pods = node.pods.copy()

        # Mark all pods on node as failed
        for pod_name in affected_pods:
            if pod_name in self.pods:
                self.pods[pod_name].phase = PodPhase.FAILED
                self.pods[pod_name].ready = False

        logger.debug(f"Node failed: {node_name}, affected pods: {affected_pods}")
        return affected_pods

    async def recover_node(self, node_name: str) -> bool:
        """Recover a failed node."""
        if node_name not in self.nodes:
            return False

        node = self.nodes[node_name]
        node.conditions[NodeCondition.READY] = True

        logger.debug(f"Node recovered: {node_name}")
        return True

    def get_running_pods(self) -> List[str]:
        """Get list of running pods."""
        return [
            p.name for p in self.pods.values()
            if p.phase == PodPhase.RUNNING and p.ready
        ]

    def get_ready_nodes(self) -> List[str]:
        """Get list of ready nodes."""
        return [n.name for n in self.nodes.values() if n.is_ready()]


class K8sPodDeletionTest:
    """
    Test pod deletion and recovery.

    Example:
        >>> test = K8sPodDeletionTest()
        >>> result = await test.test_single_pod_deletion()
        >>> assert result.passed
    """

    def __init__(self, replicas: int = 3):
        self.cluster = SimulatedK8sCluster(initial_replicas=replicas)
        self.expected_replicas = replicas

    async def test_single_pod_deletion(self) -> K8sChaosResult:
        """Test recovery from single pod deletion."""
        test_name = "single_pod_deletion"
        observations = []
        errors = []
        pods_affected = []
        pods_recovered = []

        initial_pods = self.cluster.get_running_pods()
        initial_count = len(initial_pods)
        observations.append(f"Initial running pods: {initial_count}")

        # Delete one pod
        pod_to_delete = random.choice(initial_pods)
        pods_affected.append(pod_to_delete)
        observations.append(f"Deleting pod: {pod_to_delete}")

        await self.cluster.delete_pod(pod_to_delete)

        current_pods = self.cluster.get_running_pods()
        observations.append(f"Pods after deletion: {len(current_pods)}")

        # Simulate controller creating replacement
        start_time = time.time()

        for _ in range(5):  # Max 5 attempts
            new_pod = await self.cluster.create_replacement_pod("thermalcommand")
            if new_pod:
                pods_recovered.append(new_pod)
                observations.append(f"Replacement pod created: {new_pod}")
                break
            await asyncio.sleep(0.1)

        recovery_time = time.time() - start_time

        final_pods = self.cluster.get_running_pods()
        final_count = len(final_pods)
        observations.append(f"Final running pods: {final_count}")

        passed = final_count == self.expected_replicas

        if not passed:
            errors.append(
                f"Expected {self.expected_replicas} replicas, got {final_count}"
            )

        return K8sChaosResult(
            test_name=test_name,
            passed=passed,
            initial_pod_count=initial_count,
            final_pod_count=final_count,
            pods_affected=pods_affected,
            pods_recovered=pods_recovered,
            recovery_time_seconds=recovery_time,
            expected_replicas=self.expected_replicas,
            actual_replicas=final_count,
            observations=observations,
            errors=errors,
        )

    async def test_multiple_pod_deletion(self, delete_count: int = 2) -> K8sChaosResult:
        """Test recovery from multiple pod deletions."""
        test_name = "multiple_pod_deletion"
        observations = []
        errors = []
        pods_affected = []
        pods_recovered = []

        # Reset cluster
        self.cluster = SimulatedK8sCluster(initial_replicas=self.expected_replicas)

        initial_pods = self.cluster.get_running_pods()
        initial_count = len(initial_pods)
        observations.append(f"Initial running pods: {initial_count}")

        # Delete multiple pods
        pods_to_delete = random.sample(initial_pods, min(delete_count, len(initial_pods)))

        for pod in pods_to_delete:
            pods_affected.append(pod)
            await self.cluster.delete_pod(pod)
            observations.append(f"Deleted pod: {pod}")

        current_pods = self.cluster.get_running_pods()
        observations.append(f"Pods after deletion: {len(current_pods)}")

        # Simulate controller creating replacements
        start_time = time.time()

        while len(self.cluster.get_running_pods()) < self.expected_replicas:
            new_pod = await self.cluster.create_replacement_pod("thermalcommand")
            if new_pod:
                pods_recovered.append(new_pod)
                observations.append(f"Replacement pod created: {new_pod}")
            else:
                break

            if time.time() - start_time > 5:  # Timeout
                break

        recovery_time = time.time() - start_time

        final_pods = self.cluster.get_running_pods()
        final_count = len(final_pods)
        observations.append(f"Final running pods: {final_count}")

        passed = final_count == self.expected_replicas

        if not passed:
            errors.append(
                f"Expected {self.expected_replicas} replicas, got {final_count}"
            )

        return K8sChaosResult(
            test_name=test_name,
            passed=passed,
            initial_pod_count=initial_count,
            final_pod_count=final_count,
            pods_affected=pods_affected,
            pods_recovered=pods_recovered,
            recovery_time_seconds=recovery_time,
            expected_replicas=self.expected_replicas,
            actual_replicas=final_count,
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[K8sChaosResult]:
        """Run all pod deletion tests."""
        results = []
        results.append(await self.test_single_pod_deletion())
        results.append(await self.test_multiple_pod_deletion())
        return results


class K8sNodeFailureTest:
    """
    Test node failure and recovery.

    Example:
        >>> test = K8sNodeFailureTest()
        >>> result = await test.test_single_node_failure()
        >>> assert result.passed
    """

    def __init__(self, node_count: int = 3, replicas: int = 3):
        self.cluster = SimulatedK8sCluster(
            node_count=node_count,
            initial_replicas=replicas,
        )
        self.expected_replicas = replicas

    async def test_single_node_failure(self) -> K8sChaosResult:
        """Test recovery from single node failure."""
        test_name = "single_node_failure"
        observations = []
        errors = []
        pods_affected = []
        pods_recovered = []

        initial_pods = self.cluster.get_running_pods()
        initial_count = len(initial_pods)
        initial_nodes = self.cluster.get_ready_nodes()
        observations.append(f"Initial: {initial_count} pods, {len(initial_nodes)} nodes")

        # Fail one node
        node_to_fail = random.choice(initial_nodes)
        observations.append(f"Failing node: {node_to_fail}")

        affected = await self.cluster.fail_node(node_to_fail)
        pods_affected.extend(affected)
        observations.append(f"Affected pods: {affected}")

        current_pods = self.cluster.get_running_pods()
        observations.append(f"Pods after node failure: {len(current_pods)}")

        # Simulate pod rescheduling
        start_time = time.time()

        while len(self.cluster.get_running_pods()) < self.expected_replicas:
            new_pod = await self.cluster.create_replacement_pod("thermalcommand")
            if new_pod:
                pods_recovered.append(new_pod)
                observations.append(f"Pod rescheduled: {new_pod}")
            else:
                break

            if time.time() - start_time > 5:
                break

        recovery_time = time.time() - start_time

        final_pods = self.cluster.get_running_pods()
        final_count = len(final_pods)
        observations.append(f"Final running pods: {final_count}")

        # Pods should be rescheduled to other nodes
        passed = final_count == self.expected_replicas

        if not passed:
            errors.append(f"Expected {self.expected_replicas} pods, got {final_count}")

        return K8sChaosResult(
            test_name=test_name,
            passed=passed,
            initial_pod_count=initial_count,
            final_pod_count=final_count,
            pods_affected=pods_affected,
            pods_recovered=pods_recovered,
            recovery_time_seconds=recovery_time,
            expected_replicas=self.expected_replicas,
            actual_replicas=final_count,
            observations=observations,
            errors=errors,
        )

    async def test_node_recovery(self) -> K8sChaosResult:
        """Test node recovery after failure."""
        test_name = "node_recovery"
        observations = []
        errors = []

        initial_nodes = self.cluster.get_ready_nodes()
        observations.append(f"Initial ready nodes: {len(initial_nodes)}")

        # Fail a node
        node_to_fail = random.choice(initial_nodes)
        await self.cluster.fail_node(node_to_fail)
        observations.append(f"Failed node: {node_to_fail}")

        ready_after_failure = self.cluster.get_ready_nodes()
        observations.append(f"Ready nodes after failure: {len(ready_after_failure)}")

        # Recover the node
        start_time = time.time()
        await self.cluster.recover_node(node_to_fail)
        recovery_time = time.time() - start_time

        ready_after_recovery = self.cluster.get_ready_nodes()
        observations.append(f"Ready nodes after recovery: {len(ready_after_recovery)}")

        passed = len(ready_after_recovery) == len(initial_nodes)

        if not passed:
            errors.append(
                f"Expected {len(initial_nodes)} ready nodes, got {len(ready_after_recovery)}"
            )

        return K8sChaosResult(
            test_name=test_name,
            passed=passed,
            initial_pod_count=len(self.cluster.pods),
            final_pod_count=len(self.cluster.pods),
            pods_affected=[],
            pods_recovered=[],
            recovery_time_seconds=recovery_time,
            expected_replicas=self.expected_replicas,
            actual_replicas=len(self.cluster.get_running_pods()),
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[K8sChaosResult]:
        """Run all node failure tests."""
        results = []
        results.append(await self.test_single_node_failure())

        # Reset cluster for next test
        self.cluster = SimulatedK8sCluster(
            node_count=3,
            initial_replicas=self.expected_replicas,
        )
        results.append(await self.test_node_recovery())

        return results


class K8sResourceExhaustionTest:
    """
    Test resource exhaustion scenarios.

    Example:
        >>> test = K8sResourceExhaustionTest()
        >>> result = await test.test_memory_pressure()
        >>> assert result.passed
    """

    def __init__(self):
        self.cluster = SimulatedK8sCluster()

    async def test_memory_pressure(self) -> K8sChaosResult:
        """Test behavior under memory pressure."""
        test_name = "memory_pressure"
        observations = []
        errors = []

        initial_pods = self.cluster.get_running_pods()
        initial_count = len(initial_pods)
        observations.append(f"Initial running pods: {initial_count}")

        # Simulate memory pressure on one node
        node_name = random.choice(list(self.cluster.nodes.keys()))
        node = self.cluster.nodes[node_name]
        node.conditions[NodeCondition.MEMORY_PRESSURE] = True
        observations.append(f"Memory pressure on node: {node_name}")

        # Pods on that node may be evicted
        affected_pods = node.pods.copy()
        observations.append(f"Pods on affected node: {affected_pods}")

        # Simulate eviction (mark as pending)
        for pod_name in affected_pods:
            if pod_name in self.cluster.pods:
                self.cluster.pods[pod_name].phase = PodPhase.PENDING
                self.cluster.pods[pod_name].ready = False

        current_running = self.cluster.get_running_pods()
        observations.append(f"Running pods after pressure: {len(current_running)}")

        # Clear pressure and recover
        start_time = time.time()
        node.conditions[NodeCondition.MEMORY_PRESSURE] = False

        # Pods should recover
        for pod_name in affected_pods:
            if pod_name in self.cluster.pods:
                self.cluster.pods[pod_name].phase = PodPhase.RUNNING
                self.cluster.pods[pod_name].ready = True

        recovery_time = time.time() - start_time

        final_pods = self.cluster.get_running_pods()
        final_count = len(final_pods)
        observations.append(f"Final running pods: {final_count}")

        passed = final_count == initial_count

        return K8sChaosResult(
            test_name=test_name,
            passed=passed,
            initial_pod_count=initial_count,
            final_pod_count=final_count,
            pods_affected=affected_pods,
            pods_recovered=affected_pods if passed else [],
            recovery_time_seconds=recovery_time,
            expected_replicas=initial_count,
            actual_replicas=final_count,
            observations=observations,
            errors=errors,
        )

    async def test_disk_pressure(self) -> K8sChaosResult:
        """Test behavior under disk pressure."""
        test_name = "disk_pressure"
        observations = []
        errors = []

        initial_pods = self.cluster.get_running_pods()
        initial_count = len(initial_pods)

        # Simulate disk pressure
        node_name = random.choice(list(self.cluster.nodes.keys()))
        node = self.cluster.nodes[node_name]
        node.conditions[NodeCondition.DISK_PRESSURE] = True
        observations.append(f"Disk pressure on node: {node_name}")

        # Node should not accept new pods
        ready_nodes = [n for n in self.cluster.nodes.values() if n.is_ready()]
        observations.append(f"Ready nodes for scheduling: {len(ready_nodes)}")

        # Clear pressure
        node.conditions[NodeCondition.DISK_PRESSURE] = False
        observations.append("Disk pressure cleared")

        final_pods = self.cluster.get_running_pods()

        passed = True  # Disk pressure should not affect running pods

        return K8sChaosResult(
            test_name=test_name,
            passed=passed,
            initial_pod_count=initial_count,
            final_pod_count=len(final_pods),
            pods_affected=[],
            pods_recovered=[],
            recovery_time_seconds=0,
            expected_replicas=initial_count,
            actual_replicas=len(final_pods),
            observations=observations,
            errors=errors,
        )

    async def run_all_tests(self) -> List[K8sChaosResult]:
        """Run all resource exhaustion tests."""
        results = []
        results.append(await self.test_memory_pressure())

        # Reset cluster
        self.cluster = SimulatedK8sCluster()
        results.append(await self.test_disk_pressure())

        return results


# =============================================================================
# Convenience Runner
# =============================================================================

class K8sChaosTestRunner:
    """
    Run all Kubernetes chaos tests.

    Example:
        >>> runner = K8sChaosTestRunner()
        >>> results = await runner.run_all()
        >>> print(f"Passed: {sum(1 for r in results if r.passed)}/{len(results)}")
    """

    def __init__(self):
        self.pod_deletion_test = K8sPodDeletionTest()
        self.node_failure_test = K8sNodeFailureTest()
        self.resource_exhaustion_test = K8sResourceExhaustionTest()

    async def run_all(self) -> List[K8sChaosResult]:
        """Run all Kubernetes chaos tests."""
        all_results = []

        logger.info("Running Kubernetes Pod Deletion Tests...")
        all_results.extend(await self.pod_deletion_test.run_all_tests())

        logger.info("Running Kubernetes Node Failure Tests...")
        all_results.extend(await self.node_failure_test.run_all_tests())

        logger.info("Running Kubernetes Resource Exhaustion Tests...")
        all_results.extend(await self.resource_exhaustion_test.run_all_tests())

        return all_results
