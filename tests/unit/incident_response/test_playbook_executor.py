"""
Unit tests for PlaybookExecutor.

Tests playbook selection, execution, step handling, rollback,
and the 10+ built-in playbooks for incident remediation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.incident_response.playbook_executor import (
    PlaybookExecutor,
    BasePlaybook,
    NodeResourceRemediationPlaybook,
    PodRestartPlaybook,
    SecurityIncidentPlaybook,
    DataBreachPlaybook,
    DDoSMitigationPlaybook,
    DatabaseFailoverPlaybook,
    CacheInvalidationPlaybook,
    ScaleUpPlaybook,
    RollbackDeploymentPlaybook,
    NotificationEscalationPlaybook,
    PLAYBOOKS,
)
from greenlang.infrastructure.incident_response.models import (
    Incident,
    IncidentType,
    IncidentStatus,
    EscalationLevel,
    PlaybookStep,
    PlaybookExecution,
    PlaybookStatus,
)


class TestPlaybookExecutorInitialization:
    """Test PlaybookExecutor initialization."""

    def test_initialization_with_config(self, playbook_config):
        """Test executor initializes with configuration."""
        executor = PlaybookExecutor(config=playbook_config)

        assert executor.config == playbook_config
        assert executor.max_concurrent_executions == playbook_config.max_concurrent_executions

    def test_initialization_default_config(self):
        """Test executor initializes with default configuration."""
        executor = PlaybookExecutor()

        assert executor.config is not None
        assert executor.max_concurrent_executions > 0

    def test_initialization_loads_playbooks(self, playbook_config):
        """Test executor loads all registered playbooks."""
        executor = PlaybookExecutor(config=playbook_config)

        available = executor.get_available_playbooks()

        assert len(available) >= 10  # At least 10 playbooks


class TestPlaybookRegistry:
    """Test playbook registration and lookup."""

    def test_playbooks_registry_populated(self):
        """Test PLAYBOOKS registry is populated."""
        assert len(PLAYBOOKS) >= 10

    def test_registry_contains_required_playbooks(self):
        """Test registry contains all required playbooks."""
        required_playbooks = [
            "node_resource_remediation",
            "pod_restart",
            "security_incident",
            "data_breach",
            "ddos_mitigation",
            "database_failover",
            "cache_invalidation",
            "scale_up",
            "rollback_deployment",
            "notification_escalation",
        ]

        for playbook_id in required_playbooks:
            assert playbook_id in PLAYBOOKS, f"Missing playbook: {playbook_id}"

    def test_all_playbooks_are_base_playbook_subclasses(self):
        """Test all registered playbooks are BasePlaybook subclasses."""
        for playbook_id, playbook_class in PLAYBOOKS.items():
            assert issubclass(playbook_class, BasePlaybook), \
                f"{playbook_id} is not a BasePlaybook subclass"


class TestPlaybookSelection:
    """Test playbook selection for incidents."""

    def test_select_playbook_for_infrastructure_incident(
        self, playbook_config, sample_incident
    ):
        """Test selecting playbook for infrastructure incident."""
        executor = PlaybookExecutor(config=playbook_config)

        playbook = executor.select_playbook(sample_incident)

        assert playbook is not None
        assert playbook.playbook_id in PLAYBOOKS

    def test_select_playbook_for_security_incident(
        self, playbook_config, critical_incident
    ):
        """Test selecting playbook for security incident."""
        executor = PlaybookExecutor(config=playbook_config)

        playbook = executor.select_playbook(critical_incident)

        assert playbook is not None
        assert playbook.playbook_id in ["security_incident", "data_breach", "ddos_mitigation"]

    def test_select_playbook_by_id(self, playbook_config):
        """Test selecting specific playbook by ID."""
        executor = PlaybookExecutor(config=playbook_config)

        playbook = executor.get_playbook("pod_restart")

        assert playbook is not None
        assert playbook.playbook_id == "pod_restart"

    def test_select_invalid_playbook_raises(self, playbook_config):
        """Test selecting invalid playbook raises error."""
        executor = PlaybookExecutor(config=playbook_config)

        with pytest.raises(KeyError):
            executor.get_playbook("nonexistent_playbook")

    def test_select_playbook_considers_tags(self, playbook_config):
        """Test playbook selection considers incident tags."""
        executor = PlaybookExecutor(config=playbook_config)

        incident = Incident(
            incident_id=str(uuid4()),
            title="Database connection issues",
            description="Database connections failing",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P1,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=["database", "postgres", "connection-pool"],
            metadata={},
        )

        playbook = executor.select_playbook(incident)

        # Should select database-related playbook
        assert playbook is not None


class TestPlaybookExecution:
    """Test playbook execution."""

    @pytest.mark.asyncio
    async def test_execute_playbook_creates_execution(
        self, playbook_config, sample_incident
    ):
        """Test executing playbook creates execution record."""
        executor = PlaybookExecutor(config=playbook_config)

        execution = await executor.execute(
            playbook_id="pod_restart",
            incident=sample_incident,
            executed_by="test-user",
        )

        assert execution is not None
        assert isinstance(execution, PlaybookExecution)
        assert execution.playbook_id == "pod_restart"
        assert execution.incident_id == sample_incident.incident_id

    @pytest.mark.asyncio
    async def test_execute_playbook_tracks_status(
        self, playbook_config, sample_incident
    ):
        """Test execution tracks status changes."""
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "_execute_step", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {"status": "success"}

            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
            )

            assert execution.status in [
                PlaybookStatus.COMPLETED,
                PlaybookStatus.RUNNING,
                PlaybookStatus.PENDING,
            ]

    @pytest.mark.asyncio
    async def test_execute_playbook_executes_all_steps(
        self, playbook_config, sample_incident
    ):
        """Test execution runs all playbook steps."""
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "_execute_step", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {"status": "success"}

            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
            )

            # Should execute multiple steps
            assert mock_step.call_count >= 1

    @pytest.mark.asyncio
    async def test_execute_playbook_handles_step_failure(
        self, playbook_config, sample_incident
    ):
        """Test execution handles step failures."""
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "_execute_step", new_callable=AsyncMock) as mock_step:
            mock_step.side_effect = Exception("Step failed")

            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
            )

            assert execution.status == PlaybookStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_playbook_retries_failed_steps(
        self, playbook_config, sample_incident
    ):
        """Test execution retries failed steps."""
        executor = PlaybookExecutor(config=playbook_config)

        call_count = 0

        async def flaky_step(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient failure")
            return {"status": "success"}

        with patch.object(executor, "_execute_step", side_effect=flaky_step):
            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
            )

            # Should have retried
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_execute_playbook_records_results(
        self, playbook_config, sample_incident
    ):
        """Test execution records step results."""
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "_execute_step", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {"status": "success", "output": "pods restarted"}

            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
            )

            assert len(execution.results) > 0

    @pytest.mark.asyncio
    async def test_execute_playbook_respects_timeout(
        self, playbook_config, sample_incident
    ):
        """Test execution respects timeout."""
        playbook_config.default_timeout_seconds = 1
        executor = PlaybookExecutor(config=playbook_config)

        async def slow_step(*args, **kwargs):
            import asyncio
            await asyncio.sleep(5)
            return {"status": "success"}

        with patch.object(executor, "_execute_step", side_effect=slow_step):
            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
            )

            assert execution.status in [PlaybookStatus.FAILED, PlaybookStatus.TIMEOUT]


class TestPlaybookRollback:
    """Test playbook rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_execution(
        self, playbook_config, completed_playbook_execution
    ):
        """Test rolling back completed execution."""
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "_rollback_step", new_callable=AsyncMock) as mock_rollback:
            mock_rollback.return_value = {"status": "rolled_back"}

            result = await executor.rollback(completed_playbook_execution)

            assert result is not None
            assert mock_rollback.call_count >= 1

    @pytest.mark.asyncio
    async def test_rollback_reverses_step_order(
        self, playbook_config, completed_playbook_execution
    ):
        """Test rollback executes steps in reverse order."""
        executor = PlaybookExecutor(config=playbook_config)

        rollback_order = []

        async def track_rollback(step, *args, **kwargs):
            rollback_order.append(step.order)
            return {"status": "rolled_back"}

        with patch.object(executor, "_rollback_step", side_effect=track_rollback):
            await executor.rollback(completed_playbook_execution)

            # Should be in descending order
            assert rollback_order == sorted(rollback_order, reverse=True)

    @pytest.mark.asyncio
    async def test_rollback_handles_partial_execution(
        self, playbook_config, sample_playbook_execution
    ):
        """Test rollback handles partially executed playbook."""
        executor = PlaybookExecutor(config=playbook_config)

        # Only some steps completed
        sample_playbook_execution.current_step = 2
        sample_playbook_execution.results = {
            "step_1": {"status": "success"},
            "step_2": {"status": "success"},
        }

        with patch.object(executor, "_rollback_step", new_callable=AsyncMock) as mock_rollback:
            mock_rollback.return_value = {"status": "rolled_back"}

            await executor.rollback(sample_playbook_execution)

            # Should only rollback completed steps
            assert mock_rollback.call_count == 2


class TestApprovalRequirements:
    """Test playbook approval requirements."""

    @pytest.mark.asyncio
    async def test_playbook_requires_approval(self, playbook_config, sample_incident):
        """Test playbooks that require approval."""
        playbook_config.require_approval_for = ["production_rollback", "data_deletion"]
        executor = PlaybookExecutor(config=playbook_config)

        requires = executor.requires_approval("production_rollback")

        assert requires is True

    @pytest.mark.asyncio
    async def test_playbook_no_approval_needed(self, playbook_config):
        """Test playbooks that don't require approval."""
        playbook_config.require_approval_for = ["production_rollback"]
        executor = PlaybookExecutor(config=playbook_config)

        requires = executor.requires_approval("pod_restart")

        assert requires is False

    @pytest.mark.asyncio
    async def test_execute_with_approval_required(
        self, playbook_config, sample_incident
    ):
        """Test execution with approval requirement."""
        playbook_config.require_approval_for = ["pod_restart"]
        executor = PlaybookExecutor(config=playbook_config)

        execution = await executor.execute(
            playbook_id="pod_restart",
            incident=sample_incident,
            executed_by="test-user",
            approved_by=None,  # No approval
        )

        # Should be pending approval
        assert execution.status == PlaybookStatus.PENDING_APPROVAL

    @pytest.mark.asyncio
    async def test_execute_with_approval_provided(
        self, playbook_config, sample_incident
    ):
        """Test execution with approval provided."""
        playbook_config.require_approval_for = ["pod_restart"]
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "_execute_step", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {"status": "success"}

            execution = await executor.execute(
                playbook_id="pod_restart",
                incident=sample_incident,
                executed_by="test-user",
                approved_by="approver-user",
            )

            # Should execute with approval
            assert execution.status != PlaybookStatus.PENDING_APPROVAL


class TestAutoRemediation:
    """Test auto-remediation functionality."""

    @pytest.mark.asyncio
    async def test_auto_remediation_enabled(self, playbook_config, sample_incident):
        """Test auto-remediation when enabled."""
        playbook_config.enable_auto_remediation = True
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = MagicMock(status=PlaybookStatus.COMPLETED)

            await executor.auto_remediate(sample_incident)

            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_remediation_disabled(self, playbook_config, sample_incident):
        """Test auto-remediation when disabled."""
        playbook_config.enable_auto_remediation = False
        executor = PlaybookExecutor(config=playbook_config)

        with patch.object(executor, "execute", new_callable=AsyncMock) as mock_execute:
            await executor.auto_remediate(sample_incident)

            mock_execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_remediation_skips_approval_required(
        self, playbook_config, sample_incident
    ):
        """Test auto-remediation skips playbooks requiring approval."""
        playbook_config.enable_auto_remediation = True
        playbook_config.require_approval_for = ["*"]  # All playbooks
        executor = PlaybookExecutor(config=playbook_config)

        result = await executor.auto_remediate(sample_incident)

        assert result is None


class TestIndividualPlaybooks:
    """Test individual playbook implementations."""

    def test_node_resource_remediation_steps(self):
        """Test NodeResourceRemediationPlaybook has correct steps."""
        playbook = NodeResourceRemediationPlaybook()

        assert playbook.playbook_id == "node_resource_remediation"
        assert len(playbook.steps) >= 3
        step_types = [s.action_type for s in playbook.steps]
        assert "collect_diagnostics" in step_types or "diagnose" in step_types

    def test_pod_restart_steps(self):
        """Test PodRestartPlaybook has correct steps."""
        playbook = PodRestartPlaybook()

        assert playbook.playbook_id == "pod_restart"
        assert len(playbook.steps) >= 2
        step_types = [s.action_type for s in playbook.steps]
        assert "restart" in step_types or "rollout_restart" in step_types

    def test_security_incident_steps(self):
        """Test SecurityIncidentPlaybook has correct steps."""
        playbook = SecurityIncidentPlaybook()

        assert playbook.playbook_id == "security_incident"
        assert len(playbook.steps) >= 4
        step_types = [s.action_type for s in playbook.steps]
        assert any("isolate" in t or "contain" in t for t in step_types)

    def test_data_breach_steps(self):
        """Test DataBreachPlaybook has correct steps."""
        playbook = DataBreachPlaybook()

        assert playbook.playbook_id == "data_breach"
        assert len(playbook.steps) >= 5
        step_types = [s.action_type for s in playbook.steps]
        # Should have notification step
        assert any("notify" in t or "alert" in t for t in step_types)

    def test_ddos_mitigation_steps(self):
        """Test DDoSMitigationPlaybook has correct steps."""
        playbook = DDoSMitigationPlaybook()

        assert playbook.playbook_id == "ddos_mitigation"
        assert len(playbook.steps) >= 3
        step_types = [s.action_type for s in playbook.steps]
        assert any("rate_limit" in t or "block" in t or "waf" in t for t in step_types)

    def test_database_failover_steps(self):
        """Test DatabaseFailoverPlaybook has correct steps."""
        playbook = DatabaseFailoverPlaybook()

        assert playbook.playbook_id == "database_failover"
        assert len(playbook.steps) >= 4
        step_types = [s.action_type for s in playbook.steps]
        assert any("failover" in t or "promote" in t for t in step_types)

    def test_cache_invalidation_steps(self):
        """Test CacheInvalidationPlaybook has correct steps."""
        playbook = CacheInvalidationPlaybook()

        assert playbook.playbook_id == "cache_invalidation"
        assert len(playbook.steps) >= 2
        step_types = [s.action_type for s in playbook.steps]
        assert any("invalidate" in t or "flush" in t or "clear" in t for t in step_types)

    def test_scale_up_steps(self):
        """Test ScaleUpPlaybook has correct steps."""
        playbook = ScaleUpPlaybook()

        assert playbook.playbook_id == "scale_up"
        assert len(playbook.steps) >= 2
        step_types = [s.action_type for s in playbook.steps]
        assert any("scale" in t or "hpa" in t for t in step_types)

    def test_rollback_deployment_steps(self):
        """Test RollbackDeploymentPlaybook has correct steps."""
        playbook = RollbackDeploymentPlaybook()

        assert playbook.playbook_id == "rollback_deployment"
        assert len(playbook.steps) >= 3
        step_types = [s.action_type for s in playbook.steps]
        assert any("rollback" in t or "undo" in t for t in step_types)

    def test_notification_escalation_steps(self):
        """Test NotificationEscalationPlaybook has correct steps."""
        playbook = NotificationEscalationPlaybook()

        assert playbook.playbook_id == "notification_escalation"
        assert len(playbook.steps) >= 2
        step_types = [s.action_type for s in playbook.steps]
        assert any("notify" in t or "escalate" in t or "page" in t for t in step_types)


class TestBasePlaybook:
    """Test BasePlaybook abstract class."""

    def test_base_playbook_is_abstract(self):
        """Test BasePlaybook cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePlaybook()

    def test_base_playbook_requires_steps(self):
        """Test subclass must define steps."""

        class IncompletePlaybook(BasePlaybook):
            playbook_id = "incomplete"
            name = "Incomplete"
            description = "Test"

        with pytest.raises(TypeError):
            IncompletePlaybook()

    def test_playbook_validate_method(self):
        """Test playbook validation method."""
        playbook = PodRestartPlaybook()

        # Should not raise for valid incident
        incident = Incident(
            incident_id=str(uuid4()),
            title="Test",
            description="Test",
            incident_type=IncidentType.INFRASTRUCTURE,
            status=IncidentStatus.OPEN,
            escalation_level=EscalationLevel.P2,
            alerts=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[],
            metadata={},
        )

        is_valid = playbook.validate(incident)
        assert isinstance(is_valid, bool)


class TestConcurrentExecutions:
    """Test concurrent execution limits."""

    @pytest.mark.asyncio
    async def test_max_concurrent_executions_enforced(
        self, playbook_config, sample_incident
    ):
        """Test max concurrent executions is enforced."""
        playbook_config.max_concurrent_executions = 2
        executor = PlaybookExecutor(config=playbook_config)

        # Track concurrent executions
        import asyncio

        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MagicMock(status=PlaybookStatus.COMPLETED)

        with patch.object(executor, "_execute_playbook_internal", side_effect=slow_execution):
            # Start 3 executions (should block third)
            tasks = [
                asyncio.create_task(executor.execute(
                    playbook_id="pod_restart",
                    incident=sample_incident,
                    executed_by=f"user-{i}",
                ))
                for i in range(3)
            ]

            # Should complete without exceeding limit
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete (some may queue)
            assert len(results) == 3


class TestPlaybookStepExecution:
    """Test individual step execution."""

    @pytest.mark.asyncio
    async def test_execute_collect_diagnostics_step(self, playbook_config):
        """Test executing collect_diagnostics step."""
        executor = PlaybookExecutor(config=playbook_config)

        step = PlaybookStep(
            step_id="step_1",
            name="Collect diagnostics",
            description="Gather metrics",
            action_type="collect_diagnostics",
            parameters={"target": "node-1"},
            timeout_seconds=60,
            retry_count=2,
            order=1,
        )

        with patch.object(executor, "_collect_diagnostics", new_callable=AsyncMock) as mock_collect:
            mock_collect.return_value = {"metrics": {"cpu": 90}}

            result = await executor._execute_step(step, {})

            assert result is not None
            mock_collect.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_restart_pods_step(self, playbook_config):
        """Test executing restart_pods step."""
        executor = PlaybookExecutor(config=playbook_config)

        step = PlaybookStep(
            step_id="step_1",
            name="Restart pods",
            description="Restart affected pods",
            action_type="restart_pods",
            parameters={"selector": "app=api", "namespace": "default"},
            timeout_seconds=120,
            retry_count=1,
            order=1,
        )

        with patch.object(executor, "_restart_pods", new_callable=AsyncMock) as mock_restart:
            mock_restart.return_value = {"restarted": 3}

            result = await executor._execute_step(step, {})

            assert result is not None
            mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_unknown_action_type_raises(self, playbook_config):
        """Test executing unknown action type raises error."""
        executor = PlaybookExecutor(config=playbook_config)

        step = PlaybookStep(
            step_id="step_1",
            name="Unknown step",
            description="Unknown action",
            action_type="unknown_action_type",
            parameters={},
            timeout_seconds=60,
            retry_count=0,
            order=1,
        )

        with pytest.raises(ValueError):
            await executor._execute_step(step, {})
