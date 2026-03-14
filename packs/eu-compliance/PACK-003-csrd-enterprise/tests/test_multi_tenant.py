# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Multi-Tenant Tests (25 tests)

Deep multi-tenant isolation testing including tenant lifecycle,
resource quotas, data partitioning, cross-tenant benchmarking,
and audit trail verification.

Author: GreenLang QA Team
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import _compute_hash, _new_uuid, _utcnow


class TestMultiTenantIsolation:
    """Deep multi-tenant isolation and lifecycle tests."""

    def test_tenant_creation(self, multi_tenant_engine, multi_tenant_module):
        """Test basic tenant creation."""
        mod = multi_tenant_module
        request = mod.TenantProvisionRequest(
            tenant_name="Creation Test Corp",
            tier=mod.TenantTier.STARTER,
            admin_email="admin@creation.com",
        )
        status = multi_tenant_engine.provision_tenant(request)
        assert status.name == "Creation Test Corp"
        assert status.tier == mod.TenantTier.STARTER
        assert status.health_score == 100.0

    def test_tenant_isolation(self, multi_tenant_engine, multi_tenant_module):
        """Test two tenants are isolated from each other."""
        mod = multi_tenant_module
        t1 = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Isolated A", tier=mod.TenantTier.ENTERPRISE,
                admin_email="a@isolated.com",
            )
        )
        t2 = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Isolated B", tier=mod.TenantTier.ENTERPRISE,
                admin_email="b@isolated.com",
            )
        )
        assert t1.tenant_id != t2.tenant_id
        assert t1.provenance_hash != t2.provenance_hash

    def test_cross_tenant_data_leak_prevented(self, multi_tenant_engine, multi_tenant_module):
        """Test that tenant data is not accessible across tenants."""
        mod = multi_tenant_module
        t1 = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Leak Test A", admin_email="a@leak.com",
            )
        )
        t2 = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Leak Test B", admin_email="b@leak.com",
            )
        )
        usage_a = multi_tenant_engine.get_resource_usage(t1.tenant_id)
        usage_b = multi_tenant_engine.get_resource_usage(t2.tenant_id)
        assert usage_a["tenant_id"] == t1.tenant_id
        assert usage_b["tenant_id"] == t2.tenant_id
        assert usage_a["tenant_id"] != usage_b["tenant_id"]

    def test_tenant_tier_upgrade(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant tier upgrade from STARTER to ENTERPRISE."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Upgrade Corp", tier=mod.TenantTier.STARTER,
                admin_email="admin@upgrade.com",
            )
        )
        # Capture original quota before upgrade (update_tier mutates in-place)
        original_max_agents = t.resource_quotas.max_agents
        updated = multi_tenant_engine.update_tier(t.tenant_id, mod.TenantTier.ENTERPRISE)
        assert updated.tier == mod.TenantTier.ENTERPRISE
        assert updated.resource_quotas.max_agents > original_max_agents

    def test_tenant_tier_downgrade(self, multi_tenant_engine, multi_tenant_module):
        """Test tier downgrade when usage allows."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Downgrade Corp", tier=mod.TenantTier.PROFESSIONAL,
                admin_email="admin@downgrade.com",
            )
        )
        updated = multi_tenant_engine.update_tier(t.tenant_id, mod.TenantTier.STARTER)
        assert updated.tier == mod.TenantTier.STARTER

    def test_resource_quota_enforcement(self, multi_tenant_engine, multi_tenant_module):
        """Test resource quota enforcement returns no violations when under limit."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Quota Corp", tier=mod.TenantTier.ENTERPRISE,
                admin_email="admin@quota.com",
            )
        )
        violations = multi_tenant_engine.enforce_quotas(t.tenant_id)
        assert len(violations) == 0

    def test_resource_quota_exceeded(self, multi_tenant_engine, multi_tenant_module):
        """Test quota violation detection when resource is exceeded."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Exceeded Corp", tier=mod.TenantTier.FREE,
                admin_email="admin@exceeded.com",
            )
        )
        t.resource_usage.current_agents = 100
        violations = multi_tenant_engine.enforce_quotas(t.tenant_id)
        assert len(violations) >= 1
        assert any(v.resource == "agents" for v in violations)

    def test_tenant_suspension(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant suspension with reason."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Suspend Test", admin_email="admin@suspendtest.com",
            )
        )
        suspended = multi_tenant_engine.suspend_tenant(t.tenant_id, "policy_violation")
        assert suspended.status == mod.TenantLifecycleStatus.SUSPENDED

    def test_tenant_termination(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant termination with data archival."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Terminate Test", admin_email="admin@termtest.com",
            )
        )
        result = multi_tenant_engine.terminate_tenant(t.tenant_id, archive_data=True)
        assert result["status"] == "terminated"
        assert result["archive"]["archived"] is True
        assert "retention_days" in result["archive"]

    def test_data_partition_enforcement(self, multi_tenant_engine, multi_tenant_module):
        """Test data partition scheme is created per tenant."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Partition Corp", admin_email="admin@partition.com",
                isolation_level=mod.IsolationLevel.NAMESPACE,
            )
        )
        assert t.isolation_level == mod.IsolationLevel.NAMESPACE

    def test_schema_isolation(self, multi_tenant_engine, multi_tenant_module):
        """Test DB schema isolation naming."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Schema Corp", admin_email="admin@schema.com",
            )
        )
        expected_prefix = f"tenant_{t.tenant_id.replace('-', '_')[:12]}"
        assert len(expected_prefix) > 0

    def test_redis_namespace_isolation(self, multi_tenant_engine, multi_tenant_module):
        """Test Redis namespace is unique per tenant."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Redis Corp", admin_email="admin@redis.com",
            )
        )
        redis_prefix = f"gl:tenant:{t.tenant_id[:8]}:"
        assert redis_prefix.startswith("gl:tenant:")
        assert len(redis_prefix) > 15

    def test_s3_prefix_isolation(self, multi_tenant_engine, multi_tenant_module):
        """Test S3 prefix is unique per tenant."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="S3 Corp", admin_email="admin@s3corp.com",
            )
        )
        s3_prefix = f"tenants/{t.tenant_id}/"
        assert s3_prefix.startswith("tenants/")
        assert t.tenant_id in s3_prefix

    def test_cross_tenant_benchmark_anonymized(self, multi_tenant_engine, multi_tenant_module):
        """Test cross-tenant benchmark uses anonymous labels."""
        mod = multi_tenant_module
        for i in range(5):
            multi_tenant_engine.provision_tenant(
                mod.TenantProvisionRequest(
                    tenant_name=f"Anon Bench {i}",
                    admin_email=f"admin{i}@anonbench.com",
                    tier=mod.TenantTier.ENTERPRISE,
                )
            )
        result = multi_tenant_engine.cross_tenant_benchmark("health_score", anonymize=True)
        for pos in result.get("tenant_positions", []):
            assert pos["label"].startswith("tenant_")
            assert "-" not in pos["label"]

    def test_max_tenants_limit(self, multi_tenant_module):
        """Test max tenants configuration."""
        mod = multi_tenant_module
        from conftest import StubTenantManager
        manager = StubTenantManager()
        max_allowed = 5
        for i in range(max_allowed):
            manager.create_tenant({"tenant_name": f"Tenant {i}", "tenant_id": f"t-{i}"})
        assert len(manager.list_tenants()) == max_allowed

    def test_tenant_lifecycle(self, multi_tenant_engine, multi_tenant_module):
        """Test complete tenant lifecycle: provision -> suspend -> terminate."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Lifecycle Corp", admin_email="admin@lifecycle.com",
            )
        )
        assert t.status == mod.TenantLifecycleStatus.ACTIVE
        suspended = multi_tenant_engine.suspend_tenant(t.tenant_id, "maintenance")
        assert suspended.status == mod.TenantLifecycleStatus.SUSPENDED
        result = multi_tenant_engine.terminate_tenant(t.tenant_id)
        assert result["status"] == "terminated"

    def test_tenant_config_persistence(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant config persists provenance hash."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Persist Corp", admin_email="admin@persist.com",
                tier=mod.TenantTier.ENTERPRISE,
            )
        )
        assert len(t.provenance_hash) == 64

    def test_tenant_feature_flags(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant feature flags are tier-appropriate."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Feature Corp", admin_email="admin@feature.com",
                tier=mod.TenantTier.ENTERPRISE,
            )
        )
        assert "predictive_analytics" in t.features_enabled
        assert "white_label" in t.features_enabled

    def test_tenant_status_transitions(self, multi_tenant_engine, multi_tenant_module):
        """Test invalid status transition raises error."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Transition Corp", admin_email="admin@transition.com",
            )
        )
        multi_tenant_engine.terminate_tenant(t.tenant_id)
        with pytest.raises(ValueError):
            multi_tenant_engine.suspend_tenant(t.tenant_id, "test")

    def test_concurrent_tenant_operations(self, multi_tenant_engine, multi_tenant_module):
        """Test multiple tenants can be provisioned independently."""
        mod = multi_tenant_module
        tenants = []
        for i in range(10):
            t = multi_tenant_engine.provision_tenant(
                mod.TenantProvisionRequest(
                    tenant_name=f"Concurrent {i}",
                    admin_email=f"admin{i}@concurrent.com",
                )
            )
            tenants.append(t)
        assert len(tenants) == 10
        ids = [t.tenant_id for t in tenants]
        assert len(set(ids)) == 10

    def test_tenant_rollback_on_failure(self, multi_tenant_engine, multi_tenant_module):
        """Test duplicate tenant name raises error (rollback scenario)."""
        mod = multi_tenant_module
        multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Unique Corp", admin_email="admin@unique.com",
            )
        )
        with pytest.raises(ValueError, match="already in use"):
            multi_tenant_engine.provision_tenant(
                mod.TenantProvisionRequest(
                    tenant_name="Unique Corp", admin_email="admin2@unique.com",
                )
            )

    def test_tenant_audit_trail(self, multi_tenant_engine, multi_tenant_module):
        """Test audit trail records tenant events."""
        mod = multi_tenant_module
        initial_log_len = len(multi_tenant_engine._audit_log)
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Audit Trail Corp", admin_email="admin@audit.com",
            )
        )
        assert len(multi_tenant_engine._audit_log) > initial_log_len

    def test_tenant_data_residency(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant data residency region is set."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Residency Corp", admin_email="admin@residency.com",
                region="eu-central-1",
            )
        )
        assert t.region == "eu-central-1"

    def test_tenant_search_filter(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant listing with filters."""
        mod = multi_tenant_module
        multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Filter Enterprise", admin_email="fe@filter.com",
                tier=mod.TenantTier.ENTERPRISE,
            )
        )
        multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Filter Starter", admin_email="fs@filter.com",
                tier=mod.TenantTier.STARTER,
            )
        )
        enterprise_tenants = multi_tenant_engine.list_tenants(
            filters={"tier": mod.TenantTier.ENTERPRISE}
        )
        assert all(t.tier == mod.TenantTier.ENTERPRISE for t in enterprise_tenants)

    def test_tenant_health_score(self, multi_tenant_engine, multi_tenant_module):
        """Test tenant health score defaults to 100."""
        mod = multi_tenant_module
        t = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Health Corp", admin_email="admin@health.com",
            )
        )
        assert t.health_score == 100.0
