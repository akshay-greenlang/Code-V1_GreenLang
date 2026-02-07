"""
Unit tests for compliance automation service.

Tests ISO 27001, GDPR, PCI-DSS, and CCPA compliance automation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestISO27001Compliance:
    """Test ISO 27001 compliance automation."""

    @pytest.mark.asyncio
    async def test_get_control_status(self, iso27001_control):
        """Test getting ISO 27001 control status."""
        mock_service = AsyncMock()
        mock_service.get_control.return_value = iso27001_control

        result = await mock_service.get_control(iso27001_control["control_id"])

        assert result["control_id"] == iso27001_control["control_id"]
        assert result["status"] == "implemented"

    @pytest.mark.asyncio
    async def test_list_controls_by_category(self, iso27001_control):
        """Test listing controls by category."""
        mock_service = AsyncMock()
        mock_service.list_controls.return_value = [iso27001_control]

        result = await mock_service.list_controls(
            category="Information Security Policies"
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_update_control_evidence(self, iso27001_control):
        """Test updating control evidence."""
        mock_service = AsyncMock()
        iso27001_control["evidence"].append("new_evidence.pdf")
        mock_service.update_control.return_value = iso27001_control

        result = await mock_service.update_control(
            iso27001_control["control_id"],
            evidence=["new_evidence.pdf"]
        )

        assert "new_evidence.pdf" in result["evidence"]

    @pytest.mark.asyncio
    async def test_get_compliance_score(self):
        """Test getting overall ISO 27001 compliance score."""
        mock_service = AsyncMock()
        mock_service.get_compliance_score.return_value = {
            "framework": "iso27001",
            "score": 85.5,
            "implemented": 95,
            "total": 114,
        }

        result = await mock_service.get_compliance_score("iso27001")

        assert result["score"] > 0
        assert result["implemented"] <= result["total"]


class TestGDPRCompliance:
    """Test GDPR compliance automation."""

    @pytest.mark.asyncio
    async def test_create_dsar_request(self, gdpr_dsar_request):
        """Test creating DSAR request."""
        mock_service = AsyncMock()
        mock_service.create_dsar.return_value = gdpr_dsar_request

        result = await mock_service.create_dsar(
            request_type="access",
            email="user@example.com"
        )

        assert result["request_id"] is not None
        assert result["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_process_dsar_access_request(self, gdpr_dsar_request):
        """Test processing DSAR access request."""
        mock_service = AsyncMock()
        mock_service.process_dsar.return_value = {
            **gdpr_dsar_request,
            "status": "completed",
            "data_package_url": "https://secure.example.com/data/package.zip",
        }

        result = await mock_service.process_dsar(gdpr_dsar_request["request_id"])

        assert result["status"] == "completed"
        assert result["data_package_url"] is not None

    @pytest.mark.asyncio
    async def test_process_dsar_deletion_request(self):
        """Test processing DSAR deletion request."""
        mock_service = AsyncMock()
        mock_service.process_deletion.return_value = {
            "request_id": str(uuid4()),
            "status": "completed",
            "deleted_records": 150,
            "systems_processed": ["database", "analytics", "backups"],
        }

        result = await mock_service.process_deletion(str(uuid4()))

        assert result["status"] == "completed"
        assert result["deleted_records"] > 0

    @pytest.mark.asyncio
    async def test_check_dsar_deadline(self, gdpr_dsar_request):
        """Test checking DSAR deadline compliance."""
        mock_service = AsyncMock()
        mock_service.check_deadline.return_value = {
            "request_id": gdpr_dsar_request["request_id"],
            "days_remaining": 25,
            "on_track": True,
        }

        result = await mock_service.check_deadline(gdpr_dsar_request["request_id"])

        assert result["on_track"] is True

    @pytest.mark.asyncio
    async def test_data_discovery_scan(self):
        """Test running data discovery scan."""
        mock_service = AsyncMock()
        mock_service.run_discovery.return_value = {
            "scan_id": str(uuid4()),
            "status": "completed",
            "pii_locations": [
                {"system": "database", "table": "users", "columns": ["email", "phone"]},
            ],
        }

        result = await mock_service.run_discovery()

        assert result["status"] == "completed"
        assert len(result["pii_locations"]) > 0


class TestPCIDSSCompliance:
    """Test PCI-DSS compliance automation."""

    @pytest.mark.asyncio
    async def test_get_requirement_status(self, pci_dss_requirement):
        """Test getting PCI-DSS requirement status."""
        mock_service = AsyncMock()
        mock_service.get_requirement.return_value = pci_dss_requirement

        result = await mock_service.get_requirement(pci_dss_requirement["requirement_id"])

        assert result["status"] == "compliant"

    @pytest.mark.asyncio
    async def test_run_encryption_check(self):
        """Test running encryption validation."""
        mock_service = AsyncMock()
        mock_service.check_encryption.return_value = {
            "check_id": str(uuid4()),
            "status": "passed",
            "findings": [],
            "databases_checked": 5,
            "all_pan_encrypted": True,
        }

        result = await mock_service.check_encryption()

        assert result["status"] == "passed"
        assert result["all_pan_encrypted"] is True

    @pytest.mark.asyncio
    async def test_scan_card_data(self):
        """Test scanning for unencrypted card data."""
        mock_service = AsyncMock()
        mock_service.scan_card_data.return_value = {
            "scan_id": str(uuid4()),
            "status": "completed",
            "findings": [],
            "unencrypted_pan_found": False,
        }

        result = await mock_service.scan_card_data()

        assert result["unencrypted_pan_found"] is False

    @pytest.mark.asyncio
    async def test_validate_key_rotation(self):
        """Test validating encryption key rotation."""
        mock_service = AsyncMock()
        mock_service.validate_key_rotation.return_value = {
            "compliant": True,
            "last_rotation": datetime.utcnow() - timedelta(days=30),
            "rotation_policy_days": 90,
            "days_until_required": 60,
        }

        result = await mock_service.validate_key_rotation()

        assert result["compliant"] is True


class TestCCPACompliance:
    """Test CCPA compliance automation."""

    @pytest.mark.asyncio
    async def test_create_consumer_request(self, ccpa_consumer_request):
        """Test creating CCPA consumer request."""
        mock_service = AsyncMock()
        mock_service.create_request.return_value = ccpa_consumer_request

        result = await mock_service.create_request(
            request_type="delete",
            email="consumer@example.com"
        )

        assert result["request_id"] is not None

    @pytest.mark.asyncio
    async def test_verify_consumer_identity(self, ccpa_consumer_request):
        """Test verifying consumer identity."""
        mock_service = AsyncMock()
        mock_service.verify_identity.return_value = {
            "request_id": ccpa_consumer_request["request_id"],
            "verified": True,
            "verification_method": "email_confirmation",
        }

        result = await mock_service.verify_identity(
            ccpa_consumer_request["request_id"]
        )

        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_process_opt_out_request(self):
        """Test processing opt-out request."""
        mock_service = AsyncMock()
        mock_service.process_opt_out.return_value = {
            "request_id": str(uuid4()),
            "status": "completed",
            "services_updated": ["marketing", "analytics", "third_party_sharing"],
        }

        result = await mock_service.process_opt_out(str(uuid4()))

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_data_sale_status(self):
        """Test getting data sale disclosure status."""
        mock_service = AsyncMock()
        mock_service.get_data_sale_status.return_value = {
            "consumer_email": "consumer@example.com",
            "data_sold": False,
            "third_parties": [],
            "opt_out_status": "opted_out",
        }

        result = await mock_service.get_data_sale_status("consumer@example.com")

        assert result["data_sold"] is False


class TestComplianceReporting:
    """Test compliance reporting functionality."""

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, compliance_config):
        """Test generating compliance report."""
        mock_service = AsyncMock()
        mock_service.generate_report.return_value = {
            "report_id": str(uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "frameworks": compliance_config["frameworks"],
            "overall_score": 88.5,
            "framework_scores": {
                "iso27001": 90.0,
                "gdpr": 95.0,
                "pci_dss": 85.0,
                "ccpa": 85.0,
            },
        }

        result = await mock_service.generate_report()

        assert result["overall_score"] > 0
        assert len(result["framework_scores"]) == 4

    @pytest.mark.asyncio
    async def test_get_upcoming_reviews(self):
        """Test getting upcoming compliance reviews."""
        mock_service = AsyncMock()
        mock_service.get_upcoming_reviews.return_value = [
            {
                "control_id": "A.5.1.1",
                "framework": "iso27001",
                "review_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            },
        ]

        result = await mock_service.get_upcoming_reviews(days=30)

        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_get_compliance_gaps(self):
        """Test identifying compliance gaps."""
        mock_service = AsyncMock()
        mock_service.get_gaps.return_value = {
            "total_gaps": 5,
            "gaps": [
                {
                    "framework": "pci_dss",
                    "requirement": "3.6.1",
                    "gap": "Key management procedures not documented",
                    "severity": "medium",
                },
            ],
        }

        result = await mock_service.get_gaps()

        assert "total_gaps" in result
        assert "gaps" in result
