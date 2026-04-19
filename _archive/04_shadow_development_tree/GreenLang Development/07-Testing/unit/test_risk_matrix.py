"""
Unit tests for RiskMatrix and RiskRegister modules.

Tests cover:
- 5x5 risk matrix calculations
- Risk level determination
- Risk color mapping
- IEC 61511 SIL assignment
- Risk heatmap generation
- Risk aggregation
- Risk register operations
- HAZOP and FMEA integration
- Compliance reporting
"""

import pytest
from datetime import datetime, timedelta
from greenlang.safety.risk_matrix import (
    RiskMatrix,
    RiskRegister,
    RiskData,
    RiskLevel,
    RiskCategory,
    RiskStatus,
    SafetyIntegrityLevel,
)


class TestRiskMatrix:
    """Test RiskMatrix calculations and mappings."""

    def test_calculate_risk_level_low_severity_low_likelihood(self):
        """Test LOW risk: minor severity + remote likelihood."""
        risk = RiskMatrix.calculate_risk_level(severity=1, likelihood=1)
        assert risk == RiskLevel.LOW

    def test_calculate_risk_level_medium_mixed(self):
        """Test MEDIUM risk: mixed severity/likelihood."""
        risk = RiskMatrix.calculate_risk_level(severity=2, likelihood=3)
        assert risk == RiskLevel.MEDIUM

    def test_calculate_risk_level_high_severity_high_likelihood(self):
        """Test HIGH risk: major severity + probable likelihood."""
        risk = RiskMatrix.calculate_risk_level(severity=3, likelihood=4)
        assert risk == RiskLevel.HIGH

    def test_calculate_risk_level_critical_catastrophic(self):
        """Test CRITICAL risk: catastrophic severity + almost certain."""
        risk = RiskMatrix.calculate_risk_level(severity=5, likelihood=5)
        assert risk == RiskLevel.CRITICAL

    def test_calculate_risk_level_critical_severity_4(self):
        """Test CRITICAL risk: major severity + almost certain."""
        risk = RiskMatrix.calculate_risk_level(severity=4, likelihood=5)
        assert risk == RiskLevel.CRITICAL

    def test_calculate_risk_level_invalid_severity(self):
        """Test invalid severity raises ValueError."""
        with pytest.raises(ValueError):
            RiskMatrix.calculate_risk_level(severity=6, likelihood=3)

    def test_calculate_risk_level_invalid_likelihood(self):
        """Test invalid likelihood raises ValueError."""
        with pytest.raises(ValueError):
            RiskMatrix.calculate_risk_level(severity=3, likelihood=0)

    def test_calculate_risk_level_all_combinations(self):
        """Test all 25 matrix combinations."""
        expected = [
            [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM],
            [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH],
            [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH],
            [RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL],
            [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.CRITICAL],
        ]

        for severity in range(1, 6):
            for likelihood in range(1, 6):
                risk = RiskMatrix.calculate_risk_level(severity, likelihood)
                assert risk == expected[severity - 1][likelihood - 1], \
                    f"Mismatch at ({severity},{likelihood})"

    def test_get_risk_color_low(self):
        """Test color mapping for LOW risk."""
        color = RiskMatrix.get_risk_color(RiskLevel.LOW)
        assert color == "green"

    def test_get_risk_color_medium(self):
        """Test color mapping for MEDIUM risk."""
        color = RiskMatrix.get_risk_color(RiskLevel.MEDIUM)
        assert color == "yellow"

    def test_get_risk_color_high(self):
        """Test color mapping for HIGH risk."""
        color = RiskMatrix.get_risk_color(RiskLevel.HIGH)
        assert color == "orange"

    def test_get_risk_color_critical(self):
        """Test color mapping for CRITICAL risk."""
        color = RiskMatrix.get_risk_color(RiskLevel.CRITICAL)
        assert color == "red"

    def test_get_required_sil_low(self):
        """Test SIL mapping for LOW risk."""
        sil = RiskMatrix.get_required_sil(RiskLevel.LOW)
        assert sil == SafetyIntegrityLevel.NO_SIL

    def test_get_required_sil_medium(self):
        """Test SIL mapping for MEDIUM risk."""
        sil = RiskMatrix.get_required_sil(RiskLevel.MEDIUM)
        assert sil == SafetyIntegrityLevel.SIL_1

    def test_get_required_sil_high(self):
        """Test SIL mapping for HIGH risk."""
        sil = RiskMatrix.get_required_sil(RiskLevel.HIGH)
        assert sil == SafetyIntegrityLevel.SIL_2

    def test_get_required_sil_critical(self):
        """Test SIL mapping for CRITICAL risk."""
        sil = RiskMatrix.get_required_sil(RiskLevel.CRITICAL)
        assert sil == SafetyIntegrityLevel.SIL_3

    def test_get_acceptance_days_critical(self):
        """Test IEC 61511 acceptance days for CRITICAL."""
        days = RiskMatrix.get_acceptance_days(RiskLevel.CRITICAL)
        assert days == 7

    def test_get_acceptance_days_high(self):
        """Test acceptance days for HIGH."""
        days = RiskMatrix.get_acceptance_days(RiskLevel.HIGH)
        assert days == 30

    def test_get_acceptance_days_medium(self):
        """Test acceptance days for MEDIUM."""
        days = RiskMatrix.get_acceptance_days(RiskLevel.MEDIUM)
        assert days == 90

    def test_get_acceptance_days_low(self):
        """Test acceptance days for LOW."""
        days = RiskMatrix.get_acceptance_days(RiskLevel.LOW)
        assert days == 365

    def test_generate_heatmap_empty(self):
        """Test heatmap generation with no risks."""
        heatmap = RiskMatrix.generate_heatmap([])

        assert "matrix" in heatmap
        assert "colors" in heatmap
        assert "summary" in heatmap
        assert len(heatmap["matrix"]) == 5
        assert all(len(row) == 5 for row in heatmap["matrix"])

    def test_generate_heatmap_single_risk(self):
        """Test heatmap generation with single risk."""
        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=4,
        )
        risk.risk_level = RiskLevel.CRITICAL

        heatmap = RiskMatrix.generate_heatmap([risk])

        assert heatmap["matrix"][3][3] == 1  # Position [4,4] at index [3,3]

    def test_generate_heatmap_excludes_closed_risks(self):
        """Test heatmap excludes closed risks."""
        risk1 = RiskData(
            title="Open Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=4,
            status=RiskStatus.OPEN,
        )

        risk2 = RiskData(
            title="Closed Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=4,
            status=RiskStatus.CLOSED,
        )

        heatmap = RiskMatrix.generate_heatmap([risk1, risk2])
        assert heatmap["matrix"][3][3] == 1  # Only open risk counted

    def test_aggregate_risks_empty(self):
        """Test aggregation with no risks."""
        agg = RiskMatrix.aggregate_risks([])

        assert agg["total_risks"] == 0
        assert agg["critical"] == 0
        assert agg["high"] == 0
        assert agg["medium"] == 0
        assert agg["low"] == 0

    def test_aggregate_risks_summary(self):
        """Test aggregation generates correct statistics."""
        risks = [
            RiskData(
                title="Critical Risk",
                description="Test",
                category=RiskCategory.SAFETY,
                severity=5,
                likelihood=5,
                status=RiskStatus.OPEN,
            ),
            RiskData(
                title="High Risk",
                description="Test",
                category=RiskCategory.SAFETY,
                severity=3,
                likelihood=4,
                status=RiskStatus.OPEN,
            ),
            RiskData(
                title="Medium Risk",
                description="Test",
                category=RiskCategory.SAFETY,
                severity=2,
                likelihood=3,
                status=RiskStatus.OPEN,
            ),
        ]

        # Set risk levels
        for risk in risks:
            risk.risk_level = RiskMatrix.calculate_risk_level(risk.severity, risk.likelihood)

        agg = RiskMatrix.aggregate_risks(risks)

        assert agg["total_risks"] == 3
        assert agg["critical"] == 1
        assert agg["high"] == 1
        assert agg["medium"] == 1
        assert agg["low"] == 0
        assert agg["average_severity"] == pytest.approx(10 / 3, rel=0.01)

    def test_aggregate_risks_overdue_tracking(self):
        """Test overdue risk tracking in aggregation."""
        now = datetime.utcnow()

        overdue_risk = RiskData(
            title="Overdue Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=3,
            status=RiskStatus.OPEN,
            target_mitigation_date=now - timedelta(days=5),
        )
        overdue_risk.risk_level = RiskLevel.HIGH

        current_risk = RiskData(
            title="Current Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=2,
            status=RiskStatus.OPEN,
            target_mitigation_date=now + timedelta(days=10),
        )
        current_risk.risk_level = RiskLevel.MEDIUM

        agg = RiskMatrix.aggregate_risks([overdue_risk, current_risk])

        assert agg["risks_overdue"] == 1
        assert agg["risks_at_target"] == 1


class TestRiskRegister:
    """Test RiskRegister operations."""

    def test_register_init(self):
        """Test register initialization."""
        register = RiskRegister()
        assert len(register.risks) == 0
        assert len(register.audit_trail) == 0

    def test_add_risk_basic(self):
        """Test adding risk to register."""
        register = RiskRegister()

        risk = RiskData(
            title="High Temperature Event",
            description="Potential equipment damage",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=2,
        )

        added_risk = register.add_risk(risk)

        assert added_risk.risk_id in register.risks
        assert added_risk.risk_level == RiskLevel.MEDIUM
        assert added_risk.required_sil == SafetyIntegrityLevel.SIL_1

    def test_add_risk_sets_target_date(self):
        """Test add_risk sets target mitigation date."""
        register = RiskRegister()
        now = datetime.utcnow()

        risk = RiskData(
            title="Critical Risk",
            description="Immediate action required",
            category=RiskCategory.SAFETY,
            severity=5,
            likelihood=5,
        )

        added_risk = register.add_risk(risk)

        assert added_risk.target_mitigation_date is not None
        days_diff = (added_risk.target_mitigation_date - now).days
        assert days_diff == 7  # CRITICAL: 7 days

    def test_add_risk_duplicate_raises_error(self):
        """Test adding duplicate risk raises error."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
            risk_id="RISK-001",
        )

        register.add_risk(risk)

        with pytest.raises(ValueError):
            register.add_risk(risk)

    def test_add_risk_calculates_provenance(self):
        """Test add_risk calculates provenance hash."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
        )

        added_risk = register.add_risk(risk)

        assert added_risk.provenance_hash
        assert len(added_risk.provenance_hash) == 64  # SHA-256

    def test_update_risk_status(self):
        """Test updating risk status."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
        )

        added_risk = register.add_risk(risk)
        risk_id = added_risk.risk_id

        updated_risk = register.update_risk(risk_id, {
            "status": RiskStatus.IN_PROGRESS,
            "responsible_party": "John Doe"
        })

        assert updated_risk.status == RiskStatus.IN_PROGRESS
        assert updated_risk.responsible_party == "John Doe"

    def test_update_risk_recalculates_level(self):
        """Test update_risk recalculates risk level."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=1,
            likelihood=1,
        )

        added_risk = register.add_risk(risk)
        risk_id = added_risk.risk_id

        # Increase severity
        updated_risk = register.update_risk(risk_id, {
            "severity": 5,
            "likelihood": 5,
        })

        assert updated_risk.risk_level == RiskLevel.CRITICAL
        assert updated_risk.required_sil == SafetyIntegrityLevel.SIL_3

    def test_update_nonexistent_risk_raises_error(self):
        """Test updating nonexistent risk raises error."""
        register = RiskRegister()

        with pytest.raises(ValueError):
            register.update_risk("RISK-999", {"status": RiskStatus.CLOSED})

    def test_get_open_risks(self):
        """Test filtering open risks."""
        register = RiskRegister()

        risk1 = RiskData(
            title="Open Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=3,
            status=RiskStatus.OPEN,
        )

        risk2 = RiskData(
            title="Closed Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=3,
            status=RiskStatus.CLOSED,
        )

        register.add_risk(risk1)
        register.add_risk(risk2)

        open_risks = register.get_open_risks()
        assert len(open_risks) == 1

    def test_get_open_risks_by_category(self):
        """Test filtering open risks by category."""
        register = RiskRegister()

        risk1 = RiskData(
            title="Safety Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
        )

        risk2 = RiskData(
            title="Operational Risk",
            description="Test",
            category=RiskCategory.OPERATIONAL,
            severity=3,
            likelihood=3,
        )

        register.add_risk(risk1)
        register.add_risk(risk2)

        safety_risks = register.get_open_risks(RiskCategory.SAFETY)
        assert len(safety_risks) == 1
        assert safety_risks[0].category == RiskCategory.SAFETY

    def test_get_critical_risks(self):
        """Test getting critical risks."""
        register = RiskRegister()

        risk = RiskData(
            title="Critical Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=5,
            likelihood=5,
        )

        register.add_risk(risk)

        critical_risks = register.get_critical_risks()
        assert len(critical_risks) == 1

    def test_get_overdue_risks(self):
        """Test getting overdue risks."""
        register = RiskRegister()
        now = datetime.utcnow()

        risk = RiskData(
            title="Overdue Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
            target_mitigation_date=now - timedelta(days=10),
        )

        register.add_risk(risk)

        overdue_risks = register.get_overdue_risks()
        assert len(overdue_risks) == 1

    def test_import_from_hazop(self):
        """Test importing risks from HAZOP results."""
        register = RiskRegister()

        hazop_data = [
            {
                "deviation_id": "DEV-001",
                "deviation_description": "High Temperature",
                "consequences": ["Equipment damage", "Safety hazard"],
                "severity": 4,
                "likelihood": 3,
                "recommendations": ["Install temperature sensor", "Add alarm"],
            }
        ]

        risks = register.import_from_hazop(hazop_data)

        assert len(risks) == 1
        assert risks[0].source == "HAZOP"
        assert risks[0].source_id == "DEV-001"

    def test_import_from_fmea(self):
        """Test importing risks from FMEA results."""
        register = RiskRegister()

        fmea_data = [
            {
                "fm_id": "FM-001",
                "component_name": "Pressure Relief Valve",
                "failure_mode": "Stuck Open",
                "end_effect": "Pressure loss",
                "severity": 4,
                "occurrence": 2,
                "rpn": 100,
                "recommended_action": "Replace valve",
                "responsibility": "Maintenance",
            }
        ]

        risks = register.import_from_fmea(fmea_data)

        assert len(risks) == 1
        assert risks[0].source == "FMEA"
        assert risks[0].source_id == "FM-001"

    def test_generate_report(self):
        """Test report generation."""
        register = RiskRegister()

        risk1 = RiskData(
            title="Critical Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=5,
            likelihood=5,
        )

        risk2 = RiskData(
            title="Low Risk",
            description="Test",
            category=RiskCategory.OPERATIONAL,
            severity=1,
            likelihood=1,
        )

        register.add_risk(risk1)
        register.add_risk(risk2)

        report = register.generate_report()

        assert "summary" in report
        assert report["summary"]["total_risks"] == 2
        assert report["summary"]["critical"] == 1
        assert report["summary"]["low"] == 1
        assert "heatmap" in report
        assert "critical_risks" in report

    def test_export_to_compliance_report_text(self):
        """Test exporting compliance report as text."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=4,
            likelihood=4,
        )

        register.add_risk(risk)

        report = register.export_to_compliance_report(format_type="text")

        assert "RISK REGISTER COMPLIANCE REPORT" in report
        assert "CRITICAL" in report
        assert "Immediate action required" in report

    def test_export_to_compliance_report_csv(self):
        """Test exporting compliance report as CSV."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
        )

        register.add_risk(risk)

        report = register.export_to_compliance_report(format_type="csv")

        assert "risk_id,title,category" in report
        assert "Test Risk" in report
        assert "safety" in report

    def test_export_to_compliance_report_json(self):
        """Test exporting compliance report as JSON."""
        register = RiskRegister()

        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=2,
            likelihood=2,
        )

        register.add_risk(risk)

        report = register.export_to_compliance_report(format_type="json")

        assert "summary" in report or "{" in report


class TestRiskDataValidation:
    """Test RiskData Pydantic validation."""

    def test_risk_data_valid(self):
        """Test valid RiskData creation."""
        risk = RiskData(
            title="Valid Risk",
            description="Valid description",
            category=RiskCategory.SAFETY,
            severity=3,
            likelihood=3,
        )

        assert risk.title == "Valid Risk"
        assert risk.severity == 3

    def test_risk_data_invalid_severity(self):
        """Test RiskData rejects invalid severity."""
        with pytest.raises(ValueError):
            RiskData(
                title="Invalid Risk",
                description="Test",
                category=RiskCategory.SAFETY,
                severity=6,
                likelihood=3,
            )

    def test_risk_data_invalid_likelihood(self):
        """Test RiskData rejects invalid likelihood."""
        with pytest.raises(ValueError):
            RiskData(
                title="Invalid Risk",
                description="Test",
                category=RiskCategory.SAFETY,
                severity=3,
                likelihood=0,
            )

    def test_risk_data_default_status(self):
        """Test RiskData defaults to OPEN status."""
        risk = RiskData(
            title="Test Risk",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=2,
            likelihood=2,
        )

        assert risk.status == RiskStatus.OPEN

    def test_risk_data_auto_generated_id(self):
        """Test RiskData generates unique ID."""
        risk1 = RiskData(
            title="Risk 1",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=2,
            likelihood=2,
        )

        risk2 = RiskData(
            title="Risk 2",
            description="Test",
            category=RiskCategory.SAFETY,
            severity=2,
            likelihood=2,
        )

        assert risk1.risk_id != risk2.risk_id
        assert risk1.risk_id.startswith("RISK-")
        assert risk2.risk_id.startswith("RISK-")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
