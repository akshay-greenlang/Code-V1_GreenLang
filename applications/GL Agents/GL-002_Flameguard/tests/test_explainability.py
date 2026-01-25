"""
GL-002 FLAMEGUARD - Explainability Module Tests

Comprehensive tests for SHAP/LIME-based explanations.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainability.decision_explainer import (
    DecisionExplainer,
    DecisionExplanation,
    FeatureContribution,
    ExplanationType,
    FeatureCategory,
    ImpactDirection,
    EFFICIENCY_SENSITIVITIES,
    REFERENCE_VALUES,
)
from explainability.explanation_audit import (
    ExplanationAuditLogger,
    ExplanationAuditEntry,
    ExplanationAuditEventType,
    AuditedDecisionExplainer,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def decision_explainer():
    """Create DecisionExplainer instance."""
    return DecisionExplainer(agent_id="GL-002-TEST")


@pytest.fixture
def audit_logger():
    """Create ExplanationAuditLogger instance."""
    return ExplanationAuditLogger(agent_id="GL-002-TEST")


@pytest.fixture
def audited_explainer(decision_explainer, audit_logger):
    """Create AuditedDecisionExplainer instance."""
    return AuditedDecisionExplainer(decision_explainer, audit_logger)


@pytest.fixture
def sample_process_data() -> Dict[str, float]:
    """Sample boiler process data."""
    return {
        "o2_percent": 4.0,
        "co_ppm": 35.0,
        "flue_gas_temperature_f": 365.0,
        "feedwater_temperature_f": 215.0,
        "load_percent": 75.0,
        "ambient_temperature_f": 72.0,
    }


@pytest.fixture
def sample_efficiency_result() -> Dict[str, float]:
    """Sample efficiency calculation result."""
    return {
        "efficiency_percent": 81.5,
        "stack_loss_percent": 12.3,
        "radiation_loss_percent": 1.5,
    }



# =============================================================================
# DECISION EXPLAINER TESTS
# =============================================================================


class TestDecisionExplainer:
    """Tests for DecisionExplainer class."""
    
    def test_initialization(self, decision_explainer):
        """Test explainer initialization."""
        assert decision_explainer.agent_id == "GL-002-TEST"
        assert decision_explainer._sensitivities == EFFICIENCY_SENSITIVITIES
        assert decision_explainer._reference_values == REFERENCE_VALUES
    
    def test_explain_efficiency(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test efficiency explanation generation."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        assert isinstance(explanation, DecisionExplanation)
        assert explanation.explanation_type == ExplanationType.EFFICIENCY
        assert explanation.boiler_id == "BOILER-001"
        assert explanation.current_value == 81.5
        assert len(explanation.feature_contributions) > 0
        assert explanation.lime_explanation is not None
        assert explanation.physics_grounding is not None
    
    def test_feature_contributions_sorted(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test feature contributions are sorted by impact."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        contributions = explanation.feature_contributions
        for i in range(len(contributions) - 1):
            assert abs(contributions[i].contribution) >= abs(contributions[i+1].contribution)
    
    def test_o2_contribution_physics(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test O2 contribution uses correct physics."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        o2_contrib = next(
            (c for c in explanation.feature_contributions if c.feature_name == "o2_percent"),
            None
        )
        
        assert o2_contrib is not None
        assert o2_contrib.category == FeatureCategory.COMBUSTION
        assert "ASME PTC 4.1" in o2_contrib.reference_standard
        
        expected_delta = sample_process_data["o2_percent"] - REFERENCE_VALUES["o2_percent"]
        expected_contribution = expected_delta * EFFICIENCY_SENSITIVITIES["o2_percent"]["sensitivity"]
        assert abs(o2_contrib.contribution - expected_contribution) < 0.01
    
    def test_explain_o2_trim_adjustment(self, decision_explainer):
        """Test O2 trim adjustment explanation."""
        explanation = decision_explainer.explain_o2_trim_adjustment(
            boiler_id="BOILER-001",
            current_o2=4.0,
            target_o2=3.0,
            current_co=30.0,
            load_percent=75.0,
        )
        
        assert explanation.explanation_type == ExplanationType.O2_TRIM
        assert explanation.current_value == 4.0
        assert explanation.recommended_value == 3.0
        assert explanation.expected_improvement > 0
        assert len(explanation.counterfactuals) >= 2
    
    def test_explain_o2_trim_with_high_co(self, decision_explainer):
        """Test O2 trim explanation with elevated CO."""
        explanation = decision_explainer.explain_o2_trim_adjustment(
            boiler_id="BOILER-001",
            current_o2=4.0,
            target_o2=3.5,
            current_co=150.0,
            load_percent=75.0,
        )
        
        co_contrib = next(
            (c for c in explanation.feature_contributions if c.feature_name == "co_constraint"),
            None
        )
        
        assert co_contrib is not None
        assert co_contrib.category == FeatureCategory.SAFETY
    
    def test_explain_safety_intervention(self, decision_explainer):
        """Test safety intervention explanation."""
        explanation = decision_explainer.explain_safety_intervention(
            boiler_id="BOILER-001",
            intervention_type="trip",
            trigger_value=155.0,
            setpoint=150.0,
            tag="STEAM_PRESSURE_HIGH",
            action_taken="Fuel valve closed",
        )
        
        assert explanation.explanation_type == ExplanationType.SAFETY
        assert explanation.confidence == 1.0
        assert "NFPA 85" in explanation.natural_language_summary
    
    def test_provenance_hash_generated(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test provenance hash is generated."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        assert explanation.provenance_hash != ""
        assert len(explanation.provenance_hash) == 64
    
    def test_visualization_data_generated(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test visualization data is generated."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        assert "waterfall" in explanation.visualization_data
        assert "force_plot" in explanation.visualization_data
        assert "category_breakdown" in explanation.visualization_data
    
    def test_natural_language_summary(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test natural language summary is generated."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        assert explanation.natural_language_summary != ""
        assert "BOILER-001" in explanation.natural_language_summary
    
    def test_get_explanation_by_id(
        self, decision_explainer, sample_process_data, sample_efficiency_result
    ):
        """Test retrieving explanation by ID."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        retrieved = decision_explainer.get_explanation(explanation.explanation_id)
        assert retrieved is not None
        assert retrieved.explanation_id == explanation.explanation_id
    
    def test_clear_cache(self, decision_explainer, sample_process_data, sample_efficiency_result):
        """Test clearing explanation cache."""
        decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        assert len(decision_explainer._explanations) > 0
        decision_explainer.clear_cache()
        assert len(decision_explainer._explanations) == 0



# =============================================================================
# AUDIT LOGGER TESTS
# =============================================================================


class TestExplanationAuditLogger:
    """Tests for ExplanationAuditLogger class."""
    
    def test_initialization(self, audit_logger):
        """Test audit logger initialization."""
        assert audit_logger.agent_id == "GL-002-TEST"
        assert len(audit_logger._entries) == 0
        assert audit_logger._chain_hash == "0" * 64
    
    def test_log_explanation_generated(self, audit_logger):
        """Test logging explanation generation."""
        entry = audit_logger.log_explanation_generated(
            explanation_id="exp-123",
            boiler_id="BOILER-001",
            explanation_type="efficiency",
            target_variable="efficiency_percent",
            provenance_hash="abc123" * 10 + "abcd",
            generation_time_ms=45.2,
            feature_count=5,
            counterfactual_count=3,
        )
        
        assert entry.event_type == ExplanationAuditEventType.EXPLANATION_GENERATED
        assert entry.explanation_id == "exp-123"
        assert entry.boiler_id == "BOILER-001"
        assert entry.generation_time_ms == 45.2
        assert len(audit_logger._entries) == 1
    
    def test_chain_hash_updated(self, audit_logger):
        """Test chain hash is updated on each entry."""
        initial_hash = audit_logger._chain_hash
        
        audit_logger.log_explanation_generated(
            explanation_id="exp-1",
            boiler_id="BOILER-001",
            explanation_type="efficiency",
            target_variable="efficiency_percent",
            provenance_hash="abc" * 21 + "a",
            generation_time_ms=10.0,
        )
        
        assert audit_logger._chain_hash != initial_hash
    
    def test_chain_verification(self, audit_logger):
        """Test chain hash verification."""
        for i in range(5):
            audit_logger.log_explanation_generated(
                explanation_id=f"exp-{i}",
                boiler_id="BOILER-001",
                explanation_type="efficiency",
                target_variable="efficiency_percent",
                provenance_hash="abc" * 21 + "a",
                generation_time_ms=10.0,
            )
        
        assert audit_logger._verify_chain() is True
    
    def test_statistics_tracking(self, audit_logger):
        """Test statistics are tracked."""
        for i in range(3):
            audit_logger.log_explanation_generated(
                explanation_id=f"exp-{i}",
                boiler_id="BOILER-001",
                explanation_type="efficiency",
                target_variable="efficiency_percent",
                provenance_hash="abc" * 21 + "a",
                generation_time_ms=10.0 + i,
            )
        
        audit_logger.log_explanation_accessed(
            explanation_id="exp-0",
            boiler_id="BOILER-001",
        )
        
        stats = audit_logger.get_statistics()
        assert stats["explanations_generated"] == 3
        assert stats["explanations_accessed"] == 1
        assert stats["average_generation_time_ms"] == 11.0
    
    def test_export_for_compliance(self, audit_logger):
        """Test compliance export."""
        audit_logger.log_explanation_generated(
            explanation_id="exp-1",
            boiler_id="BOILER-001",
            explanation_type="efficiency",
            target_variable="efficiency_percent",
            provenance_hash="abc" * 21 + "a",
            generation_time_ms=10.0,
        )
        
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=1)
        
        export = audit_logger.export_for_compliance(
            boiler_id="BOILER-001",
            start_time=start,
            end_time=end,
        )
        
        assert export["agent_id"] == "GL-002-TEST"
        assert export["entry_count"] == 1
        assert export["chain_verified"] is True


# =============================================================================
# AUDITED DECISION EXPLAINER TESTS
# =============================================================================


class TestAuditedDecisionExplainer:
    """Tests for AuditedDecisionExplainer wrapper."""
    
    def test_explain_efficiency_logs_audit(
        self, audited_explainer, audit_logger, sample_process_data, sample_efficiency_result
    ):
        """Test explain_efficiency logs to audit."""
        explanation = audited_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        assert len(audit_logger._entries) == 1
        entry = audit_logger._entries[0]
        assert entry.event_type == ExplanationAuditEventType.EXPLANATION_GENERATED
        assert entry.explanation_id == explanation.explanation_id
    
    def test_explain_o2_trim_logs_audit(self, audited_explainer, audit_logger):
        """Test explain_o2_trim logs to audit."""
        audited_explainer.explain_o2_trim_adjustment(
            boiler_id="BOILER-001",
            current_o2=4.0,
            target_o2=3.0,
            current_co=30.0,
            load_percent=75.0,
        )
        
        assert len(audit_logger._entries) == 1
        entry = audit_logger._entries[0]
        assert entry.explanation_type == "o2_trim"
    
    def test_get_explanation_logs_access(
        self, audited_explainer, audit_logger, sample_process_data, sample_efficiency_result
    ):
        """Test get_explanation logs access event."""
        explanation = audited_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data=sample_process_data,
            efficiency_result=sample_efficiency_result,
        )
        
        audited_explainer.get_explanation(explanation.explanation_id, accessor="operator1")
        
        assert len(audit_logger._entries) == 2
        access_entry = audit_logger._entries[1]
        assert access_entry.event_type == ExplanationAuditEventType.EXPLANATION_ACCESSED


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_process_data(self, decision_explainer):
        """Test handling empty process data."""
        explanation = decision_explainer.explain_efficiency(
            boiler_id="BOILER-001",
            process_data={},
            efficiency_result={"efficiency_percent": 82.0},
        )
        
        assert explanation is not None
        assert len(explanation.feature_contributions) == 0
    
    def test_get_nonexistent_explanation(self, decision_explainer):
        """Test getting non-existent explanation."""
        result = decision_explainer.get_explanation("nonexistent-id")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
