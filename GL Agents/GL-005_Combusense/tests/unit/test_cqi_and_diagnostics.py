# -*- coding: utf-8 -*-
"""
Unit Tests for GL-005 COMBUSENSE CQI and Diagnostics Components

Tests for:
    - CQI Calculator (Playbook Section 8)
    - Explainability Engine (Playbook Section 10)
    - Narrative Generator (Playbook Section 10.4)
    - Anomaly Detection (Playbook Section 9)
    - SSE Streaming (Playbook Section 12.3)

Author: GreenLang GL-005 Team
Version: 1.0.0
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import List

# Import CQI Calculator components
from calculators.cqi_calculator import (
    CQICalculator,
    CQIConfiguration,
    CombustionSignals,
    CQIResult,
    CQIGrade,
    OperatingMode,
    calculate_cqi_quick,
)

# Import Explainability components
from calculators.explainability import (
    ExplainabilityEngine,
    SHAPStyleExplainer,
    LIMEStyleExplainer,
    AttentionVisualizer,
    SignalSnapshot,
    SignalTimeSeries,
    ExplainabilityInput,
    ExplanationType,
    FeatureAttribution,
)

# Import Narrative Generator components
from calculators.narrative_generator import (
    NarrativeGenerator,
    NarrativeInput,
    NarrativeResult,
    EvidenceBundle,
    generate_cqi_narrative_quick,
)

# Import Anomaly Detection components
from calculators.anomaly_detection import (
    AnomalyDetector,
    AnomalyType,
    Severity,
    CombustionState,
    DetectionConfig,
    AnomalyEvent,
    detect_quick,
    ANOMALY_TAXONOMY,
)


# =============================================================================
# CQI CALCULATOR TESTS
# =============================================================================

class TestCQICalculator:
    """Tests for CQI Calculator"""

    def test_cqi_calculator_initialization(self):
        """Test CQI calculator initializes with default config"""
        calculator = CQICalculator()
        assert calculator is not None
        assert calculator.config.efficiency_weight == 0.30
        assert calculator.config.emissions_weight == 0.30
        assert calculator.config.stability_weight == 0.20
        assert calculator.config.safety_weight == 0.15
        assert calculator.config.data_weight == 0.05

    def test_cqi_weights_sum_to_one(self):
        """Test CQI weights sum to 1.0"""
        config = CQIConfiguration()
        total = (config.efficiency_weight + config.emissions_weight +
                 config.stability_weight + config.safety_weight + config.data_weight)
        assert abs(total - 1.0) < 0.001

    def test_cqi_calculation_normal_operation(self):
        """Test CQI calculation for normal operating conditions"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=30.0,
            nox_ppm=20.0,
            flame_intensity=85.0,
            operating_mode=OperatingMode.RUN
        )
        result = calculator.calculate(signals)

        assert isinstance(result, CQIResult)
        assert 0 <= result.cqi_total <= 100
        assert result.grade in CQIGrade
        assert result.confidence > 0
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) > 0

    def test_cqi_excellent_grade(self):
        """Test CQI returns excellent grade for optimal conditions"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=20.0,
            nox_ppm=15.0,
            flame_intensity=90.0,
            operating_mode=OperatingMode.RUN
        )
        result = calculator.calculate(signals)

        # Should be good or excellent
        assert result.cqi_total >= 75
        assert result.grade in [CQIGrade.EXCELLENT, CQIGrade.GOOD]

    def test_cqi_poor_grade_high_emissions(self):
        """Test CQI returns poor grade for high emissions"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=200.0,  # High CO
            nox_ppm=80.0,   # High NOx
            flame_intensity=70.0,
            operating_mode=OperatingMode.RUN
        )
        result = calculator.calculate(signals)

        assert result.cqi_total < 75
        assert result.sub_scores.emissions < 70

    def test_cqi_safety_cap_on_bypass(self):
        """Test CQI is capped when safety bypass is active"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=30.0,
            nox_ppm=20.0,
            flame_intensity=85.0,
            bypass_active=True,
            bypass_duration_s=400,  # > 5 minutes
            operating_mode=OperatingMode.RUN
        )
        result = calculator.calculate(signals)

        assert result.is_capped
        assert result.cqi_total <= 30.0
        assert result.cap_reason is not None

    def test_cqi_non_run_mode(self):
        """Test CQI returns zero for non-RUN modes"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=0.0,
            air_flow_kg_s=0.0,
            o2_percent=21.0,
            co_ppm=0.0,
            nox_ppm=0.0,
            flame_intensity=0.0,
            operating_mode=OperatingMode.STANDBY
        )
        result = calculator.calculate(signals)

        assert result.cqi_total == 0.0
        assert result.is_capped
        assert "STANDBY" in result.cap_reason

    def test_cqi_sub_scores_present(self):
        """Test all CQI sub-scores are calculated"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=30.0,
            nox_ppm=20.0,
            flame_intensity=85.0,
            operating_mode=OperatingMode.RUN
        )
        result = calculator.calculate(signals)

        assert result.sub_scores.efficiency >= 0
        assert result.sub_scores.emissions >= 0
        assert result.sub_scores.stability >= 0
        assert result.sub_scores.safety >= 0
        assert result.sub_scores.data >= 0

    def test_cqi_sse_event_format(self):
        """Test CQI result converts to SSE event format"""
        calculator = CQICalculator(CQIConfiguration(asset_id="Boiler-3"))
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=30.0,
            nox_ppm=20.0,
            flame_intensity=85.0,
            operating_mode=OperatingMode.RUN
        )
        result = calculator.calculate(signals)
        sse_event = calculator.to_sse_event(result)

        assert sse_event["schema_version"] == "1.0"
        assert sse_event["asset_id"] == "Boiler-3"
        assert "cqi_total" in sse_event
        assert "cqi_components" in sse_event
        assert len(sse_event["cqi_components"]) == 5

    def test_cqi_quick_calculation(self):
        """Test quick CQI calculation helper function"""
        result = calculate_cqi_quick(
            o2_percent=3.0,
            co_ppm=50.0,
            nox_ppm=30.0
        )
        assert isinstance(result, CQIResult)
        assert 0 <= result.cqi_total <= 100


# =============================================================================
# EXPLAINABILITY TESTS
# =============================================================================

class TestExplainability:
    """Tests for Explainability Engine"""

    def test_shap_explainer_initialization(self):
        """Test SHAP explainer initializes correctly"""
        explainer = SHAPStyleExplainer()
        assert explainer is not None
        assert "o2_percent" in explainer.reference_values

    def test_shap_feature_attribution(self):
        """Test SHAP-style feature attribution generation"""
        explainer = SHAPStyleExplainer()
        current = {"o2_percent": 5.0, "co_ppm": 100.0, "nox_ppm": 50.0}
        reference = {"o2_percent": 3.0, "co_ppm": 50.0, "nox_ppm": 30.0}

        attributions = explainer.explain(current, -15.0, reference)

        assert len(attributions) > 0
        assert all(isinstance(a, FeatureAttribution) for a in attributions)
        assert attributions[0].rank == 1  # Top driver has rank 1

    def test_shap_attribution_ranking(self):
        """Test attributions are ranked by absolute value"""
        explainer = SHAPStyleExplainer()
        current = {"o2_percent": 3.0, "co_ppm": 200.0, "nox_ppm": 30.0}

        attributions = explainer.explain(current, -20.0)

        # Verify ranking is by absolute attribution (descending)
        for i in range(len(attributions) - 1):
            assert abs(attributions[i].attribution_value) >= abs(attributions[i + 1].attribution_value)

    def test_lime_incident_explanation(self):
        """Test LIME-style incident explanation"""
        explainer = LIMEStyleExplainer()

        before = SignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            o2_percent=3.0,
            co_ppm=50.0,
            nox_ppm=30.0
        )
        after = SignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            o2_percent=2.0,
            co_ppm=150.0,
            nox_ppm=30.0
        )

        attributions, hypotheses = explainer.explain_incident(
            "CO_SPIKE", "S3", before, after, -15.0
        )

        assert len(attributions) > 0
        assert len(hypotheses) > 0
        assert hypotheses[0][1] > 0  # Confidence > 0

    def test_attention_visualizer_computation(self):
        """Test attention map computation"""
        visualizer = AttentionVisualizer()

        timestamps = [
            datetime.now(timezone.utc)
            for _ in range(60)
        ]
        time_series = SignalTimeSeries(
            timestamps=timestamps,
            o2_percent=[3.0 + i * 0.05 for i in range(60)],
            co_ppm=[50.0 + i * 2 for i in range(60)],
            flame_intensity=[80.0 - i * 0.2 for i in range(60)]
        )

        attention_map = visualizer.compute_attention_map(
            time_series,
            datetime.now(timezone.utc)
        )

        assert attention_map is not None
        assert len(attention_map.signals) > 0
        assert len(attention_map.top_signals) > 0
        assert attention_map.provenance_hash is not None

    def test_explainability_engine_cqi_change(self):
        """Test full explainability engine for CQI change"""
        engine = ExplainabilityEngine()

        input_data = ExplainabilityInput(
            event_type=ExplanationType.CQI_DEGRADATION,
            asset_id="Boiler-1",
            current_snapshot=SignalSnapshot(
                timestamp=datetime.now(timezone.utc),
                o2_percent=5.0,
                co_ppm=100.0,
                nox_ppm=40.0
            ),
            reference_snapshot=SignalSnapshot(
                timestamp=datetime.now(timezone.utc),
                o2_percent=3.0,
                co_ppm=50.0,
                nox_ppm=30.0
            ),
            cqi_current=65.0,
            cqi_previous=82.0
        )

        explanation = engine.explain_cqi_change(input_data)

        assert explanation is not None
        assert explanation.explanation_id.startswith("EXP-")
        assert len(explanation.top_drivers) <= 5
        assert explanation.event_summary is not None
        assert len(explanation.signal_deltas) > 0

    def test_evidence_bundle_generation(self):
        """Test evidence bundle generation for narratives"""
        engine = ExplainabilityEngine()

        input_data = ExplainabilityInput(
            event_type=ExplanationType.CQI_DEGRADATION,
            asset_id="Boiler-1",
            current_snapshot=SignalSnapshot(
                timestamp=datetime.now(timezone.utc),
                o2_percent=5.0,
                co_ppm=100.0
            ),
            cqi_current=65.0,
            cqi_previous=82.0
        )

        explanation = engine.explain_cqi_change(input_data)
        bundle = engine.to_evidence_bundle(explanation)

        assert "operating_mode" in bundle
        assert "signal_deltas" in bundle
        assert "top_attributions" in bundle


# =============================================================================
# NARRATIVE GENERATOR TESTS
# =============================================================================

class TestNarrativeGenerator:
    """Tests for Narrative Generator"""

    def test_narrative_generator_initialization(self):
        """Test narrative generator initializes correctly"""
        generator = NarrativeGenerator()
        assert generator is not None

    def test_cqi_narrative_generation(self):
        """Test CQI narrative generation"""
        generator = NarrativeGenerator()
        narrative = generator.generate_cqi_narrative(
            cqi=85.0,
            grade="good",
            top_driver="O2 slightly above target",
            recommendation="Monitor O2 trend"
        )

        assert len(narrative) > 0
        assert "85.0" in narrative
        assert "good" in narrative.lower()

    def test_narrative_from_input(self):
        """Test narrative generation from full input"""
        generator = NarrativeGenerator()

        input_data = NarrativeInput(
            event_type="CO_SPIKE",
            asset_id="Boiler-1",
            cqi_current=65.0,
            cqi_previous=82.0,
            o2_before=3.0,
            o2_after=2.5,
            co_before=50.0,
            co_after=150.0,
            primary_driver="Low O2 causing incomplete combustion",
            recommended_checks=["Check air damper", "Verify fuel flow"]
        )

        result = generator.generate(input_data)

        assert isinstance(result, NarrativeResult)
        assert len(result.narrative_text) > 0
        assert result.summary is not None
        assert "CO" in result.narrative_text.upper()

    def test_narrative_safety_reminder(self):
        """Test safety reminder is included"""
        generator = NarrativeGenerator()

        input_data = NarrativeInput(
            event_type="COMBUSTION_RICH",
            asset_id="Boiler-1",
            cqi_current=60.0,
            cqi_previous=80.0,
            o2_after=1.5,
            co_after=200.0
        )

        result = generator.generate(input_data)

        # Check safety reminder is present
        assert "SIS" in result.safety_reminder or "BMS" in result.safety_reminder

    def test_narrative_controlled_vocabulary(self):
        """Test controlled vocabulary is enforced"""
        generator = NarrativeGenerator()

        # Check that forbidden phrases are replaced
        text = generator._apply_controlled_vocabulary("immediately do something")
        assert "immediately do" not in text.lower()

    def test_narrative_quick_generation(self):
        """Test quick narrative generation helper"""
        narrative = generate_cqi_narrative_quick(
            cqi=75.0,
            grade="acceptable",
            driver="High excess air",
            recommendation="Check damper positions"
        )

        assert len(narrative) > 0
        assert "75.0" in narrative


# =============================================================================
# ANOMALY DETECTION TESTS
# =============================================================================

class TestAnomalyDetection:
    """Tests for Anomaly Detection"""

    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initializes correctly"""
        detector = AnomalyDetector()
        assert detector is not None
        assert detector.config is not None

    def test_anomaly_taxonomy_completeness(self):
        """Test anomaly taxonomy contains all required types"""
        required_types = [
            AnomalyType.CO_SPIKE,
            AnomalyType.NOX_SPIKE,
            AnomalyType.COMBUSTION_RICH,
            AnomalyType.COMBUSTION_LEAN,
            AnomalyType.FLAME_INSTABILITY,
            AnomalyType.FLAME_LOSS,
        ]

        for anomaly_type in required_types:
            assert anomaly_type in ANOMALY_TAXONOMY
            taxonomy = ANOMALY_TAXONOMY[anomaly_type]
            assert "description" in taxonomy
            assert "recommended_checks" in taxonomy
            assert len(taxonomy["recommended_checks"]) > 0

    def test_co_spike_detection(self):
        """Test CO spike detection"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=3.0,
            co_ppm=250.0,  # High CO
            nox_ppm=30.0,
            flame_present=True,
            operating_mode="RUN"
        )

        events = detector.detect(state)

        # Should detect CO spike
        co_events = [e for e in events if e.anomaly_type == AnomalyType.CO_SPIKE]
        assert len(co_events) > 0
        assert co_events[0].severity in [Severity.S3, Severity.S4]

    def test_rich_combustion_detection(self):
        """Test rich combustion detection"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=1.0,  # Low O2
            co_ppm=100.0,    # Elevated CO
            nox_ppm=30.0,
            flame_present=True,
            operating_mode="RUN"
        )

        events = detector.detect(state)

        rich_events = [e for e in events if e.anomaly_type == AnomalyType.COMBUSTION_RICH]
        assert len(rich_events) > 0

    def test_lean_combustion_detection(self):
        """Test lean combustion detection"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=8.0,  # High O2 (excess air)
            co_ppm=20.0,
            nox_ppm=30.0,
            flame_present=True,
            operating_mode="RUN"
        )

        events = detector.detect(state)

        lean_events = [e for e in events if e.anomaly_type == AnomalyType.COMBUSTION_LEAN]
        assert len(lean_events) > 0

    def test_flame_loss_detection(self):
        """Test flame loss detection is critical severity"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=21.0,
            co_ppm=0.0,
            nox_ppm=0.0,
            flame_present=False,  # No flame!
            operating_mode="RUN"
        )

        events = detector.detect(state)

        flame_events = [e for e in events if e.anomaly_type == AnomalyType.FLAME_LOSS]
        assert len(flame_events) > 0
        assert flame_events[0].severity == Severity.S4  # Critical

    def test_interlock_bypass_detection(self):
        """Test interlock bypass detection"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=3.0,
            co_ppm=30.0,
            nox_ppm=20.0,
            bypass_active=True,
            operating_mode="RUN"
        )

        events = detector.detect(state)

        bypass_events = [e for e in events if e.anomaly_type == AnomalyType.INTERLOCK_BYPASS]
        assert len(bypass_events) > 0

    def test_no_detection_in_quiet_mode(self):
        """Test no detection during purge/startup"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=1.0,  # Would normally trigger rich
            co_ppm=200.0,    # Would normally trigger CO spike
            nox_ppm=30.0,
            flame_present=True,
            operating_mode="PURGE"  # Quiet mode
        )

        events = detector.detect(state)

        # Should not detect anything in purge mode
        assert len(events) == 0

    def test_anomaly_lifecycle_resolution(self):
        """Test anomaly is resolved when condition clears"""
        detector = AnomalyDetector()

        # First: Create anomaly
        high_co_state = CombustionState(
            o2_percent=3.0,
            co_ppm=200.0,
            nox_ppm=30.0,
            operating_mode="RUN"
        )
        events = detector.detect(high_co_state)
        assert len(detector.get_active_anomalies()) > 0

        # Then: Clear condition
        normal_state = CombustionState(
            o2_percent=3.0,
            co_ppm=30.0,  # Normal CO
            nox_ppm=20.0,
            operating_mode="RUN"
        )
        events = detector.detect(normal_state)

        # Check for resolved events
        resolved = [e for e in events if e.status.value == "resolved"]
        assert len(resolved) > 0

    def test_anomaly_event_structure(self):
        """Test anomaly event has required fields"""
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=3.0,
            co_ppm=200.0,
            nox_ppm=30.0,
            operating_mode="RUN"
        )

        events = detector.detect(state)

        if events:
            event = events[0]
            assert event.incident_id.startswith("INC-")
            assert event.anomaly_type is not None
            assert event.severity is not None
            assert 0 <= event.confidence <= 1
            assert event.provenance_hash is not None
            assert len(event.recommended_checks) > 0

    def test_detect_quick_helper(self):
        """Test quick detection helper function"""
        events = detect_quick(
            o2=2.0,
            co=150.0,
            nox=30.0,
            flame_intensity=80.0
        )

        assert isinstance(events, list)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""

    def test_cqi_to_explanation_flow(self):
        """Test CQI calculation flows to explanation generation"""
        # Calculate CQI
        calculator = CQICalculator(CQIConfiguration(asset_id="TestBoiler"))
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=5.0,  # High - will cause efficiency penalty
            co_ppm=80.0,
            nox_ppm=40.0,
            flame_intensity=75.0,
            operating_mode=OperatingMode.RUN
        )
        cqi_result = calculator.calculate(signals)

        # Generate explanation
        engine = ExplainabilityEngine()
        input_data = ExplainabilityInput(
            event_type=ExplanationType.CQI_DEGRADATION,
            asset_id="TestBoiler",
            current_snapshot=SignalSnapshot(
                timestamp=datetime.now(timezone.utc),
                o2_percent=signals.o2_percent,
                co_ppm=signals.co_ppm,
                nox_ppm=signals.nox_ppm
            ),
            cqi_current=cqi_result.cqi_total,
            cqi_previous=85.0
        )
        explanation = engine.explain_cqi_change(input_data)

        # Generate narrative
        generator = NarrativeGenerator()
        bundle = engine.to_evidence_bundle(explanation)
        narrative = generator.generate_from_evidence_bundle(
            EvidenceBundle(
                operating_mode=bundle["operating_mode"],
                load_context=bundle["load_context"],
                signal_deltas=bundle["signal_deltas"],
                event_type=bundle["event_type"],
                severity=bundle["severity"],
                confidence=bundle["confidence"],
                top_attributions=bundle["top_attributions"],
                time_segments=bundle["time_segments"],
                safety_status={"bypass_active": False},
                recommended_checks=["Check O2 trim", "Review damper position"]
            ),
            asset_id="TestBoiler"
        )

        # Verify end-to-end flow
        assert cqi_result.cqi_total < 85.0  # Should be degraded
        assert len(explanation.top_drivers) > 0
        assert len(narrative.narrative_text) > 0

    def test_anomaly_to_narrative_flow(self):
        """Test anomaly detection flows to narrative generation"""
        # Detect anomaly
        detector = AnomalyDetector()
        state = CombustionState(
            o2_percent=1.5,
            co_ppm=180.0,
            nox_ppm=30.0,
            flame_present=True,
            operating_mode="RUN"
        )
        events = detector.detect(state)

        assert len(events) > 0
        event = events[0]

        # Generate narrative for anomaly
        generator = NarrativeGenerator()
        input_data = NarrativeInput(
            event_type=event.anomaly_type.value,
            asset_id=event.asset_id,
            incident_id=event.incident_id,
            cqi_current=70.0,
            cqi_previous=85.0,
            o2_after=state.o2_percent,
            co_after=state.co_ppm,
            primary_driver=event.description,
            recommended_checks=event.recommended_checks
        )
        narrative = generator.generate(input_data)

        assert narrative is not None
        assert event.incident_id is not None


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior"""

    def test_cqi_determinism(self):
        """Test CQI calculation is deterministic"""
        calculator = CQICalculator()
        signals = CombustionSignals(
            fuel_flow_kg_s=1.0,
            air_flow_kg_s=17.2,
            o2_percent=3.0,
            co_ppm=50.0,
            nox_ppm=30.0,
            flame_intensity=80.0,
            operating_mode=OperatingMode.RUN
        )

        # Calculate multiple times
        results = [calculator.calculate(signals) for _ in range(5)]

        # All CQI values should be identical
        cqi_values = [r.cqi_total for r in results]
        assert all(v == cqi_values[0] for v in cqi_values)

        # All sub-scores should be identical
        for i in range(1, len(results)):
            assert results[i].sub_scores.efficiency == results[0].sub_scores.efficiency
            assert results[i].sub_scores.emissions == results[0].sub_scores.emissions

    def test_explainability_determinism(self):
        """Test explanation generation is deterministic"""
        explainer = SHAPStyleExplainer()
        current = {"o2_percent": 5.0, "co_ppm": 100.0, "nox_ppm": 50.0}

        # Generate multiple times
        results = [explainer.explain(current, -15.0) for _ in range(5)]

        # Attribution values should be identical
        for i in range(1, len(results)):
            for j in range(len(results[0])):
                assert results[i][j].attribution_value == results[0][j].attribution_value

    def test_anomaly_detection_determinism(self):
        """Test anomaly detection is deterministic"""
        config = DetectionConfig()

        # Create fresh detectors each time
        state = CombustionState(
            o2_percent=1.5,
            co_ppm=180.0,
            nox_ppm=30.0,
            operating_mode="RUN"
        )

        # Detect multiple times with fresh detectors
        results = []
        for _ in range(3):
            detector = AnomalyDetector(config)
            events = detector.detect(state)
            results.append(events)

        # Same anomaly types should be detected
        for i in range(1, len(results)):
            types_0 = {e.anomaly_type for e in results[0]}
            types_i = {e.anomaly_type for e in results[i]}
            assert types_0 == types_i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
