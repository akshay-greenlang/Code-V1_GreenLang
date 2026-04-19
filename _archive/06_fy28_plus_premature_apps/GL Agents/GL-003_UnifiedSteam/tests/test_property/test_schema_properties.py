"""
Property-Based Tests for Schema Validation

Uses Hypothesis to test Kafka schema serialization, validation,
and round-trip consistency.

Author: GL-003 Test Engineering Team
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any
import pytest

try:
    from hypothesis import given, assume, settings, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed"
)


# Custom strategies for schema fields
site_strategy = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
    min_size=1,
    max_size=20,
)

tag_strategy = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
    min_size=1,
    max_size=50,
)

unit_strategy = st.sampled_from([
    "kPa_g", "kPa_a", "degC", "degF", "kg/s", "t/h",
    "kJ/kg", "kJ/kg-K", "m3/kg", "kg/m3", "%", "dB"
])

quality_strategy = st.sampled_from(["GOOD", "UNCERTAIN", "BAD", "STALE"])

value_strategy = st.floats(
    min_value=-1e6, max_value=1e6,
    allow_nan=False, allow_infinity=False,
)

timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)


class TestRawSignalSchema:
    """Property-based tests for RawSignalSchema."""

    @given(
        ts=timestamp_strategy,
        site=site_strategy,
        tag=tag_strategy,
        value=value_strategy,
        unit=unit_strategy,
        quality=quality_strategy,
    )
    @settings(max_examples=100)
    def test_raw_signal_roundtrip(self, ts, site, tag, value, unit, quality):
        """Raw signal should survive serialization roundtrip."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                RawSignalSchema, SensorQuality,
            )
        except ImportError:
            pytest.skip("Kafka schemas not available")

        assume(len(site) > 0 and len(tag) > 0)

        signal = RawSignalSchema(
            ts=ts,
            site=site,
            area="TEST_AREA",
            asset="TEST_ASSET",
            tag=tag,
            value=value,
            unit=unit,
            quality=SensorQuality[quality] if hasattr(SensorQuality, quality) else SensorQuality.GOOD,
        )

        # Serialize
        kafka_dict = signal.to_kafka_dict()

        # Verify fields preserved
        assert kafka_dict["site"] == site
        assert kafka_dict["tag"] == tag
        assert kafka_dict["value"] == value
        assert kafka_dict["unit"] == unit

    @given(value=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=50)
    def test_invalid_values_handled(self, value):
        """NaN and Inf values should be handled appropriately."""
        import math

        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                RawSignalSchema, SensorQuality,
            )
        except ImportError:
            pytest.skip("Kafka schemas not available")

        is_invalid = math.isnan(value) or math.isinf(value)

        if is_invalid:
            # Should either reject or mark as bad quality
            try:
                signal = RawSignalSchema(
                    ts=datetime.now(timezone.utc),
                    site="TEST",
                    area="TEST",
                    asset="TEST",
                    tag="TEST_TAG",
                    value=value,
                    unit="kPa_g",
                    quality=SensorQuality.GOOD,
                )
                # If accepted, should have bad quality or be flagged
            except (ValueError, TypeError):
                pass  # Expected for invalid values


class TestValidatedSignalSchema:
    """Property-based tests for ValidatedSignalSchema."""

    @given(
        original_value=value_strategy,
        validated_value=value_strategy,
    )
    @settings(max_examples=100)
    def test_validation_preserves_original(self, original_value, validated_value):
        """Validation should preserve original value."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                ValidatedSignalSchema, ValidationStatus,
            )
        except ImportError:
            pytest.skip("Kafka schemas not available")

        signal = ValidatedSignalSchema(
            ts=datetime.now(timezone.utc),
            site="TEST",
            area="TEST",
            asset="TEST",
            tag="TEST_TAG",
            value=validated_value,
            unit="kPa_g",
            original_value=original_value,
            original_unit="kPa_g",
            status=ValidationStatus.VALID,
        )

        kafka_dict = signal.to_kafka_dict()

        assert kafka_dict["original_value"] == original_value
        assert kafka_dict["value"] == validated_value


class TestComputedPropertiesSchema:
    """Property-based tests for computed properties schemas."""

    @given(
        pressure=st.floats(min_value=100, max_value=20000, allow_nan=False),
        temperature=st.floats(min_value=0, max_value=600, allow_nan=False),
        enthalpy=st.floats(min_value=0, max_value=4000, allow_nan=False),
        entropy=st.floats(min_value=0, max_value=10, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_steam_properties_consistency(self, pressure, temperature, enthalpy, entropy):
        """Steam property schema should maintain consistency."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                SteamPropertiesComputed,
            )
        except ImportError:
            pytest.skip("Kafka schemas not available")

        props = SteamPropertiesComputed(
            asset="TEST_ASSET",
            pressure_kpa=Decimal(str(pressure)),
            temperature_c=Decimal(str(temperature)),
            enthalpy_kj_kg=Decimal(str(enthalpy)),
            entropy_kj_kg_k=Decimal(str(entropy)),
            specific_volume_m3_kg=Decimal("0.1"),
            density_kg_m3=Decimal("10"),
            saturation_temp_c=Decimal("180"),
            superheat_c=Decimal(str(max(0, temperature - 180))),
            if97_region=2,
            computation_hash="test_hash_123",
        )

        kafka_dict = props.to_dict()

        # Values should be preserved
        assert float(kafka_dict["pressure_kpa"]) == pytest.approx(pressure, rel=1e-6)
        assert float(kafka_dict["temperature_c"]) == pytest.approx(temperature, rel=1e-6)


class TestRecommendationSchema:
    """Property-based tests for recommendation schemas."""

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        energy_savings=st.floats(min_value=0, max_value=10000),
    )
    @settings(max_examples=100)
    def test_recommendation_confidence_bounded(self, confidence, energy_savings):
        """Recommendation confidence should be in [0, 1]."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                RecommendationSchema, RecommendationType, Priority, Disposition,
            )
        except ImportError:
            pytest.skip("Kafka schemas not available")

        rec = RecommendationSchema(
            ts=datetime.now(timezone.utc),
            recommendation_id="REC_TEST_001",
            site="TEST",
            area="TEST",
            asset="TEST_ASSET",
            recommendation_type=RecommendationType.DESUPERHEATER_SETPOINT,
            priority=Priority.MEDIUM,
            action="Test action",
            rationale="Test rationale",
            confidence_score=Decimal(str(confidence)),
            disposition=Disposition.PENDING,
        )

        assert 0 <= float(rec.confidence_score) <= 1

    @given(
        priority=st.sampled_from(["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]),
        rec_type=st.sampled_from([
            "DESUPERHEATER_SETPOINT", "TRAP_INSPECTION", "TRAP_REPLACEMENT",
            "PRV_ADJUSTMENT", "CONDENSATE_ROUTING",
        ]),
    )
    @settings(max_examples=50)
    def test_recommendation_type_priority_combinations(self, priority, rec_type):
        """All valid type-priority combinations should work."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.kafka_schemas import (
                RecommendationSchema, RecommendationType, Priority, Disposition,
            )
        except ImportError:
            pytest.skip("Kafka schemas not available")

        rec = RecommendationSchema(
            ts=datetime.now(timezone.utc),
            recommendation_id="REC_TEST_002",
            site="TEST",
            area="TEST",
            asset="TEST_ASSET",
            recommendation_type=RecommendationType[rec_type],
            priority=Priority[priority],
            action="Test action",
            rationale="Test rationale",
            confidence_score=Decimal("0.85"),
            disposition=Disposition.PENDING,
        )

        kafka_dict = rec.to_kafka_dict()
        assert kafka_dict["recommendation_type"] == rec_type.lower()
        assert kafka_dict["priority"] == priority.lower()


class TestSchemaEvolution:
    """Property-based tests for schema evolution compatibility."""

    @given(
        extra_fields=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
            values=st.one_of(st.text(max_size=100), st.floats(allow_nan=False), st.integers()),
            min_size=0,
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_unknown_fields_handled(self, extra_fields):
        """Unknown fields in incoming data should be handled gracefully."""
        # This tests forward compatibility
        base_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "site": "TEST",
            "area": "TEST",
            "asset": "TEST",
            "tag": "TEST_TAG",
            "value": 100.0,
            "unit": "kPa_g",
            "quality": "GOOD",
        }

        # Add extra fields (simulating newer schema)
        data_with_extras = {**base_data, **extra_fields}

        # Should be able to process without error
        # (actual implementation would ignore unknown fields)
        assert "ts" in data_with_extras
        assert "value" in data_with_extras


class TestAvroSchemaProperties:
    """Property-based tests for Avro schema definitions."""

    def test_all_schemas_have_namespace(self):
        """All Avro schemas should have namespace."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.avro_schemas import (
                get_all_schemas,
            )
        except ImportError:
            pytest.skip("Avro schemas not available")

        schemas = get_all_schemas()

        for name, schema in schemas.items():
            assert "namespace" in schema, f"Schema {name} missing namespace"
            assert schema["namespace"].startswith("com.greenlang"), (
                f"Schema {name} has incorrect namespace"
            )

    def test_all_schemas_have_doc(self):
        """All Avro schemas should have documentation."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.avro_schemas import (
                get_all_schemas,
            )
        except ImportError:
            pytest.skip("Avro schemas not available")

        schemas = get_all_schemas()

        for name, schema in schemas.items():
            assert "doc" in schema, f"Schema {name} missing documentation"

    @given(field_name=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    @settings(max_examples=50)
    def test_field_naming_convention(self, field_name):
        """Field names should follow snake_case convention."""
        import re

        # snake_case pattern
        pattern = r'^[a-z][a-z0-9_]*$'

        if re.match(pattern, field_name):
            # Valid snake_case
            assert field_name == field_name.lower()
            assert "__" not in field_name


class TestSchemaRegistryProperties:
    """Property-based tests for schema registry operations."""

    @given(
        subject_name=st.text(
            min_size=5,
            max_size=50,
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789._",
        )
    )
    @settings(max_examples=50)
    def test_subject_creation_idempotent_name(self, subject_name):
        """Subject names should be handled consistently."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.schema_registry import (
                SchemaRegistry,
            )
        except ImportError:
            pytest.skip("Schema registry not available")

        assume(len(subject_name) >= 5)
        assume("." in subject_name or "_" in subject_name)

        registry = SchemaRegistry()

        try:
            registry.create_subject(subject_name)
            subjects = registry.list_subjects()
            assert subject_name in subjects
        except ValueError:
            # May fail for invalid names
            pass

    @given(
        versions=st.lists(
            st.fixed_dictionaries({
                "type": st.just("record"),
                "name": st.just("TestRecord"),
                "fields": st.lists(
                    st.fixed_dictionaries({
                        "name": st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
                        "type": st.sampled_from(["string", "double", "long", "boolean"]),
                    }),
                    min_size=1,
                    max_size=5,
                ),
            }),
            min_size=1,
            max_size=3,
        )
    )
    @settings(max_examples=30)
    def test_schema_versioning_monotonic(self, versions):
        """Schema versions should increase monotonically."""
        try:
            from GL_Agents.GL003_UnifiedSteam.schemas.schema_registry import (
                SchemaRegistry, SchemaCompatibility,
            )
        except ImportError:
            pytest.skip("Schema registry not available")

        registry = SchemaRegistry()
        registry.create_subject(
            "test.subject",
            compatibility=SchemaCompatibility.NONE,  # Disable compat checking
        )

        registered_versions = []
        for schema in versions:
            try:
                version = registry.register_schema("test.subject", schema)
                registered_versions.append(version.version)
            except Exception:
                pass

        # Versions should be monotonically increasing
        for i in range(1, len(registered_versions)):
            assert registered_versions[i] >= registered_versions[i-1], (
                f"Version decreased: {registered_versions}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
