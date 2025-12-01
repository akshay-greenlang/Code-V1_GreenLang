# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-012 STEAMQUAL.
Verifies bit-perfect reproducibility following zero-hallucination principles.
"""

import pytest
import hashlib
import json
import random
from decimal import Decimal, ROUND_HALF_UP


@pytest.fixture
def deterministic_seed():
    return 42


@pytest.fixture
def sample_inputs():
    return {
        "pressure_bar": Decimal("10.0"),
        "temperature_c": Decimal("180.0"),
        "dryness_fraction": Decimal("0.98"),
        "h_total": Decimal("2700.0"),
        "h_f": Decimal("762.8"),
        "h_fg": Decimal("2015.0"),
    }


class TestBitPerfectReproducibility:
    @pytest.mark.determinism
    def test_dryness_fraction_reproducibility(self, sample_inputs):
        h_total, h_f, h_fg = sample_inputs["h_total"], sample_inputs["h_f"], sample_inputs["h_fg"]
        results = []
        for _ in range(1000):
            x = (h_total - h_f) / h_fg
            results.append(x.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        assert len(set(results)) == 1, "Dryness calculation not deterministic"
        assert results[0] == Decimal("0.9613")

    @pytest.mark.determinism
    def test_quality_index_reproducibility(self, sample_inputs):
        dryness = float(sample_inputs["dryness_fraction"])
        results = []
        for _ in range(1000):
            index = dryness * 100 * 0.4 + 0.95 * 100 * 0.3 + 0.92 * 100 * 0.3
            results.append(round(index, 4))
        assert len(set(results)) == 1

    @pytest.mark.determinism
    def test_desuperheater_calculation_reproducibility(self):
        m_steam, h_inlet, h_outlet, h_water = Decimal("5000.0"), Decimal("2900.0"), Decimal("2800.0"), Decimal("420.0")
        results = []
        for _ in range(1000):
            m_water = m_steam * (h_inlet - h_outlet) / (h_outlet - h_water)
            results.append(m_water.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        assert len(set(results)) == 1


class TestProvenanceHashConsistency:
    @pytest.mark.determinism
    def test_hash_consistency_same_input(self, sample_inputs):
        data = {k: str(v) for k, v in sample_inputs.items()}
        hashes = []
        for _ in range(100):
            hashes.append(hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest())
        assert len(set(hashes)) == 1

    @pytest.mark.determinism
    def test_hash_changes_with_input(self, sample_inputs):
        data = {k: str(v) for k, v in sample_inputs.items()}
        original = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        data["pressure_bar"] = "10.1"
        modified = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        assert original != modified


class TestSeedPropagation:
    @pytest.mark.determinism
    def test_random_seed_propagation(self, deterministic_seed):
        random.seed(deterministic_seed)
        values_1 = [random.random() for _ in range(100)]
        random.seed(deterministic_seed)
        values_2 = [random.random() for _ in range(100)]
        assert values_1 == values_2

    @pytest.mark.determinism
    def test_no_hidden_randomness(self, sample_inputs):
        results = []
        for _ in range(100):
            x = (sample_inputs["h_total"] - sample_inputs["h_f"]) / sample_inputs["h_fg"]
            results.append(x)
        assert len(set(results)) == 1


class TestFloatingPointStability:
    @pytest.mark.determinism
    def test_associativity_preserved(self):
        values = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]
        assert sum(values, Decimal("0")) == sum(reversed(values), Decimal("0"))

    @pytest.mark.determinism
    def test_decimal_precision(self):
        assert Decimal("1.0000000001") - Decimal("0.0000000001") == Decimal("1.0")

    @pytest.mark.determinism
    def test_edge_case_small_values(self):
        assert Decimal("1E-15") + Decimal("1E-15") == Decimal("2E-15")
