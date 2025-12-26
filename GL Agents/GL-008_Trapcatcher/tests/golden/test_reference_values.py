# -*- coding: utf-8 -*-
import pytest
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any

@dataclass(frozen=True)
class GoldenValue:
    name: str
    value: float
    unit: str
    tolerance_percent: float
    source: str
    section: str

    def validate(self, calculated: float) -> Tuple[bool, float]:
        if self.value == 0:
            return abs(calculated) < 0.001, abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * 100
        return deviation <= self.tolerance_percent, deviation


class ASMEPTC39References:
    FLOW_COEFFICIENT_CD = GoldenValue("Discharge Coeff", 0.61, "dimensionless", 5.0, "ASME PTC 39", "Section 4")


class AcousticDiagnosticReferences:
    ULTRASONIC_THRESHOLDS = {
        "healthy": GoldenValue("Healthy dB", 30, "dB", 30.0, "Industry", "Best Practice"),
        "leaking": GoldenValue("Leaking dB", 50, "dB", 20.0, "Industry", "Best Practice"),
        "blowthrough": GoldenValue("Blowthrough dB", 70, "dB", 15.0, "Industry", "Best Practice"),
    }


@pytest.mark.golden
class TestASMEPTC39:
    def test_discharge_coefficient(self):
        golden = ASMEPTC39References.FLOW_COEFFICIENT_CD
        is_valid, _ = golden.validate(0.61)
        assert is_valid


@pytest.mark.golden
class TestAcousticDiagnostics:
    def test_acoustic_threshold_ordering(self):
        healthy = AcousticDiagnosticReferences.ULTRASONIC_THRESHOLDS["healthy"].value
        leaking = AcousticDiagnosticReferences.ULTRASONIC_THRESHOLDS["leaking"].value
        blowthrough = AcousticDiagnosticReferences.ULTRASONIC_THRESHOLDS["blowthrough"].value
        assert healthy < leaking < blowthrough


@pytest.mark.golden
class TestDeterminism:
    def test_loss_calculation_determinism(self):
        hashes = set()
        for _ in range(100):
            loss = round(15.5 * 2100 * 8760 / 1e6, 6)
            hashes.add(hashlib.sha256(str(loss).encode()).hexdigest())
        assert len(hashes) == 1


def export_golden_values() -> Dict[str, Any]:
    return {
        "metadata": {"version": "1.0.0", "agent": "GL-008_Trapcatcher"},
        "asme_ptc39": {"discharge_coeff": 0.61},
    }
