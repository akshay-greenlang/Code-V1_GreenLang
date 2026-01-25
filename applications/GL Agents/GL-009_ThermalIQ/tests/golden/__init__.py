"""
Golden Value Tests for GL-009 ThermalIQ
========================================

This package contains golden value tests that validate thermodynamic
calculations against authoritative reference sources:

- NIST Chemistry WebBook
- IAPWS-IF97 (Industrial Formulation for Water and Steam)
- REFPROP 10.0 (Reference Fluid Properties)

Test Modules:
- test_nist_reference.py: NIST/REFPROP validation tests

Markers:
- @pytest.mark.golden: All golden value tests
- @pytest.mark.nist: Tests using NIST reference data
- @pytest.mark.refprop: Tests using REFPROP reference data

Usage:
    pytest tests/golden/ -v -m golden
    pytest tests/golden/ -v -m nist
    pytest tests/golden/test_nist_reference.py -v
"""
