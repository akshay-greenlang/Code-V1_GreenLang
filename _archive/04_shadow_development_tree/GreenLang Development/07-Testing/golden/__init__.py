"""
GreenLang Golden Test Suite
===========================

Golden tests validate agent outputs against known-correct reference data.
These tests are critical for regulatory compliance (EUDR, CBAM, CSRD).

Features:
- JSON-based test cases with expected inputs/outputs
- Tolerance checking for floating-point calculations (+/-1%)
- Hash verification for provenance tracking
- Reproducibility validation (same input = same output)
"""

__version__ = "1.0.0"
