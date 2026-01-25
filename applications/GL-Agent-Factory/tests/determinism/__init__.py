"""
Determinism Tests for GreenLang Agents

This package contains tests that validate zero-hallucination determinism:
- Same input always produces same output
- Provenance hashes are reproducible
- Float precision is consistent
- No random or non-deterministic behavior

Test Files:
- test_seed_reproducibility.py: Tests that same inputs produce identical results
- test_hash_uniqueness.py: Tests provenance hash properties
- test_float_precision.py: Tests numerical precision handling
"""
