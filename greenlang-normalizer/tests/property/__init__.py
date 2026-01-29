"""
Property-Based Tests for GL-FOUND-X-003: GreenLang Unit & Reference Normalizer.

This package contains property-based tests using Hypothesis to verify
mathematical properties and invariants of the normalizer components.

Test Modules:
    - test_conversion_properties: Unit conversion mathematical properties
    - test_dimension_properties: Dimensional analysis algebraic properties
    - test_roundtrip_properties: Serialization and parsing roundtrip properties

Custom Strategies:
    - strategies.py: Custom Hypothesis strategies for valid inputs

Property Categories:
    1. Conversion Properties:
       - Roundtrip: convert(convert(x, A, B), B, A) == x
       - Transitivity: convert(x, A, C) == convert(convert(x, A, B), B, C)
       - Identity: convert(x, A, A) == x
       - Scaling: convert(k*x, A, B) == k*convert(x, A, B)

    2. Dimension Properties:
       - Commutativity: dim(A) * dim(B) == dim(B) * dim(A)
       - Associativity: (dim(A) * dim(B)) * dim(C) == dim(A) * (dim(B) * dim(C))
       - Identity: dim(A) * dimensionless == dim(A)
       - Inverse: dim(A) * dim(A)^-1 == dimensionless

    3. Roundtrip Properties:
       - Parse roundtrip: parse(format(parse(s))) == parse(s)
       - Serialization roundtrip: deserialize(serialize(obj)) == obj
       - Audit replay: replay(audit_record) == original_result
"""

__all__ = [
    "test_conversion_properties",
    "test_dimension_properties",
    "test_roundtrip_properties",
    "strategies",
]
