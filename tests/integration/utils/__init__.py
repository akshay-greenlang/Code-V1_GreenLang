"""
Integration test utilities.
"""
from .normalizers import (
    normalize_text,
    normalize_json,
    round_floats,
    strip_provenance,
    normalize_emissions,
    assert_numerical_invariants,
    normalize_report,
    compare_snapshots,
    format_diff
)

from .io import (
    TestIOHelper,
    load_fixture,
    save_snapshot,
    load_snapshot,
    validate_json_schema,
    OutputCapture
)

from .net_guard import (
    NetworkGuard,
    NetworkAccessError,
    block_network,
    mock_http_response,
    APICallRecorder,
    mock_api_endpoint
)

__all__ = [
    # Normalizers
    'normalize_text',
    'normalize_json',
    'round_floats',
    'strip_provenance',
    'normalize_emissions',
    'assert_numerical_invariants',
    'normalize_report',
    'compare_snapshots',
    'format_diff',
    
    # I/O
    'TestIOHelper',
    'load_fixture',
    'save_snapshot',
    'load_snapshot',
    'validate_json_schema',
    'OutputCapture',
    
    # Network Guard
    'NetworkGuard',
    'NetworkAccessError',
    'block_network',
    'mock_http_response',
    'APICallRecorder',
    'mock_api_endpoint'
]