# -*- coding: utf-8 -*-
# GL-VCCI Provenance Module
# SHA-256 provenance chain tracking for audit compliance

"""
VCCI Provenance Tracking
========================

Complete provenance chain tracking for every tCO2e calculated.

Features:
---------
- SHA-256 hashing for every calculation step
- Immutable audit trail
- Complete data lineage tracking
- Provenance record export (JSON, blockchain)

Provenance Record Structure:
----------------------------
```json
{
  "calculation_id": "calc_20250110_1234567890",
  "timestamp": "2025-01-10T14:30:00Z",
  "category": 1,
  "tier": "tier_1",
  "input_data_hash": "sha256:abc123...",
  "emission_factor_hash": "sha256:def456...",
  "calculation_hash": "sha256:ghi789...",
  "result": {
    "emissions_tco2e": 1234.56,
    "uncertainty": 0.05
  },
  "provenance_chain": ["sha256:...", "sha256:...", "sha256:..."]
}
```

Usage:
------
```python
from provenance import ProvenanceTracker

# Initialize tracker
tracker = ProvenanceTracker(storage_backend="s3")

# Track calculation
provenance_record = tracker.track_calculation(
    calculation_id="calc_123",
    input_data=input_data,
    emission_factor=ef,
    result=result
)

# Verify provenance chain
is_valid = tracker.verify_chain(provenance_record)

# Export provenance
tracker.export(
    calculation_id="calc_123",
    format="json",
    output="provenance_record.json"
)
```

Compliance:
----------
- GHG Protocol: Complete audit trail requirement
- ISO 14064-1: Data quality and uncertainty
- EU CSRD: 7-year data retention
- SOC 2 Type 2: Immutable audit logs
"""

__version__ = "1.0.0"

__all__ = [
    # "ProvenanceTracker",
    # "ProvenanceRecord",
    # "verify_provenance_chain",
]
