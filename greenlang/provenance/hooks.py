"""
GreenLang Provenance Hooks - SIM-401 Compliant Module

This module provides provenance hooks for scenario execution and seed tracking
as specified in SIM-401. The actual implementation is in greenlang.provenance.utils
for consistency with the existing provenance system, but this module provides
the spec-compliant import path.

Usage:
    from greenlang.provenance.hooks import record_seed_info, ProvenanceContext

    # Create provenance context
    ctx = ProvenanceContext(name="my_scenario")

    # Record seed information
    record_seed_info(
        ctx=ctx,
        spec=scenario_dict,
        seed_root=42,
        seed_path="scenario:test|param:x|trial:0",
        seed_child=9876543210,
        spec_type="scenario"
    )

    # Finalize provenance
    ledger_path = ctx.finalize()

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

# Import all required APIs from the actual implementation
from greenlang.provenance.utils import (
    ProvenanceContext,
    record_seed_info,
    track_provenance,
)

__all__ = [
    "ProvenanceContext",
    "record_seed_info",
    "track_provenance",
]
