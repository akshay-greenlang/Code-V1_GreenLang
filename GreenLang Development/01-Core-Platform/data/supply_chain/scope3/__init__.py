"""
Scope 3 Emission Allocation Module.

Provides emission allocation methodologies for Scope 3 reporting:
- Spend-based allocation
- Activity-based allocation
- Hybrid allocation approaches
- Category mapping (1-15)
"""

from greenlang.supply_chain.scope3.emission_allocation import (
    Scope3Allocator,
    AllocationMethod,
    EmissionAllocation,
    Scope3Category,
    SupplierEmissionProfile,
)

__all__ = [
    "Scope3Allocator",
    "AllocationMethod",
    "EmissionAllocation",
    "Scope3Category",
    "SupplierEmissionProfile",
]
