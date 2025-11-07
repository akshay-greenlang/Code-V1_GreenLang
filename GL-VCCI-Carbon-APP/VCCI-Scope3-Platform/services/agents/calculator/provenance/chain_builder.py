"""
Provenance Chain Builder
Complete tracking of calculation provenance for auditability.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..models import (
    ProvenanceChain,
    DataQualityInfo,
    EmissionFactorInfo,
)
from ..config import TierType
from .hash_utils import hash_data, hash_factor_info

logger = logging.getLogger(__name__)


class ProvenanceChainBuilder:
    """
    Builder for complete provenance chains.

    Every calculation must produce a full provenance chain for:
    - Audit trail
    - Reproducibility
    - Data lineage
    - Quality assurance
    """

    def __init__(self):
        """Initialize provenance builder."""
        logger.info("Initialized ProvenanceChainBuilder")

    async def build(
        self,
        category: int,
        tier: Optional[TierType],
        input_data: Dict[str, Any],
        emission_factor: Optional[EmissionFactorInfo],
        calculation: Dict[str, Any],
        data_quality: DataQualityInfo,
    ) -> ProvenanceChain:
        """
        Build complete provenance chain.

        Args:
            category: Scope 3 category
            tier: Calculation tier
            input_data: Input data dictionary
            emission_factor: Emission factor info
            calculation: Calculation details
            data_quality: Data quality info

        Returns:
            ProvenanceChain
        """
        # Generate unique calculation ID
        calc_id = self._generate_calculation_id(category)

        # Hash input data
        input_hash = hash_data(input_data)

        # Build provenance chain (list of hashes)
        chain = [input_hash]

        if emission_factor:
            chain.append(emission_factor.hash)

        calc_hash = hash_data(calculation)
        chain.append(calc_hash)

        # OpenTelemetry trace ID (if available)
        trace_id = self._get_trace_id()

        provenance = ProvenanceChain(
            calculation_id=calc_id,
            timestamp=datetime.utcnow(),
            category=category,
            tier=tier,
            input_data_hash=input_hash,
            emission_factor=emission_factor,
            calculation=calculation,
            data_quality=data_quality,
            provenance_chain=chain,
            opentelemetry_trace_id=trace_id
        )

        logger.debug(
            f"Built provenance chain for {calc_id}: "
            f"{len(chain)} hashes, tier={tier}"
        )

        return provenance

    def _generate_calculation_id(self, category: int) -> str:
        """Generate unique calculation ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"calc_cat{category}_{timestamp}_{unique_id}"

    def _get_trace_id(self) -> Optional[str]:
        """Get OpenTelemetry trace ID if available."""
        # In production, this would extract from OpenTelemetry context
        # For now, generate a placeholder
        try:
            trace_id = str(uuid.uuid4())
            return f"trace_{trace_id}"
        except:
            return None

    def hash_factor_info(self, value: float, source: str) -> str:
        """Hash emission factor info."""
        return hash_factor_info(value, source)


__all__ = ["ProvenanceChainBuilder"]
