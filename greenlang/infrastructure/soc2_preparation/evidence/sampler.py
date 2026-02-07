# -*- coding: utf-8 -*-
"""
Population Sampler - SEC-009 Phase 3

AICPA-compliant audit sampling for SOC 2 Type II testing.

This module implements statistical sampling methodologies based on
AICPA Audit Sampling guidance for service organization controls.

Sample Size Tables (AICPA Guidance):
    - Weekly controls: 1-2 samples for 25 occurrences, up to 25 for 250+
    - Monthly controls: 2-3 samples for 12-24 occurrences
    - Quarterly controls: 2 samples for 4 occurrences
    - Annual controls: 1 sample for single occurrence
    - On-demand controls: Variable based on population size

Sampling Methodologies:
    - Random: Simple random selection
    - Systematic: Every nth item selection
    - Stratified: Proportional selection across strata
    - Haphazard: Selection without bias pattern

Example:
    >>> sampler = PopulationSampler()
    >>> population = [...]  # List of change tickets
    >>> sample = sampler.generate_sample(
    ...     population,
    ...     control_frequency="weekly",
    ... )
    >>> doc = sampler.document_sampling(
    ...     population_size=len(population),
    ...     sample_size=len(sample),
    ...     methodology="random",
    ... )

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import random
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.soc2_preparation.evidence.models import (
    ControlFrequency,
    SamplingResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# AICPA Sample Size Tables
# ---------------------------------------------------------------------------

# Sample sizes based on AICPA Audit Sampling guidance
# Format: {population_threshold: sample_size}
# For each frequency, select sample size based on first threshold >= population

SAMPLE_SIZES: Dict[str, Dict[int, int]] = {
    # Weekly controls (52 occurrences per year)
    "weekly": {
        25: 1,      # 1-25 occurrences: sample 1
        52: 2,      # 26-52 occurrences: sample 2
        100: 15,    # 53-100 occurrences: sample 15
        250: 25,    # 101-250 occurrences: sample 25
        500: 40,    # 251-500 occurrences: sample 40
        1000: 45,   # 501-1000 occurrences: sample 45
    },

    # Monthly controls (12 occurrences per year)
    "monthly": {
        12: 2,      # 1-12 occurrences: sample 2
        24: 3,      # 13-24 occurrences: sample 3
        36: 5,      # 25-36 occurrences: sample 5
        48: 7,      # 37-48 occurrences: sample 7
    },

    # Quarterly controls (4 occurrences per year)
    "quarterly": {
        4: 2,       # 1-4 occurrences: sample 2
        8: 3,       # 5-8 occurrences: sample 3
        12: 4,      # 9-12 occurrences: sample 4
    },

    # Annual controls (1 occurrence per year)
    "annual": {
        1: 1,       # Single occurrence: sample all
    },

    # On-demand/irregular controls
    "on_demand": {
        25: 1,      # 1-25 occurrences: sample 1
        50: 3,      # 26-50 occurrences: sample 3
        100: 8,     # 51-100 occurrences: sample 8
        250: 15,    # 101-250 occurrences: sample 15
        500: 25,    # 251-500 occurrences: sample 25
        1000: 45,   # 501-1000 occurrences: sample 45
        2500: 60,   # 1001-2500 occurrences: sample 60
        5000: 75,   # 2501-5000 occurrences: sample 75
    },

    # Continuous controls - same as on_demand for sampling purposes
    "continuous": {
        25: 1,
        50: 3,
        100: 8,
        250: 15,
        500: 25,
        1000: 45,
        2500: 60,
        5000: 75,
    },

    # Hourly controls (high-frequency)
    "hourly": {
        100: 5,
        500: 15,
        1000: 25,
        5000: 45,
        10000: 60,
    },

    # Daily controls
    "daily": {
        30: 2,      # ~1 month
        90: 5,      # ~3 months
        180: 10,    # ~6 months
        365: 25,    # ~1 year
        730: 40,    # ~2 years
    },
}


class SamplingMethodology(str, Enum):
    """Sampling methodology types."""

    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    HAPHAZARD = "haphazard"
    BLOCK = "block"
    MONETARY_UNIT = "monetary_unit"


# ---------------------------------------------------------------------------
# Population Sampler
# ---------------------------------------------------------------------------


class PopulationSampler:
    """AICPA-compliant audit sampling for SOC 2 testing.

    Implements multiple sampling methodologies and generates
    audit-ready documentation of sampling procedures.

    Example:
        >>> sampler = PopulationSampler()
        >>> sample = sampler.generate_sample(
        ...     population=access_requests,
        ...     control_frequency="weekly",
        ... )
        >>> sampler.export_population(population, "access_requests.csv")
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the sampler.

        Args:
            seed: Random seed for reproducibility (optional).
        """
        self._seed = seed
        if seed is not None:
            random.seed(seed)

    def get_sample_size(
        self,
        population_size: int,
        control_frequency: str,
    ) -> int:
        """Determine appropriate sample size based on AICPA guidance.

        Args:
            population_size: Total items in population.
            control_frequency: Frequency of control operation.

        Returns:
            Recommended sample size.
        """
        frequency = control_frequency.lower().replace("-", "_")

        if frequency not in SAMPLE_SIZES:
            # Default to on_demand for unknown frequencies
            logger.warning(
                f"Unknown frequency '{frequency}', using 'on_demand' sampling"
            )
            frequency = "on_demand"

        size_table = SAMPLE_SIZES[frequency]

        # Find appropriate sample size
        for threshold in sorted(size_table.keys()):
            if population_size <= threshold:
                return size_table[threshold]

        # If population exceeds all thresholds, use largest sample
        return max(size_table.values())

    def generate_sample(
        self,
        population: List[T],
        control_frequency: str,
        methodology: SamplingMethodology = SamplingMethodology.RANDOM,
        sample_size_override: Optional[int] = None,
        stratify_by: Optional[Callable[[T], str]] = None,
    ) -> List[T]:
        """Generate a sample from the population.

        Args:
            population: Full population to sample from.
            control_frequency: Frequency of the control.
            methodology: Sampling methodology to use.
            sample_size_override: Override automatic sample size calculation.
            stratify_by: Function to extract stratum key (for stratified sampling).

        Returns:
            List of sampled items.
        """
        if not population:
            return []

        population_size = len(population)

        # Determine sample size
        if sample_size_override is not None:
            sample_size = sample_size_override
        else:
            sample_size = self.get_sample_size(population_size, control_frequency)

        # Don't sample more than population
        sample_size = min(sample_size, population_size)

        # Apply sampling methodology
        if methodology == SamplingMethodology.RANDOM:
            sample = self._random_sample(population, sample_size)
        elif methodology == SamplingMethodology.SYSTEMATIC:
            sample = self._systematic_sample(population, sample_size)
        elif methodology == SamplingMethodology.STRATIFIED:
            sample = self._stratified_sample(
                population, sample_size, stratify_by
            )
        elif methodology == SamplingMethodology.HAPHAZARD:
            sample = self._haphazard_sample(population, sample_size)
        elif methodology == SamplingMethodology.BLOCK:
            sample = self._block_sample(population, sample_size)
        else:
            # Default to random
            sample = self._random_sample(population, sample_size)

        logger.info(
            f"Generated {len(sample)} samples from {population_size} items "
            f"using {methodology.value} methodology"
        )

        return sample

    def _random_sample(
        self,
        population: List[T],
        sample_size: int,
    ) -> List[T]:
        """Simple random sampling.

        Args:
            population: Population to sample from.
            sample_size: Number of items to select.

        Returns:
            Random sample.
        """
        return random.sample(population, sample_size)

    def _systematic_sample(
        self,
        population: List[T],
        sample_size: int,
    ) -> List[T]:
        """Systematic sampling (every nth item).

        Args:
            population: Population to sample from.
            sample_size: Number of items to select.

        Returns:
            Systematic sample.
        """
        if sample_size >= len(population):
            return list(population)

        # Calculate interval
        interval = len(population) // sample_size

        # Random start point
        start = random.randint(0, interval - 1) if interval > 0 else 0

        # Select every nth item
        sample: List[T] = []
        for i in range(sample_size):
            idx = (start + i * interval) % len(population)
            sample.append(population[idx])

        return sample

    def _stratified_sample(
        self,
        population: List[T],
        sample_size: int,
        stratify_by: Optional[Callable[[T], str]] = None,
    ) -> List[T]:
        """Stratified sampling (proportional across strata).

        Args:
            population: Population to sample from.
            sample_size: Total number of items to select.
            stratify_by: Function to extract stratum key.

        Returns:
            Stratified sample.
        """
        if stratify_by is None:
            # Fall back to random sampling
            return self._random_sample(population, sample_size)

        # Group by stratum
        strata: Dict[str, List[T]] = {}
        for item in population:
            stratum = stratify_by(item)
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(item)

        # Calculate proportional sample sizes
        population_size = len(population)
        sample: List[T] = []

        for stratum, items in strata.items():
            # Proportional allocation
            stratum_size = max(
                1,
                round(len(items) / population_size * sample_size),
            )
            stratum_size = min(stratum_size, len(items))
            sample.extend(random.sample(items, stratum_size))

        # Adjust if we have too few or too many
        if len(sample) < sample_size:
            remaining = [
                item for item in population if item not in sample
            ]
            if remaining:
                additional = min(sample_size - len(sample), len(remaining))
                sample.extend(random.sample(remaining, additional))
        elif len(sample) > sample_size:
            sample = random.sample(sample, sample_size)

        return sample

    def _haphazard_sample(
        self,
        population: List[T],
        sample_size: int,
    ) -> List[T]:
        """Haphazard sampling (pseudo-random without statistical rigor).

        For audit purposes, this is similar to random but documents
        that a less formal selection process was used.

        Args:
            population: Population to sample from.
            sample_size: Number of items to select.

        Returns:
            Haphazard sample.
        """
        # Use hash-based selection for reproducibility
        scored = []
        for i, item in enumerate(population):
            # Create a score based on item hash
            item_str = str(item) if not isinstance(item, str) else item
            score = int(hashlib.md5(item_str.encode()).hexdigest()[:8], 16)
            scored.append((score, i, item))

        # Sort by score and take top N
        scored.sort(key=lambda x: x[0])
        return [item for _, _, item in scored[:sample_size]]

    def _block_sample(
        self,
        population: List[T],
        sample_size: int,
    ) -> List[T]:
        """Block sampling (consecutive items).

        Args:
            population: Population to sample from.
            sample_size: Number of items to select.

        Returns:
            Block sample.
        """
        if sample_size >= len(population):
            return list(population)

        # Random start point
        max_start = len(population) - sample_size
        start = random.randint(0, max_start)

        return population[start : start + sample_size]

    def export_population(
        self,
        population: List[Any],
        filename: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Export population to CSV file.

        Args:
            population: Population items to export.
            filename: Output filename (without path).
            output_dir: Directory for output (optional, uses temp if not set).

        Returns:
            Path to exported file.
        """
        if not population:
            return ""

        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / filename
        else:
            import tempfile
            output_path = Path(tempfile.gettempdir()) / filename

        # Ensure .csv extension
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")

        # Get field names from first item
        first_item = population[0]

        if isinstance(first_item, dict):
            fieldnames = list(first_item.keys())
            rows = population
        elif hasattr(first_item, "__dict__"):
            fieldnames = list(first_item.__dict__.keys())
            rows = [vars(item) for item in population]
        elif hasattr(first_item, "model_dump"):
            # Pydantic model
            fieldnames = list(first_item.model_dump().keys())
            rows = [item.model_dump() for item in population]
        else:
            # Simple values
            fieldnames = ["value"]
            rows = [{"value": str(item)} for item in population]

        # Write CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Convert values to strings for CSV
                str_row = {k: str(v) if v is not None else "" for k, v in row.items()}
                writer.writerow(str_row)

        logger.info(f"Exported population of {len(population)} items to {output_path}")

        return str(output_path)

    def document_sampling(
        self,
        population_size: int,
        sample_size: int,
        methodology: str,
        control_frequency: Optional[str] = None,
        control_description: Optional[str] = None,
        auditor_notes: Optional[str] = None,
    ) -> str:
        """Generate audit documentation for sampling.

        Args:
            population_size: Total items in population.
            sample_size: Number of items sampled.
            methodology: Sampling methodology used.
            control_frequency: Frequency of the control.
            control_description: Description of the control being tested.
            auditor_notes: Additional notes for auditors.

        Returns:
            Formatted documentation string.
        """
        # Calculate expected sample size
        if control_frequency:
            expected_size = self.get_sample_size(population_size, control_frequency)
        else:
            expected_size = sample_size

        # Sample rate
        sample_rate = (sample_size / population_size * 100) if population_size > 0 else 0

        doc_lines = [
            "=" * 70,
            "AUDIT SAMPLING DOCUMENTATION",
            "=" * 70,
            "",
            f"Documentation Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Documentation ID: {uuid4()}",
            "",
            "-" * 70,
            "POPULATION DETAILS",
            "-" * 70,
            f"Population Size: {population_size:,}",
        ]

        if control_frequency:
            doc_lines.append(f"Control Frequency: {control_frequency}")

        if control_description:
            doc_lines.append(f"Control Description: {control_description}")

        doc_lines.extend([
            "",
            "-" * 70,
            "SAMPLING METHODOLOGY",
            "-" * 70,
            f"Methodology: {methodology.upper()}",
            f"Expected Sample Size (AICPA): {expected_size}",
            f"Actual Sample Size: {sample_size}",
            f"Sample Rate: {sample_rate:.2f}%",
            "",
        ])

        # Methodology description
        methodology_descriptions = {
            "random": (
                "Random sampling was performed using a statistically random "
                "selection process. Each item in the population had an equal "
                "probability of being selected."
            ),
            "systematic": (
                "Systematic sampling was performed by selecting every Nth item "
                "from the population, with a random starting point."
            ),
            "stratified": (
                "Stratified sampling was performed by dividing the population "
                "into subgroups (strata) and sampling proportionally from each."
            ),
            "haphazard": (
                "Haphazard sampling was performed by selecting items without "
                "a statistically random process, but with care to avoid bias."
            ),
            "block": (
                "Block sampling was performed by selecting a consecutive "
                "block of items from the population."
            ),
        }

        method_desc = methodology_descriptions.get(
            methodology.lower(),
            "Sampling methodology as specified.",
        )
        doc_lines.append(f"Methodology Description:")
        doc_lines.append(f"  {method_desc}")
        doc_lines.append("")

        # AICPA reference
        doc_lines.extend([
            "-" * 70,
            "AICPA GUIDANCE REFERENCE",
            "-" * 70,
            "Sample sizes are based on AICPA Audit Sampling guidance for",
            "service organization controls. The sample size table considers:",
            "  - Population size",
            "  - Control frequency (annual, quarterly, monthly, weekly, etc.)",
            "  - Expected deviation rate (assumed low for operational controls)",
            "  - Desired confidence level (typically 90-95%)",
            "",
        ])

        # Deviation analysis
        if sample_size < expected_size:
            doc_lines.extend([
                "-" * 70,
                "DEVIATION FROM EXPECTED SAMPLE SIZE",
                "-" * 70,
                f"Note: Actual sample size ({sample_size}) is less than expected ({expected_size}).",
                "Auditor should document justification for reduced sample size.",
                "",
            ])
        elif sample_size > expected_size:
            doc_lines.extend([
                "-" * 70,
                "EXTENDED SAMPLE SIZE",
                "-" * 70,
                f"Note: Actual sample size ({sample_size}) exceeds expected ({expected_size}).",
                "Extended sample provides additional assurance.",
                "",
            ])

        # Auditor notes
        if auditor_notes:
            doc_lines.extend([
                "-" * 70,
                "AUDITOR NOTES",
                "-" * 70,
                auditor_notes,
                "",
            ])

        # Sign-off section
        doc_lines.extend([
            "-" * 70,
            "ATTESTATION",
            "-" * 70,
            "The sampling methodology described above was performed in accordance",
            "with AICPA professional standards for audit sampling.",
            "",
            "Prepared By: ___________________________ Date: _______________",
            "",
            "Reviewed By: ___________________________ Date: _______________",
            "",
            "=" * 70,
        ])

        return "\n".join(doc_lines)

    def generate_sampling_result(
        self,
        population: List[Any],
        control_frequency: str,
        population_name: str,
        methodology: SamplingMethodology = SamplingMethodology.RANDOM,
    ) -> SamplingResult:
        """Generate a complete sampling result with documentation.

        Args:
            population: Population to sample from.
            control_frequency: Control frequency.
            population_name: Human-readable name for the population.
            methodology: Sampling methodology.

        Returns:
            Complete SamplingResult object.
        """
        # Generate sample
        sample = self.generate_sample(
            population=population,
            control_frequency=control_frequency,
            methodology=methodology,
        )

        # Generate documentation
        documentation = self.document_sampling(
            population_size=len(population),
            sample_size=len(sample),
            methodology=methodology.value,
            control_frequency=control_frequency,
        )

        # Map control frequency to enum
        freq_map = {
            "continuous": ControlFrequency.CONTINUOUS,
            "hourly": ControlFrequency.HOURLY,
            "daily": ControlFrequency.DAILY,
            "weekly": ControlFrequency.WEEKLY,
            "monthly": ControlFrequency.MONTHLY,
            "quarterly": ControlFrequency.QUARTERLY,
            "annual": ControlFrequency.ANNUAL,
            "on_demand": ControlFrequency.ON_DEMAND,
        }

        control_freq = freq_map.get(
            control_frequency.lower().replace("-", "_"),
            ControlFrequency.ON_DEMAND,
        )

        return SamplingResult(
            population_id=uuid4(),
            population_name=population_name,
            control_frequency=control_freq,
            population_size=len(population),
            sample_size=len(sample),
            sample_items=sample,
            methodology=methodology.value,
            documentation=documentation,
            created_at=datetime.now(timezone.utc),
        )

    def validate_sample_coverage(
        self,
        population_size: int,
        sample_size: int,
        control_frequency: str,
    ) -> Dict[str, Any]:
        """Validate that sample size meets AICPA requirements.

        Args:
            population_size: Total items in population.
            sample_size: Actual sample size.
            control_frequency: Control frequency.

        Returns:
            Validation result dictionary.
        """
        expected_size = self.get_sample_size(population_size, control_frequency)

        result = {
            "population_size": population_size,
            "sample_size": sample_size,
            "expected_sample_size": expected_size,
            "control_frequency": control_frequency,
            "meets_aicpa_requirements": sample_size >= expected_size,
            "sample_rate": round(sample_size / population_size * 100, 2) if population_size > 0 else 0,
        }

        if sample_size < expected_size:
            result["deviation"] = "under"
            result["deviation_amount"] = expected_size - sample_size
            result["recommendation"] = (
                f"Sample size should be increased by {expected_size - sample_size} "
                f"to meet AICPA guidance"
            )
        elif sample_size > expected_size:
            result["deviation"] = "over"
            result["deviation_amount"] = sample_size - expected_size
            result["recommendation"] = (
                "Sample size exceeds AICPA minimum, providing additional assurance"
            )
        else:
            result["deviation"] = "none"
            result["deviation_amount"] = 0
            result["recommendation"] = "Sample size meets AICPA guidance"

        return result
