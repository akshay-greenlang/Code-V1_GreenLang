"""
GL-010 EmissionsGuardian - CEMS (Continuous Emissions Monitoring System) Module

This module provides production-grade CEMS data ingestion, normalization,
quality assurance, and hourly aggregation capabilities per EPA 40 CFR Part 75.

Standards Compliance:
    - EPA 40 CFR Part 75: Continuous Emissions Monitoring
    - EPA 40 CFR Part 75 Appendix A: Specifications and Test Procedures
    - EPA 40 CFR Part 75 Appendix B: Quality Assurance and Quality Control
    - EPA 40 CFR Part 75 Appendix D: Missing Data Substitution
    - EPA Method 19: F-factor Calculations

Zero-Hallucination Principle:
    - All emissions calculations use deterministic EPA formulas with Decimal precision
    - No LLM calls in any calculation path
    - Complete provenance tracking with SHA-256 hashes
    - Full audit trail for regulatory compliance

Module Components:
    - data_acquisition: Real-time CEMS data acquisition from OPC-UA, Modbus, DCS
    - normalization: O2 correction, moisture correction, unit conversions
    - quality_assurance: Data validation, missing data substitution, calibration drift
    - hourly_aggregation: 15-minute to hourly averaging with Part 75 compliance

Example:
    >>> from cems import CEMSDataAcquisition, CEMSNormalizer, CEMSQualityAssurance, HourlyAggregator
    >>>
    >>> # Initialize acquisition
    >>> acquisition = CEMSDataAcquisition(config)
    >>> raw_data = acquisition.poll()
    >>>
    >>> # Normalize data
    >>> normalizer = CEMSNormalizer(config)
    >>> normalized = normalizer.normalize(raw_data)
    >>>
    >>> # Apply QA/QC
    >>> qa = CEMSQualityAssurance(config)
    >>> validated = qa.validate(normalized)
    >>>
    >>> # Aggregate to hourly
    >>> aggregator = HourlyAggregator(config)
    >>> hourly = aggregator.aggregate(validated)

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from .data_acquisition import (
    CEMSDataAcquisition,
    CEMSConnectionConfig,
    CEMSRawReading,
    CEMSDataBuffer,
    ConnectionHealth,
    Protocol,
)

from .normalization import (
    CEMSNormalizer,
    NormalizedReading,
    NormalizationConfig,
    FuelType,
    FFactors,
)

from .quality_assurance import (
    CEMSQualityAssurance,
    QAResult,
    QAConfig,
    QualityFlag,
    SubstitutionMethod,
    CalibrationDriftResult,
)

from .hourly_aggregation import (
    HourlyAggregator,
    HourlyAverage,
    AggregationConfig,
    MassEmissionRate,
    DataCompleteness,
)

__all__ = [
    # Data Acquisition
    "CEMSDataAcquisition",
    "CEMSConnectionConfig",
    "CEMSRawReading",
    "CEMSDataBuffer",
    "ConnectionHealth",
    "Protocol",
    # Normalization
    "CEMSNormalizer",
    "NormalizedReading",
    "NormalizationConfig",
    "FuelType",
    "FFactors",
    # Quality Assurance
    "CEMSQualityAssurance",
    "QAResult",
    "QAConfig",
    "QualityFlag",
    "SubstitutionMethod",
    "CalibrationDriftResult",
    # Hourly Aggregation
    "HourlyAggregator",
    "HourlyAverage",
    "AggregationConfig",
    "MassEmissionRate",
    "DataCompleteness",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
