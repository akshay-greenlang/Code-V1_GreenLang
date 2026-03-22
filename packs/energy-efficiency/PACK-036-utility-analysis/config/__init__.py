"""
PACK-036 Utility Analysis Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Utility Analysis Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.energy_efficiency.PACK_036_utility_analysis.config import (
    ...     PackConfig,
    ...     UtilityAnalysisConfig,
    ...     FacilityType,
    ...     UtilityType,
    ...     get_facility_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("office_building")
    >>> print(config.pack.facility_type)
    FacilityType.OFFICE
"""
