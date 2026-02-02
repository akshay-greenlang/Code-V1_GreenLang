# -*- coding: utf-8 -*-
"""
Consent management module for GDPR, CCPA, and CAN-SPAM compliance.
"""
from .registry import ConsentRegistry
from .jurisdictions import (
    JurisdictionManager,
    JurisdictionType,
    GDPRRules,
    CCPARules,
    CANSPAMRules
)
from .opt_out_handler import OptOutHandler


__all__ = [
    "ConsentRegistry",
    "JurisdictionManager",
    "JurisdictionType",
    "GDPRRules",
    "CCPARules",
    "CANSPAMRules",
    "OptOutHandler",
]
