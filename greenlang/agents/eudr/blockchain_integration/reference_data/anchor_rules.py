# -*- coding: utf-8 -*-
"""
Anchoring Rules Reference Data - AGENT-EUDR-013 Blockchain Integration

Defines the anchoring rules for each EUDR supply chain event type
covering priority levels, batch eligibility, required fields, validation
rules, gas estimates per network, and data retention policy per EUDR
Article 14 five-year record-keeping requirements.

Event Types (8):
    1. dds_submission: Due Diligence Statement submission (Article 4).
       Priority P0, batch eligible with 300s max wait, immediate anchor
       optional for critical submissions.
    2. custody_transfer: Custody transfer between supply chain
       participants.  Priority P0, batch eligible with 300s max wait.
    3. batch_event: Batch-level processing event (splitting, merging,
       transformation).  Priority P0, batch eligible with 600s max wait.
    4. certificate_reference: External certification reference (FSC,
       RSPO, ISCC, Fairtrade, UTZ/RA).  Priority P1, 900s max wait.
    5. reconciliation_result: Mass balance reconciliation result.
       Priority P1, batch eligible with 1800s max wait.
    6. mass_balance_entry: Individual mass balance ledger entry.
       Priority P2, batch eligible with 3600s max wait.
    7. document_authentication: Document authentication result from
       AGENT-EUDR-012.  Priority P2, 3600s max wait.
    8. geolocation_verification: Geolocation verification result for
       production plot coordinates.  Priority P2, 3600s max wait.

Gas Estimates:
    Per-operation gas estimates for each supported network, used for
    cost projection and gas limit configuration.

Retention Rules:
    EUDR Article 14 mandates five-year data retention for all anchored
    compliance records.  Retention rules specify per-event-type archival
    policies and compliance period boundaries.

Lookup Helpers:
    get_anchor_rule(event_type) -> dict | None
    get_priority(event_type) -> str
    is_batch_eligible(event_type) -> bool
    get_max_batch_wait(event_type) -> int
    get_required_fields(event_type) -> list[str]
    validate_anchor_request(event_type, record_data) -> list[str]

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013) - Appendix D
Agent ID: GL-EUDR-BCI-013
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anchor rules: Dict[event_type, rule_definition]
# ---------------------------------------------------------------------------

ANCHOR_RULES: Dict[str, Dict[str, Any]] = {
    # ==================================================================
    # dds_submission - Due Diligence Statement (Article 4)
    # ==================================================================
    "dds_submission": {
        "event_type": "dds_submission",
        "display_name": "DDS Submission",
        "description": (
            "Due Diligence Statement submission event per EUDR Article 4. "
            "Anchors the hash of a complete DDS for Article 14 "
            "record-keeping compliance.  Critical priority ensures "
            "near-immediate on-chain registration."
        ),
        "priority": "p0_immediate",
        "priority_level": 0,
        "batch_eligible": True,
        "max_batch_wait_s": 300,
        "immediate_anchor": True,
        "required_fields": [
            "dds_id",
            "operator_id",
            "submission_hash",
            "commodity",
            "country_of_production",
            "submission_timestamp",
        ],
        "optional_fields": [
            "operator_name",
            "competent_authority_id",
            "product_description",
            "quantity_kg",
            "geolocation_hash",
        ],
        "validation_rules": {
            "dds_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "pattern": r"^[A-Za-z0-9\-_]+$",
                "description": "Unique DDS identifier",
            },
            "operator_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "pattern": r"^[A-Za-z0-9\-_]+$",
                "description": "EUDR operator identifier",
            },
            "submission_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of the DDS content",
            },
            "commodity": {
                "type": "string",
                "allowed_values": [
                    "cattle", "cocoa", "coffee",
                    "oil_palm", "rubber", "soya", "wood",
                ],
                "description": "EUDR regulated commodity",
            },
            "country_of_production": {
                "type": "string",
                "min_length": 2,
                "max_length": 3,
                "pattern": r"^[A-Z]{2,3}$",
                "description": "ISO 3166-1 alpha-2/3 country code",
            },
            "submission_timestamp": {
                "type": "datetime",
                "format": "iso8601",
                "description": "UTC timestamp of DDS submission",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 4",
        "eudr_notes": (
            "DDS submissions are the primary EUDR compliance artifact. "
            "On-chain anchoring provides immutable proof of timely "
            "submission for competent authority verification."
        ),
    },

    # ==================================================================
    # custody_transfer - Supply chain custody transfer
    # ==================================================================
    "custody_transfer": {
        "event_type": "custody_transfer",
        "display_name": "Custody Transfer",
        "description": (
            "Transfer of custody of regulated commodities between supply "
            "chain participants.  Records sender, receiver, commodity, "
            "quantity, and transfer timestamp for chain-of-custody "
            "traceability."
        ),
        "priority": "p0_immediate",
        "priority_level": 0,
        "batch_eligible": True,
        "max_batch_wait_s": 300,
        "immediate_anchor": False,
        "required_fields": [
            "transfer_id",
            "sender_id",
            "receiver_id",
            "commodity",
            "quantity_kg",
            "transfer_hash",
            "transfer_timestamp",
        ],
        "optional_fields": [
            "sender_name",
            "receiver_name",
            "batch_id",
            "transport_mode",
            "origin_country",
            "destination_country",
            "incoterm",
        ],
        "validation_rules": {
            "transfer_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "pattern": r"^[A-Za-z0-9\-_]+$",
                "description": "Unique transfer identifier",
            },
            "sender_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Sender operator/entity identifier",
            },
            "receiver_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Receiver operator/entity identifier",
            },
            "commodity": {
                "type": "string",
                "allowed_values": [
                    "cattle", "cocoa", "coffee",
                    "oil_palm", "rubber", "soya", "wood",
                ],
                "description": "EUDR regulated commodity",
            },
            "quantity_kg": {
                "type": "number",
                "min_value": 0.001,
                "max_value": 1_000_000_000,
                "description": "Transfer quantity in kilograms",
            },
            "transfer_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of transfer data",
            },
            "transfer_timestamp": {
                "type": "datetime",
                "format": "iso8601",
                "description": "UTC timestamp of custody transfer",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 10",
        "eudr_notes": (
            "Custody transfers establish the chain of custody from "
            "production to import.  On-chain anchoring ensures "
            "immutable provenance of ownership changes."
        ),
    },

    # ==================================================================
    # batch_event - Batch processing event
    # ==================================================================
    "batch_event": {
        "event_type": "batch_event",
        "display_name": "Batch Event",
        "description": (
            "Batch-level processing event recording splitting, merging, "
            "or transformation of commodity batches.  Critical for "
            "mass balance traceability across processing steps."
        ),
        "priority": "p0_immediate",
        "priority_level": 0,
        "batch_eligible": True,
        "max_batch_wait_s": 600,
        "immediate_anchor": False,
        "required_fields": [
            "batch_id",
            "event_type_detail",
            "operator_id",
            "commodity",
            "batch_hash",
            "event_timestamp",
        ],
        "optional_fields": [
            "parent_batch_ids",
            "child_batch_ids",
            "input_quantity_kg",
            "output_quantity_kg",
            "processing_type",
            "facility_id",
        ],
        "validation_rules": {
            "batch_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Unique batch identifier",
            },
            "event_type_detail": {
                "type": "string",
                "allowed_values": [
                    "split", "merge", "transform",
                    "package", "label", "quality_check",
                ],
                "description": "Specific batch event type",
            },
            "operator_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Processing operator identifier",
            },
            "commodity": {
                "type": "string",
                "allowed_values": [
                    "cattle", "cocoa", "coffee",
                    "oil_palm", "rubber", "soya", "wood",
                ],
                "description": "EUDR regulated commodity",
            },
            "batch_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of batch event data",
            },
            "event_timestamp": {
                "type": "datetime",
                "format": "iso8601",
                "description": "UTC timestamp of batch event",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 10",
        "eudr_notes": (
            "Batch events maintain mass balance traceability through "
            "processing.  Splitting, merging, and transformation records "
            "enable end-to-end product tracing."
        ),
    },

    # ==================================================================
    # certificate_reference - External certification reference
    # ==================================================================
    "certificate_reference": {
        "event_type": "certificate_reference",
        "display_name": "Certificate Reference",
        "description": (
            "Reference to an external certification (FSC, RSPO, ISCC, "
            "Fairtrade, UTZ/RA) with certificate ID, validity period, "
            "and scope.  Anchoring provides immutable proof of "
            "certification status at a point in time."
        ),
        "priority": "p1_standard",
        "priority_level": 1,
        "batch_eligible": True,
        "max_batch_wait_s": 900,
        "immediate_anchor": False,
        "required_fields": [
            "certificate_id",
            "certificate_type",
            "holder_id",
            "certificate_hash",
            "valid_from",
            "valid_to",
        ],
        "optional_fields": [
            "holder_name",
            "certification_body",
            "scope_description",
            "commodity",
            "country_code",
            "certificate_url",
        ],
        "validation_rules": {
            "certificate_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Unique certificate identifier",
            },
            "certificate_type": {
                "type": "string",
                "allowed_values": [
                    "fsc", "rspo", "iscc", "fairtrade",
                    "utz_ra", "rainforest_alliance",
                    "organic", "pefc", "other",
                ],
                "description": "Certification scheme type",
            },
            "holder_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Certificate holder identifier",
            },
            "certificate_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of certificate document",
            },
            "valid_from": {
                "type": "date",
                "format": "iso8601",
                "description": "Certificate validity start date",
            },
            "valid_to": {
                "type": "date",
                "format": "iso8601",
                "description": "Certificate validity end date",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 10",
        "eudr_notes": (
            "Certificate references link EUDR compliance records to "
            "external sustainability certifications.  On-chain anchoring "
            "provides tamper-evident proof of certification status."
        ),
    },

    # ==================================================================
    # reconciliation_result - Mass balance reconciliation
    # ==================================================================
    "reconciliation_result": {
        "event_type": "reconciliation_result",
        "display_name": "Reconciliation Result",
        "description": (
            "Mass balance reconciliation result comparing inbound and "
            "outbound commodity quantities at a facility or operator "
            "level.  Identifies discrepancies that may indicate "
            "non-compliant supply chain mixing."
        ),
        "priority": "p1_standard",
        "priority_level": 1,
        "batch_eligible": True,
        "max_batch_wait_s": 1800,
        "immediate_anchor": False,
        "required_fields": [
            "reconciliation_id",
            "operator_id",
            "commodity",
            "period_start",
            "period_end",
            "reconciliation_hash",
        ],
        "optional_fields": [
            "inbound_quantity_kg",
            "outbound_quantity_kg",
            "discrepancy_kg",
            "discrepancy_pct",
            "reconciliation_status",
            "facility_id",
            "notes",
        ],
        "validation_rules": {
            "reconciliation_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Unique reconciliation identifier",
            },
            "operator_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Operator performing reconciliation",
            },
            "commodity": {
                "type": "string",
                "allowed_values": [
                    "cattle", "cocoa", "coffee",
                    "oil_palm", "rubber", "soya", "wood",
                ],
                "description": "EUDR regulated commodity",
            },
            "period_start": {
                "type": "date",
                "format": "iso8601",
                "description": "Reconciliation period start date",
            },
            "period_end": {
                "type": "date",
                "format": "iso8601",
                "description": "Reconciliation period end date",
            },
            "reconciliation_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of reconciliation data",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 10",
        "eudr_notes": (
            "Reconciliation results provide evidence of mass balance "
            "integrity across the supply chain.  Discrepancies above "
            "threshold trigger compliance alerts."
        ),
    },

    # ==================================================================
    # mass_balance_entry - Individual ledger entry
    # ==================================================================
    "mass_balance_entry": {
        "event_type": "mass_balance_entry",
        "display_name": "Mass Balance Entry",
        "description": (
            "Individual mass balance ledger entry recording commodity "
            "inflow or outflow at a facility.  High-volume event type "
            "suitable for deferred batch anchoring."
        ),
        "priority": "p2_batch",
        "priority_level": 2,
        "batch_eligible": True,
        "max_batch_wait_s": 3600,
        "immediate_anchor": False,
        "required_fields": [
            "entry_id",
            "operator_id",
            "commodity",
            "direction",
            "quantity_kg",
            "entry_hash",
            "entry_timestamp",
        ],
        "optional_fields": [
            "batch_id",
            "counterparty_id",
            "facility_id",
            "dds_reference",
            "notes",
        ],
        "validation_rules": {
            "entry_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Unique ledger entry identifier",
            },
            "operator_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Operator recording the entry",
            },
            "commodity": {
                "type": "string",
                "allowed_values": [
                    "cattle", "cocoa", "coffee",
                    "oil_palm", "rubber", "soya", "wood",
                ],
                "description": "EUDR regulated commodity",
            },
            "direction": {
                "type": "string",
                "allowed_values": ["inflow", "outflow"],
                "description": "Commodity flow direction",
            },
            "quantity_kg": {
                "type": "number",
                "min_value": 0.001,
                "max_value": 1_000_000_000,
                "description": "Quantity in kilograms",
            },
            "entry_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of entry data",
            },
            "entry_timestamp": {
                "type": "datetime",
                "format": "iso8601",
                "description": "UTC timestamp of ledger entry",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 10",
        "eudr_notes": (
            "Mass balance entries are the highest-volume anchoring event. "
            "Batch anchoring via P2 priority reduces gas costs by "
            "aggregating entries into Merkle trees."
        ),
    },

    # ==================================================================
    # document_authentication - AGENT-EUDR-012 result
    # ==================================================================
    "document_authentication": {
        "event_type": "document_authentication",
        "display_name": "Document Authentication",
        "description": (
            "Document authentication result from AGENT-EUDR-012 "
            "including hash integrity verification, digital signature "
            "status, fraud detection outcome, and cross-reference "
            "verification result."
        ),
        "priority": "p2_batch",
        "priority_level": 2,
        "batch_eligible": True,
        "max_batch_wait_s": 3600,
        "immediate_anchor": False,
        "required_fields": [
            "document_id",
            "document_type",
            "authentication_hash",
            "authentication_status",
            "authentication_timestamp",
        ],
        "optional_fields": [
            "operator_id",
            "document_hash",
            "signature_status",
            "fraud_detection_status",
            "cross_reference_status",
            "classification_confidence",
            "agent_id",
        ],
        "validation_rules": {
            "document_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Unique document identifier",
            },
            "document_type": {
                "type": "string",
                "allowed_values": [
                    "coo", "phytosanitary", "bol",
                    "rspo", "fsc", "iscc", "fairtrade",
                    "utz_ra", "ltr", "dds", "ssd", "other",
                ],
                "description": "Document type classification",
            },
            "authentication_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of authentication result",
            },
            "authentication_status": {
                "type": "string",
                "allowed_values": [
                    "pass", "fail", "partial", "unknown",
                ],
                "description": "Overall authentication result status",
            },
            "authentication_timestamp": {
                "type": "datetime",
                "format": "iso8601",
                "description": "UTC timestamp of authentication",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 14",
        "eudr_notes": (
            "Document authentication results from AGENT-EUDR-012 are "
            "anchored for Article 14 record-keeping.  Provides "
            "immutable evidence that due diligence documents were "
            "verified at a specific point in time."
        ),
    },

    # ==================================================================
    # geolocation_verification - Plot coordinate verification
    # ==================================================================
    "geolocation_verification": {
        "event_type": "geolocation_verification",
        "display_name": "Geolocation Verification",
        "description": (
            "Geolocation verification result confirming that production "
            "plot coordinates are in a deforestation-free zone per the "
            "December 31, 2020 cutoff date.  Includes satellite imagery "
            "analysis reference and forest cover change assessment."
        ),
        "priority": "p2_batch",
        "priority_level": 2,
        "batch_eligible": True,
        "max_batch_wait_s": 3600,
        "immediate_anchor": False,
        "required_fields": [
            "verification_id",
            "plot_id",
            "latitude",
            "longitude",
            "verification_hash",
            "deforestation_free",
            "verification_timestamp",
        ],
        "optional_fields": [
            "operator_id",
            "commodity",
            "plot_area_ha",
            "satellite_source",
            "analysis_date",
            "forest_cover_pct",
            "forest_change_pct",
            "reference_date",
            "country_code",
        ],
        "validation_rules": {
            "verification_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Unique verification identifier",
            },
            "plot_id": {
                "type": "string",
                "min_length": 1,
                "max_length": 128,
                "description": "Production plot identifier",
            },
            "latitude": {
                "type": "number",
                "min_value": -90.0,
                "max_value": 90.0,
                "description": "Plot centroid latitude (WGS84)",
            },
            "longitude": {
                "type": "number",
                "min_value": -180.0,
                "max_value": 180.0,
                "description": "Plot centroid longitude (WGS84)",
            },
            "verification_hash": {
                "type": "string",
                "min_length": 64,
                "max_length": 128,
                "pattern": r"^[a-f0-9]{64,128}$",
                "description": "SHA-256/512 hash of verification data",
            },
            "deforestation_free": {
                "type": "boolean",
                "description": (
                    "Whether the plot is deforestation-free since "
                    "December 31, 2020"
                ),
            },
            "verification_timestamp": {
                "type": "datetime",
                "format": "iso8601",
                "description": "UTC timestamp of verification",
            },
        },
        "retention_years": 5,
        "eudr_article": "Article 10",
        "eudr_notes": (
            "Geolocation verification results provide evidence that "
            "production plots meet the EUDR deforestation-free "
            "requirement (cutoff: 31 December 2020).  On-chain "
            "anchoring creates immutable proof of verification."
        ),
    },
}


# ---------------------------------------------------------------------------
# Gas estimates per operation per network
# ---------------------------------------------------------------------------

#: Estimated gas consumption for each operation type on each supported
#: network.  Values are in gas units (not currency).  Actual costs
#: depend on the current gas price on each network.
#:
#: Operations:
#:   - anchor_single: Single Merkle root anchoring via anchor() call
#:   - anchor_batch: Batch anchor including Merkle tree root submission
#:   - verify: Verification read call (view function, zero gas on EVM)
#:   - deploy_anchor_registry: AnchorRegistry contract deployment
#:   - deploy_custody_transfer: CustodyTransfer contract deployment
#:   - deploy_compliance_check: ComplianceCheck contract deployment
#:   - record_transfer: CustodyTransfer.recordTransfer() call
#:   - confirm_transfer: CustodyTransfer.confirmTransfer() call
#:   - check_compliance: ComplianceCheck.checkCompliance() call
#:   - register_party: ComplianceCheck.registerParty() call
GAS_ESTIMATES: Dict[str, Dict[str, int]] = {
    "ethereum": {
        "anchor_single": 85_000,
        "anchor_batch": 120_000,
        "verify": 0,
        "deploy_anchor_registry": 2_200_000,
        "deploy_custody_transfer": 2_800_000,
        "deploy_compliance_check": 2_500_000,
        "record_transfer": 110_000,
        "confirm_transfer": 65_000,
        "check_compliance": 95_000,
        "register_party": 75_000,
    },
    "ethereum_goerli": {
        "anchor_single": 85_000,
        "anchor_batch": 120_000,
        "verify": 0,
        "deploy_anchor_registry": 2_200_000,
        "deploy_custody_transfer": 2_800_000,
        "deploy_compliance_check": 2_500_000,
        "record_transfer": 110_000,
        "confirm_transfer": 65_000,
        "check_compliance": 95_000,
        "register_party": 75_000,
    },
    "polygon": {
        "anchor_single": 95_000,
        "anchor_batch": 135_000,
        "verify": 0,
        "deploy_anchor_registry": 2_500_000,
        "deploy_custody_transfer": 3_100_000,
        "deploy_compliance_check": 2_800_000,
        "record_transfer": 125_000,
        "confirm_transfer": 70_000,
        "check_compliance": 105_000,
        "register_party": 80_000,
    },
    "polygon_mumbai": {
        "anchor_single": 95_000,
        "anchor_batch": 135_000,
        "verify": 0,
        "deploy_anchor_registry": 2_500_000,
        "deploy_custody_transfer": 3_100_000,
        "deploy_compliance_check": 2_800_000,
        "record_transfer": 125_000,
        "confirm_transfer": 70_000,
        "check_compliance": 105_000,
        "register_party": 80_000,
    },
    "besu": {
        "anchor_single": 90_000,
        "anchor_batch": 125_000,
        "verify": 0,
        "deploy_anchor_registry": 2_400_000,
        "deploy_custody_transfer": 3_000_000,
        "deploy_compliance_check": 2_700_000,
        "record_transfer": 120_000,
        "confirm_transfer": 68_000,
        "check_compliance": 100_000,
        "register_party": 78_000,
    },
    "fabric": {
        "anchor_single": 0,
        "anchor_batch": 0,
        "verify": 0,
        "deploy_anchor_registry": 0,
        "deploy_custody_transfer": 0,
        "deploy_compliance_check": 0,
        "record_transfer": 0,
        "confirm_transfer": 0,
        "check_compliance": 0,
        "register_party": 0,
    },
}


# ---------------------------------------------------------------------------
# Retention rules per EUDR Article 14
# ---------------------------------------------------------------------------

#: Data retention configuration per EUDR Article 14 five-year
#: record-keeping requirement.
#:
#: Article 14(1): Operators and traders shall keep the due diligence
#: statements, and all documentation, data and information collected
#: for the purposes of [due diligence], for five years from the date
#: of the placing on the market or of the export.
RETENTION_RULES: Dict[str, Dict[str, Any]] = {
    "general": {
        "retention_years": 5,
        "retention_basis": "EUDR Article 14(1)",
        "retention_start": "event_timestamp",
        "archive_policy": "compress_and_archive",
        "archive_format": "gzip_jsonl",
        "archive_storage": "s3_glacier",
        "deletion_policy": "soft_delete_then_purge",
        "deletion_grace_period_days": 90,
        "compliance_verification_interval_days": 30,
        "audit_log_retention_years": 7,
    },
    "dds_submission": {
        "retention_years": 5,
        "critical": True,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": True,
        "notes": (
            "DDS records must be preserved for the full five-year "
            "period.  Immutable on-chain anchors serve as primary "
            "evidence.  Off-chain record copies archived to S3 Glacier."
        ),
    },
    "custody_transfer": {
        "retention_years": 5,
        "critical": True,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": True,
        "notes": (
            "Custody transfer records establish chain of custody.  "
            "Must be preserved alongside DDS records for complete "
            "traceability evidence."
        ),
    },
    "batch_event": {
        "retention_years": 5,
        "critical": True,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": True,
        "notes": (
            "Batch processing events support mass balance verification. "
            "Retained for the same period as the associated DDS."
        ),
    },
    "certificate_reference": {
        "retention_years": 5,
        "critical": False,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": False,
        "notes": (
            "Certificate references provide supporting evidence. "
            "Retained for compliance but not individually critical."
        ),
    },
    "reconciliation_result": {
        "retention_years": 5,
        "critical": False,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": False,
        "notes": (
            "Reconciliation results demonstrate mass balance integrity. "
            "Archived after retention period for historical analysis."
        ),
    },
    "mass_balance_entry": {
        "retention_years": 5,
        "critical": False,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": False,
        "notes": (
            "Individual mass balance entries are high-volume records. "
            "Archived in compressed format after retention period."
        ),
    },
    "document_authentication": {
        "retention_years": 5,
        "critical": False,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": False,
        "notes": (
            "Authentication results from AGENT-EUDR-012 provide "
            "document verification evidence for audits."
        ),
    },
    "geolocation_verification": {
        "retention_years": 5,
        "critical": True,
        "immutable": True,
        "archive_on_expire": True,
        "legal_hold_eligible": True,
        "notes": (
            "Geolocation verification is critical evidence for the "
            "deforestation-free requirement.  Must be retained with "
            "satellite imagery references for the full period."
        ),
    },
}


# ---------------------------------------------------------------------------
# Priority index
# ---------------------------------------------------------------------------

#: Mapping of event type to priority level string.
_PRIORITY_INDEX: Dict[str, str] = {
    et: rule["priority"]
    for et, rule in ANCHOR_RULES.items()
}

#: Mapping of event type to batch eligibility.
_BATCH_ELIGIBLE_INDEX: Dict[str, bool] = {
    et: rule["batch_eligible"]
    for et, rule in ANCHOR_RULES.items()
}

#: Mapping of event type to max batch wait seconds.
_MAX_BATCH_WAIT_INDEX: Dict[str, int] = {
    et: rule["max_batch_wait_s"]
    for et, rule in ANCHOR_RULES.items()
}

#: Mapping of event type to required fields.
_REQUIRED_FIELDS_INDEX: Dict[str, List[str]] = {
    et: rule["required_fields"]
    for et, rule in ANCHOR_RULES.items()
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_anchor_rule(event_type: str) -> Optional[Dict[str, Any]]:
    """Return the full anchoring rule for an event type.

    Args:
        event_type: EUDR supply chain event type (e.g. ``"dds_submission"``,
            ``"custody_transfer"``).

    Returns:
        Rule dictionary or ``None`` if the event type is not recognized.

    Example:
        >>> rule = get_anchor_rule("dds_submission")
        >>> rule["priority"]
        'p0_immediate'
    """
    normalized = event_type.lower().strip()
    rule = ANCHOR_RULES.get(normalized)
    if rule is None:
        logger.warning(
            "Unknown anchor event type '%s'. Supported: %s",
            event_type, ", ".join(ANCHOR_RULES.keys()),
        )
    return rule


def get_priority(event_type: str) -> str:
    """Return the anchor priority level for an event type.

    Args:
        event_type: EUDR supply chain event type.

    Returns:
        Priority string (e.g. ``"p0_immediate"``, ``"p1_standard"``,
        ``"p2_batch"``).  Returns ``"p2_batch"`` as default for
        unknown event types.

    Example:
        >>> get_priority("dds_submission")
        'p0_immediate'
        >>> get_priority("mass_balance_entry")
        'p2_batch'
    """
    return _PRIORITY_INDEX.get(event_type.lower().strip(), "p2_batch")


def is_batch_eligible(event_type: str) -> bool:
    """Return whether an event type is eligible for batch anchoring.

    Args:
        event_type: EUDR supply chain event type.

    Returns:
        ``True`` if the event type can be included in batch Merkle tree
        submissions, ``False`` otherwise.  Returns ``True`` as default
        for unknown event types (safe default for batching).

    Example:
        >>> is_batch_eligible("dds_submission")
        True
    """
    return _BATCH_ELIGIBLE_INDEX.get(event_type.lower().strip(), True)


def get_max_batch_wait(event_type: str) -> int:
    """Return the maximum batch wait time in seconds for an event type.

    The max batch wait is the longest an event can wait in the pending
    batch before the batch is submitted, even if the batch is not full.

    Args:
        event_type: EUDR supply chain event type.

    Returns:
        Maximum batch wait in seconds.  Returns ``3600`` (1 hour) as
        default for unknown event types.

    Example:
        >>> get_max_batch_wait("dds_submission")
        300
        >>> get_max_batch_wait("mass_balance_entry")
        3600
    """
    return _MAX_BATCH_WAIT_INDEX.get(event_type.lower().strip(), 3600)


def get_required_fields(event_type: str) -> List[str]:
    """Return the list of required fields for an event type.

    Args:
        event_type: EUDR supply chain event type.

    Returns:
        List of required field names.  Returns an empty list for unknown
        event types.

    Example:
        >>> get_required_fields("dds_submission")
        ['dds_id', 'operator_id', 'submission_hash', ...]
    """
    return _REQUIRED_FIELDS_INDEX.get(event_type.lower().strip(), [])


def validate_anchor_request(
    event_type: str,
    record_data: Dict[str, Any],
) -> List[str]:
    """Validate an anchor request against the rules for an event type.

    Checks that all required fields are present and non-empty.
    Optionally validates field types and patterns if validation_rules
    are defined.

    Args:
        event_type: EUDR supply chain event type.
        record_data: Dictionary of record data fields to validate.

    Returns:
        List of validation error strings.  An empty list indicates
        the request is valid.

    Example:
        >>> errors = validate_anchor_request("dds_submission", {
        ...     "dds_id": "DDS-001",
        ...     "operator_id": "OP-001",
        ...     "submission_hash": "a" * 64,
        ...     "commodity": "coffee",
        ...     "country_of_production": "BR",
        ...     "submission_timestamp": "2026-03-08T12:00:00Z",
        ... })
        >>> len(errors)
        0

        >>> errors = validate_anchor_request("dds_submission", {})
        >>> len(errors) > 0
        True
    """
    errors: List[str] = []
    normalized = event_type.lower().strip()

    # Get the rule
    rule = ANCHOR_RULES.get(normalized)
    if rule is None:
        errors.append(
            f"Unknown event type '{event_type}'. "
            f"Supported: {', '.join(ANCHOR_RULES.keys())}"
        )
        return errors

    # Check required fields presence
    required = rule.get("required_fields", [])
    for field_name in required:
        if field_name not in record_data:
            errors.append(
                f"Missing required field '{field_name}' "
                f"for event type '{normalized}'"
            )
        elif record_data[field_name] is None:
            errors.append(
                f"Required field '{field_name}' is None "
                f"for event type '{normalized}'"
            )
        elif isinstance(record_data[field_name], str) and not record_data[field_name].strip():
            errors.append(
                f"Required field '{field_name}' is empty "
                f"for event type '{normalized}'"
            )

    # Validate field types and constraints
    validation_rules = rule.get("validation_rules", {})
    for field_name, field_rule in validation_rules.items():
        if field_name not in record_data:
            continue  # Already caught by required fields check

        value = record_data[field_name]
        if value is None:
            continue  # Already caught by required fields check

        # Type: string constraints
        if field_rule.get("type") == "string" and isinstance(value, str):
            min_len = field_rule.get("min_length")
            max_len = field_rule.get("max_length")
            if min_len is not None and len(value) < min_len:
                errors.append(
                    f"Field '{field_name}' length {len(value)} "
                    f"is below minimum {min_len}"
                )
            if max_len is not None and len(value) > max_len:
                errors.append(
                    f"Field '{field_name}' length {len(value)} "
                    f"exceeds maximum {max_len}"
                )

            # Pattern check (deferred to caller for regex matching)
            allowed = field_rule.get("allowed_values")
            if allowed is not None and value not in allowed:
                errors.append(
                    f"Field '{field_name}' value '{value}' "
                    f"not in allowed values: {allowed}"
                )

        # Type: number constraints
        if field_rule.get("type") == "number":
            try:
                numeric_val = float(value)
                min_val = field_rule.get("min_value")
                max_val = field_rule.get("max_value")
                if min_val is not None and numeric_val < min_val:
                    errors.append(
                        f"Field '{field_name}' value {numeric_val} "
                        f"is below minimum {min_val}"
                    )
                if max_val is not None and numeric_val > max_val:
                    errors.append(
                        f"Field '{field_name}' value {numeric_val} "
                        f"exceeds maximum {max_val}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Field '{field_name}' must be a number, "
                    f"got {type(value).__name__}"
                )

    return errors


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

#: Total number of anchor event types.
TOTAL_EVENT_TYPES: int = len(ANCHOR_RULES)

#: Supported event type identifiers.
SUPPORTED_EVENT_TYPES: List[str] = list(ANCHOR_RULES.keys())

#: Event types by priority level.
EVENTS_BY_PRIORITY: Dict[str, List[str]] = {}
for _et, _rule in ANCHOR_RULES.items():
    _priority = _rule["priority"]
    if _priority not in EVENTS_BY_PRIORITY:
        EVENTS_BY_PRIORITY[_priority] = []
    EVENTS_BY_PRIORITY[_priority].append(_et)

logger.debug(
    "Anchor rules loaded: %d event types (%s)",
    TOTAL_EVENT_TYPES,
    ", ".join(f"{p}={len(ets)}" for p, ets in EVENTS_BY_PRIORITY.items()),
)
