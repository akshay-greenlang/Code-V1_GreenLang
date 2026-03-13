# -*- coding: utf-8 -*-
"""
Trusted Certificate Authorities Reference Data - AGENT-EUDR-012

Curated registry of trusted certificate authorities (CAs) for validating
digital signatures on EUDR supply chain documents.  The CertificateChain-
Validator engine uses this dataset to determine whether a document's
signing certificate chains back to a known, trusted root or intermediate
CA without requiring real-time lookups to external trust stores.

Categories:
    - eIDAS TSPs: Qualified Trust Service Providers under eIDAS
      Regulation (EU) No 910/2014 for legally binding digital signatures.
    - Document signing: Commercial CAs commonly used for PDF and
      document signing across global supply chains.
    - Government: EU member state national CAs used for official
      government-issued documents (COO, phytosanitary, customs).
    - Certification bodies: CAs operated by or on behalf of
      sustainability certification bodies (FSC/ASI, RSPO, ISCC).

Each entry includes:
    - ca_name: Human-readable CA name
    - ca_category: Category classification
    - subject_dn: Distinguished name pattern (simplified)
    - key_algorithm: Public key algorithm (RSA or ECDSA)
    - key_size_bits: Public key size in bits
    - fingerprint_sha256: Placeholder SHA-256 fingerprint for
      reference (NOT a real certificate fingerprint)
    - is_root: Whether this is a root CA
    - purpose: Intended usage context

IMPORTANT: The fingerprints in this module are PLACEHOLDER values for
reference data structure purposes.  In production, these would be
replaced with actual SHA-256 fingerprints of the trusted CA certificates
loaded from the system trust store or a curated PEM bundle.

Lookup helpers:
    get_trusted_cas() -> list[dict]
    get_cas_by_category(category) -> list[dict]
    get_pinned_issuers() -> dict[str, list[str]]

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012)
Agent ID: GL-EUDR-DAV-012
Regulation: EU 2023/1115 (EUDR), eIDAS (EU) No 910/2014
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trusted CA registry
# ---------------------------------------------------------------------------

TRUSTED_CAS: List[Dict[str, Any]] = [
    # ==================================================================
    # eIDAS Qualified Trust Service Providers (TSPs)
    # ==================================================================
    {
        "ca_name": "D-TRUST Root Class 3 CA 2 2009",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=D-TRUST Root Class 3 CA 2 2009, O=D-Trust GmbH, C=DE",
        "key_algorithm": "RSA",
        "key_size_bits": 2048,
        "fingerprint_sha256": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        "is_root": True,
        "purpose": "German eIDAS qualified electronic signatures",
    },
    {
        "ca_name": "D-TRUST Qualified Signing CA 2021",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=D-TRUST Qualified Signing CA 2021, O=D-Trust GmbH, C=DE",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
        "is_root": False,
        "purpose": "German qualified electronic signatures (issuing CA)",
    },
    {
        "ca_name": "Swisscom Root CA 4",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=Swisscom Root CA 4, O=Swisscom AG, C=CH",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
        "is_root": True,
        "purpose": "Swiss eIDAS qualified electronic signatures",
    },
    {
        "ca_name": "Swisscom Qualified Signing CA 2022",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=Swisscom Qualified Signing CA 2022, O=Swisscom AG, C=CH",
        "key_algorithm": "ECDSA",
        "key_size_bits": 384,
        "fingerprint_sha256": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
        "is_root": False,
        "purpose": "Swiss qualified electronic signatures (issuing CA)",
    },
    {
        "ca_name": "DigiCert EU Qualified Root CA",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=DigiCert EU Qualified Root CA, O=DigiCert Inc, C=US",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
        "is_root": True,
        "purpose": "EU eIDAS qualified trust services via DigiCert",
    },
    {
        "ca_name": "GlobalSign EU Qualified CA - SHA384 - G4",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=GlobalSign EU Qualified CA, O=GlobalSign nv-sa, C=BE",
        "key_algorithm": "ECDSA",
        "key_size_bits": 384,
        "fingerprint_sha256": "f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7",
        "is_root": False,
        "purpose": "Belgian eIDAS qualified electronic signatures",
    },
    {
        "ca_name": "QuoVadis EU Issuing Certification Authority G4",
        "ca_category": "eidas_tsp",
        "subject_dn": "CN=QuoVadis EU Issuing CA G4, O=QuoVadis Trustlink B.V., C=NL",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8",
        "is_root": False,
        "purpose": "Dutch eIDAS qualified electronic signatures",
    },

    # ==================================================================
    # Document Signing CAs (commercial)
    # ==================================================================
    {
        "ca_name": "GlobalSign Root CA - R3",
        "ca_category": "document_signing",
        "subject_dn": "CN=GlobalSign Root CA - R3, O=GlobalSign nv-sa, C=BE",
        "key_algorithm": "RSA",
        "key_size_bits": 2048,
        "fingerprint_sha256": "b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9",
        "is_root": True,
        "purpose": "Global document signing and code signing",
    },
    {
        "ca_name": "GlobalSign Document Signing CA - SHA256 - G3",
        "ca_category": "document_signing",
        "subject_dn": "CN=GlobalSign Document Signing CA G3, O=GlobalSign nv-sa, C=BE",
        "key_algorithm": "RSA",
        "key_size_bits": 2048,
        "fingerprint_sha256": "c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0",
        "is_root": False,
        "purpose": "PDF and document signing (issuing CA)",
    },
    {
        "ca_name": "DigiCert Global Root G2",
        "ca_category": "document_signing",
        "subject_dn": "CN=DigiCert Global Root G2, O=DigiCert Inc, C=US",
        "key_algorithm": "RSA",
        "key_size_bits": 2048,
        "fingerprint_sha256": "d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1",
        "is_root": True,
        "purpose": "Global document and code signing",
    },
    {
        "ca_name": "DigiCert Document Signing CA - SHA256",
        "ca_category": "document_signing",
        "subject_dn": "CN=DigiCert Document Signing CA, O=DigiCert Inc, C=US",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2",
        "is_root": False,
        "purpose": "PDF and document signing (issuing CA)",
    },
    {
        "ca_name": "Entrust Root Certification Authority - G4",
        "ca_category": "document_signing",
        "subject_dn": "CN=Entrust Root CA - G4, O=Entrust Inc, C=US",
        "key_algorithm": "ECDSA",
        "key_size_bits": 384,
        "fingerprint_sha256": "f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3",
        "is_root": True,
        "purpose": "Global document signing and certificate services",
    },
    {
        "ca_name": "Sectigo Document Signing CA - R3",
        "ca_category": "document_signing",
        "subject_dn": "CN=Sectigo Document Signing CA R3, O=Sectigo Limited, C=GB",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4",
        "is_root": False,
        "purpose": "Document signing certificates (issuing CA)",
    },

    # ==================================================================
    # Government CAs (EU member state national CAs)
    # ==================================================================
    {
        "ca_name": "Bundesdruckerei GmbH - D-TRUST Root",
        "ca_category": "government",
        "subject_dn": "CN=D-TRUST Root CA 3 2013, O=D-Trust GmbH, C=DE",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5",
        "is_root": True,
        "purpose": "German government documents and official certificates",
    },
    {
        "ca_name": "ANSSI - Autorite de Certification Racine",
        "ca_category": "government",
        "subject_dn": "CN=IGC/A, O=PM/SGDN, C=FR",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "is_root": True,
        "purpose": "French government documents and official certificates",
    },
    {
        "ca_name": "Staat der Nederlanden Root CA - G3",
        "ca_category": "government",
        "subject_dn": "CN=Staat der Nederlanden Root CA - G3, O=Staat der Nederlanden, C=NL",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
        "is_root": True,
        "purpose": "Dutch government documents and official certificates",
    },
    {
        "ca_name": "Belgium Root CA4",
        "ca_category": "government",
        "subject_dn": "CN=Belgium Root CA4, O=Certipost s.a./n.v., C=BE",
        "key_algorithm": "ECDSA",
        "key_size_bits": 384,
        "fingerprint_sha256": "e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
        "is_root": True,
        "purpose": "Belgian government documents and eID certificates",
    },
    {
        "ca_name": "Agenzia per l'Italia Digitale - AgID Root",
        "ca_category": "government",
        "subject_dn": "CN=Certificati di Firma Qualificata, O=AgID, C=IT",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
        "is_root": True,
        "purpose": "Italian government qualified electronic signatures",
    },
    {
        "ca_name": "FNMT-RCM - Fabrica Nacional de Moneda y Timbre",
        "ca_category": "government",
        "subject_dn": "CN=AC RAIZ FNMT-RCM, O=FNMT-RCM, C=ES",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        "is_root": True,
        "purpose": "Spanish government documents and official certificates",
    },
    {
        "ca_name": "SCEE - Sistema de Certificacao Electronica do Estado",
        "ca_category": "government",
        "subject_dn": "CN=ECRaizEstado, O=SCEE, C=PT",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
        "is_root": True,
        "purpose": "Portuguese government documents and official certificates",
    },

    # ==================================================================
    # Certification Body CAs (sustainability standards)
    # ==================================================================
    {
        "ca_name": "ASI Document Signing CA (FSC accreditation)",
        "ca_category": "certification_body",
        "subject_dn": "CN=ASI Document Signing CA, O=Accreditation Services International GmbH, C=DE",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2",
        "is_root": False,
        "purpose": "FSC certificate signing by ASI-accredited bodies",
    },
    {
        "ca_name": "RSPO Secretariat Signing CA",
        "ca_category": "certification_body",
        "subject_dn": "CN=RSPO Secretariat Signing CA, O=Roundtable on Sustainable Palm Oil, C=MY",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3",
        "is_root": False,
        "purpose": "RSPO certificate signing and supply chain verification",
    },
    {
        "ca_name": "ISCC System Signing CA",
        "ca_category": "certification_body",
        "subject_dn": "CN=ISCC System Signing CA, O=ISCC System GmbH, C=DE",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4",
        "is_root": False,
        "purpose": "ISCC certificate signing and sustainability verification",
    },
    {
        "ca_name": "FLOCERT Signing CA",
        "ca_category": "certification_body",
        "subject_dn": "CN=FLOCERT Signing CA, O=FLOCERT GmbH, C=DE",
        "key_algorithm": "RSA",
        "key_size_bits": 4096,
        "fingerprint_sha256": "f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
        "is_root": False,
        "purpose": "Fairtrade certificate signing",
    },
    {
        "ca_name": "Rainforest Alliance Signing CA",
        "ca_category": "certification_body",
        "subject_dn": "CN=RA Signing CA, O=Rainforest Alliance Inc, C=US",
        "key_algorithm": "ECDSA",
        "key_size_bits": 256,
        "fingerprint_sha256": "a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6",
        "is_root": False,
        "purpose": "UTZ/Rainforest Alliance certificate signing",
    },
]

# ---------------------------------------------------------------------------
# Computed totals
# ---------------------------------------------------------------------------

TOTAL_TRUSTED_CAS: int = len(TRUSTED_CAS)

# ---------------------------------------------------------------------------
# Category index: {category: [ca_entry, ...]}
# ---------------------------------------------------------------------------

CA_CATEGORY_INDEX: Dict[str, List[Dict[str, Any]]] = {}
for _ca in TRUSTED_CAS:
    _cat = _ca["ca_category"]
    if _cat not in CA_CATEGORY_INDEX:
        CA_CATEGORY_INDEX[_cat] = []
    CA_CATEGORY_INDEX[_cat].append(_ca)

TOTAL_CATEGORIES: int = len(CA_CATEGORY_INDEX)

# ---------------------------------------------------------------------------
# Pinned issuers: {certification_standard: [expected_ca_names]}
#
# These map known certification standards to the CA names expected to
# sign their certificates.  The CertificateChainValidator uses this
# to detect unauthorized issuers (FRD-006 rule).
# ---------------------------------------------------------------------------

PINNED_ISSUERS: Dict[str, List[str]] = {
    "fsc": [
        "ASI Document Signing CA (FSC accreditation)",
        "GlobalSign Document Signing CA - SHA256 - G3",
        "DigiCert Document Signing CA - SHA256",
    ],
    "rspo": [
        "RSPO Secretariat Signing CA",
        "GlobalSign Document Signing CA - SHA256 - G3",
        "DigiCert Document Signing CA - SHA256",
    ],
    "iscc": [
        "ISCC System Signing CA",
        "GlobalSign Document Signing CA - SHA256 - G3",
    ],
    "fairtrade": [
        "FLOCERT Signing CA",
        "GlobalSign Document Signing CA - SHA256 - G3",
    ],
    "utz_ra": [
        "Rainforest Alliance Signing CA",
        "GlobalSign Document Signing CA - SHA256 - G3",
    ],
    "eu_coo": [
        "D-TRUST Root Class 3 CA 2 2009",
        "D-TRUST Qualified Signing CA 2021",
        "Bundesdruckerei GmbH - D-TRUST Root",
        "ANSSI - Autorite de Certification Racine",
        "Staat der Nederlanden Root CA - G3",
        "Belgium Root CA4",
        "Agenzia per l'Italia Digitale - AgID Root",
        "FNMT-RCM - Fabrica Nacional de Moneda y Timbre",
        "SCEE - Sistema de Certificacao Electronica do Estado",
    ],
    "eu_phytosanitary": [
        "ANSSI - Autorite de Certification Racine",
        "Bundesdruckerei GmbH - D-TRUST Root",
        "Staat der Nederlanden Root CA - G3",
        "DigiCert EU Qualified Root CA",
    ],
}

# ---------------------------------------------------------------------------
# Lookup helper functions
# ---------------------------------------------------------------------------


def get_trusted_cas() -> List[Dict[str, Any]]:
    """Return all trusted CA entries.

    Returns:
        List of trusted CA specification dictionaries.

    Example:
        >>> cas = get_trusted_cas()
        >>> len(cas) >= 20
        True
    """
    return list(TRUSTED_CAS)


def get_cas_by_category(category: str) -> List[Dict[str, Any]]:
    """Return trusted CAs filtered by category.

    Args:
        category: CA category (eidas_tsp, document_signing,
            government, certification_body).

    Returns:
        List of CA entries matching the category.
        Returns empty list if the category is not found.

    Example:
        >>> eidas = get_cas_by_category("eidas_tsp")
        >>> len(eidas) >= 5
        True
    """
    cat_lower = category.lower().strip()
    return list(CA_CATEGORY_INDEX.get(cat_lower, []))


def get_ca_by_name(ca_name: str) -> Optional[Dict[str, Any]]:
    """Return a trusted CA entry by exact name.

    Args:
        ca_name: CA name to search for.

    Returns:
        CA entry dictionary, or None if not found.

    Example:
        >>> ca = get_ca_by_name("DigiCert Global Root G2")
        >>> ca["key_algorithm"]
        'RSA'
    """
    for ca in TRUSTED_CAS:
        if ca["ca_name"] == ca_name:
            return dict(ca)
    return None


def get_root_cas() -> List[Dict[str, Any]]:
    """Return only root CA entries (is_root=True).

    Returns:
        List of root CA entries.

    Example:
        >>> roots = get_root_cas()
        >>> all(ca["is_root"] for ca in roots)
        True
    """
    return [ca for ca in TRUSTED_CAS if ca["is_root"]]


def get_intermediate_cas() -> List[Dict[str, Any]]:
    """Return only intermediate CA entries (is_root=False).

    Returns:
        List of intermediate CA entries.

    Example:
        >>> intermediates = get_intermediate_cas()
        >>> all(not ca["is_root"] for ca in intermediates)
        True
    """
    return [ca for ca in TRUSTED_CAS if not ca["is_root"]]


def get_pinned_issuers() -> Dict[str, List[str]]:
    """Return the pinned issuer mapping for certification standards.

    Returns:
        Dictionary mapping standard identifiers to lists of
        expected CA names.

    Example:
        >>> issuers = get_pinned_issuers()
        >>> "fsc" in issuers
        True
    """
    return dict(PINNED_ISSUERS)


def get_pinned_issuers_for_standard(standard: str) -> List[str]:
    """Return pinned issuer CA names for a given standard.

    Args:
        standard: Standard identifier (fsc, rspo, iscc, fairtrade,
            utz_ra, eu_coo, eu_phytosanitary).

    Returns:
        List of expected CA names for the standard.
        Returns empty list if the standard is not found.

    Example:
        >>> issuers = get_pinned_issuers_for_standard("fsc")
        >>> len(issuers) >= 1
        True
    """
    std_lower = standard.lower().strip()
    return list(PINNED_ISSUERS.get(std_lower, []))


def get_all_categories() -> List[str]:
    """Return all CA category identifiers.

    Returns:
        Sorted list of category strings.

    Example:
        >>> categories = get_all_categories()
        >>> "eidas_tsp" in categories
        True
    """
    return sorted(CA_CATEGORY_INDEX.keys())


def is_trusted_ca(ca_name: str) -> bool:
    """Check if a CA name exists in the trusted registry.

    Args:
        ca_name: CA name to check.

    Returns:
        True if the CA is in the trusted registry.

    Example:
        >>> is_trusted_ca("DigiCert Global Root G2")
        True
        >>> is_trusted_ca("Unknown CA")
        False
    """
    return any(ca["ca_name"] == ca_name for ca in TRUSTED_CAS)


# ---------------------------------------------------------------------------
# Module-level logging on import
# ---------------------------------------------------------------------------

logger.debug(
    "Trusted CAs reference data loaded: "
    "%d CAs across %d categories, %d pinned issuer standards",
    TOTAL_TRUSTED_CAS,
    TOTAL_CATEGORIES,
    len(PINNED_ISSUERS),
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "TRUSTED_CAS",
    "CA_CATEGORY_INDEX",
    "PINNED_ISSUERS",
    # Counts
    "TOTAL_TRUSTED_CAS",
    "TOTAL_CATEGORIES",
    # Lookup helpers
    "get_trusted_cas",
    "get_cas_by_category",
    "get_ca_by_name",
    "get_root_cas",
    "get_intermediate_cas",
    "get_pinned_issuers",
    "get_pinned_issuers_for_standard",
    "get_all_categories",
    "is_trusted_ca",
]
