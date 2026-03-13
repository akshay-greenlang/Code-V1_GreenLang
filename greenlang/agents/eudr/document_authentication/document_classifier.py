# -*- coding: utf-8 -*-
"""
Document Classifier Engine - AGENT-EUDR-012: Document Authentication (Engine 1)

Deterministic rule-based document classification engine for EUDR supply chain
traceability. Classifies documents into 20+ EUDR-relevant types using template
matching, field pattern recognition, header analysis, serial number extraction,
keyword detection, and multi-language support.

Zero-Hallucination Guarantees:
    - All classification logic is deterministic regex/keyword matching
    - Confidence scores are computed via weighted feature scoring
    - No ML/LLM used for classification decisions
    - All template matches use pre-registered patterns
    - SHA-256 provenance hashes on every classification operation
    - Classification history is immutable and auditable

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Due diligence obligations
    - EU 2023/1115 (EUDR) Article 10: Document verification requirements
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - eIDAS Regulation (EU) No 910/2014: Electronic document standards

Performance Targets:
    - Single document classification: <50ms
    - Batch classification (100 docs): <3 seconds
    - Template matching: <10ms per template
    - Language detection: <5ms

Supported Document Types (20):
    COO, PC, BOL, CDE, CDI, RSPO_CERT, FSC_CERT, ISCC_CERT, FT_CERT,
    UTZ_CERT, LTR, LTD, FMP, FC, WQC, DDS_DRAFT, SSD, IC, TC, WR.

Supported Languages (7):
    EN (English), FR (French), DE (German), ES (Spanish),
    PT (Portuguese), ID (Indonesian), NL (Dutch).

PRD Feature References:
    - PRD-AGENT-EUDR-012 Feature 1: Document Classification Engine
    - PRD-AGENT-EUDR-012 Feature 1.1: Template-Based Classification
    - PRD-AGENT-EUDR-012 Feature 1.2: Field Pattern Matching
    - PRD-AGENT-EUDR-012 Feature 1.3: Multi-Page Handling
    - PRD-AGENT-EUDR-012 Feature 1.4: Confidence Scoring
    - PRD-AGENT-EUDR-012 Feature 1.5: Language Detection
    - PRD-AGENT-EUDR-012 Feature 1.6: Unknown Document Flagging
    - PRD-AGENT-EUDR-012 Feature 1.7: Template Registry Management
    - PRD-AGENT-EUDR-012 Feature 1.8: Classification History

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from greenlang.agents.eudr.document_authentication.config import get_config
from greenlang.agents.eudr.document_authentication.metrics import (
    observe_classification_duration,
    record_api_error,
    record_classification,
    record_document_processed,
)
from greenlang.agents.eudr.document_authentication.models import (
    ClassificationConfidence,
    ClassificationResult,
    DocumentLanguage,
    DocumentType,
)
from greenlang.agents.eudr.document_authentication.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants: MIME type categories
# ---------------------------------------------------------------------------

#: MIME types recognized as PDF documents.
_PDF_MIME_TYPES: FrozenSet[str] = frozenset({
    "application/pdf",
    "application/x-pdf",
})

#: MIME types recognized as image documents.
_IMAGE_MIME_TYPES: FrozenSet[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/gif",
    "image/webp",
})

#: MIME types recognized as XML documents.
_XML_MIME_TYPES: FrozenSet[str] = frozenset({
    "application/xml",
    "text/xml",
})

#: MIME types recognized as JSON documents.
_JSON_MIME_TYPES: FrozenSet[str] = frozenset({
    "application/json",
    "text/json",
})

#: All supported MIME types.
_SUPPORTED_MIME_TYPES: FrozenSet[str] = (
    _PDF_MIME_TYPES
    | _IMAGE_MIME_TYPES
    | _XML_MIME_TYPES
    | _JSON_MIME_TYPES
    | frozenset({
        "application/octet-stream",
        "application/pkcs7-mime",
        "application/pkcs7-signature",
    })
)

# ---------------------------------------------------------------------------
# Constants: Language detection keyword sets
# ---------------------------------------------------------------------------

_LANGUAGE_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "certificate", "origin", "phytosanitary", "bill of lading",
        "customs", "declaration", "export", "import", "sustainability",
        "forest", "management", "felling", "invoice", "receipt",
        "shipment", "consignment", "authorized", "hereby", "certify",
        "in accordance with", "regulation", "compliance",
    ],
    "fr": [
        "certificat", "origine", "phytosanitaire", "connaissement",
        "douane", "declaration", "exportation", "importation",
        "durabilite", "forestier", "gestion", "abattage", "facture",
        "recepisse", "expedition", "autorise", "certifie",
        "conformement", "reglement",
    ],
    "de": [
        "zertifikat", "ursprung", "pflanzengesundheit", "konnossement",
        "zoll", "erklarung", "ausfuhr", "einfuhr", "nachhaltigkeit",
        "forstwirtschaft", "verwaltung", "einschlag", "rechnung",
        "quittung", "lieferung", "genehmigt", "bescheinigt",
        "gemaess", "verordnung",
    ],
    "es": [
        "certificado", "origen", "fitosanitario", "conocimiento de embarque",
        "aduana", "declaracion", "exportacion", "importacion",
        "sostenibilidad", "forestal", "gestion", "tala", "factura",
        "recibo", "envio", "autorizado", "certifico",
        "conforme", "reglamento",
    ],
    "pt": [
        "certificado", "origem", "fitossanitario", "conhecimento de embarque",
        "alfandega", "declaracao", "exportacao", "importacao",
        "sustentabilidade", "florestal", "gestao", "corte", "fatura",
        "recibo", "remessa", "autorizado", "certificamos",
        "conforme", "regulamento",
    ],
    "id": [
        "sertifikat", "asal", "karantina", "konosemen",
        "bea cukai", "deklarasi", "ekspor", "impor",
        "keberlanjutan", "kehutanan", "pengelolaan", "penebangan",
        "faktur", "kwitansi", "pengiriman", "diotorisasi",
        "menyatakan", "sesuai", "peraturan",
    ],
    "nl": [
        "certificaat", "oorsprong", "fytosanitair", "cognossement",
        "douane", "verklaring", "uitvoer", "invoer",
        "duurzaamheid", "bosbouw", "beheer", "kap", "factuur",
        "ontvangst", "zending", "bevoegd", "certificeren",
        "overeenkomstig", "verordening",
    ],
}

# ---------------------------------------------------------------------------
# Constants: Document type keyword patterns
# ---------------------------------------------------------------------------

_DOCUMENT_TYPE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "coo": {
        "keywords": [
            "certificate of origin", "certificat d'origine",
            "ursprungszeugnis", "certificado de origen",
            "certificado de origem", "sertifikat asal",
            "certificaat van oorsprong",
        ],
        "header_patterns": [
            r"certificate\s+of\s+origin",
            r"(?:form\s+)?[a-e]\s+certificate\s+of\s+origin",
            r"generalised?\s+system\s+of\s+preferences",
            r"gsp\s+form\s+[a-e]",
        ],
        "serial_patterns": [
            r"[A-Z]{2,3}[-/]\d{4,}[-/]\d{2,}",
            r"COO[-/]\d{6,}",
        ],
        "weight": 1.0,
    },
    "pc": {
        "keywords": [
            "phytosanitary certificate", "certificat phytosanitaire",
            "pflanzengesundheitszeugnis", "certificado fitosanitario",
            "certificado fitossanitario", "sertifikat karantina",
            "fytosanitair certificaat", "plant health",
        ],
        "header_patterns": [
            r"phytosanitary\s+certificate",
            r"plant\s+health\s+certificate",
            r"ippc\s+model\s+certificate",
        ],
        "serial_patterns": [
            r"PC[-/]\d{4,}",
            r"PHYTO[-/]\d{4,}",
            r"\d{2}/\d{4}/\d{4,}",
        ],
        "weight": 1.0,
    },
    "bol": {
        "keywords": [
            "bill of lading", "connaissement", "konnossement",
            "conocimiento de embarque", "conhecimento de embarque",
            "konosemen", "cognossement",
        ],
        "header_patterns": [
            r"bill\s+of\s+lading",
            r"b/?l\s+(?:no|number|#)",
            r"shipped\s+on\s+board",
            r"ocean\s+bill\s+of\s+lading",
            r"sea\s+waybill",
        ],
        "serial_patterns": [
            r"[A-Z]{4}\d{7,}",
            r"BL[-/]\d{6,}",
            r"[A-Z]{3}[-/]\d{4}[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "cde": {
        "keywords": [
            "customs declaration export", "export declaration",
            "declaration d'exportation", "ausfuhrerklarung",
            "declaracion de exportacion", "declaracao de exportacao",
            "deklarasi ekspor", "verklaring uitvoer",
        ],
        "header_patterns": [
            r"export\s+(?:customs?\s+)?declaration",
            r"single\s+administrative\s+document",
            r"sad\s+export",
            r"customs?\s+export",
        ],
        "serial_patterns": [
            r"EX[-/]\d{4,}",
            r"MRN[-/]?\d{2}[A-Z]{2}\d{10,}",
            r"SAD[-/]\d{6,}",
        ],
        "weight": 0.9,
    },
    "cdi": {
        "keywords": [
            "customs declaration import", "import declaration",
            "declaration d'importation", "einfuhrerklarung",
            "declaracion de importacion", "declaracao de importacao",
            "deklarasi impor", "verklaring invoer",
        ],
        "header_patterns": [
            r"import\s+(?:customs?\s+)?declaration",
            r"single\s+administrative\s+document",
            r"sad\s+import",
            r"customs?\s+import",
            r"entry\s+summary",
        ],
        "serial_patterns": [
            r"IM[-/]\d{4,}",
            r"MRN[-/]?\d{2}[A-Z]{2}\d{10,}",
            r"SAD[-/]\d{6,}",
        ],
        "weight": 0.9,
    },
    "rspo_cert": {
        "keywords": [
            "rspo", "roundtable on sustainable palm oil",
            "rspo supply chain certification",
            "rspo identity preserved", "rspo segregated",
            "rspo mass balance", "rspo book and claim",
            "sustainable palm oil",
        ],
        "header_patterns": [
            r"rspo\s+(?:supply\s+chain\s+)?certific",
            r"roundtable\s+on\s+sustainable\s+palm\s+oil",
            r"rspo\s+(?:ip|sg|mb|bc)\s+certif",
        ],
        "serial_patterns": [
            r"RSPO[-/]\d{4,}",
            r"[A-Z]{2,3}[-/]RSPO[-/]\d{4,}",
            r"ASI[-/]RSPO[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "fsc_cert": {
        "keywords": [
            "fsc", "forest stewardship council",
            "fsc chain of custody", "fsc-coc",
            "fsc controlled wood", "fsc mix",
            "forest management certification",
        ],
        "header_patterns": [
            r"fsc\s+(?:chain\s+of\s+custody|coc)\s+certific",
            r"forest\s+stewardship\s+council",
            r"fsc[-\s](?:c|fm|cw|mix)\d{5,}",
        ],
        "serial_patterns": [
            r"FSC[-/]C\d{5,}",
            r"FSC[-/]FM[-/]?\d{5,}",
            r"[A-Z]{2,3}[-/]COC[-/]\d{4,}",
            r"ASI[-/]FSC[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "iscc_cert": {
        "keywords": [
            "iscc", "international sustainability",
            "carbon certification", "iscc plus",
            "iscc eu", "iscc corsia",
            "biomass sustainability",
        ],
        "header_patterns": [
            r"iscc\s+(?:plus|eu|corsia)?\s*certific",
            r"international\s+sustainability\s+(?:and|&)\s+carbon",
        ],
        "serial_patterns": [
            r"ISCC[-/]\d{4,}",
            r"[A-Z]{2}[-/]ISCC[-/]\d{4,}",
            r"ISCC[-/][A-Z]{2,3}[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "ft_cert": {
        "keywords": [
            "fairtrade", "fair trade", "fairtrade international",
            "flo-cert", "flocert", "fairtrade certified",
            "fairtrade premium",
        ],
        "header_patterns": [
            r"fairtrade?\s+certific",
            r"flo[-\s]?cert",
            r"fairtrade?\s+international",
        ],
        "serial_patterns": [
            r"FLO[-/]\d{4,}",
            r"FT[-/]\d{4,}",
            r"FLOCERT[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "utz_cert": {
        "keywords": [
            "utz", "rainforest alliance", "utz certified",
            "ra certified", "rainforest alliance certified",
            "sustainable agriculture network",
        ],
        "header_patterns": [
            r"utz\s+certif",
            r"rainforest\s+alliance\s+certif",
            r"sustainable\s+agriculture\s+network",
        ],
        "serial_patterns": [
            r"UTZ[-/]\d{4,}",
            r"RA[-/]\d{4,}",
            r"SAN[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "ltr": {
        "keywords": [
            "land title", "title deed", "titre foncier",
            "grundbuch", "titulo de propiedad",
            "titulo de propriedade", "sertifikat tanah",
            "eigendomsbewijs",
        ],
        "header_patterns": [
            r"land\s+title\s+(?:register|record|deed)",
            r"title\s+deed",
            r"certificate\s+of\s+(?:title|ownership)",
        ],
        "serial_patterns": [
            r"LT[-/]\d{4,}",
            r"TITLE[-/]\d{4,}",
            r"\d{2}/\d{4}/\d{4,}",
        ],
        "weight": 0.9,
    },
    "ltd": {
        "keywords": [
            "land tenure", "tenure document", "concession",
            "land use right", "droit d'usage", "nutzungsrecht",
            "derecho de uso", "direito de uso",
            "hak guna usaha", "gebruiksrecht",
        ],
        "header_patterns": [
            r"land\s+tenure\s+(?:document|agreement|certificate)",
            r"concession\s+(?:agreement|licence|license)",
            r"land\s+use\s+(?:right|agreement|permit)",
        ],
        "serial_patterns": [
            r"LTD[-/]\d{4,}",
            r"CONC[-/]\d{4,}",
            r"HGU[-/]\d{4,}",
        ],
        "weight": 0.8,
    },
    "fmp": {
        "keywords": [
            "forest management plan", "plan d'amenagement",
            "forstbetriebsplan", "plan de manejo forestal",
            "plano de manejo florestal", "rencana pengelolaan hutan",
            "bosbeheerplan",
        ],
        "header_patterns": [
            r"forest\s+management\s+plan",
            r"sustainable\s+forest\s+management",
            r"annual\s+allowable\s+cut",
            r"forest\s+(?:inventory|harvesting)\s+plan",
        ],
        "serial_patterns": [
            r"FMP[-/]\d{4,}",
            r"IUPHHK[-/]\d{4,}",
        ],
        "weight": 0.8,
    },
    "fc": {
        "keywords": [
            "felling certificate", "felling licence", "felling license",
            "timber harvesting", "permis de coupe",
            "einschlaggenehmigung", "permiso de tala",
            "autorizacao de corte", "izin penebangan",
            "kapvergunning",
        ],
        "header_patterns": [
            r"felling\s+(?:certificate|licence|license|permit)",
            r"timber\s+harvesting\s+(?:certificate|licence|license|permit)",
            r"logging\s+(?:permit|concession)",
        ],
        "serial_patterns": [
            r"FC[-/]\d{4,}",
            r"FEL[-/]\d{4,}",
            r"THP[-/]\d{4,}",
        ],
        "weight": 0.9,
    },
    "wqc": {
        "keywords": [
            "wood quality certificate", "timber quality",
            "species identification", "wood grading",
            "certificat de qualite du bois",
            "holzqualitatszertifikat",
        ],
        "header_patterns": [
            r"wood\s+quality\s+certific",
            r"timber\s+(?:quality|grading)\s+certific",
            r"species\s+identification\s+certific",
        ],
        "serial_patterns": [
            r"WQC[-/]\d{4,}",
            r"TQC[-/]\d{4,}",
        ],
        "weight": 0.8,
    },
    "dds_draft": {
        "keywords": [
            "due diligence statement", "dds", "due diligence",
            "eudr due diligence", "deforestation-free",
            "declaration de diligence", "sorgfaltserklarung",
        ],
        "header_patterns": [
            r"due\s+diligence\s+statement",
            r"dds\s+(?:draft|submission)",
            r"eudr\s+due\s+diligence",
            r"deforestation[\s-]+free\s+(?:declaration|statement)",
        ],
        "serial_patterns": [
            r"DDS[-/]\d{4,}",
            r"EUDR[-/]DDS[-/]\d{4,}",
        ],
        "weight": 1.0,
    },
    "ssd": {
        "keywords": [
            "supplier self-declaration", "self-declaration",
            "supplier declaration", "auto-declaration",
            "selbsterklarung", "autodeclaracion",
        ],
        "header_patterns": [
            r"supplier\s+self[\s-]+declaration",
            r"self[\s-]+declaration\s+(?:form|statement)",
            r"supplier\s+(?:compliance\s+)?declaration",
        ],
        "serial_patterns": [
            r"SSD[-/]\d{4,}",
            r"SDEC[-/]\d{4,}",
        ],
        "weight": 0.8,
    },
    "ic": {
        "keywords": [
            "commercial invoice", "invoice", "facture commerciale",
            "handelsrechnung", "factura comercial",
            "fatura comercial", "faktur", "factuur",
        ],
        "header_patterns": [
            r"commercial\s+invoice",
            r"(?:pro\s*forma\s+)?invoice\s+(?:no|number|#|date)",
            r"tax\s+invoice",
        ],
        "serial_patterns": [
            r"INV[-/]\d{4,}",
            r"CI[-/]\d{4,}",
            r"[A-Z]{2,3}[-/]\d{4}[-/]\d{4,}",
        ],
        "weight": 0.7,
    },
    "tc": {
        "keywords": [
            "transit certificate", "transit document",
            "document de transit", "transitdokument",
            "certificado de transito", "surat transit",
            "transitdocument",
        ],
        "header_patterns": [
            r"transit\s+(?:certificate|document)",
            r"community\s+transit",
            r"t1\s+document",
            r"customs?\s+transit",
        ],
        "serial_patterns": [
            r"TC[-/]\d{4,}",
            r"T1[-/]\d{6,}",
            r"TRANSIT[-/]\d{4,}",
        ],
        "weight": 0.8,
    },
    "wr": {
        "keywords": [
            "weighbridge receipt", "weighbridge", "weight certificate",
            "bon de pesee", "wiegeschein", "boleta de pesaje",
            "nota de pesagem", "kwitansi timbang",
            "weegbon",
        ],
        "header_patterns": [
            r"weighbridge\s+(?:receipt|ticket|certificate)",
            r"weight\s+(?:receipt|ticket|certificate|note)",
            r"net\s+weight\s*:?\s*\d",
            r"gross\s+weight\s*:?\s*\d",
        ],
        "serial_patterns": [
            r"WR[-/]\d{4,}",
            r"WB[-/]\d{4,}",
            r"WT[-/]\d{4,}",
        ],
        "weight": 0.7,
    },
}

#: Document types that map to DocumentType enum values.
_DOCTYPE_ENUM_MAP: Dict[str, DocumentType] = {
    "coo": DocumentType.COO,
    "pc": DocumentType.PC,
    "bol": DocumentType.BOL,
    "cde": DocumentType.CDE,
    "cdi": DocumentType.CDI,
    "rspo_cert": DocumentType.RSPO_CERT,
    "fsc_cert": DocumentType.FSC_CERT,
    "iscc_cert": DocumentType.ISCC_CERT,
    "ft_cert": DocumentType.FT_CERT,
    "utz_cert": DocumentType.UTZ_CERT,
    "ltr": DocumentType.LTR,
    "ltd": DocumentType.LTD,
    "fmp": DocumentType.FMP,
    "fc": DocumentType.FC,
    "wqc": DocumentType.WQC,
    "dds_draft": DocumentType.DDS_DRAFT,
    "ssd": DocumentType.SSD,
    "ic": DocumentType.IC,
    "tc": DocumentType.TC,
    "wr": DocumentType.WR,
}

#: Language code to DocumentLanguage enum mapping.
_LANGUAGE_ENUM_MAP: Dict[str, DocumentLanguage] = {
    "en": DocumentLanguage.EN,
    "fr": DocumentLanguage.FR,
    "de": DocumentLanguage.DE,
    "es": DocumentLanguage.ES,
    "pt": DocumentLanguage.PT,
    "id": DocumentLanguage.ID,
    "nl": DocumentLanguage.NL,
}

# ---------------------------------------------------------------------------
# Template dataclass
# ---------------------------------------------------------------------------


class DocumentTemplate:
    """A registered template for a known document type and issuing authority.

    Attributes:
        template_id: Unique template identifier.
        document_type: EUDR document type this template represents.
        issuing_authority: Name of the issuing authority.
        country_code: ISO 3166-1 alpha-2 country code.
        header_patterns: Compiled regex patterns for header matching.
        serial_patterns: Compiled regex patterns for serial numbers.
        required_keywords: Keywords that must be present for a match.
        optional_keywords: Keywords that boost confidence when present.
        field_layout: Expected field positions and labels.
        version: Template version string.
        active: Whether the template is active.
        created_at: UTC creation timestamp.
        retired_at: Optional UTC retirement timestamp.
    """

    __slots__ = (
        "template_id", "document_type", "issuing_authority",
        "country_code", "header_patterns", "serial_patterns",
        "required_keywords", "optional_keywords", "field_layout",
        "version", "active", "created_at", "retired_at",
    )

    def __init__(
        self,
        document_type: str,
        issuing_authority: str,
        country_code: str,
        header_patterns: Optional[List[str]] = None,
        serial_patterns: Optional[List[str]] = None,
        required_keywords: Optional[List[str]] = None,
        optional_keywords: Optional[List[str]] = None,
        field_layout: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
        template_id: Optional[str] = None,
    ) -> None:
        """Initialize DocumentTemplate.

        Args:
            document_type: EUDR document type identifier.
            issuing_authority: Name of the issuing authority.
            country_code: ISO 3166-1 alpha-2 country code.
            header_patterns: Regex patterns for header matching.
            serial_patterns: Regex patterns for serial number matching.
            required_keywords: Keywords that must be present.
            optional_keywords: Keywords that boost confidence.
            field_layout: Expected field positions and labels.
            version: Template version string.
            template_id: Optional explicit template ID.
        """
        self.template_id = template_id or _generate_id()
        self.document_type = document_type.lower().strip()
        self.issuing_authority = issuing_authority.strip()
        self.country_code = country_code.upper().strip()
        self.header_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (header_patterns or [])
        ]
        self.serial_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (serial_patterns or [])
        ]
        self.required_keywords = [
            kw.lower() for kw in (required_keywords or [])
        ]
        self.optional_keywords = [
            kw.lower() for kw in (optional_keywords or [])
        ]
        self.field_layout = field_layout or {}
        self.version = version
        self.active = True
        self.created_at = _utcnow()
        self.retired_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize template to a dictionary for provenance.

        Returns:
            Dictionary representation of this template.
        """
        return {
            "template_id": self.template_id,
            "document_type": self.document_type,
            "issuing_authority": self.issuing_authority,
            "country_code": self.country_code,
            "header_pattern_count": len(self.header_patterns),
            "serial_pattern_count": len(self.serial_patterns),
            "required_keyword_count": len(self.required_keywords),
            "optional_keyword_count": len(self.optional_keywords),
            "version": self.version,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "retired_at": (
                self.retired_at.isoformat() if self.retired_at else None
            ),
        }


# ---------------------------------------------------------------------------
# TemplateMatchResult (internal)
# ---------------------------------------------------------------------------


class _TemplateMatchResult:
    """Internal result of a single template match attempt.

    Attributes:
        template_id: ID of the matched template.
        document_type: Detected document type.
        score: Match score (0.0-1.0).
        header_matched: Whether header patterns matched.
        serial_matched: Whether serial number patterns matched.
        required_keyword_hits: Number of required keywords found.
        optional_keyword_hits: Number of optional keywords found.
        matched_serial: The matched serial number string, if any.
    """

    __slots__ = (
        "template_id", "document_type", "score",
        "header_matched", "serial_matched",
        "required_keyword_hits", "optional_keyword_hits",
        "matched_serial",
    )

    def __init__(self) -> None:
        """Initialize with default values."""
        self.template_id: str = ""
        self.document_type: str = ""
        self.score: float = 0.0
        self.header_matched: bool = False
        self.serial_matched: bool = False
        self.required_keyword_hits: int = 0
        self.optional_keyword_hits: int = 0
        self.matched_serial: Optional[str] = None


# ---------------------------------------------------------------------------
# DocumentClassifierEngine
# ---------------------------------------------------------------------------


class DocumentClassifierEngine:
    """Deterministic rule-based document classification engine for EUDR compliance.

    Classifies documents into 20 EUDR-relevant types using a multi-signal
    approach combining template matching, header pattern recognition,
    serial number extraction, keyword detection, MIME type analysis,
    filename heuristics, and multi-language support.

    All classification logic is deterministic -- no ML or LLM is used.
    Confidence scores are computed via weighted feature scoring. Every
    classification operation produces a SHA-256 provenance hash for
    tamper-evident audit trails per EUDR Article 14.

    Thread Safety:
        All public methods are thread-safe via reentrant locking.
        The template registry and classification history are guarded
        by the same lock.

    Attributes:
        _config: Document authentication configuration singleton.
        _provenance: ProvenanceTracker for audit trail hashing.
        _templates: In-memory template registry keyed by template_id.
        _type_templates: Mapping of document_type to list of template_ids.
        _classification_history: Ordered list of classification results.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = DocumentClassifierEngine()
        >>> result = engine.classify_document(
        ...     document_bytes=b"certificate of origin...",
        ...     filename="coo_brazil_2025.pdf",
        ...     mime_type="application/pdf",
        ... )
        >>> assert result.confidence_level in ("high", "medium", "low", "unknown")
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize DocumentClassifierEngine.

        Args:
            config: Optional DocumentAuthenticationConfig override.
                If None, uses the singleton from get_config().
            provenance: Optional ProvenanceTracker override. If None,
                creates a new instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker(
            genesis_hash=self._config.genesis_hash,
        )

        # -- Template registry -------------------------------------------
        self._templates: Dict[str, DocumentTemplate] = {}
        self._type_templates: Dict[str, List[str]] = {}

        # -- Classification history --------------------------------------
        self._classification_history: List[Dict[str, Any]] = []

        # -- Compiled built-in patterns ----------------------------------
        self._builtin_header_patterns: Dict[str, List[Any]] = {}
        self._builtin_serial_patterns: Dict[str, List[Any]] = {}
        self._builtin_keywords: Dict[str, List[str]] = {}
        self._compile_builtin_patterns()

        # -- Thread safety -----------------------------------------------
        self._lock = threading.RLock()

        logger.info(
            "DocumentClassifierEngine initialized: module_version=%s, "
            "builtin_types=%d, confidence_high=%.2f, "
            "confidence_medium=%.2f",
            _MODULE_VERSION,
            len(_DOCUMENT_TYPE_PATTERNS),
            self._config.min_confidence_high,
            self._config.min_confidence_medium,
        )

    # ------------------------------------------------------------------
    # Built-in pattern compilation
    # ------------------------------------------------------------------

    def _compile_builtin_patterns(self) -> None:
        """Compile built-in regex patterns for all document types.

        Pre-compiles header and serial number regex patterns from the
        _DOCUMENT_TYPE_PATTERNS constant for efficient repeated use.
        """
        for doc_type, patterns in _DOCUMENT_TYPE_PATTERNS.items():
            self._builtin_header_patterns[doc_type] = [
                re.compile(p, re.IGNORECASE)
                for p in patterns.get("header_patterns", [])
            ]
            self._builtin_serial_patterns[doc_type] = [
                re.compile(p, re.IGNORECASE)
                for p in patterns.get("serial_patterns", [])
            ]
            self._builtin_keywords[doc_type] = [
                kw.lower() for kw in patterns.get("keywords", [])
            ]

    # ------------------------------------------------------------------
    # Public API: Single Document Classification
    # ------------------------------------------------------------------

    def classify_document(
        self,
        document_bytes: bytes,
        filename: str,
        mime_type: str = "application/octet-stream",
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """Classify a single document into an EUDR document type.

        Performs multi-signal classification using:
            1. Filename heuristics (extension, naming conventions)
            2. MIME type analysis
            3. Content text extraction (first 8KB decoded as UTF-8)
            4. Header pattern matching against 20 document types
            5. Serial number pattern matching
            6. Keyword frequency scoring
            7. Registered template matching
            8. Language detection

        The highest-scoring document type is selected, and confidence
        is computed as a weighted average of matched features.

        Args:
            document_bytes: Raw document content in bytes.
            filename: Original filename of the document.
            mime_type: MIME type of the document.
            document_id: Optional document ID. Auto-generated if not
                provided.
            metadata: Optional additional metadata for provenance.

        Returns:
            ClassificationResult with predicted type, confidence score,
            alternative types, detected features, and provenance hash.

        Raises:
            ValueError: If document_bytes is empty or filename is empty.
        """
        start_time = time.monotonic()
        doc_id = document_id or _generate_id()

        try:
            # -- Validate inputs -------------------------------------------
            self._validate_classify_inputs(document_bytes, filename)

            # -- Extract text content --------------------------------------
            text_content = self._extract_text_content(document_bytes)
            text_lower = text_content.lower()

            # -- Detect language -------------------------------------------
            detected_language = self._detect_language(text_lower)

            # -- Compute scores per document type --------------------------
            type_scores: Dict[str, float] = {}
            features_extracted: List[str] = []

            # Signal 1: Filename heuristics
            filename_scores = self._score_by_filename(filename)
            self._merge_scores(type_scores, filename_scores, weight=0.15)
            if filename_scores:
                features_extracted.append("filename_heuristic")

            # Signal 2: MIME type
            mime_scores = self._score_by_mime_type(mime_type)
            self._merge_scores(type_scores, mime_scores, weight=0.05)
            if mime_scores:
                features_extracted.append("mime_type")

            # Signal 3: Header pattern matching
            header_scores = self._score_by_header_patterns(text_lower)
            self._merge_scores(type_scores, header_scores, weight=0.30)
            if header_scores:
                features_extracted.append("header_pattern")

            # Signal 4: Serial number matching
            serial_scores, serial_str = self._score_by_serial_patterns(
                text_content,
            )
            self._merge_scores(type_scores, serial_scores, weight=0.20)
            if serial_scores:
                features_extracted.append("serial_number")

            # Signal 5: Keyword frequency scoring
            keyword_scores = self._score_by_keywords(text_lower)
            self._merge_scores(type_scores, keyword_scores, weight=0.20)
            if keyword_scores:
                features_extracted.append("keyword_frequency")

            # Signal 6: Registered template matching
            template_scores = self._score_by_templates(
                text_lower, text_content,
            )
            self._merge_scores(type_scores, template_scores, weight=0.10)
            if template_scores:
                features_extracted.append("template_match")

            # -- Select best type ------------------------------------------
            predicted_type, confidence_score, alternatives = (
                self._select_best_type(type_scores)
            )

            # -- Determine confidence level --------------------------------
            confidence_level = self._compute_confidence_level(
                confidence_score,
            )

            # -- Build result ----------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            provenance_data = {
                "document_id": doc_id,
                "predicted_type": predicted_type,
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "features": features_extracted,
                "language": detected_language,
            }
            prov_hash = _compute_hash(provenance_data)

            result = ClassificationResult(
                document_id=doc_id,
                predicted_type=_DOCTYPE_ENUM_MAP.get(
                    predicted_type, DocumentType.COO,
                ),
                confidence_score=confidence_score,
                confidence_level=ClassificationConfidence(confidence_level),
                alternative_types=alternatives,
                features_extracted=features_extracted,
                processing_time_ms=round(elapsed_ms, 2),
                provenance_hash=prov_hash,
            )

            # -- Record provenance -----------------------------------------
            if self._config.enable_provenance:
                self._provenance.record(
                    entity_type="classification",
                    action="classify",
                    entity_id=doc_id,
                    data=provenance_data,
                    metadata={
                        "document_id": doc_id,
                        "predicted_type": predicted_type,
                        "confidence_level": confidence_level,
                        "language": detected_language,
                    },
                )

            # -- Record metrics --------------------------------------------
            if self._config.enable_metrics:
                record_document_processed(predicted_type)
                record_classification(predicted_type, confidence_level)
                observe_classification_duration(elapsed_ms / 1000.0)

            # -- Record history --------------------------------------------
            self._record_classification_history(
                doc_id, predicted_type, confidence_score,
                confidence_level, detected_language, elapsed_ms,
            )

            logger.info(
                "Document classified: doc_id=%s type=%s confidence=%.3f "
                "level=%s language=%s time=%.1fms features=%d",
                doc_id[:12],
                predicted_type,
                confidence_score,
                confidence_level,
                detected_language,
                elapsed_ms,
                len(features_extracted),
            )

            return result

        except ValueError:
            raise
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "Classification failed for doc_id=%s: %s (%.1fms)",
                doc_id[:12], str(e), elapsed_ms,
                exc_info=True,
            )
            if self._config.enable_metrics:
                record_api_error("classify")
            raise

    # ------------------------------------------------------------------
    # Public API: Batch Classification
    # ------------------------------------------------------------------

    def batch_classify(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[ClassificationResult]:
        """Classify multiple documents in a single batch operation.

        Each document in the batch is classified independently. The
        batch size is limited by the configuration max_batch_size.
        Failures for individual documents are logged but do not abort
        the entire batch.

        Args:
            documents: List of document dictionaries, each containing:
                - document_bytes (bytes): Raw document content
                - filename (str): Original filename
                - mime_type (str, optional): MIME type
                - document_id (str, optional): Document ID
                - metadata (dict, optional): Additional metadata

        Returns:
            List of ClassificationResult objects, one per document.
            Failed classifications are represented with UNKNOWN
            confidence and a provenance hash noting the failure.

        Raises:
            ValueError: If documents list is empty or exceeds batch limit.
        """
        start_time = time.monotonic()

        if not documents:
            raise ValueError("documents list must not be empty")

        max_size = self._config.max_batch_size
        if len(documents) > max_size:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum "
                f"of {max_size}"
            )

        results: List[ClassificationResult] = []
        success_count = 0
        failure_count = 0

        for idx, doc in enumerate(documents):
            try:
                doc_bytes = doc.get("document_bytes", b"")
                fname = doc.get("filename", f"unknown_{idx}")
                mtype = doc.get("mime_type", "application/octet-stream")
                doc_id = doc.get("document_id")
                meta = doc.get("metadata")

                result = self.classify_document(
                    document_bytes=doc_bytes,
                    filename=fname,
                    mime_type=mtype,
                    document_id=doc_id,
                    metadata=meta,
                )
                results.append(result)
                success_count += 1

            except Exception as e:
                failure_count += 1
                error_doc_id = doc.get("document_id", _generate_id())
                logger.warning(
                    "Batch classify failed for document[%d] "
                    "doc_id=%s: %s",
                    idx, str(error_doc_id)[:12], str(e),
                )
                error_result = ClassificationResult(
                    document_id=error_doc_id,
                    predicted_type=DocumentType.COO,
                    confidence_score=0.0,
                    confidence_level=ClassificationConfidence.UNKNOWN,
                    alternative_types={},
                    features_extracted=["error"],
                    processing_time_ms=0.0,
                    provenance_hash=_compute_hash(
                        {"error": str(e), "document_id": error_doc_id},
                    ),
                )
                results.append(error_result)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        logger.info(
            "Batch classification complete: total=%d success=%d "
            "failure=%d time=%.1fms",
            len(documents), success_count, failure_count, elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: Template Registry
    # ------------------------------------------------------------------

    def register_template(
        self,
        document_type: str,
        issuing_authority: str,
        country_code: str,
        header_patterns: Optional[List[str]] = None,
        serial_patterns: Optional[List[str]] = None,
        required_keywords: Optional[List[str]] = None,
        optional_keywords: Optional[List[str]] = None,
        field_layout: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ) -> Dict[str, Any]:
        """Register a new document template for classification.

        Templates augment the built-in classification patterns with
        issuing-authority-specific patterns for higher precision
        matching. Each template is associated with a document type,
        issuing authority, and country code.

        Args:
            document_type: EUDR document type identifier.
            issuing_authority: Name of the issuing authority.
            country_code: ISO 3166-1 alpha-2 country code.
            header_patterns: Regex patterns for header matching.
            serial_patterns: Regex patterns for serial number matching.
            required_keywords: Keywords that must be present.
            optional_keywords: Keywords that boost confidence.
            field_layout: Expected field positions and labels.
            version: Template version string.

        Returns:
            Dictionary with template_id, status, and provenance_hash.

        Raises:
            ValueError: If document_type or issuing_authority are empty.
        """
        if not document_type or not document_type.strip():
            raise ValueError("document_type must not be empty")
        if not issuing_authority or not issuing_authority.strip():
            raise ValueError("issuing_authority must not be empty")
        if not country_code or not country_code.strip():
            raise ValueError("country_code must not be empty")

        template = DocumentTemplate(
            document_type=document_type,
            issuing_authority=issuing_authority,
            country_code=country_code,
            header_patterns=header_patterns,
            serial_patterns=serial_patterns,
            required_keywords=required_keywords,
            optional_keywords=optional_keywords,
            field_layout=field_layout,
            version=version,
        )

        with self._lock:
            self._templates[template.template_id] = template
            doc_type_key = template.document_type
            if doc_type_key not in self._type_templates:
                self._type_templates[doc_type_key] = []
            self._type_templates[doc_type_key].append(template.template_id)

        # -- Provenance tracking -------------------------------------------
        prov_hash = _compute_hash(template.to_dict())
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="template",
                action="register_template",
                entity_id=template.template_id,
                data=template.to_dict(),
                metadata={
                    "document_type": template.document_type,
                    "issuing_authority": template.issuing_authority,
                    "country_code": template.country_code,
                },
            )

        logger.info(
            "Template registered: id=%s type=%s authority=%s "
            "country=%s version=%s",
            template.template_id[:12],
            template.document_type,
            template.issuing_authority,
            template.country_code,
            template.version,
        )

        return {
            "template_id": template.template_id,
            "document_type": template.document_type,
            "issuing_authority": template.issuing_authority,
            "country_code": template.country_code,
            "version": template.version,
            "status": "registered",
            "provenance_hash": prov_hash,
            "created_at": template.created_at.isoformat(),
        }

    def retire_template(self, template_id: str) -> Dict[str, Any]:
        """Retire a registered template.

        Retired templates are marked inactive and no longer used for
        classification but remain in the registry for audit trail.

        Args:
            template_id: ID of the template to retire.

        Returns:
            Dictionary with template_id, status, and retired_at.

        Raises:
            ValueError: If template_id is not found.
        """
        with self._lock:
            template = self._templates.get(template_id)
            if template is None:
                raise ValueError(
                    f"Template not found: {template_id}"
                )
            template.active = False
            template.retired_at = _utcnow()

        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="template",
                action="register_template",
                entity_id=template_id,
                data={"action": "retire", "template_id": template_id},
                metadata={"retired_at": template.retired_at.isoformat()},
            )

        logger.info(
            "Template retired: id=%s type=%s",
            template_id[:12], template.document_type,
        )

        return {
            "template_id": template_id,
            "document_type": template.document_type,
            "status": "retired",
            "retired_at": template.retired_at.isoformat(),
        }

    def get_templates(
        self,
        document_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve registered templates optionally filtered by type.

        Args:
            document_type: Optional document type filter.
            active_only: Whether to return only active templates.

        Returns:
            List of template dictionaries.
        """
        with self._lock:
            templates = list(self._templates.values())

        if document_type:
            doc_type_lower = document_type.lower().strip()
            templates = [
                t for t in templates
                if t.document_type == doc_type_lower
            ]

        if active_only:
            templates = [t for t in templates if t.active]

        return [t.to_dict() for t in templates]

    # ------------------------------------------------------------------
    # Public API: Classification History
    # ------------------------------------------------------------------

    def get_classification_history(
        self,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        confidence_level: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve classification history with optional filters.

        Args:
            document_id: Filter by document ID.
            document_type: Filter by predicted document type.
            confidence_level: Filter by confidence level.
            limit: Maximum number of records to return.

        Returns:
            List of classification history dictionaries, newest first.
        """
        with self._lock:
            history = list(self._classification_history)

        if document_id:
            history = [
                h for h in history if h["document_id"] == document_id
            ]
        if document_type:
            dt_lower = document_type.lower().strip()
            history = [
                h for h in history if h["predicted_type"] == dt_lower
            ]
        if confidence_level:
            cl_lower = confidence_level.lower().strip()
            history = [
                h for h in history
                if h["confidence_level"] == cl_lower
            ]

        # Return newest first, limited
        history = history[-limit:] if len(history) > limit else history
        history.reverse()
        return history

    # ------------------------------------------------------------------
    # Public API: Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary with classification counts, template counts,
            confidence distribution, and type distribution.
        """
        with self._lock:
            total_classifications = len(self._classification_history)
            total_templates = len(self._templates)
            active_templates = sum(
                1 for t in self._templates.values() if t.active
            )

            # -- Confidence distribution -----------------------------------
            confidence_dist: Dict[str, int] = {
                "high": 0, "medium": 0, "low": 0, "unknown": 0,
            }
            type_dist: Dict[str, int] = {}

            for entry in self._classification_history:
                cl = entry.get("confidence_level", "unknown")
                confidence_dist[cl] = confidence_dist.get(cl, 0) + 1
                dt = entry.get("predicted_type", "unknown")
                type_dist[dt] = type_dist.get(dt, 0) + 1

        return {
            "total_classifications": total_classifications,
            "total_templates": total_templates,
            "active_templates": active_templates,
            "retired_templates": total_templates - active_templates,
            "confidence_distribution": confidence_dist,
            "type_distribution": type_dist,
            "module_version": _MODULE_VERSION,
        }

    # ------------------------------------------------------------------
    # Internal: Input validation
    # ------------------------------------------------------------------

    def _validate_classify_inputs(
        self,
        document_bytes: bytes,
        filename: str,
    ) -> None:
        """Validate classification inputs.

        Args:
            document_bytes: Raw document content.
            filename: Original filename.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not document_bytes:
            raise ValueError("document_bytes must not be empty")
        if not filename or not filename.strip():
            raise ValueError("filename must not be empty")

    # ------------------------------------------------------------------
    # Internal: Text extraction
    # ------------------------------------------------------------------

    def _extract_text_content(
        self,
        document_bytes: bytes,
        max_bytes: int = 8192,
    ) -> str:
        """Extract text content from the first max_bytes of the document.

        Attempts UTF-8 decoding with fallback to latin-1 for
        non-UTF-8 documents. Binary content is silently replaced
        with empty string regions.

        Args:
            document_bytes: Raw document content.
            max_bytes: Maximum bytes to decode (default 8KB).

        Returns:
            Decoded text string.
        """
        chunk = document_bytes[:max_bytes]
        try:
            return chunk.decode("utf-8", errors="replace")
        except Exception:
            try:
                return chunk.decode("latin-1", errors="replace")
            except Exception:
                return ""

    # ------------------------------------------------------------------
    # Internal: Language detection
    # ------------------------------------------------------------------

    def _detect_language(self, text_lower: str) -> str:
        """Detect document language from keyword frequency analysis.

        Counts matching keywords per language and returns the language
        with the highest hit count. Falls back to "en" if no keywords
        match or there is a tie.

        Args:
            text_lower: Lowercased document text content.

        Returns:
            ISO 639-1 language code string.
        """
        if not text_lower:
            return "en"

        scores: Dict[str, int] = {}
        for lang_code, keywords in _LANGUAGE_KEYWORDS.items():
            count = 0
            for kw in keywords:
                if kw in text_lower:
                    count += 1
            if count > 0:
                scores[lang_code] = count

        if not scores:
            return "en"

        # Return language with highest score; ties broken by English first
        best_lang = "en"
        best_score = scores.get("en", 0)
        for lang_code, score in scores.items():
            if score > best_score:
                best_score = score
                best_lang = lang_code

        return best_lang

    # ------------------------------------------------------------------
    # Internal: Filename scoring
    # ------------------------------------------------------------------

    def _score_by_filename(
        self, filename: str,
    ) -> Dict[str, float]:
        """Score document types based on filename heuristics.

        Checks filename components against known naming conventions
        for each document type.

        Args:
            filename: Original filename.

        Returns:
            Dictionary of document_type -> score (0.0-1.0).
        """
        scores: Dict[str, float] = {}
        fn_lower = filename.lower().strip()

        # -- Extension check -----------------------------------------------
        # PDFs get a small universal boost since most EUDR docs are PDF
        # No type-specific scoring from extension alone

        # -- Filename pattern matching -------------------------------------
        filename_type_hints: Dict[str, List[str]] = {
            "coo": ["coo", "certificate_of_origin", "origin_cert"],
            "pc": ["phyto", "phytosanitary", "plant_health"],
            "bol": ["bol", "bill_of_lading", "bl_", "b_l_"],
            "cde": ["cde", "export_decl", "customs_export"],
            "cdi": ["cdi", "import_decl", "customs_import"],
            "rspo_cert": ["rspo"],
            "fsc_cert": ["fsc"],
            "iscc_cert": ["iscc"],
            "ft_cert": ["fairtrade", "fair_trade", "flo"],
            "utz_cert": ["utz", "rainforest_alliance", "ra_cert"],
            "ltr": ["land_title", "title_deed"],
            "ltd": ["land_tenure", "concession"],
            "fmp": ["forest_management", "fmp"],
            "fc": ["felling_cert", "felling_lic", "timber_harvest"],
            "wqc": ["wood_quality", "timber_quality"],
            "dds_draft": ["dds", "due_diligence"],
            "ssd": ["self_decl", "supplier_decl", "ssd"],
            "ic": ["invoice", "commercial_invoice", "inv_"],
            "tc": ["transit_cert", "transit_doc"],
            "wr": ["weighbridge", "weight_receipt", "weight_cert"],
        }

        for doc_type, hints in filename_type_hints.items():
            for hint in hints:
                if hint in fn_lower:
                    scores[doc_type] = max(
                        scores.get(doc_type, 0.0), 0.7,
                    )
                    break

        return scores

    # ------------------------------------------------------------------
    # Internal: MIME type scoring
    # ------------------------------------------------------------------

    def _score_by_mime_type(
        self, mime_type: str,
    ) -> Dict[str, float]:
        """Score document types based on MIME type.

        MIME type provides a small signal: PDFs are expected for most
        formal documents while images may indicate scanned copies.

        Args:
            mime_type: MIME type string.

        Returns:
            Dictionary of document_type -> score (0.0-1.0).
        """
        scores: Dict[str, float] = {}
        mime_lower = mime_type.lower().strip()

        if mime_lower in _PDF_MIME_TYPES:
            # Most EUDR documents are PDF; small universal boost
            for doc_type in _DOCUMENT_TYPE_PATTERNS:
                scores[doc_type] = 0.1
        elif mime_lower in _IMAGE_MIME_TYPES:
            # Scanned documents are often images
            for doc_type in _DOCUMENT_TYPE_PATTERNS:
                scores[doc_type] = 0.05
        elif mime_lower in _XML_MIME_TYPES:
            # XML is common for DDS and customs declarations
            scores["dds_draft"] = 0.3
            scores["cde"] = 0.2
            scores["cdi"] = 0.2

        return scores

    # ------------------------------------------------------------------
    # Internal: Header pattern scoring
    # ------------------------------------------------------------------

    def _score_by_header_patterns(
        self, text_lower: str,
    ) -> Dict[str, float]:
        """Score document types by matching header patterns.

        Header patterns are the strongest single classification signal
        because official documents have recognizable header formats.

        Args:
            text_lower: Lowercased document text content.

        Returns:
            Dictionary of document_type -> score (0.0-1.0).
        """
        scores: Dict[str, float] = {}
        if not text_lower:
            return scores

        for doc_type, patterns in self._builtin_header_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    scores[doc_type] = max(
                        scores.get(doc_type, 0.0), 0.9,
                    )
                    break

        return scores

    # ------------------------------------------------------------------
    # Internal: Serial number scoring
    # ------------------------------------------------------------------

    def _score_by_serial_patterns(
        self, text_content: str,
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Score document types by serial number pattern matching.

        Serial numbers follow authority-specific formats that strongly
        indicate document type.

        Args:
            text_content: Document text content (case-preserved).

        Returns:
            Tuple of (scores dictionary, first matched serial string).
        """
        scores: Dict[str, float] = {}
        matched_serial: Optional[str] = None

        if not text_content:
            return scores, matched_serial

        for doc_type, patterns in self._builtin_serial_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_content)
                if match:
                    scores[doc_type] = max(
                        scores.get(doc_type, 0.0), 0.8,
                    )
                    if matched_serial is None:
                        matched_serial = match.group(0)
                    break

        return scores, matched_serial

    # ------------------------------------------------------------------
    # Internal: Keyword frequency scoring
    # ------------------------------------------------------------------

    def _score_by_keywords(
        self, text_lower: str,
    ) -> Dict[str, float]:
        """Score document types by keyword frequency analysis.

        Counts the number of matching keywords per document type and
        normalizes by the total keyword count for that type.

        Args:
            text_lower: Lowercased document text content.

        Returns:
            Dictionary of document_type -> score (0.0-1.0).
        """
        scores: Dict[str, float] = {}
        if not text_lower:
            return scores

        for doc_type, keywords in self._builtin_keywords.items():
            if not keywords:
                continue
            hit_count = sum(1 for kw in keywords if kw in text_lower)
            if hit_count > 0:
                # Normalize: score = hits / total_keywords
                # Cap at 1.0 to avoid over-weighting
                raw_score = min(hit_count / len(keywords), 1.0)
                # Apply type weight
                type_weight = _DOCUMENT_TYPE_PATTERNS.get(
                    doc_type, {},
                ).get("weight", 1.0)
                scores[doc_type] = raw_score * type_weight

        return scores

    # ------------------------------------------------------------------
    # Internal: Template matching scoring
    # ------------------------------------------------------------------

    def _score_by_templates(
        self,
        text_lower: str,
        text_content: str,
    ) -> Dict[str, float]:
        """Score document types by matching against registered templates.

        Evaluates each active template's header patterns, serial
        patterns, and required/optional keywords against the document
        text content.

        Args:
            text_lower: Lowercased document text content.
            text_content: Case-preserved document text content.

        Returns:
            Dictionary of document_type -> score (0.0-1.0).
        """
        scores: Dict[str, float] = {}

        with self._lock:
            active_templates = [
                t for t in self._templates.values() if t.active
            ]

        if not active_templates:
            return scores

        for template in active_templates:
            match_result = self._match_single_template(
                template, text_lower, text_content,
            )
            if match_result.score > 0.0:
                doc_type = match_result.document_type
                scores[doc_type] = max(
                    scores.get(doc_type, 0.0),
                    match_result.score,
                )

        return scores

    def _match_single_template(
        self,
        template: DocumentTemplate,
        text_lower: str,
        text_content: str,
    ) -> _TemplateMatchResult:
        """Match a single template against document content.

        Args:
            template: The DocumentTemplate to match.
            text_lower: Lowercased document text.
            text_content: Case-preserved document text.

        Returns:
            _TemplateMatchResult with match details.
        """
        result = _TemplateMatchResult()
        result.template_id = template.template_id
        result.document_type = template.document_type

        total_signals = 0
        matched_signals = 0

        # -- Header pattern matching -----------------------------------
        if template.header_patterns:
            total_signals += 1
            for pattern in template.header_patterns:
                if pattern.search(text_lower):
                    result.header_matched = True
                    matched_signals += 1
                    break

        # -- Serial number matching ------------------------------------
        if template.serial_patterns:
            total_signals += 1
            for pattern in template.serial_patterns:
                match = pattern.search(text_content)
                if match:
                    result.serial_matched = True
                    result.matched_serial = match.group(0)
                    matched_signals += 1
                    break

        # -- Required keywords -----------------------------------------
        if template.required_keywords:
            total_signals += 1
            hits = sum(
                1 for kw in template.required_keywords
                if kw in text_lower
            )
            result.required_keyword_hits = hits
            if hits == len(template.required_keywords):
                matched_signals += 1

        # -- Optional keywords -----------------------------------------
        if template.optional_keywords:
            hits = sum(
                1 for kw in template.optional_keywords
                if kw in text_lower
            )
            result.optional_keyword_hits = hits

        # -- Compute score ---------------------------------------------
        if total_signals > 0:
            base_score = matched_signals / total_signals
            # Bonus for optional keywords
            if template.optional_keywords:
                opt_ratio = (
                    result.optional_keyword_hits
                    / len(template.optional_keywords)
                )
                base_score = min(base_score + (opt_ratio * 0.1), 1.0)
            result.score = base_score

        return result

    # ------------------------------------------------------------------
    # Internal: Score aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_scores(
        target: Dict[str, float],
        source: Dict[str, float],
        weight: float = 1.0,
    ) -> None:
        """Merge source scores into target with weighting.

        Args:
            target: Target score dictionary (modified in-place).
            source: Source scores to merge.
            weight: Weight multiplier for source scores.
        """
        for doc_type, score in source.items():
            weighted = score * weight
            target[doc_type] = target.get(doc_type, 0.0) + weighted

    # ------------------------------------------------------------------
    # Internal: Best type selection
    # ------------------------------------------------------------------

    def _select_best_type(
        self,
        type_scores: Dict[str, float],
    ) -> Tuple[str, float, Dict[str, float]]:
        """Select the best document type from aggregated scores.

        Returns the highest-scoring type, its normalized confidence,
        and a dictionary of alternative types with their scores.

        Args:
            type_scores: Aggregated scores per document type.

        Returns:
            Tuple of (predicted_type, confidence_score, alternatives).
        """
        if not type_scores:
            return "coo", 0.0, {}

        # Sort by score descending
        sorted_types = sorted(
            type_scores.items(), key=lambda x: x[1], reverse=True,
        )

        best_type, best_score = sorted_types[0]

        # Normalize confidence to 0.0-1.0 range
        max_possible = 1.0  # Maximum theoretical weighted score
        confidence = min(best_score / max_possible, 1.0)

        # Build alternatives (top 5, excluding best)
        alternatives: Dict[str, float] = {}
        for doc_type, score in sorted_types[1:6]:
            if score > 0.0:
                alt_confidence = min(score / max_possible, 1.0)
                alternatives[doc_type] = round(alt_confidence, 4)

        return best_type, round(confidence, 4), alternatives

    # ------------------------------------------------------------------
    # Internal: Confidence level computation
    # ------------------------------------------------------------------

    def _compute_confidence_level(
        self, confidence_score: float,
    ) -> str:
        """Compute confidence level from confidence score.

        Args:
            confidence_score: Numeric confidence (0.0-1.0).

        Returns:
            Confidence level string: high, medium, low, or unknown.
        """
        if confidence_score >= self._config.min_confidence_high:
            return "high"
        elif confidence_score >= self._config.min_confidence_medium:
            return "medium"
        elif confidence_score > 0.0:
            return "low"
        else:
            return "unknown"

    # ------------------------------------------------------------------
    # Internal: Classification history
    # ------------------------------------------------------------------

    def _record_classification_history(
        self,
        document_id: str,
        predicted_type: str,
        confidence_score: float,
        confidence_level: str,
        language: str,
        processing_time_ms: float,
    ) -> None:
        """Record a classification result in the history log.

        Args:
            document_id: Classified document ID.
            predicted_type: Predicted document type.
            confidence_score: Classification confidence score.
            confidence_level: Confidence level category.
            language: Detected language code.
            processing_time_ms: Processing time in milliseconds.
        """
        entry = {
            "document_id": document_id,
            "predicted_type": predicted_type,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "language": language,
            "processing_time_ms": round(processing_time_ms, 2),
            "classified_at": _utcnow().isoformat(),
        }
        with self._lock:
            self._classification_history.append(entry)

    # ------------------------------------------------------------------
    # Internal: Header extraction (utility)
    # ------------------------------------------------------------------

    def _extract_header_patterns(
        self, text_content: str, max_lines: int = 20,
    ) -> List[str]:
        """Extract header lines from the beginning of a document.

        Args:
            text_content: Full document text content.
            max_lines: Maximum number of header lines to extract.

        Returns:
            List of header line strings.
        """
        lines = text_content.split("\n")[:max_lines]
        return [line.strip() for line in lines if line.strip()]

    def _extract_serial_number(
        self, text_content: str,
    ) -> Optional[str]:
        """Extract the first serial number found in the document.

        Attempts all built-in serial number patterns across all
        document types and returns the first match.

        Args:
            text_content: Document text content.

        Returns:
            Matched serial number string, or None.
        """
        for patterns in self._builtin_serial_patterns.values():
            for pattern in patterns:
                match = pattern.search(text_content)
                if match:
                    return match.group(0)
        return None

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            template_count = len(self._templates)
            history_count = len(self._classification_history)
        return (
            f"DocumentClassifierEngine("
            f"builtin_types={len(_DOCUMENT_TYPE_PATTERNS)}, "
            f"templates={template_count}, "
            f"classifications={history_count})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DocumentClassifierEngine",
    "DocumentTemplate",
]
