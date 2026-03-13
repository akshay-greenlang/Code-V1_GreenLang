# -*- coding: utf-8 -*-
"""
Document Templates Reference Data - AGENT-EUDR-012 Document Authentication

Known document template specifications per document type per country of
issuance.  These templates enable the DocumentClassifierEngine to identify
document types by comparing structural patterns (header keywords, field
layouts, serial number formats, issuing authority names) against a curated
library of ground-truth templates.

Each template entry includes:
    - key_indicators: Header patterns, keyword lists, and field patterns
      that distinguish this document type in a particular country.
    - serial_number_pattern: Regex for the expected serial number format.
    - required_fields: Mandatory fields expected in a conforming document.
    - issuing_authority_patterns: Known authority names that issue this
      document type in the country.
    - language: Expected primary document language (ISO 639-1).

Coverage (20+ templates across all seven EUDR commodity corridors):
    - COO: EU countries (DE, FR, NL, BE, IT, ES, PT), producer countries
      (BR, ID, MY, GH, CI)
    - Phytosanitary: IPPC member countries
    - BOL: Standard shipping line formats
    - RSPO: Standard RSPO certificate format
    - FSC: ASI-accredited certifier formats
    - ISCC: ISCC certificate format
    - Fairtrade: FLOCERT certificate format
    - UTZ/RA: Rainforest Alliance format
    - LTR: Laboratory test report formats
    - DDS: EU DDS draft format
    - SSD: Supplier self-declaration format

Lookup helpers:
    get_template(document_type, country_code) -> dict | None
    get_templates_for_type(document_type) -> dict[str, dict]
    get_all_templates() -> list[dict]

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012) - Appendix A
Agent ID: GL-EUDR-DAV-012
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document templates: Dict[document_type, Dict[country_code, template_spec]]
# ---------------------------------------------------------------------------

DOCUMENT_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ==================================================================
    # COO - Certificate of Origin
    # ==================================================================
    "coo": {
        "DE": {
            "key_indicators": {
                "header_patterns": [
                    "Ursprungszeugnis",
                    "Certificate of Origin",
                    "EUR.1",
                ],
                "keyword_list": [
                    "Handelskammer", "IHK", "Warenursprung",
                    "Ursprungsland", "Ausstellungsdatum",
                ],
                "field_patterns": [
                    r"IHK\s+\w+",
                    r"Nr\.\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^DE-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Industrie- und Handelskammer",
                "IHK Berlin", "IHK Hamburg", "IHK Frankfurt",
                "IHK M.*nchen", "IHK K.*ln",
            ],
            "language": "de",
        },
        "FR": {
            "key_indicators": {
                "header_patterns": [
                    "Certificat d'Origine",
                    "Certificate of Origin",
                    "EUR.1",
                ],
                "keyword_list": [
                    "Chambre de Commerce", "CCI", "Origine",
                    "Exportateur", "Destinataire",
                ],
                "field_patterns": [
                    r"CCI\s+\w+",
                    r"N[o\u00ba]\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^FR-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Chambre de Commerce et d'Industrie",
                "CCI Paris", "CCI Marseille", "CCI Lyon",
            ],
            "language": "fr",
        },
        "NL": {
            "key_indicators": {
                "header_patterns": [
                    "Certificaat van Oorsprong",
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "Kamer van Koophandel", "KvK", "Oorsprong",
                    "Exporteur", "Geadresseerde",
                ],
                "field_patterns": [
                    r"KvK\s+\d+",
                    r"Nr\.\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^NL-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Kamer van Koophandel",
                "KvK Amsterdam", "KvK Rotterdam",
            ],
            "language": "nl",
        },
        "BE": {
            "key_indicators": {
                "header_patterns": [
                    "Certificat d'Origine",
                    "Certificaat van Oorsprong",
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "Chambre de Commerce", "Kamer van Koophandel",
                    "Origine", "Oorsprong",
                ],
                "field_patterns": [
                    r"BCE\s*\d+",
                    r"N[o\u00ba]\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^BE-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Chambre de Commerce de Bruxelles",
                "Kamer van Koophandel Antwerpen",
            ],
            "language": "fr",
        },
        "IT": {
            "key_indicators": {
                "header_patterns": [
                    "Certificato di Origine",
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "Camera di Commercio", "CCIAA", "Origine",
                    "Esportatore", "Destinatario",
                ],
                "field_patterns": [
                    r"CCIAA\s+\w+",
                    r"N\.\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^IT-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Camera di Commercio",
                "CCIAA Milano", "CCIAA Roma", "CCIAA Genova",
            ],
            "language": "it",
        },
        "ES": {
            "key_indicators": {
                "header_patterns": [
                    "Certificado de Origen",
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "C\u00e1mara de Comercio", "Origen",
                    "Exportador", "Destinatario",
                ],
                "field_patterns": [
                    r"C\u00e1mara\s+de\s+Comercio",
                    r"N[o\u00ba]\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^ES-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "C\u00e1mara de Comercio de Barcelona",
                "C\u00e1mara de Comercio de Madrid",
            ],
            "language": "es",
        },
        "PT": {
            "key_indicators": {
                "header_patterns": [
                    "Certificado de Origem",
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "C\u00e2mara de Com\u00e9rcio", "Origem",
                    "Exportador", "Destinat\u00e1rio",
                ],
                "field_patterns": [
                    r"CCIP\s+\w*",
                    r"N[o\u00ba]\s*\d{4,}",
                ],
            },
            "serial_number_pattern": r"^PT-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "C\u00e2mara de Com\u00e9rcio e Ind\u00fastria Portuguesa",
                "CCIP",
            ],
            "language": "pt",
        },
        "BR": {
            "key_indicators": {
                "header_patterns": [
                    "Certificado de Origem",
                    "Certificate of Origin",
                    "Certificado de Origem Digital",
                ],
                "keyword_list": [
                    "Federa\u00e7\u00e3o das Ind\u00fastrias",
                    "FIESP", "FIRJAN", "Origem", "Exportador",
                ],
                "field_patterns": [
                    r"CNPJ\s*\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
                    r"CO-\d{4,}",
                ],
            },
            "serial_number_pattern": r"^BR-COO-\d{4}-\d{8,12}$",
            "required_fields": [
                "exporter_name", "exporter_cnpj", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "net_weight", "gross_weight",
                "issuing_authority", "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "FIESP", "FIRJAN", "FIESC", "FIEMG",
                "Federa\u00e7\u00e3o das Ind\u00fastrias",
            ],
            "language": "pt",
        },
        "ID": {
            "key_indicators": {
                "header_patterns": [
                    "Certificate of Origin",
                    "Surat Keterangan Asal",
                    "Form E",
                ],
                "keyword_list": [
                    "Kementerian Perdagangan", "INATRADE",
                    "Asal Barang", "Eksportir",
                ],
                "field_patterns": [
                    r"NPWP\s*\d{2}\.\d{3}\.\d{3}",
                    r"SKA-\d+",
                ],
            },
            "serial_number_pattern": r"^ID-COO-\d{4}-\d{6,10}$",
            "required_fields": [
                "exporter_name", "exporter_npwp", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number", "hs_code",
            ],
            "issuing_authority_patterns": [
                "Kementerian Perdagangan",
                "Dinas Perindustrian dan Perdagangan",
                "INATRADE",
            ],
            "language": "id",
        },
        "MY": {
            "key_indicators": {
                "header_patterns": [
                    "Certificate of Origin",
                    "Sijil Asal",
                ],
                "keyword_list": [
                    "MATRADE", "Kementerian Perdagangan Antarabangsa",
                    "Asal", "Pengeksport",
                ],
                "field_patterns": [
                    r"MATRADE\s*\d*",
                    r"MY-\d+",
                ],
            },
            "serial_number_pattern": r"^MY-COO-\d{4}-\d{6,10}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "MATRADE",
                "Malaysia External Trade Development Corporation",
            ],
            "language": "ms",
        },
        "GH": {
            "key_indicators": {
                "header_patterns": [
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "Ghana Export Promotion Authority",
                    "GEPA", "Ghana Cocoa Board", "COCOBOD",
                ],
                "field_patterns": [
                    r"GEPA\s*\d*",
                    r"GH-\d+",
                ],
            },
            "serial_number_pattern": r"^GH-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Ghana Export Promotion Authority",
                "GEPA", "Association of Ghana Industries",
            ],
            "language": "en",
        },
        "CI": {
            "key_indicators": {
                "header_patterns": [
                    "Certificat d'Origine",
                    "Certificate of Origin",
                ],
                "keyword_list": [
                    "Chambre de Commerce et d'Industrie",
                    "CCI-CI", "Conseil du Caf\u00e9-Cacao",
                ],
                "field_patterns": [
                    r"CCI-CI\s*\d*",
                    r"CI-\d+",
                ],
            },
            "serial_number_pattern": r"^CI-COO-\d{4}-\d{6,8}$",
            "required_fields": [
                "exporter_name", "exporter_address", "consignee_name",
                "country_of_origin", "description_of_goods",
                "quantity", "gross_weight", "issuing_authority",
                "date_of_issue", "serial_number",
            ],
            "issuing_authority_patterns": [
                "Chambre de Commerce et d'Industrie de C\u00f4te d'Ivoire",
                "CCI-CI",
            ],
            "language": "fr",
        },
    },

    # ==================================================================
    # PC - Phytosanitary Certificate (IPPC standard format)
    # ==================================================================
    "pc": {
        "IPPC_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Phytosanitary Certificate",
                    "Certificat Phytosanitaire",
                    "Certificado Fitosanitario",
                ],
                "keyword_list": [
                    "IPPC", "National Plant Protection Organization",
                    "NPPO", "pest free", "phytosanitary",
                    "treatment", "inspection",
                ],
                "field_patterns": [
                    r"NPPO\s+\w+",
                    r"PC-\d{4,}",
                ],
            },
            "serial_number_pattern": r"^[A-Z]{2}-PC-\d{4}-\d{6,10}$",
            "required_fields": [
                "exporter_name", "exporter_address",
                "consignee_name", "consignee_address",
                "place_of_origin", "declared_means_of_conveyance",
                "point_of_entry", "description_of_consignment",
                "botanical_name", "quantity", "treatment_type",
                "treatment_date", "nppo_name", "date_of_issue",
                "serial_number",
            ],
            "issuing_authority_patterns": [
                "National Plant Protection Organization",
                "NPPO", "Department of Agriculture",
                "Minist\u00e8re de l'Agriculture",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # BOL - Bill of Lading
    # ==================================================================
    "bol": {
        "STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Bill of Lading",
                    "B/L",
                    "Connaissement",
                ],
                "keyword_list": [
                    "Shipper", "Consignee", "Notify Party",
                    "Port of Loading", "Port of Discharge",
                    "Container No", "Seal No",
                ],
                "field_patterns": [
                    r"B/L\s*No\.?\s*\w+",
                    r"Container\s*No\.?\s*[A-Z]{4}\d{7}",
                ],
            },
            "serial_number_pattern": r"^[A-Z]{4}\d{8,12}$",
            "required_fields": [
                "shipper_name", "shipper_address",
                "consignee_name", "consignee_address",
                "notify_party", "port_of_loading",
                "port_of_discharge", "vessel_name",
                "voyage_number", "container_numbers",
                "description_of_goods", "gross_weight",
                "number_of_packages", "bl_number", "date_of_issue",
            ],
            "issuing_authority_patterns": [
                "Maersk", "MSC", "CMA CGM", "COSCO",
                "Hapag-Lloyd", "ONE", "Evergreen",
                "Yang Ming", "ZIM", "HMM",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # RSPO_CERT - RSPO Sustainability Certificate
    # ==================================================================
    "rspo_cert": {
        "RSPO_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "RSPO Supply Chain Certificate",
                    "RSPO Certificate",
                    "Roundtable on Sustainable Palm Oil",
                ],
                "keyword_list": [
                    "RSPO", "Roundtable on Sustainable Palm Oil",
                    "Supply Chain Certification", "SCCS",
                    "Identity Preserved", "Segregated",
                    "Mass Balance", "Book & Claim",
                ],
                "field_patterns": [
                    r"RSPO-\d{4,}-\d{2,}",
                    r"SCCS-\d+",
                ],
            },
            "serial_number_pattern": r"^RSPO-\d{4}-\d{4,8}$",
            "required_fields": [
                "certificate_holder", "holder_address",
                "certification_body", "scope",
                "supply_chain_model", "valid_from",
                "valid_until", "certificate_number",
                "accreditation_number",
            ],
            "issuing_authority_patterns": [
                "Control Union", "BSI Group", "SGS",
                "TUV Rheinland", "Bureau Veritas",
                "Intertek", "SCS Global Services",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # FSC_CERT - FSC Chain of Custody Certificate
    # ==================================================================
    "fsc_cert": {
        "FSC_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "FSC Chain of Custody Certificate",
                    "FSC Certificate",
                    "Forest Stewardship Council",
                ],
                "keyword_list": [
                    "FSC", "Forest Stewardship Council",
                    "Chain of Custody", "CoC",
                    "FSC-C", "FSC-STD-40-004",
                ],
                "field_patterns": [
                    r"FSC-C\d{5,6}",
                    r"[A-Z]{2,3}-COC-\d{6}",
                ],
            },
            "serial_number_pattern": r"^[A-Z]{2,3}-COC-\d{6}$",
            "required_fields": [
                "certificate_holder", "holder_address",
                "certification_body", "license_code",
                "certificate_code", "scope",
                "valid_from", "valid_until",
                "standard_version",
            ],
            "issuing_authority_patterns": [
                "SGS", "Control Union", "Bureau Veritas",
                "Rainforest Alliance", "Soil Association",
                "SCS Global", "NEPCon",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # ISCC_CERT - ISCC Sustainability Certificate
    # ==================================================================
    "iscc_cert": {
        "ISCC_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "ISCC Certificate",
                    "ISCC EU Certificate",
                    "ISCC PLUS Certificate",
                    "International Sustainability and Carbon Certification",
                ],
                "keyword_list": [
                    "ISCC", "International Sustainability",
                    "Carbon Certification", "ISCC EU",
                    "ISCC PLUS", "Sustainability Certificate",
                ],
                "field_patterns": [
                    r"ISCC-\w+-\d{4,}",
                    r"EU-ISCC-Cert-\w+",
                ],
            },
            "serial_number_pattern": r"^ISCC-\w{2,5}-\d{4}-\d{4,8}$",
            "required_fields": [
                "certificate_holder", "holder_address",
                "certification_body", "scope",
                "iscc_system", "valid_from",
                "valid_until", "certificate_number",
            ],
            "issuing_authority_patterns": [
                "Control Union", "SGS", "TUV SUD",
                "Bureau Veritas", "Peterson Control Union",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # FT_CERT - Fairtrade Certificate
    # ==================================================================
    "ft_cert": {
        "FLOCERT_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Fairtrade Certificate",
                    "FLOCERT Certificate",
                    "Fairtrade International",
                ],
                "keyword_list": [
                    "Fairtrade", "FLOCERT", "FLO-ID",
                    "Fair Trade", "Fairtrade International",
                    "Fairtrade Standards",
                ],
                "field_patterns": [
                    r"FLO-ID\s*\d{4,6}",
                    r"FLOCERT-\d+",
                ],
            },
            "serial_number_pattern": r"^FLO-\d{4,6}$",
            "required_fields": [
                "certificate_holder", "holder_address",
                "flo_id", "product_scope",
                "valid_from", "valid_until",
                "certification_body",
            ],
            "issuing_authority_patterns": [
                "FLOCERT GmbH",
                "FLOCERT",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # UTZ_CERT - UTZ/Rainforest Alliance Certificate
    # ==================================================================
    "utz_cert": {
        "RA_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Rainforest Alliance Certificate",
                    "UTZ Certificate",
                    "Rainforest Alliance Certified",
                ],
                "keyword_list": [
                    "Rainforest Alliance", "UTZ",
                    "Sustainable Agriculture Standard",
                    "SAS", "Chain of Custody",
                ],
                "field_patterns": [
                    r"RA-\d{6,8}",
                    r"UTZ-\d{4,}",
                ],
            },
            "serial_number_pattern": r"^RA-\d{6,8}$",
            "required_fields": [
                "certificate_holder", "holder_address",
                "certification_body", "product_scope",
                "certificate_type", "valid_from",
                "valid_until", "certificate_number",
            ],
            "issuing_authority_patterns": [
                "Rainforest Alliance",
                "Control Union", "SGS",
                "Bureau Veritas", "Intertek",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # LTR - Laboratory Test Report
    # ==================================================================
    "ltr": {
        "STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Laboratory Test Report",
                    "Certificate of Analysis",
                    "Test Report",
                    "Rapport d'Analyse",
                ],
                "keyword_list": [
                    "ISO 17025", "accredited laboratory",
                    "test method", "sample", "result",
                    "specification", "pass", "fail",
                ],
                "field_patterns": [
                    r"ISO\s*17025",
                    r"Report\s*No\.?\s*\w+",
                    r"Sample\s*ID\s*\w+",
                ],
            },
            "serial_number_pattern": r"^LTR-\d{4}-\d{6,10}$",
            "required_fields": [
                "laboratory_name", "laboratory_accreditation",
                "client_name", "sample_id",
                "sample_description", "test_methods",
                "results", "date_received",
                "date_tested", "date_issued",
                "report_number",
            ],
            "issuing_authority_patterns": [
                "SGS", "Bureau Veritas", "Intertek",
                "Eurofins", "TUV", "CATAS",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # DDS_DRAFT - EU Due Diligence Statement Draft
    # ==================================================================
    "dds_draft": {
        "EU_STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Due Diligence Statement",
                    "DDS",
                    "EU Deforestation Regulation",
                    "EUDR Due Diligence",
                ],
                "keyword_list": [
                    "EU 2023/1115", "EUDR", "deforestation",
                    "due diligence", "geolocation",
                    "risk assessment", "operator",
                    "commodity", "product",
                ],
                "field_patterns": [
                    r"DDS-\d{4,}",
                    r"EU\s*2023/1115",
                    r"Article\s+\d+",
                ],
            },
            "serial_number_pattern": r"^DDS-EU-\d{4}-\d{8,12}$",
            "required_fields": [
                "operator_name", "operator_address",
                "operator_eori", "commodity_type",
                "product_description", "hs_code",
                "country_of_production",
                "geolocation_coordinates",
                "quantity", "risk_assessment_result",
                "deforestation_free_statement",
                "date_of_statement",
            ],
            "issuing_authority_patterns": [
                "EU Information System",
                "Competent Authority",
            ],
            "language": "en",
        },
    },

    # ==================================================================
    # SSD - Supplier Self-Declaration
    # ==================================================================
    "ssd": {
        "STANDARD": {
            "key_indicators": {
                "header_patterns": [
                    "Supplier Self-Declaration",
                    "Supplier Declaration",
                    "Self-Declaration of Compliance",
                ],
                "keyword_list": [
                    "self-declaration", "supplier",
                    "deforestation-free", "compliance",
                    "EUDR", "EU 2023/1115",
                    "production area", "geolocation",
                ],
                "field_patterns": [
                    r"SSD-\d{4,}",
                    r"Declar\w+\s+No\.?\s*\w+",
                ],
            },
            "serial_number_pattern": r"^SSD-\d{4}-\d{6,10}$",
            "required_fields": [
                "supplier_name", "supplier_address",
                "supplier_registration", "commodity_type",
                "production_country", "production_area",
                "geolocation_coordinates",
                "deforestation_free_declaration",
                "legal_compliance_declaration",
                "date_of_declaration", "signatory_name",
                "signatory_title",
            ],
            "issuing_authority_patterns": [],
            "language": "en",
        },
    },
}

# ---------------------------------------------------------------------------
# Computed totals
# ---------------------------------------------------------------------------

TOTAL_DOCUMENT_TYPES: int = len(DOCUMENT_TEMPLATES)

TOTAL_TEMPLATES: int = sum(
    len(country_templates)
    for country_templates in DOCUMENT_TEMPLATES.values()
)

# ---------------------------------------------------------------------------
# Type-level index: {document_type: [country_code, ...]}
# ---------------------------------------------------------------------------

TEMPLATE_TYPE_INDEX: Dict[str, List[str]] = {
    doc_type: sorted(country_dict.keys())
    for doc_type, country_dict in DOCUMENT_TEMPLATES.items()
}

# ---------------------------------------------------------------------------
# Lookup helper functions
# ---------------------------------------------------------------------------


def get_template(
    document_type: str,
    country_code: str,
) -> Optional[Dict[str, Any]]:
    """Return the template specification for a document type and country.

    Performs case-insensitive lookup on the document_type and
    case-insensitive lookup on the country_code. Falls back to
    'STANDARD' or 'IPPC_STANDARD' keys if the exact country code
    is not found.

    Args:
        document_type: Document type identifier (coo, pc, bol, etc.).
        country_code: ISO 3166-1 alpha-2 country code or a standard
            key like 'STANDARD', 'IPPC_STANDARD'.

    Returns:
        Template specification dictionary, or None if not found.

    Example:
        >>> template = get_template("coo", "DE")
        >>> template["language"]
        'de'
    """
    doc_lower = document_type.lower().strip()
    country_upper = country_code.upper().strip()

    type_templates = DOCUMENT_TEMPLATES.get(doc_lower)
    if type_templates is None:
        return None

    # Exact match
    template = type_templates.get(country_upper)
    if template is not None:
        return template

    # Fallback to standard keys
    for fallback_key in ("STANDARD", "IPPC_STANDARD",
                         "RSPO_STANDARD", "FSC_STANDARD",
                         "ISCC_STANDARD", "FLOCERT_STANDARD",
                         "RA_STANDARD", "EU_STANDARD"):
        template = type_templates.get(fallback_key)
        if template is not None:
            return template

    return None


def get_templates_for_type(
    document_type: str,
) -> Dict[str, Dict[str, Any]]:
    """Return all template specifications for a given document type.

    Args:
        document_type: Document type identifier.

    Returns:
        Dictionary mapping country codes to template specifications.
        Returns empty dictionary if the document type is not found.

    Example:
        >>> templates = get_templates_for_type("coo")
        >>> "DE" in templates
        True
    """
    doc_lower = document_type.lower().strip()
    return dict(DOCUMENT_TEMPLATES.get(doc_lower, {}))


def get_all_templates() -> List[Dict[str, Any]]:
    """Return all template specifications as a flat list.

    Each entry includes the ``document_type`` and ``country_code``
    fields in addition to the template specification fields.

    Returns:
        List of template dictionaries with document_type and
        country_code injected.

    Example:
        >>> templates = get_all_templates()
        >>> len(templates) >= 20
        True
    """
    result: List[Dict[str, Any]] = []
    for doc_type, country_dict in DOCUMENT_TEMPLATES.items():
        for country_code, template in country_dict.items():
            entry = {
                "document_type": doc_type,
                "country_code": country_code,
            }
            entry.update(template)
            result.append(entry)
    return result


def get_supported_document_types() -> List[str]:
    """Return all document types that have template definitions.

    Returns:
        Sorted list of document type identifier strings.

    Example:
        >>> types = get_supported_document_types()
        >>> "coo" in types
        True
    """
    return sorted(DOCUMENT_TEMPLATES.keys())


def get_countries_for_type(document_type: str) -> List[str]:
    """Return all country codes with templates for a document type.

    Args:
        document_type: Document type identifier.

    Returns:
        Sorted list of country code strings.
        Returns empty list if document type is not found.

    Example:
        >>> countries = get_countries_for_type("coo")
        >>> "DE" in countries
        True
    """
    doc_lower = document_type.lower().strip()
    type_templates = DOCUMENT_TEMPLATES.get(doc_lower)
    if type_templates is None:
        return []
    return sorted(type_templates.keys())


# ---------------------------------------------------------------------------
# Module-level logging on import
# ---------------------------------------------------------------------------

logger.debug(
    "Document templates reference data loaded: "
    "%d document types, %d total templates",
    TOTAL_DOCUMENT_TYPES,
    TOTAL_TEMPLATES,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "DOCUMENT_TEMPLATES",
    "TEMPLATE_TYPE_INDEX",
    # Counts
    "TOTAL_DOCUMENT_TYPES",
    "TOTAL_TEMPLATES",
    # Lookup helpers
    "get_template",
    "get_templates_for_type",
    "get_all_templates",
    "get_supported_document_types",
    "get_countries_for_type",
]
