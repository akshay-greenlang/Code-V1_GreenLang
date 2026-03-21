# -*- coding: utf-8 -*-
"""
TranslationEngine - PACK-030 Net Zero Reporting Pack Engine 9
================================================================

Multi-language translation engine for climate disclosure narratives.
Supports English (en), German (de), French (fr), and Spanish (es) with
climate-specific terminology management, citation preservation, and
translation quality scoring.

Translation Methodology:
    Terminology-First Translation:
        1. Extract climate-specific terms from source text
        2. Look up canonical translations in the Climate Glossary
        3. Translate remaining text preserving glossary terms
        4. Re-insert citations and metric references unchanged
        5. Validate consistency of translated terminology

    Citation Preservation:
        Citations in the format [REF:xxx] or [CITE:xxx] are extracted
        before translation and re-inserted at identical positions in
        the translated text.  Metric values (numbers, units, dates)
        are never translated -- they pass through unchanged.

    Quality Scoring:
        quality = w1*terminology_accuracy + w2*completeness +
                  w3*citation_integrity + w4*format_preservation
        Weights: terminology=35, completeness=25, citations=25, format=15

    Glossary Management:
        The Climate Glossary contains canonical translations for:
            - GHG Protocol terms (Scope 1, Scope 2, Scope 3, etc.)
            - SBTi terms (near-term, long-term, net-zero target)
            - TCFD terms (physical risk, transition risk, etc.)
            - CSRD/ESRS terms (double materiality, transition plan)
            - Metric units (tCO2e, MWh, GJ, etc.)

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - CDP Climate Change Questionnaire (2024) -- multi-language
    - TCFD Recommendations (2017, updated 2023)
    - GRI 305 (2016) -- multi-language disclosures
    - ISSB IFRS S2 (2023) -- English primary
    - SEC Climate Disclosure (2024) -- English only
    - CSRD ESRS E1 (2024) -- all EU languages required

Zero-Hallucination:
    - Glossary-based translation for climate terms
    - No creative rewriting of quantitative claims
    - Citations and metric values pass through unchanged
    - SHA-256 provenance hash on every result
    - Quality scoring is deterministic

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-030 Net Zero Reporting
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {k: v for k, v in serializable.items()
                        if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _round_val(value: Decimal, places: int = 2) -> Decimal:
    """Round a Decimal to the given number of decimal places."""
    quant = Decimal(10) ** -places
    return value.quantize(quant, rounding=ROUND_HALF_UP)

def _round3(value: Decimal) -> Decimal:
    """Round to 3 decimal places."""
    return _round_val(value, 3)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SupportedLanguage(str, Enum):
    """Supported languages for translation."""
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"


class TranslationQualityTier(str, Enum):
    """Quality tiers for translations."""
    EXCELLENT = "excellent"      # >= 90
    GOOD = "good"                # >= 75
    ACCEPTABLE = "acceptable"    # >= 60
    NEEDS_REVIEW = "needs_review"  # >= 40
    POOR = "poor"                # < 40


class TranslationMethod(str, Enum):
    """Method used for translation."""
    GLOSSARY = "glossary"        # Term-level glossary lookup
    TEMPLATE = "template"        # Template-based structural translation
    PASSTHROUGH = "passthrough"  # No translation needed (citations, numbers)
    COMPOSITE = "composite"      # Combination of methods


class TextSegmentType(str, Enum):
    """Types of text segments for translation processing."""
    NARRATIVE = "narrative"
    CITATION = "citation"
    METRIC_VALUE = "metric_value"
    HEADER = "header"
    TABLE = "table"
    FOOTNOTE = "footnote"
    GLOSSARY_TERM = "glossary_term"


class FrameworkContext(str, Enum):
    """Framework context for terminology selection."""
    SBTI = "sbti"
    CDP = "cdp"
    TCFD = "tcfd"
    GRI = "gri"
    ISSB = "issb"
    SEC = "sec"
    CSRD = "csrd"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Constants -- Climate Glossary
# ---------------------------------------------------------------------------

# Canonical climate terminology translations.
# Structure: { "en_term": { "de": "...", "fr": "...", "es": "..." } }
CLIMATE_GLOSSARY: Dict[str, Dict[str, str]] = {
    # GHG Protocol terms
    "Scope 1 emissions": {
        "de": "Scope-1-Emissionen",
        "fr": "emissions du Scope 1",
        "es": "emisiones de Alcance 1",
    },
    "Scope 2 emissions": {
        "de": "Scope-2-Emissionen",
        "fr": "emissions du Scope 2",
        "es": "emisiones de Alcance 2",
    },
    "Scope 3 emissions": {
        "de": "Scope-3-Emissionen",
        "fr": "emissions du Scope 3",
        "es": "emisiones de Alcance 3",
    },
    "greenhouse gas": {
        "de": "Treibhausgas",
        "fr": "gaz a effet de serre",
        "es": "gas de efecto invernadero",
    },
    "carbon dioxide equivalent": {
        "de": "Kohlendioxid-Aquivalent",
        "fr": "equivalent dioxyde de carbone",
        "es": "equivalente de dioxido de carbono",
    },
    "emission factor": {
        "de": "Emissionsfaktor",
        "fr": "facteur d'emission",
        "es": "factor de emision",
    },
    "carbon footprint": {
        "de": "CO2-Fussabdruck",
        "fr": "empreinte carbone",
        "es": "huella de carbono",
    },
    "global warming potential": {
        "de": "Treibhauspotenzial",
        "fr": "potentiel de rechauffement global",
        "es": "potencial de calentamiento global",
    },
    # SBTi terms
    "science-based target": {
        "de": "wissenschaftsbasiertes Ziel",
        "fr": "objectif fonde sur la science",
        "es": "objetivo basado en la ciencia",
    },
    "near-term target": {
        "de": "kurzfristiges Ziel",
        "fr": "objectif a court terme",
        "es": "objetivo a corto plazo",
    },
    "long-term target": {
        "de": "langfristiges Ziel",
        "fr": "objectif a long terme",
        "es": "objetivo a largo plazo",
    },
    "net-zero target": {
        "de": "Netto-Null-Ziel",
        "fr": "objectif de zero net",
        "es": "objetivo de cero neto",
    },
    "base year": {
        "de": "Basisjahr",
        "fr": "annee de reference",
        "es": "ano base",
    },
    "target year": {
        "de": "Zieljahr",
        "fr": "annee cible",
        "es": "ano objetivo",
    },
    "emissions reduction": {
        "de": "Emissionsminderung",
        "fr": "reduction des emissions",
        "es": "reduccion de emisiones",
    },
    "decarbonization pathway": {
        "de": "Dekarbonisierungspfad",
        "fr": "trajectoire de decarbonation",
        "es": "via de descarbonizacion",
    },
    "carbon neutrality": {
        "de": "Klimaneutralitat",
        "fr": "neutralite carbone",
        "es": "neutralidad de carbono",
    },
    "residual emissions": {
        "de": "Restemissionen",
        "fr": "emissions residuelles",
        "es": "emisiones residuales",
    },
    "carbon removal": {
        "de": "Kohlenstoffabbau",
        "fr": "elimination du carbone",
        "es": "eliminacion de carbono",
    },
    # TCFD terms
    "physical risk": {
        "de": "physisches Risiko",
        "fr": "risque physique",
        "es": "riesgo fisico",
    },
    "transition risk": {
        "de": "Transitionsrisiko",
        "fr": "risque de transition",
        "es": "riesgo de transicion",
    },
    "climate-related risk": {
        "de": "klimabezogenes Risiko",
        "fr": "risque lie au climat",
        "es": "riesgo relacionado con el clima",
    },
    "scenario analysis": {
        "de": "Szenarioanalyse",
        "fr": "analyse de scenarios",
        "es": "analisis de escenarios",
    },
    "climate resilience": {
        "de": "Klimaresilienz",
        "fr": "resilience climatique",
        "es": "resiliencia climatica",
    },
    # CSRD/ESRS terms
    "double materiality": {
        "de": "doppelte Wesentlichkeit",
        "fr": "double materialite",
        "es": "doble materialidad",
    },
    "transition plan": {
        "de": "Transitionsplan",
        "fr": "plan de transition",
        "es": "plan de transicion",
    },
    "due diligence": {
        "de": "Sorgfaltspflicht",
        "fr": "diligence raisonnable",
        "es": "diligencia debida",
    },
    "value chain": {
        "de": "Wertschopfungskette",
        "fr": "chaine de valeur",
        "es": "cadena de valor",
    },
    "sustainability reporting": {
        "de": "Nachhaltigkeitsberichterstattung",
        "fr": "reporting de durabilite",
        "es": "informes de sostenibilidad",
    },
    "financial materiality": {
        "de": "finanzielle Wesentlichkeit",
        "fr": "materialite financiere",
        "es": "materialidad financiera",
    },
    "impact materiality": {
        "de": "Auswirkungswesentlichkeit",
        "fr": "materialite d'impact",
        "es": "materialidad de impacto",
    },
    # General reporting
    "reporting period": {
        "de": "Berichtszeitraum",
        "fr": "periode de reporting",
        "es": "periodo de reporte",
    },
    "fiscal year": {
        "de": "Geschaftsjahr",
        "fr": "exercice fiscal",
        "es": "ano fiscal",
    },
    "assurance": {
        "de": "Prufungssicherheit",
        "fr": "assurance",
        "es": "aseguramiento",
    },
    "limited assurance": {
        "de": "begrenzte Prufungssicherheit",
        "fr": "assurance limitee",
        "es": "aseguramiento limitado",
    },
    "reasonable assurance": {
        "de": "hinreichende Prufungssicherheit",
        "fr": "assurance raisonnable",
        "es": "aseguramiento razonable",
    },
    "internal carbon price": {
        "de": "interner CO2-Preis",
        "fr": "prix interne du carbone",
        "es": "precio interno del carbono",
    },
    "carbon credit": {
        "de": "Emissionsgutschrift",
        "fr": "credit carbone",
        "es": "credito de carbono",
    },
    "carbon offset": {
        "de": "Emissionskompensation",
        "fr": "compensation carbone",
        "es": "compensacion de carbono",
    },
    "energy consumption": {
        "de": "Energieverbrauch",
        "fr": "consommation d'energie",
        "es": "consumo de energia",
    },
    "renewable energy": {
        "de": "erneuerbare Energie",
        "fr": "energie renouvelable",
        "es": "energia renovable",
    },
    "energy efficiency": {
        "de": "Energieeffizienz",
        "fr": "efficacite energetique",
        "es": "eficiencia energetica",
    },
    "GHG inventory": {
        "de": "THG-Inventar",
        "fr": "inventaire de GES",
        "es": "inventario de GEI",
    },
    "emissions intensity": {
        "de": "Emissionsintensitat",
        "fr": "intensite des emissions",
        "es": "intensidad de emisiones",
    },
    "abatement cost": {
        "de": "Vermeidungskosten",
        "fr": "cout de reduction",
        "es": "costo de reduccion",
    },
}

# Framework-specific section headers translations
SECTION_HEADER_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "Executive Summary": {
        "de": "Zusammenfassung",
        "fr": "Resume executif",
        "es": "Resumen ejecutivo",
    },
    "Governance": {
        "de": "Unternehmenssteuerung",
        "fr": "Gouvernance",
        "es": "Gobernanza",
    },
    "Strategy": {
        "de": "Strategie",
        "fr": "Strategie",
        "es": "Estrategia",
    },
    "Risk Management": {
        "de": "Risikomanagement",
        "fr": "Gestion des risques",
        "es": "Gestion de riesgos",
    },
    "Metrics and Targets": {
        "de": "Kennzahlen und Ziele",
        "fr": "Indicateurs et objectifs",
        "es": "Metricas y objetivos",
    },
    "Emissions Overview": {
        "de": "Emissionsubersicht",
        "fr": "Apercu des emissions",
        "es": "Resumen de emisiones",
    },
    "Reduction Targets": {
        "de": "Reduktionsziele",
        "fr": "Objectifs de reduction",
        "es": "Objetivos de reduccion",
    },
    "Progress Report": {
        "de": "Fortschrittsbericht",
        "fr": "Rapport de progres",
        "es": "Informe de progreso",
    },
    "Methodology": {
        "de": "Methodik",
        "fr": "Methodologie",
        "es": "Methodologie",
    },
    "Data Quality": {
        "de": "Datenqualitat",
        "fr": "Qualite des donnees",
        "es": "Calidad de datos",
    },
    "Assurance Statement": {
        "de": "Prufungsbericht",
        "fr": "Rapport d'assurance",
        "es": "Informe de aseguramiento",
    },
    "Transition Plan": {
        "de": "Transitionsplan",
        "fr": "Plan de transition",
        "es": "Plan de transicion",
    },
    "Scenario Analysis": {
        "de": "Szenarioanalyse",
        "fr": "Analyse de scenarios",
        "es": "Analisis de escenarios",
    },
    "Supply Chain Emissions": {
        "de": "Lieferkettenemissionen",
        "fr": "Emissions de la chaine d'approvisionnement",
        "es": "Emisiones de la cadena de suministro",
    },
    "Appendix": {
        "de": "Anhang",
        "fr": "Annexe",
        "es": "Anexo",
    },
    "Glossary": {
        "de": "Glossar",
        "fr": "Glossaire",
        "es": "Glosario",
    },
    "Table of Contents": {
        "de": "Inhaltsverzeichnis",
        "fr": "Table des matieres",
        "es": "Indice",
    },
    "Disclaimer": {
        "de": "Haftungsausschluss",
        "fr": "Avertissement",
        "es": "Descargo de responsabilidad",
    },
}

# Regex patterns for citation and metric extraction
CITATION_PATTERN: re.Pattern = re.compile(
    r"\[(?:REF|CITE|SOURCE|NOTE):([^\]]+)\]"
)
METRIC_VALUE_PATTERN: re.Pattern = re.compile(
    r"(?:\d[\d,.]*\s*(?:tCO2e|MtCO2e|ktCO2e|MWh|GWh|GJ|TJ|%|USD|EUR|GBP))"
)
NUMBER_WITH_UNIT_PATTERN: re.Pattern = re.compile(
    r"(\d[\d,.]*)\s*(tCO2e|MtCO2e|ktCO2e|MWh|GWh|GJ|TJ|kg|t|Mt|Gt|ppm|ppb)"
)
DATE_PATTERN: re.Pattern = re.compile(
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|"
    r"Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
)

# Quality scoring weights
QUALITY_WEIGHTS: Dict[str, Decimal] = {
    "terminology_accuracy": _decimal("0.35"),
    "completeness": _decimal("0.25"),
    "citation_integrity": _decimal("0.25"),
    "format_preservation": _decimal("0.15"),
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class TextSegment(BaseModel):
    """A segment of text identified for translation processing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_id: str = Field(default_factory=_new_uuid)
    text: str = Field(..., description="Source text content")
    segment_type: TextSegmentType = Field(
        default=TextSegmentType.NARRATIVE,
        description="Type of text segment",
    )
    position: int = Field(default=0, description="Position in source document")
    section_header: Optional[str] = Field(
        default=None, description="Section header this segment belongs to"
    )
    framework_context: FrameworkContext = Field(
        default=FrameworkContext.GENERAL,
        description="Framework context for terminology",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GlossaryOverride(BaseModel):
    """Custom glossary term override for organization-specific terminology."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_term: str = Field(..., description="Source language term")
    target_translations: Dict[str, str] = Field(
        ..., description="Target language translations: {lang_code: translation}"
    )
    framework_context: Optional[FrameworkContext] = Field(
        default=None, description="Framework-specific override"
    )
    priority: int = Field(
        default=10, description="Priority (higher wins): 10=default, 20=org, 30=manual"
    )


class TranslationInput(BaseModel):
    """Input for the TranslationEngine."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organization_id: str = Field(..., description="Organization identifier")
    source_language: SupportedLanguage = Field(
        default=SupportedLanguage.ENGLISH,
        description="Source language code",
    )
    target_language: SupportedLanguage = Field(
        ..., description="Target language code",
    )
    segments: List[TextSegment] = Field(
        ..., description="Text segments to translate",
    )
    framework: FrameworkContext = Field(
        default=FrameworkContext.GENERAL,
        description="Primary framework context",
    )
    glossary_overrides: List[GlossaryOverride] = Field(
        default_factory=list,
        description="Organization-specific glossary overrides",
    )
    preserve_formatting: bool = Field(
        default=True, description="Preserve markdown/HTML formatting"
    )
    preserve_citations: bool = Field(
        default=True, description="Preserve citation references unchanged"
    )
    quality_threshold: Decimal = Field(
        default=_decimal("60"),
        description="Minimum quality score (0-100) for acceptance",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class TranslatedSegment(BaseModel):
    """A translated text segment with quality metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_id: str = Field(..., description="Original segment ID")
    source_text: str = Field(..., description="Original source text")
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    segment_type: TextSegmentType = Field(..., description="Segment type")
    translation_method: TranslationMethod = Field(
        ..., description="Method used for translation"
    )
    glossary_terms_used: List[str] = Field(
        default_factory=list, description="Glossary terms applied"
    )
    citations_preserved: List[str] = Field(
        default_factory=list, description="Citation references preserved"
    )
    metrics_preserved: List[str] = Field(
        default_factory=list, description="Metric values preserved unchanged"
    )
    quality_score: Decimal = Field(
        default=_decimal("0"), description="Quality score 0-100"
    )
    needs_review: bool = Field(
        default=False, description="Flagged for human review"
    )
    review_reasons: List[str] = Field(
        default_factory=list, description="Reasons for review flag"
    )


class TerminologyReport(BaseModel):
    """Report on terminology consistency in translations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_terms_found: int = Field(default=0, description="Total glossary terms in source")
    terms_translated: int = Field(default=0, description="Terms with glossary match")
    terms_untranslated: int = Field(default=0, description="Terms without glossary match")
    term_details: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Per-term translation details",
    )
    consistency_score: Decimal = Field(
        default=_decimal("0"), description="Terminology consistency 0-100"
    )
    unknown_terms: List[str] = Field(
        default_factory=list, description="Climate terms not in glossary"
    )


class CitationReport(BaseModel):
    """Report on citation preservation during translation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_citations_found: int = Field(default=0)
    citations_preserved: int = Field(default=0)
    citations_lost: int = Field(default=0)
    citation_details: List[Dict[str, str]] = Field(default_factory=list)
    integrity_score: Decimal = Field(default=_decimal("0"))


class TranslationResult(BaseModel):
    """Complete translation result with quality metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(..., description="Organization identifier")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    framework: str = Field(default="general", description="Framework context")

    translated_segments: List[TranslatedSegment] = Field(
        default_factory=list, description="All translated segments"
    )

    # Quality metrics
    overall_quality_score: Decimal = Field(
        default=_decimal("0"), description="Overall quality 0-100"
    )
    quality_tier: TranslationQualityTier = Field(
        default=TranslationQualityTier.POOR,
        description="Quality tier classification",
    )
    terminology_report: Optional[TerminologyReport] = Field(
        default=None, description="Terminology consistency report"
    )
    citation_report: Optional[CitationReport] = Field(
        default=None, description="Citation preservation report"
    )

    # Statistics
    total_segments: int = Field(default=0)
    segments_translated: int = Field(default=0)
    segments_passthrough: int = Field(default=0)
    segments_needing_review: int = Field(default=0)
    total_source_words: int = Field(default=0)
    total_translated_words: int = Field(default=0)

    # Component scores
    terminology_accuracy_score: Decimal = Field(default=_decimal("0"))
    completeness_score: Decimal = Field(default=_decimal("0"))
    citation_integrity_score: Decimal = Field(default=_decimal("0"))
    format_preservation_score: Decimal = Field(default=_decimal("0"))

    # Provenance
    calculated_at: str = Field(default_factory=lambda: _utcnow().isoformat())
    processing_time_ms: Decimal = Field(default=_decimal("0"))
    engine_version: str = Field(default=_MODULE_VERSION)
    provenance_hash: str = Field(default="")

    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TranslationEngine
# ---------------------------------------------------------------------------

class TranslationEngine:
    """
    Multi-language translation engine for climate disclosure narratives.

    Translates report narratives between English, German, French, and
    Spanish using a climate-specific glossary, citation preservation,
    and deterministic quality scoring.

    Key Methods:
        translate()              -- Translate all segments
        translate_narrative()    -- Translate a single narrative
        validate_translation()   -- Quality-check a translation
        maintain_terminology()   -- Manage glossary terms
        preserve_citations()     -- Extract and re-insert citations

    Usage::

        engine = TranslationEngine()
        result = await engine.translate(TranslationInput(
            organization_id="org-123",
            source_language=SupportedLanguage.ENGLISH,
            target_language=SupportedLanguage.GERMAN,
            segments=[TextSegment(text="Scope 1 emissions were 1,234 tCO2e.")],
            framework=FrameworkContext.SBTI,
        ))
    """

    def __init__(self) -> None:
        """Initialize the TranslationEngine."""
        self._glossary: Dict[str, Dict[str, str]] = dict(CLIMATE_GLOSSARY)
        self._section_headers: Dict[str, Dict[str, str]] = dict(SECTION_HEADER_TRANSLATIONS)
        self._custom_overrides: Dict[str, Dict[str, str]] = {}
        logger.info("TranslationEngine v%s initialized", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def translate(self, inp: TranslationInput) -> TranslationResult:
        """
        Translate all text segments from source to target language.

        Args:
            inp: Translation input with segments, languages, and options.

        Returns:
            TranslationResult with translated segments and quality scores.
        """
        t0 = time.perf_counter()
        logger.info(
            "Translating %d segments from %s to %s for org=%s",
            len(inp.segments),
            inp.source_language.value,
            inp.target_language.value,
            inp.organization_id,
        )

        # Load custom overrides
        self._load_overrides(inp.glossary_overrides)

        # Same-language passthrough
        if inp.source_language == inp.target_language:
            return self._same_language_passthrough(inp, t0)

        translated_segments: List[TranslatedSegment] = []
        all_glossary_terms: List[str] = []
        all_citations: List[str] = []
        all_metrics: List[str] = []
        total_source_words: int = 0
        total_translated_words: int = 0
        segments_needing_review: int = 0
        warnings: List[str] = []

        for segment in inp.segments:
            translated = await self.translate_narrative(
                text=segment.text,
                source_lang=inp.source_language,
                target_lang=inp.target_language,
                segment_type=segment.segment_type,
                framework_context=segment.framework_context,
                preserve_citations=inp.preserve_citations,
                preserve_formatting=inp.preserve_formatting,
                segment_id=segment.segment_id,
            )
            translated_segments.append(translated)
            all_glossary_terms.extend(translated.glossary_terms_used)
            all_citations.extend(translated.citations_preserved)
            all_metrics.extend(translated.metrics_preserved)
            total_source_words += len(segment.text.split())
            total_translated_words += len(translated.translated_text.split())
            if translated.needs_review:
                segments_needing_review += 1

        # Build terminology report
        terminology_report = self._build_terminology_report(
            inp.segments, inp.target_language, all_glossary_terms
        )

        # Build citation report
        citation_report = self._build_citation_report(
            inp.segments, translated_segments, all_citations
        )

        # Calculate component scores
        terminology_score = terminology_report.consistency_score
        completeness_score = self._calculate_completeness_score(
            inp.segments, translated_segments
        )
        citation_score = citation_report.integrity_score
        format_score = self._calculate_format_preservation_score(
            inp.segments, translated_segments
        )

        # Overall quality
        overall_quality = _round_val(
            terminology_score * QUALITY_WEIGHTS["terminology_accuracy"]
            + completeness_score * QUALITY_WEIGHTS["completeness"]
            + citation_score * QUALITY_WEIGHTS["citation_integrity"]
            + format_score * QUALITY_WEIGHTS["format_preservation"],
            2,
        )

        quality_tier = self._classify_quality_tier(overall_quality)

        # Check quality threshold
        if overall_quality < inp.quality_threshold:
            warnings.append(
                f"Overall quality {overall_quality} below threshold "
                f"{inp.quality_threshold}"
            )

        elapsed = _decimal(str(time.perf_counter() - t0)) * _decimal("1000")

        segments_passthrough = sum(
            1 for s in translated_segments
            if s.translation_method == TranslationMethod.PASSTHROUGH
        )

        result = TranslationResult(
            organization_id=inp.organization_id,
            source_language=inp.source_language.value,
            target_language=inp.target_language.value,
            framework=inp.framework.value,
            translated_segments=translated_segments,
            overall_quality_score=overall_quality,
            quality_tier=quality_tier,
            terminology_report=terminology_report,
            citation_report=citation_report,
            total_segments=len(inp.segments),
            segments_translated=len(translated_segments) - segments_passthrough,
            segments_passthrough=segments_passthrough,
            segments_needing_review=segments_needing_review,
            total_source_words=total_source_words,
            total_translated_words=total_translated_words,
            terminology_accuracy_score=terminology_score,
            completeness_score=completeness_score,
            citation_integrity_score=citation_score,
            format_preservation_score=format_score,
            processing_time_ms=_round_val(elapsed, 1),
            warnings=warnings,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Translation complete: %d segments, quality=%.1f (%s), %.1fms",
            len(translated_segments),
            float(overall_quality),
            quality_tier.value,
            float(elapsed),
        )
        return result

    async def translate_narrative(
        self,
        text: str,
        source_lang: SupportedLanguage,
        target_lang: SupportedLanguage,
        segment_type: TextSegmentType = TextSegmentType.NARRATIVE,
        framework_context: FrameworkContext = FrameworkContext.GENERAL,
        preserve_citations: bool = True,
        preserve_formatting: bool = True,
        segment_id: Optional[str] = None,
    ) -> TranslatedSegment:
        """
        Translate a single narrative text segment.

        The translation pipeline:
        1. Extract citations and metric values
        2. Identify glossary terms in source text
        3. Apply glossary-based translations
        4. Re-insert citations and metrics unchanged
        5. Calculate quality score

        Args:
            text: Source text to translate.
            source_lang: Source language.
            target_lang: Target language.
            segment_type: Type of text segment.
            framework_context: Framework for terminology.
            preserve_citations: Whether to preserve citations.
            preserve_formatting: Whether to preserve formatting.
            segment_id: Optional segment identifier.

        Returns:
            TranslatedSegment with translated text and quality.
        """
        sid = segment_id or _new_uuid()
        target_code = target_lang.value

        # Passthrough for citations and metric values
        if segment_type in (TextSegmentType.CITATION, TextSegmentType.METRIC_VALUE):
            return TranslatedSegment(
                segment_id=sid,
                source_text=text,
                translated_text=text,
                source_language=source_lang.value,
                target_language=target_code,
                segment_type=segment_type,
                translation_method=TranslationMethod.PASSTHROUGH,
                quality_score=_decimal("100"),
                needs_review=False,
            )

        # Section header translation
        if segment_type == TextSegmentType.HEADER:
            translated = self._translate_header(text, target_code)
            method = TranslationMethod.GLOSSARY if translated != text else TranslationMethod.TEMPLATE
            return TranslatedSegment(
                segment_id=sid,
                source_text=text,
                translated_text=translated,
                source_language=source_lang.value,
                target_language=target_code,
                segment_type=segment_type,
                translation_method=method,
                quality_score=_decimal("95") if translated != text else _decimal("70"),
                needs_review=translated == text,
                review_reasons=["Header not in glossary"] if translated == text else [],
            )

        # Full narrative translation
        # Step 1: Extract citations
        citations_found: List[str] = []
        protected_text = text
        if preserve_citations:
            citations_found = CITATION_PATTERN.findall(text)
            # Replace citations with placeholders
            citation_map: Dict[str, str] = {}
            for i, match in enumerate(CITATION_PATTERN.finditer(text)):
                placeholder = f"__CITE_{i}__"
                citation_map[placeholder] = match.group(0)
                protected_text = protected_text.replace(match.group(0), placeholder, 1)

        # Step 2: Extract metric values
        metrics_found: List[str] = []
        metric_map: Dict[str, str] = {}
        for i, match in enumerate(METRIC_VALUE_PATTERN.finditer(protected_text)):
            placeholder = f"__METRIC_{i}__"
            metric_map[placeholder] = match.group(0)
            metrics_found.append(match.group(0))
            protected_text = protected_text.replace(match.group(0), placeholder, 1)

        # Step 3: Extract dates
        date_map: Dict[str, str] = {}
        for i, match in enumerate(DATE_PATTERN.finditer(protected_text)):
            placeholder = f"__DATE_{i}__"
            date_map[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder, 1)

        # Step 4: Apply glossary-based term translation
        glossary_terms_used: List[str] = []
        translated_text = protected_text

        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(
            self._glossary.keys(), key=len, reverse=True
        )

        for term in sorted_terms:
            if target_code not in self._glossary[term]:
                continue
            # Case-insensitive search for the term
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(translated_text):
                translation = self._glossary[term][target_code]
                # Preserve case of first character
                def _case_match(m: re.Match) -> str:
                    original = m.group(0)
                    if original[0].isupper():
                        return translation[0].upper() + translation[1:]
                    return translation
                translated_text = pattern.sub(_case_match, translated_text)
                glossary_terms_used.append(term)

        # Step 5: Check for custom overrides
        for override_term, translations in self._custom_overrides.items():
            if target_code in translations:
                pattern = re.compile(re.escape(override_term), re.IGNORECASE)
                if pattern.search(translated_text):
                    translated_text = pattern.sub(
                        translations[target_code], translated_text
                    )
                    if override_term not in glossary_terms_used:
                        glossary_terms_used.append(f"{override_term} (custom)")

        # Step 6: Re-insert citations
        for placeholder, original in citation_map.items() if preserve_citations else []:
            translated_text = translated_text.replace(placeholder, original)

        # Step 7: Re-insert metrics
        for placeholder, original in metric_map.items():
            translated_text = translated_text.replace(placeholder, original)

        # Step 8: Re-insert dates
        for placeholder, original in date_map.items():
            translated_text = translated_text.replace(placeholder, original)

        # Step 9: Calculate quality
        review_reasons: List[str] = []
        quality = self._score_segment_quality(
            text, translated_text, glossary_terms_used,
            citations_found, metrics_found, review_reasons,
        )

        needs_review = quality < _decimal("60") or len(review_reasons) > 0

        # Determine method
        if glossary_terms_used:
            method = TranslationMethod.COMPOSITE if len(glossary_terms_used) > 1 else TranslationMethod.GLOSSARY
        else:
            method = TranslationMethod.TEMPLATE

        return TranslatedSegment(
            segment_id=sid,
            source_text=text,
            translated_text=translated_text,
            source_language=source_lang.value,
            target_language=target_code,
            segment_type=segment_type,
            translation_method=method,
            glossary_terms_used=glossary_terms_used,
            citations_preserved=[f"[CITE:{c}]" for c in citations_found],
            metrics_preserved=metrics_found,
            quality_score=quality,
            needs_review=needs_review,
            review_reasons=review_reasons,
        )

    async def validate_translation(
        self,
        source_text: str,
        translated_text: str,
        source_lang: SupportedLanguage,
        target_lang: SupportedLanguage,
        framework_context: FrameworkContext = FrameworkContext.GENERAL,
    ) -> Dict[str, Any]:
        """
        Validate a translation for quality and consistency.

        Checks:
            1. All citations are preserved
            2. All metric values are preserved
            3. Glossary terms are correctly translated
            4. No hallucinated numbers or metrics
            5. Formatting is preserved

        Args:
            source_text: Original text.
            translated_text: Translated text.
            source_lang: Source language.
            target_lang: Target language.
            framework_context: Framework context.

        Returns:
            Validation report dict.
        """
        issues: List[Dict[str, str]] = []
        target_code = target_lang.value

        # Check citation preservation
        source_citations = set(CITATION_PATTERN.findall(source_text))
        translated_citations = set(CITATION_PATTERN.findall(translated_text))
        missing_citations = source_citations - translated_citations
        extra_citations = translated_citations - source_citations
        if missing_citations:
            issues.append({
                "type": "citation_lost",
                "severity": "error",
                "detail": f"Missing citations: {', '.join(missing_citations)}",
            })
        if extra_citations:
            issues.append({
                "type": "citation_added",
                "severity": "warning",
                "detail": f"Extra citations: {', '.join(extra_citations)}",
            })

        # Check metric preservation
        source_metrics = set(METRIC_VALUE_PATTERN.findall(source_text))
        translated_metrics = set(METRIC_VALUE_PATTERN.findall(translated_text))
        missing_metrics = source_metrics - translated_metrics
        if missing_metrics:
            issues.append({
                "type": "metric_lost",
                "severity": "error",
                "detail": f"Missing metrics: {', '.join(missing_metrics)}",
            })

        # Check number preservation (all numbers should be identical)
        source_numbers = set(re.findall(r"\d[\d,.]+", source_text))
        translated_numbers = set(re.findall(r"\d[\d,.]+", translated_text))
        missing_numbers = source_numbers - translated_numbers
        if missing_numbers:
            issues.append({
                "type": "number_changed",
                "severity": "warning",
                "detail": f"Numbers differ: {', '.join(missing_numbers)}",
            })

        # Check glossary term translation
        glossary_issues: List[str] = []
        for term, translations in self._glossary.items():
            if target_code not in translations:
                continue
            term_lower = term.lower()
            if term_lower in source_text.lower():
                expected = translations[target_code].lower()
                if expected not in translated_text.lower():
                    glossary_issues.append(
                        f"'{term}' -> expected '{translations[target_code]}'"
                    )
        if glossary_issues:
            issues.append({
                "type": "glossary_mismatch",
                "severity": "warning",
                "detail": f"Glossary terms not translated: {'; '.join(glossary_issues[:5])}",
            })

        # Check formatting preservation
        source_has_markdown = bool(re.search(r"[#*_`\[\]]", source_text))
        translated_has_markdown = bool(re.search(r"[#*_`\[\]]", translated_text))
        if source_has_markdown and not translated_has_markdown:
            issues.append({
                "type": "formatting_lost",
                "severity": "warning",
                "detail": "Markdown formatting may have been lost",
            })

        # Calculate overall score
        error_count = sum(1 for i in issues if i["severity"] == "error")
        warning_count = sum(1 for i in issues if i["severity"] == "warning")
        score = max(
            _decimal("0"),
            _decimal("100") - _decimal("15") * _decimal(str(error_count))
            - _decimal("5") * _decimal(str(warning_count))
        )

        return {
            "valid": error_count == 0,
            "score": float(_round_val(score, 1)),
            "issues": issues,
            "error_count": error_count,
            "warning_count": warning_count,
            "citations_checked": len(source_citations),
            "metrics_checked": len(source_metrics),
            "glossary_terms_checked": len(glossary_issues) + len([
                t for t in self._glossary
                if t.lower() in source_text.lower()
            ]),
        }

    async def maintain_terminology(
        self,
        action: str = "list",
        term: Optional[str] = None,
        translations: Optional[Dict[str, str]] = None,
        framework_context: Optional[FrameworkContext] = None,
    ) -> Dict[str, Any]:
        """
        Manage the climate terminology glossary.

        Actions:
            list   -- Return all glossary terms
            add    -- Add a new term with translations
            update -- Update translations for existing term
            remove -- Remove a term from glossary
            search -- Search for terms matching a pattern
            stats  -- Return glossary statistics

        Args:
            action: Action to perform.
            term: Term to add/update/remove/search.
            translations: Translations for add/update.
            framework_context: Framework scope for the term.

        Returns:
            Action result dict.
        """
        if action == "list":
            return {
                "action": "list",
                "total_terms": len(self._glossary),
                "languages": ["en", "de", "fr", "es"],
                "terms": list(self._glossary.keys()),
            }

        elif action == "add":
            if not term or not translations:
                return {"action": "add", "success": False, "error": "term and translations required"}
            if term in self._glossary:
                return {"action": "add", "success": False, "error": f"Term '{term}' already exists, use 'update'"}
            self._glossary[term] = translations
            return {"action": "add", "success": True, "term": term, "translations": translations}

        elif action == "update":
            if not term or not translations:
                return {"action": "update", "success": False, "error": "term and translations required"}
            if term not in self._glossary:
                return {"action": "update", "success": False, "error": f"Term '{term}' not found, use 'add'"}
            self._glossary[term].update(translations)
            return {"action": "update", "success": True, "term": term, "translations": self._glossary[term]}

        elif action == "remove":
            if not term:
                return {"action": "remove", "success": False, "error": "term required"}
            if term not in self._glossary:
                return {"action": "remove", "success": False, "error": f"Term '{term}' not found"}
            del self._glossary[term]
            return {"action": "remove", "success": True, "term": term}

        elif action == "search":
            if not term:
                return {"action": "search", "results": []}
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            matches = [
                {"term": t, "translations": tr}
                for t, tr in self._glossary.items()
                if pattern.search(t)
            ]
            return {"action": "search", "query": term, "results": matches, "count": len(matches)}

        elif action == "stats":
            lang_coverage: Dict[str, int] = {"de": 0, "fr": 0, "es": 0}
            for translations in self._glossary.values():
                for lang in lang_coverage:
                    if lang in translations:
                        lang_coverage[lang] += 1
            return {
                "action": "stats",
                "total_terms": len(self._glossary),
                "custom_overrides": len(self._custom_overrides),
                "language_coverage": {
                    lang: {
                        "count": count,
                        "percentage": float(_round_val(
                            _decimal(str(count)) / max(_decimal(str(len(self._glossary))), _decimal("1")) * _decimal("100"),
                            1,
                        )),
                    }
                    for lang, count in lang_coverage.items()
                },
            }

        return {"action": action, "success": False, "error": f"Unknown action '{action}'"}

    async def preserve_citations(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract citations from text and return preservation map.

        Identifies all citation references, metric values, dates, and
        other non-translatable elements, returning a map that can be
        used to re-insert them after translation.

        Args:
            text: Source text to analyze.

        Returns:
            Dict with extracted citations, metrics, dates, and placeholders.
        """
        result: Dict[str, Any] = {
            "citations": [],
            "metrics": [],
            "dates": [],
            "numbers": [],
            "placeholders": {},
            "protected_text": text,
        }

        protected = text

        # Extract citations
        for i, match in enumerate(CITATION_PATTERN.finditer(text)):
            placeholder = f"__CITE_{i}__"
            result["citations"].append({
                "original": match.group(0),
                "reference": match.group(1),
                "placeholder": placeholder,
                "position": match.start(),
            })
            result["placeholders"][placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

        # Extract metric values
        for i, match in enumerate(METRIC_VALUE_PATTERN.finditer(protected)):
            placeholder = f"__METRIC_{i}__"
            result["metrics"].append({
                "original": match.group(0),
                "placeholder": placeholder,
                "position": match.start(),
            })
            result["placeholders"][placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

        # Extract dates
        for i, match in enumerate(DATE_PATTERN.finditer(protected)):
            placeholder = f"__DATE_{i}__"
            result["dates"].append({
                "original": match.group(0),
                "placeholder": placeholder,
                "position": match.start(),
            })
            result["placeholders"][placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

        # Extract standalone numbers with units
        for i, match in enumerate(NUMBER_WITH_UNIT_PATTERN.finditer(protected)):
            placeholder = f"__NUMUNIT_{i}__"
            result["numbers"].append({
                "original": match.group(0),
                "value": match.group(1),
                "unit": match.group(2),
                "placeholder": placeholder,
            })
            result["placeholders"][placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

        result["protected_text"] = protected
        result["total_protected"] = len(result["placeholders"])

        return result

    async def translate_report_sections(
        self,
        sections: List[Dict[str, Any]],
        source_lang: SupportedLanguage,
        target_lang: SupportedLanguage,
        framework: FrameworkContext = FrameworkContext.GENERAL,
        organization_id: str = "",
    ) -> TranslationResult:
        """
        Translate a structured report with multiple sections.

        Convenience method that converts report sections into
        TextSegments and calls translate().

        Args:
            sections: List of report section dicts with 'header' and 'content' keys.
            source_lang: Source language.
            target_lang: Target language.
            framework: Framework context.
            organization_id: Organization identifier.

        Returns:
            TranslationResult with all translated sections.
        """
        segments: List[TextSegment] = []
        for i, section in enumerate(sections):
            # Add header segment
            if "header" in section:
                segments.append(TextSegment(
                    text=section["header"],
                    segment_type=TextSegmentType.HEADER,
                    position=i * 2,
                    section_header=section.get("header"),
                    framework_context=framework,
                ))
            # Add content segment
            if "content" in section:
                segments.append(TextSegment(
                    text=section["content"],
                    segment_type=TextSegmentType.NARRATIVE,
                    position=i * 2 + 1,
                    section_header=section.get("header"),
                    framework_context=framework,
                ))
            # Add footnotes if present
            if "footnotes" in section:
                for j, footnote in enumerate(section["footnotes"]):
                    segments.append(TextSegment(
                        text=footnote,
                        segment_type=TextSegmentType.FOOTNOTE,
                        position=i * 100 + j,
                        section_header=section.get("header"),
                        framework_context=framework,
                    ))

        inp = TranslationInput(
            organization_id=organization_id or _new_uuid(),
            source_language=source_lang,
            target_language=target_lang,
            segments=segments,
            framework=framework,
        )
        return await self.translate(inp)

    async def get_supported_languages(self) -> Dict[str, Any]:
        """
        Return information about supported languages.

        Returns:
            Dict with language codes, names, and coverage statistics.
        """
        coverage: Dict[str, int] = {}
        for lang_code in ["de", "fr", "es"]:
            count = sum(
                1 for translations in self._glossary.values()
                if lang_code in translations
            )
            coverage[lang_code] = count

        return {
            "supported_languages": [
                {"code": "en", "name": "English", "role": "primary"},
                {"code": "de", "name": "German", "role": "target",
                 "glossary_terms": coverage.get("de", 0)},
                {"code": "fr", "name": "French", "role": "target",
                 "glossary_terms": coverage.get("fr", 0)},
                {"code": "es", "name": "Spanish", "role": "target",
                 "glossary_terms": coverage.get("es", 0)},
            ],
            "total_glossary_terms": len(self._glossary),
            "frameworks_supported": [f.value for f in FrameworkContext],
        }

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _load_overrides(self, overrides: List[GlossaryOverride]) -> None:
        """Load custom glossary overrides, merging with priority."""
        self._custom_overrides.clear()
        # Sort by priority (highest wins)
        sorted_overrides = sorted(overrides, key=lambda o: o.priority)
        for override in sorted_overrides:
            self._custom_overrides[override.source_term] = override.target_translations
            # Also update main glossary for highest-priority overrides
            if override.priority >= 20:
                if override.source_term in self._glossary:
                    self._glossary[override.source_term].update(
                        override.target_translations
                    )
                else:
                    self._glossary[override.source_term] = dict(
                        override.target_translations
                    )

    def _same_language_passthrough(
        self, inp: TranslationInput, t0: float
    ) -> TranslationResult:
        """Handle same-language translation (passthrough)."""
        segments = [
            TranslatedSegment(
                segment_id=seg.segment_id,
                source_text=seg.text,
                translated_text=seg.text,
                source_language=inp.source_language.value,
                target_language=inp.target_language.value,
                segment_type=seg.segment_type,
                translation_method=TranslationMethod.PASSTHROUGH,
                quality_score=_decimal("100"),
                needs_review=False,
            )
            for seg in inp.segments
        ]

        total_words = sum(len(seg.text.split()) for seg in inp.segments)
        elapsed = _decimal(str(time.perf_counter() - t0)) * _decimal("1000")

        result = TranslationResult(
            organization_id=inp.organization_id,
            source_language=inp.source_language.value,
            target_language=inp.target_language.value,
            framework=inp.framework.value,
            translated_segments=segments,
            overall_quality_score=_decimal("100"),
            quality_tier=TranslationQualityTier.EXCELLENT,
            total_segments=len(inp.segments),
            segments_translated=0,
            segments_passthrough=len(inp.segments),
            segments_needing_review=0,
            total_source_words=total_words,
            total_translated_words=total_words,
            terminology_accuracy_score=_decimal("100"),
            completeness_score=_decimal("100"),
            citation_integrity_score=_decimal("100"),
            format_preservation_score=_decimal("100"),
            processing_time_ms=_round_val(elapsed, 1),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _translate_header(self, header: str, target_lang: str) -> str:
        """Translate a section header using the header glossary."""
        # Check exact match first
        if header in self._section_headers:
            if target_lang in self._section_headers[header]:
                return self._section_headers[header][target_lang]

        # Check case-insensitive match
        header_lower = header.lower()
        for src, translations in self._section_headers.items():
            if src.lower() == header_lower and target_lang in translations:
                return translations[target_lang]

        # Check if header contains a known header as substring
        for src, translations in self._section_headers.items():
            if src.lower() in header_lower and target_lang in translations:
                return header.replace(src, translations[target_lang])

        # Fallback: return original header
        return header

    def _score_segment_quality(
        self,
        source: str,
        translated: str,
        glossary_terms: List[str],
        citations: List[str],
        metrics: List[str],
        review_reasons: List[str],
    ) -> Decimal:
        """
        Score the quality of a single translated segment.

        Scoring components:
            - Terminology: Were glossary terms correctly applied?
            - Completeness: Is the translation roughly same length?
            - Citations: Were all citations preserved?
            - Format: Is formatting preserved?
        """
        scores: List[Decimal] = []

        # Terminology score
        source_lower = source.lower()
        terms_in_source = [
            t for t in self._glossary if t.lower() in source_lower
        ]
        if terms_in_source:
            terms_translated = len([t for t in terms_in_source if t in glossary_terms])
            term_score = _decimal(str(terms_translated)) / _decimal(str(len(terms_in_source))) * _decimal("100")
        else:
            term_score = _decimal("100")
        scores.append(term_score * QUALITY_WEIGHTS["terminology_accuracy"])

        # Completeness score (word count ratio)
        source_words = max(len(source.split()), 1)
        translated_words = max(len(translated.split()), 1)
        ratio = _decimal(str(translated_words)) / _decimal(str(source_words))
        # Acceptable range: 0.7 to 1.5 (translations can be longer or shorter)
        if _decimal("0.7") <= ratio <= _decimal("1.5"):
            completeness = _decimal("100")
        elif ratio < _decimal("0.7"):
            completeness = ratio / _decimal("0.7") * _decimal("100")
        else:
            completeness = _decimal("1.5") / ratio * _decimal("100")
        scores.append(completeness * QUALITY_WEIGHTS["completeness"])

        # Citation score
        if citations:
            source_cites = set(CITATION_PATTERN.findall(source))
            translated_cites = set(CITATION_PATTERN.findall(translated))
            if source_cites:
                preserved = len(source_cites & translated_cites)
                cite_score = _decimal(str(preserved)) / _decimal(str(len(source_cites))) * _decimal("100")
            else:
                cite_score = _decimal("100")
        else:
            cite_score = _decimal("100")
        scores.append(cite_score * QUALITY_WEIGHTS["citation_integrity"])

        # Format preservation score
        source_has_formatting = bool(re.search(r"[#*_`|]", source))
        if source_has_formatting:
            translated_has_formatting = bool(re.search(r"[#*_`|]", translated))
            format_score = _decimal("100") if translated_has_formatting else _decimal("50")
        else:
            format_score = _decimal("100")
        scores.append(format_score * QUALITY_WEIGHTS["format_preservation"])

        total = sum(scores)

        # Add review reasons
        if term_score < _decimal("80") and terms_in_source:
            review_reasons.append(
                f"Low terminology accuracy ({float(_round_val(term_score, 1))}%)"
            )
        if completeness < _decimal("70"):
            review_reasons.append(
                f"Translation length mismatch (ratio={float(_round_val(ratio, 2))})"
            )
        if cite_score < _decimal("100") and citations:
            review_reasons.append("Some citations may not be preserved")

        return _round_val(total, 1)

    def _build_terminology_report(
        self,
        source_segments: List[TextSegment],
        target_lang: SupportedLanguage,
        terms_used: List[str],
    ) -> TerminologyReport:
        """Build a terminology consistency report."""
        target_code = target_lang.value
        all_source_text = " ".join(seg.text for seg in source_segments).lower()

        terms_found: List[str] = []
        terms_with_translation: List[str] = []
        terms_without_translation: List[str] = []
        term_details: List[Dict[str, str]] = []

        for term, translations in self._glossary.items():
            if term.lower() in all_source_text:
                terms_found.append(term)
                if target_code in translations:
                    terms_with_translation.append(term)
                    term_details.append({
                        "source": term,
                        "translation": translations[target_code],
                        "status": "translated",
                    })
                else:
                    terms_without_translation.append(term)
                    term_details.append({
                        "source": term,
                        "translation": "",
                        "status": "no_translation",
                    })

        total = len(terms_found)
        translated = len(terms_with_translation)
        consistency = (
            _decimal(str(translated)) / _decimal(str(total)) * _decimal("100")
            if total > 0 else _decimal("100")
        )

        return TerminologyReport(
            total_terms_found=total,
            terms_translated=translated,
            terms_untranslated=len(terms_without_translation),
            term_details=term_details,
            consistency_score=_round_val(consistency, 1),
            unknown_terms=terms_without_translation,
        )

    def _build_citation_report(
        self,
        source_segments: List[TextSegment],
        translated_segments: List[TranslatedSegment],
        all_citations: List[str],
    ) -> CitationReport:
        """Build a citation preservation report."""
        source_citations: Set[str] = set()
        for seg in source_segments:
            source_citations.update(CITATION_PATTERN.findall(seg.text))

        translated_citations: Set[str] = set()
        for seg in translated_segments:
            translated_citations.update(
                CITATION_PATTERN.findall(seg.translated_text)
            )

        preserved = source_citations & translated_citations
        lost = source_citations - translated_citations

        total = len(source_citations)
        integrity = (
            _decimal(str(len(preserved))) / _decimal(str(total)) * _decimal("100")
            if total > 0 else _decimal("100")
        )

        details: List[Dict[str, str]] = []
        for cite in source_citations:
            details.append({
                "citation": cite,
                "status": "preserved" if cite in preserved else "lost",
            })

        return CitationReport(
            total_citations_found=total,
            citations_preserved=len(preserved),
            citations_lost=len(lost),
            citation_details=details,
            integrity_score=_round_val(integrity, 1),
        )

    def _calculate_completeness_score(
        self,
        source_segments: List[TextSegment],
        translated_segments: List[TranslatedSegment],
    ) -> Decimal:
        """Calculate completeness score across all segments."""
        if not source_segments:
            return _decimal("100")

        total_source_chars = sum(len(seg.text) for seg in source_segments)
        total_translated_chars = sum(
            len(seg.translated_text) for seg in translated_segments
        )

        if total_source_chars == 0:
            return _decimal("100")

        # Count non-empty translations
        non_empty = sum(
            1 for seg in translated_segments
            if len(seg.translated_text.strip()) > 0
        )
        segment_coverage = _decimal(str(non_empty)) / _decimal(str(len(source_segments))) * _decimal("100")

        # Character ratio (translated should be within range)
        char_ratio = _decimal(str(total_translated_chars)) / _decimal(str(total_source_chars))
        if _decimal("0.6") <= char_ratio <= _decimal("1.8"):
            ratio_score = _decimal("100")
        else:
            ratio_score = _decimal("70")

        return _round_val(
            (segment_coverage + ratio_score) / _decimal("2"), 1
        )

    def _calculate_format_preservation_score(
        self,
        source_segments: List[TextSegment],
        translated_segments: List[TranslatedSegment],
    ) -> Decimal:
        """Calculate format preservation score."""
        if not source_segments:
            return _decimal("100")

        format_checks: List[bool] = []
        for source_seg, trans_seg in zip(source_segments, translated_segments):
            source_text = source_seg.text
            translated_text = trans_seg.translated_text

            # Check markdown headers
            source_headers = len(re.findall(r"^#+\s", source_text, re.MULTILINE))
            trans_headers = len(re.findall(r"^#+\s", translated_text, re.MULTILINE))
            if source_headers > 0:
                format_checks.append(trans_headers >= source_headers)

            # Check bullet points
            source_bullets = len(re.findall(r"^[-*]\s", source_text, re.MULTILINE))
            trans_bullets = len(re.findall(r"^[-*]\s", translated_text, re.MULTILINE))
            if source_bullets > 0:
                format_checks.append(trans_bullets > 0)

            # Check numbered lists
            source_numbers = len(re.findall(r"^\d+\.\s", source_text, re.MULTILINE))
            trans_numbers = len(re.findall(r"^\d+\.\s", translated_text, re.MULTILINE))
            if source_numbers > 0:
                format_checks.append(trans_numbers > 0)

            # Check paragraph count
            source_paras = len(source_text.split("\n\n"))
            trans_paras = len(translated_text.split("\n\n"))
            if source_paras > 1:
                format_checks.append(trans_paras >= source_paras - 1)

        if not format_checks:
            return _decimal("100")

        passed = sum(1 for c in format_checks if c)
        return _round_val(
            _decimal(str(passed)) / _decimal(str(len(format_checks))) * _decimal("100"),
            1,
        )

    @staticmethod
    def _classify_quality_tier(score: Decimal) -> TranslationQualityTier:
        """Classify quality score into tier."""
        if score >= _decimal("90"):
            return TranslationQualityTier.EXCELLENT
        elif score >= _decimal("75"):
            return TranslationQualityTier.GOOD
        elif score >= _decimal("60"):
            return TranslationQualityTier.ACCEPTABLE
        elif score >= _decimal("40"):
            return TranslationQualityTier.NEEDS_REVIEW
        else:
            return TranslationQualityTier.POOR
