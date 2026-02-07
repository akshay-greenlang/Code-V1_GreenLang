# -*- coding: utf-8 -*-
"""
ML-based PII Detection - SEC-007 Security Scanning Pipeline

Machine learning-based PII detection using Microsoft Presidio for
Named Entity Recognition (NER). Provides higher accuracy than regex
patterns for complex PII types like names, addresses, and context-dependent data.

Integration:
    - Microsoft Presidio Analyzer for entity detection
    - Custom recognizers for GreenLang-specific entities
    - Confidence scoring and thresholds
    - Async scanning for performance

Custom Entities:
    - GREENLANG_TENANT_ID: GreenLang tenant identifiers
    - EMISSION_DATA: Carbon emission data patterns
    - CLIMATE_METRIC: Climate-related measurements

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check Presidio Availability
# ---------------------------------------------------------------------------

try:
    from presidio_analyzer import (
        AnalyzerEngine,
        PatternRecognizer,
        Pattern,
        RecognizerResult,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.info("Microsoft Presidio not installed; ML-based PII detection unavailable")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PIIEntity:
    """A detected PII entity from ML analysis."""

    id: UUID
    entity_type: str
    start: int
    end: int
    score: float
    text_hash: str  # SHA-256 hash (never store raw)
    recognizer: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    analysis_explanation: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalysisResult:
    """Result of Presidio analysis."""

    entities: List[PIIEntity]
    text_length: int
    analysis_duration_ms: float
    recognizers_used: List[str]
    language: str


# ---------------------------------------------------------------------------
# Custom Recognizers
# ---------------------------------------------------------------------------


def _create_greenlang_recognizers() -> List[Any]:
    """Create custom recognizers for GreenLang-specific entities.

    Returns:
        List of PatternRecognizer instances.
    """
    if not PRESIDIO_AVAILABLE:
        return []

    recognizers = []

    # -------------------------------------------------------------------------
    # GreenLang Tenant ID Recognizer
    # -------------------------------------------------------------------------
    tenant_id_recognizer = PatternRecognizer(
        supported_entity="GREENLANG_TENANT_ID",
        name="greenlang_tenant_id_recognizer",
        patterns=[
            Pattern(
                name="tenant_id_uuid",
                regex=r"(?i)tenant[_-]?id['\"]?\s*[:=]\s*['\"]?([a-f0-9-]{36})['\"]?",
                score=0.7,
            ),
            Pattern(
                name="organization_id",
                regex=r"(?i)org(anization)?[_-]?id['\"]?\s*[:=]\s*['\"]?([a-f0-9-]{36})['\"]?",
                score=0.65,
            ),
        ],
        context=["tenant", "organization", "org", "customer", "client"],
    )
    recognizers.append(tenant_id_recognizer)

    # -------------------------------------------------------------------------
    # Emission Data Recognizer
    # -------------------------------------------------------------------------
    emission_recognizer = PatternRecognizer(
        supported_entity="EMISSION_DATA",
        name="greenlang_emission_recognizer",
        patterns=[
            Pattern(
                name="co2_emission",
                regex=r"(?i)(co2|carbon)\s*[:=]?\s*[\d,.]+\s*(kg|tonne|ton|mt|kt|gt)",
                score=0.6,
            ),
            Pattern(
                name="ghg_emission",
                regex=r"(?i)(ghg|greenhouse)\s*[:=]?\s*[\d,.]+\s*(co2e|tco2e|mtco2e)",
                score=0.65,
            ),
            Pattern(
                name="scope_emission",
                regex=r"(?i)scope[_\s]?[123]\s*[:=]?\s*[\d,.]+",
                score=0.55,
            ),
        ],
        context=["emission", "carbon", "ghg", "climate", "scope"],
    )
    recognizers.append(emission_recognizer)

    # -------------------------------------------------------------------------
    # Climate Metric Recognizer
    # -------------------------------------------------------------------------
    climate_recognizer = PatternRecognizer(
        supported_entity="CLIMATE_METRIC",
        name="greenlang_climate_recognizer",
        patterns=[
            Pattern(
                name="energy_consumption",
                regex=r"(?i)(energy|power)\s*[:=]?\s*[\d,.]+\s*(kwh|mwh|gwh|tj|pj)",
                score=0.55,
            ),
            Pattern(
                name="renewable_percentage",
                regex=r"(?i)renewable\s*[:=]?\s*[\d,.]+\s*%",
                score=0.5,
            ),
        ],
        context=["energy", "renewable", "consumption", "power"],
    )
    recognizers.append(climate_recognizer)

    # -------------------------------------------------------------------------
    # AWS Key Recognizer (supplement to built-in)
    # -------------------------------------------------------------------------
    aws_recognizer = PatternRecognizer(
        supported_entity="AWS_ACCESS_KEY",
        name="aws_access_key_recognizer",
        patterns=[
            Pattern(
                name="aws_access_key_id",
                regex=r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b",
                score=0.95,
            ),
        ],
        context=["aws", "amazon", "key", "access", "iam"],
    )
    recognizers.append(aws_recognizer)

    # -------------------------------------------------------------------------
    # JWT Token Recognizer
    # -------------------------------------------------------------------------
    jwt_recognizer = PatternRecognizer(
        supported_entity="JWT_TOKEN",
        name="jwt_token_recognizer",
        patterns=[
            Pattern(
                name="jwt_bearer",
                regex=r"\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b",
                score=0.9,
            ),
        ],
        context=["token", "jwt", "bearer", "authorization", "auth"],
    )
    recognizers.append(jwt_recognizer)

    return recognizers


# ---------------------------------------------------------------------------
# Presidio PII Scanner
# ---------------------------------------------------------------------------


class PresidioPIIScanner:
    """ML-based PII scanner using Microsoft Presidio.

    Provides NER-based detection for complex PII types that are difficult
    to detect with regex patterns alone. Includes custom recognizers for
    GreenLang-specific entity types.

    Example:
        >>> scanner = PresidioPIIScanner()
        >>> if scanner.is_available:
        ...     entities = await scanner.scan_async("Contact John Doe at john@example.com")
        ...     for entity in entities:
        ...         print(f"{entity.entity_type}: {entity.score}")
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        score_threshold: float = 0.5,
        use_custom_recognizers: bool = True,
    ) -> None:
        """Initialize PresidioPIIScanner.

        Args:
            languages: Languages to support (default: ["en"]).
            score_threshold: Minimum confidence threshold.
            use_custom_recognizers: Include GreenLang custom recognizers.
        """
        self._languages = languages or ["en"]
        self._score_threshold = score_threshold
        self._analyzer: Optional[Any] = None
        self._custom_recognizers: List[Any] = []

        if PRESIDIO_AVAILABLE:
            self._initialize_analyzer(use_custom_recognizers)

    @property
    def is_available(self) -> bool:
        """Check if Presidio is available.

        Returns:
            True if Presidio is installed and configured.
        """
        return PRESIDIO_AVAILABLE and self._analyzer is not None

    def _initialize_analyzer(self, use_custom_recognizers: bool) -> None:
        """Initialize the Presidio analyzer engine.

        Args:
            use_custom_recognizers: Include custom recognizers.
        """
        try:
            # Create NLP engine (uses spaCy)
            # Note: Requires spacy and a language model (e.g., en_core_web_lg)
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_lg"},
                ],
            }

            try:
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
            except Exception:
                # Fall back to default if spacy model not available
                logger.warning("spaCy model not available, using default NLP engine")
                nlp_engine = None

            # Create analyzer
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

            # Add custom recognizers
            if use_custom_recognizers:
                self._custom_recognizers = _create_greenlang_recognizers()
                for recognizer in self._custom_recognizers:
                    self._analyzer.registry.add_recognizer(recognizer)

            logger.info(
                "Presidio analyzer initialized with %d custom recognizers",
                len(self._custom_recognizers),
            )

        except Exception as e:
            logger.error("Failed to initialize Presidio analyzer: %s", e)
            self._analyzer = None

    async def scan_async(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        language: str = "en",
    ) -> List[PIIEntity]:
        """Scan text for PII asynchronously.

        Args:
            text: Text to analyze.
            entities: Entity types to detect (None = all).
            language: Language code.

        Returns:
            List of detected PIIEntity objects.
        """
        if not self.is_available:
            logger.warning("Presidio not available, returning empty results")
            return []

        # Run analysis in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._scan_sync,
            text,
            entities,
            language,
        )

    def scan(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        language: str = "en",
    ) -> List[PIIEntity]:
        """Scan text for PII synchronously.

        Args:
            text: Text to analyze.
            entities: Entity types to detect (None = all).
            language: Language code.

        Returns:
            List of detected PIIEntity objects.
        """
        return self._scan_sync(text, entities, language)

    def _scan_sync(
        self,
        text: str,
        entities: Optional[List[str]],
        language: str,
    ) -> List[PIIEntity]:
        """Internal synchronous scan implementation.

        Args:
            text: Text to analyze.
            entities: Entity types to detect.
            language: Language code.

        Returns:
            List of detected PIIEntity objects.
        """
        if not self.is_available:
            return []

        start_time = datetime.utcnow()
        pii_entities: List[PIIEntity] = []

        try:
            # Analyze text
            results = self._analyzer.analyze(
                text=text,
                language=language,
                entities=entities,
                score_threshold=self._score_threshold,
            )

            # Convert to PIIEntity objects
            for result in results:
                # Extract matched text and hash it
                matched_text = text[result.start:result.end]
                import hashlib
                text_hash = hashlib.sha256(matched_text.encode()).hexdigest()

                # Extract context
                context_start = max(0, result.start - 30)
                context_end = min(len(text), result.end + 30)
                context_before = text[context_start:result.start]
                context_after = text[result.end:context_end]

                # Build explanation
                explanation = None
                if result.analysis_explanation:
                    explanation = str(result.analysis_explanation)

                entity = PIIEntity(
                    id=uuid4(),
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    text_hash=text_hash,
                    recognizer=result.recognition_metadata.get(
                        "recognizer_name", "unknown"
                    ) if result.recognition_metadata else "unknown",
                    context_before=context_before,
                    context_after=context_after,
                    analysis_explanation=explanation,
                )
                pii_entities.append(entity)

        except Exception as e:
            logger.error("Presidio analysis failed: %s", e)

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.debug(
            "Presidio scan completed: %d entities in %.2fms",
            len(pii_entities), duration_ms
        )

        return pii_entities

    def get_supported_entities(self) -> List[str]:
        """Get list of supported entity types.

        Returns:
            List of entity type names.
        """
        if not self.is_available:
            return []

        entities = self._analyzer.get_supported_entities()
        return list(entities)

    def analyze_with_result(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        language: str = "en",
    ) -> AnalysisResult:
        """Analyze text and return detailed result.

        Args:
            text: Text to analyze.
            entities: Entity types to detect.
            language: Language code.

        Returns:
            AnalysisResult with entities and metadata.
        """
        start_time = datetime.utcnow()

        pii_entities = self._scan_sync(text, entities, language)

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Collect recognizers used
        recognizers_used = list(set(e.recognizer for e in pii_entities))

        return AnalysisResult(
            entities=pii_entities,
            text_length=len(text),
            analysis_duration_ms=duration_ms,
            recognizers_used=recognizers_used,
            language=language,
        )


# ---------------------------------------------------------------------------
# Hybrid Scanner (Regex + ML)
# ---------------------------------------------------------------------------


class HybridPIIScanner:
    """Combined regex and ML-based PII scanner.

    Uses regex patterns for high-confidence, structured data (SSN, credit cards)
    and ML for complex entities (names, addresses). Provides the best of both
    approaches with configurable combination strategies.

    Example:
        >>> scanner = HybridPIIScanner()
        >>> findings = await scanner.scan_async(
        ...     "John Doe, SSN 123-45-6789, john@example.com"
        ... )
    """

    def __init__(
        self,
        regex_scanner: Optional[Any] = None,
        ml_scanner: Optional[PresidioPIIScanner] = None,
        use_ml_for_names: bool = True,
        use_ml_for_addresses: bool = True,
        dedup_overlap: bool = True,
    ) -> None:
        """Initialize HybridPIIScanner.

        Args:
            regex_scanner: PIIScanner instance (creates default if None).
            ml_scanner: PresidioPIIScanner instance (creates if available).
            use_ml_for_names: Use ML for name detection.
            use_ml_for_addresses: Use ML for address detection.
            dedup_overlap: Remove overlapping detections.
        """
        # Import regex scanner
        from greenlang.infrastructure.security_scanning.pii_scanner import PIIScanner

        self._regex_scanner = regex_scanner or PIIScanner()
        self._ml_scanner = ml_scanner or PresidioPIIScanner()
        self._use_ml_names = use_ml_for_names
        self._use_ml_addresses = use_ml_for_addresses
        self._dedup_overlap = dedup_overlap

    async def scan_async(
        self,
        text: str,
        file_path: Optional[str] = None,
    ) -> List[Any]:
        """Scan text using both regex and ML.

        Args:
            text: Text to scan.
            file_path: Optional source file path.

        Returns:
            Combined list of findings from both scanners.
        """
        findings = []

        # Regex scan
        regex_findings = self._regex_scanner.scan_text(text)
        findings.extend(regex_findings)

        # ML scan (if available)
        if self._ml_scanner.is_available:
            ml_entities = []

            if self._use_ml_names:
                name_entities = await self._ml_scanner.scan_async(
                    text, entities=["PERSON"]
                )
                ml_entities.extend(name_entities)

            if self._use_ml_addresses:
                address_entities = await self._ml_scanner.scan_async(
                    text, entities=["LOCATION", "GPE"]
                )
                ml_entities.extend(address_entities)

            # Also get GreenLang custom entities
            custom_entities = await self._ml_scanner.scan_async(
                text,
                entities=[
                    "GREENLANG_TENANT_ID",
                    "EMISSION_DATA",
                    "CLIMATE_METRIC",
                    "AWS_ACCESS_KEY",
                    "JWT_TOKEN",
                ],
            )
            ml_entities.extend(custom_entities)

            # Convert ML entities to findings format
            from greenlang.infrastructure.security_scanning.pii_scanner import (
                PIIFinding,
                DataClassification,
                PIIType,
                DetectionMethod,
            )

            for entity in ml_entities:
                # Map entity type to PII type
                pii_type = self._map_entity_to_pii_type(entity.entity_type)

                finding = PIIFinding(
                    id=entity.id,
                    pii_type=pii_type,
                    classification=DataClassification.PII,
                    pattern_name=f"ml_{entity.entity_type.lower()}",
                    confidence_score=entity.score,
                    detection_method=DetectionMethod.ML,
                    file_path=file_path,
                    line_number=None,  # ML doesn't track lines
                    column_start=entity.start,
                    column_end=entity.end,
                    context_before=entity.context_before,
                    context_after=entity.context_after,
                    matched_text_hash=entity.text_hash,
                    exposure_risk="medium",
                )
                findings.append(finding)

        # Deduplicate overlapping findings
        if self._dedup_overlap:
            findings = self._deduplicate_overlaps(findings)

        return findings

    def _map_entity_to_pii_type(self, entity_type: str) -> Any:
        """Map Presidio entity type to PIIType enum.

        Args:
            entity_type: Presidio entity type name.

        Returns:
            Corresponding PIIType value.
        """
        from greenlang.infrastructure.security_scanning.pii_scanner import PIIType

        mapping = {
            "PERSON": PIIType.NAME,
            "LOCATION": PIIType.ADDRESS,
            "GPE": PIIType.ADDRESS,
            "PHONE_NUMBER": PIIType.PHONE,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "US_SSN": PIIType.SSN,
            "US_PASSPORT": PIIType.PASSPORT,
            "US_DRIVER_LICENSE": PIIType.DRIVER_LICENSE,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "MEDICAL_LICENSE": PIIType.MEDICAL_RECORD,
            "US_BANK_NUMBER": PIIType.FINANCIAL_ACCOUNT,
            "GREENLANG_TENANT_ID": PIIType.TENANT_ID,
            "EMISSION_DATA": PIIType.EMISSION_DATA,
            "AWS_ACCESS_KEY": PIIType.API_KEY,
            "JWT_TOKEN": PIIType.TOKEN,
        }

        return mapping.get(entity_type, PIIType.OTHER)

    def _deduplicate_overlaps(self, findings: List[Any]) -> List[Any]:
        """Remove overlapping findings, keeping highest confidence.

        Args:
            findings: List of findings.

        Returns:
            Deduplicated findings.
        """
        if not findings:
            return findings

        # Sort by confidence (highest first)
        sorted_findings = sorted(
            findings,
            key=lambda f: f.confidence_score,
            reverse=True,
        )

        deduped: List[Any] = []
        used_ranges: List[tuple] = []

        for finding in sorted_findings:
            # Check if this finding overlaps with any kept finding
            start = finding.column_start or 0
            end = finding.column_end or 0

            overlaps = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break

            if not overlaps:
                deduped.append(finding)
                used_ranges.append((start, end))

        return deduped


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_presidio_scanner: Optional[PresidioPIIScanner] = None


def get_presidio_scanner() -> Optional[PresidioPIIScanner]:
    """Get or create the global Presidio scanner instance.

    Returns:
        The global PresidioPIIScanner or None if unavailable.
    """
    global _global_presidio_scanner

    if _global_presidio_scanner is None and PRESIDIO_AVAILABLE:
        _global_presidio_scanner = PresidioPIIScanner()

    return _global_presidio_scanner


__all__ = [
    "PresidioPIIScanner",
    "HybridPIIScanner",
    "PIIEntity",
    "AnalysisResult",
    "PRESIDIO_AVAILABLE",
    "get_presidio_scanner",
]
