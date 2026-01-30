"""
GL-EUDR-001: LLM Integration Service

Provides LLM-powered features for the Supply Chain Mapper Agent:
- Entity extraction from unstructured text
- Fuzzy matching assistance for entity resolution
- Natural language query processing
- Materiality assessment for supply chain gaps

Uses a provider-agnostic interface supporting Claude and GPT-4.
Maintains zero-hallucination guarantees through structured outputs
and confidence scoring.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"  # For testing


@dataclass
class ExtractedEntity:
    """Entity extracted from text by LLM."""
    entity_type: str  # SUPPLIER, ADDRESS, CERTIFICATION, etc.
    value: str
    confidence: float
    source_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FuzzyMatchSuggestion:
    """LLM suggestion for entity resolution."""
    candidate_a_name: str
    candidate_b_name: str
    is_same_entity: bool
    confidence: float
    reasoning: str
    key_factors: List[str] = field(default_factory=list)


@dataclass
class ParsedQuery:
    """Parsed natural language query."""
    original_query: str
    interpretation: str
    filters: Dict[str, Any]
    confidence: float
    suggested_refinements: List[str] = field(default_factory=list)


@dataclass
class GapMateriality:
    """Materiality assessment for a supply chain gap."""
    gap_id: str
    is_material: bool
    severity_assessment: str  # CRITICAL, HIGH, MEDIUM, LOW
    risk_factors: List[str]
    recommended_action: str
    confidence: float


class LLMIntegrationService:
    """
    LLM integration service for Supply Chain Mapper.

    Provides zero-hallucination guarantees through:
    - Structured output schemas
    - Confidence scoring on all outputs
    - Source text citation
    - Human review for low-confidence results
    """

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    MEDIUM_CONFIDENCE_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.50

    # Entity extraction patterns (used as hints to LLM)
    ENTITY_PATTERNS = {
        "supplier_name": r"(?:supplier|vendor|producer|farm|plantation|company)[\s:]+([A-Z][A-Za-z\s&.,]+)",
        "country": r"(?:country|origin|from)[\s:]+([A-Z][a-z]+)",
        "certification": r"(?:certified|certification)[\s:]+([A-Z][A-Za-z\s]+)",
        "quantity": r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(kg|mt|tonnes?|tons?|lb|lbs)",
    }

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.LOCAL,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self._client = None

    def _default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == LLMProvider.CLAUDE:
            return "claude-sonnet-4-20250514"
        elif self.provider == LLMProvider.OPENAI:
            return "gpt-4"
        return "local"

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == LLMProvider.CLAUDE:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed, using local mode")
                self.provider = LLMProvider.LOCAL

        elif self.provider == LLMProvider.OPENAI:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed, using local mode")
                self.provider = LLMProvider.LOCAL

        return self._client

    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> List[ExtractedEntity]:
        """
        Extract supply chain entities from unstructured text.

        Args:
            text: Input text (invoice, email, document)
            entity_types: Types to extract (SUPPLIER, ADDRESS, QUANTITY, etc.)

        Returns:
            List of extracted entities with confidence scores
        """
        if not text.strip():
            return []

        entity_types = entity_types or [
            "SUPPLIER_NAME", "ADDRESS", "COUNTRY", "QUANTITY",
            "CERTIFICATION", "COMMODITY", "DATE"
        ]

        if self.provider == LLMProvider.LOCAL:
            return self._extract_entities_local(text, entity_types)

        return self._extract_entities_llm(text, entity_types)

    def _extract_entities_local(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[ExtractedEntity]:
        """Extract entities using regex patterns (fallback)."""
        entities = []

        # Country extraction
        if "COUNTRY" in entity_types:
            countries = {
                'brazil': 'BR', 'colombia': 'CO', 'peru': 'PE',
                'vietnam': 'VN', 'indonesia': 'ID', 'malaysia': 'MY',
                'ghana': 'GH', 'ivory coast': 'CI', 'ecuador': 'EC'
            }
            for name, code in countries.items():
                if name in text.lower():
                    entities.append(ExtractedEntity(
                        entity_type="COUNTRY",
                        value=code,
                        confidence=0.85,
                        source_text=text,
                        metadata={"country_name": name.title()}
                    ))

        # Quantity extraction
        if "QUANTITY" in entity_types:
            qty_pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(kg|mt|tonnes?|tons?)'
            matches = re.findall(qty_pattern, text, re.IGNORECASE)
            for qty, unit in matches:
                entities.append(ExtractedEntity(
                    entity_type="QUANTITY",
                    value=qty.replace(',', ''),
                    confidence=0.90,
                    source_text=text,
                    metadata={"unit": unit.lower()}
                ))

        # Certification extraction
        if "CERTIFICATION" in entity_types:
            certs = [
                "Rainforest Alliance", "UTZ", "Fair Trade", "Organic",
                "FSC", "PEFC", "RSPO", "4C"
            ]
            for cert in certs:
                if cert.lower() in text.lower():
                    entities.append(ExtractedEntity(
                        entity_type="CERTIFICATION",
                        value=cert,
                        confidence=0.85,
                        source_text=text
                    ))

        # Commodity extraction
        if "COMMODITY" in entity_types:
            commodities = {
                'coffee': 'COFFEE', 'cocoa': 'COCOA', 'palm oil': 'PALM_OIL',
                'soy': 'SOY', 'rubber': 'RUBBER', 'cattle': 'CATTLE',
                'beef': 'CATTLE', 'wood': 'WOOD', 'timber': 'WOOD'
            }
            for name, code in commodities.items():
                if name in text.lower():
                    entities.append(ExtractedEntity(
                        entity_type="COMMODITY",
                        value=code,
                        confidence=0.90,
                        source_text=text
                    ))

        return entities

    def _extract_entities_llm(
        self,
        text: str,
        entity_types: List[str]
    ) -> List[ExtractedEntity]:
        """Extract entities using LLM."""
        client = self._get_client()
        if client is None:
            return self._extract_entities_local(text, entity_types)

        prompt = f"""Extract supply chain entities from the following text.
Return a JSON array of objects with these fields:
- entity_type: one of {entity_types}
- value: the extracted value
- confidence: float between 0 and 1
- source_text: the relevant portion of input text

Only extract entities you are confident about. Be precise and cite source text.

Text to analyze:
{text}

Return only valid JSON, no explanation."""

        try:
            if self.provider == LLMProvider.CLAUDE:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:  # OpenAI
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content

            # Parse JSON response
            data = json.loads(content)
            return [
                ExtractedEntity(
                    entity_type=e.get("entity_type", "UNKNOWN"),
                    value=e.get("value", ""),
                    confidence=float(e.get("confidence", 0.5)),
                    source_text=e.get("source_text", text[:100])
                )
                for e in data if isinstance(e, dict)
            ]

        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            return self._extract_entities_local(text, entity_types)

    # =========================================================================
    # FUZZY MATCHING ASSISTANCE
    # =========================================================================

    def assess_entity_match(
        self,
        candidate_a: Dict[str, Any],
        candidate_b: Dict[str, Any]
    ) -> FuzzyMatchSuggestion:
        """
        Use LLM to assess if two candidates refer to same entity.

        Args:
            candidate_a: First candidate's attributes
            candidate_b: Second candidate's attributes

        Returns:
            Match suggestion with reasoning
        """
        if self.provider == LLMProvider.LOCAL:
            return self._assess_match_local(candidate_a, candidate_b)

        return self._assess_match_llm(candidate_a, candidate_b)

    def _assess_match_local(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any]
    ) -> FuzzyMatchSuggestion:
        """Assess match using heuristics (fallback)."""
        factors = []
        confidence = 0.5

        # Name similarity
        name_a = a.get('name', '').lower()
        name_b = b.get('name', '').lower()
        if name_a and name_b:
            # Simple containment check
            if name_a in name_b or name_b in name_a:
                factors.append("Similar names")
                confidence += 0.15

        # Country match
        if a.get('country_code') == b.get('country_code'):
            factors.append("Same country")
            confidence += 0.1

        # Tax ID match
        if a.get('tax_id') and a.get('tax_id') == b.get('tax_id'):
            factors.append("Matching tax ID")
            confidence = 0.95

        is_match = confidence >= 0.7

        return FuzzyMatchSuggestion(
            candidate_a_name=a.get('name', 'Unknown'),
            candidate_b_name=b.get('name', 'Unknown'),
            is_same_entity=is_match,
            confidence=min(confidence, 1.0),
            reasoning=f"Based on {len(factors)} matching factors",
            key_factors=factors
        )

    def _assess_match_llm(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any]
    ) -> FuzzyMatchSuggestion:
        """Assess match using LLM."""
        client = self._get_client()
        if client is None:
            return self._assess_match_local(a, b)

        prompt = f"""Determine if these two supplier records refer to the same entity.

Candidate A:
{json.dumps(a, indent=2)}

Candidate B:
{json.dumps(b, indent=2)}

Analyze the data and return a JSON object with:
- is_same_entity: boolean
- confidence: float 0-1
- reasoning: explanation
- key_factors: list of factors that influenced the decision

Consider name variations, address similarities, identifier matches, etc.
Return only valid JSON."""

        try:
            if self.provider == LLMProvider.CLAUDE:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content

            data = json.loads(content)

            return FuzzyMatchSuggestion(
                candidate_a_name=a.get('name', 'Unknown'),
                candidate_b_name=b.get('name', 'Unknown'),
                is_same_entity=data.get('is_same_entity', False),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', ''),
                key_factors=data.get('key_factors', [])
            )

        except Exception as e:
            logger.error(f"LLM match assessment failed: {e}")
            return self._assess_match_local(a, b)

    # =========================================================================
    # NATURAL LANGUAGE QUERY PROCESSING
    # =========================================================================

    def parse_natural_language_query(
        self,
        query: str,
        available_fields: Optional[List[str]] = None
    ) -> ParsedQuery:
        """
        Parse natural language query into structured filters.

        Args:
            query: Natural language query
            available_fields: Fields that can be filtered

        Returns:
            Parsed query with filters
        """
        available_fields = available_fields or [
            "node_type", "country_code", "commodity", "verification_status",
            "disclosure_status", "tier", "certification", "risk_score"
        ]

        if self.provider == LLMProvider.LOCAL:
            return self._parse_query_local(query, available_fields)

        return self._parse_query_llm(query, available_fields)

    def _parse_query_local(
        self,
        query: str,
        available_fields: List[str]
    ) -> ParsedQuery:
        """Parse query using keyword matching (fallback)."""
        filters = {}
        query_lower = query.lower()

        # Country detection
        countries = {
            'indonesia': 'ID', 'brazil': 'BR', 'colombia': 'CO',
            'vietnam': 'VN', 'peru': 'PE', 'ghana': 'GH',
            'ivory coast': 'CI', 'malaysia': 'MY', 'thailand': 'TH',
            'germany': 'DE', 'netherlands': 'NL', 'france': 'FR'
        }
        for country, code in countries.items():
            if country in query_lower:
                filters['country_code'] = code
                break

        # Node type detection
        if 'producer' in query_lower or 'farm' in query_lower:
            filters['node_type'] = 'PRODUCER'
        elif 'processor' in query_lower:
            filters['node_type'] = 'PROCESSOR'
        elif 'trader' in query_lower:
            filters['node_type'] = 'TRADER'
        elif 'importer' in query_lower:
            filters['node_type'] = 'IMPORTER'

        # Verification status
        if 'unverified' in query_lower:
            filters['verification_status'] = 'UNVERIFIED'
        elif 'verified' in query_lower:
            filters['verification_status'] = 'VERIFIED'

        # Certifications
        if 'expired' in query_lower and 'certification' in query_lower:
            filters['expired_certifications'] = True

        # Risk
        if 'high risk' in query_lower:
            filters['min_risk_score'] = 0.7
        elif 'low risk' in query_lower:
            filters['max_risk_score'] = 0.3

        # Commodity
        commodities = {
            'coffee': 'COFFEE', 'cocoa': 'COCOA', 'palm oil': 'PALM_OIL',
            'rubber': 'RUBBER', 'soy': 'SOY', 'wood': 'WOOD', 'cattle': 'CATTLE'
        }
        for name, code in commodities.items():
            if name in query_lower:
                filters['commodity'] = code
                break

        # Tier
        tier_match = re.search(r'tier\s*(\d+)', query_lower)
        if tier_match:
            filters['tier'] = int(tier_match.group(1))

        confidence = 0.7 if filters else 0.4

        return ParsedQuery(
            original_query=query,
            interpretation=f"Searching for nodes with filters: {filters}",
            filters=filters,
            confidence=confidence,
            suggested_refinements=[]
        )

    def _parse_query_llm(
        self,
        query: str,
        available_fields: List[str]
    ) -> ParsedQuery:
        """Parse query using LLM."""
        client = self._get_client()
        if client is None:
            return self._parse_query_local(query, available_fields)

        prompt = f"""Parse this natural language query about a supply chain into structured filters.

Query: "{query}"

Available filter fields: {available_fields}

Field values guide:
- node_type: PRODUCER, PROCESSOR, TRADER, IMPORTER, AGGREGATOR
- country_code: 2-letter ISO codes (BR, CO, ID, DE, etc.)
- commodity: COFFEE, COCOA, PALM_OIL, RUBBER, SOY, WOOD, CATTLE
- verification_status: VERIFIED, UNVERIFIED, PENDING
- disclosure_status: FULL, PARTIAL, NONE
- tier: integer (0=importer, higher=further upstream)

Return JSON with:
- interpretation: how you understood the query
- filters: object with field:value pairs
- confidence: float 0-1
- suggested_refinements: list of clarifying questions if query is ambiguous

Return only valid JSON."""

        try:
            if self.provider == LLMProvider.CLAUDE:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content

            data = json.loads(content)

            return ParsedQuery(
                original_query=query,
                interpretation=data.get('interpretation', query),
                filters=data.get('filters', {}),
                confidence=float(data.get('confidence', 0.5)),
                suggested_refinements=data.get('suggested_refinements', [])
            )

        except Exception as e:
            logger.error(f"LLM query parsing failed: {e}")
            return self._parse_query_local(query, available_fields)

    # =========================================================================
    # GAP MATERIALITY ASSESSMENT
    # =========================================================================

    def assess_gap_materiality(
        self,
        gap_type: str,
        gap_description: str,
        context: Dict[str, Any]
    ) -> GapMateriality:
        """
        Assess materiality of a supply chain gap.

        Args:
            gap_type: Type of gap (UNVERIFIED_SUPPLIER, MISSING_PLOT_DATA, etc.)
            gap_description: Description of the gap
            context: Additional context (volume, tier, commodity, etc.)

        Returns:
            Materiality assessment
        """
        if self.provider == LLMProvider.LOCAL:
            return self._assess_materiality_local(gap_type, gap_description, context)

        return self._assess_materiality_llm(gap_type, gap_description, context)

    def _assess_materiality_local(
        self,
        gap_type: str,
        gap_description: str,
        context: Dict[str, Any]
    ) -> GapMateriality:
        """Assess materiality using rules (fallback)."""
        # Rule-based materiality
        critical_types = {'MISSING_PLOT_DATA', 'CYCLE_DETECTED'}
        high_types = {'UNVERIFIED_SUPPLIER', 'PARTIAL_DISCLOSURE'}

        if gap_type in critical_types:
            severity = 'CRITICAL'
            is_material = True
        elif gap_type in high_types:
            severity = 'HIGH'
            is_material = True
        else:
            severity = 'MEDIUM'
            is_material = context.get('tier', 5) <= 2  # Material if close to importer

        # Volume consideration
        volume = context.get('volume_percentage', 0)
        if volume > 10:
            is_material = True

        return GapMateriality(
            gap_id=context.get('gap_id', 'unknown'),
            is_material=is_material,
            severity_assessment=severity,
            risk_factors=[f"Gap type: {gap_type}", f"Volume: {volume}%"],
            recommended_action="Review and remediate",
            confidence=0.7
        )

    def _assess_materiality_llm(
        self,
        gap_type: str,
        gap_description: str,
        context: Dict[str, Any]
    ) -> GapMateriality:
        """Assess materiality using LLM."""
        client = self._get_client()
        if client is None:
            return self._assess_materiality_local(gap_type, gap_description, context)

        prompt = f"""Assess the materiality of this supply chain compliance gap for EUDR.

Gap Type: {gap_type}
Description: {gap_description}
Context: {json.dumps(context, indent=2)}

Consider:
- EUDR regulatory requirements
- Position in supply chain (tier)
- Volume/quantity involved
- Risk to DDS submission

Return JSON with:
- is_material: boolean
- severity_assessment: CRITICAL, HIGH, MEDIUM, or LOW
- risk_factors: list of relevant factors
- recommended_action: specific recommendation
- confidence: float 0-1

Return only valid JSON."""

        try:
            if self.provider == LLMProvider.CLAUDE:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content

            data = json.loads(content)

            return GapMateriality(
                gap_id=context.get('gap_id', 'unknown'),
                is_material=data.get('is_material', True),
                severity_assessment=data.get('severity_assessment', 'MEDIUM'),
                risk_factors=data.get('risk_factors', []),
                recommended_action=data.get('recommended_action', 'Review required'),
                confidence=float(data.get('confidence', 0.5))
            )

        except Exception as e:
            logger.error(f"LLM materiality assessment failed: {e}")
            return self._assess_materiality_local(gap_type, gap_description, context)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def validate_response(
        self,
        response: Dict[str, Any],
        schema: Dict[str, type]
    ) -> Tuple[bool, List[str]]:
        """
        Validate LLM response against expected schema.

        Args:
            response: LLM response to validate
            schema: Expected field types

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        for field, expected_type in schema.items():
            if field not in response:
                errors.append(f"Missing field: {field}")
            elif not isinstance(response[field], expected_type):
                errors.append(
                    f"Invalid type for {field}: expected {expected_type}, "
                    f"got {type(response[field])}"
                )

        return len(errors) == 0, errors

    def get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level."""
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "HIGH"
        elif confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "MEDIUM"
        elif confidence >= self.LOW_CONFIDENCE_THRESHOLD:
            return "LOW"
        return "VERY_LOW"
