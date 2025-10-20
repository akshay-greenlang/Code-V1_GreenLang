"""
MaterialityAgent - AI-Powered Double Materiality Assessment Agent

This agent conducts double materiality assessments per ESRS 1 using AI/LLM capabilities
while maintaining complete audit trails and requiring mandatory human review.

CRITICAL: This agent uses AI/LLM for analysis - NOT ZERO-HALLUCINATION.
All AI-generated assessments MUST be reviewed by qualified sustainability professionals.

Responsibilities:
1. Impact materiality scoring (severity × scope × irremediability)
2. Financial materiality scoring (magnitude × likelihood)
3. RAG-based stakeholder consultation analysis
4. Material topic identification per ESRS 1
5. Materiality matrix generation
6. Natural language rationale generation

Key Features:
- AI-powered analysis using GPT-4 / Claude 3.5 Sonnet
- RAG-based stakeholder consultation synthesis
- <10 minutes processing time
- 80% AI automation, 20% human review required
- Complete audit trail and provenance
- Confidence tracking for all assessments

⚠️ MANDATORY HUMAN REVIEW: All AI-generated assessments must be reviewed and approved
⚠️ NOT ZERO-HALLUCINATION: LLM-based scoring requires expert validation
⚠️ LEGAL RESPONSIBILITY: Final materiality determination is the company's responsibility

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator

# Import validation utilities
import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent))
from utils.validation import (
    validate_file_size,
    validate_string_length,
    sanitize_dict_keys,
    ValidationError as InputValidationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# AI/LLM INTEGRATION
# ============================================================================

class LLMConfig(BaseModel):
    """Configuration for LLM API."""
    provider: str = Field(default="openai", description="LLM provider: 'openai' or 'anthropic'")
    model: str = Field(default="gpt-4o", description="Model name")
    api_key: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)
    timeout: int = Field(default=30, ge=1, le=300)


class LLMClient:
    """
    Client for LLM API calls (OpenAI or Anthropic).

    This client handles all AI/LLM interactions with proper error handling
    and confidence tracking.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.provider = config.provider.lower()

        # Get API key from config or environment
        self.api_key = config.api_key or os.getenv(
            "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
        )

        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}. AI features will be disabled.")
            self.enabled = False
        else:
            self.enabled = True

        # Initialize client libraries
        self.client = None
        if self.enabled:
            try:
                if self.provider == "openai":
                    import openai
                    self.client = openai.OpenAI(api_key=self.api_key)
                elif self.provider == "anthropic":
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                else:
                    logger.error(f"Unsupported LLM provider: {self.provider}")
                    self.enabled = False
            except ImportError as e:
                logger.error(f"Failed to import {self.provider} library: {e}")
                self.enabled = False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """
        Generate text using LLM.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User query prompt
            response_format: Optional response format ("json" for structured output)

        Returns:
            Tuple of (generated_text, confidence_score)
        """
        if not self.enabled:
            logger.warning("LLM client not enabled. Returning None.")
            return None, 0.0

        try:
            if self.provider == "openai":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }

                if response_format == "json":
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)

                text = response.choices[0].message.content
                # OpenAI doesn't provide confidence directly, estimate from finish_reason
                confidence = 0.85 if response.choices[0].finish_reason == "stop" else 0.5

                return text, confidence

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )

                text = response.content[0].text
                # Anthropic doesn't provide confidence, estimate based on stop reason
                confidence = 0.85 if response.stop_reason == "end_turn" else 0.5

                return text, confidence

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None, 0.0


# ============================================================================
# RAG SYSTEM (Simplified)
# ============================================================================

class RAGSystem:
    """
    Retrieval-Augmented Generation system for stakeholder analysis.

    This is a simplified RAG implementation. Production version would use
    vector databases (Pinecone, Weaviate, etc.) for scalable retrieval.
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize RAG system.

        Args:
            documents: List of document dictionaries (stakeholder input, guidance, etc.)
        """
        self.documents = documents or []
        logger.info(f"RAG system initialized with {len(self.documents)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_type: Filter by document type (optional)

        Returns:
            List of relevant documents
        """
        # Simplified keyword-based retrieval
        # Production would use vector similarity search

        query_lower = query.lower()
        scored_docs = []

        for doc in self.documents:
            if filter_type and doc.get("type") != filter_type:
                continue

            # Simple keyword matching score
            text = str(doc.get("content", "")).lower()
            score = sum(1 for word in query_lower.split() if word in text)

            if score > 0:
                scored_docs.append((score, doc))

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ImpactMaterialityScore(BaseModel):
    """Impact materiality assessment scores."""
    severity: float = Field(ge=0.0, le=10.0, description="Severity of impact (0-10)")
    scope: float = Field(ge=0.0, le=10.0, description="Scope/scale of impact (0-10)")
    irremediability: float = Field(ge=0.0, le=10.0, description="Difficulty to remediate (0-10)")
    score: float = Field(ge=0.0, le=10.0, description="Overall impact score")
    is_material: bool = Field(description="Is material from impact perspective?")
    rationale: str = Field(description="Explanation of assessment")
    confidence: float = Field(ge=0.0, le=1.0, default=0.7, description="AI confidence score")
    impact_type: List[str] = Field(default_factory=list, description="Type of impact")
    affected_stakeholders: List[str] = Field(default_factory=list)
    time_horizon: str = Field(default="medium_term")
    value_chain_stage: List[str] = Field(default_factory=list)


class FinancialMaterialityScore(BaseModel):
    """Financial materiality assessment scores."""
    magnitude: float = Field(ge=0.0, le=10.0, description="Magnitude of financial effect (0-10)")
    likelihood: float = Field(ge=0.0, le=10.0, description="Likelihood of occurrence (0-10)")
    score: float = Field(ge=0.0, le=10.0, description="Overall financial score")
    is_material: bool = Field(description="Is material from financial perspective?")
    rationale: str = Field(description="Explanation of assessment")
    confidence: float = Field(ge=0.0, le=1.0, default=0.7, description="AI confidence score")
    effect_type: List[str] = Field(default_factory=list, description="Risk or opportunity")
    financial_impact_areas: List[str] = Field(default_factory=list)
    time_horizon: str = Field(default="medium_term")


class MaterialityTopic(BaseModel):
    """Assessed sustainability topic."""
    topic_id: str
    topic_name: str
    esrs_standard: str
    topic_description: Optional[str] = None
    impact_materiality: ImpactMaterialityScore
    financial_materiality: FinancialMaterialityScore
    double_material: bool
    materiality_conclusion: str  # "material", "not_material", "borderline"
    disclosure_requirements: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    review_status: str = Field(default="pending_review")
    reviewer_comments: Optional[str] = None
    human_override: Optional[Dict[str, Any]] = None


class StakeholderPerspective(BaseModel):
    """Synthesized stakeholder perspective for a topic."""
    topic_id: str
    stakeholder_groups: List[str]
    key_concerns: List[str]
    consensus_view: Optional[str] = None
    divergent_views: List[str] = Field(default_factory=list)
    participants_count: int = 0
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class MaterialityMatrix(BaseModel):
    """Materiality matrix visualization data."""
    chart_data: List[Dict[str, Any]]
    quadrants: Dict[str, List[str]]


class AssessmentMetadata(BaseModel):
    """Metadata for materiality assessment."""
    assessment_id: str
    assessment_date: str
    reporting_period: Dict[str, Any]
    company_id: str
    version: str = "v1.0"
    status: str = "draft"  # "draft", "under_review", "approved", "final"
    approved_by: Optional[str] = None
    approval_date: Optional[str] = None
    ai_powered: bool = True
    requires_human_review: bool = True


class MethodologyInfo(BaseModel):
    """Methodology information."""
    framework: str = "ESRS_1_Double_Materiality"
    impact_threshold: float = Field(default=5.0, ge=0.0, le=10.0)
    financial_threshold: float = Field(default=5.0, ge=0.0, le=10.0)
    double_materiality_rule: str = Field(default="either_or")  # "either_or" or "both_required"
    scoring_methodology: Dict[str, str] = Field(default_factory=dict)
    stakeholder_engagement: Optional[Dict[str, Any]] = None


# ============================================================================
# MATERIALITY AGENT
# ============================================================================

class MaterialityAgent:
    """
    AI-Powered Double Materiality Assessment Agent per ESRS 1.

    This agent uses LLMs to assess impact and financial materiality of sustainability
    topics, synthesize stakeholder perspectives, and generate natural language rationales.

    ⚠️ IMPORTANT: NOT DETERMINISTIC - Uses AI/LLM for assessments
    ⚠️ MANDATORY HUMAN REVIEW REQUIRED
    ⚠️ Company is legally responsible for final materiality determinations

    Performance: <10 minutes for 10 topics
    Automation: ~80% AI, 20% human review
    """

    def __init__(
        self,
        esrs_data_points_path: Union[str, Path],
        llm_config: Optional[LLMConfig] = None,
        impact_threshold: float = 5.0,
        financial_threshold: float = 5.0,
        stakeholder_documents: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the MaterialityAgent.

        Args:
            esrs_data_points_path: Path to ESRS data points catalog JSON
            llm_config: LLM configuration (optional, uses defaults if not provided)
            impact_threshold: Threshold for impact materiality (0-10 scale)
            financial_threshold: Threshold for financial materiality (0-10 scale)
            stakeholder_documents: Stakeholder consultation documents for RAG (optional)
        """
        self.esrs_data_points_path = Path(esrs_data_points_path)
        self.impact_threshold = impact_threshold
        self.financial_threshold = financial_threshold

        # Initialize LLM client
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)

        # Initialize RAG system
        self.rag_system = RAGSystem(stakeholder_documents or [])

        # Load ESRS data points
        self.esrs_catalog = self._load_esrs_catalog()

        # ESRS topical standards
        self.esrs_topics = self._get_esrs_topics()

        # Statistics
        self.stats = {
            "topics_assessed": 0,
            "material_topics": 0,
            "impact_material": 0,
            "financial_material": 0,
            "double_material": 0,
            "llm_api_calls": 0,
            "total_confidence": 0.0,
            "start_time": None,
            "end_time": None
        }

        # Review flags
        self.review_flags: List[Dict[str, Any]] = []

        logger.info(f"MaterialityAgent initialized")
        logger.warning("⚠️ AI-POWERED AGENT: All assessments require human review")

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def _load_esrs_catalog(self) -> List[Dict[str, Any]]:
        """Load ESRS data points catalog."""
        try:
            with open(self.esrs_data_points_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and "data_points" in data:
                catalog = data["data_points"]
            else:
                catalog = data if isinstance(data, list) else []

            logger.info(f"Loaded {len(catalog)} ESRS data points")
            return catalog
        except Exception as e:
            logger.error(f"Failed to load ESRS catalog: {e}")
            return []

    def _get_esrs_topics(self) -> List[Dict[str, str]]:
        """Get list of ESRS topical standards to assess."""
        return [
            {"id": "E1", "name": "Climate Change", "description": "Climate change mitigation and adaptation"},
            {"id": "E2", "name": "Pollution", "description": "Pollution of air, water and soil"},
            {"id": "E3", "name": "Water and Marine Resources", "description": "Water and marine resources"},
            {"id": "E4", "name": "Biodiversity and Ecosystems", "description": "Biodiversity and ecosystems"},
            {"id": "E5", "name": "Resource Use and Circular Economy", "description": "Resource use and circular economy"},
            {"id": "S1", "name": "Own Workforce", "description": "Own workforce"},
            {"id": "S2", "name": "Workers in the Value Chain", "description": "Workers in the value chain"},
            {"id": "S3", "name": "Affected Communities", "description": "Affected communities"},
            {"id": "S4", "name": "Consumers and End-Users", "description": "Consumers and end-users"},
            {"id": "G1", "name": "Business Conduct", "description": "Business conduct"}
        ]

    # ========================================================================
    # IMPACT MATERIALITY ASSESSMENT (AI-POWERED)
    # ========================================================================

    def assess_impact_materiality(
        self,
        topic: Dict[str, str],
        company_context: Dict[str, Any],
        esg_data: Optional[Dict[str, Any]] = None
    ) -> ImpactMaterialityScore:
        """
        Assess impact materiality using AI/LLM.

        Args:
            topic: ESRS topic to assess
            company_context: Company profile and context
            esg_data: ESG data for context (optional)

        Returns:
            ImpactMaterialityScore with AI-generated assessment
        """
        logger.info(f"Assessing impact materiality for {topic['name']} using AI")

        # Extract company info
        company_name = company_context.get("company_info", {}).get("legal_name", "the company")
        sector = company_context.get("business_profile", {}).get("sector", "")
        business_model = company_context.get("business_profile", {}).get("business_model", "")

        # Build system prompt
        system_prompt = f"""You are an expert sustainability analyst conducting double materiality assessments per ESRS 1.

Assess the severity, scope, and irremediability of {topic['name']} impacts for {company_name} in the {sector} sector.

Use the company context and ESG data to make informed assessments. Provide scores on a 0-10 scale with clear rationale.

Respond in JSON format with this structure:
{{
    "severity": <0-10 score>,
    "scope": <0-10 score>,
    "irremediability": <0-10 score>,
    "rationale": "<100-200 word explanation>",
    "impact_type": ["actual_negative", "potential_negative", "actual_positive", "potential_positive"],
    "affected_stakeholders": ["employees", "communities", "environment", etc.],
    "time_horizon": "short_term|medium_term|long_term",
    "value_chain_stage": ["upstream", "own_operations", "downstream"]
}}"""

        # Build user prompt
        user_prompt = f"""Topic: {topic['name']}
Description: {topic['description']}
Company: {company_name}
Sector: {sector}
Business Model: {business_model}

Assess impact materiality considering:
1. Actual and potential impacts (both negative and positive)
2. Severity: How severe is the impact? (scale, intensity, nature)
3. Scope: How widespread is the impact? (number affected, geographic extent, duration)
4. Irremediability: How difficult to remediate? (reversibility, time, cost)
5. Short, medium, and long-term horizons
6. Value chain stages (upstream, own operations, downstream)

Provide your assessment in JSON format."""

        # Call LLM
        response_text, confidence = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json"
        )

        self.stats["llm_api_calls"] += 1

        # Parse response
        if response_text:
            try:
                data = json.loads(response_text)

                severity = float(data.get("severity", 5.0))
                scope = float(data.get("scope", 5.0))
                irremediability = float(data.get("irremediability", 5.0))

                # Calculate impact score: (severity × scope × irremediability) / 100
                impact_score = (severity * scope * irremediability) / 100.0
                impact_score = min(10.0, impact_score)  # Cap at 10

                is_material = impact_score >= self.impact_threshold

                return ImpactMaterialityScore(
                    severity=severity,
                    scope=scope,
                    irremediability=irremediability,
                    score=impact_score,
                    is_material=is_material,
                    rationale=data.get("rationale", ""),
                    confidence=confidence,
                    impact_type=data.get("impact_type", []),
                    affected_stakeholders=data.get("affected_stakeholders", []),
                    time_horizon=data.get("time_horizon", "medium_term"),
                    value_chain_stage=data.get("value_chain_stage", [])
                )
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")

        # Fallback: return default assessment with low confidence
        logger.warning(f"Using fallback impact assessment for {topic['name']}")
        self._flag_for_review(topic['id'], "impact_assessment_failed", "LLM assessment failed, using fallback")

        return ImpactMaterialityScore(
            severity=5.0,
            scope=5.0,
            irremediability=5.0,
            score=2.5,
            is_material=False,
            rationale="Assessment failed - requires manual review",
            confidence=0.0,
            impact_type=["unknown"],
            affected_stakeholders=[],
            time_horizon="medium_term",
            value_chain_stage=[]
        )

    # ========================================================================
    # FINANCIAL MATERIALITY ASSESSMENT (AI-POWERED)
    # ========================================================================

    def assess_financial_materiality(
        self,
        topic: Dict[str, str],
        company_context: Dict[str, Any],
        esg_data: Optional[Dict[str, Any]] = None
    ) -> FinancialMaterialityScore:
        """
        Assess financial materiality using AI/LLM.

        Args:
            topic: ESRS topic to assess
            company_context: Company profile and context
            esg_data: ESG data for context (optional)

        Returns:
            FinancialMaterialityScore with AI-generated assessment
        """
        logger.info(f"Assessing financial materiality for {topic['name']} using AI")

        # Extract financial info
        company_name = company_context.get("company_info", {}).get("legal_name", "the company")
        revenue = company_context.get("company_size", {}).get("revenue", {}).get("total_revenue", 0)
        total_assets = company_context.get("company_size", {}).get("total_assets", 0)

        # Build system prompt
        system_prompt = f"""You are a financial analyst assessing sustainability-related financial risks and opportunities per ESRS.

Evaluate the magnitude and likelihood of financial impacts from {topic['name']} for {company_name}.

Respond in JSON format with this structure:
{{
    "magnitude": <0-10 score>,
    "likelihood": <0-10 score>,
    "rationale": "<100-200 word financial explanation>",
    "effect_type": ["risk", "opportunity"],
    "financial_impact_areas": ["revenue", "costs", "assets", "liabilities", "capital", "access_to_finance"],
    "time_horizon": "short_term|medium_term|long_term"
}}"""

        # Build user prompt
        user_prompt = f"""Topic: {topic['name']}
Company: {company_name}
Revenue: {revenue:,.0f} EUR
Total Assets: {total_assets:,.0f} EUR

Assess financial materiality considering:
1. Magnitude: How large is the financial effect? (revenue impact, cost impact, asset/liability implications)
2. Likelihood: How likely is this to materialize? (probability, time horizon, external dependencies)
3. Both risks and opportunities
4. Effects on: revenue, costs, assets, liabilities, capital, access to finance
5. Short (0-1y), medium (1-5y), long-term (>5y) horizons

Provide your assessment in JSON format."""

        # Call LLM
        response_text, confidence = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json"
        )

        self.stats["llm_api_calls"] += 1

        # Parse response
        if response_text:
            try:
                data = json.loads(response_text)

                magnitude = float(data.get("magnitude", 5.0))
                likelihood = float(data.get("likelihood", 5.0))

                # Calculate financial score: (magnitude × likelihood) / 10
                financial_score = (magnitude * likelihood) / 10.0
                financial_score = min(10.0, financial_score)  # Cap at 10

                is_material = financial_score >= self.financial_threshold

                return FinancialMaterialityScore(
                    magnitude=magnitude,
                    likelihood=likelihood,
                    score=financial_score,
                    is_material=is_material,
                    rationale=data.get("rationale", ""),
                    confidence=confidence,
                    effect_type=data.get("effect_type", []),
                    financial_impact_areas=data.get("financial_impact_areas", []),
                    time_horizon=data.get("time_horizon", "medium_term")
                )
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")

        # Fallback
        logger.warning(f"Using fallback financial assessment for {topic['name']}")
        self._flag_for_review(topic['id'], "financial_assessment_failed", "LLM assessment failed, using fallback")

        return FinancialMaterialityScore(
            magnitude=5.0,
            likelihood=5.0,
            score=2.5,
            is_material=False,
            rationale="Assessment failed - requires manual review",
            confidence=0.0,
            effect_type=["unknown"],
            financial_impact_areas=[],
            time_horizon="medium_term"
        )

    # ========================================================================
    # STAKEHOLDER ANALYSIS (RAG-BASED)
    # ========================================================================

    def analyze_stakeholder_perspectives(
        self,
        topic: Dict[str, str],
        stakeholder_data: Optional[Dict[str, Any]] = None
    ) -> StakeholderPerspective:
        """
        Analyze stakeholder perspectives using RAG.

        Args:
            topic: ESRS topic
            stakeholder_data: Stakeholder consultation data (optional)

        Returns:
            StakeholderPerspective with synthesized views
        """
        logger.info(f"Analyzing stakeholder perspectives for {topic['name']}")

        # Retrieve relevant stakeholder input
        query = f"{topic['name']} stakeholder concerns consultation feedback"
        relevant_docs = self.rag_system.retrieve(query, top_k=5, filter_type="stakeholder_input")

        if not relevant_docs:
            return StakeholderPerspective(
                topic_id=topic['id'],
                stakeholder_groups=[],
                key_concerns=[],
                consensus_view=None,
                divergent_views=[],
                participants_count=0,
                confidence=0.0
            )

        # Use LLM to synthesize stakeholder perspectives
        system_prompt = f"""You are analyzing stakeholder consultation data for a materiality assessment.

Synthesize stakeholder perspectives on {topic['name']} from the provided consultation data.

Respond in JSON format:
{{
    "stakeholder_groups": ["employees", "investors", "customers", etc.],
    "key_concerns": ["concern 1", "concern 2", etc.],
    "consensus_view": "Brief summary of consensus (if any)",
    "divergent_views": ["Divergent view 1", etc.],
    "participants_count": <number>
}}"""

        # Build context from retrieved documents
        doc_context = "\n\n".join([
            f"Document {i+1}: {doc.get('content', '')}"
            for i, doc in enumerate(relevant_docs)
        ])

        user_prompt = f"""Stakeholder consultation data:\n\n{doc_context}\n\nSynthesize the stakeholder perspectives in JSON format."""

        response_text, confidence = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json"
        )

        self.stats["llm_api_calls"] += 1

        if response_text:
            try:
                data = json.loads(response_text)
                return StakeholderPerspective(
                    topic_id=topic['id'],
                    stakeholder_groups=data.get("stakeholder_groups", []),
                    key_concerns=data.get("key_concerns", []),
                    consensus_view=data.get("consensus_view"),
                    divergent_views=data.get("divergent_views", []),
                    participants_count=data.get("participants_count", len(relevant_docs)),
                    confidence=confidence
                )
            except Exception as e:
                logger.error(f"Failed to parse stakeholder analysis: {e}")

        return StakeholderPerspective(
            topic_id=topic['id'],
            stakeholder_groups=[],
            key_concerns=[],
            consensus_view=None,
            divergent_views=[],
            participants_count=0,
            confidence=0.0
        )

    # ========================================================================
    # DOUBLE MATERIALITY DETERMINATION
    # ========================================================================

    def determine_double_materiality(
        self,
        topic: Dict[str, str],
        impact_score: ImpactMaterialityScore,
        financial_score: FinancialMaterialityScore,
        methodology: MethodologyInfo
    ) -> MaterialityTopic:
        """
        Determine if topic is material (double materiality).

        Args:
            topic: ESRS topic
            impact_score: Impact materiality assessment
            financial_score: Financial materiality assessment
            methodology: Methodology configuration

        Returns:
            MaterialityTopic with determination
        """
        # Apply double materiality rule
        if methodology.double_materiality_rule == "either_or":
            is_material = impact_score.is_material or financial_score.is_material
        else:  # both_required
            is_material = impact_score.is_material and financial_score.is_material

        # Determine conclusion
        if is_material:
            materiality_conclusion = "material"
        else:
            # Check if borderline (within 1 point of threshold)
            impact_gap = abs(impact_score.score - methodology.impact_threshold)
            financial_gap = abs(financial_score.score - methodology.financial_threshold)

            if impact_gap <= 1.0 or financial_gap <= 1.0:
                materiality_conclusion = "borderline"
                self._flag_for_review(topic['id'], "borderline_case", f"Scores close to threshold")
            else:
                materiality_conclusion = "not_material"

        # Flag low confidence cases for review
        avg_confidence = (impact_score.confidence + financial_score.confidence) / 2
        if avg_confidence < 0.6:
            self._flag_for_review(topic['id'], "low_confidence", f"Average confidence: {avg_confidence:.2f}")

        # Update statistics
        self.stats["topics_assessed"] += 1
        if is_material:
            self.stats["material_topics"] += 1
        if impact_score.is_material:
            self.stats["impact_material"] += 1
        if financial_score.is_material:
            self.stats["financial_material"] += 1
        if impact_score.is_material and financial_score.is_material:
            self.stats["double_material"] += 1

        self.stats["total_confidence"] += avg_confidence

        # Get disclosure requirements
        disclosure_requirements = self._get_disclosure_requirements(topic['id'], is_material)

        return MaterialityTopic(
            topic_id=topic['id'],
            topic_name=topic['name'],
            esrs_standard=topic['id'],
            topic_description=topic['description'],
            impact_materiality=impact_score,
            financial_materiality=financial_score,
            double_material=is_material,
            materiality_conclusion=materiality_conclusion,
            disclosure_requirements=disclosure_requirements,
            data_sources=["AI assessment", "Company context", "Stakeholder input"],
            review_status="pending_review"
        )

    def _get_disclosure_requirements(self, standard: str, is_material: bool) -> List[str]:
        """Get ESRS disclosure requirements for a standard."""
        if not is_material:
            return []

        # Simplified - real implementation would query ESRS database
        requirements = {
            "E1": ["E1-1 Transition plan", "E1-6 GHG emissions", "E1-9 GHG intensity"],
            "E2": ["E2-4 Pollution metrics", "E2-5 Substances of concern"],
            "E3": ["E3-4 Water consumption", "E3-5 Water intensity"],
            "E4": ["E4-5 Biodiversity impacts", "E4-6 Ecosystem condition"],
            "E5": ["E5-5 Resource inflows", "E5-6 Resource outflows"],
            "S1": ["S1-6 Workforce characteristics", "S1-14 Health and safety"],
            "S2": ["S2-4 Value chain workers", "S2-5 Collective bargaining"],
            "S3": ["S3-4 Affected communities", "S3-5 Impacts on communities"],
            "S4": ["S4-4 Consumer impacts", "S4-5 Data privacy"],
            "G1": ["G1-4 Business conduct", "G1-5 Corruption and bribery"]
        }

        return requirements.get(standard, [])

    # ========================================================================
    # MATERIALITY MATRIX GENERATION
    # ========================================================================

    def generate_materiality_matrix(
        self,
        material_topics: List[MaterialityTopic]
    ) -> MaterialityMatrix:
        """
        Generate materiality matrix visualization data.

        Args:
            material_topics: List of assessed topics

        Returns:
            MaterialityMatrix with chart data and quadrants
        """
        chart_data = []
        quadrants = {
            "high_impact_high_financial": [],
            "high_impact_low_financial": [],
            "low_impact_high_financial": [],
            "low_impact_low_financial": []
        }

        for topic in material_topics:
            # Chart data point
            chart_data.append({
                "topic_id": topic.topic_id,
                "x_financial": round(topic.financial_materiality.score, 2),
                "y_impact": round(topic.impact_materiality.score, 2),
                "label": topic.topic_name
            })

            # Assign to quadrant
            impact_high = topic.impact_materiality.score >= 5.0
            financial_high = topic.financial_materiality.score >= 5.0

            if impact_high and financial_high:
                quadrants["high_impact_high_financial"].append(topic.topic_id)
            elif impact_high and not financial_high:
                quadrants["high_impact_low_financial"].append(topic.topic_id)
            elif not impact_high and financial_high:
                quadrants["low_impact_high_financial"].append(topic.topic_id)
            else:
                quadrants["low_impact_low_financial"].append(topic.topic_id)

        return MaterialityMatrix(
            chart_data=chart_data,
            quadrants=quadrants
        )

    # ========================================================================
    # REVIEW MANAGEMENT
    # ========================================================================

    def _flag_for_review(self, topic_id: str, flag_type: str, reason: str) -> None:
        """Flag topic for human review."""
        self.review_flags.append({
            "topic_id": topic_id,
            "flag_type": flag_type,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    def process(
        self,
        company_context: Dict[str, Any],
        esg_data: Optional[Dict[str, Any]] = None,
        stakeholder_data: Optional[Dict[str, Any]] = None,
        methodology: Optional[MethodologyInfo] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Conduct double materiality assessment.

        Args:
            company_context: Company profile and context
            esg_data: ESG data for context (optional)
            stakeholder_data: Stakeholder consultation data (optional)
            methodology: Methodology configuration (optional)
            output_file: Output file path (optional)

        Returns:
            Complete materiality assessment result
        """
        self.stats["start_time"] = datetime.now()

        methodology = methodology or MethodologyInfo(
            impact_threshold=self.impact_threshold,
            financial_threshold=self.financial_threshold
        )

        logger.info(f"Starting double materiality assessment for {len(self.esrs_topics)} topics")

        # Assess each topic
        material_topics: List[MaterialityTopic] = []
        stakeholder_perspectives: List[StakeholderPerspective] = []

        for topic in self.esrs_topics:
            logger.info(f"Assessing {topic['name']}...")

            # Impact materiality
            impact_score = self.assess_impact_materiality(topic, company_context, esg_data)

            # Financial materiality
            financial_score = self.assess_financial_materiality(topic, company_context, esg_data)

            # Stakeholder analysis
            stakeholder_view = self.analyze_stakeholder_perspectives(topic, stakeholder_data)
            stakeholder_perspectives.append(stakeholder_view)

            # Double materiality determination
            material_topic = self.determine_double_materiality(
                topic, impact_score, financial_score, methodology
            )

            material_topics.append(material_topic)

        # Generate materiality matrix
        materiality_matrix = self.generate_materiality_matrix(material_topics)

        # Calculate statistics
        self.stats["end_time"] = datetime.now()
        processing_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        avg_confidence = self.stats["total_confidence"] / self.stats["topics_assessed"] if self.stats["topics_assessed"] > 0 else 0.0

        # Build metadata
        metadata = AssessmentMetadata(
            assessment_id=str(uuid.uuid4()),
            assessment_date=self.stats["end_time"].isoformat(),
            reporting_period={
                "year": company_context.get("reporting_scope", {}).get("reporting_year", 2024)
            },
            company_id=company_context.get("company_info", {}).get("company_id", "unknown"),
            status="draft",
            ai_powered=True,
            requires_human_review=True
        )

        # Build result
        result = {
            "assessment_metadata": metadata.dict(),
            "methodology": methodology.dict(),
            "material_topics": [t.dict() for t in material_topics],
            "stakeholder_perspectives": [s.dict() for s in stakeholder_perspectives],
            "materiality_matrix": materiality_matrix.dict(),
            "summary_statistics": {
                "total_topics_assessed": self.stats["topics_assessed"],
                "material_topics_count": self.stats["material_topics"],
                "material_from_impact": self.stats["impact_material"],
                "material_from_financial": self.stats["financial_material"],
                "double_material_count": self.stats["double_material"],
                "esrs_standards_triggered": [t.esrs_standard for t in material_topics if t.double_material]
            },
            "review_flags": self.review_flags,
            "ai_metadata": {
                "llm_provider": self.llm_config.provider,
                "llm_model": self.llm_config.model,
                "total_llm_calls": self.stats["llm_api_calls"],
                "average_confidence": round(avg_confidence, 2),
                "processing_time_seconds": round(processing_time, 2),
                "processing_time_minutes": round(processing_time / 60, 1),
                "deterministic": False,
                "zero_hallucination": False,
                "requires_human_review": True
            },
            "limitations_assumptions": [
                "AI assessments are preliminary and require expert validation",
                "Materiality is company-specific - AI cannot replace human judgment",
                "Stakeholder analysis may miss nuanced perspectives",
                "Financial impact estimates are directional, not precise",
                "Topic assessments may evolve as business context changes"
            ],
            "next_review_date": None
        }

        # Write output if path provided
        if output_file:
            self.write_output(result, output_file)

        logger.info(f"Assessment complete: {self.stats['material_topics']}/{self.stats['topics_assessed']} topics material")
        logger.info(f"Processing time: {processing_time/60:.1f} minutes ({self.stats['llm_api_calls']} LLM calls)")
        logger.info(f"Average confidence: {avg_confidence:.2%}")
        logger.warning(f"⚠️ HUMAN REVIEW REQUIRED: {len(self.review_flags)} items flagged for review")

        return result

    def write_output(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write result to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Wrote materiality assessment to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSRD Double Materiality Assessment Agent (AI-Powered)")
    parser.add_argument("--esrs-catalog", required=True, help="Path to ESRS data points catalog JSON")
    parser.add_argument("--company-context", required=True, help="Path to company context JSON")
    parser.add_argument("--esg-data", help="Path to ESG data JSON (optional)")
    parser.add_argument("--stakeholder-data", help="Path to stakeholder consultation JSON (optional)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "anthropic"], help="LLM provider")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--impact-threshold", type=float, default=5.0, help="Impact materiality threshold (0-10)")
    parser.add_argument("--financial-threshold", type=float, default=5.0, help="Financial materiality threshold (0-10)")

    args = parser.parse_args()

    # Load company context
    with open(args.company_context, 'r', encoding='utf-8') as f:
        company_context = json.load(f)

    # Load ESG data if provided
    esg_data = None
    if args.esg_data:
        with open(args.esg_data, 'r', encoding='utf-8') as f:
            esg_data = json.load(f)

    # Load stakeholder data if provided
    stakeholder_data = None
    if args.stakeholder_data:
        with open(args.stakeholder_data, 'r', encoding='utf-8') as f:
            stakeholder_data = json.load(f)

    # Create LLM config
    llm_config = LLMConfig(
        provider=args.llm_provider,
        model=args.llm_model
    )

    # Create agent
    agent = MaterialityAgent(
        esrs_data_points_path=args.esrs_catalog,
        llm_config=llm_config,
        impact_threshold=args.impact_threshold,
        financial_threshold=args.financial_threshold
    )

    # Process
    result = agent.process(
        company_context=company_context,
        esg_data=esg_data,
        stakeholder_data=stakeholder_data,
        output_file=args.output
    )

    # Print summary
    print("\n" + "="*80)
    print("DOUBLE MATERIALITY ASSESSMENT SUMMARY (AI-POWERED)")
    print("="*80)
    print(f"⚠️ HUMAN REVIEW REQUIRED - AI-generated assessments")
    print("="*80)

    summary = result["summary_statistics"]
    ai_meta = result["ai_metadata"]

    print(f"\nMateriality Results:")
    print(f"  Total Topics Assessed: {summary['total_topics_assessed']}")
    print(f"  Material Topics: {summary['material_topics_count']}")
    print(f"  - Material from Impact: {summary['material_from_impact']}")
    print(f"  - Material from Financial: {summary['material_from_financial']}")
    print(f"  - Double Material: {summary['double_material_count']}")

    print(f"\nAI Processing Metrics:")
    print(f"  LLM Provider: {ai_meta['llm_provider']}")
    print(f"  LLM Model: {ai_meta['llm_model']}")
    print(f"  Total LLM API Calls: {ai_meta['total_llm_calls']}")
    print(f"  Average Confidence: {ai_meta['average_confidence']:.0%}")
    print(f"  Processing Time: {ai_meta['processing_time_minutes']:.1f} minutes")

    print(f"\nReview Requirements:")
    print(f"  Items Flagged for Review: {len(result['review_flags'])}")
    print(f"  Human Review Required: YES ⚠️")
    print(f"  Zero-Hallucination Guarantee: NO ⚠️")

    if result['review_flags']:
        print(f"\nReview Flags (first 5):")
        for flag in result['review_flags'][:5]:
            print(f"  - {flag['topic_id']}: {flag['flag_type']} - {flag['reason']}")

    print(f"\nMaterial ESRS Standards:")
    for standard in summary['esrs_standards_triggered']:
        print(f"  - {standard}")

    print("\n" + "="*80)
    print("⚠️ MANDATORY NEXT STEP: Human review and approval by qualified professionals")
    print("="*80)
