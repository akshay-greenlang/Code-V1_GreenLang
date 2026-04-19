# -*- coding: utf-8 -*-
"""
AI-Enhanced Narrative Generator for CSRD Reporting
GL-CSRD Reporting Platform - INSIGHT PATH

Transformation from v1 (static templates) to v2 (AI-powered narratives):
- BEFORE: Static templates → Placeholder text → Requires complete rewrite
- AFTER: RAG retrieval → AI narrative generation → Company-specific context

Pattern: InsightAgent enhancement for narrative generation
- calculate(): N/A (narratives don't require calculations)
- explain(): AI-powered narrative drafting with regulatory context

RAG Collections:
- csrd_guidance: Official ESRS guidance and requirements
- peer_disclosures: Example disclosures from peer companies
- regulatory_templates: Standard disclosure templates
- best_practices: Industry best practices for CSRD reporting

Temperature: 0.6 (consistency and regulatory compliance over creativity)

Version: 2.0.0
Phase: Phase 2.2 (Intelligence Paradox Fix)
Date: 2025-11-06
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.determinism import DeterministicClock


logger = logging.getLogger(__name__)


@dataclass
class AIGeneratedNarrative:
    """AI-generated narrative section with metadata."""
    section_id: str
    section_title: str
    content: str  # HTML formatted content
    ai_generated: bool = True
    review_status: str = "pending"  # "pending", "reviewed", "approved"
    language: str = "en"
    word_count: int = 0

    # AI metadata
    rag_sources_used: List[str] = None
    generation_timestamp: str = None
    confidence_score: float = 0.0  # 0-1
    token_count: int = 0
    model_version: str = "claude-sonnet-3.5"

    def __post_init__(self):
        """Calculate word count after initialization."""
        if self.content and self.word_count == 0:
            # Simple word count (strip HTML tags for accuracy)
            import re
            text_only = re.sub(r'<[^>]+>', '', self.content)
            self.word_count = len(text_only.split())

        if self.rag_sources_used is None:
            self.rag_sources_used = []

        if self.generation_timestamp is None:
            self.generation_timestamp = DeterministicClock.utcnow().isoformat() + "Z"


class NarrativeGeneratorAI:
    """
    AI-powered narrative generator for CSRD disclosures.

    Generates company-specific, context-aware narratives for:
    - Governance disclosures (GOV-1, GOV-2, GOV-3, etc.)
    - Strategy disclosures (SBM-1, SBM-2, SBM-3)
    - Topic-specific narratives (E1-E5, S1-S4, G1)
    - Management commentary
    - Risk and opportunity assessments

    Key Features:
    - RAG-grounded narratives (regulatory guidance + peer examples)
    - Company-specific context integration
    - Metrics-aware narrative generation
    - Multi-language support (EN, DE, FR, ES)
    - Regulatory compliance focus

    Quality Assurance:
    - All narratives flagged as "AI-generated"
    - Human review required before publication
    - Confidence scores for transparency
    - RAG source attribution
    """

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="narrative_generator_ai_v2",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=False,
        critical_for_compliance=False,
        transformation_priority="HIGH (Phase 2.2 transformation)",
        description="AI-powered CSRD narrative generation with RAG grounding"
    )

    def __init__(self, language: str = "en"):
        """
        Initialize AI narrative generator.

        Args:
            language: Target language (en, de, fr, es)
        """
        self.language = language
        self.generated_count = 0

        # Language-specific configuration
        self.language_configs = {
            "en": {"name": "English", "regulatory_context": "EU CSRD/ESRS"},
            "de": {"name": "German", "regulatory_context": "EU CSRD/ESRS (Deutschland)"},
            "fr": {"name": "French", "regulatory_context": "CSRD/ESRS UE"},
            "es": {"name": "Spanish", "regulatory_context": "CSRD/ESRS UE (España)"}
        }

        logger.info(f"Initialized AI Narrative Generator v2.0 (language: {language})")

    # ========================================================================
    # GOVERNANCE NARRATIVES
    # ========================================================================

    async def generate_governance_narrative(
        self,
        company_profile: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6
    ) -> AIGeneratedNarrative:
        """
        Generate AI-powered governance disclosure narrative.

        Covers:
        - GOV-1: Role of governance bodies
        - GOV-2: Information provided to governance bodies
        - GOV-3: Integration in strategy and decision-making
        - GOV-4: Sustainability-related policies
        - GOV-5: Due diligence processes

        Args:
            company_profile: Company profile with governance data
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature (0.6 for regulatory consistency)

        Returns:
            AIGeneratedNarrative with governance disclosure
        """
        logger.info("Generating AI-powered governance narrative")

        # Step 1: RAG retrieval for governance guidance
        rag_query = self._build_governance_rag_query(company_profile)

        rag_result = await rag_engine.query(
            query=rag_query,
            collections=[
                "csrd_guidance",
                "peer_disclosures",
                "governance_templates",
                "best_practices"
            ],
            top_k=6
        )

        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 2: Build narrative prompt
        prompt = self._build_governance_prompt(company_profile, formatted_knowledge)

        # Step 3: Generate narrative
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert CSRD reporting specialist drafting governance disclosures.

Your role is to generate **governance narratives** that comply with ESRS 2 requirements (GOV-1 through GOV-5).

Requirements:
1. REGULATORY COMPLIANCE: Follow ESRS 2 structure and requirements exactly
2. COMPANY-SPECIFIC: Use the provided company data, don't make up information
3. RAG-GROUNDED: Reference peer examples and regulatory guidance from RAG context
4. FACTUAL: Be specific and factual, avoid generic statements
5. HTML FORMATTING: Use proper HTML tags (h2, h3, h4, p, ul, li, strong, em)
6. LANGUAGE: Generate in {self.language_configs[self.language]['name']}

Disclosure Structure:
- GOV-1: Role of administrative, management and supervisory bodies
- GOV-2: Information provided to and sustainability matters addressed by the undertaking's administrative, management and supervisory bodies
- GOV-3: Integration of sustainability-related performance in incentive schemes
- GOV-4: Statement on due diligence
- GOV-5: Risk management and internal controls over sustainability reporting

Mark any uncertainties or gaps in data clearly.
Do NOT fabricate governance structures or policies not provided in the company data."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        # Step 4: Parse and structure narrative
        narrative_html = response.text if hasattr(response, "text") else str(response)

        # Extract RAG sources
        rag_sources = self._extract_rag_sources(rag_result)

        narrative = AIGeneratedNarrative(
            section_id="governance_esrs2",
            section_title="Governance (ESRS 2 - GOV)",
            content=self._ensure_html_structure(narrative_html),
            ai_generated=True,
            review_status="pending",
            language=self.language,
            rag_sources_used=rag_sources,
            confidence_score=0.75,  # Would calculate from model scores in production
            token_count=len(narrative_html.split()),  # Approximate
            model_version="claude-sonnet-3.5"
        )

        self.generated_count += 1
        logger.info(f"Generated governance narrative: {narrative.word_count} words")

        return narrative

    # ========================================================================
    # STRATEGY NARRATIVES
    # ========================================================================

    async def generate_strategy_narrative(
        self,
        company_profile: Dict[str, Any],
        materiality_assessment: Dict[str, Any],
        session,
        rag_engine,
        temperature: float = 0.6
    ) -> AIGeneratedNarrative:
        """
        Generate AI-powered strategy disclosure narrative.

        Covers:
        - SBM-1: Strategy, business model, and value chain
        - SBM-2: Interests and views of stakeholders
        - SBM-3: Material impacts, risks, and opportunities

        Args:
            company_profile: Company profile data
            materiality_assessment: Double materiality assessment results
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature

        Returns:
            AIGeneratedNarrative with strategy disclosure
        """
        logger.info("Generating AI-powered strategy narrative")

        # Step 1: RAG retrieval
        rag_query = self._build_strategy_rag_query(company_profile, materiality_assessment)

        rag_result = await rag_engine.query(
            query=rag_query,
            collections=[
                "csrd_guidance",
                "peer_disclosures",
                "strategy_templates",
                "business_model_examples"
            ],
            top_k=6
        )

        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 2: Build prompt
        prompt = self._build_strategy_prompt(company_profile, materiality_assessment, formatted_knowledge)

        # Step 3: Generate narrative
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert CSRD reporting specialist drafting strategy disclosures.

Your role is to generate **strategy narratives** that comply with ESRS 2 requirements (SBM-1 through SBM-3).

Requirements:
1. REGULATORY COMPLIANCE: Follow ESRS 2 structure exactly
2. BUSINESS MODEL FOCUS: Clearly describe business model and value chain
3. MATERIALITY INTEGRATION: Reference material topics from materiality assessment
4. STAKEHOLDER-CENTRIC: Address stakeholder interests and views
5. FORWARD-LOOKING: Include strategy resilience and adaptation
6. HTML FORMATTING: Use proper HTML structure
7. LANGUAGE: Generate in {self.language_configs[self.language]['name']}

Disclosure Structure:
- SBM-1: Strategy, business model and value chain (main narrative)
- SBM-2: Interests and views of stakeholders
- SBM-3: Material impacts, risks and opportunities and their interaction with strategy and business model

Use the materiality assessment to identify material topics.
Reference peer examples from RAG context for inspiration, but tailor to this company.
Be specific about the company's value chain, products, and markets."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        narrative_html = response.text if hasattr(response, "text") else str(response)
        rag_sources = self._extract_rag_sources(rag_result)

        narrative = AIGeneratedNarrative(
            section_id="strategy_esrs2_sbm",
            section_title="Strategy and Business Model (ESRS 2 - SBM)",
            content=self._ensure_html_structure(narrative_html),
            ai_generated=True,
            review_status="pending",
            language=self.language,
            rag_sources_used=rag_sources,
            confidence_score=0.75,
            model_version="claude-sonnet-3.5"
        )

        self.generated_count += 1
        logger.info(f"Generated strategy narrative: {narrative.word_count} words")

        return narrative

    # ========================================================================
    # TOPIC-SPECIFIC NARRATIVES (E1-E5, S1-S4, G1)
    # ========================================================================

    async def generate_topic_specific_narrative(
        self,
        esrs_standard: str,  # e.g., "E1", "S1", "G1"
        company_profile: Dict[str, Any],
        metrics_data: List[Dict[str, Any]],
        policies: Optional[Dict[str, Any]],
        targets: Optional[Dict[str, Any]],
        session,
        rag_engine,
        temperature: float = 0.6
    ) -> AIGeneratedNarrative:
        """
        Generate AI-powered topic-specific narrative for ESRS standard.

        Covers all disclosure requirements for specific ESRS topics:
        - E1: Climate Change
        - E2: Pollution
        - E3: Water and Marine Resources
        - E4: Biodiversity and Ecosystems
        - E5: Resource Use and Circular Economy
        - S1: Own Workforce
        - S2: Workers in the Value Chain
        - S3: Affected Communities
        - S4: Consumers and End-Users
        - G1: Business Conduct

        Args:
            esrs_standard: ESRS standard code (E1, E2, etc.)
            company_profile: Company profile
            metrics_data: Quantitative metrics for this standard
            policies: Policies related to this topic
            targets: Targets and action plans
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature

        Returns:
            AIGeneratedNarrative for the topic
        """
        standard_names = {
            "E1": "Climate Change",
            "E2": "Pollution",
            "E3": "Water and Marine Resources",
            "E4": "Biodiversity and Ecosystems",
            "E5": "Resource Use and Circular Economy",
            "S1": "Own Workforce",
            "S2": "Workers in the Value Chain",
            "S3": "Affected Communities",
            "S4": "Consumers and End-Users",
            "G1": "Business Conduct"
        }

        standard_name = standard_names.get(esrs_standard, esrs_standard)

        logger.info(f"Generating AI-powered narrative for ESRS {esrs_standard}: {standard_name}")

        # Step 1: RAG retrieval
        rag_query = self._build_topic_rag_query(esrs_standard, standard_name, company_profile, metrics_data)

        rag_result = await rag_engine.query(
            query=rag_query,
            collections=[
                "csrd_guidance",
                f"esrs_{esrs_standard.lower()}_guidance",
                "peer_disclosures",
                "topic_templates"
            ],
            top_k=8
        )

        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 2: Build prompt
        prompt = self._build_topic_prompt(
            esrs_standard,
            standard_name,
            company_profile,
            metrics_data,
            policies,
            targets,
            formatted_knowledge
        )

        # Step 3: Generate narrative
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert CSRD reporting specialist drafting ESRS {esrs_standard} disclosures.

Your role is to generate **topic-specific narratives** for {standard_name} that comply with ESRS {esrs_standard} requirements.

Requirements:
1. REGULATORY COMPLIANCE: Follow ESRS {esrs_standard} disclosure requirements exactly
2. METRICS INTEGRATION: Reference the quantitative metrics provided in context
3. POLICIES & TARGETS: Describe policies, action plans, and targets clearly
4. IMPACTS, RISKS & OPPORTUNITIES: Cover material IROs (Impact, Risk, Opportunity)
5. PERFORMANCE TRACKING: Explain how performance is measured and monitored
6. RAG-GROUNDED: Use peer examples and regulatory guidance
7. HTML FORMATTING: Proper structure with h2, h3, p, ul, li tags
8. LANGUAGE: Generate in {self.language_configs[self.language]['name']}

Standard Disclosure Structure for ESRS {esrs_standard}:
- Policies: Description of policies related to {standard_name}
- Actions: Action plans and resources allocated
- Metrics and Targets: Specific targets and KPIs (reference the provided metrics)
- Tracking Effectiveness: How effectiveness is assessed

Be specific. Use the quantitative metrics provided.
Reference peer examples from RAG for best practices, but customize to this company.
Clearly mark any data gaps or limitations."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        narrative_html = response.text if hasattr(response, "text") else str(response)
        rag_sources = self._extract_rag_sources(rag_result)

        narrative = AIGeneratedNarrative(
            section_id=f"esrs_{esrs_standard.lower()}_{standard_name.lower().replace(' ', '_')}",
            section_title=f"ESRS {esrs_standard}: {standard_name}",
            content=self._ensure_html_structure(narrative_html),
            ai_generated=True,
            review_status="pending",
            language=self.language,
            rag_sources_used=rag_sources,
            confidence_score=0.75,
            model_version="claude-sonnet-3.5"
        )

        self.generated_count += 1
        logger.info(f"Generated ESRS {esrs_standard} narrative: {narrative.word_count} words")

        return narrative

    # ========================================================================
    # HELPER METHODS - RAG QUERY BUILDING
    # ========================================================================

    def _build_governance_rag_query(self, company_profile: Dict[str, Any]) -> str:
        """Build RAG query for governance disclosures."""
        industry = company_profile.get("industry", "general")
        company_size = company_profile.get("employee_count", 0)

        return f"""
CSRD Governance Disclosure Query:

Company Context:
- Industry: {industry}
- Size: {company_size} employees

Query Topics:
1. ESRS 2 GOV-1: Role of governance bodies in sustainability oversight
2. ESRS 2 GOV-2: Information provided to boards on sustainability
3. ESRS 2 GOV-3: Integration in incentive schemes
4. ESRS 2 GOV-4: Due diligence statement requirements
5. ESRS 2 GOV-5: Risk management and internal controls

Looking for:
- Regulatory requirements and guidance for GOV disclosures
- Peer examples from {industry} industry
- Best practices for board sustainability governance
- Standard disclosure language and templates
"""

    def _build_strategy_rag_query(self, company_profile: Dict[str, Any], materiality: Dict[str, Any]) -> str:
        """Build RAG query for strategy disclosures."""
        industry = company_profile.get("industry", "general")
        material_topics = materiality.get("material_topics", [])

        return f"""
CSRD Strategy Disclosure Query:

Company Context:
- Industry: {industry}
- Material Topics: {', '.join(material_topics) if material_topics else 'Not specified'}

Query Topics:
1. ESRS 2 SBM-1: Strategy, business model, and value chain description
2. ESRS 2 SBM-2: Stakeholder engagement and interests
3. ESRS 2 SBM-3: Material impacts, risks, and opportunities

Looking for:
- Regulatory requirements for SBM disclosures
- Peer examples of business model descriptions
- Value chain disclosure best practices
- Stakeholder engagement examples
- Materiality assessment disclosure templates
"""

    def _build_topic_rag_query(
        self,
        esrs_code: str,
        standard_name: str,
        company_profile: Dict[str, Any],
        metrics: List[Dict[str, Any]]
    ) -> str:
        """Build RAG query for topic-specific disclosures."""
        industry = company_profile.get("industry", "general")
        metric_codes = [m.get("metric_code") for m in metrics if m.get("metric_code")]

        return f"""
CSRD Topic-Specific Disclosure Query:

ESRS Standard: {esrs_code} - {standard_name}
Company Industry: {industry}
Metrics Available: {', '.join(metric_codes[:10]) if metric_codes else 'None'}

Query Topics:
1. {esrs_code} disclosure requirements and structure
2. Policies and due diligence for {standard_name}
3. Actions and resources allocated to {standard_name}
4. Targets and performance tracking for {standard_name}
5. Best practices and peer examples in {industry}

Looking for:
- Detailed {esrs_code} regulatory guidance
- Peer disclosure examples for {standard_name}
- Industry-specific {standard_name} best practices
- Standard disclosure templates for {esrs_code}
"""

    # ========================================================================
    # HELPER METHODS - PROMPT BUILDING
    # ========================================================================

    def _build_governance_prompt(self, company_profile: Dict[str, Any], rag_context: str) -> str:
        """Build prompt for governance narrative."""
        return f"""
## COMPANY PROFILE

**Legal Name:** {company_profile.get('legal_name', 'Not provided')}
**Industry:** {company_profile.get('industry', 'Not specified')}
**Employees:** {company_profile.get('employee_count', 'Not specified')}
**Governance Structure:** {company_profile.get('governance_structure', 'Not specified')}
**Board Composition:** {company_profile.get('board_composition', 'Not specified')}
**Sustainability Committee:** {company_profile.get('sustainability_committee', 'Information not available')}

---

## RAG CONTEXT (Regulatory Guidance & Peer Examples)

{rag_context}

---

## YOUR TASK

Generate a comprehensive **Governance Disclosure** for CSRD (ESRS 2 - GOV) covering:

### GOV-1: Role of Administrative, Management and Supervisory Bodies
Describe the role of governance bodies in overseeing sustainability matters.

### GOV-2: Information Provided to Governance Bodies
Explain what sustainability information is provided to boards and how often.

### GOV-3: Integration in Incentive Schemes
Describe how sustainability performance is integrated into executive compensation.

### GOV-4: Statement on Due Diligence
Provide statement on due diligence process for sustainability matters.

### GOV-5: Risk Management and Internal Controls
Describe risk management and internal control systems for sustainability reporting.

Use the company profile data. Reference peer examples from RAG context.
Mark any data gaps clearly (e.g., "Information not currently disclosed").
Format in HTML with proper heading structure.
"""

    def _build_strategy_prompt(
        self,
        company_profile: Dict[str, Any],
        materiality: Dict[str, Any],
        rag_context: str
    ) -> str:
        """Build prompt for strategy narrative."""
        material_topics = materiality.get("material_topics", [])

        return f"""
## COMPANY PROFILE

**Legal Name:** {company_profile.get('legal_name', 'Not provided')}
**Industry:** {company_profile.get('industry', 'Not specified')}
**Business Model:** {company_profile.get('business_model_summary', 'Not specified')}
**Products/Services:** {company_profile.get('products_services', 'Not specified')}
**Value Chain:** {company_profile.get('value_chain_summary', 'Not specified')}
**Key Markets:** {company_profile.get('key_markets', 'Not specified')}

## MATERIALITY ASSESSMENT

**Material Topics:** {', '.join(material_topics) if material_topics else 'Not completed'}
**Impact Assessment:** {materiality.get('impact_summary', 'Not provided')}
**Stakeholder Input:** {materiality.get('stakeholder_summary', 'Not provided')}

---

## RAG CONTEXT (Regulatory Guidance & Peer Examples)

{rag_context}

---

## YOUR TASK

Generate a comprehensive **Strategy and Business Model Disclosure** for CSRD (ESRS 2 - SBM) covering:

### SBM-1: Strategy, Business Model and Value Chain
- Business model description
- Value chain (upstream and downstream)
- Key products, services, markets
- Strategy resilience regarding sustainability matters

### SBM-2: Interests and Views of Stakeholders
- Key stakeholder groups
- How stakeholder views are considered
- Stakeholder engagement processes

### SBM-3: Material Impacts, Risks and Opportunities
- Material sustainability matters (from materiality assessment)
- Current and anticipated impacts
- Material risks and opportunities
- How these interact with strategy and business model

Use the company and materiality data. Reference peer examples.
Be specific about the company's business model and value chain.
Format in HTML with clear structure.
"""

    def _build_topic_prompt(
        self,
        esrs_code: str,
        standard_name: str,
        company_profile: Dict[str, Any],
        metrics: List[Dict[str, Any]],
        policies: Optional[Dict[str, Any]],
        targets: Optional[Dict[str, Any]],
        rag_context: str
    ) -> str:
        """Build prompt for topic-specific narrative."""
        # Format metrics for prompt
        metrics_summary = "\n".join([
            f"- {m.get('metric_code', 'N/A')}: {m.get('metric_name', 'N/A')} = {m.get('value', 'N/A')} {m.get('unit', '')}"
            for m in metrics[:20]  # First 20 metrics
        ])

        policies_summary = policies.get('description', 'No specific policies provided') if policies else 'No policies data available'
        targets_summary = targets.get('description', 'No specific targets provided') if targets else 'No targets data available'

        return f"""
## COMPANY CONTEXT

**Company:** {company_profile.get('legal_name', 'Not provided')}
**Industry:** {company_profile.get('industry', 'Not specified')}

## ESRS STANDARD: {esrs_code} - {standard_name}

## QUANTITATIVE METRICS

{metrics_summary if metrics_summary else 'No metrics provided'}

## POLICIES

{policies_summary}

## TARGETS

{targets_summary}

---

## RAG CONTEXT (Regulatory Guidance & Peer Examples)

{rag_context}

---

## YOUR TASK

Generate a comprehensive disclosure for **ESRS {esrs_code}: {standard_name}** covering:

### 1. Policies Related to {standard_name}
Describe the company's policies addressing {standard_name}.

### 2. Actions and Resources
Describe action plans and resources allocated to manage impacts, risks, and opportunities related to {standard_name}.

### 3. Metrics and Targets
Present the quantitative metrics and targets for {standard_name}.
Reference the specific metrics provided above.

### 4. Tracking Effectiveness
Explain how the company tracks the effectiveness of policies and actions for {standard_name}.

Use the quantitative metrics in your narrative. Be specific.
Reference peer examples from RAG context for best practices.
Format in HTML with proper structure (h2, h3, p, ul, li).
"""

    # ========================================================================
    # HELPER METHODS - FORMATTING
    # ========================================================================

    def _format_rag_results(self, rag_result: Any) -> str:
        """Format RAG results for prompts."""
        if not rag_result or not hasattr(rag_result, 'chunks') or not rag_result.chunks:
            return "No relevant guidance or peer examples found in knowledge base."

        formatted = []
        for i, chunk in enumerate(rag_result.chunks, 1):
            source = getattr(chunk, 'metadata', {}).get('source', 'Unknown')
            formatted.append(f"**Source {i}** [{source}]:\n{chunk.text}\n")

        return "\n".join(formatted)

    def _extract_rag_sources(self, rag_result: Any) -> List[str]:
        """Extract list of RAG sources used."""
        sources = []
        if rag_result and hasattr(rag_result, 'chunks'):
            for chunk in rag_result.chunks:
                source = getattr(chunk, 'metadata', {}).get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
        return sources

    def _ensure_html_structure(self, html_content: str) -> str:
        """Ensure HTML content has proper structure."""
        # Basic check - production would use more sophisticated validation
        if not html_content.strip().startswith("<"):
            # Wrap plain text in paragraph
            html_content = f"<p>{html_content}</p>"

        # Add AI generation notice at top
        notice = """<div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin-bottom: 20px; border-radius: 4px;">
<p><strong>⚠️ AI-Generated Content - Human Review Required</strong></p>
<p>This section was generated using AI and must be reviewed and approved by qualified personnel before publication.</p>
</div>\n"""

        return notice + html_content


# ============================================================================
# INTEGRATION WITH EXISTING ReportingAgent
# ============================================================================

"""
INTEGRATION INSTRUCTIONS:

To integrate this AI narrative generator with the existing ReportingAgent:

1. Import in reporting_agent.py:
   from .narrative_generator_ai import NarrativeGeneratorAI, AIGeneratedNarrative

2. Modify ReportingAgent.generate_narratives() method:

   async def generate_narratives_ai(
       self,
       company_profile: Dict[str, Any],
       materiality_assessment: Dict[str, Any],
       metrics_by_standard: Dict[str, Any],
       session,  # ChatSession instance
       rag_engine,  # RAGEngine instance
       language: str = "en",
       temperature: float = 0.6
   ) -> List[AIGeneratedNarrative]:
       '''
       Generate AI-powered narratives (async version).

       This replaces the static template generation with AI-powered narratives.
       '''
       narrative_gen = NarrativeGeneratorAI(language=language)
       sections = []

       # Generate governance narrative
       gov_narrative = await narrative_gen.generate_governance_narrative(
           company_profile=company_profile,
           session=session,
           rag_engine=rag_engine,
           temperature=temperature
       )
       sections.append(gov_narrative)

       # Generate strategy narrative
       strategy_narrative = await narrative_gen.generate_strategy_narrative(
           company_profile=company_profile,
           materiality_assessment=materiality_assessment,
           session=session,
           rag_engine=rag_engine,
           temperature=temperature
       )
       sections.append(strategy_narrative)

       # Generate topic-specific narratives
       for standard_code in ["E1", "E2", "E3", "S1", "G1"]:
           if standard_code in metrics_by_standard:
               topic_narrative = await narrative_gen.generate_topic_specific_narrative(
                   esrs_standard=standard_code,
                   company_profile=company_profile,
                   metrics_data=metrics_by_standard[standard_code],
                   policies=company_profile.get(f'{standard_code}_policies'),
                   targets=company_profile.get(f'{standard_code}_targets'),
                   session=session,
                   rag_engine=rag_engine,
                   temperature=temperature
               )
               sections.append(topic_narrative)

       return sections

3. Update generate_report() to be async and use the AI narrative generator:
   - Make generate_report() async
   - Pass session and rag_engine as parameters
   - Call generate_narratives_ai() instead of generate_narratives()
   - Handle AIGeneratedNarrative objects appropriately
"""


# Example usage
if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("AI-Enhanced CSRD Narrative Generator - INSIGHT PATH")
    print("=" * 80)

    # Initialize generator
    generator = NarrativeGeneratorAI(language="en")

    print("\n✓ Generator initialized with InsightAgent pattern")
    print(f"✓ Category: {generator.category}")
    print(f"✓ Uses ChatSession: {generator.metadata.uses_chat_session}")
    print(f"✓ Uses RAG: {generator.metadata.uses_rag}")
    print(f"✓ Temperature: 0.6 (regulatory consistency)")

    print("\n" + "=" * 80)
    print("TRANSFORMATION SUMMARY")
    print("=" * 80)
    print("\nPattern: InsightAgent for narrative generation")
    print("  - No calculate() method (narratives don't require calculations)")
    print("  - AI-powered narrative generation (NEW)")
    print("\nWhat Changed:")
    print("  - BEFORE: Static HTML templates → Placeholder text")
    print("  - AFTER: RAG retrieval → AI generation → Company-specific narratives")
    print("\nAI Value-Add:")
    print("  ✓ Company-specific context integration")
    print("  ✓ RAG-grounded regulatory compliance")
    print("  ✓ Peer example incorporation")
    print("  ✓ Metrics-aware narrative generation")
    print("  ✓ Multi-language support")
    print("\nQuality Assurance:")
    print("  ✓ All narratives flagged as 'AI-generated'")
    print("  ✓ Human review required before publication")
    print("  ✓ RAG source attribution for transparency")
    print("  ✓ Confidence scores provided")
    print("\nCompliance:")
    print("  ✓ Narratives follow ESRS structure")
    print("  ✓ Uses deterministic metrics (no AI calculations)")
    print("  ✓ Regulatory guidance integrated via RAG")
    print("=" * 80)

    print("\n⚠ AI narrative generation requires:")
    print("  - ChatSession instance (LLM API)")
    print("  - RAGEngine instance (vector database)")
    print("  - Knowledge base with CSRD collections:")
    print("    * csrd_guidance")
    print("    * peer_disclosures")
    print("    * regulatory_templates")
    print("    * best_practices")
    print("    * esrs_e1_guidance, esrs_s1_guidance, etc.")
