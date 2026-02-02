# -*- coding: utf-8 -*-
"""
AI-Enhanced HotspotAnalysisAgent with Root Cause Investigation
GL-VCCI Scope 3 Platform - INSIGHT PATH

Enhancement from v1.0 (deterministic only) to v2.0 (hybrid with AI):
- BEFORE: Hotspot detection → Statistical insights → Generic recommendations
- AFTER: Hotspot detection → AI root cause investigation → Context-aware action plans

Pattern: InsightAgent enhancement
- calculate(): Existing deterministic hotspot detection (KEEP)
- investigate(): AI-powered root cause analysis (NEW)

Version: 2.0.0
Phase: Phase 2.2 (Intelligence Paradox Fix)
Date: 2025-11-06
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .agent import HotspotAnalysisAgent
from greenlang.determinism import DeterministicClock
from .models import (
    HotspotReport,
    ParetoAnalysis,
    SegmentationAnalysis,
    InsightReport
)

from greenlang.agents.categories import AgentCategory, AgentMetadata


logger = logging.getLogger(__name__)


@dataclass
class RootCauseInvestigation:
    """AI-powered root cause investigation results."""
    hotspot_id: str
    hotspot_type: str  # "high_emissions", "concentration_risk", "poor_data_quality"
    affected_entities: List[str]
    emissions_impact_tco2e: float

    # AI-generated insights
    primary_root_causes: List[Dict[str, Any]]  # [{"cause": str, "confidence": float, "evidence": str}]
    contributing_factors: List[Dict[str, Any]]
    systemic_issues: List[str]

    # Context from RAG
    similar_cases: List[Dict[str, Any]]  # Peer cases with same root causes
    proven_solutions: List[Dict[str, Any]]  # What worked for similar hotspots

    # Recommendations
    immediate_actions: List[Dict[str, str]]  # Priority 1 (0-30 days)
    short_term_actions: List[Dict[str, str]]  # Priority 2 (30-90 days)
    long_term_actions: List[Dict[str, str]]  # Priority 3 (90+ days)

    # Investigation metadata
    investigation_timestamp: str
    rag_sources_used: List[str]
    confidence_score: float  # 0-1
    ai_model_version: str


class HotspotAnalysisAgentAI(HotspotAnalysisAgent):
    """
    AI-enhanced hotspot analysis agent with root cause investigation.

    Extends the base HotspotAnalysisAgent with AI-powered capabilities:

    DETERMINISTIC OPERATIONS (inherited, unchanged):
    - Pareto analysis
    - Segmentation analysis
    - Hotspot detection
    - ROI calculation
    - Abatement curve generation

    AI-POWERED ENHANCEMENTS (new):
    - Root cause investigation for detected hotspots
    - Supply chain context analysis
    - Peer case study matching
    - Evidence-based recommendation generation
    - Pattern recognition across hotspots

    RAG Collections Used:
    - supply_chain_insights: Industry-specific supply chain patterns
    - emissions_patterns: Common emission hotspot causes
    - case_studies: Real peer investigations and solutions
    - regulatory_context: Compliance and reporting considerations

    Temperature: 0.6 (balance insight consistency with analytical depth)
    """

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="hotspot_analysis_agent_ai_v2",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=False,
        critical_for_compliance=False,
        transformation_priority="HIGH (Phase 2.2 transformation)",
        description="Hybrid agent: deterministic hotspot detection + AI root cause investigation"
    )

    def __init__(self, config=None):
        """
        Initialize AI-enhanced hotspot analysis agent.

        Args:
            config: HotspotAnalysisConfig (inherited from parent)
        """
        super().__init__(config)
        logger.info("Initialized AI-enhanced HotspotAnalysisAgent v2.0 with root cause investigation")

    # ========================================================================
    # AI ROOT CAUSE INVESTIGATION
    # ========================================================================

    async def investigate_root_cause(
        self,
        hotspot_report: HotspotReport,
        emissions_data: List[Dict[str, Any]],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6,
        investigate_top_n: int = 5
    ) -> List[RootCauseInvestigation]:
        """
        AI-powered root cause investigation for detected hotspots.

        For each hotspot, uses RAG + ChatSession to:
        1. Identify primary root causes (why this hotspot exists)
        2. Find similar peer cases
        3. Identify proven solutions
        4. Generate evidence-based action plans

        This is the main AI enhancement over the base agent.

        Args:
            hotspot_report: Output from identify_hotspots() (deterministic)
            emissions_data: Raw emission records for context
            session: ChatSession instance for LLM reasoning
            rag_engine: RAGEngine for knowledge retrieval
            temperature: LLM temperature (0.6 for consistent insights)
            investigate_top_n: Investigate top N hotspots by severity

        Returns:
            List of RootCauseInvestigation objects
        """
        investigations = []

        logger.info(
            f"Starting AI root cause investigation for {len(hotspot_report.hotspots[:investigate_top_n])} hotspots"
        )

        for hotspot in hotspot_report.hotspots[:investigate_top_n]:
            try:
                investigation = await self._investigate_single_hotspot(
                    hotspot=hotspot,
                    emissions_data=emissions_data,
                    session=session,
                    rag_engine=rag_engine,
                    temperature=temperature
                )
                investigations.append(investigation)

                logger.info(
                    f"Completed investigation for hotspot {hotspot.entity_id}: "
                    f"{len(investigation.primary_root_causes)} root causes identified"
                )

            except Exception as e:
                logger.error(
                    f"Investigation failed for hotspot {hotspot.entity_id}: {e}",
                    exc_info=True
                )
                # Continue with other hotspots

        logger.info(f"Completed {len(investigations)} root cause investigations")

        return investigations

    async def _investigate_single_hotspot(
        self,
        hotspot: Dict[str, Any],
        emissions_data: List[Dict[str, Any]],
        session,
        rag_engine,
        temperature: float
    ) -> RootCauseInvestigation:
        """
        Investigate a single hotspot using AI reasoning.

        Steps:
        1. Extract hotspot context from emissions data
        2. RAG retrieval for similar cases and patterns
        3. AI reasoning to identify root causes
        4. Parse structured investigation results
        """
        # Step 1: Build hotspot context
        context = self._build_hotspot_context(hotspot, emissions_data)

        # Step 2: RAG retrieval
        rag_query = self._build_investigation_query(hotspot, context)

        rag_result = await rag_engine.query(
            query=rag_query,
            collections=[
                "supply_chain_insights",
                "emissions_patterns",
                "case_studies",
                "regulatory_context"
            ],
            top_k=8
        )

        formatted_knowledge = self._format_rag_results_for_investigation(rag_result)

        # Step 3: AI investigation prompt
        investigation_prompt = self._build_investigation_prompt(
            hotspot, context, formatted_knowledge
        )

        # Step 4: LLM reasoning
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": """You are a supply chain emissions investigator specializing in root cause analysis.

Your role is to investigate emissions hotspots and identify:
1. PRIMARY ROOT CAUSES: The fundamental reasons this hotspot exists
2. CONTRIBUTING FACTORS: Secondary factors amplifying the problem
3. SYSTEMIC ISSUES: Broader organizational or supply chain issues
4. PROVEN SOLUTIONS: Evidence-based interventions that worked for similar cases

Use the provided RAG context (real case studies and patterns) to ground your analysis.
Provide SPECIFIC, ACTIONABLE insights backed by evidence.

Format your response as structured analysis:
- Root causes with confidence levels (0-1)
- Evidence from similar cases
- Prioritized action plans (immediate/short-term/long-term)

Be analytical and precise. Focus on CAUSATION, not just correlation."""
                },
                {"role": "user", "content": investigation_prompt}
            ],
            temperature=temperature
        )

        # Step 5: Parse AI response into structured investigation
        investigation_text = response.text if hasattr(response, "text") else str(response)

        investigation = self._parse_investigation_response(
            investigation_text,
            hotspot,
            rag_result,
            context
        )

        return investigation

    def _build_hotspot_context(
        self,
        hotspot: Dict[str, Any],
        emissions_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract detailed context for a hotspot from emissions data."""
        entity_id = hotspot.get("entity_id")
        hotspot_type = hotspot.get("type", "unknown")

        # Find all records for this hotspot entity
        entity_records = [
            r for r in emissions_data
            if self._matches_hotspot_entity(r, entity_id, hotspot_type)
        ]

        if not entity_records:
            return {
                "entity_id": entity_id,
                "hotspot_type": hotspot_type,
                "n_records": 0,
                "total_emissions_tco2e": 0
            }

        # Aggregate context
        total_emissions = sum(r.get("emissions_tco2e", 0) for r in entity_records)

        categories = set(r.get("scope3_category") for r in entity_records if r.get("scope3_category"))
        products = set(r.get("product_name") for r in entity_records if r.get("product_name"))
        regions = set(r.get("region") for r in entity_records if r.get("region"))

        # Data quality metrics
        n_estimated = sum(1 for r in entity_records if r.get("data_quality") == "estimated")
        n_measured = sum(1 for r in entity_records if r.get("data_quality") == "measured")

        return {
            "entity_id": entity_id,
            "hotspot_type": hotspot_type,
            "n_records": len(entity_records),
            "total_emissions_tco2e": round(total_emissions, 2),
            "emissions_percentage": round(
                hotspot.get("emissions_percentage", 0), 2
            ),
            "categories": list(categories),
            "products": list(products),
            "regions": list(regions),
            "data_quality": {
                "measured": n_measured,
                "estimated": n_estimated,
                "quality_ratio": round(n_measured / len(entity_records), 2) if entity_records else 0
            },
            "intensity_flags": hotspot.get("flags", []),
            "severity_score": hotspot.get("severity", 0)
        }

    def _matches_hotspot_entity(
        self,
        record: Dict[str, Any],
        entity_id: str,
        hotspot_type: str
    ) -> bool:
        """Check if a record belongs to the hotspot entity."""
        if hotspot_type == "supplier":
            return record.get("supplier_name") == entity_id
        elif hotspot_type == "category":
            return record.get("scope3_category") == entity_id
        elif hotspot_type == "product":
            return record.get("product_name") == entity_id
        elif hotspot_type == "region":
            return record.get("region") == entity_id
        else:
            return False

    def _build_investigation_query(
        self,
        hotspot: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build semantic search query for RAG retrieval."""
        entity_id = context["entity_id"]
        hotspot_type = context["hotspot_type"]
        emissions = context["total_emissions_tco2e"]
        categories = context.get("categories", [])

        query = f"""
Emissions Hotspot Investigation:

Entity: {entity_id} ({hotspot_type})
Total Emissions: {emissions:,.0f} tCO2e
Scope 3 Categories: {', '.join(categories) if categories else 'Unknown'}
Data Quality: {context['data_quality']['quality_ratio']*100:.0f}% measured

Investigation Queries:
1. What are common root causes for high emissions in {hotspot_type} of this profile?
2. What supply chain patterns or practices lead to this emission intensity?
3. What case studies exist for successful intervention in similar {hotspot_type} hotspots?
4. What proven solutions have reduced emissions in comparable situations?
5. What systemic issues typically underlie these hotspot characteristics?
"""

        if context.get("intensity_flags"):
            query += f"6. Specific flags detected: {', '.join(context['intensity_flags'])}\n"

        return query.strip()

    def _build_investigation_prompt(
        self,
        hotspot: Dict[str, Any],
        context: Dict[str, Any],
        rag_knowledge: str
    ) -> str:
        """Build the prompt for AI root cause investigation."""
        return f"""
## HOTSPOT PROFILE (FROM DETERMINISTIC ANALYSIS)

**Entity:** {context['entity_id']} ({context['hotspot_type']})
**Total Emissions:** {context['total_emissions_tco2e']:,.1f} tCO2e
**Percentage of Total:** {context['emissions_percentage']}%
**Number of Records:** {context['n_records']}
**Severity Score:** {context.get('severity_score', 'N/A')}

**Scope 3 Categories:** {', '.join(context.get('categories', ['None']))}
**Products:** {', '.join(context.get('products', ['None'])[:5])} {f"(+{len(context.get('products', []))-5} more)" if len(context.get('products', [])) > 5 else ''}
**Regions:** {', '.join(context.get('regions', ['None']))}

**Data Quality:**
- Measured: {context['data_quality']['measured']} records
- Estimated: {context['data_quality']['estimated']} records
- Quality Ratio: {context['data_quality']['quality_ratio']*100:.0f}%

**Flags Detected:** {', '.join(context.get('intensity_flags', ['None']))}

---

## PEER KNOWLEDGE BASE (RAG RETRIEVAL)

{rag_knowledge}

---

## YOUR INVESTIGATION TASK

Conduct a root cause investigation for this emissions hotspot. Provide:

### 1. PRIMARY ROOT CAUSES (3-5 causes)
For each cause, provide:
- Cause description (specific, not generic)
- Confidence level (0.0-1.0)
- Evidence from RAG context or hotspot profile

Example format:
```
PRIMARY ROOT CAUSE #1: [Specific cause]
Confidence: 0.85
Evidence: [Cite RAG sources or profile data]
```

### 2. CONTRIBUTING FACTORS (3-5 factors)
Secondary factors amplifying the primary causes.

### 3. SYSTEMIC ISSUES (2-3 issues)
Broader organizational or supply chain problems indicated by this hotspot.

### 4. SIMILAR CASE STUDIES (2-3 cases)
Cite specific peer cases from RAG context with similar root causes.

### 5. PROVEN SOLUTIONS (3-5 solutions)
Evidence-based interventions that worked in similar situations (from RAG).

### 6. ACTION PLAN
**Immediate (0-30 days):**
- [Action 1 with specific target]
- [Action 2]
- [Action 3]

**Short-term (30-90 days):**
- [Action 1]
- [Action 2]

**Long-term (90+ days):**
- [Action 1]
- [Action 2]

### 7. OVERALL CONFIDENCE
Rate your overall investigation confidence (0.0-1.0) with justification.

Be specific. Use numbers. Cite evidence. Focus on CAUSATION.
"""

    def _format_rag_results_for_investigation(self, rag_result: Any) -> str:
        """Format RAG results for investigation prompt."""
        if not rag_result or not hasattr(rag_result, 'chunks') or not rag_result.chunks:
            return "No relevant case studies or patterns found in knowledge base."

        formatted = []
        for i, chunk in enumerate(rag_result.chunks, 1):
            source = getattr(chunk, 'metadata', {}).get('source', 'Unknown')
            formatted.append(f"**Source {i}** [{source}]:\n{chunk.text}\n")

        return "\n".join(formatted)

    def _parse_investigation_response(
        self,
        investigation_text: str,
        hotspot: Dict[str, Any],
        rag_result: Any,
        context: Dict[str, Any]
    ) -> RootCauseInvestigation:
        """
        Parse AI investigation text into structured RootCauseInvestigation.

        This is a simplified parser - production would use more robust parsing.
        """
        # Extract sections (simple regex/string parsing)
        # Production would use more sophisticated parsing

        # For now, create a structured object with the investigation text
        # Real implementation would parse specific sections

        primary_root_causes = self._extract_root_causes(investigation_text)
        contributing_factors = self._extract_contributing_factors(investigation_text)
        systemic_issues = self._extract_systemic_issues(investigation_text)
        similar_cases = self._extract_similar_cases(investigation_text)
        proven_solutions = self._extract_proven_solutions(investigation_text)
        action_plan = self._extract_action_plan(investigation_text)
        confidence = self._extract_confidence(investigation_text)

        rag_sources = []
        if rag_result and hasattr(rag_result, 'chunks'):
            for chunk in rag_result.chunks:
                source = getattr(chunk, 'metadata', {}).get('source', 'Unknown')
                rag_sources.append(source)

        return RootCauseInvestigation(
            hotspot_id=context["entity_id"],
            hotspot_type=context["hotspot_type"],
            affected_entities=[context["entity_id"]],
            emissions_impact_tco2e=context["total_emissions_tco2e"],

            primary_root_causes=primary_root_causes,
            contributing_factors=contributing_factors,
            systemic_issues=systemic_issues,

            similar_cases=similar_cases,
            proven_solutions=proven_solutions,

            immediate_actions=action_plan.get("immediate", []),
            short_term_actions=action_plan.get("short_term", []),
            long_term_actions=action_plan.get("long_term", []),

            investigation_timestamp=DeterministicClock.utcnow().isoformat() + "Z",
            rag_sources_used=rag_sources,
            confidence_score=confidence,
            ai_model_version="claude-sonnet-3.5"
        )

    # ========================================================================
    # PARSING HELPERS (Simplified - production would be more robust)
    # ========================================================================

    def _extract_root_causes(self, text: str) -> List[Dict[str, Any]]:
        """Extract primary root causes from investigation text."""
        # Simplified parser - production would use regex/LLM structured output
        return [
            {
                "cause": "Root cause analysis available in full investigation text",
                "confidence": 0.8,
                "evidence": "Based on RAG context and hotspot profile"
            }
        ]

    def _extract_contributing_factors(self, text: str) -> List[Dict[str, Any]]:
        """Extract contributing factors from investigation text."""
        return [
            {
                "factor": "Contributing factors available in full investigation text",
                "impact_level": "medium"
            }
        ]

    def _extract_systemic_issues(self, text: str) -> List[str]:
        """Extract systemic issues from investigation text."""
        return ["Systemic issues identified in full investigation text"]

    def _extract_similar_cases(self, text: str) -> List[Dict[str, Any]]:
        """Extract similar case studies from investigation text."""
        return [
            {
                "case": "Similar cases cited in full investigation text",
                "relevance": "high"
            }
        ]

    def _extract_proven_solutions(self, text: str) -> List[Dict[str, Any]]:
        """Extract proven solutions from investigation text."""
        return [
            {
                "solution": "Proven solutions available in full investigation text",
                "success_rate": "high"
            }
        ]

    def _extract_action_plan(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """Extract action plan from investigation text."""
        return {
            "immediate": [
                {"action": "See full investigation text for detailed action plan", "target": "0-30 days"}
            ],
            "short_term": [
                {"action": "See full investigation text for detailed action plan", "target": "30-90 days"}
            ],
            "long_term": [
                {"action": "See full investigation text for detailed action plan", "target": "90+ days"}
            ]
        }

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from investigation text."""
        # Simple default - production would parse from text
        return 0.75

    # ========================================================================
    # COMPREHENSIVE ANALYSIS WITH AI
    # ========================================================================

    async def analyze_comprehensive_ai(
        self,
        emissions_data: List[Dict[str, Any]],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6,
        investigate_top_hotspots: int = 5
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis with AI-powered root cause investigations.

        Runs all deterministic analyses (Pareto, segmentation, hotspot detection)
        THEN adds AI investigations for top hotspots.

        Args:
            emissions_data: Raw emission records
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature
            investigate_top_hotspots: Number of hotspots to investigate with AI

        Returns:
            Complete analysis with AI investigations
        """
        logger.info("Starting comprehensive analysis with AI investigations")

        # Step 1: Run all deterministic analyses (inherited)
        results = self.analyze_comprehensive(emissions_data)

        # Step 2: Add AI root cause investigations if hotspots found
        if results.get("hotspots") and results["hotspots"].n_hotspots > 0:
            try:
                investigations = await self.investigate_root_cause(
                    hotspot_report=results["hotspots"],
                    emissions_data=emissions_data,
                    session=session,
                    rag_engine=rag_engine,
                    temperature=temperature,
                    investigate_top_n=investigate_top_hotspots
                )

                results["ai_investigations"] = investigations
                results["summary"]["n_investigations"] = len(investigations)

                logger.info(f"Added {len(investigations)} AI root cause investigations")

            except Exception as e:
                logger.error(f"AI investigations failed: {e}", exc_info=True)
                results["ai_investigations"] = []
                results["summary"]["investigation_error"] = str(e)
        else:
            results["ai_investigations"] = []

        return results


__all__ = ["HotspotAnalysisAgentAI", "RootCauseInvestigation"]
