# -*- coding: utf-8 -*-
"""
Intelligent Recommendation Engine - LLM-Powered Strategic Insights

FIXES THE INTELLIGENCE PARADOX:
- Transforms generic recommendations into strategic, actionable insights
- Uses LLM to generate context-aware supplier engagement strategies
- Maintains fallback to template-based recommendations
- Provides ROI estimates and prioritization

Version: 2.0.0 (Intelligence Fix)
Date: 2025-01-08
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from greenlang.determinism import deterministic_uuid, DeterministicClock
from ..models import (
    Insight,
    InsightReport,
    HotspotReport,
    ParetoAnalysis,
    SegmentationAnalysis,
    AbatementCurve,
    Hotspot
)
from ..config import InsightPriority, InsightType
from ..exceptions import InsightGenerationError

# Import LLM infrastructure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))
from greenlang.intelligence.providers.openai import OpenAIProvider
from greenlang.intelligence.providers.anthropic import AnthropicProvider
from greenlang.intelligence.providers.base import LLMProviderConfig
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.runtime.budget import Budget

logger = logging.getLogger(__name__)


class IntelligentRecommendationEngine:
    """
    LLM-powered strategic recommendation generator.

    THIS IS THE INTELLIGENCE FIX!

    Transforms raw emissions data into actionable strategic insights:
    - Supplier engagement strategies
    - Alternative supplier recommendations
    - Operational improvement opportunities
    - ROI and feasibility analysis
    - Category-specific best practices

    Hybrid Approach:
    - LLM: Strategic reasoning, natural language generation
    - Deterministic: Prioritization scoring, data validation
    - Fallback: Template-based recommendations if LLM unavailable
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_enabled: bool = True
    ):
        """
        Initialize intelligent recommendation engine.

        Args:
            llm_provider: LLM provider ('openai' or 'anthropic')
            llm_model: Model name (use gpt-4o for best strategic reasoning)
            llm_enabled: Enable LLM intelligence (disable for testing/fallback)
        """
        self.llm_enabled = llm_enabled
        self.llm_client = None

        # Initialize LLM client (if enabled)
        if self.llm_enabled:
            try:
                llm_config = LLMProviderConfig(
                    model=llm_model,
                    api_key_env="OPENAI_API_KEY" if llm_provider == "openai" else "ANTHROPIC_API_KEY",
                    timeout_s=60.0,  # Strategic reasoning may take longer
                    max_retries=2
                )

                if llm_provider == "openai":
                    self.llm_client = OpenAIProvider(llm_config)
                else:
                    self.llm_client = AnthropicProvider(llm_config)

                logger.info(f"Initialized IntelligentRecommendationEngine with {llm_provider} {llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Falling back to template mode.")
                self.llm_enabled = False

        # Statistics
        self.stats = {
            "insights_generated": 0,
            "llm_insights": 0,
            "template_insights": 0,
            "llm_api_calls": 0,
        }

        logger.info(f"Initialized IntelligentRecommendationEngine (LLM={'enabled' if self.llm_enabled else 'disabled'})")

    async def generate_insights(
        self,
        hotspot_report: HotspotReport = None,
        pareto_analysis: ParetoAnalysis = None,
        segmentation_analysis: SegmentationAnalysis = None,
        abatement_curve: AbatementCurve = None,
        company_context: Optional[Dict[str, Any]] = None
    ) -> InsightReport:
        """
        Generate comprehensive strategic insights from analysis results.

        Args:
            hotspot_report: Hotspot detection results
            pareto_analysis: Pareto analysis results
            segmentation_analysis: Segmentation analysis results
            abatement_curve: Abatement curve results
            company_context: Company context for personalized recommendations

        Returns:
            InsightReport with prioritized strategic recommendations

        Raises:
            InsightGenerationError: If generation fails critically
        """
        try:
            logger.info("Generating strategic insights (LLM-powered)")

            insights = []

            # Generate insights from hotspots (LLM-POWERED)
            if hotspot_report:
                hotspot_insights = await self._insights_from_hotspots(
                    hotspot_report,
                    company_context
                )
                insights.extend(hotspot_insights)

            # Generate insights from Pareto analysis
            if pareto_analysis:
                pareto_insights = await self._insights_from_pareto(pareto_analysis)
                insights.extend(pareto_insights)

            # Generate insights from segmentation
            if segmentation_analysis:
                seg_insights = await self._insights_from_segmentation(segmentation_analysis)
                insights.extend(seg_insights)

            # Generate insights from abatement curve
            if abatement_curve:
                abatement_insights = await self._insights_from_abatement_curve(abatement_curve)
                insights.extend(abatement_insights)

            # Sort by priority
            insights.sort(key=lambda i: self._priority_score(i.priority), reverse=True)

            # Categorize by priority
            critical = [i for i in insights if i.priority == InsightPriority.CRITICAL]
            high = [i for i in insights if i.priority == InsightPriority.HIGH]
            medium = [i for i in insights if i.priority == InsightPriority.MEDIUM]
            low = [i for i in insights if i.priority == InsightPriority.LOW]

            # Generate summary (LLM-powered if available)
            summary = await self._generate_executive_summary(insights, company_context)

            # Extract top recommendations
            top_recommendations = [i.recommendation for i in insights[:5]]

            report = InsightReport(
                total_insights=len(insights),
                critical_insights=critical,
                high_insights=high,
                medium_insights=medium,
                low_insights=low,
                all_insights=insights,
                summary=summary,
                top_recommendations=top_recommendations
            )

            self.stats["insights_generated"] = len(insights)

            logger.info(
                f"Generated {len(insights)} insights: "
                f"{len(critical)} critical, {len(high)} high, "
                f"{len(medium)} medium, {len(low)} low | "
                f"{self.stats['llm_insights']} LLM, {self.stats['template_insights']} template"
            )

            return report

        except Exception as e:
            logger.error(f"Insight generation failed: {e}", exc_info=True)
            raise InsightGenerationError(f"Insight generation failed: {e}") from e

    async def _insights_from_hotspots(
        self,
        hotspot_report: HotspotReport,
        company_context: Optional[Dict[str, Any]] = None
    ) -> List[Insight]:
        """
        Generate strategic insights from hotspot detection.

        THIS IS WHERE THE MAGIC HAPPENS - LLM generates actionable strategies!

        Args:
            hotspot_report: Hotspot report
            company_context: Company context for personalization

        Returns:
            List of strategic insights
        """
        insights = []

        for hotspot in hotspot_report.hotspots[:10]:  # Top 10 hotspots
            # Generate LLM-powered strategic recommendation
            if self.llm_enabled and hotspot.emissions_tco2e >= 100:
                # Use LLM for high-value hotspots
                insight = await self._generate_llm_hotspot_insight(hotspot, company_context)
                if insight:
                    self.stats["llm_insights"] += 1
                    insights.append(insight)
                    continue

            # Fallback to template-based insight
            insight = self._generate_template_hotspot_insight(hotspot)
            self.stats["template_insights"] += 1
            insights.append(insight)

        return insights

    async def _generate_llm_hotspot_insight(
        self,
        hotspot: Hotspot,
        company_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Insight]:
        """
        Use LLM to generate strategic, actionable insight for hotspot.

        THIS IS THE INTELLIGENCE FIX!

        Args:
            hotspot: Hotspot data
            company_context: Company context for personalization

        Returns:
            Strategic Insight or None if LLM fails
        """
        try:
            # Build context-rich prompt
            system_prompt = """You are a strategic sustainability advisor for corporations.

Your task is to analyze emissions hotspots and generate actionable, strategic recommendations for procurement and sustainability teams.

Provide:
1. Strategic insight: WHY this hotspot matters (beyond just the numbers)
2. Actionable recommendations: WHAT to do (supplier engagement, alternatives, operational changes)
3. ROI estimate: Financial impact and feasibility
4. Timeline: Short-term (0-6mo), medium-term (6-18mo), long-term (18mo+)
5. Stakeholders: Who needs to be involved

Be specific, practical, and business-focused. Avoid generic advice."""

            company_name = company_context.get("company_name", "your company") if company_context else "your company"
            target_year = company_context.get("target_year", 2030) if company_context else 2030

            user_prompt = f"""Analyze this emissions hotspot and provide strategic recommendations:

**Hotspot Details:**
- Entity: {hotspot.entity_name}
- Type: {hotspot.hotspot_type} (supplier/category/product/etc.)
- Emissions: {hotspot.emissions_tco2e:,.0f} tCO2e
- % of Total: {hotspot.percent_of_total:.1f}%
- Category: Scope 3 Category {hotspot.scope3_category or 'Unknown'}
- Industry: {getattr(hotspot, 'industry', 'Unknown')}
- Spend: ${getattr(hotspot, 'spend', 0):,.0f}

**Company Context:**
- Company: {company_name}
- Net Zero Target: {target_year}

**Your Task:**
Generate 2-3 specific, actionable recommendations. For each recommendation, provide:
1. Title (8-12 words, action-oriented)
2. Description (100-150 words)
3. Expected Impact (tCO2e reduction estimate)
4. Feasibility (High/Medium/Low)
5. Timeline (Short/Medium/Long term)
6. Key Stakeholders (Procurement, Sustainability, Operations, etc.)
7. ROI Rationale (cost savings, risk mitigation, competitive advantage)

Return JSON:
{{
    "recommendations": [
        {{
            "title": "Engage supplier on science-based targets and emissions disclosure",
            "description": "...",
            "impact_tco2e": 450,
            "feasibility": "High",
            "timeline": "Short-term (0-6 months)",
            "stakeholders": ["Procurement", "Sustainability"],
            "roi_rationale": "Low cost, high impact through supplier collaboration"
        }},
        ...
    ],
    "priority_rationale": "Why this hotspot is critical for {company_name}'s net zero strategy"
}}"""

            # Call LLM
            messages = [
                ChatMessage(role=Role.system, content=system_prompt),
                ChatMessage(role=Role.user, content=user_prompt)
            ]

            budget = Budget(max_usd=0.20)  # $0.20 per hotspot insight (worth it!)

            # Use JSON schema for structured output
            json_schema = {
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "impact_tco2e": {"type": "number"},
                                "feasibility": {"type": "string", "enum": ["High", "Medium", "Low"]},
                                "timeline": {"type": "string"},
                                "stakeholders": {"type": "array", "items": {"type": "string"}},
                                "roi_rationale": {"type": "string"}
                            },
                            "required": ["title", "description", "impact_tco2e", "feasibility"]
                        }
                    },
                    "priority_rationale": {"type": "string"}
                },
                "required": ["recommendations", "priority_rationale"],
                "additionalProperties": False
            }

            response = await self.llm_client.chat(
                messages=messages,
                json_schema=json_schema,
                budget=budget,
                temperature=0.4  # Some creativity, but not too much
            )

            self.stats["llm_api_calls"] += 1

            # Parse LLM response
            llm_output = json.loads(response.text)

            # Format recommendations as newline-separated text
            recommendations_text = "\n\n".join([
                f"**{i+1}. {rec['title']}**\n"
                f"{rec['description']}\n"
                f"• Expected Impact: ~{rec['impact_tco2e']:,.0f} tCO2e reduction\n"
                f"• Feasibility: {rec['feasibility']}\n"
                f"• Timeline: {rec.get('timeline', 'TBD')}\n"
                f"• Stakeholders: {', '.join(rec.get('stakeholders', ['TBD']))}\n"
                f"• ROI: {rec.get('roi_rationale', 'Evaluate based on context')}"
                for i, rec in enumerate(llm_output["recommendations"])
            ])

            # Calculate total estimated impact
            total_impact = sum(rec.get("impact_tco2e", 0) for rec in llm_output["recommendations"])

            # Create rich insight
            insight = Insight(
                insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                insight_type=InsightType.HIGH_EMISSIONS_SUPPLIER
                if hotspot.hotspot_type == "supplier_name"
                else InsightType.HIGH_EMISSIONS_CATEGORY,
                priority=hotspot.priority,
                title=f"Strategic Opportunity: {hotspot.entity_name}",
                description=llm_output["priority_rationale"],
                recommendation=recommendations_text,
                affected_entities=[hotspot.entity_name],
                estimated_impact_tco2e=total_impact,
                metadata={
                    "hotspot_id": hotspot.hotspot_id,
                    "llm_generated": True,
                    "llm_model": self.llm_client.config.model,
                    "llm_cost_usd": response.usage.cost_usd,
                    "num_recommendations": len(llm_output["recommendations"]),
                    "generation_timestamp": DeterministicClock.utcnow().isoformat()
                }
            )

            logger.info(
                f"Generated LLM insight for {hotspot.entity_name}: "
                f"{len(llm_output['recommendations'])} recommendations, "
                f"~{total_impact:,.0f} tCO2e potential impact"
            )

            return insight

        except Exception as e:
            logger.warning(f"LLM hotspot insight generation failed: {e}. Using template.")
            return None

    def _generate_template_hotspot_insight(self, hotspot: Hotspot) -> Insight:
        """
        Generate template-based insight (fallback when LLM unavailable).

        Args:
            hotspot: Hotspot data

        Returns:
            Template-based Insight
        """
        # Generic recommendation based on hotspot type
        if hotspot.hotspot_type == "supplier_name":
            recommendation = (
                f"**Supplier Engagement Priority**\n\n"
                f"{hotspot.entity_name} contributes {hotspot.emissions_tco2e:,.0f} tCO2e "
                f"({hotspot.percent_of_total:.1f}% of total emissions).\n\n"
                f"**Recommended Actions:**\n"
                f"1. Request Scope 1 & 2 emissions disclosure\n"
                f"2. Evaluate alternative suppliers with lower carbon intensity\n"
                f"3. Discuss carbon reduction commitments and SBTi alignment\n"
                f"4. Consider sustainability criteria in next RFP cycle"
            )
            insight_type = InsightType.HIGH_EMISSIONS_SUPPLIER
        elif hotspot.hotspot_type == "scope3_category":
            recommendation = (
                f"**Category Focus Area**\n\n"
                f"Scope 3 Category {hotspot.scope3_category}: {hotspot.emissions_tco2e:,.0f} tCO2e "
                f"({hotspot.percent_of_total:.1f}% of total).\n\n"
                f"**Recommended Actions:**\n"
                f"1. Deep-dive analysis on this category\n"
                f"2. Identify reduction opportunities and best practices\n"
                f"3. Set category-specific reduction targets\n"
                f"4. Engage relevant suppliers and value chain partners"
            )
            insight_type = InsightType.HIGH_EMISSIONS_CATEGORY
        else:
            recommendation = (
                f"**Hotspot Identified**\n\n"
                f"{hotspot.entity_name}: {hotspot.emissions_tco2e:,.0f} tCO2e.\n\n"
                f"**Recommended Actions:**\n"
                f"1. Investigate root causes of high emissions\n"
                f"2. Evaluate reduction opportunities\n"
                f"3. Engage relevant stakeholders"
            )
            insight_type = InsightType.DATA_QUALITY_ISSUE

        return Insight(
            insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
            insight_type=insight_type,
            priority=hotspot.priority,
            title=f"{hotspot.entity_name} - High Emissions Hotspot",
            description=f"Identified as hotspot: {hotspot.emissions_tco2e:,.0f} tCO2e ({hotspot.percent_of_total:.1f}% of total)",
            recommendation=recommendation,
            affected_entities=[hotspot.entity_name],
            estimated_impact_tco2e=hotspot.emissions_tco2e * 0.3,  # Assume 30% reduction potential
            metadata={
                "hotspot_id": hotspot.hotspot_id,
                "llm_generated": False,
                "template_based": True
            }
        )

    async def _insights_from_pareto(self, pareto_analysis: ParetoAnalysis) -> List[Insight]:
        """Generate insights from Pareto analysis (deterministic)."""
        insights = []

        # 80/20 rule insight
        if pareto_analysis.pareto_80_count > 0:
            insight = Insight(
                insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                insight_type=InsightType.PARETO_CONCENTRATION,
                priority=InsightPriority.HIGH,
                title=f"Pareto Principle: {pareto_analysis.pareto_80_count} Suppliers = 80% of Emissions",
                description=(
                    f"Highly concentrated emissions: just {pareto_analysis.pareto_80_count} suppliers "
                    f"account for 80% of total emissions ({pareto_analysis.pareto_80_emissions:,.0f} tCO2e). "
                    f"Focus on these key suppliers for maximum impact."
                ),
                recommendation=(
                    "**Strategic Focus Strategy**\n\n"
                    "Leverage the Pareto principle for efficient emissions reduction:\n"
                    "1. Prioritize engagement with top suppliers (80% of impact)\n"
                    "2. Develop deep partnerships and reduction roadmaps\n"
                    "3. Share best practices and collaborate on innovation\n"
                    "4. Allocate resources proportionally to impact"
                ),
                affected_entities=[],
                estimated_impact_tco2e=pareto_analysis.pareto_80_emissions * 0.4,  # 40% reduction potential
                metadata={"pareto_80_count": pareto_analysis.pareto_80_count}
            )
            insights.append(insight)

        return insights

    async def _insights_from_segmentation(self, segmentation: SegmentationAnalysis) -> List[Insight]:
        """Generate insights from segmentation analysis (deterministic)."""
        insights = []
        # Placeholder - can add segment-specific insights
        return insights

    async def _insights_from_abatement_curve(self, abatement_curve: AbatementCurve) -> List[Insight]:
        """Generate insights from abatement curve (deterministic)."""
        insights = []
        # Placeholder - can add abatement opportunity insights
        return insights

    async def _generate_executive_summary(
        self,
        insights: List[Insight],
        company_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate executive summary of insights (LLM-powered if available).

        Args:
            insights: Generated insights
            company_context: Company context

        Returns:
            Executive summary text
        """
        if not self.llm_enabled or len(insights) == 0:
            return self._generate_template_summary(insights)

        try:
            # Build summary prompt
            insights_summary = "\n".join([
                f"- {i.title}: {i.estimated_impact_tco2e:,.0f} tCO2e potential impact"
                for i in insights[:10]
            ])

            prompt = f"""Write a concise executive summary (150-200 words) of these emissions reduction insights:

{insights_summary}

Total insights: {len(insights)}
Critical: {len([i for i in insights if i.priority == InsightPriority.CRITICAL])}
High: {len([i for i in insights if i.priority == InsightPriority.HIGH])}

Structure:
1. Overview of findings (1 sentence)
2. Top priorities (2-3 key opportunities)
3. Strategic recommendation (1-2 sentences)

Tone: Executive-level, strategic, action-oriented
"""

            messages = [ChatMessage(role=Role.user, content=prompt)]
            budget = Budget(max_usd=0.05)

            response = await self.llm_client.chat(
                messages=messages,
                budget=budget,
                temperature=0.3
            )

            return response.text

        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}. Using template.")
            return self._generate_template_summary(insights)

    def _generate_template_summary(self, insights: List[Insight]) -> str:
        """Generate template-based summary (fallback)."""
        total = len(insights)
        critical = len([i for i in insights if i.priority == InsightPriority.CRITICAL])
        high = len([i for i in insights if i.priority == InsightPriority.HIGH])

        return (
            f"Analysis identified {total} emission reduction opportunities, "
            f"including {critical} critical and {high} high-priority items. "
            f"Top recommendations focus on supplier engagement, alternative sourcing, "
            f"and operational improvements. Prioritize high-impact actions for maximum ROI."
        )

    def _priority_score(self, priority: InsightPriority) -> int:
        """Convert priority to numeric score for sorting."""
        return {
            InsightPriority.CRITICAL: 4,
            InsightPriority.HIGH: 3,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 1
        }.get(priority, 0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        total = self.stats["insights_generated"] or 1
        return {
            "insights_generated": self.stats["insights_generated"],
            "llm_insights": self.stats["llm_insights"],
            "template_insights": self.stats["template_insights"],
            "llm_api_calls": self.stats["llm_api_calls"],
            "llm_usage_rate": self.stats["llm_insights"] / total * 100 if self.llm_enabled else 0.0
        }


# Backward compatibility alias
RecommendationEngine = IntelligentRecommendationEngine
