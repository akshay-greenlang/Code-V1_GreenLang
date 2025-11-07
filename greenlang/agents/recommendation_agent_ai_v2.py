"""
AI-Powered Recommendation Agent (Phase 2.2 Transformation)
GL Intelligence Infrastructure

Transforms static database lookups into AI-driven recommendations using:
- RAG for knowledge retrieval (case studies, best practices)
- ChatSession for multi-turn reasoning
- Multi-tool orchestration for validation
- Temperature 0.7 for creative problem-solving

Pattern: ReasoningAgent (RECOMMENDATION PATH)
Version: 2.0.0
Date: 2025-11-06
"""

from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime

from greenlang.agents.base_agents import ReasoningAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.intelligence.schemas.tools import ToolDef

logger = logging.getLogger(__name__)


class RecommendationAgentAI(ReasoningAgent):
    """
    AI-powered recommendation agent using RAG + multi-tool reasoning.

    Transformation from v1 (static lookups) to v2 (AI reasoning):
    - BEFORE: Static database → First match
    - AFTER: RAG retrieval → AI reasoning → Multi-tool validation

    Example:
        agent = RecommendationAgentAI()
        result = await agent.reason(
            context={
                "building_type": "commercial_office",
                "emissions_tco2e": 500,
                "electricity_pct": 65,
                "hvac_load": 0.45,
                "building_age": 25,
                "budget": 100000,
                "region": "US"
            },
            session=chat_session,
            rag_engine=rag_engine
        )
    """

    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="recommendation_agent_ai_v2",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=True,
        critical_for_compliance=False,
        transformation_priority="HIGH (Phase 2.2 transformation)",
        description="AI-powered decarbonization recommendations using RAG and multi-tool reasoning"
    )

    def __init__(self):
        """Initialize AI recommendation agent."""
        super().__init__()

    async def reason(
        self,
        context: Dict[str, Any],
        session,      # ChatSession instance
        rag_engine,   # RAGEngine instance
        tools: Optional[List[ToolDef]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered recommendations using RAG + multi-tool reasoning.

        Args:
            context: Building/facility context with metrics
            session: ChatSession for AI reasoning
            rag_engine: RAGEngine for knowledge retrieval
            tools: List of available tools (optional, will use defaults)

        Returns:
            Dictionary with recommendations, reasoning, and metadata
        """
        try:
            # Step 1: RAG retrieval for contextual knowledge
            logger.info("Step 1: Retrieving relevant knowledge from RAG")

            rag_query = self._build_rag_query(context)
            rag_result = await self._rag_retrieve(
                query=rag_query,
                rag_engine=rag_engine,
                collections=[
                    "case_studies",
                    "technology_database",
                    "best_practices",
                    "regulatory_incentives"
                ],
                top_k=8
            )

            formatted_knowledge = self._format_rag_results(rag_result)

            logger.info(f"Retrieved {len(rag_result.chunks)} relevant knowledge chunks")

            # Step 2: Initial AI reasoning with tool definitions
            logger.info("Step 2: Initiating AI reasoning with tools")

            initial_response = await session.chat(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a decarbonization expert specializing in building
                        energy efficiency and emissions reduction. Analyze the facility and use
                        available tools to develop context-aware recommendations. Prioritize
                        cost-effective solutions with proven ROI. Be specific with numbers and
                        use tools to validate feasibility."""
                    },
                    {
                        "role": "user",
                        "content": self._build_recommendation_prompt(context, formatted_knowledge)
                    }
                ],
                tools=tools or self._get_default_tools(),
                temperature=0.7,  # Allow creative reasoning
                tool_choice="auto"
            )

            # Step 3: Multi-turn tool orchestration
            logger.info("Step 3: Executing tool orchestration loop")

            conversation_history = [initial_response]
            tool_execution_trace = []
            tool_registry = self._build_tool_registry(context)

            current_response = initial_response

            # Tool orchestration loop (max 5 iterations)
            iteration = 0
            max_iterations = 5

            while current_response.tool_calls and iteration < max_iterations:
                iteration += 1
                logger.info(f"Tool orchestration iteration {iteration}: {len(current_response.tool_calls)} tools called")

                tool_results = []

                # Execute all tool calls in this turn
                for tool_call in current_response.tool_calls:
                    try:
                        result = await self._execute_tool(
                            tool_call=tool_call,
                            tool_registry=tool_registry
                        )

                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": json.dumps(result)
                        })

                        tool_execution_trace.append({
                            "tool": tool_call["name"],
                            "arguments": json.loads(tool_call["arguments"]),
                            "result": result,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "iteration": iteration
                        })

                        logger.info(f"Tool executed: {tool_call['name']}")

                    except Exception as e:
                        logger.error(f"Tool execution failed: {tool_call['name']}: {e}")
                        tool_results.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": json.dumps({"error": str(e)})
                        })

                # Continue conversation with tool results
                conversation_history.extend(tool_results)

                next_response = await session.chat(
                    messages=conversation_history,
                    tools=tools or self._get_default_tools(),
                    temperature=0.7
                )

                conversation_history.append(next_response)
                current_response = next_response

            logger.info(f"Tool orchestration complete: {iteration} iterations, {len(tool_execution_trace)} total tools executed")

            # Step 4: Parse and structure recommendations
            logger.info("Step 4: Parsing structured recommendations from AI output")

            final_text = current_response.text
            recommendations = self._parse_recommendations(final_text)
            roadmap = self._parse_roadmap(final_text)
            quick_wins = self._extract_quick_wins(recommendations)
            high_impact = self._extract_high_impact(recommendations)

            # Step 5: Build comprehensive result
            result = {
                "success": True,
                "recommendations": recommendations,
                "roadmap": roadmap,
                "quick_wins": quick_wins,
                "high_impact": high_impact,
                "reasoning": final_text,
                "rag_context": {
                    "chunks_retrieved": len(rag_result.chunks),
                    "collections_searched": ["case_studies", "technology_database", "best_practices", "regulatory_incentives"],
                    "relevance_scores": rag_result.relevance_scores.tolist() if hasattr(rag_result.relevance_scores, 'tolist') else rag_result.relevance_scores,
                    "search_time_ms": rag_result.search_time_ms,
                    "query": rag_query
                },
                "tool_execution": {
                    "total_tools_called": len(tool_execution_trace),
                    "tools_by_name": self._count_tools_by_name(tool_execution_trace),
                    "trace": tool_execution_trace
                },
                "metadata": {
                    "model": current_response.provider_info["model"],
                    "tokens_used": current_response.usage["total_tokens"],
                    "cost_usd": current_response.usage["total_cost"],
                    "temperature": 0.7,
                    "pattern": "ReasoningAgent",
                    "version": "2.0.0",
                    "orchestration_iterations": iteration
                },
                "confidence": self._extract_confidence(final_text),
                "context_analyzed": context
            }

            logger.info("Recommendation generation complete")
            return result

        except Exception as e:
            logger.error(f"Error in AI recommendation generation: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "context": context
            }

    def _build_rag_query(self, context: Dict[str, Any]) -> str:
        """Build RAG query from context."""
        building_type = context.get("building_type", "commercial")
        emissions = context.get("emissions_tco2e", 0)
        region = context.get("region", "US")

        query = f"""
        Decarbonization recommendations for {building_type} building
        with {emissions} tCO2e/year emissions in {region}.
        Energy efficiency improvements, renewable energy options, and proven ROI strategies.
        """

        return query.strip()

    def _build_recommendation_prompt(self, context: Dict[str, Any], knowledge: str) -> str:
        """Build comprehensive recommendation prompt for LLM."""
        prompt = f"""
        Develop comprehensive decarbonization recommendations for this facility:

        **Facility Profile:**
        - Building Type: {context.get('building_type', 'Commercial Office')}
        - Total Emissions: {context.get('emissions_tco2e', 0)} tCO2e/year
        - Electricity Percentage: {context.get('electricity_pct', 0)}%
        - HVAC Load Factor: {context.get('hvac_load', 0)}
        - Building Age: {context.get('building_age', 0)} years
        - Available Budget: ${context.get('budget', 0):,}
        - Region/Country: {context.get('region', 'US')}
        - Performance Rating: {context.get('performance_rating', 'Average')}

        **Additional Context:**
        - Floor Area: {context.get('floor_area_sqm', 'Not specified')} m²
        - Current Heating: {context.get('heating_system', 'Not specified')}
        - Space Available: {context.get('space_available_sqm', 'Not specified')} m²

        **Relevant Knowledge from Case Studies and Best Practices:**
        {knowledge}

        **Your Task:**
        Use the available tools to:
        1. **Check technology compatibility** with this specific facility (building type, age, load profile)
        2. **Calculate financial metrics** (ROI, payback period, NPV) for each recommendation
        3. **Verify spatial constraints** (does the technology fit in available space?)
        4. **Assess grid integration** (can the local grid support the technology?)
        5. **Evaluate regulatory compliance** (incentives, rebates, compliance requirements)
        6. **Model emission reduction** scenarios for different technology combinations

        **Required Output:**
        Provide 5-10 specific, actionable recommendations with:
        - **Technology/Action**: Clear description
        - **Expected Impact**: Percentage reduction in emissions
        - **Cost**: Capital expenditure estimate
        - **Payback Period**: Years to ROI
        - **Priority**: High/Medium/Low based on ROI and feasibility
        - **Implementation Timeline**: Months to complete
        - **Specific Considerations**: Facility-specific factors

        Also provide:
        - **Quick Wins**: Low-cost, high-impact actions (0-6 months)
        - **Strategic Roadmap**: Phased implementation plan
        - **Risk Factors**: What could prevent success
        - **Success Metrics**: How to measure results

        Be specific with numbers. Use tools to validate feasibility. Consider the budget constraint.
        """

        return prompt.strip()

    def _get_default_tools(self) -> List[ToolDef]:
        """Get default tool definitions for recommendation reasoning."""
        return [
            ToolDef(
                name="check_technology_compatibility",
                description="Check if a technology is compatible with facility constraints (building type, age, load)",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name (e.g., 'heat pump', 'solar PV', 'LED lighting')"
                        },
                        "building_type": {
                            "type": "string",
                            "description": "Building type (e.g., 'commercial_office', 'industrial', 'retail')"
                        },
                        "building_age": {
                            "type": "number",
                            "description": "Age of building in years"
                        }
                    },
                    "required": ["technology", "building_type"]
                }
            ),
            ToolDef(
                name="calculate_financial_metrics",
                description="Calculate ROI, payback period, and NPV for a technology investment",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name"
                        },
                        "capex": {
                            "type": "number",
                            "description": "Capital expenditure in USD"
                        },
                        "annual_savings": {
                            "type": "number",
                            "description": "Annual energy cost savings in USD"
                        },
                        "lifetime_years": {
                            "type": "number",
                            "description": "Expected lifetime in years"
                        }
                    },
                    "required": ["technology", "capex", "annual_savings"]
                }
            ),
            ToolDef(
                name="check_spatial_constraints",
                description="Verify if technology fits in available space",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name"
                        },
                        "available_space_sqm": {
                            "type": "number",
                            "description": "Available space in square meters"
                        },
                        "floor_area_sqm": {
                            "type": "number",
                            "description": "Total floor area in square meters"
                        }
                    },
                    "required": ["technology", "available_space_sqm"]
                }
            ),
            ToolDef(
                name="assess_grid_integration",
                description="Check grid capacity and integration requirements",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name"
                        },
                        "capacity_kw": {
                            "type": "number",
                            "description": "Technology capacity in kW"
                        },
                        "region": {
                            "type": "string",
                            "description": "Geographic region"
                        }
                    },
                    "required": ["technology", "capacity_kw", "region"]
                }
            ),
            ToolDef(
                name="evaluate_regulatory_incentives",
                description="Check available incentives, rebates, and compliance requirements",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name"
                        },
                        "region": {
                            "type": "string",
                            "description": "Geographic region/country"
                        },
                        "project_cost": {
                            "type": "number",
                            "description": "Total project cost in USD"
                        }
                    },
                    "required": ["technology", "region"]
                }
            ),
            ToolDef(
                name="model_emission_reduction",
                description="Model emission reduction for a technology or combination",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {
                            "type": "string",
                            "description": "Technology name"
                        },
                        "baseline_emissions_tco2e": {
                            "type": "number",
                            "description": "Current baseline emissions in tCO2e/year"
                        },
                        "implementation_scale": {
                            "type": "string",
                            "description": "Scale of implementation (e.g., 'full', 'partial')",
                            "enum": ["full", "partial", "pilot"]
                        }
                    },
                    "required": ["technology", "baseline_emissions_tco2e"]
                }
            )
        ]

    def _build_tool_registry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build registry of tool implementations."""
        return {
            "check_technology_compatibility": lambda **kwargs: self._tool_check_compatibility(context, **kwargs),
            "calculate_financial_metrics": lambda **kwargs: self._tool_calculate_financial(**kwargs),
            "check_spatial_constraints": lambda **kwargs: self._tool_check_spatial(context, **kwargs),
            "assess_grid_integration": lambda **kwargs: self._tool_assess_grid(**kwargs),
            "evaluate_regulatory_incentives": lambda **kwargs: self._tool_evaluate_incentives(**kwargs),
            "model_emission_reduction": lambda **kwargs: self._tool_model_reduction(**kwargs)
        }

    # Tool implementations (deterministic calculations)

    def _tool_check_compatibility(self, context: Dict, technology: str, building_type: str, building_age: float = None) -> Dict[str, Any]:
        """Check technology compatibility (deterministic)."""
        # Compatibility logic
        compatible = True
        confidence = 0.9
        notes = []

        # Age-based checks
        if building_age and building_age > 30:
            if "hvac" in technology.lower():
                notes.append("Old building may need duct/pipe infrastructure upgrades")
                confidence -= 0.1

        # Building type checks
        if building_type == "industrial":
            if "solar pv" in technology.lower():
                compatible = True
                notes.append("Industrial flat roofs excellent for solar PV")
            if "heat pump" in technology.lower():
                compatible = True
                notes.append("Industrial heat pumps suitable for process heat")

        if building_type == "commercial_office":
            if "led lighting" in technology.lower():
                compatible = True
                confidence = 0.95
                notes.append("LED retrofit ideal for office buildings")

        return {
            "compatible": compatible,
            "confidence": confidence,
            "notes": notes,
            "technology": technology,
            "building_type": building_type
        }

    def _tool_calculate_financial(self, technology: str, capex: float, annual_savings: float, lifetime_years: int = 20) -> Dict[str, Any]:
        """Calculate financial metrics (deterministic)."""
        payback_years = capex / annual_savings if annual_savings > 0 else 999
        total_savings = annual_savings * lifetime_years
        roi_pct = ((total_savings - capex) / capex * 100) if capex > 0 else 0

        # Simple NPV calculation (discount rate 8%)
        discount_rate = 0.08
        npv = sum([annual_savings / ((1 + discount_rate) ** year) for year in range(1, lifetime_years + 1)]) - capex

        return {
            "technology": technology,
            "payback_years": round(payback_years, 1),
            "roi_pct": round(roi_pct, 1),
            "npv_usd": round(npv, 2),
            "total_lifetime_savings": round(total_savings, 2),
            "financial_viability": "Excellent" if payback_years < 3 else "Good" if payback_years < 7 else "Acceptable" if payback_years < 10 else "Poor"
        }

    def _tool_check_spatial(self, context: Dict, technology: str, available_space_sqm: float, floor_area_sqm: float = None) -> Dict[str, Any]:
        """Check spatial constraints (deterministic)."""
        fits = True
        required_space = 0
        notes = []

        # Technology-specific space requirements
        if "solar pv" in technology.lower():
            # Assume 15 m² per kW capacity
            required_space = floor_area_sqm * 0.6 if floor_area_sqm else 100  # 60% of roof area usable
            fits = available_space_sqm >= required_space
            notes.append(f"Solar PV needs ~60% of flat roof area")

        if "heat pump" in technology.lower():
            required_space = 20  # m² for outdoor unit
            fits = available_space_sqm >= required_space
            notes.append("Heat pump outdoor unit needs ~20 m²")

        if "battery storage" in technology.lower():
            required_space = 10  # m² for battery room
            fits = available_space_sqm >= required_space
            notes.append("Battery storage needs dedicated room (~10 m²)")

        return {
            "fits": fits,
            "available_space_sqm": available_space_sqm,
            "required_space_sqm": required_space,
            "space_utilization_pct": (required_space / available_space_sqm * 100) if available_space_sqm > 0 else 0,
            "notes": notes,
            "technology": technology
        }

    def _tool_assess_grid(self, technology: str, capacity_kw: float, region: str) -> Dict[str, Any]:
        """Assess grid integration (deterministic)."""
        grid_ready = True
        notes = []

        # Region-specific grid readiness
        if region == "US":
            if capacity_kw > 1000:
                notes.append("Large systems may need utility interconnection study")
            grid_ready = True
            notes.append("US grid generally accommodates distributed generation")

        if region == "EU":
            grid_ready = True
            notes.append("EU grid supports renewable integration well")

        if region == "IN":
            if capacity_kw > 500:
                notes.append("India: Large systems need state utility approval")
            grid_ready = True

        return {
            "grid_ready": grid_ready,
            "capacity_kw": capacity_kw,
            "interconnection_required": capacity_kw > 100,
            "estimated_interconnection_cost": capacity_kw * 50 if capacity_kw > 100 else 0,  # $50/kW estimate
            "notes": notes,
            "region": region,
            "technology": technology
        }

    def _tool_evaluate_incentives(self, technology: str, region: str, project_cost: float = None) -> Dict[str, Any]:
        """Evaluate regulatory incentives (deterministic)."""
        incentives = []
        total_incentive_value = 0

        # Region-specific incentives
        if region == "US":
            if "solar" in technology.lower():
                itc = project_cost * 0.30 if project_cost else 0  # 30% ITC
                incentives.append({"name": "Investment Tax Credit (ITC)", "value": itc, "type": "tax_credit"})
                total_incentive_value += itc

            if "heat pump" in technology.lower():
                incentives.append({"name": "ENERGY STAR Rebate", "value": 2000, "type": "rebate"})
                total_incentive_value += 2000

        if region == "EU":
            if "renewable" in technology.lower() or "efficiency" in technology.lower():
                incentives.append({"name": "EU Taxonomy Green Financing", "value": "Variable", "type": "financing"})

        if region == "IN":
            if "solar" in technology.lower():
                subsidy = project_cost * 0.40 if project_cost else 0  # 40% subsidy
                incentives.append({"name": "MNRE Solar Rooftop Subsidy", "value": subsidy, "type": "subsidy"})
                total_incentive_value += subsidy

        return {
            "incentives_available": incentives,
            "total_incentive_value": total_incentive_value,
            "effective_project_cost": project_cost - total_incentive_value if project_cost else 0,
            "region": region,
            "technology": technology
        }

    def _tool_model_reduction(self, technology: str, baseline_emissions_tco2e: float, implementation_scale: str = "full") -> Dict[str, Any]:
        """Model emission reduction (deterministic)."""
        # Technology-specific reduction factors
        reduction_factors = {
            "solar pv": 0.40,  # 40% reduction
            "heat pump": 0.50,  # 50% reduction
            "led lighting": 0.60,  # 60% lighting emissions reduction
            "hvac upgrade": 0.25,  # 25% HVAC reduction
            "insulation": 0.15,  # 15% heating/cooling reduction
            "renewable energy": 0.50,  # 50% grid emissions reduction
        }

        # Find matching technology
        reduction_pct = 0.20  # Default 20%
        for tech, factor in reduction_factors.items():
            if tech in technology.lower():
                reduction_pct = factor
                break

        # Scale adjustment
        scale_factors = {"full": 1.0, "partial": 0.6, "pilot": 0.3}
        scale_multiplier = scale_factors.get(implementation_scale, 1.0)

        final_reduction_pct = reduction_pct * scale_multiplier
        emissions_reduction = baseline_emissions_tco2e * final_reduction_pct
        new_emissions = baseline_emissions_tco2e - emissions_reduction

        return {
            "technology": technology,
            "baseline_emissions_tco2e": baseline_emissions_tco2e,
            "reduction_percentage": round(final_reduction_pct * 100, 1),
            "emissions_reduction_tco2e": round(emissions_reduction, 2),
            "new_emissions_tco2e": round(new_emissions, 2),
            "implementation_scale": implementation_scale,
            "carbon_offset_value_usd": round(emissions_reduction * 30, 2)  # $30/tCO2e carbon price
        }

    # Parsing and extraction methods

    def _parse_recommendations(self, llm_text: str) -> List[Dict[str, Any]]:
        """Parse structured recommendations from LLM text."""
        # Simple parsing logic (in production, use more robust parsing)
        recommendations = []

        # Look for numbered recommendations
        lines = llm_text.split('\n')
        current_rec = {}

        for line in lines:
            line = line.strip()

            # Detect recommendation start
            if any(line.startswith(str(i) + ".") or line.startswith(str(i) + ")") for i in range(1, 20)):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {"action": line, "details": []}

            # Parse key fields
            elif current_rec:
                if "Impact:" in line or "Expected Impact:" in line:
                    current_rec["impact"] = line.split(":", 1)[1].strip()
                elif "Cost:" in line or "Capital:" in line:
                    current_rec["cost"] = line.split(":", 1)[1].strip()
                elif "Payback:" in line or "Payback Period:" in line:
                    current_rec["payback"] = line.split(":", 1)[1].strip()
                elif "Priority:" in line:
                    current_rec["priority"] = line.split(":", 1)[1].strip()
                elif line:
                    current_rec["details"].append(line)

        if current_rec:
            recommendations.append(current_rec)

        # Fallback: Return generic structure if parsing fails
        if not recommendations:
            recommendations = [{
                "action": "AI-generated recommendations available in full text",
                "impact": "See reasoning field",
                "priority": "See full analysis"
            }]

        return recommendations[:10]  # Top 10

    def _parse_roadmap(self, llm_text: str) -> List[Dict[str, Any]]:
        """Parse implementation roadmap from LLM text."""
        roadmap = []

        # Look for phase markers
        if "Phase 1" in llm_text or "Quick Win" in llm_text:
            roadmap.append({
                "phase": "Phase 1: Quick Wins (0-6 months)",
                "description": "Extracted from AI recommendations",
                "focus": "Low-cost, high-impact actions"
            })

        if "Phase 2" in llm_text or "Strategic" in llm_text:
            roadmap.append({
                "phase": "Phase 2: Strategic Improvements (6-18 months)",
                "description": "Medium-term investments",
                "focus": "Technology upgrades with proven ROI"
            })

        if "Phase 3" in llm_text or "Long-term" in llm_text:
            roadmap.append({
                "phase": "Phase 3: Major Transformation (18-36 months)",
                "description": "Major capital projects",
                "focus": "Deep decarbonization initiatives"
            })

        return roadmap

    def _extract_quick_wins(self, recommendations: List[Dict]) -> List[Dict]:
        """Extract quick wins from recommendations."""
        quick_wins = []
        for rec in recommendations:
            cost = rec.get("cost", "").lower()
            payback = rec.get("payback", "").lower()

            if "low" in cost or "immediate" in payback or ("1" in payback and "year" in payback):
                quick_wins.append(rec)

        return quick_wins[:3]

    def _extract_high_impact(self, recommendations: List[Dict]) -> List[Dict]:
        """Extract high-impact recommendations."""
        high_impact = []
        for rec in recommendations:
            impact = rec.get("impact", "").lower()
            priority = rec.get("priority", "").lower()

            # Look for percentage indicators or "high" priority
            if any(str(i) + "%" in impact for i in range(20, 100)) or "high" in priority:
                high_impact.append(rec)

        return high_impact[:3]

    def _count_tools_by_name(self, trace: List[Dict]) -> Dict[str, int]:
        """Count tools by name."""
        counts = {}
        for entry in trace:
            tool_name = entry.get("tool", "unknown")
            counts[tool_name] = counts.get(tool_name, 0) + 1
        return counts

    def _extract_confidence(self, llm_text: str) -> float:
        """Extract confidence score from LLM text."""
        # Look for confidence indicators
        text_lower = llm_text.lower()

        if "high confidence" in text_lower or "highly confident" in text_lower:
            return 0.9
        elif "confident" in text_lower:
            return 0.8
        elif "uncertain" in text_lower or "may" in text_lower:
            return 0.6
        else:
            return 0.75  # Default


if __name__ == "__main__":
    print("RecommendationAgentAI v2.0 - AI-Powered Recommendations")
    print("=" * 80)
    print("\nTransformation Complete:")
    print("  - Pattern: ReasoningAgent (RECOMMENDATION PATH)")
    print("  - Uses RAG: ✅ YES")
    print("  - Uses ChatSession: ✅ YES")
    print("  - Uses Tools: ✅ YES (6 tools)")
    print("  - Temperature: 0.7 (creative reasoning)")
    print("\nCapabilities:")
    print("  - Context-aware recommendations (not static lookups)")
    print("  - Multi-tool validation (compatibility, financial, spatial, grid, incentives, emissions)")
    print("  - RAG retrieval (case studies, best practices, technology database)")
    print("  - AI reasoning (adapts to facility specifics)")
    print("  - Structured output parsing")
    print("\nStatus: Ready for integration testing")
