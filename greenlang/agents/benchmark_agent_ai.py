"""
AI-Powered Benchmark Agent with Competitive Insights
GL Intelligence Infrastructure - INSIGHT PATH

Transformation from v1 (static thresholds) to v2 (AI insights):
- BEFORE: Static benchmarks → Simple rating → Generic recommendations
- AFTER: Deterministic calculations + AI competitive analysis + RAG-based insights

Pattern: InsightAgent (hybrid architecture)
- calculate(): Deterministic peer comparison, intensity, percentile
- explain(): AI-generated competitive insights with RAG context

Version: 2.0.0
Date: 2025-11-06
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import statistics

from greenlang.agents.base_agents import InsightAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata


@dataclass
class BenchmarkCalculation:
    """Deterministic benchmark calculation results."""
    carbon_intensity: float
    unit: str
    rating: str
    percentile: int
    benchmarks: Dict[str, float]
    comparison: Dict[str, float]
    building_type: str
    annualized_emissions_kg: float
    building_area_sqft: float
    calculation_trace: List[str]


class BenchmarkAgentAI(InsightAgent):
    """
    AI-powered benchmark agent with hybrid architecture.

    DETERMINISTIC CALCULATIONS (calculate method):
    - Carbon intensity computation
    - Rating assignment (excellent/good/average/poor)
    - Percentile estimation
    - Benchmark comparisons

    AI-POWERED INSIGHTS (explain method):
    - Competitive positioning analysis
    - Root cause investigation (why this rating?)
    - Peer improvement strategies (what did they do?)
    - Market context and trends
    - Actionable improvement pathways

    RAG Collections Used:
    - industry_benchmarks: Real peer data and case studies
    - best_practices: Documented improvement strategies
    - competitive_analysis: Market positioning insights
    - building_performance: Historical trends and patterns

    Temperature: 0.6 (consistency over creativity for insights)
    """

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="benchmark_agent_ai_v2",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=False,  # No tools needed for insights
        critical_for_compliance=False,
        transformation_priority="HIGH (Phase 2.2 transformation)",
        description="Hybrid agent: deterministic benchmarking + AI competitive insights"
    )

    # Static benchmark thresholds (industry standards)
    BENCHMARKS = {
        "commercial_office": {
            "excellent": 20,
            "good": 35,
            "average": 50,
            "poor": 70,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "retail": {
            "excellent": 25,
            "good": 40,
            "average": 55,
            "poor": 75,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "warehouse": {
            "excellent": 15,
            "good": 25,
            "average": 35,
            "poor": 50,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "residential": {
            "excellent": 15,
            "good": 25,
            "average": 35,
            "poor": 45,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "manufacturing": {
            "excellent": 40,
            "good": 65,
            "average": 90,
            "poor": 120,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "data_center": {
            "excellent": 80,
            "good": 120,
            "average": 160,
            "poor": 200,
            "unit": "kg_co2e_per_sqft_per_year",
        },
    }

    def __init__(self, enable_audit_trail: bool = True):
        """
        Initialize AI-powered benchmark agent.

        Args:
            enable_audit_trail: Whether to capture calculation audit trail
        """
        super().__init__(enable_audit_trail=enable_audit_trail)

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic benchmark calculations.

        This method is DETERMINISTIC and FAST:
        - Same inputs → same outputs
        - No AI, no network calls
        - Pure mathematical computations
        - Full audit trail

        Args:
            inputs: {
                "building_type": str (e.g., "commercial_office"),
                "total_emissions_kg": float,
                "building_area": float (sqft),
                "period_months": int (default: 12),
                "region": str (optional),
                "building_age": int (optional)
            }

        Returns:
            BenchmarkCalculation with all metrics
        """
        calculation_trace = []

        # Extract inputs
        building_type = inputs.get("building_type", "commercial_office")
        total_emissions_kg = inputs.get("total_emissions_kg", 0)
        building_area = inputs.get("building_area", 0)
        period_months = inputs.get("period_months", 12)

        calculation_trace.append(f"Building Type: {building_type}")
        calculation_trace.append(f"Total Emissions: {total_emissions_kg} kg CO2e")
        calculation_trace.append(f"Building Area: {building_area} sqft")
        calculation_trace.append(f"Period: {period_months} months")

        # Validation
        if building_area <= 0:
            raise ValueError("Building area must be greater than 0")

        # Step 1: Annualize emissions
        annualized_emissions = (total_emissions_kg / period_months) * 12
        calculation_trace.append(
            f"Annualized Emissions = ({total_emissions_kg} / {period_months}) * 12 = {annualized_emissions:.2f} kg CO2e/year"
        )

        # Step 2: Calculate intensity
        intensity = annualized_emissions / building_area
        calculation_trace.append(
            f"Carbon Intensity = {annualized_emissions:.2f} / {building_area} = {intensity:.2f} kg CO2e/sqft/year"
        )

        # Step 3: Get benchmarks
        benchmarks = self.BENCHMARKS.get(
            building_type, self.BENCHMARKS["commercial_office"]
        )
        calculation_trace.append(f"Benchmarks for {building_type}: {benchmarks}")

        # Step 4: Determine rating
        rating = self._get_rating(intensity, benchmarks)
        calculation_trace.append(f"Rating: {rating}")

        # Step 5: Estimate percentile
        percentile = self._estimate_percentile(intensity, benchmarks)
        calculation_trace.append(f"Percentile: {percentile}th")

        # Step 6: Calculate comparisons
        comparison = {
            "vs_excellent": round(intensity - benchmarks["excellent"], 2),
            "vs_good": round(intensity - benchmarks["good"], 2),
            "vs_average": round(intensity - benchmarks["average"], 2),
            "vs_poor": round(intensity - benchmarks["poor"], 2),
            "improvement_to_good": max(0, round(intensity - benchmarks["good"], 2)),
            "improvement_to_excellent": max(
                0, round(intensity - benchmarks["excellent"], 2)
            ),
        }
        calculation_trace.append(f"Comparisons: {comparison}")

        # Capture audit trail
        result = {
            "carbon_intensity": round(intensity, 2),
            "unit": benchmarks["unit"],
            "rating": rating,
            "percentile": percentile,
            "benchmarks": benchmarks,
            "comparison": comparison,
            "building_type": building_type,
            "annualized_emissions_kg": round(annualized_emissions, 2),
            "building_area_sqft": building_area,
            "calculation_trace": calculation_trace,
        }

        self._capture_calculation_audit(
            operation="benchmark_calculation",
            inputs=inputs,
            outputs=result,
            calculation_trace=calculation_trace,
        )

        return result

    async def explain(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6,
    ) -> str:
        """
        Generate AI-powered competitive insights.

        This method uses AI to explain WHY the building performs at this level:
        - Competitive positioning in market
        - Root causes of current rating
        - What similar peers have done to improve
        - Specific improvement pathways based on real case studies
        - Market trends and future outlook

        Args:
            calculation_result: Output from calculate() method
            context: Additional context (location, industry, goals)
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature (default 0.6 for consistency)

        Returns:
            Natural language competitive analysis and insights
        """
        # Step 1: Build RAG query
        rag_query = self._build_rag_query(calculation_result, context)

        # Step 2: RAG retrieval for peer insights
        rag_result = await self._rag_retrieve(
            query=rag_query,
            rag_engine=rag_engine,
            collections=[
                "industry_benchmarks",
                "best_practices",
                "competitive_analysis",
                "building_performance",
            ],
            top_k=6,
        )

        # Step 3: Format RAG context
        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 4: Build insight prompt
        insight_prompt = self._build_insight_prompt(
            calculation_result, context, formatted_knowledge
        )

        # Step 5: Generate insights with ChatSession
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": """You are a competitive analyst specializing in building energy performance and carbon emissions.

Your role is to provide ACTIONABLE COMPETITIVE INSIGHTS based on deterministic calculations and real peer data.

Focus on:
1. COMPETITIVE POSITIONING: Where does this building stand in the market?
2. ROOT CAUSE ANALYSIS: Why is the building at this performance level?
3. PEER STRATEGIES: What have similar buildings done to improve?
4. IMPROVEMENT PATHWAYS: Specific, evidence-based recommendations
5. MARKET TRENDS: Industry direction and future outlook

Use the provided RAG context (real case studies and peer data) to ground your insights.
Be specific, use numbers from the calculations, and cite peer examples.

DO NOT recalculate metrics - use the provided calculation results.
DO NOT provide generic advice - use the RAG context for specific insights.""",
                },
                {"role": "user", "content": insight_prompt},
            ],
            temperature=temperature,
        )

        # Step 6: Parse and structure the insights
        insights_text = response.text if hasattr(response, "text") else str(response)

        return insights_text

    def _get_rating(self, intensity: float, benchmarks: Dict) -> str:
        """Deterministic rating assignment."""
        if intensity <= benchmarks["excellent"]:
            return "Excellent"
        elif intensity <= benchmarks["good"]:
            return "Good"
        elif intensity <= benchmarks["average"]:
            return "Average"
        elif intensity <= benchmarks["poor"]:
            return "Below Average"
        else:
            return "Poor"

    def _estimate_percentile(self, intensity: float, benchmarks: Dict) -> int:
        """Deterministic percentile estimation."""
        if intensity <= benchmarks["excellent"]:
            return 90
        elif intensity <= benchmarks["good"]:
            return 70
        elif intensity <= benchmarks["average"]:
            return 50
        elif intensity <= benchmarks["poor"]:
            return 30
        else:
            return 10

    def _build_rag_query(
        self, calculation_result: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Build semantic search query for RAG retrieval."""
        building_type = calculation_result["building_type"]
        rating = calculation_result["rating"]
        intensity = calculation_result["carbon_intensity"]
        percentile = calculation_result["percentile"]

        region = context.get("region", "")
        industry = context.get("industry", "")

        # Build contextual query
        query = f"""
Building Type: {building_type}
Current Performance: {rating} ({percentile}th percentile)
Carbon Intensity: {intensity} kg CO2e/sqft/year

Query:
1. What are the typical characteristics of {building_type} buildings at the {percentile}th percentile?
2. What improvement strategies have similar buildings used to move from {rating} to better ratings?
3. What are the common root causes for {building_type} buildings performing at this level?
4. What peer case studies exist for {building_type} improvement projects?
"""

        if region:
            query += f"5. What regional factors in {region} affect building performance?\n"

        if industry:
            query += f"6. What industry-specific factors in {industry} influence emissions?\n"

        return query.strip()

    def _build_insight_prompt(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        rag_context: str,
    ) -> str:
        """Build the prompt for AI insight generation."""
        return f"""
## BENCHMARK CALCULATION RESULTS (DETERMINISTIC)

**Building Type:** {calculation_result['building_type']}
**Carbon Intensity:** {calculation_result['carbon_intensity']} kg CO2e/sqft/year
**Rating:** {calculation_result['rating']}
**Percentile:** {calculation_result['percentile']}th percentile

**Benchmark Comparisons:**
- vs. Excellent ({calculation_result['benchmarks']['excellent']}): {calculation_result['comparison']['vs_excellent']:+.2f} kg CO2e/sqft/year
- vs. Good ({calculation_result['benchmarks']['good']}): {calculation_result['comparison']['vs_good']:+.2f} kg CO2e/sqft/year
- vs. Average ({calculation_result['benchmarks']['average']}): {calculation_result['comparison']['vs_average']:+.2f} kg CO2e/sqft/year

**Improvement Needed for 'Good' Rating:** {calculation_result['comparison']['improvement_to_good']} kg CO2e/sqft/year

---

## ADDITIONAL CONTEXT

{self._format_context(context)}

---

## PEER KNOWLEDGE BASE (RAG RETRIEVAL)

{rag_context}

---

## YOUR TASK

Provide a comprehensive competitive analysis covering:

### 1. Competitive Positioning (2-3 sentences)
Where does this building stand relative to peers? What does the {calculation_result['percentile']}th percentile mean in practical terms?

### 2. Root Cause Analysis (3-4 bullet points)
Why is this building performing at this level? What are the likely drivers of current performance?

### 3. Peer Improvement Strategies (3-5 bullet points)
What have similar buildings done to improve? Cite specific examples from the RAG context.

### 4. Recommended Improvement Pathway (numbered steps)
Provide a specific, evidence-based roadmap to reach 'Good' or 'Excellent' rating.

### 5. Market Context (2-3 sentences)
Industry trends, regulatory drivers, competitive landscape.

Use specific numbers from the calculations. Ground recommendations in the RAG context.
"""

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format additional context for the prompt."""
        formatted = []
        if "region" in context:
            formatted.append(f"**Region:** {context['region']}")
        if "industry" in context:
            formatted.append(f"**Industry:** {context['industry']}")
        if "building_age" in context:
            formatted.append(f"**Building Age:** {context['building_age']} years")
        if "recent_upgrades" in context:
            formatted.append(f"**Recent Upgrades:** {context['recent_upgrades']}")
        if "improvement_goals" in context:
            formatted.append(f"**Goals:** {context['improvement_goals']}")

        return "\n".join(formatted) if formatted else "No additional context provided."


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("AI-Powered Benchmark Agent - INSIGHT PATH")
    print("=" * 80)

    # Initialize agent
    agent = BenchmarkAgentAI(enable_audit_trail=True)

    print("\n✓ Agent initialized with InsightAgent pattern")
    print(f"✓ Category: {agent.category}")
    print(f"✓ Uses ChatSession: {agent.metadata.uses_chat_session}")
    print(f"✓ Uses RAG: {agent.metadata.uses_rag}")
    print(f"✓ Temperature: 0.6 (consistency for insights)")

    # Test calculation (deterministic)
    print("\n" + "=" * 80)
    print("TEST 1: DETERMINISTIC CALCULATION")
    print("=" * 80)

    test_inputs = {
        "building_type": "commercial_office",
        "total_emissions_kg": 250000,
        "building_area": 5000,
        "period_months": 12,
    }

    print(f"\nInputs: {test_inputs}")

    result = agent.calculate(test_inputs)

    print(f"\n✓ Carbon Intensity: {result['carbon_intensity']} {result['unit']}")
    print(f"✓ Rating: {result['rating']}")
    print(f"✓ Percentile: {result['percentile']}th")
    print(f"✓ Improvement to 'Good': {result['comparison']['improvement_to_good']} kg CO2e/sqft/year")

    print("\nCalculation Trace:")
    for i, step in enumerate(result["calculation_trace"], 1):
        print(f"  {i}. {step}")

    # Test AI insights (requires ChatSession and RAGEngine)
    print("\n" + "=" * 80)
    print("TEST 2: AI INSIGHT GENERATION (requires live infrastructure)")
    print("=" * 80)

    print("\n⚠ AI insight generation requires:")
    print("  - ChatSession instance (LLM API)")
    print("  - RAGEngine instance (vector database)")
    print("  - Knowledge base with collections:")
    print("    * industry_benchmarks")
    print("    * best_practices")
    print("    * competitive_analysis")
    print("    * building_performance")

    print("\nExample async call:")
    print("""
    insights = await agent.explain(
        calculation_result=result,
        context={
            "region": "California",
            "industry": "Technology",
            "building_age": 15,
            "improvement_goals": "Reach 'Good' rating by 2026"
        },
        session=chat_session,
        rag_engine=rag_engine,
        temperature=0.6
    )
    """)

    # Verify reproducibility
    print("\n" + "=" * 80)
    print("TEST 3: REPRODUCIBILITY VERIFICATION")
    print("=" * 80)

    result2 = agent.calculate(test_inputs)
    is_reproducible = result == result2

    print(f"\n✓ Same inputs produce same outputs: {is_reproducible}")

    if agent.enable_audit_trail:
        print(f"✓ Audit trail entries: {len(agent.audit_trail)}")
        print("\nLatest audit entry:")
        latest = agent.audit_trail[-1]
        print(f"  - Timestamp: {latest.timestamp}")
        print(f"  - Operation: {latest.operation}")
        print(f"  - Input Hash: {latest.input_hash[:16]}...")
        print(f"  - Output Hash: {latest.output_hash[:16]}...")

    print("\n" + "=" * 80)
    print("TRANSFORMATION SUMMARY")
    print("=" * 80)
    print("\nPattern: InsightAgent (Hybrid Architecture)")
    print("  - calculate(): Deterministic peer comparison (KEEP)")
    print("  - explain(): AI competitive insights (NEW)")
    print("\nWhat Changed:")
    print("  - BEFORE: Static thresholds → Generic recommendations")
    print("  - AFTER: Same thresholds + AI competitive analysis")
    print("\nAI Value-Add:")
    print("  ✓ Root cause investigation (why this rating?)")
    print("  ✓ Peer improvement strategies (what did they do?)")
    print("  ✓ Competitive positioning insights")
    print("  ✓ Context-aware recommendations")
    print("  ✓ Market trends and outlook")
    print("\nCompliance:")
    print("  ✓ Calculations remain deterministic (regulatory safe)")
    print("  ✓ AI only used for narrative insights")
    print("  ✓ Full audit trail for calculations")
    print("=" * 80)
