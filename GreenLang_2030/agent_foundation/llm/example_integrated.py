"""
Comprehensive LLM Integration Example

This example demonstrates the complete GreenLang LLM infrastructure:
1. Multi-provider setup (Anthropic + OpenAI)
2. Intelligent routing with automatic failover
3. Cost tracking with budget alerts
4. Circuit breaker protection
5. Health monitoring
6. Real-world usage scenarios

Run with:
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    python example_integrated.py
"""

import asyncio
import logging
import os
from datetime import datetime

from cost_tracker import CostTracker
from llm_router import LLMRouter, RoutingStrategy
from providers.anthropic_provider import AnthropicProvider
from providers.base_provider import GenerationRequest
from providers.openai_provider import OpenAIProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ESGAnalysisAgent:
    """
    Example ESG Analysis Agent using GreenLang LLM infrastructure.

    This agent demonstrates:
    - Multi-provider routing with cost optimization
    - Budget tracking per tenant
    - Automatic failover on provider failures
    - Cost reporting and analytics
    """

    def __init__(self, tenant_id: str, monthly_budget_usd: float):
        """
        Initialize ESG Analysis Agent.

        Args:
            tenant_id: Tenant/customer ID
            monthly_budget_usd: Monthly LLM budget in USD
        """
        self.tenant_id = tenant_id
        self.agent_id = f"esg-agent-{tenant_id}"

        # Initialize router with least-cost strategy
        self.router = LLMRouter(
            strategy=RoutingStrategy.LEAST_COST,
            health_check_interval=30.0,
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0,
        )

        # Initialize cost tracker
        self.cost_tracker = CostTracker()
        self.cost_tracker.set_budget(
            tenant_id=tenant_id,
            monthly_limit_usd=monthly_budget_usd,
            alert_thresholds=[0.8, 0.9, 1.0],  # Alert at 80%, 90%, 100%
        )

        # Register alert callback
        self.cost_tracker.register_alert_callback(self._on_budget_alert)

        logger.info(
            f"Initialized ESG Agent for tenant {tenant_id} "
            f"with ${monthly_budget_usd:.2f} monthly budget"
        )

    def _on_budget_alert(
        self, tenant_id: str, threshold: float, current_cost: float
    ):
        """Handle budget alerts."""
        logger.warning(
            f"BUDGET ALERT: Tenant {tenant_id} has used {threshold * 100:.0f}% "
            f"of monthly budget (${current_cost:.2f})"
        )

    async def setup_providers(self):
        """Register LLM providers."""
        # Register Anthropic (Claude) - Priority 1
        try:
            anthropic = AnthropicProvider(
                model_id="claude-3-sonnet-20240229",  # Cost-effective model
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            self.router.register_provider("anthropic", anthropic, priority=1)
            logger.info("Registered Anthropic provider")
        except Exception as e:
            logger.warning(f"Could not register Anthropic: {e}")

        # Register OpenAI (GPT) - Priority 2 (fallback)
        try:
            openai = OpenAIProvider(
                model_id="gpt-3.5-turbo",  # Cost-effective model
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            self.router.register_provider("openai", openai, priority=2)
            logger.info("Registered OpenAI provider")
        except Exception as e:
            logger.warning(f"Could not register OpenAI: {e}")

        # Start health monitoring
        await self.router.start_health_monitoring()

    async def analyze_emissions(self, company_data: str) -> dict:
        """
        Analyze emissions data using LLM.

        Args:
            company_data: Company emissions data to analyze

        Returns:
            Analysis results with cost tracking
        """
        # Check budget before processing
        budget_status = self.cost_tracker.check_budget(self.tenant_id)
        if budget_status and budget_status.percentage_used >= 100:
            raise Exception(
                f"Monthly budget exceeded: ${budget_status.current_cost_usd:.2f}/"
                f"${budget_status.monthly_limit_usd:.2f}"
            )

        # Create generation request
        request = GenerationRequest(
            prompt=f"""Analyze the following company emissions data and provide:
1. Key findings (3-5 bullet points)
2. Risk assessment (high/medium/low)
3. Compliance recommendations

Data:
{company_data}

Provide your analysis in a structured format.""",
            temperature=0.3,  # Lower temperature for factual analysis
            max_tokens=500,
            system_prompt="You are an ESG compliance expert specializing in emissions analysis.",
        )

        # Generate with automatic provider selection and failover
        response = await self.router.generate(request)

        # Track costs
        self.cost_tracker.track_usage(
            provider=response.provider,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            model_id=response.model_id,
            usage=response.usage,
            request_id=response.metadata.get("completion_id"),
            metadata={"analysis_type": "emissions"},
        )

        logger.info(
            f"Analysis completed using {response.provider}: "
            f"${response.usage.total_cost_usd:.4f}, "
            f"{response.generation_time_ms:.0f}ms"
        )

        return {
            "analysis": response.text,
            "provider_used": response.provider,
            "model_used": response.model_id,
            "cost_usd": response.usage.total_cost_usd,
            "tokens_used": response.usage.total_tokens,
            "latency_ms": response.generation_time_ms,
        }

    async def get_usage_report(self) -> dict:
        """Get usage and cost report for this agent."""
        # Get cost summary
        summary = self.cost_tracker.get_summary(tenant_id=self.tenant_id)

        # Get budget status
        budget_status = self.cost_tracker.check_budget(self.tenant_id)

        # Get router metrics
        router_metrics = self.router.get_metrics()

        return {
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "cost_summary": {
                "total_cost_usd": summary.total_cost_usd,
                "total_tokens": summary.total_tokens,
                "total_requests": summary.total_requests,
                "by_provider": summary.breakdown_by_provider,
                "by_model": summary.breakdown_by_model,
            },
            "budget_status": {
                "monthly_limit": budget_status.monthly_limit_usd
                if budget_status
                else None,
                "current_cost": budget_status.current_cost_usd
                if budget_status
                else None,
                "percentage_used": budget_status.percentage_used
                if budget_status
                else None,
                "remaining": budget_status.remaining_usd if budget_status else None,
                "projected_monthly": budget_status.projected_monthly_cost
                if budget_status
                else None,
            },
            "router_metrics": router_metrics,
        }

    async def cleanup(self):
        """Cleanup resources."""
        await self.router.close()
        logger.info("Agent cleanup complete")


async def main():
    """Demonstrate complete LLM integration."""
    print("=" * 70)
    print("GreenLang LLM Integration - Complete Example")
    print("=" * 70)

    # Initialize agent with $100 monthly budget
    agent = ESGAnalysisAgent(tenant_id="acme-corp", monthly_budget_usd=100.0)

    try:
        # Setup providers
        print("\n[1] Setting up LLM providers...")
        await agent.setup_providers()

        # Wait for health checks
        await asyncio.sleep(2)

        # Example 1: Analyze emissions data
        print("\n[2] Analyzing emissions data...")
        company_data = """
        Company: ACME Manufacturing Inc.
        Scope 1 Emissions: 15,000 tonnes CO2e
        Scope 2 Emissions: 8,500 tonnes CO2e
        Scope 3 Emissions: 45,000 tonnes CO2e
        Total: 68,500 tonnes CO2e
        Year: 2024
        """

        result1 = await agent.analyze_emissions(company_data)
        print(f"\nAnalysis Result:")
        print(f"  Provider: {result1['provider_used']}")
        print(f"  Model: {result1['model_used']}")
        print(f"  Cost: ${result1['cost_usd']:.4f}")
        print(f"  Tokens: {result1['tokens_used']}")
        print(f"  Latency: {result1['latency_ms']:.0f}ms")
        print(f"\n{result1['analysis'][:300]}...")

        # Example 2: Multiple analyses (demonstrate cost tracking)
        print("\n[3] Running multiple analyses...")
        for i in range(3):
            result = await agent.analyze_emissions(company_data)
            print(f"  Request {i+1}: ${result['cost_usd']:.4f} via {result['provider_used']}")
            await asyncio.sleep(0.5)

        # Example 3: Get usage report
        print("\n[4] Usage Report:")
        report = await agent.get_usage_report()

        print(f"\nCost Summary:")
        print(f"  Total Cost: ${report['cost_summary']['total_cost_usd']:.4f}")
        print(f"  Total Tokens: {report['cost_summary']['total_tokens']:,}")
        print(f"  Total Requests: {report['cost_summary']['total_requests']}")

        print(f"\nBy Provider:")
        for provider, cost in report['cost_summary']['by_provider'].items():
            print(f"  {provider}: ${cost:.4f}")

        print(f"\nBudget Status:")
        print(
            f"  Monthly Limit: ${report['budget_status']['monthly_limit']:.2f}"
        )
        print(
            f"  Current Cost: ${report['budget_status']['current_cost']:.2f}"
        )
        print(
            f"  Usage: {report['budget_status']['percentage_used']:.1f}%"
        )
        print(
            f"  Remaining: ${report['budget_status']['remaining']:.2f}"
        )
        print(
            f"  Projected Monthly: ${report['budget_status']['projected_monthly']:.2f}"
        )

        print(f"\nRouter Metrics:")
        global_metrics = report['router_metrics']['global']
        print(f"  Total Requests: {global_metrics['total_requests']}")
        print(f"  Successful: {global_metrics['successful_requests']}")
        print(f"  Failed: {global_metrics['failed_requests']}")
        print(f"  Failovers: {global_metrics['failover_count']}")
        print(f"  Total Cost: ${global_metrics['total_cost_usd']:.4f}")

        # Example 4: Export cost data
        print("\n[5] Exporting cost reports...")
        agent.cost_tracker.export_csv("esg_costs.csv")
        agent.cost_tracker.export_json("esg_costs.json", include_summary=True)
        print("  Exported to esg_costs.csv and esg_costs.json")

        # Example 5: Demonstrate failover (if multiple providers)
        if len(report['router_metrics']['providers']) > 1:
            print("\n[6] Testing provider failover...")
            print("  (Disable primary provider to see automatic failover)")
            # In production, this would happen automatically on provider failures

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

    finally:
        # Cleanup
        print("\n[7] Cleaning up...")
        await agent.cleanup()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
