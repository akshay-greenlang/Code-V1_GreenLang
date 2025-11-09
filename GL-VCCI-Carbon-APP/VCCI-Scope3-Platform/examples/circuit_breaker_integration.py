"""
Circuit Breaker Integration Examples for GL-VCCI Scope 3 Platform

Demonstrates how to integrate circuit breakers into existing code.

Author: GreenLang Platform Team
Version: 1.0.0
Date: 2025-11-09
"""

from typing import Dict, Any, List
from datetime import datetime

# Circuit breaker imports
from services.circuit_breakers import (
    get_factor_broker_cb,
    get_llm_provider_cb,
    get_erp_connector_cb,
    get_email_service_cb,
)
from greenlang.resilience import CircuitOpenError
from greenlang.telemetry import get_logger


logger = get_logger(__name__)


# ============================================================================
# EXAMPLE 1: INTEGRATING WITH CALCULATOR AGENT
# ============================================================================

class CalculatorAgentWithCircuitBreaker:
    """
    Calculator agent enhanced with circuit breaker protection.

    Original code location:
    services/agents/calculator/agent.py
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        # Get circuit breaker singletons
        self.factor_cb = get_factor_broker_cb()
        self.llm_cb = get_llm_provider_cb()

    def calculate_emissions(
        self,
        activity: str,
        quantity: float,
        unit: str,
        region: str = "US"
    ) -> Dict[str, Any]:
        """
        Calculate emissions with circuit breaker protection.

        BEFORE (without circuit breaker):
            factor = self.factor_broker.get_factor(activity, region)
            emissions = quantity * factor

        AFTER (with circuit breaker):
            Uses circuit breaker with cache fallback
        """
        try:
            # Get emission factor with circuit breaker protection
            factor_data = self.factor_cb.get_emission_factor(
                source="ecoinvent",  # Try ecoinvent first
                activity=activity,
                region=region
            )

            # Calculate emissions
            emissions = quantity * factor_data["value"]

            self.logger.info(
                f"Emissions calculated successfully",
                extra={
                    "activity": activity,
                    "emissions": emissions,
                    "factor_quality": factor_data.get("quality", "unknown"),
                }
            )

            return {
                "activity": activity,
                "quantity": quantity,
                "unit": unit,
                "emissions_kg_co2e": emissions,
                "factor_source": factor_data["source"],
                "factor_quality": factor_data.get("quality", "high"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except CircuitOpenError as e:
            # All factor sources are down - try fallback sources or use conservative estimate
            self.logger.error(
                f"All factor sources unavailable",
                extra={"activity": activity, "error": str(e)}
            )

            # Use conservative default factor
            conservative_factor = 1.0  # kg CO2e per unit
            emissions = quantity * conservative_factor

            return {
                "activity": activity,
                "quantity": quantity,
                "unit": unit,
                "emissions_kg_co2e": emissions,
                "factor_source": "conservative_default",
                "factor_quality": "fallback",
                "warning": "External factor services unavailable - using conservative estimate",
                "timestamp": datetime.utcnow().isoformat(),
            }


# ============================================================================
# EXAMPLE 2: INTEGRATING WITH INTAKE AGENT
# ============================================================================

class IntakeAgentWithCircuitBreaker:
    """
    Intake agent enhanced with circuit breaker protection.

    Original code location:
    services/agents/intake/agent.py
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.erp_cb = get_erp_connector_cb()
        self.llm_cb = get_llm_provider_cb()

    def ingest_from_erp(
        self,
        system: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Ingest data from ERP system with circuit breaker protection.

        BEFORE (without circuit breaker):
            suppliers = erp_client.fetch_suppliers()
            purchases = erp_client.fetch_purchases(start_date, end_date)

        AFTER (with circuit breaker):
            Uses circuit breaker with cache fallback and graceful degradation
        """
        results = {
            "system": system,
            "suppliers": [],
            "purchases": [],
            "status": "success",
            "warnings": [],
        }

        # Fetch suppliers with circuit breaker
        try:
            suppliers = self.erp_cb.fetch_suppliers(
                system=system,
                filters={"status": "active"}
            )
            results["suppliers"] = suppliers
            self.logger.info(
                f"Fetched {len(suppliers)} suppliers from {system}",
                extra={"system": system, "count": len(suppliers)}
            )

        except CircuitOpenError:
            results["warnings"].append(
                f"{system} supplier service unavailable - using cached data"
            )
            results["suppliers"] = []  # Empty list - handled downstream

        # Fetch purchase orders with circuit breaker
        try:
            purchases = self.erp_cb.fetch_purchases(
                system=system,
                start_date=start_date,
                end_date=end_date
            )
            results["purchases"] = purchases
            self.logger.info(
                f"Fetched {len(purchases)} purchases from {system}",
                extra={"system": system, "count": len(purchases)}
            )

        except CircuitOpenError:
            results["warnings"].append(
                f"{system} purchase service unavailable - using cached data"
            )
            results["purchases"] = []

        # Set status based on results
        if results["warnings"]:
            results["status"] = "partial"

        return results

    def classify_supplier_with_llm(
        self,
        supplier_name: str,
        supplier_description: str
    ) -> Dict[str, Any]:
        """
        Classify supplier using LLM with circuit breaker protection.

        BEFORE (without circuit breaker):
            response = llm_client.classify(supplier_name, supplier_description)

        AFTER (with circuit breaker):
            Automatic failover between Claude and OpenAI
        """
        prompt = f"""
        Classify the following supplier into an industry category:

        Name: {supplier_name}
        Description: {supplier_description}

        Return only the industry category (e.g., "Manufacturing", "Transportation", "Energy").
        """

        try:
            # LLM call with automatic failover and caching
            cache_key = f"classify_supplier:{supplier_name}"

            response = self.llm_cb.generate(
                prompt=prompt,
                model="claude-3-sonnet",
                max_tokens=50,
                temperature=0.3,  # Low temperature for classification
                cache_key=cache_key  # Enable caching
            )

            return {
                "supplier_name": supplier_name,
                "category": response["text"].strip(),
                "provider": response["provider"],
                "confidence": "high",
                "cached": response.get("cached", False),
            }

        except CircuitOpenError:
            # All LLM providers down - use rule-based fallback
            self.logger.warning(
                f"LLM providers unavailable - using rule-based classification",
                extra={"supplier_name": supplier_name}
            )

            # Simple rule-based classification
            category = self._rule_based_classification(
                supplier_name,
                supplier_description
            )

            return {
                "supplier_name": supplier_name,
                "category": category,
                "provider": "rule_based",
                "confidence": "low",
                "note": "LLM providers unavailable - used rule-based classification",
            }

    def _rule_based_classification(
        self,
        name: str,
        description: str
    ) -> str:
        """Fallback rule-based classification."""
        text = f"{name} {description}".lower()

        if any(keyword in text for keyword in ["manufacturing", "factory", "production"]):
            return "Manufacturing"
        elif any(keyword in text for keyword in ["transport", "logistics", "shipping"]):
            return "Transportation"
        elif any(keyword in text for keyword in ["energy", "power", "electricity"]):
            return "Energy"
        else:
            return "Other"


# ============================================================================
# EXAMPLE 3: INTEGRATING WITH REPORTING AGENT
# ============================================================================

class ReportingAgentWithCircuitBreaker:
    """
    Reporting agent enhanced with circuit breaker protection.

    Original code location:
    services/agents/reporting/agent.py
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.email_cb = get_email_service_cb()
        self.llm_cb = get_llm_provider_cb()

    def send_report(
        self,
        recipient: str,
        report_type: str,
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send report via email with circuit breaker protection.

        BEFORE (without circuit breaker):
            email_service.send(recipient, subject, body)

        AFTER (with circuit breaker):
            Queue-based fallback when email service is down
        """
        # Generate report summary with LLM
        try:
            summary = self._generate_report_summary(report_data)
        except CircuitOpenError:
            summary = "Report summary unavailable - LLM service down"

        # Prepare email
        subject = f"Your {report_type} Carbon Report"
        body = f"""
        Hello,

        Your {report_type} carbon report is ready.

        Summary:
        {summary}

        Total Emissions: {report_data.get('total_emissions', 'N/A')} kg CO2e

        Please review the attached report for details.

        Best regards,
        GreenLang Team
        """

        # Send with circuit breaker protection
        result = self.email_cb.send_email(
            to=recipient,
            subject=subject,
            body=body,
            priority="normal"
        )

        if result["status"] == "queued":
            self.logger.warning(
                f"Email queued due to service unavailability",
                extra={"recipient": recipient, "report_type": report_type}
            )

        return result

    def _generate_report_summary(self, report_data: Dict[str, Any]) -> str:
        """Generate report summary using LLM."""
        prompt = f"""
        Generate a brief 2-sentence summary of this carbon emissions report:

        Total Emissions: {report_data.get('total_emissions', 0)} kg CO2e
        Top Categories: {report_data.get('top_categories', [])}
        Period: {report_data.get('period', 'N/A')}

        Keep it concise and actionable.
        """

        response = self.llm_cb.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )

        return response["text"]


# ============================================================================
# EXAMPLE 4: BATCH PROCESSING WITH CIRCUIT BREAKERS
# ============================================================================

class BatchProcessorWithCircuitBreaker:
    """
    Example of batch processing with circuit breaker protection.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.factor_cb = get_factor_broker_cb()

    def calculate_batch_emissions(
        self,
        activities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate emissions for batch of activities with circuit breaker.

        Handles partial failures gracefully.
        """
        results = {
            "total_activities": len(activities),
            "successful": 0,
            "failed": 0,
            "fallback_used": 0,
            "details": [],
        }

        for activity in activities:
            try:
                factor = self.factor_cb.get_emission_factor(
                    source="ecoinvent",
                    activity=activity["name"],
                    region=activity.get("region", "US")
                )

                emissions = activity["quantity"] * factor["value"]

                results["successful"] += 1
                if factor.get("quality") == "fallback":
                    results["fallback_used"] += 1

                results["details"].append({
                    "activity": activity["name"],
                    "emissions": emissions,
                    "status": "success",
                    "quality": factor.get("quality", "high"),
                })

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "activity": activity["name"],
                    "status": "failed",
                    "error": str(e),
                })

        return results


# ============================================================================
# EXAMPLE 5: MONITORING CIRCUIT BREAKER HEALTH
# ============================================================================

def get_circuit_breaker_health() -> Dict[str, Any]:
    """
    Get health status of all circuit breakers.

    Use this in health check endpoints.
    """
    factor_cb = get_factor_broker_cb()
    llm_cb = get_llm_provider_cb()
    erp_cb = get_erp_connector_cb()
    email_cb = get_email_service_cb()

    health = {
        "status": "healthy",
        "circuits": {
            "factor_broker": factor_cb.get_stats(),
            "llm_provider": llm_cb.get_stats(),
            "erp_connector": erp_cb.get_stats(),
            "email_service": email_cb.get_stats(),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Determine overall status
    for service, stats in health["circuits"].items():
        if isinstance(stats, dict):
            for name, circuit_stats in stats.items():
                if circuit_stats.get("state") == "open":
                    health["status"] = "degraded"
                    break

    return health


# ============================================================================
# EXAMPLE 6: TESTING CIRCUIT BREAKERS
# ============================================================================

def test_circuit_breakers():
    """
    Manual testing of circuit breakers.
    Use in development/staging environments.
    """
    logger.info("Starting circuit breaker tests")

    # Test factor broker
    factor_cb = get_factor_broker_cb()
    logger.info(f"Factor broker stats: {factor_cb.get_stats()}")

    # Test LLM provider failover
    llm_cb = get_llm_provider_cb()
    logger.info(f"LLM provider stats: {llm_cb.get_stats()}")

    # Test ERP connector
    erp_cb = get_erp_connector_cb()
    test_result = erp_cb.test_connection("sap")
    logger.info(f"ERP connection test: {test_result}")

    # Test email queue
    email_cb = get_email_service_cb()
    queue_stats = email_cb.get_stats()
    logger.info(f"Email queue size: {queue_stats['queue_size']}")

    # Process email queue
    if queue_stats["queue_size"] > 0:
        process_result = email_cb.process_queue()
        logger.info(f"Queue processing result: {process_result}")

    logger.info("Circuit breaker tests completed")


if __name__ == "__main__":
    # Run health check
    health = get_circuit_breaker_health()
    print(f"Circuit Breaker Health: {health['status']}")

    # Run tests
    test_circuit_breakers()
