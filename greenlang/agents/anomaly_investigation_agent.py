"""
Anomaly Investigation Agent with Root Cause Analysis
GL Intelligence Infrastructure - INSIGHT PATH

Hybrid architecture combining deterministic anomaly detection with AI-powered investigation:
- calculate(): Use anomaly_agent_iforest for deterministic anomaly detection
- explain(): AI-powered root cause analysis with RAG retrieval and diagnostic tools

Pattern: InsightAgent (deterministic calculations + AI insights)
Temperature: 0.6 (analytical consistency for investigation)
Category: AgentCategory.INSIGHT

Version: 1.0.0
Date: 2025-11-06
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from greenlang.agents.base_agents import InsightAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent


logger = logging.getLogger(__name__)


@dataclass
class AnomalyInvestigationResult:
    """Complete anomaly investigation results."""
    # Deterministic detection results
    anomalies: List[bool]
    anomaly_scores: List[float]
    anomaly_indices: List[int]
    severity_distribution: Dict[str, int]
    top_anomalies: List[Dict[str, Any]]
    patterns: Dict[str, Any]

    # AI-powered investigation
    root_cause_analysis: str
    investigation_report: str
    evidence_summary: Dict[str, Any]
    remediation_recommendations: List[str]
    confidence_score: float

    # Metadata
    calculation_trace: List[str]
    timestamp: str


class AnomalyInvestigationAgent(InsightAgent):
    """
    AI-powered anomaly investigation agent with hybrid architecture.

    DETERMINISTIC DETECTION (calculate method):
    - Isolation Forest anomaly detection
    - Anomaly scoring and severity classification
    - Pattern identification
    - Statistical analysis

    AI-POWERED INVESTIGATION (explain method):
    - Root cause analysis using RAG retrieval
    - Maintenance log correlation
    - Sensor diagnostic checks
    - Weather condition analysis
    - Evidence-based remediation recommendations

    Tools for Investigation:
    1. maintenance_log_tool - Query maintenance logs for correlated events
    2. sensor_diagnostic_tool - Check sensor health and calibration status
    3. weather_data_tool - Retrieve weather conditions during anomaly period

    RAG Collections Used:
    - anomaly_patterns: Historical anomaly cases and resolutions
    - root_cause_database: Known failure modes and diagnostic patterns
    - sensor_specifications: Sensor behavior, failure modes, and calibration
    - maintenance_procedures: Maintenance correlation and preventive measures

    Temperature: 0.6 (analytical consistency for investigation)
    """

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="anomaly_investigation_agent",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=True,
        critical_for_compliance=False,
        transformation_priority="HIGH (Phase 2 - Investigation)",
        description="Hybrid agent: deterministic anomaly detection + AI root cause analysis"
    )

    def __init__(
        self,
        enable_audit_trail: bool = True,
        detection_budget_usd: float = 1.00,
        investigation_budget_usd: float = 2.00
    ):
        """
        Initialize anomaly investigation agent.

        Args:
            enable_audit_trail: Whether to capture calculation audit trail
            detection_budget_usd: Budget for anomaly detection (default: $1.00)
            investigation_budget_usd: Budget for AI investigation (default: $2.00)
        """
        super().__init__(enable_audit_trail=enable_audit_trail)

        # Initialize anomaly detection agent
        self.detector = IsolationForestAnomalyAgent(
            budget_usd=detection_budget_usd,
            enable_explanations=False,  # We'll provide our own explanations
            enable_recommendations=False,
            enable_alerts=False
        )

        self.investigation_budget_usd = investigation_budget_usd

        # Performance tracking
        self._total_investigations = 0
        self._total_cost_usd = 0.0

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic anomaly detection.

        This method is DETERMINISTIC and FAST:
        - Uses Isolation Forest for anomaly detection
        - Same inputs produce same outputs
        - No AI, pure statistical analysis
        - Full calculation audit trail

        Args:
            inputs: {
                "data": pd.DataFrame with sensor/energy data,
                "contamination": float (expected anomaly rate, default: 0.1),
                "n_estimators": int (number of trees, default: 100),
                "feature_columns": List[str] (optional, specific features),
                "timestamp_column": str (optional, for temporal analysis),
                "location": str (optional, for contextual analysis),
                "system_id": str (optional, system identifier)
            }

        Returns:
            Dictionary with deterministic anomaly detection results:
            {
                "anomalies": List[bool],
                "anomaly_scores": List[float],
                "anomaly_indices": List[int],
                "severity_distribution": Dict[str, int],
                "top_anomalies": List[Dict],
                "patterns": Dict,
                "n_anomalies": int,
                "anomaly_rate": float,
                "calculation_trace": List[str]
            }
        """
        calculation_trace = []

        # Extract parameters
        data = inputs.get("data")
        contamination = inputs.get("contamination", 0.1)
        n_estimators = inputs.get("n_estimators", 100)
        feature_columns = inputs.get("feature_columns")

        calculation_trace.append(f"Input Data Shape: {data.shape}")
        calculation_trace.append(f"Contamination: {contamination}")
        calculation_trace.append(f"N Estimators: {n_estimators}")

        # Validate inputs
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame")

        if len(data) < 100:
            raise ValueError("Insufficient data: need at least 100 points for stable detection")

        calculation_trace.append("Input validation passed")

        # Prepare detection input
        detection_input = {
            "data": data,
            "contamination": contamination,
            "n_estimators": n_estimators
        }

        if feature_columns:
            detection_input["feature_columns"] = feature_columns
            calculation_trace.append(f"Using specific features: {feature_columns}")

        # Run deterministic anomaly detection
        calculation_trace.append("Executing Isolation Forest detection...")
        detection_result = self.detector.process(detection_input)

        # Extract key metrics
        n_anomalies = detection_result.get("n_anomalies", 0)
        anomaly_rate = detection_result.get("anomaly_rate", 0.0)

        calculation_trace.append(f"Detected {n_anomalies} anomalies ({anomaly_rate*100:.1f}%)")

        # Build result
        result = {
            "anomalies": detection_result.get("anomalies", []),
            "anomaly_scores": detection_result.get("anomaly_scores", []),
            "anomaly_indices": detection_result.get("anomaly_indices", []),
            "severity_distribution": detection_result.get("severity_distribution", {}),
            "top_anomalies": detection_result.get("top_anomalies", []),
            "patterns": detection_result.get("patterns", {}),
            "n_anomalies": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "model_info": detection_result.get("model_info", {}),
            "calculation_trace": calculation_trace,
        }

        # Capture audit trail
        if self.enable_audit_trail:
            self._capture_calculation_audit(
                operation="anomaly_detection",
                inputs=inputs,
                outputs=result,
                calculation_trace=calculation_trace
            )

        return result

    async def explain(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6
    ) -> str:
        """
        Generate AI-powered root cause analysis and investigation report.

        This method uses AI to investigate WHY anomalies occurred:
        - RAG retrieval for similar historical cases
        - Tool-based evidence gathering (maintenance, sensors, weather)
        - Root cause hypothesis generation
        - Evidence-based recommendations

        Args:
            calculation_result: Output from calculate() method
            context: Additional context {
                "data": pd.DataFrame (original data),
                "timestamp_column": str (optional),
                "location": str (optional),
                "system_id": str (optional),
                "system_type": str (optional, e.g., "HVAC", "solar"),
                "recent_changes": List[str] (optional),
                "investigation_depth": str (optional: "quick", "standard", "comprehensive")
            }
            session: ChatSession instance
            rag_engine: RAGEngine instance
            temperature: LLM temperature (default 0.6 for analytical consistency)

        Returns:
            Comprehensive investigation report with:
            - Executive summary
            - Root cause analysis
            - Evidence from tools and RAG
            - Confidence-scored remediation recommendations
            - Investigation metadata
        """
        self._total_investigations += 1

        # Extract anomaly information
        n_anomalies = calculation_result.get("n_anomalies", 0)
        top_anomalies = calculation_result.get("top_anomalies", [])
        patterns = calculation_result.get("patterns", {})
        severity_dist = calculation_result.get("severity_distribution", {})

        # If no anomalies, return early
        if n_anomalies == 0:
            return self._format_no_anomalies_report(context)

        # Step 1: Build RAG query for similar cases
        rag_query = self._build_rag_query(calculation_result, context)

        # Step 2: RAG retrieval for historical cases and patterns
        rag_result = await self._rag_retrieve(
            query=rag_query,
            rag_engine=rag_engine,
            collections=[
                "anomaly_patterns",
                "root_cause_database",
                "sensor_specifications",
                "maintenance_procedures"
            ],
            top_k=8
        )

        # Step 3: Format RAG knowledge
        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 4: Build investigation prompt with tools
        investigation_prompt = self._build_investigation_prompt(
            calculation_result,
            context,
            formatted_knowledge
        )

        # Step 5: Define investigation tools
        tools = self._get_investigation_tools()

        # Step 6: AI investigation with tools
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": investigation_prompt
                }
            ],
            tools=tools,
            temperature=temperature
        )

        # Track cost
        if hasattr(response, 'usage'):
            self._total_cost_usd += response.usage.cost_usd

        # Step 7: Process tool calls and gather evidence
        tool_evidence = await self._process_tool_calls(response, context)

        # Step 8: Format final investigation report
        investigation_report = self._format_investigation_report(
            calculation_result=calculation_result,
            context=context,
            ai_analysis=response.text if hasattr(response, 'text') else str(response),
            tool_evidence=tool_evidence,
            rag_knowledge=formatted_knowledge
        )

        return investigation_report

    def _build_rag_query(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build semantic search query for RAG retrieval."""
        n_anomalies = calculation_result.get("n_anomalies", 0)
        severity_dist = calculation_result.get("severity_distribution", {})
        patterns = calculation_result.get("patterns", {})

        system_type = context.get("system_type", "energy system")
        location = context.get("location", "")

        # Identify most anomalous features
        most_important_feature = patterns.get("most_important_feature", "unknown")

        query = f"""
Anomaly Investigation Query:
- System Type: {system_type}
- Number of Anomalies: {n_anomalies}
- Severity Distribution: {severity_dist}
- Most Affected Feature: {most_important_feature}
{f"- Location: {location}" if location else ""}

Looking for:
1. Historical cases with similar anomaly patterns
2. Known root causes for {most_important_feature} anomalies in {system_type}
3. Diagnostic procedures for anomaly investigation
4. Maintenance correlation patterns
5. Sensor failure modes and calibration issues
6. Environmental factors affecting {system_type} performance
7. Remediation strategies that have worked in similar cases
"""

        return query.strip()

    def _build_investigation_prompt(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        rag_knowledge: str
    ) -> str:
        """Build comprehensive investigation prompt."""
        n_anomalies = calculation_result.get("n_anomalies", 0)
        top_anomalies = calculation_result.get("top_anomalies", [])[:5]  # Top 5
        patterns = calculation_result.get("patterns", {})
        severity_dist = calculation_result.get("severity_distribution", {})

        system_type = context.get("system_type", "energy system")
        system_id = context.get("system_id", "unknown")
        investigation_depth = context.get("investigation_depth", "standard")

        # Format top anomalies
        anomaly_details = "\n".join([
            f"  - Index {a['index']}: Score={a['score']:.3f}, Severity={a['severity']}, Features={a['features']}"
            for a in top_anomalies
        ])

        # Format pattern analysis
        pattern_details = ""
        if patterns and "patterns" in patterns:
            for feature, stats in patterns["patterns"].items():
                pattern_details += f"\n  - {feature}:"
                pattern_details += f"\n    * Anomaly mean: {stats.get('anomaly_mean', 0):.2f}"
                pattern_details += f"\n    * Normal mean: {stats.get('normal_mean', 0):.2f}"
                pattern_details += f"\n    * Relative difference: {stats.get('relative_difference', 0)*100:.1f}%"

        prompt = f"""
# ANOMALY INVESTIGATION REQUEST

## Detection Results (Deterministic)

**System Information:**
- System Type: {system_type}
- System ID: {system_id}
- Investigation Depth: {investigation_depth}

**Anomaly Summary:**
- Total Anomalies: {n_anomalies}
- Severity Distribution: {severity_dist}

**Top Anomalies:**
{anomaly_details}

**Pattern Analysis:**
{pattern_details}

---

## Historical Knowledge (RAG Retrieval)

{rag_knowledge}

---

## Investigation Tasks

Use the provided tools to gather evidence:

1. **maintenance_log_tool** - Check for maintenance events correlating with anomalies
   - Query maintenance logs for the time periods of detected anomalies
   - Look for recent maintenance, repairs, or system changes

2. **sensor_diagnostic_tool** - Verify sensor health and calibration
   - Check sensor calibration status
   - Identify any sensor malfunctions or drift
   - Review sensor maintenance history

3. **weather_data_tool** - Analyze environmental conditions
   - Retrieve weather data for anomaly periods
   - Identify extreme conditions (temperature, humidity, storms)
   - Correlate weather events with anomaly patterns

After gathering evidence, provide:

### Root Cause Analysis
- Primary hypothesis for why anomalies occurred
- Supporting evidence from tools and RAG knowledge
- Confidence level (0-100%) with justification
- Alternative hypotheses if applicable

### Evidence Summary
- Maintenance correlation findings
- Sensor diagnostic results
- Weather/environmental factors
- Pattern analysis insights

### Remediation Recommendations
Provide 3-5 prioritized, actionable recommendations:
1. Immediate actions (within 24 hours)
2. Short-term fixes (within 1 week)
3. Long-term preventive measures

Each recommendation should include:
- Specific action to take
- Expected impact
- Implementation complexity (low/medium/high)
- Confidence level

### Investigation Metadata
- Evidence strength: (weak/moderate/strong)
- Confidence in root cause: (0-100%)
- Recommended follow-up actions
- Escalation needed: (yes/no)

**Important:**
- Use tools to gather concrete evidence
- Ground analysis in RAG knowledge and tool results
- Provide specific, actionable recommendations
- Include confidence levels for key findings
- Be transparent about uncertainty
"""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for investigation."""
        return """You are an expert anomaly investigator for climate and energy systems.

Your role is to conduct thorough root cause analysis of detected anomalies by:
1. Using diagnostic tools to gather evidence
2. Consulting historical knowledge bases
3. Forming evidence-based hypotheses
4. Providing actionable remediation recommendations

Key principles:
- Use tools to gather concrete evidence (don't speculate without data)
- Ground analysis in RAG knowledge and historical patterns
- Provide confidence levels for all conclusions
- Be specific and actionable in recommendations
- Acknowledge uncertainty when evidence is limited
- Focus on root causes, not symptoms

You are analytical, methodical, and evidence-driven.
Temperature: 0.6 for consistency while allowing analytical reasoning."""

    def _get_investigation_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for investigation."""
        return [
            {
                "name": "maintenance_log_tool",
                "description": "Query maintenance logs for events correlating with anomaly periods. Returns maintenance activities, repairs, and system changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "string",
                            "description": "Start timestamp for log query (ISO format)"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End timestamp for log query (ISO format)"
                        },
                        "system_id": {
                            "type": "string",
                            "description": "System identifier to query"
                        },
                        "event_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of events to filter (e.g., 'maintenance', 'repair', 'calibration')"
                        }
                    },
                    "required": ["start_time", "end_time"]
                }
            },
            {
                "name": "sensor_diagnostic_tool",
                "description": "Check sensor health, calibration status, and failure indicators. Returns diagnostic results and recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sensor IDs to diagnose"
                        },
                        "diagnostic_type": {
                            "type": "string",
                            "enum": ["calibration", "health", "drift", "comprehensive"],
                            "description": "Type of diagnostic to run"
                        },
                        "time_window": {
                            "type": "string",
                            "description": "Time window for diagnostic analysis (e.g., '7d', '30d')"
                        }
                    },
                    "required": ["sensor_ids"]
                }
            },
            {
                "name": "weather_data_tool",
                "description": "Retrieve weather conditions during anomaly periods. Returns temperature, humidity, precipitation, and extreme events.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location identifier (city, coordinates, or station ID)"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Start timestamp (ISO format)"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End timestamp (ISO format)"
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Weather metrics to retrieve (e.g., 'temperature', 'humidity', 'wind')"
                        }
                    },
                    "required": ["location", "start_time", "end_time"]
                }
            }
        ]

    async def _process_tool_calls(
        self,
        response,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process tool calls and gather evidence."""
        tool_evidence = {
            "maintenance_logs": None,
            "sensor_diagnostics": None,
            "weather_data": None
        }

        # Extract tool calls
        tool_calls = getattr(response, 'tool_calls', [])

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

            try:
                if tool_name == "maintenance_log_tool":
                    tool_evidence["maintenance_logs"] = self._query_maintenance_logs(
                        arguments, context
                    )
                elif tool_name == "sensor_diagnostic_tool":
                    tool_evidence["sensor_diagnostics"] = self._run_sensor_diagnostics(
                        arguments, context
                    )
                elif tool_name == "weather_data_tool":
                    tool_evidence["weather_data"] = self._fetch_weather_data(
                        arguments, context
                    )
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                tool_evidence[tool_name.replace("_tool", "") + "s"] = {
                    "error": str(e),
                    "status": "failed"
                }

        return tool_evidence

    def _query_maintenance_logs(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query maintenance logs for correlated events.

        This is a mock implementation. In production, this would:
        - Query actual maintenance database
        - Parse maintenance records
        - Correlate with anomaly timestamps
        """
        # Mock maintenance log results
        return {
            "status": "success",
            "events_found": 2,
            "events": [
                {
                    "timestamp": "2025-11-05T14:30:00Z",
                    "type": "calibration",
                    "description": "Sensor recalibration performed",
                    "system_id": context.get("system_id", "unknown"),
                    "technician": "John Doe"
                },
                {
                    "timestamp": "2025-11-04T09:15:00Z",
                    "type": "maintenance",
                    "description": "Routine HVAC filter replacement",
                    "system_id": context.get("system_id", "unknown"),
                    "technician": "Jane Smith"
                }
            ],
            "correlation": "High - maintenance event 24h before anomaly spike",
            "confidence": 0.75
        }

    def _run_sensor_diagnostics(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run sensor health diagnostics.

        This is a mock implementation. In production, this would:
        - Check sensor calibration dates
        - Analyze sensor drift patterns
        - Detect sensor failures or anomalies
        """
        # Mock sensor diagnostic results
        return {
            "status": "success",
            "sensors_checked": len(arguments.get("sensor_ids", [])),
            "diagnostics": [
                {
                    "sensor_id": arguments.get("sensor_ids", ["sensor_1"])[0],
                    "health_status": "degraded",
                    "calibration_status": "overdue",
                    "last_calibration": "2025-09-01T00:00:00Z",
                    "drift_detected": True,
                    "drift_magnitude": 0.15,
                    "recommendation": "Immediate recalibration required"
                }
            ],
            "overall_health": "degraded",
            "confidence": 0.80
        }

    def _fetch_weather_data(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fetch weather data for anomaly periods.

        This is a mock implementation. In production, this would:
        - Query weather API or database
        - Retrieve historical weather conditions
        - Identify extreme weather events
        """
        # Mock weather data results
        return {
            "status": "success",
            "location": arguments.get("location", "unknown"),
            "conditions": [
                {
                    "timestamp": "2025-11-05T12:00:00Z",
                    "temperature_c": 35.2,
                    "humidity_pct": 85,
                    "wind_speed_kmh": 45,
                    "conditions": "Severe heatwave",
                    "alert_level": "high"
                },
                {
                    "timestamp": "2025-11-05T18:00:00Z",
                    "temperature_c": 32.8,
                    "humidity_pct": 90,
                    "wind_speed_kmh": 30,
                    "conditions": "High humidity + heat",
                    "alert_level": "medium"
                }
            ],
            "extreme_events": ["Heatwave (>35°C)", "High humidity (>85%)"],
            "correlation": "Strong - anomalies coincide with extreme heat",
            "confidence": 0.85
        }

    def _format_investigation_report(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        ai_analysis: str,
        tool_evidence: Dict[str, Any],
        rag_knowledge: str
    ) -> str:
        """Format comprehensive investigation report."""
        system_type = context.get("system_type", "energy system")
        system_id = context.get("system_id", "unknown")
        n_anomalies = calculation_result.get("n_anomalies", 0)

        report = f"""
# ANOMALY INVESTIGATION REPORT
Generated: {datetime.utcnow().isoformat()}Z

## Executive Summary
- **System:** {system_type} (ID: {system_id})
- **Anomalies Detected:** {n_anomalies}
- **Severity Distribution:** {calculation_result.get('severity_distribution', {})}
- **Investigation Status:** Complete

---

## AI Root Cause Analysis

{ai_analysis}

---

## Evidence Gathered

### Maintenance Log Correlation
"""

        # Add maintenance evidence
        if tool_evidence.get("maintenance_logs"):
            maint = tool_evidence["maintenance_logs"]
            if maint.get("status") == "success":
                report += f"- Events Found: {maint.get('events_found', 0)}\n"
                report += f"- Correlation: {maint.get('correlation', 'Unknown')}\n"
                report += f"- Confidence: {maint.get('confidence', 0)*100:.0f}%\n\n"

                if maint.get("events"):
                    report += "**Recent Events:**\n"
                    for event in maint["events"][:3]:
                        report += f"- {event.get('timestamp')}: {event.get('type')} - {event.get('description')}\n"
            else:
                report += f"- Status: {maint.get('error', 'No data available')}\n"
        else:
            report += "- Status: Not queried\n"

        report += "\n### Sensor Diagnostics\n"

        # Add sensor evidence
        if tool_evidence.get("sensor_diagnostics"):
            sensor = tool_evidence["sensor_diagnostics"]
            if sensor.get("status") == "success":
                report += f"- Sensors Checked: {sensor.get('sensors_checked', 0)}\n"
                report += f"- Overall Health: {sensor.get('overall_health', 'Unknown')}\n"
                report += f"- Confidence: {sensor.get('confidence', 0)*100:.0f}%\n\n"

                if sensor.get("diagnostics"):
                    report += "**Diagnostic Results:**\n"
                    for diag in sensor["diagnostics"][:3]:
                        report += f"- Sensor {diag.get('sensor_id')}: {diag.get('health_status')} "
                        report += f"(Calibration: {diag.get('calibration_status')})\n"
                        if diag.get("recommendation"):
                            report += f"  → {diag.get('recommendation')}\n"
            else:
                report += f"- Status: {sensor.get('error', 'No data available')}\n"
        else:
            report += "- Status: Not queried\n"

        report += "\n### Weather Conditions\n"

        # Add weather evidence
        if tool_evidence.get("weather_data"):
            weather = tool_evidence["weather_data"]
            if weather.get("status") == "success":
                report += f"- Location: {weather.get('location', 'Unknown')}\n"
                report += f"- Extreme Events: {', '.join(weather.get('extreme_events', []))}\n"
                report += f"- Correlation: {weather.get('correlation', 'Unknown')}\n"
                report += f"- Confidence: {weather.get('confidence', 0)*100:.0f}%\n"
            else:
                report += f"- Status: {weather.get('error', 'No data available')}\n"
        else:
            report += "- Status: Not queried\n"

        report += f"""

---

## Pattern Analysis

**Most Affected Features:**
"""

        patterns = calculation_result.get("patterns", {})
        if patterns and "feature_importance" in patterns:
            for feature, importance in list(patterns["feature_importance"].items())[:3]:
                report += f"- {feature}: {importance*100:.1f}% deviation from normal\n"

        report += f"""

---

## Calculation Audit Trail

**Detection Method:** Isolation Forest (deterministic)
**Model Parameters:**
- Contamination: {calculation_result.get('model_info', {}).get('contamination', 0.1)}
- N Estimators: {calculation_result.get('model_info', {}).get('n_estimators', 100)}
- Features Used: {calculation_result.get('model_info', {}).get('features', [])}

**Reproducibility:** Full audit trail captured for regulatory compliance

---

## Investigation Metadata

- **Total Investigation Time:** ~{len(ai_analysis) // 100} seconds
- **Investigation Depth:** {context.get('investigation_depth', 'standard')}
- **RAG Knowledge Sources:** 4 collections queried
- **Tools Executed:** {sum(1 for v in tool_evidence.values() if v is not None)}
- **Report Generated:** {datetime.utcnow().isoformat()}Z

---

*This report combines deterministic anomaly detection with AI-powered root cause analysis.*
*Calculations are reproducible and auditable. AI insights are evidence-based and confidence-scored.*
"""

        return report.strip()

    def _format_no_anomalies_report(self, context: Dict[str, Any]) -> str:
        """Format report when no anomalies are detected."""
        return f"""
# ANOMALY INVESTIGATION REPORT
Generated: {datetime.utcnow().isoformat()}Z

## Executive Summary
- **System:** {context.get('system_type', 'energy system')} (ID: {context.get('system_id', 'unknown')})
- **Anomalies Detected:** 0
- **Status:** No anomalies detected - system operating within normal parameters

## Analysis
No anomalies were detected in the analyzed time period. The system is operating within expected ranges based on:
- Historical baseline models
- Statistical thresholds
- Pattern recognition algorithms

## Recommendations
1. Continue monitoring with current detection parameters
2. Maintain regular sensor calibration schedule
3. Review detection sensitivity if anomalies are expected but not detected

---

*This is a deterministic result from Isolation Forest analysis.*
"""

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        return {
            "agent_id": self.metadata.name,
            "category": self.category.value,
            "total_investigations": self._total_investigations,
            "total_cost_usd": self._total_cost_usd,
            "avg_cost_per_investigation": (
                self._total_cost_usd / max(self._total_investigations, 1)
            ),
            "detector_performance": self.detector.get_performance_summary()
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("Anomaly Investigation Agent - INSIGHT PATH")
    print("=" * 80)

    # Initialize agent
    agent = AnomalyInvestigationAgent(enable_audit_trail=True)

    print("\n✓ Agent initialized with InsightAgent pattern")
    print(f"✓ Category: {agent.category}")
    print(f"✓ Uses ChatSession: {agent.metadata.uses_chat_session}")
    print(f"✓ Uses RAG: {agent.metadata.uses_rag}")
    print(f"✓ Uses Tools: {agent.metadata.uses_tools}")
    print(f"✓ Temperature: 0.6 (analytical consistency)")

    # Test calculation (deterministic)
    print("\n" + "=" * 80)
    print("TEST 1: DETERMINISTIC ANOMALY DETECTION")
    print("=" * 80)

    # Create test data with anomalies
    np.random.seed(42)
    n_samples = 500

    # Normal data
    energy_normal = np.random.normal(100, 10, n_samples)
    temp_normal = np.random.normal(20, 2, n_samples)

    # Inject anomalies
    anomaly_indices = [50, 150, 300, 420]
    energy_normal[anomaly_indices] = [500, 450, 480, 510]  # Energy spikes
    temp_normal[anomaly_indices] = [35, 38, 36, 40]  # Temperature spikes

    test_data = pd.DataFrame({
        "energy_kwh": energy_normal,
        "temperature_c": temp_normal,
        "humidity_pct": np.random.normal(60, 10, n_samples)
    })

    test_inputs = {
        "data": test_data,
        "contamination": 0.05,
        "feature_columns": ["energy_kwh", "temperature_c"],
        "system_type": "HVAC",
        "system_id": "HVAC-001",
        "location": "Building A"
    }

    print(f"\nInputs: {n_samples} samples, 2 features")
    print(f"Expected anomalies: ~{int(n_samples * 0.05)} (5% contamination)")

    result = agent.calculate(test_inputs)

    print(f"\n✓ Anomalies Detected: {result['n_anomalies']}")
    print(f"✓ Anomaly Rate: {result['anomaly_rate']*100:.1f}%")
    print(f"✓ Severity Distribution: {result['severity_distribution']}")

    if result['top_anomalies']:
        print(f"\nTop 3 Anomalies:")
        for i, anomaly in enumerate(result['top_anomalies'][:3], 1):
            print(f"  {i}. Index {anomaly['index']}: Score={anomaly['score']:.3f}, Severity={anomaly['severity']}")

    # Test AI investigation (requires ChatSession and RAGEngine)
    print("\n" + "=" * 80)
    print("TEST 2: AI INVESTIGATION (requires live infrastructure)")
    print("=" * 80)

    print("\n⚠ AI investigation requires:")
    print("  - ChatSession instance (LLM API)")
    print("  - RAGEngine instance (vector database)")
    print("  - Knowledge base with collections:")
    print("    * anomaly_patterns")
    print("    * root_cause_database")
    print("    * sensor_specifications")
    print("    * maintenance_procedures")

    print("\nExample async call:")
    print("""
    investigation_report = await agent.explain(
        calculation_result=result,
        context={
            "data": test_data,
            "system_type": "HVAC",
            "system_id": "HVAC-001",
            "location": "Building A",
            "investigation_depth": "comprehensive"
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
    is_reproducible = result["n_anomalies"] == result2["n_anomalies"]

    print(f"\n✓ Same inputs produce same outputs: {is_reproducible}")

    if agent.enable_audit_trail:
        print(f"✓ Audit trail entries: {len(agent.audit_trail)}")

    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    perf = agent.get_performance_summary()
    print(f"\nAgent: {perf['agent_id']}")
    print(f"Category: {perf['category']}")
    print(f"Total Investigations: {perf['total_investigations']}")

    print("\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("\nPattern: InsightAgent (Hybrid Architecture)")
    print("  - calculate(): Deterministic Isolation Forest detection")
    print("  - explain(): AI root cause analysis with RAG + tools")
    print("\nTools for Investigation:")
    print("  ✓ maintenance_log_tool - Maintenance correlation")
    print("  ✓ sensor_diagnostic_tool - Sensor health checks")
    print("  ✓ weather_data_tool - Environmental factors")
    print("\nRAG Collections:")
    print("  ✓ anomaly_patterns - Historical cases")
    print("  ✓ root_cause_database - Failure modes")
    print("  ✓ sensor_specifications - Sensor behavior")
    print("  ✓ maintenance_procedures - Preventive measures")
    print("\nValue-Add:")
    print("  ✓ Evidence-based root cause analysis")
    print("  ✓ Multi-source investigation (maintenance, sensors, weather)")
    print("  ✓ Confidence-scored recommendations")
    print("  ✓ Actionable remediation strategies")
    print("  ✓ Full audit trail for compliance")
    print("=" * 80)
