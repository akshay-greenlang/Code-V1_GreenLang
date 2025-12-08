# -*- coding: utf-8 -*-
"""
Examples: NaturalLanguageExplainer Usage Patterns

This file demonstrates how to use the NaturalLanguageExplainer class
to generate human-readable explanations for different audiences and scenarios.
"""

from greenlang.ml.explainability.natural_language_explainer import (
    NaturalLanguageExplainer,
    Audience,
    OutputFormat,
    DecisionType,
    create_natural_language_explainer,
)


# =============================================================================
# Example 1: Basic Usage - Operator Audience
# =============================================================================

def example_operator_explanation():
    """
    Generate a simple explanation for equipment operators.

    Operators need clear, actionable guidance without technical jargon.
    """
    explainer = NaturalLanguageExplainer(default_audience=Audience.OPERATOR)

    # Simulate model output from a fouling risk predictor
    result = explainer.explain_prediction(
        prediction=0.82,
        shap_values={
            "flue_gas_temperature": 0.35,
            "days_since_cleaning": 0.28,
            "excess_air": 0.12,
            "stack_temperature": 0.07
        },
        feature_names={
            "flue_gas_temperature": "Flue Gas Temperature",
            "days_since_cleaning": "Days Since Cleaning",
            "excess_air": "Excess Air Percentage",
            "stack_temperature": "Stack Temperature"
        },
        feature_values={
            "flue_gas_temperature": 485.2,
            "days_since_cleaning": 120,
            "excess_air": 22.5,
            "stack_temperature": 325.0
        },
        decision_type=DecisionType.FOULING_RISK,
        confidence=0.88
    )

    print("OPERATOR EXPLANATION:")
    print("=" * 60)
    print(result.text_summary)
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print()


# =============================================================================
# Example 2: Engineer Audience with Technical Details
# =============================================================================

def example_engineer_explanation():
    """
    Generate a detailed technical explanation for engineers.

    Engineers need precise measurements, calculations, and full feature details.
    """
    explainer = NaturalLanguageExplainer(default_audience=Audience.ENGINEER)

    result = explainer.explain_prediction(
        prediction=0.75,
        shap_values={
            "combustion_efficiency": -0.12,
            "stack_temperature": 0.22,
            "excess_air": 0.18,
            "inlet_temperature": 0.13,
            "fuel_quality": 0.10
        },
        feature_names={
            "combustion_efficiency": "Combustion Efficiency (ASME PTC-4)",
            "stack_temperature": "Stack Temperature (F)",
            "excess_air": "Excess Air (%)",
            "inlet_temperature": "Inlet Temperature (F)",
            "fuel_quality": "Fuel Quality Index"
        },
        feature_values={
            "combustion_efficiency": 0.82,
            "stack_temperature": 385.0,
            "excess_air": 25.5,
            "inlet_temperature": 68.0,
            "fuel_quality": 0.91
        },
        decision_type=DecisionType.EFFICIENCY_DEGRADATION,
        confidence=0.91,
        baseline=0.88
    )

    print("ENGINEER EXPLANATION:")
    print("=" * 60)
    print(result.text_summary)
    print("\nMARKDOWN FORMAT:")
    print(result.markdown_summary[:200] + "...")
    print()


# =============================================================================
# Example 3: Executive Summary
# =============================================================================

def example_executive_explanation():
    """
    Generate a high-level summary for executives.

    Executives want brief status and business impact, not technical details.
    """
    explainer = NaturalLanguageExplainer(default_audience=Audience.EXECUTIVE)

    result = explainer.explain_prediction(
        prediction=0.68,
        shap_values={
            "runtime_hours": 0.40,
            "vibration_level": 0.35,
            "temperature_deviation": 0.15,
            "pressure_anomaly": 0.10
        },
        feature_names={
            "runtime_hours": "Operating Hours",
            "vibration_level": "Vibration Level",
            "temperature_deviation": "Temp Anomaly",
            "pressure_anomaly": "Pressure Anomaly"
        },
        decision_type=DecisionType.MAINTENANCE_NEEDED,
        confidence=0.85
    )

    print("EXECUTIVE SUMMARY:")
    print("=" * 60)
    print(result.text_summary)
    print("\nKEY METRICS:")
    print(f"  - Confidence: {result.confidence:.0%}")
    print(f"  - Risk Level: {result.metadata.get('risk_level')}")
    print()


# =============================================================================
# Example 4: Auditor Report with Provenance
# =============================================================================

def example_auditor_report():
    """
    Generate a compliance-focused report for auditors.

    Auditors need complete traceability, all features, and provenance hashes.
    """
    explainer = NaturalLanguageExplainer(default_audience=Audience.AUDITOR)

    result = explainer.explain_prediction(
        prediction=0.72,
        shap_values={
            "flue_gas_temperature": 0.25,
            "days_since_cleaning": 0.22,
            "excess_air": 0.15,
            "stack_temperature": 0.12,
            "pressure_drop": 0.08,
            "ambient_temp": 0.05,
            "fuel_moisture": 0.08,
            "load_factor": 0.03,
            "air_quality": 0.02
        },
        feature_names={
            "flue_gas_temperature": "Flue Gas Temperature (F)",
            "days_since_cleaning": "Days Since Cleaning",
            "excess_air": "Excess Air (%)",
            "stack_temperature": "Stack Temperature (F)",
            "pressure_drop": "Pressure Drop (PSI)",
            "ambient_temp": "Ambient Temperature (F)",
            "fuel_moisture": "Fuel Moisture Content (%)",
            "load_factor": "Load Factor (%)",
            "air_quality": "Air Quality Index"
        },
        decision_type=DecisionType.FOULING_RISK,
        confidence=0.89
    )

    print("AUDITOR COMPLIANCE REPORT:")
    print("=" * 60)
    print(result.text_summary)
    print(f"\nPROVENANCE HASH: {result.provenance_hash}")
    print(f"TIMESTAMP: {result.timestamp.isoformat()}")
    print(f"ALL FACTORS ANALYZED: {result.metadata.get('feature_count', 0)}")
    print()


# =============================================================================
# Example 5: Decision Explanation with Structured Factors
# =============================================================================

def example_decision_explanation():
    """
    Explain a decision using structured factors instead of SHAP values.

    Useful when you have domain-specific factors rather than ML features.
    """
    explainer = NaturalLanguageExplainer()

    explanation = explainer.explain_decision(
        decision_type=DecisionType.MAINTENANCE_NEEDED,
        factors={
            "Vibration Level": "7.2 mm/s (HIGH)",
            "Temperature Trend": "Rising 2C per hour",
            "Acoustic Emissions": "Above threshold",
            "Operating Hours": "4,287 hours",
            "Last Maintenance": "18 months ago"
        },
        confidence=0.94,
        audience=Audience.OPERATOR
    )

    print("STRUCTURED DECISION EXPLANATION:")
    print("=" * 60)
    print(explanation)
    print()


# =============================================================================
# Example 6: Multi-Explanation Summary
# =============================================================================

def example_multi_model_summary():
    """
    Combine explanations from multiple models into a single summary.

    Useful when you have multiple predictions (fouling, efficiency, maintenance)
    and want to create a unified status report.
    """
    explainer = NaturalLanguageExplainer()

    # Fouling risk explanation
    fouling_result = explainer.explain_prediction(
        prediction=0.82,
        shap_values={"temp": 0.5, "days_clean": 0.3},
        feature_names={"temp": "Temperature", "days_clean": "Days Since Cleaning"},
        decision_type=DecisionType.FOULING_RISK
    )

    # Efficiency explanation
    efficiency_result = explainer.explain_prediction(
        prediction=0.70,
        shap_values={"stack_temp": 0.6, "excess_air": 0.4},
        feature_names={"stack_temp": "Stack Temperature", "excess_air": "Excess Air"},
        decision_type=DecisionType.EFFICIENCY_DEGRADATION
    )

    # Maintenance explanation
    maintenance_result = explainer.explain_prediction(
        prediction=0.65,
        shap_values={"vibration": 0.5, "runtime": 0.35},
        feature_names={"vibration": "Vibration", "runtime": "Runtime Hours"},
        decision_type=DecisionType.MAINTENANCE_NEEDED
    )

    # Generate unified summary for operator
    summary = explainer.generate_summary(
        [fouling_result, efficiency_result, maintenance_result],
        audience=Audience.OPERATOR,
        output_format=OutputFormat.MARKDOWN
    )

    print("UNIFIED EQUIPMENT STATUS SUMMARY:")
    print("=" * 60)
    print(summary)
    print()


# =============================================================================
# Example 7: Output Format Conversion
# =============================================================================

def example_output_formats():
    """
    Demonstrate all output formats (text, markdown, HTML).
    """
    explainer = NaturalLanguageExplainer()

    result = explainer.explain_prediction(
        prediction=0.78,
        shap_values={"feature1": 0.5, "feature2": 0.3},
        feature_names={"feature1": "Feature 1", "feature2": "Feature 2"},
        decision_type=DecisionType.FOULING_RISK
    )

    print("OUTPUT FORMATS COMPARISON:")
    print("=" * 60)
    print("\n1. PLAIN TEXT:")
    print(result.text_summary[:150] + "...")

    print("\n2. MARKDOWN:")
    print(result.markdown_summary[:150] + "...")

    print("\n3. HTML:")
    print(result.html_summary[:150] + "...")
    print()


# =============================================================================
# Example 8: Factory Function Usage
# =============================================================================

def example_factory_function():
    """
    Use the factory function to create explainers with common configurations.
    """
    # Create operator-friendly explainer
    operator_explainer = create_natural_language_explainer(
        audience="operator",
        output_format="text"
    )

    # Create engineer explainer
    engineer_explainer = create_natural_language_explainer(
        audience="engineer",
        output_format="markdown"
    )

    print("FACTORY FUNCTION USAGE:")
    print("=" * 60)
    print(f"Operator Explainer Audience: {operator_explainer.default_audience.value}")
    print(f"Engineer Explainer Audience: {engineer_explainer.default_audience.value}")
    print()


# =============================================================================
# Example 9: Real-World Boiler Scenario
# =============================================================================

def example_real_world_boiler():
    """
    Real-world example: Steam boiler fouling prediction.

    Scenario: An industrial boiler showing signs of fouling. The ML model
    predicts 87% fouling risk. Different teams need different explanations.
    """
    print("REAL-WORLD SCENARIO: INDUSTRIAL BOILER FOULING")
    print("=" * 60)
    print("Boiler: Model MB-500, Rating: 500 BHP, Fuel: Natural Gas")
    print("Prediction: 87% fouling risk detected")
    print()

    shared_data = {
        "prediction": 0.87,
        "shap_values": {
            "flue_gas_temperature": 0.38,
            "days_since_cleaning": 0.32,
            "excess_air": 0.15,
            "stack_temperature": 0.10,
            "pressure_drop": 0.05
        },
        "feature_names": {
            "flue_gas_temperature": "Flue Gas Temperature",
            "days_since_cleaning": "Days Since Last Cleaning",
            "excess_air": "Excess Air %",
            "stack_temperature": "Stack Temperature",
            "pressure_drop": "Pressure Drop"
        },
        "feature_values": {
            "flue_gas_temperature": 510.0,
            "days_since_cleaning": 145,
            "excess_air": 28.5,
            "stack_temperature": 340.0,
            "pressure_drop": 2.1
        },
        "decision_type": DecisionType.FOULING_RISK,
        "confidence": 0.92
    }

    # Operator view
    operator_explainer = create_natural_language_explainer(
        audience="operator",
        output_format="text"
    )
    operator_result = operator_explainer.explain_prediction(**shared_data)

    print("OPERATOR VIEW (What to do):")
    print("-" * 40)
    print(operator_result.text_summary)
    print("\nActions:")
    for i, rec in enumerate(operator_result.recommendations, 1):
        print(f"  {i}. {rec}")

    # Engineer view
    engineer_explainer = create_natural_language_explainer(
        audience="engineer",
        output_format="markdown"
    )
    engineer_result = engineer_explainer.explain_prediction(**shared_data)

    print("\nENGINEER VIEW (Technical analysis):")
    print("-" * 40)
    print(engineer_result.text_summary[:300])
    print("\nTop 3 Contributing Factors:")
    for feature, contribution in engineer_result.top_factors[:3]:
        print(f"  - {feature}: {contribution:.1%} impact")

    # Executive view
    executive_explainer = create_natural_language_explainer(
        audience="executive",
        output_format="text"
    )
    executive_result = executive_explainer.explain_prediction(**shared_data)

    print("\nEXECUTIVE VIEW (Business impact):")
    print("-" * 40)
    print(executive_result.text_summary)
    print()


# =============================================================================
# Example 10: Batch Processing Multiple Explanations
# =============================================================================

def example_batch_processing():
    """
    Process multiple predictions and generate batch explanations.

    Useful for fleet management, multiple units, or daily reports.
    """
    explainer = create_natural_language_explainer(audience="manager")

    # Simulate multiple boiler units
    units = [
        {
            "name": "Unit A - Boiler",
            "prediction": 0.92,
            "top_factor": ("flue_gas_temperature", 0.45)
        },
        {
            "name": "Unit B - Furnace",
            "prediction": 0.58,
            "top_factor": ("excess_air", 0.38)
        },
        {
            "name": "Unit C - Heat Exchanger",
            "prediction": 0.31,
            "top_factor": ("inlet_temperature", 0.25)
        }
    ]

    print("BATCH REPORT - EQUIPMENT STATUS:")
    print("=" * 60)

    for unit in units:
        result = explainer.explain_prediction(
            prediction=unit["prediction"],
            shap_values={unit["top_factor"][0]: unit["top_factor"][1]},
            feature_names={unit["top_factor"][0]: unit["top_factor"][0].replace("_", " ").title()},
            decision_type=DecisionType.FOULING_RISK
        )

        status = "CRITICAL" if unit["prediction"] > 0.8 else "WARNING" if unit["prediction"] > 0.5 else "OK"
        print(f"\n{unit['name']}: {status}")
        print(f"  Risk Level: {unit['prediction']:.0%}")
        print(f"  Top Issue: {unit['top_factor'][0]}")

    print()


if __name__ == "__main__":
    # Run all examples
    example_operator_explanation()
    example_engineer_explanation()
    example_executive_explanation()
    example_auditor_report()
    example_decision_explanation()
    example_multi_model_summary()
    example_output_formats()
    example_factory_function()
    example_real_world_boiler()
    example_batch_processing()

    print("\nAll examples completed successfully!")
