# Category 1: Purchased Goods and Services - OPA Policy
# Version: 1.0.0
# GreenLang VCCI Platform
# Standard: GHG Protocol Scope 3 Standard (2011)

package vcci.scope3.category1

import future.keywords.if
import future.keywords.in

# ==============================================================================
# POLICY METADATA
# ==============================================================================
metadata := {
    "version": "1.0.0",
    "category": 1,
    "category_name": "Purchased Goods and Services",
    "standard": "GHG Protocol Scope 3 Standard (2011)",
    "coverage": "70% of typical Scope 3 emissions",
    "tiers": ["tier_1", "tier_2", "tier_3"],
    "last_updated": "2025-10-25"
}

# ==============================================================================
# INPUT VALIDATION
# ==============================================================================
# Validate required input fields
validate_input[msg] {
    not input.product
    msg := "Missing required field: product"
}

validate_input[msg] {
    not input.quantity
    msg := "Missing required field: quantity"
}

validate_input[msg] {
    not input.unit
    msg := "Missing required field: unit"
}

validate_input[msg] {
    not input.region
    msg := "Missing required field: region"
}

validate_input[msg] {
    input.quantity <= 0
    msg := "Quantity must be positive"
}

# Input is valid if no validation errors
input_valid {
    count(validate_input) == 0
}

# ==============================================================================
# TIER DETERMINATION
# ==============================================================================
# Determine calculation tier based on available data

# Tier 1: Supplier-specific PCF (highest quality)
tier_1_available {
    input.supplier_pcf
    input.supplier_pcf > 0
}

# Tier 2: Average-data (product emission factors)
tier_2_available {
    not tier_1_available
    input.product
}

# Tier 3: Spend-based (economic intensity factors)
tier_3_available {
    not tier_1_available
    not tier_2_available
    input.spend_usd
    input.spend_usd > 0
}

# Determine best available tier
tier := tier_result {
    tier_1_available
    tier_result := "tier_1"
} else := tier_result {
    tier_2_available
    tier_result := "tier_2"
} else := tier_result {
    tier_3_available
    tier_result := "tier_3"
} else := "tier_3"  # Default to tier 3

# ==============================================================================
# TIER 1 CALCULATION: SUPPLIER-SPECIFIC PCF
# ==============================================================================
# Use supplier-provided Product Carbon Footprint (PACT Pathfinder)

tier_1_calculation := result {
    tier == "tier_1"

    # Calculate emissions: quantity * supplier PCF
    emissions_kg := input.quantity * input.supplier_pcf
    emissions_tonnes := emissions_kg / 1000

    result := {
        "emissions_tco2e": emissions_tonnes,
        "method": "supplier_specific_pcf",
        "data_source": "PCF Exchange (PACT Pathfinder)",
        "tier": "tier_1",
        "data_quality_score": 95,
        "uncertainty": 0.05,  # ±5%
        "formula": "quantity * supplier_pcf",
        "inputs_used": {
            "product": input.product,
            "quantity": input.quantity,
            "unit": input.unit,
            "supplier_pcf": input.supplier_pcf
        }
    }
}

# ==============================================================================
# TIER 2 CALCULATION: AVERAGE-DATA (PRODUCT EMISSION FACTORS)
# ==============================================================================
# Use product emission factors from Factor Broker (ecoinvent, DESNZ, EPA)

tier_2_calculation := result {
    tier == "tier_2"

    # In real implementation, call Factor Broker API
    # For now, use mock values for demonstration
    emission_factor := mock_emission_factor

    # Calculate emissions: quantity * emission factor
    emissions_kg := input.quantity * emission_factor
    emissions_tonnes := emissions_kg / 1000

    result := {
        "emissions_tco2e": emissions_tonnes,
        "method": "average_data",
        "data_source": "Factor Broker (ecoinvent/DESNZ/EPA)",
        "tier": "tier_2",
        "data_quality_score": 75,
        "uncertainty": 0.20,  # ±20%
        "formula": "quantity * product_emission_factor",
        "emission_factor_used": emission_factor,
        "inputs_used": {
            "product": input.product,
            "quantity": input.quantity,
            "unit": input.unit,
            "region": input.region
        }
    }
}

# Mock emission factor (in production, query Factor Broker)
mock_emission_factor := factor {
    input.product == "Steel"
    input.region == "US"
    factor := 1.85  # kgCO2e/kg
} else := factor {
    input.product == "Aluminum"
    input.region == "US"
    factor := 9.12  # kgCO2e/kg
} else := factor {
    input.product == "Concrete"
    input.region == "US"
    factor := 0.12  # kgCO2e/kg
} else := 1.0  # Default factor

# ==============================================================================
# TIER 3 CALCULATION: SPEND-BASED (ECONOMIC INTENSITY FACTORS)
# ==============================================================================
# Use spend-based method with economic intensity factors

tier_3_calculation := result {
    tier == "tier_3"

    # In real implementation, call Factor Broker API for economic intensity
    # Economic intensity = kgCO2e per USD spent
    economic_intensity := mock_economic_intensity

    # Calculate emissions: spend * economic intensity
    emissions_kg := input.spend_usd * economic_intensity
    emissions_tonnes := emissions_kg / 1000

    result := {
        "emissions_tco2e": emissions_tonnes,
        "method": "spend_based",
        "data_source": "Factor Broker (economic intensity factors)",
        "tier": "tier_3",
        "data_quality_score": 50,
        "uncertainty": 0.50,  # ±50%
        "formula": "spend_usd * economic_intensity_factor",
        "economic_intensity_used": economic_intensity,
        "inputs_used": {
            "product": input.product,
            "spend_usd": input.spend_usd,
            "region": input.region
        }
    }
}

# Mock economic intensity (in production, query Factor Broker)
mock_economic_intensity := intensity {
    input.product == "Steel"
    intensity := 0.85  # kgCO2e/USD
} else := intensity {
    input.product == "Aluminum"
    intensity := 1.20  # kgCO2e/USD
} else := 0.50  # Default intensity for manufactured goods

# ==============================================================================
# FINAL RESULT
# ==============================================================================
# Return final calculation result based on tier

# Calculate result based on tier
calculate := result {
    input_valid
    tier == "tier_1"
    result := tier_1_calculation
} else := result {
    input_valid
    tier == "tier_2"
    result := tier_2_calculation
} else := result {
    input_valid
    tier == "tier_3"
    result := tier_3_calculation
}

# Main result with provenance
result := output {
    input_valid

    calc := calculate

    output := {
        "calculation": calc,
        "provenance": {
            "policy_name": "category_1_purchased_goods",
            "policy_version": metadata.version,
            "evaluation_timestamp": time.now_ns(),
            "tier_selected": tier,
            "gwp_standard": input.gwp_standard,
            "input_data_hash": crypto.sha256(json.marshal(input))
        },
        "validation": {
            "input_valid": true,
            "validation_errors": []
        }
    }
} else := output {
    not input_valid

    output := {
        "calculation": null,
        "provenance": null,
        "validation": {
            "input_valid": false,
            "validation_errors": validate_input
        }
    }
}

# ==============================================================================
# HELPER RULES
# ==============================================================================
# Additional helper rules for quality assurance

# Check if result is within reasonable bounds
result_reasonable {
    result.calculation.emissions_tco2e >= 0
    result.calculation.emissions_tco2e < 1000000  # Less than 1M tonnes
}

# Data quality tier classification
data_quality_tier := quality {
    result.calculation.data_quality_score >= 90
    quality := "high"
} else := quality {
    result.calculation.data_quality_score >= 70
    quality := "medium"
} else := "low"

# ==============================================================================
# TEST CASES (for OPA testing framework)
# ==============================================================================
# Example test cases for policy validation

test_tier_1_calculation {
    result.calculation.tier == "tier_1" with input as {
        "product": "Steel",
        "quantity": 1000,
        "unit": "kg",
        "region": "US",
        "supplier_pcf": 1.85,  # Supplier-specific PCF
        "gwp_standard": "AR6"
    }
}

test_tier_2_calculation {
    result.calculation.tier == "tier_2" with input as {
        "product": "Steel",
        "quantity": 1000,
        "unit": "kg",
        "region": "US",
        "gwp_standard": "AR6"
    }
}

test_tier_3_calculation {
    result.calculation.tier == "tier_3" with input as {
        "product": "Steel",
        "spend_usd": 10000,
        "region": "US",
        "gwp_standard": "AR6"
    }
}

test_input_validation_quantity {
    not input_valid with input as {
        "product": "Steel",
        "quantity": -100,  # Invalid: negative
        "unit": "kg",
        "region": "US"
    }
}

test_input_validation_missing_fields {
    not input_valid with input as {
        "product": "Steel"
        # Missing: quantity, unit, region
    }
}
