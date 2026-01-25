# Category 6: Business Travel - OPA Policy
# Version: 1.0.0
# GreenLang VCCI Platform
# Standard: GHG Protocol Scope 3 Standard (2011)

package vcci.scope3.category6

import future.keywords.if
import future.keywords.in

# ==============================================================================
# POLICY METADATA
# ==============================================================================
metadata := {
    "version": "1.0.0",
    "category": 6,
    "category_name": "Business Travel",
    "standard": "GHG Protocol Scope 3 Standard (2011)",
    "coverage": "5% of typical Scope 3 emissions",
    "methods": ["distance_based_flights", "hotel_nights", "ground_transport"],
    "last_updated": "2025-10-25"
}

# ==============================================================================
# INPUT VALIDATION
# ==============================================================================
validate_input[msg] {
    not input.travel_type
    msg := "Missing required field: travel_type"
}

validate_input[msg] {
    not input.travel_type in ["flight", "hotel", "ground_transport"]
    msg := sprintf("Invalid travel_type: %v. Must be one of: flight, hotel, ground_transport", [input.travel_type])
}

validate_input[msg] {
    input.travel_type == "flight"
    not input.distance_km
    msg := "Missing required field for flight: distance_km"
}

validate_input[msg] {
    input.travel_type == "flight"
    input.distance_km <= 0
    msg := "distance_km must be positive"
}

validate_input[msg] {
    input.travel_type == "hotel"
    not input.nights
    msg := "Missing required field for hotel: nights"
}

validate_input[msg] {
    input.travel_type == "hotel"
    input.nights <= 0
    msg := "nights must be positive"
}

# Input is valid if no validation errors
input_valid {
    count(validate_input) == 0
}

# ==============================================================================
# FLIGHT CALCULATION (DISTANCE-BASED WITH RADIATIVE FORCING INDEX)
# ==============================================================================
# Formula: distance_km * emission_factor_per_pkm * radiative_forcing_index

flight_calculation := result {
    input.travel_type == "flight"

    # Determine flight class (economy, business, first)
    class := object.get(input, "flight_class", "economy")

    # Get base emission factor
    base_ef := flight_emission_factor_per_pkm(class)

    # Determine radiative forcing index (RFI) based on distance
    rfi := radiative_forcing_index

    # Calculate emissions
    emissions_kg := input.distance_km * base_ef * rfi
    emissions_tonnes := emissions_kg / 1000

    result := {
        "emissions_tco2e": emissions_tonnes,
        "method": "distance_based_flights",
        "travel_type": "flight",
        "data_quality_score": 80,
        "uncertainty": 0.20,  # ±20%
        "formula": "distance_km * emission_factor_per_pkm * radiative_forcing_index",
        "base_emission_factor": base_ef,
        "radiative_forcing_index": rfi,
        "flight_category": flight_category,
        "inputs_used": {
            "distance_km": input.distance_km,
            "flight_class": class
        }
    }
}

# Flight category based on distance
flight_category := category {
    input.distance_km < 500
    category := "short_haul"
} else := category {
    input.distance_km >= 500
    input.distance_km <= 3000
    category := "medium_haul"
} else := "long_haul"

# Radiative Forcing Index (RFI) - accounts for non-CO2 climate impacts
radiative_forcing_index := rfi {
    flight_category == "short_haul"
    rfi := 1.0  # No RFI for short-haul
} else := rfi {
    flight_category == "medium_haul"
    rfi := 1.5  # 50% uplift for medium-haul
} else := 2.0  # 100% uplift for long-haul

# Emission factors by flight class (kgCO2e per passenger-kilometer)
flight_emission_factor_per_pkm(class) := ef {
    class == "economy"
    ef := 0.115  # kgCO2e/pkm
} else := ef {
    class == "business"
    ef := 0.230  # kgCO2e/pkm (2x economy due to more space)
} else := ef {
    class == "first"
    ef := 0.345  # kgCO2e/pkm (3x economy)
} else := 0.115  # Default to economy

# ==============================================================================
# HOTEL CALCULATION (NIGHTS-BASED)
# ==============================================================================
# Formula: nights * emission_factor_per_night

hotel_calculation := result {
    input.travel_type == "hotel"

    # Determine hotel category
    category := object.get(input, "hotel_category", "standard")

    # Get emission factor for hotel nights
    ef := hotel_emission_factor_per_night(category)

    # Calculate emissions
    emissions_kg := input.nights * ef
    emissions_tonnes := emissions_kg / 1000

    result := {
        "emissions_tco2e": emissions_tonnes,
        "method": "hotel_nights",
        "travel_type": "hotel",
        "data_quality_score": 70,
        "uncertainty": 0.30,  # ±30%
        "formula": "nights * emission_factor_per_night",
        "emission_factor": ef,
        "inputs_used": {
            "nights": input.nights,
            "hotel_category": category
        }
    }
}

# Emission factors by hotel category (kgCO2e per night)
hotel_emission_factor_per_night(category) := ef {
    category == "budget"
    ef := 15.0  # kgCO2e/night
} else := ef {
    category == "standard"
    ef := 25.0  # kgCO2e/night
} else := ef {
    category == "luxury"
    ef := 50.0  # kgCO2e/night
} else := 25.0  # Default to standard

# ==============================================================================
# GROUND TRANSPORT CALCULATION (RENTAL CARS, TAXIS)
# ==============================================================================
# Formula: distance_km * emission_factor_per_km

ground_transport_calculation := result {
    input.travel_type == "ground_transport"

    # Get emission factor
    ef := ground_transport_emission_factor_per_km

    # Calculate emissions
    emissions_kg := input.distance_km * ef
    emissions_tonnes := emissions_kg / 1000

    result := {
        "emissions_tco2e": emissions_tonnes,
        "method": "ground_transport",
        "travel_type": "ground_transport",
        "data_quality_score": 75,
        "uncertainty": 0.25,  # ±25%
        "formula": "distance_km * emission_factor_per_km",
        "emission_factor": ef,
        "inputs_used": {
            "distance_km": input.distance_km,
            "vehicle_type": object.get(input, "vehicle_type", "medium_car")
        }
    }
}

# Emission factors for ground transport (kgCO2e per km)
ground_transport_emission_factor_per_km := ef {
    input.vehicle_type == "small_car"
    ef := 0.142  # kgCO2e/km
} else := ef {
    input.vehicle_type == "medium_car"
    ef := 0.192  # kgCO2e/km
} else := ef {
    input.vehicle_type == "large_car"
    ef := 0.282  # kgCO2e/km
} else := ef {
    input.vehicle_type == "taxi"
    ef := 0.200  # kgCO2e/km (average)
} else := 0.192  # Default to medium car

# ==============================================================================
# FINAL RESULT
# ==============================================================================
calculate := result {
    input_valid
    input.travel_type == "flight"
    result := flight_calculation
} else := result {
    input_valid
    input.travel_type == "hotel"
    result := hotel_calculation
} else := result {
    input_valid
    input.travel_type == "ground_transport"
    result := ground_transport_calculation
}

result := output {
    input_valid

    calc := calculate

    output := {
        "calculation": calc,
        "provenance": {
            "policy_name": "category_6_travel",
            "policy_version": metadata.version,
            "evaluation_timestamp": time.now_ns(),
            "travel_type": input.travel_type,
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
# TEST CASES
# ==============================================================================
test_flight_short_haul_economy {
    result.calculation.travel_type == "flight" with input as {
        "travel_type": "flight",
        "distance_km": 300,
        "flight_class": "economy",
        "gwp_standard": "AR6"
    }

    result.calculation.radiative_forcing_index == 1.0 with input as {
        "travel_type": "flight",
        "distance_km": 300,
        "flight_class": "economy"
    }
}

test_flight_long_haul_business {
    result.calculation.travel_type == "flight" with input as {
        "travel_type": "flight",
        "distance_km": 10000,
        "flight_class": "business",
        "gwp_standard": "AR6"
    }

    result.calculation.radiative_forcing_index == 2.0 with input as {
        "travel_type": "flight",
        "distance_km": 10000,
        "flight_class": "business"
    }
}

test_hotel_standard {
    result.calculation.travel_type == "hotel" with input as {
        "travel_type": "hotel",
        "nights": 5,
        "hotel_category": "standard",
        "gwp_standard": "AR6"
    }
}

test_ground_transport {
    result.calculation.travel_type == "ground_transport" with input as {
        "travel_type": "ground_transport",
        "distance_km": 200,
        "vehicle_type": "medium_car",
        "gwp_standard": "AR6"
    }
}

test_input_validation_travel_type {
    not input_valid with input as {
        "travel_type": "train",  # Invalid
        "distance_km": 100
    }
}
