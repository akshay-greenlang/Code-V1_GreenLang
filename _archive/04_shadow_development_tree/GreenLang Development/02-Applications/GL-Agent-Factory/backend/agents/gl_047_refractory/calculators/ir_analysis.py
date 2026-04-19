"""
Infrared Thermal Analysis Calculator

Analysis of infrared thermography data for refractory hot spot detection
and thermal anomaly identification.

References:
    - ASTM E1933: Standard Test Methods for Measuring and Compensating for Emissivity
    - ISO 18434-1: Condition monitoring - Thermography
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8  # W/(m^2*K^4)

# Typical emissivity values for refractory surfaces
EMISSIVITY_VALUES = {
    "oxidized_steel": 0.85,
    "rusted_steel": 0.65,
    "painted_surface": 0.90,
    "bare_refractory": 0.75,
    "ceramic_fiber": 0.85,
    "insulation_cladding": 0.30,  # Bright metal
}


@dataclass
class HotSpot:
    """Detected hot spot information."""
    location_x: float
    location_y: float
    temperature_c: float
    delta_t_c: float  # Difference from background
    area_m2: float
    severity: str
    estimated_heat_loss_kw: float


def detect_hot_spots(
    temperature_matrix: List[List[float]],
    background_temp_c: float,
    threshold_delta_c: float = 20.0,
    pixel_size_m: float = 0.01
) -> List[HotSpot]:
    """
    Detect hot spots from thermal image data.

    A hot spot is a localized area significantly hotter than the
    surrounding background, indicating potential refractory damage
    or thinning.

    Args:
        temperature_matrix: 2D array of surface temperatures (C).
        background_temp_c: Expected normal surface temperature.
        threshold_delta_c: Minimum temperature rise to flag as hot spot.
        pixel_size_m: Physical size of each pixel in meters.

    Returns:
        List of detected HotSpot objects.

    Example:
        >>> temps = [[100, 100, 150], [100, 200, 100], [100, 100, 100]]
        >>> spots = detect_hot_spots(temps, 100, 30, 0.1)
        >>> print(f"Found {len(spots)} hot spots")
    """
    hot_spots = []

    if not temperature_matrix or not temperature_matrix[0]:
        return hot_spots

    rows = len(temperature_matrix)
    cols = len(temperature_matrix[0])
    pixel_area = pixel_size_m ** 2

    # Simple detection: flag individual pixels above threshold
    # In production, would use clustering/region growing
    visited = [[False] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            temp = temperature_matrix[i][j]
            delta_t = temp - background_temp_c

            if delta_t >= threshold_delta_c and not visited[i][j]:
                # Found hot spot - estimate area by flood fill
                area_pixels = _flood_fill_count(
                    temperature_matrix, visited, i, j,
                    background_temp_c, threshold_delta_c
                )

                area_m2 = area_pixels * pixel_area
                max_temp = temp  # Simplified - would find max in region

                # Estimate heat loss (radiation from elevated temp)
                heat_loss = _estimate_hotspot_heat_loss(
                    max_temp, background_temp_c, area_m2
                )

                # Determine severity
                severity = _classify_hotspot_severity(delta_t, area_m2)

                hot_spots.append(HotSpot(
                    location_x=j * pixel_size_m,
                    location_y=i * pixel_size_m,
                    temperature_c=max_temp,
                    delta_t_c=delta_t,
                    area_m2=area_m2,
                    severity=severity,
                    estimated_heat_loss_kw=heat_loss
                ))

    logger.info(f"Detected {len(hot_spots)} hot spots")
    return hot_spots


def _flood_fill_count(
    matrix: List[List[float]],
    visited: List[List[bool]],
    start_i: int,
    start_j: int,
    background: float,
    threshold: float
) -> int:
    """Count connected pixels above threshold using flood fill."""
    rows = len(matrix)
    cols = len(matrix[0])

    stack = [(start_i, start_j)]
    count = 0

    while stack:
        i, j = stack.pop()

        if (i < 0 or i >= rows or j < 0 or j >= cols):
            continue
        if visited[i][j]:
            continue

        delta = matrix[i][j] - background
        if delta < threshold:
            continue

        visited[i][j] = True
        count += 1

        # Add neighbors (4-connectivity)
        stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])

    return count


def _estimate_hotspot_heat_loss(
    hot_temp_c: float,
    background_temp_c: float,
    area_m2: float,
    emissivity: float = 0.85
) -> float:
    """Estimate additional heat loss from hot spot via radiation."""
    # Stefan-Boltzmann: P = e * sigma * A * (T_hot^4 - T_bg^4)
    t_hot_k = hot_temp_c + 273.15
    t_bg_k = background_temp_c + 273.15

    radiation_power = emissivity * STEFAN_BOLTZMANN * area_m2 * (
        t_hot_k ** 4 - t_bg_k ** 4
    )

    return radiation_power / 1000  # kW


def _classify_hotspot_severity(delta_t_c: float, area_m2: float) -> str:
    """Classify hot spot severity based on temperature rise and size."""
    # Severity matrix based on delta-T and area
    if delta_t_c > 100 or (delta_t_c > 50 and area_m2 > 0.1):
        return "CRITICAL"
    elif delta_t_c > 50 or (delta_t_c > 30 and area_m2 > 0.05):
        return "HIGH"
    elif delta_t_c > 30 or area_m2 > 0.02:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_surface_emissivity_correction(
    measured_temp_c: float,
    assumed_emissivity: float,
    actual_emissivity: float,
    reflected_temp_c: float = 20.0
) -> float:
    """
    Correct measured temperature for emissivity error.

    IR cameras measure radiance, which depends on emissivity:
        W_measured = e * sigma * T^4 + (1-e) * sigma * T_reflected^4

    Args:
        measured_temp_c: Temperature displayed by IR camera.
        assumed_emissivity: Emissivity setting used during measurement.
        actual_emissivity: True surface emissivity.
        reflected_temp_c: Ambient/reflected temperature.

    Returns:
        Corrected surface temperature in Celsius.
    """
    # Convert to Kelvin
    t_meas_k = measured_temp_c + 273.15
    t_refl_k = reflected_temp_c + 273.15

    # Calculate radiance measured
    # W = e_assumed * sigma * T_meas^4 (what camera thinks it's seeing)
    radiance_measured = assumed_emissivity * STEFAN_BOLTZMANN * t_meas_k ** 4

    # Correct for actual emissivity
    # W = e_actual * sigma * T_actual^4 + (1 - e_actual) * sigma * T_refl^4
    # T_actual^4 = (W - (1-e_actual) * sigma * T_refl^4) / (e_actual * sigma)

    reflected_component = (1 - actual_emissivity) * STEFAN_BOLTZMANN * t_refl_k ** 4
    actual_component = radiance_measured - reflected_component

    if actual_component <= 0:
        logger.warning("Invalid emissivity correction - using measured value")
        return measured_temp_c

    t_actual_k_4 = actual_component / (actual_emissivity * STEFAN_BOLTZMANN)
    t_actual_k = t_actual_k_4 ** 0.25
    t_actual_c = t_actual_k - 273.15

    logger.debug(
        f"Emissivity correction: measured={measured_temp_c:.1f}C, "
        f"corrected={t_actual_c:.1f}C "
        f"(e_assumed={assumed_emissivity}, e_actual={actual_emissivity})"
    )

    return t_actual_c


def analyze_thermal_profile(
    temperatures: List[float],
    positions_m: List[float],
    expected_profile: Optional[str] = "linear"
) -> Dict[str, any]:
    """
    Analyze temperature profile along a line/path.

    Useful for detecting localized damage or refractory thinning
    along a furnace wall or other linear feature.

    Args:
        temperatures: List of temperatures along profile.
        positions_m: List of corresponding positions in meters.
        expected_profile: Expected shape ("linear", "constant", "gradient").

    Returns:
        Dictionary with profile analysis results.
    """
    if len(temperatures) != len(positions_m):
        raise ValueError("Temperatures and positions must have same length")

    if len(temperatures) < 2:
        return {"error": "Insufficient data points"}

    # Basic statistics
    temp_min = min(temperatures)
    temp_max = max(temperatures)
    temp_avg = sum(temperatures) / len(temperatures)
    temp_range = temp_max - temp_min

    # Find anomalies (points far from neighbors)
    anomalies = []
    for i in range(1, len(temperatures) - 1):
        local_avg = (temperatures[i-1] + temperatures[i+1]) / 2
        deviation = abs(temperatures[i] - local_avg)
        if deviation > 0.2 * temp_range:  # 20% of range
            anomalies.append({
                "position_m": positions_m[i],
                "temperature_c": temperatures[i],
                "deviation_c": temperatures[i] - local_avg
            })

    # Calculate gradient
    total_length = positions_m[-1] - positions_m[0]
    if total_length > 0:
        avg_gradient = (temperatures[-1] - temperatures[0]) / total_length
    else:
        avg_gradient = 0

    # Uniformity score (100 = perfectly uniform)
    if temp_avg > 0:
        uniformity = max(0, 100 - (temp_range / temp_avg * 100))
    else:
        uniformity = 100

    return {
        "min_temp_c": temp_min,
        "max_temp_c": temp_max,
        "avg_temp_c": round(temp_avg, 1),
        "temp_range_c": temp_range,
        "gradient_c_per_m": round(avg_gradient, 2),
        "uniformity_score": round(uniformity, 1),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "profile_length_m": total_length,
    }


def calculate_anomaly_severity(
    delta_temp_c: float,
    anomaly_area_m2: float,
    growth_rate_c_per_day: float = 0.0
) -> Tuple[int, str, List[str]]:
    """
    Calculate severity score for a thermal anomaly.

    Args:
        delta_temp_c: Temperature difference from expected.
        anomaly_area_m2: Size of anomaly area.
        growth_rate_c_per_day: Rate of temperature increase (if trending).

    Returns:
        Tuple of (severity_score 0-100, severity_level, recommended_actions).
    """
    # Temperature component (0-40 points)
    if delta_temp_c > 150:
        temp_score = 40
    elif delta_temp_c > 100:
        temp_score = 30
    elif delta_temp_c > 50:
        temp_score = 20
    elif delta_temp_c > 25:
        temp_score = 10
    else:
        temp_score = 5

    # Area component (0-30 points)
    if anomaly_area_m2 > 0.5:
        area_score = 30
    elif anomaly_area_m2 > 0.2:
        area_score = 20
    elif anomaly_area_m2 > 0.05:
        area_score = 10
    else:
        area_score = 5

    # Growth rate component (0-30 points)
    if growth_rate_c_per_day > 5:
        growth_score = 30
    elif growth_rate_c_per_day > 2:
        growth_score = 20
    elif growth_rate_c_per_day > 0.5:
        growth_score = 10
    else:
        growth_score = 0

    total_score = temp_score + area_score + growth_score

    # Determine level and actions
    if total_score >= 70:
        level = "CRITICAL"
        actions = [
            "Immediate investigation required",
            "Schedule emergency repair/shutdown",
            "Increase monitoring frequency to daily"
        ]
    elif total_score >= 50:
        level = "HIGH"
        actions = [
            "Schedule investigation within 1 week",
            "Plan repair during next opportunity",
            "Increase monitoring to weekly"
        ]
    elif total_score >= 30:
        level = "MEDIUM"
        actions = [
            "Monitor weekly for changes",
            "Schedule inspection during next outage",
            "Document and track trend"
        ]
    else:
        level = "LOW"
        actions = [
            "Continue routine monitoring",
            "Document in inspection records"
        ]

    return total_score, level, actions
