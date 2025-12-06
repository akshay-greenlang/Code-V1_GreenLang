#!/usr/bin/env python3
import os
os.chdir('/c/Users/aksha/Code-V1_GreenLang')

# Read file
with open('greenlang/ml/explainability/lime_explainer.py') as f:
    lines = f.readlines()

# Find test class
test_idx = -1
for i, line in enumerate(lines):
    if 'class TestLIMEExplainer:' in line:
        test_idx = i
        break

# Keep only lines before test class
if test_idx > 0:
    content = ''.join(lines[:test_idx])
else:
    content = ''.join(lines)

# Add new classes
new_classes = '''

class ProcessHeatLIMEExplainer(LIMEExplainer):
    """Process Heat LIME Explainer with caching support."""

    def __init__(self, model, config=None, training_data=None, cache_size=1000):
        super().__init__(model, config, training_data)
        self._explanation_cache = {}
        self.cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"ProcessHeatLIMEExplainer initialized with cache_size={cache_size}")

    def _get_cache_key(self, instance):
        return hashlib.sha256(
            np.array2string(instance, precision=8, separator=",").encode()
        ).hexdigest()

    def explain_instance(self, instance, labels=None, use_cache=True):
        if isinstance(instance, list):
            instance = np.array(instance)

        cache_key = self._get_cache_key(instance)

        if use_cache and cache_key in self._explanation_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit (hits={self._cache_hits}, misses={self._cache_misses})")
            return self._explanation_cache[cache_key]

        self._cache_misses += 1
        result = super().explain_instance(instance, labels)

        if len(self._explanation_cache) < self.cache_size:
            self._explanation_cache[cache_key] = result

        return result

    def get_cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total if total > 0 else 0)
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_items": len(self._explanation_cache),
            "cache_size": self.cache_size
        }

    def clear_cache(self):
        self._explanation_cache.clear()
        logger.info("Explanation cache cleared")


class GL001LIMEExplainer(ProcessHeatLIMEExplainer):
    """LIME Explainer for GL001 Thermal Command Orchestrator."""

    def __init__(self, model, config=None, training_data=None):
        if config is None:
            config = LIMEExplainerConfig(
                feature_names=[
                    "setpoint_temp", "current_temp", "boiler_power",
                    "demand_forecast", "weather_temp", "system_efficiency",
                    "fuel_cost", "grid_price", "thermal_load", "ambient_humidity"
                ],
                class_names=["low_power", "medium_power", "high_power"],
                num_features=8
            )
        super().__init__(model, config, training_data)
        logger.info("GL001LIMEExplainer initialized for orchestrator decisions")

    def explain_decision(self, instance, decision_type="power_level"):
        result = self.explain_instance(instance)
        return {
            "decision_type": decision_type,
            "model_prediction": result.model_prediction,
            "explanation": result.local_explanation,
            "top_factors": result.feature_weights[:5],
            "confidence": result.r_squared,
            "provenance_hash": result.provenance_hash
        }


class GL010LIMEExplainer(ProcessHeatLIMEExplainer):
    """LIME Explainer for GL010 Emissions Guardian."""

    def __init__(self, model, config=None, training_data=None):
        if config is None:
            config = LIMEExplainerConfig(
                feature_names=[
                    "fuel_type", "fuel_quantity", "combustion_efficiency",
                    "emission_factor", "co2_content", "ch4_content",
                    "n2o_content", "operating_hours", "temperature", "oxygen_level"
                ],
                class_names=["low_emissions", "medium_emissions", "high_emissions"],
                num_features=8
            )
        super().__init__(model, config, training_data)
        logger.info("GL010LIMEExplainer initialized for emissions predictions")

    def explain_emission_prediction(self, instance, emission_scope="scope1"):
        result = self.explain_instance(instance)
        return {
            "emission_scope": emission_scope,
            "predicted_emissions": result.model_prediction,
            "local_model_emissions": result.local_prediction,
            "contributing_factors": result.local_explanation,
            "top_contributors": result.feature_weights[:5],
            "model_reliability": result.r_squared,
            "provenance_hash": result.provenance_hash
        }


class GL013LIMEExplainer(ProcessHeatLIMEExplainer):
    """LIME Explainer for GL013 Predictive Maintenance."""

    def __init__(self, model, config=None, training_data=None):
        if config is None:
            config = LIMEExplainerConfig(
                feature_names=[
                    "equipment_age", "operating_hours", "vibration_level",
                    "temperature_trend", "pressure_diff", "motor_current",
                    "efficiency_decline", "maintenance_history", "failure_rate",
                    "component_condition"
                ],
                class_names=["healthy", "warning", "failure_imminent"],
                num_features=8
            )
        super().__init__(model, config, training_data)
        logger.info("GL013LIMEExplainer initialized for failure predictions")

    def explain_failure_prediction(self, instance, equipment_id=""):
        result = self.explain_instance(instance)
        failure_probability = result.model_prediction
        return {
            "equipment_id": equipment_id,
            "failure_probability": failure_probability,
            "failure_risk_level": self._classify_risk(failure_probability),
            "contributing_factors": result.local_explanation,
            "top_risk_factors": result.feature_weights[:5],
            "model_confidence": result.r_squared,
            "provenance_hash": result.provenance_hash
        }

    @staticmethod
    def _classify_risk(probability):
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"
'''

with open('greenlang/ml/explainability/lime_explainer.py', 'w') as f:
    f.write(content + new_classes)

print('LIME explainer enhanced successfully')
print(f'Final file size: {os.path.getsize("greenlang/ml/explainability/lime_explainer.py")} bytes')
