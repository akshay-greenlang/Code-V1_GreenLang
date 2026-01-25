# -*- coding: utf-8 -*-
"""
Counterfactual Explanation Generator for ML Explainability Framework.

This module provides counterfactual explanations for Process Heat ML models,
helping users understand what minimal changes would be needed to achieve
different predictions (e.g., reduce fouling risk, improve efficiency).

Counterfactual explanations answer questions like:
- "What would need to change to reduce fouling risk from HIGH to LOW?"
- "How much would I need to increase flow rate to achieve optimal efficiency?"
- "What if the flue gas temperature was 50F lower?"

The module follows zero-hallucination principles: all counterfactuals are
generated using optimization algorithms on the actual model, not synthesized.

Example:
    >>> from greenlang.ml.explainability import CounterfactualExplainer
    >>> explainer = CounterfactualExplainer(
    ...     model=fouling_model,
    ...     feature_names=["temperature", "pressure", "flow_rate"],
    ...     feature_ranges={"temperature": (100, 500), "pressure": (10, 100)}
    ... )
    >>> result = explainer.generate_counterfactual(
    ...     instance={"temperature": 450, "pressure": 80, "flow_rate": 1000},
    ...     target_prediction=0.3  # Reduce fouling risk to 30%
    ... )
    >>> print(result.changes_required)

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hashlib
import logging
import numpy as np

from .schemas import (
    CounterfactualResult,
    WhatIfResult,
    ExplainerType,
    ProcessHeatContext,
    compute_provenance_hash,
)

logger = logging.getLogger(__name__)

# Conditional imports for optimization libraries
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install with: pip install scipy")


class CounterfactualExplainer:
    """
    Counterfactual Explanation Generator.

    Generates counterfactual explanations showing minimal changes needed
    to alter model predictions. Uses optimization to find the closest
    valid counterfactual in feature space.

    This class implements zero-hallucination principles:
    - All counterfactuals are optimized using the actual model
    - Changes are constrained to realistic feature ranges
    - Feasibility scores indicate how realistic changes are

    Attributes:
        model: ML model with predict method
        feature_names: Names of input features
        feature_ranges: Valid ranges for each feature
        immutable_features: Features that cannot be changed
        process_heat_context: Domain context for process heat

    Example:
        >>> explainer = CounterfactualExplainer(
        ...     model=risk_model,
        ...     feature_names=["temp", "pressure", "flow"],
        ...     feature_ranges={"temp": (100, 500), "pressure": (10, 100)}
        ... )
        >>> cf = explainer.generate_counterfactual(
        ...     instance={"temp": 400, "pressure": 80, "flow": 1000},
        ...     target_prediction=0.2
        ... )
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        immutable_features: Optional[List[str]] = None,
        categorical_features: Optional[Dict[str, List[Any]]] = None,
        process_heat_context: Optional[ProcessHeatContext] = None,
        max_iterations: int = 1000,
        tolerance: float = 0.01,
        random_state: int = 42,
    ):
        """
        Initialize counterfactual explainer.

        Args:
            model: ML model with predict/predict_proba method
            feature_names: Names of input features
            feature_ranges: Valid (min, max) ranges for each feature
            immutable_features: Features that cannot be changed
            categorical_features: Categorical features with allowed values
            process_heat_context: Domain context for process heat
            max_iterations: Maximum optimization iterations
            tolerance: Tolerance for target prediction
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}
        self.immutable_features = set(immutable_features or [])
        self.categorical_features = categorical_features or {}
        self.process_heat_context = process_heat_context
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        # Validate model has prediction method
        if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
            raise ValueError("Model must have 'predict' or 'predict_proba' method")

        # Set default ranges if not provided
        self._set_default_ranges()

        logger.info(
            f"CounterfactualExplainer initialized with {len(feature_names)} features, "
            f"{len(self.immutable_features)} immutable"
        )

    def _set_default_ranges(self) -> None:
        """Set default feature ranges if not provided."""
        for name in self.feature_names:
            if name not in self.feature_ranges:
                # Default to wide range, will be refined with actual data
                self.feature_ranges[name] = (-1e6, 1e6)

    def _get_prediction(self, X: np.ndarray) -> float:
        """
        Get model prediction for input.

        Args:
            X: Input array

        Returns:
            Prediction value
        """
        if hasattr(self.model, "predict_proba"):
            pred = self.model.predict_proba(X.reshape(1, -1))
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                return float(pred[0, 1])  # Binary classification positive class
            return float(pred[0])
        else:
            pred = self.model.predict(X.reshape(1, -1))
            return float(pred[0])

    def _dict_to_array(self, instance: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array."""
        return np.array([instance.get(name, 0.0) for name in self.feature_names])

    def _array_to_dict(self, X: np.ndarray) -> Dict[str, float]:
        """Convert feature array to dict."""
        return {name: float(X[i]) for i, name in enumerate(self.feature_names)}

    def generate_counterfactual(
        self,
        instance: Union[Dict[str, float], np.ndarray],
        target_prediction: Optional[float] = None,
        target_class: Optional[str] = None,
        max_features_to_change: int = 5,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> CounterfactualResult:
        """
        Generate counterfactual explanation.

        Finds the minimal changes to the instance that would result in
        the target prediction. Uses constrained optimization to ensure
        changes are realistic and minimal.

        Args:
            instance: Original feature values (dict or array)
            target_prediction: Target prediction value to achieve
            target_class: Target class label (for classification)
            max_features_to_change: Maximum features to modify
            feature_weights: Weights for prioritizing certain features

        Returns:
            CounterfactualResult with required changes

        Raises:
            ValueError: If no valid counterfactual can be found
        """
        start_time = datetime.now()

        # Convert instance to array if dict
        if isinstance(instance, dict):
            original_dict = instance.copy()
            X_original = self._dict_to_array(instance)
        else:
            X_original = instance.copy()
            original_dict = self._array_to_dict(X_original)

        # Get original prediction
        original_prediction = self._get_prediction(X_original)

        # Determine target
        if target_prediction is None:
            # Default: flip prediction (useful for classification)
            if original_prediction > 0.5:
                target_prediction = 0.2
            else:
                target_prediction = 0.8

        # Set up feature weights (prefer changing certain features)
        weights = feature_weights or {}
        for name in self.feature_names:
            if name not in weights:
                weights[name] = 1.0
            if name in self.immutable_features:
                weights[name] = 1e10  # Very high cost for immutable

        # Run optimization
        if SCIPY_AVAILABLE:
            counterfactual_X = self._optimize_counterfactual(
                X_original,
                target_prediction,
                weights,
                max_features_to_change
            )
        else:
            # Fallback: simple gradient-free search
            counterfactual_X = self._simple_search_counterfactual(
                X_original,
                target_prediction,
                max_features_to_change
            )

        # Get counterfactual prediction
        counterfactual_prediction = self._get_prediction(counterfactual_X)

        # Calculate changes
        changes_required = self._calculate_changes(X_original, counterfactual_X)

        # Calculate change magnitude
        change_magnitude = sum(
            abs(new - old) / (abs(old) + 1e-10)
            for _, (old, new) in changes_required.items()
        )

        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility(changes_required)

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            original_dict,
            changes_required,
            original_prediction,
            counterfactual_prediction
        )

        # Compute provenance
        provenance_data = {
            "original_instance": original_dict,
            "target_prediction": target_prediction,
            "counterfactual": self._array_to_dict(counterfactual_X),
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = compute_provenance_hash(provenance_data)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Counterfactual generated in {elapsed_ms:.2f}ms, "
            f"changed {len(changes_required)} features, "
            f"feasibility: {feasibility_score:.2f}"
        )

        return CounterfactualResult(
            original_prediction=original_prediction,
            original_class=None,
            counterfactual_prediction=counterfactual_prediction,
            counterfactual_class=target_class,
            changes_required=changes_required,
            change_magnitude=change_magnitude,
            num_features_changed=len(changes_required),
            feasibility_score=feasibility_score,
            explanation_text=explanation_text,
            provenance_hash=provenance_hash,
        )

    def _optimize_counterfactual(
        self,
        X_original: np.ndarray,
        target: float,
        weights: Dict[str, float],
        max_changes: int
    ) -> np.ndarray:
        """
        Optimize to find counterfactual using scipy.

        Args:
            X_original: Original feature values
            target: Target prediction
            weights: Feature change weights
            max_changes: Maximum features to change

        Returns:
            Counterfactual feature values
        """
        # Build bounds
        bounds = []
        for i, name in enumerate(self.feature_names):
            if name in self.immutable_features:
                # Fix immutable features
                bounds.append((X_original[i], X_original[i]))
            elif name in self.feature_ranges:
                bounds.append(self.feature_ranges[name])
            else:
                # Default bounds around original value
                val = X_original[i]
                bounds.append((val * 0.1, val * 10 if val > 0 else val * 10))

        weight_array = np.array([weights.get(name, 1.0) for name in self.feature_names])

        def objective(X: np.ndarray) -> float:
            """
            Objective: minimize weighted distance + prediction error.

            This is a multi-objective function that balances:
            1. Distance from original (want minimal changes)
            2. Prediction error from target (want to hit target)
            3. Sparsity (prefer fewer features changed)
            """
            # Prediction error
            pred = self._get_prediction(X)
            pred_error = (pred - target) ** 2

            # Weighted distance from original
            distance = np.sum(weight_array * ((X - X_original) / (np.abs(X_original) + 1e-10)) ** 2)

            # Sparsity penalty (number of features changed)
            n_changed = np.sum(np.abs(X - X_original) > 1e-6)
            sparsity_penalty = max(0, n_changed - max_changes) * 10

            return pred_error * 100 + distance + sparsity_penalty

        # Try multiple random starts
        best_result = None
        best_score = float("inf")

        np.random.seed(self.random_state)

        for _ in range(5):
            # Random starting point between original and bounds
            x0 = X_original.copy()
            for i, name in enumerate(self.feature_names):
                if name not in self.immutable_features:
                    x0[i] += np.random.uniform(-0.1, 0.1) * x0[i]
                    x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])

            try:
                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.max_iterations}
                )

                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x

            except Exception as e:
                logger.warning(f"Optimization attempt failed: {e}")

        if best_result is None:
            # Return original if optimization failed
            logger.warning("Counterfactual optimization failed, returning original")
            return X_original.copy()

        return best_result

    def _simple_search_counterfactual(
        self,
        X_original: np.ndarray,
        target: float,
        max_changes: int
    ) -> np.ndarray:
        """
        Simple gradient-free counterfactual search (fallback).

        Args:
            X_original: Original feature values
            target: Target prediction
            max_changes: Maximum features to change

        Returns:
            Counterfactual feature values
        """
        np.random.seed(self.random_state)
        X_best = X_original.copy()
        best_error = abs(self._get_prediction(X_original) - target)

        # Get mutable feature indices
        mutable_indices = [
            i for i, name in enumerate(self.feature_names)
            if name not in self.immutable_features
        ]

        # Random search with increasing perturbation
        for step_size in [0.01, 0.05, 0.1, 0.2, 0.5]:
            for _ in range(100):
                X_candidate = X_original.copy()

                # Randomly select features to change
                n_change = min(max_changes, len(mutable_indices))
                features_to_change = np.random.choice(
                    mutable_indices, size=n_change, replace=False
                )

                for idx in features_to_change:
                    name = self.feature_names[idx]
                    val = X_candidate[idx]

                    # Random perturbation
                    delta = np.random.uniform(-step_size, step_size) * abs(val + 1e-10)
                    new_val = val + delta

                    # Clip to bounds
                    if name in self.feature_ranges:
                        new_val = np.clip(
                            new_val,
                            self.feature_ranges[name][0],
                            self.feature_ranges[name][1]
                        )

                    X_candidate[idx] = new_val

                # Check if this is better
                pred = self._get_prediction(X_candidate)
                error = abs(pred - target)

                if error < best_error:
                    best_error = error
                    X_best = X_candidate.copy()

                    if error < self.tolerance:
                        return X_best

        return X_best

    def _calculate_changes(
        self,
        X_original: np.ndarray,
        X_counterfactual: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate which features changed and by how much.

        Args:
            X_original: Original values
            X_counterfactual: Counterfactual values

        Returns:
            Dict of {feature_name: (original_value, new_value)}
        """
        changes = {}
        for i, name in enumerate(self.feature_names):
            original = X_original[i]
            new = X_counterfactual[i]

            # Only include meaningful changes
            if abs(new - original) > 1e-6:
                changes[name] = (float(original), float(new))

        return changes

    def _calculate_feasibility(
        self,
        changes: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Calculate feasibility score for changes.

        Args:
            changes: Required changes

        Returns:
            Feasibility score 0-1 (1 = highly feasible)
        """
        if not changes:
            return 1.0

        feasibility_scores = []

        for name, (original, new) in changes.items():
            # Check if within valid range
            if name in self.feature_ranges:
                low, high = self.feature_ranges[name]
                if low <= new <= high:
                    range_score = 1.0
                else:
                    range_score = 0.5
            else:
                range_score = 0.8  # Unknown range, moderate confidence

            # Check change magnitude (smaller = more feasible)
            rel_change = abs(new - original) / (abs(original) + 1e-10)
            magnitude_score = max(0.0, 1.0 - rel_change)

            # Check if immutable
            if name in self.immutable_features:
                immutable_score = 0.0
            else:
                immutable_score = 1.0

            feature_score = 0.5 * range_score + 0.3 * magnitude_score + 0.2 * immutable_score
            feasibility_scores.append(feature_score)

        return sum(feasibility_scores) / len(feasibility_scores)

    def _generate_explanation_text(
        self,
        original: Dict[str, float],
        changes: Dict[str, Tuple[float, float]],
        original_prediction: float,
        counterfactual_prediction: float
    ) -> str:
        """
        Generate human-readable explanation of counterfactual.

        Args:
            original: Original feature values
            changes: Required changes
            original_prediction: Original prediction
            counterfactual_prediction: Counterfactual prediction

        Returns:
            Human-readable explanation
        """
        if not changes:
            return "No changes needed - the prediction already meets the target."

        # Build explanation
        lines = [
            f"To change the prediction from {original_prediction:.2%} to {counterfactual_prediction:.2%}, "
            f"the following changes would be needed:"
        ]

        for i, (name, (old, new)) in enumerate(changes.items(), 1):
            change = new - old
            direction = "increase" if change > 0 else "decrease"
            pct_change = abs(change) / (abs(old) + 1e-10) * 100

            # Get units from context if available
            units = ""
            if self.process_heat_context and self.process_heat_context.feature_units:
                units = self.process_heat_context.feature_units.get(name, "")

            readable_name = name.replace("_", " ").title()

            lines.append(
                f"  {i}. {readable_name}: {direction} from {old:.2f}{units} to {new:.2f}{units} "
                f"({pct_change:.1f}% change)"
            )

        return "\n".join(lines)

    def explain_what_if(
        self,
        instance: Union[Dict[str, float], np.ndarray],
        feature_changes: Dict[str, float],
        scenario_name: str = "what-if"
    ) -> WhatIfResult:
        """
        Analyze what-if scenario with specific feature changes.

        This method lets users explore specific scenarios:
        "What if temperature increased by 50F?"
        "What if we doubled the flow rate?"

        Args:
            instance: Original feature values
            feature_changes: Changes to apply {feature: new_value}
            scenario_name: Name for this scenario

        Returns:
            WhatIfResult with prediction changes
        """
        start_time = datetime.now()

        # Convert instance to array if dict
        if isinstance(instance, dict):
            original_dict = instance.copy()
            X_original = self._dict_to_array(instance)
        else:
            X_original = instance.copy()
            original_dict = self._array_to_dict(X_original)

        # Get original prediction
        original_prediction = self._get_prediction(X_original)

        # Apply changes
        X_modified = X_original.copy()
        applied_changes = {}

        for name, new_value in feature_changes.items():
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                old_value = X_original[idx]
                X_modified[idx] = new_value
                applied_changes[name] = (float(old_value), float(new_value))
            else:
                logger.warning(f"Unknown feature '{name}' in what-if scenario")

        # Get modified prediction
        modified_prediction = self._get_prediction(X_modified)
        prediction_change = modified_prediction - original_prediction

        # Calculate sensitivity (how much prediction changes per unit feature change)
        sensitivity = {}
        for name, (old, new) in applied_changes.items():
            if abs(new - old) > 1e-10:
                sensitivity[name] = prediction_change / (new - old)

        # Generate explanation
        direction = "increased" if prediction_change > 0 else "decreased"
        explanation_text = (
            f"In scenario '{scenario_name}', the prediction {direction} "
            f"from {original_prediction:.2%} to {modified_prediction:.2%} "
            f"(change of {prediction_change:+.2%})."
        )

        # Provenance
        provenance_data = {
            "scenario_name": scenario_name,
            "original_prediction": original_prediction,
            "modified_prediction": modified_prediction,
            "changes": applied_changes,
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = compute_provenance_hash(provenance_data)

        return WhatIfResult(
            scenario_name=scenario_name,
            original_prediction=original_prediction,
            modified_prediction=modified_prediction,
            prediction_change=prediction_change,
            feature_changes=applied_changes,
            sensitivity=sensitivity,
            explanation_text=explanation_text,
            provenance_hash=provenance_hash,
        )

    def generate_multiple_counterfactuals(
        self,
        instance: Union[Dict[str, float], np.ndarray],
        target_prediction: float,
        num_counterfactuals: int = 5,
        diversity_weight: float = 0.3
    ) -> List[CounterfactualResult]:
        """
        Generate diverse set of counterfactual explanations.

        Useful for giving users multiple options to achieve a goal.

        Args:
            instance: Original feature values
            target_prediction: Target prediction to achieve
            num_counterfactuals: Number of counterfactuals to generate
            diversity_weight: Weight for diversity in optimization

        Returns:
            List of diverse CounterfactualResults
        """
        results = []
        previously_changed = set()

        for i in range(num_counterfactuals):
            # Increase weight for previously changed features
            weights = {}
            for name in self.feature_names:
                if name in previously_changed:
                    weights[name] = 1.0 + diversity_weight * i
                else:
                    weights[name] = 1.0

            result = self.generate_counterfactual(
                instance=instance,
                target_prediction=target_prediction,
                feature_weights=weights
            )

            results.append(result)

            # Track which features were changed
            for name in result.changes_required:
                previously_changed.add(name)

        return results

    def set_feature_ranges_from_data(self, X: np.ndarray) -> None:
        """
        Set feature ranges based on observed data.

        Args:
            X: Training/reference data (samples x features)
        """
        for i, name in enumerate(self.feature_names):
            col = X[:, i]
            # Use percentiles to exclude outliers
            low = float(np.percentile(col, 1))
            high = float(np.percentile(col, 99))
            self.feature_ranges[name] = (low, high)

        logger.info(f"Updated feature ranges from {len(X)} samples")


class ProcessHeatCounterfactualExplainer(CounterfactualExplainer):
    """
    Process Heat-specific Counterfactual Explainer.

    Extends CounterfactualExplainer with domain knowledge for
    process heat applications (boilers, furnaces, heat exchangers).

    Includes:
    - Pre-defined feature ranges for process heat equipment
    - Domain-specific immutable features
    - Process heat explanation templates
    """

    # Default ranges for common process heat features
    PROCESS_HEAT_RANGES = {
        # Temperature features (Fahrenheit)
        "flue_gas_temperature": (200.0, 800.0),
        "stack_temperature": (250.0, 600.0),
        "inlet_temperature": (50.0, 300.0),
        "outlet_temperature": (100.0, 500.0),
        "ambient_temperature": (-20.0, 120.0),

        # Pressure features (PSI)
        "steam_pressure": (15.0, 300.0),
        "operating_pressure": (0.0, 500.0),
        "differential_pressure": (0.0, 50.0),

        # Flow features
        "flow_rate": (100.0, 100000.0),
        "fuel_flow_rate": (0.0, 10000.0),
        "air_flow_rate": (0.0, 50000.0),

        # Efficiency/performance
        "efficiency": (0.5, 0.99),
        "excess_air": (0.0, 50.0),
        "fouling_factor": (0.0, 1.0),

        # Time features
        "days_since_cleaning": (0, 365),
        "operating_hours": (0, 8760),
        "cycles": (0, 10000),
    }

    # Features that typically cannot be changed
    DEFAULT_IMMUTABLE = [
        "equipment_type",
        "equipment_age",
        "fuel_type",
        "design_capacity",
    ]

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        equipment_type: str = "boiler",
        **kwargs
    ):
        """
        Initialize process heat counterfactual explainer.

        Args:
            model: ML model
            feature_names: Feature names
            equipment_type: Type of equipment (boiler, furnace, etc.)
            **kwargs: Additional arguments for base class
        """
        # Set up domain-specific ranges
        feature_ranges = kwargs.pop("feature_ranges", {})
        for name in feature_names:
            if name not in feature_ranges and name in self.PROCESS_HEAT_RANGES:
                feature_ranges[name] = self.PROCESS_HEAT_RANGES[name]

        # Set up immutable features
        immutable = kwargs.pop("immutable_features", [])
        immutable = list(set(immutable + [
            f for f in self.DEFAULT_IMMUTABLE if f in feature_names
        ]))

        # Create process heat context
        context = ProcessHeatContext(
            equipment_type=equipment_type,
            process_type="steam" if "steam" in equipment_type.lower() else "combustion",
            feature_units={
                "temperature": "F",
                "pressure": "PSI",
                "flow_rate": "lb/hr",
                "efficiency": "%",
            }
        )

        super().__init__(
            model=model,
            feature_names=feature_names,
            feature_ranges=feature_ranges,
            immutable_features=immutable,
            process_heat_context=context,
            **kwargs
        )

        self.equipment_type = equipment_type

        logger.info(
            f"ProcessHeatCounterfactualExplainer initialized for {equipment_type}"
        )

    def suggest_maintenance_actions(
        self,
        instance: Dict[str, float],
        target_efficiency: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> CounterfactualResult:
        """
        Suggest maintenance actions to improve efficiency or reduce risk.

        Args:
            instance: Current operating conditions
            target_efficiency: Desired efficiency level
            target_risk: Desired risk level (lower is better)

        Returns:
            Counterfactual with maintenance recommendations
        """
        # Determine target based on what's provided
        if target_efficiency is not None:
            target_prediction = target_efficiency
        elif target_risk is not None:
            target_prediction = target_risk
        else:
            # Default: improve by 10%
            original_pred = self._get_prediction(self._dict_to_array(instance))
            target_prediction = original_pred * 0.9  # 10% improvement

        # Only allow maintenance-related features to change
        maintenance_features = [
            "days_since_cleaning",
            "fouling_factor",
            "cycles",
            "excess_air",
        ]

        feature_weights = {}
        for name in self.feature_names:
            if name in maintenance_features:
                feature_weights[name] = 0.5  # Prefer changing these
            else:
                feature_weights[name] = 10.0  # Avoid changing these

        return self.generate_counterfactual(
            instance=instance,
            target_prediction=target_prediction,
            feature_weights=feature_weights
        )
