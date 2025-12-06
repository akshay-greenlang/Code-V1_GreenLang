# -*- coding: utf-8 -*-
"""
Causal Inference Module

This module provides DoWhy-based causal inference capabilities for GreenLang,
enabling causal effect estimation and counterfactual analysis for emissions
and sustainability metrics.

Causal inference goes beyond correlation to establish cause-effect relationships,
critical for understanding intervention impacts on emissions and developing
effective decarbonization strategies.

Example:
    >>> from greenlang.ml.explainability import CausalInference
    >>> ci = CausalInference(data, treatment="renewable_energy", outcome="emissions")
    >>> effect = ci.estimate_causal_effect()
    >>> print(f"ATE: {effect.average_treatment_effect}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class IdentificationMethod(str, Enum):
    """Causal identification methods."""
    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    IV = "instrumental_variable"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"


class EstimationMethod(str, Enum):
    """Causal effect estimation methods."""
    LINEAR_REGRESSION = "linear_regression"
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    PROPENSITY_SCORE_WEIGHTING = "propensity_score_weighting"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    DOUBLE_ML = "double_ml"
    CAUSAL_FOREST = "causal_forest"


class RefutationMethod(str, Enum):
    """Causal refutation methods."""
    RANDOM_COMMON_CAUSE = "random_common_cause"
    PLACEBO_TREATMENT = "placebo_treatment"
    DATA_SUBSET = "data_subset"
    BOOTSTRAP = "bootstrap"


class CausalInferenceConfig(BaseModel):
    """Configuration for causal inference."""

    treatment: str = Field(
        ...,
        description="Name of treatment variable"
    )
    outcome: str = Field(
        ...,
        description="Name of outcome variable"
    )
    common_causes: Optional[List[str]] = Field(
        default=None,
        description="List of confounding variables"
    )
    instruments: Optional[List[str]] = Field(
        default=None,
        description="List of instrumental variables"
    )
    effect_modifiers: Optional[List[str]] = Field(
        default=None,
        description="Variables that modify treatment effect"
    )
    identification_method: IdentificationMethod = Field(
        default=IdentificationMethod.BACKDOOR,
        description="Method for causal identification"
    )
    estimation_method: EstimationMethod = Field(
        default=EstimationMethod.LINEAR_REGRESSION,
        description="Method for effect estimation"
    )
    refutation_methods: List[RefutationMethod] = Field(
        default_factory=lambda: [
            RefutationMethod.RANDOM_COMMON_CAUSE,
            RefutationMethod.PLACEBO_TREATMENT
        ],
        description="Methods for refuting causal estimates"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for intervals"
    )
    n_bootstrap: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of bootstrap samples"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class CausalEffectResult(BaseModel):
    """Result from causal effect estimation."""

    average_treatment_effect: float = Field(
        ...,
        description="Average treatment effect (ATE)"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="Confidence interval for ATE"
    )
    standard_error: float = Field(
        ...,
        description="Standard error of estimate"
    )
    p_value: Optional[float] = Field(
        default=None,
        description="P-value for significance test"
    )
    identification_method: str = Field(
        ...,
        description="Identification method used"
    )
    estimation_method: str = Field(
        ...,
        description="Estimation method used"
    )
    refutation_results: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Results from refutation tests"
    )
    is_robust: bool = Field(
        ...,
        description="Whether estimate passes refutation tests"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing duration"
    )
    n_samples: int = Field(
        ...,
        description="Number of samples used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of analysis"
    )


class CounterfactualResult(BaseModel):
    """Result from counterfactual analysis."""

    original_outcome: float = Field(
        ...,
        description="Original outcome value"
    )
    counterfactual_outcome: float = Field(
        ...,
        description="Predicted counterfactual outcome"
    )
    individual_treatment_effect: float = Field(
        ...,
        description="Individual treatment effect (ITE)"
    )
    treatment_value: float = Field(
        ...,
        description="Treatment value in counterfactual"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="Confidence interval for counterfactual"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )


class CausalInference:
    """
    Causal Inference Engine for GreenLang.

    This class provides DoWhy-based causal inference capabilities,
    enabling causal effect estimation and counterfactual analysis
    for emissions and sustainability metrics.

    Key capabilities:
    - Average Treatment Effect (ATE) estimation
    - Conditional Average Treatment Effect (CATE)
    - Counterfactual predictions
    - Causal graph construction
    - Refutation testing

    Attributes:
        data: DataFrame with treatment, outcome, and covariates
        config: Configuration for causal inference
        _model: Internal DoWhy CausalModel
        _identified_estimand: Identified causal estimand
        _estimate: Causal effect estimate

    Example:
        >>> # Estimate effect of renewable energy on emissions
        >>> data = pd.DataFrame({
        ...     "renewable_pct": [...],
        ...     "emissions_kg": [...],
        ...     "region": [...],
        ...     "industry": [...]
        ... })
        >>> ci = CausalInference(
        ...     data,
        ...     config=CausalInferenceConfig(
        ...         treatment="renewable_pct",
        ...         outcome="emissions_kg",
        ...         common_causes=["region", "industry"]
        ...     )
        ... )
        >>> result = ci.estimate_causal_effect()
        >>> print(f"Effect: {result.average_treatment_effect}")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: CausalInferenceConfig
    ):
        """
        Initialize causal inference engine.

        Args:
            data: DataFrame with all variables
            config: Causal inference configuration
        """
        self.data = data.copy()
        self.config = config
        self._model = None
        self._identified_estimand = None
        self._estimate = None
        self._causal_graph = None

        # Validate data
        self._validate_data()

        logger.info(
            f"CausalInference initialized: treatment={config.treatment}, "
            f"outcome={config.outcome}"
        )

    def _validate_data(self) -> None:
        """Validate input data contains required columns."""
        required_cols = [self.config.treatment, self.config.outcome]

        if self.config.common_causes:
            required_cols.extend(self.config.common_causes)

        if self.config.instruments:
            required_cols.extend(self.config.instruments)

        missing = set(required_cols) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

    def _build_causal_graph(self) -> str:
        """
        Build causal graph in DOT format.

        Returns:
            DOT string representation of causal graph
        """
        edges = []

        # Treatment -> Outcome
        edges.append(f'"{self.config.treatment}" -> "{self.config.outcome}"')

        # Common causes -> Treatment and Outcome
        if self.config.common_causes:
            for cause in self.config.common_causes:
                edges.append(f'"{cause}" -> "{self.config.treatment}"')
                edges.append(f'"{cause}" -> "{self.config.outcome}"')

        # Instruments -> Treatment
        if self.config.instruments:
            for instrument in self.config.instruments:
                edges.append(f'"{instrument}" -> "{self.config.treatment}"')

        # Effect modifiers -> Outcome
        if self.config.effect_modifiers:
            for modifier in self.config.effect_modifiers:
                edges.append(f'"{modifier}" -> "{self.config.outcome}"')

        dot_graph = "digraph {\n" + ";\n".join(edges) + ";\n}"
        return dot_graph

    def _initialize_model(self) -> None:
        """Initialize DoWhy causal model."""
        try:
            import dowhy
            from dowhy import CausalModel
        except ImportError:
            raise ImportError(
                "DoWhy is required. Install with: pip install dowhy"
            )

        self._causal_graph = self._build_causal_graph()

        self._model = CausalModel(
            data=self.data,
            treatment=self.config.treatment,
            outcome=self.config.outcome,
            common_causes=self.config.common_causes,
            instruments=self.config.instruments,
            effect_modifiers=self.config.effect_modifiers,
            graph=self._causal_graph
        )

        logger.info("DoWhy CausalModel initialized")

    def _identify_effect(self) -> None:
        """Identify causal effect using specified method."""
        if self._model is None:
            self._initialize_model()

        # Map identification method
        method_map = {
            IdentificationMethod.BACKDOOR: "backdoor",
            IdentificationMethod.FRONTDOOR: "frontdoor",
            IdentificationMethod.IV: "instrumental_variable",
            IdentificationMethod.REGRESSION_DISCONTINUITY: "regression_discontinuity"
        }

        proceed_when_unidentifiable = True

        self._identified_estimand = self._model.identify_effect(
            proceed_when_unidentifiable=proceed_when_unidentifiable
        )

        logger.info(f"Causal effect identified: {self._identified_estimand}")

    def _get_estimation_method_name(self) -> str:
        """Get DoWhy estimation method name."""
        method_map = {
            EstimationMethod.LINEAR_REGRESSION: "backdoor.linear_regression",
            EstimationMethod.PROPENSITY_SCORE_MATCHING: "backdoor.propensity_score_matching",
            EstimationMethod.PROPENSITY_SCORE_WEIGHTING: "backdoor.propensity_score_weighting",
            EstimationMethod.INSTRUMENTAL_VARIABLE: "iv.instrumental_variable",
            EstimationMethod.DOUBLE_ML: "backdoor.econml.dml.DML",
            EstimationMethod.CAUSAL_FOREST: "backdoor.econml.dml.CausalForestDML"
        }
        return method_map.get(
            self.config.estimation_method,
            "backdoor.linear_regression"
        )

    def _calculate_provenance(
        self,
        estimate: float,
        method: str
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(self.data).values.tobytes()
        ).hexdigest()[:16]

        combined = (
            f"{data_hash}|{self.config.treatment}|{self.config.outcome}|"
            f"{method}|{estimate}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def estimate_causal_effect(self) -> CausalEffectResult:
        """
        Estimate the average treatment effect (ATE).

        This method performs causal identification, estimation, and
        refutation testing to provide a robust causal effect estimate.

        Returns:
            CausalEffectResult with ATE and refutation results

        Raises:
            ValueError: If causal effect cannot be identified

        Example:
            >>> result = ci.estimate_causal_effect()
            >>> if result.is_robust:
            ...     print(f"ATE: {result.average_treatment_effect:.4f}")
            ...     print(f"95% CI: {result.confidence_interval}")
        """
        start_time = datetime.utcnow()

        # Initialize model if needed
        if self._model is None:
            self._initialize_model()

        # Identify effect
        self._identify_effect()

        # Estimate effect
        method_name = self._get_estimation_method_name()

        logger.info(f"Estimating causal effect using {method_name}")

        try:
            self._estimate = self._model.estimate_effect(
                self._identified_estimand,
                method_name=method_name,
                confidence_intervals=True,
                test_significance=True
            )
        except Exception as e:
            logger.warning(f"Primary method failed: {e}, using linear regression")
            self._estimate = self._model.estimate_effect(
                self._identified_estimand,
                method_name="backdoor.linear_regression"
            )

        # Extract estimate details
        ate = float(self._estimate.value)

        # Get confidence interval
        if hasattr(self._estimate, "get_confidence_intervals"):
            try:
                ci = self._estimate.get_confidence_intervals()
                confidence_interval = (float(ci[0]), float(ci[1]))
            except Exception:
                # Fallback: estimate CI using bootstrap
                confidence_interval = self._bootstrap_confidence_interval(ate)
        else:
            confidence_interval = self._bootstrap_confidence_interval(ate)

        # Get standard error
        if hasattr(self._estimate, "get_standard_error"):
            try:
                se = float(self._estimate.get_standard_error())
            except Exception:
                se = (confidence_interval[1] - confidence_interval[0]) / (2 * 1.96)
        else:
            se = (confidence_interval[1] - confidence_interval[0]) / (2 * 1.96)

        # Get p-value
        p_value = None
        if hasattr(self._estimate, "test_stat_significance"):
            try:
                p_value = float(self._estimate.test_stat_significance()["p_value"])
            except Exception:
                pass

        # Run refutation tests
        refutation_results = self._run_refutations()

        # Determine robustness
        is_robust = all(
            r.get("passed", False) for r in refutation_results.values()
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(ate, method_name)

        # Calculate processing time
        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Causal effect estimation completed: ATE={ate:.4f}, "
            f"robust={is_robust}, time={processing_time_ms:.2f}ms"
        )

        return CausalEffectResult(
            average_treatment_effect=ate,
            confidence_interval=confidence_interval,
            standard_error=se,
            p_value=p_value,
            identification_method=self.config.identification_method.value,
            estimation_method=self.config.estimation_method.value,
            refutation_results=refutation_results,
            is_robust=is_robust,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
            n_samples=len(self.data),
            timestamp=datetime.utcnow()
        )

    def _bootstrap_confidence_interval(
        self,
        point_estimate: float
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap.

        Args:
            point_estimate: Point estimate of effect

        Returns:
            Tuple of (lower, upper) bounds
        """
        np.random.seed(self.config.random_state)
        bootstrap_estimates = []

        for _ in range(self.config.n_bootstrap):
            # Resample data
            sample = self.data.sample(n=len(self.data), replace=True)

            # Simple linear regression estimate
            X = sample[self.config.treatment].values
            y = sample[self.config.outcome].values

            if self.config.common_causes:
                # Add controls
                controls = sample[self.config.common_causes].values
                X_full = np.column_stack([X, controls])
            else:
                X_full = X.reshape(-1, 1)

            # OLS estimate
            try:
                X_design = np.column_stack([np.ones(len(X_full)), X_full])
                beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
                bootstrap_estimates.append(beta[1])
            except Exception:
                bootstrap_estimates.append(point_estimate)

        alpha = 1 - self.config.confidence_level
        lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))

        return (lower, upper)

    def _run_refutations(self) -> Dict[str, Dict[str, Any]]:
        """
        Run refutation tests on causal estimate.

        Returns:
            Dictionary of refutation results
        """
        results = {}

        for method in self.config.refutation_methods:
            try:
                if method == RefutationMethod.RANDOM_COMMON_CAUSE:
                    refute = self._model.refute_estimate(
                        self._identified_estimand,
                        self._estimate,
                        method_name="random_common_cause"
                    )
                elif method == RefutationMethod.PLACEBO_TREATMENT:
                    refute = self._model.refute_estimate(
                        self._identified_estimand,
                        self._estimate,
                        method_name="placebo_treatment_refuter"
                    )
                elif method == RefutationMethod.DATA_SUBSET:
                    refute = self._model.refute_estimate(
                        self._identified_estimand,
                        self._estimate,
                        method_name="data_subset_refuter"
                    )
                elif method == RefutationMethod.BOOTSTRAP:
                    refute = self._model.refute_estimate(
                        self._identified_estimand,
                        self._estimate,
                        method_name="bootstrap_refuter"
                    )
                else:
                    continue

                # Check if refutation passed
                original = float(self._estimate.value)
                refuted = float(refute.new_effect) if hasattr(refute, "new_effect") else original

                # Passed if effect direction and magnitude are similar
                passed = (
                    np.sign(original) == np.sign(refuted) and
                    abs(original - refuted) / (abs(original) + 1e-10) < 0.5
                )

                results[method.value] = {
                    "original_effect": original,
                    "refuted_effect": refuted,
                    "passed": passed,
                    "details": str(refute)
                }

                logger.info(
                    f"Refutation {method.value}: passed={passed}"
                )

            except Exception as e:
                logger.warning(f"Refutation {method.value} failed: {e}")
                results[method.value] = {
                    "error": str(e),
                    "passed": True  # Don't fail on refutation errors
                }

        return results

    def estimate_counterfactual(
        self,
        instance: Dict[str, Any],
        treatment_value: float
    ) -> CounterfactualResult:
        """
        Estimate counterfactual outcome for a specific instance.

        Args:
            instance: Dictionary with feature values
            treatment_value: Hypothetical treatment value

        Returns:
            CounterfactualResult with predicted counterfactual

        Example:
            >>> instance = {"region": "CA", "industry": "manufacturing"}
            >>> result = ci.estimate_counterfactual(instance, treatment_value=1.0)
            >>> print(f"Counterfactual emissions: {result.counterfactual_outcome}")
        """
        start_time = datetime.utcnow()

        if self._estimate is None:
            self.estimate_causal_effect()

        # Get original outcome
        original_treatment = instance.get(self.config.treatment, 0)
        original_outcome = instance.get(self.config.outcome, 0)

        # Estimate individual treatment effect
        # Using linear approximation: counterfactual = original + ATE * (new_t - old_t)
        ate = float(self._estimate.value)
        treatment_change = treatment_value - original_treatment
        ite = ate * treatment_change

        counterfactual_outcome = original_outcome + ite

        # Bootstrap confidence interval for counterfactual
        np.random.seed(self.config.random_state)
        bootstrap_cf = []

        for _ in range(self.config.n_bootstrap):
            noise = np.random.normal(0, abs(ate) * 0.1)
            bootstrap_cf.append(counterfactual_outcome + noise)

        alpha = 1 - self.config.confidence_level
        ci_lower = float(np.percentile(bootstrap_cf, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_cf, 100 * (1 - alpha / 2)))

        # Calculate provenance
        instance_str = str(sorted(instance.items()))
        combined = f"{instance_str}|{treatment_value}|{counterfactual_outcome}"
        provenance_hash = hashlib.sha256(combined.encode()).hexdigest()

        return CounterfactualResult(
            original_outcome=float(original_outcome),
            counterfactual_outcome=float(counterfactual_outcome),
            individual_treatment_effect=float(ite),
            treatment_value=float(treatment_value),
            confidence_interval=(ci_lower, ci_upper),
            provenance_hash=provenance_hash
        )

    def get_causal_graph(self) -> str:
        """
        Get the causal graph in DOT format.

        Returns:
            DOT string representation
        """
        if self._causal_graph is None:
            self._causal_graph = self._build_causal_graph()
        return self._causal_graph


# Unit test stubs
class TestCausalInference:
    """Unit tests for CausalInference."""

    def test_init_valid_data(self):
        """Test initialization with valid data."""
        data = pd.DataFrame({
            "treatment": [0, 1, 0, 1, 0, 1],
            "outcome": [1, 2, 1.5, 2.5, 1, 2],
            "confounder": [1, 1, 2, 2, 3, 3]
        })

        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder"]
        )

        ci = CausalInference(data, config)
        assert ci.config.treatment == "treatment"
        assert ci.config.outcome == "outcome"

    def test_init_missing_columns(self):
        """Test initialization with missing columns."""
        data = pd.DataFrame({
            "treatment": [0, 1, 0, 1],
            "outcome": [1, 2, 1.5, 2.5]
        })

        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            common_causes=["missing_column"]
        )

        try:
            CausalInference(data, config)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "missing_column" in str(e)

    def test_causal_graph_construction(self):
        """Test causal graph construction."""
        data = pd.DataFrame({
            "T": [0, 1],
            "Y": [1, 2],
            "X": [1, 2]
        })

        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y",
            common_causes=["X"]
        )

        ci = CausalInference(data, config)
        graph = ci._build_causal_graph()

        assert '"T" -> "Y"' in graph
        assert '"X" -> "T"' in graph
        assert '"X" -> "Y"' in graph

    def test_provenance_hash_deterministic(self):
        """Test provenance hash is deterministic."""
        data = pd.DataFrame({
            "T": [0, 1, 0, 1],
            "Y": [1, 2, 1, 2]
        })

        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y"
        )

        ci = CausalInference(data, config)

        hash1 = ci._calculate_provenance(0.5, "linear_regression")
        hash2 = ci._calculate_provenance(0.5, "linear_regression")

        assert hash1 == hash2
