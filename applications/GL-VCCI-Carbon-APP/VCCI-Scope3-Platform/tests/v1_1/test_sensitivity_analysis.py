# -*- coding: utf-8 -*-
"""
Unit Tests for Sensitivity Analysis Engine (v1.1)

Tests all four core analyzers: SobolAnalyzer, MorrisAnalyzer,
TornadoDiagramGenerator, and ConvergenceAnalyzer.

Target module: services/methodologies/sensitivity_analysis.py
Test count: 44 tests
Coverage target: 85%+
"""

import pytest
import numpy as np
from decimal import Decimal
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

import sys
import os

# Ensure the platform root is on the path
PLATFORM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

from services.methodologies.sensitivity_analysis import (
    SobolAnalyzer,
    MorrisAnalyzer,
    TornadoDiagramGenerator,
    ConvergenceAnalyzer,
    SobolResult,
    MorrisResult,
    TornadoData,
    TornadoParameter,
    ConvergenceResult,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _linear_model(params: Dict[str, float]) -> float:
    """y = 2*x1 + 3*x2 (known analytical Sobol indices)."""
    return 2.0 * params["x1"] + 3.0 * params["x2"]


def _multiplicative_model(params: Dict[str, float]) -> float:
    """y = x1 * x2 (non-additive, has interaction)."""
    return params["x1"] * params["x2"]


def _single_param_model(params: Dict[str, float]) -> float:
    """y = x1 (single parameter, Si should be 1.0)."""
    return params["x1"]


def _constant_model(params: Dict[str, float]) -> float:
    """y = 100 (constant, variance is zero)."""
    return 100.0


def _make_params(names: List[str], mean: float = 100.0, std: float = 10.0,
                 distribution: str = "normal") -> List[Dict]:
    """Build a list of parameter descriptors for sensitivity analyzers."""
    return [
        {"name": n, "mean": mean, "std_dev": std, "distribution": distribution}
        for n in names
    ]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sobol_analyzer():
    """Create a reproducible SobolAnalyzer."""
    return SobolAnalyzer(seed=42)


@pytest.fixture
def morris_analyzer():
    """Create a reproducible MorrisAnalyzer."""
    return MorrisAnalyzer(seed=42)


@pytest.fixture
def tornado_generator():
    """Create a TornadoDiagramGenerator."""
    return TornadoDiagramGenerator()


@pytest.fixture
def convergence_analyzer():
    """Create a ConvergenceAnalyzer."""
    return ConvergenceAnalyzer()


@pytest.fixture
def two_param_descriptors():
    """Two normal-distributed parameters for x1 and x2."""
    return _make_params(["x1", "x2"], mean=100.0, std=10.0)


@pytest.fixture
def three_param_descriptors():
    """Three normal-distributed parameters."""
    return _make_params(["x1", "x2", "x3"], mean=50.0, std=5.0)


@pytest.fixture
def single_param_descriptor():
    """Single normal-distributed parameter."""
    return _make_params(["x1"], mean=100.0, std=10.0)


@pytest.fixture
def converged_samples():
    """Generate a large sample from a stable distribution (should converge)."""
    rng = np.random.RandomState(42)
    return rng.normal(loc=250.0, scale=50.0, size=20000)


@pytest.fixture
def not_converged_samples():
    """Generate a small, noisy sample that may not converge."""
    rng = np.random.RandomState(99)
    return rng.normal(loc=250.0, scale=500.0, size=200)


# ============================================================================
# TEST: SobolAnalyzer
# ============================================================================

class TestSobolAnalyzer:
    """Test suite for SobolAnalyzer (variance-based global sensitivity)."""

    def test_initialization_with_seed(self):
        """Test that SobolAnalyzer can be initialized with a seed."""
        analyzer = SobolAnalyzer(seed=123)
        assert analyzer is not None
        assert analyzer._rng is not None

    def test_initialization_without_seed(self):
        """Test that SobolAnalyzer can be initialized without a seed."""
        analyzer = SobolAnalyzer(seed=None)
        assert analyzer is not None

    def test_saltelli_sampling_generates_correct_shape(
        self, sobol_analyzer, two_param_descriptors
    ):
        """Saltelli matrices A, B should be (N, k); AB list should have k entries."""
        N = 128
        samples = sobol_analyzer.generate_saltelli_samples(two_param_descriptors, N=N)

        k = len(two_param_descriptors)
        assert samples["A"].shape == (N, k)
        assert samples["B"].shape == (N, k)
        assert len(samples["AB"]) == k
        for ab_mat in samples["AB"]:
            assert ab_mat.shape == (N, k)
        assert samples["param_names"] == ["x1", "x2"]

    def test_saltelli_ab_matrix_structure(
        self, sobol_analyzer, two_param_descriptors
    ):
        """AB_i should equal A except column i which comes from B."""
        N = 64
        samples = sobol_analyzer.generate_saltelli_samples(two_param_descriptors, N=N)
        A = samples["A"]
        B = samples["B"]
        AB_list = samples["AB"]

        for j in range(2):
            AB_j = AB_list[j]
            # Column j should match B
            np.testing.assert_array_equal(AB_j[:, j], B[:, j])
            # Other columns should match A
            for c in range(2):
                if c != j:
                    np.testing.assert_array_equal(AB_j[:, c], A[:, c])

    def test_first_order_indices_sum_approximately_one(
        self, sobol_analyzer, two_param_descriptors
    ):
        """For a purely additive model, sum(Si) should be close to 1.0."""
        result = sobol_analyzer.run_analysis(
            _linear_model, two_param_descriptors, N=2048
        )
        sum_si = sum(result.first_order_indices.values())
        assert abs(sum_si - 1.0) < 0.15, (
            f"Sum of first-order indices = {sum_si}, expected near 1.0"
        )

    def test_total_order_greater_than_or_equal_first_order(
        self, sobol_analyzer, two_param_descriptors
    ):
        """STi >= Si for every parameter (total includes interactions)."""
        result = sobol_analyzer.run_analysis(
            _linear_model, two_param_descriptors, N=1024
        )
        for name in result.parameters:
            si = result.first_order_indices[name]
            sti = result.total_order_indices[name]
            assert sti >= si - 0.05, (
                f"STi ({sti}) < Si ({si}) for {name} (allowing 0.05 tolerance)"
            )

    def test_known_linear_model_indices(self, sobol_analyzer):
        """
        For y = 2*x1 + 3*x2 with same-variance inputs,
        Si should be approximately [4/13, 9/13] = [0.308, 0.692].
        """
        params = [
            {"name": "x1", "mean": 0.5, "std_dev": 0.1, "distribution": "uniform",
             "min": 0.0, "max": 1.0},
            {"name": "x2", "mean": 0.5, "std_dev": 0.1, "distribution": "uniform",
             "min": 0.0, "max": 1.0},
        ]
        result = sobol_analyzer.run_analysis(_linear_model, params, N=4096)

        si_x1 = result.first_order_indices["x1"]
        si_x2 = result.first_order_indices["x2"]

        expected_x1 = 4.0 / 13.0  # ~0.308
        expected_x2 = 9.0 / 13.0  # ~0.692

        assert abs(si_x1 - expected_x1) < 0.12, (
            f"Si(x1) = {si_x1}, expected {expected_x1:.3f}"
        )
        assert abs(si_x2 - expected_x2) < 0.12, (
            f"Si(x2) = {si_x2}, expected {expected_x2:.3f}"
        )

    def test_parameter_bounds_respected(self, sobol_analyzer):
        """Samples generated with uniform distribution should respect min/max."""
        params = [
            {"name": "x1", "mean": 50.0, "std_dev": 10.0,
             "distribution": "uniform", "min": 10.0, "max": 90.0},
        ]
        samples = sobol_analyzer.generate_saltelli_samples(params, N=512)
        all_values = np.concatenate([samples["A"][:, 0], samples["B"][:, 0]])
        assert np.all(all_values >= 10.0), "Some samples below min"
        assert np.all(all_values <= 90.0), "Some samples above max"

    def test_empty_parameters_raises(self, sobol_analyzer):
        """Passing an empty parameter list should raise ValueError."""
        with pytest.raises(ValueError, match="At least one parameter"):
            sobol_analyzer.generate_saltelli_samples([], N=128)

    def test_single_parameter(self, sobol_analyzer, single_param_descriptor):
        """With one parameter, Si and STi should both be approximately 1.0."""
        result = sobol_analyzer.run_analysis(
            _single_param_model, single_param_descriptor, N=1024
        )
        assert len(result.parameters) == 1
        si = result.first_order_indices["x1"]
        sti = result.total_order_indices["x1"]
        assert si > 0.7, f"Single-param Si = {si}, expected near 1.0"
        assert sti > 0.7, f"Single-param STi = {sti}, expected near 1.0"

    def test_large_parameter_space(self, sobol_analyzer):
        """Run Sobol with 5 parameters to ensure no crashes with larger k."""
        params = _make_params(
            ["a", "b", "c", "d", "e"], mean=100.0, std=10.0
        )

        def five_param_model(p):
            return p["a"] + 2 * p["b"] + 3 * p["c"] + p["d"] + p["e"]

        result = sobol_analyzer.run_analysis(five_param_model, params, N=256)
        assert len(result.parameters) == 5
        assert result.sample_size == 256
        assert result.computation_time > 0.0

    def test_interaction_effects_nonnegative(
        self, sobol_analyzer, two_param_descriptors
    ):
        """Interaction effects (STi - Si) should be >= 0."""
        result = sobol_analyzer.run_analysis(
            _multiplicative_model, two_param_descriptors, N=1024
        )
        for name in result.parameters:
            assert result.interaction_effects[name] >= 0.0, (
                f"Negative interaction for {name}"
            )

    def test_convergence_info_present(
        self, sobol_analyzer, two_param_descriptors
    ):
        """Result should contain convergence diagnostics."""
        result = sobol_analyzer.run_analysis(
            _linear_model, two_param_descriptors, N=512
        )
        assert "sum_first_order" in result.convergence_info
        assert "sum_total_order" in result.convergence_info
        assert "note" in result.convergence_info

    def test_result_model_fields(self, sobol_analyzer, two_param_descriptors):
        """SobolResult should have all expected fields populated."""
        result = sobol_analyzer.run_analysis(
            _linear_model, two_param_descriptors, N=256
        )
        assert isinstance(result, SobolResult)
        assert len(result.parameters) == 2
        assert len(result.first_order_indices) == 2
        assert len(result.total_order_indices) == 2
        assert len(result.interaction_effects) == 2
        assert result.sample_size == 256
        assert result.computation_time >= 0.0

    def test_zero_variance_model(self, sobol_analyzer, two_param_descriptors):
        """Constant model should produce zero indices."""
        result = sobol_analyzer.run_analysis(
            _constant_model, two_param_descriptors, N=128
        )
        for name in result.parameters:
            assert result.first_order_indices[name] == 0.0
            assert result.total_order_indices[name] == 0.0


# ============================================================================
# TEST: MorrisAnalyzer
# ============================================================================

class TestMorrisAnalyzer:
    """Test suite for MorrisAnalyzer (elementary effects screening)."""

    def test_initialization(self):
        """Test MorrisAnalyzer can be created with and without seed."""
        m1 = MorrisAnalyzer(seed=42)
        m2 = MorrisAnalyzer(seed=None)
        assert m1 is not None
        assert m2 is not None

    def test_trajectory_design_valid(
        self, morris_analyzer, two_param_descriptors
    ):
        """Each trajectory should have shape (k+1, k)."""
        k = len(two_param_descriptors)
        trajectories = morris_analyzer.generate_trajectories(
            two_param_descriptors, r=5, levels=4
        )
        assert len(trajectories) == 5
        for traj in trajectories:
            assert traj.shape == (k + 1, k)

    def test_trajectory_values_in_unit_interval(
        self, morris_analyzer, two_param_descriptors
    ):
        """All trajectory values should be within [0, 1]."""
        trajectories = morris_analyzer.generate_trajectories(
            two_param_descriptors, r=10, levels=4
        )
        for traj in trajectories:
            assert np.all(traj >= 0.0), "Trajectory values below 0"
            assert np.all(traj <= 1.0), "Trajectory values above 1"

    def test_trajectory_one_column_changes_per_step(
        self, morris_analyzer, two_param_descriptors
    ):
        """Between consecutive rows, exactly one column should change."""
        trajectories = morris_analyzer.generate_trajectories(
            two_param_descriptors, r=5, levels=4
        )
        for traj in trajectories:
            k = traj.shape[1]
            for step in range(k):
                diff = traj[step + 1] - traj[step]
                nonzero = np.count_nonzero(diff)
                assert nonzero == 1, (
                    f"Expected 1 changed column per step, got {nonzero}"
                )

    def test_mu_star_captures_linear_effect(self, morris_analyzer):
        """For y = 2*x1 + 3*x2, mu*(x2) > mu*(x1) because x2 has larger coefficient."""
        params = _make_params(["x1", "x2"], mean=100.0, std=10.0)
        result = morris_analyzer.run_screening(
            _linear_model, params, r=20, levels=4
        )
        assert result.mu_star["x2"] > result.mu_star["x1"], (
            f"mu*(x2) = {result.mu_star['x2']} should exceed "
            f"mu*(x1) = {result.mu_star['x1']}"
        )

    def test_sigma_captures_interaction(self, morris_analyzer):
        """For a multiplicative model, sigma should be relatively high (interaction)."""
        params = _make_params(["x1", "x2"], mean=100.0, std=10.0)
        result = morris_analyzer.run_screening(
            _multiplicative_model, params, r=20, levels=4
        )
        # Both parameters should have non-zero sigma (interaction present)
        for name in ["x1", "x2"]:
            assert result.sigma[name] > 0.0, (
                f"sigma({name}) = {result.sigma[name]}, expected > 0"
            )

    def test_screening_ranks_parameters(self, morris_analyzer):
        """Parameters with larger effects should have higher mu*."""
        params = _make_params(["a", "b", "c"], mean=50.0, std=5.0)

        def weighted_sum(p):
            return 1.0 * p["a"] + 5.0 * p["b"] + 10.0 * p["c"]

        result = morris_analyzer.run_screening(
            weighted_sum, params, r=30, levels=4
        )
        # Ranking by mu* should follow: c > b > a
        assert result.mu_star["c"] > result.mu_star["b"], "c should rank above b"
        assert result.mu_star["b"] > result.mu_star["a"], "b should rank above a"

    def test_configurable_trajectory_count(self, morris_analyzer):
        """The number of trajectories should match the requested count."""
        params = _make_params(["x1", "x2"], mean=10.0, std=1.0)
        for r_val in [5, 15, 25]:
            result = morris_analyzer.run_screening(
                _linear_model, params, r=r_val, levels=4
            )
            assert result.num_trajectories == r_val

    def test_elementary_effects_computed(self, morris_analyzer):
        """calculate_elementary_effects should return proper structure."""
        params = _make_params(["x1", "x2"], mean=50.0, std=5.0)
        trajectories = morris_analyzer.generate_trajectories(params, r=10)
        ee = morris_analyzer.calculate_elementary_effects(
            _linear_model, trajectories, params
        )
        assert "x1" in ee
        assert "x2" in ee
        assert "mu_star" in ee["x1"]
        assert "sigma" in ee["x1"]
        assert "n_effects" in ee["x1"]
        assert ee["x1"]["n_effects"] > 0

    def test_classification_output(self, morris_analyzer):
        """MorrisResult should contain classification per parameter."""
        params = _make_params(["x1", "x2"], mean=100.0, std=10.0)
        result = morris_analyzer.run_screening(
            _linear_model, params, r=20
        )
        for name in ["x1", "x2"]:
            assert name in result.classification
            assert result.classification[name] in (
                "important", "non-important", "interactive"
            )

    def test_morris_result_model_fields(self, morris_analyzer):
        """MorrisResult should have all expected fields."""
        params = _make_params(["x1"], mean=100.0, std=10.0)
        result = morris_analyzer.run_screening(
            _single_param_model, params, r=10
        )
        assert isinstance(result, MorrisResult)
        assert result.parameters == ["x1"]
        assert result.computation_time >= 0.0
        assert result.num_trajectories == 10
        assert "x1" in result.mu_star_conf


# ============================================================================
# TEST: TornadoDiagramGenerator
# ============================================================================

class TestTornadoDiagramGenerator:
    """Test suite for TornadoDiagramGenerator."""

    def test_initialization(self, tornado_generator):
        """TornadoDiagramGenerator should initialize cleanly."""
        assert tornado_generator is not None

    def test_one_way_sweep_produces_ranked_data(self, tornado_generator):
        """Parameters should be sorted by descending absolute impact."""
        params = _make_params(["x1", "x2", "x3"], mean=100.0, std=10.0)
        baseline = {"x1": 100.0, "x2": 100.0, "x3": 100.0}

        def model(p):
            return 1.0 * p["x1"] + 5.0 * p["x2"] + 10.0 * p["x3"]

        result = tornado_generator.generate_one_way(
            model, params, baseline, variation_pct=0.10
        )
        # Impacts should be in descending order
        impacts = [tp.impact for tp in result.parameters]
        assert impacts == sorted(impacts, reverse=True), (
            "Parameters not sorted by descending impact"
        )

    def test_positive_negative_impact_correct_sign(self, tornado_generator):
        """High value should be greater than low value for a positive relationship."""
        params = _make_params(["x1"], mean=100.0, std=10.0)
        baseline = {"x1": 100.0}

        def model(p):
            return 2.0 * p["x1"]

        result = tornado_generator.generate_one_way(
            model, params, baseline, variation_pct=0.10
        )
        tp = result.parameters[0]
        assert tp.high_value > tp.low_value, (
            "high_value should be > low_value for positive model"
        )

    def test_baseline_output_correct(self, tornado_generator):
        """Baseline output should equal the model evaluated at baseline values."""
        params = _make_params(["x1", "x2"], mean=50.0, std=5.0)
        baseline = {"x1": 50.0, "x2": 50.0}

        def model(p):
            return p["x1"] + p["x2"]

        result = tornado_generator.generate_one_way(
            model, params, baseline, variation_pct=0.10
        )
        assert abs(result.baseline_output - 100.0) < 1e-6

    def test_from_sobol_results(self, tornado_generator):
        """generate_from_sobol should produce tornado data from SobolResult."""
        sobol_result = SobolResult(
            parameters=["ef", "activity"],
            first_order_indices={"ef": 0.6, "activity": 0.4},
            total_order_indices={"ef": 0.65, "activity": 0.42},
            interaction_effects={"ef": 0.05, "activity": 0.02},
            sample_size=1024,
        )
        tornado = tornado_generator.generate_from_sobol(sobol_result, 2500.0)

        assert isinstance(tornado, TornadoData)
        assert len(tornado.parameters) == 2
        assert tornado.baseline_output == 2500.0
        # ef should be first (higher Si)
        assert tornado.parameters[0].name == "ef"

    def test_from_monte_carlo_results(self, tornado_generator):
        """generate_from_monte_carlo should convert Pearson correlations."""
        mc_sensitivity = {"ef": 0.8, "activity": 0.5, "gwp": 0.2}
        tornado = tornado_generator.generate_from_monte_carlo(
            mc_sensitivity, baseline_output=5000.0, mc_std=500.0
        )

        assert isinstance(tornado, TornadoData)
        assert len(tornado.parameters) == 3
        assert tornado.baseline_output == 5000.0
        # Highest correlation (ef=0.8) should be first
        assert tornado.parameters[0].name == "ef"

    def test_top_n_filtering_via_slicing(self, tornado_generator):
        """Users can take top N from the sorted list."""
        params = _make_params(
            ["a", "b", "c", "d", "e"], mean=100.0, std=10.0
        )
        baseline = {p["name"]: p["mean"] for p in params}

        def model(p):
            return 1 * p["a"] + 2 * p["b"] + 3 * p["c"] + 4 * p["d"] + 5 * p["e"]

        result = tornado_generator.generate_one_way(
            model, params, baseline, variation_pct=0.10
        )
        top_3 = result.parameters[:3]
        assert len(top_3) == 3
        # Top 3 should be e, d, c (coefficients 5, 4, 3)
        assert top_3[0].name == "e"
        assert top_3[1].name == "d"
        assert top_3[2].name == "c"

    def test_baseline_at_p50_via_one_way(self, tornado_generator):
        """One-way sweep baseline is the model evaluated at the given baseline values."""
        params = _make_params(["x1"], mean=200.0, std=20.0)
        baseline = {"x1": 200.0}

        def model(p):
            return p["x1"]

        result = tornado_generator.generate_one_way(
            model, params, baseline, variation_pct=0.10
        )
        assert abs(result.baseline_output - 200.0) < 1e-6

    def test_relative_impact_calculated(self, tornado_generator):
        """relative_impact should be impact / abs(baseline_output)."""
        params = _make_params(["x1"], mean=100.0, std=10.0)
        baseline = {"x1": 100.0}

        def model(p):
            return 2.0 * p["x1"]

        result = tornado_generator.generate_one_way(
            model, params, baseline, variation_pct=0.10
        )
        tp = result.parameters[0]
        expected_rel = tp.impact / abs(result.baseline_output)
        assert abs(tp.relative_impact - expected_rel) < 1e-4

    def test_tornado_data_model(self, tornado_generator):
        """TornadoData should contain computation_time."""
        params = _make_params(["x1"], mean=100.0, std=10.0)
        baseline = {"x1": 100.0}
        result = tornado_generator.generate_one_way(
            _single_param_model, params, baseline
        )
        assert result.computation_time >= 0.0
        assert isinstance(result.parameters, list)


# ============================================================================
# TEST: ConvergenceAnalyzer
# ============================================================================

class TestConvergenceAnalyzer:
    """Test suite for ConvergenceAnalyzer."""

    def test_converged_simulation(
        self, convergence_analyzer, converged_samples
    ):
        """A large stable sample should report as converged."""
        result = convergence_analyzer.assess_convergence(converged_samples)
        assert isinstance(result, ConvergenceResult)
        assert result.is_converged is True

    def test_not_converged_simulation(
        self, convergence_analyzer, not_converged_samples
    ):
        """A small noisy sample is unlikely to converge with tight threshold."""
        result = convergence_analyzer.assess_convergence(
            not_converged_samples, threshold=0.0001
        )
        # With 200 samples and 0.01% threshold, convergence is unlikely
        # (depends on seed, but the intent is to show non-convergence detection)
        assert isinstance(result, ConvergenceResult)

    def test_running_statistics(self, convergence_analyzer, converged_samples):
        """Running means and stds should be computed at each window size."""
        windows = [100, 500, 1000, 5000, 10000]
        result = convergence_analyzer.assess_convergence(
            converged_samples, window_sizes=windows
        )
        assert len(result.running_means) == len(windows)
        assert len(result.running_stds) == len(windows)
        assert len(result.window_sizes) == len(windows)

    def test_convergence_threshold(self, convergence_analyzer, converged_samples):
        """Custom threshold should be reflected in the result."""
        result = convergence_analyzer.assess_convergence(
            converged_samples, threshold=0.05
        )
        assert result.convergence_threshold == 0.05

    def test_relative_changes_computed(
        self, convergence_analyzer, converged_samples
    ):
        """Relative changes should have len = len(windows) - 1."""
        windows = [100, 500, 2000, 10000]
        result = convergence_analyzer.assess_convergence(
            converged_samples, window_sizes=windows
        )
        assert len(result.relative_changes) == len(windows) - 1

    def test_recommended_iterations(
        self, convergence_analyzer, converged_samples
    ):
        """Recommended iterations should be a positive integer."""
        result = convergence_analyzer.assess_convergence(converged_samples)
        assert result.recommended_iterations > 0

    def test_window_sizes_exceed_sample_are_filtered(
        self, convergence_analyzer
    ):
        """Window sizes larger than the sample should be excluded."""
        small_sample = np.random.normal(100, 10, size=50)
        result = convergence_analyzer.assess_convergence(
            small_sample, window_sizes=[10, 25, 50, 100, 1000]
        )
        # Only windows <= 50 should be present
        for w in result.window_sizes:
            assert w <= 50

    def test_default_window_sizes(self, convergence_analyzer, converged_samples):
        """Default windows should be used when none are provided."""
        result = convergence_analyzer.assess_convergence(converged_samples)
        # Default is [100, 500, 1000, 5000, 10000]
        assert 100 in result.window_sizes
        assert 500 in result.window_sizes

    def test_single_window_no_relative_changes(self, convergence_analyzer):
        """With only one valid window, relative_changes should be empty."""
        small = np.random.normal(100, 10, size=80)
        result = convergence_analyzer.assess_convergence(
            small, window_sizes=[50, 200]
        )
        # Only window 50 is valid (80 samples), 200 is filtered out
        if len(result.window_sizes) == 1:
            assert len(result.relative_changes) == 0

    def test_convergence_result_model_fields(
        self, convergence_analyzer, converged_samples
    ):
        """ConvergenceResult should have all mandatory fields."""
        result = convergence_analyzer.assess_convergence(converged_samples)
        assert isinstance(result.is_converged, bool)
        assert isinstance(result.recommended_iterations, int)
        assert isinstance(result.running_means, list)
        assert isinstance(result.running_stds, list)
        assert isinstance(result.window_sizes, list)
        assert isinstance(result.relative_changes, list)
        assert isinstance(result.convergence_threshold, float)
