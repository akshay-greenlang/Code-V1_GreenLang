"""GL-013 Explainability Tests - Author: GL-TestEngineer"""
import pytest
import numpy as np

class TestSHAPValueConsistency:
    def test_shap_values_sum_to_prediction(self, sample_shap_values):
        base_value = sample_shap_values["base_value"]
        shap_values = sample_shap_values["shap_values"]
        prediction = sample_shap_values["prediction"]
        computed = base_value + sum(shap_values)
        assert abs(computed - prediction) < 1e-6

    def test_shap_feature_count_matches(self, sample_shap_values):
        feature_names = sample_shap_values["feature_names"]
        shap_values = sample_shap_values["shap_values"]
        feature_values = sample_shap_values["feature_values"]
        assert len(feature_names) == len(shap_values) == len(feature_values)

    def test_shap_values_bounded(self, sample_shap_values):
        shap_values = sample_shap_values["shap_values"]
        base_value = sample_shap_values["base_value"]
        for sv in shap_values:
            assert abs(sv) < abs(base_value)

class TestLIMEExplanationStability:
    def test_lime_local_model_score(self, sample_lime_explanation):
        score = sample_lime_explanation["local_model_score"]
        assert 0 <= score <= 1

    def test_lime_feature_weights_present(self, sample_lime_explanation):
        weights = sample_lime_explanation["feature_weights"]
        assert len(weights) > 0

    def test_lime_weights_ordered_by_importance(self, sample_lime_explanation):
        weights = sample_lime_explanation["feature_weights"]
        abs_weights = [abs(w[1]) for w in weights]
        assert abs_weights == sorted(abs_weights, reverse=True)

class TestCausalGraphValidation:
    def test_causal_graph_has_nodes(self, sample_causal_graph):
        nodes = sample_causal_graph["nodes"]
        assert len(nodes) > 0

    def test_causal_graph_has_edges(self, sample_causal_graph):
        edges = sample_causal_graph["edges"]
        assert len(edges) > 0

    def test_causal_edges_reference_valid_nodes(self, sample_causal_graph):
        nodes = sample_causal_graph["nodes"]
        edges = sample_causal_graph["edges"]
        node_ids = {n["id"] for n in nodes}
        for edge in edges:
            assert edge["source"] in node_ids
            assert edge["target"] in node_ids

    def test_causal_edge_weights_bounded(self, sample_causal_graph):
        edges = sample_causal_graph["edges"]
        for edge in edges:
            assert -1 <= edge["weight"] <= 1

    def test_no_self_loops(self, sample_causal_graph):
        edges = sample_causal_graph["edges"]
        for edge in edges:
            assert edge["source"] != edge["target"]

class TestExplainabilityProvenance:
    def test_explanation_has_prediction_id(self, sample_lime_explanation):
        assert "prediction_id" in sample_lime_explanation
        assert len(sample_lime_explanation["prediction_id"]) > 0

class TestFeatureImportance:
    def test_most_important_feature_identified(self, sample_shap_values):
        shap_values = sample_shap_values["shap_values"]
        feature_names = sample_shap_values["feature_names"]
        abs_values = [abs(v) for v in shap_values]
        max_idx = np.argmax(abs_values)
        most_important = feature_names[max_idx]
        assert most_important is not None
