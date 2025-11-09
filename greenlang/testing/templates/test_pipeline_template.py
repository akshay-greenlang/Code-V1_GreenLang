"""
Pipeline Test Template
======================

Template for testing GreenLang pipelines.

Copy this template and customize for your specific pipeline.
"""

from greenlang.testing import PipelineTestCase
# Import your pipeline here
# from your_module import YourPipeline


class TestYourPipeline(PipelineTestCase):
    """Test suite for YourPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Initialize your pipeline
        # self.pipeline = YourPipeline()

        # Load test data
        self.test_input = {
            "data": "input_data",
        }

    def test_full_pipeline(self):
        """Test complete pipeline execution."""
        result = self.run_pipeline(self.pipeline, self.test_input)

        # Assert pipeline success
        self.assert_pipeline_success(result)

        # Assert all stages completed
        self.assert_all_stages_completed(result)

    def test_pipeline_performance(self):
        """Test pipeline performance."""
        result = self.run_pipeline(self.pipeline, self.test_input)

        # Assert performance
        self.assert_pipeline_performance(result, max_time=10.0)

    def test_individual_stages(self):
        """Test individual pipeline stages."""
        result = self.run_pipeline(self.pipeline, self.test_input)

        # Test stage 1
        self.assert_stage_output(
            result,
            "stage_1_name",
            {"type": "object"}
        )

        # Test stage 2
        self.assert_stage_output(
            result,
            "stage_2_name",
            {"type": "object"}
        )

    def test_pipeline_with_mocks(self):
        """Test pipeline with mocked components."""
        # Set up mocks
        self.mock_chat.add_response("Stage 1 response")
        self.mock_chat.add_response("Stage 2 response")

        with self.mock_infrastructure():
            result = self.run_pipeline(self.pipeline, self.test_input)
            self.assert_pipeline_success(result)

    def test_pipeline_error_recovery(self):
        """Test pipeline error handling and recovery."""
        # Provide input that causes an error in stage 2
        error_input = {"invalid": "data"}

        result = self.run_pipeline(self.pipeline, error_input)

        # Assert pipeline failed gracefully
        self.assertFalse(result['success'])
        self.assertIsNotNone(result.get('error'))

    def test_pipeline_with_fixtures(self):
        """Test pipeline with fixture data."""
        fixture_data = self.load_fixture('sample_emissions_data.json')

        result = self.run_pipeline(self.pipeline, fixture_data)

        self.assert_pipeline_success(result)
        self.assert_all_stages_completed(result)


if __name__ == '__main__':
    import unittest
    unittest.main()
