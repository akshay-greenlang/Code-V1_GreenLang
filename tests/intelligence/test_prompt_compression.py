"""
Tests for Prompt Compression

Tests:
- Whitespace normalization
- Abbreviation application
- Token counting
- Compression ratio
- Semantic preservation
"""

import pytest

from greenlang.intelligence.prompt_compression import (
    PromptCompressor,
    TokenCounter,
    get_compression_metrics,
)


class TestTokenCounter:
    """Test token counting"""

    @pytest.fixture
    def counter(self):
        """Create token counter"""
        return TokenCounter(model="gpt-4")

    def test_count_tokens(self, counter):
        """Test token counting"""
        text = "What is the carbon footprint of natural gas?"
        tokens = counter.count_tokens(text)

        assert tokens > 0
        assert tokens < 100  # Should be reasonable

    def test_count_messages_tokens(self, counter):
        """Test counting tokens in messages"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is carbon footprint?"},
        ]

        tokens = counter.count_messages_tokens(messages)

        assert tokens > 0


class TestPromptCompressor:
    """Test prompt compressor"""

    @pytest.fixture
    def compressor(self):
        """Create compressor"""
        return PromptCompressor(compression_threshold=0)

    def test_init(self, compressor):
        """Test initialization"""
        assert compressor.compression_threshold == 0
        assert compressor.aggressive_mode is False

    def test_whitespace_normalization(self, compressor):
        """Test whitespace normalization"""
        prompt = "What  is   the\n\n\ncarbon   footprint?"

        result = compressor.compress(prompt, force=True)

        assert result.compressed_prompt.count("  ") == 0
        assert result.compressed_prompt.count("\n\n\n") == 0

    def test_abbreviation(self, compressor):
        """Test abbreviation application"""
        prompt = "What is the carbon dioxide emissions from natural gas?"

        # Enable abbreviations
        compressor.aggressive_mode = True
        result = compressor.compress(prompt, force=True, preserve_quality=False)

        assert "CO2" in result.compressed_prompt or "carbon dioxide" in result.compressed_prompt

    def test_token_reduction(self, compressor):
        """Test token reduction"""
        prompt = """
        Please can you tell me what is the carbon dioxide emissions from natural gas
        heating systems in residential buildings? I would like to know this as accurately
        as possible. Also, could you provide information about how this compares to other
        heating options? Thank you very much.
        """

        result = compressor.compress(prompt, force=True)

        assert result.token_reduction > 0
        assert result.compressed_tokens < result.original_tokens

    def test_compression_threshold(self):
        """Test compression threshold"""
        compressor = PromptCompressor(compression_threshold=1000)

        short_prompt = "What is carbon footprint?"

        # Should not compress (below threshold)
        result = compressor.compress(short_prompt, force=False)

        assert result.compression_ratio == 1.0

    def test_preserve_quality(self, compressor):
        """Test quality preservation"""
        prompt = "What is the carbon footprint of natural gas heating?"

        result1 = compressor.compress(prompt, force=True, preserve_quality=True)
        result2 = compressor.compress(prompt, force=True, preserve_quality=False)

        # Preserving quality should result in less compression
        assert result1.compression_ratio >= result2.compression_ratio

    def test_compress_messages(self, compressor):
        """Test message compression"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant for climate calculations. Please provide accurate information."},
            {"role": "user", "content": "What is the carbon dioxide emissions from natural gas?"},
        ]

        compressed_messages, result = compressor.compress_messages(messages, force=True)

        assert len(compressed_messages) == len(messages)
        assert result.token_reduction > 0

    def test_optimize_system_prompt(self, compressor):
        """Test system prompt optimization"""
        system_prompt = """
        You are a helpful assistant specializing in climate change and carbon emissions.
        Please provide accurate and detailed information about greenhouse gas emissions.
        Try to be as precise as possible. Thank you.
        """

        optimized = compressor.optimize_system_prompt(system_prompt)

        assert len(optimized) < len(system_prompt)

    def test_cost_savings_calculation(self, compressor):
        """Test cost savings calculation"""
        prompt = "a" * 1000  # Long prompt

        result = compressor.compress(prompt, force=True)

        assert result.cost_savings_usd > 0

    def test_metrics_tracking(self):
        """Test global metrics tracking"""
        metrics = get_compression_metrics()

        # Reset for test
        metrics.total_requests = 0
        metrics.total_original_tokens = 0
        metrics.total_compressed_tokens = 0

        compressor = PromptCompressor()

        # Compress some prompts
        for i in range(3):
            result = compressor.compress(f"Test prompt {i}" * 10, force=True)
            metrics.update(result)

        assert metrics.total_requests == 3
        assert metrics.total_original_tokens > 0
        assert metrics.avg_compression_ratio < 1.0


class TestCompressionQuality:
    """Test compression preserves semantic meaning"""

    @pytest.fixture
    def compressor(self):
        """Create compressor"""
        return PromptCompressor()

    def test_semantic_preservation(self, compressor):
        """Test semantic meaning is preserved"""
        test_cases = [
            ("What is the carbon footprint?", ["carbon", "footprint"]),
            ("Calculate electricity emissions", ["electricity", "emissions"]),
            ("Recommend solar panel options", ["solar", "panel", "options"]),
        ]

        for prompt, keywords in test_cases:
            result = compressor.compress(prompt, force=True)

            # Check keywords are preserved
            compressed_lower = result.compressed_prompt.lower()
            for keyword in keywords:
                assert keyword in compressed_lower or keyword[:3] in compressed_lower


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v", "-s"])
