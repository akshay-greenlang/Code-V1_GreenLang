"""
Tests for Streaming LLM Responses

Tests:
- SSE token streaming
- Metrics tracking
- Error handling
- Demo provider
"""

import asyncio

import pytest

from greenlang.intelligence.streaming import (
    DemoStreamingProvider,
    StreamBuffer,
    StreamingProvider,
    StreamMetrics,
    StreamToken,
    stream_chat_completion,
)


class TestStreamToken:
    """Test stream token"""

    def test_init(self):
        """Test initialization"""
        token = StreamToken(token="Hello", delta=50.0, index=0)

        assert token.token == "Hello"
        assert token.delta == 50.0
        assert token.index == 0

    def test_to_sse_event(self):
        """Test SSE event formatting"""
        token = StreamToken(token="Hello", delta=50.0, index=0)

        sse = token.to_sse_event()

        assert "data:" in sse
        assert "Hello" in sse
        assert "\n\n" in sse  # SSE event terminator

    def test_finish_reason(self):
        """Test finish reason"""
        token = StreamToken(
            token="",
            delta=0,
            index=10,
            finish_reason="stop",
        )

        assert token.finish_reason == "stop"


class TestStreamMetrics:
    """Test stream metrics"""

    def test_init(self):
        """Test initialization"""
        metrics = StreamMetrics()

        assert metrics.total_tokens == 0
        assert metrics.first_token_time is None

    def test_record_token(self):
        """Test recording tokens"""
        metrics = StreamMetrics()

        metrics.record_token(50.0)
        metrics.record_token(100.0)

        assert metrics.total_tokens == 2
        assert metrics.first_token_time == 50.0
        assert metrics.last_token_time == 100.0

    def test_finalize(self):
        """Test finalization"""
        metrics = StreamMetrics()

        metrics.record_token(50.0)
        metrics.record_token(100.0)
        metrics.record_token(150.0)
        metrics.finalize()

        assert metrics.total_time == 150.0
        assert metrics.avg_token_latency == 50.0


class TestDemoStreamingProvider:
    """Test demo streaming provider"""

    @pytest.fixture
    def provider(self):
        """Create demo provider"""
        return DemoStreamingProvider(tokens_per_second=100)

    @pytest.mark.asyncio
    async def test_stream_completion(self, provider):
        """Test streaming completion"""
        messages = [{"role": "user", "content": "Test"}]

        tokens = []
        async for token in provider.stream_completion(messages):
            tokens.append(token)

        assert len(tokens) > 0

        # Check first token
        assert tokens[0].index == 0
        assert tokens[0].token != ""

        # Check last token has finish reason
        assert tokens[-1].finish_reason is not None

    @pytest.mark.asyncio
    async def test_max_tokens(self, provider):
        """Test max tokens limit"""
        messages = [{"role": "user", "content": "Test"}]

        tokens = []
        async for token in provider.stream_completion(messages, max_tokens=5):
            if token.token:
                tokens.append(token)

        assert len(tokens) <= 5

    @pytest.mark.asyncio
    async def test_streaming_latency(self, provider):
        """Test streaming latency"""
        messages = [{"role": "user", "content": "Test"}]

        first_token_delta = None
        async for token in provider.stream_completion(messages):
            if token.index == 0 and token.token:
                first_token_delta = token.delta
                break

        # First token should arrive quickly
        assert first_token_delta is not None


class TestStreamBuffer:
    """Test stream buffer"""

    def test_init(self):
        """Test initialization"""
        buffer = StreamBuffer(buffer_size=5, timeout_ms=100)

        assert buffer.buffer_size == 5
        assert len(buffer.buffer) == 0

    def test_buffering(self):
        """Test token buffering"""
        buffer = StreamBuffer(buffer_size=3, timeout_ms=10000)

        # Add tokens
        result1 = buffer.add("Hello")
        assert result1 is None  # Not full yet

        result2 = buffer.add(" ")
        assert result2 is None

        result3 = buffer.add("World")
        assert result3 == "Hello World"  # Buffer flushed

    def test_flush(self):
        """Test manual flush"""
        buffer = StreamBuffer()

        buffer.add("Hello")
        buffer.add(" ")
        buffer.add("World")

        flushed = buffer.flush()

        assert flushed == "Hello World"
        assert len(buffer.buffer) == 0


class TestStreamIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_stream_chat_completion(self):
        """Test stream_chat_completion convenience function"""
        messages = [{"role": "user", "content": "Test"}]

        tokens = []
        async for token in stream_chat_completion(messages, use_demo=True, max_tokens=10):
            tokens.append(token)

        assert len(tokens) > 0
        assert any(t.finish_reason is not None for t in tokens)

    @pytest.mark.asyncio
    async def test_progressive_response(self):
        """Test progressive response building"""
        messages = [{"role": "user", "content": "Test"}]

        full_response = ""
        token_count = 0

        async for token in stream_chat_completion(messages, use_demo=True, max_tokens=20):
            if token.token:
                full_response += token.token
                token_count += 1

        assert token_count > 0
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test collecting metrics during streaming"""
        messages = [{"role": "user", "content": "Test"}]

        metrics = StreamMetrics()

        async for token in stream_chat_completion(messages, use_demo=True, max_tokens=10):
            if token.token:
                import time
                current_time = time.time()
                metrics.record_token((current_time - metrics.start_time.timestamp()) * 1000)

        metrics.finalize()

        assert metrics.total_tokens > 0
        assert metrics.first_token_time is not None

    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """Test multiple concurrent streams"""
        messages1 = [{"role": "user", "content": "Query 1"}]
        messages2 = [{"role": "user", "content": "Query 2"}]

        async def collect_tokens(messages):
            tokens = []
            async for token in stream_chat_completion(messages, use_demo=True, max_tokens=5):
                tokens.append(token)
            return tokens

        # Run concurrently
        results = await asyncio.gather(
            collect_tokens(messages1),
            collect_tokens(messages2),
        )

        assert len(results) == 2
        assert all(len(r) > 0 for r in results)


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v", "-s"])
