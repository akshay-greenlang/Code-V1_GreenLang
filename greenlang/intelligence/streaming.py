# -*- coding: utf-8 -*-
"""
Streaming LLM Responses with Server-Sent Events (SSE)

Implements streaming for better UX and perceived latency:
- Server-Sent Events (SSE) for real-time token streaming
- Progressive UI updates as tokens arrive
- Reduced perceived latency (first token < 500ms)
- Better experience for long responses

Architecture:
    Client -> SSE Request -> Stream Tokens -> Progressive Render
                |
           LLM API (stream=True)
                |
           Token by Token
                |
           Buffer & Yield

Benefits:
- First token < 500ms (vs waiting for full response)
- Progressive rendering improves UX
- Users see progress immediately
- Better for long responses (>500 tokens)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")


logger = logging.getLogger(__name__)


@dataclass
class StreamToken:
    """
    Single token in a stream

    Attributes:
        token: Token text
        delta: Time since last token (ms)
        index: Token index in stream
        finish_reason: Reason for stream completion (if last token)
        metadata: Additional metadata
    """
    token: str
    delta: float
    index: int
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sse_event(self) -> str:
        """
        Convert to SSE event format

        Returns:
            SSE event string
        """
        data = {
            "token": self.token,
            "delta": self.delta,
            "index": self.index,
        }

        if self.finish_reason:
            data["finish_reason"] = self.finish_reason

        if self.metadata:
            data["metadata"] = self.metadata

        return f"data: {json.dumps(data)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "token": self.token,
            "delta": self.delta,
            "index": self.index,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


@dataclass
class StreamMetrics:
    """
    Metrics for streaming performance

    Attributes:
        start_time: Stream start timestamp
        first_token_time: Time to first token (ms)
        last_token_time: Time to last token (ms)
        total_tokens: Total tokens streamed
        avg_token_latency: Average latency per token (ms)
        total_time: Total streaming time (ms)
    """
    start_time: datetime = field(default_factory=datetime.now)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    total_tokens: int = 0
    avg_token_latency: float = 0.0
    total_time: float = 0.0

    def record_token(self, token_time: float):
        """Record token timing"""
        if self.first_token_time is None:
            self.first_token_time = token_time

        self.last_token_time = token_time
        self.total_tokens += 1

    def finalize(self):
        """Calculate final metrics"""
        if self.first_token_time and self.last_token_time:
            self.total_time = self.last_token_time
            if self.total_tokens > 0:
                self.avg_token_latency = self.total_time / self.total_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "first_token_time_ms": self.first_token_time,
            "total_time_ms": self.total_time,
            "total_tokens": self.total_tokens,
            "avg_token_latency_ms": self.avg_token_latency,
        }


class StreamingProvider:
    """
    Base class for streaming LLM providers

    Handles:
    - Token streaming
    - SSE formatting
    - Error handling
    - Metrics tracking
    """

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamToken]:
        """
        Stream LLM completion

        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature (0-1)
            max_tokens: Maximum tokens to generate

        Yields:
            Stream tokens
        """
        raise NotImplementedError("Subclasses must implement stream_completion")

    async def stream_to_sse(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream completion as SSE events

        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature
            max_tokens: Maximum tokens

        Yields:
            SSE event strings
        """
        async for token in self.stream_completion(messages, model, temperature, max_tokens):
            yield token.to_sse_event()


class OpenAIStreamingProvider(StreamingProvider):
    """
    Streaming provider for OpenAI

    Uses OpenAI's streaming API to get tokens as they're generated.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI streaming provider

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")

        self.client = AsyncOpenAI(api_key=api_key)

        logger.info("OpenAI streaming provider initialized")

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamToken]:
        """
        Stream OpenAI completion

        Args:
            messages: Chat messages
            model: Model name
            temperature: Temperature
            max_tokens: Maximum tokens

        Yields:
            Stream tokens
        """
        metrics = StreamMetrics()
        start_time = time.time()
        index = 0
        last_time = start_time

        try:
            # Create streaming completion
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            # Process stream
            async for chunk in stream:
                current_time = time.time()
                delta = (current_time - last_time) * 1000  # Convert to ms

                # Extract token
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    token_text = choice.delta.content or ""

                    if token_text:
                        # Create stream token
                        token = StreamToken(
                            token=token_text,
                            delta=delta,
                            index=index,
                        )

                        # Record metrics
                        metrics.record_token((current_time - start_time) * 1000)

                        yield token

                        index += 1
                        last_time = current_time

                    # Check for finish
                    if choice.finish_reason:
                        # Send final token with finish reason
                        final_token = StreamToken(
                            token="",
                            delta=delta,
                            index=index,
                            finish_reason=choice.finish_reason,
                            metadata=metrics.to_dict(),
                        )
                        yield final_token

            # Finalize metrics
            metrics.finalize()
            logger.info(f"Stream complete: {metrics.total_tokens} tokens, first token: {metrics.first_token_time:.0f}ms")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Yield error token
            error_token = StreamToken(
                token="",
                delta=0,
                index=index,
                finish_reason="error",
                metadata={"error": str(e)},
            )
            yield error_token


class DemoStreamingProvider(StreamingProvider):
    """
    Demo streaming provider for testing without API keys

    Simulates streaming by yielding tokens with artificial delays.
    """

    def __init__(self, tokens_per_second: int = 20):
        """
        Initialize demo streaming provider

        Args:
            tokens_per_second: Simulated token generation rate
        """
        self.tokens_per_second = tokens_per_second
        self.delay = 1.0 / tokens_per_second

        logger.info(f"Demo streaming provider initialized ({tokens_per_second} tokens/s)")

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "demo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamToken]:
        """
        Stream demo completion

        Args:
            messages: Chat messages
            model: Model name (ignored)
            temperature: Temperature (ignored)
            max_tokens: Maximum tokens

        Yields:
            Stream tokens
        """
        # Generate demo response
        demo_response = (
            "Natural gas has a carbon footprint of approximately 0.185 kg CO2 per kWh. "
            "This includes both direct combustion emissions and upstream methane leakage from extraction and transport. "
            "For a typical residential heating system consuming 10,000 kWh per year, this translates to about 1.85 tonnes of CO2 emissions annually. "
            "By comparison, heat pumps can reduce emissions by 60-80% depending on grid carbon intensity."
        )

        # Split into tokens (simple word splitting)
        tokens = demo_response.split()

        if max_tokens:
            tokens = tokens[:max_tokens]

        start_time = time.time()
        last_time = start_time

        for i, token in enumerate(tokens):
            # Simulate network delay
            await asyncio.sleep(self.delay)

            current_time = time.time()
            delta = (current_time - last_time) * 1000

            # Yield token
            stream_token = StreamToken(
                token=token + " ",
                delta=delta,
                index=i,
            )
            yield stream_token

            last_time = current_time

        # Send finish token
        final_token = StreamToken(
            token="",
            delta=0,
            index=len(tokens),
            finish_reason="stop",
        )
        yield final_token


class StreamBuffer:
    """
    Buffer for streaming tokens

    Collects tokens and yields when buffer is full or timeout occurs.
    Useful for reducing SSE overhead by batching small tokens.
    """

    def __init__(
        self,
        buffer_size: int = 5,
        timeout_ms: float = 100,
    ):
        """
        Initialize stream buffer

        Args:
            buffer_size: Number of tokens to buffer
            timeout_ms: Timeout to flush buffer (ms)
        """
        self.buffer_size = buffer_size
        self.timeout_ms = timeout_ms / 1000  # Convert to seconds

        self.buffer: List[str] = []
        self.last_flush = time.time()

    def add(self, token: str) -> Optional[str]:
        """
        Add token to buffer

        Args:
            token: Token text

        Returns:
            Flushed buffer text if buffer is full, None otherwise
        """
        self.buffer.append(token)

        # Check buffer size
        if len(self.buffer) >= self.buffer_size:
            return self.flush()

        # Check timeout
        if time.time() - self.last_flush > self.timeout_ms:
            return self.flush()

        return None

    def flush(self) -> str:
        """
        Flush buffer

        Returns:
            Buffered text
        """
        text = "".join(self.buffer)
        self.buffer = []
        self.last_flush = time.time()
        return text


async def stream_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    use_demo: bool = False,
) -> AsyncIterator[StreamToken]:
    """
    Stream chat completion

    Convenience function for streaming completions.

    Args:
        messages: Chat messages
        model: Model name
        temperature: Temperature
        max_tokens: Maximum tokens
        api_key: API key (OpenAI)
        use_demo: Use demo provider

    Yields:
        Stream tokens
    """
    if use_demo:
        provider = DemoStreamingProvider()
    else:
        provider = OpenAIStreamingProvider(api_key=api_key)

    async for token in provider.stream_completion(messages, model, temperature, max_tokens):
        yield token


async def stream_to_sse(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    use_demo: bool = False,
) -> AsyncIterator[str]:
    """
    Stream chat completion as SSE events

    Convenience function for SSE streaming.

    Args:
        messages: Chat messages
        model: Model name
        temperature: Temperature
        max_tokens: Maximum tokens
        api_key: API key (OpenAI)
        use_demo: Use demo provider

    Yields:
        SSE event strings
    """
    if use_demo:
        provider = DemoStreamingProvider()
    else:
        provider = OpenAIStreamingProvider(api_key=api_key)

    async for event in provider.stream_to_sse(messages, model, temperature, max_tokens):
        yield event


# Example FastAPI endpoint (for reference)
FASTAPI_EXAMPLE = """
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from greenlang.intelligence.streaming import stream_to_sse

app = FastAPI()

@app.get("/api/agents/{agent_id}/stream")
async def stream_agent_response(agent_id: str, query: str):
    '''
    Stream agent response using SSE

    Args:
        agent_id: Agent ID
        query: User query

    Returns:
        SSE stream
    '''
    messages = [
        {"role": "system", "content": f"You are agent {agent_id}"},
        {"role": "user", "content": query},
    ]

    return StreamingResponse(
        stream_to_sse(messages, use_demo=True),
        media_type="text/event-stream",
    )
"""

# Example client-side JavaScript
JAVASCRIPT_EXAMPLE = """
// Client-side JavaScript for consuming SSE stream

function streamAgentResponse(agentId, query) {
    const url = `/api/agents/${agentId}/stream?query=${encodeURIComponent(query)}`;
    const eventSource = new EventSource(url);

    const responseDiv = document.getElementById('response');
    let fullResponse = '';

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.token) {
            // Append token to response
            fullResponse += data.token;
            responseDiv.textContent = fullResponse;
        }

        if (data.finish_reason) {
            // Stream complete
            console.log('Stream complete:', data.metadata);
            eventSource.close();
        }
    };

    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        eventSource.close();
    };
}
"""


if __name__ == "__main__":
    """
    Demo and testing
    """
    import asyncio

    print("=" * 80)
    print("GreenLang Streaming Demo")
    print("=" * 80)

    async def demo():
        # Test messages
        messages = [
            {"role": "system", "content": "You are a climate and carbon expert."},
            {"role": "user", "content": "What is the carbon footprint of natural gas?"},
        ]

        print("\n1. Demo streaming (simulated):")
        print("   Query: What is the carbon footprint of natural gas?")
        print("   Response: ", end="", flush=True)

        # Stream demo response
        full_response = ""
        first_token_time = None
        start_time = time.time()

        async for token in stream_chat_completion(messages, use_demo=True, max_tokens=30):
            if token.token:
                print(token.token, end="", flush=True)
                full_response += token.token

                if first_token_time is None:
                    first_token_time = (time.time() - start_time) * 1000

            if token.finish_reason:
                total_time = (time.time() - start_time) * 1000
                print(f"\n\n   Metrics:")
                print(f"   - First token: {first_token_time:.0f}ms")
                print(f"   - Total time: {total_time:.0f}ms")
                print(f"   - Total tokens: {token.index}")
                print(f"   - Finish reason: {token.finish_reason}")

        print("\n2. SSE Event Format:")
        print("   Example SSE event:")
        token = StreamToken(token="Hello", delta=50.5, index=0)
        print(f"   {token.to_sse_event().strip()}")

    # Run demo
    asyncio.run(demo())

    print("\n" + "=" * 80)
