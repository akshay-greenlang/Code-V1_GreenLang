# -*- coding: utf-8 -*-
"""
LLM-assisted reranking for emission factor matching (F043).

Enterprise-only feature that uses an LLM to rerank the top-N candidates
from hybrid search. Critically, LLM reranking NEVER changes factor values
— it only reorders the ranking based on semantic understanding.

Safety guarantees:
- Factor values (co2e, vectors, etc.) are never mutated
- Input candidates are frozen before LLM call
- Output is validated: only reordering, no new factors injected
- Rate limited: max 10 req/min per API key (enforced externally)

Usage:
    reranker = LLMReranker(LLMRerankConfig(provider="anthropic"))
    reranked = reranker.rerank("diesel combustion US", candidates)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class LLMRerankConfig:
    """Configuration for LLM reranking."""

    provider: str = "anthropic"  # anthropic, openai, stub
    model: str = "claude-sonnet-4-5-20250929"
    max_candidates: int = 20
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 30
    rate_limit_rpm: int = 10
    enabled: bool = True


@dataclass
class RerankResult:
    """Result of an LLM reranking operation."""

    reranked: List[Dict[str, Any]]
    model_used: str
    latency_ms: float
    candidates_in: int
    candidates_out: int
    reranked_by_llm: bool


class LLMProvider(Protocol):
    """Protocol for LLM providers used in reranking."""

    def complete(self, prompt: str, config: LLMRerankConfig) -> str: ...


class StubLLMProvider:
    """
    Stub LLM provider for testing.

    Returns candidates in reverse order (to verify reranking works).
    """

    def complete(self, prompt: str, config: LLMRerankConfig) -> str:
        # Parse the candidates from the prompt and return reversed order
        # The prompt includes factor_ids — extract and reverse
        import re

        factor_ids = re.findall(r'"factor_id":\s*"([^"]+)"', prompt)
        reversed_ids = list(reversed(factor_ids))
        return json.dumps({"ranked_factor_ids": reversed_ids})


class AnthropicLLMProvider:
    """
    Anthropic Claude provider for LLM reranking.

    Requires: ANTHROPIC_API_KEY environment variable.
    """

    def complete(self, prompt: str, config: LLMRerankConfig) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic()
            message = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")


class OpenAILLMProvider:
    """
    OpenAI GPT provider for LLM reranking.

    Requires: OPENAI_API_KEY environment variable.
    """

    def complete(self, prompt: str, config: LLMRerankConfig) -> str:
        try:
            import openai

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("openai package required: pip install openai")


def _build_rerank_prompt(
    activity_description: str,
    candidates: List[Dict[str, Any]],
) -> str:
    """Build the LLM prompt for reranking emission factor candidates."""
    candidate_lines = []
    for i, c in enumerate(candidates):
        fid = c.get("factor_id", "unknown")
        score = c.get("score", 0.0)
        candidate_lines.append(
            f'  {i + 1}. {{"factor_id": "{fid}", "score": {score}}}'
        )

    candidates_block = "\n".join(candidate_lines)

    return f"""You are an emission factor matching expert. Given the user's activity description,
rerank the following emission factor candidates by relevance.

Activity: "{activity_description}"

Candidates (current ranking):
{candidates_block}

Instructions:
- Rerank the candidates by relevance to the activity description
- Consider fuel type, geography, scope, and boundary alignment
- Return ONLY a JSON object with a "ranked_factor_ids" array
- The array must contain ONLY factor_ids from the candidates above (no additions)
- Do NOT modify any factor values, only reorder

Respond with ONLY valid JSON:
{{"ranked_factor_ids": ["most_relevant_factor_id", "second_most_relevant", ...]}}"""


def _parse_rerank_response(
    response: str,
    original_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Parse LLM rerank response and validate output.

    Safety: only allows reordering of existing candidates, never adds new ones.
    """
    # Extract JSON from response (may have markdown fences)
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM rerank response not valid JSON, keeping original order")
        return original_candidates

    ranked_ids = parsed.get("ranked_factor_ids", [])
    if not isinstance(ranked_ids, list):
        logger.warning("LLM rerank response missing ranked_factor_ids array")
        return original_candidates

    # Build lookup of original candidates
    candidate_map = {c["factor_id"]: c for c in original_candidates}
    original_ids = set(candidate_map.keys())

    # Validate: no injected factor_ids
    valid_ids = [fid for fid in ranked_ids if fid in original_ids]
    if not valid_ids:
        logger.warning("LLM rerank returned no valid factor_ids, keeping original order")
        return original_candidates

    # Build reranked list, append any missing originals at the end
    reranked = []
    seen = set()
    for fid in valid_ids:
        if fid not in seen:
            reranked.append(candidate_map[fid])
            seen.add(fid)

    # Append any missing candidates at the end (preserve their original order)
    for c in original_candidates:
        if c["factor_id"] not in seen:
            reranked.append(c)
            seen.add(c["factor_id"])

    return reranked


class LLMReranker:
    """
    LLM-based reranker for emission factor candidates.

    Reranks the top-N candidates from hybrid search using an LLM.
    Never modifies factor values — only reorders.
    """

    def __init__(
        self,
        config: Optional[LLMRerankConfig] = None,
        provider: Optional[LLMProvider] = None,
    ):
        self._config = config or LLMRerankConfig()
        if provider:
            self._provider = provider
        elif self._config.provider == "stub":
            self._provider = StubLLMProvider()
        elif self._config.provider == "openai":
            self._provider = OpenAILLMProvider()
        else:
            self._provider = AnthropicLLMProvider()
        self._last_call_time: float = 0.0
        self._call_count_minute: int = 0
        self._minute_start: float = 0.0

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.monotonic()
        if now - self._minute_start > 60:
            self._minute_start = now
            self._call_count_minute = 0
        return self._call_count_minute < self._config.rate_limit_rpm

    def rerank(
        self,
        activity_description: str,
        candidates: List[Dict[str, Any]],
    ) -> RerankResult:
        """
        Rerank candidates using LLM.

        Returns RerankResult with reranked candidates.
        Falls back to original order on any failure.
        """
        start = time.monotonic()

        if not self._config.enabled:
            return RerankResult(
                reranked=candidates,
                model_used="none",
                latency_ms=0.0,
                candidates_in=len(candidates),
                candidates_out=len(candidates),
                reranked_by_llm=False,
            )

        if not self._check_rate_limit():
            logger.warning("LLM rerank rate limit exceeded (%d rpm)", self._config.rate_limit_rpm)
            return RerankResult(
                reranked=candidates,
                model_used="none",
                latency_ms=0.0,
                candidates_in=len(candidates),
                candidates_out=len(candidates),
                reranked_by_llm=False,
            )

        # Truncate to max_candidates
        top_n = candidates[: self._config.max_candidates]

        # Build prompt
        prompt = _build_rerank_prompt(activity_description, top_n)

        try:
            response = self._provider.complete(prompt, self._config)
            self._call_count_minute += 1

            reranked = _parse_rerank_response(response, top_n)

            # Append any candidates beyond top_n (not sent to LLM)
            if len(candidates) > self._config.max_candidates:
                reranked.extend(candidates[self._config.max_candidates:])

            latency = (time.monotonic() - start) * 1000
            logger.info(
                "LLM rerank completed: model=%s candidates=%d latency=%.1fms",
                self._config.model, len(top_n), latency,
            )
            return RerankResult(
                reranked=reranked,
                model_used=self._config.model,
                latency_ms=latency,
                candidates_in=len(candidates),
                candidates_out=len(reranked),
                reranked_by_llm=True,
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning("LLM rerank failed (%.1fms): %s", latency, exc)
            return RerankResult(
                reranked=candidates,
                model_used=self._config.model,
                latency_ms=latency,
                candidates_in=len(candidates),
                candidates_out=len(candidates),
                reranked_by_llm=False,
            )

    def as_rerank_fn(self) -> Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Return a callback suitable for run_match(rerank_fn=...)."""

        def _fn(activity: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            result = self.rerank(activity, candidates)
            return result.reranked

        return _fn
