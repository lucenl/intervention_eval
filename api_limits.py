"""Request rate limiting and token usage tracking for OASIS simulations."""

import asyncio
import time
from dataclasses import dataclass

import oasis


class RateLimiter:
    """Async context manager that caps concurrent requests and requests per minute.

    Requests are spaced evenly: at most one request starts every
    60 / requests_per_minute seconds, and at most max_concurrent run at once.
    """

    def __init__(self, max_concurrent: int, requests_per_minute: int):
        if max_concurrent < 1 or requests_per_minute < 1:
            raise ValueError("max_concurrent and requests_per_minute must be >= 1")
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._interval = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            wait = self._next_start - now
            self._next_start = max(now, self._next_start) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()
        return False


@dataclass
class TokenUsage:
    """Running totals of tokens used by the model."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def cost(self, price_per_1m_input: float, price_per_1m_output: float) -> float:
        return (self.prompt_tokens * price_per_1m_input
                + self.completion_tokens * price_per_1m_output) / 1e6


def track_usage(model) -> TokenUsage:
    """Wrap the shared model backend so every response adds to a TokenUsage."""
    usage = TokenUsage()
    original_arun = model.arun

    async def arun(*args, **kwargs):
        response = await original_arun(*args, **kwargs)
        if getattr(response, "usage", None) is not None:
            usage.prompt_tokens += response.usage.prompt_tokens or 0
            usage.completion_tokens += response.usage.completion_tokens or 0
        return response

    model.arun = arun
    return usage


def make_env(agent_graph, database_path: str, max_concurrent: int,
             requests_per_minute: int):
    """Create a Twitter OASIS environment whose LLM calls go through a RateLimiter."""
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=database_path,
    )
    env.llm_semaphore = RateLimiter(max_concurrent, requests_per_minute)
    return env
