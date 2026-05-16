"""API manager with a Token Bucket rate limiter and ChatAgent usage tracking."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
import threading
from functools import wraps

# ==============================================================================
# Core: enhanced semaphore
# ==============================================================================


class ChatAgentEnhancedSemaphore:
    """Concurrency semaphore backed by a Token Bucket rate limiter."""

    def __init__(self, permits: int = 150, api_limits: Dict = None):
        self.base_semaphore = asyncio.Semaphore(permits)

        self.api_limits = api_limits or {"rpm": 30_000, "tpm": 150_000_000}

        self._rpm_limit = self.api_limits.get("rpm", 30_000)
        self._tpm_limit = self.api_limits.get("tpm", 150_000_000)

        self._left_requests = self._rpm_limit
        self._left_tokens = self._tpm_limit

        self._last_tick = time.monotonic()

        self._rpm_rate = self._rpm_limit / 60.0
        self._tpm_rate = self._tpm_limit / 60.0

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.consecutive_errors = 0

        self.request_history = deque(maxlen=1)
        self.current_api_limits = {}

        self.lock = asyncio.Lock()
        self.sync_lock = threading.Lock()
        self.logger = logging.getLogger("chatagent_enhanced_semaphore")

    async def __aenter__(self):
        await self.base_semaphore.acquire()
        await self.acquire(tokens=0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._record_request_result(exc_type, exc_val)
        self.base_semaphore.release()

    async def acquire(self, tokens: int = 0):
        async with self.lock:
            while True:
                now = time.monotonic()
                delta = now - self._last_tick
                self._last_tick = now

                if delta > 0:
                    self._left_requests = min(self._rpm_limit, self._left_requests + delta * self._rpm_rate)
                    self._left_tokens = min(self._tpm_limit, self._left_tokens + delta * self._tpm_rate)

                cost_req = 1
                cost_tok = tokens

                if self._left_requests >= cost_req and self._left_tokens >= cost_tok:
                    self._left_requests -= cost_req
                    self._left_tokens -= cost_tok

                    self.request_history.append({'timestamp': time.time(), 'tokens': tokens})
                    return

                wait_req = (cost_req - self._left_requests) / self._rpm_rate if self._left_requests < cost_req else 0
                wait_tok = (cost_tok - self._left_tokens) / self._tpm_rate if self._left_tokens < cost_tok else 0

                wait_time = max(wait_req, wait_tok)

                if wait_time > 0.01:
                    pass

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

    def _record_request_result(self, exc_type, exc_val):
        success = exc_type is None
        if not success and exc_val:
            error_msg = str(exc_val).lower()
            if any(keyword in error_msg for keyword in ['rate limit', '429', 'quota']):
                self.consecutive_errors += 1
                self.logger.warning(f"Rate limit detected (429): {exc_val}")
        else:
            self.consecutive_errors = 0

    def update_from_chatagent_response(self, agent_response):
        """Extract usage from a ChatAgent response and update global counters."""
        if not hasattr(agent_response, 'info'):
            return

        usage_dict = agent_response.info.get('usage_dict', {})

        p_tokens = usage_dict.get('prompt_tokens', 0)
        c_tokens = usage_dict.get('completion_tokens', 0)

        with self.sync_lock:
            self.total_prompt_tokens += p_tokens
            self.total_completion_tokens += c_tokens
            self.total_requests += 1

    def update_from_api_headers(self, headers: Dict[str, str]):
        """Update remaining quota from API response headers."""
        try:
            self.current_api_limits = {
                'rpm': int(headers.get('x-ratelimit-limit-requests', self.api_limits['rpm'])),
                'remaining': int(headers.get('x-ratelimit-remaining-requests', 0)),
            }
        except Exception:
            pass

# ==============================================================================
# Helpers: response tracker and environment wrapper
# ==============================================================================


class ChatAgentResponseTracker:
    """Monkey-patches an agent to intercept responses and extract token usage."""

    def __init__(self, semaphore: ChatAgentEnhancedSemaphore):
        self.semaphore = semaphore
        self.logger = logging.getLogger("chatagent_response_tracker")

    def enhance_agent_for_tracking(self, agent):
        if hasattr(agent, '_tracking_enhanced'):
            return

        original_get_model_response = agent._get_model_response
        original_aget_model_response = agent._aget_model_response

        def enhanced_get_model_response(*args, **kwargs):
            response = original_get_model_response(*args, **kwargs)
            self._extract_tracking_info(response)
            return response

        async def enhanced_aget_model_response(*args, **kwargs):
            response = await original_aget_model_response(*args, **kwargs)
            self._extract_tracking_info(response)
            return response

        agent._get_model_response = enhanced_get_model_response
        agent._aget_model_response = enhanced_aget_model_response
        agent._tracking_enhanced = True

    def _extract_tracking_info(self, model_response):
        try:
            if hasattr(model_response, 'usage_dict') and model_response.usage_dict:
                self.semaphore.update_from_chatagent_response(
                    type('MockResponse', (), {'info': {'usage_dict': model_response.usage_dict}})()
                )

            if hasattr(model_response, 'response'):
                raw = model_response.response
                if hasattr(raw, 'headers'):
                    self.semaphore.update_from_api_headers(dict(raw.headers))

        except Exception as e:
            self.logger.debug(f"Tracking error: {e}")


class OASISChatAgentCompatibleEnv:
    """OASIS environment wrapper that swaps in the enhanced semaphore."""

    def __init__(self, original_env, api_limits: Dict = None, semaphore_permits: int = 50):
        self.env = original_env
        self.logger = logging.getLogger("oasis_chatagent_compatible")

        self.enhanced_semaphore = ChatAgentEnhancedSemaphore(permits=semaphore_permits, api_limits=api_limits)

        self.response_tracker = ChatAgentResponseTracker(self.enhanced_semaphore)

        self.env.llm_semaphore = self.enhanced_semaphore

        self._enhance_all_agents()

    def _enhance_all_agents(self):
        try:
            for _, agent in self.env.agent_graph.get_agents():
                self.response_tracker.enhance_agent_for_tracking(agent)
            self.logger.info("Enhanced all agents with Token Bucket tracking")
        except Exception as e:
            self.logger.warning(f"Could not enhance agents: {e}")

    async def step(self, *args, **kwargs):
        return await self.env.step(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)


# ==============================================================================
# Factory
# ==============================================================================

def create_chatagent_enhanced_oasis_env(agent_graph, platform, database_path,
                                        api_tier: str = "tier5", semaphore: int = 100):
    """Create an enhanced OASIS environment with Token Bucket rate limiting."""
    import oasis

    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=database_path,
        semaphore=semaphore
    )

    tiers = {
        "tier1": {"rpm": 500, "tpm": 200_000},
        "tier2": {"rpm": 5000, "tpm": 2_000_000},
        "tier3": {"rpm": 5000, "tpm": 4_000_000},
        "tier4": {"rpm": 10_000, "tpm": 10_000_000},
        "tier5": {"rpm": 30_000, "tpm": 150_000_000},
    }
    limits = tiers.get(api_tier, tiers["tier5"])

    enhanced_env = OASISChatAgentCompatibleEnv(env, api_limits=limits, semaphore_permits=semaphore)

    print(f"[API Manager] Created Token Bucket Enhanced Environment")
    print(f"   Limits: {limits['rpm']} RPM | {limits['tpm']} TPM")
    print(f"   Algorithm: Token Bucket (High Concurrency Optimized)")

    return enhanced_env
