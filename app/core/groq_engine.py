"""
Groq LLM Engine for PromptBeast-V2.
Implements:
- Only 3 allowed models: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b
- 950-token output cap baseline per request
- Effective-cap memory: drop cap to 800 for 60s after 429
- 2s pacing between requests
- Key-wait rule: retry same key if wait <= 3s, rotate if > 3s
- Header-driven multi-key rotation (remaining tokens & requests)
- Circuit breaker: 3 consecutive 429s parks key for 60s
- Strict structured output (JSON schema)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

ALLOWED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b"
]


def _parse_duration_string(val: str) -> Optional[float]:
    """Parses duration strings like '202ms', '17.4s', '7m12s', or '15' into seconds."""
    if not val:
        return None
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        pass
    total = 0.0
    matched = False
    m = re.search(r"([\d\.]+)\s*m(?:in)?(?![a-z])", val, re.IGNORECASE)
    if m:
        total += float(m.group(1)) * 60.0
        matched = True
    s = re.search(r"([\d\.]+)\s*s(?:ec)?(?![a-z])", val, re.IGNORECASE)
    if s:
        total += float(s.group(1))
        matched = True
    ms = re.search(r"([\d\.]+)\s*ms", val, re.IGNORECASE)
    if ms:
        total += float(ms.group(1)) / 1000.0
        matched = True
    return total if matched else None


def parse_groq_retry_after(headers: Optional[Dict[str, str]] = None, text: Optional[str] = None) -> Optional[float]:
    """Extracts wait time in seconds from Groq 429 response body or headers."""
    if text:
        m_s = re.search(r"try again in ([\d\.]+)\s*s", text, re.IGNORECASE)
        if m_s:
            return max(0.1, float(m_s.group(1)))
        m_ms = re.search(r"try again in ([\d\.]+)\s*ms", text, re.IGNORECASE)
        if m_ms:
            return max(0.1, float(m_ms.group(1)) / 1000.0)

    if headers:
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            try:
                return max(0.1, float(ra))
            except ValueError:
                pass
        reset_req = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
        if reset_req:
            dur = _parse_duration_string(reset_req)
            if dur is not None and dur > 0:
                return max(0.1, dur)

    return None


@dataclass
class KeyStats:
    remaining_tokens: int = 1000000
    remaining_requests: int = 10000
    consecutive_429: int = 0
    parked_until: float = 0.0
    cooldown_until: float = 0.0


class GroqMultiKeyManager:
    """Header-driven multi-key scheduler with rate-limit tracking and circuit breaker."""

    def __init__(self, keys: Optional[List[str]] = None):
        self.keys: List[str] = []
        if keys:
            self.keys = [k.strip() for k in keys if k and k.strip().startswith("gsk_")]

        if not self.keys:
            self._auto_load_keys()

        self._stats: Dict[str, KeyStats] = {k: KeyStats() for k in self.keys}
        self._current_index = 0

    def _auto_load_keys(self):
        env_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
        if env_keys:
            for k in env_keys.split(","):
                k = k.strip()
                if k.startswith("gsk_") and k not in self.keys:
                    self.keys.append(k)

        candidate_files = [
            Path("data/groq api.txt"),
            Path("groq api.txt"),
            Path("../groq api.txt"),
            Path("d:/AntiGravity/groq api.txt"),
        ]
        for cf in candidate_files:
            if cf.exists():
                try:
                    with open(cf, "r", encoding="utf-8") as f:
                        for line in f:
                            k = line.strip()
                            if k.startswith("gsk_") and k not in self.keys:
                                self.keys.append(k)
                except Exception:
                    pass

        self._stats = {k: KeyStats() for k in self.keys}

    def total_keys(self) -> int:
        return len(self.keys)

    def get_available_key(self) -> Tuple[str, float]:
        """
        Picks the available key with the most remaining tokens.
        If all are waiting/parked, returns the one becoming ready earliest with wait time.
        """
        if not self.keys:
            return "", 0.0

        now = time.time()
        # Find candidates not parked and not in cooldown
        candidates = []
        for k in self.keys:
            stat = self._stats.get(k, KeyStats())
            if now >= stat.parked_until and now >= stat.cooldown_until:
                candidates.append((k, stat.remaining_tokens))

        if candidates:
            # Pick key with maximum remaining tokens
            candidates.sort(key=lambda x: x[1], reverse=True)
            chosen_key = candidates[0][0]
            self._current_index = self.keys.index(chosen_key)
            return chosen_key, 0.0

        # All keys in cooldown or parked; find minimum ready time
        min_ready = float("inf")
        best_k = self.keys[0]
        for k in self.keys:
            stat = self._stats.get(k, KeyStats())
            ready = max(stat.parked_until, stat.cooldown_until)
            if ready < min_ready:
                min_ready = ready
                best_k = k

        wait = max(0.1, min_ready - now)
        return best_k, wait

    def record_success(self, key: str, headers: Dict[str, str]):
        stat = self._stats.setdefault(key, KeyStats())
        stat.consecutive_429 = 0
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        if rem_tok:
            try:
                stat.remaining_tokens = int(rem_tok)
            except ValueError:
                pass
        rem_req = headers.get("x-ratelimit-remaining-requests")
        if rem_req:
            try:
                stat.remaining_requests = int(rem_req)
            except ValueError:
                pass

    def record_429(self, key: str, wait_seconds: float):
        stat = self._stats.setdefault(key, KeyStats())
        stat.consecutive_429 += 1
        now = time.time()
        # Circuit breaker: 3 consecutive 429s -> park for 60s
        if stat.consecutive_429 >= 3:
            stat.parked_until = now + 60.0
            stat.cooldown_until = now + 60.0
        else:
            stat.cooldown_until = now + max(1.0, wait_seconds)

    def mark_invalid(self, key: str):
        stat = self._stats.setdefault(key, KeyStats())
        stat.parked_until = time.time() + 86400.0


class GroqEngine:
    """Production Groq LLM client adhering strictly to PromptBeast-V2 specs."""

    API_BASE = "https://api.groq.com/openai/v1"
    BASELINE_MAX_TOKENS = 950
    REDUCED_MAX_TOKENS = 800
    EFFECTIVE_CAP_RECOVERY_SECONDS = 60.0
    REQUEST_PACING_SECONDS = 2.0
    SHORT_DELAY_RETRY_SECONDS = 3.0

    def __init__(
        self,
        key_manager: Optional[GroqMultiKeyManager] = None,
        default_model: str = "openai/gpt-oss-120b",
        timeout: float = 60.0
    ):
        if default_model not in ALLOWED_MODELS:
            raise ValueError(f"Model '{default_model}' is not in allowed models: {ALLOWED_MODELS}")

        self.key_manager = key_manager or GroqMultiKeyManager()
        self.default_model = default_model
        self.timeout = timeout
        self._effective_cap = self.BASELINE_MAX_TOKENS
        self._effective_cap_until = 0.0
        self._last_request_time = 0.0

    @property
    def api_keys(self) -> List[str]:
        return self.key_manager.keys

    @api_keys.setter
    def api_keys(self, keys: List[str]) -> None:
        self.key_manager.keys = [k.strip() for k in keys if k and k.strip()]
        self.key_manager.current_index = 0

    def current_effective_max_tokens(self) -> int:
        if time.time() < self._effective_cap_until:
            return self._effective_cap
        return self.BASELINE_MAX_TOKENS

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        target_model = model or self.default_model
        if target_model not in ALLOWED_MODELS:
            raise ValueError(f"Model '{target_model}' is not in allowed models: {ALLOWED_MODELS}")

        num_keys = max(1, self.key_manager.total_keys())
        max_attempts = min(8, num_keys * 2)
        max_wait_budget = 120.0
        accumulated_wait = 0.0
        last_error = None

        for attempt in range(max_attempts):
            key, wait_needed = self.key_manager.get_available_key()
            if not key:
                raise RuntimeError("No Groq API keys available or all keys exhausted.")

            key_id = f"...{key[-6:]}"

            if wait_needed > 0.0:
                if accumulated_wait + wait_needed > max_wait_budget:
                    raise RuntimeError(f"Rate limit wait budget exceeded ({max_wait_budget}s).")
                time.sleep(wait_needed)
                accumulated_wait += wait_needed

            # 2.0s request pacing
            gap = time.monotonic() - self._last_request_time
            if gap < self.REQUEST_PACING_SECONDS:
                time.sleep(self.REQUEST_PACING_SECONDS - gap)

            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            effective_tok = min(
                max_tokens if max_tokens is not None else self.BASELINE_MAX_TOKENS,
                self.current_effective_max_tokens()
            )

            payload: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_tok
            }
            if response_format:
                payload["response_format"] = response_format

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    self._last_request_time = time.monotonic()
                    resp = client.post(f"{self.API_BASE}/chat/completions", headers=headers, json=payload)

                    if resp.status_code == 200:
                        self.key_manager.record_success(key, dict(resp.headers))
                        data = resp.json()
                        choices = data.get("choices", [])
                        msg = choices[0].get("message", {}) if choices else {}
                        return msg.get("content") or ""

                    elif resp.status_code == 429:
                        raw_wait = parse_groq_retry_after(dict(resp.headers), resp.text)
                        self._effective_cap = self.REDUCED_MAX_TOKENS
                        self._effective_cap_until = time.time() + self.EFFECTIVE_CAP_RECOVERY_SECONDS

                        if raw_wait is not None and raw_wait <= self.SHORT_DELAY_RETRY_SECONDS:
                            time.sleep(raw_wait)
                            accumulated_wait += raw_wait
                            continue

                        wait_sec = max(3.0, (raw_wait or 0.0) + 1.5)
                        self.key_manager.record_429(key, wait_sec)
                        last_error = f"HTTP 429: {resp.text}"
                        continue

                    elif resp.status_code in (401, 403):
                        self.key_manager.mark_invalid(key)
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        continue

                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        continue

            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(f"Groq generation failed after {max_attempts} attempts. Last error: {last_error}")

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str],
        schema: Dict[str, Any],
        schema_name: str = "structured_output",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Performs structured JSON generation using Groq's json_schema response format."""
        resp_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema
            }
        }
        raw_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            response_format=resp_format
        )
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            # Fallback regex search for json block if any wrapping occurred
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise RuntimeError(f"Failed to parse structured JSON from Groq output: {raw_text}") from e

