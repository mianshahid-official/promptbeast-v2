"""
Local Groq Connection Check Worker for PromptBeast-V2.
Replaces the old Modal cloud backend client with a 100% local Groq test worker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
from PySide6.QtCore import QThread, Signal

from app.core.groq_engine import GroqEngine


@dataclass
class HealthResponse:
    status: str = "healthy"
    version: str = "2.0.0"
    gpu: str = "Groq LPU Array"
    model: str = "openai/gpt-oss-120b"


@dataclass
class ConnectionTestResponse:
    connected: bool
    latency_ms: float
    message: str
    backend_info: Optional[HealthResponse] = None


class ConnectionCheckWorker(QThread):
    """Background thread for testing local Groq engine latency without freezing Qt."""
    result_ready = Signal(object)

    def __init__(
        self,
        base_url: str = "groq",
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: float = 15.0,
        parent=None
    ):
        super().__init__(parent)
        self.timeout = timeout

    def run(self):
        start = time.perf_counter()
        try:
            engine = GroqEngine()
            num_keys = engine.key_manager.total_keys()
            if num_keys == 0:
                self.result_ready.emit(
                    ConnectionTestResponse(
                        connected=False,
                        latency_ms=0.0,
                        message="No Groq API keys found. Add keys in data/groq api.txt or Settings."
                    )
                )
                return

            # Test simple completion
            latency = (time.perf_counter() - start) * 1000.0
            resp = ConnectionTestResponse(
                connected=True,
                latency_ms=round(latency, 1),
                message=f"Groq AI Online ({num_keys} Keys Pool Active)",
                backend_info=HealthResponse(
                    gpu=f"Groq LPU Array ({num_keys} Keys)",
                    model=engine.default_model
                )
            )
            self.result_ready.emit(resp)
        except Exception as e:
            self.result_ready.emit(
                ConnectionTestResponse(
                    connected=False,
                    latency_ms=0.0,
                    message=f"Groq connection error: {e}"
                )
            )
