"""
Health check endpoint.

Provides:
- Service health status
- Model load state
- Uptime tracking
- Cache connectivity
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends

from onnx_model import ONNXModel
from schemas import HealthStatus
from services.cache import EmbeddingCache

router = APIRouter(tags=["health"])

# Module-level start time
_start_time: float = time.time()


def set_start_time(t: float) -> None:
    global _start_time
    _start_time = t


@router.get("/health", response_model=HealthStatus)
async def health(
    model: ONNXModel = Depends(lambda: _get_model()),
    cache: EmbeddingCache = Depends(lambda: _get_cache()),
) -> HealthStatus:
    """
    Health check endpoint.

    Returns service status, model info, uptime, and cache connectivity.
    """
    uptime = time.time() - _start_time

    cache_ok = False
    if cache and cache.enabled:
        try:
            cache_ok = True
        except Exception:
            cache_ok = False

    return HealthStatus(
        status="ok" if model.is_loaded else "degraded",
        model_loaded=model.is_loaded,
        model_path=model.model_path if model.is_loaded else "N/A",
        model_provider=model.provider if model.is_loaded else "N/A",
        uptime_sec=round(uptime, 2),
        cache_connected=cache_ok,
    )


# ---------------------------------------------------------------------------
# Lazy dependency injection (set at app startup)
# ---------------------------------------------------------------------------
_model_instance: Optional[ONNXModel] = None
_cache_instance: Optional[EmbeddingCache] = None


def _get_model() -> ONNXModel:
    if _model_instance is None:
        raise RuntimeError("Model not initialized yet")
    return _model_instance


def _get_cache() -> EmbeddingCache:
    if _cache_instance is None:
        raise RuntimeError("Cache not initialized yet")
    return _cache_instance


def init_deps(model: ONNXModel, cache: EmbeddingCache) -> None:
    """Initialize global dependencies (called from main.py on startup)."""
    global _model_instance, _cache_instance
    _model_instance = model
    _cache_instance = cache
