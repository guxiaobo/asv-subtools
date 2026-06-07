"""
FastAPI application entry point for ASV (Automatic Speaker Verification) API.

Usage:
    # Development
    python -m api.main

    # Production
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

    # Custom config
    ASV_CONFIG_PATH=/path/to/config.yaml python -m api.main
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import AppConfig, dump_config, load_config
from onnx_model import ONNXModel
from routers import health as health_router
from routers import recordings as recordings_router
from routers import verify as verify_router
from services.audio import AudioLoader, AudioLoadError, InsufficientAudioError
from services.cache import create_cache
from services.fetcher import AudioFetcher
from services.recording_db import init_db
from services.verifier import SpeakerVerifier

logger = logging.getLogger("asv-api")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app_config: AppConfig = load_config()
model: ONNXModel
verifier: SpeakerVerifier
audio_loader: AudioLoader


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan: startup and shutdown."""
    global app_config, model, verifier, audio_loader

    startup_time = time.time()

    # 1. Log configuration
    cfg_dump = dump_config(app_config)
    logger.info("Starting ASV API with config: %s", cfg_dump)

    # 2. Initialize audio fetcher (plugin: local_file / s3 / redis)
    #    config.yaml 只指定 fetcher.type，具体参数由子类自身维护
    fetcher_type = app_config.fetcher_type
    try:
        fetcher = AudioFetcher.create(fetcher_type)
        logger.info("AudioFetcher created: type=%s", fetcher_type)
    except ValueError as e:
        logger.warning("Fetcher init failed: %s — proceeding without ID-based fetch", e)
        fetcher = None

    # 3. Initialize audio loader
    audio_loader = AudioLoader(
        audio_config=app_config.audio,
        storage_config=app_config.storage,
        fetcher=fetcher,
    )
    logger.info("AudioLoader initialized (target_sr=%d fetcher=%s)",
                app_config.audio.target_sample_rate, fetcher_type)

    # 4. Load ONNX model
    model_path = app_config.model.path
    if not os.path.exists(model_path):
        logger.warning(
            "Model file not found at '%s'. "
            "Set ASV_MODEL_PATH or update config.yaml. "
            "Health check will report 'degraded'.",
            model_path,
        )

    model = ONNXModel(
        model_path=model_path,
        provider=app_config.model.provider,
        provider_options=app_config.model.provider_options or {},
        inter_op_threads=app_config.model.inter_op_threads,
        intra_op_threads=app_config.model.intra_op_threads,
        hot_reload_interval_sec=app_config.model.hot_reload_interval_sec,
    )

    # Start hot-reload background thread
    if model.is_loaded:
        model.start_hot_reload()
        logger.info("Model loaded and hot-reload enabled: %s", model_path)
    else:
        logger.warning("Model not loaded at startup (will retry via hot-reload)")

    # 5. Initialize cache
    cache = create_cache(app_config.cache)
    logger.info("Cache initialized: enabled=%s backend=%s", cache.enabled, type(cache).__name__)

    # 5. Initialize verifier
    verifier = SpeakerVerifier(
        config=app_config,
        model=model,
        audio_loader=audio_loader,
        cache=cache,
    )

    # 6. Inject dependencies into routers
    health_router.init_deps(model, cache)
    health_router.set_start_time(startup_time)
    verify_router.init_verifier(verifier)

    # 7. Initialize training database (for recording push)
    try:
        await init_db(app_config.training.db_path)
        logger.info("Training DB initialized: %s", app_config.training.db_path)
    except Exception as e:
        logger.warning("Training DB init failed: %s — recordings push will be unavailable", e)

    logger.info(
        "ASV API ready in %.2fs | %s | threshold=%.3f",
        time.time() - startup_time,
        "model_loaded" if model.is_loaded else "model_deferred",
        app_config.verification.default_threshold,
    )

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down ASV API...")
    model.stop_hot_reload()
    # Note: no global DB connection to close — each request uses its own.


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ASV Speaker Verification API",
    version="0.1.0",
    description="""
    Automatic Speaker Verification (ASV) API for bank customer service and
    debt collection scenarios.

    Supports:
    - **Direct mode**: Upload two audio files (WAV, ulaw, alaw, etc.)
    - **Indirect mode**: Reference audio by ID from NAS/S3/Redis
    - **Scenario-specific thresholds**: customer_service, debt_collection, audit
    - **Embedding cache**: Optional Redis for high-frequency speakers
    - **Model hot-reload**: Swap models without restart
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router.router)
app.include_router(verify_router.router)
app.include_router(recordings_router.router)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(AudioLoadError)
async def audio_load_error_handler(request: Request, exc: AudioLoadError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": getattr(exc, "code", "AUDIO_ERROR"),
                "message": str(exc),
            },
        },
    )


@app.exception_handler(InsufficientAudioError)
async def insufficient_audio_handler(request: Request, exc: InsufficientAudioError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "INSUFFICIENT_AUDIO",
                "message": str(exc),
            },
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
            },
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the API server via `python -m api.main`."""
    host = app_config.server.host
    port = app_config.server.port
    log_level = app_config.server.log_level

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting uvicorn on %s:%s (log_level=%s)", host, port, log_level)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=1,  # Single-process for dev; use `uvicorn --workers N` for production
        reload=False,
    )


if __name__ == "__main__":
    main()
