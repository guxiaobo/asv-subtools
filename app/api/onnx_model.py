"""
ONNX model wrapper with hot-reload support.

Provides:
- ONNX session management
- Model hot-reload (periodic checksum comparison)
- Multi-provider support (CPU, CUDA, CoreML)
- Thread-safe inference
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXModel:
    """
    Wrapper around ONNX Runtime inference session with hot-reload support.

    Usage::

        model = ONNXModel(
            model_path="/path/to/model.onnx",
            provider="CPUExecutionProvider",
            intra_op_threads=4,
            hot_reload_interval_sec=30,
        )
        model.start_hot_reload()
        output = model.infer(input_dict)
        model.stop_hot_reload()
    """

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        provider_options: Optional[Dict[str, Any]] = None,
        inter_op_threads: int = 4,
        intra_op_threads: int = 4,
        hot_reload_interval_sec: int = 30,
    ) -> None:
        self._model_path = Path(model_path)
        self._provider = provider
        self._provider_options = provider_options or {}
        self._inter_op_threads = inter_op_threads
        self._intra_op_threads = intra_op_threads
        self._hot_reload_interval_sec = hot_reload_interval_sec

        # Session and metadata — protected by _lock
        self._lock = threading.RLock()
        self._session: Optional[ort.InferenceSession] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, List[int]] = {}
        self._input_types: Dict[str, np.dtype] = {}

        # Hot-reload state
        self._last_checksum: Optional[str] = None
        self._last_checked_at: float = 0.0
        self._hot_reload_active = False
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load initial session
        self._load_session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the model session is active."""
        return self._session is not None

    @property
    def model_path(self) -> str:
        return str(self._model_path)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    @property
    def input_shapes(self) -> Dict[str, List[int]]:
        return dict(self._input_shapes)

    def infer(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Run inference with the provided feed dict.

        Args:
            feed_dict: Dict mapping input names to numpy arrays.

        Returns:
            List of output numpy arrays matching the model's output order.
        """
        session = self._session
        if session is None:
            raise RuntimeError("ONNX model session is not loaded")

        # Validate input names
        for name in feed_dict:
            if name not in self._input_names:
                logger.warning("Unknown input '%s'; valid inputs: %s", name, self._input_names)

        # ONNX Runtime releases the GIL during Run() — true parallelism
        # when called from multiple threads.
        outputs = session.run(self._output_names, feed_dict)
        return outputs

    def get_input_meta(self, name: str) -> Optional[Tuple[List[int], np.dtype]]:
        """Get shape and dtype for a named input."""
        if name in self._input_shapes:
            return list(self._input_shapes[name]), self._input_types[name]
        return None

    def reload(self) -> bool:
        """Force-reload the model from disk. Returns True on success."""
        logger.info("Reloading model from %s", self._model_path)
        try:
            self._load_session()
            logger.info("Model reloaded successfully")
            return True
        except Exception:
            logger.exception("Failed to reload model")
            return False

    def start_hot_reload(self) -> None:
        """Start background thread that checks for model changes periodically."""
        if self._hot_reload_active:
            return
        self._hot_reload_active = True
        self._stop_event.clear()
        self._reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            name="onnx-hot-reload",
            daemon=True,
        )
        self._reload_thread.start()
        logger.info(
            "Hot-reload started (interval=%ds)", self._hot_reload_interval_sec
        )

    def stop_hot_reload(self) -> None:
        """Stop the hot-reload background thread."""
        self._hot_reload_active = False
        self._stop_event.set()
        if self._reload_thread and self._reload_thread.is_alive():
            self._reload_thread.join(timeout=5)
        logger.info("Hot-reload stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_session(self) -> None:
        """Load/reload the ONNX session with current config."""
        model_path_str = str(self._model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path_str}")

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = self._inter_op_threads
        session_options.intra_op_num_threads = self._intra_op_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_profiling = False
        session_options.log_severity_level = 3  # 0:verbose .. 4:fatal

        # Resolve execution provider
        providers = self._resolve_providers()
        logger.info("Loading ONNX model with providers=%s", providers)

        session = ort.InferenceSession(
            model_path_str,
            sess_options=session_options,
            providers=providers,
        )

        # Extract metadata
        input_names = []
        input_shapes = {}
        input_types = {}
        for inp in session.get_inputs():
            input_names.append(inp.name)
            input_shapes[inp.name] = list(inp.shape) if inp.shape else []
            input_types[inp.name] = inp.type

        output_names = [o.name for o in session.get_outputs()]

        # Update state under lock
        with self._lock:
            self._session = session
            self._input_names = input_names
            self._output_names = output_names
            self._input_shapes = input_shapes
            self._input_types = input_types
            # Record checksum
            self._last_checksum = self._compute_file_checksum()
            self._last_checked_at = time.time()

        logger.info(
            "Model loaded: inputs=%s, outputs=%s",
            input_names,
            output_names,
        )

    def _resolve_providers(self) -> List[str]:
        """Resolve execution provider list, falling back to available ones."""
        requested = self._provider
        available = ort.get_available_providers()

        if requested in available:
            if self._provider_options:
                return [(requested, self._provider_options)]  # type: ignore[return-value]
            return [requested]

        # Fallback: try to find a suitable provider
        logger.warning(
            "Requested provider '%s' not available. Available: %s. Falling back.",
            requested,
            available,
        )
        priority = ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                     "MIGraphXExecutionProvider", "CPUExecutionProvider"]
        for p in priority:
            if p in available:
                logger.info("Falling back to provider: %s", p)
                return [p]
        return ["CPUExecutionProvider"]

    def _compute_file_checksum(self) -> str:
        """Compute MD5 checksum of the model file."""
        hasher = hashlib.md5()
        with open(self._model_path, "rb") as f:
            # Read in 64KB chunks
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _check_model_changed(self) -> bool:
        """Check if the model file has changed since last load."""
        if not self._model_path.exists():
            return False
        current = self._compute_file_checksum()
        return current != self._last_checksum

    def _hot_reload_loop(self) -> None:
        """Background loop that checks and reloads on changes."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self._hot_reload_interval_sec)
            if self._stop_event.is_set():
                break
            try:
                if self._check_model_changed():
                    logger.info("Model file changed, triggering hot-reload")
                    self.reload()
            except Exception:
                logger.exception("Error in hot-reload check")
