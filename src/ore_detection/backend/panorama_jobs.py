"""Background panorama prediction jobs for the stdlib backend."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ore_detection.inference.model_prediction import load_simple_unet_checkpoint
from ore_detection.inference.tiled_prediction import (
    PanoramaPredictionCancelled,
    save_tiled_selected_model_prediction,
)


@dataclass(frozen=True)
class PanoramaPredictionRequest:
    """Resolved request for one panorama prediction job."""

    image_path: Path
    binary_model_path: Path
    ore_model_path: Path | None = None
    model_kind: str = "binary"
    include_ore_model: bool = False
    device: str = "auto"
    binary_threshold: float = 0.5
    tile_size: int = 512
    overlap: int = 0
    batch_size: int = 16


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class PanoramaJobManager:
    """Small in-process job runner with disk-backed progress snapshots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def start(
        self,
        *,
        request: PanoramaPredictionRequest,
        jobs_root: str | Path,
        predictions_root: str | Path,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        jobs_root = Path(jobs_root)
        predictions_root = Path(predictions_root)
        job_dir = jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        cancel_event = threading.Event()
        started_at = time.perf_counter()
        status = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "image_path": str(request.image_path),
            "binary_model_path": str(request.binary_model_path),
            "ore_model_path": str(request.ore_model_path) if request.ore_model_path is not None else None,
            "model_kind": request.model_kind,
            "include_ore_model": request.include_ore_model,
            "tile_size": request.tile_size,
            "overlap": request.overlap,
            "stride": request.tile_size - request.overlap,
            "batch_size": request.batch_size,
            "binary_threshold": request.binary_threshold,
            "device": request.device,
            "processed_tiles": 0,
            "total_tiles": 0,
            "processed_batches": 0,
            "total_batches": 0,
            "elapsed_sec": 0.0,
            "eta_sec": None,
            "tile_per_sec": 0.0,
            "batch_per_sec": 0.0,
            "timings": {},
            "artifacts": {},
            "sample_dir": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "_started_at_monotonic": started_at,
            "_progress_path": str(job_dir / "progress.json"),
        }
        with self._lock:
            self._jobs[job_id] = status
            self._cancel_events[job_id] = cancel_event
            self._write_status_locked(job_id)
        thread = threading.Thread(
            target=self._run_job,
            kwargs={
                "job_id": job_id,
                "request": request,
                "jobs_root": jobs_root,
                "predictions_root": predictions_root,
                "cancel_event": cancel_event,
            },
            daemon=True,
        )
        thread.start()
        return self.status(job_id, jobs_root=jobs_root)

    def status(self, job_id: str, *, jobs_root: str | Path) -> dict[str, Any]:
        with self._lock:
            if job_id in self._jobs:
                return self._public_status(self._jobs[job_id])
        path = Path(jobs_root) / job_id / "progress.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown panorama job: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def cancel(self, job_id: str, *, jobs_root: str | Path) -> dict[str, Any]:
        with self._lock:
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
                if self._jobs.get(job_id, {}).get("status") in {"queued", "running"}:
                    self._jobs[job_id]["phase"] = "cancelling"
                    self._write_status_locked(job_id)
        return self.status(job_id, jobs_root=jobs_root)

    def _run_job(
        self,
        *,
        job_id: str,
        request: PanoramaPredictionRequest,
        jobs_root: Path,
        predictions_root: Path,
        cancel_event: threading.Event,
    ) -> None:
        try:
            self._update(job_id, status="running", phase="loading_models")
            resolved_device = _resolve_device(request.device)
            if request.model_kind in {"binary", "ct_unet"}:
                selected_model = load_simple_unet_checkpoint(request.binary_model_path, device=resolved_device)
            elif request.model_kind == "ore":
                if request.ore_model_path is None:
                    raise FileNotFoundError("ore_model_path is required for ore model inference")
                selected_model = load_simple_unet_checkpoint(request.ore_model_path, device=resolved_device)
            else:
                raise ValueError("model_kind must be `binary`, `ct_unet`, or `ore`")
            self._update(job_id, phase="opening_image", device=resolved_device)

            def progress_callback(values: dict[str, Any]) -> None:
                self._update(job_id, **values)

            effective_model_kind = "ore" if request.model_kind == "ore" or selected_model.metadata.task == "multiclass" else "binary"
            artifacts = save_tiled_selected_model_prediction(
                request.image_path,
                model=selected_model,
                model_kind=effective_model_kind,
                output_root=predictions_root / "panorama",
                binary_threshold=request.binary_threshold,
                tile_size=request.tile_size,
                overlap=request.overlap,
                batch_size=request.batch_size,
                sample_id=job_id,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            self._update(
                job_id,
                status="completed",
                phase="completed",
                sample_dir=str(artifacts.sample_dir),
                artifacts=metadata.get("artifacts", {}),
                metadata_path=str(artifacts.metadata_path),
                effective_model_kind=effective_model_kind,
            )
        except PanoramaPredictionCancelled as exc:
            self._update(job_id, status="cancelled", phase="cancelled", error=str(exc))
        except Exception as exc:  # keep backend alive and expose actionable UI status
            selected_path = request.ore_model_path if request.model_kind == "ore" else request.binary_model_path
            self._update(
                job_id,
                status="failed",
                phase="failed",
                error=f"{request.model_kind} panorama prediction failed with {type(exc).__name__}: {exc}",
                selected_model_path=str(selected_path) if selected_path is not None else None,
            )
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._write_status_locked(job_id)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            status = self._jobs[job_id]
            status.update(values)
            self._refresh_rates_locked(status)
            self._write_status_locked(job_id)

    def _refresh_rates_locked(self, status: dict[str, Any]) -> None:
        started = float(status.get("_started_at_monotonic", time.perf_counter()))
        elapsed = max(0.0, time.perf_counter() - started)
        status["elapsed_sec"] = elapsed
        processed_tiles = int(status.get("processed_tiles") or 0)
        total_tiles = int(status.get("total_tiles") or 0)
        processed_batches = int(status.get("processed_batches") or 0)
        status["tile_per_sec"] = (processed_tiles / elapsed) if elapsed > 0 and processed_tiles else 0.0
        status["batch_per_sec"] = (processed_batches / elapsed) if elapsed > 0 and processed_batches else 0.0
        if processed_tiles > 0 and total_tiles > processed_tiles:
            status["eta_sec"] = (elapsed / processed_tiles) * (total_tiles - processed_tiles)
        elif total_tiles and processed_tiles >= total_tiles:
            status["eta_sec"] = 0.0
        else:
            status["eta_sec"] = None

    def _write_status_locked(self, job_id: str) -> None:
        status = self._jobs[job_id]
        path = Path(str(status["_progress_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._public_status(status), indent=2), encoding="utf-8")

    def _public_status(self, status: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in status.items() if not key.startswith("_")}


PANORAMA_JOB_MANAGER = PanoramaJobManager()
