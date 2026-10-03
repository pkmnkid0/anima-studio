"""
Anima Studio's own training backend - server layer.

Implements the exact HTTP contract app/services/backend_client.py already
speaks (reverse-engineered from the vendored LoRA_Easy_Training_Scripts
backend's own main.py/utils/validation.py - see NOTES.md in this folder),
so the existing Train tab works against this unchanged: just point
"Backend URL" at wherever this is running instead.

This does NOT shell out to separate training scripts the way the vendored
backend does. /validate keeps the parsed config in memory; /train runs
the matching trainer function (sdxl_trainer.run or anima_trainer.run) on
a background thread and tracks its progress for /is_training to report.

Endpoints:
  POST /validate       body: {"args", "dataset", "accelerate"} -> {"tags": {}}
  GET  /train           params: train_mode, sdxl, flux, anima, accelerate_*
  GET  /is_training     -> {"training": bool, "errored": bool}
  GET  /stop_training    params: force
  GET  /check_path       params: path -> {"exists": bool}
  GET  /status           (extra, not part of the original contract - step/
                           loss/epoch progress for a richer Monitor tab)
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, request

_lock = threading.Lock()
_validated: dict | None = None  # last successful /validate payload
_thread: threading.Thread | None = None
_state = {"training": False, "errored": False, "error_detail": None}
_progress = {"step": 0, "total_steps": None, "epoch": 0, "loss": None}
_stop_requested = threading.Event()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/validate")
    def validate():
        global _validated
        body = request.get_json(force=True) or {}
        args = body.get("args")
        dataset = body.get("dataset")
        if not isinstance(args, dict) or not isinstance(dataset, dict):
            return jsonify({"detail": "args and dataset are required"}), 400

        errors = _validate_payload(args, dataset)
        if errors:
            return jsonify({"detail": errors}), 400

        with _lock:
            _validated = {"args": args, "dataset": dataset, "accelerate": body.get("accelerate") or {}}
        return jsonify({"tags": {}})

    @app.get("/train")
    def train():
        global _thread
        with _lock:
            if _state["training"]:
                return jsonify({"detail": "Training Already Running"}), 409
            if _validated is None:
                return jsonify({"detail": "No Previously Validated Args"}), 400
            validated = _validated

        is_sdxl = request.args.get("sdxl", "False") == "True"
        is_flux = request.args.get("flux", "False") == "True"
        is_anima = request.args.get("anima", "False") == "True"
        train_mode = request.args.get("train_mode", "lora")

        runner = _pick_runner(train_mode, is_sdxl, is_flux, is_anima)
        if runner is None:
            return jsonify({
                "detail": "Invalid Train Parameters",
                "sdxl": is_sdxl, "flux": is_flux, "anima": is_anima, "train_mode": train_mode,
            }), 400

        _stop_requested.clear()
        with _lock:
            _state["training"] = True
            _state["errored"] = False
            _state["error_detail"] = None
            _progress.update(step=0, total_steps=None, epoch=0, loss=None)
        _thread = threading.Thread(target=_run_training, args=(runner, validated), daemon=True)
        _thread.start()
        return jsonify({"detail": "Training Started", "training": True})

    @app.get("/is_training")
    def is_training():
        with _lock:
            return jsonify({"training": _state["training"], "errored": _state["errored"]})

    @app.get("/status")
    def status():
        with _lock:
            return jsonify({**_state, **_progress})

    @app.get("/stop_training")
    def stop_training():
        with _lock:
            if not _state["training"]:
                return jsonify({"detail": "Not Currently Training"}), 400
        _stop_requested.set()
        return jsonify({"detail": "Training Thread Requested to Die"})

    @app.get("/check_path")
    def check_path():
        path = request.args.get("path", "")
        return jsonify({"exists": bool(path) and Path(path).exists()})

    return app


def _validate_payload(args: dict, dataset: dict) -> list[str]:
    errors = []
    subsets = dataset.get("subsets") or []
    if not subsets:
        errors.append("dataset.subsets is empty - add at least one image folder.")
    for i, s in enumerate(subsets):
        image_dir = s.get("image_dir")
        if not image_dir:
            errors.append(f"subset {i} has no image_dir.")
        elif not Path(image_dir).is_dir():
            errors.append(f"subset {i}: '{image_dir}' is not a folder.")
    saving = args.get("saving_args", {})
    if not saving.get("output_dir"):
        errors.append("saving_args.output_dir is required.")
    return errors


def _pick_runner(train_mode: str, is_sdxl: bool, is_flux: bool, is_anima: bool):
    if train_mode != "lora":
        return None  # this backend only implements LoRA training for now
    if is_sdxl and not is_flux and not is_anima:
        from . import sdxl_trainer
        return sdxl_trainer.run
    if is_anima and not is_sdxl and not is_flux:
        from . import anima_trainer
        return anima_trainer.run
    return None  # flux, or an inconsistent combination - not implemented here


def _run_training(runner, validated: dict) -> None:
    def on_progress(**kwargs) -> None:
        with _lock:
            _progress.update(kwargs)

    def should_stop() -> bool:
        return _stop_requested.is_set()

    try:
        runner(validated["args"], validated["dataset"], on_progress=on_progress, should_stop=should_stop)
    except Exception as exc:  # noqa: BLE001 - reported via /is_training and /status
        traceback.print_exc()
        with _lock:
            _state["errored"] = True
            _state["error_detail"] = str(exc)
    finally:
        with _lock:
            _state["training"] = False


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    create_app().run(host="127.0.0.1", port=port, threaded=True)
