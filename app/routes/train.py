from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

from flask import Blueprint, jsonify, request, Response

from .. import config, schema, train_config
from ..toml_writer import dumps as toml_dumps
from ..services import backend_client, backend_manager, backend_setup

bp = Blueprint("train", __name__)


@bp.post("/parse_toml")
def parse_toml():
    """Load TOML: accepts either raw TOML text (JSON body {"text": "..."})
    or an uploaded file (multipart "file"), parses it, and returns the
    Train tab's UI JSON shape so the form can populate itself."""
    text = None
    if request.files.get("file"):
        text = request.files["file"].read().decode("utf-8", errors="replace")
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text")

    if not text:
        return jsonify({"error": "No TOML content received."}), 400

    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return jsonify({"error": f"That doesn't look like valid TOML: {exc}"}), 400

    try:
        canonical = train_config.from_toml_document(doc)
        ui = train_config.to_ui_payload(canonical)
    except Exception as exc:  # noqa: BLE001 - surface a clean message either way
        return jsonify({"error": f"Couldn't read that config: {exc}"}), 400

    return jsonify({"ui": ui})


@bp.post("/load_toml_from_disk")
def load_toml_from_disk():
    """Like /parse_toml, but reads the file straight off the server's
    filesystem (via the file browser's path) instead of a browser upload -
    for loading a .toml that already lives on disk, e.g. from a previous
    session or written by another tool."""
    body = request.get_json(force=True) or {}
    path = body.get("path", "")
    p = Path(path)
    if not p.is_file():
        return jsonify({"error": f"No such file: {path}"}), 400
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": f"Couldn't read {path}: {exc}"}), 400

    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return jsonify({"error": f"That doesn't look like valid TOML: {exc}"}), 400

    try:
        canonical = train_config.from_toml_document(doc)
        ui = train_config.to_ui_payload(canonical)
    except Exception as exc:  # noqa: BLE001 - surface a clean message either way
        return jsonify({"error": f"Couldn't read that config: {exc}"}), 400

    return jsonify({"ui": ui, "path": str(p)})


@bp.post("/save_toml_to_disk")
def save_toml_to_disk():
    """Writes the current config straight to a folder on the server's
    filesystem (chosen via the folder browser) instead of only offering
    it as a browser download."""
    body = request.get_json(force=True) or {}
    ui = body.get("ui", {})
    directory = body.get("dir", "")
    filename = (body.get("filename") or "").strip()
    if not directory:
        return jsonify({"error": "Choose a destination folder first."}), 400

    payload = train_config.build_backend_payload(ui)
    doc = train_config.to_toml_document(payload)
    text = toml_dumps(doc)

    if not filename:
        filename = (payload["args"].get("saving_args", {}).get("output_name") or "anima_lora_config") + ".toml"
    if not filename.lower().endswith(".toml"):
        filename += ".toml"

    d = Path(directory)
    try:
        d.mkdir(parents=True, exist_ok=True)
        target = d / filename
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": f"Couldn't write to {directory}: {exc}"}), 400

    return jsonify({"ok": True, "path": str(target)})


@bp.get("/schema")
def get_schema():
    return jsonify(schema.full_schema())


def _validation_warnings(payload: dict) -> list[str]:
    warnings = []
    general = payload["args"].get("general_args", {})
    anima = payload["args"].get("anima_args", {})
    saving = payload["args"].get("saving_args", {})
    if not anima.get("pretrained_model_name_or_path"):
        warnings.append("No DiT model path set.")
    if not anima.get("qwen3"):
        warnings.append("No Qwen3 text encoder path set.")
    if not anima.get("vae"):
        warnings.append("No VAE path set.")
    if not saving.get("output_dir"):
        warnings.append("No output folder set.")
    if not payload["subsets"]:
        warnings.append("No dataset subsets added - training needs at least one image folder.")
    else:
        for s in payload["subsets"]:
            if not s.get("image_dir"):
                warnings.append(f"Subset '{s.get('name')}' has no image folder set.")
    return warnings


@bp.post("/build")
def build():
    """Convert the UI payload into the backend structure + TOML text,
    without saving or training - used for live preview and for the
    "Download TOML" action."""
    ui = request.get_json(force=True) or {}
    payload = train_config.build_backend_payload(ui)
    doc = train_config.to_toml_document(payload)
    text = toml_dumps(doc)
    return jsonify({
        "payload": payload,
        "toml": text,
        "warnings": _validation_warnings(payload),
    })


@bp.post("/download_toml")
def download_toml():
    ui = request.get_json(force=True) or {}
    payload = train_config.build_backend_payload(ui)
    doc = train_config.to_toml_document(payload)
    text = toml_dumps(doc)
    filename = (payload["args"].get("saving_args", {}).get("output_name") or "anima_lora_config") + ".toml"
    return Response(
        text, mimetype="application/toml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- Presets (native JSON - exact round trip of the UI's own shape) ----

@bp.get("/presets")
def list_presets():
    presets = []
    for f in sorted(config.PRESETS_DIR.glob("*.json")):
        presets.append({"name": f.stem, "modified": f.stat().st_mtime})
    return jsonify({"presets": presets})


@bp.get("/presets/<name>")
def load_preset(name: str):
    path = _preset_path(name)
    if not path.exists():
        return jsonify({"error": "Preset not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@bp.post("/presets/<name>")
def save_preset(name: str):
    ui = request.get_json(force=True) or {}
    path = _preset_path(name)
    path.write_text(json.dumps(ui, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "name": name})


@bp.delete("/presets/<name>")
def delete_preset(name: str):
    path = _preset_path(name)
    if path.exists():
        path.unlink()
    return jsonify({"ok": True})


def _preset_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "preset"
    return config.PRESETS_DIR / f"{safe}.json"


# --- Backend control -----------------------------------------------------

@bp.get("/backend/ping")
def backend_ping():
    url = request.args.get("url", config.DEFAULT_BACKEND_URL)
    return jsonify(backend_client.ping(url))


@bp.post("/backend/validate")
def backend_validate():
    body = request.get_json(force=True) or {}
    ui = body.get("ui", {})
    url = body.get("backend_url", config.DEFAULT_BACKEND_URL)
    payload = train_config.build_backend_payload(ui)
    warnings = _validation_warnings(payload)
    # The backend's own /validate expects "subsets" nested inside "dataset"
    # (confirmed against its actual validate_dataset_args, which does
    # `for item in args["subsets"]`) - payload keeps them as separate top-
    # level keys internally (the TOML file format wants subsets at its own
    # top level instead), so they're merged here for this specific call.
    dataset_with_subsets = {**payload["dataset"], "subsets": payload["subsets"]}
    try:
        result = backend_client.validate(url, payload["args"], dataset_with_subsets)
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc), "warnings": warnings}), 502
    return jsonify({"result": result, "warnings": warnings})


@bp.post("/backend/start")
def backend_start():
    body = request.get_json(force=True) or {}
    ui = body.get("ui", {})
    url = body.get("backend_url", config.DEFAULT_BACKEND_URL)
    payload = train_config.build_backend_payload(ui)
    dataset_with_subsets = {**payload["dataset"], "subsets": payload["subsets"]}

    try:
        backend_client.validate(url, payload["args"], dataset_with_subsets)
        backend_client.start_training(
            url,
            train_mode=payload["train_mode"],
            has_sdxl=bool(payload["args"].get("general_args", {}).get("sdxl")),
            has_flux=bool(payload["args"].get("flux_args")),
            has_anima=bool(payload["args"].get("anima_args")),
            accelerate=ui.get("accelerate", {}),
        )
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"ok": True, "started_at": time.time()})


@bp.get("/backend/status")
def backend_status():
    url = request.args.get("url", config.DEFAULT_BACKEND_URL)
    try:
        return jsonify(backend_client.is_training(url))
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.post("/backend/stop")
def backend_stop():
    body = request.get_json(force=True) or {}
    url = body.get("backend_url", config.DEFAULT_BACKEND_URL)
    try:
        backend_client.stop_training(url)
    except backend_client.BackendError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"ok": True})


# --- Local backend process launcher ---------------------------------------
# Distinct from the /backend/* routes above (which talk *to* an already
# running backend over HTTP) - these start/stop that backend as a child
# process of Anima Studio itself, so the user doesn't have to open a
# second terminal for it.

@bp.get("/backend/process/settings")
def backend_process_settings():
    return jsonify(backend_manager.get_settings())


@bp.post("/backend/process/settings")
def backend_process_save_settings():
    body = request.get_json(force=True) or {}
    saved = backend_manager.save_settings(
        body.get("dir", ""), body.get("command", ""), bool(body.get("autostart", False)),
    )
    return jsonify(saved)


@bp.get("/backend/process/status")
def backend_process_status():
    return jsonify(backend_manager.status())


@bp.post("/backend/process/start")
def backend_process_start():
    body = request.get_json(force=True) or {}
    directory = body.get("dir", "")
    command = body.get("command", "")
    # Starting it is also a good time to remember these settings, so a
    # restart of Anima Studio doesn't lose them even without autostart on.
    current = backend_manager.get_settings()
    backend_manager.save_settings(directory, command, current.get("autostart", False))
    try:
        result = backend_manager.start(directory, command)
    except backend_manager.BackendLaunchError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@bp.post("/backend/process/stop")
def backend_process_stop():
    return jsonify(backend_manager.stop())


# --- In-app backend installer -----------------------------------------
# Runs the same clone + install steps as setup_backend.sh/.bat, from a
# button instead of a separate terminal. The installer itself can ask
# interactive questions, so /setup/input lets the UI act as a tiny
# terminal for whatever it asks rather than guessing blind.

@bp.post("/backend/setup/start")
def backend_setup_start():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(backend_setup.start(body.get("target_dir", "")))
    except backend_setup.SetupError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/backend/setup/status")
def backend_setup_status():
    return jsonify(backend_setup.status())


@bp.post("/backend/setup/input")
def backend_setup_input():
    body = request.get_json(force=True) or {}
    text = body.get("text", "")
    try:
        return jsonify(backend_setup.send_input(text))
    except backend_setup.SetupError as exc:
        return jsonify({"error": str(exc)}), 400
