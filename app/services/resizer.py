"""
Generates smaller, model-friendly-resolution copies of a whole dataset
tree - lower-resolution training data trains noticeably faster (fewer
pixels per step), which is what you'd want before a first-pass or
low-res training run. Adapted from a script the user shared, with three
bugs fixed along the way:

- Transparent images (RGBA/LA/P) are now composited onto a white
  background before the alpha channel is dropped, instead of just
  discarding alpha and keeping whatever RGB values happened to sit
  underneath it - which for a lot of PNG exporters is black, and showed
  up as dark fringing around transparent edges on cutout-style art.
- The "already small enough, just round to a multiple of 64" path now
  derives height from the rounded width via the aspect ratio, instead
  of rounding each axis independently - the old way could visibly skew
  small/thin images.
- Output filenames now include the source extension (art.png ->
  art_png.png, art.jpg -> art_jpg.png) so a folder that happens to have
  both art.png and art.jpg doesn't have one silently overwrite the
  other's resized copy (same fix applied to the copied caption file).

Runs on a thread pool, not a process pool like the original script did -
this executes inside the already-running Anima Studio server, and
spawning subprocesses from inside a long-running app is a real way to
hang or duplicate the app, particularly on Windows, where multiprocessing
re-imports the launching script in each worker unless very carefully
guarded. Pillow releases the GIL during its own resize/decode/encode
work, so threads still get real parallelism for this without that risk.
"""
from __future__ import annotations

import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

OUTPUT_FOLDER_NAME = "resized"
DEFAULT_RESOLUTIONS = [512, 1024, 1536]
DIVISIBLE = 64
_VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_lock = threading.Lock()
_status: dict = {"stage": "idle", "done": 0, "total": 0, "errors": [], "folder": None}


class ResizeError(RuntimeError):
    pass


def status() -> dict:
    with _lock:
        return dict(_status, errors=list(_status["errors"]))


def is_busy() -> bool:
    with _lock:
        return _status["stage"] == "running"


def _set(**kwargs) -> None:
    with _lock:
        _status.update(kwargs)


def start(folder: str, resolutions: list[int]) -> dict:
    if is_busy():
        raise ResizeError("A resize job is already running - wait for it to finish.")
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise ResizeError(f"Not a directory: {folder}")

    clean_resolutions = sorted({int(r) for r in resolutions if int(r) > 0})
    if not clean_resolutions:
        raise ResizeError("Choose at least one resolution.")

    _set(stage="running", done=0, total=0, errors=[], folder=str(base))
    threading.Thread(target=_run, args=(base, clean_resolutions), daemon=True).start()
    return status()


def _find_images(base: Path) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(base):
        # Don't descend into any previous output folder, at any depth, so
        # re-running this doesn't reprocess its own output.
        dirs[:] = [d for d in dirs if d != OUTPUT_FOLDER_NAME]
        for fname in files:
            if Path(fname).suffix.lower() in _VALID_EXTENSIONS:
                found.append(Path(root) / fname)
    return found


def _run(base: Path, resolutions: list[int]) -> None:
    try:
        images = _find_images(base)
        _set(total=len(images))
        if not images:
            _set(stage="done")
            return

        targets = {res: res * res for res in resolutions}
        errors: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            futures = {pool.submit(_process_one, path, targets): path for path in images}
            for future in as_completed(futures):
                done += 1
                err = future.result()
                if err:
                    errors.append(err)
                _set(done=done, errors=errors)
        _set(stage="done")
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via status()
        _set(stage="error", errors=[str(exc)])


def _process_one(path: Path, targets: dict[int, int]) -> str | None:
    try:
        source_dir = path.parent
        base_name = path.stem
        ext_tag = path.suffix.lower().lstrip(".")
        caption_path = source_dir / f"{base_name}.txt"

        with Image.open(path) as raw:
            img = _flatten_to_rgb(raw)
            orig_w, orig_h = img.size

            for res, target_area in targets.items():
                target_w, target_h = _target_shape(orig_w, orig_h, target_area)
                is_downscale = (target_w * target_h) <= (orig_w * orig_h)
                resample = Image.Resampling.BOX if is_downscale else Image.Resampling.LANCZOS
                resized = img.resize((target_w, target_h), resample)

                dest_dir = source_dir / OUTPUT_FOLDER_NAME / str(res)
                dest_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{base_name}_{ext_tag}.png"
                resized.save(dest_dir / out_name, format="PNG")

                if caption_path.exists():
                    shutil.copy2(caption_path, dest_dir / f"{base_name}_{ext_tag}.txt")
        return None
    except Exception as exc:  # noqa: BLE001 - collected into the job's error list
        return f"{path.name}: {exc}"


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Drops alpha the safe way: composite onto white first. Plain
    .convert('RGB') on an RGBA/LA/P image keeps whatever color values
    sat under the transparent pixels (often black), which shows up as
    dark fringing around transparent edges once alpha is gone."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _target_shape(width: int, height: int, target_area: int, divisible: int = DIVISIBLE) -> tuple[int, int]:
    """New size preserving aspect ratio, capped at target_area, rounded
    to the nearest multiple of `divisible` (most training pipelines want
    both dimensions divisible by 64)."""
    current_area = width * height
    aspect_ratio = width / height

    if current_area <= target_area:
        # Already small enough - just round to a valid size instead of
        # upscaling it. Height is derived from the rounded width via the
        # aspect ratio rather than rounded independently, so a small/thin
        # image doesn't visibly skew (64px can be a large fraction of a
        # small image's own dimensions).
        new_w = max(divisible, round(width / divisible) * divisible)
        new_h = max(divisible, round((new_w / aspect_ratio) / divisible) * divisible)
        return new_w, new_h

    target_h = math.sqrt(target_area / aspect_ratio)
    target_w = aspect_ratio * target_h
    new_w = max(divisible, round(target_w / divisible) * divisible)
    new_h = max(divisible, round(target_h / divisible) * divisible)
    return new_w, new_h
