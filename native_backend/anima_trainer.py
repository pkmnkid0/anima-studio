"""
Anima LoRA training for Anima Studio's native backend.

Unlike sdxl_trainer.py (a genuinely new training loop built on official
diffusers/peft primitives), this deliberately does NOT reimplement
Anima's training loop from scratch. Anima is an obscure, very new
architecture - a Diffusion Transformer with a Qwen3 LLM text encoder
bridged through a custom adapter, trained via rectified flow - and
mainstream libraries have no support for it. The vendored backend at
training_backend/sd_scripts/ does (anima_train_network.py +
library/anima_models.py, Apache-2.0), and re-deriving that from
fragments read out of context would risk getting a subtle, unverifiable
detail wrong. What's actually new here: translating Anima Studio's own
config shape into what that real trainer expects, and invoking it
in-process (a direct Python call) instead of as a subprocess.

Concretely: the real training entry point
(anima_train_network.py's `if __name__ == "__main__"` block) builds its
`args` by parsing real CLI defaults, then does a couple of small fixups,
then calls `AnimaNetworkTrainer().train(args)`. This mirrors that exact
sequence - args come from the real parser's own defaults
(`setup_parser().parse_args([])`), not a hand-built Namespace guessing
at field names, so anything this code doesn't explicitly override still
gets a real, tested default rather than a missing-attribute crash deep
inside the trainer.

Dataset handoff: the base trainer only reads datasets from a TOML file
path (`--dataset_config`), verified against its actual schema in
library/config_util.py (top-level "datasets" list of DreamBooth-style
subsets, "image_dir" the only required field) - so Anima Studio's
dataset dict is written out as a real TOML file here, not passed
in-memory, since there's no in-memory entry point to hook into.

Honest limitation: this has NOT been run against real Anima model
weights (Qwen3 + the DiT checkpoint + the Qwen-Image VAE) - those are
large, separately-hosted downloads outside what could be fetched and
verified in the environment this was built in. What IS verified:
argument translation produces a valid, schema-correct config (tested
against the real parser and the real TOML schema), and the import path
into the vendored trainer resolves correctly.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import dataset as ds

_SD_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "training_backend" / "sd_scripts"


class TrainerError(RuntimeError):
    pass


def _get(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_sd_scripts_importable() -> None:
    if not _SD_SCRIPTS_DIR.is_dir():
        raise TrainerError(
            f"Anima training needs the bundled backend's sd_scripts folder, not found at {_SD_SCRIPTS_DIR}. "
            "Run \"Set up backend\" in the Train tab first."
        )
    if str(_SD_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SD_SCRIPTS_DIR))


def write_dataset_toml(dataset: dict, resolution: int, dest_path: Path) -> Path:
    """Anima Studio's dataset dict -> a real kohya-style TOML file, the
    only format the vendored trainer's dataset loader actually reads
    (see library/config_util.py's ConfigSanitizer - DB_SUBSET_DISTINCT_SCHEMA
    requires image_dir; everything else here is optional with the
    trainer's own defaults)."""
    import toml  # ships with the vendored backend's own venv

    subsets = dataset.get("subsets") or []
    if not subsets:
        raise TrainerError("No subsets in dataset config.")

    toml_subsets = []
    for s in subsets:
        image_dir = s.get("image_dir")
        if not image_dir or not Path(image_dir).is_dir():
            raise TrainerError(f"Subset image_dir is missing or not a folder: {image_dir!r}")
        entry = {
            "image_dir": str(image_dir),
            "num_repeats": int(s.get("num_repeats", 1)),
            # The vendored trainer's own default is ".caption", not ".txt" -
            # Anima Studio's own dataset convention (and every other tab in
            # this app) writes ".txt" captions, so this has to be explicit
            # or captions get silently dropped with no error (confirmed by
            # actually running this against the real dataset loader, not
            # just read from its source - see native_backend/NOTES.md).
            "caption_extension": s.get("caption_extension") or ".txt",
        }
        toml_subsets.append(entry)

    doc = {"datasets": [{"resolution": resolution, "batch_size": 1, "subsets": toml_subsets}]}
    dest_path.write_text(toml.dumps(doc), encoding="utf-8")
    return dest_path


def build_args(args: dict, dataset_toml_path: Path, output_dir: Path):
    """A real argparse.Namespace from the vendored trainer's own parser -
    every field starts at its real, tested default; only what Anima
    Studio's UI actually configures gets overridden on top. See module
    docstring for why this is safer than hand-building the Namespace."""
    _ensure_sd_scripts_importable()
    import anima_train_network

    parser = anima_train_network.setup_parser()
    ns = parser.parse_args([])

    anima_args = args.get("anima_args", {})
    general = args.get("general_args", {})
    optimizer = args.get("optimizer_args", {})
    network = args.get("network_args", {})
    saving = args.get("saving_args", {})

    ns.pretrained_model_name_or_path = anima_args.get("pretrained_model_name_or_path")
    ns.qwen3 = anima_args.get("qwen3")
    ns.vae = anima_args.get("vae")
    missing = [k for k in ("pretrained_model_name_or_path", "qwen3", "vae") if not getattr(ns, k)]
    if missing:
        raise TrainerError(f"Anima training needs anima_args.{', anima_args.'.join(missing)} set.")

    ns.network_module = "networks.lora_anima"
    ns.network_dim = int(_get(network, "network_dim", default=16))
    ns.network_alpha = float(_get(network, "network_alpha", default=8.0))

    if optimizer.get("learning_rate"):
        ns.learning_rate = float(optimizer["learning_rate"])
    if optimizer.get("optimizer_type"):
        ns.optimizer_type = optimizer["optimizer_type"]
    ns.max_train_steps = int(general.get("max_train_steps") or ns.max_train_steps or 1600)
    ns.mixed_precision = general.get("mixed_precision", ns.mixed_precision)

    ns.dataset_config = str(dataset_toml_path)
    ns.output_dir = str(output_dir)
    ns.output_name = saving.get("output_name", "anima_lora")
    if saving.get("save_every_n_steps"):
        ns.save_every_n_epochs = None
        ns.save_every_n_steps = int(saving["save_every_n_steps"])

    if getattr(ns, "attn_mode", None) == "sdpa":
        ns.attn_mode = "torch"  # backward-compat fixup, matches the real script's __main__ block
    ns._anima_model = True  # bypasses a generic "rectified flow" log/guard in the shared base trainer

    return ns


def run(args: dict, dataset: dict, on_progress, should_stop) -> None:
    _ensure_sd_scripts_importable()
    import anima_train_network
    from library import train_util

    saving = args.get("saving_args", {})
    output_dir_str = saving.get("output_dir")
    if not output_dir_str:
        raise TrainerError("saving_args.output_dir is required.")
    output_dir = Path(output_dir_str).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = int(_get(args, "general_args", "resolution", default=1024))
    # Fail fast on a genuinely missing/empty folder before handing off to
    # the vendored trainer, which would otherwise report a much less
    # specific error from deep inside its own dataset loader.
    ds.build_dataset_items(dataset)

    with tempfile.TemporaryDirectory(prefix="anima_studio_dataset_") as tmp:
        toml_path = write_dataset_toml(dataset, resolution, Path(tmp) / "dataset.toml")
        ns = build_args(args, toml_path, output_dir)
        ns = train_util.read_config_from_file(ns, anima_train_network.setup_parser())

        on_progress(step=0, total_steps=ns.max_train_steps, loss=None)

        # The real trainer reports its own progress via tqdm/accelerate,
        # not a plain callback - patching those internals would be
        # fragile across versions, so progress here is start/finish only.
        # (Anima Studio's Monitor tab already reads sample images and
        # checkpoints from output_dir directly for the vendored-backend
        # path too, so this matches that, not a regression from it.)
        trainer = anima_train_network.AnimaNetworkTrainer()
        trainer.train(ns)

        on_progress(step=ns.max_train_steps, total_steps=ns.max_train_steps, loss=None)
