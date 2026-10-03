"""
Caption file operations for the tag editor: reading/writing individual
`.txt` captions, and bulk operations (find & replace, add tag, remove
tag, prepend trigger word, underscore -> space cleanup) across a whole
folder at once.
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import config
from .imaging import is_backup_file, read_text_robust


def _is_dataset_image(f: Path) -> bool:
    return f.suffix.lower() in config.IMAGE_EXTENSIONS and not is_backup_file(f)


def _caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def read_caption(image_path: str) -> str:
    p = _caption_path(Path(image_path))
    if p.exists():
        return read_text_robust(p)
    return ""


def write_caption(image_path: str, caption: str) -> None:
    p = _caption_path(Path(image_path))
    p.write_text(caption.strip(), encoding="utf-8")


def _split(caption: str) -> list[str]:
    # Comma-separated is this app's own convention, but some datasets
    # tagged by other tools write one tag per line instead - without
    # this, every bulk operation (add/remove tag, find & replace) would
    # see a whole newline-separated caption as a single unmatched "tag"
    # and silently no-op on it.
    return [t.strip() for t in re.split(r"[,\n]+", caption) if t.strip()]


def _join(tags: list[str]) -> str:
    return ", ".join(tags)


def add_tag_to_images(image_paths: list[str], tag: str, position: str = "end") -> int:
    """Add `tag` to exactly the given images (not the whole folder) -
    used to undo a Tag Frequency "exclude" click by restoring the tag
    to precisely the images it was removed from."""
    tag = tag.strip()
    if not tag:
        return 0
    changed = 0
    for image_path in image_paths:
        caption_path = _caption_path(Path(image_path))
        current = _split(read_text_robust(caption_path)) if caption_path.exists() else []
        if tag in current:
            continue
        current = [tag] + current if position == "start" else current + [tag]
        caption_path.write_text(_join(current), encoding="utf-8")
        changed += 1
    return changed


def bulk_add_tag(folder: str, tag: str, position: str = "end") -> int:
    """Add `tag` to every caption in the folder that doesn't already have
    it. position: "start" | "end". Returns the number of files changed."""
    tag = tag.strip()
    if not tag:
        return 0
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        current = _split(read_text_robust(caption_path)) if caption_path.exists() else []
        if tag in current:
            continue
        current = [tag] + current if position == "start" else current + [tag]
        caption_path.write_text(_join(current), encoding="utf-8")
        changed += 1
    return changed


def bulk_remove_tag(folder: str, tag: str) -> int:
    tag = tag.strip()
    if not tag:
        return 0
    return bulk_remove_tags(folder, [tag])


def bulk_remove_tags(folder: str, tags: list[str]) -> int:
    """Remove any number of tags from every caption in one pass - used
    by the Tag Frequency view's multi-select "exclude" action so
    excluding N tags costs one folder scan instead of N."""
    tag_set = {t.strip() for t in tags if t and t.strip()}
    if not tag_set:
        return 0
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        current = _split(read_text_robust(caption_path))
        new_tags = [t for t in current if t not in tag_set]
        if len(new_tags) != len(current):
            caption_path.write_text(_join(new_tags), encoding="utf-8")
            changed += 1
    return changed


def bulk_find_replace(folder: str, find: str, replace: str, whole_tag_only: bool = True) -> int:
    """If whole_tag_only, only replaces a tag that matches `find` exactly
    (renaming a tag). Otherwise does a substring replace within each
    caption's raw text."""
    find = find.strip()
    if not find:
        return 0
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        original = read_text_robust(caption_path)
        if whole_tag_only:
            tags = _split(original)
            if find not in tags:
                continue
            new_tags = [replace.strip() if t == find else t for t in tags]
            new_tags = [t for t in new_tags if t]  # allow "replace" empty to delete
            new_text = _join(new_tags)
        else:
            if find not in original:
                continue
            new_text = original.replace(find, replace)
        if new_text != original:
            caption_path.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def bulk_prepend_trigger(folder: str, trigger_word: str) -> int:
    return bulk_add_tag(folder, trigger_word, position="start")


def bulk_underscores_to_spaces(folder: str) -> int:
    changed = 0
    for f in Path(folder).iterdir():
        if not _is_dataset_image(f):
            continue
        caption_path = _caption_path(f)
        if not caption_path.exists():
            continue
        original = read_text_robust(caption_path)
        tags = _split(original)
        # keep emoticon-style tags with underscores intact, e.g. ">_<", "._.":
        new_tags = [t if t.replace("_", "") == "" else t.replace("_", " ") for t in tags]
        new_text = _join(new_tags)
        if new_text != original:
            caption_path.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def rebuild_from_sidecar(image_entry: dict, trigger_word: str = "",
                          tag_order: tuple[str, ...] = ("artist", "copyright", "character", "general", "meta")
                          ) -> str:
    """Rebuild a caption from an image's `.tags.json` sidecar (written at
    download time), e.g. after the user edits the tag-order preference."""
    from .booru import flatten_tags
    sidecar = image_entry.get("sidecar") or {}
    tags = flatten_tags(sidecar.get("tags", {}), tag_order)
    tags = [t.replace("_", " ") for t in tags]
    if trigger_word:
        tags = [trigger_word] + tags
    return _join(tags)
