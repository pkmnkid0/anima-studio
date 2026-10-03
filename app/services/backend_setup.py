"""
In-app installer for the venv Anima Studio's native backend
(native_backend/) runs training with. The bundled sd_scripts library at
training_backend/sd_scripts/ (see THIRD_PARTY_NOTICES.md - vendored
from 67372a/LoRA_Easy_Training_Scripts, not written by Anima Studio)
ships with Anima Studio already; what's still a separate step is
installing dependencies into its venv - first that library's own
(PyTorch, xformers, etc, via its own installer, a multi-GB download
that has to match your GPU/CUDA setup), then native_backend's own
SDXL-path dependencies (diffusers, peft, etc) into that same venv
right after, so native_backend/server.py has everything it needs for
both training paths in one place. This module runs both of those steps
from a button instead of a terminal, with output streamed live.

The sd_scripts installer itself is interactive (it can ask things like
"are you using this remotely? (y/n)" - auto-answered below, see
_run_interactive - and depending on the machine, follow-up questions
about CUDA/xformers versions) - this deliberately does not try to
script past unknown ones blind, since guessing wrong is worse than
asking: the log below is a real (very small) terminal, and whatever it
asks beyond what's auto-handled, type the answer in the box and click
Send. The native_backend dependency install that runs afterward is
plain and non-interactive, so it just runs straight through.

The input box only marks itself "ready" once the installer has gone
quiet for a bit while still running - not just "the process is alive" -
since that's the whole runtime of a multi-minute install. Without that,
someone reasonably sends an answer to what they *see* on screen, but if
nothing's actually blocked on a read() right then, that input just sits
in the pipe buffer and gets silently fed to whatever the *next* real
prompt turns out to be - answering a question they never got to read.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import config

_MAX_LOG_LINES = 800
_IDLE_SECONDS_BEFORE_LIKELY_WAITING = 2.5

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_stage = "idle"  # idle | installing | done | error
_log_lines: list[str] = []
_target_dir: str | None = None
_error: str | None = None
_last_output_at: float = 0.0


class SetupError(RuntimeError):
    pass


def bundled_backend_dir() -> Path:
    """Where the vendored training_backend/ ships, right next to this
    app's own code - the default (and normally only) install target."""
    return config.APP_DIR / "training_backend"


def is_bundled() -> bool:
    return (bundled_backend_dir() / "main.py").exists()


def is_installed() -> bool:
    """Whether the bundled backend's own dependencies have actually been
    installed - checked by looking for the venv its installer creates,
    not just whether *this process* has run an install this session.
    Without this, restarting Anima Studio after a previous, successful
    install would show "idle"/"not started" again, even though it's
    already done."""
    return (bundled_backend_dir() / "sd_scripts" / "venv").exists()


def status() -> dict:
    with _lock:
        running = _proc is not None and _proc.poll() is None
        idle_for = time.time() - _last_output_at if running else 0.0
        return {
            "stage": _stage,
            "installed": is_installed(),
            "waiting_for_input": running and idle_for > _IDLE_SECONDS_BEFORE_LIKELY_WAITING,
            "target_dir": _target_dir,
            "error": _error,
            "log_tail": list(_log_lines[-200:]),
            "bundled": is_bundled(),
            "bundled_dir": str(bundled_backend_dir()),
        }


def _append(line: str) -> None:
    global _last_output_at
    with _lock:
        _log_lines.append(line.rstrip("\n"))
        if len(_log_lines) > _MAX_LOG_LINES:
            del _log_lines[: len(_log_lines) - _MAX_LOG_LINES]
        _last_output_at = time.time()


def _set_stage(stage: str, error: str | None = None) -> None:
    global _stage, _error
    with _lock:
        _stage = stage
        _error = error


def is_busy() -> bool:
    with _lock:
        return _stage == "installing"


def maybe_autostart_install() -> None:
    """Called once when Anima Studio itself starts up (see app/__init__.py):
    if the bundled backend's source is here but its dependencies have
    never been installed, kick off that install automatically in the
    background - so by the time someone gets to the Train tab, it's
    often already done (or visibly in progress there), instead of
    needing to separately notice and click "Install dependencies" first.

    Fire-and-forget, same as backend_manager.maybe_autostart - never
    raises, never blocks app startup, and does nothing at all if the
    backend isn't bundled, is already installed, or is already running
    (e.g. this got called twice)."""
    if not is_bundled() or is_installed() or is_busy():
        return
    try:
        start()
    except SetupError:
        pass  # best-effort - the Train tab's card still works manually either way


def start(target_dir: str = "") -> dict:
    """Runs the bundled backend's own installer (dependencies only - the
    training code itself is already here). `target_dir` is only used as
    a fallback for the rare case someone points this at a different
    checkout instead of the bundled one."""
    global _target_dir
    if is_busy():
        raise SetupError("Setup is already running - check the log below, or wait for it to finish.")

    target = Path(target_dir).expanduser() if target_dir else bundled_backend_dir()
    if not target.exists():
        raise SetupError(f"'{target}' doesn't exist.")

    with _lock:
        _log_lines.clear()
        global _last_output_at
        _last_output_at = time.time()
    _target_dir = str(target)
    _set_stage("installing")

    threading.Thread(target=_run_install, args=(target,), daemon=True).start()
    return status()


def _run_install(target: Path) -> None:
    try:
        installer = _find_installer(target)
        if installer is None:
            _append(f"Couldn't find an install script (install.sh/install.bat) in {target}.")
            _set_stage("error", "No installer script found.")
            return

        _append(f"Running {installer.name} in {target} - this downloads PyTorch and the rest")
        _append("of the ML stack, so it can take a while.")
        _append("")
        _run_interactive(installer, target)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user via status()
        _append(f"Unexpected error: {exc}")
        _set_stage("error", str(exc))


def _find_installer(target: Path):
    name = "install.bat" if sys.platform == "win32" else "install.sh"
    candidate = target / name
    return candidate if candidate.exists() else None


def _run_interactive(script: Path, cwd: Path) -> None:
    global _proc
    args = ["cmd", "/c", str(script)] if sys.platform == "win32" else ["bash", str(script)]
    proc = subprocess.Popen(
        args, cwd=str(cwd),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        # No visible console window for this child process (see the same
        # fix in backend_manager.py) - doesn't affect the piped stdin/
        # stdout this function relies on to feed it answers.
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    with _lock:
        _proc = proc

    try:
        # The installer's first question is "are you using this remotely?
        # (y/n)" - answering "y" means "yes, remote", which leads to a
        # different set of follow-up questions (ngrok/cloudflared tunnels)
        # that don't apply here, since Anima Studio always runs locally.
        # Auto-answering this one specific, always-the-same-answer question
        # avoids relying on someone reading the literal wording correctly
        # ("are you remote" reads easy to answer backwards if you're
        # thinking "yes, I'm local").
        proc.stdin.write("n\n")
        proc.stdin.flush()
        _append('(Auto-answered "are you using this remotely?" with "n" - Anima Studio')
        _append("always runs locally. If it asks anything else, use the box below.)")
        _append("")
    except OSError:
        pass  # if this fails, the manual input box below still covers it

    def _reader() -> None:
        global _proc
        try:
            for line in proc.stdout:
                _append(line)
        except Exception:  # noqa: BLE001 - the process going away mid-read is normal
            pass
        finally:
            code = proc.poll()
            with _lock:
                _proc = None
            if code == 0:
                _install_native_backend_deps(cwd)
                _append("")
                _append("Installer finished. 'Backend folder' below is already set to this")
                _append("folder - click Launch backend, then Check connection.")
                _set_stage("done")
            else:
                _set_stage("error", f"Installer exited with code {code} - scroll up to see why.")

    threading.Thread(target=_reader, daemon=True).start()


def _install_native_backend_deps(sd_scripts_dir: Path) -> None:
    """After the interactive installer above finishes setting up
    sd_scripts' own venv, also installs native_backend's SDXL-path
    dependencies (diffusers/peft/etc) into that SAME venv - so one venv
    covers both training paths native_backend/server.py can run,
    instead of needing a second, separate environment just for SDXL."""
    venv_python = sd_scripts_dir / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    requirements = bundled_backend_dir().parent / "native_backend" / "requirements.txt"
    if not venv_python.exists() or not requirements.exists():
        return  # non-fatal - SDXL training just won't work until this is sorted out manually
    _append("")
    _append("Installing native_backend's own SDXL-path dependencies (diffusers, peft, ...)")
    _append("into the same virtual environment...")
    try:
        proc = subprocess.Popen(
            [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        for line in proc.stdout:
            _append(line)
        proc.wait()
    except OSError as exc:
        _append(f"Couldn't install native_backend's dependencies automatically: {exc}")
        _append(f"You can run this yourself: {venv_python} -m pip install -r {requirements}")


def send_input(text: str) -> dict:
    with _lock:
        proc = _proc
    if proc is None or proc.poll() is not None or proc.stdin is None:
        raise SetupError("Nothing is waiting for input right now.")
    try:
        proc.stdin.write(text.rstrip("\n") + "\n")
        proc.stdin.flush()
    except OSError as exc:
        raise SetupError(f"Couldn't send that: {exc}") from exc
    _append(f"> {text}")
    return status()
