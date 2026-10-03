from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB, generous for JSON payloads

    from .routes.pages import bp as pages_bp
    from .routes.collect import bp as collect_bp
    from .routes.caption import bp as caption_bp
    from .routes.train import bp as train_bp
    from .routes.files import bp as files_bp
    from .routes.edit import bp as edit_bp
    from .routes.monitor import bp as monitor_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(collect_bp, url_prefix="/api/collect")
    app.register_blueprint(caption_bp, url_prefix="/api/caption")
    app.register_blueprint(train_bp, url_prefix="/api/train")
    app.register_blueprint(files_bp, url_prefix="/api/files")
    app.register_blueprint(edit_bp, url_prefix="/api/edit")
    app.register_blueprint(monitor_bp, url_prefix="/api/monitor")

    @app.errorhandler(Exception)
    def handle_error(exc):  # noqa: ANN001
        # Keep unhandled errors as clean JSON instead of an HTML traceback
        # page - this app is consumed entirely by its own frontend's fetch()
        # calls, which expect JSON either way.
        from flask import jsonify, request
        if request.path.startswith("/api/"):
            app.logger.exception("Unhandled error on %s", request.path)
            return jsonify({"error": str(exc)}), 500
        raise exc

    import atexit
    from .services import backend_manager
    # Fire-and-forget: don't block app startup on a slow backend boot, and
    # don't crash app startup if the configured launch command is wrong -
    # the Train tab's launcher status/log will just show why it failed.
    import threading
    threading.Thread(target=backend_manager.maybe_autostart, daemon=True).start()
    atexit.register(backend_manager.stop)  # don't leave an orphaned backend process behind

    # NOTE: this app deliberately does NOT auto-install the bundled
    # training backend's dependencies (torch etc.) on startup - that's a
    # multi-GB, potentially multi-minute download, and doing it silently
    # the moment someone opens Anima Studio (even if all they want is
    # Collect/Caption/Edit) was judged more surprising than helpful.
    # Installing it is a deliberate, visible action: the "Set up backend"
    # card in the Train tab, on demand only.

    return app
