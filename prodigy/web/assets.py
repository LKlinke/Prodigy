"""Versioned URLs for the GUI's static files, so that an upgrade loads every ES module anew.

The page loads its scripts and styles from ``/assets/<token>/<path>`` instead of ``/static/<path>``.
The token changes with Prodigy's version and with the content of ``static`` (paths, sizes and
modification times of its files), so a new release or an edited file gives every file a new URL. The
ES modules import each other with relative paths (``./setup.js``, ``../vendor/codemirror.js``), so
they inherit the token of ``main.js`` without a build step or an import map; ``tests/web`` checks that
every import is relative.

Files under the current token never change, so the browser may keep them (``immutable``). A request
with another token (a page that is still open from before an upgrade) gets the current file with
``no-cache``, so that nothing is cached under a token that does not match its content. The page itself
is revalidated on every load (``no-cache``), so a reload always sees the current token.
"""
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from flask import Flask, send_from_directory, url_for

#: How long a token is reused before the files are looked at again, in seconds: one page load asks
#: for some 50 files, which need not walk the folder each.
TOKEN_SECONDS = 1.0
IMMUTABLE = "public, max-age=31536000, immutable"
REVALIDATE = "no-cache"


def asset_token(static_folder: str, version: str) -> str:
    """A short token for `version` and the files under `static_folder` (relative path, size and
    modification time of each): it changes with a new version and with any added, removed or edited
    file."""
    digest = hashlib.sha256(version.encode())
    root = Path(static_folder)
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        for name in sorted(files):
            path = Path(directory, name)
            stat = path.stat()
            digest.update(f"\0{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def _cached_token(static_folder: str, version: str) -> Callable[[], str]:
    """`asset_token`, recomputed at most every TOKEN_SECONDS."""
    cache: Dict[str, Tuple[float, str]] = {}

    def current() -> str:
        now = time.monotonic()
        last = cache.get("token")
        if last is None or now - last[0] > TOKEN_SECONDS:
            last = (now, asset_token(static_folder, version))
            cache["token"] = last
        return last[1]

    return current


def register_assets(app: Flask, version: str) -> None:
    """The route ``/assets/<token>/<path>`` and the template function ``asset(path)`` that names a
    static file under the current token."""
    static_folder = app.static_folder
    if static_folder is None:
        raise ValueError("The app has no static folder.")
    current = _cached_token(static_folder, version)

    @app.get("/assets/<token>/<path:filename>")
    def asset(token: str, filename: str) -> Any:
        response = send_from_directory(static_folder, filename)
        response.headers["Cache-Control"] = IMMUTABLE if token == current() else REVALIDATE
        return response

    @app.context_processor
    def _asset_url() -> Dict[str, Callable[[str], str]]:
        token = current()
        return {"asset": lambda filename: url_for("asset", token=token, filename=filename)}
