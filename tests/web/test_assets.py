"""Versioned URLs of the GUI's static files (prodigy/web/assets.py): an upgrade loads every ES module anew."""
import re
from pathlib import Path

import pytest

from prodigy.web import assets
from prodigy.web.app import create_app

from tests.web.test_app import HOST

STATIC = Path(__file__).parents[2] / "prodigy" / "web" / "static"
IMPORT = re.compile(r'^\s*(?:import|export)\b[^"\';]*?\bfrom\s*["\']([^"\']+)["\']|^\s*import\s*["\']([^"\']+)["\']'
                    r'|\bimport\(\s*["\']([^"\']+)["\']', re.M)


@pytest.fixture(name="client")
def fixture_client():
    app = create_app(max_jobs=1)
    with app.test_client() as client:
        yield client
    app.extensions["prodigy_jobs"].shutdown()


def page_urls(client):
    page = client.get("/", headers=HOST).get_data(as_text=True)
    return re.findall(r'(?:href|src)="([^"]+)"', page)


def test_the_page_loads_every_file_under_the_current_token(client):
    """Scripts, styles and the icon come from /assets/<token>/; no file is named without the token."""
    urls = page_urls(client)
    assert urls and all(url.startswith("/assets/") for url in urls), urls
    tokens = {url.split("/")[2] for url in urls}
    assert len(tokens) == 1
    assert "/assets/{}/app/main.js".format(*tokens) in urls
    assert not any("?v=" in url for url in urls)
    response = client.get("/", headers=HOST)
    assert response.headers["Cache-Control"] == "no-cache"


def test_every_module_imports_the_others_relatively():
    """A relative import resolves under the token of main.js, so every module inherits it; an absolute
    path or a bare name would bypass it."""
    sources = list((STATIC / "app").rglob("*.js"))
    assert len(sources) > 30
    for source in sources:
        for match in IMPORT.finditer(source.read_text(encoding="utf-8")):
            specifier = next(group for group in match.groups() if group)
            assert specifier.startswith(("./", "../")), (source.name, specifier)


def test_an_upgrade_gives_every_file_a_new_url(tmp_path):
    """The token changes with the version and with any added, removed or edited file."""
    (tmp_path / "app").mkdir()
    module = tmp_path / "app" / "main.js"
    module.write_text("export const a = 1;\n", encoding="utf-8")
    token = assets.asset_token(str(tmp_path), "0.7.0")
    assert token == assets.asset_token(str(tmp_path), "0.7.0")
    assert token != assets.asset_token(str(tmp_path), "0.7.1")
    module.write_text("export const a = 22;\n", encoding="utf-8")
    edited = assets.asset_token(str(tmp_path), "0.7.0")
    assert edited != token
    (tmp_path / "app" / "parts.js").write_text("", encoding="utf-8")
    added = assets.asset_token(str(tmp_path), "0.7.0")
    assert added not in (token, edited)


def test_a_new_version_names_every_module_anew(monkeypatch):
    """Two versions of the server name main.js (and through it every module) under different URLs."""
    def main_url(version):
        monkeypatch.setattr("prodigy.analysis.result.prodigy_version", lambda: version)
        app = create_app(max_jobs=1)
        try:
            with app.test_client() as client:
                return next(url for url in page_urls(client) if url.endswith("/app/main.js"))
        finally:
            app.extensions["prodigy_jobs"].shutdown()

    assert main_url("0.7.0") != main_url("0.7.1")


def test_files_under_the_current_token_are_immutable_and_others_are_revalidated(client):
    main = next(url for url in page_urls(client) if url.endswith("/app/main.js"))
    response = client.get(main, headers=HOST)
    assert response.status_code == 200 and "immutable" in response.headers["Cache-Control"]
    response.close()
    # an import of main.js under the same token
    response = client.get(main.replace("/app/main.js", "/app/components/format.js"), headers=HOST)
    assert response.status_code == 200 and "immutable" in response.headers["Cache-Control"]
    response.close()
    # a page still open from before an upgrade: the current file, not cached under the old token
    response = client.get("/assets/0000000000000000/app/main.js", headers=HOST)
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-cache"
    response.close()
    # nothing outside the static folder
    response = client.get(main.replace("/app/main.js", "/..%2F..%2F..%2Fpyproject.toml"), headers=HOST)
    assert response.status_code == 404
    response.close()
