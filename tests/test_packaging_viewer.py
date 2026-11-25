import asyncio
from pathlib import Path

import pytest


pytest.importorskip("fastapi")

from starlette.requests import ClientDisconnect

from viz import server


ROOT = Path(__file__).parent.parent
STATIC = ROOT / "src" / "viz" / "web" / "static"


def test_viewer_is_self_contained_and_handles_lane_zero():
    index = (STATIC / "index.html").read_text()
    css = (STATIC / "style.css").read_text()
    app = (STATIC / "app.js").read_text()

    assert "deck.gl-9.0.0.min.js" in index
    assert "http://" not in index
    assert "https://" not in index
    assert "@import" not in css
    assert "pf.source !== null" in app
    assert "pf.source === null" in app
    assert "1000 * state.scenario.metadata.dt / state.speed" in app
    assert (STATIC / "deck.gl-9.0.0.LICENSE.txt").is_file()


def _export_request(*headers, receive=None):
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "server": ("testserver", 80),
        "path": "/api/export/mp4",
        "query_string": b"",
        "headers": [(b"host", b"testserver"), *headers],
    }
    return server.Request(scope, receive) if receive else server.Request(scope)


def test_csp_and_export_media_limit(monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    with pytest.raises(server.HTTPException) as media_error:
        asyncio.run(server.export_mp4(_export_request((b"content-type", b"application/octet-stream"))))

    headers = [(b"content-type", b"video/webm"), (b"content-length", str(server.EXPORT_MAX_BYTES + 1).encode())]
    with pytest.raises(server.HTTPException) as size_error:
        asyncio.run(server.export_mp4(_export_request(*headers)))

    async def response_with_csp():
        middleware = server.NoCacheStaticMiddleware(server.app)
        return await middleware.dispatch(
            _export_request(*headers), lambda request: asyncio.sleep(0, result=server.Response())
        )

    response = asyncio.run(response_with_csp())
    assert "object-src 'none'" in response.headers["content-security-policy"]
    assert media_error.value.status_code == 415
    assert size_error.value.status_code == 413


def test_export_cleans_up_disconnected_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    tempdir = tmp_path / "export"

    def make_tempdir(prefix):
        tempdir.mkdir()
        return str(tempdir)

    async def disconnect():
        return {"type": "http.disconnect"}

    monkeypatch.setattr(server.tempfile, "mkdtemp", make_tempdir)

    with pytest.raises(ClientDisconnect):
        asyncio.run(server.export_mp4(_export_request((b"content-type", b"video/webm"), receive=disconnect)))

    assert not tempdir.exists()


def test_docker_context_is_selective():
    dockerfile = (ROOT / "docker_tools" / "dockerfiles" / "123drive.Dockerfile").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text()

    assert "COPY . " not in dockerfile
    assert "COPY src ./src" in dockerfile
    assert dockerignore.startswith("**\n")
    assert "!src/**" in dockerignore
    assert "!docker_tools/gcp" not in dockerignore
