"""FastAPI server for Puffer visualization."""

import argparse
import asyncio
import re
import shutil
import tempfile
from pathlib import Path


try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.background import BackgroundTask
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Install viz dependencies first: uv sync --extra viz") from exc

from viz.utils import as_json_dict


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.add_middleware(NoCacheStaticMiddleware)

SCENARIO_DIR: Path | None = None


def _require_scenario_dir() -> Path:
    if SCENARIO_DIR is None:
        raise HTTPException(status_code=500, detail="Scenario directory is not configured")
    return SCENARIO_DIR


def _resolve_scenario_path(filename: str) -> Path:
    base = _require_scenario_dir()
    path = (base / filename).resolve()
    if not path.is_relative_to(base) or path.suffix != ".bin":
        raise HTTPException(status_code=400, detail="Invalid scenario filename")
    return path


def _input_suffix(content_type: str) -> str:
    lowered = content_type.lower()
    if "mp4" in lowered:
        return ".mp4"
    if "webm" in lowered:
        return ".webm"
    return ".dat"


def _sanitize_export_name(name: str) -> str:
    collapsed = re.sub(r"[\\/]+", "__", name)
    stem = re.sub(r"\.(bin|mp4)$", "", collapsed, flags=re.IGNORECASE)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{safe or 'scenario'}.mp4"


@app.get("/api/types")
def get_types():
    return as_json_dict()


@app.get("/api/scenarios")
def list_scenarios():
    scenario_dir = _require_scenario_dir()
    files = sorted(scenario_dir.rglob("*.bin"))
    return [str(p.relative_to(scenario_dir)) for p in files]


@app.get("/api/scenario/{filename:path}")
def get_scenario(filename: str):
    path = _resolve_scenario_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Scenario not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/api/export/mp4")
async def export_mp4(request: Request, name: str = "scenario"):
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg is not available on the server")

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Missing recording payload")

    tempdir = Path(tempfile.mkdtemp(prefix="viz-export-"))
    input_path = tempdir / f"input{_input_suffix(request.headers.get('content-type', ''))}"
    output_path = tempdir / "output.mp4"
    input_path.write_bytes(payload)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not output_path.exists():
        shutil.rmtree(tempdir, ignore_errors=True)
        detail = stderr.decode(errors="replace").strip() or "ffmpeg export failed"
        raise HTTPException(status_code=500, detail=detail)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=_sanitize_export_name(name),
        background=BackgroundTask(shutil.rmtree, tempdir, ignore_errors=True),
    )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="123Drive Server")
    parser.add_argument("--dir", required=True, help="Directory with .bin scenario files")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    args = parser.parse_args()

    global SCENARIO_DIR
    SCENARIO_DIR = Path(args.dir).resolve()
    if not SCENARIO_DIR.is_dir():
        raise ValueError(f"Not a directory: {SCENARIO_DIR}")

    static_dir = Path(__file__).parent / "web" / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    print(f"Serving scenarios from: {SCENARIO_DIR}")
    print(f"Starting server at http://localhost:{args.port}")
    uvicorn.run(app, host="localhost", port=args.port)


if __name__ == "__main__":
    main()
