"""FastAPI server for Puffer visualization."""

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from bin_factory.convert.types import as_json_dict


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
    base = _require_scenario_dir().resolve()
    path = (base / filename).resolve()
    if path.parent != base or path.suffix != ".bin":
        raise HTTPException(status_code=400, detail="Invalid scenario filename")
    return path


@app.get("/api/types")
def get_types():
    return as_json_dict()


@app.get("/api/scenarios")
def list_scenarios():
    scenario_dir = _require_scenario_dir()
    files = sorted(scenario_dir.glob("*.bin"), key=lambda p: p.name)
    return [p.name for p in files]


@app.get("/api/scenario/{filename}")
def get_scenario(filename: str):
    path = _resolve_scenario_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Scenario not found")
    return FileResponse(path, media_type="application/octet-stream")


def main():
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
