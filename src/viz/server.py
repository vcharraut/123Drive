"""FastAPI server for Puffer visualization."""

import argparse
from pathlib import Path
from threading import Lock

import numpy as np
import orjson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.viz.binary_loader import load_puffer_binary
from src.viz.web.utils import build_lane_map, compute_route_polyline


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.add_middleware(NoCacheStaticMiddleware)

SCENARIO_DIR: Path = None
_SCENARIO_CACHE: dict[str, tuple[int, int, bytes]] = {}
_SCENARIO_CACHE_LOCK = Lock()


def _arr(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def serialize_scenario(data: dict) -> dict:
    agents = []
    road_elements = data.get("road_map_elements", [])
    lane_map = build_lane_map(road_elements)

    for agent in data.get("agents", []):
        states = agent.get("states", {})
        xyz = states.get("xyz", np.array([]))
        start_pos = xyz[0] if len(xyz) > 0 else None

        routes = agent.get("routes", [])
        route_polyline = None
        if routes:
            pts = compute_route_polyline(routes[0], lane_map, start_pos)
            if pts is not None:
                route_polyline = pts.tolist()

        agents.append(
            {
                "id": agent["id"],
                "type": agent["type"],
                "xyz": _arr(xyz),
                "heading": _arr(states.get("heading")),
                "velocity": _arr(states.get("velocity")),
                "length": _arr(states.get("length")),
                "width": _arr(states.get("width")),
                "height": _arr(states.get("height")),
                "valid": _arr(states.get("valid")),
                "routes": agent.get("routes", []),
                "route_polyline": route_polyline,
            }
        )

    roads = []
    for elem in road_elements:
        xyz = elem.get("xyz")
        roads.append(
            {
                "id": elem["id"],
                "type": elem["type"],
                "xyz": _arr(xyz),
                "entry_lanes": elem.get("entry_lanes", []),
                "exit_lanes": elem.get("exit_lanes", []),
                "speed_limit": elem.get("speed_limit", 0.0),
            }
        )

    traffic = []
    for elem in data.get("traffic_control_elements", []):
        traffic.append(
            {
                "id": elem["id"],
                "type": elem["type"],
                "xyz": _arr(elem.get("xyz")),
                "states": elem.get("states", []),
                "controlled_lanes": elem.get("controlled_lanes", []),
            }
        )

    return {
        "scenario_id": data.get("scenario_id", ""),
        "agents": agents,
        "road_map_elements": roads,
        "traffic_control_elements": traffic,
        "metadata": data.get("metadata", {}),
    }


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


def _get_serialized_response_bytes(path: Path) -> bytes:
    stat = path.stat()
    cache_key = str(path)
    cache_token = (stat.st_mtime_ns, stat.st_size)
    with _SCENARIO_CACHE_LOCK:
        cached = _SCENARIO_CACHE.get(cache_key)
    if cached and cached[0] == cache_token[0] and cached[1] == cache_token[1]:
        return cached[2]

    data = load_puffer_binary(path)
    serialized = serialize_scenario(data)
    payload = orjson.dumps(serialized)

    with _SCENARIO_CACHE_LOCK:
        _SCENARIO_CACHE[cache_key] = (cache_token[0], cache_token[1], payload)
    return payload


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
    return Response(content=_get_serialized_response_bytes(path), media_type="application/json")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Puffer Viz Server")
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
