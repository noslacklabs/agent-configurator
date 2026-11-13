"""FastAPI app for the hand-off service (port 8003).

Run with ``python -m uvicorn agent_configurator.handoff.app:app --port 8003``
from the repository root, so ``.env`` and ``frontend/dist`` are found.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_configurator.handoff.service import HandoffError, HandoffService, Settings


class EvaluateRequest(BaseModel):
    include_agents: bool = True
    test: bool = True


def create_app(service: HandoffService | None = None, frontend_dir: Path | None = None) -> FastAPI:
    if service is None:
        load_dotenv(Path.cwd() / ".env")
        service = HandoffService(Settings.from_env())
    if frontend_dir is None:
        frontend_dir = Path(os.getenv("HANDOFF_FRONTEND_DIR", Path.cwd() / "frontend" / "dist"))

    app = FastAPI(title="Agent Configurator hand-off", version="0.1.0")
    # Only needed for `npm run dev`; the built page is served from this origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(HandoffError)
    async def handoff_error(_request, exc: HandoffError):
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.get("/api/health")
    async def health():
        return await service.health()

    @app.get("/api/flows")
    async def flows():
        return await service.list_flows()

    @app.post("/api/flows/{flow_id}/evaluate")
    async def evaluate(flow_id: str, request: EvaluateRequest | None = None):
        request = request or EvaluateRequest()
        return await service.evaluate(flow_id, include_agents=request.include_agents, test=request.test)

    if (frontend_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

        @app.get("/")
        async def index():
            return FileResponse(frontend_dir / "index.html")
    else:
        @app.get("/")
        async def index_missing():
            return JSONResponse({
                "detail": "The page is not built yet. Run `npm install` and `npm run build` in frontend/, "
                          "or use the API: GET /api/flows, POST /api/flows/{id}/evaluate.",
            })

    return app


app = create_app()
