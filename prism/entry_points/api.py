"""
PRISM API - Compute interface for ORTHON.

ORTHON commands → PRISM computes → ORTHON SQL

Endpoints:
    POST /compute     - Run discipline-specific computation
    GET  /health      - Status check
    GET  /files       - List available parquet files
    GET  /read/{file} - Read parquet as JSON
    GET  /disciplines - List available disciplines
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import io
import json
import subprocess
import sys
import os

app = FastAPI(title="PRISM", version="0.2.0", description="Compute engine for ORTHON")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class ComputeRequest(BaseModel):
    """Request from ORTHON to run computation."""
    discipline: str
    input_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None


class ComputeResponse(BaseModel):
    """Response after computation."""
    status: str
    job_id: str
    discipline: str
    output_path: Optional[str] = None
    message: Optional[str] = None
    timestamp: str


class JobStatus(BaseModel):
    """Status of a compute job."""
    job_id: str
    status: str  # pending, running, completed, failed
    discipline: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# State (in-memory for now, could be Redis/DB)
# =============================================================================

_jobs: Dict[str, JobStatus] = {}


# =============================================================================
# Helpers
# =============================================================================

def _get_data_dir() -> Path:
    """Get the PRISM data directory."""
    return Path(os.path.expanduser("~/prism-mac/data"))


def _get_inbox_dir() -> Path:
    """Get the PRISM inbox directory."""
    return Path(os.path.expanduser("~/prism-inbox"))


def _generate_job_id() -> str:
    """Generate a unique job ID."""
    import uuid
    return str(uuid.uuid4())[:8]


def _run_compute(job_id: str, discipline: str, input_path: str, config: Dict, output_dir: str):
    """Run computation in background."""
    _jobs[job_id].status = "running"
    _jobs[job_id].started_at = datetime.now().isoformat()

    try:
        # Write config to temp file
        config_path = _get_inbox_dir() / f"config_{job_id}.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump({"discipline": discipline, **config}, f)

        # Run PRISM compute
        cmd = [
            sys.executable, "-m", "prism.entry_points.compute",
            "--discipline", discipline,
            "--config", str(config_path),
        ]
        if input_path:
            cmd.extend(["--input", input_path])
        if output_dir:
            cmd.extend(["--output", output_dir])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent.parent))

        if result.returncode != 0:
            _jobs[job_id].status = "failed"
            _jobs[job_id].error = result.stderr
        else:
            _jobs[job_id].status = "completed"
            _jobs[job_id].output_path = output_dir or str(_get_data_dir())

        _jobs[job_id].completed_at = datetime.now().isoformat()

        # Write status for ORTHON
        status_file = _get_data_dir() / "job_status.json"
        with open(status_file, 'w') as f:
            json.dump(_jobs[job_id].dict(), f)

    except Exception as e:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)
        _jobs[job_id].completed_at = datetime.now().isoformat()


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    from prism import __version__
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/disciplines")
async def list_disciplines():
    """List available disciplines."""
    from prism.disciplines import DISCIPLINES
    return {
        "disciplines": list(DISCIPLINES.keys()),
        "details": {k: v.get("name", k) for k, v in DISCIPLINES.items()},
    }


@app.get("/disciplines/{discipline}")
async def get_discipline(discipline: str):
    """Get discipline requirements and engines."""
    from prism.disciplines import DISCIPLINES
    from prism.disciplines.requirements import get_requirements_text, check_requirements

    if discipline not in DISCIPLINES:
        raise HTTPException(404, f"Unknown discipline: {discipline}")

    return {
        "discipline": discipline,
        "info": DISCIPLINES[discipline],
        "requirements_text": get_requirements_text(discipline),
    }


@app.post("/compute", response_model=ComputeResponse)
async def compute(request: ComputeRequest, background_tasks: BackgroundTasks):
    """
    Run PRISM computation.

    ORTHON sends discipline + config, PRISM computes, returns job ID.
    Poll /jobs/{job_id} for status.
    """
    from prism.disciplines import DISCIPLINES

    # Validate discipline
    if request.discipline not in DISCIPLINES:
        raise HTTPException(400, f"Unknown discipline: {request.discipline}")

    # Create job
    job_id = _generate_job_id()
    _jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        discipline=request.discipline,
    )

    # Run in background
    background_tasks.add_task(
        _run_compute,
        job_id,
        request.discipline,
        request.input_path,
        request.config or {},
        request.output_dir,
    )

    return ComputeResponse(
        status="accepted",
        job_id=job_id,
        discipline=request.discipline,
        message="Computation started. Poll /jobs/{job_id} for status.",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    return _jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": list(_jobs.values())}


@app.get("/files")
async def list_files():
    """List available parquet files."""
    data_dir = _get_data_dir()
    files = {}
    for f in ['observations', 'data', 'vector', 'geometry', 'dynamics', 'physics']:
        path = data_dir / f"{f}.parquet"
        if path.exists():
            files[f] = {
                "exists": True,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        else:
            files[f] = {"exists": False}
    return files


@app.get("/read/{filename}")
async def read_file(filename: str, limit: int = 100, offset: int = 0):
    """Read a parquet file and return as JSON."""
    import polars as pl

    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    df = pl.read_parquet(path)
    return {
        "file": filename,
        "total_rows": len(df),
        "columns": df.columns,
        "offset": offset,
        "limit": limit,
        "data": df.slice(offset, limit).to_dicts(),
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a parquet file."""
    if not filename.endswith('.parquet'):
        filename = f"{filename}.parquet"

    path = _get_data_dir() / filename
    if not path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return StreamingResponse(
        io.BytesIO(path.read_bytes()),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/trigger-github")
async def trigger_github(request: ComputeRequest):
    """
    Trigger GitHub Actions workflow (alternative to local compute).

    Requires GITHUB_TOKEN env var.
    """
    import httpx

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise HTTPException(500, "GITHUB_TOKEN not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.github.com/repos/prism-engines/prism/dispatches",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={
                "event_type": "compute",
                "client_payload": {
                    "discipline": request.discipline,
                    "config_path": request.input_path,
                    "input_path": request.input_path,
                }
            }
        )

    if response.status_code != 204:
        raise HTTPException(response.status_code, f"GitHub API error: {response.text}")

    return {"status": "triggered", "message": "GitHub Actions workflow dispatched"}


# =============================================================================
# CLI
# =============================================================================

def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="PRISM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port (default: 8100)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    print(f"Starting PRISM API at http://{args.host}:{args.port}")
    print(f"Docs at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "prism.entry_points.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
