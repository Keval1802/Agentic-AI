"""
FastAPI Web Application - Production-grade game development API.
Enhanced with validation status, game history, and quality metrics.
"""

import sys
import asyncio

# FORCE PROACTOR LOOP FOR WINDOWS (Fixes Playwright NotImplementedError)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from queue import Queue
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .chains.game_chain import GameDevelopmentChain, PipelineStep

load_dotenv()

# Per-session progress tracking for SSE (isolates concurrent users)
progress_sessions: Dict[str, Queue] = {}
progress_lock = Lock()

def make_progress_callback(session_id: str):
    """Create a per-session progress callback."""
    def on_progress(step: PipelineStep):
        with progress_lock:
            if session_id in progress_sessions:
                progress_sessions[session_id].put({
                    "name": step.name,
                    "status": step.status,
                    "message": step.message
                })
    return on_progress

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI Game Developer",
    description="Production-grade AI game development using multi-agent architecture",
    version="2.0.0"
)

# CORS middleware - restrict origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
GENERATED_DIR = STATIC_DIR / "generated"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize chain (progress callback set per-request to isolate sessions)
chain = GameDevelopmentChain(
    include_design_step=True, 
    include_validation=True, 
    on_progress=None,
    include_visual_testing=True
)


# ============ REQUEST/RESPONSE MODELS ============

class GameRequest(BaseModel):
    """Enhanced request model for game generation."""
    prompt: str = Field(..., min_length=5, max_length=2000, description="Game description")
    include_design: bool = Field(True, description="Include design phase")
    include_validation: bool = Field(True, description="Validate output")
    include_visual_testing: bool = Field(True, description="Run visual tests in browser")
    quick_mode: bool = Field(False, description="Skip design and validation")
    session_id: str = Field("", description="Client-generated session ID for progress tracking")


class ValidationInfo(BaseModel):
    """Validation details."""
    is_valid: bool
    score: int
    issue_count: int
    summary: str


class VisualTestInfo(BaseModel):
    """Visual test results."""
    passed: bool
    score: int
    issues: List[dict]
    summary: str



class GameResponse(BaseModel):
    """Enhanced response with validation and metrics."""
    success: bool
    game_url: Optional[str] = None
    game_id: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None
    generation_time: Optional[float] = None
    quality_score: Optional[int] = None
    validation: Optional[ValidationInfo] = None
    visual_test: Optional[VisualTestInfo] = None
    steps_completed: List[str] = []


class PlanRequest(BaseModel):
    prompt: str


class PlanResponse(BaseModel):
    success: bool
    plan: Optional[str] = None
    error: Optional[str] = None


# ============ API ENDPOINTS ============

def _validate_game_id(game_id: str) -> str:
    """Validate game_id to prevent path traversal attacks."""
    import re as _re
    if not _re.match(r'^[a-zA-Z0-9_-]+$', game_id):
        raise HTTPException(status_code=400, detail="Invalid game ID")
    return game_id


def _safe_game_path(game_id: str) -> Path:
    """Resolve game path and verify it stays within GENERATED_DIR."""
    _validate_game_id(game_id)
    resolved = (GENERATED_DIR / game_id).resolve()
    if not str(resolved).startswith(str(GENERATED_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    return resolved


@app.get("/api/progress")
async def progress_stream(session_id: str = ""):
    """SSE endpoint for real-time pipeline progress updates (per-session)."""
    import time as _time
    import asyncio as _asyncio

    # Ensure concurrent access doesn't create race conditions
    with progress_lock:
        queue = progress_sessions.get(session_id)
        if not queue:
            # Create a new queue and REGISTER it so generate_game can use it
            queue = Queue()
            progress_sessions[session_id] = queue
            print(f"📡 New SSE connection: Registered session {session_id}")
        else:
            print(f"📡 Reconnecting SSE: Found existing session {session_id}")

    async def event_generator():
        start = _time.time()
        timeout = 600  # 10 minute max
        try:
            while _time.time() - start < timeout:
                # Use a non-blocking check
                has_event = False
                while not queue.empty():
                    event = queue.get_nowait()
                    has_event = True
                    yield f"data: {json.dumps(event)}\n\n"
                    # Check for completion signals
                    if isinstance(event, dict): 
                        if event.get('name') == 'Complete' or event.get('status') == 'failed':
                            return
                    elif hasattr(event, 'name'): # Handle PipelineStep object
                        if event.name == 'Complete' or event.status == 'failed':
                            return

                if not has_event:
                    await _asyncio.sleep(0.1)
                    
            yield f"data: {json.dumps({'name': 'Timeout', 'status': 'failed', 'message': 'SSE timeout'})}\n\n"
        except Exception as e:
            print(f"⚠️ SSE error for {session_id}: {e}")
        finally:
            # Only remove if WE created it and it's done? 
            # Actually, generate_game handles cleanup mostly, but we should be safe.
            # We don't remove it here because generate_game might still be writing?
            # But if client disconnects, we should probably leave it for a bit or rely on generate_game to finish.
            # If we pop it here, generate_game might crash writing to None?
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate", response_model=GameResponse)
async def generate_game(request: GameRequest):
    """
    Generate a complete game with the multi-agent pipeline.
    
    Pipeline: Plan → Design → Code → Validate
    """
    try:
        # Use client-provided session_id or generate one
        session_id = request.session_id or uuid.uuid4().hex
        
        # Check if SSE endpoint already created a queue for this session
        with progress_lock:
            session_queue = progress_sessions.get(session_id)
            if not session_queue:
                session_queue = Queue()
                progress_sessions[session_id] = session_queue
                print(f"🎮 Generator created new session: {session_id}")
            else:
                print(f"🔗 Generator attached to existing session: {session_id}")
        progress_callback = make_progress_callback(session_id)

        # Thread-safe: set the progress callback and per-request config
        chain.on_progress = progress_callback
        # Use per-request configuration instead of mutating shared state
        original_design = chain.include_design_step
        original_validation = chain.include_validation
        original_visual = chain.include_visual_testing
        try:
            chain.include_design_step = request.include_design
            chain.include_validation = request.include_validation
            chain.include_visual_testing = request.include_visual_testing

            # Run generation in background thread to allow SSE to stream
            if request.quick_mode:
                result = await asyncio.to_thread(chain.run_quick, request.prompt)
            else:
                result = await asyncio.to_thread(chain.run, request.prompt)
        finally:
            # Restore shared state after request completes
            chain.include_design_step = original_design
            chain.include_validation = original_validation
            chain.include_visual_testing = original_visual
            chain.on_progress = None
        
        if not result.success:
            return GameResponse(
                success=False,
                error=result.error or "Failed to generate game",
                steps_completed=result.steps_completed
            )
        
        # Create game folder with 3 separate files
        game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        game_folder = GENERATED_DIR / game_id
        game_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if multi-file output is available
        if result.game_files and result.game_files.get('html'):
            # Save 3 separate files
            with open(game_folder / "index.html", "w", encoding="utf-8") as f:
                f.write(result.game_files['html'])
            with open(game_folder / "style.css", "w", encoding="utf-8") as f:
                f.write(result.game_files.get('css', '/* No styles */'))
            with open(game_folder / "script.js", "w", encoding="utf-8") as f:
                f.write(result.game_files.get('js', '// No script'))
            print(f"📁 Saved game to folder: {game_folder}")
        else:
            # Fallback: save combined HTML as before
            with open(game_folder / "index.html", "w", encoding="utf-8") as f:
                f.write(result.game_code)
            print(f"📁 Saved combined HTML to: {game_folder}/index.html")
        
        # Build validation info
        validation_info = None
        if result.validation:
            validation_info = ValidationInfo(
                is_valid=result.validation.is_valid,
                score=result.validation.score,
                issue_count=len(result.validation.issues),
                summary=result.validation.summary
            )
        # Run Visual Testing (if enabled and validation passed)
        visual_info = None
        if request.include_visual_testing and chain.visual_tester and result.success:
            try:
                # Infer game type from prompt for specific tests
                game_type = "custom"
                prompt_lower = request.prompt.lower()
                if "chess" in prompt_lower: game_type = "chess"
                elif "snake" in prompt_lower: game_type = "snake"
                elif "tic" in prompt_lower: game_type = "tictactoe"
                elif "pong" in prompt_lower: game_type = "pong"
                elif "breakout" in prompt_lower: game_type = "breakout"
                
                # Notify UI
                progress_callback(PipelineStep(name="Visual Testing", status="running", message="Running browser tests..."))
                
                # Run the test
                index_path = str(game_folder / "index.html")
                visual_result = await chain.visual_tester.test_game_async(index_path, game_type)
                
                visual_info = VisualTestInfo(
                    passed=visual_result.passed,
                    score=visual_result.score,
                    issues=visual_result.issues,
                    summary=visual_result.summary
                )
                
                progress_callback(PipelineStep(
                    name="Visual Testing", 
                    status="completed" if visual_result.passed else "failed",
                    message=f"Visual Score: {visual_result.score}/100"
                ))
            except Exception as e:
                print(f"⚠️ Visual testing failed: {e}")
                progress_callback(PipelineStep(name="Visual Testing", status="failed", message=f"Error: {str(e)[:50]}"))
        
        return GameResponse(
            success=True,
            game_url=f"/static/generated/{game_id}/index.html",
            game_id=game_id,
            title=result.metadata.get("title", "Generated Game"),
            generation_time=round(result.generation_time, 2),
            quality_score=result.metadata.get("quality_score"),
            validation=validation_info,
            visual_test=visual_info,
            steps_completed=result.steps_completed
        )
        
    except Exception as e:
        return GameResponse(
            success=False,
            error=f"Error generating game: {str(e)}"
        )


@app.post("/api/plan", response_model=PlanResponse)
async def generate_plan(request: PlanRequest):
    """Generate only a game plan."""
    try:
        plan = chain.plan_only(request.prompt)
        return PlanResponse(success=True, plan=plan)
    except Exception as e:
        return PlanResponse(success=False, error=str(e))


@app.get("/api/games")
async def list_games():
    """List all generated games with metadata."""
    games = []
    # Check folder-based games (new format: game_id/index.html)
    for folder in GENERATED_DIR.iterdir():
        if folder.is_dir():
            index_file = folder / "index.html"
            if index_file.exists():
                stat = folder.stat()
                games.append({
                    "id": folder.name,
                    "filename": "index.html",
                    "url": f"/static/generated/{folder.name}/index.html",
                    "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size_kb": round(sum(f.stat().st_size for f in folder.iterdir() if f.is_file()) / 1024, 1)
                })
    # Also check legacy single-file games (old format: game_id.html)
    for file in GENERATED_DIR.glob("*.html"):
        stat = file.stat()
        games.append({
            "id": file.stem,
            "filename": file.name,
            "url": f"/static/generated/{file.name}",
            "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size_kb": round(stat.st_size / 1024, 1)
        })
    return {
        "games": sorted(games, key=lambda x: x["created"], reverse=True),
        "total": len(games)
    }


@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    """Get game details by ID."""
    safe_path = _safe_game_path(game_id)
    
    # Check folder-based game first
    if safe_path.is_dir():
        index_file = safe_path / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Game not found")
        content = index_file.read_text(encoding="utf-8")
        return {
            "id": game_id,
            "url": f"/static/generated/{game_id}/index.html",
            "content": content,
            "size_kb": round(len(content.encode()) / 1024, 1)
        }
    
    # Fallback: legacy single-file
    filepath = GENERATED_DIR / f"{game_id}.html"
    filepath = filepath.resolve()
    if not str(filepath).startswith(str(GENERATED_DIR.resolve())) or not filepath.exists():
        raise HTTPException(status_code=404, detail="Game not found")
    
    content = filepath.read_text(encoding="utf-8")
    return {
        "id": game_id,
        "url": f"/static/generated/{game_id}.html",
        "content": content,
        "size_kb": round(len(content.encode()) / 1024, 1)
    }


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """Delete a generated game."""
    import shutil
    safe_path = _safe_game_path(game_id)
    
    # Check folder-based game
    if safe_path.is_dir():
        shutil.rmtree(safe_path)
        return {"success": True, "message": f"Game {game_id} deleted"}
    
    # Fallback: legacy single-file
    filepath = GENERATED_DIR / f"{game_id}.html"
    filepath = filepath.resolve()
    if not str(filepath).startswith(str(GENERATED_DIR.resolve())) or not filepath.exists():
        raise HTTPException(status_code=404, detail="Game not found")
    
    filepath.unlink()
    return {"success": True, "message": f"Game {game_id} deleted"}


@app.get("/health")
async def health_check():
    """Health check with system status."""
    # Count both folder-based games (new) and legacy single-file games
    folder_count = sum(1 for d in GENERATED_DIR.iterdir() if d.is_dir() and (d / "index.html").exists())
    legacy_count = len(list(GENERATED_DIR.glob("*.html")))
    game_count = folder_count + legacy_count
    return {
        "status": "healthy",
        "version": "2.0.0",
        "llm_provider": "Multi-Provider (NVIDIA NIM / Groq / Gemini)",
        "api_key_configured": bool(os.getenv("NVIDIA_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "games_generated": game_count,
        "features": {
            "validation": True,
            "design_system": True,
            "retry_logic": True,
            "model_fallback": True,
            "visual_testing": True
        }
    }


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
