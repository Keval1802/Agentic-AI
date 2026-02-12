"""
Enhanced Game Development Chain - Production-grade orchestration.
Implements validation pipeline, retry logic, and quality gates.
"""

import sys
import os
import io
import time
import random
import json
import builtins

from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

from ..agents.planner import PlannerAgent
from ..agents.designer import DesignerAgent
from ..agents.coder import CoderAgent
from ..agents.validator import ValidatorAgent, ValidationResult
from ..agents.visual_tester import VisualTesterAgent, VisualTestResult
from ..parsers.game_parser import GameCodeParser
from ..utils.code_patcher import FunctionPatcher
from ..graphs.coder_validator_graph import CoderValidatorGraph

# Ensure UTF-8 output on Windows to avoid 'charmap' errors when printing emojis
try:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    # Always wrap to force UTF-8 on Windows consoles
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="backslashreplace", line_buffering=True)
except Exception:
    pass

def safe_print(*args, **kwargs):
    safe_args = []
    for a in args:
        s = str(a)
        s = s.encode("utf-8", "backslashreplace").decode("utf-8")
        safe_args.append(s)
    return builtins.print(*safe_args, **kwargs)

print = safe_print

load_dotenv()


@dataclass
class GameResult:
    """Comprehensive result of the game development pipeline."""
    game_code: str  # Combined HTML (for backwards compatibility)
    game_plan: str
    design_spec: Optional[str]
    metadata: Dict[str, Any]
    validation: Optional[ValidationResult]
    success: bool
    error: Optional[str] = None
    generation_time: float = 0.0
    steps_completed: list = field(default_factory=list)
    # Multi-file output
    game_files: Dict[str, str] = field(default_factory=dict)  # {'html': ..., 'css': ..., 'js': ...}
    game_folder: Optional[str] = None  # Folder path where files are saved
    visual_test: Optional[Any] = None  # VisualTestResult from browser testing


@dataclass 
class PipelineStep:
    """Represents a single step in the generation pipeline."""
    name: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    message: str = ""
    duration: float = 0.0


class GameDevelopmentChain:
    """
    Production-Grade Game Development Pipeline.
    
    Features:
    - Multi-agent orchestration (Planner → Designer → Coder → Validator)
    - Quality gates between each stage
    - Retry logic with exponential backoff for 429 Errors
    - Progress callbacks for UI updates
    """
    
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 10.0  # Seconds
    STEP_DELAY = 2.0  # Delay between logic steps to avoid RPM spikes
    
    def __init__(
        self,
        include_design_step: bool = True,
        include_validation: bool = True,
        model: Optional[str] = None,
        on_progress: Optional[Callable[[PipelineStep], None]] = None,
        use_langgraph: bool = True,  # Enable LangGraph-based coder-validator loop
        include_visual_testing: bool = True  # Enable browser_use visual testing
    ):
        """
        Initialize the enhanced game development chain.
        
        Args:
            include_design_step: Include visual design generation
            include_validation: Include code validation and fixing
            model: LLM model to use
            on_progress: Progress callback for UI updates
            use_langgraph: Use LangGraph for coder-validator conversation (recommended)
        """
        self.include_design_step = include_design_step
        self.include_validation = include_validation
        self.on_progress = on_progress
        self.use_langgraph = use_langgraph
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "0") == "1"
        
        # Initialize agents - each agent now uses its optimal Groq model by default
        # Planner: llama-3.3-70b-versatile (best reasoning)
        # Designer: llama-3.1-8b-instant (fast)
        # Coder: qwen/qwen3-32b (excellent for code)
        # Validator: openai/gpt-oss-120b (best bug detection)
        self.planner = PlannerAgent()
        self.designer = DesignerAgent() if include_design_step else None
        self.coder = CoderAgent(model="qwen/qwen3-32b")
        self.validator = ValidatorAgent() if include_validation else None
        self.parser = GameCodeParser()
        
        # Initialize Visual Tester (browser + NVIDIA NIM)
        self.visual_tester = VisualTesterAgent() if include_visual_testing else None
        self.include_visual_testing = include_visual_testing
        
        # Initialize LangGraph coder-validator graph (if enabled)
        self.coder_validator_graph = None
        if use_langgraph and self.validator:
            self.coder_validator_graph = CoderValidatorGraph(
                coder=self.coder,
                validator=self.validator,
                on_progress=lambda msg: print(msg)  # Simple progress logging
            )
        
        # Pipeline state
        self.steps: list[PipelineStep] = []
    
    def _notify_progress(self, step: PipelineStep):
        """Notify progress if callback is set."""
        self.steps.append(step)
        if self.on_progress:
            try:
                self.on_progress(step)
            except:
                pass
    
    def _log_parsed_snapshot(self, label: str, raw_code: str) -> None:
        """Log parsed HTML/CSS/JS lengths for debugging."""
        if not self.verbose_logs:
            return
        try:
            files = self.parser.parse_multi_file(raw_code or "")
            html_len = len(files.get("html", ""))
            css_len = len(files.get("css", ""))
            js_len = len(files.get("js", ""))
            print(f"📦 Code Memory [{label}]: HTML={html_len} CSS={css_len} JS={js_len} chars")
        except Exception as e:
            print(f"⚠️ Failed to parse code snapshot [{label}]: {e}")
    
    def run(self, user_request: str) -> GameResult:
        """
        Execute the complete game development pipeline.
        """
        start_time = datetime.now()
        self.steps = []
        
        try:
            # ============ STEP 1: PLANNING ============
            print(f"\n{'='*50}")
            print(f"📋 STEP 1: PLANNING - Model: {self.planner.model}")
            print(f"{'='*50}")
            self._notify_progress(PipelineStep(
                name="Planning",
                status="running",
                message=f"Creating game architecture..."
            ))
            
            # Use dynamic lookup for agent access to support fallback replacement
            game_plan = self._run_with_backoff(
                lambda: getattr(self, 'planner').plan(user_request),
                "Planning"
            )
            print(f"✅ Planning completed - Plan length: {len(game_plan)} chars")
            
            if not game_plan or len(game_plan) < 100:
                return self._error_result("Failed to generate game plan", start_time)
            
            self._notify_progress(PipelineStep(
                name="Planning",
                status="completed",
                message="Architecture created"
            ))
            
            # Save agent outputs for inspection
            # self._save_agent_outputs(user_request, game_plan, None)

            # Small delay between steps to avoid hitting RPM limits
            time.sleep(self.STEP_DELAY)
            
            # ============ STEP 2: DESIGN (Optional) ============
            design_spec = None
            if self.include_design_step and self.designer:
                print(f"\n{'='*50}")
                print(f"🎨 STEP 2: DESIGNING - Model: {self.designer.model}")
                print(f"{'='*50}")
                self._notify_progress(PipelineStep(
                    name="Design",
                    status="running",
                    message="Creating visual design system..."
                ))
                
                design_spec = self._run_with_backoff(
                    lambda: getattr(self, 'designer').design(game_plan),
                    "Design"
                )
                print(f"✅ Designing completed - Design length: {len(design_spec) if design_spec else 0} chars")
                
                self._notify_progress(PipelineStep(
                    name="Design",
                    status="completed",
                    message="Design system created"
                ))

                # Save updated agent outputs with design
                # self._save_agent_outputs(user_request, game_plan, design_spec)
                # Save agent outputs ONCE per request (after planning/design)
                self._save_agent_outputs(user_request, game_plan, design_spec)
                
                time.sleep(self.STEP_DELAY)
            
            # ============ STEP 3 & 4: CODING + VALIDATION ============
            # Use LangGraph for stateful coder-validator conversation (if enabled)
            
            if self.use_langgraph and self.coder_validator_graph:
                # ============ LANGGRAPH APPROACH ============
                print(f"\n{'='*50}")
                print(f"� STEPS 3-4: CODING + VALIDATION (LangGraph)")
                print(f"{'='*50}")
                
                self._notify_progress(PipelineStep(
                    name="Coding",
                    status="running",
                    message="Starting LangGraph coder-validator loop..."
                ))
                
                # Run the coder-validator graph
                import uuid
                thread_id = f"game_{uuid.uuid4().hex[:8]}"
                
                graph_result = self.coder_validator_graph.run(
                    game_plan=game_plan,
                    design_spec=design_spec or "",
                    max_iterations=None,
                    thread_id=thread_id
                )
                
                # Extract results from graph state
                game_code = graph_result.get("current_code", "")
                validation_result = None
                
                if graph_result.get("validation_result"):
                    vr = graph_result["validation_result"]
                    validation_result = ValidationResult(
                        is_valid=vr.get("is_valid", False),
                        score=vr.get("score", 0),
                        issues=vr.get("issues", []),
                        summary=vr.get("summary", "LangGraph validation complete")
                    )
                
                # Check for failure
                if graph_result.get("status") == "failed" or not game_code:
                    return self._error_result(
                        graph_result.get("error", "LangGraph pipeline failed"),
                        start_time
                    )
                
                self._notify_progress(PipelineStep(
                    name="Coding",
                    status="completed",
                    message="Game code generated"
                ))
                
                if validation_result:
                    self._notify_progress(PipelineStep(
                        name="Validation",
                        status="completed",
                        message=f"Quality score: {validation_result.score}/100"
                    ))
                
                if self.verbose_logs:
                    # Log conversation history for debugging
                    conversation = self.coder_validator_graph.get_conversation_history(thread_id)
                    if conversation:
                        print(f"\n📜 Conversation history ({len(conversation)} messages):")
                        for msg in conversation[-5:]:  # Last 5 messages
                            print(f"   {msg}")
                
            else:
                # ============ LEGACY APPROACH (Original while-loop) ============
                print(f"\n{'='*50}")
                print(f"💻 STEP 3: CODING - Model: {self.coder.model}")
                print(f"{'='*50}")
                self._notify_progress(PipelineStep(
                    name="Coding",
                    status="running",
                    message="Generating game code..."
                ))
                
                game_code = self._run_with_backoff(
                    lambda: getattr(self, 'coder').code(game_plan, design_spec or ""),
                    "Coding"
                )
                print(f"✅ Coding completed - Code length: {len(game_code)} chars")
                self._log_parsed_snapshot("Coder Output", game_code)
                
                # Check for empty skeleton or too short code
                is_too_short = not game_code or len(game_code) < 1000
                has_default_body = '<div id="game-container"><h1>Game</h1></div>' in game_code
                is_empty_skeleton = is_too_short or (len(game_code) < 5000 and has_default_body)
                
                if is_empty_skeleton:
                    print("⚠️ Generated code is empty/skeleton. Attempting fallback...")
                    if self._switch_to_fallback():
                        self._notify_progress(PipelineStep(
                            name="Coding",
                            status="running",
                            message=f"Retrying with fallback model..."
                        ))
                        game_code = self._run_with_backoff(
                            lambda: getattr(self, 'coder').code(game_plan, design_spec or ""),
                            "Coding (Fallback)"
                        )
                        is_empty_skeleton = not game_code or len(game_code) < 200
                
                if is_empty_skeleton:
                    return self._error_result("Failed to generate game code (Empty Skeleton)", start_time)
                
                self._notify_progress(PipelineStep(
                    name="Coding",
                    status="completed",
                    message="Game code generated"
                ))
                
                time.sleep(self.STEP_DELAY)
                
                # STEP 4: VALIDATION (Legacy while-loop)
                validation_result = None
                MAX_FIX_ITERATIONS = None
                
                if self.include_validation and self.validator:
                    print(f"\n{'='*50}")
                    print(f"🔍 STEP 4: VALIDATING - Model: {self.validator.model}")
                    print(f"{'='*50}")
                    
                    iteration = 0
                    current_code = game_code
                    patcher = FunctionPatcher()
                    temp_file = patcher.save_temp_file(current_code, "game_validation")
                    previous_issue_count = float('inf')
                    stall_count = 0
                    MAX_STALLS = 3
                    
                    while True:
                        iteration += 1
                        
                        self._notify_progress(PipelineStep(
                            name="Validation",
                            status="running",
                            message=f"Validating quality (attempt {iteration})..."
                        ))
                        
                        current_code = patcher.read_file(temp_file)
                        self._log_parsed_snapshot(f"Validation Iteration {iteration} (Before Validate)", current_code)
                        
                        validation_result = self._run_with_backoff(
                            lambda: getattr(self, 'validator').validate(current_code),
                            f"Validation (Attempt {iteration})"
                        )
                        
                        current_issue_count = len(validation_result.issues)
                        print(f"📊 Validation {iteration}: Score={validation_result.score}/100, Issues={current_issue_count}")
                        
                        if len(validation_result.issues) == 0:
                            print(f"✅ Validation PASSED on attempt {iteration}")
                            game_code = current_code
                            break
                        
                        if current_issue_count >= previous_issue_count:
                            stall_count += 1
                            if stall_count >= MAX_STALLS:
                                print(f"🛑 Stall detected. Exiting.")
                                game_code = current_code
                                break
                        else:
                            stall_count = 0
                        
                        previous_issue_count = current_issue_count
                        
                        if MAX_FIX_ITERATIONS is not None and iteration >= MAX_FIX_ITERATIONS:
                            game_code = current_code
                            break
                        
                        # Use fix_code for targeted fixes with full context
                        function_names = patcher.extract_function_names_from_issues(validation_result.issues)
                        if function_names:
                            targeted = []
                            for fn in function_names:
                                matched_issue = next(
                                    (
                                        issue for issue in validation_result.issues
                                        if fn.lower() in str(issue.get('location', '')).lower()
                                        or fn.lower() in str(issue.get('issue', '')).lower()
                                        or fn.lower() in str(issue.get('fix', '')).lower()
                                    ),
                                    None,
                                )
                                if matched_issue:
                                    targeted.append(
                                        f"- Fix {fn}: {matched_issue.get('issue', 'Fix this function')} → {matched_issue.get('fix', 'Implement properly')}"
                                    )
                                else:
                                    targeted.append(f"- Fix {fn}: Ensure this function is fully implemented and callable")

                            fix_instructions = "\n".join(targeted)
                            if not fix_instructions.strip():
                                fix_instructions = self._format_fix_instructions(validation_result.issues)
                            
                            time.sleep(1)
                            current_code = self._run_with_backoff(
                                lambda: getattr(self, 'coder').fix_code(current_code, fix_instructions),
                                f"Fixing {len(function_names)} functions"
                            )
                            patcher.write_file(temp_file, current_code)
                            self._log_parsed_snapshot(f"Validation Iteration {iteration} (After Fix)", current_code)
                        else:
                            fix_instructions = self._format_fix_instructions(validation_result.issues)
                            full_fix_plan = f"""## ORIGINAL GAME PLAN:
{game_plan}

## DESIGN SPEC:
{design_spec or 'Use modern styling'}

## CRITICAL FIXES NEEDED:
{fix_instructions}

## REQUIREMENTS:
1. Fix ALL the issues listed above
2. Return COMPLETE, working HTML file"""
                            
                            current_code = self._run_with_backoff(
                                lambda: getattr(self, 'coder').code(full_fix_plan, design_spec or ""),
                                f"Full Regeneration (Iteration {iteration})"
                            )
                            patcher.write_file(temp_file, current_code)
                            self._log_parsed_snapshot(f"Validation Iteration {iteration} (After Regen)", current_code)
                    
                    patcher.cleanup_temp_files(keep_last=3)
                    
                    self._notify_progress(PipelineStep(
                        name="Validation",
                        status="completed",
                        message=f"Quality score: {validation_result.score}/100"
                    ))
            
            # ============ BUILD RESULT ============
            # Parse for multi-file output
            game_files = self.parser.parse_multi_file(game_code)
            
            metadata = {
                "title": self.parser.get_metadata().title,
                "has_html": self.parser.get_metadata().has_html,
                "has_css": self.parser.get_metadata().has_css,
                "has_js": self.parser.get_metadata().has_js,
                "coder_model": self.coder.model,
                "planner_model": self.planner.model,
                "included_design": self.include_design_step,
                "validated": self.include_validation,
                "quality_score": validation_result.score if validation_result else None
            }
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            self._notify_progress(PipelineStep(
                name="Complete",
                status="completed",
                message=f"Game ready! ({generation_time:.1f}s)"
            ))
            
            return GameResult(
                game_code=game_code,  # Keep original for backwards compatibility
                game_plan=game_plan,
                design_spec=design_spec,
                metadata=metadata,
                validation=validation_result,
                success=True,
                generation_time=generation_time,
                steps_completed=[s.name for s in self.steps if s.status == "completed"],
                game_files=game_files  # New: separate HTML, CSS, JS
            )
            
        except Exception as e:
            return self._error_result(str(e), start_time)

    def _save_agent_outputs(self, prompt: str, game_plan: str, design_spec: Optional[str]) -> None:
        """Persist planner/designer outputs for inspection."""
        try:
            base_dir = Path(__file__).parent.parent.parent
            out_dir = base_dir / "logs" / "agent_outputs"
            out_dir.mkdir(parents=True, exist_ok=True)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plan_path = out_dir / f"plan_{stamp}.txt"
            plan_path.write_text(game_plan or "", encoding="utf-8")

            design_text = design_spec or ""
            design_path = out_dir / f"design_{stamp}.txt"
            design_path.write_text(design_text, encoding="utf-8")

            meta = {
                "timestamp": stamp,
                "prompt": prompt,
                "plan": game_plan or "",
                "design": design_text,
            }
            meta_path = out_dir / f"agent_outputs_{stamp}.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Failed to save agent outputs: {e}")

    def _switch_to_fallback(self) -> bool:
        """Switch to Gemini 2.5 Flash fallback when primary quota is exhausted."""
        FALLBACK_MODEL = "gemini-2.5-flash"  # Reliable cloud fallback
        
        # Check if coder is already using fallback (coder is the main concern)
        if hasattr(self, 'coder') and self.coder.model == FALLBACK_MODEL:
            if self.verbose_logs:
                print(f"DEBUG: Already using fallback model {FALLBACK_MODEL}")
            return False  # Already using fallback
            
        if self.verbose_logs:
            print(f"DEBUG: Switching coder from {self.coder.model} to {FALLBACK_MODEL}")
        
        # Reinitialize all agents with fallback model (use_groq=False)
        try:
            self.planner = PlannerAgent(model=FALLBACK_MODEL, use_groq=False)
            if self.designer: 
                self.designer = DesignerAgent(model=FALLBACK_MODEL, use_groq=False)
            self.coder = CoderAgent(model=FALLBACK_MODEL, use_groq=False)
            if self.validator: 
                self.validator = ValidatorAgent(model=FALLBACK_MODEL, use_groq=False)
            
            if self.verbose_logs:
                print(f"DEBUG: Fallback agents initialized with {FALLBACK_MODEL}")
                print(f"DEBUG: Coder model is now: {self.coder.model}")
            return True
            
        except Exception as e:
            if self.verbose_logs:
                print(f"DEBUG: Failed to initialize fallback: {e}")
            return False

    def _run_with_backoff(self, operation: Callable, step_name: str):
        """Execute operation with exponential backoff and jitter."""
        last_error = None
        current_backoff = self.INITIAL_BACKOFF
        rate_limit_on_fallback = 0  # Track 429s when already on fallback
        MAX_FALLBACK_RETRIES = 3  # Exit after 3 consecutive 429s on fallback
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return operation()
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check for rate limit errors (429)
                if "429" in error_msg or "resource_exhausted" in error_msg:
                    if self.verbose_logs:
                        print(f"DEBUG: 429 Error encountered in {step_name}. Attempting fallback...")
                    # Try switching to fallback model for daily limits
                    if self._switch_to_fallback():
                        self._notify_progress(PipelineStep(
                            name=step_name,
                            status="running",
                            message=f"Switched to fallback model {self.coder.model}..."
                        ))
                        time.sleep(2.0) # Small pause for switch
                        if self.verbose_logs:
                            print(f"DEBUG: Retrying {step_name} with new model {self.coder.model}")
                        rate_limit_on_fallback = 0  # Reset counter after successful switch
                        # Continue to next iteration - operation will use updated self.coder
                        continue
                    else:
                        # Already on fallback, increment counter
                        rate_limit_on_fallback += 1
                        if self.verbose_logs:
                            print(f"DEBUG: Rate limit on fallback ({rate_limit_on_fallback}/{MAX_FALLBACK_RETRIES})")
                        
                        if rate_limit_on_fallback >= MAX_FALLBACK_RETRIES:
                            print(f"🛑 API quota exhausted on fallback model. Cannot continue.")
                            raise Exception(f"API quota exhausted. Please wait or use a different API key.")

                    if attempt < self.MAX_RETRIES:
                        # Add jitter to backoff
                        jitter = random.uniform(0.8, 1.2)
                        wait_time = current_backoff * jitter
                        
                        self._notify_progress(PipelineStep(
                            name=step_name,
                            status="running",
                            message=f"Rate limit hit. Waiting {wait_time:.1f}s before retry..."
                        ))
                        
                        time.sleep(wait_time)
                        current_backoff *= 2.0  # Exponential increase
                        continue
                
                # For basic retries (if result was None/Failed but not rate limited)
                if attempt < self.MAX_RETRIES:
                    time.sleep(2)  # Short wait for general errors
                    continue
                    
        raise last_error

    def _format_issues(self, issues: list) -> str:
        """Format validation issues as feedback for retry."""
        if not issues: return ""
        return "\n".join([f"- [{i.get('severity', 'issue')}] {i.get('issue', '')}: {i.get('fix', '')}" for i in issues[:5]])
    
    def _format_fix_instructions(self, issues: list) -> str:
        """Format validation issues as CONCISE fix instructions with EXACT LOCATIONS."""
        if not issues:
            return "No issues found."
        
        instructions = ["## LOGIC ERRORS TO FIX:\n"]
        
        for idx, issue in enumerate(issues, 1):
            location = issue.get('location', 'Unknown location')
            problem = issue.get('issue', 'Unknown issue')
            fix = issue.get('fix', 'Fix this')
            
            instructions.append(f"""{idx}. **{location}**
   - Problem: {problem}
   - Fix: {fix}
""")
        
        return "\n".join(instructions)
    
    def _error_result(self, error: str, start_time) -> GameResult:
        """Create an error result."""
        self._notify_progress(PipelineStep(name="Error", status="failed", message=error))
        return GameResult(
            game_code="", game_plan="", design_spec=None, metadata={},
            validation=None, success=False, error=error,
            generation_time=(datetime.now() - start_time).total_seconds(),
            steps_completed=[s.name for s in self.steps if s.status == "completed"]
        )
    
    def run_quick(self, user_request: str) -> GameResult:
        """Quick generation without validation (design stays enabled by default)."""
        original_design = self.include_design_step
        original_validation = self.include_validation
        try:
            self.include_design_step = False
            self.include_validation = False
            result = self.run(user_request)
        finally:
            self.include_design_step = original_design
            self.include_validation = original_validation
        return result
    
    def plan_only(self, user_request: str) -> str:
        """Only generate a game plan."""
        return self._run_with_backoff(lambda: self.planner.plan(user_request), "Planning")
    
    def code_from_plan(self, game_plan: str) -> str:
        """Generate code from an existing plan."""
        return self._run_with_backoff(lambda: self.coder.code(game_plan), "Coding")
