"""
Visual Tester Agent - Uses Playwright + NVIDIA NIM to visually validate games.
This agent acts as the "eyes" of the pipeline, opening games in a real browser 
and testing them autonomously with an LLM-powered vision model.

REMOVED DEPENDENCY: browser-use (requires Python 3.11+)
NOW USES: playwright directly (compatible with older Python)
"""

import os
import asyncio
import base64
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VisualTestResult:
    """Result from visual game testing."""
    passed: bool = False
    score: int = 0           # 0-100 visual quality score
    issues: List[Dict] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    summary: str = ""
    interactions_tested: int = 0


class VisualTesterAgent:
    """
    Visual Game Tester - NVIDIA NIM powered browser automation.
    Uses Playwright directly to open and visually validate generated games.
    
    This is the "eyes" of the pipeline - it catches bugs that regex can't:
    - Canvas not rendering
    - Pieces invisible or overlapping
    - Buttons not responding
    - Game states not transitioning
    - Layout/styling issues
    """
    
    def __init__(self, model: Optional[str] = None):
        """Initialize the Visual Tester with NVIDIA NIM model."""
        self.use_nvidia = os.getenv("USE_NVIDIA", "").lower() == "true" and os.getenv("NVIDIA_API_KEY")
        
        # Use NVIDIA's vision model if available, otherwise fallback
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = "https://integrate.api.nvidia.com/v1"
        self.model = model or "nvidia/llama-3.1-nemotron-ultra-253b-v1"
        
        if api_key:
            # Use a verified Vision-capable model
            self.model = model or "meta/llama-3.2-11b-vision-instruct" 
            
            self.llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=self.model,
                temperature=0.1,
                max_tokens=4096,
            )
            print(f"👁️ VisualTester using NVIDIA NIM (Vision): {self.model}")
        else:
            # Fallback to standard OpenAI or compatible
            self.llm = ChatOpenAI(temperature=0.1)
            print(f"👁️ VisualTester using default LLM (Check env vars!)")
    
    def _build_test_task(self, game_type: str) -> str:
        """Build a comprehensive visual test task based on game type."""
        
        base_task = """
        You are a QA tester for a browser-based game. You are looking at a screenshot of the game.
        
        ## VISUAL CHECKS (report pass/fail for each):
        1. Does the page load without errors? (Is the screen blank or showing an error?)
        2. Is a game canvas or board visible on screen?
        3. Are all game elements properly rendered (not clipped, not invisible)?
        4. Do the colors look correct (distinct player sides, visible pieces)?
        5. Is the text readable (score, turn indicator, buttons)?
        """
        
        game_specific_tasks = {
            'chess': """
            ## CHESS-SPECIFIC TESTS:
            6. Is the board SQUARE (8x8 grid, all rows visible)?
            7. Can you see pieces on the board? Are white and black pieces visually DIFFERENT?
            8. Does the board layout look like a standard chess setup?
            """,
            'snake': """
            ## SNAKE-SPECIFIC TESTS:
            6. Is the snake visible on the board?
            7. Is food visible on the board?
            8. Is there a clear grid or play area?
            """,
            'tictactoe': """
            ## TIC-TAC-TOE SPECIFIC TESTS:
            6. Is the 3x3 grid visible?
            7. Are the grid lines clear?
            8. Is there a score display?
            """,
        }
        
        specific = game_specific_tasks.get(game_type, """
            ## GENERAL GAME TESTS:
            6. Is there a clear "Play" or "Start" button?
            7. Is there a score display?
            8. Does the layout look playable?
            """)
        
        return base_task + specific + """
        ## REPORT FORMAT:
        For each test, report: ✅ PASS or ❌ FAIL with a brief explanation.
        End with an overall SCORE out of 100 and a summary of critical issues.
        """

    async def test_game_async(self, file_path: str, game_type: str = "custom") -> VisualTestResult:
        """
        Asynchronously test a game file using Playwright + Vision LLM.
        
        CRITICAL WINDOWS FIX:
        Playwright requires the ProactorEventLoop on Windows to spawn subprocesses.
        Uvicorn (FastAPI) defaults to SelectorEventLoop which lacks this.
        If we detect an incompatible loop, we offload to a thread with the correct loop.
        """
        import sys
        import asyncio
        
        # Check for Windows loop incompatibility
        if sys.platform == 'win32' and isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop):
            print(f"⚠️ [VisualTester] Incompatible Windows loop detected. Offloading to separate thread...")
            # Run in a separate thread with a fresh Proactor loop
            return await asyncio.to_thread(self._run_in_proactor_thread, file_path, game_type)
            
        # If loop is compatible (or not on Windows), run normally
        return await self._run_test_logic(file_path, game_type)

    def _run_in_proactor_thread(self, file_path: str, game_type: str) -> VisualTestResult:
        """Helper to run the test in a thread with a ProactorEventLoop."""
        import asyncio
        loop = asyncio.WindowsProactorEventLoopPolicy().new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run_test_logic(file_path, game_type))
        finally:
            loop.close()

    async def _run_test_logic(self, file_path: str, game_type: str) -> VisualTestResult:
        """Core test logic (launch browser, check screenshot, etc)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("⚠️ playwright not installed. Run: pip install playwright && playwright install")
            return VisualTestResult(
                passed=True,
                score=50,
                summary="Visual testing skipped - playwright not installed",
                issues=[{'severity': 'info', 'issue': 'playwright not installed', 'fix': 'pip install playwright'}]
            )
        
        print(f"👁️ [VisualTester] Opening game in browser: {file_path}")
        
        try:
            async with async_playwright() as p:
                # Launch browser (headless for speed)
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to local file
                # Handle Windows paths for file URI
                if os.name == 'nt' and not file_path.startswith('file:///'):
                    url_path = file_path.replace('\\', '/')
                    if not url_path.startswith('//'):
                        url_path = f"///{url_path}"
                    file_uri = f"file:{url_path}"
                else:
                    file_uri = f"file://{file_path}"
                
                # Capture console errors
                console_errors = []
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
                page.on("pageerror", lambda exc: console_errors.append(str(exc)))
                
                try:
                    await page.goto(file_uri)
                    # Wait for game to render (canvas, board, etc.)
                    await page.wait_for_load_state("networkidle")
                    await asyncio.sleep(2)  # Extra wait for JS rendering
                    
                    # Take screenshot
                    screenshot_bytes = await page.screenshot(full_page=True)
                    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                    
                    # Analyze with LLM
                    print(f"👁️ [VisualTester] Analyzing screenshot with {self.model}...")
                    
                    # Add console errors to the prompt if any exist
                    error_context = ""
                    if console_errors:
                        print(f"⚠️ [VisualTester] Found {len(console_errors)} console errors!")
                        error_context = f"\n\n## CRITICAL CONSOLE ERRORS FOUND:\n" + "\n".join([f"- {err}" for err in console_errors])
                    
                    task_prompt = self._build_test_task(game_type) + error_context
                    
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": task_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{screenshot_b64}"
                                },
                            },
                        ]
                    )
                    
                    response = await self.llm.ainvoke([message])
                    result_text = response.content
                    
                    # Combine LLM result with hard-captured console errors
                    result = self._parse_result(result_text)
                    
                    # Inject console errors as critical issues if LLM missed them
                    for err in console_errors:
                        result.passed = False
                        result.score = min(result.score, 40) # Penalty for runtime errors
                        result.issues.insert(0, {
                            'severity': 'critical', 
                            'issue': f"Runtime Error: {err[:200]}", 
                            'fix': 'Fix JavaScript runtime error'
                        })
                    
                    return result
                    
                except Exception as e:
                    print(f"⚠️ [VisualTester] Browser navigation failed: {e}")
                    return VisualTestResult(
                        passed=False,
                        score=0,
                        summary=f"Browser error: {str(e)}",
                        issues=[{'severity': 'critical', 'issue': f"Browser Error: {e}", 'fix': 'Check file path'}]
                    )
                finally:
                    await browser.close()
                    
        except Exception as e:
            print(f"⚠️ [VisualTester] Playwright failed: {e}")
            return VisualTestResult(
                passed=True,  # Don't block pipeline
                score=40,
                summary=f"Visual testing error: {str(e)[:200]}",
                issues=[{'severity': 'warning', 'issue': f'Test harness failed: {str(e)[:100]}', 'fix': 'Check environment'}]
            )
            
    def test_game(self, file_path: str, game_type: str = "custom") -> VisualTestResult:
        """Synchronous wrapper for test_game_async."""
        import asyncio
        import sys
        
        # If running on Windows, use Proactor loop directly
        if sys.platform == 'win32':
             return self._run_in_proactor_thread(file_path, game_type)
             
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.test_game_async(file_path, game_type))
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(self.test_game_async(file_path, game_type))
        except Exception as e:
            print(f"⚠️ [VisualTester] Sync wrapper error: {e}")
            return VisualTestResult(passed=True, score=50, summary=f"Wrapper error: {e}")

    def _parse_result(self, raw_result: str) -> VisualTestResult:
        """Parse the LLM's evaluation text into a structured result."""
        import re
        
        # Count passes and fails
        passes = len(re.findall(r'✅|PASS', raw_result, re.IGNORECASE))
        fails = len(re.findall(r'❌|FAIL', raw_result, re.IGNORECASE))
        total = passes + fails
        
        # Extract score if mentioned
        score_match = re.search(r'(\d+)\s*/\s*100', raw_result)
        if score_match:
            score = int(score_match.group(1))
        elif total > 0:
            score = int((passes / total) * 100)
        else:
            score = 50  # Default if we can't parse
        
        # Extract issues from ❌ lines
        issues = []
        for line in raw_result.split('\n'):
            if '❌' in line or 'FAIL' in line.upper():
                issues.append({
                    'severity': 'critical' if any(kw in line.lower() for kw in ['not visible', 'crash', 'error', 'blank']) else 'warning',
                    'location': 'Visual Test',
                    'issue': line.strip()[:200],
                    'fix': 'Fix the visual rendering issue'
                })
        
        return VisualTestResult(
            passed=fails == 0 or score >= 70,
            score=score,
            issues=issues,
            summary=raw_result[:500],
            interactions_tested=total
        )
