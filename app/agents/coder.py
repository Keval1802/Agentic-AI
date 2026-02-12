"""
Enhanced Coder Agent - Multi-phase production-grade code generation.
Generates complete, polished HTML games with proper architecture.
Primary: NVIDIA NIM with Mistral Devstral 2 123B (dedicated code model).
Fallback: Groq → Gemini.
"""

import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from ..prompts.templates import CODER_PROMPT
from ..parsers.game_parser import GameCodeParser

load_dotenv()

FIX_FUNCTION_PROMPT = """You are a surgical code fixer.
Your goal is to fix a SINGLE function or method without breaking external references.

## CONTEXT (Global Variables & HTML):
{context}

## BROKEN FUNCTION:
```javascript
{existing_code}
```

## FIX INSTRUCTIONS:
{fix_instructions}

## RULES:
1. Return the **FULL FUNCTION** (Signature + Body).
2. If the function is MISSING, generating a new one is allowed.
3. Maintain the exact same signature (name + params) if it exists.
4. Do NOT wrap in `class Game {{ ... }}` unless it was originally inside braces.
5. USE the provided context to prevent ReferenceError.
6. NO Markdown backticks. Just the raw code.
"""

ENHANCED_CODER_PROMPT = """You are an ELITE game developer. Generate 100% COMPLETE, WORKING code.

## Game Specs:
{game_plan}

## Design:
{design_spec}

## ⚠️ MANDATORY RULES:

### NO PLACEHOLDERS - FORBIDDEN:
❌ "// logic here", "// TODO", "// implement", "// add code"
✅ Every function MUST have real, working code

### PREVENT ReferenceError:
1. Use `this.` for ALL class properties: `this.score`, `this.player`
2. Declare variables BEFORE using them
3. Use arrow functions in callbacks: `(e) => this.handleKey(e)`
4. Every method you call MUST exist

### REQUIRED FEATURES:
- Start screen with Play button
- Game loop with requestAnimationFrame
- P=pause, R=restart
- Score with localStorage high score
- Game over screen

## CRITICAL STEP - THINK FIRST:
Before writing code, you MUST output a <thinking> block:
1. Analyze the requirements.
2. Plan the class structure (Game, Entity, etc.).
3. Identify potential ReferenceErrors (variable scope).
4. Define the game loop logic step-by-step.
5. List required HTML overlays.

## OUTPUT FORMAT:
<thinking>
...your detailed plan...
</thinking>

followed by 3 code blocks:

```html
<!DOCTYPE html>...
```

```css
/* Complete styles */...
```

```javascript
/* Complete logic */...
```

```html
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Game</title>
<link rel="stylesheet" href="style.css">
</head><body>
<div class="game-container">
  <div id="startOverlay"><h1>Title</h1><button onclick="game.start()">Play</button></div>
  <canvas id="gameCanvas"></canvas>
  <div id="pauseOverlay" style="display:none;">Paused</div>
  <div id="gameOverOverlay" style="display:none;"></div>
</div>
<script src="script.js"></script>
</body></html>
```

```css
/* Complete styles with colors, animations */
```

```javascript
class Game {{
  constructor() {{
    this.canvas = document.getElementById('gameCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.canvas.width = 800;
    this.canvas.height = 600;
    this.gameActive = false;
    this.score = 0;
    this.highScore = parseInt(localStorage.getItem('highScore')) || 0;
    // Initialize ALL game objects here with this.
    this.setupEventListeners();
  }}
  
  setupEventListeners() {{
    document.addEventListener('keydown', (e) => this.handleKey(e));
  }}
  
  handleKey(e) {{
    if (e.key === 'p' || e.key === 'P') this.togglePause();
    if (e.key === 'r' || e.key === 'R') this.restart();
    // Handle movement keys
  }}
  
  start() {{
    this.gameActive = true;
    this.score = 0;
    // Reset game objects
    document.getElementById('startOverlay').style.display = 'none';
    this.lastTime = performance.now();
    this.gameLoop();
  }}
  
  gameLoop(t = performance.now()) {{
    if (!this.gameActive) return;
    const dt = (t - this.lastTime) / 1000;
    this.lastTime = t;
    this.update(dt);
    this.render();
    requestAnimationFrame((t) => this.gameLoop(t));
  }}
  
  update(dt) {{ /* REAL game logic here */ }}
  render() {{ this.ctx.clearRect(0,0,800,600); /* REAL drawing here */ }}
  togglePause() {{ /* pause logic */ }}
  gameOver() {{ 
    this.gameActive = false;
    if (this.score > this.highScore) localStorage.setItem('highScore', this.score);
    document.getElementById('gameOverOverlay').style.display = 'flex';
  }}
  restart() {{ this.start(); }}
}}
const game = new Game();
```

## FINAL CHECK:
✅ All variables declared with this.
✅ All methods exist before calling
✅ Arrow functions in event listeners
✅ No placeholder comments
"""


class CoderAgent:
    """
    Game Coding Agent - NVIDIA NIM with Mistral Devstral 2 123B.
    Dedicated Code Model: State-of-the-art open code model with deep reasoning,
    256k context, and unmatched efficiency for game code generation.
    Fallback: Groq → Gemini.
    """
    
    def __init__(self, model: Optional[str] = None, use_groq: bool = True):
        """Initialize the Coder Agent.
        
        Priority: NVIDIA NIM (Devstral 2 123B) → Groq → Gemini.
        """
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "0") == "1"
        self.use_nvidia = os.getenv("USE_NVIDIA", "").lower() == "true" and os.getenv("NVIDIA_API_KEY")
        self.use_groq = use_groq and os.getenv("GROQ_API_KEY")

        if model and model.startswith("qwen/"):
            self.use_nvidia = False
        
        if self.use_nvidia:
            # PRIMARY: NVIDIA NIM with Devstral 2 123B (Dedicated Code Model)
            self.model = model or "mistralai/devstral-2-123b-instruct-2512"
            self.llm = ChatOpenAI(
                model=self.model,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=0.3,
                max_tokens=20000,
            )
            print(f"🧠 CoderAgent using NVIDIA NIM: {self.model} (Dedicated Code Model)")
        elif self.use_groq:
            # FALLBACK 1: Groq with Qwen 3 32B
            self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
            self.llm = ChatGroq(
                model=self.model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3,  
                max_tokens=40768,
            )
            print(f"🚀 CoderAgent using Groq: {self.model}")
        else:
            # FALLBACK 2: Gemini
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3, 
                max_output_tokens=32768,
                convert_system_message_to_human=True
            )
            print(f"🔄 CoderAgent using Gemini: {self.model}")
        
        self.prompt = ChatPromptTemplate.from_template(ENHANCED_CODER_PROMPT)
        self.parser = GameCodeParser()
    
    def code(self, game_plan: str, design_spec: str = "") -> str:
        """Generate complete, production-ready game code.
        
        Returns the RAW LLM output (with thinking tokens stripped).
        The caller should use GameCodeParser for final parsing.
        """
        messages = self.prompt.format_messages(
            game_plan=game_plan,
            design_spec=design_spec or "Use modern, attractive default styling"
        )
        if self.verbose_logs:
            print(f"DEBUG CODER: Invoking LLM ({self.model})...")
        response = self.llm.invoke(messages)
        raw_code = self._extract_content(response)
        
        # Post-process: fix common LLM formatting issues
        raw_code = self._fix_formatting(raw_code)
        
        if self.verbose_logs:
            print(f"DEBUG CODER: Raw response length: {len(raw_code)}")
        
        # Return RAW output - let game_chain use parse_multi_file
        # This preserves separate ```html, ```css, ```javascript blocks
        return raw_code
    
    def code_with_examples(self, game_plan: str, example_code: Optional[str] = None, design_spec: str = "") -> str:
        """Generate code with example reference."""
        if example_code:
            enhanced_plan = f"""
{game_plan}

## Reference Example (match this quality):
```
html
{example_code[:3000]}
```
"""
            return self.code(enhanced_plan, design_spec)
        return self.code(game_plan, design_spec)
    
    def _strip_thinking_tokens(self, content: str) -> str:
        """Strip Qwen 3's <think>...</think> blocks from response."""
        import re
        # Remove <think>...</think> blocks (Qwen 3's reasoning)
        clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return clean.strip()
    
    def _fix_formatting(self, code: str) -> str:
        """Fix common LLM code formatting issues that cause runtime errors."""
        import re
        
        # 1. Fix concatenated method declarations: }methodName( → }\n\n    methodName(
        # Only match when } is at end-of-line (avoid corrupting JSON/object literals)
        code = re.sub(
            r'\}\s*\n(\s*)((?:async\s+)?(?:get |set |static )?[a-zA-Z_]\w*\s*\()',
            r'}\n\n\1\2',
            code
        )
        
        # 2. Fix non-square canvas for board games (chess, checkers, etc.)
        board_keywords = ['chess', 'checkers', 'draughts', 'othello', 'reversi', 'go ']
        is_board_game = any(kw in code.lower() for kw in board_keywords)
        if is_board_game:
            # Find canvas width and height
            w_match = re.search(r'canvas\.width\s*=\s*(\d+)', code)
            h_match = re.search(r'canvas\.height\s*=\s*(\d+)', code)
            if w_match and h_match:
                w, h = int(w_match.group(1)), int(h_match.group(1))
                if w != h:
                    # Make canvas square using the width value
                    code = code.replace(
                        f'canvas.height = {h}',
                        f'canvas.height = {w}'
                    )
                    print(f"🔧 CODER FIX: Fixed non-square canvas {w}x{h} → {w}x{w}")
            
        return code
    
    def _extract_content(self, response) -> str:
        """Safely extract content from response and strip thinking tokens."""
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            else:
                content = str(content)
        else:
            content = str(response)
        
        # Strip Qwen's thinking tokens
        content = self._strip_thinking_tokens(content)
        
        return content
    
    def get_metadata(self):
        """Get metadata about the last generated game."""
        return self.parser.get_metadata()
    
    def fix_code(self, existing_code: str, fix_instructions: str, context: str = "") -> str:
        """
        Fix code based on instructions.
        
        Mode 1 (Context Provided): Surgical function patch.
        - existing_code: The function body/signature only.
        - context: Global vars/HTML IDs.
        - Returns: Fixed function code only.
        
        Mode 2 (No Context): Full file fix.
        - existing_code: The entire file.
        - Returns: The entire file (HTML/CSS/JS).
        """
        if context:
            # Mode 1: Surgical Fix
            prompt = ChatPromptTemplate.from_template(FIX_FUNCTION_PROMPT)
            messages = prompt.format_messages(
                existing_code=existing_code,
                fix_instructions=fix_instructions,
                context=context
            )
        else:
            # Mode 2: Full File Fix
            prompt = ChatPromptTemplate.from_template(FIX_CODE_PROMPT)
            messages = prompt.format_messages(
                existing_code=existing_code,
                fix_instructions=fix_instructions
            )
            
        if self.verbose_logs:
            print(f"DEBUG CODER FIX: Applying {'surgical' if context else 'full'} fixes...")
            
        response = self.llm.invoke(messages)
        fixed_code = self._extract_content(response)
        
        # Post-process: fix common LLM formatting issues
        fixed_code = self._fix_formatting(fixed_code)
        
        if self.verbose_logs:
            print(f"DEBUG CODER FIX: Fixed code length: {len(fixed_code)}")
        return fixed_code

# Separate prompt for targeted fixes (legacy - full file approach)
FIX_CODE_PROMPT = """You are a code fixer. Your job is to fix SPECIFIC functions in the existing code.

## EXISTING CODE (DO NOT REWRITE EVERYTHING):
```html
{existing_code}
```

## FIX INSTRUCTIONS (ONLY FIX THESE):
{fix_instructions}

## YOUR TASK:
1. Find each function, method, OR variable mentioned in the fix instructions.
2. If it EXISTS: Replace ONLY that part with working code.
3. If it is MISSING: **ADD IT** to the appropriate class or scope.
4. Keep ALL other code EXACTLY as it is (Copy-Paste untouched parts).
4. Do NOT change HTML structure, CSS, or unmentioned functions
5. STRICTLY precise changes.

## OUTPUT (THREE CODE BLOCKS ONLY):
Return the COMPLETE corrected game code.
```html
...
```
```css
...
```
```javascript
...
```
"""
