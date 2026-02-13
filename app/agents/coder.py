"""
Enhanced Coder Agent - Clean Version
Generates complete HTML game files with inline CSS and JS.
Primary: NVIDIA NIM (Mistral Devstral 2 123B)
Fallback: Groq → Gemini
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from ..parsers.game_parser import GameCodeParser

load_dotenv()

# Simplified, single-output HTML game generation prompt
ENHANCED_CODER_PROMPT = """You are a professional game developer. Generate a single, 100% complete HTML game file.

## Game Specification:
{game_plan}

## Design Specification:
{design_spec}

## OUTPUT REQUIREMENTS:
- Return ONE HTML file only.
- Include all CSS inside <style> tags.
- Include all JavaScript inside <script> tags.
- Must start with <!DOCTYPE html> and end with </html>.
- Do NOT include markdown, backticks, or commentary.
- ZERO explanations, ZERO reasoning, ZERO thinking — output ONLY the HTML file.

## UI/UX REQUIREMENTS (MANDATORY):
1. Import a modern Google Font (Inter, Outfit, or Poppins) via <link> in <head>.
2. Use a curated dark color palette — NO plain red/blue/green. Use HSL-based or gradient combos.
3. Background: subtle gradient (e.g. linear-gradient(135deg, #0f0c29, #302b63, #24243e)).
4. Buttons: rounded corners (8-12px), box-shadow, gradient fill, hover scale(1.05) + glow effect.
5. Overlays (start/pause/game-over): glassmorphism — backdrop-filter: blur(10px), semi-transparent bg.
6. Score/HUD: fixed position, semi-transparent panel, rounded, with subtle shadow.
7. Canvas: centered, rounded corners, subtle border or box-shadow glow.
8. Animations: fadeIn for overlays, pulse for score changes, smooth transitions (0.3s ease).
9. Responsive: game container must adapt to viewport (max-width + margin: auto).
10. Typography: use the imported font everywhere, good spacing, readable sizes.

## GAME REQUIREMENTS:
1. Start screen with Play button.
2. Game loop using requestAnimationFrame().
3. Keyboard controls:
   - P = Pause
   - R = Restart
4. Score system with localStorage high score.
5. Game Over screen with restart option.

## CODING RULES:
1. No placeholders or incomplete logic.
   ❌ "// TODO", "// logic here"
   ✅ Every function must contain full, working logic.
2. Use `this.` for all class properties.
3. Declare variables before use.
4. All called methods must exist.
5. Use arrow functions for event callbacks.
6. Code must run without ReferenceError.
7. Register input listeners once (constructor/init), never inside update/draw/gameLoop.
8. Do not access a property after setting it to null in the same logical block.

## OUTPUT FORMAT:
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Game</title>
<style>
/* Full CSS here */
</style>
</head>
<body>
<div class="game-container">
  <div id="startOverlay"><h1>Game Title</h1><button onclick="game.start()">Play</button></div>
  <canvas id="gameCanvas"></canvas>
  <div id="pauseOverlay" style="display:none;">Paused</div>
  <div id="gameOverOverlay" style="display:none;">Game Over</div>
</div>
<script>
/* Full JavaScript logic here */
class Game {{
  // Complete implementation
}}
const game = new Game();
</script>
</body>
</html>

/no_think
"""

# Surgical fix prompt (kept minimal but effective)
FIX_FUNCTION_PROMPT = """You are a precise code fixer.
Fix the SINGLE function or method below using the provided context.

## CONTEXT:
{context}

## EXISTING FUNCTION:
```javascript
{existing_code}
FIX INSTRUCTIONS:
{fix_instructions}

RULES:
Return the full function (signature + body).
Keep the same name and parameters.
Use context to prevent ReferenceError.
Output ONLY the raw code (no markdown or backticks).
ZERO explanations, ZERO reasoning, ZERO commentary — code ONLY.
Preserve existing UI/UX styling — do NOT remove CSS classes, animations, or visual effects.

/no_think

Full-file fix mode """
FIX_CODE_PROMPT = """You are a code fixer.
Your job is to correct or complete functions as described below.

EXISTING CODE:
{existing_code}
FIX INSTRUCTIONS:
{fix_instructions}

RULES:
Modify only relevant parts.
Keep all other code unchanged.
Return ONE full HTML file starting with <!DOCTYPE html> and ending with </html>.
No markdown, backticks, or commentary.
ZERO explanations, ZERO reasoning, ZERO thinking — output the HTML file ONLY.
Preserve ALL existing UI/UX styling — do NOT remove CSS, animations, gradients, fonts, or visual effects.

/no_think
"""

class CoderAgent:
    """Game Coding Agent
    Multi-model system for generating complete, production-ready HTML games.
    Priority: NVIDIA NIM (Devstral 2 123B) → Groq → Gemini
    """

    def __init__(self, model: Optional[str] = None, use_groq: bool = True):
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "0") == "1"
        self.use_nvidia = os.getenv("USE_NVIDIA", "").lower() == "true" and os.getenv("NVIDIA_API_KEY")
        self.use_groq = use_groq and os.getenv("GROQ_API_KEY_2")

        if model and model.startswith("qwen/"):
            self.use_nvidia = False

        if self.use_groq:
            # Groq fallback
            self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
            self.llm = ChatGroq(
                model=self.model,
                api_key=os.getenv("GROQ_API_KEY_2"),
                temperature=0.3,
                max_tokens=40768,
            )
            print(f"🚀 Using Groq: {self.model}")
        else:
            # Gemini fallback
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,
                max_output_tokens=32768,
                convert_system_message_to_human=True
            )
            print(f"🔄 Using Gemini: {self.model}")

        self.prompt = ChatPromptTemplate.from_template(ENHANCED_CODER_PROMPT)
        self.parser = GameCodeParser()

    def code(self, game_plan: str, design_spec: str = "") -> str:
        """Generate full HTML game file."""
        messages = self.prompt.format_messages(
            game_plan=game_plan,
            design_spec=design_spec or "Use clean, modern design with minimal CSS."
        )
        if self.verbose_logs:
            print(f"Invoking LLM ({self.model})...")
        response = self.llm.invoke(messages)
        raw_code = self._extract_content(response)
        raw_code = self._fix_formatting(raw_code)
        return raw_code

    def code_with_examples(self, game_plan: str, example_code: Optional[str] = None, design_spec: str = "") -> str:
        """Generate code with reference example (optional)."""
        if example_code:
            combined_plan = f"{game_plan}\n\nReference Example:\n{example_code[:3000]}"
            return self.code(combined_plan, design_spec)
        return self.code(game_plan, design_spec)

    def fix_code(self, existing_code: str, fix_instructions: str, context: str = "") -> str:
        """Apply targeted or full-file code fixes."""
        if context:
            prompt = ChatPromptTemplate.from_template(FIX_FUNCTION_PROMPT)
            messages = prompt.format_messages(
                existing_code=existing_code,
                fix_instructions=fix_instructions,
                context=context
            )
        else:
            prompt = ChatPromptTemplate.from_template(FIX_CODE_PROMPT)
            messages = prompt.format_messages(
                existing_code=existing_code,
                fix_instructions=fix_instructions
            )

        if self.verbose_logs:
            print(f"Applying {'function-level' if context else 'full-file'} fix...")
        response = self.llm.invoke(messages)
        fixed_code = self._extract_content(response)
        return self._fix_formatting(fixed_code)

    def _extract_content(self, response) -> str:
        """Safely extract model output and strip thinking tokens."""
        content = getattr(response, 'content', str(response))
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        content = self._strip_thinking_tokens(content)
        return content.strip()

    def _strip_thinking_tokens(self, content: str) -> str:
        """Strip Qwen 3's <think>...</think> blocks from response."""
        import re
        clean = re.sub(r'<(?:think|thinking)>.*?</(?:think|thinking)>', '', content, flags=re.DOTALL | re.IGNORECASE)
        return clean.strip()

    def _fix_formatting(self, code: str) -> str:
        """Repair common LLM formatting issues."""
        import re
        # Ensure closing HTML
        if not code.strip().endswith("</html>"):
            code += "\n</html>"
        # Fix misjoined JS methods
        code = re.sub(
            r'\}\s*\n(\s*)((?:async\s+)?[a-zA-Z_]\w*\s*\()',
            r'}\n\n\1\2',
            code
        )
        return code.strip()

    def get_metadata(self):
        """Retrieve metadata about last parsed game."""
        return self.parser.get_metadata()