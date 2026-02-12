"""
Planner Agent - Creates detailed game specifications with implementation logic.
Uses Groq with Llama 3.3 70B for excellent reasoning capability.
Outputs structured requirements with code examples for coder agent.
"""

import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

ENHANCED_PLANNER_PROMPT = """You are a game architect instructing a coder. Specify WHAT to build.

## User Request:
{user_request}

## OUTPUT FORMAT:

### [GAME]
Title | Genre | Core mechanic in one sentence

### [OBJECTS]
List every game entity with: name, emoji, properties (x, y, speed, size, lives), behavior

### [MECHANICS]
- Movement: which keys do what, speed values, boundary rules
- Collision: which objects collide, what happens on each collision
- Scoring: what actions give points, exact values
- Win/Lose: exact conditions for winning and losing

### [CONTROLS]
List all key bindings (movement, action, pause, restart)

### [STATES]
Game state flow: START → PLAYING → PAUSED → GAME_OVER

### [RESOURCES]
- Use: Canvas API for rendering, requestAnimationFrame for game loop
- Use: localStorage for high score persistence
- Use: class-based architecture (single Game class)
- Use: deltaTime for frame-rate independent movement
- For board games: Canvas MUST be SQUARE (width = height = 800). NOT 800x600!
- For board games: each side MUST have visually DISTINCT piece symbols
- For chess: use Unicode chess pieces ♔♕♖♗♘♙ (white) vs ♚♛♜♝♞♟ (black)
- For chess: MUST implement pawn promotion, check, checkmate
- For chess: do NOT show "Lives" - use turn indicator and captured pieces
- For physics games: specify gravity, friction, bounce values

## RULES:
- Be SPECIFIC with numbers (speed=300, lives=3, points=+10)
- NO code, NO examples - just plain text instructions
- 100-200 words max
"""

# Game-specific mandatory constraints injected into plans
GAME_CONSTRAINTS = {
    'chess': """
### [MANDATORY CONSTRAINTS - CHESS]
- Canvas MUST be 800x800 (SQUARE, not 800x600!)
- White pieces: ♔♕♖♗♘♙  |  Black pieces: ♚♛♜♝♞♟ (DO NOT substitute with emojis)
- MUST implement: pawn promotion (auto-queen), check detection, checkmate
- SHOULD implement: en passant, castling, stalemate
- HUD: turn indicator + captured pieces (NO "Lives" counter)
- AI opponent with capture-priority logic
""",
    'checkers': """
### [MANDATORY CONSTRAINTS - CHECKERS]
- Canvas MUST be 800x800 (SQUARE board)
- Red pieces vs Black pieces (distinct colors)
- Kings marked with crown symbol
- Mandatory capture rule: if a jump is available, player MUST take it
- Multi-jump support in single turn
""",
}


class PlannerAgent:
    """
    Game Planning Agent - Uses Groq with Llama 3.3 70B for best reasoning.
    Outputs detailed technical specifications with implementation hints.
    """
    
    def __init__(self, model: Optional[str] = None, use_groq: bool = True):
        """Initialize the Planner Agent."""
        self.use_groq = use_groq and os.getenv("GROQ_API_KEY")
        
        if self.use_groq:
            # Use Groq with Llama 3.3 70B (Best for reasoning)
            self.model = model or os.getenv("GROQ_PLANNER_MODEL", "llama-3.3-70b-versatile")
            self.llm = ChatGroq(
                model=self.model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.5,  # Lower for more structured output
                max_tokens=2048,  # Increased for detailed specs
            )
            print(f"📋 PlannerAgent using Groq: {self.model}")
        else:
            # Fallback to Gemini
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.5,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
            print(f"🔄 PlannerAgent using Gemini: {self.model}")
        
        self.prompt = ChatPromptTemplate.from_template(ENHANCED_PLANNER_PROMPT)
    
    def plan(self, user_request: str) -> str:
        """Generate detailed game specification with game-specific constraints."""
        messages = self.prompt.format_messages(user_request=user_request)
        response = self.llm.invoke(messages)
        plan = self._extract_content(response)
        
        # Inject game-specific mandatory constraints
        plan = self._inject_constraints(user_request, plan)
        
        return plan
    
    def _extract_content(self, response) -> str:
        """Safely extract content from response."""
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                return "\n".join(str(part) for part in content)
            return str(content)
        return str(response)
    
    def plan_with_refinement(self, user_request: str, feedback: Optional[str] = None) -> str:
        """Generate or refine a game plan based on feedback."""
        if feedback:
            enhanced_request = f"""
Original Request: {user_request}

User Feedback: {feedback}

Please create an updated, more detailed game plan addressing this feedback.
"""
            return self.plan(enhanced_request)
        return self.plan(user_request)
    
    def _inject_constraints(self, user_request: str, plan: str) -> str:
        """Inject game-specific mandatory constraints into the plan."""
        request_lower = user_request.lower()
        
        for game_type, constraints in GAME_CONSTRAINTS.items():
            if game_type in request_lower:
                plan += constraints
                print(f"📋 Planner: Injected {game_type} mandatory constraints")
                break
        
        return plan
