"""
Game Prompt Templates - Structured prompts for specific game types.
Uses detailed specifications and code examples to get complete implementations.
"""

# Base template with anti-placeholder rules
BASE_RULES = """
## ABSOLUTE REQUIREMENTS - FAILURE MEANS REJECTION:

### TOP PRIORITY:
- NO Uncaught ReferenceError from ANY HTML button (all handlers/functions must exist and be bound)

### COMPLETENESS RULES (MANDATORY):
1. **EVERY function MUST have FULL implementation** - NO placeholder comments
2. **FORBIDDEN phrases** - NEVER write these:
   - "// logic here"
   - "// implement this"
   - "// TODO"
   - "// add code here"
   - "// your code"
   - "// complete this"
3. If a function exists, it MUST contain working code that does something
4. ALL game mechanics must be FULLY implemented

## OUTPUT FORMAT:
- Provide THREE code blocks: HTML, CSS, and JavaScript
- NO explanatory text, ONLY code
- ZERO explanations, ZERO reasoning, ZERO thinking — output ONLY code
- HTML block must start with <!DOCTYPE html>
"""

def get_game_prompt(game_type: str, custom_requirements: str = "") -> str:
    """
    Get a structured prompt for a specific game type.
    
    Args:
        game_type: Type of game (chess, tictactoe, snake, breakout, pong, checkers, memory)
        custom_requirements: Additional requirements to append
        
    Returns:
        Complete structured prompt for the game
    """
    # Always return the generic template (no static per-game prompts)
    return f"""Create a COMPLETE, FULLY FUNCTIONAL {game_type} game in HTML5/CSS/JavaScript.

## REQUIREMENTS:
- Implement ALL core game mechanics completely
- Visual feedback for all player actions
- Start, pause, and restart functionality
- Score/state display
- Keyboard and/or mouse controls
- UI/UX design with clear visual hierarchy

{custom_requirements if custom_requirements else ''}

{BASE_RULES}
"""

def detect_game_type(user_input: str) -> str:
    """Detect game type from user input."""
    # Static categorization removed; always treat as custom.
    return "custom"
