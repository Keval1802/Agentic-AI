"""
Designer Agent - Creates visual design specifications and UI/UX requirements.
Uses Groq with Llama 3.1 8B for fast design generation.
Outputs CSS design system + UI component specifications.
"""

import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

ENHANCED_DESIGNER_PROMPT = """You are a UI designer instructing a coder. Specify HOW it should look.

## Game Plan:
{game_plan}



### [THEME]
- Visual theme (2–6 words)
- Mood (1–3 words)

### [COLORS]
List CSS variable names with exact hex values:
- primary, secondary, bg-dark, bg-light, text, success, danger, accent

### [TYPOGRAPHY]
- Google Font import URL
- Title size (px), HUD size (px), body size (px), button size (px)
- Font weights 

### [LAYOUT]
- Canvas/container size rules (min/max or responsive)
- UI placement: score, lives, level, timer (exact corners/positions)
- Game size must be based on window size like full screen = 100% height and 100% width and in centralized position.
- Panel widths/heights (px or %)

### [SCREENS]
Describe layout for each screen in plain text:
- Start Screen: layout, title style, button style, background treatment
- Game HUD: layout, alignment, spacing, icons
- Pause Overlay: layout, backdrop opacity (0–1), blur (px), text style
- Game Over: layout, score, high score, CTA buttons

### [COMPONENTS]
- Buttons: radius (px), shadow (exact), hover effect
- Cards/Panels: radius, border, shadow
- Progress/Health bar: height (px), colors

### [ANIMATIONS]
List which animations to use and where:
- fadeIn for overlays
- pulse for score changes
- shake for damage/collisions

### [STYLE]
- Button hover effect (scale, color change, shadow)
- Background gradient direction and colors
- Border radius, box shadow values
- Transition duration

## RULES:
- Use EXACT values (hex colors, px sizes, seconds)
- NO code, NO CSS blocks - just plain text instructions
- 80–140 words max
- ZERO explanations, ZERO reasoning, ZERO thinking — output ONLY the design spec
- Design must feel PREMIUM and modern — glassmorphism, gradients, micro-animations, curated palettes

/no_think
"""

class DesignerAgent:
    """
    Game Design Agent - Uses Groq with Llama 3.1 8B for fast design generation.
    Creates complete visual specifications including colors, animations, and UI components.
    """
    
    def __init__(self, model: Optional[str] = None, use_groq: bool = True):
        """Initialize the Designer Agent."""
        self.use_groq = use_groq and os.getenv("GROQ_API_KEY")
        
        if self.use_groq:
            # Use Groq with Llama 3.1 8B (Fast for design)
            self.model = model or os.getenv("GROQ_DESIGNER_MODEL", "llama-3.1-8b-instant")
            self.llm = ChatGroq(
                model=self.model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7,
                max_tokens=2048,  # Increased for detailed specs
            )
            print(f"🎨 DesignerAgent using Groq: {self.model}")
        else:
            # Fallback to Gemini
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
            print(f"🔄 DesignerAgent using Gemini: {self.model}")
        
        self.prompt = ChatPromptTemplate.from_template(ENHANCED_DESIGNER_PROMPT)
    
    def design(self, game_plan: str) -> str:
        """Generate comprehensive visual design specification."""
        messages = self.prompt.format_messages(game_plan=game_plan)
        response = self.llm.invoke(messages)
        return self._extract_content(response)
    
    def _extract_content(self, response) -> str:
        """Safely extract content from response."""
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                return "\n".join(str(part) for part in content)
            return str(content)
        return str(response)
