"""
Prompt templates for the game development agents.
These carefully crafted prompts guide the AI through game creation.
"""

PLANNER_PROMPT = """You are a professional game designer and architect. Your task is to create a detailed game design document.

## User's Game Request:
{user_request}

## Your Task:
Create a comprehensive game plan that includes:

### 1. Game Overview
- Game title
- Brief description
- Target audience

### 2. Core Mechanics
- Main gameplay loop
- Player controls (keyboard/mouse)
- Win/lose conditions
- Scoring system (if applicable)

### 3. Visual Design
- Color scheme suggestions
- Layout description
- UI elements needed

### 4. Technical Requirements
- Key JavaScript functions needed
- Data structures required
- Game state management approach

### 5. Features List
- Core features (must have)
- Nice-to-have features

Be specific and detailed. This plan will be used by a developer to implement the game.

## Game Design Document:
"""

CODER_PROMPT = """You are an expert web game developer. Create a complete, playable HTML game based on this game plan.

## Game Plan:
{game_plan}

## Requirements:
1. Create a SINGLE, self-contained HTML file
2. Include ALL CSS styles inline within <style> tags
3. Include ALL JavaScript within <script> tags
4. The game MUST be fully playable immediately when opened in a browser
5. Use modern, attractive styling with gradients and shadows
6. Implement smooth animations where appropriate
7. Make it responsive and visually appealing

## Code Quality Standards:
- Clean, well-commented code
- Proper error handling
- Smooth 60fps gameplay (use requestAnimationFrame for animations)
- Clear visual feedback for user actions

## Output Format:
Provide the complete game code in exactly this format:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Meta tags, title, and styles here -->
</head>
<body>
    <!-- Game HTML structure here -->
    <script>
        // Game JavaScript here
    </script>
</body>
</html>
```

Now create the complete game:
"""

VALIDATOR_PROMPT = """You are a QA engineer reviewing a game. Check the following game code for issues:

## Game Code:
{game_code}

## Check for:
1. Syntax errors in HTML/CSS/JavaScript
2. Missing game functionality
3. Potential runtime errors
4. UI/UX issues

## Validation Report:
Provide a brief report on any issues found. If the code is good, say "VALIDATED".
"""
