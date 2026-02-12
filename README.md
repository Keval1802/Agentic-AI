# Agentic AI Game Developer

An AI-powered game development agent that creates playable HTML/CSS/JavaScript games from natural language descriptions.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   - Copy `.env` and add your API key
   - Supports OpenAI (GPT-4) or Google Gemini

3. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open browser:**
   Navigate to `http://localhost:8000`

## Usage

1. Enter a game description (e.g., "Create a snake game")
2. Click "Generate Game"
3. Wait for the AI to create your game
4. Play the generated game in the preview
5. Download the HTML file

## Architecture

```
User Request → Planner Agent → Coder Agent → Parser → Playable Game
```

## Supported Game Types

- Puzzle games (Tic-tac-toe, Memory, Sudoku)
- Arcade games (Snake, Breakout, Pong)
- Board games (Chess, Checkers)
- Casual games (Clicker, Idle games)
