import os
import json
import time
import random
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# --- Advanced Coder Topics ---
CODER_TOPICS = [
    "Grid-based pathfinding (A*) with obstacle avoidance",
    "Finite State Machine (FSM) for a Boss with 3 distinct phases",
    "Inventory system with weight limits, stacking, and drag-drop UI",
    "Quadratic Bezier curves for smooth arrow/projectile trajectories",
    "Dynamic lighting system using 2D Canvas globalCompositeOperation",
    "Procedural infinite terrain generation using Perlin Noise",
    "Object Pooling system to optimize bullet/particle performance",
    "Collision detection using SAT (Separating Axis Theorem) for polygons",
    "Game Save/Load manager with versioning and data encryption",
    "Entity-Component-System (ECS) architecture for thousands of units"
]

class CoderDataGenerator:
    def __init__(self, provider="groq"):
        self.provider = provider
        if provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_example(self, topic: str, complexity: str):
        system_prompt = "You are an expert Game Developer. You write high-performance, production-ready ES6+ JavaScript code."
        
        user_prompt = f"""Task: Create a complex implementation of {topic}.
        Complexity: {complexity}
        
        Requirements:
        1. Use ES6 Class syntax.
        2. Include a 'Plan' section explaining the logic.
        3. Provide the full Implementation.
        4. Add a 'Usage' example at the end.

        Return ONLY valid JSON:
        {{
            "instruction": "A clear coding challenge",
            "content": "A string containing the Plan, Implementation, and Usage"
        }}
        """

        try:
            if self.provider == "groq":
                resp = self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[{"role": "user", "content": f"{system_prompt}\n{user_prompt}"}],
                response_format={"type": "json_object"}
                )
                return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"Error generating: {e}")
            return None

if __name__ == "__main__":
    generator = CoderDataGenerator(provider="groq")
    output_file = "game_coding_dataset.jsonl"
    
    # Levels of complexity to ensure diversity
    complexities = ["Intermediate", "Advanced", "Professional/Architect"]
    
    with open(output_file, "a") as f:
        for i in range(200):
            topic = random.choice(CODER_TOPICS)
            comp = random.choice(complexities)
            print(f"[{i+1}/200] Generating {comp} code for: {topic}")
            
            data = generator.generate_example(topic, comp)
            if data:
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are an expert Game Developer. Write complete, high-performance ES6 JavaScript."},
                        {"role": "user", "content": data["instruction"]},
                        {"role": "assistant", "content": data["content"]}
                    ]
                }
                f.write(json.dumps(entry) + "\n")
                time.sleep(3) # Respect Gemini Pro rate limits