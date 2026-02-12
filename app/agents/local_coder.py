"""
Local Fine-Tuned Model Coder Agent
===================================
Uses your fine-tuned LoRA model for game code generation.

Requirements:
- pip install transformers peft accelerate bitsandbytes torch
- Your fine-tuned model in models/game-coder-lora/

For CPU usage (slow but works):
- Set USE_CPU=True in .env

For Ollama (recommended for CPU):
- Install Ollama and run: ollama run game-coder
"""

import os
import torch
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LocalCoderAgent:
    """Uses locally fine-tuned model for code generation."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the local coder agent.
        
        Args:
            model_path: Path to the fine-tuned LoRA model (defaults to .env setting)
        """
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH", "model/game-coder-lora")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if using Ollama instead
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        
        if self.use_ollama:
            print("🦙 Using Ollama for local inference")
            self._setup_ollama()
        else:
            print(f"🔧 Using Transformers on {self.device}")
            if self.device == "cpu":
                print("⚠️ CPU inference will be slow. Consider using Ollama instead.")
            self._load_model()
    
    def _setup_ollama(self):
        """Setup Ollama client."""
        try:
            import ollama
            self.ollama_client = ollama
            self.ollama_model = os.getenv("OLLAMA_LOCAL_MODEL", "game-coder")
            print(f"✅ Ollama ready with model: {self.ollama_model}")
        except ImportError:
            print("❌ Ollama not installed. Run: pip install ollama")
            raise
    
    def _load_model(self):
        """Load the fine-tuned model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            print(f"📥 Loading model from {self.model_path}...")
            
            # Base model name (must match what was fine-tuned)
            base_model_name = "Qwen/Qwen2.5-Coder-1.5B"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load base model
            print("📥 Loading base model...")
            if self.device == "cuda":
                # GPU: Use 4-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # CPU: Load in float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            
            # Load LoRA adapter
            print("🔧 Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            
            # Merge for faster inference (optional but recommended)
            print("⚡ Merging adapter for faster inference...")
            self.model = self.model.merge_and_unload()
            
            self.model.eval()
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate code from a prompt."""
        
        if self.use_ollama:
            return self._generate_ollama(prompt)
        else:
            return self._generate_transformers(prompt, max_tokens)
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama."""
        response = self.ollama_client.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 2048,
            }
        )
        return response["response"]
    
    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate using Transformers."""
        
        # Format prompt
        formatted = f"""<|im_start|>system
You are an expert game developer. Create complete, working HTML5 games.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(formatted, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        
        return response.strip()
    
    def code(self, game_plan: str, design_spec: str = "") -> str:
        """
        Generate game code (compatible with existing coder interface).
        
        Args:
            game_plan: The game plan from planner
            design_spec: The design specification from designer
            
        Returns:
            Generated HTML/CSS/JS code
        """
        prompt = f"""Create a complete HTML5 game based on this plan:

## Game Plan:
{game_plan}

## Design Specification:
{design_spec}

Generate the complete game with:
1. HTML (index.html) with canvas and overlays
2. CSS (style.css) with modern styling
3. JavaScript (script.js) with game logic

Output the code in separate blocks: ```html, ```css, ```javascript"""

        return self.generate(prompt)


def test_local_model():
    """Test the local model."""
    print("\n" + "="*50)
    print("🧪 Testing Local Fine-Tuned Model")
    print("="*50 + "\n")
    
    try:
        agent = LocalCoderAgent()
        
        prompt = "Create a simple Snake game with arrow key controls"
        print(f"📝 Prompt: {prompt}\n")
        print("⏳ Generating (this may take a while on CPU)...\n")
        
        response = agent.generate(prompt)
        
        print("📄 Generated Code:")
        print("-"*50)
        print(response[:2000])  # First 2000 chars
        print("-"*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_local_model()
