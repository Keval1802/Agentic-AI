"""
Unsloth Fine-Tuning Script for Game Coding Model
=================================================
Fine-tune Qwen or Llama models on game coding data using Unsloth (4x faster than standard).

REQUIREMENTS:
1. NVIDIA GPU with 8GB+ VRAM (RTX 3060, 3070, 3080, 4070, etc.)
2. Windows WSL2 or Linux (Unsloth doesn't work on native Windows)
3. CUDA toolkit installed

SETUP INSTRUCTIONS:
-------------------
# Option A: If you have WSL2 on Windows (Recommended)
1. Install WSL2: wsl --install -d Ubuntu
2. Open WSL2 terminal: wsl
3. Install CUDA in WSL2: 
   sudo apt update && sudo apt install -y nvidia-cuda-toolkit
4. Install Python packages:
   pip install unsloth transformers datasets accelerate bitsandbytes

# Option B: Use Google Colab (Free GPU)
1. Copy this script to Colab
2. Enable GPU: Runtime > Change runtime type > T4 GPU
3. Run: !pip install unsloth

USAGE:
------
# From WSL2 or Linux:
python training/finetune_unsloth.py --dataset training/game_coding_dataset.jsonl --output models/game-coder

# Quick test with small model:
python training/finetune_unsloth.py --model unsloth/Qwen2.5-Coder-1.5B-bnb-4bit --epochs 1
"""

import os
import sys
import argparse
import json
from typing import Optional


def check_environment():
    """Check if the environment is suitable for training."""
    # Check if running on Windows (not WSL)
    if sys.platform == "win32":
        print("""
╔════════════════════════════════════════════════════════════════╗
║  ❌ Unsloth doesn't work on native Windows!                   ║
╠════════════════════════════════════════════════════════════════╣
║  OPTIONS:                                                      ║
║  1. Use WSL2: wsl --install -d Ubuntu                         ║
║     Then run this script in WSL                                ║
║  2. Use Google Colab (free GPU): bit.ly/unsloth-colab         ║
║  3. Use vast.ai or runpod.io for cloud GPU                    ║
╚════════════════════════════════════════════════════════════════╝
""")
        return False
    
    # Check for GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ No GPU detected. Training will be very slow.")
            print("   For GPU access, use Google Colab or cloud services.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
    except ImportError:
        print("⚠️ PyTorch not installed. Run: pip install torch")
        return False
    
    return True


def install_dependencies():
    """Install Unsloth and dependencies."""
    try:
        import unsloth
        print("✅ Unsloth is installed")
        return True
    except ImportError:
        print("📦 Installing Unsloth (this may take a few minutes)...")
        os.system("pip install unsloth transformers datasets accelerate bitsandbytes peft trl")
        return False


def load_dataset(dataset_path: str):
    """Load the JSONL training dataset."""
    from datasets import Dataset
    
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"📊 Loaded {len(data)} examples from {dataset_path}")
    
    # Convert to chat format for training
    formatted_data = []
    for item in data:
        messages = item.get("messages", [])
        # Format as conversation
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        formatted_data.append({"text": "\n".join(text_parts)})
    
    return Dataset.from_list(formatted_data)


def finetune(
    dataset_path: str,
    output_dir: str,
    model_name: str = "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
):
    """Fine-tune a model using Unsloth with LoRA."""
    
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    print(f"\n🚀 Starting fine-tuning with Unsloth")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    print(f"   Epochs: {epochs}")
    print("-" * 50)
    
    # Load model with 4-bit quantization
    print("📥 Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    print("🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir,
        save_strategy="epoch",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Train
    print("\n🏋️ Starting training...")
    trainer_stats = trainer.train()
    
    # Save model
    print(f"\n💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also save as GGUF for Ollama
    print(f"📦 Converting to GGUF format...")
    try:
        model.save_pretrained_gguf(
            f"{output_dir}_gguf",
            tokenizer,
            quantization_method="q4_k_m"
        )
        print(f"✅ GGUF saved to {output_dir}_gguf")
    except Exception as e:
        print(f"⚠️ GGUF conversion failed: {e}")
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║  ✅ Training Complete!                                         ║
╠════════════════════════════════════════════════════════════════╣
║  Model saved to: {output_dir:<40} ║
║                                                                ║
║  TO USE THE MODEL:                                             ║
║  1. With Unsloth (Python):                                     ║
║     from unsloth import FastLanguageModel                      ║
║     model, tokenizer = FastLanguageModel.from_pretrained(      ║
║         "{output_dir}")                                        ║
║                                                                ║
║  2. With Ollama (GGUF):                                        ║
║     ollama create game-coder -f {output_dir}_gguf/Modelfile    ║
║     ollama run game-coder                                      ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    return trainer_stats


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for game coding")
    parser.add_argument("--dataset", type=str, default="training/game_coding_dataset.jsonl",
                       help="Path to training dataset")
    parser.add_argument("--output", type=str, default="models/game-coder",
                       help="Output directory for the fine-tuned model")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-Coder-7B-bnb-4bit",
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        return 1
    
    # Install dependencies if needed
    install_dependencies()
    
    # Run fine-tuning
    try:
        finetune(
            dataset_path=args.dataset,
            output_dir=args.output,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_seq_length=args.max_length,
            lora_r=args.lora_r,
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
