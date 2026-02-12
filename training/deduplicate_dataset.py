"""
Deduplicate Training Dataset
=============================
Removes duplicate or near-duplicate entries from the training dataset.
"""

import json
import hashlib
from pathlib import Path


def deduplicate_dataset(input_file: str, output_file: str = None):
    """Remove duplicate training examples based on instruction similarity."""
    
    if output_file is None:
        output_file = input_file.replace('.jsonl', '_deduped.jsonl')
    
    print(f"📂 Loading: {input_file}")
    
    # Load all examples
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"📊 Total examples: {len(examples)}")
    
    # Deduplicate based on instruction text
    seen_instructions = set()
    unique_examples = []
    duplicates = 0
    
    for ex in examples:
        # Extract instruction from messages
        instruction = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
                break
        
        # Create a normalized key (lowercase, strip, first 200 chars)
        key = instruction.lower().strip()[:200]
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        if key_hash not in seen_instructions:
            seen_instructions.add(key_hash)
            unique_examples.append(ex)
        else:
            duplicates += 1
    
    print(f"🔄 Duplicates found: {duplicates}")
    print(f"✅ Unique examples: {len(unique_examples)}")
    
    # Save deduplicated dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in unique_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved to: {output_file}")
    
    return len(unique_examples)


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "training/game_coding_dataset.jsonl"
    
    # Also check root directory
    if not Path(input_file).exists():
        input_file = "game_coding_dataset.jsonl"
    
    if Path(input_file).exists():
        deduplicate_dataset(input_file)
    else:
        print(f"❌ File not found: {input_file}")
