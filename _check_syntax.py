import ast
import sys

files = [
    'app/agents/planner.py',
    'app/agents/coder.py',
    'app/agents/validator.py',
    'app/graphs/coder_validator_graph.py',
    'app/chains/game_chain.py',
]

all_ok = True
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            ast.parse(fh.read())
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"FAIL: {f} -> {e}")
        all_ok = False

if all_ok:
    print("\nAll 5 files: syntax OK")
else:
    print("\nSome files have syntax errors!")
    sys.exit(1)
