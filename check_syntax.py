import ast
import sys

files = [
    'app/main.py',
    'app/agents/coder.py', 
    'app/chains/game_chain.py'
]

all_ok = True
for f in files:
    try:
        with open(f, encoding='utf-8') as fh:
            ast.parse(fh.read())
        print(f'{f}: SYNTAX OK')
    except SyntaxError as e:
        print(f'{f}: SYNTAX ERROR - {e}')
        all_ok = False

sys.exit(0 if all_ok else 1)
