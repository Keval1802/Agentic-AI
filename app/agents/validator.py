"""
Validator Agent - Quality assurance for generated games.
Primary: NVIDIA NIM with Llama 3.1 Nemotron Ultra 253B (Instruction Following King).
Fallback: Groq → Gemini.
"""

import os
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

VALIDATOR_PROMPT = """You are a STRICT game code reviewer. Find ALL bugs. ReferenceError = INSTANT FAIL.

## Full Game Code (Single HTML File):
{game_code}

## 🚨 ZERO TOLERANCE - REFERENCEERROR:
ANY of these = CRITICAL issue:
- Variable used before declaration
- Missing `this.` on class properties (score vs this.score)
- Method called but never defined
- `function()` callback inside class (breaks `this`)
- Arrow function NOT used in addEventListener

## CHECK JAVASCRIPT:
1. Every function has REAL logic (no stubs, no empty bodies)"""
VALIDATOR_PROMPT = """You are a Senior Code Auditor.
Your job is to CRITIQUE the provided game code. Do NOT be polite. Be rigorous.

## STRUCTURED AUDIT INSTRUCTIONS:
Review the code for:
1. **Logic Errors**: Infinite loops, broken state management, unbeatable conditions.
2. **Security**: XSS vulnerabilities, unsafe HTML injection.
3. **Edge Cases**: Zero inputs, max inputs, window resizing, mobile touch support.
4. **Completeness**: Are all planned features implemented?
5. **Structure**: Is it a SINGLE valid HTML file with inline CSS (<style>) and JS (<script>)?

## OUTPUT FORMAT (JSON ONLY):
You MUST return a valid JSON object. No markdown, no preambles.
{
    "logic_errors": ["Error 1 description", "Error 2..."],
    "security_score": (1-10 integer),
    "edge_cases_missed": ["Missed case 1", ...],
    "status": "APPROVED" | "REJECTED",
    "fix_directive": "Specific instructions for the Coder to fix the REJECTED status. If APPROVED, leave empty."
}

## CRITICAL RULES:
- If `security_score` < 9, status MUST be "REJECTED".
- If any `logic_errors` exist, status MUST be "REJECTED".
- If `ReferenceError` is likely, status MUST be "REJECTED".
- `fix_directive` must be technical and actionable (e.g., "Add bounds check to move() function", not "Fix bugs").
"""


@dataclass
class ValidationResult:
    """Result of game code validation."""
    is_valid: bool
    score: int
    issues: List[Dict]
    summary: str
    fixed_code: Optional[str] = None


class ValidatorAgent:
    """
    Game Validation Agent - NVIDIA NIM with Llama 3.1 Nemotron Ultra 253B.
    Instruction Following King: NVIDIA's Nemotron Ultra is their flagship model 
    for strict constraint enforcement, coding accuracy, and bug identification.
    ReferenceError = ZERO TOLERANCE.
    Fallback: Groq → Gemini.
    """
    
    def __init__(self, model: Optional[str] = None, use_groq: bool = True):
        """Initialize the Validator Agent.
        
        Priority: NVIDIA NIM (Nemotron Ultra 253B) → Groq → Gemini.
        """
        self.use_nvidia = os.getenv("USE_NVIDIA", "").lower() == "true" and os.getenv("NVIDIA_API_KEY")
        self.use_groq = use_groq and os.getenv("GROQ_API_KEY")

        if model and model.startswith("qwen/"):
             self.use_nvidia = False
        
        if self.use_nvidia:
            # PRIMARY: NVIDIA NIM with Nemotron Ultra 253B (Instruction Following King)
            self.model = model or "nvidia/llama-3.1-nemotron-ultra-253b-v1"
            self.llm = ChatOpenAI(
                model=self.model,
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=0.1,
                max_tokens=4096
            )
            print(f"🎯 ValidatorAgent using NVIDIA NIM: {self.model} (Instruction Following King)")
        elif self.use_groq:
            # FALLBACK 1: Groq with Qwen 3 32B (as requested)
            self.model = model or "qwen/qwen3-32b"
            self.llm = ChatGroq(
                model=self.model,
                api_key=os.getenv("GROQ_API_KEY_2"),
                temperature=0.1,
                max_tokens=2048,
            )
            print(f"🔍 ValidatorAgent using Groq: {self.model}")
        else:
            # FALLBACK 2: Gemini
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1,
                max_output_tokens=1024,
                convert_system_message_to_human=True
            )
            print(f"🔄 ValidatorAgent using Gemini: {self.model}")
        
        self.prompt = ChatPromptTemplate.from_template(VALIDATOR_PROMPT)
    
    def _extract_js(self, code: str) -> str:
        """Extract JavaScript from HTML or markdown code blocks."""
        script_match = re.search(r'<script[^>]*>(.*?)</script>', code, re.DOTALL | re.IGNORECASE)
        if script_match:
            return script_match.group(1).strip()
        js_match = re.search(r'```javascript\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
        if js_match:
            return js_match.group(1).strip()
        return code
    
    def _extract_html(self, code: str) -> str:
        """Extract HTML structure (without script/style content)."""
        html = code
        # Remove script content but keep tags
        html = re.sub(r'<script[^>]*>.*?</script>', '<script></script>', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove style content but keep tags
        html = re.sub(r'<style[^>]*>.*?</style>', '<style></style>', html, flags=re.DOTALL | re.IGNORECASE)
        return html
    
    def _extract_css(self, code: str) -> str:
        """Extract CSS from HTML or markdown code blocks."""
        style_match = re.search(r'<style[^>]*>(.*?)</style>', code, re.DOTALL | re.IGNORECASE)
        if style_match:
            return style_match.group(1).strip()
        css_match = re.search(r'```css\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
        if css_match:
            return css_match.group(1).strip()
        return ""

    def _strip_strings_and_comments(self, code: str) -> str:
        """Remove strings/comments to reduce false positives in regex checks."""
        code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)
        code = re.sub(r'//.*', ' ', code)
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
        # Fix: handle single quotes better to avoid stripping contractions
        code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code) 
        code = re.sub(r'`(?:\\.|[^`])*`', '``', code, flags=re.DOTALL)
        return code

    def _find_line_number(self, code: str, target: str) -> int:
        """Find the first line number containing the target string."""
        for idx, line in enumerate(code.splitlines(), 1):
            if target in line:
                return idx
        return 0
    
    def _map_code_structure(self, code: str) -> Dict[str, int]:
        """
        Scan code line-by-line to map function names to valid line numbers.
        Returns: {'functionName': line_number}
        """
        structure = {}
        lines = code.split('\n')
        
        # Regex to find function definitions
        # Matches: function foo(), foo = function(), foo() {, async foo() {
        patterns = [
            r'function\s+(\w+)\s*\(',
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)',
            r'^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{',  # Method start of line
            r'\n\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{'   # Method after newline
        ]
        
        for idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    if func_name not in structure:
                        structure[func_name] = idx
                        
        return structure

    def validate(self, game_code: str, contract: str = "") -> ValidationResult:
        """Validate ALL parts: HTML structure, CSS styling, and JavaScript logic."""
        # Extract all 3 parts
        js_code = self._extract_js(game_code)
        html_code = self._extract_html(game_code)
        css_code = self._extract_css(game_code)
        
        # === STRUCTURAL CHECKS (regex-based, no LLM) ===
        js_issues = self._validate_js(js_code)
        html_issues = self._validate_html(html_code)
        css_issues = self._validate_css(css_code)

        # === CONTRACT ADHERENCE CHECK ===
        contract_issues = []
        if contract:
            contract_issues = self._validate_contract(js_code, contract)

        # === LINE-BY-LINE SCAN (deterministic) ===
        line_issues = []
        line_issues += self._scan_placeholder_lines(html_code, "html")
        line_issues += self._scan_placeholder_lines(css_code, "css")
        line_issues += self._scan_placeholder_lines(js_code, "js")
        
        # === LLM DEEP VALIDATION (full code) ===
        llm_result = self._llm_validation(game_code)
        
        # Combine ALL issues
        all_issues = js_issues + html_issues + css_issues + contract_issues + line_issues + llm_result.get('issues', [])
        
        # Strict Mode: Zero Tolerance for both Critical Errors AND Warnings
        # User feedback: "warning is also not acceptable"
        
        # Count all issues equally
        critical_count = len([i for i in all_issues if i.get('severity') == 'critical'])
        warning_count = len([i for i in all_issues if i.get('severity') == 'warning'])
        total_issues = len(all_issues)
        
        # FAIL if ANY issue exists (ReferenceError, Logic Error, OR Warning)
        is_valid = total_issues == 0 and len(js_code) > 200
        
        # Score heavily penalized for ANY issue
        # Previously warnings were -5, now they act like critical errors (-25)
        score = 100 - (total_issues * 25)
        score = max(0, min(100, score))
        
        summary = llm_result.get('summary', f'Found {total_issues} issues ({critical_count} critical, {warning_count} warnings)')
        if not is_valid and total_issues > 0:
            summary = f"FAILED: {summary} (Zero Tolerance Mode)"
            
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=all_issues,
            summary=summary,
            fixed_code=None
        )

    def _scan_placeholder_lines(self, code: str, kind: str) -> List[Dict]:
        """Scan code line-by-line to flag placeholder-only lines."""
        issues: List[Dict] = []
        if not code:
            return issues

        placeholders = [f"existing {kind}", f"exesisting {kind}", f"exesting {kind}"]
        for idx, line in enumerate(code.splitlines(), start=1):
            lowered = line.strip().lower()
            if not lowered:
                continue
            # Normalize to avoid false positives in long lines
            normalized = re.sub(r"[^a-z]+", " ", lowered).strip()
            if normalized in placeholders:
                issues.append({
                    "severity": "warning",
                    "location": f"{kind.upper()} line {idx}",
                    "issue": f"Placeholder line detected: '{normalized}'",
                    "fix": "Remove placeholder line and keep prior code memory",
                    "code": "placeholder",
                    "file": kind,
                    "line": idx
                })

        return issues
    
    def _validate_contract(self, js_code: str, contract: str) -> List[Dict]:
        """Check if code follows the Planner's function/variable contract.
        
        Only flags MISSING contract functions as critical issues.
        Hallucinated functions are logged but NOT flagged — the Coder
        needs freedom to create helpers the planner didn't foresee.
        """
        import re
        issues = []
        
        if not contract or not js_code:
            return issues
        
        # === Parse contract for expected function names ===
        contract_functions = set()
        for line in contract.splitlines():
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('GLOBAL') or line.startswith('CLASS') or line.startswith('FUNCTIONS'):
                continue
            fn_match = re.match(r'^\s*(?:this\.)?(\w+)\s*\(', line)
            if fn_match:
                fn_name = fn_match.group(1)
                if fn_name not in ('let', 'const', 'var', 'if', 'for', 'while', 'return', 'new'):
                    contract_functions.add(fn_name)
        
        if not contract_functions:
            return issues
        
        # === Extract actual function/method names from JS code ===
        defined_methods = set(re.findall(
            r'^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', js_code, re.MULTILINE
        ))
        defined_methods.update(re.findall(
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)', js_code
        ))
        noise = {'if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'else', 'function'}
        defined_methods -= noise
        
        # === Only flag MISSING contract functions (critical) ===
        missing = contract_functions - defined_methods
        for fn in sorted(missing):
            issues.append({
                'severity': 'critical',
                'location': fn,
                'issue': f"CONTRACT BREACH: Function '{fn}' is in the contract but MISSING from code",
                'fix': f"Implement the '{fn}' function/method as specified in the contract"
            })
        
        # === Log hallucinated functions (info only, NOT issues) ===
        if len(contract_functions) >= 3:
            hallucinated = defined_methods - contract_functions
            safe_names = {'constructor', 'render', 'init', 'setup', 'main', 'animate',
                          'resize', 'destroy', 'dispose', 'reset', 'toString', 'valueOf'}
            hallucinated -= safe_names
            if hallucinated:
                print(f"ℹ️ Contract Info: {len(hallucinated)} extra functions not in contract (OK): {sorted(hallucinated)}")
        
        if missing:
            print(f"📋 Contract Check: {len(missing)} MISSING functions: {sorted(missing)}")
        else:
            print(f"✅ Contract Check: All {len(contract_functions)} contract functions found in code")
        
        return issues

    def _validate_js(self, js_code: str) -> List[Dict]:
        """Validate JavaScript code. ReferenceError = ZERO TOLERANCE."""
        issues = []
        structure = self._map_code_structure(js_code)
        
        if not js_code or len(js_code.strip()) < 50:
            issues.append({
                'severity': 'critical',
                'location': 'JavaScript',
                'issue': 'ReferenceError risk: No JavaScript code found or code is too short',
                'fix': 'Generate complete JavaScript game code'
            })
            return issues
        
        # ============ REFERENCEERROR PREVENTION (ZERO TOLERANCE) ============
        
        # 1. Check for function() callbacks inside class (breaks this = ReferenceError)
        func_callback_pattern = r'addEventListener\s*\(\s*[\'"][^"\']+[\'"]\s*,\s*function\s*\('
        for idx, line in enumerate(js_code.splitlines(), 1):
             if re.search(func_callback_pattern, line):
                issues.append({
                    'severity': 'critical',
                    'location': f'JavaScript Line {idx}',
                    'line': idx,
                    'issue': 'ReferenceError: Using function() in event listener - this will be undefined inside class',
                    'fix': 'Replace function() with arrow function () => to preserve this context'
                })
        
        # 2. Check for methods called via this. that are NOT defined in the class
        if 'class ' in js_code:
            # Find all this.methodName( calls
            called_methods = set(re.findall(r'this\.(\w+)\s*\(', js_code))
            # Find all method definitions (methodName( or methodName =)
            defined_methods = set(re.findall(r'^\s+(\w+)\s*\([^)]*\)\s*\{', js_code, re.MULTILINE))
            defined_methods.update(re.findall(r'(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)', js_code))
            # Also add constructor and common built-ins
            defined_methods.update(['constructor', 'toString', 'valueOf', 'bind', 'call', 'apply',
                                     'getElementById', 'querySelector', 'getContext', 'getItem', 'setItem',
                                     'clearRect', 'fillRect', 'strokeRect', 'fillText', 'beginPath',
                                     'arc', 'fill', 'stroke', 'drawImage', 'moveTo', 'lineTo',
                                     'addEventListener', 'removeEventListener', 'preventDefault',
                                     'push', 'pop', 'shift', 'splice', 'filter', 'map', 'forEach', 'find',
                                     'includes', 'indexOf', 'slice', 'concat', 'sort', 'reverse',
                                     'keys', 'values', 'entries', 'assign', 'freeze',
                                     'parse', 'stringify', 'floor', 'ceil', 'round', 'random', 'abs',
                                     'min', 'max', 'sqrt', 'pow', 'sin', 'cos', 'atan2', 'PI',
                                     'log', 'warn', 'error', 'now', 'toFixed', 'replace', 'split',
                                     'trim', 'toLowerCase', 'toUpperCase', 'charAt', 'substring',
                                     'startsWith', 'endsWith', 'match', 'test', 'exec'])
            
            undefined_methods = called_methods - defined_methods - set(structure.keys())
            for method in undefined_methods:
                # Find line number for the undefined method call
                line_num = 0
                for idx, line in enumerate(js_code.splitlines(), 1):
                    if f"this.{method}(" in line:
                        line_num = idx
                        break
                
                issues.append({
                    'severity': 'critical',
                    'location': f'JavaScript Line {line_num or "?"}',
                    'line': line_num,
                    'issue': f'ReferenceError: Method this.{method}() is called but NEVER defined in the class',
                    'fix': f'Define {method}() method in the class or fix the method name'
                })
        
        # 3. Check for missing this. on class properties
        if 'class ' in js_code:
            constructor_match = re.search(r'constructor\s*\([^)]*\)\s*\{(.*?)\n\s{2,4}\}', js_code, re.DOTALL)
            if constructor_match:
                constructor_body = constructor_match.group(1)
                this_props = set(re.findall(r'this\.(\w+)\s*=', constructor_body))
                
                common_props = ['score', 'lives', 'level', 'gameActive', 'paused', 'player', 'enemies', 'board']
                for prop in common_props:
                    if prop in this_props:
                        bare_pattern = rf'(?<!this\.)(?<!\w){prop}(?!\w)(?!\s*[=:])'
                        # Scan line-by-line relative to whole file
                        for idx, line in enumerate(js_code.splitlines(), 1):
                            # Skip constructor definition itself if needed, but simple check is okay
                            if 'constructor' in line: continue
                            
                            # Remove strings/comments from line to avoid false positives
                            clean_line = self._strip_strings_and_comments(line)
                            if re.search(bare_pattern, clean_line):
                                issues.append({
                                    'severity': 'critical',
                                    'location': f'JavaScript Line {idx}',
                                    'line': idx,
                                    'issue': f'ReferenceError: "{prop}" used without this. prefix - will crash at runtime',
                                    'fix': f'Use this.{prop} instead of bare {prop} on line {idx}'
                                })

        # 4. Check for event listeners registered inside hot loops (update/draw/gameLoop)
        loop_methods = ('update', 'draw', 'gameLoop', 'tick', 'loop', 'render')
        in_loop_method = None
        brace_depth = 0
        for idx, line in enumerate(js_code.splitlines(), 1):
            clean_line = self._strip_strings_and_comments(line)
            method_match = re.match(r'^\s*(?:async\s+)?(' + '|'.join(loop_methods) + r')\s*\([^)]*\)\s*\{', clean_line)
            if method_match:
                in_loop_method = method_match.group(1)
                brace_depth = clean_line.count('{') - clean_line.count('}')
                continue
            if in_loop_method:
                brace_depth += clean_line.count('{') - clean_line.count('}')
                if 'addEventListener' in clean_line:
                    issues.append({
                        'severity': 'critical',
                        'location': f'JavaScript Line {idx}',
                        'line': idx,
                        'issue': f'Event listener registered inside {in_loop_method}() - will rebind every frame',
                        'fix': 'Move addEventListener calls to constructor/init so they run once'
                    })
                if brace_depth <= 0:
                    in_loop_method = None

        # 5. Check for use-after-null on common state props
        def _use_after_null(prop: str, window: int = 30) -> None:
            lines = js_code.splitlines()
            needle_assign = f"this.{prop} = null"
            use_pattern = re.compile(rf'\bthis\.{prop}\s*\.')
            assign_pattern = re.compile(rf'\bthis\.{prop}\s*=')
            for idx, line in enumerate(lines, 1):
                clean_line = self._strip_strings_and_comments(line)
                if needle_assign in clean_line:
                    for j in range(idx + 1, min(idx + window, len(lines)) + 1):
                        look = self._strip_strings_and_comments(lines[j - 1])
                        if assign_pattern.search(look):
                            break
                        if use_pattern.search(look):
                            issues.append({
                                'severity': 'critical',
                                'location': f'JavaScript Line {j}',
                                'line': j,
                                'issue': f'Use-after-null: this.{prop} accessed after being set to null',
                                'fix': f'Delay setting this.{prop} = null until after all uses, or reassign before access'
                            })
                            break

        _use_after_null('selectedPiece')
        _use_after_null('sourceSquare')
        
        # ============ GENERAL JS CHECKS ============
        # Check for game loop
        if 'requestAnimationFrame' not in js_code and 'setInterval' not in js_code:
            issues.append({
                'severity': 'warning',
                'location': 'Game loop',
                'issue': 'No game loop detected',
                'fix': 'Add requestAnimationFrame for game loop'
            })
        
        # Check for event listeners
        if 'addEventListener' not in js_code and 'onclick' not in js_code.lower():
            issues.append({
                'severity': 'warning',
                'location': 'Event handling',
                'issue': 'No event listeners found',
                'fix': 'Add keyboard/mouse event listeners'
            })
        
        # ============ BOARD GAME SPECIFIC CHECKS ============
        board_keywords = ['chess', 'checkers', 'draughts', 'othello', 'reversi']
        is_board_game = any(kw in js_code.lower() for kw in board_keywords)
        is_chess = 'chess' in js_code.lower()
        
        # Check 1: Non-square canvas for board games
        if is_board_game:
            w_match = re.search(r'canvas\.width\s*=\s*(\d+)', js_code)
            h_match = re.search(r'canvas\.height\s*=\s*(\d+)', js_code)
            if w_match and h_match:
                w, h = int(w_match.group(1)), int(h_match.group(1))
                if w != h:
                    issues.append({
                        'severity': 'critical',
                        'location': 'Canvas dimensions',
                        'issue': f'Board game canvas is NOT square ({w}x{h}) - bottom rows will be clipped',
                        'fix': f'Set canvas.width = canvas.height = {w} for a proper square board'
                    })
        
        # Check 2: Duplicate symbols for opposing players
        if is_board_game:
            # Extract all symbol/emoji definitions per player
            player_symbols = re.findall(r"player['\"]?\s*,\s*symbol\s*:\s*['\"]([^'\"]+)['\"]", js_code)
            enemy_symbols = re.findall(r"enemy['\"]?\s*,\s*symbol\s*:\s*['\"]([^'\"]+)['\"]", js_code)
            if not player_symbols:
                player_symbols = re.findall(r"'player'.*?symbol\s*:\s*['\"]([^'\"]+)['\"]", js_code)
            if not enemy_symbols:
                enemy_symbols = re.findall(r"'enemy'.*?symbol\s*:\s*['\"]([^'\"]+)['\"]", js_code)
            
            if player_symbols and enemy_symbols:
                shared = set(player_symbols) & set(enemy_symbols)
                if shared:
                    issues.append({
                        'severity': 'critical',
                        'location': 'Game - Piece Symbols',
                        'issue': f'Both players use IDENTICAL symbols {shared} - cannot distinguish sides',
                        'fix': 'Use Unicode chess pieces: ♔♕♖♗♘♙ (white) vs ♚♛♜♝♞♟ (black)'
                    })
        
        # Check 3: Chess-specific missing features
        if is_chess:
            # Missing pawn promotion (warning, not critical — it's a missing feature, not a crash)
            if 'promot' not in js_code.lower():
                issues.append({
                    'severity': 'warning',
                    'location': 'Chess - Pawn Promotion',
                    'issue': 'No pawn promotion logic found - pawns reaching end row stay as pawns',
                    'fix': 'Add promotion: when pawn reaches row 0/7, convert to queen'
                })
            
            # Missing en passant (warning, not critical)
            if 'passant' not in js_code.lower():
                issues.append({
                    'severity': 'warning',
                    'location': 'Chess - En Passant',
                    'issue': 'No en passant capture logic found',
                    'fix': 'Track last pawn double-move and allow diagonal en passant capture'
                })
            
            # Missing castling
            if 'castl' not in js_code.lower():
                issues.append({
                    'severity': 'warning',
                    'location': 'Chess - Castling',
                    'issue': 'No castling logic found',
                    'fix': 'Add castling: king moves 2 squares toward rook, rook jumps over king'
                })
        
        # Check 4: Misleading HUD for board games (e.g., "Lives" in chess)
        if is_board_game:
            if 'lives' in js_code.lower() and 'livesdisplay' in js_code.lower().replace(' ', ''):
                issues.append({
                    'severity': 'warning',
                    'location': 'Game HUD',
                    'issue': 'Board game shows "Lives" counter which is not applicable',
                    'fix': 'Replace Lives display with captured pieces, turn indicator, or move counter'
                })
        
        # ============ PLACEHOLDER/INCOMPLETE CODE DETECTION ============
        
        # Check for "// Implement..." or "// TODO" comments (indicates unfinished code)
        placeholder_patterns = [
            (r'//\s*implement', 'Implementation placeholder comment'),
            (r'//\s*todo', 'TODO marker indicates unfinished code'),
            (r'//\s*add\s+\w+\s+logic', 'Logic placeholder comment'),
            (r'//\s*placeholder', 'Placeholder marker'),
            (r'//\s*fix\s*me', 'FIXME marker indicates broken code'),
        ]
        
        for pattern, description in placeholder_patterns:
            matches = re.findall(pattern, js_code, re.IGNORECASE)
            if matches:
                issues.append({
                    'severity': 'critical',
                    'location': 'JavaScript',
                    'issue': f'{description} found ({len(matches)} occurrences)',
                    'fix': 'Implement the actual logic instead of leaving placeholders'
                })
        
        # Check for empty switch case bodies
        empty_case_pattern = r'case\s+[\'\"][^"\']+[\'\"]\s*:\s*(break;|//)'
        empty_cases = re.findall(empty_case_pattern, js_code)
        if len(empty_cases) >= 3:  # Multiple empty cases = suspicious
            issues.append({
                'severity': 'critical',
                'location': 'JavaScript - Switch cases',
                'issue': f'Found {len(empty_cases)} empty switch case bodies',
                'fix': 'Implement logic for each case statement'
            })
        
        # Check for stub functions that just return true/false
        stub_patterns = [
            (r'function\s+\w+\s*\([^)]*\)\s*\{\s*return\s+true\s*;?\s*\}', 'Stub function returning true'),
            (r'function\s+\w+\s*\([^)]*\)\s*\{\s*return\s+false\s*;?\s*\}', 'Stub function returning false'),
            (r'\w+\s*\([^)]*\)\s*\{\s*return\s+true\s*;?\s*\}', 'Stub method returning true'),
        ]
        
        for pattern, description in stub_patterns:
            if re.search(pattern, js_code, re.IGNORECASE):
                issues.append({
                    'severity': 'warning',
                    'location': 'JavaScript',
                    'issue': f'{description} - likely placeholder',
                    'fix': 'Implement actual validation/game logic'
                })
        
        # Check for empty function bodies (functions with just return [] or {})
        empty_func_pattern = r'function\s+\w+\s*\([^)]*\)\s*\{\s*(const\s+\w+\s*=\s*\[\];?)?\s*return\s+\w+\s*;?\s*\}'
        if re.search(empty_func_pattern, js_code):
            issues.append({
                'severity': 'warning',
                'location': 'JavaScript',
                'issue': 'Found functions with minimal/empty implementation',
                'fix': 'Add actual implementation logic'
            })
        
        # Check minimum code complexity for games
        code_lines = len([l for l in js_code.split('\n') if l.strip() and not l.strip().startswith('//')])
        if code_lines < 50:
            issues.append({
                'severity': 'critical',
                'location': 'JavaScript',
                'issue': f'JS code too short ({code_lines} lines) - likely incomplete',
                'fix': 'Add complete game implementation'
            })
        
        # ============ JAVASCRIPT SYNTAX VALIDATION ============
        
        # Check brace balance
        open_braces = js_code.count('{')
        close_braces = js_code.count('}')
        if open_braces != close_braces:
            issues.append({
                'severity': 'critical',
                'location': 'JavaScript syntax',
                'issue': f'Unbalanced braces: {open_braces} opening, {close_braces} closing',
                'fix': f'Add {abs(open_braces - close_braces)} {"closing" if open_braces > close_braces else "opening"} braces'
            })
            
        # Check for malformed function definitions (}functionName pattern)
        malformed_pattern = r'\}[a-zA-Z_][a-zA-Z0-9_]*\s*\('
        malformed_matches = re.findall(malformed_pattern, js_code)
        if malformed_matches:
            issues.append({
                'severity': 'critical',
                'location': 'JavaScript syntax',
                'issue': f'Malformed code structure: missing whitespace/newlines between functions',
                'fix': 'Add proper closing braces and newlines between method definitions'
            })
            
        # Check parenthesis balance
        open_parens = js_code.count('(')
        close_parens = js_code.count(')')
        if open_parens != close_parens:
            issues.append({
                'severity': 'critical',
                'location': 'JavaScript syntax',
                'issue': f'Unbalanced parentheses: {open_parens} opening, {close_parens} closing',
                'fix': f'Add {abs(open_parens - close_parens)} {"closing" if open_parens > close_parens else "opening"} parentheses'
            })
        
        # Check for required game mechanics keywords
        game_keywords = ['score', 'lives', 'level', 'update', 'draw', 'collision', 'win', 'lose', 'game']
        keyword_count = sum(1 for kw in game_keywords if kw in js_code.lower())
        if keyword_count < 3:
            issues.append({
                'severity': 'warning',
                'location': 'Game mechanics',
                'issue': f'Missing common game keywords (found {keyword_count}/9)',
                'fix': 'Implement scoring, lives, and game state management'
            })
        
        # Check for hardcoded dummy AI/movement
        dummy_patterns = [
            (r'aiMove\s*=\s*\{\s*fromSquare:\s*\{\s*x:\s*0,\s*y:\s*0\s*\}', 'Hardcoded AI move detected'),
            (r'setTimeout\s*\(\s*\(\)\s*=>\s*\{[^}]{1,50}\}', 'Suspiciously simple timeout logic'),
        ]
        
        for pattern, description in dummy_patterns:
            if re.search(pattern, js_code, re.IGNORECASE):
                issues.append({
                    'severity': 'critical',
                    'location': 'JavaScript - AI/Logic',
                    'issue': description,
                    'fix': 'Implement proper AI algorithm (minimax, alpha-beta, etc.)'
                })
        
        # ============ GAME LOGIC COMPLETENESS CHECKS ============
        
        # Check 1: Incomplete move/action functions (simplified/demo comments)
        simplified_patterns = [
            (r'//\s*simplified', 'Simplified logic comment found - incomplete implementation'),
            (r'//\s*for\s+demo', 'Demo-only implementation detected'),
            (r'//\s*basic\s+\w+\s+only', 'Basic-only implementation detected'),
            (r'//\s*only\s+handles?\s+\w+', 'Partial implementation detected'),
            (r'return\s*\[\s*\]\s*;?\s*//.*partial', 'Function returns empty array with partial comment'),
        ]
        
        for pattern, description in simplified_patterns:
            if re.search(pattern, js_code, re.IGNORECASE):
                issues.append({
                    'severity': 'critical',
                    'location': 'JavaScript - Game Logic',
                    'issue': description,
                    'fix': 'Implement complete game logic for all cases'
                })
        
        # Check 2: Missing turn/player system for board games
        board_game_indicators = ['chess', 'checkers', 'board', 'turn', 'player']
        is_board_game = any(indicator in js_code.lower() for indicator in board_game_indicators)
            
        if is_board_game:
            # Check for turn tracking
            if 'turn' not in js_code.lower() and 'isplayerturn' not in js_code.lower().replace(' ', ''):
                issues.append({
                    'severity': 'warning',
                    'location': 'JavaScript - Turn System',
                    'issue': 'Board game detected but no turn tracking found',
                    'fix': 'Add turn-based system (e.g., isPlayerTurn, currentPlayer)'
                })
            
            # Check for AI/opponent logic
            ai_patterns = ['makeaimove', 'aimove', 'opponent', 'computer', 'enemy']
            has_ai = any(pattern in js_code.lower().replace(' ', '') for pattern in ai_patterns)
            if not has_ai:
                issues.append({
                    'severity': 'warning',
                    'location': 'JavaScript - AI',
                    'issue': 'Board game detected but no AI/opponent logic found',
                    'fix': 'Implement AI opponent or two-player mode'
                })
            
        # Check 3: Missing source position tracking (common bug in move systems)
        if 'selectedsquare' in js_code.lower().replace(' ', '') or 'selectedpiece' in js_code.lower().replace(' ', ''):
            # Check if source position is saved before moving
            if 'sourcesquare' not in js_code.lower().replace(' ', '') and \
               'sourcerow' not in js_code.lower().replace(' ', '') and \
               'fromrow' not in js_code.lower().replace(' ', '') and \
               'from[' not in js_code.lower() and \
               'pieceposition' not in js_code.lower().replace(' ', ''):
                issues.append({
                    'severity': 'warning',
                    'location': 'JavaScript - State Tracking',
                    'issue': 'Selection system found but source position may not be tracked',
                    'fix': 'Store source position when selecting pieces (e.g., sourceSquare = [row, col])'
                })
        
        # Check 4: Incomplete piece/entity movement (only handles some cases)
        if 'getvalidmoves' in js_code.lower().replace(' ', '') or 'getmoves' in js_code.lower().replace(' ', ''):
            # Count how many piece types are handled
            piece_patterns = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
            handled_pieces = sum(1 for p in piece_patterns if p in js_code.lower())
            
            # If it's a chess game, all pieces should be handled
            if 'chess' in js_code.lower() and handled_pieces < 6:
                issues.append({
                    'severity': 'critical',
                    'location': 'JavaScript - Move Logic',
                    'issue': f'Chess game but only {handled_pieces}/6 piece types have move logic',
                    'fix': 'Implement move logic for ALL piece types (pawn, rook, knight, bishop, queen, king)'
                })
        
        # Check 5: Move function returns empty or near-empty results
        move_func_pattern = r'(getValidMoves|getMoves|calculateMoves)\s*\([^)]*\)\s*\{([^}]{0,500})\}'
        move_funcs = re.findall(move_func_pattern, js_code, re.IGNORECASE)
        for func_name, func_body in move_funcs:
            # Check if function body is too short
            if len(func_body.strip()) < 100:
                issues.append({
                    'severity': 'warning',
                    'location': f'JavaScript - {func_name}()',
                    'issue': f'Move function {func_name}() has suspiciously short implementation',
                    'fix': 'Implement complete move validation logic'
                })
        
        # Check 6: Selection initialization on game start
        if 'selectedsquare' in js_code.lower().replace(' ', ''):
            start_func_match = re.search(r'start\s*\(\s*\)\s*\{([^}]{0,1000})\}', js_code, re.IGNORECASE)
            if start_func_match:
                start_body = start_func_match.group(1)
                if 'selectedsquare' not in start_body.lower().replace(' ', '') and \
                   'selectsquare' not in start_body.lower().replace(' ', ''):
                    issues.append({
                        'severity': 'warning',
                        'location': 'JavaScript - start()',
                        'issue': 'Game uses selection system but start() does not initialize selection',
                        'fix': 'Initialize selectedSquare in start() method'
                    })
        
        # Check 7: Render/update after moves
        if 'makemove' in js_code.lower().replace(' ', '') or 'movePiece' in js_code.lower():
            move_func_match = re.search(r'(makeMove|movePiece)\s*\([^)]*\)\s*\{([^}]{0,2000})\}', js_code, re.IGNORECASE)
            if move_func_match:
                move_body = move_func_match.group(2)
                if 'render' not in move_body.lower() and 'update' not in move_body.lower() and 'draw' not in move_body.lower():
                    issues.append({
                        'severity': 'warning',
                        'location': 'JavaScript - Move Function',
                        'issue': 'Move function does not call render/update after moving',
                        'fix': 'Call renderBoard() or similar after making a move'
                    })
        
        return issues
    
    # ============ HTML VALIDATION ============
    
    def _validate_html(self, html_code: str) -> List[Dict]:
        """Validate HTML structure for game requirements."""
        issues = []
        
        if not html_code or len(html_code.strip()) < 20:
            return issues  # No HTML to check (might be separate files)
        
        # Check DOCTYPE
        if '<!DOCTYPE' not in html_code and '<!doctype' not in html_code:
            issues.append({
                'severity': 'warning',
                'location': 'HTML - DOCTYPE',
                'issue': 'Missing <!DOCTYPE html> declaration',
                'fix': 'Add <!DOCTYPE html> at the beginning'
            })
        
        # Check for canvas element
        if '<canvas' not in html_code.lower():
            issues.append({
                'severity': 'warning',
                'location': 'HTML - Canvas',
                'issue': 'No <canvas> element found',
                'fix': 'Add <canvas id="gameCanvas"></canvas> for game rendering'
            })
        elif 'id=' not in html_code[html_code.lower().find('<canvas'):html_code.lower().find('<canvas') + 100]:
            issues.append({
                'severity': 'critical',
                'location': 'HTML - Canvas',
                'issue': 'ReferenceError: Canvas element has no ID - getElementById will return null',
                'fix': 'Add id="gameCanvas" to the canvas element'
            })
        
        # Check for overlay divs
        required_overlays = {
            'startOverlay': 'Start screen overlay',
            'pauseOverlay': 'Pause screen overlay',
            'gameOverOverlay': 'Game over screen overlay'
        }
        for overlay_id, description in required_overlays.items():
            if overlay_id not in html_code and overlay_id.lower() not in html_code.lower():
                issues.append({
                    'severity': 'warning',
                    'location': f'HTML - {description}',
                    'issue': f'Missing {description} (id="{overlay_id}")',
                    'fix': f'Add <div id="{overlay_id}"> for the {description.lower()}'
                })
        
        # Check for script and style links
        # Check for script and style (inline or external)
        if '<script' not in html_code.lower():
            issues.append({
                'severity': 'critical',
                'location': 'HTML - Script',
                'issue': 'No <script> tags found - game logic missing',
                'fix': 'Add <script>...</script> block with game logic'
            })
        
        if '<style' not in html_code.lower():
            issues.append({
                'severity': 'warning',
                'location': 'HTML - Stylesheet',
                'issue': 'No <style> tags found',
                'fix': 'Add <style>...</style> block with CSS'
            })
        
        # Check for mismatched IDs between HTML and script references
        html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html_code))
        if html_ids:
            # Check if any common game IDs are missing
            expected_ids = ['gameCanvas']
            for eid in expected_ids:
                found = any(eid.lower() == hid.lower() for hid in html_ids)
                if not found and 'canvas' in html_code.lower():
                    # Canvas exists but with wrong ID
                    actual_canvas_id = re.search(r'<canvas[^>]*id=["\']([^"\']+)["\']', html_code, re.IGNORECASE)
                    if actual_canvas_id and actual_canvas_id.group(1) != eid:
                        issues.append({
                            'severity': 'warning',
                            'location': 'HTML - Canvas ID',
                            'issue': f'Canvas ID is "{actual_canvas_id.group(1)}" but code may reference "{eid}"',
                            'fix': f'Ensure canvas ID matches what JavaScript uses in getElementById()'
                        })
        
        return issues
    
    # ============ CSS VALIDATION ============
    
    def _validate_css(self, css_code: str) -> List[Dict]:
        """Validate CSS for game styling requirements."""
        issues = []
        
        if not css_code or len(css_code.strip()) < 10:
            issues.append({
                'severity': 'warning',
                'location': 'CSS',
                'issue': 'No CSS styles found or CSS is too short',
                'fix': 'Add complete game styling'
            })
            return issues
        
        # Check for game container styling
        if 'game-container' not in css_code and 'gameContainer' not in css_code:
            issues.append({
                'severity': 'warning',
                'location': 'CSS - Game Container',
                'issue': 'No game container styles found',
                'fix': 'Add .game-container { position: relative; width: 800px; margin: auto; }'
            })
        
        # Check for overlay positioning
        overlay_keywords = ['overlay', 'Overlay']
        has_overlay_styles = any(kw in css_code for kw in overlay_keywords)
        if has_overlay_styles:
            if 'position' not in css_code:
                issues.append({
                    'severity': 'warning',
                    'location': 'CSS - Overlays',
                    'issue': 'Overlay styles found but no positioning (absolute/fixed)',
                    'fix': 'Add position: absolute or position: fixed to overlay styles'
                })
        else:
            issues.append({
                'severity': 'warning',
                'location': 'CSS - Overlays',
                'issue': 'No overlay styles found (start, pause, game over screens)',
                'fix': 'Add styles for #startOverlay, #pauseOverlay, #gameOverOverlay'
            })
        
        # Check for animations
        if '@keyframes' not in css_code:
            issues.append({
                'severity': 'warning',
                'location': 'CSS - Animations',
                'issue': 'No @keyframes animations defined',
                'fix': 'Add fadeIn, pulse, or shake animations for better UX'
            })
        
        # Check for font import
        if '@import' not in css_code and 'font-family' not in css_code:
            issues.append({
                'severity': 'warning',
                'location': 'CSS - Typography',
                'issue': 'No font import or font-family found',
                'fix': 'Import a Google Font and set font-family on body'
            })
        
        # Check for canvas styling
        if 'canvas' not in css_code:
            issues.append({
                'severity': 'warning',
                'location': 'CSS - Canvas',
                'issue': 'No canvas styles found',
                'fix': 'Add canvas { display: block; } and optional border/shadow'
            })
        
        # Check for responsive/centered layout
        if 'margin' not in css_code and 'center' not in css_code and 'flex' not in css_code:
            issues.append({
                'severity': 'warning',
                'location': 'CSS - Layout',
                'issue': 'Game may not be centered on screen',
                'fix': 'Add margin: auto or display: flex; justify-content: center to center the game'
            })
        
        return issues
    
    def _llm_validation(self, game_code: str) -> Dict:
        """Use LLM for deeper code validation with STRUCTURED AUDIT."""
        try:
            # Truncate code if too long
            code_preview = game_code[:12000] if len(game_code) > 12000 else game_code
            messages = self.prompt.format_messages(game_code=code_preview)
            response = self.llm.invoke(messages)
            
            content = self._extract_content(response)
            
            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                import json
                try:
                    data = json.loads(json_match.group())
                    
                    # Convert Structured Audit to Standard Issues List
                    issues = []
                    
                    # 1. Logic Errors (Critical)
                    for err in data.get('logic_errors', []):
                        issues.append({
                            'severity': 'critical',
                            'location': 'Logic Check',
                            'issue': str(err),
                            'fix': data.get('fix_directive', 'Fix logic error')
                        })
                        
                    # 2. Security Score
                    sec_score = data.get('security_score', 10)
                    if sec_score < 9:
                        issues.append({
                            'severity': 'critical',
                            'location': 'Security Audit',
                            'issue': f'Security Score {sec_score}/10 (Threshold: 9)',
                            'fix': 'Review input handling and HTML injection risks'
                        })
                        
                    # 3. Edge Cases (Warning)
                    for edge in data.get('edge_cases_missed', []):
                        issues.append({
                            'severity': 'warning',
                            'location': 'Edge Case',
                            'issue': str(edge),
                            'fix': 'Implement handling for this edge case'
                        })
                        
                    # 4. Status Check
                    if data.get('status') == 'REJECTED' and not issues:
                        # Fallback if status rejected but no specific errors listed (rare)
                        issues.append({
                            'severity': 'critical',
                            'location': 'General Audit',
                            'issue': 'Code REJECTED by auditor',
                            'fix': data.get('fix_directive', 'Address code quality issues')
                        })
                        
                    return {
                        'issues': issues,
                        'score': sec_score * 10,  # Approximate score base
                        'summary': f"Audit: {data.get('status')} ({len(issues)} issues)"
                    }
                    
                except json.JSONDecodeError:
                    return {'issues': [], 'summary': 'Validation JSON Parse Error'}
            
            return {'issues': [], 'summary': 'LLM validation completed'}
            
        except Exception as e:
            return {
                'issues': [{
                    'severity': 'info',
                    'location': 'Validation',
                    'issue': f'LLM validation error: {str(e)}',
                    'fix': 'Manual review recommended'
                }],
                'summary': 'Partial validation completed'
            }
    
    def _extract_content(self, response) -> str:
        """Safely extract content from response."""
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            else:
                 content = str(content)
        else:
             content = str(response)
             
        return self._strip_thinking_tokens(content)

    def _strip_thinking_tokens(self, content: str) -> str:
        """Strip Qwen 3's <think>...</think> blocks from response."""
        # Remove <think>...</think> blocks (Qwen 3's reasoning)
        clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return clean.strip()
    
    def quick_validate(self, game_code: str) -> Tuple[bool, str]:
        """Quick validation without LLM - structural checks on JS, HTML, CSS."""
        js_code = self._extract_js(game_code)
        html_code = self._extract_html(game_code)
        css_code = self._extract_css(game_code)
        
        all_issues = self._validate_js(js_code) + self._validate_html(html_code) + self._validate_css(css_code)
        critical = [i for i in all_issues if i.get('severity') == 'critical']
        
        if critical:
            return False, f"Found {len(critical)} critical issues (ReferenceError risks)"
        return True, "Basic structure valid"
