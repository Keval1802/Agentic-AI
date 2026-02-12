"""
Coder-Validator Graph - LangGraph implementation for iterative code generation and validation.
This graph manages the conversation between the Coder and Validator agents,
automatically fixing issues until the code passes validation or max iterations are reached.
"""

import os
import re
import time
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from dataclasses import dataclass
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..agents.coder import CoderAgent
from ..agents.validator import ValidatorAgent, ValidationResult
from ..utils.code_patcher import FunctionPatcher


# ============ STATE DEFINITION ============

class GraphState(TypedDict):
    """State that flows through the coder-validator graph.
    
    Key Feature: code_memory maintains the FULL context of all code files
    across iterations, solving the stateless problem by ensuring each
    LLM call has access to the complete, synchronized codebase.
    """
    # Input
    game_plan: str
    design_spec: str
    
    # Code state - unified memory for all files
    current_code: str  # Combined HTML (for backwards compatibility)
    game_files: Dict[str, str]  # {'html': ..., 'css': ..., 'js': ...}
    
    # Code Memory - stores full context for each file type
    code_memory: Dict[str, str]  # {'html': full_html, 'css': full_css, 'js': full_js}
    
    # Validation state
    validation_result: Optional[Dict[str, Any]]
    issues: List[Dict[str, Any]]
    is_valid: bool
    score: int
    
    # Iteration tracking
    iteration: int
    max_iterations: Optional[int]
    stall_count: int
    previous_issue_count: int
    previous_issue_signature: Optional[str]
    previous_score: int
    
    # Conversation history (for debugging/logging)
    messages: Annotated[List[str], add]
    
    # Fix history - tracks what was fixed to avoid repeating
    fix_history: Dict[str, str]
    
    # Final status
    status: str  # 'coding', 'validating', 'patching', 'complete', 'failed'
    error: Optional[str]


# ============ GRAPH NODES ============

class CoderValidatorGraph:
    """
    LangGraph-based orchestration of Coder and Validator agents.
    
    Flow:
        START → coder → validator → [is_valid?]
                                        ├── Yes → END
                                        └── No → patcher → validator → ...
    
    Features:
    - Stateful iteration with automatic stall detection
    - Function-level patching for targeted fixes
    - Conversation history for debugging
    - Checkpointing support for resume capability
    """
    
    MAX_ITERATIONS = None
    MAX_STALLS = 3
    
    def __init__(
        self,
        coder: CoderAgent,
        validator: ValidatorAgent,
        on_progress: Optional[callable] = None
    ):
        """Initialize the coder-validator graph."""
        self.coder = coder
        self.validator = validator
        self.patcher = FunctionPatcher()
        self.on_progress = on_progress
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "0") == "1"
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Memory saver for checkpointing
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create graph with state schema
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("coder", self._coder_node)
        graph.add_node("validator", self._validator_node)
        graph.add_node("patcher", self._patcher_node)
        
        # Set entry point
        graph.set_entry_point("coder")
        
        # Add edges
        graph.add_edge("coder", "validator")
        
        # Conditional edge from validator
        graph.add_conditional_edges(
            "validator",
            self._should_continue,
            {
                "complete": END,
                "patch": "patcher",
                "failed": END
            }
        )
        
        # Patcher loops back to validator
        graph.add_edge("patcher", "validator")
        
        return graph
    
    def _notify(self, message: str):
        """Send progress notification if callback is set."""
        safe_message = str(message).encode("utf-8", "backslashreplace").decode("utf-8")
        if self.on_progress:
            try:
                self.on_progress(safe_message)
                return
            except:
                pass
        print(safe_message)
    
    # ============ NODE IMPLEMENTATIONS ============
    
    def _extract_code_parts(self, game_code: str) -> Dict[str, str]:
        """Extract HTML, CSS, and JavaScript from LLM output into separate memory.
        
        Uses GameCodeParser for robust extraction of markdown code blocks.
        """
        from ..parsers.game_parser import GameCodeParser
        
        parser = GameCodeParser()
        
        # Use parse_multi_file for proper extraction
        files = parser.parse_multi_file(game_code)
        
        html = files.get('html', '')
        css = files.get('css', '')
        js = files.get('js', '')

        # If the model returns placeholders like "existing CSS/JS/HTML",
        # leave that part empty so merge logic keeps prior memory.
        if self._is_existing_placeholder(html, "html"):
            html = ""
        if self._is_existing_placeholder(css, "css"):
            css = ""
        if self._is_existing_placeholder(js, "js"):
            js = ""

        code_memory = {
            'html': html,
            'css': css,
            'js': js,
            'full': game_code  # Keep original for reference
        }
        
        return code_memory

    def _is_existing_placeholder(self, content: str, kind: str) -> bool:
        """Detect placeholder blocks like "existing CSS" to preserve previous memory."""
        if not content:
            return False

        lowered = content.strip().lower()
        if not lowered:
            return False

        # Normalize common typo variants: exesisting/exesting
        for prefix in ["existing", "exesisting", "exesting"]:
            if f"{prefix} {kind}" in lowered:
                return True

        # Handle comment-only placeholders
        comment_only = re.sub(r"[^a-z]+", " ", lowered).strip()
        if comment_only in {f"existing {kind}", f"exesisting {kind}", f"exesting {kind}"}:
            return True

        return False

    def _build_minimal_html_with_js(self, js_code: str) -> str:
        """Build a tiny HTML wrapper to reduce token usage during JS-only fixes."""
        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Game</title>
    <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
    <canvas id=\"gameCanvas\"></canvas>
    <script>
{js_code}
    </script>
</body>
</html>"""

    def _merge_code_memory(self, previous: Dict[str, str], patched: Dict[str, str]) -> Dict[str, str]:
        """Prefer patched JS but keep previous HTML/CSS when patch output is minimal."""
        merged = {
            "html": patched.get("html") or previous.get("html", ""),
            "css": patched.get("css") or previous.get("css", ""),
            "js": patched.get("js") or previous.get("js", ""),
            "full": patched.get("full") or previous.get("full", ""),
        }
        return merged

    def _apply_placeholder_line_fixes(self, code_memory: Dict[str, str], issues: List[Dict[str, Any]]) -> tuple:
        """
        Remove placeholder lines (e.g. '// existing css') from code_memory
        and strip out the corresponding issues so they don't persist.
        Returns (updated_code_memory, remaining_issues, count_of_applied_fixes).
        """
        applied = 0
        remaining_issues = []

        # Group placeholder issues by file type
        placeholder_issues_by_file: Dict[str, List[int]] = {}  # file -> list of 1-based line numbers
        for issue in issues:
            if issue.get("code") == "placeholder":
                file_type = issue.get("file", "")
                line_num = issue.get("line")
                if file_type and line_num:
                    placeholder_issues_by_file.setdefault(file_type, []).append(line_num)
                    applied += 1
                else:
                    remaining_issues.append(issue)
            else:
                remaining_issues.append(issue)

        # Remove the flagged lines from each file in code_memory
        for file_type, line_numbers in placeholder_issues_by_file.items():
            content = code_memory.get(file_type, "")
            if not content:
                continue
            lines = content.splitlines()
            line_set = set(line_numbers)  # 1-based
            new_lines = [
                line for idx, line in enumerate(lines, start=1)
                if idx not in line_set
            ]
            code_memory[file_type] = "\n".join(new_lines)

        return code_memory, remaining_issues, applied

    def _issue_signature(self, issues: List[Dict[str, Any]]) -> str:
        """Create a stable signature for issue comparisons across iterations."""
        parts = []
        for issue in issues:
            parts.append(
                "|".join([
                    str(issue.get("severity", "")),
                    str(issue.get("location", "")),
                    str(issue.get("issue", "")),
                    str(issue.get("fix", ""))
                ])
            )
        return "||".join(sorted(parts))

    def _coder_node(self, state: GraphState) -> dict:
        """
        Coder Node: Generate initial game code from plan and design spec.
        Stores code in unified memory for stateful iteration.
        """
        self._notify(f"💻 [Coder] Generating game code...")
        
        try:
            game_code = self.coder.code(
                state["game_plan"],
                state.get("design_spec", "")
            )
            
            if not game_code or len(game_code) < 500:
                return {
                    "current_code": "",
                    "code_memory": {},
                    "status": "failed",
                    "error": "Generated code is too short or empty",
                    "messages": ["❌ Coder failed to generate valid code"]
                }
            
            # Extract and store code parts in memory
            code_memory = self._extract_code_parts(game_code)
            
            self._notify(f"✅ [Coder] Generated {len(game_code)} chars of code")
            self._notify(f"   📄 HTML: {len(code_memory.get('html', ''))} chars")
            self._notify(f"   🎨 CSS: {len(code_memory.get('css', ''))} chars")
            self._notify(f"   ⚡ JS: {len(code_memory.get('js', ''))} chars")
            
            return {
                "current_code": game_code,
                "code_memory": code_memory,
                "iteration": 1,
                "status": "validating",
                "fix_history": [],
                "messages": [f"📝 Generated initial code ({len(game_code)} chars)"]
            }
            
        except Exception as e:
            return {
                "current_code": "",
                "code_memory": {},
                "status": "failed",
                "error": f"Coder error: {str(e)}",
                "messages": [f"❌ Coder error: {str(e)}"]
            }
    
    def _validator_node(self, state: GraphState) -> dict:
        """
        Validator Node: Check code quality and identify issues.
        """
        iteration = state.get("iteration", 1)
        self._notify(f"🔍 [Validator] Validating code (iteration {iteration})...")
        
        try:
            current_code = state["current_code"]
            
            # Run validation
            result = self.validator.validate(current_code)
            
            # Track issue count for stall detection
            current_issue_count = len(result.issues)
            previous_count = state.get("previous_issue_count", float('inf'))
            previous_signature = state.get("previous_issue_signature")
            previous_score = state.get("previous_score", -1)
            current_signature = self._issue_signature(result.issues)
            
            # Update stall counter
            stall_count = state.get("stall_count", 0)
            if (
                current_issue_count >= previous_count and
                current_signature == previous_signature and
                result.score <= previous_score
            ):
                stall_count += 1
            else:
                stall_count = 0
            
            self._notify(
                f"📊 [Validator] Score: {result.score}/100, "
                f"Issues: {current_issue_count}, Valid: {result.is_valid}"
            )
            
            return {
                "validation_result": {
                    "is_valid": result.is_valid,
                    "score": result.score,
                    "issues": result.issues,
                    "summary": result.summary
                },
                "issues": result.issues,
                "is_valid": len(result.issues) == 0,  # Zero issues = valid
                "score": result.score,
                "previous_issue_count": current_issue_count,
                "previous_issue_signature": current_signature,
                "previous_score": result.score,
                "stall_count": stall_count,
                "messages": [
                    f"🔍 Validation {iteration}: {current_issue_count} issues, score={result.score}"
                ]
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [],
                "score": 0,
                "error": f"Validation error: {str(e)}",
                "messages": [f"❌ Validator error: {str(e)}"]
            }
    
    def _patcher_node(self, state: GraphState) -> dict:
        """
        Patcher Node: Fix identified issues in the code.
        Uses code_memory to maintain full context across iterations.
        """
        iteration = state.get("iteration", 1)
        issues = state.get("issues", [])
        current_code = state["current_code"]
        code_memory = state.get("code_memory", {})
        fix_history = state.get("fix_history", {})
        if isinstance(fix_history, list):
            fix_history = {name: "" for name in fix_history}
        
        self._notify(f"🔧 [Patcher] Fixing {len(issues)} issues (iteration {iteration})...")
        
        if self.verbose_logs:
            self._notify(
                f"   📦 Code Memory: HTML={len(code_memory.get('html',''))} "
                f"CSS={len(code_memory.get('css',''))} "
                f"JS={len(code_memory.get('js',''))} chars"
            )

        try:
            # Apply deterministic line fixes before LLM patching
            try:
                code_memory, issues, applied = self._apply_placeholder_line_fixes(code_memory, issues)
                if applied:
                    from ..parsers.game_parser import GameCodeParser
                    parser = GameCodeParser()
                    current_code = parser._combine(
                        code_memory.get("html", ""),
                        code_memory.get("css", ""),
                        code_memory.get("js", ""),
                    )
                    self._notify(f"🧹 [Patcher] Removed {applied} placeholder line(s)")
            except Exception as placeholder_err:
                self._notify(f"⚠️ [Patcher] Placeholder fix error (non-fatal): {placeholder_err}")

            # === CRITIQUE-LOOP: GLOBAL REJECTION CHECK ===
            # Check if Validator rejected code entirely (Structured Audit)
            audit_rejection = next((i for i in issues if i.get('location') == 'General Audit' and 'REJECTED' in i.get('issue', '')), None)
            
            if audit_rejection:
                fix_directive = audit_rejection.get('fix', 'Resolve critical logic flaws.')
                self._notify(f"🚨 [Patcher] AUDIT REJECTION received. Bypass surgical patch.")
                self._notify(f"   📝 Directive: {fix_directive}")
                
                # Force global fix with the directive
                fix_instructions = f"## AUDITOR REJECTION:\nThe code was REJECTED. You must fix this CRITICAL issue:\n\n{fix_directive}\n\nReview the logic from scratch and rewrite the broken parts."
                
                # Reconstruct full code
                full_code = current_code
                from ..parsers.game_parser import GameCodeParser  # Ensure import
                if code_memory.get("html") or code_memory.get("css") or code_memory.get("js"):
                    parser = GameCodeParser()
                    full_code = parser._combine(
                        code_memory.get("html", ""),
                        code_memory.get("css", ""),
                        code_memory.get("js", ""),
                    )
                
                new_code = self.coder.fix_code(full_code, fix_instructions)
                
                updated_memory = self._extract_code_parts(new_code)
                updated_memory = self._merge_code_memory(code_memory, updated_memory)
                parser = GameCodeParser()
                new_code = parser._combine(
                    updated_memory.get("html", ""),
                    updated_memory.get("css", ""),
                    updated_memory.get("js", ""),
                )
                
                return {
                    "current_code": new_code,
                    "code_memory": updated_memory,
                    "fix_history": fix_history,
                    "iteration": iteration + 1,
                    "status": "validating",
                    "messages": [f"🔧 Applied Audit Fixes (Global Rewrite)"]
                }

            # Extract function names from issues
            function_names = self.patcher.extract_function_names_from_issues(issues)
            
            # === DEPENDENCY MAPPING (KEYHOLE FIX) ===
            try:
                from ..utils.dependency_mapper import DependencyMapper
                # Use current JS to find dependencies
                current_js = code_memory.get('js', '')
                if not current_js:
                     js_match = re.search(r'<script[^>]*>(.*?)</script>', current_code, re.DOTALL | re.IGNORECASE)
                     current_js = js_match.group(1) if js_match else current_code

                mapper = DependencyMapper(current_js)
                dependent_functions = set()
                
                # Find callers for each targeted function
                for fn in function_names:
                    callers = mapper.get_callers(fn)
                    if callers:
                        self._notify(f"🔗 [Patcher] Found callers of '{fn}': {callers}")
                        dependent_functions.update(callers)
                
                # Add dependents to patch list if not already there
                for dep in dependent_functions:
                    if dep not in function_names:
                        function_names.append(dep)
                        # Add a synthetic issue so the patcher knows WHY it's patching this
                        issues.append({
                            'location': dep, 
                            'issue': f"Dependency Update: Calls modified function(s). Check for signature mismatch.",
                            'fix': "Update call arguments if needed."
                        })
                        self._notify(f"🔗 [Patcher] Added '{dep}' to patch list (Cascading Update)")

            except Exception as dep_err:
                self._notify(f"⚠️ [Patcher] Dependency mapping failed: {dep_err}")

            # Filter out already fixed functions to avoid loops
            new_functions = []
            for fn in function_names:
                signature = ""
                for issue in issues:
                    if fn in issue.get("location", ""):
                        signature = self._issue_signature([issue])
                        break
                if fix_history.get(fn) != signature:
                    new_functions.append(fn)
            
            if new_functions:
                # Targeted function patching with SURGICAL PRECISION
                self._notify(f"🎯 [Patcher] Surgically patching functions: {new_functions}")
                
                # Use code_memory for JavaScript context
                current_js = code_memory.get('js', '')
                if not current_js:
                    # Try to extract from full code if memory empty
                    js_match = re.search(r'<script[^>]*>(.*?)</script>', current_code, re.DOTALL | re.IGNORECASE)
                    current_js = js_match.group(1) if js_match else current_code

                # Map function names to their issues
                issue_map = {}
                for issue in issues:
                    for fn in new_functions:
                        if fn in issue.get('location', ''):
                            issue_map[fn] = issue
                            break
                
                # EXTRACT CONTEXT for surgical fix
                # This prevents the LLM from hallucinating variables or losing references
                context = self._get_code_context(code_memory)

                patched_count = 0
                
                # Iterate and patch each function
                for fn in new_functions:
                    # Extract function data to get precise location and body
                    func_data = self.patcher.extract_function(current_js, fn)
                    
                    if not func_data:
                        self._notify(f"⚠️ [Patcher] Function '{fn}' not found in code, skipping.")
                        continue
                        
                    # Prepare specific instructions
                    issue = issue_map.get(fn, {})
                    fix_instructions = f"Fix {fn}: {issue.get('issue', 'Fix logic error')} -> {issue.get('fix', 'Implement correctly')}"
                    
                    # Call LLM with Context-Aware Surgical Fix
                    try:
                        patched_body = self.coder.fix_code(
                             func_data.body, 
                             fix_instructions, 
                             context=context
                        )
                        
                        # Clean up response (remove markdown and strict whitespace)
                        patched_body = patched_body.replace("```javascript", "").replace("```", "").strip()
                        
                        # INTELLIGENT WRAPPING: Check if LLM returned the full function or just body
                        # Check for function-like patterns at start (ignoring comments/whitespace)
                        # We use a simple regex to check if the code *starts* with a function definition
                        clean_start = re.sub(r'//.*?\n|/\*.*?\*/', '', patched_body, flags=re.DOTALL).strip()
                        is_full_function = any(clean_start.startswith(kw) for kw in ['function', 'async', 'class', 'static', 'get ', 'set ', func_data.name])
                        
                        if is_full_function:
                            new_function = patched_body
                        else:
                            # It's just the body, so wrap it
                            new_function = f"{func_data.signature} {{\n{patched_body}\n}}"
                        
                        # Apply patch using exact string replacement
                        if func_data.full_match in current_js:
                            current_js = current_js.replace(func_data.full_match, new_function)
                            patched_count += 1
                            self._notify(f"✅ [Patcher] Fixed '{fn}'")
                            
                            # Update fix history
                            fix_history[fn] = self._issue_signature([issue])
                            
                    except Exception as e:
                        print(f"⚠️ [Patcher] Failed to patch {fn}: {e}")

                if patched_count > 0:
                    # Update code memory
                    code_memory["js"] = current_js
                    
                    # Rebuild full code
                    from ..parsers.game_parser import GameCodeParser
                    parser = GameCodeParser()
                    final_code = parser._combine(
                        code_memory.get("html", ""),
                        code_memory.get("css", ""),
                        code_memory.get("js", ""),
                    )
                    
                    return {
                        "current_code": final_code,
                        "code_memory": code_memory,
                        "fix_history": fix_history,
                        "iteration": iteration + 1,
                        "status": "validating",
                        "messages": [f"🔧 Surgically patched {patched_count} functions"]
                    }
                else:
                    self._notify("⚠️ [Patcher] Surgical patch failed (functions not found), falling back to global fix.")
            
            # Fallback (or if no specific functions identified): Global Fix
            self._notify(f"🔧 [Patcher] Applying global fixes to {len(issues)} issues...")
            
            fix_instructions = self._format_fix_instructions(issues)
            
            # Inject Stall Warning to break loops
            stall_count = state.get("stall_count", 0)
            if stall_count > 0:
                fix_instructions = (
                    f"## ⚠️ REPEATED FAILURE WARNING ({stall_count} attempts):\n"
                    f"You have failed to fix these issues multiple times.\n"
                    f"Your previous fixes were ineffective or reverted.\n"
                    f"YOU MUST TRY A DRASTICALLY DIFFERENT APPROACH.\n"
                    f"If you were being conservative, BE BOLD.\n"
                    f"If you were using a library, try manual implementation.\n\n"
                    + fix_instructions
                )
            
            # Reconstruct full code for context
            full_code = current_code
            from ..parsers.game_parser import GameCodeParser  # Ensure import
            if code_memory.get("html") or code_memory.get("css") or code_memory.get("js"):
                parser = GameCodeParser()
                full_code = parser._combine(
                    code_memory.get("html", ""),
                    code_memory.get("css", ""),
                    code_memory.get("js", ""),
                )
            
            # Use improved global fix (stable prompt)
            new_code = self.coder.fix_code(full_code, fix_instructions)
            
            # Update code_memory
            extracted_parts = self._extract_code_parts(new_code)
            
            # DEBUG: Check if we actually got anything
            if not any(extracted_parts.values()):
                self._notify(f"⚠️ [Patcher] Global Fix returned no valid code blocks! Raw length: {len(new_code)}")
                # Fallback: Try to use new_code as a single file if it looks like HTML
                if "<html" in new_code.lower():
                     extracted_parts["html"] = new_code
                     self._notify("⚠️ [Patcher] Used raw output as HTML file.")
            
            updated_memory = self._merge_code_memory(code_memory, extracted_parts)
            
            # Check if effective change happened
            changes = []
            if len(updated_memory.get('js','')) != len(code_memory.get('js','')): changes.append('JS')
            if len(updated_memory.get('html','')) != len(code_memory.get('html','')): changes.append('HTML')
            if len(updated_memory.get('css','')) != len(code_memory.get('css','')): changes.append('CSS')
            
            if not changes:
                 self._notify("⚠️ [Patcher] Global Fix resulted in NO CHANGES. LLM output ignored?")
            else:
                 self._notify(f"✅ [Patcher] Global Fix changed: {', '.join(changes)}")

            parser = GameCodeParser()
            new_code = parser._combine(
                updated_memory.get("html", ""),
                updated_memory.get("css", ""),
                updated_memory.get("js", ""),
            )
            
            return {
                "current_code": new_code,
                "code_memory": updated_memory,
                "iteration": iteration + 1,
                "status": "validating",
                "messages": [f"🔧 Applied global fixes"]
            }
                
        except Exception as e:
            return {
                "iteration": iteration + 1,
                "status": "validating",  # Continue anyway
                "messages": [f"⚠️ Patcher error: {str(e)}"],
                "error": str(e)
            }
    
    def _format_fix_instructions(self, issues: List[Dict]) -> str:
        """Format validation issues as fix instructions."""
        if not issues:
            return "No issues found."
        
        instructions = ["## LOGIC ERRORS TO FIX:\n"]
        
        for idx, issue in enumerate(issues, 1):
            location = issue.get('location', 'Unknown location')
            problem = issue.get('issue', 'Unknown issue')
            fix = issue.get('fix', 'Fix this')
            
            instructions.append(f"""{idx}. **{location}**
   - Problem: {problem}
   - Fix: {fix}
""")
        
        return "\n".join(instructions)
    
    # ============ CONDITIONAL EDGE ============
    
    def _get_code_context(self, code_memory: Dict[str, str]) -> str:
        """Extract global context (HTML IDs, JS vars) to help the LLM patch functions safely."""
        import re
        html = code_memory.get("html", "")
        js = code_memory.get("js", "")
        
        # 1. Extract HTML IDs
        ids = re.findall(r'id=["\']([^"\']+)["\']', html)
        
        # 2. Extract Top-Level JS Variables (const/let/var at start of line)
        # simplistic regex, but effective for typical game code structure
        global_vars = re.findall(r'^(?:const|let|var)\s+(\w+)', js, re.MULTILINE)
        
        # 3. Extract Class Names
        classes = re.findall(r'class\s+(\w+)', js)
        
        context_str = "## GLOBAL CONTEXT:\n"
        if ids:
            context_str += f"HTML Elements (IDs): {', '.join(ids)}\n"
        if global_vars:
            context_str += f"Global Variables: {', '.join(global_vars)}\n"
        if classes:
            context_str += f"Classes: {', '.join(classes)}\n"
            
        return context_str

    def _should_continue(self, state: GraphState) -> Literal["complete", "patch", "failed"]:
        """
        Decide whether to continue patching, complete, or fail.
        
        Returns:
            - "complete": Validation passed (0 issues)
            - "patch": Continue fixing issues
            - "failed": Max iterations or stalls reached
        """
        # Check for validation success
        if state.get("is_valid", False):
            self._notify(f"✅ [Graph] Validation PASSED!")
            return "complete"
        
        # Check for fatal errors
        if state.get("status") == "failed":
            self._notify(f"❌ [Graph] Pipeline failed: {state.get('error', 'Unknown error')}")
            return "failed"
        
        # Check iteration limit
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", self.MAX_ITERATIONS)
        if max_iter is not None and iteration >= max_iter:
            self._notify(f"⚠️ [Graph] Max iterations ({max_iter}) reached. Using best effort.")
            return "complete"  # Return what we have
        
        # Check if only warnings remain (no critical issues) — accept the code
        issues = state.get("issues", [])
        if issues:
            has_critical = any(i.get("severity") == "critical" for i in issues)
            if not has_critical:
                self._notify(
                    f"✅ [Graph] Only {len(issues)} warning(s) remain (no critical). "
                    f"Accepting code with score {state.get('score', 0)}/100."
                )
                return "complete"
        
        # Check stall detection
        stall_count = state.get("stall_count", 0)
        if stall_count >= self.MAX_STALLS:
            self._notify(f"⚠️ [Graph] Stall detected ({stall_count} iterations without improvement). Stopping.")
            return "complete"  # Return what we have
        
        # Continue patching
        return "patch"
    
    # ============ PUBLIC INTERFACE ============
    
    def run(
        self,
        game_plan: str,
        design_spec: str = "",
        max_iterations: Optional[int] = None,
        thread_id: str = "default"
    ) -> GraphState:
        """
        Execute the coder-validator graph.
        
        Args:
            game_plan: The game specification from the planner
            design_spec: Visual design specification (optional)
            max_iterations: Maximum fix iterations before stopping (None = unlimited)
            thread_id: Thread ID for checkpointing
        
        Returns:
            Final GraphState with code and validation results
        """
        # Initialize state
        initial_state: GraphState = {
            "game_plan": game_plan,
            "design_spec": design_spec,
            "current_code": "",
            "game_files": {},
            "validation_result": None,
            "issues": [],
            "is_valid": False,
            "score": 0,
            "iteration": 0,
            "max_iterations": max_iterations,
            "stall_count": 0,
            "previous_issue_count": 999999,
            "previous_issue_signature": None,
            "previous_score": -1,
            "messages": [],
            "code_memory": {},  # Will store extracted HTML, CSS, JS separately
            "fix_history": {},  # Tracks what functions were already fixed
            "status": "coding",
            "error": None
        }
        
        # Run the graph with checkpointing
        config = {"configurable": {"thread_id": thread_id}}
        
        self._notify(f"\n{'='*50}")
        self._notify(f"🚀 [Graph] Starting Coder-Validator Graph")
        self._notify(f"{'='*50}\n")
        
        # Execute graph
        final_state = None
        for step in self.compiled_graph.stream(initial_state, config):
            # step is a dict with node_name: output
            for node_name, output in step.items():
                if isinstance(output, dict):
                    # Merge output into state tracking
                    if "current_code" in output:
                        final_state = output
        
        # Get final state from checkpointer
        checkpoint = self.memory.get(config)
        if checkpoint and "channel_values" in checkpoint:
            final_state = checkpoint["channel_values"]
        
        self._notify(f"\n{'='*50}")
        self._notify(f"🏁 [Graph] Complete - Score: {final_state.get('score', 0)}/100")
        self._notify(f"{'='*50}\n")
        
        return final_state
    
    def get_conversation_history(self, thread_id: str = "default") -> List[str]:
        """Retrieve conversation history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.memory.get(config)
        if checkpoint and "channel_values" in checkpoint:
            return checkpoint["channel_values"].get("messages", [])
        return []
