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
    contract: str  # Planner's variable/function contract + pseudo-code logic
    
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
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("coder", self._coder_node)
        graph.add_node("validator", self._validator_node)
        graph.add_node("patcher", self._patcher_node)
        graph.add_node("reset_coder", self._reset_coder_node)
        
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
                "reset_coder": "reset_coder",
                "failed": END
            }
        )
        
        # Patcher loops back to validator
        graph.add_edge("patcher", "validator")
        # Reset coder also goes to validator
        graph.add_edge("reset_coder", "validator")
        
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
        if patched.get("html") and "<!doctype html>" in patched.get("html", "").lower():
             # Single-file override!
             # If the patch returned a full HTML file, it includes all CSS/JS inline.
             # We must clear the separate CSS/JS memory to avoid double-injection or stale code.
             merged = {
                 "html": patched.get("html"),
                 "css": "", 
                 "js": "",
                 "full": patched.get("full")
             }
        else:
             # Legacy/Split-mode merge
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
                state.get("design_spec", ""),
                contract=state.get("contract", "")
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
            result = self.validator.validate(current_code, contract=state.get("contract", ""))
            
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
        Patcher Node: Fix identified issues using Search/Replace diffs.
        
        Primary strategy: LLM outputs <<<< SEARCH / ==== / >>>> REPLACE blocks
        that are applied as precise string replacements on the master code.
        Fallback: Full-file fix_code() if no patches can be applied.
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
            # ========== STEP 0: Deterministic placeholder fixes ==========
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

            # ========== STEP 1: Audit Rejection → full rewrite ==========
            audit_rejection = next(
                (i for i in issues if i.get('location') == 'General Audit' and 'REJECTED' in i.get('issue', '')),
                None
            )
            
            if audit_rejection:
                fix_directive = audit_rejection.get('fix', 'Resolve critical logic flaws.')
                self._notify(f"🚨 [Patcher] AUDIT REJECTION. Full rewrite triggered.")
                
                fix_instructions = (
                    f"## AUDITOR REJECTION:\n"
                    f"The code was REJECTED. Fix this CRITICAL issue:\n\n"
                    f"{fix_directive}\n\n"
                    f"Review the logic from scratch and rewrite the broken parts."
                )
                
                full_code = current_code
                from ..parsers.game_parser import GameCodeParser
                if code_memory.get("html") or code_memory.get("css") or code_memory.get("js"):
                    parser = GameCodeParser()
                    full_code = parser._combine(
                        code_memory.get("html", ""),
                        code_memory.get("css", ""),
                        code_memory.get("js", ""),
                    )
                
                new_code = self.coder.fix_code(full_code, fix_instructions, contract=state.get("contract", ""))
                
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

            # ========== STEP 2: Search/Replace Patching (PRIMARY) ==========
            if issues:
                self._notify(f"🔍 [Patcher] Using Search/Replace diff strategy for {len(issues)} issues...")
                
                # Reconstruct full code from memory
                full_code = current_code
                from ..parsers.game_parser import GameCodeParser
                if code_memory.get("html") or code_memory.get("css") or code_memory.get("js"):
                    parser = GameCodeParser()
                    full_code = parser._combine(
                        code_memory.get("html", ""),
                        code_memory.get("css", ""),
                        code_memory.get("js", ""),
                    )
                
                # Call LLM with search/replace prompt
                try:
                    patch_response = self.coder.patch_code(
                        full_code=full_code,
                        issues=issues,
                        contract=state.get("contract", "")
                    )
                    
                    # Apply the patches
                    from ..utils.code_patcher import apply_search_replace_patch
                    patched_code, applied_count, failed_count = apply_search_replace_patch(
                        full_code, patch_response
                    )
                    
                    self._notify(
                        f"📋 [Patcher] Search/Replace: {applied_count} applied, {failed_count} failed"
                    )
                    
                    if applied_count > 0:
                        # Update code memory from patched result
                        extracted_parts = self._extract_code_parts(patched_code)
                        updated_memory = self._merge_code_memory(code_memory, extracted_parts)
                        
                        parser = GameCodeParser()
                        final_code = parser._combine(
                            updated_memory.get("html", ""),
                            updated_memory.get("css", ""),
                            updated_memory.get("js", ""),
                        )
                        
                        return {
                            "current_code": final_code,
                            "code_memory": updated_memory,
                            "fix_history": fix_history,
                            "iteration": iteration + 1,
                            "status": "validating",
                            "messages": [f"🔧 Applied {applied_count} search/replace patches"]
                        }
                    else:
                        self._notify("⚠️ [Patcher] No search/replace patches applied, falling back to full-file fix...")
                        
                except Exception as patch_err:
                    self._notify(f"⚠️ [Patcher] Search/Replace error: {patch_err}, falling back to full-file fix...")

            # ========== STEP 3: Fallback — full-file fix_code ==========
            self._notify(f"🔧 [Patcher] Fallback: Applying full-file fix for {len(issues)} issues...")
            
            fix_instructions = self._format_fix_instructions(issues)
            
            # Inject stall warning to break loops
            stall_count = state.get("stall_count", 0)
            if stall_count > 0:
                fix_instructions = (
                    f"## ⚠️ REPEATED FAILURE WARNING ({stall_count} attempts):\n"
                    f"Your previous fixes were ineffective.\n"
                    f"YOU MUST TRY A DRASTICALLY DIFFERENT APPROACH.\n\n"
                    + fix_instructions
                )
            
            full_code = current_code
            from ..parsers.game_parser import GameCodeParser
            if code_memory.get("html") or code_memory.get("css") or code_memory.get("js"):
                parser = GameCodeParser()
                full_code = parser._combine(
                    code_memory.get("html", ""),
                    code_memory.get("css", ""),
                    code_memory.get("js", ""),
                )
            
            new_code = self.coder.fix_code(full_code, fix_instructions, contract=state.get("contract", ""))
            
            extracted_parts = self._extract_code_parts(new_code)
            
            if not any(extracted_parts.values()):
                self._notify(f"⚠️ [Patcher] Fallback returned no valid code! Raw length: {len(new_code)}")
                if "<html" in new_code.lower():
                    extracted_parts["html"] = new_code
            
            updated_memory = self._merge_code_memory(code_memory, extracted_parts)
            
            # Check for effective change
            changes = []
            if len(updated_memory.get('js','')) != len(code_memory.get('js','')): changes.append('JS')
            if len(updated_memory.get('html','')) != len(code_memory.get('html','')): changes.append('HTML')
            if len(updated_memory.get('css','')) != len(code_memory.get('css','')): changes.append('CSS')
            
            if not changes:
                self._notify("⚠️ [Patcher] Fallback resulted in NO CHANGES.")
            else:
                self._notify(f"✅ [Patcher] Fallback changed: {', '.join(changes)}")

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
                "messages": [f"🔧 Applied fallback fixes"]
            }
                
        except Exception as e:
            return {
                "iteration": iteration + 1,
                "status": "validating",
                "messages": [f"⚠️ Patcher error: {str(e)}"],
                "error": str(e)
            }
    
    def _reset_coder_node(self, state: GraphState) -> dict:
        """
        Circuit Breaker Node: Wipe poisoned context and regenerate from scratch.
        
        Called when the patcher has failed repeatedly (stall_count >= 2, iteration >= 3).
        Uses the original game_plan and contract to produce a fresh codebase.
        """
        self._notify("🧹 [Reset] Wiping context and regenerating from original plan...")
        
        game_plan = state.get("game_plan", "")
        design_spec = state.get("design_spec", "")
        contract = state.get("contract", "")
        
        try:
            # Add failure context so the coder knows what went wrong
            reset_plan = (
                f"{game_plan}\n\n"
                f"## ⚠️ REGENERATION NOTICE:\n"
                f"The previous code generation attempt failed after {state.get('iteration', 0)} "
                f"patching iterations. The patcher could not fix the bugs.\n"
                f"Generate a CLEAN, COMPLETE implementation from scratch.\n"
                f"Do NOT repeat the same mistakes."
            )
            
            new_code = self.coder.code(reset_plan, design_spec, contract=contract)
            
            if not new_code or len(new_code) < 500:
                self._notify("⚠️ [Reset] Regenerated code too short. Keeping previous code.")
                return {
                    "iteration": 1,  # Reset counter
                    "stall_count": 0,
                    "fix_history": {},
                    "status": "validating",
                    "messages": ["⚠️ Reset coder returned no valid code"]
                }
            
            code_memory = self._extract_code_parts(new_code)
            
            self._notify(f"✅ [Reset] Fresh code generated: {len(new_code)} chars")
            
            return {
                "current_code": new_code,
                "code_memory": code_memory,
                "iteration": 1,  # Reset counter to give fresh code a fair chance
                "stall_count": 0,
                "previous_issue_count": 999999,
                "previous_issue_signature": None,
                "previous_score": -1,
                "fix_history": {},
                "status": "validating",
                "messages": ["🧹 Circuit Breaker: Fresh code generated from original plan"]
            }
            
        except Exception as e:
            self._notify(f"⚠️ [Reset] Error: {e}")
            return {
                "iteration": 1,
                "stall_count": 0,
                "status": "validating",
                "messages": [f"⚠️ Reset error: {str(e)}"],
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

    def _should_continue(self, state: GraphState) -> Literal["complete", "patch", "reset_coder", "failed"]:
        """
        Decide whether to continue patching, reset, complete, or fail.
        
        Circuit Breaker: After 3 failed patcher iterations, route to reset_coder
        to wipe context and regenerate from the original plan.
        
        Returns:
            - "complete": Validation passed (0 issues)
            - "patch": Continue fixing issues
            - "reset_coder": Wipe and regenerate (circuit breaker triggered)
            - "failed": Max iterations or stalls reached
        """
        # 1. Success: Validation passed
        if state.get("is_valid", False):
            self._notify(f"✅ [Graph] Validation PASSED!")
            return "complete"
        
        # 2. Fatal error
        if state.get("status") == "failed":
            self._notify(f"❌ [Graph] Pipeline failed: {state.get('error', 'Unknown error')}")
            return "failed"
        
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations") or 15
        
        # 3. Hard stop: max iterations
        if iteration >= max_iter:
            self._notify(f"⚠️ [Graph] Max iterations ({max_iter}) reached. Stopping.")
            return "complete"
        
        # 4. Circuit Breaker: after 3 patcher iterations, wipe and retry
        issues = state.get("issues", [])
        stall_count = state.get("stall_count", 0)
        
        if iteration >= 3 and stall_count >= 2 and issues:
            self._notify(
                f"🛑 [Graph] Circuit Breaker: {stall_count} stalls after {iteration} iterations. "
                f"Wiping context and regenerating from original plan."
            )
            return "reset_coder"
        
        # 5. Zero Tolerance: any issues → keep patching
        if issues:
            self._notify(
                f"⚠️ [Graph] {len(issues)} issue(s) remain. Continuing (Zero Tolerance)."
            )
            return "patch"
        
        # 6. Stall detection (no issues but not valid?)
        if stall_count >= self.MAX_STALLS:
            self._notify(f"⚠️ [Graph] Stall limit ({stall_count}) reached. Stopping.")
            return "complete"
        
        return "patch"
    
    # ============ PUBLIC INTERFACE ============
    
    def run(
        self,
        game_plan: str,
        design_spec: str = "",
        contract: str = "",
        max_iterations: Optional[int] = 15,
        thread_id: str = "default"
    ) -> GraphState:
        """
        Execute the coder-validator graph.
        
        Args:
            game_plan: The game specification from the planner
            design_spec: Visual design specification (optional)
            contract: Planner's variable/function contract (optional)
            max_iterations: Maximum fix iterations before stopping (None = unlimited)
            thread_id: Thread ID for checkpointing
        
        Returns:
            Final GraphState with code and validation results
        """
        # Initialize state
        initial_state: GraphState = {
            "game_plan": game_plan,
            "design_spec": design_spec,
            "contract": contract,
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
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}
        
        self._notify(f"\n{'='*50}")
        self._notify(f"🚀 [Graph] Starting Coder-Validator Graph")
        self._notify(f"{'='*50}\n")
        
        # Execute graph — catch recursion limit gracefully
        final_state = None
        try:
            for step in self.compiled_graph.stream(initial_state, config):
                # step is a dict with node_name: output
                for node_name, output in step.items():
                    if isinstance(output, dict):
                        # Merge output into state tracking
                        if "current_code" in output:
                            final_state = output
        except Exception as graph_err:
            err_str = str(graph_err).lower()
            if "recursion" in err_str or "recursion_limit" in err_str:
                self._notify(f"⚠️ [Graph] Recursion limit reached. Returning best code so far.")
            else:
                self._notify(f"⚠️ [Graph] Error: {graph_err}. Returning best code so far.")
        
        # Get final state from checkpointer
        checkpoint = self.memory.get(config)
        if checkpoint and "channel_values" in checkpoint:
            final_state = checkpoint["channel_values"]
        
        # Fallback: if final_state is still None, return initial state with error
        if final_state is None:
            final_state = initial_state
            final_state["status"] = "failed"
            final_state["error"] = "Graph produced no output"
        
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
