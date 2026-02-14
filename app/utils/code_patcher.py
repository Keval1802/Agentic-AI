"""
Code Patcher Utility - Regex-based function extraction and patching.
Provides surgical patching of JavaScript functions in HTML game files.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


def normalize_whitespace(text: str) -> str:
    """Strips all whitespace/newlines for fuzzy comparison."""
    return re.sub(r'\s+', '', text)


def apply_search_replace_patch(original_code: str, patch_response: str) -> Tuple[str, int, int]:
    """
    Parse SEARCH/REPLACE blocks from LLM response and apply them to source code.
    Uses progressive matching: exact first, then fuzzy whitespace match.
    
    Returns:
        (patched_code, applied_count, failed_count)
    """
    # 1. Strip markdown code blocks if the LLM hallucinated them
    patch_response = re.sub(r'```[a-z]*\n|```', '', patch_response)
    
    # 2. Extract the SEARCH and REPLACE blocks
    pattern = r"<{4}\s*SEARCH\n(.*?)\n={4}\n(.*?)\n>{4}\s*REPLACE"
    matches = list(re.finditer(pattern, patch_response, re.DOTALL))
    
    if not matches:
        return original_code, 0, 0
    
    updated_code = original_code
    applied = 0
    failed = 0
    
    for match in matches:
        search_block = match.group(1).strip()
        replace_block = match.group(2).strip()
        
        if not search_block:
            print("⚠️ Patch: Empty SEARCH block, skipping")
            failed += 1
            continue
        
        # Strategy A: Exact Match
        if search_block in updated_code:
            updated_code = updated_code.replace(search_block, replace_block, 1)
            applied += 1
            preview = search_block[:60].replace('\n', ' ')
            print(f"✅ Patch Applied (exact): '{preview}...'")
            continue
        
        # Strategy B: Fuzzy Whitespace Match
        print("⚠️ Exact match failed. Attempting fuzzy whitespace match...")
        normalized_search = normalize_whitespace(search_block)
        
        original_lines = updated_code.split('\n')
        search_lines = search_block.split('\n')
        window_size = len(search_lines)
        found = False
        
        for i in range(len(original_lines)):
            for j in range(i + 1, min(i + window_size + 3, len(original_lines) + 1)):
                chunk = '\n'.join(original_lines[i:j])
                if normalize_whitespace(chunk) == normalized_search:
                    updated_code = updated_code.replace(chunk, replace_block, 1)
                    applied += 1
                    print(f"✅ Patch Applied (fuzzy): lines {i+1}-{j}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            preview = search_block[:80].replace('\n', ' ')
            print(f"⚠️ Patch FAILED: SEARCH block not found: '{preview}...'")
            failed += 1
    
    return updated_code, applied, failed


@dataclass
class ExtractedFunction:
    """Represents an extracted function from code."""
    name: str
    full_match: str  # The entire matched function
    signature: str   # Function signature (name + params)
    body: str        # Function body content
    start_pos: int   # Start position in original code
    end_pos: int     # End position in original code


class FunctionPatcher:
    """
    Extracts and patches JavaScript functions in HTML files.
    Supports regular functions, arrow functions, and class methods.
    """
    
    # Patterns for different function types (order matters - try most specific first)
    PATTERNS = {
        # Regular function: [async] function name(...) { ... }
        'regular': r'(?:async\s+)?function\s+{name}\s*\([^)]*\)\s*\{{',
        
        # Arrow function assigned: const/let/var name = [async] (...) => { ... }
        'arrow': r'(?:const|let|var)\s+{name}\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>\s*\{{',
        
        # Class method with newline: [async] name(...) {\n
        'method_newline': r'\n\s*(?:async\s+)?{name}\s*\([^)]*\)\s*\{{',
        
        # Class method: [async] name(...) { ... }
        'method': r'^\s*(?:async\s+)?{name}\s*\([^)]*\)\s*\{{',
        
         # Class field arrow: name = [async] (...) => { ... }
        'class_arrow': r'^\s*{name}\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>\s*\{{',
        
        # Method after another method: } [async] name(...) {
        'method_after': r'\}}\s*\n\s*(?:async\s+)?{name}\s*\([^)]*\)\s*\{{',
        
        # Object method: name: [async] function(...) { ... }
        'object': r'{name}\s*:\s*(?:async\s+)?function\s*\([^)]*\)\s*\{{',
        
        # Object arrow: name: [async] (...) => { ... }
        'object_arrow': r'{name}\s*:\s*(?:async\s+)?(?:\([^)]*\)|[^:=])\s*=>\s*\{{'
    }
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize the patcher with optional temp directory."""
        self.temp_dir = temp_dir or Path("temp_games")
        self.temp_dir.mkdir(exist_ok=True)
    
    def extract_function(self, code: str, function_name: str) -> Optional[ExtractedFunction]:
        """
        Extract a function by name from the code.
        Handles nested braces properly.
        """
        # Try each pattern type
        for pattern_type, pattern_template in self.PATTERNS.items():
            pattern = pattern_template.format(name=re.escape(function_name))
            # Enable MULTILINE mode so ^ matches start of line, not just start of string
            # Enable DOTALL is NOT desired here as . shouldn't match newlines in these patterns normally, 
            # but our patterns don't use . much. 
            match = re.search(pattern, code, re.MULTILINE)
            
            if match:
                # Found the function start, now find the matching closing brace
                start_pos = match.start()
                brace_start = match.end() - 1  # Position of opening brace
                
                # Count braces to find matching close
                body_end = self._find_matching_brace(code, brace_start)
                
                if body_end != -1:
                    full_match = code[start_pos:body_end + 1]
                    signature = match.group(0).rstrip('{').strip()
                    body = code[brace_start + 1:body_end]
                    
                    return ExtractedFunction(
                        name=function_name,
                        full_match=full_match,
                        signature=signature,
                        body=body.strip(),
                        start_pos=start_pos,
                        end_pos=body_end + 1
                    )
        
        return None
    
    def _find_matching_brace(self, code: str, start: int) -> int:
        """Find the position of the matching closing brace."""
        if code[start] != '{':
            return -1
        
        count = 1
        pos = start + 1
        in_string = False
        string_char = None
        
        while pos < len(code) and count > 0:
            char = code[pos]
            prev_char = code[pos - 1] if pos > 0 else ''
            
            # Handle strings (ignore braces inside strings)
            if char in '"\'`' and prev_char != '\\':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            
            # Count braces (only outside strings)
            if not in_string:
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
            
            pos += 1
        
        return pos - 1 if count == 0 else -1
    
    def patch_function(self, code: str, function_name: str, new_body: str) -> Optional[str]:
        """
        Replace a function's body with new content.
        Returns the patched code or None if function not found.
        Handles both pure JS and HTML with embedded script tags.
        """
        import re
        
        # Check if this is HTML with script tags
        script_match = re.search(r'(<script[^>]*>)(.*?)(</script>)', code, re.DOTALL | re.IGNORECASE)
        
        if script_match:
            # Extract JS from script tag
            script_open = script_match.group(1)
            js_code = script_match.group(2)
            script_close = script_match.group(3)
            script_start = script_match.start()
            script_end = script_match.end()
            
            # Try to extract function from JS code
            extracted = self.extract_function(js_code, function_name)
            
            if extracted:
                # Patch within the JS code
                new_body = new_body.strip()
                if not new_body.startswith('{'):
                    new_body = '{\n' + new_body
                if not new_body.endswith('}'):
                    new_body = new_body + '\n}'
                
                new_function = f"{extracted.signature} {new_body}"
                
                # Replace function in JS code
                new_js = (
                    js_code[:extracted.start_pos] +
                    new_function +
                    js_code[extracted.end_pos:]
                )
                
                # Rebuild full HTML with patched script
                new_code = (
                    code[:script_start] +
                    script_open + new_js + script_close +
                    code[script_end:]
                )
                
                print(f"✅ Patched function '{function_name}' ({len(extracted.body)} → {len(new_body)} chars)")
                return new_code
        
        # Fallback: try extracting directly from code (for pure JS files)
        extracted = self.extract_function(code, function_name)
        
        if not extracted:
            print(f"⚠️ Function '{function_name}' not found in code")
            return None
        
        # Build new function with original signature + new body
        new_body = new_body.strip()
        if not new_body.startswith('{'):
            new_body = '{\n' + new_body
        if not new_body.endswith('}'):
            new_body = new_body + '\n}'
        
        # Construct the new function
        new_function = f"{extracted.signature} {new_body}"
        
        # Replace in code
        new_code = (
            code[:extracted.start_pos] +
            new_function +
            code[extracted.end_pos:]
        )
        
        print(f"✅ Patched function '{function_name}' ({len(extracted.body)} → {len(new_body)} chars)")
        return new_code
    
    def extract_function_names_from_issues(self, issues: List[Dict]) -> List[str]:
        """Extract likely function names from validator issue payloads.

        Communication loophole fixed:
        - Validator often reports the function name in `issue` or `fix`, not only `location`.
        - If we only inspect `location`, surgical patching is skipped and we fall back to full
          regeneration loops.
        """
        function_names: List[str] = []

        # Match: someFn(), this.someFn(), "Define someFn()", "Fix someFn method".
        patterns = [
            r'(?:this\.)?(\w+)\s*\(\)',
            r'\bdefine\s+(\w+)\b',
            r'\bfix\s+(\w+)\b',
            r'\b(\w+)\s+method\b',
            r'\b(\w+)\s+function\b',
            r'→\s*(\w+)\s*\(',
        ]

        stop_words = {
            'function', 'method', 'script', 'javascript', 'html', 'css', 'code',
            'this', 'game', 'line', 'referenceerror', 'fix', 'define'
        }

        for issue in issues:
            # Check all relevant channels where validator may mention function identifiers.
            candidate_texts = [
                str(issue.get('location', '')),
                str(issue.get('issue', '')),
                str(issue.get('fix', '')),
            ]

            for text in candidate_texts:
                for pattern in patterns:
                    for match in re.findall(pattern, text, flags=re.IGNORECASE):
                        name = str(match).strip()
                        if (
                            len(name) < 2
                            or name.lower() in stop_words
                            or not re.match(r'^[A-Za-z_]\w*$', name)
                        ):
                            continue
                        if name not in function_names:
                            function_names.append(name)

        return function_names
    
    def save_temp_file(self, code: str, prefix: str = "game") -> Path:
        """Save code to a temp file and return the path."""
        import uuid
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.html"
        filepath = self.temp_dir / filename
        filepath.write_text(code, encoding='utf-8')
        print(f"📁 Saved temp file: {filepath}")
        return filepath
    
    def read_file(self, filepath: Path) -> str:
        """Read code from file."""
        return Path(filepath).read_text(encoding='utf-8')
    
    def write_file(self, filepath: Path, code: str) -> None:
        """Write code to file."""
        Path(filepath).write_text(code, encoding='utf-8')
        print(f"💾 Updated file: {filepath}")
    
    def cleanup_temp_files(self, keep_last: int = 5) -> None:
        """Clean up old temp files, keeping the most recent ones."""
        files = sorted(self.temp_dir.glob("*.html"), key=lambda f: f.stat().st_mtime)
        for f in files[:-keep_last]:
            f.unlink()
            print(f"🗑️ Deleted old temp file: {f}")
