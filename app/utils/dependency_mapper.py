import re
from typing import Dict, List, Set, Optional

class DependencyMapper:
    """
    Analyzes JavaScript code to build a dependency graph of functions.
    Maps:
    - Definitions: Where functions are defined.
    - Call Graph: Which functions call which other functions.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.functions: Dict[str, str] = {}  # name -> body
        self.calls: Dict[str, Set[str]] = {} # caller -> set(callees)
        self.callers: Dict[str, Set[str]] = {} # callee -> set(callers)
        self._analyze()

    def _analyze(self):
        """Build the function map and call graph."""
        # 1. Component: Identify all function definitions
        # Matches: function foo(), const foo = (), async foo(), foo() {
        def_pattern = r'(?:function\s+|const\s+|let\s+|var\s+|async\s+)?(?:\s+)?(\w+)\s*(?:=|:|\()\s*(?:async\s+)?(?:\([^)]*\)|function\s*\([^)]*\))\s*(?:=>)?\s*\{'
        
        matches = list(re.finditer(def_pattern, self.code, re.MULTILINE))
        
        for match in matches:
            name = match.group(1)
            if name in ['if', 'for', 'while', 'switch', 'catch']: continue
            
            # Simple body extraction (heuristic: match braces)
            start = match.start()
            end = self._find_matching_brace(start)
            if end != -1:
                body = self.code[start:end+1]
                self.functions[name] = body

        # 2. Component: Analyze calls within each function body
        for func_name, body in self.functions.items():
            self.calls[func_name] = set()
            
            # Search for usages of other known functions
            # distinct from definition, look for "name(" or "name (" or "name,"
            for other_func in self.functions:
                if other_func == func_name: continue
                
                # Regex for "call": name followed by (
                # OR usage as value: name followed by , or ) or space
                # This is a heuristic.
                if re.search(r'\b' + re.escape(other_func) + r'\b', body):
                    self.calls[func_name].add(other_func)
                    
                    if other_func not in self.callers:
                        self.callers[other_func] = set()
                    self.callers[other_func].add(func_name)

    def _find_matching_brace(self, start_idx: int) -> int:
        """Find the closing brace for the block starting at start_idx."""
        # Find first '{'
        try:
            open_idx = self.code.find('{', start_idx)
            if open_idx == -1: return -1
            
            count = 1
            for i in range(open_idx + 1, len(self.code)):
                char = self.code[i]
                if char == '{':
                    count += 1
                elif char == '}':
                    count -= 1
                    if count == 0:
                        return i
            return -1
        except:
            return -1

    def get_callers(self, func_name: str) -> List[str]:
        """Get list of functions that call/use the given function."""
        return list(self.callers.get(func_name, []))

    def get_users(self, var_name: str) -> List[str]:
        """Get list of functions that use a specific global variable (heuristic)."""
        users = []
        for func_name, body in self.functions.items():
            if re.search(r'\b' + re.escape(var_name) + r'\b', body):
                users.append(func_name)
        return users
