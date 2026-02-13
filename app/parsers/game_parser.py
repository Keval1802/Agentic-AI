"""Game Code Parser - Extracts HTML/CSS/JS from LLM output."""

import re
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field


class GameMetadata(BaseModel):
    """Metadata about the generated game."""
    title: str = Field(default="Generated Game", description="Game title")
    description: str = Field(default="", description="Game description")
    has_html: bool = Field(default=False)
    has_css: bool = Field(default=False)
    has_js: bool = Field(default=False)


class GameCodeParser:
    """
    Parser for extracting game code from LLM responses.

    Priorities:
    1) Explicit fenced blocks: ```html, ```css, ```javascript
    2) Complete HTML document with inline <style>/<script>
    3) Single untagged fence (best-effort detection)
    """

    def __init__(self):
        self.metadata = GameMetadata()

    def _strip_thinking_tokens(self, text: str) -> str:
        clean = re.sub(r"<(?:thinking|think)>.*?</(?:thinking|think)>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return clean.strip()

    def _looks_like_html(self, content: str) -> bool:
        if not content:
            return False
        lowered = content.lower()
        return "<!doctype" in lowered or "<html" in lowered or "<body" in lowered

    def _looks_like_css(self, content: str) -> bool:
        if not content or "<" in content:
            return False
        return "{" in content and "}" in content and ":" in content

    def _looks_like_js(self, content: str) -> bool:
        if not content:
            return False
        lowered = content.lower()
        if "<!doctype" in lowered or "<html" in lowered:
            return False
        return any(kw in content for kw in ["function", "=>", "const ", "let ", "var ", "class "])

    def _extract_fenced_blocks(self, text: str) -> List[Tuple[str, str]]:
        pattern = r"```(?P<lang>\w+)?\s*\n(?P<body>.*?)```"
        blocks = []
        for match in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
            lang = (match.group("lang") or "").strip().lower()
            body = match.group("body").strip()
            if body:
                blocks.append((lang, body))
        return blocks

    def _extract_complete_html(self, text: str) -> Optional[str]:
        text = text.strip()
        block_pattern = r"```(?:html|xml)?\s*(<!DOCTYPE\s+html>.*?</html>)\s*```"
        match = re.search(block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        raw_pattern = r"(<!DOCTYPE\s+html>.*?</html>)"
        match = re.search(raw_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def parse(self, llm_output) -> str:
        files = self.parse_multi_file(llm_output)
        return self._combine(files.get("html", ""), files.get("css", ""), files.get("js", ""))

    def parse_multi_file(self, llm_output) -> dict:
        if isinstance(llm_output, list):
            llm_output = "\n".join(str(part) for part in llm_output)
        llm_output = self._strip_thinking_tokens(str(llm_output).strip())

        html = ""
        css = ""
        js = ""

        blocks = self._extract_fenced_blocks(llm_output)

        for lang, body in blocks:
            if not html and lang in ["html", "xml"]:
                html = body
            elif not css and lang in ["css", "style"]:
                css = body
            elif not js and lang in ["javascript", "js"]:
                js = body

        if not html:
            complete_html = self._extract_complete_html(llm_output)
            if complete_html:
                # SINGLE FILE DETECTED: Keep it intact!
                # Do NOT strip styles/scripts. Do NOT extract to separate vars.
                html = complete_html
                
                # We can still populate metadata by inspecting (non-destructively)
                self.metadata.has_css = "<style" in html.lower()
                self.metadata.has_js = "<script" in html.lower()

        if not html and blocks:
            untagged = [b for (lang, b) in blocks if not lang]
            if len(untagged) == 1:
                candidate = untagged[0]
                if self._looks_like_html(candidate):
                    html = candidate
                elif self._looks_like_css(candidate):
                    css = candidate
                elif self._looks_like_js(candidate):
                    js = candidate

        # Only inject external links if we actually have SEPARATE code and no inline equivalents
        is_complete_doc = "<html" in html.lower()
        has_inline_style = "<style" in html.lower()
        has_inline_script = "<script" in html.lower()

        # Inject CSS link only if we have CSS but no inline styles, and it's not already linked
        if css and not has_inline_style and "<link" not in html.lower():
             html = re.sub(
                r"(</head>)",
                '    <link rel="stylesheet" href="style.css">\n\\1',
                html,
                flags=re.IGNORECASE
            )
            
        # Inject JS script only if we have JS but no inline scripts, and it's not already linked
        if js and not has_inline_script and 'src="script.js"' not in html.lower():
             html = re.sub(
                r"(</body>)",
                '    <script src="script.js"></script>\n\\1',
                html,
                flags=re.IGNORECASE
            )

        if js and len(js.strip()) < 50:
            js = ""

        self.metadata.has_html = bool(html)
        self.metadata.has_css = bool(css)
        self.metadata.has_js = bool(js)
        self.metadata.title = self._extract_title(html) if html else "Generated Game"



        return {
            "html": html or self._default_html(),
            "css": css, # Return empty if empty (don't inject comments)
            "js": js,   # Return empty if empty
        }

    def _default_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Game</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="game-container">
        <h1>Game</h1>
    </div>
    <script src="script.js"></script>
</body>
</html>'''

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE)
        if match:
            return re.sub(r"<[^>]+>", "", match.group(1)).strip()
        return "Generated Game"

    def _combine(self, html: str, css: str, js: str) -> str:
        if html and "<html" in html.lower():
            return self._inject_into_html(html, css, js)
        body_content = html if html else '<div id="game-container"><h1>Game</h1></div>'
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Game</title>
    <style>
        {css}
    </style>
</head>
<body>
    {body_content}
    <script>
        {js}
    </script>
</body>
</html>'''

    def _inject_into_html(self, html: str, css: str, js: str) -> str:
        # Remove external references since we are injecting inline
        html = re.sub(r'<link[^>]*href=["\']style\.css["\'][^>]*>', '', html, flags=re.IGNORECASE)
        html = re.sub(r'<script[^>]*src=["\']script\.js["\'][^>]*>\s*</script>', '', html, flags=re.IGNORECASE)

        if css and "<style" not in html.lower():
            style_tag = f"<style>\n{css}\n</style>"
            html = re.sub(r"</head>", f"{style_tag}\n</head>", html, flags=re.IGNORECASE)
        if js and js not in html:
            script_tag = f"<script>\n{js}\n</script>"
            html = re.sub(r"</body>", f"{script_tag}\n</body>", html, flags=re.IGNORECASE)
        self.metadata.title = self._extract_title(html)
        return html

    def get_metadata(self) -> GameMetadata:
        return self.metadata
