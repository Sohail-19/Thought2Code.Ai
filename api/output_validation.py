"""Deterministic offline checks for generated web output.

HTMLParser plus a conservative element-stack check detects common malformed
HTML nesting. CSS checking validates balanced delimiters, strings, and comments;
it is not a complete CSS grammar parser. Node's ``--check`` parses JavaScript
from standard input without executing it. Missing required scripts/stylesheets/imports, unsafe
references, detected malformed source, and unavailable or failed required syntax
checks are blocking. Missing optional assets are warnings. External, data, and
fragment URLs are ignored. Only ``.js/.mjs/.cjs`` receive JavaScript syntax parsing;
inline scripts are also checked. JSX and TypeScript syntax is not checked.

Static JavaScript import/export-from declarations are found with a bounded
scanner that ignores comments, quoted strings, templates, and recognizable
regular-expression literals. Relative and root-relative imports are checked within the job.
Extensionless imports resolve only when exactly one of the exact file, `.js`,
`.mjs`, `.cjs`, or (for a directory) `index.js/.mjs/.cjs` candidates exists;
ambiguous matches block completion. Dynamic imports are deliberately not
resolved. If the scanner cannot safely tokenize a source file, that file blocks
completion. This scanner is not a full ECMAScript parser.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit


MAX_VALIDATION_FILE_BYTES = 1_048_576
MAX_JAVASCRIPT_SCAN_TOKENS = 100_000
MAX_JAVASCRIPT_IMPORTS_PER_FILE = 1_000
MAX_JAVASCRIPT_IMPORT_SCAN_STEPS = 500_000
JS_SYNTAX_EXTENSIONS = {".js", ".mjs", ".cjs"}
JS_MODULE_EXTENSIONS = JS_SYNTAX_EXTENSIONS | {".jsx", ".ts", ".tsx"}
JS_EXTENSIONLESS_SUFFIXES = (".js", ".mjs", ".cjs")
HTML_EXTENSIONS = {".html", ".htm"}
CSS_EXTENSIONS = {".css"}
OPTIONAL_HTML_ASSETS = {
    ("img", "src"),
    ("img", "srcset"),
    ("source", "src"),
    ("source", "srcset"),
    ("audio", "src"),
    ("video", "src"),
    ("video", "poster"),
    ("track", "src"),
    ("object", "data"),
    ("embed", "src"),
    ("image", "href"),
    ("image", "xlink:href"),
    ("a", "href"),
}
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
JAVASCRIPT_MIME_TYPES = {
    "application/ecmascript",
    "application/javascript",
    "text/ecmascript",
    "text/javascript",
}
OPTIONAL_END_TAGS = {
    "dd", "dt", "li", "optgroup", "option", "p", "rb", "rp", "rt", "rtc",
    "tbody", "td", "tfoot", "th", "thead", "tr",
}
BLOCK_STARTS_ENDING_P = {
    "address", "article", "aside", "blockquote", "div", "dl", "fieldset",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "main", "nav", "ol", "p", "pre", "section", "table", "ul",
}
END_TAG_CLOSING_OPTIONAL = {
    "body": OPTIONAL_END_TAGS,
    "html": OPTIONAL_END_TAGS,
    "ol": {"li"},
    "ul": {"li"},
    "dl": {"dt", "dd"},
    "select": {"option", "optgroup"},
    "optgroup": {"option"},
    "table": {"td", "th", "tr", "tbody", "thead", "tfoot"},
    "tr": {"td", "th"},
}


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    severity: str
    path: str
    reference: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[ValidationFinding, ...]
    warnings: tuple[ValidationFinding, ...]

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(sorted({finding.code for finding in self.errors}))

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(sorted({finding.code for finding in self.warnings}))


@dataclass(frozen=True)
class _JavaScriptToken:
    kind: str
    value: str


def _skip_quoted_javascript(source: str, index: int) -> tuple[int, bool]:
    quote = source[index]
    index += 1
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
        elif char == quote:
            return index + 1, True
        elif char in "\r\n":
            return index, False
        else:
            index += 1
    return index, False


def _decode_javascript_string(value: str) -> str | None:
    """Decode common ECMAScript string escapes used in import specifiers."""
    decoded: list[str] = []
    index = 0
    simple_escapes = {
        "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
        "v": "\v", "0": "\0", "\\": "\\", "'": "'", '"': '"', "/": "/",
    }
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        index += 1
        if index >= len(value):
            return None
        escape = value[index]
        if escape in "\r\n":
            if escape == "\r" and index + 1 < len(value) and value[index + 1] == "\n":
                index += 1
            index += 1
            continue
        if escape in simple_escapes:
            decoded.append(simple_escapes[escape])
            index += 1
            continue
        if escape == "x":
            digits = value[index + 1:index + 3]
            if len(digits) != 2 or not re.fullmatch(r"[0-9a-fA-F]{2}", digits):
                return None
            decoded.append(chr(int(digits, 16)))
            index += 3
            continue
        if escape == "u":
            if index + 1 < len(value) and value[index + 1] == "{":
                end = value.find("}", index + 2)
                digits = value[index + 2:end] if end >= 0 else ""
                if not digits or not re.fullmatch(r"[0-9a-fA-F]{1,6}", digits):
                    return None
                codepoint = int(digits, 16)
                if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                    return None
                decoded.append(chr(codepoint))
                index = end + 1
                continue
            digits = value[index + 1:index + 5]
            if len(digits) != 4 or not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                return None
            codepoint = int(digits, 16)
            if 0xD800 <= codepoint <= 0xDFFF:
                return None
            decoded.append(chr(codepoint))
            index += 5
            continue
        if escape in "123456789":
            return None
        # ECMAScript NonEscapeCharacter identity escape.
        decoded.append(escape)
        index += 1
    return "".join(decoded)


def _skip_template_expression(source: str, index: int) -> tuple[int, bool]:
    """Skip a template literal including nested ${...} expressions."""
    index += 1
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
        elif char == "`":
            return index + 1, True
        elif source.startswith("${", index):
            index, complete = _skip_javascript_braces(source, index + 2)
            if not complete:
                return index, False
        else:
            index += 1
    return index, False


def _skip_javascript_braces(source: str, index: int) -> tuple[int, bool]:
    depth = 1
    while index < len(source):
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                return len(source), False
            index = end + 2
        elif source[index] in {"'", '"'}:
            index, complete = _skip_quoted_javascript(source, index)
            if not complete:
                return index, False
        elif source[index] == "`":
            index, complete = _skip_template_expression(source, index)
            if not complete:
                return index, False
        elif source[index] == "{":
            depth += 1
            index += 1
        elif source[index] == "}":
            depth -= 1
            index += 1
            if depth == 0:
                return index, True
        else:
            index += 1
    return index, False


def _skip_javascript_regex(source: str, index: int) -> tuple[int, bool]:
    index += 1
    in_character_class = False
    while index < len(source) and source[index] not in "\r\n":
        char = source[index]
        if char == "\\":
            index += 2
        elif char == "[":
            in_character_class = True
            index += 1
        elif char == "]":
            in_character_class = False
            index += 1
        elif char == "/" and not in_character_class:
            index += 1
            while index < len(source) and (source[index].isalpha() or source[index].isdigit()):
                index += 1
            return index, True
        else:
            index += 1
    return index, False


def _javascript_regex_can_start(tokens: list[_JavaScriptToken]) -> bool:
    if not tokens:
        return True
    previous = tokens[-1].value
    if previous == ")":
        depth = 0
        for index in range(len(tokens) - 1, -1, -1):
            value = tokens[index].value
            if value == ")":
                depth += 1
            elif value == "(":
                depth -= 1
                if depth == 0:
                    return index > 0 and tokens[index - 1].value in {
                        "if", "while", "for", "with", "switch", "catch",
                    }
    return previous in {
        "(", "[", "{", ",", ";", ":", "=", "!", "?", "~", "+", "-",
        "*", "%", "&", "|", "^", "<", ">", "return", "case", "throw",
        "typeof", "void", "delete", "in", "of", "yield", "await", "=>",
    }


def _scan_javascript_tokens(source: str) -> tuple[list[_JavaScriptToken], bool]:
    """Tokenize enough JS syntax to locate static imports without executing code."""
    tokens: list[_JavaScriptToken] = []
    index = 0
    complete = True
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
        elif source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                complete = False
                break
            index = end + 2
        elif char in {"'", '"'}:
            end, closed = _skip_quoted_javascript(source, index)
            if not closed:
                complete = False
                break
            raw = source[index + 1:end - 1]
            decoded = _decode_javascript_string(raw)
            if decoded is None:
                complete = False
                tokens.append(_JavaScriptToken("escaped-string", raw))
            else:
                tokens.append(_JavaScriptToken("string", decoded))
            index = end
        elif char == "`":
            index, closed = _skip_template_expression(source, index)
            if not closed:
                complete = False
                break
        elif char == "/" and _javascript_regex_can_start(tokens):
            index, closed = _skip_javascript_regex(source, index)
            if not closed:
                complete = False
                break
        elif char.isalpha() or char in "_$":
            start = index
            index += 1
            while index < len(source) and (source[index].isalnum() or source[index] in "_$"):
                index += 1
            tokens.append(_JavaScriptToken("identifier", source[start:index]))
        else:
            tokens.append(_JavaScriptToken("punctuation", char))
            index += 1
        if len(tokens) > MAX_JAVASCRIPT_SCAN_TOKENS:
            return tokens, False
    return tokens, complete


def _javascript_static_imports(source: str) -> tuple[list[str], bool]:
    tokens, complete = _scan_javascript_tokens(source)
    references: list[str] = []
    scan_steps = 0

    def add_reference(value: str) -> bool:
        references.append(value)
        return len(references) <= MAX_JAVASCRIPT_IMPORTS_PER_FILE

    for index, token in enumerate(tokens):
        if token.kind != "identifier" or token.value not in {"import", "export"}:
            continue
        following = tokens[index + 1:index + 2]
        if token.value == "import":
            if following and following[0].value in {".", "("}:
                # import.meta and dynamic imports are outside this check.
                continue
            if following and following[0].kind == "string":
                if not add_reference(following[0].value):
                    return list(dict.fromkeys(references)), False
                continue

        if token.value == "export" and not (
            following and following[0].value in {"*", "{"}
        ):
            continue

        cursor = index + 1
        while cursor < len(tokens):
            scan_steps += 1
            if scan_steps > MAX_JAVASCRIPT_IMPORT_SCAN_STEPS:
                return list(dict.fromkeys(references)), False
            current = tokens[cursor]
            if current.value == ";":
                break
            if (
                current.kind == "identifier"
                and current.value == "from"
                and cursor + 1 < len(tokens)
                and tokens[cursor + 1].kind == "string"
            ):
                if not add_reference(tokens[cursor + 1].value):
                    return list(dict.fromkeys(references)), False
                break
            cursor += 1
    return list(dict.fromkeys(references)), complete


class _HTMLStructureParser(HTMLParser):
    def __init__(self, path: str):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.stack: list[tuple[str, int]] = []
        self.references: list[tuple[str, str, int]] = []
        self.inline_styles: list[tuple[str, int]] = []
        self.inline_scripts: list[tuple[str, int]] = []
        self.active_inline_script: tuple[int, list[str]] | None = None
        self.errors: list[ValidationFinding] = []

    def _error(self, code: str) -> None:
        self.errors.append(ValidationFinding(code, "error", self.path))

    def _close_optional_for_start(self, tag: str) -> None:
        closable = set()
        if tag == "li":
            closable = {"li"}
        elif tag in {"dt", "dd"}:
            closable = {"dt", "dd"}
        elif tag == "option":
            closable = {"option"}
        elif tag == "optgroup":
            closable = {"option", "optgroup"}
        elif tag == "tr":
            closable = {"td", "th", "tr"}
        elif tag in {"td", "th"}:
            closable = {"td", "th"}
        elif tag in {"tbody", "thead", "tfoot"}:
            closable = {"td", "th", "tr", "tbody", "thead", "tfoot"}
        elif tag in BLOCK_STARTS_ENDING_P:
            closable = {"p"}
        if self.stack and self.stack[-1][0] in closable:
            self.stack.pop()

    def _handle_tag(self, tag: str, attrs: list[tuple[str, str | None]], closed: bool):
        tag = tag.lower()
        line = self.getpos()[0]
        self._close_optional_for_start(tag)
        values = {name.lower(): value or "" for name, value in attrs}
        if tag == "script" and "src" in values:
            self.references.append((values["src"], "required", line))
        elif tag == "script" and not closed:
            script_type = values.get("type", "").lower().split(";", 1)[0].strip()
            if not script_type or script_type == "module" or script_type in JAVASCRIPT_MIME_TYPES:
                self.active_inline_script = (line, [])
        elif tag == "link" and "href" in values:
            rels = set(values.get("rel", "").lower().split())
            if "stylesheet" in rels:
                self.references.append((values["href"], "required", line))
            elif rels & {"icon", "manifest", "preload", "modulepreload"}:
                self.references.append((values["href"], "optional", line))

        for asset_tag, attribute in OPTIONAL_HTML_ASSETS:
            value = values.get(attribute)
            if tag == asset_tag and value:
                if attribute == "srcset":
                    if "data:" in value.lower():
                        continue
                    for candidate in value.split(","):
                        reference = candidate.strip().split()
                        if reference:
                            self.references.append((reference[0], "optional", line))
                else:
                    self.references.append((value, "optional", line))

        if values.get("style"):
            self.inline_styles.append((values["style"], line))
        if not closed and tag not in VOID_TAGS:
            self.stack.append((tag, line))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_tag(tag, attrs, False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_tag(tag, attrs, True)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "script" and self.active_inline_script is not None:
            line, chunks = self.active_inline_script
            self.inline_scripts.append(("".join(chunks), line))
            self.active_inline_script = None
        closable = END_TAG_CLOSING_OPTIONAL.get(tag, set())
        while self.stack and self.stack[-1][0] in closable:
            self.stack.pop()
        if tag in OPTIONAL_END_TAGS and tag not in {item[0] for item in self.stack}:
            return
        matching = next(
            (index for index in range(len(self.stack) - 1, -1, -1) if self.stack[index][0] == tag),
            None,
        )
        if matching is None:
            self._error("html_unmatched_closing_tag")
        else:
            if matching != len(self.stack) - 1:
                self._error("html_mismatched_tag_nesting")
            del self.stack[matching:]

    def handle_data(self, data: str) -> None:
        if self.active_inline_script is not None:
            self.active_inline_script[1].append(data)
        elif self.stack and self.stack[-1][0] == "style" and data.strip():
            self.inline_styles.append((data, self.getpos()[0]))

    def finish(self) -> None:
        if self.active_inline_script is not None:
            line, chunks = self.active_inline_script
            self.inline_scripts.append(("".join(chunks), line))
            self.active_inline_script = None
        for tag, _line in self.stack:
            if tag not in OPTIONAL_END_TAGS:
                self._error("html_unclosed_tag")


def _check_css_structure(source: str, path: str) -> list[ValidationFinding]:
    findings = []
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = None
    escaped = False
    comment = False
    index = 0
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if comment:
            if char == "*" and following == "/":
                comment = False
                index += 2
                continue
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and following == "*":
            comment = True
            index += 2
            continue
        elif char in {"'", '"'}:
            quote = char
        elif char in "({[":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                findings.append(ValidationFinding("css_unbalanced_delimiter", "error", path))
                return findings
        index += 1
    if quote or comment or stack:
        findings.append(ValidationFinding("css_unclosed_construct", "error", path))
    return findings


def _read_css_string(source: str, index: int) -> tuple[str, int]:
    quote = source[index]
    index += 1
    value = []
    while index < len(source):
        char = source[index]
        if char == "\\" and index + 1 < len(source):
            value.append(source[index + 1])
            index += 2
        elif char == quote:
            return "".join(value), index + 1
        else:
            value.append(char)
            index += 1
    return "".join(value), index


def _read_css_url(source: str, index: int) -> tuple[str, int] | None:
    if index and (source[index - 1].isalnum() or source[index - 1] in "_-"):
        return None
    match = re.match(r"url\s*\(", source[index:], re.IGNORECASE)
    if not match:
        return None
    index += match.end()
    while index < len(source) and source[index].isspace():
        index += 1
    if index < len(source) and source[index] in {"'", '"'}:
        value, index = _read_css_string(source, index)
        while index < len(source) and source[index].isspace():
            index += 1
        if index >= len(source) or source[index] != ")":
            return None
        return value, index + 1

    start = index
    escaped = False
    while index < len(source):
        char = source[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ")":
            return source[start:index].strip(), index + 1
        index += 1
    return None


def _css_references(source: str) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    index = 0
    while index < len(source):
        if source.startswith("/*", index):
            comment_end = source.find("*/", index + 2)
            if comment_end == -1:
                break
            index = comment_end + 2
            continue
        if source[index] in {"'", '"'}:
            _value, index = _read_css_string(source, index)
            continue
        if source[index] == "@":
            import_match = re.match(r"@import\b", source[index:], re.IGNORECASE)
            if import_match:
                index += import_match.end()
                while index < len(source) and source[index].isspace():
                    index += 1
                url_result = _read_css_url(source, index)
                if url_result:
                    reference, index = url_result
                    references.append((reference, "required"))
                    continue
                if index < len(source) and source[index] in {"'", '"'}:
                    reference, index = _read_css_string(source, index)
                    references.append((reference, "required"))
                    continue
                continue
        url_result = _read_css_url(source, index)
        if url_result:
            reference, index = url_result
            references.append((reference, "optional"))
            continue
        index += 1
    return list(dict.fromkeys(references))


def _resolve_local_reference(
    job_root: pathlib.Path, source_path: pathlib.Path, reference: str
) -> tuple[pathlib.Path | None, str | None]:
    raw = reference.strip()
    if not raw or raw.startswith("#") or raw.startswith("//"):
        return None, None
    if re.match(r"^[a-zA-Z]:[\\/]", raw):
        return None, "unsafe"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None, "unsafe"
    if parsed.scheme or parsed.netloc:
        return None, None

    decoded_path = unquote(parsed.path)
    if not decoded_path:
        return None, None
    if "\x00" in decoded_path or "\\" in decoded_path or re.match(r"^[a-zA-Z]:", decoded_path):
        return None, "unsafe"

    if decoded_path.startswith("/"):
        parts = PurePosixPath(decoded_path.lstrip("/")).parts
    else:
        parts = (
            *source_path.relative_to(job_root).parent.parts,
            *PurePosixPath(decoded_path).parts,
        )

    normalized_parts: list[str] = []
    for part in parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                return None, "unsafe"
            normalized_parts.pop()
        else:
            normalized_parts.append(part)

    candidate = job_root
    for part in normalized_parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return None, "unsafe"

    # The normalized components were constrained above; returning the lexical
    # path avoids resolving or following anything outside this job directory.
    return candidate, None


def _inspect_reference(
    job_root: pathlib.Path,
    source_path: pathlib.Path,
    reference: str,
    requiredness: str,
) -> ValidationFinding | None:
    if not reference.strip():
        if requiredness == "required":
            return ValidationFinding(
                "required_reference_missing",
                "error",
                source_path.relative_to(job_root).as_posix(),
            )
        return None
    target, problem = _resolve_local_reference(job_root, source_path, reference)
    if problem == "unsafe":
        return ValidationFinding(
            "unsafe_local_reference",
            "error",
            source_path.relative_to(job_root).as_posix(),
            reference,
        )
    elif target is not None:
        try:
            metadata = target.lstat()
        except OSError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            return ValidationFinding(
                "unsafe_local_reference",
                "error",
                source_path.relative_to(job_root).as_posix(),
            reference,
            )
        if metadata is not None and stat.S_ISREG(metadata.st_mode):
            return None
        severity = "error" if requiredness == "required" else "warning"
        code = "required_reference_missing" if severity == "error" else "optional_asset_missing"
        return ValidationFinding(
            code, severity, source_path.relative_to(job_root).as_posix(), reference
        )
    return None


def _inspect_html(
    source: str, path: str
) -> tuple[
    list[ValidationFinding],
    list[tuple[str, str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
]:
    parser = _HTMLStructureParser(path)
    try:
        parser.feed(source)
        parser.close()
        parser.finish()
    except Exception:
        parser.errors.append(ValidationFinding("html_parse_error", "error", path))
    return parser.errors, parser.references, parser.inline_styles, parser.inline_scripts


def _inspect_javascript(source: str, path: str) -> list[ValidationFinding]:
    node = shutil.which("node")
    if node is None:
        return [ValidationFinding("javascript_parser_unavailable", "error", path)]
    suffix = pathlib.PurePosixPath(path).suffix.lower()
    input_type = "commonjs" if suffix == ".cjs" else "module"
    try:
        result = subprocess.run(
            [node, f"--input-type={input_type}", "--check", "-"],
            input=source,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
            env={"PATH": os.defpath, "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return [ValidationFinding("javascript_parser_timeout", "error", path)]
    except OSError:
        return [ValidationFinding("javascript_parser_unavailable", "error", path)]
    if result.returncode:
        if result.returncode < 0:
            return [ValidationFinding("javascript_parser_failed", "error", path)]
        return [ValidationFinding("javascript_syntax_invalid", "error", path)]
    return []


def _inspect_javascript_imports(
    root: pathlib.Path, source_path: pathlib.Path, source: str
) -> list[ValidationFinding]:
    references, scan_complete = _javascript_static_imports(source)
    findings: list[ValidationFinding] = []
    relative_source = source_path.relative_to(root).as_posix()
    if not scan_complete:
        findings.append(
            ValidationFinding("javascript_import_scan_incomplete", "error", relative_source)
        )
    for reference in references:
        # Package/bare specifiers (including #package-import aliases), data,
        # and external URLs are intentionally outside local file resolution.
        if (
            reference.lower().startswith("file:")
            or re.match(r"^[A-Za-z]:[\\/]", reference)
            or reference.startswith("\\")
            or ("\\" in reference and reference.startswith((".", "/")))
        ):
            findings.append(
                ValidationFinding("unsafe_local_reference", "error", relative_source, reference)
            )
            continue
        if not reference.startswith(("./", "../", "/")) or reference.startswith("//"):
            continue
        target, problem = _resolve_local_reference(root, source_path, reference)
        if problem == "unsafe":
            findings.append(
                ValidationFinding("unsafe_local_reference", "error", relative_source, reference)
            )
            continue
        if target is None:
            continue

        def candidate_state(candidate: pathlib.Path) -> str:
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                return "unsafe"
            current = root
            for part in relative.parts:
                current = current / part
                try:
                    metadata = current.lstat()
                except FileNotFoundError:
                    return "missing"
                except OSError:
                    return "unsafe"
                if stat.S_ISLNK(metadata.st_mode):
                    return "unsafe"
                if current != candidate and not stat.S_ISDIR(metadata.st_mode):
                    return "missing"
                if current == candidate:
                    if stat.S_ISREG(metadata.st_mode):
                        return "file"
                    if stat.S_ISDIR(metadata.st_mode):
                        return "directory"
                    return "unsafe"
            return "missing"

        state = candidate_state(target)
        if state == "unsafe":
            findings.append(
                ValidationFinding("unsafe_local_reference", "error", relative_source, reference)
            )
            continue
        if target.suffix:
            candidates = [target] if state == "file" else []
            unsafe_candidate = False
        elif state == "file":
            # An exact extensionless file is accepted as the sole exact match.
            candidates = [target]
            unsafe_candidate = False
        else:
            suffixes = JS_EXTENSIONLESS_SUFFIXES
            bases = [target / "index"] if state == "directory" else [target]
            candidates = []
            unsafe_candidate = False
            for base in bases:
                for suffix in suffixes:
                    candidate = pathlib.Path(f"{base}{suffix}")
                    candidate_result = candidate_state(candidate)
                    if candidate_result == "file":
                        candidates.append(candidate)
                    elif candidate_result == "unsafe":
                        unsafe_candidate = True

        if unsafe_candidate:
            findings.append(
                ValidationFinding("unsafe_local_reference", "error", relative_source, reference)
            )
        elif len(candidates) > 1:
            findings.append(
                ValidationFinding("ambiguous_local_import", "error", relative_source, reference)
            )
        elif not candidates:
            findings.append(
                ValidationFinding("required_reference_missing", "error", relative_source, reference)
            )
    return findings


def validate_generated_output(
    job_root: pathlib.Path, relative_files: list[str]
) -> ValidationReport:
    """Check local web references and available parser-level structure/syntax."""
    errors: list[ValidationFinding] = []
    warnings: list[ValidationFinding] = []
    seen: set[tuple[str, str, str, str | None]] = set()

    def add(finding: ValidationFinding):
        identity = (finding.code, finding.severity, finding.path, finding.reference)
        if identity not in seen:
            seen.add(identity)
            (errors if finding.severity == "error" else warnings).append(finding)

    root = job_root.resolve(strict=True)
    source_files = [
        path
        for path in relative_files
        if pathlib.PurePosixPath(path).suffix.lower()
        in HTML_EXTENSIONS | CSS_EXTENSIONS | JS_SYNTAX_EXTENSIONS | {".jsx", ".ts", ".tsx"}
    ]
    for relative in source_files:
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or "\\" in relative
            or "\x00" in relative
            or re.match(r"^[a-zA-Z]:", relative)
            or any(part in {".", ".."} for part in pure.parts)
        ):
            add(ValidationFinding("unsafe_generated_path", "error", ""))
            continue
        candidate = root
        unsafe = False
        for part in pure.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                unsafe = True
                break
        if unsafe or candidate == root:
            add(ValidationFinding("unsafe_generated_path", "error", ""))
            continue
        try:
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                add(ValidationFinding("unsafe_generated_path", "error", ""))
                continue
            if metadata.st_size > MAX_VALIDATION_FILE_BYTES:
                add(ValidationFinding("source_file_too_large_to_validate", "error", relative))
                continue
            source = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            add(ValidationFinding("source_file_unreadable", "error", relative))
            continue

        suffix = candidate.suffix.lower()
        if suffix in HTML_EXTENSIONS:
            html_errors, html_refs, inline_styles, inline_scripts = _inspect_html(
                source, relative
            )
            for finding in html_errors:
                add(finding)
            for reference, requiredness, _line in html_refs:
                finding = _inspect_reference(
                    root, candidate, reference, requiredness
                )
                if finding:
                    add(finding)
            for inline_css, _line in inline_styles:
                for finding in _check_css_structure(inline_css, relative):
                    add(finding)
                for reference, requiredness in _css_references(inline_css):
                    finding = _inspect_reference(
                        root, candidate, reference, requiredness
                    )
                    if finding:
                        add(finding)
            for inline_script, _line in inline_scripts:
                for finding in _inspect_javascript(inline_script, relative):
                    add(finding)
                for finding in _inspect_javascript_imports(
                    root, candidate, inline_script
                ):
                    add(finding)
        elif suffix in CSS_EXTENSIONS:
            for finding in _check_css_structure(source, relative):
                add(finding)
            for reference, requiredness in _css_references(source):
                finding = _inspect_reference(
                    root, candidate, reference, requiredness
                )
                if finding:
                    add(finding)
        elif suffix in JS_SYNTAX_EXTENSIONS:
            for finding in _inspect_javascript(source, relative):
                add(finding)
            for finding in _inspect_javascript_imports(root, candidate, source):
                add(finding)
        else:
            add(ValidationFinding("javascript_parser_unsupported", "warning", relative))
            if suffix in JS_MODULE_EXTENSIONS:
                for finding in _inspect_javascript_imports(root, candidate, source):
                    add(finding)
    return ValidationReport(tuple(errors), tuple(warnings))
