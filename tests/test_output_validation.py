import subprocess

import pytest

from api.output_validation import validate_generated_output


def make_job(tmp_path, files):
    root = tmp_path / "job"
    root.mkdir()
    for relative, content in files.items():
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    return root


def test_valid_html_css_javascript_and_local_assets_have_no_findings(tmp_path):
    files = {
        "index.html": """<!doctype html>
<html><head><link rel="stylesheet" href="css/site.css"></head>
<body><img src="assets/logo.svg"><script src="js/app.js"></script></body></html>""",
        "css/site.css": "@import \"base.css\"; body { background: url('../assets/pattern.svg'); }",
        "css/base.css": "body { color: navy; }",
        "js/app.js": "document.body.dataset.ready = 'yes';",
        "assets/logo.svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        "assets/pattern.svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()
    assert report.warnings == ()


def test_external_data_fragment_and_scheme_urls_are_ignored(tmp_path):
    files = {
        "index.html": """<html><head>
<link rel="stylesheet" href="https://cdn.example/site.css">
<link rel="stylesheet" href="//cdn.example/other.css">
</head><body><img src="data:image/svg+xml,%3Csvg%3E">
<a href="#section">Jump</a><script src="mailto:script.js"></script>
<script type="application/json">{ invalid JavaScript }</script></body></html>""",
        "site.css": "body { background: url(https://cdn.example/bg.png); content: 'url(fake.png)'; background-image: myurl(fake.png); }",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()
    assert report.warnings == ()


def test_missing_required_script_stylesheet_and_css_import_are_blocking(tmp_path):
    files = {
        "index.html": "<html><head><link rel='stylesheet' href='missing.css'></head>"
        "<body><script src='missing.js'></script></body></html>",
        "site.css": "@import 'missing-base.css';",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert set(report.error_codes) == {"required_reference_missing"}
    assert report.warnings == ()


def test_missing_optional_images_and_fonts_are_warnings_only(tmp_path):
    files = {
        "index.html": "<html><body><img src='missing.png'></body></html>",
        "site.css": "@font-face { src: url('missing.woff2'); }",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()
    assert report.warning_codes == ("optional_asset_missing",)


def test_path_traversal_and_symlink_references_are_blocking(tmp_path):
    files = {
        "index.html": "<html><body><script src='../outside.js'></script></body></html>",
    }
    root = make_job(tmp_path, files)
    outside = tmp_path / "outside.js"
    outside.write_text("// outside job", encoding="utf-8")
    (root / "linked.js").symlink_to(outside)
    (root / "style.css").write_text(".x { background: url('linked.js'); }", encoding="utf-8")

    report = validate_generated_output(root, [*files, "style.css"])

    assert report.error_codes == ("unsafe_local_reference",)
    assert report.warnings == ()


def test_malformed_html_and_css_structure_are_blocking(tmp_path):
    files = {
        "index.html": "<html><body><main><div></main></body></html>",
        "site.css": ".card { color: red;",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert "html_mismatched_tag_nesting" in report.error_codes
    assert "css_unclosed_construct" in report.error_codes


def test_malformed_inline_javascript_is_detected_without_execution(tmp_path, monkeypatch):
    files = {
        "index.html": "<html><body><script>const value = ;</script></body></html>",
    }
    root = make_job(tmp_path, files)
    monkeypatch.setattr("api.output_validation.shutil.which", lambda _name: "/mock/node")
    monkeypatch.setattr(
        "api.output_validation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    report = validate_generated_output(root, list(files))

    assert report.error_codes == ("javascript_syntax_invalid",)


@pytest.mark.parametrize("suffix", [".js", ".mjs", ".cjs"])
def test_valid_javascript_uses_syntax_only_node_check(tmp_path, monkeypatch, suffix):
    files = {f"app{suffix}": "const value = 1;"}
    root = make_job(tmp_path, files)
    calls = []
    monkeypatch.setattr("api.output_validation.shutil.which", lambda _name: "/mock/node")

    def syntax_check(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("api.output_validation.subprocess.run", syntax_check)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()
    assert report.warnings == ()
    assert len(calls) == 1
    args, kwargs = calls[0]
    expected_mode = "commonjs" if suffix == ".cjs" else "module"
    assert args[1:] == [f"--input-type={expected_mode}", "--check", "-"]
    assert kwargs["shell"] is False
    assert kwargs["input"] == files[f"app{suffix}"]


@pytest.mark.parametrize(
    ("node_path", "run_result", "finding_code"),
    [
        (None, None, "javascript_parser_unavailable"),
        ("/mock/node", subprocess.TimeoutExpired("node", 5), "javascript_parser_timeout"),
        ("/mock/node", OSError("must not be exposed"), "javascript_parser_unavailable"),
        ("/mock/node", subprocess.CompletedProcess(["node"], -9), "javascript_parser_failed"),
    ],
)
def test_unavailable_or_failed_javascript_parser_blocks(
    tmp_path, monkeypatch, node_path, run_result, finding_code
):
    files = {"app.js": "const value = 1;"}
    root = make_job(tmp_path, files)
    monkeypatch.setattr("api.output_validation.shutil.which", lambda _name: node_path)

    def syntax_check(*_args, **_kwargs):
        if isinstance(run_result, BaseException):
            raise run_result
        return run_result

    monkeypatch.setattr("api.output_validation.subprocess.run", syntax_check)

    report = validate_generated_output(root, list(files))

    assert report.error_codes == (finding_code,)
    assert report.warnings == ()


def test_valid_relative_static_imports_and_reexports_are_resolved(tmp_path):
    files = {
        "src/app.js": (
            'import value, { item } from "./lib/module.js";\n'
            'import "./setup.js";\n'
            'export { value } from "./lib/re-export.js";\n'
            'export * from "./lib/all.js";\n'
        ),
        "src/lib/module.js": "export const value = 1; export const item = 2;",
        "src/setup.js": "globalThis.ready = true;",
        "src/lib/re-export.js": "export const value = 1;",
        "src/lib/all.js": "export const another = 2;",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()


def test_missing_relative_import_is_blocking(tmp_path):
    files = {"app.js": 'import value from "./missing.js";'}
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.error_codes == ("required_reference_missing",)
    assert report.errors[0].path == "app.js"
    assert report.errors[0].reference == "./missing.js"


def test_relative_import_traversal_and_symlink_escape_are_blocking(tmp_path):
    files = {
        "src/app.js": (
            'import outside from "../../outside.js";\n'
            'import linked from "./linked.js";\n'
            'import absolute from "C:/outside.js";\n'
            'import fileUrl from "file:///outside.js";'
        ),
        "src/linked.js": "export default 1;",
    }
    root = make_job(tmp_path, files)
    outside = tmp_path / "outside.js"
    outside.write_text("export default 2;", encoding="utf-8")
    (root / "src" / "linked.js").unlink()
    (root / "src" / "linked.js").symlink_to(outside)

    report = validate_generated_output(root, ["src/app.js"])

    assert report.error_codes == ("unsafe_local_reference",)
    assert {finding.reference for finding in report.errors} == {
        "../../outside.js",
        "./linked.js",
        "C:/outside.js",
        "file:///outside.js",
    }


def test_external_and_bare_imports_are_ignored(tmp_path):
    files = {
        "app.js": (
            'import React from "react";\n'
            'import remote from "https://cdn.example/module.js";\n'
            'import data from "data:text/javascript,export%20default%201";\n'
            'export { React } from "@scope/package";\n'
            'void import("./dynamic-missing.js");\n'
            'import local from "./local.js?variant=1#entry";'
        ),
        "local.js": "export default 1;",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()


def test_import_like_comments_strings_templates_and_regex_are_ignored(tmp_path):
    files = {
        "app.js": r'''/* import "./comment-missing.js"; */
const text = 'import "./string-missing.js"';
const template = `import "./template-missing.js" ${`inner ${1}`}`;
const pattern = /import "\.\/regex-missing\.js"/;
if (true) /import "\.\/control-regex-missing\.js"/.test(text);
import "./real.js";
''',
        "real.js": "export {};",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()


def test_extensionless_imports_use_unique_js_and_directory_index_rules(tmp_path):
    files = {
        "app.js": 'import module from "./module"; import directory from "./directory";',
        "module.js": "export default 1;",
        "directory/index.mjs": "export default 2;",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()


def test_ambiguous_extensionless_import_is_blocking(tmp_path):
    files = {
        "app.js": 'import value from "./module";',
        "module.js": "export default 1;",
        "module.mjs": "export default 2;",
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.error_codes == ("ambiguous_local_import",)


def test_circular_imports_are_checked_without_recursive_traversal(tmp_path):
    files = {
        "a.js": 'import "./b.js"; export const a = 1;',
        "b.js": 'import "./a.js"; export const b = 2;',
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.errors == ()


def test_unreadable_or_oversized_javascript_is_blocked_safely(tmp_path, monkeypatch):
    root = tmp_path / "job"
    root.mkdir()
    (root / "invalid.js").write_bytes(b"import \xff;")
    report = validate_generated_output(root, ["invalid.js"])
    assert report.error_codes == ("source_file_unreadable",)

    (root / "large.js").write_text('import "./missing.js";', encoding="utf-8")
    monkeypatch.setattr("api.output_validation.MAX_VALIDATION_FILE_BYTES", 4)
    report = validate_generated_output(root, ["large.js"])
    assert report.error_codes == ("source_file_too_large_to_validate",)


def test_escaped_module_specifier_is_decoded_before_safe_resolution(tmp_path):
    files = {
        "app.js": r'import value from "./missing\u002ejs";',
    }
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert report.error_codes == ("required_reference_missing",)
    assert report.errors[0].reference == "./missing.js"


def test_unterminated_string_blocks_syntax_and_marks_import_scan_incomplete(tmp_path):
    files = {"app.js": 'import value from "./missing.js'}
    root = make_job(tmp_path, files)

    report = validate_generated_output(root, list(files))

    assert "javascript_syntax_invalid" in report.error_codes
    assert report.error_codes == ("javascript_import_scan_incomplete", "javascript_syntax_invalid")
    assert report.warnings == ()


def test_import_scanner_token_budget_fails_closed(tmp_path, monkeypatch):
    files = {"app.js": 'import value from "./missing.js";'}
    root = make_job(tmp_path, files)
    monkeypatch.setattr("api.output_validation.MAX_JAVASCRIPT_SCAN_TOKENS", 2)

    report = validate_generated_output(root, list(files))

    assert "javascript_import_scan_incomplete" in report.error_codes
    assert "required_reference_missing" not in report.error_codes
