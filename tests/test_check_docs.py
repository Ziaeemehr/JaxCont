"""Docs audit tool on small synthetic pages and ledgers (gate G6 machinery)."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_docs", Path(__file__).resolve().parents[1] / "scripts" / "check_docs.py"
)
check_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_docs)


def test_rst_blocks_are_dedented_options_skipped_and_markers_read(tmp_path):
    page = tmp_path / "page.rst"
    page.write_text(
        "Intro\n\n"
        ".. code-block:: python\n"
        "   :caption: first\n\n"
        "   x = 1\n"
        "   if x:\n"
        "       y = 2\n\n"
        "Text.\n\n"
        ".. illustrative\n\n"
        ".. code-block:: python\n\n"
        "   result = undefined_name\n"
    )
    (line1, code1, ill1), (line2, code2, ill2) = check_docs.python_blocks(page)
    assert code1 == "x = 1\nif x:\n    y = 2\n" and not ill1
    assert code2 == "result = undefined_name\n" and ill2
    assert line1 < line2


def test_markdown_blocks_and_html_marker(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "```python\nimport math\n```\n\n"
        "<!-- illustrative -->\n```python\nplaceholder()\n```\n"
        "```bash\nls\n```\n"
    )
    blocks = check_docs.python_blocks(page)
    assert [(code, ill) for _, code, ill in blocks] == [("import math\n", False), ("placeholder()\n", True)]


def _ledger(tmp_path, findings, dispositions):
    text = (
        "# Loop ledger\n\n## Findings\n\n| ID | Severity | Evidence | Status |\n|---|---|---|---|\n"
        + "".join(f"| {fid} | Minor | e | {status} |\n" for fid, status in findings)
        + "\n## Changelog dispositions\n\n| Entry | Summary | Disposition | Docs | Test |\n|---|---|---|---|---|\n"
        + "".join(f"| {entry} | s | {disp} | d | t |\n" for entry, disp in dispositions)
    )
    path = tmp_path / "ledger.md"
    path.write_text(text)
    return path


def test_ledger_requires_every_finding_closed_or_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert check_docs.check_ledger(_ledger(tmp_path, [("L1", "closed"), ("L2", "Accepted (owner)")], []))
    assert not check_docs.check_ledger(_ledger(tmp_path, [("L1", "closed"), ("L2", "open")], []))
    assert not check_docs.check_ledger(_ledger(tmp_path, [("L1", "")], []))


def test_changelog_requires_a_disposition_per_entry(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [9.9.0] - Unreleased\n\n### Breaking\n\n- Removed x.\n  continued line\n\n"
        "### Fixed\n\n- Fixed y.\n- Fixed z.\n\n## [9.8.0]\n\n### Added\n\n- Old entry.\n"
    )
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    assert [entry for entry, _ in check_docs.changelog_entries("9.9.0")] == ["B01", "F01", "F02"]
    complete = _ledger(tmp_path, [("L1", "closed")], [("B01", "documented"), ("F01", "n/a"), ("F02", "test")])
    assert check_docs.check_changelog("9.9.0", complete)
    partial = _ledger(tmp_path, [("L1", "closed")], [("B01", "documented"), ("F01", "")])
    assert not check_docs.check_changelog("9.9.0", partial)
    with pytest.raises(SystemExit, match="no changelog entries"):
        check_docs.changelog_entries("1.0.0")


def test_current_changelog_has_entries_for_every_section():
    entries = check_docs.changelog_entries("0.5.0")
    assert {entry[0] for entry, _ in entries} == {"B", "A", "F", "K"}
    assert len({entry for entry, _ in entries}) == len(entries)


def test_ledger_outside_the_repository_is_reported_not_crashed(tmp_path, capsys):
    # ROOT is the real repository here; the ledger lives in a temp directory.
    assert check_docs.check_ledger(_ledger(tmp_path, [("L1", "closed")], []))
    assert str(tmp_path) in capsys.readouterr().out
