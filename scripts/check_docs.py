"""Docs audit checks (gate G6).

  --snippets             run every Python code block in README.md and the
                         authored docs pages, each after the page's earlier
                         blocks; a block preceded by an invisible
                         ``.. illustrative`` (RST) or ``<!-- illustrative -->``
                         (Markdown) comment, or whose first line is
                         ``# illustrative``, is listed but not run
  --list-changelog VER   print the changelog entries of VER with stable IDs
                         (B01, A01, F01, K01 for Breaking/Added/Fixed/Known
                         limitations), to seed a ledger's disposition table
  --changelog VER        require a disposition row in the ledger for every
                         changelog entry of VER (needs --ledger)
  --ledger FILE          require every finding in FILE to be closed or accepted

Ledger format (Markdown tables; extra columns are allowed):

  ## Findings
  | ID | Severity | File/line | Evidence | Fix | Status |
  ## Changelog dispositions
  | Entry | Summary | Disposition | Docs | Test |

Exit status is nonzero if any requested check fails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"
# Generated or non-authored pages are never audited as snippets.
SKIP_DIRS = {"auto_examples", "auto_tutorials", "auto_validation", "_build", "_static", "_templates"}
SKIP_FILES = {"changelog.md", "sg_execution_times.rst"}
SECTIONS = {"Breaking": "B", "Added": "A", "Fixed": "F", "Known limitations": "K"}
CLOSED = {"closed", "accepted"}


def authored_pages() -> list[Path]:
    pages = [ROOT / "README.md"]
    for path in sorted(DOCS.rglob("*")):
        if path.suffix not in (".md", ".rst") or path.name in SKIP_FILES:
            continue
        if SKIP_DIRS & set(path.relative_to(DOCS).parts):
            continue
        pages.append(path)
    return pages


_MD_BLOCK = re.compile(r"^```(?:python|py)\s*\n(.*?)^```", re.S | re.M)
_RST_BLOCK = re.compile(r"^([ \t]*)\.\. code-block:: (?:python|py)[ \t]*\n((?:\1[ \t]+:.*\n)*)[ \t]*\n((?:(?:\1[ \t]+.*)?\n)+)", re.M)


_MARKERS = ("<!-- illustrative -->", ".. illustrative")


def _marked_illustrative(text: str, start: int) -> bool:
    """True if the last non-blank line before ``start`` is an invisible marker."""
    before = [line.strip() for line in text[:start].splitlines() if line.strip()]
    return bool(before) and before[-1] in _MARKERS


def python_blocks(path: Path) -> list[tuple[int, str, bool]]:
    """``(line, code, illustrative)`` for each Python block in a Markdown or RST page."""
    text = path.read_text(encoding="utf-8")
    blocks = []
    if path.suffix == ".md":
        for match in _MD_BLOCK.finditer(text):
            code = match.group(1)
            blocks.append((text.count("\n", 0, match.start()) + 1, code,
                           _marked_illustrative(text, match.start()) or code.lstrip().startswith("# illustrative")))
    else:
        for match in _RST_BLOCK.finditer(text):
            code = textwrap.dedent(match.group(3)).strip("\n") + "\n"
            blocks.append((text.count("\n", 0, match.start()) + 1, code,
                           _marked_illustrative(text, match.start()) or code.lstrip().startswith("# illustrative")))
    return blocks


def run_snippets(timeout: float) -> bool:
    ok = True
    for page in authored_pages():
        # A page is read top to bottom, so each block runs after the page's
        # earlier runnable blocks (as a reader would have run them).
        earlier = ""
        for line, code, illustrative in python_blocks(page):
            where = f"{page.relative_to(ROOT)}:{line}"
            if illustrative:
                print(f"SKIP  {where} (illustrative)")
                continue
            with tempfile.TemporaryDirectory() as workdir:
                try:
                    completed = subprocess.run(
                        [sys.executable, "-c", earlier + code], cwd=workdir, stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=timeout,
                    )
                    failed, detail = completed.returncode != 0, completed.stderr.strip()[-400:]
                except subprocess.TimeoutExpired:
                    failed, detail = True, f"timed out after {timeout:g} s"
            print(f"{'FAIL' if failed else 'PASS'}  {where}" + (f"\n      {detail}" if failed else ""))
            ok &= not failed
            earlier += code + "\n"
    return ok


def changelog_entries(version: str) -> list[tuple[str, str]]:
    """``(ID, first line)`` for each top-level bullet of ``version``'s sections."""
    lines = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    entries, inside, prefix, count = [], False, None, {}
    for line in lines:
        if line.startswith("## "):
            inside = line[3:].lstrip("[").startswith(version)
            prefix = None
            continue
        if not inside:
            continue
        if line.startswith("### "):
            prefix = SECTIONS.get(line[4:].strip())
            continue
        if prefix and line.startswith("- "):
            count[prefix] = count.get(prefix, 0) + 1
            entries.append((f"{prefix}{count[prefix]:02d}", line[2:].strip()))
    if not entries:
        raise SystemExit(f"no changelog entries found for {version}")
    return entries


def table_rows(text: str, heading: str) -> list[list[str]]:
    """Body rows of the first Markdown table under ``## heading``."""
    section = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not section:
        return []
    rows = [line for line in section.group(1).splitlines() if line.strip().startswith("|")]
    cells = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    return [row for row in cells[2:] if any(row)]  # drop header and separator


def _display(path: Path) -> str:
    """Repository-relative path when possible; ledgers may live elsewhere."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check_ledger(path: Path) -> bool:
    rows = table_rows(path.read_text(encoding="utf-8"), "Findings")
    if not rows:
        print(f"FAIL  {_display(path)}: no Findings table")
        return False
    ok = True
    for row in rows:
        words = row[-1].lower().split()
        if not words or words[0] not in CLOSED:
            print(f"OPEN  {row[0]}: status {row[-1]!r}")
            ok = False
    print(f"{'PASS' if ok else 'FAIL'}  {len(rows)} findings in {_display(path)}")
    return ok


def check_changelog(version: str, ledger: Path) -> bool:
    disposed = {row[0] for row in table_rows(ledger.read_text(encoding="utf-8"), "Changelog dispositions")
                if len(row) >= 3 and row[2]}
    missing = [(entry, text) for entry, text in changelog_entries(version) if entry not in disposed]
    for entry, text in missing:
        print(f"MISSING  {entry}: {text[:100]}")
    print(f"{'PASS' if not missing else 'FAIL'}  changelog {version}: "
          f"{len(changelog_entries(version)) - len(missing)} of {len(changelog_entries(version))} entries disposed")
    return not missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snippets", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--list-changelog", metavar="VER")
    parser.add_argument("--changelog", metavar="VER")
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args(argv)
    if args.list_changelog:
        for entry, text in changelog_entries(args.list_changelog):
            print(f"| {entry} | {text[:90]} |  |  |  |")
        return 0
    if args.changelog and not args.ledger:
        parser.error("--changelog needs --ledger")
    if not (args.snippets or args.ledger):
        parser.error("nothing to check")
    ok = True
    if args.snippets:
        ok &= run_snippets(args.timeout)
    if args.ledger:
        ok &= check_ledger(args.ledger)
        if args.changelog:
            ok &= check_changelog(args.changelog, args.ledger)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
