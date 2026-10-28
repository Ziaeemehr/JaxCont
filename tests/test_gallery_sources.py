"""Structural gallery whitelist and executed Sphinx acceptance gate."""

import ast
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tokenize
import io

import pytest
from tests.test_offline_examples import ENTRIES, ROOT, offline_environment


def configuration(monkeypatch, rtd, execute):
    monkeypatch.setenv("READTHEDOCS", rtd)
    monkeypatch.setenv("JAXCONT_DOCS_EXECUTE_GALLERY", execute)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    return runpy.run_path(str(ROOT / "docs/source/conf.py"))


@pytest.mark.parametrize(
    ("rtd", "execute", "expected"),
    [("False", "0", False), ("False", "1", True), ("True", "0", True), ("True", "1", True)],
)
def test_execution_semantics_survive_root_split(monkeypatch, rtd, execute, expected):
    cfg = configuration(monkeypatch, rtd, execute)
    gallery = cfg["sphinx_gallery_conf"]
    assert gallery["plot_gallery"] is expected
    assert gallery["abort_on_example_error"] is True
    assert gallery["download_all_examples"] is True
    assert gallery["within_subsection_order"] == "FileNameSortKey"
    if rtd == "True":
        assert os.environ["JAX_PLATFORMS"] == "cpu"
    else:
        assert "JAX_PLATFORMS" not in os.environ


def test_only_examples_tutorials_and_examples_validation_gallery_plot_sources_become_pages(
    monkeypatch,
):
    conf = configuration(monkeypatch, "False", "0")["sphinx_gallery_conf"]
    assert conf["examples_dirs"] == [
        "../../examples/tutorials",
        "../../examples/validation/gallery",
    ]
    assert conf["gallery_dirs"] == ["auto_tutorials", "auto_validation"]
    assert len(ENTRIES) == 22
    for entry in ENTRIES:
        assert re.search(conf["filename_pattern"], entry.as_posix())
        assert not re.search(conf["ignore_pattern"], entry.as_posix())
        text = entry.read_text()
        assert ast.get_docstring(ast.parse(text))
        assert "# %%" in text
    for root in conf["examples_dirs"]:
        assert (ROOT / "docs/source" / root / "GALLERY_HEADER.rst").exists()


@pytest.mark.parametrize(
    "entry", sorted((ROOT / "examples/tutorials").glob("plot_*.py")), ids=lambda p: p.stem
)
def test_tutorial_executable_line_budget(entry):
    lines = {
        t.start[0]
        for t in tokenize.generate_tokens(io.StringIO(entry.read_text()).readline)
        if t.type
        not in (
            tokenize.STRING,
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        )
    }
    periodic = any(entry.name.startswith(f"plot_{i:02d}_") for i in [9, 10, 12, 13, 14, 15, 16])
    assert len(lines) <= (90 if periodic else 60)
    if len(lines) > 60:
        assert "90-line allowance" in entry.read_text()


@pytest.mark.slow
def test_sphinx_build_executes_and_lists_only_the_two_examples_roots(tmp_path):
    sentinels = [
        ROOT / "examples/validation" / sub / "plot_outside_sentinel.py"
        for sub in ["cases", "reference", "regenerate"]
    ]
    for sentinel in sentinels:
        sentinel.write_text(
            '"""Forbidden gallery source\n========================\n"""\n'
            'raise RuntimeError("OUTSIDE_ROOT_EXECUTED")\n'
        )
    try:
        output = tmp_path / "html"
        env = offline_environment(tmp_path)
        env["JAXCONT_DOCS_EXECUTE_GALLERY"] = "1"
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-E",
                "-a",
                "-W",
                "--keep-going",
                "-b",
                "html",
                "docs/source",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=2400,
        )
        (tmp_path / "sphinx.log").write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout[-5000:] + run.stderr
        for sub, source_root in [
            ("auto_tutorials", "tutorials"),
            ("auto_validation", "validation/gallery"),
        ]:
            expected = {p.name for p in (ROOT / "examples" / source_root).glob("plot_*.py")}
            generated = {p.name for p in (ROOT / "docs/source" / sub).glob("*.py")}
            assert generated == expected
            for source in expected:
                html = output / sub / (Path(source).stem + ".html")
                assert html.is_file()
                assert list((output / "_images").glob(f"sphx_glr_{Path(source).stem}_[0-9][0-9][0-9].png"))
            import zipfile
            for archive in (output / sub).glob("*.zip"):
                with zipfile.ZipFile(archive) as bundle:
                    for name in bundle.namelist():
                        assert "sentinel" not in name
                        if name.endswith(".py"):
                            assert Path(name).name in expected
        for path in output.rglob("*"):
            assert "plot_outside_sentinel" not in path.name
        for page in output.rglob("*.html"):
            assert "OUTSIDE_ROOT_EXECUTED" not in page.read_text()
        for redirect in json.loads((ROOT / "docs/source/examples-migration.json").read_text())[
            "gallery_redirects"
        ]:
            assert redirect["new_url"] in (output / redirect["old_url"]).read_text()
    finally:
        for sentinel in sentinels:
            sentinel.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "entry", [p for p in ENTRIES if "gallery" in p.parts], ids=lambda p: p.stem
)
def test_comparison_page_has_the_six_sections_and_report_table(entry):
    text = entry.read_text()
    sections = [
        "The system",
        "What to look for",
        "The JaxCont computation",
        "Comparison figures",
        "Quantitative summary",
        "Takeaway",
    ]
    positions = [text.index(section) for section in sections]
    assert positions == sorted(positions)
    assert "render_report_table(report)" in text
    assert "save_checked_figure" in text


def test_failed_report_figure_and_sidecar_share_status_and_hash(tmp_path):
    import matplotlib.pyplot as plt
    from examples.validation.reporting import ValidationReport, save_checked_figure
    from examples.validation.visualize import render_report_table

    report = ValidationReport(
        "synthetic",
        "FAIL",
        [dict(metric="difference", measured=2.0, limit=1.0, passed=False)],
        {"failed_check": False},
        None,
        "synthetic mismatch",
        "float32",
        {},
    )
    figure = render_report_table(report)
    target = tmp_path / "failed.png"
    save_checked_figure(figure, report, target)
    record = json.loads(target.with_suffix(".checks.json").read_text())
    assert record == report.record()
    assert record["status"] == "FAIL"
    assert "FAIL" in " ".join(t.get_text() for t in figure.texts)
    assert record["report_hash"][:12] in " ".join(t.get_text() for t in figure.texts)
    plt.close(figure)
