"""Execute the bootstrap cells actually distributed in gallery notebooks."""
from pathlib import Path
import sys
import tempfile

import jax
import pytest
from sphinx_gallery import py_source_parser
from sphinx_gallery.gen_rst import _clean_script_blocks

ROOT = Path(__file__).resolve().parents[1]
GALLERY = sorted((ROOT / "examples/validation/gallery").glob("plot_*.py"))
TUTORIALS = sorted((ROOT / "examples/tutorials").glob("plot_*.py"))


@pytest.mark.parametrize("entry", GALLERY + TUTORIALS, ids=lambda p: p.stem)
@pytest.mark.parametrize("mode", ["notebook", "script"])
def test_distributed_bootstrap_accepts_kernel_args_and_script_output(entry, mode, monkeypatch, tmp_path):
    _, blocks = py_source_parser.split_code_and_text_blocks(entry)
    namespace = {"__name__": "__main__"}
    if mode == "notebook":
        blocks, _ = _clean_script_blocks(
            {"remove_config_comments": True}, py_source_parser, blocks, [""] * len(blocks))
        monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "-f", "kernel.json"])
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        expected = tmp_path / "jaxcont/generated"
    else:
        namespace["__file__"] = str(entry)
        expected = tmp_path / "custom-output"
        monkeypatch.setattr(sys, "argv", [str(entry), "--output-dir", str(expected)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    initial_x64 = jax.config.x64_enabled
    try:
        bootstrap = next(block.content for block in blocks if block.type == "code")
        exec(compile(bootstrap, str(entry), "exec"), namespace)
        assert namespace["options"].output_dir == expected
        assert namespace["Path"] is Path
        if entry in GALLERY:
            assert namespace["reference_dir"].is_dir()
        if entry.stem in {"plot_lorenz84", "plot_torbpc"}:
            assert jax.config.x64_enabled
    finally:
        jax.config.update("jax_enable_x64", initial_x64)
