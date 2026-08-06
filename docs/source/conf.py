# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Read the Docs builders are CPU-only. Force hosted documentation builds onto
# the CPU backend before importing JaxCont/JAX so an installed CUDA plugin
# cannot emit a no-device initialization traceback into Sphinx-Gallery output.
# Local gallery builds retain JAX's normal device selection and can use a GPU.
if os.environ.get('READTHEDOCS') == 'True':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

sys.path.insert(0, os.path.abspath('../../src'))

# Import version from the package
from jaxcont._version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JaxCont'
copyright = '2026, Abolfazl Ziaeemehr and JaxCont Contributors'
author = 'JaxCont Contributors'
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_gallery.gen_gallery',
]

# -- Sphinx-Gallery ----------------------------------------------------------
# Parse every user-facing example and demo into downloadable Python/notebook
# forms. Read the Docs executes the gallery so the hosted documentation
# includes figures and captured output. Local execution remains opt-in because
# compilation timings make it unsuitable for every development build.
execute_gallery = (
    os.environ.get('READTHEDOCS') == 'True'
    or os.environ.get('JAXCONT_DOCS_EXECUTE_GALLERY', '0') == '1'
)

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    # examples/MatCont/{artifacts,compare,registry,run_validation}.py are the
    # validation suite's library modules, not narrated example scripts --
    # sphinx-gallery lists every .py file it finds regardless of
    # filename_pattern (that only gates execution), so without this they turn
    # into title-less gallery pages and break the `fail_on_warning` build.
    'ignore_pattern': (
        r'(__init__|check_installation|profile_continuation'
        r'|artifacts|compare|registry|run_validation)\.py'
    ),
    'filename_pattern': r'/(example_|demo_)',
    'within_subsection_order': 'FileNameSortKey',
    'plot_gallery': execute_gallery,
    'abort_on_example_error': True,
    'download_all_examples': True,
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# MyST parser settings
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
]

# Templates path
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'api/core.rst',
    'development.rst',
    'roadmap.rst',
    'tutorials/index.rst',
    'user_guide/index.rst',
    'examples/index.rst',
]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# NOTE: do not add '../../examples/images' here -- it's a runtime-generated,
# git-untracked directory (created only after the example scripts have been
# run locally). Referencing it works on a dev machine that happens to have
# run the examples, but breaks fail_on_warning builds on a fresh checkout
# (e.g. Read the Docs' `git clone --depth 1`), where it never exists.
html_static_path = ['_static']

# Theme options (matches the sibling lyapax project's docs style)
html_theme_options = {
    'repository_url': 'https://github.com/Ziaeemehr/JaxCont',
    'use_repository_button': True,
    'show_toc_level': 2,
    'navigation_with_keys': True,
}
html_context = {
    'default_mode': 'light',
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/jaxcont_logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = '_static/favicon.ico'

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'JaxContdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
    ''',
}

# The docs and docstrings use Unicode math/Greek symbols and box-drawing
# characters freely (theta, tau, alpha, arrows, box-drawing banners, etc.).
# The classical pdfLaTeX engine needs each such glyph registered by hand via
# inputenc, which is an unbounded whack-a-mole; XeLaTeX renders Unicode
# natively through the system font stack instead.
latex_engine = 'xelatex'
# Sphinx defaults to `xindy` for the index when using xelatex/lualatex, but
# xindy's LATIN9 language module isn't available in every TeX distribution
# (including the one this was tested against) -- fall back to the classic,
# more portable `makeindex`.
latex_use_xindy = False

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'JaxCont.tex', 'JaxCont Documentation',
     'JaxCont Contributors', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'jaxcont', 'JaxCont Documentation',
     [author], 1)
]

# -- Options for EPUB output --------------------------------------------------

# sphinx.ext.githubpages writes a `.nojekyll` marker for the HTML/GitHub
# Pages target; the EPUB packager doesn't recognize its mimetype and warns
# (fatal under fail_on_warning) unless it's excluded here.
epub_exclude_files = ['.nojekyll']

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'JaxCont', 'JaxCont Documentation',
     author, 'JaxCont', 'High-performance continuation and bifurcation analysis in JAX.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Technical presentation build ---------------------------------------------
# The Beamer technical presentation lives under notes/technical_presentation/
# and its PDF is repo-wide `.gitignore`d like every other generated artifact.
# Build a fresh copy into _static/ whenever the docs build so presentation.md
# can link to and embed it without ever committing a binary to Git. Missing
# LaTeX tooling degrades to a skipped page rather than a failed docs build,
# since the rest of the documentation does not depend on it.
import shutil
import subprocess

from sphinx.util import logging as _sphinx_logging

_presentation_logger = _sphinx_logging.getLogger(__name__)


def _build_technical_presentation(app):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    presentation_dir = os.path.join(repo_root, 'notes', 'technical_presentation')
    pdf_name = 'jaxcont_technical_presentation.pdf'
    built_pdf = os.path.join(presentation_dir, pdf_name)
    static_dir = os.path.join(app.srcdir, '_static')
    dest_pdf = os.path.join(static_dir, pdf_name)

    if shutil.which('latexmk') is None or shutil.which('make') is None:
        _presentation_logger.info(
            'latexmk/make not found -- skipping technical presentation PDF build; '
            'presentation.md will link to a missing file.'
        )
        return

    result = subprocess.run(
        ['make', '-C', presentation_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _presentation_logger.info(
            'Technical presentation PDF build failed (exit %d); the download '
            'link on presentation.md may be stale or missing.\n%s',
            result.returncode,
            (result.stdout or '') + (result.stderr or ''),
        )

    if os.path.exists(built_pdf):
        os.makedirs(static_dir, exist_ok=True)
        shutil.copy2(built_pdf, dest_pdf)
    else:
        _presentation_logger.info(
            'Technical presentation PDF not found at %s after build attempt', built_pdf
        )


def setup(app):
    app.connect('builder-inited', _build_technical_presentation)
