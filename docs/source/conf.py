# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

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
# Parse every numbered example into downloadable Python/notebook forms. Gallery
# execution is opt-in because compilation timings and accelerator availability
# make it unsuitable for deterministic documentation builds.
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'ignore_pattern': (
        r'(__init__|check_installation|demo_jit_speedup|profile_continuation)\.py'
    ),
    'filename_pattern': r'/example_',
    'within_subsection_order': 'FileNameSortKey',
    'plot_gallery': os.environ.get('JAXCONT_DOCS_EXECUTE_GALLERY', '0') == '1',
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
