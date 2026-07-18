Building the Documentation
==========================

Prerequisites
-------------

Install documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Or manually:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-gallery myst-parser sphinx-autobuild

Building HTML
-------------

.. code-block:: bash

   cd docs
   make html

The generated HTML will be in ``docs/build/html/``. Open ``index.html`` in your browser.

Building PDF
------------

Requires LaTeX installation:

.. code-block:: bash

   cd docs
   make latexpdf

Live Rebuild
------------

For development, use auto-rebuild:

.. code-block:: bash

   cd docs
   make livehtml

This watches for changes and automatically rebuilds. Access at http://localhost:8000

Clean Build
-----------

Remove all build artifacts:

.. code-block:: bash

   cd docs
   make clean

Building the Example Gallery
----------------------------

Sphinx-Gallery always generates downloadable Python scripts and notebooks.
Example execution is disabled for deterministic default builds. To execute the
gallery and capture plots/output:

.. code-block:: bash

   JAXCONT_DOCS_EXECUTE_GALLERY=1 make html

Troubleshooting
---------------

**Import Errors**

Ensure jaxcont is installed:

.. code-block:: bash

   pip install -e .

**Missing Modules**

Install missing Sphinx extensions:

.. code-block:: bash

   pip install -e ".[docs]"

**Gallery Errors**

Run the relevant source example directly before rebuilding:

.. code-block:: bash

   python ../examples/example_08_vmap_sweep.py

Deployment
----------

GitHub Pages
^^^^^^^^^^^^

Configure in your repository settings to serve from ``docs/build/html/``.

Read the Docs
^^^^^^^^^^^^^

1. Connect your GitHub repository
2. RTD will automatically build on push
3. Configure ``.readthedocs.yml`` if needed

Manual Hosting
^^^^^^^^^^^^^^

Upload contents of ``docs/build/html/`` to your web server.
