Building the Documentation
==========================

Prerequisites
-------------

Install documentation dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Or manually:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme nbsphinx myst-parser sphinx-autobuild

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

Building Notebooks
------------------

To execute notebooks during build:

.. code-block:: python

   # In conf.py
   nbsphinx_execute = 'always'

To skip notebook execution:

.. code-block:: python

   nbsphinx_execute = 'never'

Troubleshooting
---------------

**Import Errors**

Ensure jaxcont is installed:

.. code-block:: bash

   pip install -e .

**Missing Modules**

Install missing Sphinx extensions:

.. code-block:: bash

   pip install <missing-extension>

**Notebook Errors**

Check notebook execution with:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute notebook.ipynb

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
