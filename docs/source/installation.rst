Installation
============

Requirements
------------

JaxCont requires:

- Python >= 3.9
- JAX >= 0.3.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

Installing from PyPI
--------------------

The simplest way to install JaxCont is using pip:

.. code-block:: bash

   pip install jaxcont

Installing from Source
----------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/Ziaeemehr/JaxCont.git
   cd JaxCont
   pip install -e ".[dev]"

This installs JaxCont in editable mode with development dependencies.

GPU Support
-----------

To use GPU acceleration, install JAX with CUDA support:

.. code-block:: bash

   pip install --upgrade "jax[cuda12]"

See the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for current platform-specific commands.

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import jaxcont
   print(jaxcont.__version__)

Run the test suite:

.. code-block:: bash

   pytest tests/

Conda Environment (Recommended)
-------------------------------

Create a dedicated conda environment:

.. code-block:: bash

   conda create -n jaxcont python=3.9
   conda activate jaxcont
   pip install jaxcont

Optional Dependencies
---------------------

For Jupyter notebook support:

.. code-block:: bash

   pip install jupyter notebook

For building documentation:

.. code-block:: bash

   pip install -e ".[docs]"

Troubleshooting
---------------

**Import Errors**

If you encounter import errors, ensure JAX is properly installed:

.. code-block:: bash

   python -c "import jax; print(jax.__version__)"

**GPU Not Detected**

Check if JAX detects your GPU:

.. code-block:: python

   import jax
   print(jax.devices())

If no GPU is found, reinstall JAX with CUDA support.

**Version Conflicts**

If you have version conflicts, create a fresh environment:

.. code-block:: bash

   conda create -n jaxcont-clean python=3.9
   conda activate jaxcont-clean
   pip install jaxcont
