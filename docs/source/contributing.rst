Contributing to JaxCont
=======================

We welcome contributions to JaxCont! This guide will help you get started.

Ways to Contribute
------------------

- Report bugs and issues
- Suggest new features
- Improve documentation
- Submit bug fixes
- Add new continuation methods
- Implement new bifurcation types
- Add examples and tutorials

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/JaxCont.git
      cd JaxCont

3. Create a development environment:

   .. code-block:: bash

      conda create -n jaxcont-dev python=3.9
      conda activate jaxcont-dev
      pip install -e ".[dev]"

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

Development Workflow
--------------------

1. Make your changes
2. Add tests for new functionality
3. Run the test suite:

   .. code-block:: bash

      pytest tests/ -v

4. Check code formatting:

   .. code-block:: bash

      black src/
      isort src/
      flake8 src/

5. Update documentation if needed
6. Commit your changes with clear messages
7. Push to your fork and submit a pull request

Code Style
----------

- Follow PEP 8 guidelines
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints where appropriate
- Write docstrings for all public functions

Docstring Format
^^^^^^^^^^^^^^^^

Use NumPy-style docstrings:

.. code-block:: python

   def my_function(param1: Array, param2: float) -> Array:
       """
       Short description of function.
       
       Longer description with more details about what the function
       does and how it works.
       
       Parameters
       ----------
       param1 : Array
           Description of param1
       param2 : float
           Description of param2
       
       Returns
       -------
       Array
           Description of return value
       
       Examples
       --------
       >>> result = my_function(arr, 0.5)
       """
       pass

Testing
-------

- Write tests for all new features
- Aim for >80% code coverage
- Use pytest for testing
- Place tests in the ``tests/`` directory

Test Structure
^^^^^^^^^^^^^^

.. code-block:: python

   import pytest
   import jax.numpy as jnp
   from jaxcont import MyClass

   def test_my_feature():
       """Test description."""
       # Setup
       obj = MyClass()
       
       # Execute
       result = obj.do_something()
       
       # Assert
       assert result is not None

Documentation
-------------

- Update relevant .rst files in ``docs/source/``
- Add examples for new features
- Include mathematical descriptions where appropriate
- Build docs locally to test:

  .. code-block:: bash

     cd docs
     make html
     # Open docs/build/html/index.html

Pull Request Guidelines
-----------------------

- Provide a clear description of changes
- Reference related issues
- Include tests and documentation updates
- Ensure all tests pass
- Keep pull requests focused on a single feature/fix

Code Review Process
-------------------

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

Areas for Contribution
----------------------

Priority areas:

- Implementing additional bifurcation types
- Improving numerical stability
- Adding more examples and tutorials
- GPU acceleration optimizations
- Performance benchmarking

Community Guidelines
--------------------

- Be respectful and constructive
- Help others in discussions
- Follow the code of conduct
- Ask questions if unclear

Getting Help
------------

- Open an issue for bugs or feature requests
- Join discussions on GitHub
- Check existing issues and PRs

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank You!
----------

Your contributions make JaxCont better for everyone. We appreciate your help!
