# Contributing to JaxCont

Thank you for your interest in contributing to JaxCont! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Ziaeemehr/JaxCont.git
cd JaxCont
```

2. Create a virtual environment and install development dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

3. Run tests to ensure everything works:
```bash
pytest tests/
```

## Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black src/`
- Use isort for import sorting: `isort src/`
- Type hints are encouraged
- Add docstrings to all public functions and classes

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Run tests with: `pytest tests/ -v --cov=jaxcont`

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes and add tests
3. Ensure code passes all tests and linting
4. Update documentation if needed
5. Submit a pull request with a clear description

## Areas for Contribution

- Implementing additional bifurcation types
- Improving numerical stability
- Adding more examples
- Enhancing documentation
- Performance optimizations
- GPU acceleration features

## Questions?

Feel free to open an issue for questions or discussions.
