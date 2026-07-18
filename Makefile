# Makefile for JaxCont

.PHONY: help install install-dev test test-all test-cov lint format clean docs dist examples

help:
	@echo "JaxCont Development Commands:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make examples     - Run all examples"
	@echo "  make docs         - Build Sphinx documentation"
	@echo "  make dist         - Build and validate sdist/wheel"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/          # fast: no coverage, slow tests excluded

test-all:
	pytest tests/ -m ""    # everything, including slow tests

test-cov:
	pytest tests/ -m "" --cov=jaxcont --cov-report=html --cov-report=term

lint:
	flake8 src/jaxcont
	mypy src/jaxcont

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

examples:
	@echo "Running Example 1: Pitchfork Bifurcation"
	python examples/example_01_pitchfork.py
	@echo "\nRunning Example 2: Lorenz System"
	python examples/example_02_lorenz.py
	@echo "\nRunning Example 3: Van der Pol Oscillator"
	python examples/example_03_van_der_pol.py

docs:
	$(MAKE) -C docs html SPHINXOPTS="-W --keep-going"

dist:
	python -m build
	python -m twine check dist/*
