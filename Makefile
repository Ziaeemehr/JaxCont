# Makefile for JaxCont

.PHONY: help install install-dev test test-all test-cov test-gpu lint format clean docs dist examples

help:
	@echo "JaxCont Development Commands:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make test         - Run tests (CPU, parallel)"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make test-gpu     - Run GPU smoke tests (real GPU backend, serial)"
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

# JAX_PLATFORMS=cpu + `-n auto` is the intended default for the non-GPU suite,
# and the two settings are a package deal. Each xdist worker is a separate
# process, and every JAX process preallocates a large fraction of the GPU on
# first use -- so `-n auto` against the GPU backend has the workers fight over
# one device (observed on an RTX A5000: one worker took 18GB, another 5GB, the
# rest were starved at ~250MB and thrashed; the suite had not finished after 20
# minutes). Pinned to CPU it is ~40s for the same 211 tests. None of this suite
# needs the GPU -- CI is CPU-only too -- and the GPU story has its own serial
# target below.
test:
	JAX_PLATFORMS=cpu pytest tests/ -n auto          # fast: no coverage, slow tests excluded

test-all:
	JAX_PLATFORMS=cpu pytest tests/ -m "" -n auto    # everything, including slow tests

test-cov:
	JAX_PLATFORMS=cpu pytest tests/ -m "" -n auto --cov=jaxcont --cov-report=html --cov-report=term

# Deliberately serial and NOT pinned to CPU -- the point is to exercise the real
# GPU backend. Do not add `-n auto` here (see the note above).
test-gpu:
	pytest tests/ -m gpu

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
