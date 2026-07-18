# Releasing JaxCont

The package version is defined in `src/jaxcont/_version.py`. A release version
must also match `CITATION.cff` and the changelog.

## One-time setup

1. Create protected GitHub environments named `testpypi` and `pypi`.
2. On each package index, configure a trusted publisher for this repository,
   workflow `.github/workflows/publish.yml`, and the matching environment.
3. Enable the repository in Zenodo's GitHub integration. No DOI should be put
   in the README until Zenodo has reserved or minted it.

## Release sequence

1. From a clean checkout, run:

   ```bash
   env PYTHONPATH=src python -m pytest
   make docs
   python -m build
   python -m twine check dist/*
   ```

2. Run the **Publish distribution** workflow with `testpypi`. Install the
   uploaded wheel in a fresh environment and run a smoke continuation.
3. Run the same workflow with `pypi`, approving the protected environment.
4. Tag the exact published commit (`v0.1.0`) and create a GitHub release using
   the matching changelog section.
5. Confirm Zenodo archived the GitHub release. Add the minted DOI badge and DOI
   to `CITATION.cff` in the next commit; use the concept DOI when citing the
   project across versions and the version DOI for a specific release.

PyPI and TestPyPI do not allow replacing an uploaded version. If validation
fails after upload, increment the version and rebuild rather than trying to
overwrite an artifact.
