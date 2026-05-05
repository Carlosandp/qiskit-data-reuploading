# Contributing to qiskit-data-reuploading

Thank you for considering a contribution!

## Setup

```bash
git clone https://github.com/Carlosandp/qiskit-data-reuploading.git
cd qiskit-data-reuploading
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -v --cov=qdr
```

## Code style

- PEP 8, enforced via `ruff check qdr/`
- Type hints on all public functions
- NumPy-style docstrings on all public classes

## Pull request checklist

- [ ] Tests added / updated for new functionality
- [ ] `ruff check qdr/` passes
- [ ] `mypy qdr/` passes (or suppression justified)
- [ ] Docstrings complete for any new public API
- [ ] CITATION.cff updated if new academic reference is added

## Reporting issues

Open a GitHub Issue with a minimal reproducible example and your Qiskit / Python versions.
