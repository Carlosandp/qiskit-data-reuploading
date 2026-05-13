"""Smoke tests for example scripts."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_examples_import_without_running_training():
    for path in sorted((ROOT / "examples").glob("*.py")):
        module = _load_example(path)
        assert callable(module.main)


def test_examples_avoid_pre_split_scaler_fit_transform():
    for path in sorted((ROOT / "examples").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "fit_transform(X)" not in source
        assert "if __name__ == \"__main__\":" in source
