# qiskit-data-reuploading

[![CI](https://github.com/Carlosandp/qiskit-data-reuploading/actions/workflows/ci.yml/badge.svg)](https://github.com/Carlosandp/qiskit-data-reuploading/actions)
[![PyPI](https://img.shields.io/pypi/v/qiskit-data-reuploading)](https://pypi.org/project/qiskit-data-reuploading/)
[![Coverage](https://codecov.io/gh/Carlosandp/qiskit-data-reuploading/branch/main/graph/badge.svg)](https://codecov.io/gh/Carlosandp/qiskit-data-reuploading)
[![Python](https://img.shields.io/pypi/pyversions/qiskit-data-reuploading)](https://pypi.org/project/qiskit-data-reuploading/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The first pip-installable, sklearn-compatible implementation of data re-uploading quantum classifiers for Qiskit 2.x.**

Implements the technique from Pérez-Salinas et al. (2019) as a production-quality Python library — complete with V2 primitives, benchmarking tools, and hardware integration.

> Compatible with Qiskit — not affiliated with IBM.

---

## Installation

```bash
pip install qiskit-data-reuploading
```

For IBM Quantum hardware support:
```bash
pip install "qiskit-data-reuploading[hardware]"
```

**Requirements:** Python ≥ 3.10, Qiskit ≥ 2.0, qiskit-machine-learning ≥ 0.9.0

---

## Quick Start

```python
from qdr.models import DataReuploadingClassifier

model = DataReuploadingClassifier(
    n_qubits=2,
    n_layers=5,
    encoding="rx_ry_rz",       # "rx" | "ry" | "rz" | "rx_ry_rz"
    entanglement="full",        # "none" | "linear" | "circular" | "full"
    optimizer="COBYLA",         # "COBYLA" | "SPSA" | "ADAM"
    backend=None,               # None = StatevectorEstimator (local)
    shots=None,                 # None = exact, int = noisy simulation
    max_iter=100,
)

model.fit(X_train, y_train)
preds  = model.predict(X_test)
proba  = model.predict_proba(X_test)
score  = model.score(X_test, y_test)
model.save("model.pkl")

# Reload
loaded = DataReuploadingClassifier.load("model.pkl")
```

### Direct circuit access

```python
from qdr.circuits import DataReuploadingCircuit

circuit = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=2)
circuit.build_circuit()
circuit.draw("mpl")
```

### Benchmarking

```python
from qdr.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(cv_folds=5)
runner.run(X, y, include_svm=True, include_mlp=True)
df = runner.summary()   # pandas DataFrame: accuracy, f1, train_time_s, …
```

### Visualization

```python
from qdr.visualization import plot_decision_boundary, plot_loss_curve

plot_loss_curve(model.loss_history_)
plot_decision_boundary(model, X, y)
```

---

## Existing Ecosystem Analysis

This library fills a gap that remained open as of mid-2025:

| What exists | Status |
|---|---|
| PR #668 in qiskit-community/qiskit-machine-learning | DRAFT, abandoned ~2024, never merged |
| PennyLane data-reuploading tutorial | Maintained demo, not a library |
| Academic Qiskit notebook (arxiv:2211.13191) | Didactic, Qiskit 1.x, no pip install |

**What did NOT exist before this library:**
- A pip-installable `DataReuploadingClassifier` with sklearn API
- Native data reuploading support in qiskit-machine-learning
- A dedicated feature map in `circuit.library`
- Reproducible benchmarks (DR vs MLP/SVM) on Qiskit 2.x V2 primitives

**Deprecated approaches avoided:**
- `execute()` / `Aer.get_backend()` legacy APIs (removed in Qiskit 2.x)
- V1 primitives (`StatevectorSimulator`, `algorithm_globals`)
- `BlueprintCircuit` (deprecated upstream)

This library uses **exclusively V2 primitives**: `StatevectorEstimator`, `StatevectorSampler`, and `qiskit_ibm_runtime.EstimatorV2` for hardware.

---

## Architecture

```
qdr/
├── circuits/       # DataReuploadingCircuit, ReuploadingFeatureMap
├── models/         # DataReuploadingClassifier, DataReuploadingRegressor
├── training/       # SPSA, COBYLA, ADAM, ParameterShiftGradient
├── benchmarks/     # BenchmarkRunner (vs MLP / SVM)
├── visualization/  # decision boundaries, loss curves, Bloch sphere
├── hardware/       # IBM Quantum backend integration
└── utils/          # encoding helpers
```

---

## API Reference

Full API documentation is at `docs/api/`. Key classes:

| Class | Module | Description |
|---|---|---|
| `DataReuploadingClassifier` | `qdr.models` | sklearn-compatible classifier |
| `DataReuploadingRegressor` | `qdr.models` | sklearn-compatible regressor |
| `DataReuploadingCircuit` | `qdr.circuits` | parameterized quantum circuit |
| `ReuploadingFeatureMap` | `qdr.circuits` | fixed-weight feature map |
| `ParameterShiftGradient` | `qdr.training` | exact quantum gradients |
| `SPSA` / `COBYLA` / `ADAM` | `qdr.training` | optimizer wrappers |
| `BenchmarkRunner` | `qdr.benchmarks` | vs MLP and SVM |

---

## Scientific Background

This library implements the **data re-uploading** technique:

> "A single qubit can be used as a universal quantum classifier by re-uploading
> classical data at each layer of the circuit, interleaved with trainable
> rotation gates."

**Key insight:** Unlike conventional quantum feature maps (which encode data once), data re-uploading applies encoding gates repeatedly, alternated with trainable parameters, allowing a single qubit to approximate any function.

**Reference:**

Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J.I. (2020).
*Data re-uploading for a universal quantum classifier.*
**Quantum**, 4, 226. https://doi.org/10.22331/q-2020-02-06-226

---

## Supported Python and Qiskit versions

| Python | Qiskit | qiskit-machine-learning | qiskit-aer |
|--------|--------|------------------------|------------|
| 3.10   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |
| 3.11   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |
| 3.12   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |
| 3.13   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All PRs welcome — especially:
- Additional encoding schemes
- Noise-aware training
- Hardware experiment results
- More benchmark datasets

---

## Citation

If you use this library in research, please cite:

```bibtex
@article{perez2020data,
  title={Data re-uploading for a universal quantum classifier},
  author={Pérez-Salinas, Adrián and Cervera-Lierta, Alba and Gil-Fuster, Elies and Latorre, José Ignacio},
  journal={Quantum},
  volume={4},
  pages={226},
  year={2020},
  doi={10.22331/q-2020-02-06-226}
}
```

And this software:
```bibtex
@software{andrade2026qdr,
  title={qiskit-data-reuploading},
  author={Andrade, Carlos},
  year={2026},
  url={https://github.com/Carlosandp/qiskit-data-reuploading},
  license={MIT}
}
```

---

## Disclaimer

This project is compatible with Qiskit but is **not affiliated with, endorsed by, or
maintained by IBM**.  Qiskit is a trademark of IBM.
