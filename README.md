# qiskit-data-reuploading

<div align="center">

[![CI](https://github.com/Carlosandp/qiskit-data-reuploading/actions/workflows/ci.yml/badge.svg)](https://github.com/Carlosandp/qiskit-data-reuploading/actions)
[![PyPI version](https://img.shields.io/pypi/v/qiskit-data-reuploading.svg)](https://pypi.org/project/qiskit-data-reuploading/)
[![PyPI downloads](https://img.shields.io/pypi/dm/qiskit-data-reuploading.svg)](https://pypi.org/project/qiskit-data-reuploading/)
[![Python](https://img.shields.io/pypi/pyversions/qiskit-data-reuploading.svg)](https://pypi.org/project/qiskit-data-reuploading/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY.txt)
[![Qiskit](https://img.shields.io/badge/compatible%20with-Qiskit%202.x%20V2%20primitives-6929c4)](https://www.ibm.com/quantum/qiskit)
[![arXiv](https://img.shields.io/badge/arXiv-2605.19233-b31b1b.svg)](https://arxiv.org/abs/2605.19233)

**A PyPI-installable, scikit-learn-compatible Python library for data
re-uploading quantum classifiers, built on Qiskit 2.x V2 primitives.**

*Implements the architecture from Pérez-Salinas et al. (2020) as a
research-grade, benchmarkable, and hardware-ready open-source package.*

> Compatible with Qiskit; not affiliated with, endorsed by, or maintained by IBM.

**PyPI:** <https://pypi.org/project/qiskit-data-reuploading/0.1.0/>

</div>

---

## Authors

This package is developed in connection with the companion UAV/QML research
effort by:

| Name | Role | Affiliation |
|---|---|---|
| **Carlos A. Durán Paredes** | Primary package author and maintainer | Corporation for Aerospace Initiatives, Research and Innovation (CASIRI), Popayán, Colombia |
| **Javier E. León Calderón** | Theory/code audit and package collaborator | Department of Electronics Engineering, Universidad Nacional de Colombia, Manizales, Colombia |
| **Nicolás Sánchez Perea** | Research co-author and experimental contributor | Department of Electronics Engineering, Universidad del Cauca, Popayán, Colombia |
| **German Darío Díaz** | Research co-author and physics contributor | Department of Physics, Universidad del Cauca, Popayán, Colombia |
| **Camilo Segura Quintero** | Research co-author and CASIRI contributor | Corporation for Aerospace Initiatives, Research and Innovation (CASIRI), Popayán, Colombia |

Package contact: <caduranpd@gmail.com>

---

## Table of Contents

- [Authors](#authors)
- [What is Data Re-uploading?](#what-is-data-re-uploading)
- [Existing Ecosystem Analysis](#existing-ecosystem-analysis)
- [Related Research Article](#related-research-article)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [Architecture](#architecture)
- [Supported Versions](#supported-versions)
- [Hardware Integration](#hardware-integration)
- [Benchmarking](#benchmarking)
- [Scientific Background](#scientific-background)
- [Licensing](#licensing)
- [Citation](#citation)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## What is Data Re-uploading?

Classical machine learning encodes data **once** before processing it. Data re-uploading
breaks this assumption: input features are injected at **every layer** of the quantum
circuit, interleaved with trainable rotation gates.

```
Layer 1:  [ Encode(x) → Train(θ) ]
Layer 2:  [ Encode(x) → Train(θ) ] ← same x, new θ
Layer N:  [ Encode(x) → Train(θ) ]
               ↓
          Measure → classify
```

A single qubit with enough layers can approximate any continuous function — a quantum
analog of the universal approximation theorem. This makes it a uniquely compact
variational model for NISQ hardware.

---

## Existing Ecosystem Analysis

This library addresses a gap that remained open in the Qiskit ecosystem as of mid-2025:

| What exists | Framework | Status |
|---|---|---|
| PR #668 "Implement Data-reuploading classifier" | qiskit-machine-learning | **DRAFT, abandoned ~2024, never merged** |
| Data-reuploading tutorial | PennyLane | Maintained demo — not a library |
| Academic Qiskit notebook (arXiv:2211.13191) | Qiskit 1.x | Didactic, no pip install, legacy APIs |

**What did not exist before this library:**

- A PyPI-installable `DataReuploadingClassifier` with sklearn-compatible API
- Native data re-uploading support in `qiskit-machine-learning`
- A dedicated feature map in Qiskit's `circuit.library`
- Reproducible benchmarks (DR vs. LogReg/SVM/RF/MLP/XGBoost) on Qiskit 2.x V2 primitives

**Deprecated approaches this library explicitly avoids:**

- `execute()` and `Aer.get_backend()` — removed in Qiskit 2.x
- V1 primitives (`StatevectorSimulator`, `algorithm_globals`)
- `BlueprintCircuit` — deprecated upstream in favor of constructor methods

This library uses **Qiskit V2 primitives**: `StatevectorEstimator` for exact
local simulation, `qiskit_aer.primitives.EstimatorV2` for finite-shot local
simulation, and `qiskit_ibm_runtime.EstimatorV2` for IBM Quantum hardware.

---

## Related Research Article

This library supports the open Qiskit 2.x implementation associated with:

> Carlos A. Durán Paredes, Javier E. León Calderón, Nicolás Sánchez Perea,
> German Darío Díaz, and Camilo Segura Quintero. **Quantum Machine Learning for
> Cyber-Physical Anomaly Detection in Unmanned Aerial Vehicles: A Leakage-Free
> Evaluation with Proxy-Audited Feature Sets.** arXiv:2605.19233, submitted
> May 19, 2026. <https://arxiv.org/abs/2605.19233>

The paper evaluates quantum machine learning for UAV cyber-physical anomaly
detection and frames this package as a reproducible Qiskit 2.x benchmark
implementation for data re-uploading workflows.

---

## Installation

**Stable release from PyPI:**
```bash
pip install qiskit-data-reuploading
```

**With IBM Quantum hardware support:**
```bash
pip install "qiskit-data-reuploading[hardware]"
```

**Editable source install (development):**
```bash
git clone https://github.com/Carlosandp/qiskit-data-reuploading.git
cd qiskit-data-reuploading
pip install -e ".[dev]"
```

The public PyPI release `0.1.0` is available at
<https://pypi.org/project/qiskit-data-reuploading/0.1.0/>.

**Requirements:** Python ≥ 3.10 · Qiskit ≥ 2.0 · qiskit-machine-learning ≥ 0.9.0 · qiskit-aer ≥ 0.15

---

## Quick Start

### Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qdr.models import DataReuploadingClassifier

X, y = load_iris(return_X_y=True)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

scaler = MinMaxScaler(feature_range=(-3.14159, 3.14159))
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

model = DataReuploadingClassifier(
    n_qubits=3,              # Multiclass: one local Z observable per class
    n_layers=5,
    encoding="rx_ry_rz",    # "rx" | "ry" | "rz" | "rx_ry_rz"
    entanglement="full",     # "none" | "linear" | "circular" | "full"
    optimizer="COBYLA",      # "COBYLA" | "SPSA" | "ADAM"
    backend=None,            # fit() requires None; use qdr.hardware for IBM backends
    shots=None,              # None = exact; int = noisy simulation
    max_iter=150,
)

model.fit(X_train, y_train)

preds  = model.predict(X_test)
proba  = model.predict_proba(X_test)
score  = model.score(X_test, y_test)

print(f"Accuracy: {score:.4f}")

model.save("iris_model.pkl")
loaded = DataReuploadingClassifier.load("iris_model.pkl")
```

### Circuit inspection

```python
from qdr.circuits import DataReuploadingCircuit

circuit = DataReuploadingCircuit(n_qubits=2, n_layers=3, n_features=4)
circuit.build_circuit()
circuit.draw("mpl")           # matplotlib figure
circuit.draw("text")          # ASCII
print(circuit.circuit.parameters)
```

### Benchmarking against classical baselines

```python
from qdr.benchmarks import BenchmarkRunner
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.1)

runner = BenchmarkRunner(cv_folds=5)
runner.run(X, y, include_logreg=True, include_svm=True, include_mlp=True, include_rf=True)

df = runner.summary()
# Returns pandas DataFrame: model, accuracy, f1, train_time_s, predict_time_s, cv_mean, cv_std
print(df.to_string(index=False))
```

### Visualization

```python
from qdr.visualization import (
    plot_decision_boundary,
    plot_loss_curve,
)

plot_loss_curve(model.loss_history_)
plot_decision_boundary(model, X_test, y_test)
```

For datasets with more than two features, `plot_decision_boundary` plots the
selected `feature_indices` and fixes all non-plotted features at their empirical
mean in `X`. This produces a well-defined 2D slice of the fitted model.

### IBM Quantum hardware

```python
from qdr.hardware import run_on_ibm_backend

parameter_values = model._circuit_.make_param_batch(model.weights_, X_test)
result = run_on_ibm_backend(
    circuit=model._circuit_.circuit,
    observable=model._observables_[0],
    parameter_values=parameter_values,
    backend_name="ibm_brisbane",
    resilience_level=1,       # 0=off; 1/2 use IBM Runtime mitigation presets
    default_shots=4096,        # or use precision=...
)
```

---

## API Overview

| Class / Function | Module | Description |
|---|---|---|
| `DataReuploadingClassifier` | `qdr.models` | sklearn-compatible multi-class classifier |
| `DataReuploadingRegressor` | `qdr.models` | sklearn-compatible regressor |
| `DataReuploadingCircuit` | `qdr.circuits` | parameterized re-uploading circuit |
| `ReuploadingFeatureMap` | `qdr.circuits` | fixed-weight feature map (no training) |
| `ParameterShiftGradient` | `qdr.training` | parameter-shift gradients for expectation values |
| `SPSA` | `qdr.training` | gradient-free stochastic optimizer |
| `COBYLA` | `qdr.training` | derivative-free local optimizer |
| `ADAM` | `qdr.training` | adaptive moment estimation optimizer |
| `BenchmarkRunner` | `qdr.benchmarks` | benchmarks vs. LogReg, SVM, Random Forest, MLP, optional XGBoost |
| `plot_decision_boundary` | `qdr.visualization` | 2D decision boundary plot |
| `plot_loss_curve` | `qdr.visualization` | training loss over iterations |
| `plot_benchmark_comparison` | `qdr.visualization` | benchmark metric comparison |
| `plot_bloch_sphere` | `qdr.visualization` | single-qubit Bloch sphere rendering |
| `run_on_ibm_backend` | `qdr.hardware` | execution on IBM Quantum real hardware |

API details are documented in the public class/function docstrings, with
executable usage examples under [`examples/`](examples/).

---

## Architecture

```
qdr/
├── circuits/
│   ├── data_reuploading.py     # DataReuploadingCircuit
│   └── feature_maps.py         # ReuploadingFeatureMap
├── models/
│   ├── classifier.py           # DataReuploadingClassifier
│   └── regressor.py            # DataReuploadingRegressor
├── training/
│   ├── gradients.py            # ParameterShiftGradient
│   └── optimizers.py           # SPSA, COBYLA, ADAM wrappers
├── benchmarks/
│   └── runner.py               # BenchmarkRunner
├── visualization/
│   └── plots.py                # all plotting utilities
├── hardware/
│   └── ibm_backend.py          # IBM Quantum integration (V2 Runtime)
└── utils/
    └── encoding.py             # RX/RY/RZ encoding helpers
```

All modules are independent. No circular imports. Each subpackage can be used in
isolation without loading the full library.

---

## Supported Versions

| Python | Qiskit | qiskit-machine-learning | qiskit-aer |
|--------|--------|------------------------|------------|
| 3.10   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |
| 3.11   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |
| 3.12   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |
| 3.13   | ≥ 2.0  | ≥ 0.9.0                | ≥ 0.15     |

---

## Hardware Integration

The library supports three execution modes:

| Mode | Backend | Noise | Use case |
|---|---|---|---|
| Exact simulation | `StatevectorEstimator` | None | Development, small circuits |
| Noisy simulation | `AerSimulator` + noise model | Configurable | Pre-hardware testing |
| Real hardware | `qiskit_ibm_runtime.EstimatorV2` | Device noise | Production experiments |

Training with `fit()` uses local estimators only. Hardware execution is exposed
through `qdr.hardware.run_on_ibm_backend()` so that authentication, transpilation,
queue time, and real-device cost are explicit.

`run_on_ibm_backend()` uses Qiskit Runtime `EstimatorV2(mode=backend)`, transpiles
the circuit to the selected backend, applies the ISA layout to the observable,
and returns one expectation value per parameter row. `default_shots` and
`precision` are mutually exclusive because Runtime shot settings override
precision targets.

Parameter-shift gradients are usually impractical on real hardware: with `P`
trainable weights, one MSE gradient call requires `2P + 1` batched circuit
evaluations per observable (`2P` shifted evaluations plus one current-prediction
evaluation for the residuals). The default classifier (`n_qubits=2`,
`n_layers=5`, `encoding="rx_ry_rz"`) has `P = 5 * 2 * 3 = 30`, so one gradient
call requires 61 executions. With `n_qubits=6`, `n_layers=5`, the model has
`P = 90`, requiring 181 executions per gradient call. For real IBM Quantum runs,
prefer SPSA or COBYLA unless you have explicitly budgeted for parameter-shift
execution.

---

## Benchmarking

`BenchmarkRunner` evaluates models using stratified k-fold cross-validation and
reports: accuracy, weighted F1-score, training time, prediction time, and
cross-validation mean/std when enabled.

The runner accepts any NumPy-compatible feature matrix and label vector; the
examples use Iris and Moons datasets from scikit-learn.

`BenchmarkRunner` does not transform features for the quantum circuit. For QDR,
pass the same feature representation intended for rotation angles, commonly a
compact range such as `[-pi, pi]`. Classical baselines that require scaling use
their own `sklearn` pipelines.

Baselines: `sklearn` Logistic Regression, SVM (RBF kernel), Random Forest, MLP,
and optional XGBoost (`include_xgboost=True`, requiring the optional `xgboost`
package).

---

## Scientific Background

This library implements the **data re-uploading** technique introduced in:

> Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J.I. (2020).
> *Data re-uploading for a universal quantum classifier.*
> **Quantum**, 4, 226.
> https://doi.org/10.22331/q-2020-02-06-226

**Key result from the paper:** A single-qubit circuit with re-uploading layers can
approximate any continuous function on a compact domain, making it a universal
classifier with minimal qubit overhead. Entanglement across multiple qubits extends
this expressibility with improved sample efficiency.

**Implementation notes:**
- In each gate, the angle is the sum of a trainable weight and one input
  feature: `angle = w + x`. This mixing inside the same gate is the defining
  feature of data re-uploading and distinguishes this architecture from
  standard VQCs where the feature map and ansatz are separate sequential blocks.
- Parameter-shift rules compute analytic gradients without finite-difference
  approximation when the estimator is exact; with finite shots, the same formula
  produces a stochastic gradient estimate.
- Scalability is limited by current NISQ hardware and by the `2P + 1` batched
  evaluations required by MSE parameter-shift gradients.
- `DataReuploadingRegressor` is single-output: it maps one `<Z_0>` expectation
  value from `[-1, 1]` into the target range observed during `fit`. Multi-output
  regression requires a separate observable design and is not implemented.

---

## Licensing

This project uses a **dual license**:

| Component | License |
|---|---|
| Source code (`qdr/`, `tests/`, `examples/`) | [MIT License](LICENSE) |
| Documentation, notebooks, tutorials (`docs/`, `notebooks/`) | [CC BY 4.0](LICENSE-CC-BY.txt) |

**What this means in practice:**
- You can use, modify, and redistribute the code freely under MIT terms.
- If you reuse or adapt the documentation or notebooks, you must credit the author.
- Academic publications using this library should include the citation below.

---

## Citation

If you use this library in research or academic work, please cite the original
data re-uploading method, the companion UAV/QML study, and this software:

**Original method:**
```bibtex
@article{perez2020data,
  title     = {Data re-uploading for a universal quantum classifier},
  author    = {P{\'{e}}rez-Salinas, Adri{\'{a}}n and Cervera-Lierta, Alba
               and Gil-Fuster, Elies and Latorre, Jos{\'{e}} Ignacio},
  journal   = {Quantum},
  volume    = {4},
  pages     = {226},
  year      = {2020},
  doi       = {10.22331/q-2020-02-06-226},
  url       = {https://doi.org/10.22331/q-2020-02-06-226}
}
```

**Companion UAV/QML study using this implementation:**
```bibtex
@misc{duranparedes2026uavqml,
  title         = {Quantum Machine Learning for Cyber-Physical Anomaly Detection
                   in Unmanned Aerial Vehicles: A Leakage-Free Evaluation with
                   Proxy-Audited Feature Sets},
  author        = {Carlos A. Dur{\'a}n Paredes and Javier E. Le{\'o}n Calder{\'o}n
                   and Nicol{\'a}s S{\'a}nchez Perea and German Dar{\'i}o D{\'i}az
                   and Camilo Segura Quintero},
  year          = {2026},
  eprint        = {2605.19233},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2605.19233},
  url           = {https://arxiv.org/abs/2605.19233}
}
```

**This software:**
```bibtex
@software{duranleon2026qdr,
  title     = {qiskit-data-reuploading: A PyPI-installable sklearn-compatible library for data re-uploading quantum classifiers on Qiskit 2.x},
  author    = {Carlos Andres Duran Paredes and Javier E. Le{\'o}n Calder{\'o}n
               and Nicol{\'a}s S{\'a}nchez Perea and German Dar{\'i}o D{\'i}az
               and Camilo Segura Quintero},
  year      = {2026},
  url       = {https://github.com/Carlosandp/qiskit-data-reuploading},
  license   = {MIT (code) / CC BY 4.0 (docs)},
  note      = {Compatible with Qiskit 2.x. Not affiliated with IBM.}
}
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This project adheres to the [Qiskit Code of Conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md)
and includes a local [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for contributors.

Priority areas:
- Additional encoding schemes (ZZ, Pauli, custom)
- Noise-aware training methods
- Hardware experiment results and calibration data
- Additional benchmark datasets
- Contributions toward a potential upstream PR to `qiskit-machine-learning`

Before opening a PR, run the test suite:
```bash
pytest tests/ --cov=qdr --cov-report=term-missing
```

---

## Disclaimer

This project is **compatible with Qiskit** but is not affiliated with, endorsed by,
or maintained by IBM. Qiskit is a registered trademark of IBM. The use of Qiskit
in this project's name and documentation is solely to indicate technical compatibility.
