# qiskit-data-reuploading

<div align="center">

[![CI](https://github.com/Carlosandp/qiskit-data-reuploading/actions/workflows/ci.yml/badge.svg)](https://github.com/Carlosandp/qiskit-data-reuploading/actions)
[![PyPI](https://img.shields.io/badge/PyPI-pre--release-orange)](https://pypi.org/project/qiskit-data-reuploading/)
[![Coverage](https://codecov.io/gh/Carlosandp/qiskit-data-reuploading/branch/main/graph/badge.svg)](https://codecov.io/gh/Carlosandp/qiskit-data-reuploading)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/qiskit-data-reuploading/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY)
[![Qiskit](https://img.shields.io/badge/compatible%20with-Qiskit%202.x-6929c4)](https://qiskit.org)

**The first pip-installable, sklearn-compatible Python library for data re-uploading quantum classifiers, built on Qiskit 2.x V2 primitives.**

*Implements the architecture from Pérez-Salinas et al. (2020) as a production-quality,
benchmarkable, and hardware-ready open-source package.*

> Compatible with Qiskit — not affiliated with, endorsed by, or maintained by IBM.

</div>

---

## Authors

This library is developed and maintained by:

| Name | Affiliation | Contact |
|---|---|---|
| **Carlos A. Durán Paredes** | Corporation for Aerospace Initiatives, Research and Innovation (CASIRI), Popayán, Colombia | caduranpd@gmail.com |
| **Javier E. León Calderón** | Department of Electronics Engineering, Universidad Nacional de Colombia, Manizales, Colombia | javleonca@unal.edu.co |
| **Nicolás Sánchez Perea** | Department of Electronics Engineering, Universidad del Cauca, Popayán, Colombia | nicolassp@unicauca.edu.co |
| **German Darío Díaz** | Department of Physics, Universidad del Cauca, Popayán, Colombia | germandiaz@unicauca.edu.co |
| **Camilo Segura** | Corporation for Aerospace Initiatives, Research and Innovation (CASIRI), Popayán, Colombia | camilosegura6@gmail.com |

---

## Table of Contents

- [What is Data Re-uploading?](#what-is-data-re-uploading)
- [Existing Ecosystem Analysis](#existing-ecosystem-analysis)
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

- A pip-installable `DataReuploadingClassifier` with sklearn-compatible API
- Native data re-uploading support in `qiskit-machine-learning`
- A dedicated feature map in Qiskit's `circuit.library`
- Reproducible benchmarks (DR vs. MLP/SVM) on Qiskit 2.x V2 primitives

**Deprecated approaches this library explicitly avoids:**

- `execute()` and `Aer.get_backend()` — removed in Qiskit 2.x
- V1 primitives (`StatevectorSimulator`, `algorithm_globals`)
- `BlueprintCircuit` — deprecated upstream in favor of constructor methods

This library uses **exclusively V2 primitives**: `StatevectorEstimator`,
`StatevectorSampler` for local simulation, and `qiskit_ibm_runtime.EstimatorV2` for
IBM Quantum hardware.

---

## Installation

**Standard:**
```bash
pip install qiskit-data-reuploading
```

**With IBM Quantum hardware support:**
```bash
pip install "qiskit-data-reuploading[hardware]"
```

**From source (latest development):**
```bash
git clone https://github.com/Carlosandp/qiskit-data-reuploading.git
cd qiskit-data-reuploading
pip install -e ".[dev]"
```

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
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DataReuploadingClassifier(
    n_qubits=2,
    n_layers=5,
    encoding="rx_ry_rz",    # "rx" | "ry" | "rz" | "rx_ry_rz"
    entanglement="full",     # "none" | "linear" | "circular" | "full"
    optimizer="COBYLA",      # "COBYLA" | "SPSA" | "ADAM"
    backend=None,            # None → StatevectorEstimator (local, exact)
    shots=None,              # None → exact; int → noisy simulation
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
print(circuit.get_parameters())
```

### Benchmarking against classical baselines

```python
from qdr.benchmarks import BenchmarkRunner
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.1)

runner = BenchmarkRunner(cv_folds=5)
runner.run(X, y, include_svm=True, include_mlp=True, include_lr=True)

df = runner.summary()
# Returns pandas DataFrame: model, accuracy, f1, precision, recall, mcc, train_time_s
print(df.to_string(index=False))
```

### Visualization

```python
from qdr.visualization import (
    plot_decision_boundary,
    plot_loss_curve,
    plot_bloch_trajectory,
    plot_parameter_landscape,
)

plot_loss_curve(model.loss_history_)
plot_decision_boundary(model, X_test, y_test)
plot_bloch_trajectory(circuit, X_train[0])
```

### IBM Quantum hardware

```python
from qdr.hardware import run_on_ibm_backend

result = run_on_ibm_backend(
    model=model,
    X=X_test,
    backend_name="ibm_brisbane",   # or "fake_manila" for noise simulation
    shots=1024,
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
| `ParameterShiftGradient` | `qdr.training` | exact quantum gradients via parameter shift |
| `SPSA` | `qdr.training` | gradient-free stochastic optimizer |
| `COBYLA` | `qdr.training` | derivative-free local optimizer |
| `ADAM` | `qdr.training` | adaptive moment estimation optimizer |
| `BenchmarkRunner` | `qdr.benchmarks` | benchmarks vs. MLP, SVM, logistic regression |
| `plot_decision_boundary` | `qdr.visualization` | 2D decision boundary plot |
| `plot_loss_curve` | `qdr.visualization` | training loss over iterations |
| `plot_bloch_trajectory` | `qdr.visualization` | qubit state on Bloch sphere per layer |
| `plot_parameter_landscape` | `qdr.visualization` | 2D cost landscape scan |
| `run_on_ibm_backend` | `qdr.hardware` | execution on IBM Quantum real hardware |

Full API documentation: [`docs/api/`](docs/api/)

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

All modes share the same API — switching is a single `backend=` argument.

---

## Benchmarking

`BenchmarkRunner` evaluates models using stratified k-fold cross-validation and
reports: accuracy, F1-score (macro), precision, recall, MCC, and training time.

Supported datasets out of the box: Iris, Moons, Circles, Wine, reduced MNIST.

Baselines: `sklearn` MLP, SVM (RBF kernel), and Logistic Regression.

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
- Encoding and trainable parameters are kept strictly separate in the circuit
- Parameter shift rules compute exact gradients without finite-difference approximation
- Barren plateau risk is discussed in `docs/barren_plateaus.md`
- Scalability is limited by current NISQ hardware; see `docs/nisq_limitations.md`

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

If you use this library in research or academic work, please cite both the original
paper and this software. If your work involves UAV anomaly detection or the QML
benchmark evaluation, also cite the associated study below.

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

**This software:**
```bibtex
@software{duranleon2026qdr,
  title     = {qiskit-data-reuploading: A pip-installable sklearn-compatible library
               for data re-uploading quantum classifiers on Qiskit 2.x},
  author    = {Carlos Andr{\'e}s Dur{\'a}n Paredes and
               Javier Esteban Le{\'o}n Calder{\'o}n and
               Nicol{\'a}s S{\'a}nchez Perea and
               German Dar{\'i}o D{\'i}az and
               Camilo Segura},
  year      = {2026},
  url       = {https://github.com/Carlosandp/qiskit-data-reuploading},
  license   = {MIT (code) / CC BY 4.0 (docs)},
  note      = {Compatible with Qiskit 2.x. Not affiliated with IBM.}
}
```

**Associated study — UAV anomaly detection benchmark:**
```bibtex
@article{duran2026qml,
  title     = {Quantum Machine Learning for Cyber-Physical Anomaly Detection in
               Unmanned Aerial Vehicles: A Leakage-Free Evaluation with
               Proxy-Audited Feature Sets},
  author    = {Dur{\'a}n Paredes, Carlos A. and
               Le{\'o}n Calder{\'o}n, Javier E. and
               S{\'a}nchez Perea, Nicol{\'a}s and
               D{\'i}az, German Dar{\'i}o and
               Segura Quintero, Camilo},
  year      = {2026},
  eprint    = {2605.19233},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi       = {10.48550/arXiv.2605.19233},
  url       = {https://arxiv.org/abs/2605.19233},
  note      = {10 pages, 7 figures, 1 table; open Qiskit 2.x implementation
               available at https://github.com/Carlosandp/qiskit-data-reuploading}
}
```

> **Preprint:** Durán Paredes et al. (2026). *Quantum Machine Learning for
> Cyber-Physical Anomaly Detection in Unmanned Aerial Vehicles: A Leakage-Free
> Evaluation with Proxy-Audited Feature Sets.* arXiv:2605.19233 [cs.CR].
> https://arxiv.org/abs/2605.19233

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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
