"""qdr - Qiskit Data Reuploading library.

Implements the data re-uploading quantum classifier (Perez-Salinas et al., 2020)
with a full sklearn-compatible API, Qiskit 2.x V2 primitives, and benchmarking
tools.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("qiskit-data-reuploading")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

from qdr.circuits.data_reuploading import DataReuploadingCircuit
from qdr.circuits.feature_maps import ReuploadingFeatureMap
from qdr.models.classifier import DataReuploadingClassifier
from qdr.models.regressor import DataReuploadingRegressor

__all__ = [
    "DataReuploadingCircuit",
    "DataReuploadingClassifier",
    "DataReuploadingRegressor",
    "ReuploadingFeatureMap",
    "__version__",
]
