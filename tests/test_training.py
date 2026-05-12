"""Tests for training utilities: optimizers and parameter-shift gradients."""

import numpy as np
import pytest

from qdr.training.optimizers import ADAM, COBYLA, SPSA, OptimizeResult, get_optimizer


class TestSPSA:
    def test_invalid_hyperparameters_raise(self):
        with pytest.raises(ValueError, match="maxiter must be >= 1"):
            SPSA(maxiter=0)
        with pytest.raises(ValueError, match="c must be > 0"):
            SPSA(c=0.0)

    def test_minimize_reduces_loss(self):
        # Use a simple sphere function where SPSA reliably converges
        def sphere(x):
            return float(np.sum(x ** 2))

        x0 = np.array([3.0, -3.0])
        spsa = SPSA(maxiter=300, a=0.3, c=0.05, A=5, seed=0)
        result = spsa.minimize(sphere, x0)
        assert isinstance(result, OptimizeResult)
        assert result.fun < sphere(x0)  # must improve on the starting point

    def test_quadratic_minimum(self):
        """SPSA should approach x=0 for f(x)=||x||^2."""
        spsa = SPSA(maxiter=500, a=0.3, c=0.05, A=5, seed=42)
        x0 = np.array([2.0, -2.0])
        result = spsa.minimize(lambda x: float(np.sum(x**2)), x0)
        assert result.fun < 2.0  # should improve substantially

    def test_loss_history_length(self):
        spsa = SPSA(maxiter=20, seed=0)
        result = spsa.minimize(lambda x: float(np.sum(x**2)), np.ones(3))
        assert len(result.loss_history) == 20

    def test_no_extra_final_objective_evaluation(self):
        calls = 0

        def sphere(x):
            nonlocal calls
            calls += 1
            return float(np.sum(x**2))

        spsa = SPSA(maxiter=4, seed=0)
        result = spsa.minimize(sphere, np.ones(2))

        assert calls == 12
        assert result.nfev == 12
        assert result.fun == result.loss_history[-1]

    def test_callback_called(self):
        calls = []
        spsa = SPSA(maxiter=5, seed=0)
        spsa.minimize(lambda x: float(np.sum(x**2)), np.ones(2), callback=lambda w: calls.append(1))
        assert len(calls) == 5


class TestCOBYLA:
    def test_invalid_hyperparameters_raise(self):
        with pytest.raises(ValueError, match="maxiter must be >= 1"):
            COBYLA(maxiter=0)
        with pytest.raises(ValueError, match="rhobeg must be > 0"):
            COBYLA(rhobeg=0.0)

    def test_minimize_sphere(self):
        cobyla = COBYLA(maxiter=200)
        result = cobyla.minimize(lambda x: float(np.sum(x**2)), np.array([1.0, 1.0]))
        assert result.fun < 0.1

    def test_returns_optimize_result(self):
        cobyla = COBYLA(maxiter=10)
        result = cobyla.minimize(lambda x: float(np.sum(x**2)), np.ones(2))
        assert isinstance(result, OptimizeResult)
        assert result.x.shape == (2,)

    def test_loss_history_non_empty(self):
        cobyla = COBYLA(maxiter=20)
        result = cobyla.minimize(lambda x: float(np.sum(x**2)), np.ones(2))
        assert len(result.loss_history) > 0
        assert result.nfev == len(result.loss_history)


class TestADAM:
    def test_invalid_hyperparameters_raise(self):
        with pytest.raises(ValueError, match="lr must be > 0"):
            ADAM(lr=0.0)
        with pytest.raises(ValueError, match="beta1 must satisfy"):
            ADAM(beta1=1.0)
        with pytest.raises(ValueError, match="eps must be > 0"):
            ADAM(eps=0.0)

    def test_minimize_sphere(self):
        x0 = np.array([2.0, -2.0])
        adam = ADAM(maxiter=500, lr=0.1)
        result = adam.minimize(
            lambda x: float(np.sum(x**2)),
            x0,
            gradient_fn=lambda x: 2 * x,
        )
        assert result.fun < 0.1

    def test_loss_decreases(self):
        x0 = np.ones(3) * 3.0
        adam = ADAM(maxiter=100, lr=0.05)
        result = adam.minimize(
            lambda x: float(np.sum(x**2)),
            x0,
            gradient_fn=lambda x: 2 * x,
        )
        assert result.loss_history[-1] < result.loss_history[0]

    def test_callback(self):
        calls = []
        adam = ADAM(maxiter=5, lr=0.01)
        adam.minimize(
            lambda x: float(np.sum(x**2)),
            np.ones(2),
            gradient_fn=lambda x: 2 * x,
            callback=lambda w: calls.append(1),
        )
        assert len(calls) == 5

    def test_no_extra_final_objective_evaluation(self):
        fun_calls = 0
        grad_calls = 0

        def sphere(x):
            nonlocal fun_calls
            fun_calls += 1
            return float(np.sum(x**2))

        def grad(x):
            nonlocal grad_calls
            grad_calls += 1
            return 2 * x

        adam = ADAM(maxiter=3, lr=0.05)
        result = adam.minimize(sphere, np.ones(2), gradient_fn=grad)

        assert fun_calls == 3
        assert grad_calls == 3
        assert result.nfev == 3
        assert result.njev == 3
        assert result.fun == result.loss_history[-1]

    def test_rejects_bad_gradient_shape(self):
        adam = ADAM(maxiter=1)
        with pytest.raises(ValueError, match="gradient_fn returned shape"):
            adam.minimize(
                lambda x: float(np.sum(x**2)),
                np.ones(2),
                gradient_fn=lambda x: np.ones(3),
            )

    def test_callback_cannot_mutate_optimizer_state(self):
        def mutate(w):
            w[:] = 999.0

        adam = ADAM(maxiter=1, lr=0.1)
        result = adam.minimize(
            lambda x: float(np.sum(x**2)),
            np.array([1.0]),
            gradient_fn=lambda x: 2 * x,
            callback=mutate,
        )
        assert result.x[0] == pytest.approx(0.9)


class TestGetOptimizer:
    def test_get_spsa(self):
        opt = get_optimizer("SPSA", maxiter=10)
        assert isinstance(opt, SPSA)
        assert opt.maxiter == 10

    def test_get_cobyla(self):
        opt = get_optimizer("COBYLA", maxiter=50)
        assert isinstance(opt, COBYLA)

    def test_get_adam(self):
        opt = get_optimizer("ADAM", lr=0.05)
        assert isinstance(opt, ADAM)
        assert opt.lr == 0.05

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="optimizer"):
            get_optimizer("SGD")


class TestParameterShiftGradient:
    def test_invalid_shift_raises(self):
        from qiskit.quantum_info import SparsePauliOp

        from qdr.circuits import DataReuploadingCircuit
        from qdr.training.gradients import ParameterShiftGradient

        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=1, encoding="ry")
        drc.build_circuit()
        with pytest.raises(ValueError, match="sin\\(shift\\) is zero"):
            ParameterShiftGradient(drc, SparsePauliOp("Z"), shift=np.pi)

    def test_gradient_shape(self):
        from qiskit.quantum_info import SparsePauliOp

        from qdr.circuits import DataReuploadingCircuit
        from qdr.training.gradients import ParameterShiftGradient

        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=1, encoding="ry")
        drc.build_circuit()
        obs = SparsePauliOp("Z")
        psr = ParameterShiftGradient(drc, obs)

        rng = np.random.default_rng(0)
        weights = rng.uniform(-np.pi, np.pi, drc.n_weights)
        X = rng.uniform(-1, 1, (5, 1))
        y = np.ones(5)

        grad = psr.compute(weights, X, y)
        assert grad.shape == (drc.n_weights,)

    def test_gradient_finite_difference_consistency(self):
        """Check parameter-shift vs numerical finite difference on a simple circuit."""
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator

        from qdr.circuits import DataReuploadingCircuit
        from qdr.training.gradients import ParameterShiftGradient

        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=1, encoding="ry")
        drc.build_circuit()
        obs = SparsePauliOp("Z")
        estimator = StatevectorEstimator()
        psr = ParameterShiftGradient(drc, obs, estimator=estimator)

        rng = np.random.default_rng(7)
        weights = rng.uniform(-np.pi, np.pi, drc.n_weights)
        X = np.array([[0.5]])
        y = np.array([1.0])

        grad_psr = psr.compute(weights, X, y)

        # Numerical FD for comparison
        eps = 1e-5
        grad_fd = np.zeros_like(weights)
        for i in range(len(weights)):
            wp = weights.copy()
            wp[i] += eps
            wm = weights.copy()
            wm[i] -= eps
            evp = psr._eval(wp, X)[0]
            evm = psr._eval(wm, X)[0]
            # dL/dw_i = 2*(ev - y) * d(ev)/dw_i
            grad_fd[i] = 2 * (psr._eval(weights, X)[0] - y[0]) * (evp - evm) / (2 * eps)

        np.testing.assert_allclose(grad_psr, grad_fd, atol=1e-4)

    def test_non_default_shift_matches_finite_difference(self):
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp

        from qdr.circuits import DataReuploadingCircuit
        from qdr.training.gradients import ParameterShiftGradient

        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=1, encoding="ry")
        drc.build_circuit()
        obs = SparsePauliOp("Z")
        psr = ParameterShiftGradient(drc, obs, estimator=StatevectorEstimator(), shift=np.pi / 3)

        weights = np.array([0.4])
        X = np.array([[0.2]])
        y = np.array([0.1])
        grad_psr = psr.compute(weights, X, y)

        eps = 1e-6
        ev0 = psr._eval(weights, X)[0]
        evp = psr._eval(weights + eps, X)[0]
        evm = psr._eval(weights - eps, X)[0]
        grad_fd = np.array([2 * (ev0 - y[0]) * (evp - evm) / (2 * eps)])
        np.testing.assert_allclose(grad_psr, grad_fd, atol=1e-5)

    def test_compute_rejects_y_shape_mismatch(self):
        from qiskit.quantum_info import SparsePauliOp

        from qdr.circuits import DataReuploadingCircuit
        from qdr.training.gradients import ParameterShiftGradient

        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=1, encoding="ry")
        drc.build_circuit()
        psr = ParameterShiftGradient(drc, SparsePauliOp("Z"))
        with pytest.raises(ValueError, match="y must have shape"):
            psr.compute(np.zeros(drc.n_weights), np.zeros((2, 1)), np.zeros(1))

    def test_compute_rejects_non_finite_y(self):
        from qiskit.quantum_info import SparsePauliOp

        from qdr.circuits import DataReuploadingCircuit
        from qdr.training.gradients import ParameterShiftGradient

        drc = DataReuploadingCircuit(n_qubits=1, n_layers=1, n_features=1, encoding="ry")
        drc.build_circuit()
        psr = ParameterShiftGradient(drc, SparsePauliOp("Z"))
        with pytest.raises(ValueError, match="y contains NaN or Inf"):
            psr.compute(np.zeros(drc.n_weights), np.zeros((2, 1)), np.array([0.0, np.nan]))
