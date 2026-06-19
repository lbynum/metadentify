import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from scipy import stats

from metadentify.mechanisms import (
    NeuralNetMechanism,
    RandomLinearMechanism,
)


def test_neural_net_mechanism_deterministic():
    mech = NeuralNetMechanism(hidden_dim=5)
    X = np.random.randn(10, 3)

    y1 = mech.predict(X)
    y2 = mech.predict(X)

    np.testing.assert_allclose(
        y1, y2, err_msg='NeuralNetMechanism should be deterministic across multiple calls with same input.'
    )


def test_linear_mechanism_deterministic():
    mech = RandomLinearMechanism()
    X = np.random.randn(10, 2)

    y1 = mech.predict(X)
    y2 = mech.predict(X)

    np.testing.assert_allclose(
        y1,
        y2,
        err_msg='RandomLinearMechanism should be deterministic across multiple calls with same input.',
    )


def test_random_linear_mechanism_linearity():
    np.random.seed(123)

    mech_bias = RandomLinearMechanism(bias=True)
    mech_nobias = RandomLinearMechanism(bias=False)

    for mech in [mech_bias, mech_nobias]:
        X_a = np.random.randn(10, 5)
        X_b = np.random.randn(10, 5)
        X_mid = (X_a + X_b) / 2.0

        y_a = mech.predict(X_a)
        y_b = mech.predict(X_b)
        y_mid = mech.predict(X_mid)

        y_mid_expected = (y_a + y_b) / 2.0

        np.testing.assert_allclose(
            y_mid,
            y_mid_expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg='RandomLinearMechanism failed linearity test: f((A+B)/2) != (f(A)+f(B))/2',
        )

        assert y_a.shape == (10,)
