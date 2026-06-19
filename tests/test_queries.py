import dowhy.gcm as gcm
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from metadentify.queries import (
    make_cate_query_fn,
    make_ite_query_fn,
    make_pate_query_fn,
    make_sate_query_fn,
)


@pytest.fixture
def causal_model_and_data():
    causal_graph = nx.DiGraph([('X', 'T'), ('X', 'Y'), ('T', 'Y')])
    causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

    np.random.seed(123)
    n = 500
    X = np.random.normal(0, 1, n)
    T = 0.5 * X + np.random.normal(0, 0.1, n)
    Y = 1.5 * T + 0.8 * X + np.random.normal(0, 0.1, n)
    data = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

    gcm.auto.assign_causal_mechanisms(causal_model, data)
    gcm.fit(causal_model, data)

    return causal_model, data


@pytest.fixture
def interventions():
    treat = {'T': lambda x: 1.0}
    ctrl = {'T': lambda x: 0.0}
    return treat, ctrl


def test_make_pate_query_fn(causal_model_and_data, interventions):
    causal_model, _ = causal_model_and_data
    treat, ctrl = interventions

    ate_fn = make_pate_query_fn('Y', treat, ctrl, num_mc_samples=1000)

    result = ate_fn(causal_model)

    assert isinstance(result, float)
    assert 1.2 < result < 1.8


def test_make_cate_query_fn(causal_model_and_data, interventions):
    causal_model, _ = causal_model_and_data
    treat, ctrl = interventions

    cate_fn = make_cate_query_fn('Y', treat, ctrl, num_mc_samples=50, query_feature_names=['X'])

    query_point_dict = {'X': 1.0}
    result_dict = cate_fn(causal_model, query_point_dict)

    assert isinstance(result_dict, np.ndarray)
    assert len(result_dict) == 1

    query_point_df = pd.DataFrame({'X': [-1.0, 0.0, 1.0]})
    result_df = cate_fn(causal_model, query_point_df)

    assert isinstance(result_df, np.ndarray)
    assert len(result_df) == 3


def test_make_sate_query_fn(causal_model_and_data, interventions):
    causal_model, data = causal_model_and_data
    treat, ctrl = interventions

    sate_fn = make_sate_query_fn('Y', treat, ctrl)

    observed_sample = data.sample(50, random_state=123)
    result = sate_fn(causal_model, observed_sample)

    assert isinstance(result, float)
    assert 1.0 < result < 2.0


def test_make_ite_query_fn(causal_model_and_data, interventions):
    causal_model, data = causal_model_and_data
    treat, ctrl = interventions

    ite_fn = make_ite_query_fn('Y', treat, ctrl)

    observed_sample = data.sample(10, random_state=123)
    result = ite_fn(causal_model, observed_sample)

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 1)


def test_sate_ite_match(causal_model_and_data, interventions):
    causal_model, data = causal_model_and_data
    treat, ctrl = interventions

    sate_fn = make_sate_query_fn('Y', treat, ctrl)
    ite_fn = make_ite_query_fn('Y', treat, ctrl)

    observed_sample = data.sample(50, random_state=123)
    sate_result = sate_fn(causal_model, observed_sample)
    ite_result = ite_fn(causal_model, observed_sample)

    assert np.isclose(sate_result, ite_result.mean())
