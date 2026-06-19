import dowhy.gcm as gcm
import networkx as nx
import numpy as np
import pandas as pd
import torch

from metadentify.mixture import CausalMixtureDistribution, PicklableLambdaWrapper, _worker_generate_task


def test_worker_standardization():
    def dummy_prior_fn():
        causal_graph = nx.DiGraph([('X', 'T'), ('T', 'Y'), ('X', 'Y')])
        causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)
        return causal_model

    prior_fn_wrapped = PicklableLambdaWrapper(dummy_prior_fn)

    def dummy_mixture_fn(scm):
        mixture = CausalMixtureDistribution(scm)
        np.random.seed(123)
        n = 500
        X = np.random.normal(100, 20, n)
        T = np.random.normal(-50, 5, n)
        Y = 2.0 * X + 0.5 * T + np.random.normal(10, 2, n)
        df = pd.DataFrame({'X': X, 'T': T, 'Y': Y})
        mixture.add_empirical(df, weight=1.0)
        return mixture

    mixture_fn_wrapped = PicklableLambdaWrapper(dummy_mixture_fn)

    def dummy_query_fn(scm, query_df=None):
        if query_df is not None:
            return (0.5 * query_df['T']).values
        return 0.5

    query_fn_wrapped = PicklableLambdaWrapper(dummy_query_fn)

    num_context_points = 500
    num_query_points = 20
    outcome_name = 'Y'
    treatment_name = 'T'
    observed_feature_names = ['X', 'T', 'Y']
    query_feature_names = ['X', 'T', 'Y']
    query_x_type = 'sample-from-context'

    np.random.seed(123)
    res_raw = _worker_generate_task(
        prior_fn_wrapped,
        mixture_fn_wrapped,
        query_fn_wrapped,
        num_context_points,
        num_query_points,
        outcome_name,
        treatment_name,
        observed_feature_names,
        query_feature_names,
        query_x_type,
        standardize=False,
    )

    np.random.seed(123)
    res_std = _worker_generate_task(
        prior_fn_wrapped,
        mixture_fn_wrapped,
        query_fn_wrapped,
        num_context_points,
        num_query_points,
        outcome_name,
        treatment_name,
        observed_feature_names,
        query_feature_names,
        query_x_type,
        standardize=True,
    )

    assert not res_raw['standardized'].item()
    assert res_raw['y_std'].item() == 1.0

    raw_X = res_raw['x_features'][:, 0]
    raw_T = res_raw['x_features'][:, 1]
    raw_Y = res_raw['x_features'][:, 2]
    raw_query_value = res_raw['query_value']

    assert raw_X.mean() > 90
    assert raw_T.mean() < -40
    assert raw_Y.mean() > 100

    assert res_std['standardized'].item()

    std_X = res_std['x_features'][:, 0]
    std_T = res_std['x_features'][:, 1]
    std_Y = res_std['x_features'][:, 2]
    std_query_value = res_std['query_value']

    sigma_Y = res_std['y_std'].item()

    assert np.isclose(sigma_Y, raw_Y.std(unbiased=True).item(), atol=1e-4) or np.isclose(
        sigma_Y, raw_Y.std(unbiased=False).item(), atol=1e-4
    )

    assert np.isclose(std_X.mean().item(), 0, atol=1e-5)
    assert np.isclose(std_X.std(unbiased=False).item(), 1.0, atol=1e-5)

    assert np.isclose(std_T.mean().item(), 0, atol=1e-5)
    assert np.isclose(std_T.std(unbiased=False).item(), 1.0, atol=1e-5)

    assert np.isclose(std_Y.mean().item(), 0, atol=1e-5)
    assert np.isclose(std_Y.std(unbiased=False).item(), 1.0, atol=1e-5)

    sigma_Y = res_std['y_std'].item()
    sigma_T = res_std['t_std'].item()

    assert torch.allclose(std_query_value, (raw_query_value / sigma_Y) * sigma_T, atol=1e-4)


def test_worker_standardization_binary_ignored():

    def dummy_prior_fn():
        causal_graph = nx.DiGraph([('X', 'T'), ('T', 'Y'), ('X', 'Y')])
        return gcm.InvertibleStructuralCausalModel(causal_graph)

    prior_fn_wrapped = PicklableLambdaWrapper(dummy_prior_fn)


    def dummy_query_fn(scm, query_df=None):
        if query_df is not None:
            return (0.5 * query_df['T']).values
        return 0.5

    query_fn_wrapped = PicklableLambdaWrapper(dummy_query_fn)

    num_context_points = 500
    num_query_points = 20
    outcome_name = 'Y'
    treatment_name = 'T'
    observed_feature_names = ['X', 'T', 'Y']
    query_feature_names = ['X', 'T', 'Y']
    query_x_type = 'sample-from-context'

    def dummy_mixture_fn_continuous_binary(scm):
        mixture = CausalMixtureDistribution(scm)
        np.random.seed(123)
        n = 500
        X = np.random.normal(100, 20, n)
        T = np.random.binomial(1, 0.5, 500).astype(float) 
        Y = 2.0 * X + 0.5 * T + np.random.normal(10, 2, n)
        df = pd.DataFrame({'X': X, 'T': T, 'Y': Y})
        mixture.add_empirical(df, weight=1.0)
        return mixture
    
    def dummy_mixture_fn_discrete_binary(scm):
        mixture = CausalMixtureDistribution(scm)
        np.random.seed(123)
        n = 500
        X = np.random.normal(100, 20, n)
        T = np.random.binomial(1, 0.5, 500).astype(int) 
        Y = 2.0 * X + 0.5 * T + np.random.normal(10, 2, n)
        df = pd.DataFrame({'X': X, 'T': T, 'Y': Y})
        mixture.add_empirical(df, weight=1.0)
        return mixture
    
    for dummy_mixture_fn in [dummy_mixture_fn_continuous_binary, dummy_mixture_fn_discrete_binary]:
        mixture_fn_wrapped = PicklableLambdaWrapper(dummy_mixture_fn)

        np.random.seed(123)
        res_raw = _worker_generate_task(
            prior_fn_wrapped,
            mixture_fn_wrapped,
            query_fn_wrapped,
            num_context_points,
            num_query_points,
            outcome_name,
            treatment_name,
            observed_feature_names,
            query_feature_names,
            query_x_type,
            standardize=False,
        )

        np.random.seed(123)
        res_std = _worker_generate_task(
            prior_fn_wrapped,
            mixture_fn_wrapped,
            query_fn_wrapped,
            num_context_points,
            num_query_points,
            outcome_name,
            treatment_name,
            observed_feature_names,
            query_feature_names,
            query_x_type,
            standardize=True,
        )

        std_T = res_std['x_features'][:, 1]
        raw_T = res_raw['x_features'][:, 1]

        assert torch.allclose(std_T, raw_T)
        assert torch.all((std_T == 0.0) | (std_T == 1.0))
