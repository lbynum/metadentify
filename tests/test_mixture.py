import dowhy.gcm as gcm
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from metadentify.mixture import CausalMixtureDistribution


@pytest.fixture
def dummy_data():
    np.random.seed(123)
    X = np.random.normal(0, 1, 100)
    Y = 2.0 * X + np.random.normal(0, 0.1, 100)
    return pd.DataFrame({'X': X, 'Y': Y})


@pytest.fixture
def dummy_graph():
    return nx.DiGraph([('X', 'Y')])


@pytest.fixture
def fitted_scm(dummy_data, dummy_graph):
    causal_model = gcm.ProbabilisticCausalModel(dummy_graph)
    gcm.auto.assign_causal_mechanisms(causal_model, dummy_data)
    gcm.fit(causal_model, dummy_data)
    return causal_model


def test_mixture_sampler_empty(fitted_scm):
    sampler = CausalMixtureDistribution(fitted_scm)
    with pytest.raises(ValueError, match='No components added'):
        sampler.sample_batch(100)


def test_mixture_sampler_adds_components(fitted_scm, dummy_data):
    sampler = CausalMixtureDistribution(fitted_scm)

    sampler.add_empirical(dummy_data, weight=0.5, name='Name1')
    sampler.add_interventional({'X': lambda _: 5.0}, weight=0.3, name='Name2')

    assert len(sampler.components) == 2
    assert sampler.weights == [0.5, 0.3]
    assert sampler.names == ['Name1', 'Name2']


def test_mixture_sampler_batch_generation(fitted_scm, dummy_data):
    sampler = CausalMixtureDistribution(fitted_scm)
    sampler.add_empirical(dummy_data, weight=0.8)
    sampler.add_interventional({'X': lambda _: 1.0}, weight=0.2)

    total_samples = 500
    batch_df = sampler.sample_batch(total_samples)

    assert len(batch_df) == total_samples

    assert 'source_component' in batch_df.columns
    unique_sources = batch_df['source_component'].unique()
    assert 0 in unique_sources
    assert 1 in unique_sources

    empirical_count = (batch_df['source_component'] == 0).sum()
    assert 350 < empirical_count < 450