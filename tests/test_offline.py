from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from dowhy import gcm
from scipy import stats

from metadentify.mixture import CausalMixtureDistribution, OfflineCausalDataGenerator, OfflineCausalDataModule, PicklableLambdaWrapper, _worker_generate_task


@pytest.fixture
def dummy_mixture_components():
    dag = nx.DiGraph([('X', 'T'), ('T', 'Y')])

    def prior_fn():
        scm = gcm.InvertibleStructuralCausalModel(dag)
        scm.set_causal_mechanism('X', gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
        scm.set_causal_mechanism(
            'T',
            gcm.AdditiveNoiseModel(
                gcm.ml.create_linear_regressor(), gcm.ScipyDistribution(stats.norm, loc=0, scale=0.1)
            ),
        )
        scm.set_causal_mechanism(
            'Y',
            gcm.AdditiveNoiseModel(
                gcm.ml.create_linear_regressor(), gcm.ScipyDistribution(stats.norm, loc=0, scale=0.1)
            ),
        )
        data = pd.DataFrame(np.random.randn(20, 3), columns=['X', 'T', 'Y'])
        gcm.fit(scm, data)
        return scm

    def mixture_fn(scm):
        mixture = CausalMixtureDistribution(scm)
        mixture.add_observational(weight=1.0)
        return mixture

    def query_fn(scm, query_df=None):
        return 1.0

    return prior_fn, mixture_fn, query_fn


@pytest.fixture
def datamodule(tmp_path):
    for split in ['train', 'val', 'test']:
        split_dir = tmp_path / split
        split_dir.mkdir()
        torch.save({'dummy': 0}, split_dir / 'dummy.pt')

    offline_datamodule = OfflineCausalDataModule(data_dir=tmp_path, batch_size=4, num_workers=4)
    offline_datamodule.setup()
    return offline_datamodule


def test_offline_generator_full_flow(tmp_path, dummy_mixture_components):
    prior_fn, mixture_fn, query_fn = dummy_mixture_components

    gen = OfflineCausalDataGenerator(
        causal_model_prior_fn=prior_fn,
        mixture_builder_fn=mixture_fn,
        causal_query_fn=query_fn,
        outcome_name='Y',
        treatment_name='T',
        observed_feature_names=['X', 'T', 'Y'],
        query_feature_names=['X', 'T', 'Y'],
        query_x_type='none',
        num_context_points=10,
        num_query_points=5,
        standardize=True,
    )

    save_dir = tmp_path / 'data'
    gen.generate_and_save(total_tasks=6, tasks_per_file=3, save_dir=str(save_dir), n_workers=1)

    chunks = sorted(list(save_dir.glob('chunk_*.pt')))
    assert len(chunks) == 2

    data = torch.load(chunks[0], weights_only=False)
    assert len(data) == 3
    assert 'x_features' in data[0]
    assert data[0]['x_features'].shape == (10, 3)
    assert data[0]['query_value'].shape == (5,)


def test_worker_generate_task_unit(dummy_mixture_components):
    prior_fn, mixture_fn, query_fn = dummy_mixture_components

    res = _worker_generate_task(
        prior_fn_wrapped=PicklableLambdaWrapper(prior_fn),
        mixture_fn_wrapped=PicklableLambdaWrapper(mixture_fn),
        query_fn_wrapped=PicklableLambdaWrapper(query_fn),
        num_context_points=10,
        num_query_points=5,
        outcome_name='Y',
        treatment_name='T',
        observed_feature_names=['X', 'T', 'Y'],
        query_feature_names=['X', 'T', 'Y'],
        query_x_type='none',
        standardize=True,
    )

    assert 'x_features' in res
    assert 'query_value' in res
    assert res['standardized'].item() is True
    assert res['query_value'].shape == (5,)

    sigma_y = res['y_std'].item()
    sigma_t = res['t_std'].item()
    assert torch.allclose(res['query_value'], torch.tensor([(1.0 / sigma_y) * sigma_t] * 5))


def test_test_dataloader_num_workers_override(datamodule):
    dl = datamodule.test_dataloader()

    if isinstance(dl, list):
        assert dl[0].num_workers == 4
    else:
        assert dl.num_workers == 4

    dl_safe = datamodule.test_dataloader(num_workers=0)
    if isinstance(dl_safe, list):
        assert dl_safe[0].num_workers == 0
    else:
        assert dl_safe.num_workers == 0


def test_generate_and_save_recycling(tmp_path):
    with patch('metadentify.mixture.multiprocessing.Pool') as MockPool:
        MockPool.return_value.__enter__.return_value = MagicMock()

        generator = OfflineCausalDataGenerator(
            causal_model_prior_fn=lambda: None,
            mixture_builder_fn=lambda _: None,
            causal_query_fn=lambda _a, _b: None,
            outcome_name='Y',
            treatment_name='T',
            observed_feature_names=['X', 'T', 'Y'],
            query_feature_names=['X', 'T'],
            query_x_type='none',
        )

        generator.generate_and_save(
            total_tasks=10,
            tasks_per_file=7,
            save_dir=str(tmp_path),
            n_workers=2,
        )

        args, kwargs = MockPool.call_args
        assert kwargs['maxtasksperchild'] == 7
