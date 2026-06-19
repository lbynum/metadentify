import pytest
import torch

from metadentify.modules import CausalMetaModel, build_backbone


@pytest.fixture
def shapes():
    return {
        'dim_context_features': 4,
        'dim_query_features': 2,
        'num_source_types': 3,
        'batch_size': 2,
        'num_context_points': 10,
        'num_queries': 4,
    }


@pytest.fixture
def dummy_batch(shapes):
    return {
        'x_features': torch.randn(shapes['batch_size'], shapes['num_context_points'], shapes['dim_context_features']),
        'x_sources': torch.zeros(shapes['batch_size'], shapes['num_context_points'], dtype=torch.long),
        'query_x': torch.randn(shapes['batch_size'], shapes['num_queries'], shapes['dim_query_features']),
        'query_value': torch.randn(shapes['batch_size'], shapes['num_queries']),
    }


@pytest.fixture
def backbone_config():
    return {
        'embed_dim': 16,
        'num_layers': 2,
        'num_heads': 2,
        'num_tau_samples': 2,
        'source_embed_dim': 1,
        'output_dim': 1,
        'num_inducing_points': 2,
        'dropout': 0.0,
    }


def get_trainer(backbone_type, shapes, backbone_config, tmp_path):
    backbone = build_backbone(
        config=backbone_config,
        backbone_type=backbone_type,
        predict_ate=False,
        dim_context_features=shapes['dim_context_features'],
        dim_query_features=shapes['dim_query_features'],
        num_source_types=shapes['num_source_types'],
    )
    model = CausalMetaModel(backbone=backbone)
    return model


@pytest.mark.parametrize(
    'backbone_type',
    ['q-cnp', 'q-tnp'],
)
def test_model_forward_shapes(backbone_type, shapes, backbone_config, dummy_batch, tmp_path):
    model = get_trainer(backbone_type, shapes, backbone_config, tmp_path)
    if 'quantile' in backbone_type:
        taus = torch.linspace(0.1, 0.9, 3, device=model.device).view(1, 3, 1).expand(shapes['batch_size'], -1, -1)
        res, out_taus = model(dummy_batch['x_features'], dummy_batch['x_sources'], dummy_batch['query_x'], taus=taus)
        assert res.shape == (shapes['batch_size'], shapes['num_queries'], 3)
    else:
        res = model(
            dummy_batch['x_features'],
            dummy_batch['x_sources'],
            dummy_batch['query_x'],
        )
        if 'gaussian' in backbone_type:
            assert res[0].shape == (shapes['batch_size'], shapes['num_queries'])
            assert res[1].shape == (shapes['batch_size'], shapes['num_queries'])
        elif 'point' in backbone_type:
            assert res.shape == (shapes['batch_size'], shapes['num_queries'])


@pytest.mark.parametrize(
    'backbone_type',
    ['q-cnp', 'q-tnp'],
)
def test_predict_step_output(backbone_type, shapes, backbone_config, dummy_batch, tmp_path):
    model = get_trainer(backbone_type, shapes, backbone_config, tmp_path)
    res = model.predict_step(dummy_batch, 0)

    assert 'y_pred' in res
    assert 'y_true' in res

    y_pred = res['y_pred']
    assert y_pred.shape == (shapes['batch_size'] * shapes['num_queries'], 3)


@pytest.mark.parametrize(
    'backbone_type',
    ['q-cnp', 'q-tnp'],
)
def test_ate_mode_predict_step(backbone_type, shapes, backbone_config, dummy_batch, tmp_path):
    backbone = build_backbone(
        config=backbone_config,
        backbone_type=backbone_type,
        predict_ate=True,
        dim_context_features=shapes['dim_context_features'],
        dim_query_features=shapes['dim_query_features'],
        num_source_types=shapes['num_source_types'],
    )
    model = CausalMetaModel(backbone=backbone)
    res = model.predict_step(dummy_batch, 0)

    assert res['y_pred'].shape == (shapes['batch_size'], 3)
    assert res['y_true'].shape == (shapes['batch_size'],)

