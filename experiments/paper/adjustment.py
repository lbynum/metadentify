import networkx as nx
import numpy as np
from dowhy import gcm
from scipy import stats

from metadentify.mechanisms import NeuralNetMechanism, build_synthetic_causal_model
from metadentify.mixture import CausalMixtureDistribution
from metadentify.queries import make_pate_query_fn


def resample_node_function_dict(scm_function_class):
    if scm_function_class == "nn":
        u_tuple = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
        z_1_tuple = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
        z_2_tuple = (
            NeuralNetMechanism(),
            gcm.ScipyDistribution(stats.norm, loc=0, scale=1),
        )
        t_tuple = (
            NeuralNetMechanism(),
            gcm.ScipyDistribution(stats.norm, loc=0, scale=1),
        )
        y_tuple = (
            NeuralNetMechanism(),
            gcm.ScipyDistribution(stats.norm, loc=0, scale=1),
        )
    elif scm_function_class == "binary":
        p_u = np.random.uniform(low=0.0, high=1.0)
        u_tuple = (None, gcm.ScipyDistribution(stats.bernoulli, p=p_u))

        p_z1 = np.random.uniform(low=0.0, high=1.0)
        z_1_tuple = (None, gcm.ScipyDistribution(stats.bernoulli, p=p_z1))

        p_z2_11 = np.random.uniform(low=0.0, high=1.0)
        p_z2_10 = np.random.uniform(low=0.0, high=1.0)
        p_z2_01 = np.random.uniform(low=0.0, high=1.0)
        p_z2_00 = np.random.uniform(low=0.0, high=1.0)

        def z_2_function(x):
            # alphabetical: u, z_1
            u, z_1 = x[:, 0], x[:, 1]
            p = (
                p_z2_11 * u * z_1
                + p_z2_10 * u * (1 - z_1)
                + p_z2_01 * (1 - u) * z_1
                + p_z2_00 * (1 - u) * (1 - z_1)
            )
            z_2 = np.random.binomial(n=1, p=p).astype(int)
            return z_2

        z_2_tuple = (z_2_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=0))

        p_t_0 = np.random.uniform(low=0.0, high=1.0)
        p_t_1 = np.random.uniform(low=0.0, high=1.0)

        def t_function(x):
            z_1 = x[:, 0]
            p = p_t_1 * z_1 + p_t_0 * (1 - z_1)
            t = np.random.binomial(n=1, p=p).astype(int)
            return t

        t_tuple = (t_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=0))

        p_y_11 = np.random.uniform(low=0.0, high=1.0)
        p_y_10 = np.random.uniform(low=0.0, high=1.0)
        p_y_01 = np.random.uniform(low=0.0, high=1.0)
        p_y_00 = np.random.uniform(low=0.0, high=1.0)

        def y_function(x):
            t, u = x[:, 0], x[:, 1]
            p = (
                p_y_11 * u * t
                + p_y_10 * u * (1 - t)
                + p_y_01 * (1 - u) * t
                + p_y_00 * (1 - u) * (1 - t)
            )
            y = np.random.binomial(n=1, p=p).astype(int)
            return y

        y_tuple = (y_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=0))
    else:
        raise ValueError(f"Unknown setting: {scm_function_class}")

    node_function_dict = {
        "u": u_tuple,
        "z_1": z_1_tuple,
        "z_2": z_2_tuple,
        "t": t_tuple,
        "y": y_tuple,
    }
    return node_function_dict


causal_graph = nx.DiGraph(
    [
        ("u", "z_2"),
        ("u", "y"),
        ("z_1", "z_2"),
        ("z_1", "t"),
        ("t", "y"),
    ]
)


def causal_model_prior_fn(scm_function_class):
    node_function_dict = resample_node_function_dict(
        scm_function_class=scm_function_class
    )
    causal_model = build_synthetic_causal_model(
        causal_graph=causal_graph, node_function_dict=node_function_dict
    )
    return causal_model


def build_mixture(scm):
    mix = CausalMixtureDistribution(causal_model=scm)
    mix.add_observational(weight=1.0)
    return mix


def add_args(parser):
    parser.add_argument(
        "--scm_function_class", type=str, choices=["linear", "nn", "binary"]
    )


def _build_optimal_adjustment_setup(config, observed_covariates, experiment_name):
    queries = {
        "PATE": make_pate_query_fn(
            outcome="y",
            treatment_intervention={"t": lambda x: 1},
            control_intervention={"t": lambda x: 0},
            num_mc_samples=1000,
        ),
    }

    return {
        "experiment_name": experiment_name,
        "observed_covariates": observed_covariates,
        "outcome_name": "y",
        "treatment_name": "t",
        "causal_model_prior_fn": lambda: causal_model_prior_fn(
            scm_function_class=config["scm_function_class"]
        ),
        "mixture_builder_fn": lambda scm: build_mixture(scm),
        "queries": queries,
        "source_names": ["Observational"],
        "empirical_max_len": None,
        "data_generation_keys": ["scm_function_class"],
    }


_EXPERIMENT_SPECS = {
    "optimal_adjustment_both": {
        "experiment_name": "optimal_adjustment_both",
        "observed_covariates": ["y", "t", "z_1", "z_2"],
    },
    "optimal_adjustment_z1": {
        "experiment_name": "optimal_adjustment_z1",
        "observed_covariates": ["y", "t", "z_1"],
    },
    "optimal_adjustment_z2": {
        "experiment_name": "optimal_adjustment_z2",
        "observed_covariates": ["y", "t", "z_2"],
    },
    "optimal_adjustment_none": {
        "experiment_name": "optimal_adjustment_none",
        "observed_covariates": ["y", "t"],
    },
}

experiment_setup = {}
for key, spec in _EXPERIMENT_SPECS.items():

    def custom_setup(
        config,
        observed_covariates=spec["observed_covariates"],
        experiment_name=spec["experiment_name"],
    ):
        return _build_optimal_adjustment_setup(
            config, observed_covariates, experiment_name
        )

    experiment_setup[key] = custom_setup
