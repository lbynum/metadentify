import networkx as nx
import numpy as np
from dowhy import gcm
from scipy import stats

from metadentify.mechanisms import (
    NeuralNetMechanism,
    RandomLinearMechanism,
    build_synthetic_causal_model,
)
from metadentify.mixture import CausalMixtureDistribution
from metadentify.queries import (
    make_cate_query_fn,
    make_ite_query_fn,
    make_pate_query_fn,
    make_sate_query_fn,
)


def resample_node_function_dict(hidden_confounder_weight, mechanism_type):
    x_noise = gcm.ScipyDistribution(stats.norm, loc=0, scale=1)
    t_noise = gcm.ScipyDistribution(stats.norm, loc=0, scale=1)
    y_noise = gcm.ScipyDistribution(stats.norm, loc=0, scale=1)

    x_tuple = (None, x_noise)

    if mechanism_type == "nn":
        t_mechanism = NeuralNetMechanism()
        y_mechanism = NeuralNetMechanism()
    elif mechanism_type == "linear":
        t_mechanism = RandomLinearMechanism()
        y_mechanism = RandomLinearMechanism()
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")

    def t_function(x):
        u_t = x[:, 0]
        x_array = x[:, 1]
        t_input = (
            hidden_confounder_weight * x_array
            + np.sqrt(1 - hidden_confounder_weight**2) * u_t
        )
        return t_mechanism.predict(t_input)

    t_tuple = (t_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=0))

    def y_function(x):
        t_array = x[:, 0]
        u_y = x[:, 1]
        x_array = x[:, 2]
        y_input = np.c_[
            t_array,
            hidden_confounder_weight * x_array
            + np.sqrt(1 - hidden_confounder_weight**2) * u_y,
        ]
        return y_mechanism.predict(y_input)

    y_tuple = (y_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=0))

    node_function_dict = {
        "x": x_tuple,
        "u_t": (None, t_noise),
        "t": t_tuple,
        "u_y": (None, y_noise),
        "y": y_tuple,
    }
    return node_function_dict


causal_graph = nx.DiGraph(
    [
        ("x", "y"),
        ("x", "t"),
        ("u_t", "t"),
        ("u_y", "y"),
        ("t", "y"),
    ]
)


def causal_model_prior_fn(hidden_confounder_weight, mechanism_type):
    node_function_dict = resample_node_function_dict(
        hidden_confounder_weight=hidden_confounder_weight, mechanism_type=mechanism_type
    )
    causal_model = build_synthetic_causal_model(
        causal_graph=causal_graph, node_function_dict=node_function_dict
    )
    return causal_model


def build_mixture(scm, config):
    mix = CausalMixtureDistribution(causal_model=scm)
    mix.add_observational(weight=1.0 - config["frac_interventional"])
    if config["frac_interventional"] > 0:
        mix.add_interventional(
            {"t": lambda x: np.random.uniform(-1, 1)},
            weight=config["frac_interventional"],
            name="Interventional",
        )
    return mix


def _build_hidden_confounder_setup(config, observed_covariates, experiment_name):
    queries = {
        "PATE": make_pate_query_fn(
            "y", {"t": lambda x: x + 1.0}, {"t": lambda x: x}, num_mc_samples=1000
        ),
        "SATE": make_sate_query_fn("y", {"t": lambda x: x + 1.0}, {"t": lambda x: x}),
        "CATE": make_cate_query_fn(
            "y",
            {"t": lambda x: x + 1.0},
            {"t": lambda x: x},
            num_mc_samples=1000,
            query_feature_names=["x"],
        ),
        "ITE": make_ite_query_fn("y", {"t": lambda x: x + 1.0}, {"t": lambda x: x}),
    }

    return {
        "experiment_name": experiment_name,
        "observed_covariates": observed_covariates,
        "outcome_name": "y",
        "treatment_name": "t",
        "causal_model_prior_fn": lambda: causal_model_prior_fn(
            hidden_confounder_weight=config["hidden_confounder_weight"],
            mechanism_type=config["mechanism_type"],
        ),
        "mixture_builder_fn": lambda scm: build_mixture(scm, config),
        "queries": queries,
        "source_names": ["Observational", "Interventional"]
        if config["frac_interventional"] > 0
        else ["Observational"],
        "empirical_max_len": None,
        "data_generation_keys": ["frac_interventional", "hidden_confounder_weight"],
    }


def add_args(parser):
    parser.add_argument("--frac_interventional", type=float, default=0.0)
    parser.add_argument("--hidden_confounder_weight", type=float, default=0.0)
    parser.add_argument("--mechanism_type", choices=["nn", "linear"], default="nn")


_EXPERIMENT_SPECS = {
    "interventional_mixture": {
        "experiment_name": "interventional_mixture",
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
        return _build_hidden_confounder_setup(
            config, observed_covariates, experiment_name
        )

    experiment_setup[key] = custom_setup
