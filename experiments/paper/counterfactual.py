import networkx as nx
import numpy as np
from dowhy import gcm
from scipy import stats

from metadentify.mechanisms import build_synthetic_causal_model
from metadentify.mixture import CausalMixtureDistribution
from metadentify.queries import make_cate_query_fn, make_ite_query_fn


def resample_node_function_dict(cf_noise_std):
    x_tuple = (None, gcm.ScipyDistribution(stats.uniform, loc=-2, scale=4))
    u_tuple = (None, gcm.ScipyDistribution(stats.uniform, loc=-2, scale=4))
    t_tuple = (None, gcm.ScipyDistribution(stats.bernoulli, p=0.5))

    coef_x = np.random.uniform(-3, 3, size=1)
    coef_t = np.random.uniform(-3, 3, size=1)
    coef_xt = np.random.uniform(-1, 1, size=1)
    coef_ut = np.random.uniform(-3, 3, size=1)

    def y_function(x_array):
        t = x_array[:, 0]
        u = x_array[:, 1]
        x = x_array[:, 2]
        y = coef_x * x + coef_t * t + coef_xt * x * t + coef_ut * u * t
        return y

    y_tuple = (y_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=cf_noise_std))

    node_function_dict = {
        "x": x_tuple,
        "u": u_tuple,
        "t": t_tuple,
        "y": y_tuple,
    }
    return node_function_dict


causal_graph = nx.DiGraph(
    [
        ("x", "y"),
        ("t", "y"),
        ("u", "y"),
    ]
)


def causal_model_prior_fn(cf_noise_std):
    node_function_dict = resample_node_function_dict(cf_noise_std=cf_noise_std)
    causal_model = build_synthetic_causal_model(
        causal_graph=causal_graph, node_function_dict=node_function_dict
    )
    return causal_model


def build_mixture(scm, config):
    empirical_data = gcm.draw_samples(scm, num_samples=config["dataset_size"])
    mix = CausalMixtureDistribution(causal_model=scm)

    cf_intervention = {"t": lambda x: 1 - x}
    paired_counterfactual_weight = config["frac_paired_counterfactual"]
    paired_interventional_weight = config["frac_paired_interventional"]
    if paired_counterfactual_weight > 0 and paired_interventional_weight > 0:
        total_weight = paired_counterfactual_weight + paired_interventional_weight
        paired_counterfactual_weight /= total_weight
        paired_interventional_weight /= total_weight
        paired_counterfactual_weight *= 2 / 3
        paired_interventional_weight *= 2 / 3
        empirical_weight = 1 / 3
    else:
        empirical_weight = (
            1 - paired_counterfactual_weight - paired_interventional_weight
        )

    if paired_counterfactual_weight > 0:
        mix.add_paired_counterfactual(
            evidence_data=empirical_data,
            hypothetical_intervention=cf_intervention,
            weight=paired_counterfactual_weight,
        )
    if paired_interventional_weight > 0:
        mix.add_paired_interventional(
            evidence_data=empirical_data,
            intervention=cf_intervention,
            weight=paired_interventional_weight,
            observed_variable_names=["x", "t", "y"],
        )

    mix.add_empirical(data=empirical_data, weight=empirical_weight)

    if config["frac_counterfactual"] > 0:
        mix.add_counterfactual(
            evidence_data=empirical_data,
            hypothetical_intervention=cf_intervention,
            weight=empirical_weight * config["frac_counterfactual"],
        )

    if config["frac_interventional"] > 0:
        mix.add_interventional(
            intervention=cf_intervention,
            weight=empirical_weight * config["frac_interventional"],
        )
    return mix


def _build_cf_int_setup(config, observed_covariates, experiment_name):
    queries = {
        "CATE": make_cate_query_fn(
            "y",
            {"t": lambda x: 1},
            {"t": lambda x: 0},
            num_mc_samples=1000,
            query_feature_names=["x"],
        ),
        "ITE": make_ite_query_fn("y", {"t": lambda x: 1}, {"t": lambda x: 0}),
    }

    source_names = ["Empirical"]
    if config["frac_paired_interventional"] > 0:
        source_names = ["PairedEmpirical", "PairedInterventional"] + source_names
    if config["frac_paired_counterfactual"] > 0:
        source_names = ["PairedEmpirical", "PairedCounterfactual"] + source_names

    if config["frac_counterfactual"] > 0:
        source_names.append("Counterfactual")
    if config["frac_interventional"] > 0:
        source_names.append("Interventional")

    return {
        "experiment_name": experiment_name,
        "observed_covariates": observed_covariates,
        "outcome_name": "y",
        "treatment_name": "t",
        "causal_model_prior_fn": lambda: causal_model_prior_fn(
            cf_noise_std=config["cf_noise_std"]
        ),
        "mixture_builder_fn": lambda scm: build_mixture(scm, config),
        "queries": queries,
        "source_names": source_names,
        "empirical_max_len": None,
        "data_generation_keys": [
            "frac_paired_interventional",
            "frac_paired_counterfactual",
            "frac_interventional",
            "frac_counterfactual",
            "cf_noise_std",
        ],
    }


def add_args(parser):
    parser.add_argument("--frac_interventional", type=float, default=0.0)
    parser.add_argument("--frac_counterfactual", type=float, default=0.0)
    parser.add_argument("--frac_paired_counterfactual", type=float, default=0.0)
    parser.add_argument("--frac_paired_interventional", type=float, default=0.0)
    parser.add_argument("--cf_noise_std", type=float, default=1.0)


_EXPERIMENT_SPECS = {
    "cf_vs_int": {
        "experiment_name": "cf_vs_int",
        "observed_covariates": ["y", "t", "x"],
    },
}

experiment_setup = {}
for key, spec in _EXPERIMENT_SPECS.items():

    def custom_setup(
        config,
        observed_covariates=spec["observed_covariates"],
        experiment_name=spec["experiment_name"],
    ):
        return _build_cf_int_setup(config, observed_covariates, experiment_name)

    experiment_setup[key] = custom_setup
