import networkx as nx
import numpy as np
from dowhy import gcm
from scipy import stats
from scipy.special import expit

from metadentify.mechanisms import build_synthetic_causal_model
from metadentify.mixture import CausalMixtureDistribution
from metadentify.queries import (
    make_cate_query_fn,
    make_ite_query_fn,
    make_pate_query_fn,
    make_sate_query_fn,
)
from metadentify.utils import disjoint_uniform


def resample_node_function_dict(setting, binary_treatment, hidden_confounder_weight):
    gamma_x = disjoint_uniform(low_left=-2, high_left=-1, low_right=1, high_right=2, size=1)
    beta_x = disjoint_uniform(low_left=-2, high_left=-1, low_right=1, high_right=2, size=1)

    delta_1 = np.random.uniform(low=0, high=1, size=1)
    delta_2 = np.random.uniform(low=0, high=1, size=1)

    delta_invalid_ocp_1 = np.random.uniform(low=0, high=1, size=1)
    delta_invalid_ocp_2 = np.random.uniform(low=0, high=1, size=1)

    delta_invalid_tcp = np.random.uniform(low=0, high=1, size=1)

    gamma_t = np.random.uniform(low=-1, high=1, size=1)
    beta_t = np.random.uniform(low=-1, high=1, size=1)
    beta_y = np.random.uniform(low=-1, high=1, size=1)

    if setting in ["double-invalid-ocp", "t-only-double-invalid-ocp"]:
        beta_invalid_ocp_1 = np.random.uniform(low=-1, high=1, size=1)
        beta_invalid_ocp_2 = np.random.uniform(low=-1, high=1, size=1)
        gamma_invalid_tcp_t = np.array([0.0])
        beta_invalid_tcp_y = np.array([0.0])
    elif setting in ["iv-invalid-tcp", "t-only-iv-invalid-tcp"]:
        beta_invalid_ocp_1 = np.array([0.0])
        beta_invalid_ocp_2 = np.array([0.0])
        gamma_invalid_tcp_t = np.random.uniform(low=-1, high=1, size=1)
        beta_invalid_tcp_y = np.random.uniform(low=-1, high=1, size=1)
    else:
        beta_invalid_ocp_1 = np.array([0.0])
        beta_invalid_ocp_2 = np.array([0.0])
        gamma_invalid_tcp_t = np.array([0.0])
        beta_invalid_tcp_y = np.array([0.0])

    if binary_treatment:

        def t_function(x):
            invalid_w_tcp, u_t, x, x_hidden = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            x_hidden_t = gamma_x * hidden_confounder_weight * x_hidden
            t = np.random.binomial(
                n=1,
                p=expit(
                    x_hidden_t + gamma_x * x + gamma_t * u_t + gamma_invalid_tcp_t * invalid_w_tcp
                ),
            )
            return t
    else:

        def t_function(x):
            invalid_w_tcp, u_t, x, x_hidden = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            x_hidden_t = gamma_x * hidden_confounder_weight * x_hidden
            t = x_hidden_t + gamma_x * x + gamma_t * u_t + gamma_invalid_tcp_t * invalid_w_tcp
            return t

    t_noise = None

    if setting == "linear-cate":

        def y_function(x):
            invalid_w_ocp_1, invalid_w_ocp_2, invalid_w_tcp, t, x, x_hidden = (
                x[:, 0],
                x[:, 1],
                x[:, 2],
                x[:, 3],
                x[:, 4],
                x[:, 5],
            )
            x_hidden_y = beta_x * hidden_confounder_weight * x_hidden
            y = (
                x_hidden_y
                + beta_x * x * t
                + beta_t * t
                + beta_invalid_tcp_y * invalid_w_tcp
                + beta_invalid_ocp_1 * invalid_w_ocp_1
                + beta_invalid_ocp_2 * invalid_w_ocp_2
            )
            return y
    elif setting == "nonlinear-cate":

        def y_function(x):
            invalid_w_ocp_1, invalid_w_ocp_2, invalid_w_tcp, t, x, x_hidden = (
                x[:, 0],
                x[:, 1],
                x[:, 2],
                x[:, 3],
                x[:, 4],
                x[:, 5],
            )
            x_hidden_y = beta_x * hidden_confounder_weight * x_hidden
            y = (
                x_hidden_y
                + beta_x * np.abs(x) * t
                + beta_t * t
                + beta_invalid_tcp_y * invalid_w_tcp
                + beta_invalid_ocp_1 * invalid_w_ocp_1
                + beta_invalid_ocp_2 * invalid_w_ocp_2
            )
            return y
    elif setting in [
        "instrument",
        "confounder",
        "iv",
        "iv-invalid-tcp",
        "proxy",
        "t-only",
        "t-only-double-invalid-ocp",
        "t-only-iv-invalid-tcp",
        "confounder-and-iv",
    ]:

        def y_function(x):
            invalid_w_ocp_1, invalid_w_ocp_2, invalid_w_tcp, t, x, x_hidden = (
                x[:, 0],
                x[:, 1],
                x[:, 2],
                x[:, 3],
                x[:, 4],
                x[:, 5],
            )
            x_hidden_y = beta_x * hidden_confounder_weight * x_hidden
            y = (
                x_hidden_y
                + beta_x * x
                + beta_t * t
                + beta_invalid_tcp_y * invalid_w_tcp
                + beta_invalid_ocp_1 * invalid_w_ocp_1
                + beta_invalid_ocp_2 * invalid_w_ocp_2
            )
            return y
    else:
        raise ValueError(f"Unsupported setting: {setting}")

    y_noise = gcm.ScipyDistribution(stats.norm, loc=beta_y, scale=1)

    def w_function(x):
        return x

    def invalid_w_tcp_function(x):
        return x

    def invalid_w_ocp_function(x):
        t = x[:, 0]
        x = x[:, 1]
        return t + x

    x_tuple = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
    x_hidden_tuple = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
    u_t_tuple = (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
    w_1_tuple = (w_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=delta_1))
    w_2_tuple = (w_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=delta_2))
    invalid_w_tcp_tuple = (
        invalid_w_tcp_function,
        gcm.ScipyDistribution(stats.norm, loc=0, scale=delta_invalid_tcp),
    )
    invalid_w_ocp_1_tuple = (
        invalid_w_ocp_function,
        gcm.ScipyDistribution(stats.norm, loc=0, scale=delta_invalid_ocp_1),
    )
    invalid_w_ocp_2_tuple = (
        invalid_w_ocp_function,
        gcm.ScipyDistribution(stats.norm, loc=0, scale=delta_invalid_ocp_2),
    )
    t_tuple = (t_function, t_noise)
    y_tuple = (y_function, y_noise)

    node_function_dict = {
        "x": x_tuple,
        "x_hidden": x_hidden_tuple,
        "w_1": w_1_tuple,
        "w_2": w_2_tuple,
        "invalid_w_tcp": invalid_w_tcp_tuple,
        "invalid_w_ocp_1": invalid_w_ocp_1_tuple,
        "invalid_w_ocp_2": invalid_w_ocp_2_tuple,
        "u_t": u_t_tuple,
        "t": t_tuple,
        "y": y_tuple,
    }
    return node_function_dict


causal_graph = nx.DiGraph(
    [
        ("x", "y"),
        ("x", "t"),
        ("x", "w_1"),
        ("x", "w_2"),
        ("x", "invalid_w_tcp"),
        ("x", "invalid_w_ocp_1"),
        ("x", "invalid_w_ocp_2"),
        ("x_hidden", "y"),
        ("x_hidden", "t"),
        ("u_t", "t"),
        ("invalid_w_tcp", "t"),
        ("invalid_w_tcp", "y"),
        ("invalid_w_ocp_1", "y"),
        ("invalid_w_ocp_2", "y"),
        ("t", "invalid_w_ocp_1"),
        ("t", "invalid_w_ocp_2"),
        ("t", "y"),
    ]
)


def causal_model_prior_fn(setting, hidden_confounder_weight=0.0):
    node_function_dict = resample_node_function_dict(
        setting=setting,
        binary_treatment=False,
        hidden_confounder_weight=hidden_confounder_weight,
    )
    causal_model = build_synthetic_causal_model(
        causal_graph=causal_graph, node_function_dict=node_function_dict
    )
    return causal_model


def build_mixture(scm, config=None):
    if config is None:
        config = {}
    frac_int = config.get("frac_interventional", 0.0)

    mix = CausalMixtureDistribution(causal_model=scm)
    mix.add_observational(weight=1.0 - frac_int)
    if frac_int > 0:
        mix.add_interventional(
            {"t": lambda x: np.random.uniform(-1, 1)},
            weight=frac_int,
            name="Interventional",
        )
    return mix


def add_args(parser):
    parser.add_argument("--hidden_confounder_weight", type=float, default=0.0)
    return parser


def _build_linear_setup(config, setting, observed_covariates, experiment_name):
    queries = {
        "PATE": make_pate_query_fn(
            "y", {"t": lambda x: x + 1.0}, {"t": lambda x: x}, num_mc_samples=1000
        ),
        "STATE": make_sate_query_fn("y", {"t": lambda x: x + 1.0}, {"t": lambda x: x}),
        "CATE": make_cate_query_fn(
            "y",
            {"t": lambda x: x + 1.0},
            {"t": lambda x: x},
            num_mc_samples=1000,
            query_feature_names=observed_covariates,
        ),
        "ITE": make_ite_query_fn("y", {"t": lambda x: x + 1.0}, {"t": lambda x: x}),
    }

    return {
        "experiment_name": experiment_name,
        "observed_covariates": observed_covariates,
        "outcome_name": "y",
        "treatment_name": "t",
        "causal_model_prior_fn": lambda: causal_model_prior_fn(
            setting, hidden_confounder_weight=config["hidden_confounder_weight"]
        ),
        "mixture_builder_fn": lambda scm: build_mixture(scm, config),
        "queries": queries,
        "source_names": ["Observational", "Interventional"]
        if config.get("frac_interventional", 0.0) > 0
        else ["Observational"],
        "empirical_max_len": None,
        "data_generation_keys": [
            "frac_interventional",
            "hidden_confounder_weight",
            "binary_treatment",
        ],
    }


_LINEAR_SPECS = {
    "t-only": {
        "experiment_name": "t-only",
        "observed_covariates": ["y", "t"],
        "setting": "t-only",
    },
    "confounder": {
        "experiment_name": "confounder",
        "observed_covariates": ["y", "t", "x"],
        "setting": "confounder",
    },
    "iv": {
        "experiment_name": "iv",
        "observed_covariates": ["y", "t", "u_t"],
        "setting": "iv",
    },
    "proxy": {
        "experiment_name": "proxy",
        "observed_covariates": ["y", "t", "w_1", "w_2"],
        "setting": "proxy",
    },
}

experiment_setup = {}
for key, spec in _LINEAR_SPECS.items():

    def custom_linear_setup(
        config,
        setting=spec["setting"],
        observed_covariates=spec["observed_covariates"],
        experiment_name=spec["experiment_name"],
    ):
        return _build_linear_setup(config, setting, observed_covariates, experiment_name)

    experiment_setup[key] = custom_linear_setup
