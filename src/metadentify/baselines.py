from typing import Any

import numpy as np
from econml.dml import CausalForestDML, LinearDML
from econml.iv.dml import DMLIV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor


def confounder_linear_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    dataset_x = np.c_[t, x]
    linmod = LinearRegression().fit(X=dataset_x, y=y)
    estimate = linmod.coef_[0]

    if query_x is not None:
        estimate = np.ones((query_x.shape[0],)) * estimate

    return estimate


def confounder_ridge_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    dataset_x = np.c_[t, x]
    linmod = Ridge().fit(X=dataset_x, y=y)
    estimate = linmod.coef_[0]

    if query_x is not None:
        estimate = np.ones((query_x.shape[0],)) * estimate

    return estimate


def confounder_mlp_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    dataset_x = np.c_[t, x]
    mlp = MLPRegressor(max_iter=3000).fit(X=dataset_x, y=y)
    if np.array_equal(np.unique(t), [0, 1]):
        t_sample = t * 0
    else:
        t_sample = np.random.uniform(low=t.min(), high=t.max(), size=x.shape[0])
    control_x = np.c_[t_sample, x]
    treatment_x = np.c_[t_sample + 1, x]
    estimate = np.mean(mlp.predict(X=treatment_x) - mlp.predict(X=control_x))

    if query_x is not None:
        estimate = np.ones((query_x.shape[0],)) * estimate

    return estimate


def treatment_only_linear_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    linmod = LinearRegression().fit(X=t.reshape(-1, 1), y=y)
    estimate = linmod.coef_[0]

    if query_x is not None:
        estimate = np.ones((query_x.shape[0],)) * estimate

    return estimate


def treatment_only_ridge_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    linmod = Ridge().fit(X=t.reshape(-1, 1), y=y)
    estimate = linmod.coef_[0]

    if query_x is not None:
        estimate = np.ones((query_x.shape[0],)) * estimate

    return estimate


def tsls_linear_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    stage1linmod = LinearRegression().fit(X=x, y=t)
    predicted_t = stage1linmod.predict(X=x)
    stagt2linmod = LinearRegression().fit(X=predicted_t.reshape(-1, 1), y=y)
    estimate = stagt2linmod.coef_[0]
    return estimate


def tsls_ridge_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    stage1linmod = Ridge().fit(X=x, y=t)
    predicted_t = stage1linmod.predict(X=x)
    stagt2linmod = Ridge().fit(X=predicted_t.reshape(-1, 1), y=y)
    estimate = stagt2linmod.coef_[0]
    return estimate


def tsls_mlp_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    stage1model = MLPRegressor().fit(X=x, y=t.ravel())
    predicted_t = stage1model.predict(X=x)
    predicted_t = np.expand_dims(predicted_t, -1)
    stage2model = Ridge().fit(X=predicted_t, y=y)
    estimate = stage2model.coef_[0]
    return estimate


def proxy_linear_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    proxy_1 = x[:, [0]]
    proxy_2 = x[:, [1]]
    treatment_and_proxy_2 = np.c_[proxy_2, t]
    stage1model = LinearRegression().fit(X=treatment_and_proxy_2, y=proxy_1)
    predicted_proxy_1 = stage1model.predict(X=treatment_and_proxy_2)
    treatment_and_predicted_proxy = np.c_[t, predicted_proxy_1]
    stage2model = LinearRegression().fit(X=treatment_and_predicted_proxy, y=y)
    estimate = stage2model.coef_[0]
    return estimate


def proxy_ridge_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    proxy_1 = x[:, [0]]
    proxy_2 = x[:, [1]]
    treatment_and_proxy_2 = np.c_[proxy_2, t]
    stage1model = Ridge().fit(X=treatment_and_proxy_2, y=proxy_1)
    predicted_proxy_1 = stage1model.predict(X=treatment_and_proxy_2)
    treatment_and_predicted_proxy = np.c_[t, predicted_proxy_1]
    stage2model = Ridge().fit(X=treatment_and_predicted_proxy, y=y)
    estimate = stage2model.coef_[0]
    return estimate


def proxy_mlp_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    proxy_1 = x[:, [0]]
    proxy_2 = x[:, [1]]
    treatment_and_proxy_2 = np.c_[proxy_2, t]
    stage1model = MLPRegressor().fit(X=treatment_and_proxy_2, y=proxy_1)
    predicted_proxy_1 = stage1model.predict(X=treatment_and_proxy_2)
    treatment_and_predicted_proxy = np.c_[t, predicted_proxy_1]
    stage2model = Ridge().fit(X=treatment_and_predicted_proxy, y=y)
    estimate = stage2model.coef_[0]
    return estimate


def confounder_dml_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
    type: str = "forest",
    binary_t: bool = False,
) -> float | np.ndarray:
    if type == "forest":
        model = CausalForestDML(
            discrete_treatment=binary_t,
            discrete_outcome=False,
            n_jobs=None,
        )
    elif type == "linear":
        model = LinearDML(
            discrete_treatment=binary_t,
            discrete_outcome=False,
        )
    model.fit(Y=y, T=t, X=x, W=None)
    estimate = model.ate(X=x)
    return estimate


def iv_dml_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray | None = None,
    x_sources: np.ndarray | None = None,
    type: str = "forest",
    binary_t: bool = False,
) -> float | np.ndarray:
    if type == "forest":
        model = DMLIV(
            discrete_treatment=binary_t,
            discrete_instrument=False,
            model_y_xw="forest",
            model_t_xw="forest",
            model_t_xwz="forest",
        )
    elif type == "linear":
        model = DMLIV(
            discrete_treatment=binary_t,
            discrete_instrument=False,
            model_y_xw="linear",
            model_t_xw="linear",
            model_t_xwz="linear",
        )
    model.fit(Y=y, T=t, Z=x, X=None, W=None)
    estimate = model.ate()
    return estimate


def cate_confounder_linear_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    dataset_x = np.c_[x * t, t]
    linmod = LinearRegression().fit(X=dataset_x, y=y)
    estimate = linmod.coef_[0] * query_x + linmod.coef_[1]
    return estimate


def cate_confounder_ridge_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray,
    x_sources: np.ndarray | None = None,
) -> float | np.ndarray:
    dataset_x = np.c_[x * t, t]
    linmod = Ridge().fit(X=dataset_x, y=y)
    estimate = query_x * linmod.coef_[0] + linmod.coef_[1]
    return estimate


def cate_confounder_mlp_baseline(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    query_x: np.ndarray,
    num_t_samples: int = 1000,
    x_sources: np.ndarray | None = None,
) -> list[float]:
    dataset_x = np.c_[t, x]
    mlp = MLPRegressor(max_iter=3000).fit(X=dataset_x, y=y)
    if np.array_equal(np.unique(t), [0, 1]):
        t_sample = np.zeros((num_t_samples, 1))
    else:
        t_sample = np.random.uniform(low=t.min(), high=t.max(), size=(num_t_samples, 1))

    estimates = []
    for i in range(query_x.shape[0]):
        query_row = np.array(query_x[i]).reshape(1, -1)
        query_point = np.repeat(query_row, repeats=num_t_samples, axis=0)
        control_x = np.c_[t_sample, query_point]
        treatment_x = np.c_[t_sample + 1, query_point]
        estimate = np.mean(mlp.predict(X=treatment_x) - mlp.predict(X=control_x))
        estimates.append(estimate)

    return estimates


def get_baselines(baseline_setting: str) -> list[dict[str, Any]]:
    t_only_dict = {
        "model": treatment_only_ridge_baseline,
        "name": "t_only_ridge",
        "alias": "T-Only-Ridge",
    }

    baselines = [t_only_dict]

    if baseline_setting in [
        "t-only",
        "t-only-double-invalid-ocp",
        "t-only-iv-invalid-tcp",
    ]:
        return baselines

    elif baseline_setting == "confounder":
        baseline_1_model = confounder_ridge_baseline
        baseline_1_name = "ridge_regression"
        baseline_1_alias = "Reg-Ridge"

        baseline_2_model = confounder_mlp_baseline
        baseline_2_name = "mlp_regression"
        baseline_2_alias = "Reg-MLP"

    elif baseline_setting == "iv":
        baseline_1_model = tsls_ridge_baseline
        baseline_1_name = "tsls_ridge"
        baseline_1_alias = "TSLS-Ridge"

        baseline_2_model = tsls_mlp_baseline
        baseline_2_name = "mlp_tsls"
        baseline_2_alias = "TSLS-MLP"

    elif baseline_setting in ["proxy", "double-invalid-ocp"]:
        baseline_1_model = proxy_ridge_baseline
        baseline_1_name = "proxy_tsls_ridge"
        baseline_1_alias = "PrTSLS-Ridge"

        baseline_2_model = proxy_mlp_baseline
        baseline_2_name = "mlp_proxy_tsls"
        baseline_2_alias = "PrTSLS-MLP"

    elif baseline_setting == "confounder-linreg":
        baseline_1_model = confounder_linear_baseline
        baseline_1_name = "linear_regression"
        baseline_1_alias = "Reg-Lin"

        baseline_2_model = confounder_mlp_baseline
        baseline_2_name = "mlp_regression"
        baseline_2_alias = "Reg-MLP"

    elif baseline_setting == "iv-linreg":
        baseline_1_model = tsls_linear_baseline
        baseline_1_name = "tsls_linear"
        baseline_1_alias = "TSLS-Lin"

        baseline_2_model = tsls_mlp_baseline
        baseline_2_name = "mlp_tsls"
        baseline_2_alias = "TSLS-MLP"

    elif baseline_setting in ["proxy-linreg", "double-invalid-ocp-linreg"]:
        baseline_1_model = proxy_linear_baseline
        baseline_1_name = "proxy_tsls_linear"
        baseline_1_alias = "PrTSLS-Lin"

        baseline_2_model = proxy_mlp_baseline
        baseline_2_name = "mlp_proxy_tsls"
        baseline_2_alias = "PrTSLS-MLP"

    elif baseline_setting == "confounder-and-iv":
        pass

    elif baseline_setting == "iv-invalid-tcp":
        pass

    elif baseline_setting == "confounder-binary-t-dml":

        def baseline_1_model(x, t, y, query_x):
            return confounder_dml_baseline(x, t, y, query_x, type="linear", binary_t=True)

        baseline_1_name = "linear_dml"
        baseline_1_alias = "DML-Lin"

        def baseline_2_model(x, t, y, query_x):
            return confounder_dml_baseline(x, t, y, query_x, type="forest", binary_t=True)

        baseline_2_name = "forest_dml"
        baseline_2_alias = "DML-RF"

    elif baseline_setting == "confounder-continuous-t-dml":

        def baseline_1_model(x, t, y, query_x):
            return confounder_dml_baseline(x, t, y, query_x, type="linear", binary_t=False)

        baseline_1_name = "linear_dml"
        baseline_1_alias = "DML-Lin"

        def baseline_2_model(x, t, y, query_x):
            return confounder_dml_baseline(x, t, y, query_x, type="forest", binary_t=False)

        baseline_2_name = "forest_dml"
        baseline_2_alias = "DML-RF"

    elif baseline_setting == "iv-binary-t-dml":

        def baseline_1_model(x, t, y, query_x):
            return iv_dml_baseline(x, t, y, query_x, type="linear", binary_t=True)

        baseline_1_name = "linear_dmliv"
        baseline_1_alias = "DMLIV-Lin"

        def baseline_2_model(x, t, y, query_x):
            return iv_dml_baseline(x, t, y, query_x, type="forest", binary_t=True)

        baseline_2_name = "forest_dmliv"
        baseline_2_alias = "DMLIV-RF"

    elif baseline_setting == "iv-continuous-t-dml":

        def baseline_1_model(x, t, y, query_x):
            return iv_dml_baseline(x, t, y, query_x, type="linear", binary_t=False)

        baseline_1_name = "linear_dmliv"
        baseline_1_alias = "DMLIV-Lin"

        def baseline_2_model(x, t, y, query_x):
            return iv_dml_baseline(x, t, y, query_x, type="forest", binary_t=False)

        baseline_2_name = "forest_dmliv"
        baseline_2_alias = "DMLIV-RF"

    elif baseline_setting == "cate-confounder-linear":
        baseline_1_model = cate_confounder_ridge_baseline
        baseline_1_name = "cate_ridge"
        baseline_1_alias = "CATE-Ridge"

        baseline_2_model = cate_confounder_mlp_baseline
        baseline_2_name = "cate_mlp"
        baseline_2_alias = "CATE-MLP"

    elif baseline_setting == "cate-confounder-nonlinear":
        baseline_1_model = cate_confounder_ridge_baseline
        baseline_1_name = "cate_ridge"
        baseline_1_alias = "CATE-Ridge"

        baseline_2_model = cate_confounder_mlp_baseline
        baseline_2_name = "cate_mlp"
        baseline_2_alias = "CATE-MLP"

    else:
        raise ValueError(f"Unsupported baseline_setting: {baseline_setting}")

    baselines.append(dict(model=baseline_1_model, name=baseline_1_name, alias=baseline_1_alias))
    baselines.append(dict(model=baseline_2_model, name=baseline_2_name, alias=baseline_2_alias))

    return baselines
