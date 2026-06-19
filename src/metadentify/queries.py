from collections.abc import Callable
from typing import Any

import dowhy.gcm as gcm
import numpy as np
import pandas as pd


def make_pate_query_fn(
    outcome: str,
    treatment_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
    control_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
    num_mc_samples: int = 1000,
) -> Callable[[gcm.ProbabilisticCausalModel, Any], float | np.ndarray]:
    def query_fn(
        causal_model: gcm.ProbabilisticCausalModel,
        query_point: pd.DataFrame | dict[str, np.ndarray] = None,
    ) -> float | np.ndarray:
        if query_point is None:
            treated = gcm.interventional_samples(
                causal_model=causal_model,
                num_samples_to_draw=num_mc_samples,
                interventions=treatment_intervention,
            )[outcome]

            control = gcm.interventional_samples(
                causal_model=causal_model,
                num_samples_to_draw=num_mc_samples,
                interventions=control_intervention,
            )[outcome]
            return float(treated.mean() - control.mean())
        else:
            if isinstance(query_point, dict):
                query_point = pd.DataFrame([query_point])

            grouping = np.repeat(query_point.index, num_mc_samples)

            treated = gcm.interventional_samples(
                causal_model=causal_model,
                num_samples_to_draw=len(grouping),
                interventions=treatment_intervention,
            )[[outcome]]

            control = gcm.interventional_samples(
                causal_model=causal_model,
                num_samples_to_draw=len(grouping),
                interventions=control_intervention,
            )[[outcome]]

            treated["group"] = grouping
            control["group"] = grouping

            return treated.groupby("group").mean().values - control.groupby("group").mean().values

    return query_fn


def make_cate_query_fn(
    outcome: str,
    treatment_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
    control_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
    num_mc_samples: int = 100,
    query_feature_names: list[str] = None,
) -> Callable[[gcm.ProbabilisticCausalModel, pd.DataFrame | dict[str, np.ndarray]], np.ndarray]:
    def query_fn(
        causal_model: gcm.ProbabilisticCausalModel,
        query_point: pd.DataFrame | dict[str, np.ndarray],
    ) -> np.ndarray:
        if isinstance(query_point, dict):
            query_point = pd.DataFrame([query_point])

        grouping = np.repeat(query_point.index, num_mc_samples)
        repeated_samples = query_point.loc[grouping].reset_index(drop=True)

        df_to_condition_on = gcm.draw_samples(causal_model=causal_model, num_samples=len(grouping))
        df_to_condition_on[query_feature_names] = repeated_samples[query_feature_names]

        treated = gcm.interventional_samples(
            causal_model=causal_model,
            observed_data=df_to_condition_on,
            interventions=treatment_intervention,
        )[[outcome]]

        control = gcm.interventional_samples(
            causal_model=causal_model,
            observed_data=df_to_condition_on,
            interventions=control_intervention,
        )[[outcome]]

        treated["group"] = grouping
        control["group"] = grouping

        return treated.groupby("group").mean().values - control.groupby("group").mean().values

    return query_fn


def make_sate_query_fn(
    outcome: str,
    treatment_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
    control_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> Callable[[gcm.InvertibleStructuralCausalModel, pd.DataFrame], float]:
    def query_fn(
        causal_model: gcm.InvertibleStructuralCausalModel, observed_data: pd.DataFrame
    ) -> float:
        treated = gcm.counterfactual_samples(
            causal_model=causal_model,
            observed_data=observed_data,
            interventions=treatment_intervention,
        )[outcome]

        control = gcm.counterfactual_samples(
            causal_model=causal_model,
            observed_data=observed_data,
            interventions=control_intervention,
        )[outcome]

        ites = treated.values - control.values
        return float(np.mean(ites))

    return query_fn


def make_ite_query_fn(
    outcome: str,
    treatment_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
    control_intervention: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> Callable[[gcm.InvertibleStructuralCausalModel, pd.DataFrame], np.ndarray]:
    def query_fn(
        causal_model: gcm.InvertibleStructuralCausalModel, observed_data: pd.DataFrame
    ) -> np.ndarray:
        treated = gcm.counterfactual_samples(
            causal_model=causal_model,
            observed_data=observed_data,
            interventions=treatment_intervention,
        )[outcome]

        control = gcm.counterfactual_samples(
            causal_model=causal_model,
            observed_data=observed_data,
            interventions=control_intervention,
        )[outcome]

        ites = np.expand_dims(treated.values - control.values, axis=1)
        return ites

    return query_fn
