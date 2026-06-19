import bisect
import multiprocessing
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, AnyStr

import cloudpickle
import dowhy.gcm as gcm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from metadentify.utils import set_worker_random_seed, set_worker_single_thread


class PicklableLambdaWrapper:
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        return cloudpickle.dumps(self.func)

    def __setstate__(self, state):
        self.func = cloudpickle.loads(state)


def _draw_emp(n: int, data: pd.DataFrame) -> pd.DataFrame:
    return data.sample(n=n, replace=True).reset_index(drop=True)


def _draw_obs(n: int, causal_model: gcm.ProbabilisticCausalModel) -> pd.DataFrame:
    return gcm.draw_samples(causal_model, num_samples=n)


def _draw_int(
    n: int,
    causal_model: gcm.ProbabilisticCausalModel,
    intervention: dict[str, Callable],
) -> pd.DataFrame:
    return gcm.interventional_samples(
        causal_model, interventions=intervention, num_samples_to_draw=n
    )


def _draw_cf(
    n: int,
    causal_model: gcm.ProbabilisticCausalModel,
    evidence_data: pd.DataFrame,
    hypothetical_intervention: dict[str, Callable],
) -> pd.DataFrame:
    evidence_batch = evidence_data.sample(n=n, replace=True)
    return gcm.counterfactual_samples(
        causal_model,
        observed_data=evidence_batch,
        interventions=hypothetical_intervention,
    ).reset_index(drop=True)


def _draw_paired_cf(
    n: int,
    causal_model: gcm.ProbabilisticCausalModel,
    evidence_data: pd.DataFrame,
    hypothetical_intervention: dict[str, Callable],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    n_pairs = (n + 1) // 2
    observed_batch = evidence_data.sample(n=n_pairs, replace=True).reset_index(drop=True)
    cf_batch = gcm.counterfactual_samples(
        causal_model,
        observed_data=observed_batch,
        interventions=hypothetical_intervention,
    ).reset_index(drop=True)

    observed_batch["pair_id"] = np.arange(n_pairs)
    cf_batch["pair_id"] = np.arange(n_pairs)

    if n % 2 != 0:
        cf_batch = cf_batch.iloc[:-1]

    return observed_batch, cf_batch


def _draw_paired_int(
    n: int,
    causal_model: gcm.ProbabilisticCausalModel,
    evidence_data: pd.DataFrame,
    intervention: dict[str, Callable],
    observed_variable_names: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    n_pairs = (n + 1) // 2
    observed_batch = evidence_data.sample(n=n_pairs, replace=True).reset_index(drop=True)

    if observed_variable_names is not None:
        batch_to_condition_on = gcm.draw_samples(causal_model=causal_model, num_samples=n_pairs)
        batch_to_condition_on[observed_variable_names] = observed_batch[observed_variable_names]
    else:
        batch_to_condition_on = observed_batch

    int_batch = gcm.interventional_samples(
        causal_model, interventions=intervention, observed_data=batch_to_condition_on
    ).reset_index(drop=True)

    batch_to_condition_on["pair_id"] = np.arange(n_pairs)
    int_batch["pair_id"] = np.arange(n_pairs)

    if n % 2 != 0:
        int_batch = int_batch.iloc[:-1]

    return batch_to_condition_on, int_batch


class CausalMixtureDistribution:
    def __init__(self, causal_model: gcm.ProbabilisticCausalModel) -> None:
        self.causal_model = causal_model
        self.components: list[Callable[[int], pd.DataFrame | tuple[pd.DataFrame, ...]]] = []
        self.weights: list[float] = []
        self.names: list[str] = []
        self.component_source_counts: list[int] = []

    def add_empirical(self, data: pd.DataFrame, weight: float, name: str = "Empirical") -> None:
        self.components.append(partial(_draw_emp, data=data))
        self.weights.append(weight)
        self.names.append(name)
        self.component_source_counts.append(1)

    def add_observational(self, weight: float, name: str = "Observational") -> None:
        self.components.append(partial(_draw_obs, causal_model=self.causal_model))
        self.weights.append(weight)
        self.names.append(name)
        self.component_source_counts.append(1)

    def add_interventional(
        self,
        intervention: dict[str, Callable],
        weight: float,
        name: str = "Interventional",
    ) -> None:
        picklable_intervention = {k: PicklableLambdaWrapper(v) for k, v in intervention.items()}
        self.components.append(
            partial(
                _draw_int,
                causal_model=self.causal_model,
                intervention=picklable_intervention,
            )
        )
        self.weights.append(weight)
        self.names.append(name)
        self.component_source_counts.append(1)

    def add_counterfactual(
        self,
        evidence_data: pd.DataFrame,
        hypothetical_intervention: dict[str, Callable],
        weight: float,
        name: str = "Counterfactual",
    ) -> None:
        picklable_intervention = {
            k: PicklableLambdaWrapper(v) for k, v in hypothetical_intervention.items()
        }
        self.components.append(
            partial(
                _draw_cf,
                causal_model=self.causal_model,
                evidence_data=evidence_data,
                hypothetical_intervention=picklable_intervention,
            )
        )
        self.weights.append(weight)
        self.names.append(name)
        self.component_source_counts.append(1)

    def add_paired_counterfactual(
        self,
        evidence_data: pd.DataFrame,
        hypothetical_intervention: dict[str, Callable],
        weight: float,
        names: list[str] | None = None,
    ) -> None:
        if names is None:
            names = ["PairedEmpirical", "PairedCounterfactual"]
        picklable_intervention = {
            k: PicklableLambdaWrapper(v) for k, v in hypothetical_intervention.items()
        }
        self.components.append(
            partial(
                _draw_paired_cf,
                causal_model=self.causal_model,
                evidence_data=evidence_data,
                hypothetical_intervention=picklable_intervention,
            )
        )
        self.weights.append(weight)
        self.names.extend(names)
        self.component_source_counts.append(len(names))

    def add_paired_interventional(
        self,
        evidence_data: pd.DataFrame,
        intervention: dict[str, Callable],
        weight: float,
        names: list[str] | None = None,
        observed_variable_names: list[str] = None,
    ) -> None:
        if names is None:
            names = ["PairedEmpirical", "PairedInterventional"]
        picklable_intervention = {k: PicklableLambdaWrapper(v) for k, v in intervention.items()}
        self.components.append(
            partial(
                _draw_paired_int,
                causal_model=self.causal_model,
                evidence_data=evidence_data,
                intervention=picklable_intervention,
                observed_variable_names=observed_variable_names,
            )
        )
        self.weights.append(weight)
        self.names.extend(names)
        self.component_source_counts.append(len(names))

    def sample_batch(self, total_samples: int) -> pd.DataFrame:
        if not self.components:
            raise ValueError("No components added to the mixture sampler.")

        probs = np.array(self.weights) / np.sum(self.weights)
        component_counts = np.random.multinomial(total_samples, probs)

        batch_pieces = []
        source_idx = 0
        for idx, count in enumerate(component_counts):
            num_sources = self.component_source_counts[idx]
            if count > 0:
                component_sample = self.components[idx](count)
                if isinstance(component_sample, (list, tuple)):
                    for i, df_piece in enumerate(component_sample):
                        df_piece["source_component"] = source_idx + i
                        batch_pieces.append(df_piece)
                else:
                    component_sample["source_component"] = source_idx
                    batch_pieces.append(component_sample)

            source_idx += num_sources

        return pd.concat(batch_pieces, ignore_index=True)

    def plot_covariate(
        self,
        covariate_name: str,
        sample_df: pd.DataFrame | None = None,
        num_samples: int | None = None,
    ) -> None:
        if sample_df is None and num_samples is None:
            raise ValueError("Must specify either sample_df or num_samples")
        if sample_df is None:
            sample_df = self.sample_batch(total_samples=num_samples)
        num_legend_cols = min(len(sample_df["source_component"].unique()), 3)
        plt.figure(figsize=(4, 2))
        name_mapping = dict(enumerate(self.names))
        g = sns.histplot(
            data=sample_df.assign(source_component=sample_df["source_component"].map(name_mapping)),
            x=covariate_name,
            hue="source_component",
            kde=True,
            stat="density",
            linewidth=0.5,
        )
        sns.move_legend(
            obj=g,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.65),
            ncol=num_legend_cols,
            title=None,
            frameon=False,
        )
        plt.show()


class LazyOfflineCausalDataset(Dataset):
    def __init__(self, data_dir: str, cache_size: int = 4) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_files = sorted(self.data_dir.glob("*.pt"))
        self.cache_size = cache_size
        self._chunk_cache: dict[int, Any] = {}

        if not self.chunk_files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        self.chunk_lengths = []
        for file in self.chunk_files:
            chunk = torch.load(file, weights_only=True)
            self.chunk_lengths.append(len(chunk))

        self.cumulative_sizes = [
            sum(self.chunk_lengths[: i + 1]) for i in range(len(self.chunk_lengths))
        ]

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _get_chunk(self, chunk_idx: int) -> list[Any]:
        if chunk_idx not in self._chunk_cache:
            if len(self._chunk_cache) >= self.cache_size:
                oldest_key = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest_key]

            self._chunk_cache[chunk_idx] = torch.load(
                self.chunk_files[chunk_idx], weights_only=True
            )

        return self._chunk_cache[chunk_idx]

    def __getitem__(self, idx):
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if chunk_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[chunk_idx - 1]
        chunk_data = self._get_chunk(chunk_idx)
        return chunk_data[local_idx]


def _worker_generate_task(
    prior_fn_wrapped: PicklableLambdaWrapper,
    mixture_fn_wrapped: PicklableLambdaWrapper,
    query_fn_wrapped: PicklableLambdaWrapper,
    num_context_points: int,
    num_query_points: int,
    outcome_name: AnyStr,
    treatment_name: AnyStr,
    observed_feature_names: list[AnyStr],
    query_feature_names: list[AnyStr],
    query_x_type: str,
    standardize: bool = False,
) -> dict[str, torch.Tensor]:
    set_worker_single_thread()
    set_worker_random_seed()

    scm = prior_fn_wrapped()

    mixture = mixture_fn_wrapped(scm)
    df_x = mixture.sample_batch(total_samples=num_context_points)
    df_x_for_query = df_x.copy(deep=True)

    sigma_y = 1.0
    sigma_t = 1.0
    if standardize:
        df_y = df_x[[outcome_name]].values
        mu_y = float(df_y.mean())
        sigma_y = float(df_y.std())
        if sigma_y == 0.0 or np.isnan(sigma_y):
            sigma_y = 1.0
            mu_y = 0.0

        df_x[outcome_name] = (df_x[outcome_name] - mu_y) / sigma_y

        covariates = [c for c in observed_feature_names if c not in [treatment_name, outcome_name]]
        if covariates:
            cov_scaler = StandardScaler().fit(df_x[covariates].values)
            df_x[covariates] = cov_scaler.transform(df_x[covariates].values)

        if df_x[treatment_name].dtype == float and df_x[treatment_name].nunique() > 2:
            treat_scaler = StandardScaler().fit(df_x[[treatment_name]].values)
            sigma_t = float(treat_scaler.scale_[0])
            if sigma_t == 0.0 or np.isnan(sigma_t):
                sigma_t = 1.0
            df_x[[treatment_name]] = treat_scaler.transform(df_x[[treatment_name]].values)
        else:
            treat_scaler = None

    if "source_component" in df_x.columns:
        x_sources = torch.tensor(df_x["source_component"].values, dtype=torch.int64)
        x_features = torch.tensor(df_x[observed_feature_names].values, dtype=torch.float32)
    else:
        x_sources = torch.zeros(len(df_x), dtype=torch.int64)
        x_features = torch.tensor(df_x[observed_feature_names].values, dtype=torch.float32)

    if query_x_type == "none":
        query_x = torch.full(
            size=(num_query_points, 1), fill_value=float("nan"), dtype=torch.float32
        )
        query_value = query_fn_wrapped(scm)

    elif query_x_type == "full-context-dummy":
        query_x = torch.full(
            size=(num_query_points, 1), fill_value=float("nan"), dtype=torch.float32
        )
        query_value = query_fn_wrapped(scm, df_x_for_query)

    elif query_x_type == "sample-from-context":
        query_df = df_x_for_query.sample(n=num_query_points, replace=True, ignore_index=True)
        query_value = query_fn_wrapped(scm, query_df)

        if standardize:
            query_df[outcome_name] = (query_df[outcome_name] - mu_y) / sigma_y
            query_covs = [c for c in query_feature_names if c not in [treatment_name, outcome_name]]
            if query_covs and covariates:
                query_df[query_covs] = cov_scaler.transform(query_df[query_covs].values)
            if treatment_name in query_feature_names and treat_scaler is not None:
                query_df[[treatment_name]] = treat_scaler.transform(
                    query_df[[treatment_name]].values
                )

        query_x = torch.tensor(query_df[query_feature_names].values, dtype=torch.float32)

    elif query_x_type == "sample-from-scm":
        query_df = gcm.draw_samples(scm, num_samples=num_query_points)
        query_value = query_fn_wrapped(scm, query_df)

        if standardize:
            query_df[outcome_name] = (query_df[outcome_name] - mu_y) / sigma_y
            query_covs = [c for c in query_feature_names if c not in [treatment_name, outcome_name]]
            if query_covs and covariates:
                query_df[query_covs] = cov_scaler.transform(query_df[query_covs].values)
            if treatment_name in query_feature_names and treat_scaler is not None:
                query_df[[treatment_name]] = treat_scaler.transform(
                    query_df[[treatment_name]].values
                )

        query_x = torch.tensor(query_df[query_feature_names].values, dtype=torch.float32)

    if standardize:
        query_value = (query_value / sigma_y) * sigma_t

    qv_tensor = torch.tensor(query_value, dtype=torch.float32)

    if qv_tensor.dim() == 2 and qv_tensor.size(1) == 1:
        qv_tensor = qv_tensor.squeeze(-1)
    elif qv_tensor.dim() == 0:
        qv_tensor = qv_tensor.expand(num_query_points)

    x_index = list(range(len(observed_feature_names)))
    t_index = observed_feature_names.index(treatment_name)
    y_index = observed_feature_names.index(outcome_name)
    for idx in sorted([t_index, y_index], reverse=True):
        del x_index[idx]

    x_index_tensor = torch.tensor(np.array(x_index), dtype=torch.int64)
    t_index_tensor = torch.tensor(np.array([t_index]), dtype=torch.int64)
    y_index_tensor = torch.tensor(np.array([y_index]), dtype=torch.int64)

    if query_feature_names:
        query_x_index = list(range(len(query_feature_names)))
        query_t_index = query_feature_names.index(treatment_name)
        query_y_index = query_feature_names.index(outcome_name)
        for idx in sorted([query_t_index, query_y_index], reverse=True):
            del query_x_index[idx]
        query_x_index_tensor = torch.tensor(np.array(query_x_index), dtype=torch.int64)
    else:
        query_x_index = []
        query_x_index_tensor = torch.tensor([[]], dtype=torch.int64)

    return {
        "x_features": x_features,
        "x_sources": x_sources,
        "query_x": query_x,
        "query_value": qv_tensor,
        "x_index": x_index_tensor,
        "t_index": t_index_tensor,
        "y_index": y_index_tensor,
        "query_x_index": query_x_index_tensor,
        "y_std": torch.tensor([sigma_y], dtype=torch.float32),
        "t_std": torch.tensor([sigma_t], dtype=torch.float32),
        "standardized": torch.tensor([standardize], dtype=torch.bool),
    }


class OfflineCausalDataGenerator:
    def __init__(
        self,
        causal_model_prior_fn: Callable[[], gcm.ProbabilisticCausalModel],
        mixture_builder_fn: Callable[[gcm.ProbabilisticCausalModel], Any],
        causal_query_fn: Callable[[gcm.ProbabilisticCausalModel, pd.DataFrame], Any],
        outcome_name: AnyStr,
        treatment_name: AnyStr,
        observed_feature_names: list[AnyStr],
        query_feature_names: list[AnyStr],
        query_x_type: AnyStr,
        num_context_points: int = 500,
        num_query_points: int = 20,
        standardize: bool = False,
    ) -> None:
        self.prior_fn = PicklableLambdaWrapper(causal_model_prior_fn)
        self.mixture_fn = PicklableLambdaWrapper(mixture_builder_fn)
        self.query_fn = PicklableLambdaWrapper(causal_query_fn)
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name
        self.observed_feature_names = observed_feature_names
        self.query_feature_names = query_feature_names
        self.num_context_points = num_context_points
        self.num_query_points = num_query_points
        self.query_x_type = query_x_type
        self.standardize = standardize

    def generate_and_save(
        self, total_tasks: int, tasks_per_file: int, save_dir: str, n_workers: int = -1
    ) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if n_workers <= 0:
            raise ValueError(f"Invalid n_workers: {n_workers}")

        _ = self.prior_fn()

        current_chunk = []
        chunk_index = 0
        recruit_threshold = tasks_per_file

        with multiprocessing.Pool(processes=n_workers, maxtasksperchild=recruit_threshold) as pool:
            worker_args = (
                self.prior_fn,
                self.mixture_fn,
                self.query_fn,
                self.num_context_points,
                self.num_query_points,
                self.outcome_name,
                self.treatment_name,
                self.observed_feature_names,
                self.query_feature_names,
                self.query_x_type,
                self.standardize,
            )

            tasks_iterator = (worker_args for _ in range(total_tasks))

            for task_data in tqdm(
                pool.imap_unordered(_worker_unpack_wrapper, tasks_iterator, chunksize=100),
                total=total_tasks,
                desc="Generating Tasks",
            ):
                try:
                    current_chunk.append(task_data)

                    if len(current_chunk) >= tasks_per_file:
                        file_path = save_path / f"chunk_{chunk_index:04d}.pt"
                        torch.save(current_chunk, file_path)
                        current_chunk = []
                        chunk_index += 1

                except Exception as e:
                    print(f"\nWorker failure: {e}")

            if current_chunk:
                file_path = save_path / f"chunk_{chunk_index:04d}.pt"
                torch.save(current_chunk, file_path)


class OnlineCausalDataset(IterableDataset):
    def __init__(
        self,
        causal_model_prior_fn: Callable[[], gcm.ProbabilisticCausalModel],
        mixture_builder_fn: Callable[[gcm.ProbabilisticCausalModel], Any],
        causal_query_fn: Callable[[gcm.ProbabilisticCausalModel, pd.DataFrame], Any],
        outcome_name: AnyStr,
        treatment_name: AnyStr,
        observed_feature_names: list[AnyStr],
        query_feature_names: list[AnyStr],
        query_x_type: AnyStr,
        num_context_points: int = 500,
        num_query_points: int = 20,
        standardize: bool = False,
    ) -> None:
        super().__init__()
        self.prior_fn = PicklableLambdaWrapper(causal_model_prior_fn)
        self.mixture_fn = PicklableLambdaWrapper(mixture_builder_fn)
        self.query_fn = PicklableLambdaWrapper(causal_query_fn)
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name
        self.observed_feature_names = observed_feature_names
        self.query_feature_names = query_feature_names
        self.num_context_points = num_context_points
        self.num_query_points = num_query_points
        self.query_x_type = query_x_type
        self.standardize = standardize

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        set_worker_random_seed()
        set_worker_single_thread()

        while True:
            yield _worker_generate_task(
                self.prior_fn,
                self.mixture_fn,
                self.query_fn,
                self.num_context_points,
                self.num_query_points,
                self.outcome_name,
                self.treatment_name,
                self.observed_feature_names,
                self.query_feature_names,
                self.query_x_type,
                self.standardize,
            )


class OnlineCausalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        causal_model_prior_fn: Callable[[], gcm.ProbabilisticCausalModel],
        mixture_builder_fn: Callable[[gcm.ProbabilisticCausalModel], Any],
        causal_query_fn: Callable[[gcm.ProbabilisticCausalModel, pd.DataFrame], Any],
        outcome_name: AnyStr,
        treatment_name: AnyStr,
        observed_feature_names: list[AnyStr],
        query_feature_names: list[AnyStr],
        query_x_type: AnyStr,
        num_context_points: int = 500,
        num_query_points: int = 20,
        standardize: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        data_dir: str | None = None,
        empirical_data_dir: str | None = None,
        empirical_parser_fn: Callable[[pd.DataFrame], dict[str, torch.Tensor]] | None = None,
        empirical_max_len: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_kwargs = {
            "causal_model_prior_fn": causal_model_prior_fn,
            "mixture_builder_fn": mixture_builder_fn,
            "causal_query_fn": causal_query_fn,
            "outcome_name": outcome_name,
            "treatment_name": treatment_name,
            "observed_feature_names": observed_feature_names,
            "query_feature_names": query_feature_names,
            "query_x_type": query_x_type,
            "num_context_points": num_context_points,
            "num_query_points": num_query_points,
            "standardize": standardize,
        }
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.data_dir = Path(data_dir) if data_dir else None
        self.empirical_data_dir = empirical_data_dir
        self.empirical_parser_fn = empirical_parser_fn
        self.empirical_max_len = empirical_max_len

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = OnlineCausalDataset(**self.dataset_kwargs)

        if self.data_dir:
            self.val_ds = LazyOfflineCausalDataset(self.data_dir / "val", cache_size=4)
            self.test_ds = LazyOfflineCausalDataset(self.data_dir / "test", cache_size=4)
        else:
            self.val_ds = OnlineCausalDataset(**self.dataset_kwargs)
            self.test_ds = OnlineCausalDataset(**self.dataset_kwargs)

        self.empirical_ds = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self, num_workers: int | None = None) -> DataLoader | list[DataLoader]:
        nw = num_workers if num_workers is not None else self.num_workers
        simulated_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
        )
        if self.empirical_ds is not None:
            empirical_dl = DataLoader(
                self.empirical_ds,
                batch_size=1,
                num_workers=nw,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=nw > 0,
            )
            return [simulated_dl, empirical_dl]
        else:
            return simulated_dl


class OfflineCausalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        empirical_data_dir: str | None = None,
        empirical_parser_fn: Callable[[pd.DataFrame], dict[str, torch.Tensor]] | None = None,
        empirical_max_len: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.empirical_data_dir = empirical_data_dir
        self.empirical_parser_fn = empirical_parser_fn
        self.empirical_max_len = empirical_max_len

    def setup(self, stage=None):
        self.train_ds = LazyOfflineCausalDataset(self.data_dir / "train", cache_size=8)
        self.val_ds = LazyOfflineCausalDataset(self.data_dir / "val", cache_size=4)
        self.test_ds = LazyOfflineCausalDataset(self.data_dir / "test", cache_size=4)

        self.empirical_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self, num_workers: int | None = None) -> DataLoader | list[DataLoader]:
        nw = num_workers if num_workers is not None else self.num_workers
        simulated_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=nw,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
        )
        if self.empirical_ds is not None:
            empirical_dl = DataLoader(
                self.empirical_ds,
                batch_size=1,
                num_workers=nw,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=nw > 0,
            )
            return [simulated_dl, empirical_dl]
        else:
            return simulated_dl


def _worker_unpack_wrapper(args: tuple) -> dict[str, torch.Tensor]:
    return _worker_generate_task(*args)
