import warnings
from typing import Any

import numpy as np
import pytorch_lightning as pl
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from metadentify.baselines import get_baselines


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim: int, num_cosines: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.register_buffer("i_pi", torch.arange(1, num_cosines + 1).float() * np.pi)

    def forward(self, taus):
        return self.net(torch.cos(taus * self.i_pi))


class MAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_out, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)
        self.ffn = nn.Sequential(
            nn.Linear(dim_out, dim_out * 4), nn.ReLU(), nn.Linear(dim_out * 4, dim_out)
        )

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, y_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_out, _ = self.mha(query=X, key=Y, value=Y, key_padding_mask=y_mask)
        H = self.ln1(X + attn_out)
        return self.ln2(H + self.ffn(H))


class ISAB(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inducing_points: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inducing_points, dim_out))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(dim_out, dim_in, num_heads, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, num_heads, dropout=dropout)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.size(0)
        I_expanded = self.I.expand(B, -1, -1)
        H = self.mab0(I_expanded, X, y_mask=x_mask)
        return self.mab1(X, H, y_mask=None)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, num_heads, dropout=dropout)

    def forward(self, X: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.size(0)
        S_expanded = self.S.expand(B, -1, -1)
        return self.mab(S_expanded, X, y_mask=x_mask)


class CausalMetaModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        lambda_crossing_penalty: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_lr_scheduler: bool = False,
        run_baselines: bool = False,
        baseline_setting: str = "confounder",
        confidence_level: float = 0.95,
        val_metrics_normalized: bool = True,
        plot_results: bool = False,
        config: dict = None,
        custom_baselines: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "config", "custom_baselines"])
        self.backbone = backbone
        self.config = config or {}
        self.lambda_crossing_penalty = lambda_crossing_penalty
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler

        self.gaussian_nll = None
        if self.estimator_type == "gaussian":
            self.gaussian_nll = nn.GaussianNLLLoss(eps=1e-6)

        self.run_baselines = run_baselines
        self.baseline_setting = baseline_setting
        self.val_metrics_normalized = val_metrics_normalized
        self.confidence_level = confidence_level
        self.plot_results = plot_results

        self.baselines = None
        if self.run_baselines:
            if custom_baselines is not None:
                self.baselines = custom_baselines
            else:
                self.baselines = get_baselines(baseline_setting=self.baseline_setting)

        self.validation_step_outputs: list[dict[str, Any]] = []
        self.test_step_outputs: list[dict[str, Any]] = []

        self.conformal_q = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "config" in checkpoint:
            self.config = checkpoint["config"]

    @property
    def estimator_type(self) -> str:
        return getattr(self.backbone, "estimator_type", None)

    def forward(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        query_x: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> Any:
        return self.backbone(x_features, x_sources, query_x, taus, lengths)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        lengths = batch.get("lengths", None)

        if self.estimator_type == "gaussian":
            if self.backbone.predict_ate:
                mean, var = self(batch["x_features"], batch["x_sources"], lengths=lengths)
                target = batch["query_value"].view(mean.size(0), -1).mean(dim=1)
            else:
                mean, var = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"]
            loss = self.gaussian_nll(mean, target, var)

        elif self.estimator_type == "point":
            if self.backbone.predict_ate:
                preds = self(batch["x_features"], batch["x_sources"], lengths=lengths)
                target = batch["query_value"].view(preds.size(0), -1).mean(dim=1)
            else:
                preds = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"]
            loss = F.huber_loss(preds, target, delta=1.0)

        elif self.estimator_type == "quantile":
            if self.backbone.predict_ate:
                preds, taus = self(batch["x_features"], batch["x_sources"], lengths=lengths)
                target_val = batch["query_value"].view(preds.size(0), -1).mean(dim=1, keepdim=True)
                target = target_val.expand_as(preds)
            else:
                preds, taus = self(
                    batch["x_features"],
                    batch["x_sources"],
                    batch["query_x"],
                    lengths=lengths,
                )
                target = batch["query_value"].unsqueeze(2).expand_as(preds)

            error = target - preds
            huber_loss = F.huber_loss(preds, target, reduction="none", delta=1.0)
            tau_reshaped = (
                taus.squeeze(-1)
                if self.backbone.predict_ate
                else taus.view(batch["x_features"].size(0), 1, self.backbone.num_tau_samples)
            )

            base_loss = (torch.abs(tau_reshaped - (error.detach() < 0).float()) * huber_loss).mean()

            dim_to_sort = 1 if self.backbone.predict_ate else 2
            _, sort_idx = torch.sort(taus.squeeze(-1), dim=1)
            sort_idx_expanded = (
                sort_idx
                if self.backbone.predict_ate
                else sort_idx.unsqueeze(1).expand(-1, preds.size(1), -1)
            )
            sorted_preds = torch.gather(preds, dim=dim_to_sort, index=sort_idx_expanded)

            diffs = (
                sorted_preds[:, 1:] - sorted_preds[:, :-1]
                if self.backbone.predict_ate
                else sorted_preds[:, :, 1:] - sorted_preds[:, :, :-1]
            )
            crossing_penalty = F.relu(-diffs).mean()

            loss = base_loss + (self.lambda_crossing_penalty * crossing_penalty)

            self.log("train/base_loss", base_loss, prog_bar=True)
            self.log("train/crossing_penalty", crossing_penalty, prog_bar=True)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _evaluate_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, stage: str
    ) -> dict[str, torch.Tensor]:
        lengths = batch.get("lengths", None)
        x_features, x_sources = batch["x_features"], batch["x_sources"]
        query_x = batch.get("query_x", None)

        if self.backbone.predict_ate:
            y_true = batch["query_value"].view(x_features.size(0), -1).mean(dim=1)
        else:
            y_true = batch["query_value"]

        if self.estimator_type == "point":
            if self.backbone.predict_ate:
                preds = self(x_features, x_sources, lengths=lengths)
            else:
                preds = self(x_features, x_sources, query_x, lengths=lengths)

            pred_median = preds
            pred_lower = preds
            pred_upper = preds

        elif self.estimator_type == "gaussian":
            if self.backbone.predict_ate:
                mean, var = self(x_features, x_sources, lengths=lengths)
            else:
                mean, var = self(x_features, x_sources, query_x, lengths=lengths)

            std = torch.sqrt(var)
            pred_median = mean
            z = sp.stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            pred_lower = mean - z * std
            pred_upper = mean + z * std

        elif self.estimator_type == "quantile":
            lower_tau = (1 - self.confidence_level) / 2
            upper_tau = 1 - lower_tau
            eval_taus = torch.tensor([lower_tau, 0.50, upper_tau], device=self.device).view(1, 3, 1)
            eval_taus = eval_taus.expand(x_features.size(0), -1, -1)

            if self.backbone.predict_ate:
                preds, _ = self(x_features, x_sources, taus=eval_taus, lengths=lengths)
                pred_lower, pred_median, pred_upper = (
                    preds[:, 0],
                    preds[:, 1],
                    preds[:, 2],
                )
            else:
                preds, _ = self(x_features, x_sources, query_x, taus=eval_taus, lengths=lengths)
                pred_lower, pred_median, pred_upper = (
                    preds[:, :, 0],
                    preds[:, :, 1],
                    preds[:, :, 2],
                )

        if self.conformal_q and self.estimator_type in [
            "quantile",
            "gaussian",
        ]:
            pred_lower = pred_lower - self.conformal_q
            pred_upper = pred_upper + self.conformal_q

            pred_lower = torch.min(pred_lower, pred_median)
            pred_upper = torch.max(pred_upper, pred_median)

        if batch.get("standardized", torch.tensor([False])).any():
            if stage.startswith("test") or (stage == "val" and not self.val_metrics_normalized):
                y_std = batch.get("y_std", torch.ones(x_features.size(0), 1, device=self.device))
                t_std = batch.get("t_std", torch.ones(x_features.size(0), 1, device=self.device))
                if self.backbone.predict_ate:
                    y_std = y_std.squeeze(-1)
                    t_std = t_std.squeeze(-1)

                inverse_scale = y_std / t_std
                y_true = y_true * inverse_scale
                pred_median = pred_median * inverse_scale
                pred_lower = pred_lower * inverse_scale
                pred_upper = pred_upper * inverse_scale

        return {
            "y_true": y_true.detach().cpu(),
            "pred_lower": pred_lower.detach().cpu(),
            "pred_median": pred_median.detach().cpu(),
            "pred_upper": pred_upper.detach().cpu(),
        }

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        out = self._evaluate_step(batch, batch_idx, "val")
        self.validation_step_outputs.append(out)
        return out

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        out = self._evaluate_step(batch, batch_idx, f"test_{dataloader_idx}")

        out_dict = {str(k): v for k, v in out.items()}
        out_dict["dataloader_idx"] = dataloader_idx

        if self.run_baselines and self.baselines:
            warnings.filterwarnings("ignore")

            required_keys = ["x_index", "t_index", "y_index"]
            if not all(k in batch for k in required_keys):
                raise KeyError(f"Missing indices for baselines: {required_keys}")

            x_features_np = batch["x_features"].cpu().numpy()
            x_sources_np = batch["x_sources"].cpu().numpy()
            batch_size = x_features_np.shape[0]

            def _parse_indices(idx_data):
                if (
                    isinstance(idx_data, (list, tuple))
                    and len(idx_data) == 1
                    and isinstance(idx_data[0], torch.Tensor)
                ):
                    return idx_data[0].cpu().tolist()

                elif isinstance(idx_data, torch.Tensor):
                    return idx_data.cpu().tolist()

                elif isinstance(idx_data, (list, tuple)):
                    if not isinstance(idx_data[0], int):
                        raise ValueError(f"Unsupported index type: {type(idx_data[0])}")
                    return list(idx_data)

                return [idx_data] * batch_size

            x_idx_list = _parse_indices(batch["x_index"])
            t_idx_list = _parse_indices(batch["t_index"])
            y_idx_list = _parse_indices(batch["y_index"])
            query_x_idx_list = _parse_indices(batch["query_x_index"])

            query_x_np = None
            if "query_x" in batch and batch["query_x"] is not None:
                query_x_np = batch["query_x"].cpu().numpy()
                if np.isnan(query_x_np).all():
                    query_x_np = None

            batch_y_baselines = {b["name"]: [] for b in self.baselines}

            for i in range(batch_size):
                xi = x_idx_list[i]
                ti = t_idx_list[i]
                yi = y_idx_list[i]
                qxi = query_x_idx_list[i] if query_x_np is not None else None

                x_i = x_features_np[i, ..., xi].T
                t_i = x_features_np[i, ..., ti].T
                y_i = x_features_np[i, ..., yi].T
                qx_i = query_x_np[i, ..., qxi].T if query_x_np is not None else None
                if self.backbone.predict_ate:
                    qx_i = None

                x_sources_i = x_sources_np[i]

                for b in self.baselines:
                    pred = b["model"](x=x_i, t=t_i, y=y_i, query_x=qx_i, x_sources=x_sources_i)
                    batch_y_baselines[b["name"]].append(pred)

            warnings.filterwarnings("default")

            for b in self.baselines:
                y_baseline = np.array(batch_y_baselines[b["name"]])
                if self.backbone.predict_ate:
                    y_baseline = y_baseline.reshape(batch_size)
                else:
                    y_baseline = y_baseline.reshape(batch_size, -1)

                if batch.get("standardized", torch.tensor([False])).any():
                    y_std_np = batch.get("y_std", torch.ones(batch_size, 1)).cpu().numpy()
                    t_std_np = batch.get("t_std", torch.ones(batch_size, 1)).cpu().numpy()
                    if self.backbone.predict_ate:
                        y_std_np = y_std_np.squeeze(-1)
                        t_std_np = t_std_np.squeeze(-1)
                    y_baseline = (y_baseline * y_std_np) / t_std_np

                out_dict[f"y_{b['name']}"] = y_baseline

        self.test_step_outputs.append(out_dict)
        return out_dict

    def _log_epoch_metrics_and_plots(
        self,
        outputs: list[dict[str, Any]],
        stage: str,
        dataloader_idx: int | None = None,
        with_plot: bool = False,
    ) -> None:
        if not outputs:
            return

        if dataloader_idx is not None:
            outputs = [o for o in outputs if o.get("dataloader_idx") == dataloader_idx]
            if not outputs:
                return
            stage_prefix = (
                f"{stage}/empirical"
                if dataloader_idx > 0
                else f"{stage}/simulated"
                if stage.startswith("test")
                else stage
            )
        else:
            stage_prefix = stage

        y_true = torch.cat([x["y_true"].flatten() for x in outputs]).numpy()
        y_lower = torch.cat([x["pred_lower"].flatten() for x in outputs]).numpy()
        y_median = torch.cat([x["pred_median"].flatten() for x in outputs]).numpy()
        y_upper = torch.cat([x["pred_upper"].flatten() for x in outputs]).numpy()

        mae = np.mean(np.abs(y_median - y_true))
        rmse = np.sqrt(np.mean((y_median - y_true) ** 2))
        r2 = r2_score(y_true, y_median)

        metrics = {
            f"{stage_prefix}/mae": mae,
            f"{stage_prefix}/rmse": rmse,
            f"{stage_prefix}/r2": r2,
        }

        if self.estimator_type in ["quantile", "gaussian"]:
            coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            mpiw = np.mean(y_upper - y_lower)
            inversion_rate = np.mean(y_lower > y_upper)

            metrics.update(
                {
                    f"{stage_prefix}/coverage_95": coverage,
                    f"{stage_prefix}/interval_width": mpiw,
                    f"{stage_prefix}/inversion_rate": inversion_rate,
                }
            )

        self.log_dict(metrics, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics_and_plots(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        dl_indices = set(x.get("dataloader_idx", 0) for x in self.test_step_outputs)
        for idx in dl_indices:
            self._log_epoch_metrics_and_plots(
                self.test_step_outputs, "test", dataloader_idx=idx, with_plot=True
            )

        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mae",
                    "frequency": 1,
                },
            }
        return optimizer

    @torch.no_grad()
    def calibrate(self, dataloader: Any, alpha: float | None = None) -> None:
        self.eval()
        device = self.device
        scores = []

        if alpha is None:
            alpha = 1 - self.confidence_level

        for batch in dataloader:
            lengths = batch.get("lengths", None)
            x_features = batch["x_features"].to(device)
            x_sources = batch["x_sources"].to(device)
            query_x = batch.get("query_x", None)
            if query_x is not None:
                query_x = query_x.to(device)

            if self.estimator_type == "gaussian":
                mean, var = self(x_features, x_sources, query_x, lengths=lengths)
                std = torch.sqrt(var)
                z_score = sp.stats.norm.ppf(1 - alpha / 2)
                pred_lower = mean - z_score * std
                pred_upper = mean + z_score * std

            elif self.estimator_type == "quantile":
                quantiles = [alpha / 2, 0.5, 1.0 - (alpha / 2)]
                eval_taus = (
                    torch.tensor(quantiles, device=device)
                    .view(1, 3, 1)
                    .expand(x_features.size(0), -1, -1)
                )
                preds, _ = self(x_features, x_sources, query_x, taus=eval_taus, lengths=lengths)

                dim_to_sort = 1 if self.backbone.predict_ate else 2
                preds, _ = torch.sort(preds, dim=dim_to_sort)
                pred_lower, pred_upper = preds[..., 0], preds[..., 2]
            else:
                raise ValueError("Conformal prediction requires an interval estimator")

            if self.backbone.predict_ate:
                y_true = batch["query_value"].view(x_features.size(0), -1).mean(dim=1).to(device)
            else:
                y_true = batch["query_value"].to(device)
                y_true = y_true.view_as(pred_lower)

            batch_scores = torch.max(pred_lower - y_true, y_true - pred_upper)
            scores.append(batch_scores.flatten())

        scores = torch.cat(scores).cpu().numpy()
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)

        self.conformal_q = np.quantile(scores, q_level)

    @torch.no_grad()
    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
        inverse_transform: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        lengths = batch.get("lengths", None)
        x_features, x_sources = batch["x_features"], batch["x_sources"]
        query_x = batch.get("query_x", None)
        B = x_features.size(0)

        if self.backbone.predict_ate:
            y_true = batch["query_value"].view(B, -1).mean(dim=1).cpu().numpy()
        else:
            y_true = batch["query_value"].view(-1).cpu().numpy()

        if self.estimator_type == "point":
            point_estimates = self(x_features, x_sources, query_x, lengths=lengths)
            if self.backbone.predict_ate:
                dummy_intervals = point_estimates.unsqueeze(1).expand(-1, 3)
            else:
                dummy_intervals = point_estimates.view(-1).unsqueeze(1).expand(-1, 3)

            if inverse_transform and batch.get("standardized", torch.tensor([False])).any():
                y_std = batch.get("y_std", torch.ones(B, 1, device=self.device))
                t_std = batch.get("t_std", torch.ones(B, 1, device=self.device))
                if self.backbone.predict_ate:
                    y_std = y_std.squeeze(-1)
                    t_std = t_std.squeeze(-1)

                scale_factor = y_std / t_std
                dummy_intervals = (
                    dummy_intervals * scale_factor.unsqueeze(-1)
                    if self.backbone.predict_ate
                    else dummy_intervals * scale_factor
                )
                y_true = y_true * scale_factor.cpu().numpy().squeeze()
            return {"y_pred": dummy_intervals.cpu().numpy().squeeze(), "y_true": y_true}

        elif self.estimator_type == "gaussian":
            if self.backbone.predict_ate:
                mean, var = self(x_features, x_sources, lengths=lengths)
            else:
                mean, var = self(x_features, x_sources, query_x, lengths=lengths)
            std = torch.sqrt(var)
            z_score = sp.stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            lower = mean - z_score * std
            upper = mean + z_score * std
            sorted_preds = torch.stack([lower, mean, upper], dim=-1)

        elif self.estimator_type == "quantile":
            lower_quantile = (1 - self.confidence_level) / 2
            upper_quantile = 1.0 - lower_quantile
            quantiles = [lower_quantile, 0.5, upper_quantile]
            eval_taus = torch.tensor(quantiles, device=self.device).view(1, 3, 1).expand(B, -1, -1)
            preds, taus = self(x_features, x_sources, query_x, taus=eval_taus, lengths=lengths)

            dim_to_sort = 1 if self.backbone.predict_ate else 2
            _, sort_idx = torch.sort(taus.squeeze(-1), dim=1)
            sort_idx_expanded = (
                sort_idx
                if self.backbone.predict_ate
                else sort_idx.unsqueeze(1).expand(-1, preds.size(1), -1)
            )
            sorted_preds = torch.gather(preds, dim=dim_to_sort, index=sort_idx_expanded)

        if self.conformal_q:
            sorted_preds[..., 0] -= self.conformal_q
            sorted_preds[..., 2] += self.conformal_q
            sorted_preds[..., 0] = torch.min(sorted_preds[..., 0], sorted_preds[..., 1])
            sorted_preds[..., 2] = torch.max(sorted_preds[..., 2], sorted_preds[..., 1])

        if inverse_transform and batch.get("standardized", torch.tensor([False])).any():
            y_std = batch.get("y_std", torch.ones(B, 1, device=self.device))
            t_std = batch.get("t_std", torch.ones(B, 1, device=self.device))
            transpose_dim = -1 if self.backbone.predict_ate else 1
            if self.backbone.predict_ate:
                y_std = y_std.squeeze(-1)
                t_std = t_std.squeeze(-1)

            scale_factor = y_std / t_std
            sorted_preds = sorted_preds * scale_factor.unsqueeze(transpose_dim)
            y_true = y_true * scale_factor.cpu().numpy().squeeze()

        if not self.backbone.predict_ate:
            sorted_preds = sorted_preds.view(-1, 3)

        return {"y_pred": sorted_preds.cpu().numpy().squeeze(), "y_true": y_true}


class QCNP(nn.Module):
    def __init__(
        self,
        predict_ate: bool,
        dim_context_features: int,
        dim_query_features: int,
        num_source_types: int = 4,
        source_embed_dim: int = 16,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_tau_samples: int = 8,
        dropout: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_tau_samples = num_tau_samples
        self.predict_ate = predict_ate
        self.estimator_type = "quantile"

        self.source_embedding = nn.Embedding(num_source_types, source_embed_dim)

        ctx_layers = [nn.Linear(dim_context_features + source_embed_dim, embed_dim)]
        for _ in range(num_layers - 1):
            if dropout > 0:
                ctx_layers.append(nn.Dropout(dropout))
            ctx_layers.append(nn.ReLU())
            ctx_layers.append(nn.Linear(embed_dim, embed_dim))
        self.context_encoder = nn.Sequential(*ctx_layers)

        self.quantile_embedding = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

        if self.predict_ate:
            ate_layers = [nn.Linear(embed_dim * 2, embed_dim if num_layers > 1 else output_dim)]
            for i in range(num_layers - 1):
                if dropout > 0:
                    ate_layers.append(nn.Dropout(dropout))
                ate_layers.append(nn.ReLU())
                ate_layers.append(
                    nn.Linear(embed_dim, embed_dim if i < num_layers - 2 else output_dim)
                )
            self.ate_decoder = nn.Sequential(*ate_layers)
        else:
            query_layers = [nn.Linear(dim_query_features, embed_dim)]
            for _ in range(num_layers - 1):
                if dropout > 0:
                    query_layers.append(nn.Dropout(dropout))
                query_layers.append(nn.ReLU())
                query_layers.append(nn.Linear(embed_dim, embed_dim))
            self.query_encoder = nn.Sequential(*query_layers)

            cate_layers = [nn.Linear(embed_dim * 3, embed_dim if num_layers > 1 else output_dim)]
            for i in range(num_layers - 1):
                if dropout > 0:
                    cate_layers.append(nn.Dropout(dropout))
                cate_layers.append(nn.ReLU())
                cate_layers.append(
                    nn.Linear(embed_dim, embed_dim if i < num_layers - 2 else output_dim)
                )
            self.cate_decoder = nn.Sequential(*cate_layers)

    def forward(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        query_x: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_features.size(0)
        device = x_features.device

        if taus is None:
            taus = torch.rand(B, self.num_tau_samples, 1, device=device)
        N_tau = taus.size(1)

        source_emb = self.source_embedding(x_sources)
        combined_context = torch.cat([x_features, source_emb], dim=-1)
        encoded_ctx = self.context_encoder(combined_context)

        if lengths is not None:
            max_len = encoded_ctx.size(1)
            valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            masked_ctx = encoded_ctx * valid_mask.unsqueeze(-1)
            r_agg = masked_ctx.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)
        else:
            r_agg = torch.mean(encoded_ctx, dim=1)

        tau_embed = self.quantile_embedding(taus)

        if self.predict_ate:
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, N_tau, -1)
            decoder_input = torch.cat([r_agg_exp, tau_embed], dim=-1)
            return self.ate_decoder(decoder_input).squeeze(-1), taus
        else:
            N_q = query_x.size(1)
            q_i = self.query_encoder(query_x)
            r_agg_exp = r_agg.view(B, 1, 1, -1).expand(-1, N_q, N_tau, -1)
            q_i_exp = q_i.unsqueeze(2).expand(-1, -1, N_tau, -1)
            tau_emb_exp = tau_embed.unsqueeze(1).expand(-1, N_q, -1, -1)

            decoder_input = torch.cat([r_agg_exp, q_i_exp, tau_emb_exp], dim=-1)
            return self.cate_decoder(decoder_input).squeeze(-1), taus


class QTNP(nn.Module):
    def __init__(
        self,
        predict_ate: bool,
        dim_context_features: int,
        dim_query_features: int,
        num_source_types: int = 4,
        source_embed_dim: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_inducing_points: int = 32,
        num_tau_samples: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tau_samples = num_tau_samples
        self.predict_ate = predict_ate
        self.estimator_type = "quantile"

        self.source_embedding = nn.Embedding(num_source_types, source_embed_dim)
        self.ctx_embedding = nn.Sequential(
            nn.Linear(dim_context_features + source_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.context_encoder = nn.ModuleList(
            [
                ISAB(
                    embed_dim,
                    embed_dim,
                    num_heads,
                    num_inducing_points,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pma = PMA(embed_dim, num_heads, num_seeds=1, dropout=dropout)
        self.quantile_embedding = CosineEmbeddingNetwork(embedding_dim=embed_dim)

        if self.predict_ate:
            self.ate_decoder = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.query_embedding = nn.Sequential(
                nn.Linear(dim_query_features, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cate_decoder = nn.Sequential(
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )

    def forward(
        self,
        x_features: torch.Tensor,
        x_sources: torch.Tensor,
        query_x: torch.Tensor | None = None,
        taus: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, max_len, _ = x_features.shape
        device = x_features.device

        padding_mask = None
        if lengths is not None:
            seq_range = torch.arange(max_len, device=device).unsqueeze(0)
            padding_mask = seq_range >= lengths.unsqueeze(1)

        source_emb = self.source_embedding(x_sources)
        combined_context = torch.cat([x_features, source_emb], dim=-1)
        ctx_mapped = self.ctx_embedding(combined_context)

        enc_out = ctx_mapped
        for layer in self.context_encoder:
            enc_out = layer(enc_out, x_mask=padding_mask)

        r_agg = self.pma(enc_out, x_mask=padding_mask).squeeze(1)

        if taus is None:
            taus = torch.rand(B, self.num_tau_samples, 1, device=device)
        N_tau = taus.size(1)
        tau_embed = self.quantile_embedding(taus)

        if self.predict_ate:
            r_agg_exp = r_agg.unsqueeze(1).expand(-1, N_tau, -1)
            decoder_input = torch.cat([r_agg_exp, tau_embed], dim=-1)
            preds = self.ate_decoder(decoder_input).squeeze(-1)
            return preds, taus
        else:
            N_q = query_x.size(1)
            queries = self.query_embedding(query_x)

            query_features, _ = self.cross_attention(
                query=queries,
                key=ctx_mapped,
                value=ctx_mapped,
                key_padding_mask=padding_mask,
            )

            r_agg_exp = r_agg.view(B, 1, 1, -1).expand(-1, N_q, N_tau, -1)
            q_exp = query_features.unsqueeze(2).expand(-1, -1, N_tau, -1)
            tau_exp = tau_embed.unsqueeze(1).expand(-1, N_q, -1, -1)

            query_orig_exp = queries.unsqueeze(2).expand(-1, -1, N_tau, -1)
            decoder_input = torch.cat([r_agg_exp, query_orig_exp, q_exp, tau_exp], dim=-1)
            preds = self.cate_decoder(decoder_input).squeeze(-1)
            return preds, taus


def build_backbone(
    config: dict[str, Any],
    backbone_type: str,
    predict_ate: bool,
    dim_context_features: int,
    dim_query_features: int,
    num_source_types: int,
) -> nn.Module:
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    num_tau_samples = config["num_tau_samples"]
    source_embed_dim = config["source_embed_dim"]
    output_dim = config["output_dim"]
    num_inducing_points = config["num_inducing_points"]
    dropout = config["dropout"]

    if backbone_type == "q-cnp":
        return QCNP(
            predict_ate=predict_ate,
            dim_context_features=dim_context_features,
            dim_query_features=dim_query_features,
            num_source_types=num_source_types,
            source_embed_dim=source_embed_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_tau_samples=num_tau_samples,
            dropout=dropout,
            output_dim=output_dim,
        )

    elif backbone_type == "q-tnp":
        return QTNP(
            predict_ate=predict_ate,
            dim_context_features=dim_context_features,
            dim_query_features=dim_query_features,
            num_source_types=num_source_types,
            source_embed_dim=source_embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_inducing_points=num_inducing_points,
            num_tau_samples=num_tau_samples,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


BACKBONE_REGISTRY = {
    "q-cnp": QCNP,
    "q-tnp": QTNP,
}
