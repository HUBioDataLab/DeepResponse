"""Base training strategy with shared training infrastructure."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from config.constants import (
    ALLOWED_CHECKPOINT_METRICS,
    BINARY_THRESHOLD,
    CACHE_EMBEDDING_BATCH_MAX,
    CACHE_EMBEDDING_BATCH_MIN,
    COSINE_ETA_MIN_FLOOR,
    COSINE_ETA_MIN_SCALE,
    DIR_CHECKPOINTS,
    DIR_LOGS,
    EARLY_STOP_MIN_DELTA,
    GRAD_CLIP_NORM,
    SAMPLE_WEIGHT_EPS,
    WARM_RESTART_T_0,
    WARM_RESTART_T_MULT,
)
from config.defaults import DefaultConfig
from src.evaluation import compute_metrics


def _numpy_to_native(obj):
    """Convert numpy scalars to Python types for json.dump."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def pairwise_ranking_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 1.0,
    margin: float = 1e-6,
    max_ranking_samples: int = 128,
) -> tuple[torch.Tensor, int]:
    """Pairwise ranking loss within the same group."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    group_ids = group_ids.reshape(-1)
    n_items = y_true.shape[0]
    if n_items < 2:
        return torch.tensor(0.0, device=y_true.device), 0

    if n_items > max_ranking_samples:
        idx = torch.randperm(n_items, device=y_true.device)[:max_ranking_samples]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        group_ids = group_ids[idx]
        n_items = max_ranking_samples

    diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    sign_true = torch.sign(diff_true)

    same_group = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
    informative = torch.abs(diff_true) > margin

    row_idx = torch.arange(n_items, device=y_true.device).unsqueeze(1)
    col_idx = torch.arange(n_items, device=y_true.device).unsqueeze(0)
    upper_triangle = row_idx < col_idx

    valid_pairs = same_group & informative & upper_triangle
    n_valid = int(valid_pairs.sum().item())
    if n_valid == 0:
        return torch.tensor(0.0, device=y_true.device), 0

    rank_logits = -sign_true * (diff_pred / temperature)
    pair_losses = torch.nn.functional.softplus(rank_logits)
    masked_losses = pair_losses * valid_pairs.float()
    ranking = masked_losses.sum() / n_valid
    return ranking, n_valid


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(
        self,
        patience: int = DefaultConfig().patience,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Return True when training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            return False

        return (epoch - self.best_epoch) >= self.patience


@dataclass
class BestValidationState:
    """Best validation snapshot tracked during training."""

    val_loss: float
    val_pcc: float
    val_r2: float
    monitor_score: float


class BaseTrainingStrategy(ABC):
    """Abstract base class for training strategies."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = DIR_CHECKPOINTS
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        self._run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
        self._log_run_id = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_pid{os.getpid()}"
        self.prediction_manager = None

    @abstractmethod
    def train_and_evaluate_model(self, strategy_creator, dataset_input, comet_logger):
        """Train and evaluate the model."""

    @staticmethod
    @abstractmethod
    def _resolve_ranking_group_mode(split_type: str, ranking_group_mode: str) -> str:
        """Resolve the effective ranking group mode for this strategy."""

    @abstractmethod
    def _resolve_batch_group_ids(
        self,
        split_type: str,
        ranking_group_mode: str,
        smiles: list[str],
        group_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Resolve group IDs for a training batch."""

    @staticmethod
    def _resolve_checkpoint_metric(checkpoint_metric: str) -> str:
        """Resolve and validate checkpoint/early-stop monitor metric."""
        metric = str(checkpoint_metric or "auto").lower()
        if metric == "auto":
            return "val_loss"
        if metric in ALLOWED_CHECKPOINT_METRICS:
            return metric
        logging.warning(
            "Unknown checkpoint metric '%s'. Falling back to 'val_loss'. Allowed: %s",
            metric,
            sorted(ALLOWED_CHECKPOINT_METRICS),
        )
        return "val_loss"

    @staticmethod
    def _monitor_score_from_validation(
        monitor_metric: str,
        val_loss: float,
        val_metrics: dict[str, float],
    ) -> float:
        """Extract monitor score from validation outputs."""
        score = (
            float(val_metrics.get("r2", float("nan")))
            if monitor_metric == "val_r2"
            else float(val_loss)
        )
        if np.isfinite(score):
            return score
        fallback = -float("inf") if monitor_metric == "val_r2" else float("inf")
        logging.warning(
            "Validation monitor '%s' is non-finite (value=%s). Using fallback=%s.",
            monitor_metric,
            score,
            fallback,
        )
        return fallback

    @classmethod
    def _build_checkpoint_policy(
        cls,
        checkpoint_metric: str,
        patience: int,
        min_delta: float,
    ) -> tuple[str, str, EarlyStopping, BestValidationState]:
        """Create monitor policy, early stopper, and best-state container."""
        monitor_metric = cls._resolve_checkpoint_metric(checkpoint_metric)
        monitor_mode = "max" if monitor_metric == "val_r2" else "min"
        early_stopping = EarlyStopping(
            patience=patience,
            mode=monitor_mode,
            min_delta=min_delta,
        )
        best_state = BestValidationState(
            val_loss=float("inf"),
            val_pcc=-float("inf"),
            val_r2=-float("inf"),
            monitor_score=(-float("inf") if monitor_mode == "max" else float("inf")),
        )
        return monitor_metric, monitor_mode, early_stopping, best_state

    @staticmethod
    def _compute_train_epoch_metrics(
        train_preds_list: list[np.ndarray],
        train_targets_list: list[np.ndarray],
    ) -> tuple[dict[str, float], float]:
        """Compute train metrics and mse from collected batch outputs."""
        if not train_preds_list or not train_targets_list:
            raise ValueError("Training DataLoader yielded no batches.")

        train_preds_np = np.concatenate(train_preds_list, axis=0).flatten()
        train_targets_np = np.concatenate(train_targets_list, axis=0).flatten()
        train_metrics = compute_metrics(
            train_targets_np,
            train_preds_np,
            binary_threshold=BINARY_THRESHOLD,
        )
        train_mse = float(np.mean((train_targets_np - train_preds_np) ** 2))
        train_metrics["mse"] = train_mse
        return train_metrics, train_mse

    def _update_best_checkpoint(
        self,
        *,
        monitor_metric: str,
        monitor_mode: str,
        val_loss: float,
        val_metrics: dict[str, float],
        best_state: BestValidationState,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        checkpoint_path: str,
    ) -> tuple[float, bool]:
        """Update best-state and persist checkpoint when monitor improves."""
        val_pcc = float(val_metrics["pcc"])
        val_r2 = float(val_metrics["r2"])
        monitor_score = self._monitor_score_from_validation(
            monitor_metric=monitor_metric,
            val_loss=float(val_loss),
            val_metrics=val_metrics,
        )
        improved = (
            monitor_score > best_state.monitor_score
            if monitor_mode == "max"
            else monitor_score < best_state.monitor_score
        )
        if not improved:
            return monitor_score, False

        best_state.monitor_score = monitor_score
        best_state.val_loss = float(val_loss)
        best_state.val_pcc = val_pcc
        best_state.val_r2 = val_r2
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "val_loss": val_loss,
                "val_pcc": val_pcc,
                "val_r2": val_r2,
                "monitor_metric": monitor_metric,
                "monitor_score": float(monitor_score),
            },
            checkpoint_path,
        )
        return monitor_score, True

    @staticmethod
    def _log_best_model_saved(
        monitor_metric: str,
        best_state: BestValidationState,
    ) -> None:
        """Log best-model checkpoint event."""
        logging.info(
            "  ✓ Best model saved (%s=%.4f, Val Loss=%.4f, R2=%.4f, PCC=%.4f)",
            monitor_metric,
            best_state.monitor_score,
            best_state.val_loss,
            best_state.val_r2,
            best_state.val_pcc,
        )

    @staticmethod
    def _log_early_stopping(
        epoch: int,
        early_stopping: EarlyStopping,
        monitor_metric: str,
    ) -> None:
        """Log early stopping event."""
        logging.info(
            "Early stopping at epoch %d (best epoch=%d, best %s=%.4f)",
            epoch,
            early_stopping.best_epoch,
            monitor_metric,
            (
                float(early_stopping.best_score)
                if early_stopping.best_score is not None
                else float("nan")
            ),
        )

    @staticmethod
    def _best_state_to_summary(
        monitor_metric: str,
        best_state: BestValidationState,
    ) -> dict[str, float | str]:
        """Build train summary payload from best-state values."""
        return {
            "best_val_loss": float(best_state.val_loss),
            "best_val_pcc": float(best_state.val_pcc),
            "best_val_r2": float(best_state.val_r2),
            "best_monitor_metric": str(monitor_metric),
            "best_monitor_score": float(best_state.monitor_score),
        }

    @staticmethod
    def _attach_best_summary_to_test_metrics(
        test_metrics: dict,
        train_summary: dict[str, float | str],
    ) -> None:
        """Attach best-validation fields into test metrics."""
        keys = (
            "best_val_loss",
            "best_val_pcc",
            "best_val_r2",
            "best_monitor_metric",
            "best_monitor_score",
        )
        for key in keys:
            if key not in train_summary:
                continue
            value = train_summary[key]
            test_metrics[key] = str(value) if key == "best_monitor_metric" else float(value)

    def _resolve_sample_weight(self, batch) -> torch.Tensor | None:
        """Extract sample weights from batch."""
        sample_weight = batch.get("sample_weight")
        if sample_weight is None:
            return None
        return sample_weight.to(self.device)

    def _resolve_group_ids(self, batch) -> torch.Tensor | None:
        """Extract group ids from batch."""
        group_ids = batch.get("group_id")
        if group_ids is None:
            return None
        return group_ids.to(self.device).long()

    @staticmethod
    def _optimizer_step(
        *,
        loss: torch.Tensor,
        optimizer: optim.Optimizer,
        scaler: torch.amp.GradScaler,
        model: nn.Module,
        grad_clip_norm: float,
    ) -> bool:
        """Run backward + optimizer step."""
        if scaler.is_enabled():
            prev_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            return not (new_scale < prev_scale)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        return True

    @staticmethod
    def _sync_scheduler_lr_lists(scheduler, optimizer: optim.Optimizer) -> None:
        """Ensure scheduler lr lists match optimizer param-group count."""
        if scheduler is None:
            return

        n_groups = len(optimizer.param_groups)
        for attr in ("base_lrs", "max_lrs", "_last_lr"):
            values = getattr(scheduler, attr, None)
            if values is None:
                continue
            if not isinstance(values, list):
                try:
                    values = list(values)
                except TypeError:
                    continue
            if len(values) >= n_groups:
                setattr(scheduler, attr, values)
                continue
            for idx in range(len(values), n_groups):
                values.append(float(optimizer.param_groups[idx].get("lr", 0.0)))
            setattr(scheduler, attr, values)

    @staticmethod
    def _staged_unfreeze(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        unfreeze_epoch: int,
        unfreeze_layers: int,
        unfreeze_lr_factor: float,
        scheduler=None,
    ):
        """Optionally unfreeze encoder layers at a specified epoch."""
        if unfreeze_epoch < 0 or epoch != unfreeze_epoch or unfreeze_layers <= 0:
            return

        drug_encoder = getattr(model, "drug_encoder", None)
        if drug_encoder is None or not hasattr(drug_encoder, "_configure_trainable_layers"):
            logging.warning("Staged unfreeze requested but drug_encoder missing _configure_trainable_layers")
            return

        try:
            tracked_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
            drug_encoder._configure_trainable_layers(unfreeze_layers)

            new_params = [
                p for p in drug_encoder.parameters()
                if p.requires_grad and id(p) not in tracked_params
            ]
            if new_params:
                base_lr = optimizer.param_groups[0]["lr"]
                new_group = {"params": new_params, "lr": base_lr}
                for key in (
                    "initial_lr",
                    "max_lr",
                    "min_lr",
                    "max_momentum",
                    "base_momentum",
                    "swa_lr",
                ):
                    if key in optimizer.param_groups[0]:
                        new_group[key] = optimizer.param_groups[0][key]
                optimizer.add_param_group(new_group)
                BaseTrainingStrategy._sync_scheduler_lr_lists(scheduler, optimizer)
                logging.info("Added %d newly unfrozen params to optimizer", len(new_params))

            if unfreeze_lr_factor > 0 and unfreeze_lr_factor != 1.0:
                for group in optimizer.param_groups:
                    group["lr"] *= unfreeze_lr_factor
                if scheduler is not None:
                    if hasattr(scheduler, "base_lrs"):
                        scheduler.base_lrs = [lr * unfreeze_lr_factor for lr in scheduler.base_lrs]
                    if hasattr(scheduler, "max_lrs"):
                        scheduler.max_lrs = [lr * unfreeze_lr_factor for lr in scheduler.max_lrs]
                    if hasattr(scheduler, "_last_lr"):
                        scheduler._last_lr = [lr * unfreeze_lr_factor for lr in scheduler._last_lr]
            logging.info("Staged unfreeze applied at epoch %d: layers=%d, lr_factor=%.4f", epoch, unfreeze_layers, unfreeze_lr_factor)
        except Exception as exc:
            logging.warning("Failed to apply staged unfreeze at epoch %d: %s", epoch, exc)

    @staticmethod
    def _compute_weighted_huber(
        criterion_train: nn.Module,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute sample-weighted Huber loss."""
        loss_matrix = criterion_train(predictions, targets)
        if loss_matrix.ndim > 1:
            reduce_dims = tuple(range(1, loss_matrix.ndim))
            per_sample = loss_matrix.mean(dim=reduce_dims)
        else:
            per_sample = loss_matrix

        if sample_weight is None:
            return per_sample.mean()

        weights = sample_weight.reshape(-1).float()
        weights = weights / weights.mean().clamp(min=SAMPLE_WEIGHT_EPS)
        return (per_sample * weights).mean()

    @staticmethod
    def _extract_targets_from_train_loader(train_loader: DataLoader) -> np.ndarray:
        """Extract all target values from a training DataLoader."""
        dataset = getattr(train_loader, "dataset", None)
        targets = getattr(dataset, "targets", None)
        if targets is not None:
            return np.asarray(targets, dtype=np.float32).reshape(-1)

        target_chunks = []
        for batch in train_loader:
            target_chunks.append(batch["response"].detach().cpu().numpy().reshape(-1))
        if not target_chunks:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(target_chunks, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _group_ids_from_smiles(smiles: list[str], device: str | torch.device) -> torch.Tensor:
        """Map SMILES strings to integer group IDs."""
        ids: dict[str, int] = {}
        mapped = []
        for smi in smiles:
            if smi not in ids:
                ids[smi] = len(ids)
            mapped.append(ids[smi])
        return torch.tensor(mapped, dtype=torch.long, device=device)

    def _configure_bounded_output(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        bounded_output: str,
        bounded_output_mode: str,
        bounded_output_center: float,
        bounded_output_scale: float,
        bounded_output_tau: float,
        bounded_output_std_factor: float,
        bounded_output_min_scale: float,
    ) -> None:
        """Configure tanh output bounding from training statistics."""
        if bounded_output != "tanh":
            return
        if not hasattr(model, "set_output_bounds"):
            return

        center = bounded_output_center
        scale = bounded_output_scale
        tau = max(bounded_output_tau, 1e-6)

        if bounded_output_mode == "train_stats_fixed":
            targets = self._extract_targets_from_train_loader(train_loader)
            finite = np.isfinite(targets)
            if finite.any():
                valid_targets = targets[finite]
                center = float(valid_targets.mean())
                scale = max(
                    bounded_output_min_scale,
                    bounded_output_std_factor * float(valid_targets.std()),
                )
            else:
                center = 0.0
                scale = max(bounded_output_min_scale, 1.0)
        else:
            scale = max(scale, bounded_output_min_scale)

        model.set_output_bounds(center=center, scale=scale, tau=tau)
        logging.info("Bounded output configured: mode=%s center=%.4f scale=%.4f tau=%.4f", bounded_output_mode, center, scale, tau)

    def _create_optimizer(self, model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create AdamW optimizer, skipping frozen params."""
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "drug_encoder" in name or "cell_encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": lr * 0.1})
        param_groups.append({"params": head_params, "lr": lr})
        return optim.AdamW(param_groups, weight_decay=weight_decay)

    def _create_scheduler(self, strategy_creator, optimizer, steps_per_epoch: int):
        """Create learning rate scheduler."""
        eta_min = max(
            strategy_creator.learning_rate * COSINE_ETA_MIN_SCALE,
            COSINE_ETA_MIN_FLOOR,
        )

        if strategy_creator.trainable_encoder_layers == 0:
            logging.info("Scheduler selected: CosineAnnealingLR (epoch-step, stl=0)")
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=strategy_creator.epochs,
                eta_min=eta_min,
            )
            return scheduler, "epoch"

        t_0 = int(getattr(strategy_creator, "warm_restarts_t0", WARM_RESTART_T_0))
        t_mult = int(getattr(strategy_creator, "warm_restarts_t_mult", WARM_RESTART_T_MULT))
        logging.info(
            "Scheduler selected: CosineAnnealingWarmRestarts "
            "(epoch-step, T_0=%d, T_mult=%d, eta_min=%.2e)",
            t_0, t_mult, eta_min,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=eta_min,
        )
        return scheduler, "epoch"

    def _resolve_drug_embedding(self, batch, model: nn.Module) -> torch.Tensor:
        """Resolve drug embedding from cache or encoder."""
        cached = batch.get("drug_embedding")
        if cached is not None:
            return cached.to(self.device)
        return model.drug_encoder.encode_smiles(batch["smiles"]).to(self.device)

    def _forward_batch(self, model: nn.Module, batch) -> torch.Tensor:
        """Run a single model forward pass for a prepared batch."""
        cell_features = batch["cell_features"].to(self.device)
        drug_emb = self._resolve_drug_embedding(batch, model)
        return model(drug_emb, cell_features)

    @torch.no_grad()
    def _update_bn_for_swa(
        self,
        loader: DataLoader,
        swa_model: AveragedModel,
        base_model: nn.Module,
    ) -> None:
        """Recompute BatchNorm statistics for the SWA-averaged model."""
        momenta: dict[nn.Module, float | None] = {}
        for module in swa_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
                momenta[module] = module.momentum
                module.momentum = None

        if not momenta:
            return

        swa_model.train()
        for batch in loader:
            cell_features = batch["cell_features"].to(self.device)
            drug_emb = self._resolve_drug_embedding(batch, base_model)
            swa_model.module(drug_emb, cell_features)

        for module, momentum in momenta.items():
            module.momentum = momentum

    def _batch_targets(self, batch) -> torch.Tensor:
        """Extract response targets from a batch."""
        return batch["response"].to(self.device).reshape(-1, 1)

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Detach and move tensor to cpu numpy array."""
        return tensor.detach().float().cpu().numpy()

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, data_loader: DataLoader, criterion_eval: nn.Module):
        """Evaluate model on validation/test data."""
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0

        for batch in data_loader:
            targets = self._batch_targets(batch)
            predictions = self._forward_batch(model, batch)
            loss = criterion_eval(predictions, targets)

            total_loss += float(loss.item())
            n_batches += 1
            all_predictions.append(self._tensor_to_numpy(predictions))
            all_targets.append(self._tensor_to_numpy(targets))

        if not all_predictions:
            raise ValueError("Evaluation DataLoader yielded no batches.")

        all_predictions_np = np.concatenate(all_predictions, axis=0).flatten()
        all_targets_np = np.concatenate(all_targets, axis=0).flatten()
        avg_loss = total_loss / max(1, n_batches)
        metrics = compute_metrics(all_targets_np, all_predictions_np, binary_threshold=BINARY_THRESHOLD)
        metrics["loss"] = float(avg_loss)
        return float(avg_loss), metrics

    @torch.no_grad()
    def _predict(self, model: nn.Module, data_loader: DataLoader) -> np.ndarray:
        """Run inference and return predictions as numpy array."""
        model.eval()
        outputs = []
        for batch in data_loader:
            predictions = self._forward_batch(model, batch)
            outputs.append(self._tensor_to_numpy(predictions))
        if not outputs:
            raise ValueError("Prediction DataLoader yielded no batches.")
        return np.concatenate(outputs, axis=0).flatten()

    def _enable_cached_drug_embeddings(
        self,
        strategy_creator,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        """Pre-compute and cache drug embeddings when encoder is frozen."""
        has_staged_unfreeze = int(strategy_creator.unfreeze_epoch) >= 0 and int(strategy_creator.unfreeze_layers) > 0
        if int(strategy_creator.trainable_encoder_layers) != 0 or has_staged_unfreeze:
            return

        loaders = [train_loader, val_loader, test_loader]
        smiles_set: set[str] = set()
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            smiles_values = getattr(dataset, "smiles", None)
            if smiles_values is None:
                continue
            for value in smiles_values:
                smiles_set.add(value.decode("utf-8") if isinstance(value, bytes) else str(value))
        unique_smiles = sorted(smiles_set)
        if not unique_smiles:
            return

        cache_batch_size = max(
            CACHE_EMBEDDING_BATCH_MIN,
            min(CACHE_EMBEDDING_BATCH_MAX, int(strategy_creator.batch_size)),
        )
        logging.info("Frozen encoder cache enabled: precomputing %d unique drug embeddings (batch=%d).", len(unique_smiles), cache_batch_size)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            embeddings = model.encode_drugs(unique_smiles, batch_size=cache_batch_size)
        if was_training:
            model.train()

        cache = {
            smi: emb.detach().cpu().to(dtype=torch.float32)
            for smi, emb in zip(unique_smiles, embeddings)
        }
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "cached_drug_embeddings"):
                dataset.cached_drug_embeddings = cache
        logging.info("Frozen encoder cache ready: %d embeddings attached to dataloaders.", len(cache))

    @staticmethod
    def _get_dataloader(data) -> DataLoader:
        """Validate that data is a DataLoader."""
        if isinstance(data, DataLoader):
            return data
        raise ValueError(f"Expected DataLoader, got {type(data)}")

    def _run_prefix(self, strategy_creator) -> str:
        """Prefix used for checkpoint directories."""
        parts = [strategy_creator.data_source]
        if strategy_creator.evaluation_source:
            parts.append(f"to_{strategy_creator.evaluation_source}")
        parts.extend([strategy_creator.split_type, f"stl{strategy_creator.trainable_encoder_layers}"])
        parts.append(self._run_id)
        return "_".join(parts)

    def _log_prefix(self, strategy_creator) -> str:
        """Prefix used for result artifacts (matches log filename format)."""
        parts = [strategy_creator.data_source]
        if strategy_creator.evaluation_source:
            parts.append(f"to_{strategy_creator.evaluation_source}")
        parts.extend([strategy_creator.split_type, self._log_run_id])
        return "_".join(parts)

    def _get_run_dir(self, strategy_creator, create: bool = False) -> Path:
        """Resolve per-run output directory path."""
        run_dir = Path(self.checkpoint_dir) / self._run_prefix(strategy_creator)
        if create:
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _get_checkpoint_path(self, strategy_creator, fold_idx: int) -> str:
        """Generate checkpoint path for a fold."""
        return str(self._get_run_dir(strategy_creator, create=True) / f"fold{fold_idx}.pt")

    def _get_fold_results_dir(self, strategy_creator, fold_idx: int) -> Path:
        return self._get_run_dir(strategy_creator, create=False) / "results" / str(fold_idx)

    @staticmethod
    def _compute_cv_stats(all_fold_results: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
        """Compute per-metric mean and std across folds."""
        cv_mean: dict[str, float] = {}
        cv_std: dict[str, float] = {}
        for key in all_fold_results[0]:
            values = [
                float(result[key]) for result in all_fold_results
                if isinstance(result.get(key), (int, float, np.generic))
            ]
            if values:
                cv_mean[key] = float(np.mean(values))
                cv_std[key] = float(np.std(values))
        return cv_mean, cv_std

    def _save_artifacts(self, strategy_creator, all_fold_results: list[dict]):
        """Save results JSON."""
        if not all_fold_results:
            return

        log_dir = Path(DIR_LOGS)
        log_dir.mkdir(parents=True, exist_ok=True)
        cv_mean, cv_std = self._compute_cv_stats(all_fold_results)

        payload = {
            "args": vars(strategy_creator.args),
            "fold_count": len(all_fold_results),
            "fold_results": all_fold_results,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }
        if len(all_fold_results) == 1:
            payload["test_metrics"] = all_fold_results[0]

        results_path = log_dir / f"{self._log_prefix(strategy_creator)}_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True, default=_numpy_to_native)
        logging.info("Saved results artifact: %s", results_path)

    def _log_final_cv_results(self, all_fold_results: list[dict], comet_logger):
        """Log aggregated CV results."""
        if not all_fold_results:
            return

        cv_mean, cv_std = self._compute_cv_stats(all_fold_results)
        fold_count = len(all_fold_results)
        logging.info("%s", "=" * 60)
        logging.info("Cross-Validation Results Summary" if fold_count > 1 else "Single-Fold Results Summary")
        logging.info("%s", "=" * 60)

        for key in cv_mean:
            logging.info("%s: %.4f ± %.4f", key, cv_mean[key], cv_std[key])
            if comet_logger:
                comet_logger.log_metrics({f"cv_mean_{key}": cv_mean[key], f"cv_std_{key}": cv_std[key]})

    def _run_training_loop(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        n_epochs: int,
        checkpoint_path: str,
        fold_idx: int,
        comet_logger,
        use_amp: bool,
        patience: int,
        checkpoint_metric: str,
        bounded_output: str,
        bounded_output_mode: str,
        bounded_output_center: float,
        bounded_output_scale: float,
        bounded_output_tau: float,
        bounded_output_std_factor: float,
        bounded_output_min_scale: float,
        use_ranking: bool,
        ranking_weight: float,
        ranking_group_mode: str,
        split_type: str,
        unfreeze_epoch: int,
        unfreeze_layers: int,
        unfreeze_lr_factor: float,
        swa_start_pct: float = 0.0,
        swa_lr: float = 1e-5,
    ) -> dict[str, float | str]:
        """Shared training loop used by all training strategies."""
        criterion_train = nn.SmoothL1Loss(reduction="none")
        criterion_eval = nn.SmoothL1Loss()
        monitor_metric, monitor_mode, early_stopping, best_state = (
            self._build_checkpoint_policy(
                checkpoint_metric=checkpoint_metric,
                patience=patience,
                min_delta=EARLY_STOP_MIN_DELTA,
            )
        )
        effective_group_mode = self._resolve_ranking_group_mode(
            split_type=split_type,
            ranking_group_mode=ranking_group_mode,
        )
        logging.info(
            "Checkpoint/Early-stop policy: monitor=%s mode=%s",
            monitor_metric,
            monitor_mode,
        )

        self._configure_bounded_output(
            model=model,
            train_loader=train_loader,
            bounded_output=bounded_output,
            bounded_output_mode=bounded_output_mode,
            bounded_output_center=bounded_output_center,
            bounded_output_scale=bounded_output_scale,
            bounded_output_tau=bounded_output_tau,
            bounded_output_std_factor=bounded_output_std_factor,
            bounded_output_min_scale=bounded_output_min_scale,
        )

        if use_ranking:
            logging.info(
                "Ranking regularization: ENABLED (weight=%.4f, group_mode=%s, split=%s)",
                ranking_weight,
                effective_group_mode,
                split_type,
            )
        else:
            logging.info(
                "Ranking regularization: DISABLED (split=%s)",
                split_type,
            )

        amp_enabled = bool(use_amp and self.device == "cuda")
        amp_device = "cuda" if self.device == "cuda" else "cpu"
        scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)

        swa_enabled = 0.0 < swa_start_pct < 1.0
        swa_start_epoch = int(n_epochs * swa_start_pct) if swa_enabled else n_epochs + 1
        swa_model: AveragedModel | None = None
        swa_scheduler: SWALR | None = None
        swa_n_averaged = 0
        if swa_enabled:
            swa_model = AveragedModel(model, device=self.device)
            swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
            try:
                for group in optimizer.param_groups:
                    if "swa_lr" not in group:
                        raise KeyError(
                            f"Param group missing 'swa_lr' key before training starts. "
                            f"Group keys: {list(group.keys())}"
                        )
            except Exception as exc:
                raise RuntimeError(f"SWA setup validation failed: {exc}") from exc
            logging.info(
                "SWA enabled: averaging begins at epoch %d/%d (pct=%.2f, swa_lr=%.2e)",
                swa_start_epoch, n_epochs, swa_start_pct, swa_lr,
            )

        for epoch in range(1, n_epochs + 1):
            if hasattr(model, "set_training_progress"):
                model.set_training_progress(epoch=epoch, total_epochs=n_epochs)
            self._staged_unfreeze(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                unfreeze_epoch=unfreeze_epoch,
                unfreeze_layers=unfreeze_layers,
                unfreeze_lr_factor=unfreeze_lr_factor,
            )
            start_time = time.time()
            model.train()

            train_loss_total = 0.0
            huber_total = 0.0
            rank_total = 0.0
            n_batches = 0
            train_preds_list: list[np.ndarray] = []
            train_targets_list: list[np.ndarray] = []

            for batch in train_loader:
                targets = self._batch_targets(batch)
                smiles = batch["smiles"]
                sample_weight = self._resolve_sample_weight(batch)

                group_ids = self._resolve_group_ids(batch)
                rank_group_ids = self._resolve_batch_group_ids(
                    split_type=split_type,
                    ranking_group_mode=effective_group_mode,
                    smiles=smiles,
                    group_ids=group_ids,
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    predictions = self._forward_batch(model, batch)
                    huber_loss = self._compute_weighted_huber(
                        criterion_train,
                        predictions,
                        targets,
                        sample_weight,
                    )

                    rank_loss = torch.tensor(0.0, device=self.device)
                    if use_ranking and rank_group_ids is not None:
                        rank_loss, _ = pairwise_ranking_loss(
                            targets.squeeze(-1),
                            predictions.squeeze(-1),
                            rank_group_ids,
                        )

                    loss = huber_loss + (ranking_weight * rank_loss)

                self._optimizer_step(
                    loss=loss,
                    optimizer=optimizer,
                    scaler=scaler,
                    model=model,
                    grad_clip_norm=GRAD_CLIP_NORM,
                )

                train_loss_total += float(loss.item())
                huber_total += float(huber_loss.item())
                rank_total += float(rank_loss.item())
                n_batches += 1
                train_preds_list.append(self._tensor_to_numpy(predictions))
                train_targets_list.append(self._tensor_to_numpy(targets))

            train_loss = train_loss_total / max(1, n_batches)
            train_huber = huber_total / max(1, n_batches)
            train_rank = rank_total / max(1, n_batches)

            train_metrics, train_mse = self._compute_train_epoch_metrics(
                train_preds_list=train_preds_list,
                train_targets_list=train_targets_list,
            )

            val_loss, val_metrics = self._evaluate(model, val_loader, criterion_eval)

            if swa_enabled and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                for group in optimizer.param_groups:
                    if "swa_lr" not in group:
                        group["swa_lr"] = swa_lr
                swa_scheduler.step()
                swa_n_averaged += 1
            else:
                scheduler.step()

            elapsed = time.time() - start_time
            current_lr = max(group.get("lr", 0.0) for group in optimizer.param_groups)
            val_mse = float(val_metrics.get("mse", float("nan")))
            logging.info(
                "[Fold %d | Epoch %d/%d] Time: %.1fs | LR: %.2e | "
                "Train {Loss: %.4f, MSE: %.4f, R2: %.4f, PCC: %.4f} | "
                "Val {Loss: %.4f, MSE: %.4f, R2: %.4f, PCC: %.4f}",
                fold_idx,
                epoch,
                n_epochs,
                elapsed,
                current_lr,
                train_loss,
                train_mse,
                train_metrics["r2"],
                train_metrics["pcc"],
                val_loss,
                val_mse,
                val_metrics["r2"],
                val_metrics["pcc"],
            )

            if comet_logger:
                prefix = f"fold_{fold_idx}"
                metrics_payload = {
                    f"{prefix}/train_loss": train_loss,
                    f"{prefix}/train_mse": train_mse,
                    f"{prefix}/train_huber": train_huber,
                    f"{prefix}/train_r2": train_metrics["r2"],
                    f"{prefix}/train_pcc": train_metrics["pcc"],
                    f"{prefix}/val_loss": val_loss,
                    f"{prefix}/val_mse": val_mse,
                    f"{prefix}/val_r2": val_metrics["r2"],
                    f"{prefix}/val_pcc": val_metrics["pcc"],
                    f"{prefix}/lr": current_lr,
                }
                if use_ranking:
                    metrics_payload[f"{prefix}/train_rank"] = train_rank
                comet_logger.log_metrics(metrics_payload, epoch=epoch)

            monitor_score, improved = self._update_best_checkpoint(
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                val_loss=float(val_loss),
                val_metrics=val_metrics,
                best_state=best_state,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                checkpoint_path=checkpoint_path,
            )
            if improved:
                self._log_best_model_saved(monitor_metric, best_state)

            if early_stopping(monitor_score, epoch):
                self._log_early_stopping(epoch, early_stopping, monitor_metric)
                break

        if swa_enabled and swa_model is not None and swa_n_averaged > 0:
            logging.info(
                "Finalizing SWA: %d snapshots averaged, updating batch-norm statistics …",
                swa_n_averaged,
            )
            try:
                self._update_bn_for_swa(train_loader, swa_model, model)
            except Exception as exc:
                logging.warning("SWA BN update failed (non-fatal): %s", exc)

            model.load_state_dict(swa_model.module.state_dict())
            existing_ckpt: dict = {}
            if os.path.isfile(checkpoint_path):
                try:
                    existing_ckpt = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                except Exception as exc:
                    logging.warning("Could not load existing checkpoint for SWA re-save: %s", exc)
            existing_ckpt["model_state_dict"] = model.state_dict()
            existing_ckpt["swa"] = True
            torch.save(existing_ckpt, checkpoint_path)
            logging.info("SWA-averaged weights saved to checkpoint: %s", checkpoint_path)
        elif swa_enabled and swa_n_averaged == 0:
            logging.info(
                "SWA was enabled but training ended before epoch %d; "
                "best checkpoint kept unchanged.",
                swa_start_epoch,
            )

        return self._best_state_to_summary(monitor_metric, best_state)
