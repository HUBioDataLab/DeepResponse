"""Stratified split training strategy for cell, drug, and drug-cell splits."""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterable as IterableABC

import numpy as np
import torch
import torch.nn as nn

from src.evaluation import evaluate_test_metrics
from src.models import create_model
from src.repurposing.inference_engine import RepurposingInferenceEngine
from src.training.base_training_strategy import (
    BaseTrainingStrategy,
)


class StratifiedSplitTrainingStrategy(BaseTrainingStrategy):
    """Training strategy for stratified splits."""

    def train_and_evaluate_model(
        self,
        strategy_creator,
        dataset_iterator,
        comet_logger,
    ) -> list[dict[str, float | str]]:
        """Train and evaluate model across stratified folds."""
        logging.info("Run isolation ID: %s", self._run_id)
        self.prediction_manager = RepurposingInferenceEngine(
            data_source=str(strategy_creator.data_source),
            device=self.device,
        )

        all_fold_results: list[dict[str, float | str]] = []

        model = None
        initial_weights = None

        for fold_idx, data_fold in enumerate(self._iter_data_folds(dataset_iterator), start=1):
            (
                dims,
                _train_data,
                _valid_data,
                _test_data,
                _y_test_df,
                _fold_metadata,
            ) = self._unpack_fold_data(data_fold, fold_idx)
            _, cell_input_shape = dims

            if model is None:
                model = create_model(
                    cell_input_shape=cell_input_shape,
                    hidden_dim=strategy_creator.hidden_dim,
                    cell_embed_dim=strategy_creator.cell_embed_dim,
                    trainable_layers=strategy_creator.trainable_encoder_layers,
                    pooling=strategy_creator.encoder_pooling,
                    latent_dim=strategy_creator.latent_dim,
                    rank_dim=strategy_creator.rank_dim,
                    dropout=strategy_creator.dropout,
                    force_cell_blind=strategy_creator.force_cell_blind,
                    fusion_type=strategy_creator.fusion_type,
                    modality_dropout_drug=strategy_creator.modality_dropout_drug,
                    modality_dropout_cell=strategy_creator.modality_dropout_cell,
                    modality_dropout_schedule=strategy_creator.modality_dropout_schedule,
                    modality_dropout_final_scale=strategy_creator.modality_dropout_final_scale,
                    bounded_output=strategy_creator.bounded_output,
                    output_center=strategy_creator.bounded_output_center,
                    output_scale=strategy_creator.bounded_output_scale,
                    output_tau=strategy_creator.bounded_output_tau,
                    device=self.device,
                )
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                initial_weights = buf
            else:
                initial_weights.seek(0)
                model.load_state_dict(torch.load(initial_weights, map_location="cpu", weights_only=True))

            all_fold_results.append(
                self._train_and_evaluate_single_fold(
                    strategy_creator=strategy_creator,
                    data_fold=data_fold,
                    fold_idx=fold_idx,
                    comet_logger=comet_logger,
                    model=model,
                )
            )

        self._log_final_cv_results(all_fold_results, comet_logger)
        self._save_artifacts(strategy_creator, all_fold_results)
        return all_fold_results

    def _train_and_evaluate_single_fold(
        self,
        strategy_creator,
        data_fold,
        fold_idx: int,
        comet_logger,
        model: nn.Module | None = None,
    ) -> dict[str, float | str]:
        """Run one stratified fold end-to-end and return test metrics."""
        logging.info("%s", "=" * 60)
        logging.info("Starting CV Fold %d", fold_idx)
        logging.info("%s", "=" * 60)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (
            dims,
            train_data,
            valid_data,
            test_data,
            y_test_df,
            fold_metadata,
        ) = self._unpack_fold_data(data_fold, fold_idx)
        _, cell_input_shape = dims
        split_type = str(strategy_creator.split_type)
        ranking_weight = float(strategy_creator.ranking_weight)

        if model is None:
            model = create_model(
                cell_input_shape=cell_input_shape,
                hidden_dim=strategy_creator.hidden_dim,
                cell_embed_dim=strategy_creator.cell_embed_dim,
                trainable_layers=strategy_creator.trainable_encoder_layers,
                pooling=strategy_creator.encoder_pooling,
                latent_dim=strategy_creator.latent_dim,
                rank_dim=strategy_creator.rank_dim,
                dropout=strategy_creator.dropout,
                force_cell_blind=strategy_creator.force_cell_blind,
                fusion_type=strategy_creator.fusion_type,
                modality_dropout_drug=strategy_creator.modality_dropout_drug,
                modality_dropout_cell=strategy_creator.modality_dropout_cell,
                modality_dropout_schedule=strategy_creator.modality_dropout_schedule,
                modality_dropout_final_scale=strategy_creator.modality_dropout_final_scale,
                bounded_output=strategy_creator.bounded_output,
                output_center=strategy_creator.bounded_output_center,
                output_scale=strategy_creator.bounded_output_scale,
                output_tau=strategy_creator.bounded_output_tau,
                device=self.device,
            )

        train_loader = self._get_dataloader(train_data)
        val_loader = self._get_dataloader(valid_data)
        test_loader = self._get_dataloader(test_data)
        self._enable_cached_drug_embeddings(
            strategy_creator,
            model,
            train_loader,
            val_loader,
            test_loader,
        )
        optimizer = self._create_optimizer(
            model=model,
            lr=strategy_creator.learning_rate,
            weight_decay=strategy_creator.weight_decay,
        )
        scheduler, _scheduler_mode = self._create_scheduler(
            strategy_creator, optimizer, len(train_loader)
        )

        checkpoint_path = self._get_checkpoint_path(strategy_creator, fold_idx)
        train_summary = self._run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=strategy_creator.epochs,
            checkpoint_path=checkpoint_path,
            fold_idx=fold_idx,
            comet_logger=comet_logger,
            use_amp=bool(strategy_creator.use_amp),
            patience=int(strategy_creator.patience),
            checkpoint_metric=str(strategy_creator.checkpoint_metric),
            use_ranking=self._ranking_enabled(split_type, ranking_weight),
            ranking_weight=ranking_weight,
            ranking_group_mode=str(strategy_creator.ranking_group_mode),
            split_type=split_type,
            bounded_output=str(strategy_creator.bounded_output),
            bounded_output_mode=str(strategy_creator.bounded_output_mode),
            bounded_output_center=float(strategy_creator.bounded_output_center),
            bounded_output_scale=float(strategy_creator.bounded_output_scale),
            bounded_output_tau=float(strategy_creator.bounded_output_tau),
            bounded_output_std_factor=float(
                strategy_creator.bounded_output_std_factor
            ),
            bounded_output_min_scale=float(
                strategy_creator.bounded_output_min_scale
            ),
            unfreeze_epoch=int(strategy_creator.unfreeze_epoch),
            unfreeze_layers=int(strategy_creator.unfreeze_layers),
            unfreeze_lr_factor=float(strategy_creator.unfreeze_lr_factor),
            swa_start_pct=float(strategy_creator.swa_start_pct),
            swa_lr=float(strategy_creator.swa_lr),
        )

        checkpoint = self._load_best_checkpoint(model, checkpoint_path)
        if checkpoint is not None:
            logging.info("Loaded best model from epoch %s", checkpoint.get("epoch", "n/a"))

        y_pred = self._predict(model, test_loader)
        y_pred = self._apply_residual_target_inverse(y_pred, fold_metadata)
        if fold_metadata.get("residual_target"):
            logging.info(
                "Applied residual-target inverse transform (fold=%d).",
                fold_idx,
            )

        y_true = self._flatten_targets(y_test_df)
        test_metrics = evaluate_test_metrics(
            y_true,
            y_pred,
            comet_logger,
            split_type=split_type,
            trainable_encoder_layers=int(strategy_creator.trainable_encoder_layers),
            data_source=str(strategy_creator.data_source),
            fold_idx=fold_idx,
            output_dir=self._get_fold_results_dir(strategy_creator, fold_idx),
        )
        self._attach_best_summary_to_test_metrics(test_metrics, train_summary)

        try:
            self.prediction_manager.log_predictions(
                model=model,
                fold_idx=fold_idx,
                output_dir=self._get_fold_results_dir(strategy_creator, fold_idx),
            )
        except Exception as exc:
            logging.warning(
                "Repurposing prediction export failed for fold %d (%s).",
                fold_idx,
                exc,
            )

        return test_metrics

    @staticmethod
    def _unpack_fold_data(data_fold, fold_idx: int):
        """Validate and unpack stratified fold tuple."""
        if isinstance(data_fold, (tuple, list)) and len(data_fold) >= 5:
            dims, train_data, valid_data, test_data, y_test_df = data_fold[:5]
            metadata_raw = data_fold[5] if len(data_fold) > 5 else {}
            fold_metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            return dims, train_data, valid_data, test_data, y_test_df, fold_metadata
        raise ValueError(
            f"Unexpected data fold format at fold {fold_idx}: {type(data_fold)}"
        )

    @staticmethod
    def _iter_data_folds(dataset_input):
        """Yield stratified folds from dataset iterator."""
        if isinstance(dataset_input, tuple):
            yield dataset_input
            return

        if isinstance(dataset_input, IterableABC):
            yielded = False
            for fold in dataset_input:
                yielded = True
                yield fold
            if not yielded:
                raise ValueError("Dataset iterator yielded no folds.")
            return

        raise ValueError(f"Unexpected dataset_input format: {type(dataset_input)}")

    @staticmethod
    def _ranking_enabled(split_type: str, ranking_weight: float) -> bool:
        """Return whether ranking loss is active for this split."""
        return (
            split_type in {"cell_stratified", "drug_stratified", "drug_cell_stratified"}
            and ranking_weight > 0
        )

    def _load_best_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> dict | None:
        """Load best-checkpoint weights into model; raises if checkpoint is missing."""
        if not os.path.exists(checkpoint_path):
            logging.critical("Checkpoint file not found: %s", checkpoint_path)
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    @staticmethod
    def _apply_residual_target_inverse(
        predictions: np.ndarray,
        fold_metadata: dict,
    ) -> np.ndarray:
        """Inverse-transform residual target predictions when configured."""
        if not fold_metadata.get("residual_target"):
            return predictions
        target_mean = float(
            fold_metadata.get(
                "target_mean",
                fold_metadata.get("fallback_global_mean", 0.0),
            )
        )
        return predictions + target_mean

    @staticmethod
    def _flatten_targets(y_test_df) -> np.ndarray:
        """Convert test targets to a flattened numpy vector."""
        if hasattr(y_test_df, "values"):
            return y_test_df.values.flatten()
        return np.asarray(y_test_df).flatten()

    @staticmethod
    def _resolve_ranking_group_mode(split_type: str, ranking_group_mode: str) -> str:
        mode = str(ranking_group_mode or "auto").lower()
        default_mode = "drug" if split_type == "cell_stratified" else "cell"
        if mode == "auto":
            return default_mode
        if mode in {"cell", "drug"}:
            return mode
        return default_mode

    def _resolve_batch_group_ids(
        self,
        split_type: str,
        ranking_group_mode: str,
        smiles: list[str],
        group_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if ranking_group_mode == "drug":
            return self._group_ids_from_smiles(smiles, device=self.device)
        if ranking_group_mode == "cell":
            if group_ids is not None:
                return group_ids
            if split_type == "cell_stratified":
                return self._group_ids_from_smiles(smiles, device=self.device)
            return None
        return group_ids
