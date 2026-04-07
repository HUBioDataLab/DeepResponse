"""Strategy resolution and instantiation for the DeepResponse pipeline."""

from __future__ import annotations

import logging

from config.constants import PROJECT_ROOT
from config.defaults import DefaultConfig
from src.comet.initialize import create_comet_experiment
from src.dataset.cell_stratified_dataset_strategy import CellStratifiedDatasetStrategy
from src.dataset.cross_domain_dataset_strategy import CrossDomainDatasetStrategy
from src.dataset.drug_cell_stratified_dataset_strategy import DrugCellStratifiedDatasetStrategy
from src.dataset.drug_stratified_dataset_strategy import DrugStratifiedDatasetStrategy
from src.dataset.random_split_dataset_strategy import RandomSplitDatasetStrategy
from src.training import RandomSplitTrainingStrategy, StratifiedSplitTrainingStrategy

_DEFAULTS = DefaultConfig()


class StrategyResolver:
    """Resolve and instantiate strategies from CLI arguments."""

    def __init__(self, args) -> None:
        self.args = args

    def __getattr__(self, name: str):
        """Fall back to *args* then *DefaultConfig* for any attribute."""
        if name == "args":
            raise AttributeError(name)
        args = object.__getattribute__(self, "args")
        try:
            return getattr(args, name)
        except AttributeError:
            pass
        try:
            return getattr(_DEFAULTS, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}' "
                f"(not in args or DefaultConfig)"
            ) from None

    def get_effective_learning_rate(self):
        """Return learning rate string, showing the post-unfreeze rate if applicable."""
        lr = self.learning_rate
        if (
            self.unfreeze_epoch >= 0
            and self.unfreeze_layers > 0
            and self.unfreeze_lr_factor not in (None, 1.0)
        ):
            return f"{lr} -> {lr * self.unfreeze_lr_factor}"
        return lr

    def should_use_ranking_regularization(self) -> bool:
        """Return True if ranking regularization should be applied for this run."""
        return (
            self.split_type in {"cell_stratified", "drug_stratified", "drug_cell_stratified"}
            and float(self.ranking_weight) > 0
        )

    def get_comet_strategy(self):
        """Return a Comet ML experiment or None."""
        return create_comet_experiment(self.use_comet)

    def get_dataset_path(self) -> str:
        """Return the primary dataset CSV path for the configured data source."""
        return str(
            PROJECT_ROOT / "dataset_creator" / self.data_source / "processed" / "drug_response_features.csv"
        )

    def get_evaluation_dataset_path(self) -> str:
        """Return the evaluation dataset CSV path for cross-domain runs."""
        if self.evaluation_source is None:
            raise ValueError("evaluation_source must be provided for cross_domain split type.")
        return str(
            PROJECT_ROOT / "dataset_creator" / self.evaluation_source / "processed" / "drug_response_features.csv"
        )

    def _validate_config(self) -> None:
        """Warn about misconfigured options that would have no effect."""
        if (
            str(getattr(self, "modality_dropout_schedule", "")) == "warmup_decay"
            and float(getattr(self, "modality_dropout_drug", 0.0)) == 0.0
            and float(getattr(self, "modality_dropout_cell", 0.0)) == 0.0
        ):
            logging.warning(
                "modality_dropout_schedule is 'warmup_decay' but both "
                "modality_dropout_drug and modality_dropout_cell are 0.0 — "
                "the schedule has no effect."
            )

    def get_split_strategy(self) -> dict:
        """Instantiate and return the dataset and training strategy pair."""
        self._validate_config()
        split_type = self.split_type
        dataset_path = self.get_dataset_path()

        dataset_kwargs = dict(
            n_splits=self.n_splits,
            hard_validation=self.hard_validation,
            ood_weighting=self.ood_weighting,
            residual_target=self.residual_target,
            omics_mask=getattr(self, "omics_mask", "1,1,1,1"),
        )

        if split_type == "random":
            return {
                "dataset": RandomSplitDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": RandomSplitTrainingStrategy(),
            }
        if split_type == "cell_stratified":
            return {
                "dataset": CellStratifiedDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": StratifiedSplitTrainingStrategy(),
            }
        if split_type == "drug_stratified":
            return {
                "dataset": DrugStratifiedDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": StratifiedSplitTrainingStrategy(),
            }
        if split_type == "drug_cell_stratified":
            return {
                "dataset": DrugCellStratifiedDatasetStrategy(dataset_path, **dataset_kwargs),
                "training": StratifiedSplitTrainingStrategy(),
            }
        if split_type == "cross_domain":
            return {
                "dataset": CrossDomainDatasetStrategy(
                    dataset_path,
                    self.get_evaluation_dataset_path(),
                    **dataset_kwargs,
                ),
                "training": RandomSplitTrainingStrategy(),
            }
        raise ValueError(f"Unknown split_type: {split_type!r}")
