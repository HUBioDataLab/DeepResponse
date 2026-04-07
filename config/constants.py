"""Project-wide constants for options, splitting, training, and evaluation."""

from pathlib import Path
from typing import Set

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_SOURCES: Set[str] = {"depmap", "gdsc", "ccle"}
SPLIT_TYPES: Set[str] = {
    "random",
    "cell_stratified",
    "drug_stratified",
    "drug_cell_stratified",
    "cross_domain",
}
ENCODER_POOLING_CHOICES: Set[str] = {
    "mean",
    "cls",
    "max",
}
FUSION_TYPES: Set[str] = {
    "concat",
    "film_bilinear",
    "xattn_dcn_residual",
}
BOUNDED_OUTPUT_CHOICES: Set[str] = {
    "none",
    "tanh",
}
BOUNDED_OUTPUT_MODES: Set[str] = {
    "train_stats_fixed",
    "global",
}
MODALITY_DROPOUT_SCHEDULES: Set[str] = {
    "constant",
    "warmup_decay",
}
RANKING_GROUP_MODES: Set[str] = {
    "auto",
    "cell",
    "drug",
}
ALLOWED_CHECKPOINT_METRICS: Set[str] = {"auto", "val_loss", "val_r2"}

DEFAULT_BOUNDED_OUTPUT_MODE: str = "train_stats_fixed"
DEFAULT_BOUNDED_OUTPUT_CENTER: float = 0.0
DEFAULT_BOUNDED_OUTPUT_SCALE: float = 10.0
DEFAULT_BOUNDED_OUTPUT_STD_FACTOR: float = 3.0
DEFAULT_BOUNDED_OUTPUT_MIN_SCALE: float = 1.0
DEFAULT_LAYERWISE_LR_DECAY: float = 1.0
DEFAULT_LAYERWISE_LR_MIN_SCALE: float = 1.0

BINARY_THRESHOLD: float = 6.0
TEST_SPLIT_RATIO: float = 0.10
VALIDATION_SPLIT_RATIO: float = 0.10
DEFAULT_NUM_WORKERS: int = 4

GRAD_CLIP_NORM: float = 1.0
EARLY_STOP_MIN_DELTA: float = 1e-4
SAMPLE_WEIGHT_EPS: float = 1e-8
COSINE_ETA_MIN_SCALE: float = 0.01
COSINE_ETA_MIN_FLOOR: float = 1e-7
WARM_RESTART_T_0: int = 10
WARM_RESTART_T_MULT: int = 2
CACHE_EMBEDDING_BATCH_MIN: int = 16
CACHE_EMBEDDING_BATCH_MAX: int = 256

DIR_LOGS: str = "logs"
DIR_CHECKPOINTS: str = "checkpoints"
DIR_PRETRAINED_SELFORMER: Path = PROJECT_ROOT / "pretrained" / "selformer"
