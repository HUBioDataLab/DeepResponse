"""Comet ML integration."""

from __future__ import annotations

import logging
import os
from typing import Any


def create_comet_experiment(use_comet: bool) -> Any | None:
    """Create a Comet experiment if enabled, otherwise return None."""
    if not use_comet:
        logging.info("Comet integration was skipped.")
        return None

    from comet_ml import Experiment
    from dotenv import load_dotenv

    load_dotenv("./dev.env")
    logging.info("Comet was integrated successfully.")
    return Experiment(
        api_key=os.environ.get("api_key"),
        project_name=os.environ.get("project_name"),
        workspace=os.environ.get("workspace"),
        auto_histogram_tensorboard_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
    )
