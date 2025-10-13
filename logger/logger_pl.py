from typing import Tuple
from datetime import datetime

from comet_ml import Experiment
from lightning.pytorch.loggers import CometLogger


def configure_logger_pl(
        model_name: str,
        disable_logging: bool,
        save_dir: str,
) -> Tuple[Experiment, str]:
    """comet logger factory

    Args:
        model_name (str): modelname to be added as a tag of comet experiment
        disable_logging (bool): disable comet Experiment object
        save_dir (str): dir to save comet log

    Returns:
        comet_ml.Experiment: logger
        str: experiment name of comet.ml
    """

    exp_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    exp_name = exp_name.replace(" ", "_")

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    comet_logger = CometLogger(
        save_dir=save_dir,
        experiment_name=exp_name,
        parse_args=True,
        disabled=disable_logging,
    )
    comet_logger.experiment.add_tag(model_name)

    return comet_logger, exp_name
