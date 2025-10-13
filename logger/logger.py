from datetime import datetime
from comet_ml import Experiment


def configure_logger(
    logged_params: dict,
    model_name: str,
    disable_logging: bool,
) -> Experiment:
    """comet logger factory

    Args:
        logged_params (dict): hyperparameters to be logged in comet.
            Typically "vars(args)" for logging all parameters in "args".
        model_name (str): modelname to be added as a tag of comet experiment
        disable_logging (bool): disable comet Experiment object

    Returns:
        comet_ml.Experiment: logger
    """

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    experiment_logger = Experiment(disabled=disable_logging)

    exp_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    experiment_logger.set_name(exp_name)
    experiment_logger.add_tag(model_name)

    experiment_logger.log_parameters(logged_params)

    return experiment_logger
