"""
Logging and configuration utility functions.
"""
import os
import datetime
import transformers
import diffusers


def create_logging(logging_module, logger, accelerator):
    """Create logging configuration."""
    logging_module.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_module.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    """Set verbose logging for accelerate, transformers, and diffusers."""
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def create_output_folders(output_dir, config):
    """Create output folders for training."""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    # OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir
