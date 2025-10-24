# Third-party library imports
from train.utils.lora_utils import param_optim
from train.utils.optimization_utils import create_optimizer_params, get_optimizer
from diffusers.optimization import get_scheduler

# First-party imports
from train.utils.lora_handler import LoraHandler

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def setup_lora_components(
    unet,
    train_temporal_lora: bool,
    single_spatial_lora: bool,
    train_dataset,
    use_unet_lora: bool,
    lora_unet_dropout: float,
    lora_path: str,
    lora_rank: int,
    learning_rate: float,
    extra_unet_params: dict,
    adam_beta1: float,
    adam_beta2: float,
    adam_weight_decay: float,
    adam_epsilon: float,
    lr_scheduler: str,
    lr_warmup_steps: int,
    gradient_accumulation_steps: int,
    max_train_steps: int,
    use_8bit_adam: bool
):
    """
    Temporal LoRAとSpatial LoRAsの設定を行う

    Returns:
        tuple: (lora_manager_temporal, unet_lora_params_temporal, unet_negation_temporal,
                optimizer_temporal, lr_scheduler_temporal,
                lora_managers_spatial, unet_lora_params_spatial_list,
                optimizer_spatial_list, lr_scheduler_spatial_list,
                unet_negation_all)
    """

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Temporal LoRA
    if train_temporal_lora:
        # one temporal lora
        lora_manager_temporal = LoraHandler(
            use_unet_lora=use_unet_lora, unet_replace_modules=["TransformerTemporalModel"])

        unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
            use_unet_lora, unet, lora_manager_temporal.unet_replace_modules, lora_unet_dropout,
            lora_path + '/temporal/lora/', r=lora_rank)

        optimizer_temporal = optimizer_cls(
            create_optimizer_params([param_optim(unet_lora_params_temporal, use_unet_lora, is_lora=True,
                                                 extra_params={
                                                     **{"lr": learning_rate}, **extra_unet_params}
                                                 )], learning_rate),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        lr_scheduler_temporal = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_temporal,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
    else:
        lora_manager_temporal = None
        unet_lora_params_temporal, unet_negation_temporal = [], []
        optimizer_temporal = None
        lr_scheduler_temporal = None

    # Spatial LoRAs
    if single_spatial_lora:
        spatial_lora_num = 1
    else:
        # Check if this is a motion dataset
        if hasattr(train_dataset, '__getname__') and train_dataset.__getname__() == 'motions':
            # For motion dataset, one spatial lora for each verb
            spatial_lora_num = train_dataset.__len__()
        else:
            # one spatial lora for each video
            spatial_lora_num = train_dataset.__len__()

    lora_managers_spatial = []
    unet_lora_params_spatial_list = []
    optimizer_spatial_list = []
    lr_scheduler_spatial_list = []

    for _ in range(spatial_lora_num):
        lora_manager_spatial = LoraHandler(
            use_unet_lora=use_unet_lora, unet_replace_modules=["Transformer2DModel"])
        lora_managers_spatial.append(lora_manager_spatial)
        unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
            use_unet_lora, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
            lora_path + '/spatial/lora/', r=lora_rank)

        unet_lora_params_spatial_list.append(unet_lora_params_spatial)

        optimizer_spatial = optimizer_cls(
            create_optimizer_params([param_optim(unet_lora_params_spatial, use_unet_lora, is_lora=True,
                                                 extra_params={
                                                     **{"lr": learning_rate}, **extra_unet_params}
                                                 )], learning_rate),
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        optimizer_spatial_list.append(optimizer_spatial)

        # Scheduler
        lr_scheduler_spatial = get_scheduler(
            lr_scheduler,
            optimizer=optimizer_spatial,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        lr_scheduler_spatial_list.append(lr_scheduler_spatial)

    # Calculate unet_negation_all from the last spatial lora
    if len(lora_managers_spatial) > 0:
        unet_negation_all = unet_negation_spatial + unet_negation_temporal
    else:
        unet_negation_all = unet_negation_temporal

    return (
        lora_manager_temporal, unet_lora_params_temporal, unet_negation_temporal,
        optimizer_temporal, lr_scheduler_temporal,
        lora_managers_spatial, unet_lora_params_spatial_list,
        optimizer_spatial_list, lr_scheduler_spatial_list,
        unet_negation_all, spatial_lora_num
    )
