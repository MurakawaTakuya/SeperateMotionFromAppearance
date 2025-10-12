# Standard library imports
import argparse
import inspect
import math
import random
import logging
import os
import sys

# Third-party library imports
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, TextToVideoSDPipeline

# First-party imports
from utils.lora import extract_lora_child_module

# Local imports
from .utils.logging_utils import (
    create_logging, accelerate_set_verbose, create_output_folders
)
from .utils.validation_utils import (
    should_sample, export_to_video, save_pipe
)
from .utils.dataset_utils import get_train_dataset, extend_datasets
from .utils.training_utils import (
    handle_trainable_modules, handle_cache_latents
)
from .utils.model_utils import (
    load_primary_models, freeze_models, unet_and_text_g_c,
    handle_memory_attention, tensor_to_vae_latent, sample_noise,
    enforce_zero_terminal_snr, is_mixed_precision, cast_to_gpu_and_type
)
from .model.lora_setup import setup_lora_components
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility functions


logger = get_logger(__name__, log_level="INFO")


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        single_spatial_lora: bool = False,
        train_temporal_lora: bool = True,
        random_hflip_img: float = -1,
        extra_train_data: list = [],
        dataset_types: Tuple[str] = ('json'),
        validation_steps: int = 1000,
        trainable_modules: Tuple[str] = None,  # Eg: ("attn1", "attn2")
        extra_unet_params=None,
        train_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 5e-5,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        text_encoder_gradient_checkpointing: bool = False,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        resume_step: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        enable_torch_2_attn: bool = False,
        seed: Optional[int] = None,
        use_offset_noise: bool = False,
        rescale_schedule: bool = False,
        offset_noise_strength: float = 0.1,
        extend_dataset: bool = False,
        cache_latents: bool = False,
        cached_latent_dir=None,
        use_unet_lora: bool = False,
        unet_lora_modules: Tuple[str] = [],
        text_encoder_lora_modules: Tuple[str] = [],
        save_pretrained_model: bool = True,
        lora_rank: int = 16,
        lora_path: str = '',
        lora_unet_dropout: float = 0.1,
        logger_type: str = 'tensorboard',
        **kwargs
):
    *_, config = inspect.getargvalues(inspect.currentframe())
    # breakpoint()

    # TODO: ここからモデルクラスのinitに入れる
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # Handle the output folder creation
    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(
        pretrained_model_path)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(
        enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    # Get the training dataset based on types (json, single_video, image)
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    # If you have extra train data, you can add a list of however many you would like.
    # Eg: extra_train_data: [{: {dataset_types, train_data: {etc...}}}]
    try:
        if extra_train_data is not None and len(extra_train_data) > 0:
            for dataset in extra_train_data:
                d_t, t_d = dataset['dataset_types'], dataset['train_data']
                train_datasets += get_train_dataset(d_t, t_d, tokenizer)

    except Exception as e:
        print(f"Could not process extra train datasets due to an error : {e}")

    # Extend datasets that are less than the greatest one. This allows for more balanced training.
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    # Process one dataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]

    # Process many datasets
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    # Setup LoRA components
    (lora_manager_temporal, unet_lora_params_temporal, unet_negation_temporal,
     optimizer_temporal, lr_scheduler_temporal,
     lora_managers_spatial, unet_lora_params_spatial_list,
     optimizer_spatial_list, lr_scheduler_spatial_list,
     unet_negation_all, spatial_lora_num) = setup_lora_components(
        unet=unet,
        train_temporal_lora=train_temporal_lora,
        single_spatial_lora=single_spatial_lora,
        train_dataset=train_dataset,
        use_unet_lora=use_unet_lora,
        lora_unet_dropout=lora_unet_dropout,
        lora_path=lora_path,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        extra_unet_params=extra_text_encoder_params,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_weight_decay=adam_weight_decay,
        adam_epsilon=adam_epsilon,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_train_steps=max_train_steps,
        use_8bit_adam=use_8bit_adam
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False
    )
    # Latents caching
    cached_data_loader = handle_cache_latents(
        cache_latents,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        pretrained_model_path,
        validation_data.noise_prior,
        cached_latent_dir,
    )

    if cached_data_loader is not None:
        train_dataloader = cached_data_loader

    # Prepare everything with our `accelerator`.
    unet, optimizer_spatial_list, optimizer_temporal, train_dataloader, lr_scheduler_spatial_list, lr_scheduler_temporal, text_encoder = accelerator.prepare(
        unet,
        optimizer_spatial_list, optimizer_temporal,
        train_dataloader,
        lr_scheduler_spatial_list, lr_scheduler_temporal,
        text_encoder
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet,
        text_encoder,
        gradient_checkpointing,
        text_encoder_gradient_checkpointing
    )

    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # Fix noise schedules to predcit light and dark areas if available.
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(
            noise_scheduler.betas)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * \
        accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def finetune_unet(batch, step, mask_spatial_lora=False, mask_temporal_lora=False):
        nonlocal use_offset_noise
        nonlocal rescale_schedule

        # Unfreeze UNET Layers
        if global_step == 0:
            unet.train()
            handle_trainable_modules(
                unet,
                trainable_modules,
                is_enabled=True,
                negation=unet_negation_all
            )

        # Convert videos to latent space
        if not cache_latents:
            latents = tensor_to_vae_latent(batch["pixel_values"], vae)
        else:
            latents = batch["latents"]

        # Sample noise that we'll add to the latents
        use_offset_noise = use_offset_noise and not rescale_schedule
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        # Sample a random timestep for each video
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # *Potentially* Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if kwargs.get('eval_train', False):
            unet.eval()
            text_encoder.eval()

        # Encode text embeddings
        token_ids = batch['prompt_ids']
        encoder_hidden_states = text_encoder(token_ids)[0]
        detached_encoder_state = encoder_hidden_states.clone().detach()

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise

        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        encoder_hidden_states = detached_encoder_state

        if mask_spatial_lora:
            loras = extract_lora_child_module(
                unet, target_replace_module=["Transformer2DModel"])
            for lora_i in loras:
                lora_i.scale = 0.
            loss_spatial = None
        else:
            loras = extract_lora_child_module(
                unet, target_replace_module=["Transformer2DModel"])

            if spatial_lora_num == 1:
                for lora_i in loras:
                    lora_i.scale = 1.
            else:
                for lora_i in loras:
                    lora_i.scale = 0.

                for lora_idx in range(0, len(loras), spatial_lora_num):
                    loras[lora_idx + step].scale = 1.

            loras = extract_lora_child_module(unet, target_replace_module=[
                                              "TransformerTemporalModel"])
            if len(loras) > 0:
                for lora_i in loras:
                    lora_i.scale = 0.

            ran_idx = torch.randint(0, noisy_latents.shape[2], (1,)).item()

            if random.uniform(0, 1) < random_hflip_img:
                pixel_values_spatial = transforms.functional.hflip(
                    batch["pixel_values"][:, ran_idx, :, :, :]).unsqueeze(1)
                latents_spatial = tensor_to_vae_latent(
                    pixel_values_spatial, vae)
                noise_spatial = sample_noise(
                    latents_spatial, offset_noise_strength, use_offset_noise)
                noisy_latents_input = noise_scheduler.add_noise(
                    latents_spatial, noise_spatial, timesteps)
                target_spatial = noise_spatial
                model_pred_spatial = unet(noisy_latents_input, timesteps,
                                          encoder_hidden_states=encoder_hidden_states).sample
                loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                          target_spatial[:, :, 0, :, :].float(), reduction="mean")
            else:
                noisy_latents_input = noisy_latents[:, :, ran_idx, :, :]
                target_spatial = target[:, :, ran_idx, :, :]
                model_pred_spatial = unet(noisy_latents_input.unsqueeze(2), timesteps,
                                          encoder_hidden_states=encoder_hidden_states).sample
                loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                          target_spatial.float(), reduction="mean")

        if mask_temporal_lora:
            loras = extract_lora_child_module(unet, target_replace_module=[
                                              "TransformerTemporalModel"])
            for lora_i in loras:
                lora_i.scale = 0.
            loss_temporal = None
        else:
            loras = extract_lora_child_module(unet, target_replace_module=[
                                              "TransformerTemporalModel"])
            for lora_i in loras:
                lora_i.scale = 1.
            model_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states=encoder_hidden_states).sample
            loss_temporal = F.mse_loss(
                model_pred.float(), target.float(), reduction="mean")

            beta = 1
            alpha = (beta ** 2 + 1) ** 0.5
            ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()
            model_pred_decent = alpha * model_pred - beta * \
                model_pred[:, :, ran_idx, :, :].unsqueeze(2)
            target_decent = alpha * target - beta * \
                target[:, :, ran_idx, :, :].unsqueeze(2)
            loss_ad_temporal = F.mse_loss(
                model_pred_decent.float(), target_decent.float(), reduction="mean")
            loss_temporal = loss_temporal + loss_ad_temporal

        return loss_spatial, loss_temporal, latents, noise

    for epoch in range(first_epoch, num_train_epochs):
        train_loss_spatial = 0.0
        train_loss_temporal = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):

                text_prompt = batch['text_prompt'][0]

                for optimizer_spatial in optimizer_spatial_list:
                    optimizer_spatial.zero_grad(set_to_none=True)

                if optimizer_temporal is not None:
                    optimizer_temporal.zero_grad(set_to_none=True)

                if train_temporal_lora:
                    mask_temporal_lora = False
                else:
                    mask_temporal_lora = True
                mask_spatial_lora = random.uniform(
                    0, 1) < 0.2 and not mask_temporal_lora

                with accelerator.autocast():
                    loss_spatial, loss_temporal, latents, init_noise = finetune_unet(
                        batch, step, mask_spatial_lora=mask_spatial_lora, mask_temporal_lora=mask_temporal_lora)

                # Gather the losses across all processes for logging (if we use distributed training).
                if not mask_spatial_lora:
                    avg_loss_spatial = accelerator.gather(
                        loss_spatial.repeat(train_batch_size)).mean()
                    train_loss_spatial += avg_loss_spatial.item() / gradient_accumulation_steps

                if not mask_temporal_lora and train_temporal_lora:
                    avg_loss_temporal = accelerator.gather(
                        loss_temporal.repeat(train_batch_size)).mean()
                    train_loss_temporal += avg_loss_temporal.item() / gradient_accumulation_steps

                # Backpropagate
                if not mask_spatial_lora:
                    accelerator.backward(loss_spatial, retain_graph=True)
                    if spatial_lora_num == 1:
                        optimizer_spatial_list[0].step()
                    else:
                        optimizer_spatial_list[step].step()

                if not mask_temporal_lora and train_temporal_lora:
                    accelerator.backward(loss_temporal)
                    optimizer_temporal.step()

                if spatial_lora_num == 1:
                    lr_scheduler_spatial_list[0].step()
                else:
                    lr_scheduler_spatial_list[step].step()
                if lr_scheduler_temporal is not None:
                    lr_scheduler_temporal.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss": train_loss_temporal}, step=global_step)
                train_loss_temporal = 0.0
                if global_step % checkpointing_steps == 0 and global_step > 0:
                    save_pipe(
                        pretrained_model_path,
                        global_step,
                        accelerator,
                        unet,
                        text_encoder,
                        vae,
                        output_dir,
                        lora_managers_spatial[0] if lora_managers_spatial else None,
                        lora_manager_temporal,
                        unet_lora_modules,
                        text_encoder_lora_modules,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )

                if should_sample(global_step, validation_steps, validation_data):
                    if accelerator.is_main_process:
                        # Set seed for validation if specified
                        if hasattr(validation_data, 'seed') and validation_data.seed is not None:
                            torch.manual_seed(validation_data.seed)
                            logger.info(f"Using validation seed: {validation_data.seed}")

                        with accelerator.autocast():
                            unet.eval()
                            text_encoder.eval()
                            unet_and_text_g_c(unet, text_encoder, False, False)
                            loras = extract_lora_child_module(
                                unet, target_replace_module=["Transformer2DModel"])
                            for lora_i in loras:
                                lora_i.scale = validation_data.spatial_scale

                            if validation_data.noise_prior > 0:
                                preset_noise = (validation_data.noise_prior) ** 0.5 * batch['inversion_noise'] + (
                                    1 - validation_data.noise_prior) ** 0.5 * torch.randn_like(batch['inversion_noise'])
                            else:
                                preset_noise = None

                            pipeline = TextToVideoSDPipeline.from_pretrained(
                                pretrained_model_path,
                                text_encoder=text_encoder,
                                vae=vae,
                                unet=unet
                            )

                            diffusion_scheduler = DDIMScheduler.from_config(
                                pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            prompt_list = text_prompt if len(
                                validation_data.prompt) <= 0 else validation_data.prompt
                            for prompt in prompt_list:
                                save_filename = f"{global_step}_{prompt.replace('.', '')}"

                                out_file = f"{output_dir}/samples/{save_filename}.mp4"

                                with torch.no_grad():
                                    video_frames = pipeline(
                                        prompt,
                                        width=validation_data.width,
                                        height=validation_data.height,
                                        num_frames=validation_data.num_frames,
                                        num_inference_steps=validation_data.num_inference_steps,
                                        guidance_scale=validation_data.guidance_scale,
                                        latents=preset_noise
                                    ).frames
                                export_to_video(
                                    video_frames, out_file, train_data.get('fps', 8))
                                logger.info(
                                    f"Saved a new sample to {out_file}")
                            del pipeline
                            torch.cuda.empty_cache()

                    unet_and_text_g_c(
                        unet,
                        text_encoder,
                        gradient_checkpointing,
                        text_encoder_gradient_checkpointing
                    )

            if loss_temporal is not None:
                accelerator.log(
                    {"loss_temporal": loss_temporal.detach().item()}, step=step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
            pretrained_model_path,
            global_step,
            accelerator,
            unet,
            text_encoder,
            vae,
            output_dir,
            lora_managers_spatial[0] if lora_managers_spatial else None,
            lora_manager_temporal,
            unet_lora_modules,
            text_encoder_lora_modules,
            is_checkpoint=False,
            save_pretrained_model=save_pretrained_model
        )
    accelerator.end_training()


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default='./configs/config_multi_videos.yaml')
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))


if __name__ == "__main__":
    main_cli()
