# Standard library imports
import inspect
import math
import random
import logging
import os
import sys
from typing import Dict, Optional, Tuple

# Third-party library imports
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, TextToVideoSDPipeline

# First-party imports
from utils.lora import extract_lora_child_module

# Local imports
from train.utils.logging_utils import (
    create_logging, accelerate_set_verbose, create_output_folders
)
from train.utils.validation_utils import (
    should_sample, export_to_video, save_pipe
)
from train.utils.dataset_utils import get_train_dataset, extend_datasets
from train.utils.training_utils import (
    handle_cache_latents
)
from train.utils.model_utils import (
    load_primary_models, freeze_models, unet_and_text_g_c,
    handle_memory_attention,
    enforce_zero_terminal_snr, is_mixed_precision, cast_to_gpu_and_type
)
from train.model.lora_setup import setup_lora_components
from train.model.lora_training import train_lora_unet

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

logger = get_logger(__name__, log_level="INFO")


class TextToVideoModel:
    """Text-to-Video fine-tuning model class"""

    def __init__(
        self,
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
        trainable_modules: Tuple[str] = None,
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
        # Store all parameters as instance variables
        *_, config = inspect.getargvalues(inspect.currentframe())
        self.config = config
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.train_data = train_data
        self.validation_data = validation_data
        self.single_spatial_lora = single_spatial_lora
        self.train_temporal_lora = train_temporal_lora
        self.random_hflip_img = random_hflip_img
        self.extra_train_data = extra_train_data
        self.dataset_types = dataset_types
        self.validation_steps = validation_steps
        self.trainable_modules = trainable_modules
        self.extra_unet_params = extra_unet_params
        self.train_batch_size = train_batch_size
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.text_encoder_gradient_checkpointing = text_encoder_gradient_checkpointing
        self.checkpointing_steps = checkpointing_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_step = resume_step
        self.mixed_precision = mixed_precision
        self.use_8bit_adam = use_8bit_adam
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
        self.enable_torch_2_attn = enable_torch_2_attn
        self.seed = seed
        self.use_offset_noise = use_offset_noise
        self.rescale_schedule = rescale_schedule
        self.offset_noise_strength = offset_noise_strength
        self.extend_dataset = extend_dataset
        self.cache_latents = cache_latents
        self.cached_latent_dir = cached_latent_dir
        self.use_unet_lora = use_unet_lora
        self.unet_lora_modules = unet_lora_modules
        self.text_encoder_lora_modules = text_encoder_lora_modules
        self.save_pretrained_model = save_pretrained_model
        self.lora_rank = lora_rank
        self.lora_path = lora_path
        self.lora_unet_dropout = lora_unet_dropout
        self.logger_type = logger_type
        self.kwargs = kwargs

        # Initialize components
        self._setup_accelerator()
        self._setup_logging()
        self._setup_output_folders()
        self._load_models()
        self._setup_training_components()
        self._setup_data_loader()
        self._prepare_models()
        self._setup_scheduler()
        self._calculate_training_steps()
        self._init_trackers()

    def _setup_accelerator(self):
        """Setup accelerator"""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=self.logger_type,
            project_dir=self.output_dir
        )

    def _setup_logging(self):
        """Setup logging"""
        create_logging(logging, logger, self.accelerator)
        accelerate_set_verbose(self.accelerator)

    def _setup_output_folders(self):
        """Setup output folders"""
        if self.accelerator.is_main_process:
            self.output_dir = create_output_folders(self.output_dir, self.config)

    def _load_models(self):
        """Load primary models"""
        self.noise_scheduler, self.tokenizer, self.text_encoder, self.vae, self.unet = load_primary_models(
            self.pretrained_model_path)

        # Freeze models
        freeze_models([self.vae, self.text_encoder, self.unet])

        # Handle memory attention
        handle_memory_attention(
            self.enable_xformers_memory_efficient_attention,
            self.enable_torch_2_attn,
            self.unet
        )

    def _setup_training_components(self):
        """Setup training datasets and LoRA components"""
        # Get training datasets
        self.train_datasets = get_train_dataset(self.dataset_types, self.train_data, self.tokenizer)

        # Process extra training data
        try:
            if self.extra_train_data is not None and len(self.extra_train_data) > 0:
                for dataset in self.extra_train_data:
                    d_t, t_d = dataset['dataset_types'], dataset['train_data']
                    self.train_datasets += get_train_dataset(d_t, t_d, self.tokenizer)
        except Exception as e:
            print(f"Could not process extra train datasets due to an error : {e}")

        # Extend datasets if needed
        attrs = ['train_data', 'frames', 'image_dir', 'video_files']
        extend_datasets(self.train_datasets, attrs, extend=self.extend_dataset)

        # Process datasets
        if len(self.train_datasets) == 1:
            self.train_dataset = self.train_datasets[0]
        else:
            self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets)

        # Setup LoRA components
        extra_text_encoder_params = self.extra_unet_params if self.extra_unet_params is not None else {}

        (self.lora_manager_temporal, self.unet_lora_params_temporal, self.unet_negation_temporal,
         self.optimizer_temporal, self.lr_scheduler_temporal,
         self.lora_managers_spatial, self.unet_lora_params_spatial_list,
         self.optimizer_spatial_list, self.lr_scheduler_spatial_list,
         self.unet_negation_all, self.spatial_lora_num) = setup_lora_components(
            unet=self.unet,
            train_temporal_lora=self.train_temporal_lora,
            single_spatial_lora=self.single_spatial_lora,
            train_dataset=self.train_dataset,
            use_unet_lora=self.use_unet_lora,
            lora_unet_dropout=self.lora_unet_dropout,
            lora_path=self.lora_path,
            lora_rank=self.lora_rank,
            learning_rate=self.learning_rate,
            extra_unet_params=extra_text_encoder_params,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_weight_decay=self.adam_weight_decay,
            adam_epsilon=self.adam_epsilon,
            lr_scheduler=self.lr_scheduler,
            lr_warmup_steps=self.lr_warmup_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_train_steps=self.max_train_steps,
            use_8bit_adam=self.use_8bit_adam
        )

    def _setup_data_loader(self):
        """Setup data loader and handle cache latents"""
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False
        )

        # Handle cache latents
        cached_data_loader = handle_cache_latents(
            self.cache_latents,
            self.output_dir,
            self.train_dataloader,
            self.train_batch_size,
            self.vae,
            self.unet,
            self.pretrained_model_path,
            self.validation_data.noise_prior,
            self.cached_latent_dir,
        )

        if cached_data_loader is not None:
            self.train_dataloader = cached_data_loader

    def _prepare_models(self):
        """Prepare models with accelerator"""
        self.unet, self.optimizer_spatial_list, self.optimizer_temporal, self.train_dataloader, self.lr_scheduler_spatial_list, self.lr_scheduler_temporal, self.text_encoder = self.accelerator.prepare(
            self.unet,
            self.optimizer_spatial_list, self.optimizer_temporal,
            self.train_dataloader,
            self.lr_scheduler_spatial_list, self.lr_scheduler_temporal,
            self.text_encoder
        )

        # Setup gradient checkpointing
        unet_and_text_g_c(
            self.unet,
            self.text_encoder,
            self.gradient_checkpointing,
            self.text_encoder_gradient_checkpointing
        )

        # Enable VAE slicing for memory efficiency
        self.vae.enable_slicing()

        # Cast models to appropriate dtype and GPU
        weight_dtype = is_mixed_precision(self.accelerator)
        models_to_cast = [self.text_encoder, self.vae]
        cast_to_gpu_and_type(models_to_cast, self.accelerator, weight_dtype)

    def _setup_scheduler(self):
        """Setup noise scheduler"""
        if not self.use_offset_noise and self.rescale_schedule:
            self.noise_scheduler.betas = enforce_zero_terminal_snr(
                self.noise_scheduler.betas)

    def _calculate_training_steps(self):
        """Calculate training steps"""
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps)
        self.num_train_epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

    def _init_trackers(self):
        """Initialize trackers"""
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("text2video-fine-tune")

    def train(self):
        """Main training loop"""
        # Training setup
        total_batch_size = self.train_batch_size * \
            self.accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Progress bar
        progress_bar = tqdm(range(global_step, self.max_train_steps),
                            disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        # Training loop
        for epoch in range(first_epoch, self.num_train_epochs):
            train_loss_spatial = 0.0
            train_loss_temporal = 0.0

            for step, batch in enumerate(self.train_dataloader):
                # Skip steps if resuming from checkpoint
                if self.resume_from_checkpoint and epoch == first_epoch and step < self.resume_step:
                    if step % self.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                global_step = self.train_step(step, batch, global_step, progress_bar,
                                              train_loss_spatial, train_loss_temporal)

                # Check if training is complete
                if global_step >= self.max_train_steps:
                    break

            # Check if training is complete
            if global_step >= self.max_train_steps:
                break

        # Save final model
        self._save_final_model(global_step)
        self.accelerator.end_training()

    def train_step(self, step, batch, global_step, progress_bar, train_loss_spatial, train_loss_temporal):
        """Train for one step"""
        with self.accelerator.accumulate(self.unet), self.accelerator.accumulate(self.text_encoder):
            text_prompt = batch['text_prompt'][0]

            # Setup masking
            mask_temporal_lora, mask_spatial_lora = self._setup_masking()

            # Forward and backward pass
            loss_spatial, loss_temporal = self._forward_backward_pass(
                step, batch, global_step, mask_spatial_lora, mask_temporal_lora
            )

            # Gather losses for logging
            train_loss_spatial, train_loss_temporal = self._gather_losses(
                loss_spatial, loss_temporal, mask_spatial_lora, mask_temporal_lora,
                train_loss_spatial, train_loss_temporal
            )

        # Check if accelerator performed optimization step
        if self.accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            self.accelerator.log(
                {"train_loss": train_loss_temporal}, step=global_step)
            train_loss_temporal = 0.0

            # Checkpoint and validation
            self._checkpoint_and_validation(global_step, batch, text_prompt)

        # Log temporal loss
        if loss_temporal is not None:
            self.accelerator.log(
                {"loss_temporal": loss_temporal.detach().item()}, step=step)

        return global_step

    def _setup_masking(self):
        """Setup masking for LoRA training"""
        if self.train_temporal_lora:
            mask_temporal_lora = False
        else:
            mask_temporal_lora = True
        mask_spatial_lora = random.uniform(0, 1) < 0.2 and not mask_temporal_lora
        return mask_temporal_lora, mask_spatial_lora

    def _forward_backward_pass(self, step, batch, global_step, mask_spatial_lora, mask_temporal_lora):
        """Perform forward and backward pass"""
        # Zero gradients
        for optimizer_spatial in self.optimizer_spatial_list:
            optimizer_spatial.zero_grad(set_to_none=True)

        if self.optimizer_temporal is not None:
            self.optimizer_temporal.zero_grad(set_to_none=True)

        # Forward pass
        with self.accelerator.autocast():
            loss_spatial, loss_temporal, _, _ = train_lora_unet(
                batch=batch,
                step=step,
                unet=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder,
                noise_scheduler=self.noise_scheduler,
                global_step=global_step,
                trainable_modules=self.trainable_modules,
                unet_negation_all=self.unet_negation_all,
                cache_latents=self.cache_latents,
                use_offset_noise=self.use_offset_noise,
                rescale_schedule=self.rescale_schedule,
                offset_noise_strength=self.offset_noise_strength,
                spatial_lora_num=self.spatial_lora_num,
                random_hflip_img=self.random_hflip_img,
                mask_spatial_lora=mask_spatial_lora,
                mask_temporal_lora=mask_temporal_lora,
                **self.kwargs
            )

        # Backward pass
        if not mask_spatial_lora:
            self.accelerator.backward(loss_spatial, retain_graph=True)
            if self.spatial_lora_num == 1:
                self.optimizer_spatial_list[0].step()
            else:
                self.optimizer_spatial_list[step].step()

        if not mask_temporal_lora and self.train_temporal_lora:
            self.accelerator.backward(loss_temporal)
            self.optimizer_temporal.step()

        # Update learning rate schedulers
        if self.spatial_lora_num == 1:
            self.lr_scheduler_spatial_list[0].step()
        else:
            self.lr_scheduler_spatial_list[step].step()
        if self.lr_scheduler_temporal is not None:
            self.lr_scheduler_temporal.step()

        return loss_spatial, loss_temporal

    def _gather_losses(self, loss_spatial, loss_temporal, mask_spatial_lora, mask_temporal_lora,
                       train_loss_spatial, train_loss_temporal):
        """Gather losses for logging"""
        if not mask_spatial_lora:
            avg_loss_spatial = self.accelerator.gather(
                loss_spatial.repeat(self.train_batch_size)).mean()
            train_loss_spatial += avg_loss_spatial.item() / self.gradient_accumulation_steps

        if not mask_temporal_lora and self.train_temporal_lora:
            avg_loss_temporal = self.accelerator.gather(
                loss_temporal.repeat(self.train_batch_size)).mean()
            train_loss_temporal += avg_loss_temporal.item() / self.gradient_accumulation_steps

        return train_loss_spatial, train_loss_temporal

    def _checkpoint_and_validation(self, global_step, batch, text_prompt):
        """Handle checkpoint saving and validation"""
        # Save checkpoint
        if global_step % self.checkpointing_steps == 0 and global_step > 0:
            save_pipe(
                self.pretrained_model_path,
                global_step,
                self.accelerator,
                self.unet,
                self.text_encoder,
                self.vae,
                self.output_dir,
                self.lora_managers_spatial[0] if self.lora_managers_spatial else None,
                self.lora_manager_temporal,
                self.unet_lora_modules,
                self.text_encoder_lora_modules,
                is_checkpoint=True,
                save_pretrained_model=self.save_pretrained_model
            )

        # Validation sampling
        if should_sample(global_step, self.validation_steps, self.validation_data):
            self._run_validation(global_step, batch, text_prompt)

    def _save_final_model(self, global_step):
        """Save final model"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_pipe(
                self.pretrained_model_path,
                global_step,
                self.accelerator,
                self.unet,
                self.text_encoder,
                self.vae,
                self.output_dir,
                self.lora_managers_spatial[0] if self.lora_managers_spatial else None,
                self.lora_manager_temporal,
                self.unet_lora_modules,
                self.text_encoder_lora_modules,
                is_checkpoint=False,
                save_pretrained_model=self.save_pretrained_model
            )

    def _run_validation(self, global_step, batch, text_prompt):
        """Run validation sampling"""
        if self.accelerator.is_main_process:
            # Set validation seed if specified
            if hasattr(self.validation_data, 'seed') and self.validation_data.seed is not None:
                torch.manual_seed(self.validation_data.seed)
                logger.info(f"Using validation seed: {self.validation_data.seed}")

            with self.accelerator.autocast():
                self.unet.eval()
                self.text_encoder.eval()
                unet_and_text_g_c(self.unet, self.text_encoder, False, False)

                # Extract and scale LoRA modules
                loras = extract_lora_child_module(
                    self.unet, target_replace_module=["Transformer2DModel"])
                for lora_i in loras:
                    lora_i.scale = self.validation_data.spatial_scale

                # Setup noise
                if self.validation_data.noise_prior > 0:
                    preset_noise = (self.validation_data.noise_prior) ** 0.5 * batch['inversion_noise'] + (
                        1 - self.validation_data.noise_prior) ** 0.5 * torch.randn_like(batch['inversion_noise'])
                else:
                    preset_noise = None

                # Create pipeline
                pipeline = TextToVideoSDPipeline.from_pretrained(
                    self.pretrained_model_path,
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    unet=self.unet
                )

                # Setup scheduler
                diffusion_scheduler = DDIMScheduler.from_config(
                    pipeline.scheduler.config)
                pipeline.scheduler = diffusion_scheduler

                # Generate samples
                prompt_list = text_prompt if len(
                    self.validation_data.prompt) <= 0 else self.validation_data.prompt
                for prompt in prompt_list:
                    save_filename = f"{global_step}_{prompt.replace('.', '')}"
                    out_file = f"{self.output_dir}/samples/{save_filename}.mp4"

                    with torch.no_grad():
                        video_frames = pipeline(
                            prompt,
                            width=self.validation_data.width,
                            height=self.validation_data.height,
                            num_frames=self.validation_data.num_frames,
                            num_inference_steps=self.validation_data.num_inference_steps,
                            guidance_scale=self.validation_data.guidance_scale,
                            latents=preset_noise
                        ).frames
                    export_to_video(
                        video_frames, out_file, self.train_data.get('fps', 8))
                    logger.info(f"Saved a new sample to {out_file}")

                del pipeline
                torch.cuda.empty_cache()

        # Restore gradient checkpointing
        unet_and_text_g_c(
            self.unet,
            self.text_encoder,
            self.gradient_checkpointing,
            self.text_encoder_gradient_checkpointing
        )
