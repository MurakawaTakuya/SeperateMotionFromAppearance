# Standard library imports
import argparse
import os
import sys

# Third-party library imports
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

# Local imports
from train.model.model import TextToVideoModel

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def main(
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        validation_data: Dict,
        single_spatial_lora: bool = False,
        train_temporal_lora: bool = True,
        random_hflip_img: float = -1,
        extra_train_data: list = None,
        dataset_types: Tuple[str] = None,
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
        unet_lora_modules: Tuple[str] = None,
        text_encoder_lora_modules: Tuple[str] = None,
        save_pretrained_model: bool = True,
        lora_rank: int = 16,
        lora_path: str = '',
        lora_unet_dropout: float = 0.1,
        logger_type: str = 'tensorboard',
        disable_comet: bool = False,
        save_preview: bool = False,
        **kwargs
):
    """
    Main function for text-to-video fine-tuning.
    Creates and trains a TextToVideoModel instance.
    """
    # Create model instance (initialization includes all setup)
    if extra_train_data is None:
        extra_train_data = []
    if unet_lora_modules is None:
        unet_lora_modules = []
    if text_encoder_lora_modules is None:
        text_encoder_lora_modules = []
    if dataset_types is None:
        dataset_types = ('json',)

    # TODO: モデル名変える
    model = TextToVideoModel(
        pretrained_model_path=pretrained_model_path,
        output_dir=output_dir,
        train_data=train_data,
        validation_data=validation_data,
        single_spatial_lora=single_spatial_lora,
        train_temporal_lora=train_temporal_lora,
        random_hflip_img=random_hflip_img,
        extra_train_data=extra_train_data,
        dataset_types=dataset_types,
        validation_steps=validation_steps,
        trainable_modules=trainable_modules,
        extra_unet_params=extra_unet_params,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_weight_decay=adam_weight_decay,
        adam_epsilon=adam_epsilon,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        text_encoder_gradient_checkpointing=text_encoder_gradient_checkpointing,
        checkpointing_steps=checkpointing_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_step=resume_step,
        mixed_precision=mixed_precision,
        use_8bit_adam=use_8bit_adam,
        enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
        enable_torch_2_attn=enable_torch_2_attn,
        seed=seed,
        use_offset_noise=use_offset_noise,
        rescale_schedule=rescale_schedule,
        offset_noise_strength=offset_noise_strength,
        extend_dataset=extend_dataset,
        cache_latents=cache_latents,
        cached_latent_dir=cached_latent_dir,
        use_unet_lora=use_unet_lora,
        unet_lora_modules=unet_lora_modules,
        text_encoder_lora_modules=text_encoder_lora_modules,
        save_pretrained_model=save_pretrained_model,
        lora_rank=lora_rank,
        lora_path=lora_path,
        lora_unet_dropout=lora_unet_dropout,
        logger_type=logger_type,
        disable_comet=disable_comet,
        save_preview=save_preview,
        **kwargs
    )

    # Start training
    model.train()


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default='./configs/config_multi_videos.yaml')
    parser.add_argument("--disable_comet", action="store_true",
                        help="Disable Comet logging")
    parser.add_argument("--dc", action="store_true",
                        help="Short form for --disable_comet")
    parser.add_argument("--save_preview", action="store_true",
                        help="Enable video preview saving during training")
    parser.add_argument("--preview_dir", type=str, default=None,
                        help="Directory to save preview videos of trimmed training videos (default: output_dir/preview)")
    args = parser.parse_args()

    # Load config from file
    config = OmegaConf.load(args.config)

    # Override disable_comet if specified via command line
    if args.disable_comet or args.dc:
        config.disable_comet = True

    # Override save_preview if specified via command line
    if args.save_preview:
        config.train_data.save_preview = True

    # Override preview_dir if specified via command line
    if args.preview_dir:
        config.train_data.preview_dir = args.preview_dir

    main(**config)


if __name__ == "__main__":
    main_cli()
