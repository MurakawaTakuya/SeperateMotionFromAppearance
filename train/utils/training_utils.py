"""
Training-related utility functions.
"""
import os
import copy
import sys
import torch
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, TextToVideoSDPipeline

# First-party imports
from train.utils.ddim_utils import ddim_inversion
from train.utils.dataset import CachedDataset

# Local imports
from .model_utils import tensor_to_vae_latent
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


already_printed_trainables = False


def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    """Handle trainable modules configuration."""
    global already_printed_trainables

    # This can most definitely be refactored :-)
    unfrozen_params = 0
    if trainable_modules is not None:
        for name, module in model.named_modules():
            for tm in tuple(trainable_modules):
                if tm == 'all':
                    model.requires_grad_(is_enabled)
                    unfrozen_params = len(list(model.parameters()))
                    break

                if tm in name and 'lora' not in name:
                    for m in module.parameters():
                        m.requires_grad_(is_enabled)
                        if is_enabled:
                            unfrozen_params += 1

    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True
        print(f"{unfrozen_params} params have been unfrozen for training.")


def inverse_video(pipe, latents, num_steps):
    """Perform DDIM inversion on video latents."""
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)

    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent


def handle_cache_latents(
        should_cache,
        output_dir,
        train_dataloader,
        train_batch_size,
        vae,
        unet,
        pretrained_model_path,
        noise_prior,
        cached_latent_dir=None,
):
    """
    Cache latents by storing them in VRAM.
    Speeds up training and saves memory by not encoding during the train loop.
    """
    if not should_cache:
        return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        vae=vae,
        unet=copy.deepcopy(unet).to('cuda', dtype=torch.float16)
    )
    pipe.text_encoder.to('cuda', dtype=torch.float16)

    cached_latent_dir = (
        os.path.abspath(
            cached_latent_dir) if cached_latent_dir is not None else None
    )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path = f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to(
                'cuda', dtype=torch.float16)
            batch['latents'] = tensor_to_vae_latent(pixel_values, vae)
            if noise_prior > 0.:
                batch['inversion_noise'] = inverse_video(
                    pipe, batch['latents'], 50)
            for k, v in batch.items():
                batch[k] = v[0]

            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir),
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0
    )
