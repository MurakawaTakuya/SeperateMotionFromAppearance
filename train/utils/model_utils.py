"""
Model-related utility functions for training.
"""
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from diffusers.models import AutoencoderKL
from einops import rearrange

from models.unet_3d_condition import UNet3DConditionModel


def load_primary_models(pretrained_model_path):
    """Load primary models (scheduler, tokenizer, text_encoder, vae, unet)."""
    noise_scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(
        pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def freeze_models(models_to_freeze):
    """Freeze model parameters."""
    for model in models_to_freeze:
        if model is not None:
            model.requires_grad_(False)


def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    """Set gradient checkpointing for UNet and text encoder."""
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder)


def is_attn(name):
    """Check if module name is an attention module."""
    return name.split('.')[-1] in ('attn1', 'attn2')


def set_processors(attentions):
    """Set attention processors."""
    for attn in attentions:
        attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    """Set Torch 2.0 attention for UNet."""
    optim_count = 0

    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet):
    """Handle memory efficient attention settings."""
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn

        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")

        if enable_torch_2:
            set_torch_2_attn(unet)

    except Exception as e:
        print(f"Could not enable memory efficient attention for xformers or Torch 2.0: {e}")


def tensor_to_vae_latent(t, vae):
    """Convert tensor to VAE latent representation."""
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents


def sample_noise(latents, noise_strength, use_offset_noise=False):
    """Sample noise for training."""
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents


def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def is_mixed_precision(accelerator):
    """Determine weight dtype based on mixed precision setting."""
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype


def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    """Cast models to GPU and specified dtype."""
    for model in model_list:
        if model is not None:
            model.to(accelerator.device, dtype=weight_dtype)
