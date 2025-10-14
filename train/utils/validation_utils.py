"""
Validation and sampling utility functions.
"""
import os
import gc
import copy
import sys
import torch
import imageio
import numpy as np
from diffusers import TextToVideoSDPipeline

# First-party imports
from train.utils.lora_handler import LoraHandler
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def should_sample(global_step, validation_steps, validation_data):
    """Check if we should sample at this step."""
    return global_step % validation_steps == 0 and validation_data.sample_preview


def export_to_video(video_frames, output_video_path, fps):
    """Export video frames to video file."""
    video_writer = imageio.get_writer(output_video_path, fps=fps)
    for img in video_frames:
        video_writer.append_data(np.array(img))
    video_writer.close()


def save_pipe(
        path,
        global_step,
        accelerator,
        unet,
        text_encoder,
        vae,
        output_dir,
        lora_manager_spatial: LoraHandler,
        lora_manager_temporal: LoraHandler,
        unet_target_replace_module=None,
        text_target_replace_module=None,
        is_checkpoint=False,
        save_pretrained_model=True
):
    """Save pipeline with LoRA weights."""
    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype

    # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_out = copy.deepcopy(accelerator.unwrap_model(
        unet.cpu(), keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(
        text_encoder.cpu(), keep_fp32_wrapper=False))

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)

    lora_manager_spatial.save_lora_weights(model=copy.deepcopy(
        pipeline), save_path=save_path + '/spatial', step=global_step)
    if lora_manager_temporal is not None:
        lora_manager_temporal.save_lora_weights(model=copy.deepcopy(
            pipeline), save_path=save_path + '/temporal', step=global_step)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [
            (unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    from accelerate.logging import get_logger
    logger = get_logger(__name__, log_level="INFO")
    logger.info(f"Saved model at {save_path} on step {global_step}")

    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()
