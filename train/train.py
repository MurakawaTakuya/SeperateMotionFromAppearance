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
    handle_cache_latents
)
from .utils.model_utils import (
    load_primary_models, freeze_models, unet_and_text_g_c,
    handle_memory_attention,
    enforce_zero_terminal_snr, is_mixed_precision, cast_to_gpu_and_type
)
from .model.lora_setup import setup_lora_components
from .model.lora_training import train_lora_unet
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

    # デバッグ用に各プロセスで設定をログ出力
    create_logging(logging, logger, accelerator)

    # accelerate、transformers、diffusersの警告を初期化
    accelerate_set_verbose(accelerator)

    # 出力フォルダの作成を処理
    if accelerator.is_main_process:
        output_dir = create_output_folders(output_dir, config)

    # スケジューラー、トークナイザー、モデルを読み込み
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(
        pretrained_model_path)

    # 必要なモデルを凍結
    freeze_models([vae, text_encoder, unet])

    # 利用可能な場合はxformersを有効化
    handle_memory_attention(
        enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    # タイプに基づいてトレーニングデータセットを取得（json、single_video、image）
    train_datasets = get_train_dataset(dataset_types, train_data, tokenizer)

    # 追加のトレーニングデータがある場合、任意の数だけリストに追加可能
    # 例: extra_train_data: [{: {dataset_types, train_data: {etc...}}}]
    try:
        if extra_train_data is not None and len(extra_train_data) > 0:
            for dataset in extra_train_data:
                d_t, t_d = dataset['dataset_types'], dataset['train_data']
                train_datasets += get_train_dataset(d_t, t_d, tokenizer)

    except Exception as e:
        print(f"Could not process extra train datasets due to an error : {e}")

    # 最大のデータセットより小さいデータセットを拡張。これによりよりバランスの取れたトレーニングが可能
    attrs = ['train_data', 'frames', 'image_dir', 'video_files']
    extend_datasets(train_datasets, attrs, extend=extend_dataset)

    # 1つのデータセットを処理
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]

    # 複数のデータセットを処理
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # 条件付きで最適化するパラメータを作成（"condition"がtrueの場合、最適化する）
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}
    extra_text_encoder_params = extra_unet_params if extra_unet_params is not None else {}

    # LoRAコンポーネントの設定
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

    # データローダーの作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False
    )
    # 潜在変数のキャッシュ
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

    # `accelerator`ですべてを準備
    unet, optimizer_spatial_list, optimizer_temporal, train_dataloader, lr_scheduler_spatial_list, lr_scheduler_temporal, text_encoder = accelerator.prepare(
        unet,
        optimizer_spatial_list, optimizer_temporal,
        train_dataloader,
        lr_scheduler_spatial_list, lr_scheduler_temporal,
        text_encoder
    )

    # 有効な場合は勾配チェックポイントを使用
    unet_and_text_g_c(
        unet,
        text_encoder,
        gradient_checkpointing,
        text_encoder_gradient_checkpointing
    )

    # メモリ節約のためVAEスライシングを有効化
    vae.enable_slicing()

    # 混合精度トレーニングでは、text_encoderとvaeの重みを半精度にキャスト
    # これらのモデルは推論にのみ使用されるため、重みを全精度で保持する必要はない
    weight_dtype = is_mixed_precision(accelerator)

    # テキストエンコーダーとVAEをGPUに移動
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # 利用可能な場合は明暗領域を予測するためにノイズスケジュールを修正
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(
            noise_scheduler.betas)

    # トレーニングデータローダーのサイズが変更された可能性があるため、総トレーニングステップ数を再計算する必要がある
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)
    # トレーニングエポック数を再計算
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # 使用するトラッカーを初期化し、設定も保存する必要がある
    # トラッカーはメインプロセスで自動的に初期化される
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # ここから学習開始
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

    # 各マシンでプログレスバーを一度だけ表示
    progress_bar = tqdm(range(global_step, max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss_spatial = 0.0
        train_loss_temporal = 0.0

        for step, batch in enumerate(train_dataloader):
            # 再開ステップに到達するまでステップをスキップ
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
                    loss_spatial, loss_temporal, latents, init_noise = train_lora_unet(
                        batch=batch,
                        step=step,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        noise_scheduler=noise_scheduler,
                        global_step=global_step,
                        trainable_modules=trainable_modules,
                        unet_negation_all=unet_negation_all,
                        cache_latents=cache_latents,
                        use_offset_noise=use_offset_noise,
                        rescale_schedule=rescale_schedule,
                        offset_noise_strength=offset_noise_strength,
                        spatial_lora_num=spatial_lora_num,
                        random_hflip_img=random_hflip_img,
                        mask_spatial_lora=mask_spatial_lora,
                        mask_temporal_lora=mask_temporal_lora,
                        **kwargs
                    )

                # 分散トレーニングを使用する場合、ログ用にすべてのプロセスから損失を収集
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

            # アクセラレーターが裏で最適化ステップを実行したかチェック
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
                        # 指定されている場合はバリデーション用のシードを設定
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

    # 訓練されたモジュールを使用してパイプラインを作成し、保存
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
