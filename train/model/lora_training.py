# Standard library imports
from train.utils.model_utils import tensor_to_vae_latent, sample_noise
from train.utils.training_utils import handle_trainable_modules
from utils.lora import extract_lora_child_module
import random

# Third-party library imports
import torch
import torch.nn.functional as F
from torchvision import transforms

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# First-party imports

# Local imports


def train_lora_unet(
    batch,
    step,
    unet,
    vae,
    text_encoder,
    noise_scheduler,
    global_step,
    trainable_modules,
    unet_negation_all,
    cache_latents,
    use_offset_noise,
    rescale_schedule,
    offset_noise_strength,
    spatial_lora_num,
    random_hflip_img,
    mask_spatial_lora=False,
    mask_temporal_lora=False,
    **kwargs
):
    """
    UNetのLoRAアダプターを学習する関数

    Args:
        batch: バッチデータ
        step: 現在のステップ
        unet: UNetモデル（LoRAアダプターが追加済み）
        vae: VAEモデル
        text_encoder: テキストエンコーダー
        noise_scheduler: ノイズスケジューラー
        global_step: グローバルステップ
        trainable_modules: 訓練可能なモジュール
        unet_negation_all: UNet否定パラメータ
        cache_latents: 潜在変数キャッシュフラグ
        use_offset_noise: オフセットノイズ使用フラグ
        rescale_schedule: スケジュール再スケールフラグ
        offset_noise_strength: オフセットノイズ強度
        spatial_lora_num: 空間LoRA数
        random_hflip_img: ランダム水平フリップ確率
        mask_spatial_lora: 空間LoRAマスクフラグ
        mask_temporal_lora: 時間LoRAマスクフラグ
        **kwargs: その他の引数

    Returns:
        tuple: (loss_spatial, loss_temporal, latents, noise)
    """

    # UNETレイヤーの凍結を解除
    if global_step == 0:
        unet.train()
        handle_trainable_modules(
            unet,
            trainable_modules,
            is_enabled=True,
            negation=unet_negation_all
        )

    # 動画を潜在空間に変換
    if not cache_latents:
        latents = tensor_to_vae_latent(batch["pixel_values"], vae)
    else:
        latents = batch["latents"]

    # 潜在変数に追加するノイズをサンプリング
    use_offset_noise = use_offset_noise and not rescale_schedule
    noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
    bsz = latents.shape[0]

    # 各動画に対してランダムなタイムステップをサンプリング
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # 各タイムステップでのノイズの大きさに応じて潜在変数にノイズを追加
    # （これは順方向拡散プロセス）
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 勾配チェックポイント訓練を修正する可能性がある(potentially)
    # 参照: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    if kwargs.get('eval_train', False):
        unet.eval()
        text_encoder.eval()

    # テキスト埋め込みをエンコード
    token_ids = batch['prompt_ids']
    encoder_hidden_states = text_encoder(token_ids)[0]
    detached_encoder_state = encoder_hidden_states.clone().detach()

    # 予測タイプに応じて損失のターゲットを取得
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
