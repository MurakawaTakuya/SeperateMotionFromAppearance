import os
import torch
import numpy as np
from einops import rearrange
import imageio


def save_video_preview(video_tensor, output_path, fps=8):
    """
    動画テンソルをプレビュー動画として保存する

    Args:
        video_tensor: 動画テンソル (f, c, h, w) または (f, h, w, c)
        output_path: 保存先パス
        fps: フレームレート
    """
    # テンソルの形状を確認し、必要に応じて変換
    if len(video_tensor.shape) == 4:
        if video_tensor.shape[1] == 3 or video_tensor.shape[1] == 1:  # (f, c, h, w)
            video_tensor = rearrange(video_tensor, "f c h w -> f h w c")
        # (f, h, w, c) の形状にする

    # テンソルをnumpy配列に変換
    if isinstance(video_tensor, torch.Tensor):
        video_np = video_tensor.detach().cpu().numpy()
    else:
        video_np = video_tensor

    # 値を0-255の範囲に正規化
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = np.clip(video_np, 0, 255).astype(np.uint8)

    # ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 動画として保存
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in video_np:
            writer.append_data(frame)


def create_preview_filename(step, global_step, video_path, output_dir, preview_dir=None):
    """
    プレビュー動画のファイル名を生成する

    Args:
        step: 現在のステップ
        global_step: グローバルステップ
        video_path: 元の動画パス
        output_dir: 出力ディレクトリ
        preview_dir: プレビューディレクトリ（指定されない場合はoutput_dir/previewを使用）

    Returns:
        str: プレビュー動画の保存パス
    """
    # 元の動画ファイル名を取得
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # プレビューファイル名を生成
    preview_filename = f"step_{global_step:06d}_batch_{step:04d}_{video_name}.mp4"

    # プレビューディレクトリのパス（デフォルトはoutput_dir/preview）
    if preview_dir is None:
        preview_dir = os.path.join(output_dir, "preview")

    return os.path.join(preview_dir, preview_filename)
