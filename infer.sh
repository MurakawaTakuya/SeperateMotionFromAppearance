python -m infer.infer \
    --model "models/zeroscope_v2_576w" \
    --prompt "A panda is diving in the garden." \
    --checkpoint_folder outputs/train/train_2025-10-25T17-13-16 \
    --checkpoint_index 100 \
    --output_dir ./inference_results/ \
    --noise_prior 0. \
    --seed 8551187 \
    --num-frames 16 \
    --lora_scale 1
