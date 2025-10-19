python -m infer.infer \
    --model "models/zeroscope_v2_576w" \
    --prompt "A person is golfing in the golf course." \
    --checkpoint_folder outputs/train/train_2025-10-19T01-39-09 \
    --checkpoint_index 1000 \
    --output_dir ./inference_results/ \
    --noise_prior 0. \
    --seed 8551187 \
    --num-frames 16 \
    --lora_scale 1
