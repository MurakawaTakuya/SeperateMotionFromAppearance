python3 infer/infer.py \
  --model "models/zeroscope_v2_576w" \
  --prompt "A horse is running in the forest." \
  --checkpoint_folder outputs/train/car/pretrained \
  --checkpoint_index 800 \
  --seed 8551187 \
  --output_dir ./inference_results
