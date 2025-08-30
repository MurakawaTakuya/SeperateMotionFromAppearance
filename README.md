
# Separate motion from appearance: Customizing motion via customizing text-to-video diffusion models

><p align="center"> <span style="color:#137cf3; font-family: Gill Sans">Huijie Liu</span><sup>1,2</sup>,</a>  <span style="color:#137cf3; font-family: Gill Sans">Jingyun Wang</span><sup>1</sup>,</a> <span style="color:#137cf3; font-family: Gill Sans">Shuai Ma</span><sup>1</sup>,</a>  <span style="color:#137cf3; font-family: Gill Sans">Jie Hu</span><sup>2</sup>,</a>  <span style="color:#137cf3; font-family: Gill Sans">Xiaoming Wei</span><sup>2</sup>, </a>  <span style="color:#137cf3; font-family: Gill Sans">Guoliang Kang</span><sup>1</sup></a> </a> <br> 
><span style="font-size: 16px">Beihang University</span><sup>1</sup>,Meituan</span><sup>2</sup></span></p>


## 📖 Abstract
Motion customization aims to adapt the diffusion model (DM) to generate videos with the motion specified by a set of video clips with the same motion concept. To realize this goal, the adaptation of DM should be possible to model the specified motion concept, without compromising the ability to generate diverse appearances. Thus, the key to solving this problem lies in how to separate the motion concept from the appearance in the adaptation process of DM. Typical previous works explore different ways to represent and insert a motion concept into large-scale pretrained text-to-video diffusion models, e.g., learning a motion LoRA, using latent noise residuals, etc. While those methods can encode the motion concept, they also inevitably encode the appearance in the reference videos, resulting in weakened appearance generation capability. In this paper, we follow the typical way to learn a motion LoRA to encode the motion concept, but propose two novel strategies to enhance motion-appearance separation, including temporal attention purification (TAP) and appearance highway (AH). Specifically, we assume that in the temporal attention module, the pretrained Value embeddings are sufficient to serve as basic components needed by producing a new motion. Thus, in TAP, we choose only to reshape the temporal attention with motion LoRAs so that Value embeddings can be reorganized to produce a new motion. Further, in AH, we alter the starting point of each skip connection in U-Net from the output of each temporal attention module to the output of each spatial attention module. Extensive experiments demonstrate that compared to previous works, our method can generate videos with appearance more aligned with the text descriptions and motion more consistent with the reference videos.

## ⚡️ Quick Start

### 🔧 Requirements and Installation


Install the requirements
```bash
pip install -r requirements.txt
```

### ⏬ Download
To access the Text-to-Video foundation model(ZeroScope), please click [here](https://huggingface.co/cerspense/zeroscope_v2_576w).

### Train
Please configure your config file according to the path of the downloaded Text-to-Video foundation model, and then execute:
```bash
cd train
python train.py --config configs/car.yaml
```

If you don't want to train a LoRA model, you can also use the checkpoint we provide.
| Types | checkpoints |
|--------|--------|
| car-turn | [Google Drive](https://drive.google.com/file/d/1nV7LaLVnX1fy9Dcm0PwsyXUc9mT731mr/view?usp=drive_link) |
| zoom-in |[Google Drive](https://drive.google.com/file/d/1g8uG4VSnaBxiDaMer46XQcjfY02asjXR/view?usp=drive_link) |
| zoom-out | [Google Drive](https://drive.google.com/file/d/1V_N1wpKT-PdECwbSupdGV2rYS_oLG5qS/view?usp=drive_link) |
| dolly | [Google Drive](https://drive.google.com/file/d/1WpH48b9jRHzDQXOY0jEU71z1Qwd47iwD/view?usp=drive_link) |


### Inference

```bash
cd ..
cd infer
CUDA_VISIBLE_DEVICES=7 python3 infer.py --model your/foundation/model/path/ \
    --prompt your/prompt/ \
    --checkpoint_folder your/checkpoints/folder \
    --checkpoint_index 1000 --seed xxx --output_dir outputs/infer/
```

##  Citation
If our work is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our paper:
```bibtex
@article{liu2025separate,
  title={Separate motion from appearance: Customizing motion via customizing text-to-video diffusion models},
  author={Liu, Huijie and Wang, Jingyun and Ma, Shuai and Hu, Jie and Wei, Xiaoming and Kang, Guoliang},
  journal={arXiv preprint arXiv:2501.16714},
  year={2025}
}
```
