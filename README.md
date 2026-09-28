# PPM-CLIP

## 1. Installation

```bash
git clone https://github.com/bandaidssssss/PPM_CLIP.git
cd PPM_CLIP

conda create -n ppmclip python=3.10 -y
conda activate ppmclip

pip install -r requirements.txt
```

---

## 2. Dataset

This project mainly supports evaluation on **GenImage** and **UniversalFakeDetect / Ojha et al.** datasets.

### GenImage

- Official repository: https://github.com/GenImage-Dataset/GenImage
- Paper: https://arxiv.org/abs/2306.08571
- Dataset download: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing

### UniversalFakeDetect / Ojha et al.

- Official repository: https://github.com/WisconsinAIVision/UniversalFakeDetect
- Paper: *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023

After downloading the dataset, modify the corresponding paths in:

```text
data_loading.py
```

---

## 3. Checkpoint

Download our pretrained checkpoint here:

**Checkpoint:** [PPM-CLIP Checkpoint](https://modelscope.cn/models/Noone1/PPM_CLIP)

Place the checkpoint under:

```text
checkpoints/
```



---

## 4. Test

Run evaluation with:

```bash
python test.py \
    --dataset genimage \
    --gpu 0 \
    --ckpt_path checkpoints/ppm_clip_genimage.pt
```

