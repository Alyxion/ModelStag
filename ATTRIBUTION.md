# Model Attribution

This document provides detailed attribution for all AI models used in ModelStag.

---

## Background Removal Models (via rembg)

These models are accessed through the [rembg](https://github.com/danielgatis/rembg) library.

### U2-Net

| Field | Value |
|-------|-------|
| **Model** | U2-Net (U²-Net) |
| **Authors** | Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar Zaiane, Martin Jagersand |
| **Institution** | University of Alberta |
| **Paper** | "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection" (2020) |
| **Paper URL** | https://arxiv.org/abs/2005.09007 |
| **Code Repository** | https://github.com/xuebinqin/U-2-Net |
| **License** | Apache License 2.0 |
| **Hugging Face** | https://huggingface.co/briaai/RMBG-1.4 (rembg distribution) |
| **Used For** | `u2net`, `u2net_human_seg` models |

### BiRefNet (Bilateral Reference Network)

| Field | Value |
|-------|-------|
| **Model** | BiRefNet |
| **Authors** | Zhengyi Lu, Zongze Wu, Zheng Zhang |
| **Institution** | Nankai University, ByteDance |
| **Paper** | "Bilateral Reference for High-Resolution Dichotomous Image Segmentation" (2024) |
| **Paper URL** | https://arxiv.org/abs/2401.03407 |
| **Code Repository** | https://github.com/ZhengPeng7/BiRefNet |
| **License** | MIT License |
| **Hugging Face** | https://huggingface.co/ZhengPeng7/BiRefNet |
| **Used For** | `birefnet-general`, `birefnet-portrait`, `birefnet-hrsod`, `birefnet-dis` models |

**Variants:**
- `birefnet-general` - General purpose background removal
- `birefnet-portrait` - Optimized for portraits and people
- `birefnet-hrsod` - High-resolution salient object detection
- `birefnet-dis` - Dichotomous image segmentation (fine details like hair, fur)

### IS-Net (Intermediate Supervision Network)

| Field | Value |
|-------|-------|
| **Model** | IS-Net |
| **Authors** | Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, Luc Van Gool |
| **Institution** | University of Alberta, ETH Zurich, IIAI |
| **Paper** | "Highly Accurate Dichotomous Image Segmentation" (ECCV 2022) |
| **Paper URL** | https://arxiv.org/abs/2203.03041 |
| **Code Repository** | https://github.com/xuebinqin/DIS |
| **License** | Apache License 2.0 |
| **Hugging Face** | Distributed via rembg |
| **Used For** | `isnet-general-use`, `isnet-anime` models |

**Variants:**
- `isnet-general-use` - General object segmentation
- `isnet-anime` - Optimized for anime/illustration segmentation

---

## Segment Anything Model (SAM)

| Field | Value |
|-------|-------|
| **Model** | Segment Anything Model (SAM) |
| **Authors** | Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick |
| **Institution** | Meta AI Research (FAIR) |
| **Paper** | "Segment Anything" (2023) |
| **Paper URL** | https://arxiv.org/abs/2304.02643 |
| **Project Page** | https://segment-anything.com/ |
| **Code Repository** | https://github.com/facebookresearch/segment-anything |
| **License** | Apache License 2.0 |
| **Model Weights** | https://dl.fbaipublicfiles.com/segment_anything/ |

**Variants:**
| Model | Checkpoint | Size |
|-------|-----------|------|
| `sam` (vit_b) | `sam_vit_b_01ec64.pth` | 375 MB |
| `sam_large` (vit_l) | `sam_vit_l_0b3195.pth` | 1.2 GB |
| `sam_huge` (vit_h) | `sam_vit_h_4b8939.pth` | 2.4 GB |

---

## Depth Anything

| Field | Value |
|-------|-------|
| **Model** | Depth Anything |
| **Authors** | Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao |
| **Institution** | The University of Hong Kong, ByteDance |
| **Paper** | "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data" (2024) |
| **Paper URL** | https://arxiv.org/abs/2401.10891 |
| **Project Page** | https://depth-anything.github.io/ |
| **Code Repository** | https://github.com/LiheYoung/Depth-Anything |
| **License** | Apache License 2.0 |
| **Hugging Face** | https://huggingface.co/LiheYoung |

**Variants:**
| Model | Hugging Face ID |
|-------|-----------------|
| `depth_small` | `LiheYoung/depth-anything-small-hf` |
| `depth_base` | `LiheYoung/depth-anything-base-hf` |
| `depth_large` | `LiheYoung/depth-anything-large-hf` |

---

## YOLO-World

| Field | Value |
|-------|-------|
| **Model** | YOLO-World |
| **Authors** | Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan |
| **Institution** | Tencent AI Lab, Huazhong University of Science and Technology |
| **Paper** | "YOLO-World: Real-Time Open-Vocabulary Object Detection" (2024) |
| **Paper URL** | https://arxiv.org/abs/2401.17270 |
| **Code Repository** | https://github.com/AILab-CVC/YOLO-World |
| **License** | GPLv3 (Ultralytics) / Apache 2.0 (research) |
| **Distributed By** | Ultralytics (https://github.com/ultralytics/ultralytics) |
| **Ultralytics License** | AGPL-3.0 (open source) or Ultralytics Enterprise License |
| **Docs** | https://docs.ultralytics.com/models/yolo-world/ |

**Variants:**
| Model | Checkpoint |
|-------|-----------|
| `detect_small` | `yolov8s-worldv2.pt` |
| `detect_medium` | `yolov8m-worldv2.pt` |
| `detect_large` | `yolov8l-worldv2.pt` |

**Note:** YOLO-World uses open-vocabulary detection, allowing detection of arbitrary objects specified at runtime without retraining.

---

## Florence-2

| Field | Value |
|-------|-------|
| **Model** | Florence-2 |
| **Authors** | Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan |
| **Institution** | Microsoft |
| **Paper** | "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" (2023) |
| **Paper URL** | https://arxiv.org/abs/2311.06242 |
| **Code Repository** | https://github.com/microsoft/Florence-2 |
| **License** | MIT License |
| **Hugging Face** | https://huggingface.co/microsoft/Florence-2-base |

**Variants:**
| Model | Hugging Face ID |
|-------|-----------------|
| `caption_base` | `microsoft/Florence-2-base` |
| `caption_large` | `microsoft/Florence-2-large` |

**Capabilities:**
- Image captioning (brief, detailed, verbose)
- OCR (text extraction)
- Object detection
- Dense region captioning
- Visual question answering

---

## Pose Estimation Models

### MediaPipe Pose (BlazePose)

| Field | Value |
|-------|-------|
| **Model** | BlazePose / MediaPipe Pose |
| **Authors** | Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, Matthias Grundmann |
| **Institution** | Google |
| **Paper** | "BlazePose: On-device Real-time Body Pose tracking" (2020) |
| **Paper URL** | https://arxiv.org/abs/2006.10204 |
| **Code Repository** | https://github.com/google-ai-edge/mediapipe |
| **License** | Apache License 2.0 |
| **Docs** | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker |

**Features:**
- 33 body landmarks
- Real-time on mobile/edge devices
- Single-person tracking
- Includes body segmentation mask

### RTMO (Real-Time Multi-person One-stage)

| Field | Value |
|-------|-------|
| **Model** | RTMO |
| **Authors** | OpenMMLab Team |
| **Institution** | Shanghai AI Laboratory, SenseTime |
| **Paper** | "RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation" (2023) |
| **Paper URL** | https://arxiv.org/abs/2312.07526 |
| **Code Repository** | https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo |
| **License** | Apache License 2.0 |
| **Docs** | https://mmpose.readthedocs.io/ |

**Variants:**
| Model | Performance |
|-------|-------------|
| `pose_rtmo_s` | Fastest, good accuracy |
| `pose_rtmo_m` | Balanced speed/accuracy |
| `pose_rtmo_l` | 74.8% AP COCO, 141 FPS |

**Features:**
- One-stage multi-person detection (no separate person detector needed)
- Best for crowded scenes (83.8% AP CrowdPose)
- Real-time performance

### RTMW (Real-Time Multi-person Wholebody)

| Field | Value |
|-------|-------|
| **Model** | RTMW |
| **Authors** | OpenMMLab Team |
| **Institution** | Shanghai AI Laboratory |
| **Paper** | "RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation" (2024) |
| **Paper URL** | https://arxiv.org/abs/2407.08634 |
| **Code Repository** | https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose |
| **License** | Apache License 2.0 |

**Variants:**
| Model | Performance |
|-------|-------------|
| `pose_rtmw_m` | Medium wholebody |
| `pose_rtmw_l` | 70.2% mAP COCO-Wholebody (SOTA) |

**Features:**
- 133 keypoints (body + hands + face)
- First open-source model to exceed 70% mAP on COCO-Wholebody
- Real-time performance

---

## Hand Tracking Models

### MediaPipe Hands

| Field | Value |
|-------|-------|
| **Model** | MediaPipe Hands |
| **Authors** | Fan Zhang, Valentin Bazarevsky, Andrey Vakunov, Andrei Tkachenka, George Sung, Chuo-Ling Chang, Matthias Grundmann |
| **Institution** | Google |
| **Paper** | "MediaPipe Hands: On-device Real-time Hand Tracking" (2020) |
| **Paper URL** | https://arxiv.org/abs/2006.10214 |
| **Code Repository** | https://github.com/google-ai-edge/mediapipe |
| **License** | Apache License 2.0 |
| **Docs** | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker |

**Features:**
- 21 hand landmarks per hand
- Supports up to 2 hands
- Real-time on CPU
- Handedness detection (left/right)

### HaMeR (Hand Mesh Recovery)

| Field | Value |
|-------|-------|
| **Model** | HaMeR |
| **Authors** | Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, Jitendra Malik |
| **Institution** | UC Berkeley, University of Michigan |
| **Paper** | "Reconstructing Hands in 3D with Transformers" (CVPR 2024) |
| **Paper URL** | https://arxiv.org/abs/2312.05251 |
| **Project Page** | https://geopavlakos.github.io/hamer/ |
| **Code Repository** | https://github.com/geopavlakos/hamer |
| **License** | MIT License |

**Features:**
- Full 3D hand mesh reconstruction
- MANO parameter estimation
- Transformer-based architecture
- 2nd place Ego-Exo4D Challenge 2024

---

## Python Libraries

### rembg

| Field | Value |
|-------|-------|
| **Library** | rembg |
| **Author** | Daniel Gatis |
| **Repository** | https://github.com/danielgatis/rembg |
| **License** | MIT License |
| **PyPI** | https://pypi.org/project/rembg/ |
| **Purpose** | Background removal wrapper for U2-Net, BiRefNet, IS-Net models |

### Transformers

| Field | Value |
|-------|-------|
| **Library** | transformers |
| **Author** | Hugging Face |
| **Repository** | https://github.com/huggingface/transformers |
| **License** | Apache License 2.0 |
| **PyPI** | https://pypi.org/project/transformers/ |
| **Purpose** | Model loading and inference for Depth Anything, Florence-2 |

### Ultralytics

| Field | Value |
|-------|-------|
| **Library** | ultralytics |
| **Author** | Ultralytics |
| **Repository** | https://github.com/ultralytics/ultralytics |
| **License** | AGPL-3.0 |
| **PyPI** | https://pypi.org/project/ultralytics/ |
| **Purpose** | YOLO-World model loading and inference |

### segment-anything

| Field | Value |
|-------|-------|
| **Library** | segment-anything |
| **Author** | Meta AI |
| **Repository** | https://github.com/facebookresearch/segment-anything |
| **License** | Apache License 2.0 |
| **PyPI** | https://pypi.org/project/segment-anything/ |
| **Purpose** | SAM model loading and inference |

### MediaPipe

| Field | Value |
|-------|-------|
| **Library** | mediapipe |
| **Author** | Google |
| **Repository** | https://github.com/google-ai-edge/mediapipe |
| **License** | Apache License 2.0 |
| **PyPI** | https://pypi.org/project/mediapipe/ |
| **Purpose** | Pose estimation and hand tracking (MediaPipe Pose, MediaPipe Hands) |

### MMPose

| Field | Value |
|-------|-------|
| **Library** | mmpose |
| **Author** | OpenMMLab |
| **Repository** | https://github.com/open-mmlab/mmpose |
| **License** | Apache License 2.0 |
| **PyPI** | https://pypi.org/project/mmpose/ |
| **Purpose** | Pose estimation framework (RTMO, RTMW, RTMPose) |

### HaMeR

| Field | Value |
|-------|-------|
| **Library** | hamer |
| **Author** | Georgios Pavlakos |
| **Repository** | https://github.com/geopavlakos/hamer |
| **License** | MIT License |
| **Install** | `pip install git+https://github.com/geopavlakos/hamer.git` |
| **Purpose** | 3D hand mesh reconstruction |

---

## License Summary

| Model Family | License | Commercial Use |
|--------------|---------|----------------|
| U2-Net | Apache 2.0 | Yes |
| BiRefNet | MIT | Yes |
| IS-Net | Apache 2.0 | Yes |
| SAM | Apache 2.0 | Yes |
| Depth Anything | Apache 2.0 | Yes |
| YOLO-World | AGPL-3.0 / Enterprise | Requires AGPL compliance or license |
| Florence-2 | MIT | Yes |
| MediaPipe Pose | Apache 2.0 | Yes |
| RTMO | Apache 2.0 | Yes |
| RTMW | Apache 2.0 | Yes |
| MediaPipe Hands | Apache 2.0 | Yes |
| HaMeR | MIT | Yes |

**Important:** YOLO-World via Ultralytics is licensed under AGPL-3.0, which requires source code disclosure for derivative works. For commercial use without AGPL requirements, consider Ultralytics Enterprise License.

---

## Citation

If you use these models in research, please cite the original papers:

```bibtex
@inproceedings{qin2020u2net,
  title={U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
  author={Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
  booktitle={Pattern Recognition},
  year={2020}
}

@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and others},
  journal={arXiv preprint arXiv:2401.03407},
  year={2024}
}

@inproceedings{qin2022dis,
  title={Highly Accurate Dichotomous Image Segmentation},
  author={Qin, Xuebin and others},
  booktitle={ECCV},
  year={2022}
}

@article{kirillov2023sam,
  title={Segment Anything},
  author={Kirillov, Alexander and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}

@article{yang2024depthanything,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and others},
  journal={arXiv preprint arXiv:2401.10891},
  year={2024}
}

@article{cheng2024yoloworld,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and others},
  journal={arXiv preprint arXiv:2401.17270},
  year={2024}
}

@article{xiao2023florence,
  title={Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author={Xiao, Bin and others},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}

@article{bazarevsky2020blazepose,
  title={BlazePose: On-device Real-time Body Pose tracking},
  author={Bazarevsky, Valentin and others},
  journal={arXiv preprint arXiv:2006.10204},
  year={2020}
}

@article{lu2023rtmo,
  title={RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
  author={Lu, Peng and others},
  journal={arXiv preprint arXiv:2312.07526},
  year={2023}
}

@article{jiang2024rtmw,
  title={RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation},
  author={Jiang, Tao and others},
  journal={arXiv preprint arXiv:2407.08634},
  year={2024}
}

@article{zhang2020mediapipe,
  title={MediaPipe Hands: On-device Real-time Hand Tracking},
  author={Zhang, Fan and others},
  journal={arXiv preprint arXiv:2006.10214},
  year={2020}
}

@inproceedings{pavlakos2024hamer,
  title={Reconstructing Hands in 3D with Transformers},
  author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
  booktitle={CVPR},
  year={2024}
}
```
