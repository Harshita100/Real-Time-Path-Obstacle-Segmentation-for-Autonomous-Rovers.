# 🛰️ NavSeg: Real-Time Path & Obstacle Segmentation for Autonomous Rovers

> **Lightweight YOLO26 segmentation models that tell a rover where it can drive and what is in the way, fast enough for edge hardware.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-informational)
![Task](https://img.shields.io/badge/Task-Semantic%20%26%20Instance%20Segmentation-purple)
![Framework](https://img.shields.io/badge/PyTorch%20%7C%20TensorFlow-orange)
![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab%20(T4)-yellow)

---

## 📖 Overview

Autonomous rovers can't navigate safely without understanding their surroundings frame by frame:
**where is the drivable path, and what is blocking it?** NavSeg answers both by segmenting every image into
`path` and `obstacle` regions, in real time and with models small enough to run on embedded hardware.

Developed as part of an **ISRO** project, this repository compares five approaches: semantic segmentation
(YOLO26n-sem), instance segmentation (YOLO26n-seg) and a U-Net baseline, across outdoor (IGVC), indoor and
planetary-style terrain. The best models reach **0.94+ mIoU** on outdoor semantic segmentation and
**0.80 mask mAP50** on indoor instance segmentation, with only 1.6M to 2.7M parameters and inference
in single-digit to low-double-digit milliseconds on a T4 GPU.

Each model lives in its own notebook and covers the full pipeline: dataset download, augmentation,
label conversion, training, evaluation and inference.

**Keywords:** `Computer Vision` · `Semantic Segmentation` · `Instance Segmentation` · `YOLO26` ·
`Ultralytics` · `U-Net` · `Obstacle Detection` · `Path Segmentation` · `Autonomous Navigation` ·
`Data Augmentation` · `Albumentations` · `Roboflow` · `COCO → YOLO` · `Edge-Friendly Models`

---

## 🧪 Models at a Glance

| # | Notebook | Task | Architecture | Data | Status |
|---|---|---|---|---|---|
| 1 | `model_1.ipynb` | Semantic seg. (outdoor) | YOLO26n-sem | IGVC-Segmentation (augmented) | ✅ Strong |
| 2 | `model_2.ipynb` | Semantic seg. (indoor) | YOLO26n-sem | Self-annotated + IGVC | ⚠️ Underperforms |
| 3 | `model_3.ipynb` | Binary seg. baseline | U-Net (TensorFlow/Keras) | Custom | ❌ Did not converge |
| 4 | `model_4.ipynb` | Instance seg. (indoor) | YOLO26n-seg | Roboflow `indoor_obstacle_model` | ✅ Good |
| 5 | `model_5.ipynb` | Instance seg. (indoor + planetary) | YOLO26n-seg | Roboflow | 🚧 Results to be added |

> Rename the notebook filenames above to match what you upload.

---

## 📊 Results

### Model 1: Semantic segmentation on IGVC (outdoor)

Trained for 100 epochs (4.8 h on a Tesla T4) · 1.6M parameters · 4.3 GFLOPs · **~4–7 ms inference per image**

| Split | Class | IoU | Pixel Acc. |
|---|---|---|---|
| Validation (806 imgs) | obstacle | 0.977 | 0.987 |
| | path | 0.996 | 0.999 |
| | **mIoU (all)** | **0.985** | **0.996** |
| Test (62 imgs) | obstacle | 0.902 | 0.989 |
| | path | 0.985 | 0.986 |

**Test precision / recall / F1 (from the confusion matrix):** obstacle P 98.9 % · R 91.1 % · F1 94.8 %, path P 98.6 % · R 99.8 % · F1 99.2 %.

> ⚠️ **Known issue:** the model carries a third, empty `background` class that has no ground truth in the test set.
> Ultralytics averages it in as 0 IoU, so the reported test mIoU of 0.629 is misleading. Averaging only
> the two real classes gives about **0.94**. Fix: train with `nc: 2` and no background class.

### Model 2: Semantic segmentation on self-annotated data

| Class | IoU |
|---|---|
| clean_path | 0.143 |
| wall_obstacle | 0.116 |
| mIoU (all) | 0.126 |

Only 20 epochs were run (plus a resume), and **702 label files are empty** because unannotated areas were assumed to be `clean_path`. Treat this as an experiment on the road to Model 4, not a usable model.

### Model 3: U-Net (TensorFlow)

A from-scratch U-Net (encoder/decoder with skip connections). Training logged `loss: nan` from epoch 1, so there are no valid metrics. Kept in the repo as a documented failed baseline.

### Model 4: Instance segmentation (indoor obstacles)

YOLO26n-seg · 2.7M parameters · 9.0 GFLOPs · 130 epochs (0.83 h) · dataset split 762 / 164 / 163 (train / val / test)

| Test split | Class | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|
| 90 imgs | Clean_Path | 0.925 | 0.899 |
| | Obstacle | 0.670 | 0.501 |
| | **All** | **0.798** | **0.700** |

Box mAP50 is 0.771 and box mAP50-95 is 0.684. Inference takes about 22 ms per image on a T4.

### Model 5: Instance segmentation (indoor + planetary)

_Add results here once this notebook has been run on its own dataset._

---

## 🔬 Pipeline

```
 Dataset ──▶ Augmentation ──▶ Mask/COCO → YOLO polygons ──▶ Train ──▶ Evaluate ──▶ Infer
 (HF / Roboflow)  (Albumentations)    (OpenCV contours)        (Ultralytics)  (mIoU / mAP)
```

1. **Data:** IGVC-Segmentation from Hugging Face (`ngres/IGVC-Segmentation`) and indoor/planetary data annotated and exported from Roboflow.
2. **Augmentation (Model 1):** horizontal flip, affine (scale, translate, rotate), brightness/contrast, Gaussian noise and blur using Albumentations. The training set grows from 1,024 to 2,400 images.
3. **Label conversion:** binary masks become YOLO polygon labels via OpenCV contours. `path` is derived from the inverted obstacle mask. COCO exports are converted with `convert_coco(use_segments=True)`.
4. **Training:** pretrained YOLO26 nano weights, image size 640, batch size 8.
5. **Evaluation:** mIoU, pixel accuracy, per-class precision/recall/F1 for semantic models; mAP50 and mAP50-95 for instance models.
6. **Inference:** one helper runs any saved checkpoint over a folder of images on GPU or CPU.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A CUDA GPU is recommended (the notebooks were run on Google Colab with a Tesla T4)

### Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install ultralytics albumentations opencv-python datasets roboflow matplotlib
# Model 3 only:
pip install tensorflow
```

### Run a notebook

Open any notebook in Google Colab and run the cells top to bottom. Each notebook mounts Google Drive for datasets and checkpoints, so update the paths near the top to match your own Drive layout.

### Inference with a trained model

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(source="test_image.jpg", conf=0.25)
results[0].show()
```

---

## 🔐 Roboflow API key

Models 4 and 5 download their dataset from Roboflow. Set your own key as an environment variable instead of hard-coding it:

```python
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
```

---

## 🛣️ Roadmap

- [ ] Retrain Model 1 with exactly two classes (no phantom `background`)
- [ ] Re-annotate Model 2's data so unlabeled regions are not silently marked as path
- [ ] Fix the NaN loss in the U-Net (check mask scaling and label range)
- [ ] Merge outdoor, indoor and planetary data into one generalized model
- [ ] Export to ONNX / TensorRT and benchmark on an embedded board
- [ ] Add a live video/webcam inference demo

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.

## 📄 License

Add your license here (e.g., MIT). Check the licenses of the IGVC dataset and your Roboflow data before redistributing.

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- IGVC-Segmentation dataset by `ngres` on Hugging Face
- [Roboflow](https://roboflow.com) for annotation and dataset management
- ISRO
