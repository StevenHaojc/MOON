
# Non-Contrast CT Esophageal Varices Grading through Clinical Prior-Enhanced Multi-Organ Analysis

## Environment Dependencies

```txt
numpy==1.26.2
scikit-learn==1.3.2
scipy==1.11.4
SimpleITK==2.4.0
timm==0.9.16
torch==2.1.0
torchaudio==2.1.0
torchvision==0.16.0
tqdm==4.65.2
PyYAML==6.0.1
tensorboard==2.16.2
imageio==2.33.1
transformers==4.47.1
monai==1.3.0
nibabel==5.2.0
opencv-python==4.9.0.80
matplotlib==3.8.2
pandas==2.1.4
scikit-image==0.22.0
seaborn==0.13.2
Pillow==9.5.0
pydicom==2.4.4
batchgenerators==0.25
dynamic-network-architectures==0.3.1
nnunetv2==2.5.1
einops==0.8.0
```

## Installation

```bash
pip install -r requirements.txt
pip install transformers
```

## Data Format

### File System Structure
```
Data/
├── esophagus/     # Esophagus CT images
│   ├── V0_0779_0000.nii.gz
│   └── ...
├── liver/         # Liver CT images
│   ├── V0_0779_0000.nii.gz
│   └── ...
├── spleen/        # Spleen CT images
│   ├── V0_0779_0000.nii.gz
│   └── ...
└── full/          # Full abdominal CT images
    ├── V0_0779_0000.nii.gz
    └── ...
```

### Data File Format
Each line in txt file contains: `[Image ID] [Label] [Text Description]`
```
V0_0779_0000 0 Abdominal CT scan showing a liver of normal size...
```

### Dataset Organization
- **Multi-organ Processing**: Each sample contains images of 4 organs (esophagus, liver, spleen, full)
- **Classification Labels**:
  - 0: Normal
  - 1: Mild abnormality
  - 2: Moderate abnormality
  - 3: Severe abnormality
- **Multimodal Data**: Image + Text description
- **Flexible Dimensions**: Different organs can have different target sizes
- **Data Augmentation**: Supports 3D spatial and intensity transformations

## Quick Start

```bash
bash run_all.sh
```

## Manual Training

```bash
python main/trainer3D_multi.py \
    --model_use Inter \
    --fusion_method concat \
    --fold 1 \
    --batch_size 16 \
    --gpus 0 \
    --num_epochs 100 \
    --data_dir ../Data
```

## Inference

```bash
# Validation set
python main/infer3D_multi.py \
    --model_use Inter \
    --dataset_type valid \
    --fold 1 \
    --model_path best_model_f1.pth

# Test set
python main/infer3D_multi.py \
    --model_use Inter \
    --dataset_type test \
    --fold 1 \
    --model_path best_model_f1.pth
```

## Evaluation

```bash
python main/postprocess/evaluate_one_fold.py \
    --rootpath results/.../fold1
```

## Project Structure

```
├── main/
│   ├── trainer3D_multi.py          # Training script
│   ├── infer3D_multi.py            # Inference script
│   ├── network/
│   │   ├── datasets/               # Data processing
│   │   ├── models/M3D/Uniformer/   # Model definitions
│   │   └── ...
│   └── postprocess/                # Post-processing
├── requirements.txt
├── run_all.sh
└── README.md
```
