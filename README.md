# Pediatric Congenital Heart Disease Classification from Chest X-rays Using a Hybrid CNN

This repository contains the implementation of a hybrid deep learning model (ResNet50 + DenseNet121) for multi-class classification of pediatric chest X-ray images into four categories: **Normal**, **ASD** (Atrial Septal Defect), **VSD** (Ventricular Septal Defect), and **PDA** (Patent Ductus Arteriosus).

The model extracts complementary features from two convolutional backbones, fuses them after global average pooling, and performs four-class classification. Performance is evaluated with Accuracy, Precision, Recall, F1-Score, Specificity, confusion matrix, and ROC-AUC.

> **Note**: This project focuses on classification of already acquired chest radiographs. It does **not** perform diagnosis of congenital heart disease in a clinical setting, nor does it replace echocardiography or specialist interpretation.

## Overview

- **Task**: Four-class classification of pediatric CXR images (Normal / ASD / VSD / PDA)
- **Architecture**: Dual-stream hybrid network (ResNet50 + DenseNet121)
- **Feature fusion**: Concatenation after Global Average Pooling
- **Training**: 30 epochs, weighted Cross-Entropy loss, Adam optimizer with cosine annealing
- **Evaluation**: Accuracy, per-class Precision/Recall/F1, confusion matrix, macro ROC-AUC

## Method

### Data preprocessing
Images are resized to 224×224. During training the following augmentations are applied:
- Random crop
- Random affine transformation (rotation, translation, shear)
- Horizontal flip
- Color jitter (brightness & contrast)

All images are normalized with ImageNet mean and standard deviation. Missing or low-quality samples should be filtered before creating the ImageFolder structure.

### Modeling approach
Two pretrained backbones are used in parallel:

- **ResNet50**: residual connections help gradient flow in deeper layers.
- **DenseNet121**: dense connectivity reuses features across layers and produces compact representations.

Feature maps from both streams are reduced with Adaptive Average Pooling, flattened, concatenated, and passed through a two-layer classifier head (512 → 4) with dropout.

### Algorithms & techniques
- Transfer learning from ImageNet weights
- Weighted Cross-Entropy loss (to mitigate mild class imbalance)
- Gradient clipping
- Cosine annealing learning-rate schedule
- Standard stratified train/test split (70/30)

### Evaluation
The model is evaluated on a held-out test set. Reported metrics include:
- Overall accuracy
- Per-class Precision, Recall (Sensitivity), F1-Score
- Confusion matrix
- Macro-averaged ROC-AUC (one-vs-rest)

In the experiments reported in the thesis, the hybrid model reached an overall test accuracy of **81.53%**, with class-wise accuracies of approximately 79.6% (Normal), 86.4% (ASD), 87.0% (VSD) and 73.1% (PDA). Macro ROC-AUC was around **92.95%**.

## Repository structure

```
pediatric-chd-cxr-hybrid-cnn/
├── hybrid_chd_classifier.py   # Main training & evaluation script
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/armanahangari/pediatric-chd-cxr-hybrid-cnn.git
cd pediatric-chd-cxr-hybrid-cnn
pip install -r requirements.txt
```

Place your dataset in an ImageFolder layout:

```
dataset/
├── Normal/
├── ASD/
├── VSD/
└── PDA/
```

Then run:

```bash
python hybrid_chd_classifier.py
```

A checkpoint will be saved under `checkpoints/`.

## Thesis / Paper

Arman Ahangari Kiasari, Parvaneh Asghari, “A Hybrid Convolutional Neural Network for Pediatric Congenital Heart Disease Classification from Chest X-rays,” (In preparation)

## Citation

If you use this code or the associated work, please cite the thesis / forthcoming paper above.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Arman Ahangari Kiasari  
[GitHub](https://github.com/armanahangari) · [ORCID](https://orcid.org/0009-0002-0346-6204)
