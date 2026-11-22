# Pediatric Congenital Heart Disease Classification from Chest X-rays Using a Hybrid CNN

This project performs multi-class classification of pediatric chest X-ray images into four categories: Normal, ASD, VSD, and PDA. A hybrid deep learning model based on ResNet50 and DenseNet121 extracts complementary features from two parallel pathways. The features are fused after global average pooling and used for four-class classification. The model is evaluated with accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
Data preprocessing: Images are resized to 224×224 and normalized with ImageNet statistics. Training augmentations include random crop, affine transformation, horizontal flip, and color jitter.
Modeling approach: Two pretrained backbones (ResNet50 and DenseNet121) extract features in parallel. After global average pooling, the feature vectors are concatenated and passed through a fully connected classifier with dropout.
Algorithms used: Hybrid CNN (ResNet50 + DenseNet121), weighted cross-entropy loss, Adam optimizer with cosine annealing, and gradient clipping.
Evaluation: Held-out test set evaluation using overall accuracy, per-class metrics, confusion matrix, and macro ROC-AUC. The hybrid model achieved 81.53% overall accuracy on the test set.

## Paper

Arman Ahangari Kiasari, Parvaneh Asghari, “A Hybrid Convolutional Neural Network for Pediatric Congenital Heart Disease Classification from Chest X-rays,” (In preparation)
