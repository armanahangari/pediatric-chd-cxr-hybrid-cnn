"""
Hybrid ResNet50 + DenseNet121 classifier for pediatric congenital heart disease
from chest X-ray images (Normal / ASD / VSD / PDA).

This script trains a two-stream CNN, fuses the features after global average
pooling, and evaluates the model with standard multi-class metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from tqdm import tqdm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATASET_DIR = "./dataset"          # ImageFolder structure expected
BATCH_SIZE = 24
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 4
SEED = 42

CLASS_NAMES = ["Normal", "ASD", "VSD", "PDA"]
# Class weights roughly inverse to frequency (can be adjusted)
CLASS_WEIGHTS = torch.tensor([1.93, 1.80, 1.95, 2.00])


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(200),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # For evaluation we keep a deterministic transform
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def prepare_dataloaders(dataset_dir: str = DATASET_DIR):
    train_tf, eval_tf = get_transforms()

    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_tf)

    # Force consistent class order
    full_dataset.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    full_dataset.classes = CLASS_NAMES

    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_test = n_total - n_train

    set_seed(SEED)
    train_set, test_set = random_split(full_dataset, [n_train, n_test])

    # Use evaluation transform for the test set
    test_set.dataset.transform = eval_tf

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class HybridResNetDenseNet(nn.Module):
    """
    Two-stream feature extractor:
      - ResNet-50 backbone (without final FC)
      - DenseNet-121 backbone (feature layers only)
    Features are pooled, concatenated and passed through a small classifier head.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()

        # ResNet-50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        resnet_dim = resnet.fc.in_features          # 2048

        # DenseNet-121
        densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.densenet_features = densenet.features
        densenet_dim = densenet.classifier.in_features  # 1024

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(resnet_dim + densenet_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        r = self.resnet_features(x)
        d = self.densenet_features(x)

        r = self.pool(r).flatten(1)
        d = self.pool(d).flatten(1)

        fused = torch.cat([r, d], dim=1)
        return self.classifier(fused)


# ------------------------------------------------------------------
# Training & Evaluation helpers
# ------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Macro ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "macro_auc": auc,
        "preds": all_preds,
        "labels": all_labels,
    }


def print_metrics(metrics, class_names=CLASS_NAMES):
    print(f"\nOverall Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"Macro ROC-AUC    : {metrics['macro_auc']*100:.2f}%\n")

    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for i, name in enumerate(class_names):
        print(f"{name:<10} {metrics['precision'][i]*100:>9.2f}% "
              f"{metrics['recall'][i]*100:>9.2f}% "
              f"{metrics['f1'][i]*100:>9.2f}%")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = prepare_dataloaders()

    model = HybridResNetDenseNet(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print("Starting training ...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, epoch)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | Train loss: {train_loss:.4f}")

    print("\nEvaluating on test set ...")
    metrics = evaluate(model, test_loader, device)
    print_metrics(metrics)

    # Optional: save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "class_names": CLASS_NAMES,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("preds", "labels", "confusion_matrix")},
    }, "checkpoints/hybrid_chd_best.pt")
    print("\nModel checkpoint saved to checkpoints/hybrid_chd_best.pt")


if __name__ == "__main__":
    main()
