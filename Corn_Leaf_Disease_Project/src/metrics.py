import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from src.dataloader import get_dataloaders
from models.attention_model import CornDiseaseAttentionNet


# -------------------------
# DEVICE
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# LOAD DATA
# -------------------------

train_loader, val_loader, test_loader, class_names = get_dataloaders()


# -------------------------
# LOAD MODEL
# -------------------------

model = CornDiseaseAttentionNet(
    num_classes=len(class_names)
).to(device)

model.load_state_dict(
    torch.load(
        "models/best_corn_attention_model.pth",
        map_location=device
    )
)

model.eval()


# -------------------------
# PREDICTIONS
# -------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# -------------------------
# CLASSIFICATION REPORT
# -------------------------

print("\nClassification Report")
print("=" * 50)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names
)

print(report)


# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Corn Leaf Disease Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

print("\nConfusion matrix saved successfully!")
print("Location: results/confusion_matrix.png")

plt.show()