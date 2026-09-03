import torch
from src.dataloader import get_dataloaders
from models.attention_model import CornDiseaseAttentionNet

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Load data
train_loader, val_loader, test_loader, class_names = get_dataloaders()

# Create model
model = CornDiseaseAttentionNet(
    num_classes=len(class_names)
)

# Load trained model
model.load_state_dict(
    torch.load(
        "models/best_corn_attention_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Variables
correct = 0
total = 0

# Disable gradient calculation
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print("\nTest Results")
print("-" * 30)

print("Total Test Images:", total)

print(f"Test Accuracy: {accuracy:.2f}%")

print("\nClass Names:")
print(class_names)