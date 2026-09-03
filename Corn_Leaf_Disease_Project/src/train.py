import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataloader import get_dataloaders
from models.attention_model import CornDiseaseAttentionNet


# --------------------------------
# DEVICE
# --------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------
# CREATE REQUIRED FOLDERS
# --------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# --------------------------------
# LOAD DATA
# --------------------------------

train_loader, val_loader, test_loader, class_names = get_dataloaders()


print("\nClasses:")
print(class_names)


# --------------------------------
# CREATE MODEL
# --------------------------------

model = CornDiseaseAttentionNet(
    num_classes=len(class_names)
).to(device)


# --------------------------------
# LOSS FUNCTION
# --------------------------------

# Give extra importance to Gray Leaf Spot

# Class weights
# Extra importance to Gray_Leaf_Spot because it has fewer samples
class_weights = torch.tensor(
    [1.0, 1.0, 2.5, 1.0],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# --------------------------------
# OPTIMIZER
# --------------------------------

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)

# LEARNING RATE SCHEDULER
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# --------------------------------
# TRAINING SETTINGS
# --------------------------------

epochs = 30


# Lists for graphs

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


# Best validation accuracy

best_val_accuracy = 0.0



# =================================
# TRAINING LOOP
# =================================

for epoch in range(epochs):

    print("\n" + "=" * 50)
    print(f"Epoch {epoch + 1}/{epochs}")
    print("=" * 50)


    # -----------------------------
    # TRAINING
    # -----------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)


        # Forward pass

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # Statistics

        running_loss += (
            loss.item() * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    # Training results

    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        100 * correct / total
    )


    train_losses.append(
        train_loss
    )


    train_accuracies.append(
        train_accuracy
    )


    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item() * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)


            val_correct += (
                predicted == labels
            ).sum().item()


    # Validation results

    val_loss = (
        val_running_loss / val_total
    )


    val_accuracy = (
        100 * val_correct / val_total
    )

    # Update learning rate according to validation accuracy
    scheduler.step(val_accuracy)

    val_losses.append(
        val_loss
    )


    val_accuracies.append(
        val_accuracy
    )


    # -----------------------------
    # PRINT RESULTS
    # -----------------------------

    print("\nEpoch Results")

    print(
        f"Training Loss: {train_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            "models/best_corn_attention_model.pth"
        )


        print(
            "\nBest model saved!"
        )

        print(
            f"Best Validation Accuracy: "
            f"{best_val_accuracy:.2f}%"
        )


# =================================
# SAVE FINAL MODEL
# =================================

torch.save(
    model.state_dict(),
    "models/corn_attention_model.pth"
)


print("\n" + "=" * 50)

print("Training completed!")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print("=" * 50)


# =================================
# PLOT LOSS GRAPH
# =================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    range(1, epochs + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)


plt.plot(
    range(1, epochs + 1),
    val_losses,
    marker="o",
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid()


plt.savefig(
    "results/loss_graph.png",
    dpi=300
)


plt.close()


# =================================
# PLOT ACCURACY GRAPH
# =================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    range(1, epochs + 1),
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)


plt.plot(
    range(1, epochs + 1),
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid()


plt.savefig(
    "results/accuracy_graph.png",
    dpi=300
)


plt.close()


print("\nGraphs saved successfully!")

print(
    "results/loss_graph.png"
)

print(
    "results/accuracy_graph.png"
)