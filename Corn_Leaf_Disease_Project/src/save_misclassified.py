import os
import shutil
import torch

from src.dataloader import get_dataloaders
from models.attention_model import CornDiseaseAttentionNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load data
train_loader, val_loader, test_loader, class_names = get_dataloaders()


# Load best model
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


# Class indexes
gray_index = class_names.index("Gray_Leaf_Spot")
blight_index = class_names.index("Blight")


# Output folder
output_folder = r"results\misclassified\gray_as_blight"

os.makedirs(
    output_folder,
    exist_ok=True
)


# We need the original test image paths
test_dataset = test_loader.dataset

saved_count = 0


with torch.no_grad():

    for batch_start, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu()


        for i in range(len(labels)):

            true_label = labels[i].item()
            predicted_label = predictions[i].item()


            # Gray Leaf Spot wrongly predicted as Blight
            if (
                true_label == gray_index
                and predicted_label == blight_index
            ):

                dataset_index = (
                    batch_start * test_loader.batch_size + i
                )

                original_path, _ = (
                    test_dataset.samples[dataset_index]
                )

                filename = os.path.basename(
                    original_path
                )

                destination = os.path.join(
                    output_folder,
                    filename
                )

                shutil.copy2(
                    original_path,
                    destination
                )

                saved_count += 1


print("\nFinished!")

print(
    f"Gray Leaf Spot predicted as Blight: {saved_count}"
)

print(
    "Saved in:",
    output_folder
)