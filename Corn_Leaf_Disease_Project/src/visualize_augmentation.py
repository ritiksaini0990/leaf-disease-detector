import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Dataset path
dataset_path = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\processed\train"

# Select one class
class_name = "Blight"

class_path = os.path.join(dataset_path, class_name)

# Get all images
images = [
    img for img in os.listdir(class_path)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Select one random image
random_image = random.choice(images)

image_path = os.path.join(class_path, random_image)

# Open original image
original_image = Image.open(image_path).convert("RGB")

# Augmentation
augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15)
])

# Create augmented versions
fig, axes = plt.subplots(1, 5, figsize=(18, 5))

# Original image
axes[0].imshow(original_image)
axes[0].set_title("Original")
axes[0].axis("off")

# Generate 4 augmented versions
for i in range(1, 5):
    augmented_image = augmentation(original_image)

    axes[i].imshow(augmented_image)
    axes[i].set_title(f"Augmented {i}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()