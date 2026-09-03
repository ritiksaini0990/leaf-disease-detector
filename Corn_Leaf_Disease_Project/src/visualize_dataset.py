import os
import random
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\archive\data"

classes = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for i, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)

    images = os.listdir(class_path)
    random_image = random.choice(images)

    image_path = os.path.join(class_path, random_image)

    image = Image.open(image_path)

    axes[i].imshow(image)
    axes[i].set_title(class_name)
    axes[i].axis("off")

plt.tight_layout()
plt.show()