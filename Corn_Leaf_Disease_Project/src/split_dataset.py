import os
import shutil
import random

random.seed(42)

source_dir = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\archive\data"

output_dir = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\processed"

classes = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

train_ratio = 0.70
val_ratio = 0.15

for class_name in classes:

    source_class_dir = os.path.join(source_dir, class_name)

    images = [
        img for img in os.listdir(source_class_dir)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = os.path.join(
            output_dir,
            split_name,
            class_name
        )

        os.makedirs(destination, exist_ok=True)

        for image_name in split_images:

            source_path = os.path.join(
                source_class_dir,
                image_name
            )

            destination_path = os.path.join(
                destination,
                image_name
            )

            shutil.copy2(source_path, destination_path)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed successfully!")