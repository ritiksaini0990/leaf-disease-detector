import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders():

    # Dataset path
    base_path = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\processed"

    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")
    test_path = os.path.join(base_path, "test")

    # Training transformations
    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.3),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

    # Validation and test transformations
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_path,
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=val_test_transform
    )

    # DataLoaders
    batch_size = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Get class names
    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names


# This runs only when you directly execute dataloader.py
if __name__ == "__main__":

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Classes:", class_names)

    print("\nDataset sizes:")
    print("Training:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\nOne batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)