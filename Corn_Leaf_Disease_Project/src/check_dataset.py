import os

dataset_path = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\archive\data"

classes = os.listdir(dataset_path)

print("Dataset Classes:")
print("-" * 30)

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        print(f"{class_name}: {len(images)} images")