import os

base_path = r"C:\Users\madha\OneDrive\Desktop\Corn_Leaf_Disease_Project\dataset\processed"

splits = ["train", "val", "test"]

total_images = 0

for split in splits:
    split_path = os.path.join(base_path, split)
    split_total = 0

    print(f"\n{split.upper()} SET")

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)

        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            print(f"{class_name}: {count}")
            split_total += count

    print(f"Total {split}: {split_total}")
    total_images += split_total

print("\n" + "=" * 30)
print(f"TOTAL IMAGES: {total_images}")