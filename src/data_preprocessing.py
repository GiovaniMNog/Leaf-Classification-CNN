import os
import shutil
import random
from PIL import Image
import torchvision.transforms as transforms

# Original dataset directory
original_dataset_dir = 'data/Folio'

# Directories for the train, validation and test sets
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Creation of the directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Configuration of the train, validation and test ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# List of classes
classes = os.listdir(original_dataset_dir)

# Define a data augmentation pipeline
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# Save transformed images
def save_transformed_image(image_path, save_dir, transform, num_augmentations=5):
    image = Image.open(image_path)
    for i in range(num_augmentations):
        augmented_image = transform(image)
        augmented_image = transforms.ToPILImage()(augmented_image)
        base, ext = os.path.splitext(os.path.basename(image_path))
        augmented_image.save(os.path.join(save_dir, f"{base}_aug{i}{ext}"))

for class_name in classes:
    class_dir = os.path.join(original_dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # List all images in the class directory
        images = os.listdir(class_dir)
        
        # Shuffle the images
        random.shuffle(images)
        
        # Calculate the number of images in each set
        num_images = len(images)
        train_size = int(train_ratio * num_images)
        val_size = int(val_ratio * num_images)
        
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Create the class directories in the train, val and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Move the images to the corresponding directories
        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_dir, class_name, image)
            shutil.copyfile(src, dst)
            save_transformed_image(src, os.path.join(train_dir, class_name), data_augmentation)
        
        for image in val_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(val_dir, class_name, image)
            shutil.copyfile(src, dst)
        
        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_dir, class_name, image)
            shutil.copyfile(src, dst)

print("Dataset division and augmentation concluded!")
