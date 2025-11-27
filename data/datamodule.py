from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_DATA_DIR = "datasets/Rock-Paper-Scissors/train"
VAL_DATA_DIR = "datasets/Rock-Paper-Scissors/test"

IMG_SIZE = 128
BATCH_SIZE = 32

# Data augmentation and normalization for training
# Just normalization for validation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),  # data augmentation
    transforms.RandomRotation(15),       # data augmentation
    transforms.ToTensor(),  
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (good default)
        std=[0.229, 0.224, 0.225]
    ),
])

# No data augmentation for validation
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (good default)
        std=[0.229, 0.224, 0.225]
    ),
])

# Datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=train_transforms)
validation_dataset = datasets.ImageFolder(root=VAL_DATA_DIR, transform=val_transforms)

# Class names and mapping (folder names => indices)
class_names = train_dataset.classes      # ['paper', 'rock', 'scissors']
class_to_idx = train_dataset.class_to_idx    # {'paper': 0, 'rock': 1, 'scissors': 2}
print("Classes", class_names)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)