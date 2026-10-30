import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# Transformations definitions
data_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(
    data_dir='./tiny-imagenet/tiny-imagenet-200',
    batch_size=64,
    num_workers=2
):
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    
    # Load dataset and apply transformations
    train_dataset = ImageFolder(
      root=train_path, transform=data_transforms
    )
    val_dataset = ImageFolder(
        root=val_path, transform=data_transforms
    )

    # Create Dataloaders
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_dataset, val_dataset