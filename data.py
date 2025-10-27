import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Function that:
# 1. Loads images from two specified directories (domainA and domainB).
# 2. Applies necessary transformations (resizing, normalization).
# 3. Returns DataLoader objects for both datasets.

def get_dataloaders(
    root_dir="datasets",
    domainA="cityData", # Domain A folder name
    domainB="vanGogh", # Domain B folder name
    img_size=256,
    batch_size=1,
    num_workers=2,
    shuffle=True
):
   
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] scaling
    ])

    path_A = os.path.join(root_dir, domainA)
    path_B = os.path.join(root_dir, domainB)

    # Generate ImageFolder datasets
    dataset_A = datasets.ImageFolder(root=path_A, transform=transform)
    dataset_B = datasets.ImageFolder(root=path_B, transform=transform)

    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader_A, loader_B