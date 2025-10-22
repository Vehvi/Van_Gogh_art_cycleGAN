from data import get_dataloaders

# Simple test to verify dataloader functionality
if __name__ == "__main__":
    loader_A, loader_B = get_dataloaders(num_workers=0) # Set num_workers=0 for Windows compatibility
    for imgs, labels in loader_A:
        print(imgs.shape)
        break