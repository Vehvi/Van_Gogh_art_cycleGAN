from data import get_dataloaders

# Simple test to verify dataloader functionality
if __name__ == "__main__":
    loader_A, loader_B = get_dataloaders(num_workers=0)  # tai pidä 2, mutta tämä helpottaa testissä
    for imgs, labels in loader_A:
        print(imgs.shape)
        break