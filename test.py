import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from nets import Generator
from data import get_dataloaders

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and settings
checkpoint_path = "checkpoints/G_AB_epoch100.pth"  # modify to the correct file name (latest/best checkpoint)
output_dir = "results_test"
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained generator G_AB
G_AB = Generator().to(device)
G_AB.load_state_dict(torch.load(checkpoint_path, map_location=device))
G_AB.eval()

print("Generator G_AB loaded successfully!")


# Load test data (city own images)
loader_A, _ = get_dataloaders(
    root_dir="datasets",
    domainA="ownImages", 
    domainB="vanGogh",    
    batch_size=1,
    num_workers=0,
    shuffle=False
)

# Generate A -> B samples
@torch.no_grad()
def generate_samples():
    for i, (real_A, _) in enumerate(loader_A):
        
        real_A = real_A.to(device)

        # A -> B
        fake_B = G_AB(real_A)

        # Rescale
        fake_B = (fake_B * 0.5) + 0.5

        save_path = os.path.join(output_dir, f"fake_{i+1:04d}.png")
        save_image(fake_B, save_path)

        if (i + 1) % 50 == 0:
            print(f"Saved {i+1} images...")

    print("Testing ready:", output_dir)


if __name__ == "__main__":
    generate_samples()