import os
import torch
from torchvision.utils import save_image
from nets import Generator
from data import get_dataloaders

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
output_dir = "results_test_epochs"
os.makedirs(output_dir, exist_ok=True)

# Epochs to test
epochs_to_test = [1, 20, 40, 60, 80, 100]

# Load test data (only one image)
loader_A, _ = get_dataloaders(
    root_dir="datasets",
    domainA="ownImages",
    domainB="vanGogh",
    batch_size=1,
    num_workers=0,
    shuffle=False
)

@torch.no_grad()
def generate_samples_for_epochs():
    # Grab the single test image
    real_A, _ = next(iter(loader_A))
    real_A = real_A.to(device)

    for epoch in epochs_to_test:
        print(f"Processing epoch {epoch}...")

        # Load generators
        G_AB = Generator().to(device)
        G_BA = Generator().to(device)

        path_AB = f"checkpoints/G_AB_epoch{epoch}.pth"
        path_BA = f"checkpoints/G_BA_epoch{epoch}.pth"

        G_AB.load_state_dict(torch.load(path_AB, map_location=device))
        G_BA.load_state_dict(torch.load(path_BA, map_location=device))

        G_AB.eval()
        G_BA.eval()

        # Create subdirectory for this epoch
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Generate images
        fake_B = G_AB(real_A)
        recov_A = G_BA(fake_B)

        # Rescale to [0,1]
        real_A_disp = (real_A * 0.5) + 0.5
        fake_B_disp = (fake_B * 0.5) + 0.5
        recov_A_disp = (recov_A * 0.5) + 0.5

        # Save images
        save_image(real_A_disp, os.path.join(epoch_dir, "realA.png"))
        save_image(fake_B_disp, os.path.join(epoch_dir, "fakeB.png"))
        save_image(recov_A_disp, os.path.join(epoch_dir, "recovA.png"))

        print(f"Epoch {epoch} completed. Images saved in {epoch_dir}")

if __name__ == "__main__":
    generate_samples_for_epochs()