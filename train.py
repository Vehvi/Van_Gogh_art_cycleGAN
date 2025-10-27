import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from nets import Generator, Discriminator
from data import get_dataloaders

# Options and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 1
lr = 0.0002
lambda_cycle = 10.0
lambda_identity = 5.0

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Load Data
loader_A, loader_B = get_dataloaders(
    root_dir="datasets",
    domainA="cityData",
    domainB="vanGogh",
    batch_size=batch_size,
    num_workers=0,
)

# Create Models
G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Loss Functions and Optimizers
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = Adam(
    list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999)
)
optimizer_D_A = Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(1, epochs + 1):
    for i, ((real_A, _), (real_B, _)) in enumerate(zip(loader_A, loader_B)):

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # GENERATOR
        optimizer_G.zero_grad()

        # A -> B and back
        fake_B = G_AB(real_A)
        recov_A = G_BA(fake_B)

        # B -> A and back
        fake_A = G_BA(real_B)
        recov_B = G_AB(fake_A)

        # Identity loss
        same_B = G_AB(real_B)
        same_A = G_BA(real_A)
        loss_identity = (
            criterion_identity(same_B, real_B) + criterion_identity(same_A, real_A)
        ) * lambda_identity * 0.5

        # Adversarial loss
        loss_GAN_AB = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        loss_GAN_BA = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) * 0.5

        # Cycle consistency loss
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle * 0.5

        # Total generator loss
        loss_G = loss_GAN + loss_cycle + loss_identity
        loss_G.backward()
        optimizer_G.step()

        # DISCRIMINATOR
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # D_A: real A vs. generated
        loss_D_A_real = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
        loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
        loss_D_A_total = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A_total.backward()
        optimizer_D_A.step()

        # D_B: real B vs. generated
        loss_D_B_real = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
        loss_D_B_fake = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
        loss_D_B_total = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B_total.backward()
        optimizer_D_B.step()

        # Visualization and Logging
        if (i + 1) % 500 == 0: 
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i+1}] "
                f"Loss_G: {loss_G.item():.4f}, "
                f"Loss_D_A: {loss_D_A_total.item():.4f}, "
                f"Loss_D_B: {loss_D_B_total.item():.4f}"
            )

    # Save model weights after each epoch
    torch.save(G_AB.state_dict(), os.path.join(save_dir, f"G_AB_epoch{epoch}.pth"))
    torch.save(G_BA.state_dict(), os.path.join(save_dir, f"G_BA_epoch{epoch}.pth"))

    # Save some example images
    with torch.no_grad():
        if epoch % 10 == 0:
            fake_B = G_AB(real_A)
            recov_A = G_BA(fake_B)
            save_image((fake_B * 0.5 + 0.5), f"results_fakeB_epoch{epoch}.png")
            save_image((recov_A * 0.5 + 0.5), f"results_recoveredA_epoch{epoch}.png")

print("-----Training complete!-----")