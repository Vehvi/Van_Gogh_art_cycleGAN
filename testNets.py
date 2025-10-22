from nets import Generator, Discriminator
import torch

if __name__ == "__main__":
    G_AB = Generator()
    G_BA = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    x = torch.randn(1, 3, 128, 128)
    fake_B = G_AB(x)
    out = D_B(fake_B)

    print("Fake_B shape:", fake_B.shape)
    print("Discriminator output:", out.shape)