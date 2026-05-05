import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import nn
import torch.optim as optim
import random
import string

class ImageDataset(Dataset):
    def __init__(self, n=200, size=128, mode=1):
        self.n = n
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.alphabet = string.ascii_uppercase

    def random_text(self, length):
        return ''.join(random.choice(self.alphabet) for _ in range(length))

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        image = Image.new("L", (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if self.mode == 1:
            text = "ABC"
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)

        elif self.mode == 2:
            text = self.random_text(3)
            x, y = 40, 40

        elif self.mode == 3:
            length = np.random.randint(1, 6)
            text = self.random_text(length)
            x, y = 40, 40

        elif self.mode == 4:
            length = np.random.randint(1, 6)
            text = self.random_text(length)
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transform(image)

        return tensor, tensor


class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x


if __name__ == "__main__":

    mode = 4

    ds = ImageDataset(2000, 256, mode=mode)
    dataloader = DataLoader(ds, batch_size=32, shuffle=True)

    encoder = Encoder()
    decoder = Decoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(decoder.parameters()))

    encoder.train()
    decoder.train()

    epochs = 10

    for epoch in range(epochs):
        epoch_loss = 0.0

        for imgs, _ in dataloader:
            imgs = imgs.to(device)

            optimizer.zero_grad()

            latent = encoder(imgs)
            output = decoder(latent)

            loss = criterion(imgs, output)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"mode={mode}, epoch={epoch}, loss={avg_loss:.4f}")

    torch.save(encoder.state_dict(), f"encoder_mode{mode}.pth")
    torch.save(decoder.state_dict(), f"decoder_mode{mode}.pth")