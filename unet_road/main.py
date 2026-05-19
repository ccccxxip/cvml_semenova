import torch
import numpy as np
import matplotlib.pyplot as plt

from unet_road import UNet
from unet_road import RoadsDataset
from unet_road import path


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = UNet(
    in_channels=3,
    out_channels=1,
    features=[64, 128, 256, 512]
).to(device)

model.load_state_dict(
    torch.load(
        "unet_road_model.pth",
        map_location=device
    )
)

model.eval()

ds = RoadsDataset(path)

image, mask = ds[0]

with torch.no_grad():

    pred = model(
        image.unsqueeze(0).to(device)
    )

    pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()


image = image.permute(1, 2, 0).numpy()

mask = mask.squeeze().numpy()

pred = pred.squeeze().cpu().numpy()

diff = np.abs(mask - pred)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Image")

plt.subplot(1, 4, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 4, 3)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")

plt.subplot(1, 4, 4)
plt.imshow(diff, cmap="hot")
plt.title("Difference")

plt.show()