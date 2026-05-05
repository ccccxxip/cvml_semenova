from train import Decoder, Encoder, ImageDataset
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.figure(figsize=(10, 10))

for i, mode in enumerate([1, 2, 3, 4]):

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    encoder.load_state_dict(torch.load(f"encoder_mode{mode}.pth"))
    decoder.load_state_dict(torch.load(f"decoder_mode{mode}.pth"))

    encoder.eval()
    decoder.eval()

    dataset = ImageDataset(10, 256, mode=mode)
    image, _ = dataset[0]
    image = image.to(device)

    with torch.no_grad():
        latent = encoder(image.unsqueeze(0))
        result = decoder(latent)

    plt.subplot(4, 3, i*3 + 1)
    plt.imshow(image.cpu().squeeze())
    plt.title(f"Mode {mode}")

    plt.subplot(4, 3, i*3 + 2)
    plt.imshow(result.cpu().squeeze())
    plt.title("восстановление")

    plt.subplot(4, 3, i*3 + 3)
    plt.imshow((image.cpu().squeeze() - result.cpu().squeeze()))
    plt.title("разница")

plt.tight_layout()
plt.show()