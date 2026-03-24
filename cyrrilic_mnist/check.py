import numpy as np
from PIL import Image
from pathlib import Path

img_path = Path("Cyrillic/I/58b1d04f8aa15.png")

# RGBA
img = Image.open(img_path).convert('RGBA')

img_array = np.array(img) # в np массив

# проверка каждого канала

for channel in range(4):
    channel_data = img_array[:, :, channel]
    channel_name = ['R', 'G', 'B', 'A'][channel]
    
    print(f"\nКанал {channel_name}:")
    print("min", channel_data.min())
    print("max", channel_data.max())
    print("mean", channel_data.mean())
    print("std", channel_data.std())


# с помощью этой проверки стало понятно, что данные о букве находятся в альфа канале ( прозрачном ), 
# тк остальные три канала содержат только нули