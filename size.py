from PIL import Image
import os

input_path = "image-data"
output_path = "o-image-data"
target_size = (28, 28)

for filename in os.listdir(input_path):
    img_path = os.path.join(input_path, filename)
    img = Image.open(img_path)
    img = img.resize(target_size)
    img.save(os.path.join(output_path, filename))
