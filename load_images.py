import torch
import torchvision
from tqdm import tqdm

n_images = 15744 #pour pouvoir faire un maximum de batch de 64

images = torch.zeros((n_images, 64, 64))

for i in tqdm(range(n_images)):
    img = torchvision.io.read_image(f"cats/{i+1}.jpg")
    img = img.float() #change dtype for normalization to work

    transform1 = torchvision.transforms.Grayscale()
    img = transform1(img)

    img = img / 255
    transform2 = torchvision.transforms.Normalize((0.5,), (0.5,))
    img = transform2(img)

    img = img.reshape(64, 64)

    images[i] = img

torch.save(images, "dataset.pt")