import torch
from model import generator
import matplotlib.pyplot as plt

# torch.manual_seed(42)

images = torch.load("dataset.pt")

G = generator()
G = torch.load("model_parameters/Generator_epoch_250.pt")
# G = torch.load("model_parameters2/Generator_epoch_105.pt")


noise = (torch.rand(100, 128) - 0.5) / 0.5
fake_imgs = G(noise)

fake_imgs = fake_imgs.reshape(100, 64, 64)

for i in range(fake_imgs.shape[0]):
    fake_img = fake_imgs[i].detach()
    plt.subplot(1, 2, 1)
    plt.imshow(images[i], cmap="gray")
    plt.title("a real image")
    plt.subplot(1, 2, 2)
    plt.imshow(fake_img, cmap="gray")
    plt.title("a generated image")
    plt.show()