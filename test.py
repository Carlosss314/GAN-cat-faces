import torch
from model import generator
import matplotlib.pyplot as plt

# torch.manual_seed(42)

images = torch.load("dataset.pt")

G = generator()
G = torch.load("model_parameters/Generator_epoch_255.pt", map_location=torch.device('cpu'))

batch_size = 64
noise = (torch.rand(batch_size, 128, 1, 1) - 0.5) / 0.5
fake_imgs = G(noise)


for i in range(fake_imgs.shape[0]):
    img = images[i].permute(1, 2, 0)
    img = ((img + 1) * (255/2)).int() #reverse normalization
    fake_img = fake_imgs[i].detach().permute(1, 2, 0)
    fake_img = ((fake_img + 1) * (255/2)).int() #reverse normalization
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("a real image")
    plt.subplot(1, 2, 2)
    plt.imshow(fake_img)
    plt.title("a generated image")
    plt.show()
