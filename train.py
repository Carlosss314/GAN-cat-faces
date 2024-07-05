import torch
from model import discriminator, generator
from tqdm import tqdm
import matplotlib.pyplot as plt


#hyperparameter settings
epochs = 500
lr = 0.0002
batch_size = 64
loss = torch.nn.BCELoss()


#model
G = generator()
D = discriminator()

G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


#data
images = torch.load("dataset.pt")
loader = torch.split(images, batch_size)

#show some images
# for i in range(15):
#     img = loader[0][i].permute(1, 2, 0) #reshape from (3, 64, 64) to (64, 64, 3)
#     img = ((img + 1) * (255/2)).int() #reverse normalization
#     plt.imshow(img)
#     plt.show()


D_loss_list = []
G_loss_list = []

for epoch in tqdm(range(epochs)):
    for i in tqdm(range(int(images.shape[0] / batch_size))):

        # Training the discriminator
        real_inputs = loader[i]
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1)

        noise = (torch.rand(real_inputs.shape[0], 128, 1, 1) - 0.5) / 0.5
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()


        # Training the generator
        noise = (torch.rand(real_inputs.shape[0], 128, 1, 1) - 0.5) / 0.5
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1])

        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()


        if (i+1)%50 == 0:
            print(f'Epoch {epoch+1} Image {i+1}/{int(images.shape[0]/batch_size)}: discriminator_loss {D_loss.item():.3f} generator_loss {G_loss.item():.3f}')
        i += 1


    D_loss_list.append(D_loss.item())
    G_loss_list.append(G_loss.item())

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(D_loss_list, c="blue", linewidth=1)
    plt.plot(G_loss_list, c="red", linewidth=1)
    plt.subplot(1, 2, 2)
    plt.imshow(fake_inputs[0].detach().permute(1, 2, 0))
    plt.pause(0.001)


    if (epoch+1) % 5 == 0:
        torch.save(G, f'model_parameters/Generator_epoch_{epoch+1}.pt')
