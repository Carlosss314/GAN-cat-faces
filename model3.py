# model utilisant la convolution

import torch.nn as nn

# class discriminator(nn.Module):
#     def __init__(self):
#         super(discriminator, self).__init__()
#         self.layer1 = nn.Sequential(
#             # in: 1 x 64 x 64
#             nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#             # out: 4 x 31 x 31
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#             # out: 8 x 14 x 14
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#             # out: 12 x 6 x 6
#         )
#         self.out = nn.Linear(12*6*6, 1)
#         # out: 1

#     def forward(self, x):
#         x = x.reshape(-1, 1, 64, 64) #reshape from (batch_size, 64, 64) to (batch_size, 1, 64, 64) for convolutional layer
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.reshape(x.shape[0], -1) #reshape from (batch_size, 6, 14, 14) to (batch_size, 6*14*14) for fully-connected layer
#         x = self.out(x)
#         return nn.Sigmoid()(x)




# class generator(nn.Module):
#     def __init__(self):
#         super(generator, self).__init__()
#         self.conv1 = nn.Sequential(
#             # in: 128 x 1 x 1
#             nn.ConvTranspose2d(128, 32, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             # out: 32 x 4 x 4
#         )
#         self.conv2 = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             # out: 16 x 8 x 8
#         )
#         self.conv3 = nn.Sequential(
#             nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             # out: 8 x 16 x 16
#         )
#         self.conv4 = nn.Sequential(
#             nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(4),
#             nn.ReLU(True),
#             # out: 4 x 32 x 32
#         )
#         self.conv5 = nn.Sequential(
#             nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.Tanh()
#             # out: 1 x 64 x 64
#         )

    # def forward(self, x):
    #     x = x.reshape(x.shape[0], x.shape[1], 1, 1)
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #     x = self.conv4(x)
    #     x = self.conv5(x)
    #     return x














class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            # in: 1 x 64 x 64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 64, 64) #reshape from (batch_size, 64, 64) to (batch_size, 1, 64, 64) for convolutional layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = nn.Flatten()(x)
        return nn.Sigmoid()(x)




class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.conv1 = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # out: 512 x 4 x 4
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 1 x 64 x 64
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x