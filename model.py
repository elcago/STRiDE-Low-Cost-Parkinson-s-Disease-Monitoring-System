import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=128, channels=2, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.start = image_size // 8
        self.fc = nn.Linear(latent_dim, 256 * self.start * self.start)
        self.net = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, self.start, self.start)
        return self.net(x)


class DiscriminatorClassifier(nn.Module):
    def __init__(self, channels=2, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.real_fake = nn.Linear(256, 1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.real_fake(h), self.classifier(h)


class STGAN(nn.Module):
    def __init__(self, latent_dim=128, channels=2, n_classes=4, image_size=64):
        super().__init__()
        self.generator = Generator(latent_dim, channels, image_size)
        self.discriminator = DiscriminatorClassifier(channels, n_classes)
