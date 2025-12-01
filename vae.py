import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_var = nn.Linear(flatten_dim, latent_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()

        self.proj = nn.Linear(latent_dim, 256 * 4 * 4)
        self.relu = nn.ReLU()
        self.conv_t1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_t2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_t3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv_t4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)
        x = x.reshape(x.size(0), 256, 4, 4)

        x = self.relu(self.bn1(self.conv_t1(x)))
        x = self.relu(self.bn2(self.conv_t2(x)))
        x = self.relu(self.bn3(self.conv_t3(x)))
        x = self.sigmoid(self.bn4(self.conv_t4(x)))

        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x = self.decoder(z)

        return x, mu, log_var