import torch
import torch.nn as nn
from torchvision.models import resnet18

# Custom Model with ResNet18 Encoder and VAE Reparameterization
class ResNet18VAE(nn.Module):
    def __init__(self, latent_dim=64, input_channels=3, image_size=224):
        super(ResNet18VAE, self).__init__()
        # Load ResNet18 and remove fully connected layer
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()  # Removing FC layer for feature extraction

        # Linear layers for latent space
        self.compress = nn.Linear(512, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)  # Learned mean
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)  # Learned log variance

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * (image_size // 32) ** 2),
            nn.Unflatten(1, (512, image_size // 32, image_size // 32)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output pixel values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)  # Extract features
        features = self.compress(features)
        mu = self.fc_mu(features)  # Mean
        logvar = self.fc_logvar(features)  # Log variance
        z = self.reparameterize(mu, logvar)  # New latent vector
        reconstructed_image = self.decoder(z)  # Generate image
        return reconstructed_image, mu, logvar  # Output reconstructed image and latent parameters


