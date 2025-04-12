import torch
import torch.nn as nn
from timm.models._registry import register_model
import torchvision

import torch
import torch.nn as nn
from timm.models._registry import register_model
import torchvision

# Custom Model with ResNet50 Encoder and VAE Reparameterization
import torch
import torch.nn as nn
from timm.models._registry import register_model
import torchvision

# Custom Model with ResNet50 Encoder and VAE Reparameterization


class ConditionalVAE(nn.Module):
    def __init__(self, board_dim, trajectory_dim, latent_dim):
        super(ConditionalVAE, self).__init__()

        self.board_dim = board_dim
        self.trajectory_dim = trajectory_dim
        self.latent_dim = latent_dim

        # head to encode the image (convolution layer)

        # Encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(board_dim + trajectory_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(64, latent_dim)  # Log variance vector

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + board_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, trajectory_dim),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z ~ N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image_vector, trajectory_vector):
        # Combine image vector and trajectory vector as input to the encoder
        encoder_input = torch.cat((image_vector, trajectory_vector), dim=1)

        # Encode to latent space
        encoded = self.encoder_fc(encoder_input)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decode the latent variable z concatenated with the image vector
        decoder_input = torch.cat((z, image_vector), dim=1)
        reconstructed_trajectory = self.decoder_fc(decoder_input)

        return reconstructed_trajectory, mu, logvar

    def inference(self, image_vector, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image_vector.shape[0]
        image_vector = image_vector.repeat(
            n_samples, 1
        )  # Repeat across batch dimension
        z = torch.randn(n_batch * n_samples, self.latent_dim).to(
            image_vector.device
        )  # Sample latent vectors
        decoder_input = torch.cat(
            (z, image_vector), dim=1
        )  # Concatenate latent vectors with board coordinates
        generated_trajectories = self.decoder_fc(decoder_input)  # Generate trajectories
        return generated_trajectories.view(n_batch, n_samples, self.trajectory_dim)
