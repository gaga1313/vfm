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


class CVAE(nn.Module):
    def __init__(self, board_dim, trajectory_dim, latent_dim):
        super(CVAE, self).__init__()

        self.board_dim = board_dim
        self.trajectory_dim = trajectory_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(board_dim + trajectory_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log variance vector

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + board_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Bottleneck before heads
            nn.ReLU(),
        )

        # 16 heads, each outputting a 896-dimensional vector
        self.heads = nn.ModuleList([nn.Linear(1024, 896) for _ in range(16)])

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
        decoded = self.decoder_fc(decoder_input)

        # Pass through each head
        outputs = torch.stack(
            [head(decoded) for head in self.heads], dim=1
        )  # Shape: (batch_size, 16, 896)

        return outputs, mu, logvar

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
        decoded = self.decoder_fc(decoder_input)  # Generate trajectories
        generated_trajectories = torch.stack(
            [head(decoded) for head in self.heads], dim=1
        )
        generated_trajectories = torch.argmax(generated_trajectories, dim=-1)
        return generated_trajectories.view(n_batch, n_samples, 16)


class CConvVAE(nn.Module):
    def __init__(self, image_channels, image_size, trajectory_dim, latent_dim):
        super(CConvVAE, self).__init__()

        self.image_channels = image_channels
        self.trajectory_dim = trajectory_dim
        self.latent_dim = latent_dim

        # Encoder
        self.image_encoder1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 512),
            nn.ReLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(512 + trajectory_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log variance vector

        self.image_encoder2 = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 512),
            nn.ReLU(),
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Bottleneck before heads
            nn.ReLU(),
        )

        # 16 heads, each outputting a 896-dimensional vector
        self.heads = nn.ModuleList([nn.Linear(1024, 896) for _ in range(16)])

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z ~ N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, trajectory_vector):
        # Combine image vector and trajectory vector as input to the encoder
        image_vector1 = self.image_encoder1(image)
        encoder_input = torch.cat((image_vector1, trajectory_vector), dim=1)

        # Encode to latent space
        encoded = self.encoder_fc(encoder_input)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        image_vector2 = self.image_encoder2(image)
        # Decode the latent variable z concatenated with the image vector
        decoder_input = torch.cat((image_vector2, z), dim=1)
        decoded = self.decoder_fc(decoder_input)

        # Pass through each head
        outputs = torch.stack(
            [head(decoded) for head in self.heads], dim=1
        )  # Shape: (batch_size, 16, 896)

        return outputs, mu, logvar

    def inference(self, image, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image.shape[0]
        image = image.repeat_interleave(
            n_samples, dim=0
        )  # Expands batch size instead of using repeat

        # Sample latent vectors
        z = torch.randn(n_batch * n_samples, self.latent_dim).to(image.device)

        # Encode image
        image_vector = self.image_encoder2(
            image
        )  # Ensure image_encoder2 can handle this shape

        # Concatenate latent vector with encoded image
        decoder_input = torch.cat((image_vector, z), dim=1)

        # Generate trajectories
        decoded = self.decoder_fc(decoder_input)
        generated_trajectories = torch.stack(
            [head(decoded) for head in self.heads], dim=1
        )

        # Convert to indices using argmax
        generated_trajectories = torch.argmax(generated_trajectories, dim=-1)
        return generated_trajectories.view(n_batch, n_samples, 16)


class CConvVAE2(nn.Module):
    def __init__(self, image_channels, image_size, trajectory_dim, latent_dim):
        super(CConvVAE2, self).__init__()

        self.image_channels = image_channels
        self.trajectory_dim = trajectory_dim
        self.latent_dim = latent_dim

        # Encoder
        self.image_encoder1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 512),
            nn.ReLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(512 + trajectory_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(256, latent_dim)  # Log variance vector

        self.image_encoder2 = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 512),
            nn.ReLU(),
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Bottleneck before heads
            nn.ReLU(),
        )

        # 16 heads, each outputting a 896-dimensional vector
        self.heads = nn.ModuleList([nn.Linear(1024, 100) for _ in range(32)])

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z ~ N(mu, sigma^2)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def forward(self, image, trajectory_vector):
    #     # Combine image vector and trajectory vector as input to the encoder
    #     image_vector1 = self.image_encoder1(image)
    #     encoder_input = torch.cat((image_vector1, trajectory_vector), dim=1)
    #     # Encode to latent space
    #     encoded = self.encoder_fc(encoder_input)
    #     mu = self.fc_mu(encoded)
    #     logvar = self.fc_logvar(encoded)

    #     # Reparameterization
    #     z = self.reparameterize(mu, logvar)

    #     image_vector2 = self.image_encoder2(image)
    #     # Decode the latent variable z concatenated with the image vector
    #     decoder_input = torch.cat((image_vector2, z), dim=1)
    #     decoded = self.decoder_fc(decoder_input)
    #     # Pass through each head
    #     outputs = torch.stack([head(decoded) for head in self.heads], dim=1)  # Shape: (batch_size, 16, 896)

    #     return outputs, mu, logvar

    # captum forward
    def forward(self, image):
        # Combine image vector and trajectory vector as input to the encoder
        z = torch.normal(mean=0, std=2, size=(32, self.latent_dim)).to(image.device)
        image_vector2 = self.image_encoder2(image)
        # Decode the latent variable z concatenated with the image vector
        decoder_input = torch.cat((image_vector2, z), dim=1)
        decoded = self.decoder_fc(decoder_input)
        # Pass through each head
        outputs = torch.stack(
            [head(decoded) for head in self.heads], dim=1
        )  # Shape: (batch_size, 16, 896)

        return outputs

    def inference(self, image, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image.shape[0]
        image = image.repeat_interleave(
            n_samples, dim=0
        )  # Expands batch size instead of using repeat

        # Sample latent vectors
        z = torch.normal(mean=0, std=1, size=(n_batch * n_samples, self.latent_dim)).to(
            image.device
        )

        # Encode image
        image_vector = self.image_encoder2(
            image
        )  # Ensure image_encoder2 can handle this shape

        # Concatenate latent vector with encoded image
        decoder_input = torch.cat((image_vector, z), dim=1)

        # Generate trajectories
        decoded = self.decoder_fc(decoder_input)
        generated_trajectories = torch.stack(
            [head(decoded) for head in self.heads], dim=1
        )

        # Convert to indices using argmax
        generated_trajectories = torch.argmax(generated_trajectories, dim=-1)
        return generated_trajectories.view(n_batch, n_samples, 32)

    def inference2(self, image, n_samples=1000, chunk_size=10):
        """Generates n_samples new trajectories while keeping computations on GPU
        and saving outputs on CPU to manage memory usage efficiently."""

        n_batch = image.shape[0]
        device = image.device  # Keep operations on the same device as the model

        all_generated_trajectories = []

        with torch.no_grad():
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                curr_n_samples = end - start  # Number of samples in this chunk

                # Expand batch size (stays on GPU)
                image_chunk = image.repeat_interleave(curr_n_samples, dim=0)

                # Sample latent vectors (on GPU)
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=(n_batch * curr_n_samples, self.latent_dim),
                    device=device,
                )

                # Encode image (on GPU)
                image_vector = self.image_encoder2(image_chunk)

                # Concatenate latent vector with encoded image (on GPU)
                decoder_input = torch.cat((image_vector, z), dim=1)

                # Generate trajectories (on GPU)
                decoded = self.decoder_fc(decoder_input)
                generated_trajectories = torch.stack(
                    [head(decoded) for head in self.heads], dim=1
                )

                # Convert to indices using argmax (on GPU)
                generated_trajectories = torch.argmax(generated_trajectories, dim=-1)

                # Move result to CPU to save GPU memory
                all_generated_trajectories.append(
                    generated_trajectories.view(n_batch, curr_n_samples, 32).cpu()
                )

        # Concatenate final output on CPU
        return torch.cat(all_generated_trajectories, dim=1)
