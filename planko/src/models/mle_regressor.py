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


class MLEREG(nn.Module):
    def __init__(self, board_dim, trajectory_dim):
        super(MLEREG, self).__init__()

        self.trajectory_dim = trajectory_dim
        # head to encode the image (convolution layer)

        # Encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(board_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, trajectory_dim),
        )

    def forward(self, image_vector):
        # Combine image vector and trajectory vector as input to the encoder

        reconstructed_trajectory = self.encoder_fc(image_vector)
        return reconstructed_trajectory

    def inference(self, image_vector, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image_vector.shape[0]
        image_vector = image_vector.repeat(
            n_samples, 1
        )  # Repeat across batch dimension
        generated_trajectories = self.encoder_fc(image_vector)  # Generate trajectories
        return generated_trajectories.view(n_batch, n_samples, self.trajectory_dim)


class ConvREG(nn.Module):
    def __init__(self, image_channels, image_size):
        super(ConvREG, self).__init__()

        self.image_channels = image_channels

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
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Bottleneck before heads
            nn.ReLU(),
        )

        # 16 heads, each outputting a 896-dimensional vector
        self.heads = nn.ModuleList([nn.Linear(1024, 896) for _ in range(16)])

    def forward(self, image):
        # Combine image vector and trajectory vector as input to the encoder
        encoded = self.encoder_fc(self.image_encoder1(image))

        # Pass through each head
        outputs = torch.stack(
            [head(encoded) for head in self.heads], dim=1
        )  # Shape: (batch_size, 16, 896)

        return outputs

    def inference(self, image, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image.shape[0]
        image = image.repeat_interleave(
            n_samples, dim=0
        )  # Expands batch size instead of using repeat

        # Encode image
        encoded = self.encoder_fc(
            self.image_encoder1(image)
        )  # Ensure image_encoder2 can handle this shape

        outputs = torch.stack([head(encoded) for head in self.heads], dim=1)

        # Convert to indices using argmax
        generated_trajectories = torch.argmax(outputs, dim=-1)
        return generated_trajectories.view(n_batch, n_samples, 16)


class Conv2REG(nn.Module):
    def __init__(self, image_channels, image_size):
        super(Conv2REG, self).__init__()

        self.image_channels = image_channels

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
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # Bottleneck before heads
            nn.ReLU(),
        )

        # 16 heads, each outputting a 896-dimensional vector
        self.heads = nn.ModuleList([nn.Linear(1024, 100) for _ in range(32)])

    def forward(self, image):
        # Combine image vector and trajectory vector as input to the encoder
        encoded = self.encoder_fc(self.image_encoder1(image))

        # Pass through each head
        outputs = torch.stack(
            [head(encoded) for head in self.heads], dim=1
        )  # Shape: (batch_size, 16, 896)

        return outputs
        # return outputs

    def inference(self, image, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image.shape[0]
        image = image.repeat_interleave(
            n_samples, dim=0
        )  # Expands batch size instead of using repeat

        # Encode image
        encoded = self.encoder_fc(
            self.image_encoder1(image)
        )  # Ensure image_encoder2 can handle this shape

        outputs = torch.stack([head(encoded) for head in self.heads], dim=1)

        # Convert to indices using argmax
        generated_trajectories = torch.argmax(outputs, dim=-1)
        return generated_trajectories.view(n_batch, n_samples, 32)

    def inference2(self, image, n_samples=100):
        """Generates n_samples new trajectories conditioned on board coordinates."""
        n_batch = image.shape[0]
        image = image.repeat_interleave(
            n_samples, dim=0
        )  # Expands batch size instead of using repeat

        # Encode image
        encoded = self.encoder_fc(
            self.image_encoder1(image)
        )  # Ensure image_encoder2 can handle this shape

        outputs = torch.stack([head(encoded) for head in self.heads], dim=1)

        # Convert to indices using argmax
        generated_trajectories = torch.argmax(outputs, dim=-1)
        return generated_trajectories.view(n_batch, n_samples, 32)
