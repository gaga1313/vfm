import torch
import torch.nn as nn
from timm.models._registry import register_model
import torchvision


# Custom Model with ResNet50 Encoder and VAE Reparameterization
class ResNet50VAEClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super(ResNet50VAEClassifier, self).__init__()
        # Load ResNet50 and remove fully connected layer
        self.encoder = torchvision.models.resnet.resnet50(weights=None)
        self.encoder.fc = nn.Identity()  # Removing FC layer for feature extraction
        self.compress = nn.Linear(2048, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)  # Learned mean
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)  # Learned log variance
        self.fc_decision = nn.Linear(latent_dim, num_classes)  # Decision layer

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        feature_maps = []

        features = self.encoder(x)  # Extract features
        # Go through layers of ResNet50 and store outputs of the last 5 layers
        for name, layer in list(self.encoder.named_children())[
            :-1
        ]:  # Exclude the final FC layer
            x = layer(x)
            feature_maps.append(x)

        # # Keep only the last 5 feature maps
        feature_maps = feature_maps[-4]
        features = self.compress(features)
        mu = self.fc_mu(features)  # Mean
        logvar = self.fc_logvar(features)  # Log variance
        z = self.reparameterize(mu, logvar)  # New latent vector
        return (
            feature_maps,
            self.fc_decision(z),
            mu,
            logvar,
        )  # Output logits and latent parameters
        # return self.fc_decision(z)


# __all__ = []


# @register_model
# def resnet50vaeclassifier(pretrained=False, **kwargs):
#     model = ResNet50VAEClassifier(n_classes=1)
#     return model


class ResNet50VAERegressor(nn.Module):
    def __init__(self, latent_dim=1024, num_classes=10, inference=False, n_samples=100):
        super(ResNet50VAERegressor, self).__init__()
        # Load ResNet50 and remove fully connected layer
        self.encoder = torchvision.models.resnet.resnet50(weights=None)
        self.encoder.fc = nn.Identity()  # Removing FC layer for feature extraction
        self.compress = nn.Linear(2048, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)  # Learned mean
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)  # Learned log variance
        self.fc_decision = nn.Linear(latent_dim, num_classes)  # Decision layer
        self.inference = inference
        self.n_samples = n_samples

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        feature_maps = []
        features = self.encoder(x)  # Extract features
        # Go through layers of ResNet50 and store outputs of the last 5 layers
        if not self.inference:
            for name, layer in list(self.encoder.named_children())[
                :-1
            ]:  # Exclude the final FC layer
                x = layer(x)
                feature_maps.append(x)

            # # Keep only the last 5 feature maps
            feature_maps = feature_maps[-4]
        features = self.compress(features)
        mu = self.fc_mu(features)  # Mean
        logvar = self.fc_logvar(features)  # Log variance
        trajectories = []
        z = self.reparameterize(mu, logvar)  # New latent vector
        if self.inference:
            for i in range(self.n_samples):
                z = self.reparameterize(mu, logvar)  # New latent vector
                trajectories.append(self.fc_decision(z))
            return trajectories
        else:
            return (
                feature_maps,
                self.fc_decision(z),
                mu,
                logvar,
            )  # Output logits and latent parameters
