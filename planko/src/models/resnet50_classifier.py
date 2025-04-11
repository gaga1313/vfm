import torch
import torch.nn as nn
from timm.models._registry import register_model
import torchvision

# Custom Model with ResNet50 Encoder and VAE Reparameterization
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Classifier, self).__init__()
        
        # Load ResNet50 and remove the fully connected layer
        self.encoder = torchvision.models.resnet50(weights=None)
        self.encoder.fc = nn.Identity()  # Removing FC layer for feature extraction
        
        # Decision layer
        self.fc_decision = nn.Linear(2048, num_classes)  # Classification layer

    def forward(self, x):
        feature_maps = []  # To store the outputs of the last 5 layers
        
        # Go through layers of ResNet50 and store outputs of the last 5 layers
        for name, layer in list(self.encoder.named_children())[:-1]:  # Exclude the final FC layer
            x = layer(x)
            feature_maps.append(x)
        
        # # Keep only the last 5 feature maps
        # feature_maps = feature_maps[-5:]
        
        # Final decision layer output
        logits = self.fc_decision(x.flatten(1))  # Flatten before feeding into FC
        
        return logits, feature_maps[-3]
        # return logits
    
# __all__ = []


# @register_model
# def resnet50vaeclassifier(pretrained=False, **kwargs):
#     model = ResNet50VAEClassifier(n_classes=1)
#     return model
    
class ResNet50Regressor(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Regressor, self).__init__()
        
        # Load ResNet50 and remove the fully connected layer
        self.encoder = torchvision.models.resnet50(weights=None)
        self.encoder.fc = nn.Identity()  # Removing FC layer for feature extraction
        
        # Decision layer
        self.fc_decision = nn.Linear(2048, num_classes)  # Classification layer

    def forward(self, x):
        feature_maps = []  # To store the outputs of the last 5 layers
        
        # Go through layers of ResNet50 and store outputs of the last 5 layers
        for name, layer in list(self.encoder.named_children())[:-1]:  # Exclude the final FC layer
            x = layer(x)
            feature_maps.append(x)
        
        # # Keep only the last 5 feature maps
        # feature_maps = feature_maps[-5:]
        
        # Final decision layer output
        logits = self.fc_decision(x.flatten(1))  # Flatten before feeding into FC
        
        return logits, feature_maps[-3]