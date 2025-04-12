from torchvision import transforms

# Data Preparation
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
