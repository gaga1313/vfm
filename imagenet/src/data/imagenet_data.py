from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset split folder (e.g., 'data/train')
            transform (callable, optional): Optional transform to apply to each image
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get all class folders
        self.classes = sorted(
            entry.name for entry in os.scandir(root_dir) if entry.is_dir()
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Store all (image_path, label) pairs
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(cls_folder, img_name)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel
        if self.transform:
            image = self.transform(image)
        return image, label


# if __name__ == "__main__":
#     transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     val_dataset   = CustomImageDataset(root_dir="/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/mae/data/imagenet/val", transform=transform)
#     # train_dataset = CustomImageDataset(root_dir="/gpfs/data/shared/imagenet/ILSVRC2012/train", transform=transform)

#     # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
#     val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     for (images, labels) in val_loader:
#         print(labels)
