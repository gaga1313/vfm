import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

sys.path.append( "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_vfm/vfm/linear_proble")

from src.utils import get_circle_kernel
from src.config import IMAGE_ITERATOR, KERNEL_SIZE, KERNEL_SIZE_SIGMA

CLICKME_DATA_ROOT = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/human_clickme_data_processing/assets/"

# A) For images WITHOUT heatmaps in training mode
image_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# B) For images WITHOUT heatmaps in validation mode
image_transforms_val = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# C) For images WITH heatmaps => resize,crop,tensor,normalize
image_transforms_with_heatmap = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# D) For heatmaps => resize,crop,tensor, but NO normalization
clickmap_transforms_no_norm = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def file_exists(sample):
    image_ok = os.path.exists(sample['image_path'])
    heatmap_ok = True if not sample['has_heatmap'] else os.path.exists(sample['heatmap_path'])
    return image_ok and heatmap_ok


class ClickMe(Dataset):
    def __init__(self, image_folder, label_to_category_map, is_training=True):
        super().__init__()
        self.image_folder = image_folder
        self.label_to_category_map = label_to_category_map
        self.is_training = is_training
        self.data = []
        self.circle_kernel = get_circle_kernel(KERNEL_SIZE, KERNEL_SIZE_SIGMA)

        self.root_dir = "/oscar/data/tserre/Shared/"

        training_images = os.listdir(os.path.join(CLICKME_DATA_ROOT, "co3d_train"))
        validation_images = os.listdir(os.path.join(CLICKME_DATA_ROOT, "co3d_val"))
        self.overlapping_images = set(training_images).intersection(set(validation_images))

        category_index = 0

        if is_training:
            print("Processing data...")
            co3d_path = "co3d_train"
            text_file = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/co3d-harmonization-pytorch/data_lists/co3d_train.txt"
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    path = path.split()[1]
                    files = parts[4].split()
                    if label not in self.label_to_category_map:
                        self.label_to_category_map[label] = category_index
                        category_index += 1
                    for i in range(0, 50, IMAGE_ITERATOR):
                        full_path = os.path.join(self.root_dir, path, files[i].strip())
                        entry = {
                            'image_path': full_path,
                            'heatmap_path': None,
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                        }
                        if file_exists(entry):
                            self.data.append(entry)
            print(f"1/4: Done processing training images WITHOUT ClickMaps. There are {len(self.data)} images.")

            data = os.listdir(os.path.join(CLICKME_DATA_ROOT, co3d_path))
            for file in data:
                if file not in self.overlapping_images:
                    image_name, object_class = self._process_file_name(file)
                    clickmap_path = os.path.join(CLICKME_DATA_ROOT, co3d_path, file)
                    image_path = os.path.join(self.root_dir, image_name)
                    if object_class not in self.label_to_category_map:
                        self.label_to_category_map[object_class] = category_index
                        category_index += 1
                    entry = {
                        'image_path': image_path,
                        'heatmap_path': clickmap_path,
                        'category_label': self.label_to_category_map[object_class],
                        'has_heatmap': True,
                    }
                    if file_exists(entry):
                        self.data.append(entry)
            print(f"2/4: Done processing training images WITH ClickMaps.")

        else:
            print('Processing val data...')
            co3d_path = "co3d_val"
            text_file = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/co3d-harmonization-pytorch/data_lists/co3d_val.txt"
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    path = path.split()[1]
                    files = parts[4].split()
                    if label not in self.label_to_category_map:
                        self.label_to_category_map[label] = category_index
                        category_index += 1
                    for i in range(0, 50, IMAGE_ITERATOR):
                        full_path = os.path.join(self.root_dir, path, files[i].strip())
                        entry = {
                            'image_path': full_path,
                            'heatmap_path': None,
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                        }
                        if file_exists(entry):
                            self.data.append(entry)
            print(f"3/4: Done processing validation images WITHOUT ClickMaps. There are {len(self.data)} images.")

            data = os.listdir(os.path.join(CLICKME_DATA_ROOT, co3d_path))
            for file in data:
                image_name, object_class = self._process_file_name(file)
                clickmap_path = os.path.join(CLICKME_DATA_ROOT, co3d_path, file)
                image_path = os.path.join(self.root_dir, image_name)
                if object_class not in self.label_to_category_map:
                    self.label_to_category_map[object_class] = category_index
                    category_index += 1
                entry = {
                    'image_path': image_path,
                    'heatmap_path': clickmap_path,
                    'category_label': self.label_to_category_map[object_class],
                    'has_heatmap': True,
                }
                if file_exists(entry):
                    self.data.append(entry)
            print(f"4/4: Done processing validation images WITH ClickMaps.")

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(sample['image_path']).convert("RGB")
        if sample['has_heatmap']:
            heatmap = np.load(sample['heatmap_path'])
            heatmap = torch.tensor(heatmap).mean(dim=0)
        else:
            heatmap = torch.from_numpy(np.zeros((256, 256)))

        label = torch.tensor(sample['category_label'], dtype=torch.long)

        hmp_np = heatmap.numpy() if isinstance(heatmap, torch.Tensor) else np.array(heatmap, dtype=np.float32)
        if hmp_np.ndim == 3 and hmp_np.shape[0] > 1:
            hmp_np = hmp_np.mean(axis=0)

        hmp_img = Image.fromarray(hmp_np, mode='F')

        if sample['has_heatmap']:
            image = image_transforms_with_heatmap(image)
        else:
            image = image_transforms_train(image) if self.is_training else image_transforms_val(image)

        heatmap = clickmap_transforms_no_norm(hmp_img)
        return image, heatmap, label, sample['has_heatmap']

    def __len__(self):
        return len(self.data)

    def _process_file_name(self, file):
        file_name = file.split("_")
        object_category = file_name[0]
        sequence_id = file_name[1:4]
        frame_number = file_name[5].split(".")[0]
        image_file_name = "binocular_trajectory/" + object_category + "/" + "_".join(sequence_id) + "/renders/" + frame_number + ".png"
        return image_file_name, object_category

def collate_fn(batch):
    images, heatmaps, labels, image_names = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    heatmaps = torch.stack(heatmaps, dim=0)
    return images, heatmaps, labels, image_names


if __name__ == '__main__':
    from tqdm import tqdm
    # val_dataset = ClickMe(image_folder="val", label_to_category_map={}, is_training=False)
    # val_loader = DataLoader(val_dataset, batch_size=32)

    train_dataset = ClickMe(image_folder="train", label_to_category_map={}, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=32)
    print(len(train_dataset))
    for image, heatmap, label, has_heatmap in tqdm(train_loader):
        x = 0