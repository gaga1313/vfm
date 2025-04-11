import os
import time
import torch
import torchvision

from torch.utils.data import Dataset

def load_image(file_path):
    return torchvision.io.read_image(file_path)


class OrigPlank(Dataset):
    def __init__(
        self, path, train = False, transform = None, **kwargs
    ):
        super().__init__()
        self.path = path
        self.train = train
        self.transform = transform
        print("started loading")
        t0 = time.time()
        print(self.path)
        if os.path.exists(self.path):
            self.file_list = [(os.path.join(self.path, f.split("_")[0]+"_start.png"), 0 if "b1" in f else 1) for f in os.listdir(self.path) if "b" in f]
        if not self.train:
           self.file_list = sorted(self.file_list)
        t1 = time.time()
        print("finished loading in ")
        print(t1-t0)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        file_name, label = self.file_list[index]
        img = load_image(file_name)
        if self.transform:
            img = self.transform(img)
        return img, label
