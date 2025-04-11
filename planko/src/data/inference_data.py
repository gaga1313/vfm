from omegaconf import DictConfig, ValueNode, ListConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
from tqdm import tqdm
import cv2
# from transforms import transform
import pandas as pd
import dgread
import os

def load_image(directory):
    return Image.open(directory).convert('RGB')

class TestPlank(Dataset):
    def __init__(
        self, path: ValueNode,  transform = None, **kwargs
    ):
        super().__init__()
        # self.cfg = cfg
        self.path = path
        self.transform = transform
        self.ball_pos_idx = 16 #self.cfg.ball_pos_train_idx # number between 1 to 16
        import time
        print("started loading")
        t0 = time.time()
        #self.file_list = [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f))]
        self.id_to_side = {1:'right', 0:'left'}
        self.df = pd.read_csv(self.path)

        t1 = time.time()
        print("finished loading in ")
        print(t1-t0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        
        row = self.df.iloc[index]
        file_path = row['root_path']
        file_path = file_path.replace("/media/data_cifs_lrs", "/files22_lrsresearch/CLPS_Serre_Lab")
        name = 'board__' + row['id'] + "__" + self.id_to_side[row['label']] + '.png'
        file_name = os.path.join(file_path, name)
        # dgz_name = file_name.replace("train", "world").replace("test", "world").replace(".png", ".dgz").replace("png", "world")
        # import ipdb; ipdb.set_trace()
        img = load_image(file_name)
        if self.transform:
            img = self.transform(img)
        
        label = row['label']
        uncertainty = row['sim_uncertainty']
        rt = row['RT']

        return img, label, uncertainty, rt
    

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"

# if __name__ == "__main__":
#     test_dataset = TestPlank("/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_vfm/vfm/data/RT_Planko_RNN.csv", transform = transform)
    
#     test_dataloader = DataLoader(test_dataset, batch_size=32)

#     progress_bar = tqdm(test_dataloader)
#     for img, label, uncertainty, rt in progress_bar:
#         import ipdb; ipdb.set_trace()
#         x = 0

    
    