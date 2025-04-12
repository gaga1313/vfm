from omegaconf import DictConfig, ValueNode, ListConfig
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
from tqdm import tqdm
import cv2

# from transforms import transform
import pandas as pd
import os


def load_image(directory):
    return Image.open(directory).convert("RGB")


class EyeTrackData(Dataset):
    def __init__(self, path: ValueNode, transform=None, **kwargs):
        super().__init__()
        # self.cfg = cfg
        self.path = path
        self.board_path = path.replace("s20_board_eye_tracking", "human_boardss20/png")
        self.transform = transform
        self.ball_pos_idx = 16  # self.cfg.ball_pos_train_idx # number between 1 to 16
        import time

        print("started loading")
        t0 = time.time()
        self.eye_file_list = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if os.path.isfile(os.path.join(self.path, f))
        ]
        self.board_file_list = [
            os.path.join(self.board_path, f)
            for f in os.listdir(self.board_path)
            if os.path.isfile(os.path.join(self.board_path, f))
        ]

        self.eye_file_list.sort()
        self.board_file_list.sort()

        self.id_to_side = {1: "right", 0: "left"}

        t1 = time.time()
        print("finished loading in ")
        print(t1 - t0)

    def __len__(self) -> int:
        return len(self.eye_file_list)

    def __getitem__(self, index: int):
        file_name = self.eye_file_list[index]
        file_name = file_name.replace(
            "/media/data_cifs_lrs", "/files22_lrsresearch/CLPS_Serre_Lab"
        )

        # dgz_name = file_name.replace("train", "world").replace("test", "world").replace(".png", ".dgz").replace("png", "world")
        # import ipdb; ipdb.set_trace()
        # /media/data_cifs_lrs/projects/prj_vis_sim/s20_board_eye_tracking/s20_board_000.png
        # /media/data_cifs_lrs/projects/prj_vis_sim/human_boardss20/png/board_001_13079_left.png
        board_file_name = self.board_file_list[index]
        board_img = load_image(board_file_name)
        eye_img = load_image(file_name)
        if self.transform:
            board_img = self.transform(board_img)
            eye_img = self.transform(eye_img)

        label = board_file_name.split("_")[-1][:-4]

        return board_img, eye_img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


# if __name__ == "__main__":
#     test_dataset = EyeTrackData("/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_vis_sim/s20_board_eye_tracking", transform = transform)

#     test_dataloader = DataLoader(test_dataset, batch_size=32)

#     progress_bar = tqdm(test_dataloader)
#     for board_img, eye_img, label in progress_bar:
#         import ipdb; ipdb.set_trace()
#         x = 0
