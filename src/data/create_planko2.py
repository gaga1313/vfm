from omegaconf import DictConfig, ValueNode, ListConfig
from torch.utils.data import Dataset, DataLoader
import torch
from os import listdir, scandir
from os.path import isfile, join
from PIL import Image
# import random
import dgread
import numpy as np
# from tqdm import tqdm
# import cv2
# from transforms import transform

def load_image(directory):
    return Image.open(directory).convert('RGB')

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_equi_points(world_desc):
    x_positions = [world_desc['ball_pos_x'][0]]
    y_positions = [world_desc['ball_pos_y'][0]]
    for i in range(0,len(world_desc['ball_pos_y'])):
        # if world_desc['ball_pos_y'][i] < -8.5:
        #   break
        dist = distance(x_positions[-1], y_positions[-1], world_desc['ball_pos_x'][i], world_desc['ball_pos_y'][i])
        if dist > 1:
            x_positions.append(world_desc['ball_pos_x'][i])
            y_positions.append(world_desc['ball_pos_y'][i])
            #print(i)
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    return x_positions, y_positions


def get_equi_points_ART(world_desc):

    xes = []
    yes = []
    for (x,y) in zip(world_desc['ball_pos_x'], world_desc['ball_pos_y']):
        xes.append(x)
        yes.append(y)
        if y < -8.5:
            break

    xes = np.array(xes)
    yes = np.array(yes)

    total_dist = np.sum(np.sqrt((xes[1:] - xes[:-1])**2 + (yes[1:] - yes[:-1])**2))
    sep_distance = total_dist / 18
    x_positions = [xes[0]]
    y_positions = [yes[0]]
    for i in range(0,len(yes)):
        if world_desc['ball_pos_y'][i] < -8.5:
          break
        dist = distance(x_positions[-1], y_positions[-1], xes[i], yes[i])
        if dist >= sep_distance:
            x_positions.append(xes[i])
            y_positions.append(yes[i])

    for i in range(0, 16-len(x_positions)):
        x_positions.append(x_positions[-1])
        y_positions.append(y_positions[-1])
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    return x_positions, y_positions


class OrigPlank2(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, transform = None, **kwargs
    ):
        super().__init__()
        # self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.ball_pos_idx = 16 #self.cfg.ball_pos_train_idx # number between 1 to 16

        import time
        print("started loading")
        t0 = time.time()
        #self.file_list = [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f))]
        if isinstance(self.path, ListConfig):
            self.file_list = []
            for p in self.path:
                self.file_list += [join(p, f) for f in scandir(p) if f.is_file()] #[:20000]
        else:
            self.file_list = [join(self.path, f) for f in scandir(self.path) if f.is_file()]
        # if not self.train:
        #    self.file_list = sorted(self.file_list)
        #print(self.file_list[:20])
        t1 = time.time()
        print("finished loading in ")
        print(t1-t0)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        file_name = self.file_list[index]
        img = load_image(file_name)
        if self.transform:
            img = self.transform(img)
        
        label = [0 if "left" in file_name else 1][0]
        #label = int(random.random() >0.5)

        dgz_name = file_name.replace("train", "world").replace("test", "world").replace(".png", ".dgz").replace("png", "world")
        # print(dgz_name)
        # import pdb; pdb.set_trace()
        world_desc = dgread.dgread(dgz_name)
        # x_pos = world_desc['ball_pos_x'][self.ball_pos_idx*38]
        # y_pos = world_desc['ball_pos_y'][self.ball_pos_idx*38]

        # x_positions, y_positions = get_equi_points(world_desc)
        x_positions, y_positions = get_equi_points_ART(world_desc)
        
        # print(len(x_positions))
        # if len(x_positions) <= self.ball_pos_idx:
        #     # if self.cfg.write_inference_to_file:
        #     #     return img, [torch.tensor([x_positions[-1], y_positions[-1]]), file_name.split('/')[-1]]

        #     return img, torch.tensor([x_positions[-1], y_positions[-1]])

        # x_pos = x_positions[self.ball_pos_idx]
        # y_pos = y_positions[self.ball_pos_idx]
        # import ipdb; ipdb.set_trace()
        #[print(world_desc['ball_pos_x'][i], world_desc['ball_pos_y'][i]) for i in range(100)]
        
        # if self.cfg.write_inference_to_file:
        #     return img, [torch.tensor([x_pos, y_pos]), file_name.split('/')[-1]]

        board_coordinates = np.hstack([world_desc['tx'], world_desc['ty'], world_desc['sx'], world_desc['sy'], world_desc['spin']])

        positions = []
        for i in range(16):
            positions.append(x_positions[i])
            positions.append(y_positions[i])
        positions = np.array(positions)

        
        
        return img, label, torch.tensor(positions), torch.tensor(board_coordinates)
    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"

# if __name__ == "__main__":
#     train_dataset = OrigPlank2("/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/train", train = False, transform = transform)

#     train_dataloader = DataLoader(train_dataset, batch_size=32)

#     progress_bar = tqdm(train_dataloader)

#     for img, label, pos, board, world in progress_bar:
#         import ipdb;ipdb.set_trace()
#         # for i in range(32):
#         #     cv2.imwrite(f"../../trash/{i}.png", cv2.cvtColor(img[i].permute(1,2,0).numpy()*255, cv2.COLOR_RGB2BGR))
#         #     print('image_saved')
#         # import ipdb;ipdb.set_trace()

    
    