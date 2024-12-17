import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import os

from src.data import OrigPlank, OrigPlank2, transform
from src.models import ResNet50Regressor, ResNet50VAERegressor, ResNet18VAERegressor
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import cv2

torch.manual_seed(42)

root_dir = "./trajectories"
os.makedirs(root_dir, exist_ok = True)

def test(model, dataloader, device='cuda', modelname = "resnet50vae_np_reg"):
    os.makedirs(os.path.join(root_dir, modelname + "_latent_128"), exist_ok = True)
    model.eval()

    progress_bar = tqdm(dataloader)
    count = 0
    with torch.no_grad():
        for images, labels, pos in progress_bar:
            images, labels, pos = images.to(device).float(), labels.to(device), pos.to(device)
            # feature_maps, outputs, _, _ = model(images)
            mu, logvar, predicted_trajectories = model(images)
        
            images = images * 255
            images = images.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)

            for i, img in enumerate(images):
                filename = f"img_{count}_basket{labels[i] + 1}.png"
                fig, ax = plt.subplots()
                ax.imshow(img[::-1], origin='lower')
    
                curr_pos = pos[i].cpu().numpy().reshape(-1, 2)
                plt.plot((curr_pos[:,0] * 11.2 +111.5), (curr_pos[:, 1] *11.2 + 112), "o-", linewidth=4, markersize=3, alpha=0.3, markerfacecolor=(0.0, 1.0, 1.0, 0.1), color='#ed0000')
                # plt.savefig(os.path.join(root_dir, modelname, filename), bbox_inches='tight', pad_inches=0)
                # Loop over each trajectory for this image
                for traj in predicted_trajectories[:30, i, :]:  # Access each trajectory for this image
                    traj = traj.cpu().numpy().reshape(-1, 2)  # Reshape to pairs of (x, y)

                    # Plot the trajectory as a line connecting the points
                    plt.plot((traj[:, 0] * 11.2 +111.5), (traj[:, 1] *11.2 + 112), "o-", linewidth=4, markersize=3, alpha=0.3, markerfacecolor=(0.0, 1.0, 1.0, 0.0), color='#000000ff')
                    # ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, color='blue', linewidth=1)

                plt.axis('off')
                plt.savefig(os.path.join(root_dir, modelname + "_latent_128", filename), bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                count += 1

def main():


    device='cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = OrigPlank2("/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/test", train = False, transform = transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    modelname = "resnet18vae_np_reg"
    model = ResNet18VAERegressor(num_classes=32, inference = True).to("cuda:0")

    load_path = os.path.join("pretrained_models", modelname, "trial_5_best_model.pth")
    modelname += "_trail_5_30"
    model.load_state_dict(torch.load(load_path, map_location = "cuda:0"))

    # import ipdb; ipdb.set_trace()
    test(model, test_dataloader, device, modelname)

    
if __name__ == "__main__":
    main()
    
    
