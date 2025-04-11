import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import os
from argparse import ArgumentParser
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.data import EyeTrackData, transform
from src.models import CConvVAE2, Conv2REG
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

parser = ArgumentParser(description = "Visual Foundation Model Training")
parser.add_argument("--vae_test", action = "store_true", default = False, help = "training strategy")

xbins = np.arange(-28, 28.56, 0.56)
ybins = np.arange(-8, 8.16, 0.16)

x_bin_coor = {idx: xbins[i] for idx, i in enumerate(range(100))}
y_bin_coor = {idx: ybins[i] for idx, i in enumerate(range(100))}

def unwrap_model(model):
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    return model

import io
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image

def plot_trajectory(model, images, eye_images, labels, n_samples=1):
    predicted_trajectories = model.inference2(images, n_samples=n_samples)
    
    images = (images * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    eye_images = (eye_images * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    wandb_images = []
    
    for i, img in enumerate(images):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[1].imshow(eye_images[i][::-1], origin='lower')
        axes[0].imshow(img[::-1], origin='lower')

        first_traj = True  # Reset for each image
        has_legend = False  # Track if any labeled trajectory is added

        for traj in predicted_trajectories[i, :]:  
            indices = traj.detach().cpu().numpy()
            ybin, xbin = indices[16:], indices[:16]
            traj = np.array([[x_bin_coor[x], y_bin_coor[ybin[idx]]] for idx, x in enumerate(xbin)])

            label = f'Basket: {labels[i]}' if first_traj else "_"  # Assign an actual label only once
            first_traj = False  # Only the first trajectory gets a label
            has_legend = True  # At least one labeled trajectory exists

            axes[0].plot(
                (traj[:, 0] * 11.2 + 111.5), (traj[:, 1] * 11.2 + 112),
                "o-", linewidth=4, markersize=3, alpha=0.3, 
                markerfacecolor=(0.0, 1.0, 1.0, 0.0), color='#eb8c34', label=label
            )
        
        if has_legend:  # Add legend only if at least one labeled trajectory is plotted
            axes[0].legend(loc='upper right', fontsize=10)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        wandb_img = wandb.Image(Image.open(buf))
        wandb_images.append(wandb_img)

    return wandb_images


import numpy as np
import matplotlib.pyplot as plt

def plot_variance_heatmap(model, images, eye_images, n_samples):

    predicted_trajectories = model.inference2(images, n_samples=n_samples)
    
    images = (images * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    eye_images = (eye_images * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    wandb_images = []
 
    for i, img in enumerate(images):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[1].imshow(eye_images[i][::-1], origin='lower')
        axes[0].imshow(img[::-1], origin='lower')

        # Convert trajectories to numpy
        all_traj = []
        for traj in predicted_trajectories[i]:
            indices = traj.detach().cpu().numpy()
            ybin, xbin = indices[16:], indices[:16]
            traj_points = np.array([[x_bin_coor[x], y_bin_coor[ybin[idx]]] for idx, x in enumerate(xbin)])
            all_traj.append(traj_points)

        all_traj = np.array(all_traj)  # Shape: (num_trajectories, num_points, 2)

        # Compute mean and variance per step
        mean_traj = np.mean(all_traj, axis=0)
        var_traj = np.var(all_traj, axis=0)
        # import ipdb; ipdb.set_trace()
        # Normalize variance for visualization
        var_norm = var_traj / np.max(var_traj)

        # Plot variance-based circles
        for j, (x, y) in enumerate(mean_traj):
            circle_size = var_norm[j] * 500  # Scale size for visibility
            alpha_value = var_norm[j] * 0.7  # Control transparency
            axes[0].scatter(x * 11.2 + 111.5, y * 11.2 + 112, s=max(max(circle_size), 5), alpha=max(max(alpha_value), 0.25), color='#396ca3')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        wandb_img = wandb.Image(Image.open(buf))
        wandb_images.append(wandb_img)
    return wandb_images


def test(
    args, 
    model, 
    dataloader, 
    device="cuda", 
):
    model.eval()
    progress_bar = tqdm(dataloader)

    with torch.no_grad():
        for step, (board_images, eye_images, labels) in enumerate(progress_bar):
            board_images = board_images.to(device).float()
            eye_images = eye_images.float()
                
            logs = {}
            logs['Eye_Track Images'] = plot_variance_heatmap(model, board_images, eye_images, n_samples=50)
            wandb.log(
                    logs,
                    step = step
                )        
    return

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    batch_size = 32
    latent_size = 4
    kld = 1.0
    print(f"Is VAE testing: {args.vae_test}")
    test_dataset = EyeTrackData("/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_vis_sim/s20_board_eye_tracking", transform = transform)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    if args.vae_test:
        modelname = "ccvae2_grid"
        model = CConvVAE2(image_channels= 3, image_size= 224,  trajectory_dim = 32, latent_dim = latent_size).to(device)
        load_path = os.path.join("pretrained_models", modelname, "trial_2_best_model.pth")
        modelname = "S20_eye_tracking" + str(latent_size) + "_" + str(kld) + "_8_" + modelname
        model.load_state_dict(torch.load(load_path, map_location = "cuda:0"))

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    print(model)
  
    wandb.init(project="vfm", entity="gaga13", name=modelname)

    test(
        args,
        model,
        test_dataloader,
        device,
    )



    wandb.finish()

if __name__ == "__main__":
    main()