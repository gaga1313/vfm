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
from src.data import TestPlank, transform
from src.models import CConvVAE2, Conv2REG
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

parser = ArgumentParser(description="Visual Foundation Model Training")
parser.add_argument(
    "--vae_test", action="store_true", default=False, help="training strategy"
)

xbins = np.arange(-28, 28.56, 0.56)
ybins = np.arange(-8, 8.16, 0.16)

x_bin_coor = {idx: xbins[i] for idx, i in enumerate(range(100))}
y_bin_coor = {idx: ybins[i] for idx, i in enumerate(range(100))}


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def plot_trajectory(model, images, uncertain, labels, n_samples=1):
    predicted_trajectories = model.inference2(images, n_samples=n_samples)
    images = images * 255
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

    wandb_images = []
    for i, img in enumerate(images):
        fig, ax = plt.subplots()
        ax.imshow(img[::-1], origin="lower")

        for traj in predicted_trajectories[
            i, :
        ]:  # Access each trajectory for this image
            indices = traj.detach().cpu().numpy()  # Reshape to pairs of (x, y)

            # indices = np.where(traj >= 0.5)[0]

            ybin, xbin = indices[16:], indices[:16]
            traj = np.array(
                [[x_bin_coor[x], y_bin_coor[ybin[idx]]] for idx, x in enumerate(xbin)]
            )

            # Plot the trajectory as a line connecting the points
            plt.plot(
                (traj[:, 0] * 11.2 + 111.5),
                (traj[:, 1] * 11.2 + 112),
                "o-",
                linewidth=4,
                markersize=3,
                alpha=0.3,
                markerfacecolor=(0.0, 1.0, 1.0, 0.0),
                color="#000000ff",
            )
            # ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, color='blue', linewidth=1)

        buf = io.BytesIO()
        plt.legend([f"uncertain: {uncertain[i]}", f"basket: {labels[i]}"])
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        wandb_img = wandb.Image(Image.open(buf))
        wandb_images.append(wandb_img)

    return wandb_images


def trajectory_corr(predicted_trajectories):
    variance = []
    trajectory_difference = np.zeros(predicted_trajectories.shape[0])
    trajectory_length = np.zeros(predicted_trajectories.shape[0])
    for i in range(32):
        image_traj = predicted_trajectories[i, :]
        pairwise_distances = pdist(image_traj, metric="cosine")
        trajectory_difference[i] = 1 - np.mean(pairwise_distances)

        trajectory_var = np.var(image_traj, axis=0)
        # # trajectory_var = np.hstack([trajectory_var[8:16], trajectory_var[24:32]])
        variance.append(trajectory_var)

    return (
        np.mean(np.array(variance), axis=1),
        trajectory_difference,
    )  # Convert list to numpy array


import numpy as np


def compute_average_trajectory_length(pred_trajectories):
    """
    Compute the average length of trajectory across 1000 samples for each image.

    Parameters:
    - pred_trajectories: A numpy array of shape (32, 1000, 32),
      where each trajectory consists of 32 points.

    Returns:
    - avg_lengths: A numpy array of shape (32,) containing the average trajectory length
      for each image.
    """
    num_images, num_samples, num_points = pred_trajectories.shape  # (32, 1000, 32)

    avg_lengths = np.zeros(num_images)

    for img_idx in range(num_images):
        total_length = 0

        for sample_idx in range(num_samples):
            # Extract x and y indices for the trajectory
            x_indices = pred_trajectories[img_idx, sample_idx, :16]
            y_indices = pred_trajectories[img_idx, sample_idx, 16:]

            # Convert to Cartesian coordinates
            x_coords = np.array([x_bin_coor[idx] for idx in x_indices])
            y_coords = np.array([y_bin_coor[idx] for idx in y_indices])

            # Compute pairwise Euclidean distances between consecutive points
            distances = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)

            # Sum the distances to get the total trajectory length
            trajectory_length = np.sum(distances)
            total_length += trajectory_length

        # Compute the average length across 1000 samples
        avg_lengths[img_idx] = total_length / num_samples

    return avg_lengths


def test2(args, model, dataloader, n_samples, device="cuda"):
    model.eval()
    progress_bar = tqdm(dataloader)

    all_uncertain = []
    all_rt = []
    all_variance = []
    all_difference = []
    avg_traj_len = []
    with torch.no_grad():
        for step, (images, labels, uncertain, rt) in enumerate(progress_bar):
            images = images.to(device).float()
            predicted_trajectories = (
                model.inference2(images, n_samples=n_samples).detach().cpu().numpy()
            )
            images = (
                (images * 255)
                .permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            # Collect variances
            # variance, difference = trajectory_corr(predicted_trajectories)
            trajectory_length = compute_average_trajectory_length(
                predicted_trajectories
            )
            # all_variance.append(variance)
            all_uncertain.append(uncertain.cpu().numpy())
            all_rt.append(rt.cpu().numpy())
            # all_difference.append(difference)
            avg_traj_len.append(trajectory_length)

        # Convert lists to arrays for plotting
        # all_variance = np.concatenate(all_variance, axis=0)
        all_uncertain = np.concatenate(all_uncertain, axis=0)
        all_rt = np.concatenate(all_rt, axis=0)
        # all_difference = np.concatenate(all_difference, axis = 0)
        avg_traj_len = np.concatenate(avg_traj_len, axis=0)

        # Create scatter plots with best-fit line
        # corr_uncertain, _ = pearsonr(all_variance, all_uncertain)
        # corr_rt, _ = pearsonr(all_variance, all_rt)

        # diff_corr_uncertain, _ = pearsonr(all_difference, all_uncertain)
        # diff_corr_rt, _ = pearsonr(all_difference, all_rt)

        tl_corr_uncertain, _ = pearsonr(avg_traj_len, all_uncertain)
        tl_corr_rt, _ = pearsonr(avg_traj_len, all_rt)

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # sns.regplot(x=all_variance, y=all_uncertain, ax=axes[0], scatter_kws={"s": 10})
        # axes[0].set_title("Variance vs. Uncertainty")
        # axes[0].set_xlabel("Variance")
        # axes[0].set_ylabel("Uncertainty")
        # axes[0].text(0.05, 0.9, f"Corr: {corr_uncertain:.2f}", transform=axes[0].transAxes, fontsize=12, color="red")

        # sns.regplot(x=all_variance, y=all_rt, ax=axes[1], scatter_kws={"s": 10})
        # axes[1].set_title("Variance vs. RT")
        # axes[1].set_xlabel("Variance")
        # axes[1].set_ylabel("RT")
        # axes[1].text(0.05, 0.9, f"Corr: {corr_rt:.2f}", transform=axes[1].transAxes, fontsize=12, color="red")

        # # Log to wandb
        # log = {"Correlation test": [wandb.Image(fig)]}
        # plt.close(fig)  # Free memory

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # sns.regplot(x=all_difference, y=all_uncertain, ax=axes[0], scatter_kws={"s": 10})
        # axes[0].set_title("Cosine vs. Uncertainty")
        # axes[0].set_xlabel("Cosine")
        # axes[0].set_ylabel("Uncertainty")
        # axes[0].text(0.05, 0.9, f"Corr: {diff_corr_uncertain:.2f}", transform=axes[0].transAxes, fontsize=12, color="red")

        # sns.regplot(x=all_difference, y=all_rt, ax=axes[1], scatter_kws={"s": 10})
        # axes[1].set_title("Cosine vs. RT")
        # axes[1].set_xlabel("Cosine")
        # axes[1].set_ylabel("RT")
        # axes[1].text(0.05, 0.9, f"Corr: {diff_corr_rt:.2f}", transform=axes[1].transAxes, fontsize=12, color="red")

        # # Log to wandb
        # log["Correlation test"].append(wandb.Image(fig))
        # plt.close(fig)  # Free memory

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.regplot(x=avg_traj_len, y=all_uncertain, ax=axes[0], scatter_kws={"s": 10})
        axes[0].set_title("Trajectory Length vs. Uncertainty")
        axes[0].set_xlabel("Trajectory Length")
        axes[0].set_ylabel("Uncertainty")
        axes[0].text(
            0.05,
            0.9,
            f"Corr: {tl_corr_uncertain:.2f}",
            transform=axes[0].transAxes,
            fontsize=12,
            color="red",
        )

        sns.regplot(x=avg_traj_len, y=all_rt, ax=axes[1], scatter_kws={"s": 10})
        axes[1].set_title("Trajectory Length vs. RT")
        axes[1].set_xlabel("Trajectory Length")
        axes[1].set_ylabel("RT")
        axes[1].text(
            0.05,
            0.9,
            f"Corr: {tl_corr_rt:.2f}",
            transform=axes[1].transAxes,
            fontsize=12,
            color="red",
        )

        # Log to wandb
        log = {"Correlation test": [wandb.Image(fig)]}
        plt.close(fig)  # Free memory

        wandb.log(log)

    return


def test(
    args,
    model,
    dataloader,
    n_samples,
    device="cuda",
):
    model.eval()
    progress_bar = tqdm(dataloader)

    with torch.no_grad():
        for step, (images, labels, uncertain, rt) in enumerate(progress_bar):
            images = images.to(device).float()

            logs = {}
            logs["Test Images"] = plot_trajectory(
                model, images, uncertain, labels, n_samples=n_samples
            )
            wandb.log(logs, step=step)
    return


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    batch_size = 32
    latent_size = 4
    kld = 1.0
    print(f"Is VAE testing: {args.vae_test}")
    test_dataset = TestPlank(
        "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_vfm/vfm/data/RT_Planko_RNN.csv",
        transform=transform,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if args.vae_test:
        modelname = "ccvae2_grid"
        model = CConvVAE2(
            image_channels=3, image_size=224, trajectory_dim=32, latent_dim=latent_size
        ).to(device)
        load_path = os.path.join(
            "pretrained_models", modelname, "trial_3_best_model.pth"
        )
        modelname = (
            "trajectorylen_coor" + str(latent_size) + "_" + str(kld) + "_1_" + modelname
        )
        model.load_state_dict(torch.load(load_path, map_location="cuda:0"))
    else:
        modelname = "conv2_det_grid"
        model = Conv2REG(image_channels=3, image_size=224).to(device)
        load_path = os.path.join(
            "pretrained_models", modelname, "trial_0_best_model.pth"
        )
        modelname = "trajectorylen_coor" + str(latent_size) + "_" + modelname
        model.load_state_dict(torch.load(load_path, map_location="cuda:0"))

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
        50,
        device,
    )

    test2(
        args,
        model,
        test_dataloader,
        50,
        device,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
