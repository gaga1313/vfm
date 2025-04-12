import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import wandb
import os
from argparse import ArgumentParser
import json
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.data import OrigPlank3, OrigPlank4, OrigPlank5, transform
from src.models import CVAE, MLEREG, CConvVAE, ConvREG, CConvVAE2, Conv2REG
from src.utils import MetricLogger, KLD

parser = ArgumentParser(description="Visual Foundation Model Training")
parser.add_argument(
    "--vae_test", action="store_true", default=False, help="training strategy"
)
parser.add_argument("--w2", default=1e-4, help="KLD loss weight")

kld = KLD(std=1.0).kld
xbins = np.arange(-28, 28.56, 0.56)
ybins = np.arange(-8, 8.16, 0.16)

x_bin_coor = {idx: xbins[i] for idx, i in enumerate(range(100))}
y_bin_coor = {idx: ybins[i] for idx, i in enumerate(range(100))}


def plot_trajectory(model, images, pos, n_samples=1):
    predicted_trajectories = model.inference(images, n_samples=n_samples)
    images = images * 255
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

    wandb_images = []
    for i, img in enumerate(images):
        fig, ax = plt.subplots()
        ax.imshow(img[::-1], origin="lower")

        indices = pos[i].cpu().numpy()
        # indices = np.where(curr_labels == 1)[0]

        # ybin, xbin = indices // 56, indices % 56
        ybin, xbin = indices[16:], indices[:16]
        curr_pos = np.array(
            [[x_bin_coor[x], y_bin_coor[ybin[idx]]] for idx, x in enumerate(xbin)]
        )

        plt.scatter((curr_pos[:, 0] * 11.2 + 111.5), (curr_pos[:, 1] * 11.2 + 112))
        # plt.savefig(os.path.join(root_dir, modelname, filename), bbox_inches='tight', pad_inches=0)
        # Loop over each trajectory for this image
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

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        wandb_img = wandb.Image(Image.open(buf))
        wandb_images.append(wandb_img)

    return wandb_images


def test(
    args,
    model,
    dataloader,
    criterion,
    metric_logger,
    device="cuda",
    global_step=None,
    log_interval=10,
    save_images=True,
):
    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Epoch {1}/{1}")

    with torch.no_grad():
        for step, (images, labels, pos, board) in enumerate(progress_bar):
            images, labels, pos = (
                images.to(device).float(),
                labels.to(device),
                pos.to(device),
            )

            if args.vae_test:
                outputs, mu, logvar = model(images, pos)
            else:
                outputs = model(images)

            cce_loss = criterion(outputs.permute(0, 2, 1), pos.long())
            metric_logger.add("test_cce_loss", cce_loss.item())

            if args.vae_test:
                loss_kld = args.w2 * kld(mu, logvar) / labels.size(0)
                final_loss = cce_loss + loss_kld
                batch_loss = final_loss.item()
                metric_logger.add("test_kld_loss", loss_kld.item())
                metric_logger.add("test_loss", final_loss.item())

            if (step + 1) % log_interval == 0:
                global_step += 1
                if args.vae_test:
                    logs = {
                        "test_cce_loss": cce_loss.item(),
                        "test_kld_loss": loss_kld.item(),
                        "test_loss": batch_loss,
                    }
                else:
                    logs = {
                        "test_cce_loss": cce_loss.item(),
                    }
                if save_images and (step + 1) % 10 == 0:
                    logs["Test Images"] = plot_trajectory(
                        model, images[:12], pos[:12], n_samples=1
                    )

                wandb.log(logs, step=global_step)

            if args.vae_test:
                progress_bar.set_postfix(
                    {
                        "Test Loss": batch_loss,
                        "Test KLD": loss_kld.item(),
                        "Test CCE Loss": cce_loss.item(),
                    }
                )
            else:
                progress_bar.set_postfix({"Test CCE Loss": cce_loss.item()})

    avg_loss = metric_logger.global_average("test_loss")

    return avg_loss, global_step


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_step = 0
    args = parser.parse_args()
    args.w2 = float(args.w2)
    batch_size = 32
    latent_size = 4
    noise = False
    normalization = False

    print(f"Is VAE test: {args.vae_test}")

    test_dataset = OrigPlank5(
        "/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/singbaskv3/test",
        train=False,
        is_val=False,
        transform=transform,
        noise=noise,
        norm=normalization,
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
        modelname = "test_sbk" + str(latent_size) + "_1.0_" + "_1_" + modelname
        model.load_state_dict(torch.load(load_path, map_location="cuda:0"))
    else:
        modelname = "conv2_det_grid"
        model = Conv2REG(image_channels=3, image_size=224).to(device)
        load_path = os.path.join(
            "pretrained_models", modelname, "trial_1_best_model.pth"
        )
        modelname = "test_sbk" + str(latent_size) + "_" + modelname
        model.load_state_dict(torch.load(load_path, map_location="cuda:0"))

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    print(model)

    wandb.init(project="vfm", entity="gaga13", name=modelname)

    criterion = nn.CrossEntropyLoss()

    metric_logger = MetricLogger()

    # Training the model
    test_loss, global_step = test(
        args,
        model,
        test_dataloader,
        criterion,
        metric_logger,
        device,
        global_step=global_step,
        save_images=True,
    )


if __name__ == "__main__":
    main()
