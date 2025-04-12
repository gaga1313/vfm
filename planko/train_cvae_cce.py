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
from src.data import OrigPlank3, transform
from src.models import CVAE, MLEREG
from src.utils import MetricLogger, KLD

parser = ArgumentParser(description="Visual Foundation Model Training")
parser.add_argument(
    "--vae_training", action="store_true", default=False, help="training strategy"
)
parser.add_argument("--w2", default=1e-4, help="KLD loss weight")
parser.add_argument(
    "--trial",
    default=None,
    help="How many times have u've trained conditional vae before?",
)

kld = KLD(std=1.0).kld

x_bin_coor = {idx: i + 0.5 for idx, i in enumerate(range(-28, 28))}
y_bin_coor = {idx: i + 0.5 for idx, i in enumerate(range(-8, 8))}


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def plot_trajectory(model, images, pos, board, n_samples=1):
    predicted_trajectories = model.module.inference(board, n_samples=n_samples)
    images = images * 255
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

    wandb_images = []
    for i, img in enumerate(images):
        fig, ax = plt.subplots()
        ax.imshow(img[::-1], origin="lower")

        indices = pos[i].cpu().numpy()
        # indices = np.where(curr_labels == 1)[0]

        ybin, xbin = indices // 56, indices % 56
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

            ybin, xbin = indices // 56, indices % 56
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
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        wandb_img = wandb.Image(Image.open(buf))
        wandb_images.append(wandb_img)

    return wandb_images


# Training function with a progress bar
def train_one_epoch(
    args,
    model,
    dataloader,
    criterion,
    optimizer,
    metric_logger,
    epoch,
    epochs=10,
    device="cuda",
    rank=0,
    log_interval=10,
    global_step=None,
    save_images=False,
):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)

    for step, (images, labels, pos, board) in enumerate(progress_bar):
        images, labels, pos, board = (
            images.to(device).float(),
            labels.to(device),
            pos.to(device),
            board.to(device),
        )

        optimizer.zero_grad()

        if args.vae_training:
            outputs, mu, logvar = model(board, pos)
        else:
            outputs = model(board)

        cce_loss = criterion(outputs.permute(0, 2, 1), pos.long())

        metric_logger.add("train_cce_loss", cce_loss.item())

        if args.vae_training:
            loss_kld = args.w2 * kld(mu, logvar) / labels.size(0)
            final_loss = cce_loss + loss_kld
            batch_loss = final_loss.item()
            final_loss.backward()
            metric_logger.add("train_loss", batch_loss)
            metric_logger.add("train_kld_loss", loss_kld.item())
        else:
            cce_loss.backward()

        optimizer.step()

        if rank == 0 and (step + 1) % log_interval == 0:
            global_step += 1
            if args.vae_training:
                logs = {
                    "train_cce_loss": cce_loss.item(),
                    "epoch": epoch + 1,
                    "train_kld_loss": loss_kld.item(),
                    "train_loss": batch_loss,
                }
            else:
                logs = {
                    "train_cce_loss": cce_loss.item(),
                    "epoch": epoch + 1,
                }
            if save_images:
                logs["Train Images"] = plot_trajectory(
                    model, images[:12], pos[:12], board[:12], n_samples=5
                )

            wandb.log(logs, step=global_step)

        if args.vae_training:
            progress_bar.set_postfix(
                {
                    "Train CCE Loss": cce_loss.item(),
                    "Train KLD": loss_kld.item(),
                    "Train Loss": batch_loss,
                }
            )
            avg_loss = metric_logger.global_average("train_loss")
        else:
            progress_bar.set_postfix({"Train CCE Loss": cce_loss.item()})
            avg_loss = metric_logger.global_average("train_cce_loss")

    return avg_loss, global_step


def test(
    args,
    model,
    dataloader,
    criterion,
    metric_logger,
    epoch,
    epochs,
    device="cuda",
    rank=0,
    global_step=None,
):
    model.eval()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)

    with torch.no_grad():
        for step, (images, labels, pos, board) in enumerate(progress_bar):
            images, labels, pos, board = (
                images.to(device).float(),
                labels.to(device),
                pos.to(device),
                board.to(device),
            )

            if args.vae_training:
                outputs, mu, logvar = model(board, pos)
            else:
                outputs = model(board)

            cce_loss = criterion(outputs.permute(0, 2, 1), pos.long())
            metric_logger.add("test_cce_loss", cce_loss.item())

            if args.vae_training:
                loss_kld = args.w2 * kld(mu, logvar) / labels.size(0)
                final_loss = cce_loss + loss_kld
                batch_loss = final_loss.item()
                metric_logger.add("test_kld_loss", loss_kld.item())
                metric_logger.add("test_loss", final_loss.item())

            if args.vae_training:
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
    avg_cce_loss = metric_logger.global_average("test_cce_loss")
    avg_kld_loss = metric_logger.global_average("test_kld_loss")
    if rank == 0:
        print(f"Test CCE Loss: {avg_cce_loss:.4f}")
        global_step += 1
        if args.vae_training:
            wandb.log(
                {
                    "test_cce_loss": avg_cce_loss,
                    "test_kld_loss": avg_kld_loss,
                    "test_loss": avg_loss,
                },
                step=global_step,
            )
        else:
            wandb.log({"test_cce_loss": avg_cce_loss}, step=global_step)
    return avg_loss, global_step


def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_step = 0
    args = parser.parse_args()
    args.w2 = float(args.w2)
    epochs = 50
    batch_size = 32
    latent_size = 128
    learning_rate = 0.001
    noise = False
    normalization = False
    save_images = True

    print(f"Is VAE training: {args.vae_training}")

    # shifted to new data
    train_dataset = OrigPlank3(
        "/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/train",
        train=True,
        transform=transform,
        noise=noise,
        norm=normalization,
    )
    test_dataset = OrigPlank3(
        "/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/test",
        train=False,
        transform=transform,
        noise=noise,
        norm=normalization,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler
    )

    if args.vae_training:
        modelname = "cvae_grid"
        model = CVAE(board_dim=105, trajectory_dim=16, latent_dim=latent_size).to(
            device
        )
    else:
        modelname = "mle_trajectory"
        model = MLEREG(board_dim=105, trajectory_dim=16).to(device)

    save_path = os.path.join("pretrained_models", modelname)

    metadata_path = os.path.join("metadata", modelname)
    os.makedirs(metadata_path, exist_ok=True)

    if not args.trial:
        os.makedirs(save_path, exist_ok=True)
        trial_number = str(len(os.listdir(save_path)))

    else:
        trial_number = str(args.trial)

    if rank == 0:
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params}")

        meta_data = {
            "batch_size": batch_size,
            "trail_number": trial_number,
            "latent_dimension": latent_size,
            "learning rate": learning_rate,
            "kld_coe(w2)": args.w2,
            "kld_std": 1.0,
            "trajectory noise": noise,
            "trajectory normalization": normalization,
            "loss function": "CCE",
            "trainable_params": trainable_params,
        }

        with open(os.path.join(metadata_path, f"{trial_number}.json"), "w") as file:
            json.dump(meta_data, file)
        print("Meta Data saved!!")

        print(model)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    save_path = os.path.join("pretrained_models", modelname)
    dataset_name = "og_planko_trajectory_board_xys"

    modelname = modelname + "_" + dataset_name + "_" + trial_number + "_" + str(args.w2)

    if rank == 0:
        print(f"Total Train Images: {len(train_dataset)}")
        print(f"Total Test Images: {len(test_dataset)}")
        wandb.init(project="vfm", entity="gaga13", name=modelname)
        wandb.watch(model, criterion, log="all", log_freq=10)

    metric_logger = MetricLogger()

    # Training the model
    for e in range(epochs):
        train_sampler.set_epoch(e)
        train_loss, global_step = train_one_epoch(
            args,
            model,
            train_dataloader,
            criterion,
            optimizer,
            metric_logger,
            e,
            epochs=epochs,
            device=device,
            rank=rank,
            global_step=global_step,
            save_images=save_images,
        )

        if rank == 0:
            print(f"Epoch {e+1}/{epochs} - Train CCE Loss: {train_loss:.4f}")
        test_loss, global_step = test(
            args,
            model,
            test_dataloader,
            criterion,
            metric_logger,
            e,
            epochs,
            device,
            rank=rank,
            global_step=global_step,
        )

        if rank == 0 and test_loss < best_loss:
            best_loss = test_loss
            torch.save(
                unwrap_model(model).state_dict(),
                os.path.join(save_path, "trial_" + trial_number + "_best_model.pth"),
            )
            print(f"Best model saved with Test Loss: {best_loss:.2f}")

    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
