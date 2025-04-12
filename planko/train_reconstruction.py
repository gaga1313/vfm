import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import wandb
import os
from argparse import ArgumentParser

from src.data import OrigPlank, OrigPlank2, transform
from src.models import (
    ResNet50Regressor,
    ResNet50VAERegressor,
    ResNet18VAERegressor,
    ResNet18VAE,
)
from src.utils import MetricLogger, KLD

parser = ArgumentParser(description="Visual Foundation Model Training")
parser.add_argument(
    "--vae_training", action="store_true", default=False, help="training strategy"
)
parser.add_argument("--w2", type=float, default=1e-3, help="KLD loss weight")

kld = KLD(std=1.414).kld


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
):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)

    for step, (images, labels, pos) in enumerate(progress_bar):
        images, labels, pos = (
            images.to(device).float(),
            labels.to(device),
            pos.to(device),
        )

        optimizer.zero_grad()

        if args.vae_training:
            outputs, mu, logvar = model(images)
        else:
            outputs, _ = model(images)

        mse_loss = criterion(outputs, images)
        metric_logger.add("train_mse_loss", mse_loss.item())

        if args.vae_training:
            loss_kld = args.w2 * kld(mu, logvar) / labels.size(0)
            final_loss = mse_loss + loss_kld
            batch_loss = final_loss.item()
            final_loss.backward()
            metric_logger.add("train_loss", batch_loss)
            metric_logger.add("train_kld_loss", loss_kld.item())
        else:
            mse_loss.backward()

        optimizer.step()

        if rank == 0 and (step + 1) % log_interval == 0:
            global_step += 1
            if args.vae_training:
                wandb.log(
                    {
                        "train_mse_loss": mse_loss.item(),
                        "epoch": epoch + 1,
                        "train_kld_loss": loss_kld.item(),
                        "train_loss": batch_loss,
                    },
                    step=global_step,
                )
            else:
                wandb.log(
                    {
                        "train_mse_loss": mse_loss.item(),
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
        if args.vae_training:
            progress_bar.set_postfix(
                {
                    "Train MSE Loss": mse_loss.item(),
                    "Train KLD": loss_kld.item(),
                    "Train Loss": batch_loss,
                }
            )
            avg_loss = metric_logger.global_average("train_loss")
        else:
            progress_bar.set_postfix({"Train MSE Loss": mse_loss.item()})
            avg_loss = metric_logger.global_average("train_mse_loss")

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
        for step, (images, labels, pos) in enumerate(progress_bar):
            images, labels, pos = (
                images.to(device).float(),
                labels.to(device),
                pos.to(device),
            )

            if args.vae_training:
                outputs, mu, logvar = model(images)
            else:
                outputs, _ = model(images)

            mse_loss = criterion(outputs, images)
            metric_logger.add("test_mse_loss", mse_loss.item())

            if args.vae_training:
                loss_kld = args.w2 * kld(mu, logvar) / labels.size(0)
                final_loss = mse_loss + loss_kld
                batch_loss = final_loss.item()
                metric_logger.add("test_kld_loss", loss_kld.item())
                metric_logger.add("test_loss", final_loss.item())

            if args.vae_training:
                progress_bar.set_postfix(
                    {
                        "Test Loss": batch_loss,
                        "Test KLD": loss_kld.item(),
                        "Test MSE Loss": mse_loss.item(),
                    }
                )
            else:
                progress_bar.set_postfix({"Test MSE Loss": mse_loss.item()})

    avg_loss = metric_logger.global_average("test_loss")
    avg_mse_loss = metric_logger.global_average("test_mse_loss")
    avg_kld_loss = metric_logger.global_average("test_kld_loss")
    if rank == 0:
        print(f"Test MSE Loss: {avg_mse_loss:.4f}")
        global_step += 1
        if args.vae_training:
            wandb.log(
                {
                    "test_mse_loss": avg_mse_loss,
                    "test_kld_loss": avg_kld_loss,
                    "test_loss": avg_loss,
                },
                step=global_step,
            )
        else:
            wandb.log({"test_mse_loss": avg_mse_loss}, step=global_step)
    return avg_loss, global_step


def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_step = 0
    args = parser.parse_args()

    epochs = 100
    print(f"Is VAE training: {args.vae_training}")

    # shifted to new data
    train_dataset = OrigPlank2(
        "/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/train",
        train=True,
        transform=transform,
    )
    test_dataset = OrigPlank2(
        "/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/test",
        train=False,
        transform=transform,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

    if args.vae_training:
        modelname = "resnet18vae_np_recon"
        model = ResNet18VAE().to(device)
    else:
        modelname = "resnet18_np_recon"
        import ipdb

        ipdb.set_trace()
        model = ResNet18VAE().to(device)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=False
    )

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float("inf")
    save_path = os.path.join("pretrained_models", modelname)
    dataset_name = "board_reconstruction"
    os.makedirs(save_path, exist_ok=True)
    trial_number = str(len(os.listdir(save_path)))
    modelname = modelname + "_" + dataset_name + "_" + trial_number

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
        )

        if rank == 0:
            print(f"Epoch {e+1}/{epochs} - Train MSE Loss: {train_loss:.4f}")
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
