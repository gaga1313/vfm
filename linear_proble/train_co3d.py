import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from diffusers import AutoencoderKL
import wandb
from tqdm import tqdm
import os
from datetime import datetime

from src.data import ClickMe
from src.models import Head


def setup():
    print(">> Inside setup")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f">> RANK: {rank} initialized!")
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


def train_one_epoch(model, vae, train_loader, criterion, optimizer, epoch, device, rank, log_interval=100, global_step=0):
    model.train()
    total, correct, running_loss = 0, 0, 0

    progress_bar = tqdm(train_loader, position=rank)
    for i, (images, _, labels, _) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215
        latents = latents.view(latents.size(0), -1)
        latents.requires_grad = True
        optimizer.zero_grad()
        outputs = model(latents)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('done optimizer step')

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        accuracy = correct / total
        avg_loss_so_far = running_loss / total
        progress_bar.set_postfix({
            "Train Acc": f"{accuracy:.4f}",
            "Train Loss": f"{avg_loss_so_far:.4f}"
        })

        # if rank == 0 and (i + 1) % log_interval == 0:
        #     wandb.log({"train_step_loss": loss.item(), "train_step_acc": correct / total}, step=global_step + i)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, global_step + len(train_loader)


def evaluate(model, vae, loader, criterion, device, rank, step, log_interval=100):
    model.eval()
    total, correct, running_loss = 0, 0, 0

    with torch.no_grad():
        progress_bar = tqdm(loader, position=rank)
        for i, (images, _, labels, _) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            latents = vae.encode(images).latent_dist.sample() * 0.18215
            latents = latents.view(latents.size(0), -1)
            outputs = model(latents)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            accuracy = correct / total
            avg_loss_so_far = running_loss / total
            progress_bar.set_postfix({
                "Val Acc": f"{accuracy:.4f}",
                "Val Loss": f"{avg_loss_so_far:.4f}"
            })

            # if rank == 0 and (i + 1) % log_interval == 0:
            #     wandb.log({"val_step_loss": loss.item(), "val_step_acc": correct / total}, step=step + i)

    return running_loss / total, correct / total


def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%m%d%y_%H%M")
    dataset_name = "clickme"
    run_id = f"vae_probe_{timestamp}_{dataset_name}"

    label_map = {}
    val_dataset = ClickMe(image_folder="val", label_to_category_map=label_map, is_training=False)
    train_dataset = ClickMe(image_folder="train", label_to_category_map=label_map, is_training=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, drop_last=True, sampler=val_sampler)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, drop_last=True, sampler=train_sampler)


    print("DataLoader Created!")

    model = Head().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(device)
    for param in vae.parameters():
        param.requires_grad = False

    print("Model Created!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # if rank == 0:
    #     wandb.init(project="vae_linear_probe", name=run_id)

    global_step = 0
    best_val_loss = float("inf")
    save_path = "pretrained_models"
    os.makedirs(save_path, exist_ok=True)

    print(f"Training Started for rank {rank} !!!")
    num_epochs = 100
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc, global_step = train_one_epoch(model, vae, train_loader, criterion, optimizer, epoch, device, rank, global_step=global_step)

        # if rank == 0:
        #     wandb.log({"train_epoch_loss": train_loss, "train_epoch_acc": train_acc}, step=global_step)

        val_loss, val_acc = evaluate(model, vae, val_loader, criterion, device, rank, global_step)

        if rank == 0:
        #     wandb.log({"val_epoch_loss": val_loss, "val_epoch_acc": val_acc}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(unwrap_model(model).state_dict(), os.path.join(save_path, run_id))
                print(f">> Saved best model with loss: {best_val_loss:.4f}")

    # if rank == 0:
    #     # wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()