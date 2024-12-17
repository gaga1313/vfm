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
from src.models import ResNet50Classifier, ResNet50VAEClassifier
from src.utils import MetricLogger, kld

parser = ArgumentParser(description = "Visual Foundation Model Training")
parser.add_argument("--vae_training", action = "store_true", default = False, help = "training strategy")
parser.add_argument("--w2", default = 1e-5, help = "KLD loss weight")


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()

def unwrap_model(model):
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    return model

# Training function with a progress bar
def train_one_epoch(
    args, model,
    dataloader,
    criterion,
    optimizer,
    metric_logger,
    epoch,
    epochs=10,
    device="cuda",
    rank=0,
    log_interval=10,
):
    model.train()
    running_loss = correct = total = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)

    for step, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device).float(), labels.to(device)
        
        optimizer.zero_grad()
        
        if args.vae_training:
            _, outputs, mu, logvar = model(images)
        else:
            outputs, _ = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        batch_loss = loss.item()
        
        if args.vae_training:
            loss_kld = args.w2 * kld(mu, logvar)
            final_loss = loss + loss_kld
            final_loss.backward()
            running_loss += loss_kld.item()
            batch_loss += loss_kld.item()
        else:
            loss.backward()
        
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        batch_loss /= labels.size(0)
        batch_accuracy = 100 * correct / total

        metric_logger.add("train_loss", batch_loss)
        metric_logger.add("train_accuracy", batch_accuracy)

        if rank == 0 and (step + 1) % log_interval == 0:
            wandb.log(
                {
                    "train_loss": batch_loss,
                    "train_accuracy": batch_accuracy,
                    "epoch": epoch + 1,
                }
            )
        if args.vae_training:
            progress_bar.set_postfix(
                {"Train Loss": batch_loss, "Train KLD": loss_kld.item(), "Train Accuracy": batch_accuracy}
            )
        else:
            progress_bar.set_postfix(
                {"Train Loss": batch_loss, "Train Accuracy": batch_accuracy}
            )


    avg_loss = metric_logger.average("train_loss")
    accuracy = metric_logger.average("train_accuracy")
    return avg_loss, accuracy


def test(
    args, 
    model, 
    dataloader, 
    criterion, 
    metric_logger, 
    epoch, 
    epochs, 
    device="cuda", 
    rank=0
):
    model.eval()
    total = correct = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", position=rank)

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device).float(), labels.to(device)

            if args.vae_training:
                _, outputs, mu, logvar = model(images)
            else:
                outputs, _ = model(images)
            
            loss = criterion(outputs, labels)
            batch_loss = loss.item()
            # running_loss += loss.item()

            if args.vae_training:
                loss_kld = args.w2 * kld(mu, logvar)
                batch_loss += loss_kld.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_loss /= labels.size(0)
            batch_accuracy = 100 * correct / total

            metric_logger.add("test_loss", batch_loss)
            metric_logger.add("test_accuracy", batch_accuracy)

            if rank == 0:
                wandb.log({"test_loss": batch_loss, "test_accuracy": batch_accuracy})

            if args.vae_training:
                progress_bar.set_postfix(
                    {"Test Loss": batch_loss,  "Test KLD": loss_kld.item(), "Test Accuracy": batch_accuracy}
                )
            else:
                progress_bar.set_postfix(
                    {"Test Loss": batch_loss, "Test Accuracy": batch_accuracy}
                )

    avg_loss = metric_logger.global_average("test_loss")
    accuracy = metric_logger.global_average("test_accuracy")
    if rank == 0:
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()

    epochs = 10
    dataset_dir = "data/dataset/"
    dataset = OrigPlank(path=dataset_dir, transform=None)

    print(f"Is VAE training: {args.vae_training}")
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # shifted to new data
    train_dataset = OrigPlank2("/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/train", train = True, transform = transform)
    test_dataset = OrigPlank2("/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/test", train = False, transform = transform)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)


    if args.vae_training:
        modelname = "resnet50vae_np_cls"
        model = ResNet50VAEClassifier(num_classes = 2).to(device)
    else:
        modelname = "resnet50_np_cls"
        model = ResNet50Classifier(num_classes = 2).to(device)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0
    save_path = os.path.join("pretrained_models", modelname)
    dataset_name = "og_planko"
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
        train_loss, train_acc = train_one_epoch(
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
        )

        if rank == 0:
            print(
                f"Epoch {e+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%"
            )
        _, test_acc = test(
            args,
            model,
            test_dataloader,
            criterion,
            metric_logger,
            e,
            epochs,
            device,
            rank=rank,
        )
        
        if rank == 0 and  test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(
                unwrap_model(model).state_dict(),
                os.path.join(save_path, "trial_" + trial_number + "_best_model.pth"),
            )
            print(f"Best model saved with Test Accuracy: {best_accuracy:.2f}%")

    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
