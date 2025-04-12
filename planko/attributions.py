import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from captum.attr import IntegratedGradients, Occlusion

from src.data import OrigPlank5, transform
from src.models import CConvVAE2, Conv2REG
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import cv2
from argparse import ArgumentParser

# Seed for reproducibility
torch.manual_seed(42)

parser = ArgumentParser(description="Visual Foundation Model Training")
parser.add_argument(
    "--vae_test", action="store_true", default=False, help="training strategy"
)

# Directory for saving Integrated Gradients overlayed images
output_dir = "./attributions/sbk/ccvae2_grid2/integrated_gradients/"
os.makedirs(output_dir, exist_ok=True)


def compute_and_save_ig(model, dataloader, device="cuda", explainer="ig"):
    model.eval()

    # Initialize Integrated Gradients
    if explainer == "ig":
        exp = IntegratedGradients(model)
    else:
        exp = Occlusion(model)

    progress_bar = tqdm(dataloader)
    count = 0
    for step, (images, labels, pos, board) in enumerate(progress_bar):
        images = images.to(device).float().requires_grad_()
        pos = pos.to(device)
        # Compute attributions using Integrated Gradients
        if explainer == "ig":
            attributions = exp.attribute(
                images,
                target=[(31, int(x[31])) for x in pos],
                n_steps=50,
                internal_batch_size=32,
            )
        else:
            attributions = exp.attribute(
                images, target=(), sliding_window_shapes=(1, 30, 40)
            )

        # Convert images and attributions for overlaying and saving
        images = images * 255
        images_np = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        attributions = attributions.permute(0, 2, 3, 1).detach().cpu().numpy()

        # outputs = model(images)
        # _, predicted = torch.max(outputs, 1)
        # correct_indices = (predicted == labels).cpu().numpy()

        for i, (img, attr) in enumerate(zip(images_np, attributions)):
            # Normalize and colorize the attribution map
            attr_map = np.abs(attr).mean(axis=-1)  # Aggregate across color channels
            attr_map = cv2.normalize(attr_map, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            attr_map_colored = cv2.applyColorMap(attr_map, cv2.COLORMAP_HOT)

            # Overlay attribution map on the original image
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            alpha = 0.85
            overlayed_image = cv2.addWeighted(
                img_bgr, 1 - alpha, attr_map_colored, alpha, 0
            )

            # Convert back to RGB and save the overlayed image
            overlayed_image_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
            filename = f"img_{count}.png"
            output_path = os.path.join(output_dir, filename)
            plt.imsave(output_path, overlayed_image_rgb)
            count += 1


def main():
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Dataset and DataLoader setup
    batch_size = 32
    latent_size = 4
    test_dataset = OrigPlank5(
        "/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/singbaskv3/test",
        train=False,
        is_val=False,
        transform=transform,
        noise=False,
        norm=False,
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
        model.load_state_dict(torch.load(load_path, map_location="cuda:0"))
    else:
        modelname = "conv2_det_grid"
        model = Conv2REG(image_channels=3, image_size=224).to(device)
        load_path = os.path.join(
            "pretrained_models", modelname, "trial_1_best_model.pth"
        )
        model.load_state_dict(torch.load(load_path, map_location="cuda:0"))

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    # Run Integrated Gradients overlay and save results
    compute_and_save_ig(model, test_dataloader, device, "ig")


if __name__ == "__main__":
    main()
