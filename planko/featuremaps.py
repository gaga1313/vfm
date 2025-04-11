import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
# import wandb
import os

from src.data import OrigPlank, OrigPlank2, transform
from src.models import ResNet50Classifier, ResNet50VAEClassifier
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import cv2

torch.manual_seed(42)

root_dir = "./feature_maps/og/layer_-3"
os.makedirs(root_dir, exist_ok = True)

def test(model, dataloader, device='cuda', modelname = "resnet50vae_np_cls"):
    os.makedirs(os.path.join(root_dir, modelname), exist_ok = True)
    model.eval()

    progress_bar = tqdm(dataloader)
    count = 0
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device).float(), labels.to(device)
            # feature_maps, outputs, _, _ = model(images)
            outputs, feature_maps = model(images)
            images = images * 255
            images = images.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)

            _, predicted = torch.max(outputs, 1)
            correct_indices = (predicted == labels).cpu().numpy()
            labels = labels.cpu().numpy()
            # for k in range(5):
            fmps = feature_maps.permute(0,2,3,1).detach().cpu().mean(dim = -1)
            fm = F.resize(fmps, [224, 224]).numpy()
            fm = cv2.normalize(fm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for i, img in enumerate(images):
                filename = "img_" + str(count) + "_basket" + str(labels[i]+1) + "_correct_" + str(correct_indices[i]) + ".png"
                colored_feature_map = cv2.applyColorMap(fm[i], cv2.COLORMAP_MAGMA)
                image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                alpha = 0.5  # Adjust transparency
                overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_feature_map, alpha, 0)

                # Convert overlay back to RGB for saving with Matplotlib
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                # Save the overlay as a PNG file
                output_path = os.path.join(root_dir, modelname, filename)
                plt.imsave(output_path, overlay_rgb)
                count+=1
    return

def main():


    device='cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = "data/dataset/"
    # dataset = OrigPlank(path=dataset_dir, transform=None)

    # train_size = int(0.8 * len(dataset))
    # test_size  = len(dataset) - train_size
    # _, test_dataset = random_split(dataset, [train_size, test_size])
    test_dataset = OrigPlank2("/cifs/data/tserre_lrs/projects/projects/prj_vis_sim/plankdatasets/originalv1/test", train = False, transform = transform)

    test_dataloader = DataLoader(test_dataset, batch_size=32)


    modelname = "resnet50_np_cls"
    model = ResNet50Classifier(num_classes=2).to("cuda:0")

    load_path = os.path.join("pretrained_models", modelname, "trial_6_best_model.pth") 
    model.load_state_dict(torch.load(load_path, map_location = "cuda:0"))

    # import ipdb; ipdb.set_trace()
    test(model, test_dataloader, device, modelname)

    
if __name__ == "__main__":
    main()
    
    
