
---

# 🌌 Visual Foundation Model with Bayesian Inference

**_Real-world Image Classification with Distributed PyTorch Training_**

![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red)
![Bayesian Inference](https://img.shields.io/badge/Bayesian%20Inference-Applied-blue)
![Distributed Training](https://img.shields.io/badge/Distributed%20Training-MultiGPU-green)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

> A robust foundation model for visual inference and classification tasks built on real-world images. This model leverages **ResNet50** with **Bayesian Inference** for uncertainty estimation, distributed training with **NCCL backend**, and logging through **Weights and Biases**.

---

## 🔥 Key Features

- **Distributed Training**: Multi-GPU training across devices using PyTorch's NCCL backend.
- **Bayesian Inference**: Real-time uncertainty estimation for robust model outputs.
- **Efficient Data Handling**: Supports large-scale real-world image datasets.
- **Automatic Logging**: Logs training metrics and model checkpoints with Weights & Biases.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [License](#license)

---

## ⚙️ Requirements

- **Python 3.8+**
- **CUDA 11.0+** for GPU-based training
- **PyTorch >= 1.10**
- **NCCL backend** for distributed training
- **Weights & Biases** for logging metrics

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🛠️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/gaga1313/vfm.git
   cd visual-foundation-model
   ```

2. **Set up your dataset**

   Place your dataset under `data/dataset/`. The dataset should contain images and corresponding labels.

3. **Environment Setup**

   Create an `.env` file with your [Weights & Biases](https://wandb.ai/) credentials if you’d like to enable logging.

---

## 🚀 Quick Start

**1. Initialize Distributed Training**

Ensure your GPUs are correctly set up and launch training with:

```bash
pyython -m torch.distributed.run --nproc_per_node=<num_gpus> train.py
```


## 🧠 Model Architecture

Our model uses the **ResNet50** architecture, customized with:

- **Bayesian Inference Layer**: Enables uncertainty estimation for each prediction.
- **Metric Logger**: Logs losses, accuracies, and Bayesian confidence.
- **Multi-Process DataLoader**: Optimized for distributed data loading in multi-GPU environments.

## 📊 Training

The training loop is distributed across available GPUs using **DistributedSampler**. Metrics are logged for every epoch via **Weights & Biases**.

Run the training script:

```bash
torchrun --nproc_per_node=<num_gpus> main.py --mode train
```

**Flags**:
- `--nproc_per_node`: Number of GPUs
- `--batch_size`: Batch size for each process
- `--epochs`: Total epochs
- `--log_interval`: Logging frequency

## 🔍 Testing

Evaluate the trained model on a separate test dataset. To run the test evaluation, use:

```bash
torchrun --nproc_per_node=<num_gpus> main.py --mode test
```

---

## 📈 Results

Sample results and logged metrics can be found in the `results/` directory. Metrics such as **accuracy**, **loss**, and **Bayesian confidence** are logged for detailed analysis.

Example Weights & Biases Dashboard (replace `<your_wandb_project_link>`):
[![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-Project%20Dashboard-blue)](<your_wandb_project_link>)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

Happy Training! 💪
