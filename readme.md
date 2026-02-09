# Deep‑Learning Tutorial (dl_tutorial)

## Overview
This repository contains a small end‑to‑end tutorial that demonstrates how to build, train, and evaluate a **ConvNeXt‑style convolutional neural network** using PyTorch. The tutorial is organized into a series of Jupyter notebooks that walk you through the workflow step‑by‑step, supported by a few utility scripts and a collection of figures.

## File purpose

| File / Directory | Description |
|------------------|-------------|
| `part1_introduction.ipynb` | **Step 1 – Introduction**: Sets random seeds, creates synthetic linear data, and visualises basic concepts (plots, basic regression). |
| `part2_dl_wb2.ipynb` | **Step 2 – Download**: Download WeatherBench2 data, needed for training and testing. |
| `part2_deep_learning_training.ipynb` | **Step 3 – Training**: Configures device (`cuda`/`cpu`), optimizer, and training loop. Includes progress bars (`tqdm`) and visualises training loss. |
| `part2_deep_learning_testing.ipynb` | **Step 4 – Model testing**: Loads the network, runs a quick forward pass on a small dataset, and demonstrates loss computation without training. |
| `part2_estimate_norm.ipynb` | **Step 5 – Normalisation analysis**: Estimates and visualises the normalisation statistics (`in_mean`, `in_std`, etc.) used in `constants.py`. |
| `Figures/` | PNG assets referenced by the notebooks (architecture sketches, activation‑function diagrams, dataset samples, etc.). |
| `constants.py` | Defines pre‑computed mean/std tensors (`in_mean`, `in_std`, `res_mean`, `res_std`) and a latitude weighting vector (`weights_lat`). These constants are used for normalising input data and loss scaling. |
| `neural_net.py` | Implements the core model:<br>• `ConvNeXtBlock` – a building block with circular padding, depthwise convolution, and a learnable scaling parameter (`γ`).<br>• `get_net` – a helper that assembles a sequential network of *n_layers* blocks and returns a ready‑to‑use `torch.nn.Module`. |

## Execution order

1. **Download WeatherBench2 data** – run `part2_dl_wb2.ipynb` to download the needed WeatherBench2 data for training and test.
2. **Train the model** – execute `part2_deep_learning_training.ipynb`. Adjust the `device` and `dtype` cells if you have a GPU.
3. **Test the model** – open `part2_deep_learning_testing.ipynb` to see a forward pass and loss calculation using `neural_net.get_net`.

## Setup

```bash
# Install dependencies (see requirements.txt)
pip install -r requirements.txt

# (Optional) Verify you have a GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

After installing the requirements, launch Jupyter (or VS Code’s notebook UI) and open the notebooks in the order above.