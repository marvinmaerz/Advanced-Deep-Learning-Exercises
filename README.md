# Advanced Deep Learning Exercises
Programming exercises of the Advanced Deep Learning (ADL) course at FAU.

# Structure & Getting Started
Each exercise is its own top-level directory under the repository root.

## Installation
### On the cip machines:
Activate `conda` environment:
```bash
conda activate /proj/aimi-adl/envs/adl23_2
```

### Local development using `hatch`:
Create environment from `pyproject.toml`:
```bash
hatch env create
```

Activate and use the environment:
```bash
hatch shell 
# or
hatch run python 
```

# Collaboration
Each person initially does work on their own branch with name `exerciseX-person`. Once work is finished on individual branches, merge and test on `main`.

# Exercises
## 1. Vision Transformer and Cross Vision Transformer
Tasks: 
1. PyTorch basics
2. VisionTransformer (ViT): Self-attention for image classification
3. Cross-Attention Multi-Scale ViT
4. Visualizing ViT decision-making