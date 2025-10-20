
# **What this folder contains**

Minimal, modular scripts to **train**, **evaluate**, and **inference** serve a classifier that detects common apple-leaf conditions:

* healthy
* scab
* black_rot
* rust

The scripts are framework-agnostic on purpose but assume a PyTorch image-classification workflow:  
```pgsql
src/
├── train.py        # Train a model and save checkpoints & metrics
├── evaluate.py     # Evaluate a trained checkpoint on a holdout set
└── inference.py    # Load a checkpoint and run single/batch predictions
