# **AI-Powered Apple Leaf Specialist**

This project is a part of the Machine Learning Operations (AAI-540-02) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Completed**

## **Introduction**

AI-Powered Apple Leaf Specialist helps growers and hobbyists understand what’s going on when apple leaves show spots, rust, or other symptoms. The app looks at a few photos, predicts the likely condition, and gives clear, safety-first steps the user can take.

* Model objective: classify an apple leaf photo into a small set of conditions and return a short action plan.
* ML problem type: supervised multi-class image classification with an unsure path for low-confidence cases.

## **Objectives**

* Provide a condition label and a confidence score for each uploaded photo.
* Offer plain-language guidance for each condition, plus a prevention tip.
* Keep costs low by training in Colab and deploying a short demo on SageMaker (serverless).
* Start with PlantVillage (Apple subset) and, if resources allow, add Plant Pathology 2020 and PlantDoc later.

## **Dataset**

This project uses the **[PlantVillage (Apple)](https://www.kaggle.com/datasets/emmarex/plantdisease)** subset, a research dataset of labeled apple leaf images for disease recognition. Each image is a single leaf with a class label. For apple, the core classes are:

- **healthy**
- **scab** (Venturia inaequalis)
- **rust** (cedar apple rust)
- **black_rot** (Botryosphaeria obtusa)

**Why this dataset**
- Consistent images (single leaf, plain backgrounds) make ingestion and training straightforward.
- Labels map cleanly to a small, meaningful set of conditions.
- Available via **TensorFlow Datasets (TFDS)**, which keeps the pipeline reproducible.

**What to expect**
- Clean backgrounds help the model learn disease patterns quickly.
- Less representative of “in-the-wild” photos; strong augmentations and careful validation help address this.
- No personal data is expected (plant leaves, not people).

**How it's used here**
- Serves as the **sole source** for the baseline model in this project.
- Labels are aligned to a canonical set: `healthy, scab, rust, black_rot` (with an `unknown` route at inference time for low confidence).
- Images are standardized to a fixed size and split **70/15/15** (train/val/

## **Methods Used**

* Data wrangling and label harmonization (Colab)
* Exploratory Data Analysis (EDA) & data sanity checks
* Class imbalance handling (class-aware augmentation; optional focal loss/weights)
* Image augmentation (brightness/contrast jitter, blur/noise, crop, perspective, MixUp)
* Transfer learning (ResNet-18) and probability calibration (temperature scaling)
* Hyperparameter tuning (seeded grid) and threshold selection (τ=0.65)
* Error analysis with confusion matrices & per-class metrics
* Batch inference for MLOps smoke tests (n=20 cost-bounded)

## **Technologies**

* Python, PyTorch/TorchVision
* scikit-learn (metrics, calibration helpers)
* Google Colab (wrangling/EDA/augmentation)
* AWS SageMaker (training jobs, Model Registry, Batch Transform)
* Amazon CloudWatch (infra dashboards & alarms)
* SageMaker Clarify (bias/fairness reports)
* Amazon S3 (datasets, manifests, reports)
* ONNX / TorchScript (export); Streamlit (app)

## **Results**

* Test accuracy ~0.989, macro-F1 ~0.986; strong per-class precision/recall.
* Batch smoke test (n=20) shows high confidence, with unknown routing for low-confidence cases.
* Clarify shows no major slice-wise disparity across brightness_bin; findings captured in the model card.
* CloudWatch dashboard confirms low p90 latency and zero 5XX during dev runs.

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thanks to Professor Jules Malin for guidance and feedback.
Credit to the PlantVillage team and the Plant Pathology 2020 organizers for making data available for research and education.


