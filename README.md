<p align="left">
  <!-- Language / Core -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="TorchVision" src="https://img.shields.io/badge/TorchVision-0.17%2B-orange">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-export-005CED?logo=onnx&logoColor=white">
  <img alt="TorchScript" src="https://img.shields.io/badge/TorchScript-export-000000">

  <!-- Data / Wrangling -->
  <img alt="Google Colab" src="https://img.shields.io/badge/Google%20Colab-wrangling-FFC107?logo=googlecolab&logoColor=black">
  <img alt="PlantVillage" src="https://img.shields.io/badge/Dataset-PlantVillage%20(Apple)-6DB33F">

  <!-- AWS (dev/MLOps) -->
  <img alt="SageMaker" src="https://img.shields.io/badge/AWS%20SageMaker-dev-232F3E?logo=amazonaws&logoColor=white">
  <img alt="Model Registry" src="https://img.shields.io/badge/SageMaker-Model%20Registry-232F3E?logo=amazonaws&logoColor=white">
  <img alt="Batch Transform" src="https://img.shields.io/badge/SageMaker-Batch%20Transform-232F3E?logo=amazonaws&logoColor=white">
  <img alt="Clarify" src="https://img.shields.io/badge/SageMaker-Clarify%20Bias%20Report-5A3EC8">
  <img alt="CloudWatch" src="https://img.shields.io/badge/AWS-CloudWatch%20Dashboards-5A3EC8?logo=amazonaws&logoColor=white">
  <img alt="Amazon S3" src="https://img.shields.io/badge/Amazon%20S3-datasets%20%26%20reports-569A31?logo=amazons3&logoColor=white">

  <!-- App -->
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Camera or Upload" src="https://img.shields.io/badge/Input-camera%20or%20upload-444">

  <!-- Model policy / classes -->
  <img alt="Classes" src="https://img.shields.io/badge/Classes-healthy%7Cscab%7Crust%7Cblack__rot%7Cunknown-0E8A16">
  <img alt="Threshold" src="https://img.shields.io/badge/Decision%20threshold-%CF%84%3D0.65-0366D6">

  <!-- Results -->
  <img alt="Accuracy" src="https://img.shields.io/badge/Test%20accuracy-98.9%25-brightgreen">
  <img alt="Macro F1" src="https://img.shields.io/badge/Macro--F1-0.986-brightgreen">

  <!-- Model -->
  <img alt="ResNet-18" src="https://img.shields.io/badge/Model-ResNet-18-0069A1?logo=keras&logoColor=white">
</p>



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


