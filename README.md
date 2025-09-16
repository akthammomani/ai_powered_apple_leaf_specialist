# **AI Powered Apple Leaf Specialist**

This project is a part of the Machine Learning Operations (AAI-540-02) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Ongoing**

## **Introduction**

AI-Powered Apple Leaf Specialist helps growers and hobbyists understand what’s going on when apple leaves show spots, rust, or other symptoms. The app looks at a few photos, predicts the likely condition, and gives clear, safety-first steps the user can take.

* Model objective: classify an apple leaf photo into a small set of conditions and return a short action plan.
* ML problem type: supervised multi-class image classification with an unsure path for low-confidence cases.

## **Objectives##

* Provide a condition label and a confidence score for each uploaded photo.
* Offer plain-language guidance for each condition, plus a prevention tip.
* Keep costs low by training in Colab and deploying a short demo on SageMaker (serverless).
* Start with PlantVillage (Apple subset) and, if resources allow, add Plant Pathology 2020 and PlantDoc later.

## **Methods Used**

* Data wrangling and label harmonization
* Exploratory Data Analysis (EDA)
* Data visualization and sanity checks
* Class imbalance handling (class weights or focal loss)
* Image augmentation (brightness/contrast jitter, blur/noise, crop, perspective, mixup)
* Transfer learning and probability calibration
* Hyperparameter tuning and threshold selection
* Error analysis with confusion matrices

## **Technologies**

* Python for the end-to-end workflow
* PyTorch and TorchVision for modeling
* TensorFlow Datasets (TFDS) to pull PlantVillage
* Image augmentation
* scikit-learn for metrics and calibration
* AWS SageMaker Model Registry and Serverless Inference for the demo endpoint
* Amazon S3 for final datasets and reports
* ONNX or TorchScript for export

## **Features**

* Condition detection: predicts among healthy, scab, rust, black_rot, unknown
* Actionable guidance: short, safe steps and one prevention tip per condition
* Confidence-aware: routes low-confidence cases to unknown with a prompt to try another photo
* Lean deployment: Colab training with a serverless SageMaker demo endpoint and S3-backed manifests

## **Dataset**

* PlantVillage (Apple subset) — core training data with labeled apple leaf images in clean conditions
* Plant Pathology 2020 (FGVC7) — optional later add for real-orchard validation and quick fine-tune
* PlantDoc (Apple) — optional later “in-the-wild” check to test generalization
* Start with PlantVillage only due to time and budget; add the others if resources allow.

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thanks to Professor Jules Malin for guidance and feedback.
Credit to the PlantVillage team and the Plant Pathology 2020 organizers for making data available for research and education.


