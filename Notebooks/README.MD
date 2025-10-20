
# **Data Wrangling & Preprocessing**

In this notebook, we'll prepare the training data for AI-Powered Apple Leaf Specialist. The goal is to pull PlantVillage Apple dataset, harmonize their labels, clean and normalize the images, and produce final train, validation, and test splits where can used for modeling. Along the way we'll keep everything reproducible with a simple config.yaml and a tidy folder layout so it is easy to hand off to AWS later:

* Pull sources: PlantVillage Apple.

* Normalize labels into one shared set: healthy, scab, rust, black_rot, unknown:

* Remove obvious duplicates, fix filenames, and standardize image sizes:

* Write final splits (train, val and test) to disk and export CSV manifests for each split.

* Data Augmentation (train only): To help the model generalize and avoid overfitting to the majority class, train data augmentation is applied.

# **Modeling**
AI-Powered Apple Leaf Specialist is an end-to-end system that detects apple leaf diseases from images with production-grade reliability. The goal is to train a PyTorch classifier and validate it rigorously, and deploy it using Github and streamlit (Cloud) due to Amazon low budget.

**Importance**

Early detection helps reduce crop loss and chemical overuse. An accurate, lightweight model that scales on demand is practical for growers, agronomists, and mobile field apps.

This notebook implements the practical path from data to a live endpoint:
* It prepares data in S3.
* Trains a PyTorch model on SageMaker.
* Packages inference code.
* Deploys a serverless endpoint.
* Smoke-tests predictions with real images.
* Evaluates on the full test set, and sets up basic monitoring.
* It closes with concise guidance on cost and performance trade-offs.
