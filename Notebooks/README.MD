
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

# **Feature Store & Batch Inference**

This notebook builds a small, end-to-end feature store workflow for the apple leaf project. It creates a feature group for image metadata, ingests a demo table, validates online and offline stores, runs Athena queries, and demonstrates batch inference with a PyTorch model using JSONL input.

Here're the main steps in this notebooks:

* Stand up a Feature Group for image-level features.
* Ingest and validate records in online and offline stores.
* Query features via Athena for analysis.
* Run a SageMaker Batch Transform job with an existing model artifact.

# **Model Registry & CI/CD Pipeline (DAG)**

This notebook builds a CI-style SageMaker pipeline that trains, evaluates, gates on quality, and registers an Apple leaf classifier in the SageMaker Model Registry, with approval and deployment.

Here're the main steps in this notebooks:

* Define a reusable pipeline that produces versioned model packages.
* Enforce a validation-accuracy threshold before registration.
* Persist evaluation metrics alongside each registered version.
* Finally, approve and deploy a chosen package to an endpoint.

# **Model Monitor Bias Report**

This notebook operationalizes post-training bias analysis for the Apple Leaf classifier using **Amazon SageMaker Clarify**. It deploys the trained PyTorch model with request/response capture, builds an evaluation table with predictions and a simple “brightness” facet, and generates a Clarify bias report.

**We'll cover the following item:**

* Model deployment with `DataCaptureConfig`.
* Request generation and capture verification in S3.
* Local batch inference to create `eval.csv`.
* Clarify configuration: `DataConfig`, `BiasConfig`, `ModelPredictedLabelConfig`.
* Executing `run_post_training_bias` and retrieving the report artifacts.
* Reading key metrics and known caveats (e.g., CDDPL requires group variable).
