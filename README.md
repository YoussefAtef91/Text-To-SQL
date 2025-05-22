# Text-to-SQL with Finetuned BART

This project demonstrates how to fine-tune a pretrained BART model for the task of translating natural language queries into SQL. It includes model training, versioning, API deployment, and production-ready containerization.

## 🔧 What I Did

* **Model Fine-Tuning:**
  Fine-tuned the [BART](https://arxiv.org/abs/1910.13461) transformer model for the text-to-SQL task using a custom dataset.

* **Data and Model Versioning:**
  Used [DVC](https://dvc.org/) to track data and model artifacts across training stages.
  Pushed all versions to [DagsHub](https://dagshub.com/) for reproducibility and collaboration.

* **Serving the Model:**
  Built a scalable API using [LitServe](https://github.com/Lightning-AI/litserve) to expose the model for inference.

* **Containerization:**
  Created a Docker image to package the API and its dependencies for consistent deployment.

* **Deployment:**
  Pushed the image to [DockerHub](https://hub.docker.com/) and deployed the containerized API on [Lightning AI](https://lightning.ai/), enabling cloud-based, production-ready serving.

## 🚀 Tech Stack

* `Transformers` (HuggingFace)
* `DVC` + `DagsHub`
* `LitServe`
* `Docker` + `DockerHub`
* `Lightning AI` for deployment