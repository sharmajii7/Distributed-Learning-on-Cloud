# Distributed Learning on Cloud

This repository hosts a federated learning (FL) framework for credit card fraud detection, implemented on Microsoft Azure. It simulates a decentralized banking environment where multiple institutions (data silos) collaboratively train a fraud detection model without sharing sensitive transaction data. The framework leverages Azure Machine Learning (Azure ML) for orchestration, ensuring privacy through differential privacy and secure data isolation via virtual networks (VNets) and private endpoints. This README provides a comprehensive guide to setting up, running, and understanding the project, including detailed file descriptions and execution instructions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setup and Deployment](#setup-and-deployment)
4. [Running the Project](#running-the-project)
5. [File Structure and Descriptions](#file-structure-and-descriptions)
6. [How It Works](#how-it-works)
7. [Key Features](#key-features)
8. [Future Extensions](#future-extensions)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

The project aims to enable privacy-preserving machine learning for financial applications, specifically credit card fraud detection. Using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), it trains a Long Short-Term Memory (LSTM) model across distributed silos, each representing a bank. The system ensures data remains localized, with model updates aggregated securely by a central orchestrator. Key aspects include:

- **Privacy**: Differential privacy protects sensitive data during training.
- **Security**: An "eyes-off" configuration restricts data access to compute instances.
- **Scalability**: Azure’s cloud infrastructure supports distributed training.
- **Efficiency**: Parallel processing reduces training time.

The framework includes two pipelines:
- **Data Upload Pipeline**: Distributes the dataset across silos.
- **Training Pipeline**: Trains the model using federated learning.

## Prerequisites

To run this project, you need:
- An active Azure subscription (e.g., [Azure for Students](https://azure.microsoft.com/en-us/free/students/)).
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed and configured.
- Python 3.8+ installed.
- [Git](https://git-scm.com/downloads) for cloning the repository.
- A Kaggle account with an API token for dataset access ([Kaggle API](https://www.kaggle.com/docs/api)).
- Basic familiarity with Azure ML and cloud computing concepts.

## Setup and Deployment

### Step 1: Clone the Repository
Clone the repository to your local machine or an Azure ML compute instance:
```bash
git clone https://github.com/adit4443ya/Distributed-Learning-on-Cloud.git
cd Distributed-Learning-on-Cloud
```

### Step 2: Configure Azure Resources
1. **Create a Resource Group**:
   - In the [Azure Portal](https://portal.azure.com/), create a resource group named `First_Group` in the East US region.
   - Or use Azure CLI:
     ```bash
     az group create --name First_Group --location eastus
     ```

2. **Deploy the Infrastructure**:
   - The project uses a secure "eyes-off" configuration with private endpoints and VNets to restrict data access.
   - Deploy resources using a custom Azure Resource Manager (ARM) template (not included in this repo; adapt from your Azure setup).
   - Example deployment parameters:
     - **Demo Base Name**: `fldemo-adit4443ya`
     - **Orchestrator Region**: East US
     - **Silo Regions**: East US, West Europe, Australia East
     - **VM Size**: `Standard_DS2_v2` (2 vCPUs, 7 GiB RAM)
     - **Kaggle Credentials**: Store username and API key in Azure Key Vault
   - Run the deployment command (modify as needed):
     ```bash
     az deployment group create --resource-group First_Group --template-file azuredeploy.json --parameters demoBaseName=fldemo-adit4443ya orchestratorRegion=eastus siloRegions="eastus westeurope australiaeast"
     ```

3. **Verify Resources**:
   - Ensure the following are created in `First_Group`:
     - **Azure ML Workspace**: `fl-demo-adit4443ya-workspace`
     - **Compute Clusters**: `orchestrator-01`, `silo0-01`, `silo1-01`, `silo2-01`
     - **Storage Accounts**: `datastore_orchestrator`, `datastore_silo0`, `datastore_silo1`, `datastore_silo2`
     - **Key Vault**: `kv-fldemo-adit4443ya`
     - **Virtual Network**: Configured with private endpoints

### Step 3: Authenticate with Azure
- On your compute instance or local machine, authenticate:
  ```bash
  az login
  ```
- Follow the prompts to sign in, ensuring access to your subscription (e.g., ID `5e7283a7-33ca-46ab-8a0c-6e60f7b3b7ec`).

### Step 4: Create a Compute Instance
- In Azure ML Studio ([ml.azure.com](https://ml.azure.com/)), navigate to **Compute** > **Compute Instances**.
- Create a new instance (e.g., `adit4443ya1`) with a standard VM size (e.g., `Standard_DS3_v2`).
- Use this instance to run pipelines and manage the project.

## Running the Project

### Step 1: Set Up the Environment
- Navigate to the repository root:
  ```bash
  cd Distributed-Learning-on-Cloud
  ```
- Install dependencies:
  ```bash
  pip install -r pipelines/requirements.txt
  ```
- Alternatively, set up a Conda environment:
  ```bash
  conda env create -f pipelines/environment.yml
  conda activate fl_experiment_conda_env
  ```

### Step 2: Configure Pipelines
- Update configuration files in `pipelines/upload_data/` and `pipelines/ccfraud/` with your Azure details:
  - **Subscription ID**: `5e7283a7-33ca-46ab-8a0c-6e60f7b3b7ec`
  - **Resource Group**: `First_Group`
  - **Workspace Name**: `fl-demo-adit4443ya-workspace`
  - **Key Vault URL**: Obtain from Azure Portal (e.g., `https://kv-fldemo-adit4443ya.vault.azure.net/`)
- Example configuration (do not include in code snippets; edit manually):
  - `pipelines/upload_data/config.yaml`
  - `pipelines/ccfraud/config.yaml`

### Step 3: Run the Data Upload Pipeline
- This pipeline downloads the Kaggle dataset and distributes it across silos.
- Navigate to the pipeline directory:
  ```bash
  cd pipelines/upload_data
  ```
- Execute:
  ```bash
  python submit.py --example CCFRAUD
  ```
- Monitor progress in Azure ML Studio under **Pipelines** > `upload_data`.

### Step 4: Run the Training Pipeline
- This pipeline trains the LSTM model using federated learning.
- Navigate to the pipeline directory:
  ```bash
  cd ../ccfraud
  ```
- Execute:
  ```bash
  python submit.py
  ```
- Monitor progress in Azure ML Studio under **Pipelines** > `ccfraud`.

### Step 5: Monitor Results
- Access Azure ML Studio ([ml.azure.com](https://ml.azure.com/)).
- Go to **Pipelines** to view pipeline status and logs.
- Check **Jobs** for metrics like accuracy, precision, recall, and training time.

## File Structure and Descriptions

The repository is organized into two main directories: `components/` and `pipelines/`. Below is the structure with detailed descriptions:

| **Path** | **Description** |
|----------|-----------------|
| `components/` | Contains reusable Azure ML components for pipeline stages. |
| `components/aggregatemodelweights/` | Aggregates model weights from silos for federated averaging. |
| `components/aggregatemodelweights/conda.yaml` | Conda environment for the aggregation component. |
| `components/aggregatemodelweights/run.py` | Executes federated averaging on model updates. |
| `components/aggregatemodelweights/spec.yaml` | Defines the component’s inputs, outputs, and compute requirements. |
| `components/preprocessing/` | Preprocesses data for each silo (e.g., normalization, encoding). |
| `components/preprocessing/conda.yaml` | Conda environment for preprocessing. |
| `components/preprocessing/confidential_io.py` | Handles secure data input/output with encryption support. |
| `components/preprocessing/run.py` | Performs data preprocessing tasks. |
| `components/preprocessing/spec.yaml` | Component specification for preprocessing. |
| `components/traininsilo/` | Trains the LSTM model locally on each silo. |
| `components/traininsilo/conda.yaml` | Conda environment for training. |
| `components/traininsilo/confidential_io.py` | Manages secure data access during training. |
| `components/traininsilo/datasets.py` | Defines dataset loading and preprocessing logic. |
| `components/traininsilo/models.py` | Implements the LSTM model architecture. |
| `components/traininsilo/run.py` | Executes local training with differential privacy. |
| `components/traininsilo/spec.yaml` | Component specification for training. |
| `components/upload_data/` | Uploads and distributes the dataset to silos. |
| `components/upload_data/conda.yaml` | Conda environment for data upload. |
| `components/upload_data/confidential_io.py` | Ensures secure dataset handling. |
| `components/upload_data/run.py` | Downloads and partitions the dataset. |
| `components/upload_data/spec.yaml` | Component specification for data upload. |
| `components/upload_data/us_regions.csv` | Lists regions for silo deployment (e.g., East US). |
| `pipelines/` | Contains pipeline definitions and dependencies. |
| `pipelines/ccfraud/` | Training pipeline for fraud detection. |
| `pipelines/ccfraud/config.yaml` | Configures the training pipeline (e.g., model parameters, compute targets). |
| `pipelines/ccfraud/submit.py` | Submits the training pipeline to Azure ML. |
| `pipelines/upload_data/` | Data upload pipeline. |
| `pipelines/upload_data/config.yaml` | Configures the data upload pipeline (e.g., workspace, Key Vault). |
| `pipelines/upload_data/submit.py` | Submits the data upload pipeline to Azure ML. |
| `pipelines/environment.yml` | Conda environment for running pipelines. |
| `pipelines/requirements.txt` | Python dependencies for the project. |

## How It Works

### Data Flow
1. **Data Upload Pipeline**:
   - Downloads the Kaggle dataset using credentials from Azure Key Vault.
   - Partitions the dataset into three subsets for silos (`datastore_silo0`, `datastore_silo1`, `datastore_silo2`).
   - Performs a local train-test split (80% training, 20% testing) per silo.
   - Stores data securely using private endpoints.

2. **Training Pipeline**:
   - Initializes a global LSTM model on `orchestrator-01`.
   - Distributes the model to silos for local training.
   - Each silo trains the model for 3 epochs (batch size 64, learning rate 0.001).
   - Applies differential privacy (noise multiplier 1.1, epsilon ≈ 1.5).
   - Sends encrypted parameter updates to the orchestrator.
   - Aggregates updates using federated averaging over 5 rounds.
   - Updates the global model iteratively.

3. **Monitoring**:
   - Tracks pipeline execution and metrics (e.g., accuracy, recall) in Azure ML Studio.
   - Provides logs and visualizations for debugging and analysis.

### Workflow
- **Preprocessing**: Normalizes transaction amounts and encodes temporal features.
- **Training**: Uses an LSTM with two layers (128 hidden units each) for sequence-based fraud detection.
- **Aggregation**: Computes weighted averages of model parameters, proportional to silo dataset sizes.
- **Privacy**: Ensures data protection through local training and differential privacy.

## Key Features

- **Privacy-Preserving**: Data stays within silos, with differential privacy protecting updates.
- **Secure Architecture**: VNets and private endpoints restrict data access.
- **Scalable**: Parallel training across silos reduces computation time.
- **Flexible**: Supports custom datasets and model configurations.
- **Transparent**: Azure ML Studio provides real-time monitoring.

## Future Extensions

- **Graph Analytics**: Adapt for Graph Neural Networks (GNNs) to solve problems like link prediction.
- **Model Expansion**: Test SimpleLinear and SimpleVAE models for comparison.
- **Automation**: Develop end-to-end pipelines for data ingestion and deployment.
- **Advanced Privacy**: Integrate homomorphic encryption for enhanced security.
- **Broader Applications**: Apply to social networks, healthcare, or IoT analytics.


---

**Key Citations**:
- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Microsoft Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure CLI Installation Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Kaggle API Documentation for Dataset Access](https://www.kaggle.com/docs/api)
- [Azure for Students Free Subscription](https://azure.microsoft.com/en-us/free/students/)