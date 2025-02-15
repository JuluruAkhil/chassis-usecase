# Fraud Detection Model with ChassisML

This project implements a machine learning model for credit card fraud detection, containerized using ChassisML for easy deployment and inference.

## Project Overview

The system uses a Random Forest classifier to predict potential credit card fraud based on various features including:

- Income
- Security code
- Card expiry
- Professional background

The model is packaged as a Docker container using ChassisML, exposing a gRPC interface for predictions.

## Project Structure

```
.
├── README.md
├── train_and_save_model.py    # Script for training and saving the model
├── load_and_use_model.py      # Utility script for loading and using the saved model
├── index.py                   # Main ChassisML model definition and Docker build
├── .condarc                   # Conda environment configuration
└── docker_test.py            # Test script for the containerized model using gRPC
```

## Prerequisites

- Python 3.7+
- Docker
- Conda (Miniconda or Anaconda)

## Setup and Installation

### 1. Set up the Conda Environment

Create and activate a new Conda environment using the provided `.condarc` file:

```bash
# Create new environment
conda env create -f .condarc.yaml -n fraud-detection

# Activate the environment
conda activate fraud-detection
```

### 2. Train the Model

With the environment activated, train the model:

```bash
python train_and_save_model.py
```

### 3. Build the Docker Container

Build the container image:

```bash
python index.py
```

## Usage

### Docker Container

The model is packaged as a Docker container and exposes a gRPC endpoint on port 45000.

To run the container:

```bash
docker run -d -p 45000:45000 fraud-detection-model:1.0.0
```

### Making Predictions

The model is accessed via gRPC using the ChassisML client. Here's an example using the provided test script:

```python
from chassis.client import OMIClient
import json

async with OMIClient("localhost", 45000) as client:
    # Format your input data
    input_data = json.dumps([[5000, 35, 4, 1, 0]])  # [income, security_code, months_to_expiry, is_engineer, is_lawyer]

    # Make prediction
    res = await client.run([{"input": input_data.encode("utf-8")}])
    result = res.outputs[0].output["results.json"]
```

The input data should be formatted as a JSON array of arrays, where each inner array contains:

```python
[income, security_code, months_to_expiry, is_engineer, is_lawyer]
```

Example input:

```python
[[5000, 35, 4, 1, 0]]  # Single prediction
[[5000, 35, 4, 1, 0], [7000, 50, 3, 0, 1]]  # Multiple predictions
```

The model returns predictions in the following format:

```json
[
  {
    "predicted_class": 0,
    "fraud_probability": 0.12
  }
]
```

### Local Testing

You can test the containerized model using the provided test script:

```bash
python docker_test.py
```

This script demonstrates how to connect to and use the gRPC API.

## Model Details

- Algorithm: Random Forest Classifier
- Features: Income, Security Code, Months to Expiry, Professional Background
- Handling Class Imbalance: SMOTE (Synthetic Minority Over-sampling Technique)
- Evaluation Metrics: Classification Report including Precision, Recall, and F1-Score
