# Databricks notebook source
import os
import pandas as pd
import requests
import json
from transformers import pipeline
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from mlflow.tracking import MlflowClient

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

import requests
import json

# Set the name of the MLflow endpoint
endpoint_name = "radllama2_7b"

# Name of the registered MLflow model
model_name = "ang_nara_catalog.rad_llm.radllama2_7b_test"

# Get the latest version of the MLflow model
model_version = 3

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_MEDIUM"

# Specify the scale-out size of compute (Small, Medium, Large, etc.)
workload_size = "Small"

# Specify Scale to Zero (only supported for CPU endpoints)
scale_to_zero = False

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# send the POST request to create the serving endpoint

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "workload_type": workload_type,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers
)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------



# COMMAND ----------


