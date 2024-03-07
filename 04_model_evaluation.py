# Databricks notebook source
# MAGIC
# MAGIC %md # Notebook Parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "hls_healthcare") 
dbutils.widgets.text("database", "hls_dev")
dbutils.widgets.text("model_version", "1")

# COMMAND ----------

batch_tablename = (dbutils.widgets.get("catalog") + 
                    "." + dbutils.widgets.get("database") +
                    ".batch_prediction_result")
                    
eval_table = (dbutils.widgets.get("catalog") + 
                    "." + dbutils.widgets.get("database") +
                    ".batch_pred_eval") 
                  
model_version = dbutils.widgets.get("model_version")
MODEL_NAME = dbutils.widgets.get("catalog") + "." + dbutils.widgets.get("database") + ".rad-meditron"

# COMMAND ----------

#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12
!pip install -q unstructured["local-inference"]==0.7.4 pillow
!pip install pydantic==1.8.2
!pip install --upgrade mlflow 
!pip install sentence-transformers

# COMMAND ----------

import os
import torch
import pandas as pd
import re
import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from pyspark.sql.functions import udf,col
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType
from sentence_transformers import SentenceTransformer, util

# COMMAND ----------

#read batch dataframe
test_data = spark.table(batch_tablename)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### We are using paraphrase-MiniLM-L6-v2 to perform semantic similarity between predictions and ground_truth. Semantic similarity is often considered a better metric for raw text labeling compared to other metrics because it takes into account the meaning and context of the text rather than relying solely on superficial features or word overlap. Semantic similarity considers the overall context and meaning of the text, allowing for a more nuanced understanding. This is crucial in tasks like text labeling where the goal is to accurately capture the intended message or content.

# COMMAND ----------

# Load pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define a UDF to calculate similarity
@udf(DoubleType())
def calculate_similarity(ground_truth, predictions):
    embeddings_ground_truth = model.encode([ground_truth])[0]
    embeddings_predictions = model.encode([predictions])[0]
    return float(util.pytorch_cos_sim(embeddings_ground_truth, embeddings_predictions).cpu().numpy().diagonal()[0])

# Apply the UDF to calculate similarity and add a new column
test_data = test_data.withColumn('semantic_similarity', calculate_similarity(test_data['ground_truth'], test_data['predictions']))

for c_name, c_type in test_data.dtypes:
    if c_type in ('double', 'float'):
        test_data = test_data.withColumn(c_name, F.round(c_name, 2))

# COMMAND ----------

#write eval dataframe to delta table
test_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(eval_table)
