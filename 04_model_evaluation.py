# Databricks notebook source
<<<<<<< Updated upstream
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
                  
incorrect_pred_table = (dbutils.widgets.get("catalog") + 
                    "." + dbutils.widgets.get("database") +
                    ".batch_incorrec_pred") 
                  
model_version = dbutils.widgets.get("model_version")
MODEL_NAME = dbutils.widgets.get("catalog") + "." + dbutils.widgets.get("database") + ".rad-meditron"
=======
dbutils.widgets.text("model_version", "3")
dbutils.widgets.text("eval_table", "ang_nara_catalog.rad_llm.batch_pred_eval")
>>>>>>> Stashed changes

# COMMAND ----------

#install libraries
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 guardrail-ml==0.0.12
!pip install -q unstructured["local-inference"]==0.7.4 pillow
!pip install pydantic==1.8.2
!pip install --upgrade mlflow 
!pip install sentence-transformers

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_intersect, expr

# Example data as PySpark DataFrame
data = [("CT Urography History", "CT Urography"),
        ("CT Scan History", "CT Scan"),
        ("MRI History", "MRI")]

# Create a PySpark DataFrame
df = spark.createDataFrame(data, ["predictions", "ground_truth"])

# Split the predicitons into words
df = df.withColumn("predicted_words", expr("split(predictions, ' ')"))

# Find the common words between predictions and ground truth
df = df.withColumn("common_words", array_intersect(col("predicted_words"), expr("split(ground_truth, ' ')")))

# Join the common words to form the cleaned predictions
df = df.withColumn("cleaned_predictions", expr("concat_ws(' ', common_words)"))

# Select relevant columns
result_df = df.select("predictions", "ground_truth", "cleaned_predictions")

# Show the result
result_df.show(truncate=False)


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

test_data = spark.table(batch_tablename)

# COMMAND ----------

<<<<<<< Updated upstream
# Load pyfunc model from UC
mlflow.set_registry_uri("databricks-uc")
model_uri = "models:/{MODEL_NAME}/{model_version}".format(model_name=MODEL_NAME, model_version=model_version)
model =  mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
test_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(eval_table)

# COMMAND ----------

filtered_dataframe = test_data.filter(col("semantic_similarity") < 0.50)

# Write the filtered DataFrame to the Delta table
filtered_dataframe.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(incorrect_pred_table)

# COMMAND ----------

#TODO add stats / visuals here to easily consume output
=======
eval_table = dbutils.widgets.get("eval_table")

# COMMAND ----------

test_data.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(eval_table)
>>>>>>> Stashed changes
