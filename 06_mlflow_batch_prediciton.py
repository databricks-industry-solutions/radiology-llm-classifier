# Databricks notebook source
!pip install mlflow --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

import pandas as pd
data = pd.read_csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/batch_prediction_20_notes.csv')
notes  = data[['clinical_notes']]

# COMMAND ----------

# You can update the catalog and schema name containing the model in Unity Catalog if needed
CATALOG_NAME = "ang_nara_catalog"
SCHEMA_NAME = "rad_llm"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.radllama2_7b_test"

# COMMAND ----------

import mlflow.pyfunc
model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

prediction = notes.apply(lambda row: model.predict(row), axis=1)
data['predicition'] = prediction
spark_data = spark.createDataFrame(data)

# COMMAND ----------

spark_data.write.format("delta").mode("overwrite").saveAsTable("ang_nara_catalog.rad_llm.rad_batch_pred_20_notes")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ang_nara_catalog.rad_llm.rad_batch_pred_20_notes
