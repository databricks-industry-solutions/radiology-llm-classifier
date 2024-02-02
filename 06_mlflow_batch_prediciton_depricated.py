# Databricks notebook source
pip install --upgrade "mlflow-skinny[databricks]>=2.4.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

data = spark.read.csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/test_batch_data.csv', header=True).limit(20)

# COMMAND ----------

# You can update the catalog and schema name containing the model in Unity Catalog if needed
CATALOG_NAME = "ang_nara_catalog"
SCHEMA_NAME = "rad_llm"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.radllama2_7b_test"

# COMMAND ----------

# Load pyfunc model from UC
model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Prediction using PySpark dataframe though Vectorized Pandas UDF (Faster than a PySpark UDF)

# COMMAND ----------

from datetime import datetime
class ModelWrapper(): 
  def __init__(self, model_uri):
    self.model_uri = model_uri
    ModelWrapper.model = None

  @staticmethod
  def get_model(model_uri = None):
    if ModelWrapper.model is None and model_uri is not None:
      print(str(datetime.now()) + "::ADZ Startup of get_model with model_uri param " + model_uri)
      import mlflow
      mlflow.set_registry_uri("databricks-uc")
      ModelWrapper.model = mlflow.pyfunc.load_model(model_uri, suppress_warnings=True)
    print(str(datetime.now()) +  "::ADZ model finished loading with model_uri param " + model_uri)
    return None

  def evaluate_udf(self, val):
    print(str(datetime.now()) + "::ADZ model evaluate called with val " + val)
    ModelWrapper.get_model(self.model_uri)
    return ModelWrapper.model.predict(val)

model = ModelWrapper(model_uri)
model_udf = udf(model.evaluate_udf)
load_model_udf = udf(model.get_model)

# COMMAND ----------

from pyspark.sql.functions import lit
#time taken to load model on all executors ~6 minutes
#data.withColumn("dummy", load_model_udf(model_uri)).show()

# COMMAND ----------

#time taken running predictions
prediction_df = data.withColumn("model_prediction", model_udf(data["clinical_notes"]))
prediction_df.show()

# COMMAND ----------

model

# COMMAND ----------

"""
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
#original: 10.90 minutes runtime

@udf
def model_predict_udf(val):
    return loaded_model.predict(val)

# Apply the Pandas UDF to the DataFrame
prediction_df = data.withColumn("model_prediction", model_predict_udf(data['clinical_notes']))

# Display the result
display(prediction_df)
"""

# COMMAND ----------


