# Databricks notebook source
pip install --upgrade "mlflow-skinny>=2.4.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
data = spark.read.csv('/Volumes/ang_nara_catalog/rad_llm/clinical_data/test_batch_data.csv', header=True).limit(20)

# COMMAND ----------

# You can update the catalog and schema name containing the model in Unity Catalog if needed
CATALOG_NAME = "ang_nara_catalog"
SCHEMA_NAME = "rad_llm"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.radllama2_7b_test"

# COMMAND ----------

data.write.mode("overwrite").saveAsTable("hls_healthcare.hls_dev.rad_data_input")

# COMMAND ----------

# MAGIC %md ## UDF() & ModelWrapper 

# COMMAND ----------

from datetime import datetime
model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)

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
  
  @staticmethod
  def rdd_mapPartitions(rows):
    return {**row.asDict(), **{"model_prediction": model.predict(row.asDict().get("clinical_notes"))}}
  
  def evaluate_udf(self, val):
    print(str(datetime.now()) + "::ADZ model evaluate called with val " + val)
    ModelWrapper.get_model(self.model_uri)
    return ModelWrapper.model.predict(val)

#model = ModelWrapper(model_uri)
#model_udf = udf(model.evaluate_udf)
#load_model_udf = udf(model.get_model)

# COMMAND ----------

# MAGIC %md ### RDD / ModelWrapper

# COMMAND ----------

model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)

def rdd_map(row, wrapper):
    print(str(datetime.now()) + "::ADZ  start of prediction for " + str(row))
    result = {**row.asDict(), **{"model_prediction": wrapper.model.predict(row.asDict().get("clinical_notes"))}}
    #result = {**row.asDict(), **{"model_prediction": row.asDict().get("clinical_notes")}}
    print(str(datetime.now()) + "::ADZ  end of prediction for " + str(row))
    return result

def rdd_mapPartitions(rows):
  import mlflow
  mlflow.set_registry_uri("databricks-uc")
  model = ModelWrapper(model_uri)
  print(str(datetime.now()) + "::ADZ  call to load model ")
  ModelWrapper.get_model(model_uri)
  print(str(datetime.now()) + "::ADZ  done loading model ")
  return map(lambda row: rdd_map(row, model), rows)

# COMMAND ----------

# DBTITLE 1,Reduce Logging 
# MAGIC %scala
# MAGIC sc.parallelize(Seq("")).foreachPartition(x => {
# MAGIC   import org.apache.log4j.{LogManager, Level}
# MAGIC   import org.apache.commons.logging.LogFactory
# MAGIC  
# MAGIC   LogManager.getRootLogger().setLevel(Level.ERROR)
# MAGIC   val log = LogFactory.getLog("EXECUTOR-LOG:")
# MAGIC   log.debug("START EXECUTOR ERROR LOG LEVEL")
# MAGIC })

# COMMAND ----------

rdd_with_prediction = spark.table("hls_healthcare.hls_dev.rad_data_input").rdd.repartition(20).mapPartitions(rdd_mapPartitions)
rdd_with_prediction.take(20) 


# COMMAND ----------

#rdd_with_prediction = spark.table("hls_healthcare.hls_dev.rad_data_input").rdd.mapPartitions(rdd_mapPartitions)
#rdd_with_prediction.take(20) 
#without repartition Command took 1.03 hours -- by aaron.zavora@databricks.com at 2/13/2024, 10:47:42 AM on Aaron Zavora's GPU Cluster (clone)

# COMMAND ----------

# MAGIC %md ## Srijit pandas suggestion

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from typing import Iterator

@pandas_udf('string', PandasUDFType.SCALAR_ITER)
def answer_question_udf(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
  import mlflow
  mlflow.set_registry_uri("databricks-uc")
  model = ModelWrapper(model_uri)
  ModelWrapper.get_model(model_uri)

  for texts in batches:
      answers = model.model.predict(texts)
      yield pd.Series(answers)

# COMMAND ----------

#2 minutes to load model
prediction_df = data.withColumn("model_prediction", answer_question_udf(data["clinical_notes"]))
display(prediction_df)

# COMMAND ----------

data = data.repartition(20)
prediction_df = data.withColumn("model_prediction", answer_question_udf(data["clinical_notes"]))
display(prediction_df)

# COMMAND ----------

# MAGIC %md ## pandas_udf() without model predict

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Load pyfunc model from UC
model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)

@pandas_udf('string', PandasUDFType.SCALAR_ITER)
def model_eval(it: pd.Series):
  import mlflow
  mlflow.set_registry_uri("databricks-uc")
  model = ModelWrapper(model_uri)
  ModelWrapper.get_model(model_uri)
  return map(lambda x: x, it)

# COMMAND ----------

#2 minutes to load model
prediction_df = data.withColumn("model_prediction", model_eval(data["clinical_notes"]))
prediction_df.show()

# COMMAND ----------

# MAGIC %md ## pands_udf() with predict

# COMMAND ----------




# Load pyfunc model from UC
model_uri = "models:/{model_name}@radllama".format(model_name=MODEL_NAME)

@pandas_udf('string', PandasUDFType.SCALAR_ITER)
def model_eval(it: pd.Series):
  import mlflow
  mlflow.set_registry_uri("databricks-uc")
  model = ModelWrapper(model_uri)
  ModelWrapper.get_model(model_uri)
  return map(lambda x: model.predict(x), it)

# COMMAND ----------

#time taken running predictions 58 minutes
prediction_df = data.withColumn("model_prediction", model_eval(data["clinical_notes"]))
prediction_df.show()

# COMMAND ----------


