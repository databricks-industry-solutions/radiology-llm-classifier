# Databricks notebook source
import pyspark.pandas as ps
import pyspark.sql.utils
import pandas as pd
import re
import dlt
from pyspark.sql.functions import *
ps.set_option('compute.ops_on_diff_frames', True)

# COMMAND ----------

@dlt.table
def load_data():
  data = pd.read_csv(
        "//Volumes/ang_nara_catalog/rad_llm/clinical_data/12k_handwritten_clinical_notes.csv"
    )
  data = data.drop(['Unnamed: 0'], axis=1)
  return spark.createDataFrame(data)


# COMMAND ----------

@dlt.table
def remove_label_counts_less_than_50():
  df = dlt.read('load_data')
  df.write.format("delta").mode("overwrite").option("overwriteSchema",True).saveAsTable("ang_nara_catalog.rad_llm.delta_rad")
  df = spark.sql("""
      SELECT t.input, t.radiology_labels
      FROM (
         SELECT t.*, COUNT(*) OVER (PARTITION BY radiology_labels) AS cnt
         FROM ang_nara_catalog.rad_llm.delta_rad t
      ) t
      WHERE cnt > 50
""")
  return df

# COMMAND ----------

@dlt.table
def filtered_table():
  df = dlt.read('remove_label_counts_less_than_50')
  df = df.withColumn("instruction", lit('predict radiology label for the clinical notes')) 
  df.write.format("delta").mode("overwrite").option("overwriteSchema",True).saveAsTable("ang_nara_catalog.rad_llm.delta_rad_filtered")
  return df
