# Databricks notebook source
import os
from pyspark.sql.functions import *
#
# load handwritten notes from csv
#
def load_data(path="/data/12k_handwritten_clinical_notes.csv"):
  return (spark.read.format("csv")
          .option("header",True)
          .load("file:///" + os.getcwd() +  path)
  )

# COMMAND ----------

#
# Only use where there exists 50 or more labels
#
def filtered_table():
  df = load_data()
  df.createOrReplaceTempView("radiology_data")
  return spark.sql("""
    SELECT t.input, t.radiology_labels
      FROM (
         SELECT t.*, COUNT(*) OVER (PARTITION BY radiology_labels) AS cnt
         FROM radiology_data t
      ) t
      WHERE cnt > 50
  """).withColumn("instruction", lit('predict radiology label for the clinical notes')) 

# COMMAND ----------

data = df_label_counts_less_than_50()
data.show()
