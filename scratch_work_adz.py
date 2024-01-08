# Databricks notebook source
dbutils.fs.ls("/Volumes/ang_nara_catalog/rad_llm/clinical_data/12k_handwritten_clinical_notes.csv")

# COMMAND ----------

dbutils.fs.cp("/Volumes/ang_nara_catalog/rad_llm/clinical_data/12k_handwritten_clinical_notes.csv", 
              "/Workspace/Repos/aaron.zavora@databricks.com/radiology-llm-classifier/data/12k_handwritten_clinical_notes.csv")

# COMMAND ----------

dbutils.fs.ls("/Workspace/Repos/aaron.zavora@databricks.com/radiology-llm-classifier/data/")

# COMMAND ----------

# MAGIC %sh
# MAGIC cp /Volumes/ang_nara_catalog/rad_llm/clinical_data/filtered_clinical_notes.json /Workspace/Repos/aaron.zavora@databricks.com/radiology-llm-classifier/data/filtered_clinical_notes.json

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -ltr /Workspace/Repos/aaron.zavora@databricks.com/radiology-llm-classifier/data/12k_handwritten_clinical_notes.csv

# COMMAND ----------


