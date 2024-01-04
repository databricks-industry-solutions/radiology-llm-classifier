-- Databricks notebook source
select 
count (*)
from uc_demos_angelina_leigh.nara_ang_demo.delta_med_icd10_filtered

-- COMMAND ----------

select * from system.billing.usage where usage_metadata.cluster_id = '0601-182128-dcbte59m' and usage_date = '2023-11-03'

-- COMMAND ----------

select * from john_snow_labs_icd_9_icd_10_and_clinical_classification_codes.icd_9_icd_10_and_clinical_classification_codes.clinical_classification_software_for_icd_10_cm

-- COMMAND ----------

use catalog uc_demos_angelina_leigh;
use schema nara_ang_demo

-- COMMAND ----------

--create table icd10_jsl as 
select *
from
  john_snow_labs_icd_9_icd_10_and_clinical_classification_codes.icd_9_icd_10_and_clinical_classification_codes.clinical_classification_software_for_icd_10_cm
group by
  all
having count(ICD10CM_Code) >= 5

-- COMMAND ----------

select * from icd10_jsl

-- COMMAND ----------

-- MAGIC %python
-- MAGIC (trainingData, testData) = df.randomSplit([0.7, 0.3], seed = 100)
-- MAGIC print("Training Dataset Count: " + str(trainingData.count()))
-- MAGIC print("Test Dataset Count: " + str(testData.count()))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col
-- MAGIC
-- MAGIC trainingData.groupBy("protocol_labels") \
-- MAGIC     .count() \
-- MAGIC     .orderBy(col("count").desc()) \
-- MAGIC     .show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC !pip install spark-nlp==5.1.3

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import mlflow
-- MAGIC import mlflow.spark
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC from sparknlp.base import *
-- MAGIC from sparknlp.annotator import *
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC import pandas as pd
-- MAGIC import os
-- MAGIC
-- MAGIC with mlflow.start_run(run_name="JSL-Text-Classification") as run:
-- MAGIC     # Define pipeline
-- MAGIC     document_assembler = DocumentAssembler() \
-- MAGIC         .setInputCol("studydesc+procedurecode+reasonofstudy") \
-- MAGIC         .setOutputCol("document")
-- MAGIC
-- MAGIC     tokenizer = Tokenizer() \
-- MAGIC         .setInputCols(["document"]) \
-- MAGIC         .setOutputCol("token")
-- MAGIC
-- MAGIC     normalizer = Normalizer() \
-- MAGIC         .setInputCols(["token"]) \
-- MAGIC         .setOutputCol("normalized")
-- MAGIC
-- MAGIC     stopwords_cleaner = StopWordsCleaner()\
-- MAGIC         .setInputCols("normalized")\
-- MAGIC         .setOutputCol("cleanTokens")\
-- MAGIC         .setCaseSensitive(False)
-- MAGIC
-- MAGIC     lemma = LemmatizerModel.pretrained('lemma_antbnc') \
-- MAGIC         .setInputCols(["cleanTokens"]) \
-- MAGIC         .setOutputCol("lemma")
-- MAGIC
-- MAGIC     glove_embeddings = WordEmbeddingsModel().pretrained() \
-- MAGIC         .setInputCols(["document",'lemma'])\
-- MAGIC         .setOutputCol("embeddings")\
-- MAGIC         .setCaseSensitive(False)
-- MAGIC
-- MAGIC     embeddingsSentence = SentenceEmbeddings() \
-- MAGIC         .setInputCols(["document", "embeddings"]) \
-- MAGIC         .setOutputCol("sentence_embeddings") \
-- MAGIC         .setPoolingStrategy("AVERAGE")
-- MAGIC
-- MAGIC     classsifierdl = ClassifierDLApproach()\
-- MAGIC         .setInputCols(["sentence_embeddings"])\
-- MAGIC         .setOutputCol("class")\
-- MAGIC         .setLabelColumn("protocol_labels")\
-- MAGIC         .setMaxEpochs(3)\
-- MAGIC         .setEnableOutputLogs(True)
-- MAGIC         #.setOutputLogsPath('logs')
-- MAGIC
-- MAGIC     clf_pipeline = Pipeline(
-- MAGIC         stages=[
-- MAGIC             document_assembler,
-- MAGIC             tokenizer,
-- MAGIC             normalizer,
-- MAGIC             stopwords_cleaner,
-- MAGIC             lemma,
-- MAGIC             glove_embeddings,
-- MAGIC             embeddingsSentence,
-- MAGIC             classsifierdl
-- MAGIC         ]
-- MAGIC     )
-- MAGIC
-- MAGIC     clf_pipelineModel = clf_pipeline.fit(trainingData)
-- MAGIC
-- MAGIC     # Log parameters
-- MAGIC     mlflow.log_param("label", "protocol_labels")
-- MAGIC     mlflow.log_param("features", "studydesc+procedurecode+reasonofstudy")
-- MAGIC
-- MAGIC     # Log model
-- MAGIC     mlflow.spark.log_model(clf_pipelineModel, "model") 
-- MAGIC
-- MAGIC     # Evaluate predictions
-- MAGIC     preds = clf_pipelineModel.transform(testData)
-- MAGIC     preds.select('protocol_labels','studydesc+procedurecode+reasonofstudy',"class.result").show(10, truncate=80)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC preds_df = preds.select('protocol_labels','studydesc+procedurecode+reasonofstudy',"class.result").toPandas()
-- MAGIC preds_df['result'] = preds_df['result'].apply(lambda x : x[0])

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from sklearn.metrics import classification_report
-- MAGIC
-- MAGIC print (classification_report(preds_df['protocol_labels'], preds_df['result']))
