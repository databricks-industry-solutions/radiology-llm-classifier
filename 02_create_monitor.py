# Databricks notebook source
# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.3.6-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks import lakehouse_monitoring as lm
info = lm.create_monitor(
    table_name=f"ang_nara_catalog.rad_llm.delta_rad_filtered",
    profile_type=lm.Snapshot(),
    output_schema_name=f"ang_nara_catalog.rad_llm"
)

# COMMAND ----------

import time
 
# Wait for monitor to be created
while info.status == lm.MonitorStatus.PENDING:
  info = lm.get_monitor(table_name=f"ang_nara_catalog.rad_llm.delta_rad_filtered")
  time.sleep(10)
 
assert(info.status == lm.MonitorStatus.ACTIVE)

# COMMAND ----------

# A metric refresh will automatically be triggered on creation
refreshes = lm.list_refreshes(table_name=f"ang_nara_catalog.rad_llm.delta_rad_filtered")
assert(len(refreshes) > 0)
 
run_info = refreshes[0]
while run_info.state in (lm.RefreshState.PENDING, lm.RefreshState.RUNNING):
  run_info = lm.get_refresh(table_name=f"ang_nara_catalog.rad_llm.delta_rad_filtered", refresh_id=run_info.refresh_id)
  time.sleep(30)
 
assert(run_info.state == lm.RefreshState.SUCCESS)

# COMMAND ----------

lm.get_monitor(table_name=f"ang_nara_catalog.rad_llm.delta_rad_filtered")
