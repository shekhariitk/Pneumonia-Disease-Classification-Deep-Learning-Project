from src.logger import logging
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline




STAGE_NAME = "Data Ingestion stage"
try:
  logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
  data_ingestion = DataIngestionTrainingPipeline()
  data_ingestion.main()
  logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e


STAGE_NAME = "Prepare base model"
try: 
   logging.info(f"*******************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e


