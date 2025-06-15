from src.logger import logging
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.pipeline.stage_04_model_evaluation import EvaluationPipeline




def main():
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
        logging.info(f"\n{'*'*40}")
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n{'x'*40}")
    except Exception as e:
        logging.exception(e)
        raise e

    STAGE_NAME = "Model Training"
    try: 
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
    
    STAGE_NAME = "Evaluation stage"
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evalution = EvaluationPipeline()
        model_evalution.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logging.exception(e)
        raise e
    
    

if __name__ == "__main__":
    main()




