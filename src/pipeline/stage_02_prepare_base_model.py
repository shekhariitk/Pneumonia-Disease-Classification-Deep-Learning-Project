from src.config.configuration import ConfigurationManager
from src.components.prepare_base_model import PrepareBaseModel
from src.logger import logging

STAGE_NAME = "Prepare Base Model with Custom CNN"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logging.info(f"\n{'='*40}\n>>> Stage {STAGE_NAME} started <<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logging.info(f">>> Stage {STAGE_NAME} completed <<<\n{'='*40}")
    except Exception as e:
        logging.exception(e)
        raise e
