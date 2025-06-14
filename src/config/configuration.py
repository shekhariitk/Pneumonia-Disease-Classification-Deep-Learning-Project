from src.constants import *
import os
from src.utils.common import read_yaml, create_directories,save_json
from src.entity.config_entity import (DataIngestionConfig,
                                         PrepareBaseModelConfig,
                                         TrainingConfig,
                                         EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.PARAMS_IMAGE_SIZE,
            params_learning_rate=self.params.PARAMS_LEARNING_RATE,
            params_include_top=self.params.PARAMS_INCLUDE_TOP,
            params_weights=self.params.PARAMS_WEIGHTS,
            params_classes=self.params.PARAMS_CLASSES
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "chest_xray")
        create_directories([
            Path(training.root_dir)
        ])
        create_directories([
            Path(training.model_dir)
        ])


        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.PARAMS_EPOCHS,
            params_batch_size=params.PARAMS_BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.PARAMS_IMAGE_SIZE,
            best_trained_model_path=Path(training.best_trained_model_path)


        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.config.training.best_trained_model_path),
            training_data= os.path.join(self.config.data_ingestion.unzip_dir, "chest_xray","test"),
            mlflow_uri="https://dagshub.com/shekhariitk/Pneumonia-Disease-Classification-Deep-Learning-Project.mlflow",
            all_params=self.params,
            params_image_size=self.params.PARAMS_IMAGE_SIZE,
            params_batch_size=self.params.PARAMS_BATCH_SIZE
        )
        return eval_config
  