import tensorflow as tf
from pathlib import Path
import mlflow
import os
import mlflow.keras
from dotenv import load_dotenv
from urllib.parse import urlparse

from dagshub import dagshub_logger  # ✅ You missed this import
from src.entity.config_entity import EvaluationConfig
from src.utils.common import read_yaml, create_directories, save_json

from src.logger import logging

# ✅ Load environment variables from .env file
load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")

# ✅ Initialize Dagshub logger (required for secure logging)
#dagshub_logger(repo_name=REPO_NAME, repo_owner=DAGSHUB_USERNAME, token=DAGSHUB_TOKEN)



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="binary"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}

        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)  # optional, set this only if needed
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log all hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })

            # Log model (without registry, for DagsHub compatibility)
            mlflow.keras.log_model(self.model, "model")
            logging.info("Model and metrics logged into MLflow successfully.")
