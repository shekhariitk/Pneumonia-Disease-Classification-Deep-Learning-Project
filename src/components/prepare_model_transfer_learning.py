import tensorflow as tf
from pathlib import Path
import os
from src.logger import logging
from src.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModelTransferLearning:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        input_shape = tuple(self.config.params_image_size)
        logging.info(f"Loading MobileNetV2 with input shape: {input_shape}")

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'  # use pretrained weights
        )
        base_model.trainable = False  # freeze base model

        self.base_model = base_model

        logging.info("Base MobileNetV2 model loaded and frozen")
        os.makedirs(self.config.base_model_path.parent, exist_ok=True)
        self.save_model(self.config.base_model_path, base_model)

        logging.info(f"Base model saved to {self.config.base_model_path}")
        return base_model

    def update_base_model(self):
        logging.info("Building full model with custom classification head")

        x = self.base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        if self.config.params_classes == 1:
            output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            output = tf.keras.layers.Dense(self.config.params_classes, activation="softmax")(x)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        full_model = tf.keras.models.Model(inputs=self.base_model.input, outputs=output)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=loss_fn,
            metrics=["accuracy"]
        )

        self.full_model = full_model

        logging.info("Full model created and compiled")
        full_model.summary()

        os.makedirs(self.config.updated_base_model_path.parent, exist_ok=True)
        self.save_model(self.config.updated_base_model_path, full_model)

        logging.info(f"Updated model saved to {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logging.info(f"Model saved to {path}")
