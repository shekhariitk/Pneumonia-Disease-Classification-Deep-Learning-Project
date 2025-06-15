import tensorflow as tf
from pathlib import Path
import os
from src.logger import logging
from src.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        input_shape = tuple(self.config.params_image_size)

        logging.info(f"Creating base model with input shape: {input_shape}")

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten()
        ])

        logging.info("Base model created successfully")
        logging.info(f"Base model summary:\n{self.model.summary()}")

        os.makedirs(self.config.base_model_path.parent, exist_ok=True)
        self.save_model(self.config.base_model_path, self.model)

        logging.info(f"Base model saved to {self.config.base_model_path}")
        return self.model

    def update_base_model(self):
        x = self.model.output

        logging.info("Updating base model with additional layers")
        # Add custom layers to the base model

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        if self.config.params_classes == 1:
            output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            output = tf.keras.layers.Dense(self.config.params_classes, activation="softmax")(x)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        full_model = tf.keras.models.Model(inputs=self.model.input, outputs=output)

        logging.info("Full model created with custom layers")
        logging.info(f"Full model summary:\n{full_model.summary()}")

        # ✅ Compile full model - all layers trainable
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=loss_fn,
            metrics=["accuracy"]
        )

        self.full_model = full_model
        full_model.summary()

        os.makedirs(self.config.updated_base_model_path.parent, exist_ok=True)
        self.save_model(self.config.updated_base_model_path, full_model)

        logging.info(f"Updated base model saved to {self.config.updated_base_model_path}")  
        

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logging.info(f"Model saved to {path}")