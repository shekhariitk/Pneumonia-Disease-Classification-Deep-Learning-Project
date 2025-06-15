import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from src.entity.config_entity import TrainingConfig
from src.components.callbacks import Callbacks,ClassWeightCalculator
from src.logger import logging


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="binary",
            color_mode="rgb",
            shuffle=True
        )
        logging.info("Creating validation and training data generators...")
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data,"test"),
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            logging.info("Creating training data generator with augmentation...")
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            logging.info("Using validation data generator for training as augmentation is disabled.")
            logging.info("Creating training data generator without augmentation...")
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data,"train"),
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    # Load the callbacks
    logging.info("Loading callbacks for model training...")
    def load_callbacks(self):
        callbacks = Callbacks(
            model_dir=self.config.best_trained_model_path,
            patience=5,
            monitor='val_loss',
            mode='min'
        )
        return callbacks.get_callbacks()
    
    # compute class weights
    def compute_class_weights(self):
        logging.info("Calculating class weights for imbalanced dataset...")
        class_weight_calculator = ClassWeightCalculator(
            train_classes=self.train_generator.classes
        )
        return class_weight_calculator.compute_class_weights()
    




    
    def train(self):
        logging.info("Starting model training...")
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        logging.info(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")


        logging.info("Fitting the model...")
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=self.load_callbacks(),
            verbose=1,
            class_weight=self.compute_class_weights()

        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
