# creating callbacks for the model training
import tensorflow as tf
from typing import List
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from src.logger import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class Callbacks:
    def __init__(self, model_dir: str, patience: int = 5, monitor: str = 'val_loss', mode: str = 'min') -> None:
        self.model_dir = model_dir
        self.patience = patience
        self.monitor = monitor
        self.mode = mode

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        logging.info("Creating callbacks for model training...")

        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        logging.info(f"Model directory created at: {self.model_dir}")
        # Define the callbacks

        logging.info("Creating EarlyStopping")
        # Early stopping to prevent overfitting

        early_stopping = EarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            restore_best_weights=True
        )

        logging.info("Creating ReduceLROnPlateau")
        # Reduce learning rate when a metric has stopped improving

        reduce_lr = ReduceLROnPlateau(
            monitor=self.monitor,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            mode=self.mode
        )
        logging.info("Creating ModelCheckpoint")
        # Save the best model based on the monitored metric

        
        model_checkpoint = ModelCheckpoint(
            filepath=self.model_dir,
            monitor=self.monitor,
            save_best_only=True,
            mode=self.mode
        )

        logging.info("Returning the list of callbacks.")
        logging.info(f"Callbacks: {[early_stopping, reduce_lr, model_checkpoint]}")
        logging.info("Callbacks creation completed.")

        callbacks_list = [early_stopping, reduce_lr, model_checkpoint]
        logging.info(f"Callbacks list: {callbacks_list}")

        return  callbacks_list
        logging.info("Callbacks creation completed.")

class ClassWeightCalculator:
    def __init__(self,train_classes) -> None:
        self.train_classes = train_classes
        """
        Initialize the ClassWeightCalculator.
        This class is used to compute class weights for imbalanced datasets.
        """
        logging.info("ClassWeightCalculator initialized.")

    def compute_class_weights(self) -> dict:
        """
        Compute class weights for imbalanced datasets.
        Args:
            train_classes: Array-like of class labels (e.g., train.classes from ImageDataGenerator).
        Returns:
            dict: Class weights in the format expected by Keras.
        """
        unique_classes = np.unique(self.train_classes)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=self.train_classes
        )

        logging.info(f"Computed class weights: {dict(zip(unique_classes, weights))}")
        logging.info("Class weights computed successfully.")
        class_weights = dict(zip(unique_classes, weights)) 
        return class_weights
        logging.info("ClassWeightCalculator compute_class_weights method completed.")





