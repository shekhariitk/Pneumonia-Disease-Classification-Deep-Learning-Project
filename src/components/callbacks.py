# callbacks.py
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

        # Construct file path to save the best model
        model_path = os.path.join(self.model_dir, "best_model.h5")
        logging.info(f"Model will be saved to: {model_path}")

        # Callback: EarlyStopping
        logging.info("Creating EarlyStopping")
        early_stopping = EarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            restore_best_weights=True
        )

        # Callback: ReduceLROnPlateau
        logging.info("Creating ReduceLROnPlateau")
        reduce_lr = ReduceLROnPlateau(
            monitor=self.monitor,
            factor=0.2,
            patience=2,
            min_lr=1e-8,
            mode=self.mode
        )

        # Callback: ModelCheckpoint
        logging.info("Creating ModelCheckpoint")
        model_checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor=self.monitor,
            save_best_only=True,
            mode=self.mode
        )

        callbacks_list = [early_stopping, reduce_lr, model_checkpoint]
        logging.info(f"Callbacks list created: {callbacks_list}")
        return callbacks_list


class ClassWeightCalculator:
    def __init__(self, train_classes) -> None:
        self.train_classes = train_classes
        logging.info("ClassWeightCalculator initialized.")

    def compute_class_weights(self) -> dict:
        """
        Compute class weights for imbalanced datasets.
        Returns:
            dict: Class weights in the format expected by Keras.
        """
        unique_classes = np.unique(self.train_classes)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=self.train_classes
        )
        class_weights = dict(zip(unique_classes, weights))
        logging.info(f"Computed class weights: {class_weights}")
        return class_weights




