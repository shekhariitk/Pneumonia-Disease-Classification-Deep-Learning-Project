# 🧠 Pneumonia Detection using Deep Learning (TensorFlow)

This project is designed to detect **pneumonia** in chest X-ray images using a custom-built convolutional neural network (CNN). It follows a modular and class-based architecture suitable for both experimentation and production.

---

## 🚀 Features

- Custom CNN architecture with dense layers
- Modular code with reusable components
- Config-driven (YAML-based) setup
- Class-weight balancing for imbalanced data
- Augmentation support
- Logging system
- Callback support (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Model saving and checkpointing
- Easily extensible and production-ready

---

## 🛠️ Technologies Used

- Python 3.10
- TensorFlow 2.x
- NumPy, Matplotlib, Scikit-learn
- YAML, logging
- Object-Oriented Programming

---

## 🧩 Project Modules

### 1. `prepare_base_model.py`
- Creates a base CNN with Conv2D, MaxPooling, and Flatten layers.
- Adds fully connected (Dense) layers and compiles the model.
- Saves the base and updated model.

### 2. `training.py`
- Loads model from `updated_base_model_path`.
- Sets up training and test data generators.
- Handles augmentation, class weights, and training logic.
- Saves the final trained model.

### 3. `callbacks.py`
- Creates callbacks for EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- Computes class weights using `sklearn.utils.class_weight`.

### 4. `config_entity.py`
- Data classes for type-safe access to configuration values.

### 5. `configuration.py`
- Reads configuration and parameter values from `config.yaml` and `params.yaml`.

### 6. `logger.py`
- Centralized logging setup for tracking the pipeline.

---

## 📁 Folder Structure

- `src/` — Main source code
- `artifacts/` — Stores models and intermediate outputs
- `configs/` — Contains `config.yaml` and `params.yaml`
- `notebook/` — Jupyter notebook for experimentation
- `main.py` — Pipeline execution entry point
- `app.py` — (Optional) For deploying the model as a web service
- `README.md` — Documentation
- `requirements.txt` — Python dependencies

---

## ⚙️ Configuration Files

### `config.yaml`
Stores paths and structural settings.

###  ` Workflows Of the projects`
- Update config.yaml
- Update secrets.yaml [Optional]
- Update params.yaml
- Update the entity
- Update the configuration manager in src config
- Update the components
- Update the pipeline
- Update the main.py
- Update the dvc.yaml
- app.py

