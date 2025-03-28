# MIT License
# Copyright (c) 2025 Doncilă Denis

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical # type: ignore
from preprocess import load_data
from model import build_model

# Load the data
features, labels = load_data("data/galaxies.csv")

# Normalize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert the labels into a usable format
labels_encoded = np.argmax(labels.values, axis=1)
labels_categorical = to_categorical(labels_encoded)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_categorical, test_size=0.2, random_state=42)

# Build the model
model = build_model()

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save("model/galaxy_classifier.h5")
