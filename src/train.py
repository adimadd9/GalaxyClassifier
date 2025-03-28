# MIT License
# Copyright (c) 2025 Doncilă Denis

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical # type: ignore
from preprocess import load_data
from model import build_model

# Încarcă datele
features, labels = load_data("data/galaxies.csv")

# Normalizează datele
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convertirea labelurilor într-un format utilizabil
labels_encoded = np.argmax(labels.values, axis=1)
labels_categorical = to_categorical(labels_encoded)

# Împărțirea setului de date în antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_categorical, test_size=0.2, random_state=42)

# Construirea modelului
model = build_model()

# Antrenarea modelului
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Salvarea modelului antrenat
model.save("model/galaxy_classifier.h5")
