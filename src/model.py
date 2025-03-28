# MIT License
# Copyright (c) 2025 Doncilă Denis

import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(7,)),  # 7 features
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes (SPIRAL, ELLIPTICAL, UNCERTAIN)
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
