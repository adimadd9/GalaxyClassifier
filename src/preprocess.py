# MIT License
# Copyright (c) 2025 Doncilă Denis

import pandas as pd

def load_data(file_path):
    # Citirea datelor
    df = pd.read_csv(file_path)
    
    # Selectarea caracteristicilor relevante pentru clasificare
    features = df[['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG', 'P_CS',]]
    labels = df[['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']]
    
    return features, labels

if __name__ == "__main__":
    features, labels = load_data("data/galaxies.csv")
    print(features.head())  # Afișează primele 5 rânduri
