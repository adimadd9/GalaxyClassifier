# MIT License
# Copyright (c) 2025 Doncilă Denis

import os
import sys
import customtkinter as ctk
from tkinter import messagebox, filedialog, PhotoImage
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocess import load_data
import matplotlib.pyplot as plt
from PIL import Image

# Setting the theme for the interface
ctk.set_appearance_mode("dark")  # Mode: "dark" or "light"
ctk.set_default_color_theme("blue") # Default color theme: "blue"

# Returns the model file path, considering whether the application is run as an executable or a Python script.
def get_model_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, "model/galaxy_classifier.h5")
    else:
        return "model/galaxy_classifier.h5"

# Load the model
model_path = get_model_path()
model = tf.keras.models.load_model(model_path)

# Load the dataset for normalization
features, _ = load_data("data/galaxies.csv")
scaler = StandardScaler()
scaler.fit(features)

# Creating the main window
root = ctk.CTk()
root.title("Clasificare Galaxii")
root.geometry("500x600")
root.resizable(False, False)
root.iconbitmap("img/icon.ico")

# Loading the background image
bg_image = ctk.CTkImage(light_image=Image.open("img/bg.jpg"),
                        dark_image=Image.open("img/bg.jpg"),
                        size=(500, 600))

# Adding the background image
bg_label = ctk.CTkLabel(root, image=bg_image, text="")  
bg_label.place(relwidth=1, relheight=1)

# Creating a semi-transparent frame over the background
main_frame = ctk.CTkFrame(root, corner_radius=15, fg_color=("gray85", "gray20"), width=450, height=550)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

# Title
title_label = ctk.CTkLabel(main_frame, text="Clasificare Galaxii", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

# List of features and entries
feature_names = ['P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG', 'P_CS']
entries = []
categories = ["SPIRAL", "ELLIPTICAL", "UNCERTAIN"]

entry_frame = ctk.CTkFrame(main_frame, corner_radius=15, fg_color="transparent")
entry_frame.pack(pady=5, padx=20)

for feature in feature_names:
    row = ctk.CTkFrame(entry_frame, fg_color="transparent")
    row.pack(fill="x", pady=2)
    ctk.CTkLabel(row, text=feature, width=150, anchor="w").pack(side="left")
    entry = ctk.CTkEntry(row, width=150)
    entry.pack(side="right", padx=10)
    entries.append(entry)

# Prediction function
def predict():
    try:
        input_data = np.array([[float(entry.get()) for entry in entries]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        predicted_class = categories[np.argmax(prediction)]
        messagebox.showinfo("Predicție", f"Tipul obiectului cosmic: {predicted_class}")
    except ValueError:
        messagebox.showerror("Eroare", "Completează toate câmpurile cu valori numerice valide.")

# CSV loading function
def load_csv():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("Fișierul CSV este gol")
            messagebox.showinfo("Fișier Încărcat", f"{len(df)} rânduri încărcate.")
            process_csv(df)
    except Exception as e:
        messagebox.showerror("Eroare", f"Nu s-a putut încărca fișierul: {str(e)}")

# CSV processing function
def process_csv(df):
    try:
        if df is None or df.empty:
            raise ValueError("DataFrame-ul este gol sau invalid")
        
        input_data = df[feature_names].values
        input_data_scaled = scaler.transform(input_data)
        predictions = model.predict(input_data_scaled, verbose=0)
        df['Predicted_Class'] = [categories[np.argmax(p)] for p in predictions]
        
        # Check if the directory exists and create it if not
        if not os.path.exists("data"):
            os.makedirs("data")
        
        df.to_csv("data/predictions.csv", index=False)
        messagebox.showinfo("Succes", "Predicțiile au fost salvate în predictions.csv!")
    except Exception as e:
        messagebox.showerror("Eroare", str(e))

# Graph display function
def show_results():
    df = pd.read_csv("data/predictions.csv")
    counts = df["Predicted_Class"].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=["gray", "red", "blue"])
    plt.xlabel("Tipul Obiectului Cosmic")
    plt.ylabel("Număr de Predicții")
    plt.title("Distribuția Clasificărilor")
    plt.show()

# Predictions file download function
def download_predictions():
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")],
                                                 title="Salvează fișierul cu predicții")
        if file_path:
            df = pd.read_csv("data/predictions.csv")
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Succes", "Fișierul a fost salvat cu succes!")
    except Exception as e:
        messagebox.showerror("Eroare", "Nu există predicții disponibile pentru descărcare!")

# Creating the buttons
button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
button_frame.pack(pady=10)

ctk.CTkButton(button_frame, text="Clasifică", fg_color="#53316c", hover_color="#432857", command=predict, width=200).pack(pady=5)
ctk.CTkButton(button_frame, text="Încarcă CSV", fg_color="#53316c", hover_color="#432857", command=load_csv, width=200).pack(pady=5)
ctk.CTkButton(button_frame, text="Vezi Grafic", fg_color="#53316c", hover_color="#432857", command=show_results, width=200).pack(pady=5)
ctk.CTkButton(button_frame, text="Descarcă predicțiile", fg_color="#53316c", hover_color="#432857", command=download_predictions, width=200).pack(pady=5)

# Starting the application
root.mainloop()