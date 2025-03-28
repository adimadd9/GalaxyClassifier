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

# Configurare tema pentru interfață
ctk.set_appearance_mode("dark")  # Mod: "dark" sau "light"
ctk.set_default_color_theme("blue")

def get_model_path():
    """Returnează calea fișierului modelului, luând în considerare dacă aplicația este rulată ca executabil sau script Python."""
    if getattr(sys, 'frozen', False):  # Dacă aplicația este rulată ca executabil
        # Căutăm modelul în directorul unde se află fișierul executabil
        return os.path.join(sys._MEIPASS, "model/galaxy_classifier.h5")
    else:
        # Dacă aplicația este rulată dintr-un script, utilizăm calea relativă
        return "model/galaxy_classifier.h5"

# Încarcă modelul
model_path = get_model_path()
model = tf.keras.models.load_model(model_path)

# Încarcă setul de date pentru normalizare
features, _ = load_data("data/galaxies.csv")
scaler = StandardScaler()
scaler.fit(features)

# Crearea ferestrei principale
root = ctk.CTk()
root.title("Clasificare Galaxii")
root.geometry("500x600")
root.resizable(False, False)
root.iconbitmap("img/icon.ico")

# Încărcarea imaginii de fundal
bg_image = ctk.CTkImage(light_image=Image.open("img/bg.jpg"),
                        dark_image=Image.open("img/bg.jpg"),
                        size=(500, 600))

# Adăugarea imaginii de fundal
bg_label = ctk.CTkLabel(root, image=bg_image, text="")  
bg_label.place(relwidth=1, relheight=1)  # Fundal pe toată fereastra

# Crearea unui Frame semi-transparent peste fundal
main_frame = ctk.CTkFrame(root, corner_radius=15, fg_color=("gray85", "gray20"), width=450, height=550)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

# Titlu
title_label = ctk.CTkLabel(main_frame, text="Clasificare Galaxii", font=("Arial", 18, "bold"))
title_label.pack(pady=15)

# Lista de caracteristici și intrări
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

# Funcția de predicție
def predict():
    try:
        input_data = np.array([[float(entry.get()) for entry in entries]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        predicted_class = categories[np.argmax(prediction)]
        messagebox.showinfo("Predicție", f"Tipul obiectului cosmic: {predicted_class}")
    except ValueError:
        messagebox.showerror("Eroare", "Completează toate câmpurile cu valori numerice valide.")

# Funcția de încărcare CSV
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


# Funcția de procesare CSV
def process_csv(df):
    try:
        if df is None or df.empty:
            raise ValueError("DataFrame-ul este gol sau invalid")
        
        input_data = df[feature_names].values
        input_data_scaled = scaler.transform(input_data)
        predictions = model.predict(input_data_scaled, verbose=0)
        df['Predicted_Class'] = [categories[np.argmax(p)] for p in predictions]
        
        # Verifică dacă directorul există și creează-l dacă nu
        if not os.path.exists("data"):
            os.makedirs("data")
        
        df.to_csv("data/predictions.csv", index=False)
        messagebox.showinfo("Succes", "Predicțiile au fost salvate în predictions.csv!")
    except Exception as e:
        messagebox.showerror("Eroare", str(e))

# Funcția de afișare grafic
def show_results():
    df = pd.read_csv("data/predictions.csv")
    counts = df["Predicted_Class"].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=["gray", "red", "blue"])
    plt.xlabel("Tipul Obiectului Cosmic")
    plt.ylabel("Număr de Predicții")
    plt.title("Distribuția Clasificărilor")
    plt.show()

# Funcția de descărcare a fișierului de predicții
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

# Crearea butoanelor
button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
button_frame.pack(pady=10)

ctk.CTkButton(button_frame, text="Clasifică", fg_color="#53316c", hover_color="#432857", command=predict, width=200).pack(pady=5)
ctk.CTkButton(button_frame, text="Încarcă CSV", fg_color="#53316c", hover_color="#432857", command=load_csv, width=200).pack(pady=5)
ctk.CTkButton(button_frame, text="Vezi Grafic", fg_color="#53316c", hover_color="#432857", command=show_results, width=200).pack(pady=5)
ctk.CTkButton(button_frame, text="Descarcă predicțiile", fg_color="#53316c", hover_color="#432857", command=download_predictions, width=200).pack(pady=5)

# Pornirea aplicației
root.mainloop()
