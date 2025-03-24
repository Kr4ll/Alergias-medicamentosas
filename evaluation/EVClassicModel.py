import joblib
import numpy as np
import spacy
from datasets import tqdm
from keras.src.utils import pad_sequences
import os
import re

from sklearn.metrics import classification_report, confusion_matrix

textos = []
etiquetas = []
def cargar_datos(ruta,textos,etiquetas, value):

    for archivo in os.listdir(ruta):
        with open(os.path.join(ruta, archivo), 'r', encoding='utf-8') as file:
            textos.append(file.read())
            etiquetas.append(value)

print("Loading data...")

ruta_datos_alergico = "../data/ByPatients/EVAAlergico"
ruta_datos_non = "../data/ByPatients/EVANoAlergico"
cargar_datos(ruta_datos_alergico,textos,etiquetas, "Allergic")
cargar_datos(ruta_datos_non,textos,etiquetas, "Non allergic")
models = []
modelsName = []
# Load the trained models
for archivo in os.listdir("../models/Classic_models"):
    models.append(joblib.load(f"../models/Classic_models/{archivo}"))
    modelsName.append(archivo[:-7])
tokenizer = joblib.load("../models/Classic_models/tokenizer.joblib")  

print("Text preprocessing...")
# Step 2: Text preprocessing
nlp = spacy.load("es_core_news_sm")

def preprocesar(texto):
    texto = re.sub(r'\*\*.*?\*\*', '', texto)
    doc = nlp(texto)
    tokens_lemmatizados = [token.lemma_ for token in doc if token.text.isalnum() or token.text.isspace()]
    return " ".join(tokens_lemmatizados)

textos_preprocesados = [preprocesar(texto) for texto in textos]

# Tokenize and pad sequences
X_new = tokenizer.transform(textos_preprocesados)

c=0
# Make predictions
for model in tqdm(models):
    y_pred = model.predict(X_new)

    report = classification_report(etiquetas, y_pred)
    conf_matrix_report = confusion_matrix(etiquetas, y_pred)
    TN, FP, FN, TP = conf_matrix_report.ravel()

    #Report saving

    report_file_path = os.path.join("../evaluation", f"{modelsName[c]}_classification_report.txt")
    with open(report_file_path, 'w') as report_file:
        report_file.write(report)
        print(conf_matrix_report)
        report_file.write(f"True Positives (TP): {TP}")
        report_file.write(f"True Negatives (TN): {TN}")
        report_file.write(f"False Positives (FP): {FP}")
        report_file.write(f"False Negatives (FN): {FN}")
    print(f"Classification report saved as: {report_file_path}")

    c=c+1