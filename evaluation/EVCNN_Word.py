import joblib
import numpy as np
import spacy
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

# Load the trained model
model_file_path = "../models/CNN_model/WordCNN.joblib"
word_cnn_model = joblib.load(model_file_path)
tokenizer = joblib.load("../models/CNN_model/WordCNNTokenizer.joblib")  

label_index = {'Allergic': 1, 'Non allergic': 0}
y = np.array([label_index[label] for label in etiquetas])

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
new_sequences = tokenizer.texts_to_sequences(textos_preprocesados)
max_sequence_length = word_cnn_model.input_shape[1]  
X_new = pad_sequences(new_sequences, maxlen=max_sequence_length)
predictions = word_cnn_model.predict(X_new)

# Make predictions

y_pred = (predictions > 0.5).astype(int)

report = classification_report(y, y_pred)
conf_matrix_report = confusion_matrix(y, y_pred)
TN, FP, FN, TP = conf_matrix_report.ravel()
print("Predictions:", y_pred)

model_name = "CNN"
#Report saving

report_file_path = os.path.join("../evaluation", f"{model_name}_classification_report.txt")
with open(report_file_path, 'w') as report_file:
    report_file.write(report)
    print(conf_matrix_report)
    report_file.write(f"True Positives (TP): {TP}")
    report_file.write(f"True Negatives (TN): {TN}")
    report_file.write(f"False Positives (FP): {FP}")
    report_file.write(f"False Negatives (FN): {FN}")
print(f"Classification report saved as: {report_file_path}")