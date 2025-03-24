import joblib
import spacy
import re
import sys
import os


def cargar_modelos():
    modelos = []
    nombres_modelos = []
    modelo_path = "../models/Classic_models"
    for archivo in os.listdir(modelo_path):
        if "tokenizer" not in archivo:
            modelos.append(joblib.load(os.path.join(modelo_path, archivo)))
            nombres_modelos.append(archivo[:-7])
    tokenizer = joblib.load("../models/Classic_models/tokenizer.joblib")
    return modelos, nombres_modelos, tokenizer


def preprocesar(texto, nlp):
    texto = re.sub(r'\*\*.*?\*\*', '', texto)  # Eliminar texto entre ** **
    doc = nlp(texto)
    tokens_lemmatizados = [token.lemma_ for token in doc if token.text.isalnum() or token.text.isspace()]
    return " ".join(tokens_lemmatizados)


def evaluar_texto(archivo_txt):
    # Cargar modelo y tokenizer
    modelos, nombres_modelos, tokenizer = cargar_modelos()

    # Cargar el modelo de lenguaje
    nlp = spacy.load("es_core_news_sm")

     # Leer el archivo
    with open(archivo_txt, 'r', encoding='utf-8') as file:
        texto = file.read()

    # Preprocesar el texto
    texto_preprocesado = preprocesar(texto, nlp)

    # Tokenizar el texto
    X_new = tokenizer.transform([texto_preprocesado])

    # Evaluar con cada modelo
    for i, modelo in enumerate(modelos):
        y_pred = modelo.predict(X_new)[0]
        resultado = "Alérgico" if y_pred == "Allergic" else "No alérgico"
        print(f"Modelo {nombres_modelos[i]}: {resultado}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python evaluar_alergia.py <archivo_txt>")
    else:
        evaluar_texto(sys.argv[1])

