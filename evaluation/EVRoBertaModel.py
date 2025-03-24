import evaluate
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import Dataset
import os
import numpy as np
import spacy
import re
from sklearn.metrics import classification_report, confusion_matrix

# Loading the saved tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('../models/RoBerta_model/roberta_tokenizer')
model = RobertaForSequenceClassification.from_pretrained('../models/RoBerta_model/roberta_model')

textos = []
etiquetas = []

def cargar_datos(ruta, textos, etiquetas, value):
    for archivo in os.listdir(ruta):
        with open(os.path.join(ruta, archivo), 'r', encoding='utf-8') as file:
            textos.append(file.read())
            etiquetas.append(value)

print("Loading data...")

ruta_datos_alergico = "../data/ByNotes/EVAAlergico"
ruta_datos_non = "../data/ByNotes/EVANoAlergico"
cargar_datos(ruta_datos_alergico, textos, etiquetas, "Allergic")
cargar_datos(ruta_datos_non, textos, etiquetas, "Non allergic")

# Convert labels to indices
label_index = {'Allergic': 0, 'Non allergic': 1}
y = np.array([label_index[label] for label in etiquetas])

# Load Spacy for preprocessing
nlp = spacy.load("es_core_news_sm")


textos_preprocesados = [re.sub(r'\*\*.*?\*\*', '', texto) for texto in textos]

# Tokenize the text data

# Accuracy metric
metric = evaluate.load("accuracy")

model.eval()  
predictions = []
labels = y.tolist() 
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)


data = {"text": textos_preprocesados, "label": labels}
dataset = Dataset.from_dict(data)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    report = classification_report(labels, predictions, target_names=["Allergic", "Non allergic"], output_dict=True)
    conf_matrix_report = confusion_matrix(labels, predictions, labels=[0, 1])
    TN, FP, FN, TP = conf_matrix_report.ravel()
    print(report)
    with open("RoBerta_classification_report.txt", "w") as f:
        f.write(classification_report(labels, predictions, target_names=["Allergic", "Non allergic"]))
        f.write("\n")
        f.write(f"True Positives (TP): {TP} \n")
        f.write(f"True Negatives (TN): {TN} \n")
        f.write(f"False Positives (FP): {FP}\n")
        f.write(f"False Negatives (FN): {FN}")

    return  {
        "accuracy": report['accuracy'],
        "f1_allergic": report['Allergic']['f1-score'],
        "f1_non_allergic": report['Non allergic']['f1-score'],
        "precision_allergic": report['Allergic']['precision'],
        "precision_non_allergic": report['Non allergic']['precision'],
        "recall_allergic": report['Allergic']['recall'],
        "recall_non_allergic": report['Non allergic']['recall']
    }


trainer = Trainer(
    model=model,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)


eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")