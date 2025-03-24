import os
import torch
import spacy
import re

from sklearn.metrics import classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_metric
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import evaluate

import tensorflow as tf
# Step 1: Data load
textos = []
etiquetas = []

def cargar_datos(ruta, textos, etiquetas, value):
    for archivo in os.listdir(ruta):
        with open(os.path.join(ruta, archivo), 'r', encoding='utf-8') as file:
            textos.append(file.read())
            etiquetas.append(value)

ruta_datos = "../data/ByPatients/Alergico"
ruta_datos1 = "../data/ByPatients/NoAlergico"

print(tf.config.list_physical_devices('GPU'))
cargar_datos(ruta_datos, textos, etiquetas, "Allergic")
cargar_datos(ruta_datos1, textos, etiquetas, "Non allergic")

nlp = spacy.load("es_core_news_sm")


textos_preprocesados = [re.sub(r'\*\*.*?\*\*', '', texto) for texto in textos]

# Convert the data to a Hugging Face Dataset
data = {"text": textos_preprocesados, "label": [0 if label == "Allergic" else 1 for label in etiquetas]}
dataset = Dataset.from_dict(data)

# Split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict(train_test_split)

# Balance train class
texts = dataset['train']['text']
labels = dataset['train']['label']

texts = [[text] for text in texts]
ros = RandomOverSampler(random_state=42)
resampled_texts, resampled_labels = ros.fit_resample(texts, labels)

resampled_texts = [text[0] for text in resampled_texts]
resampled_dataset = Dataset.from_dict({'text': resampled_texts, 'label': resampled_labels})
dataset['train'] = resampled_dataset


# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = RobertaForSequenceClassification.from_pretrained('PlanTL-GOB-ES/roberta-base-bne', num_labels=2)


# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=2,
    save_steps=0,
)

# Load the accuracy metric
metric = evaluate.load("accuracy")
torch.cuda.empty_cache()

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

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)


# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the trained model
model.save_pretrained("../models/RoBerta_model/roberta_model")
tokenizer.save_pretrained("../models/RoBerta_model/roberta_tokenizer")