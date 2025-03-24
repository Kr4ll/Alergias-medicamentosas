import os
import torch
from transformers import RobertaTokenizer, Trainer, TrainingArguments, RobertaModel
from datasets import Dataset, DatasetDict, load_metric
from sklearn.model_selection import train_test_split
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

ruta_datos = "../data/limp/ByNotes/Alergico"
ruta_datos1 = "../data/limp/ByNotes/NoAlergico"

print(tf.config.list_physical_devices('GPU'))
cargar_datos(ruta_datos, textos, etiquetas, "Allergic")
cargar_datos(ruta_datos1, textos, etiquetas, "Non allergic")

# Convert the data to a Hugging Face Dataset
data = {"text": textos, "label": [0 if label == "Alergic" else 1 for label in etiquetas]}
dataset = Dataset.from_dict(data)

# Split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict(train_test_split)

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')


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
)

# Load the accuracy metric
metric = evaluate.load("accuracy")
torch.cuda.empty_cache()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

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
model.save_pretrained("../models/roberta_model")
tokenizer.save_pretrained("../models/roberta_tokenizer")