import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re


# Function to load the text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Preprocess the text (you can modify this according to your needs)
def preprocess_text(text):
    # Remove unwanted patterns (if any) - Adjust as necessary for your task
    text = re.sub(r'\*\*.*?\*\*', '', text)  # Example of removing certain patterns
    return text


# Function to classify a given text file
def classify_text(clinicNote, model, tokenizer):
    # Load and preprocess the text
    text = clinicNote
    preprocessed_text = preprocess_text(text)

    # Tokenize the input text
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True)

    # Make sure the model is in evaluation mode
    model.eval()

    # Perform the classification (disable gradient calculation for faster inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label (argmax of logits)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Map class IDs to labels (assuming 0 -> Allergic, 1 -> Non-allergic)
    label_map = {0: "Allergic", 1: "Non-allergic"}
    predicted_label = label_map[predicted_class_id]

    return predicted_label


if __name__ == "__main__":
    # Define the paths for the saved model and tokenizer
    model_path = "../models/RoBerta_model/ByNotes/roberta_model"
    tokenizer_path = "../models/RoBerta_model/ByNotes/roberta_tokenizer"

    # Load the model and tokenizer
    print("Loading the model and tokenizer...")
    model = RobertaForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    # Path to the text file you want to classify
    file_path="../programs/Alergic/"
    clinicNotee = input("Introduzca la nota clínica: ")

    print("Classifying the text...")
    predicted_label = classify_text(clinicNotee, model, tokenizer)

    # Output the result
    print(f"The predicted class for the text is: {predicted_label}")
