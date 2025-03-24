import os
import joblib
import numpy as np
import spacy
import re
from keras.src.utils import pad_sequences

# Load model and tokenizer
print("Loading CNN model and tokenizer...")
model_path = "../models/CNN_model/WordCNN.joblib"
tokenizer_path = "../models/CNN_model/WordCNNTokenizer.joblib"
word_cnn_model = joblib.load(model_path)
tokenizer = joblib.load(tokenizer_path)

# Load NLP model for preprocessing
nlp = spacy.load("es_core_news_sm")


def preprocess_text(text):
    text = re.sub(r'\*\*.*?\*\*', '', text)  # Remove unwanted patterns
    doc = nlp(text)
    tokens_lemmatized = [token.lemma_ for token in doc if token.text.isalnum() or token.text.isspace()]
    return " ".join(tokens_lemmatized)


def classify_text(text, model, tokenizer):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    max_sequence_length = model.input_shape[1]
    X_input = pad_sequences(sequence, maxlen=max_sequence_length)

    prediction = model.predict(X_input)
    predicted_class = "Allergic" if prediction > 0.5 else "Non-allergic"

    return predicted_class


if __name__ == "__main__":
    # Take input and classify
    clinic_note = input("Enter the clinical note: ")
    prediction = classify_text(clinic_note, word_cnn_model, tokenizer)
    print(f"Predicted class: {prediction}")
