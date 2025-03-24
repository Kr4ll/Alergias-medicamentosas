import os
import numpy as np
import spacy
from keras import Model
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score,confusion_matrix
import joblib
import re

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import tensorflow as tf
from keras._tf_keras.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Step 1: Load data
textos = []
etiquetas = []
def cargar_datos(ruta, textos, etiquetas, value):
    for archivo in os.listdir(ruta):
        with open(os.path.join(ruta, archivo), 'r', encoding='utf-8') as file:
            textos.append(file.read())
            etiquetas.append(value)

print("Loading data")
ruta_datos_allergic = "../data/ByPatients/Alergico" 
ruta_datos_non = "../data/ByPatients/NoAlergico"

cargar_datos(ruta_datos_allergic, textos, etiquetas, "Allergic")
cargar_datos(ruta_datos_non, textos, etiquetas, "Non allergic")

print("Text preprocess")
# Step 2: Text preprocessing
nlp = spacy.load("es_core_news_sm")

def preprocesar(texto):
    texto = re.sub(r'\*\*.*?\*\*', '', texto)
    doc = nlp(texto)
    tokens_lemmatizados = [token.lemma_ for token in doc if token.text.isalnum() or token.text.isspace()]
    return " ".join(tokens_lemmatizados)

textos_preprocesados = [preprocesar(texto) for texto in textos]

print("Word-level tokenization and padding")

# Step 3: Word-level tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(textos_preprocesados)
sequences = tokenizer.texts_to_sequences(textos_preprocesados)
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to numpy arrays
label_index = {'Allergic': 1, 'Non allergic': 0}
y = np.array([label_index[label] for label in etiquetas])

print("Defining model")
# Step 4: Define Word-level CNN model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50

input_layer = Input(shape=(None,))
embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_sequence_length)(input_layer)
conv_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
pooling_layer = GlobalMaxPooling1D()(conv_layer)
dense_layer = Dense(10, activation='relu')(pooling_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)


word_cnn_model = Model(input_layer, output_layer)
word_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
word_cnn_model.summary()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the clases
ros = RandomOverSampler(random_state=42)
X_resampled_train, y_resampled_train = ros.fit_resample(X_train, y_train)

print("training model")

# Train the word-level CNN model
word_cnn_model.fit(X_resampled_train, y_resampled_train, epochs=10, batch_size=16)

model_name = "WordCNN"

# Save the trained model
model_file_path = os.path.join("../models/CNN_model", f"{model_name}.joblib")
joblib.dump(word_cnn_model, model_file_path)
tokenizer_file_path = os.path.join("../models/CNN_model", f"{model_name}Tokenizer.joblib")
joblib.dump(tokenizer, tokenizer_file_path)
print(f"Trained model saved as: {model_file_path} and tokenizer saved as: {tokenizer_file_path}")

# Evaluation of the model
char_features_test = word_cnn_model.predict(X_test)
y_pred = (char_features_test > 0.5).astype(int)
report = classification_report(y_test, y_pred)
conf_matrix_report = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix_report.ravel()
print('Accuracy:', report)

# Save the classification report
report_file_path = os.path.join("../results", f"{model_name}_classification_report.txt")
with open(report_file_path, 'w') as report_file:
    report_file.write(report)
    print(conf_matrix_report)
    report_file.write(f"True Positives (TP): {TP}")
    report_file.write(f"True Negatives (TN): {TN}")
    report_file.write(f"False Positives (FP): {FP}")
    report_file.write(f"False Negatives (FN): {FN}")
print(f"Classification report saved as: {report_file_path}")