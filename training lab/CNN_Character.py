import os
import re
import numpy as np
import spacy
from keras import Model
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras._tf_keras.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Step 1: Load data
textos = []
etiquetas = []
def cargar_datos(ruta, textos, etiquetas, value):
    for archivo in os.listdir(ruta):
        with open(os.path.join(ruta, archivo), 'r', encoding='utf-8') as file:
            textos.append(file.read())
            etiquetas.append(value)

ruta_datos_alergico = "../data/ByNotes/Alergico"
ruta_datos_non = "../data/ByNotes/NoAlergico"

cargar_datos(ruta_datos_alergico, textos, etiquetas, "Allergic")
cargar_datos(ruta_datos_non, textos, etiquetas, "Non allergic")

# Step 2: Preprocessing text
nlp = spacy.load("es_core_news_sm")

def preprocesar(texto):
    texto = re.sub(r'\*\*.*?\*\*', '', texto)
    doc = nlp(texto)
    tokens_lemmatizados = [token.lemma_ for token in doc if token.text.isalnum() or token.text.isspace()]
    return " ".join(tokens_lemmatizados)

textos_preprocesados = [preprocesar(texto) for texto in textos]

# Step 3: Prepare character-level data
max_len = 100  # Maximum length of the character sequence
char_tokenizer = Tokenizer(char_level=True)
char_tokenizer.fit_on_texts(textos_preprocesados)

def text_to_char_sequences(texts):
    sequences = char_tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, padding='post')

X_char = text_to_char_sequences(textos_preprocesados)

# Convert labels to numpy arrays
label_index = {'Allergic': 0, 'Non allergic': 1}
y = np.array([label_index[label] for label in etiquetas])

# Step 4: Define character-level CNN model
vocab_size = len(char_tokenizer.word_index) + 1
embedding_dim = 50

input_layer = Input(shape=(None,))
embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)
conv_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
pooling_layer = GlobalMaxPooling1D()(conv_layer)
dense_layer = Dense(10, activation='relu')(pooling_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)


char_cnn_model = Model(input_layer, output_layer)
char_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['F1-score'])
char_cnn_model.summary()


# Split data into training and test sets
X_train_char, X_test_char, y_train, y_test = train_test_split(X_char, y, test_size=0.2, random_state=42)

# Balance the clases
ros = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_char, y_train)

# Train the character-level CNN model
char_cnn_model.fit(X_resampled, y_resampled, epochs=10, batch_size=16)

# Get character-level features
char_features_test = char_cnn_model.predict(X_test_char)
y_pred = (char_features_test > 0.5).astype(int)
report = classification_report(y_test, y_pred)
print('Accuracy:', report)



model_name = char_cnn_model.__class__.__name__

# Save the trained model
model_file_path = os.path.join("../models", f"{model_name}.joblib")
joblib.dump(char_cnn_model, model_file_path)
print(f"Trained model saved as: {model_file_path}")

# Evaluation of the model
char_features_test = char_cnn_model.predict(X_test_char)

# Save the classification report
report_file_path = os.path.join("../models", f"{model_name}_classification_report.txt")
with open(report_file_path, 'w') as report_file:
    y_pred = (char_features_test > 0.5).astype(int)
    report = classification_report(y_test, y_pred)
    report_file.write(report)
print(f"Classification report saved as: {report_file_path}")