import os
import re
import joblib
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Data load into variables textos and etiquetas
textos = []
etiquetas = []
def cargar_datos(ruta, textos, etiquetas, value):
    for archivo in os.listdir(ruta):
        with open(os.path.join(ruta, archivo), 'r', encoding='utf-8') as file:
            textos.append(file.read())
            etiquetas.append(value)

print("Loading data...")

ruta_datos_alergico = "../data/ByNotes/Alergico"
ruta_datos_non = "../data/ByNotes/NoAlergico"

cargar_datos(ruta_datos_alergico, textos, etiquetas, "Allergic")
cargar_datos(ruta_datos_non, textos, etiquetas, "Non allergic")

# Text preprocessing
print("Text preprocessing...")

nlp = spacy.load("es_core_news_sm")

def preprocesar(texto):
    texto = re.sub(r'\*\*.*?\*\*', '', texto)
    doc = nlp(texto)
    tokens_lemmatizados = [token.lemma_ for token in doc if token.text.isalnum() or token.text.isspace()]
    return tokens_lemmatizados

textos_preprocesados = [preprocesar(texto) for texto in textos]

# Train Word2Vec model with the dataset
print("Training Word2Vec model...")

word2vec_model = Word2Vec(sentences=textos_preprocesados, vector_size=200, window=5, min_count=1, sg=1)
word2vec_model.save("word2vec_model.model")

# Create feature vectors for each document by averaging word vectors
def document_vector(tokens):
    # Remove out-of-vocabulary words
    tokens = [word for word in tokens if word in word2vec_model.wv.key_to_index]
    if len(tokens) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[tokens], axis=0)

X = np.array([document_vector(tokens) for tokens in textos_preprocesados])

# Convert labels to numpy arrays
label_index = {'Allergic': 0, 'Non allergic': 1}
y = np.array([label_index[label] for label in etiquetas])

# Separation of the data between training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the class distribution
ros = RandomUnderSampler(random_state=42)
X_resampled_train, y_resampled_train = ros.fit_resample(X_train, y_train)

# Classification model training
models = [
    RandomForestClassifier(random_state=42),
    SVC(kernel='linear', random_state=42),
    SVC(kernel='rbf', random_state=42),
    SVC(kernel='poly', random_state=42),
    GradientBoostingClassifier(random_state=42)
]

# With StratifiedKFold, the folds are made by preserving the percentage of samples for each class.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scoring = ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']

# Iterative loop to print metrics from each model
counter = 0
for model in tqdm(models):
    model_name = model.__class__.__name__
    result = cross_validate(model, X_resampled_train, y_resampled_train, cv=kf, scoring=scoring)
    print("%s: Mean Accuracy = %.2f%%; Mean F1-macro = %.2f%%; Mean recall-macro = %.2f%%; Mean precision-macro = %.2f%%"
          % (model_name,
             result['test_accuracy'].mean()*100,
             result['test_f1_macro'].mean()*100,
             result['test_recall_macro'].mean()*100,
             result['test_precision_macro'].mean()*100))
    model.fit(X_resampled_train, y_resampled_train)

    # Save the trained model
    model_file_path = os.path.join("../models", f"{model_name}{counter}.joblib")
    joblib.dump(model, model_file_path)
    print(f"Trained model saved as: {model_file_path}")

    # Evaluation of the model
    predicciones = model.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    print("Model Precision:", precision)

    conf_matrix_report = confusion_matrix(y_test, predicciones)
    TN, FP, FN, TP = conf_matrix_report.ravel()

    # Save the classification report
    report_file_path = os.path.join("../models", f"{model_name}{counter}_classification_report.txt")
    with open(report_file_path, 'w') as report_file:
        report = classification_report(y_test, predicciones)
        conf_matrix_report = confusion_matrix(y_test, predicciones)

        report_file.write(report)
        print(conf_matrix_report)
        report_file.write(f"True Positives (TP): {TP}")
        report_file.write(f"True Negatives (TN): {TN}")
        report_file.write(f"False Positives (FP): {FP}")
        report_file.write(f"False Negatives (FN): {FN}")
    print(f"Classification report saved as: {report_file_path}")

    # Extract feature importances or coefficients
    if hasattr(model, 'coef_'):  # SVC with linear kernel
        pass
    elif hasattr(model, 'feature_importances_'):  # RandomForest or GradientBoosting
        importances = model.feature_importances_
        top_features = np.argsort(importances)[-10:]
        with open(report_file_path, 'a') as report_file:
            report_file.write("\n")
            report_file.write(f"Palabras con mas influencia en el modelo {model_name}: \n")
            for i in top_features:
                report_file.write(f"Feature {i}: {importances[i]} \n")

    print()
    counter += 1
