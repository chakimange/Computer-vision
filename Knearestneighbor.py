import pandas as pd
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf

# ETAPE 1 : Charger CIFAR-10 officiel
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train ,test_size=0.2 ,random_state=42,stratify=y_train)

print(f"Train: {len(X_train)} images")
print(f"Test: {len(X_test)} images")

y_train = y_train.flatten()
y_test = y_test.flatten()

# ETAPE 2 : Préparer les données 
X_train_normalise = X_train_split.astype('float32') / 255.0
X_val_normalise = X_val.astype('float32') / 255.0
X_test_normalise = X_test.astype('float32') / 255.0

# Normaisation
X_train_flat = X_train_normalise.reshape(len(X_train_normalise), -1)
X_val_flat = X_val_normalise.reshape(len(X_val_normalise), -1)
X_test_flat = X_test_normalise.reshape(len(X_test_normalise), -1)

print(f"Forme données: {X_train_flat.shape}")

# Noms des classes CIFAR-10
class_names = ['avion', 'voiture', 'oiseau', 'chat', 'cerf', 
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Entraîner k-NN
knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn.fit(X_train_flat, y_train_split)
print("Modèle entraîné!")


# ETAPE 4 : Evaluer sur la validation
val_predictions = knn.predict(X_val_flat)
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions, average='weighted')
val_recall = recall_score(y_val, val_predictions, average='weighted')
val_f1 = f1_score(y_val, val_predictions, average='weighted')

print(f"Accuracy (validation): {val_accuracy:.3f}")
print(f"Précision (validation): {val_precision:.3f}")
print(f"Recall (validation): {val_recall:.3f}")
print(f"F1 (validation): {val_f1:.3f}")

# ETAPE 3 : Evaluer
predictions = knn.predict(X_test_flat)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')  
recall = recall_score(y_test, predictions, average='weighted')        
f1 = f1_score(y_test, predictions, average='weighted')               

print(f"Accuracy: {accuracy:.3f}")
print(f"Précision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")