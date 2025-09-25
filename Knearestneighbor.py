import pandas as pd
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#ÉTAPE 1 : Charger les données
csv_path = r"C:\Users\Dell\Documents\Computer_Vision\ImagesDatasets\cifar-10 (2)\trainLabels.csv"
df = pd.read_csv(csv_path)
print(f"Labels chargés: {len(df)} entrées")
dossier_images = r"C:\Users\Dell\Documents\Computer_Vision\ImagesDatasets\cifar-10 (2)\train\train"
images = []
labels = []

for index, row in df.iterrows():
    num_image = row['id']
    label = row['label']
    
    # Charger l'image 
    chemin_image = os.path.join(dossier_images, f"{num_image}.png")
    
    if os.path.exists(chemin_image):
        img = cv2.imread(chemin_image)
        images.append(img)
        labels.append(label)
    else:
        print(f"Image manquante: {chemin_image}")

X = np.array(images)
y = np.array(labels)

print(f"Images chargées: {len(X)}")

#ÉTAPE 2 : Diviser en train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40, shuffle=True
)

print(f"Train: {len(X_train)} images")
print(f"Test: {len(X_test)} images")

# ÉTAPE 3 : Préparer les données 
X_train_normalise = X_train.astype('float32') / 255.0
X_test_normalise = X_test.astype('float32') / 255.0
# Aplatir les images, les transformer en liste simple
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Convertir les labels en nombres (chat->1,chien->2, etc...)
labels_uniques = list(set(y))
label_to_num = {label: i for i, label in enumerate(labels_uniques)}

y_train_num = [label_to_num[label] for label in y_train]
y_test_num = [label_to_num[label] for label in y_test]

print(f"Classes: {labels_uniques}")

#  Entraîner k-NN
knn = KNeighborsClassifier(n_neighbors=5,weights="distance")
knn.fit(X_train_flat, y_train_num)
print("Modèle entraîné!")

#ÉTAPE 4 : Évaluer
predictions = knn.predict(X_test_flat)
precision = accuracy_score(y_test_num, predictions)
print(f"Précision: {precision:.3f}")

# ÉTAPE 5 : Fonction de prédiction
def predire_image(num_image, modele, label_dict):
    chemin_image = os.path.join(dossier_images, f"{num_image}.png")
    img = cv2.imread(chemin_image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_flat = img.reshape(1, -1)
    prediction_num = modele.predict(img_flat)[0]
    
    # Trouver le label
    label_prediction = list(label_dict.keys())[list(label_dict.values()).index(prediction_num)]
    
    # Affichage simple
    plt.imshow(img_rgb)
    plt.title(f"Prédiction: {label_prediction}")
    plt.axis('off')
    plt.show()
    
    print(f"Prédiction: {label_prediction}")
    return label_prediction

predire_image(1000, knn, label_to_num)


# k=3 p=0.324, k=5 p=0.340 , k=15 p=0.320, k=7 p=0.330

# k=3 p=0.324, k=5 p=0.339 , k=15 p=0.322, k=7 p=0.331

#weight=dist k=3 p=0.344, k=5 p=0.352 , k=15 p=0.322, k=7 p=0.331


import matplotlib.pyplot as plt
from collections import Counter

# Vérifier la distribution des classes
distribution = Counter(y)
print("Distribution des classes:")
for classe, count in distribution.items():
    print(f"{classe}: {count} images ({count/len(y)*100:.1f}%)")

# Afficher quelques images avec leurs vrais labels
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i+1)
    idx = np.random.randint(len(X_train))
    plt.imshow(cv2.cvtColor(X_train[idx], cv2.COLOR_BGR2RGB))
    plt.title(f"Vrai: {y_train[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()