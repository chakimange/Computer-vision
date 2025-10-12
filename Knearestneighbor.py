from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold,train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf

# ETAPE 1 : Charger CIFAR-10 officiel
(X_train_e, y_train_e), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(f"Train: {len(X_train_e)} images")
print(f"Test: {len(X_test)} images")

y_train_e = y_train_e.flatten()
y_test = y_test.flatten()

X_train, X_val, y_train, y_val = train_test_split(X_train_e, y_train_e,test_size = 0.2, random_state=42, stratify = y_train_e)

# ETAPE 2 : Préparer les données 
X_train_normalise = X_train.astype('float32') / 255.0
X_val_normalise = X_val.astype('float32') / 255.0
X_test_normalise = X_test.astype('float32') / 255.0

# Normaisation
X_train_flat = X_train_normalise.reshape(len(X_train_normalise), -1)
X_val_flat = X_val_normalise.reshape(len(X_val_normalise), -1)
X_test_flat = X_test_normalise.reshape(len(X_test_normalise), -1)


class_names = ['avion', 'voiture', 'oiseau', 'chat', 'cerf', 
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']

f1_score_list=[]

# Entraîner k-NN
for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train_flat, y_train)


    # ETAPE 3 : Evaluer sur la validation croisée
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(knn, X_train_flat, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
    cv_precision = cross_val_score(knn, X_train_flat, y_train, cv=kfold, scoring='precision_weighted', n_jobs=-1)
    cv_recall = cross_val_score(knn, X_train_flat, y_train, cv=kfold, scoring='recall_weighted', n_jobs=-1)
    cv_f1 = cross_val_score(knn, X_train_flat, y_train, cv=kfold, scoring='f1_weighted', n_jobs=-1)

    """print(f"Accuracy moyenne: {cv_accuracy.mean():.3f}")
    print(f"Précision moyenne: {cv_precision.mean():.3f}")
    print(f"Recall moyen: {cv_recall.mean():.3f}")
    print(f"F1 moyen: {cv_f1.mean():.3f}")"""
    print(k)
    f1_score_list.append(cv_f1.mean())
    
    
plt.plot(range(1,20),f1_score_list)
plt.xlabel("k")
plt.ylabel("F1 score")
plt.title("PLot of the accuracy with respect to k")
plt.show()

#starts running at 9h30
#finished running at 13h50