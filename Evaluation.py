import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(model, dataloader, device, num_classes=4):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    metrics = {
        'accuracy': np.mean(all_preds == all_labels),
        'confusion_matrix': cm,
        'precision_per_class': [],
        'recall_per_class': [],
        'f1_per_class': [],
        'support_per_class': []
    }
    
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
          
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision_per_class'].append(precision)
        metrics['recall_per_class'].append(recall)
        metrics['f1_per_class'].append(f1)
        metrics['support_per_class'].append(cm[i, :].sum())
    
    metrics['precision_macro'] = np.mean(metrics['precision_per_class'])
    metrics['recall_macro'] = np.mean(metrics['recall_per_class'])
    metrics['f1_macro'] = np.mean(metrics['f1_per_class'])
    
    total_samples = sum(metrics['support_per_class'])
    weights = [support / total_samples for support in metrics['support_per_class']]
    
    metrics['precision_weighted'] = np.average(metrics['precision_per_class'], weights=weights)
    metrics['recall_weighted'] = np.average(metrics['recall_per_class'], weights=weights)
    metrics['f1_weighted'] = np.average(metrics['f1_per_class'], weights=weights)
    
    return metrics

def print_detailed_report(metrics, class_names=None):
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['precision_per_class']))]
    
    print("=" * 60)
    
    print(f"\n Accuracy globale: {metrics['accuracy']*100:.2f}%")
    print(f"Total échantillons: {sum(metrics['support_per_class'])}")
    
    print(f"\n{'Classe':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    
    for i, name in enumerate(class_names):
        print(f"{name:<15} {metrics['precision_per_class'][i]:<12.3f} "
              f"{metrics['recall_per_class'][i]:<12.3f} "
              f"{metrics['f1_per_class'][i]:<12.3f} "
              f"{metrics['support_per_class'][i]:<10}")
    
    print(f"Macro avg  : Precision={metrics['precision_macro']:.3f}, "
          f"Recall={metrics['recall_macro']:.3f}, F1={metrics['f1_macro']:.3f}")
    print(f"Weighted avg: Precision={metrics['precision_weighted']:.3f}, "
          f"Recall={metrics['recall_weighted']:.3f}, F1={metrics['f1_weighted']:.3f}")

def plot_confusion_matrix(cm, class_names=None, title="Matrice de Confusion"):
    if class_names is None:
        class_names = [f'Classe {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=16)
    plt.ylabel('Vraie étiquette', fontsize=12)
    plt.xlabel('Étiquette prédite', fontsize=12)
    plt.tight_layout()
    plt.show()

def get_classification_report_sklearn(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if class_names is None:
        class_names = [str(i) for i in range(len(set(all_labels)))]
    
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, digits=3)
    return report


def calculate_accuracy(model, dataloader, device):
    """Calcule uniquement l'accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def evaluate_model_comprehensive(model, train_loader, test_loader, device, class_names=None):
    """
    Évaluation complète sur train et test sets
    """
    print("ÉVALUATION COMPLÈTE DU MODÈLE")
    print("=" * 60)
    
    
    print("\nSUR LE TRAIN SET:")
    train_metrics = calculate_metrics(model, train_loader, device)
    print(f"Train Accuracy: {train_metrics['accuracy']*100:.2f}%")
    print(f"Train F1 Macro: {train_metrics['f1_macro']:.3f}")
    
   
    print("\nSUR LE TEST SET:")
    test_metrics = calculate_metrics(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1 Macro: {test_metrics['f1_macro']:.3f}")
    
    
    print("\nRAPPORT DÉTAILLÉ TEST SET:")
    print_detailed_report(test_metrics, class_names)
    
    
    print("\nMATRICE DE CONFUSION (Test Set):")
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names)
    
    
    print("\nRAPPORT SKLEARN (Test Set):")
    report = get_classification_report_sklearn(model, test_loader, device, class_names)
    print(report)
    
    return train_metrics, test_metrics

