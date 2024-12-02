import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_split_distribution(train_df, val_df, test_df, save_path):
    splits = ['Train', 'Validation', 'Test']
    real = [
        (train_df.label == 0).sum(),
        (val_df.label == 0).sum(),
        (test_df.label == 0).sum()
    ]
    fake = [
        (train_df.label == 1).sum(),
        (val_df.label == 1).sum(),
        (test_df.label == 1).sum()
    ]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(splits))
    width = 0.35
    
    plt.bar(x - width/2, real, width, label='Real')
    plt.bar(x + width/2, fake, width, label='Fake')
    
    plt.xlabel('Dataset Split')
    plt.ylabel('Number of Images')
    plt.title('Dataset Distribution')
    plt.xticks(x, splits)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(models_predictions, labels, save_path):
    plt.figure(figsize=(10, 8))
    
    for model_name, preds in models_predictions.items():
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 