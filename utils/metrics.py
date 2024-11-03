from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
from utils.data_viz import binarize_predictions

import matplotlib.pyplot as plt

def dice_coefficient(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    return dice

def volume_similarity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    vs = 1 - abs(fn - fp) / (2 * tp + fp + fn + 1e-7)
    return vs


def plot_violin(predictions, ground_truth, model_names, filename):
    accuracies = {model: [] for model in model_names}
    precisions = {model: [] for model in model_names}
    recalls = {model: [] for model in model_names}
    f1_scores = {model: [] for model in model_names}
    dice_scores = {model: [] for model in model_names}
    volume_similarities = {model: [] for model in model_names}
    
    binarized_predictions = [binarize_predictions(pred) for pred in predictions]

    for model_idx, model_name in enumerate(model_names):
        for i in range(ground_truth.shape[0]):
            true_flat = ground_truth[i].flatten()
            pred_flat = binarized_predictions[model_idx][i].flatten()
            
            accuracies[model_name].append(accuracy_score(true_flat, pred_flat))
            precisions[model_name].append(precision_score(true_flat, pred_flat, zero_division=0))
            recalls[model_name].append(recall_score(true_flat, pred_flat, zero_division=0))
            f1_scores[model_name].append(f1_score(true_flat, pred_flat, zero_division=0))
            dice_scores[model_name].append(dice_coefficient(true_flat, pred_flat))
            volume_similarities[model_name].append(volume_similarity(true_flat, pred_flat))

    data = {
        'Model': [],
        'Metric': [],
        'Value': []
    }

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Dice', 'Volume Similarity']
    metric_dicts = [accuracies, precisions, recalls, f1_scores, dice_scores, volume_similarities]

    for metric_name, metric_values in zip(metrics, metric_dicts):
        for model_name in model_names:
            data['Model'].extend([model_name] * len(metric_values[model_name]))
            data['Metric'].extend([metric_name] * len(metric_values[model_name]))
            data['Value'].extend(metric_values[model_name])

    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 18))

    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        sns.violinplot(x='Model', y='Value', data=df[df['Metric'] == metric])
        plt.title(metric)
        # Dynamically set the y-axis limit
        y_min = max(df[df['Metric'] == metric]['Value'].min() - 0.1, 0)
        plt.ylim(y_min, 1.1)

    plt.tight_layout()
    plt.savefig(filename + ".svg")
    plt.show()
