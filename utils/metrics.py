from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
from utils.data_viz import binarize_predictions

import matplotlib.pyplot as plt


def plot_violin(predictions, ground_truth, model_names, filename):
    
    accuracies = {model: [] for model in model_names}
    precisions = {model: [] for model in model_names}
    recalls = {model: [] for model in model_names}
    f1_scores = {model: [] for model in model_names}
    
    
    binarized_predictions = [binarize_predictions(pred) for pred in predictions]

    # Compute metrics for each image for each model
    for model_idx, model_name in enumerate(model_names):
        for i in range(ground_truth.shape[0]):
            true_flat = ground_truth[i].flatten()
            pred_flat = binarized_predictions[model_idx][i].flatten()
            
            accuracies[model_name].append(accuracy_score(true_flat, pred_flat))
            precisions[model_name].append(precision_score(true_flat, pred_flat, zero_division=0))
            recalls[model_name].append(recall_score(true_flat, pred_flat, zero_division=0))
            f1_scores[model_name].append(f1_score(true_flat, pred_flat, zero_division=0))

    # Create a DataFrame for plotting
    data = {
        'Model': [],
        'Metric': [],
        'Value': []
    }

    for model_name in model_names:
        data['Model'].extend([model_name] * ground_truth.shape[0])
        data['Metric'].extend(['Accuracy'] * ground_truth.shape[0])
        data['Value'].extend(accuracies[model_name])
        
        data['Model'].extend([model_name] * ground_truth.shape[0])
        data['Metric'].extend(['Precision'] * ground_truth.shape[0])
        data['Value'].extend(precisions[model_name])
        
        data['Model'].extend([model_name] * ground_truth.shape[0])
        data['Metric'].extend(['Recall'] * ground_truth.shape[0])
        data['Value'].extend(recalls[model_name])
        
        data['Model'].extend([model_name] * ground_truth.shape[0])
        data['Metric'].extend(['F1 Score'] * ground_truth.shape[0])
        data['Value'].extend(f1_scores[model_name])

    df = pd.DataFrame(data)

    # Plot using Seaborn
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(12, 24))  # Adjusted figsize for better visualization

    for i, metric in enumerate(metrics):
        plt.subplot(4, 1, i + 1)
        sns.violinplot(x='Model', y='Value', data=df[df['Metric'] == metric])
        plt.title(metric)

    plt.tight_layout()
    plt.savefig(filename + ".svg")
    plt.show()