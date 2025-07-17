import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score
import io
import boto3
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from itertools import product
import gc
import re

# plot 1 tsne with plotly
def plot_tsne(df_cluster: pd.DataFrame, col_label: str, opacity_value: float, titles: str):
    # เลือกชุดสีที่ต้องการ
    colors = px.colors.qualitative.Set1  # เลือกชุดสี Set1

    # Define a color mapping (0 -> blue, 1 -> red)
    color_map = {0: colors[1], 1: colors[0]}  # Fixed: closed the dictionary

    fig = go.Figure()

    for label in [0, 1]:
        mask = df_cluster[col_label].astype(int) == label
        fig.add_trace(go.Scatter(
            x=df_cluster.loc[mask, 'x_tsne'],
            y=df_cluster.loc[mask, 'y_tsne'],
            mode='markers',
            name=f'{col_label}: {label}',  # Legend entry
            marker=dict(
                color=color_map[label],
                opacity=opacity_value
            )
        ))

    fig.update_layout(
        title=titles,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_white",
        width=600,
        height=600,
        legend=dict(title=col_label)
    )

    fig.show()
    
# plot all tsne with matplotlib
def rgb_string_to_hex(rgb_string):
    """Convert 'rgb(r, g, b)' to hex format '#rrggbb'."""
    r, g, b = map(int, re.findall(r'\d+', rgb_string))
    return mcolors.to_hex([r / 255, g / 255, b / 255])

# plot all tsne with matplotlib
def plot_tsne_grid(tsne_result, df_cluster, n_cols=9, point_size=5, opacity=1, figsize=None):
    """Plot t-SNE results in a grid of subplots based on cluster labels in df_cluster."""
    
    # Convert Plotly Set1 colors to hex
    colors = px.colors.qualitative.Set1
    color_map = {
        0: rgb_string_to_hex(colors[1]),  # Blue
        1: rgb_string_to_hex(colors[0])   # Red
    }

    df_experiment = df_cluster.copy()
    n_experiments = df_experiment.shape[1]
    n_rows = (n_experiments + n_cols - 1) // n_cols

    # Determine figure size
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    for i in range(n_experiments):
        ax = axs[i]
        labels = df_experiment.iloc[:, i]
        
        # Map cluster labels to colors
        colors_assigned = labels.map(color_map)
        blue_mask = colors_assigned == color_map[0]
        red_mask = colors_assigned == color_map[1]

        # Plot blue before red for layering
        ax.scatter(tsne_result[blue_mask, 0], tsne_result[blue_mask, 1], 
                   c=color_map[0], s=point_size, alpha=opacity)
        ax.scatter(tsne_result[red_mask, 0], tsne_result[red_mask, 1], 
                   c=color_map[1], s=point_size, alpha=opacity)
        
        ax.set_title(df_experiment.columns[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for i in range(n_experiments, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# dbscan_grid_search
def dbscan_grid_search(x_scaled, index, bucket_name, s3_prefix,
                       metric='euclidean', eps_range=(0.1, 1.1, 0.1), min_samples_range=(2, 11)):
    eps_values = np.arange(*eps_range)
    min_samples_values = range(*min_samples_range)

    results = []
    label_dict = {}
    
    s3 = boto3.client('s3')  # Initialize boto3 S3 client
    
    for eps, min_samples in tqdm(product(eps_values, min_samples_values)):
        key = f"eps_{eps:.1f}|min_samples_{min_samples}"
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        dbscan.fit(x_scaled) # Train model

        # Save model to S3
        buffer = io.BytesIO()
        joblib.dump(dbscan, buffer)
        buffer.seek(0)
        s3_path = f"{s3_prefix}{key}.pkl"
        s3.upload_fileobj(buffer, bucket_name, s3_path)
        
        # Save labels each experiment
        labels = dbscan.labels_
        label_dict[key] = labels

    # df_labels
    df_labels = pd.DataFrame(label_dict,index=index)
    df_labels = (df_labels == -1).astype(int)
    df_labels = df_labels.reset_index()
    # df_labels = df_labels.rename(columns={'index':'sales_id'})

    return df_labels

def isolationforest_grid_search(x_scaled, index, bucket_name, s3_prefix,
                                n_estimators=(100, 200+1, 50), 
                                max_samples=['auto', 0.8, 0.5], 
                                contamination=(0.01, 0.05, 0.1),
                                max_features=(0.8, 1.0+0.1, 0.2),
                                bootstrap=[False, True]):
    eps_values = np.arange(*eps_range)
    min_samples_values = range(*min_samples_range)

    results = []
    label_dict = {}
    
    s3 = boto3.client('s3')  # Initialize boto3 S3 client
    
    for eps, min_samples in tqdm(product(eps_values, min_samples_values)):
        key = f"eps_{eps:.1f}|min_samples_{min_samples}"
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        dbscan.fit(x_scaled) # Train model

        # Save model to S3
        buffer = io.BytesIO()
        joblib.dump(dbscan, buffer)
        buffer.seek(0)
        s3_path = f"{s3_prefix}{key}.pkl"
        s3.upload_fileobj(buffer, bucket_name, s3_path)
        
        # Save labels each experiment
        labels = dbscan.labels_
        label_dict[key] = labels

    # df_labels
    df_labels = pd.DataFrame(label_dict,index=index)
    df_labels = (df_labels == -1).astype(int)
    df_labels = df_labels.reset_index()
    # df_labels = df_labels.rename(columns={'index':'sales_id'})

    return df_labels

# evaluate_fraud_predictions
def evaluate_fraud_predictions(x_scaled, df_lables ,true_fraud_list):
    print('Total input:', df_lables.shape[0])
    df_lables['true_fraud'] = 0
    df_lables.loc[df_lables.index.isin(true_fraud_list), 'true_fraud'] = 1
    y_true = df_lables['true_fraud']
    df_lables = df_lables.drop(columns='true_fraud')
    
    # Create an empty list to collect results
    results = []

    for col in tqdm(df_lables.columns):
        
        y_pred = df_lables[col]

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # print('-------' + col + '-------')
        # print("Confusion Matrix:")
        # print(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))
        
        noise_count = np.sum(y_pred == 1)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
        # score = silhouette_score(x_scaled, y_pred) # ks or js -> พวก kl

        # Store results in the list
        results.append({
            'experiments': col,
            'total_alert': noise_count,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f0.5': f05,
            'f1': f1
            # ,'Silhouette_Score':score
        })

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df