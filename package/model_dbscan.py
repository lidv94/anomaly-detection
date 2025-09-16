import pandas as pd
import numpy as np
import io
import boto3
import joblib
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from itertools import product
# import gc
# import re
from package.utils import timer
import datetime
import uuid
import json
import pickle
from io import BytesIO
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from package.evaluation import calculate_performance,upload_bytes,make_df,evaluate_percentile_cuts_generic

def dbscan_predict(dbscan_model, X_new, metric):
    # Pairwise distance (shape: [n_new, n_core])
    dist_matrix = cdist(X_new, dbscan_model.components_, metric=metric)
    nearest_core = dist_matrix.argmin(axis=1)
    nearest_dist = dist_matrix[np.arange(len(X_new)), nearest_core]
    y_new = np.full(shape=len(X_new), fill_value=-1, dtype=int)
    mask = nearest_dist < dbscan_model.eps
    core_labels = dbscan_model.labels_[dbscan_model.core_sample_indices_]
    y_new[mask] = core_labels[nearest_core[mask]]
    return y_new

@timer
def dbscan_grid_search(
        X_train,
        y_train=None,
        X_test=None,
        y_test=None,
        transformer=None,
        s3_bucket=None,
        s3_prefix="dbscan_experiments",
        eps_list=[0.5, 1.0],
        min_samples_list=[5, 10],
        metric="euclidean",
        pct_list=[75,80,85,90, 95, 97, 99],
    ):
    """
    DBSCAN grid search with train/test performance and S3 upload.
    """
    X_train_scaled = transformer.transform(X_train)
    if X_test is not None:
        X_test_scaled = transformer.transform(X_test)

    results = []

    for eps, min_samples in tqdm(list(product(eps_list, min_samples_list))):
        run_id = f"{str(uuid.uuid4())}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        prefix = f"{s3_prefix}/{run_id}" if s3_prefix else run_id
        
        # Fit DBSCAN
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels_train = clusterer.fit_predict(X_train_scaled)
        y_pred_train = (labels_train==-1).astype(int)
        
        # Train performance
        perf_train = calculate_performance(y_pred_train, y_train, suffix="train")

        # Test performance
        if X_test is not None:
            labels_test = dbscan_predict(clusterer, X_test_scaled, metric=metric)
            y_pred_test = (labels_test==-1).astype(int)
            perf_test = calculate_performance(y_pred_test, y_test, suffix="test")
        else:
            perf_test = {}

        perf_metrics = {**perf_train, **perf_test}
        perf_metrics['experiments'] = {
            'eps': eps, 
            'min_samples': min_samples, 
            'metric': metric, 
            'pct':0,
            'thresh': 0
        }
        perf_metrics['n_features'] = X_train.shape[1]
        perf_metrics['id'] = run_id

        df = pd.concat([
            make_df(X_train_scaled, y_pred_train, y_train, 'train_set'),
            make_df(X_test_scaled, y_pred_test, y_test, 'test_set')], axis=0, ignore_index=True)

        # Generate binary flags + evaluate
        cut_rows = evaluate_percentile_cuts_generic(
            df=df,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
            pct_list=pct_list,
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=None,
            metric=metric,
            run_id=run_id,
            n_features=X_train.shape[1],
            algo="dbscan")

        # Combine baseline + cut rows
        all_perf_rows = [perf_metrics] + cut_rows

        # Upload files
        upload_bytes(all_perf_rows, s3_bucket, f"{prefix}/performance.json")
        upload_bytes(X_train.columns.tolist(), s3_bucket, f"{prefix}/feature.json")
        upload_bytes(clusterer, s3_bucket, f"{prefix}/model.pkl", is_pickle=True)
        upload_bytes(transformer, s3_bucket, f"{prefix}/transformer.pkl", is_pickle=True)
        # Upload distance dataframe
        # upload_bytes(df[['y_pred','y_true','data_set','dist']], s3_bucket, f"{prefix}/distance_cut_percentile.csv")
        df[['y_pred','y_true','data_set','dist']].to_csv(f"s3://{s3_bucket}/{s3_prefix}/{run_id}/distance_cut_percentile.csv", index=False)
        results.append({'s3_prefix': prefix, **perf_metrics})

# @timer
# def dbscan_grid_search(
#     x_scaled,
#     index,
#     bucket_name,
#     s3_prefix,
#     metric='euclidean',
#     eps_range=(0.1, 1.1, 0.1),
#     min_samples_range=(2, 11),col_name="db"):
#     """
#     Runs a DBSCAN grid‑search, uploads each model to S3, and returns
#       • df_labels      – anomaly flags per experiment (db_1, …)
#       • df_experiment  – two‑column lookup: experiment | parameter
#     """
#     eps_values         = np.arange(*eps_range)
#     min_samples_values = range(*min_samples_range)
#     grid               = list(product(eps_values, min_samples_values))

#     s3          = boto3.client('s3')
#     label_dict  = {}
#     exp_meta    = []

#     for i, (eps, min_samples) in enumerate(tqdm(grid, desc="DBSCAN grid")):
#         exp_name  = f"{col_name}_{i+1}"
#         s3_key     = f"{s3_prefix}{exp_name}.pkl"
#         parameter  = f"eps:{eps:.1f}|min_samples:{min_samples}"

#         # fit & save model
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(x_scaled)
#         buffer = io.BytesIO()
#         joblib.dump(dbscan, buffer)
#         buffer.seek(0)
#         s3.upload_fileobj(buffer, bucket_name, s3_key)

#         # collect outputs
#         label_dict[exp_name] = dbscan.labels_
#         exp_meta.append({'experiment': exp_name, 'parameter': parameter})

#     # labels: convert –1→1 anomaly flag
#     df_labels = pd.DataFrame(label_dict).eq(-1).astype(int)
#     df_labels = pd.concat([index,df_labels],axis=1)

#     # experiment table: exactly two columns
#     df_experiment = pd.DataFrame(exp_meta, columns=['experiment', 'parameter'])
#     return df_labels, df_experiment

# def db_cut_threshold(X, df_labels):
#     df_cut_threshold = df_labels.iloc[:, :2].copy()
#     df_exceed = df_labels.copy()
#     col_exceed = (df_labels.iloc[:, 2:].sum(axis=0) != len(df_labels)).index
#     df_exceed = df_exceed[col_exceed]

#     for col in df_exceed.columns:
#         df_cluster = pd.concat(
#             [
#                 pd.DataFrame(X).reset_index(drop=True),
#                 df_exceed[[col]].reset_index(drop=True)
#             ],
#             axis=1
#         )

#         # Separate feature sets by label
#         X0 = df_cluster[df_cluster[col] == 0].drop(columns=col).to_numpy()
#         X1 = df_cluster[df_cluster[col] == 1].drop(columns=col).to_numpy()

#         # ✅ ข้ามคอลัมน์นี้ถ้าไม่มี sample ใน X0 หรือ X1
#         if X0.shape[0] == 0 or X1.shape[0] == 0:
#             continue

#         # Compute total distance from each point in X1 to all points in X0
#         dist = euclidean_distances(X1, X0).sum(axis=1)

#         # Assign distances back to the original df_cluster
#         df_cluster['dist'] = 0
#         df_cluster.loc[df_cluster[col] == 1, 'dist'] = dist

#         # Generate binary flags based on distance percentiles
#         for q in [75,80,85,90, 95, 97, 99]:
#             percentile_val = np.percentile(dist, q)
#             df_cut_threshold[f'{col}_p{q}'] = (df_cluster['dist'] >= percentile_val).astype(int)

#     return df_cut_threshold

