import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
# from sklearn.metrics import silhouette_score
from package.utils import timer
import boto3
import joblib
import io
from io import BytesIO
import json
import pickle
from tqdm import tqdm
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

def calculate_performance(y_pred, y_true=None, suffix=""):
    
    y_pred_binary = np.array(y_pred)
    y_true_binary = np.array(y_true)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    f05 = fbeta_score(y_true_binary, y_pred_binary, beta=0.5, zero_division=0)
    
    prefix = f"{suffix}_" if suffix else ""
    return {
        f'{prefix}total_alert': int(tp + fp),
        f'{prefix}tp': int(tp),
        f'{prefix}fp': int(fp),
        f'{prefix}tn': int(tn),
        f'{prefix}fn': int(fn),
        f'{prefix}precision': float(precision),
        f'{prefix}recall': float(recall),
        f'{prefix}f0.5': float(f05),
        f'{prefix}f1': float(f1)
    }


def upload_bytes(obj,s3_bucket, key, is_pickle=False):
    bio = BytesIO()
    if is_pickle:
        pickle.dump(obj, bio)
    else:
        bio.write(json.dumps(obj, indent=4).encode("utf-8"))
    bio.seek(0)
    s3_client = boto3.client("s3")
    s3_client.upload_fileobj(bio, s3_bucket, key)
    
@timer
def read_performance(s3_bucket: str, prefix: str):
    """
    Read all performance.json files from S3 and return a normalized DataFrame.
    """
    s3 = boto3.client("s3")
    
    # List all objects in the experiments folder
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
    if "Contents" not in response:
        return pd.DataFrame()  # no files
    
    records = []
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith("performance.json"):
            # Read file
            file_obj = s3.get_object(Bucket=s3_bucket, Key=key)
            content = file_obj["Body"].read()
            data = json.loads(content)

            # # add uuid info
            # uuid = key.split("/")[1]
            # data["uuid"] = uuid

            records.append(data)
    
    # Normalize JSON into DataFrame
    df = pd.json_normalize(records)
    return df

@timer
def read_performance_to_df(s3_bucket, s3_prefix):
    """
    Read all performance.json files from S3 and return a single DataFrame.
    """
    s3_client = boto3.client("s3")
    rows = []
    continuation_token = None

    while True:
        list_kwargs = {"Bucket": s3_bucket, "Prefix": s3_prefix}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        if "Contents" not in response:
            break

        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith("performance.json"):
                file_obj = s3_client.get_object(Bucket=s3_bucket, Key=key)
                content = file_obj["Body"].read().decode("utf-8")
                perf_data = json.loads(content)
                
                # performance.json is a list of dicts
                if isinstance(perf_data, list):
                    rows.extend(perf_data)
                else:
                    rows.append(perf_data)

        # Handle pagination
        if response.get("IsTruncated"):
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return pd.DataFrame(rows)

def prep_eval_sent(df_raw,top_n,top_col):
    df = df_raw.copy()
    df = df.sort_values(by=top_col,ascending=False).iloc[:top_n,9:-2]
    df.columns = [col.replace("test_", "") if col != "experiments" else col for col in df.columns]
    return df

@timer
def load_model(s3_bucket: str, model_key: str):
    """
    Load IsolationForest model.pkl from S3.
    """
    s3 = boto3.client("s3")

    # Download model.pkl
    file_obj = s3.get_object(Bucket=s3_bucket, Key=model_key)
    body = file_obj["Body"].read()

    # Load with pickle
    model = pickle.load(BytesIO(body))
    return model


def make_df(X_scaled, y_pred, y_true, dataset_name):
    df = pd.DataFrame(X_scaled)
    df['y_pred'] = y_pred
    df['y_true'] = y_true
    df['data_set'] = dataset_name
    return df


def evaluate_percentile_cuts_generic(df,X_train,y_train,y_test,pct_list,run_id,n_features,algo="dbscan",eps=None,min_samples=None,min_cluster_size=None,metric="euclidean"):
    # Separate feature sets by label - X0 is train only
    X0 = df[(df['data_set']=='train_set')&(df['y_pred']==0)].drop(columns=['y_pred','y_true','data_set']).to_numpy() 
    # Separate feature sets by label X0train vs X1train and X0train vs X1test
    X1_train = df[(df['data_set']=='train_set')&(df['y_pred']==1)].drop(columns=['y_pred','y_true','data_set']).to_numpy()
    X1_test = df[(df['data_set']=='test_set')&(df['y_pred']==1)].drop(columns=['y_pred','y_true','data_set']).to_numpy()
    
    # Compute distances (only if both groups non-empty)
    dist_train = euclidean_distances(X1_train, X0).sum(axis=1) if (X0.shape[0] > 0 and X1_train.shape[0] > 0) else np.array([])
    dist_test  = euclidean_distances(X1_test, X0).sum(axis=1)  if (X0.shape[0] > 0 and X1_test.shape[0] > 0)  else np.array([])
    
    # Assign distances back
    df['dist'] = 0.0
    df.loc[(df['y_pred'] == 1) & (df['data_set'] == 'train_set'), 'dist'] = dist_train
    df.loc[(df['y_pred'] == 1) & (df['data_set'] == 'test_set'), 'dist']  = dist_test

    # Generate binary flags based on percentiles (train anomalies only)
    cut_rows = []
    train_anom_dist = df.loc[df['data_set'] == 'train_set', 'dist'].to_numpy()
    for q in pct_list:
        if np.isnan(train_anom_dist).any():
            df[f'p{q}'] = 0
            thresh = None
        else:
            thresh = np.percentile(train_anom_dist, q)
            df[f'p{q}'] = (df['dist'] >= thresh).astype(int)
            
        # Train set
        y_pred_train_q = df.loc[df['data_set'] == 'train_set', f'p{q}']
        perf_train_q = calculate_performance(y_pred_train_q, y_train, suffix="train")
        
        # Test set
        y_pred_test_q = df.loc[df['data_set'] == 'test_set', f'p{q}']
        perf_test_q = calculate_performance(y_pred_test_q, y_test, suffix="test")
    
        # Merge results
        perf_q = {**perf_train_q, **perf_test_q}
        # Store experiment params depending on algo
        if algo == "dbscan":
            perf_q['experiments'] = {
                'eps': eps,
                'min_samples': min_samples,
                'metric': metric,
                'pct': q,
                'thresh': thresh
            }
        elif algo == "hdbscan":
            perf_q['experiments'] = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'metric': metric,
                'pct': q,
                'thresh': thresh
            }
        else:
            raise ValueError(f"Unknown algo: {algo}")
            
        perf_q['n_features'] = X_train.shape[1]
        perf_q['id'] = run_id
        cut_rows.append(perf_q)
    return cut_rows

# @timer
# def predict_all_models(bucket_name, prefix, test_set,index,is_dbscan):
#     s3 = boto3.client('s3')
    
#     # 1. List all model files in the S3 bucket
#     response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
#     model_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pkl')]
    
#     predictions_dict = {}
    
#     # 2. Loop through each model file
#     for model_file in tqdm(model_files):
#         # Read model file into memory
#         buffer = io.BytesIO()
#         s3.download_fileobj(Bucket=bucket_name, Key=model_file, Fileobj=buffer)
#         buffer.seek(0)
        
#         # Load the model
#         model = joblib.load(buffer)
        
#         # 3. Predict using the test set
#         if is_dbscan == True:
#             y_pred = dbscan_predict(dbscan_model=model, X_new=test_set, metric=sp.spatial.distance.euclidean)
#         else:
#             y_pred = model.predict(test_set)
            
#         y_pred = (y_pred == -1).astype(int)
#         # 4. Store predictions with filename as column name (no prefix path)
#         filename = model_file.split('/')[-1]
#         predictions_dict[filename] = y_pred
    
#     # 5. Create DataFrame with models as columns
#     df_predictions = pd.DataFrame(predictions_dict)
#     df_predictions = pd.concat([index,df_predictions],axis=1)
    
#     return df_predictions

# @timer
# def evaluate_fraud_predictions(df_lables ,y):
#     print('Total input:', df_lables.shape[0])
#     # df_lables['true_fraud'] = 0
#     # df_lables.loc[df_lables.index.isin(true_fraud_list), 'true_fraud'] = 1
#     # y_true = df_lables['true_fraud']
#     y_true = y
#     # df_lables = df_lables.drop(columns='true_fraud')
    
#     # Create an empty list to collect results
#     results = []

#     for col in tqdm(df_lables.columns):
        
#         y_pred = df_lables[col]

#         cm = confusion_matrix(y_true, y_pred)
#         tn, fp, fn, tp = cm.ravel()

#         # print('-------' + col + '-------')
#         # print("Confusion Matrix:")
#         # print(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))
        
#         noise_count = np.sum(y_pred == 1)
#         precision = precision_score(y_true, y_pred, zero_division=0)
#         recall = recall_score(y_true, y_pred, zero_division=0)
#         f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
#         f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
#         # score = silhouette_score(x_scaled, y_pred) # ks or js -> พวก kl

#         # Store results in the list
#         results.append({
#             'experiments': col,
#             'total_alert': noise_count,
#             'tp': tp,
#             'fp': fp,
#             'tn': tn,
#             'fn': fn,
#             'precision': precision,
#             'recall': recall,
#             'f0.5': f05,
#             'f1': f1
#             # ,'Silhouette_Score':score
#         })

#     # Convert the results into a DataFrame
#     results_df = pd.DataFrame(results)

#     return results_df



# @timer
# def evaluate_percentile_cuts_hdb(df,X_train,y_train,y_test, pct_list, min_cluster_size, min_samples, metric, run_id, n_features):
#     # Separate feature sets by label - X0 is train only
#     X0 = df[(df['data_set']=='train_set')&(df['y_pred']==0)].drop(columns=['y_pred','y_true','data_set']).to_numpy() 
#     # Separate feature sets by label X0train vs X1train and X0train vs X1test
#     X1_train = df[(df['data_set']=='train_set')&(df['y_pred']==1)].drop(columns=['y_pred','y_true','data_set']).to_numpy()
#     X1_test = df[(df['data_set']=='test_set')&(df['y_pred']==1)].drop(columns=['y_pred','y_true','data_set']).to_numpy()
    
#     # Compute distances (only if both groups non-empty)
#     dist_train = euclidean_distances(X1_train, X0).sum(axis=1) if (X0.shape[0] > 0 and X1_train.shape[0] > 0) else np.array([])
#     dist_test  = euclidean_distances(X1_test, X0).sum(axis=1)  if (X0.shape[0] > 0 and X1_test.shape[0] > 0)  else np.array([])
    
#     # Assign distances back
#     df['dist'] = 0.0
#     df.loc[(df['y_pred'] == 1) & (df['data_set'] == 'train_set'), 'dist'] = dist_train
#     df.loc[(df['y_pred'] == 1) & (df['data_set'] == 'test_set'), 'dist']  = dist_test

#     # Generate binary flags based on percentiles (train anomalies only)
#     cut_rows = []
#     train_anom_dist = df.loc[df['data_set'] == 'train_set', 'dist'].to_numpy()
#     for q in pct_list:
#         if np.isnan(train_anom_dist).any():
#             df[f'p{q}'] = 0
#         else:
#             thresh = np.percentile(train_anom_dist, q)
#             df[f'p{q}'] = (df['dist'] >= thresh).astype(int)
            
#         y_pred_train_q = df.loc[df['data_set'] == 'train_set', f'p{q}']
#         perf_train_q = calculate_performance(y_pred_train_q, y_train, suffix="train")
    
#         y_pred_test_q = df.loc[df['data_set'] == 'test_set', f'p{q}']
#         perf_test_q = calculate_performance(y_pred_test_q, y_test, suffix="test")
    
#         perf_q = {**perf_train_q, **perf_test_q}
#         perf_q['experiment'] = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples, 'metric': metric, 'pct':q,'thresh':thresh}
#         perf_q['n_features'] = X_train.shape[1]
#         perf_q['id'] = run_id
#         cut_rows.append(perf_q)
#     return cut_rows


# def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
#     """
#     Assign labels to new data based on the core samples from a trained DBSCAN model.
    
#     Parameters:
#     - dbscan_model: fitted sklearn.cluster.DBSCAN model
#     - X_new: numpy array or list of new input points
#     - metric: distance function (default: euclidean)

#     Returns:
#     - y_new: predicted cluster labels for X_new
#     """
#     if not hasattr(dbscan_model, 'components_'):
#         raise ValueError("DBSCAN model has not been fit or has no core samples.")

#     y_new = np.full(len(X_new), -1, dtype=int)

#     for j, x_new in enumerate(X_new):
#         for i, x_core in enumerate(dbscan_model.components_):
#             if metric(x_new, x_core) < dbscan_model.eps:
#                 y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
#                 break

#     return y_new

# def dbscan_predict(dbscan_model, X_new, metric):
#     # Pairwise distance (shape: [n_new, n_core])
#     dist_matrix = cdist(X_new, dbscan_model.components_, metric=metric)

#     # For each new point, find nearest core point
#     nearest_core = dist_matrix.argmin(axis=1)
#     nearest_dist = dist_matrix[np.arange(len(X_new)), nearest_core]

#     # Start with all as noise (-1)
#     y_new = np.full(shape=len(X_new), fill_value=-1, dtype=int)

#     # Assign labels if within eps
#     mask = nearest_dist < dbscan_model.eps
#     core_labels = dbscan_model.labels_[dbscan_model.core_sample_indices_]
#     y_new[mask] = core_labels[nearest_core[mask]]

#     return y_new