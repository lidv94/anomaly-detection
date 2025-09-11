import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report,accuracy_score,precision_score, recall_score,fbeta_score,confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from mpl_toolkits.mplot3d import Axes3D  # Only used if 3D
from package.utils import timer
import time 
from tqdm import tqdm
import os
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import io
import boto3

s3_client = boto3.client("s3")

@timer
def set_seed(seed=42):
    # print(f"\n[INFO] Setting global seed to {seed}")

    # Python random
    random.seed(seed)
    # print(f"[INFO] Python random: {random.randint(0, 100)}")

    # Numpy
    np.random.seed(seed)
    # print(f"[INFO] NumPy random: {np.random.randint(0, 100)}")

    # Torch
    torch.manual_seed(seed)
    # print(f"[INFO] Torch random: {torch.randint(0, 100, (1,)).item()}")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # print(f"[INFO] Torch CUDA random: {torch.randint(0, 100, (1,), device='cuda').item()}")

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print("[INFO] CUDNN deterministic set to True, benchmark set to False")

@timer
def generate_mock_data(n_sales=300, n_rules=10, fraud_ratio=0.05, seed=42):
    np.random.seed(seed)
    
    timeunits = ['l3', 'l6', 'l9', 'l12', 'l24', 'l36']
    sales_ids = np.arange(n_sales)
    all_data = []

    # Base date
    base_date = pd.to_datetime("2025-06-30")

    # Assign fraud at sales_id level (consistent across timeunits)
    is_fraud = np.random.choice([0, 1], size=n_sales, p=[1-fraud_ratio, fraud_ratio])
    sales_id_fraud_map = dict(zip(sales_ids, is_fraud))

    # Assign expected date per sales_id (spread out for variety)
    sales_id_date_map = {
        s_id: base_date - pd.to_timedelta(np.random.randint(0, 365), unit="D")
        for s_id in sales_ids
    }

    for t in timeunits:
        for s_id in sales_ids:
            row = {
                'sales_id': s_id,
                'expected_dt': sales_id_date_map[s_id],
                'timeunit': t,
                'flag_fraud': sales_id_fraud_map[s_id]
            }
            for i in range(1, n_rules + 1):
                base_val = np.random.normal(loc=0, scale=1)
                if sales_id_fraud_map[s_id] == 1:
                    base_val += np.random.normal(loc=3, scale=1.5)
                row[f'rule_{i}'] = base_val
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    return df



@timer    
def expand_rule_columns(df, rule_prefix="rule_"):
    """
    Expand rule columns so each is suffixed with its corresponding timeunit.
    Example: rule_1 in row with timeunit='l3' becomes rule_1_l3.
    """
    # Identify rule columns
    rule_cols = [col for col in df.columns if col.startswith(rule_prefix)]

    # Melt to long format
    df_long = df.melt(id_vars=["sales_id",'expected_dt', "timeunit", "flag_fraud"],
                      value_vars=rule_cols,
                      var_name="rule",
                      value_name="value")

    # Append timeunit to rule name
    df_long["rule_time"] = df_long["rule"] + "_" + df_long["timeunit"]

    # Pivot to wide format
    df_wide = df_long.pivot(
        index=["sales_id",'expected_dt', "flag_fraud"],
        columns="rule_time",
        values="value"
    )

    # Flatten the columns (remove MultiIndex)
    df_wide.columns = df_wide.columns.tolist()

    # Bring index back as columns
    df_wide = df_wide.reset_index()

    return df_wide

@timer    
def subset_by_timeunit(df, timeunits,keep_cols_list=["sales_id",'expected_dt', "flag_fraud"]):
    """
    Subset DataFrame to only include sales_id, flag_fraud, 
    and rule columns for the selected timeunit(s).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned from expand_rule_columns().
    timeunits : list or str
        Timeunit suffix(es) to keep (e.g., ['l3', 'l36'] or 'l3').

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only the selected timeunit columns.
    """
    # Ensure list format
    if isinstance(timeunits, str):
        timeunits = [timeunits]

    # Always keep these
    keep_cols = keep_cols_list 

    # Add rule columns for each timeunit
    for t in timeunits:
        keep_cols.extend([col for col in df.columns if col.endswith(f"_{t}")])

    # Ensure no duplicates
    keep_cols = list(dict.fromkeys(keep_cols))

    return df[keep_cols]

    
@timer
def split_train_test(df, pk="sales_id",date_col='expected_dt', target="flag_fraud", test_size=0.2, random_state=42):
    """
    Split the dataframe into train/test sets (80-20) stratified by target.
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target], 
        random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


@timer    
def prepare_train_val_test_data(train_df, 
                                test_df, 
                                pk="sales_id", 
                                date_col="expected_dt",
                                target="flag_fraud",
                                val_size=0.2, 
                                random_state=42, 
                                scale=True):
    """
    Prepare training, validation, and testing data for novelty detection.
    
    Training data uses only non-fraud samples (flag_fraud == 0).  
    Validation split is taken from the non-fraud training data.  
    Testing data contains both fraud and non-fraud samples.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame after initial train/test split.
    test_df : pd.DataFrame
        Testing DataFrame after initial train/test split.
    pk : str, default "sales_id"
        Primary key column name to drop from features.
    date_col : str, default "expected_dt"
        Date column name to drop from features.
    target : str, default "flag_fraud"
        Target column name.
    val_size : float, default 0.2
        Fraction of non-fraud training data to use for validation.
    random_state : int, default 42
        Random seed for reproducibility.
    scale : bool, default False
        Whether to scale features using StandardScaler.
        
    Returns
    -------
    X_train : pd.DataFrame
        Training features containing only non-fraud samples.
    X_val : pd.DataFrame
        Validation features containing only non-fraud samples.
    X_test : pd.DataFrame
        Testing features containing all samples (fraud and non-fraud).
    y_test : pd.Series
        Testing target labels.
    X_train_nonfraud : pd.DataFrame
        Training (non-fraud only) features, scaled if requested.
    y_train_nonfraud : pd.Series
        Training (non-fraud only) target labels.
    all_df : pd.DataFrame
        Concatenated train_df and test_df.
    X_all : pd.DataFrame
        All features (scaled if requested).
    y_all : pd.Series
        All target labels.
    """
    # Separate features (drop primary key and target)
    exclude_cols = {pk, date_col, target} & set(train_df.columns)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Concatenate train + test for "all data"
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    X_all = all_df[feature_cols]
    y_all = all_df[target].copy()

    # All train data (both fraud, nf)
    train_all_df = train_df.copy()
    X_train_all = train_all_df[feature_cols].reset_index(drop=True)
    y_train_all = train_all_df[target].reset_index(drop=True)
    
    # Filter only non-fraud
    train_nonfraud_df = train_df[train_df[target] == 0]
    X_train_nonfraud = train_nonfraud_df[feature_cols].reset_index(drop=True)
    y_train_nonfraud = train_nonfraud_df[target].reset_index(drop=True)

    # ----- Split non-fraud training data into pure train and validation sets -----
    pure_train_nonfraud, val_nonfraud = train_test_split(
        train_nonfraud_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_nonfraud_df[target]  # stratify even if all zeros for consistency
    )

    pure_train_nonfraud = pure_train_nonfraud.reset_index(drop=True)
    val_nonfraud = val_nonfraud.reset_index(drop=True)
    X_train = pure_train_nonfraud[feature_cols]
    X_val = val_nonfraud[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target].copy()


    # Optional scaling
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index) # Fit the scaler only on pure X_train (non-fraud).
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
        X_train_all = pd.DataFrame(scaler.transform(X_train_all), columns=feature_cols, index=X_train_all.index)
        X_train_nonfraud = pd.DataFrame(scaler.transform(X_train_nonfraud), columns=feature_cols, index=X_train_nonfraud.index)
        X_all = pd.DataFrame(scaler.transform(X_all), columns=feature_cols, index=X_all.index)

    print("Data Prepared:")
    print(f"all_df            : {all_df.shape}")
    print(f"X_all             : {X_all.shape}, y_all: {y_all.shape}")
    print(f"train_all_df      : {train_all_df.shape}")
    print(f"X_train_all       : {X_train_all.shape}, y_train_all: {y_train_all.shape}")
    print(f"train_nonfraud_df : {train_nonfraud_df.shape}")
    print(f"X_train_nonfraud  : {X_train_nonfraud.shape}, y_train_nonfraud: {y_train_nonfraud.shape}")
    print(f"X_train           : {X_train.shape}")
    print(f"X_val             : {X_val.shape}")
    print(f"X_test            : {X_test.shape}, y_test: {y_test.shape}")
    print("-" * 50)
    
    return (all_df, X_all, y_all , 
            train_all_df, X_train_all,y_train_all,
            train_nonfraud_df, X_train_nonfraud, y_train_nonfraud, 
            X_train, X_val, X_test, y_test, 
            )

def init_weights(layer, seed=42):
    """
    Deterministic per-layer initialization.
    Setting the seed during weight initialization ensures that your model always starts from the exact same point.

    That means your experiments are reproducible, results are fairly comparable, and debugging becomes much easier.
    
    """
    if isinstance(layer, nn.Linear):
        torch.manual_seed(seed)  # reset RNG for reproducibility
        nn.init.xavier_uniform_(layer.weight)  # you can change to kaiming, normal, etc.
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

class Autoencoder(nn.Module):
    """Autoencoder neural network with customizable architecture and optional dropout."""

    def __init__(self, input_dim, encoding_dim=4, hidden_layers=[8], 
                 activation=nn.ReLU, dropout=0.0, verbose=False, seed=42):
        super().__init__()
        self.verbose = verbose
        self.activation_cls = activation
        self.dropout = dropout
        self.seed = seed

        # Encoder
        encoder = []
        prev_dim = input_dim
        for idx, h in enumerate(hidden_layers):
            encoder.append(nn.Linear(prev_dim, h))
            encoder.append(activation())
            if dropout > 0:
                encoder.append(nn.Dropout(p=dropout)) 
            prev_dim = h
        encoder.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = []
        prev_dim = encoding_dim
        for idx, h in enumerate(reversed(hidden_layers)):
            decoder.append(nn.Linear(prev_dim, h))
            decoder.append(activation())
            if dropout > 0:
                decoder.append(nn.Dropout(p=dropout)) 
            prev_dim = h
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

        # Apply deterministic initialization per layer
        self.encoder.apply(lambda l: init_weights(l, seed=self.seed))
        self.decoder.apply(lambda l: init_weights(l, seed=self.seed))

    def forward(self, x):
        if self.verbose:
            print(f"Input: {x.shape}")
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.verbose:
                print(f"Encoder layer {i}: {x.shape}")
        latent = x
        if self.verbose:
            print(f"Latent space: {latent.shape}")
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if self.verbose:
                print(f"Decoder layer {i}: {x.shape}")
        if self.verbose:
            print(f"Output: {x.shape}")
        return x

    def encode(self, x):
        if self.verbose:
            print(f"Input to encoder: {x.shape}")
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.verbose:
                print(f"Encoder layer {i}: {x.shape}")
        if self.verbose:
            print(f"Latent embedding: {x.shape}")
        return x


#####################

def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")

def _split_s3_path(s3_path: str):
    """Split s3://bucket/key into bucket, key"""
    assert s3_path.startswith("s3://")
    path_parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ""
    return bucket, key

@timer
def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir, prefix="checkpoint"):
    """
    Save model checkpoint to local disk or S3.
    """
    checkpoint_data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }

    filename = f"{prefix}_epoch{epoch}.pkl"

    if _is_s3_path(run_dir):
        bucket, prefix_key = _split_s3_path(run_dir)
        key = f"{prefix_key}/{filename}" if prefix_key else filename

        buffer = io.BytesIO()
        pickle.dump(checkpoint_data, buffer)
        buffer.seek(0)
        s3_client.upload_fileobj(buffer, bucket, key)
        print(f"✅ Saved checkpoint to S3: s3://{bucket}/{key}")

    else:
        os.makedirs(run_dir, exist_ok=True)
        file_path = os.path.join(run_dir, filename)
        with open(file_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"✅ Saved checkpoint locally: {file_path}")


@timer
def load_checkpoint(file_path, model=None, optimizer=None, device=None):
    """
    Load model & optimizer state from a checkpoint stored locally or in S3.

    Parameters
    ----------
    file_path : str
        Local path or S3 URI (e.g., "s3://bucket/key").
    model : torch.nn.Module, optional
        If provided, loads the state_dict into this model.
    optimizer : torch.optim.Optimizer, optional
        If provided, loads the optimizer state_dict.
    device : torch.device, optional
        Device to map model to. Defaults to CUDA if available, else CPU.

    Returns
    -------
    start_epoch : int
    train_losses : list
    val_losses : list
    run_dir : str
        Directory or S3 path prefix where checkpoint was loaded from.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _is_s3_path(file_path):
        bucket, key = _split_s3_path(file_path)
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        checkpoint_data = pickle.load(buffer)
        print(f"🔄 Loaded checkpoint from S3: {file_path} (epoch {checkpoint_data['epoch']})")
        run_dir = f"s3://{bucket}/{os.path.dirname(key)}"
    else:
        with open(file_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        print(f"🔄 Loaded checkpoint locally: {file_path} (epoch {checkpoint_data['epoch']})")
        run_dir = os.path.dirname(file_path)

    if model is not None and "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])
        model.to(device)

    if optimizer is not None and "optimizer_state" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])

    return (
        checkpoint_data["epoch"],
        checkpoint_data.get("train_losses", []),
        checkpoint_data.get("val_losses", []),
        run_dir
    )

#####################

# --- Loss function selector ---
def get_loss_function(loss_name: str):
    """
    Return a PyTorch loss function based on the provided name.

    Recommended usage in anomaly detection / autoencoder training:

    - "mse": Mean Squared Error (default for autoencoders).  
        Use when input and output are continuous and you want to minimize squared reconstruction errors.  
        Common for anomaly detection with numeric features.

    - "mae": Mean Absolute Error.  
        More robust to outliers compared to MSE.  
        Use when your data has heavy-tailed noise or you want absolute deviation instead of squared error.

    - "huber" / "smoothl1": Huber Loss (Smooth L1).  
        A compromise between MSE and MAE.  
        Use when you want robustness to outliers but still care about squared error for small deviations.

    - "bce": Binary Cross Entropy.  
        Use when reconstructing binary input features (e.g., one-hot encoded categorical variables).

    - "bce_logits": BCE with Logits.  
        Same as BCE but more numerically stable (expects raw logits instead of probabilities).  
        Use for binary features when your decoder does not apply a sigmoid activation.

    - "kldiv": Kullback-Leibler Divergence.  
        Often used in Variational Autoencoders (VAE) for regularization.  
        Not usually applied to plain reconstruction error, but combined with another loss.

    - "cosine": Cosine Embedding Loss.  
        Use when you care about the *direction* (cosine similarity) of the reconstructed vector rather than magnitude.  
        Less common, but can be useful when embeddings are normalized or when angular similarity is more meaningful.

    Parameters
    ----------
    loss_name : str
        Name of the loss function (e.g., "mse", "mae", "huber", "bce", "bce_logits", "kldiv", "cosine").

    Returns
    -------
    torch.nn.Module
        Corresponding PyTorch loss function.

    Raises
    ------
    ValueError
        If the provided loss_name is not supported.
    """
    loss_name = loss_name.lower()

    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name in ["huber", "smoothl1"]:
        return nn.SmoothL1Loss(beta=1.0)
    elif loss_name == "bce":
        return nn.BCELoss()
    elif loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "kldiv":
        return nn.KLDivLoss(reduction="batchmean")
    elif loss_name == "cosine":
        return nn.CosineEmbeddingLoss()
    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Choose from ['mse', 'mae', 'huber', 'bce', 'bce_logits', 'kldiv', 'cosine']"
        )

# --- Optimizer selector for autoencoders ---
def get_optimizer(model: nn.Module, optimizer_name: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0):
    """
    Return a PyTorch optimizer for autoencoder training.

    | Optimizer                     | Description / When to Use                                                                 |
    | ----------------------------- | ----------------------------------------------------------------------------------------- |
    | **Adam** (`optim.Adam`)       | Default choice; adaptive LR, stable, fast.                                               |
    | **AdamW** (`optim.AdamW`)     | Variant of Adam with better weight decay handling; improves regularization.              |
    | **RMSProp** (`optim.RMSprop`) | Helps stabilize training if gradients are noisy; sometimes useful for deeper autoencoders. |

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    optimizer_name : str
        Name of optimizer ("adam", "adamw", "rmsprop").
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.

    Returns
    -------
    torch.optim.Optimizer
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Choose from ['adam', 'adamw', 'rmsprop']"
        )


@timer
def get_scheduler(optimizer, scheduler_name="plateau", factor=0.5, patience=5, step_size=10):
    if scheduler_name.lower() == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    elif scheduler_name.lower() == "steplr":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=factor)
    else:
        scheduler = None
    return scheduler


# --- Training function ---
@timer
def train_autoencoder_checkpoint(model, 
                                 X_train, 
                                 X_val,
                                 epochs=50, 
                                 batch_size=32, 
                                 lr=1e-3,
                                 checkpoint_every=100,
                                 checkpoint_path=None,
                                 run_dir=None,
                                 early_stopping_patience=None,
                                 weight_decay=0.0,
                                 sparsity_lambda=0.0,
                                 loss_name="mae", 
                                 optimizer_name="adam",
                                 scheduler_name="plateau",
                                 scheduler_params=None, 
                                 verbose=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # ---- Ensure reproducibility for DataLoader ----
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        print(f"[INFO] Worker {worker_id} seed -> Torch: {worker_seed}, "
              f"Numpy: {np.random.randint(0, 100)}, Random: {random.randint(0, 100)}")

    g = torch.Generator()
    g.manual_seed(42)  # <- use the same seed as set_seed()

    print("[INFO] DataLoader seed info:")
    print(f"       Global generator manual_seed: 42")
    print(f"       Torch initial seed: {torch.initial_seed()}")
    print(f"       Generator state (deterministic): {torch.backends.cudnn.deterministic}")
    print(f"       Benchmark mode: {torch.backends.cudnn.benchmark}")

    # Prepare data
    X_train = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val.values if hasattr(X_val, "values") else X_val, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        X_train,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,  # <- NEW
        generator=g                  # <- NEW
    )
    val_loader = torch.utils.data.DataLoader(
        X_val,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,  # <- NEW
        generator=g                  # <- NEW
    )

    # --- Flexible loss ---
    criterion = get_loss_function(loss_name)
    print(f"[INFO] Using loss function: {criterion.__class__.__name__}")
    
    # --- Flexible optimizer ---
    optimizer = get_optimizer(model, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)
    last_lr = optimizer.param_groups[0]['lr']
    print(f"[INFO] Using optimizer: {optimizer.__class__.__name__}, LR: {last_lr:.6f}")

    # --- Scheduler setup ---
    if scheduler_name is not None:
        default_scheduler_params = {"factor": 0.5, "patience": 5, "step_size": 10}
        if scheduler_params is None:
            scheduler_params = default_scheduler_params
        else:
            for k, v in default_scheduler_params.items():
                scheduler_params.setdefault(k, v)
    
        scheduler = get_scheduler(
            optimizer,
            scheduler_name=scheduler_name,
            factor=scheduler_params.get("factor"),
            patience=scheduler_params.get("patience"),
            step_size=scheduler_params.get("step_size")
        )
        print(f"[INFO] Using scheduler: {scheduler.__class__.__name__} with params: {scheduler_params}" if scheduler is not None else "[INFO] Scheduler not used.")
    else:
        scheduler = None
        print("[INFO] Scheduler not used.")

    # --- cont epoch ---    
    start_epoch = 0
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # --- Resume logic ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, train_losses, val_losses, run_dir = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
    else:
        if run_dir is None:
            run_dir = f"model_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            run_dir = os.path.join(run_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            if not _is_s3_path(run_dir):
                os.makedirs(run_dir, exist_ok=True)

    best_state_dict = None
    
    for epoch in range(start_epoch, epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            base_loss = criterion(output, batch)
            
            # --- Sparsity regularization ---
            if sparsity_lambda > 0.0:
                latent = model.encode(batch)  # get latent representation
                sparsity_loss = sparsity_lambda * torch.mean(torch.abs(latent))
                loss = base_loss + sparsity_loss
            else:
                loss = base_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                base_loss = criterion(output, batch)
                
                if sparsity_lambda > 0.0:
                    latent = model.encode(batch)
                    sparsity_loss = sparsity_lambda * torch.mean(torch.abs(latent))
                    loss = base_loss + sparsity_loss
                else:
                    loss = base_loss

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Scheduler step ---
        if scheduler is not None:
            if scheduler_name.lower() == "plateau":
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # --- Print LR only if changed ---
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"[INFO] Current LR: {current_lr:.6f}")
            last_lr = current_lr

        # --- Save checkpoint ---
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir)

        # # --- Early stopping & best-model save (guarded) ---
        if early_stopping_patience is not None:
            if epoch == 0 and verbose:  
                print(f"✅ Early stopping is enabled (patience={early_stopping_patience})")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_state_dict = model.state_dict()  # keep in memory
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

    if best_state_dict is not None:
        save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir,
                        prefix="best_model")

    return model, train_losses, val_losses

@timer
def load_pickled_autoencoder(checkpoint_path, model=None, device=None):
    """
    Load a pickled autoencoder checkpoint (from local or S3).
    
    Args:
        checkpoint_path (str): Local path or S3 URI (e.g., s3://bucket/key).
        model (torch.nn.Module, optional): Autoencoder model to load weights into.
        device (torch.device, optional): Torch device. Defaults to CUDA if available, else CPU.
    
    Returns:
        model (torch.nn.Module): Model with loaded weights (if provided).
        train_losses (list): Training loss history.
        val_losses (list): Validation loss history.
        last_epoch (int): Last trained epoch.
    """
    
    # Default device handling
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read from S3 or local
    if checkpoint_path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = checkpoint_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        checkpoint = pickle.load(io.BytesIO(obj["Body"].read()))
    else:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

    train_losses = checkpoint.get("train_losses", None)
    val_losses = checkpoint.get("val_losses", None)
    last_epoch = checkpoint.get("epoch", None)

    # Load weights into model
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
    return model, train_losses, val_losses, last_epoch



@timer
def plot_learning_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

@timer
def get_reconstruction_error(model, X_scaled, loss_name="mse"):
    """
    Compute per-sample reconstruction error for an autoencoder using the specified loss.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder.
    X_scaled : np.array or pd.DataFrame
        Features scaled data.
    loss_name : str
        Loss function to use ("mse", "mae", "huber", etc.).

    Returns
    -------
    np.array
        Reconstruction error for each sample.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_tensor = torch.tensor(X_scaled.values if hasattr(X_scaled, "values") else X_scaled,
                            dtype=torch.float32, device=device)
    
    loss_fn = get_loss_function(loss_name)
    # Set reduction='none' for per-element loss
    if hasattr(loss_fn, 'reduction'):
        loss_fn.reduction = 'none'
    
    with torch.no_grad():
        X_pred = model(X_tensor)
        # Compute element-wise loss
        elem_loss = loss_fn(X_pred, X_tensor)
        # Average over features to get per-sample reconstruction error
        recon_error = elem_loss.mean(dim=1).cpu().numpy()

    print(f"[INFO] Reconstruction error calculated using {loss_name.upper()} for {X_tensor.shape[0]} samples")
    print(f"[INFO] Error stats -> min: {recon_error.min():.4f}, max: {recon_error.max():.4f}, mean: {recon_error.mean():.4f}")
    
    return recon_error

@timer
# Flag anomalies based on a chosen threshold
def flag_anomalies(recon_error, threshold):
    """
    Flag samples as anomaly/fraud if recon_error > threshold.

    Returns:
    - y_pred: np.array of 0/1 flags
    """
    y_pred = (recon_error > threshold).astype(int)
    return y_pred

@timer
def extract_embeddings(model, X):
    """
    Extract latent embeddings from the encoder of an Autoencoder.

    Parameters:
    - model: Autoencoder with `.encode()` method
    - X: np.array or pd.DataFrame (input data)

    Returns:
    - embeddings: np.array of shape (n_samples, encoding_dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32).to(device)
    print(f"[INFO] Input shape: {X_tensor.shape}")

    with torch.no_grad():
        embeddings = model.encode(X_tensor).cpu().numpy()

    print(f"[INFO] Extracted embeddings shape: {embeddings.shape}")
    return embeddings

# ----------------- 2D version -----------------
@timer
def plot_latent_space_2d(embeddings, y_test, y_pred, index_df=None, hover_col=None, marker_size=5, use_tsne=True):
    """
    Plot 2D latent space with ground truth and predicted labels side by side.
    Fraud = red, Normal = blue, shared legend across subplots.
    Hover shows index or specified column from index_df.
    Automatically reduces embeddings >2D to 2D using t-SNE.

    Parameters:
    - embeddings: np.array of shape (n_samples, n_features)
    - y_test: np.array of ground truth labels (0=Normal, 1=Fraud)
    - y_pred: np.array of predicted labels (0=Normal, 1=Fraud)
    - index_df: pd.DataFrame corresponding to embeddings for hover info (optional)
    - hover_col: str, column in index_df to display on hover (optional)
    - marker_size: int, size of markers
    - use_tsne: bool, if True, reduce >2D embeddings to 2D using t-SNE
    """

    if embeddings.shape[1] > 2:
        if use_tsne:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            raise ValueError("Embeddings have more than 2 dimensions. Set use_tsne=True to reduce them.")
    else:
        embeddings_2d = embeddings

    df = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
    df['hover'] = index_df[hover_col].values if index_df is not None and hover_col is not None else df.index.astype(str)

    color_map = {0: 'lightblue', 1: 'red'}
    label_map = {0: 'Normal', 1: 'Fraud'}

    def _add_scatter(fig, labels, row=1, col=1, first_occurrence=False):
        labels = np.array(labels)
        for val in [0, 1]:
            idx = labels == val
            color = color_map[val]
            name = label_map[val]
            showlegend = first_occurrence
            hover = [f"{df.loc[i, 'hover']} (Class: {name})" for i in df.loc[idx].index]

            fig.add_trace(
                go.Scatter(
                    x=df.loc[idx, 'Dim1'],
                    y=df.loc[idx, 'Dim2'],
                    mode='markers',
                    marker=dict(color=color, size=marker_size, opacity=0.3),
                    name=name,
                    legendgroup=name,
                    showlegend=showlegend,
                    hovertext=hover,
                    hoverinfo='text+name'
                ),
                row=row, col=col
            )

    # Always assume both y_test and y_pred are provided
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'xy'}, {'type':'xy'}]],
                        subplot_titles=("Latent Space (Y True Flag)", "Latent Space (Y Predicted Flag)"))
    _add_scatter(fig, y_test, row=1, col=1, first_occurrence=True)
    _add_scatter(fig, y_pred, row=1, col=2, first_occurrence=False)

    fig.update_layout(
            height=600,
            width=1200,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
    fig.show()

# ----------------- 3D version -----------------
@timer
def plot_latent_space_3d(embeddings, y_test, y_pred, index_df=None, hover_col=None, marker_size=5):
    """
    Plot 3D latent space with ground truth and predicted labels.
    Fraud = red, Normal = blue, shared legend across subplots.
    Hover shows index or specified column from index_df.

    Parameters:
    - embeddings: np.array of shape (n_samples, 3)
    - y_test: np.array of ground truth labels (0=Normal, 1=Fraud)
    - y_pred: np.array of predicted labels (0=Normal, 1=Fraud)
    - index_df: pd.DataFrame corresponding to embeddings for hover info (optional)
    - hover_col: str, column in index_df to display on hover (optional)
    """
    if embeddings.shape[1] != 3:
        raise ValueError("Embeddings must have 3 dimensions for 3D plot.")
    
    df = pd.DataFrame(embeddings, columns=['Dim1', 'Dim2', 'Dim3'])
    df['hover'] = index_df[hover_col].values if index_df is not None and hover_col is not None else df.index.astype(str)

    color_map = {0: 'lightblue', 1: 'red'}
    label_map = {0: 'Normal', 1: 'Fraud'}

    def _add_scatter3d(fig, labels, row=1, col=1, first_occurrence=False):
        labels = np.array(labels)
        for val in [0, 1]:
            idx = labels == val
            color = color_map[val]
            name = label_map[val]
            showlegend = first_occurrence
            hover = [f"{df.loc[i, 'hover']} (Class: {name})" for i in df.loc[idx].index]

            fig.add_trace(
                go.Scatter3d(
                    x=df.loc[idx, 'Dim1'],
                    y=df.loc[idx, 'Dim2'],
                    z=df.loc[idx, 'Dim3'],
                    mode='markers',
                    marker=dict(color=color, size=marker_size, opacity=0.3),
                    name=name,
                    legendgroup=name,
                    showlegend=showlegend,
                    hovertext=hover,
                    hoverinfo='text+name'
                ),
                row=row, col=col
            )

    # Only handle the case where both y_test and y_pred are provided
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]],
                        subplot_titles=("Latent Space (Y True Flag)", "Latent Space (Y Predicted Flag)"))
    _add_scatter3d(fig, y_test, row=1, col=1, first_occurrence=True)
    _add_scatter3d(fig, y_pred, row=1, col=2, first_occurrence=False)
    fig.update_layout(
        height=600,
        width=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.show()


        
@timer
def create_pack_results(full_df, y_pred, experiment_name
                        , id_col='sales_id'
                        # , date_col = 'expected_dt'
                       ):
    """
    Create a wide-format DataFrame with sales_id and one experiment column for predictions.

    Parameters:
    - full_df: pd.DataFrame with the full cols data (to get sales_id and date_col)
    - y_pred: array-like predicted labels
    - experiment_name: str, the column name for predictions (e.g., 'ae_1')
    - id_col: str, name of the ID column in test_df (default 'sales_id')

    Returns:
    - pd.DataFrame with columns ['sales_id', experiment_name]
    """
    df_results = pd.DataFrame({
        id_col: full_df[id_col].values,
        # date_col:full_df[date_col].values,
        experiment_name: y_pred
    })
    return df_results

@timer
def create_pack_results_date(full_df, y_pred, experiment_name
                        , id_col='sales_id'
                        , date_col = 'expected_dt'
                       ):

    df_results = pd.DataFrame({
        id_col: full_df[id_col].values,
        date_col:full_df[date_col].values,
        experiment_name: y_pred
    })
    return df_results
    

@timer
# evaluate_fraud_predictions
def evaluate_fraud_predictions(x_scaled, df_lables ,true_fraud_list):
    print('Total input:', df_lables.shape[0])
    df_lables['true_fraud'] = 0
    df_lables.loc[df_lables.index.isin(true_fraud_list), 'true_fraud'] = 1
    y_true = df_lables['true_fraud']
    df_lables = df_lables.drop(columns=['true_fraud'])
    
    # Create an empty list to collect results
    results = []

    for col in tqdm(df_lables.columns):
        
        y_pred = df_lables[col]
        # print(f'ypred{y_pred}')
        # print(f'ytrue{y_true}')
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

@timer
#### add loop each percentile ####
def evaluate_thresholds(x_scaled,
                        train_all_df , 
                        y_train_all, 
                        recon_error, 
                        true_fraud_list, 
                        exp_name ='default_all' ,               
                        percentiles=None,
                       ):
    if percentiles is None:
        percentiles = [10,20,30,40,50,60,70,80,85,90,95,97,99]

    df_lables_all = pd.DataFrame(index=train_all_df['sales_id'])
    for p in percentiles:
        # threshold = np.percentile(recon_error, p)
        threshold = np.percentile(recon_error[y_train_all == 0], p)
        y_pred = flag_anomalies(recon_error, threshold)

        experiment_name = 'ae_' + exp_name + f'_p{p}'
        df_labels_tmp = create_pack_results(full_df=train_all_df, y_pred=y_pred, experiment_name=experiment_name)
        df_labels_tmp = df_labels_tmp.set_index(['sales_id'])

        df_lables_all[experiment_name] = df_labels_tmp[experiment_name]

    # Run evaluation on all thresholds at once
    results_df = evaluate_fraud_predictions(
        x_scaled=x_scaled,
        df_lables=df_lables_all.copy(),
        true_fraud_list=true_fraud_list
    )

    return results_df
    
@timer
def get_thresholds(recon_error, y_train_all, exp_name="default_all"):
    """
    Calculate threshold cutoffs for each percentile (1–100) 
    using only non-fraud (y=0) reconstruction errors.
    
    Returns
    -------
    thresholds_df : pd.DataFrame
        experiment_name | percentile | threshold_value
    """
    thresholds = []
    for p in range(1, 101):  # every percentile from 1 to 100
        threshold = np.percentile(recon_error[y_train_all == 0], p)
        experiment_name = f"ae_{exp_name}_p{p}"
        thresholds.append({
            # "experiment_name": experiment_name,
            "percentile": p,
            "threshold_value": threshold
        })

    thresholds_df = pd.DataFrame(thresholds)# .set_index("experiment_name")
    return thresholds_df

@timer
def map_errors_to_percentiles(recon_error, thresholds_df):
    """
    Map each reconstruction error to its corresponding percentile 
    based on the precomputed threshold cutoffs.
    
    Parameters
    ----------
    recon_error_all : np.ndarray
        Array of reconstruction errors (shape: n_samples,).
    thresholds_df : pd.DataFrame
        DataFrame with columns ['percentile', 'threshold_value'].
        Must be sorted by percentile in ascending order.
    
    Returns
    -------
    percentiles : np.ndarray
        Array of same shape as recon_error_all, each element is the
        percentile where the error belongs.
    """
    # Ensure sorted
    thresholds = thresholds_df.sort_values("percentile")
    cutoffs = thresholds["threshold_value"].values
    percentiles = thresholds["percentile"].values
    
    # For each error, find index of the highest threshold it passes
    indices = np.searchsorted(cutoffs, recon_error, side="right") - 1
    
    # Clip to valid range
    indices = np.clip(indices, 0, len(percentiles) - 1)

    mapped_percentiles = percentiles[indices].astype(float)

    # Convert percentile to probability
    probs = mapped_percentiles / 100.0
    return probs

    
################### Pipeline ########################
# ----------------- Function 1: Training Pipeline -----------------
@timer
def run_training_pipeline(
    input_dim,
    encoding_dim,
    hidden_layers_list,
    dropout,
    X_scaled_train,
    X_scaled_val,
    seed: int = 42,
    epochs: int = 20,
    checkpoint_every: int = 10,
    early_stopping_patience: int = 10,
    run_dir: str = "model_checkpoint",
    loss_name: str = "mae",
    optimizer_name: str = "adam",
    scheduler_name="plateau",
    scheduler_params={"factor": 0.2, "patience": 5},
):
    set_seed(seed)

    # ---- set architecture ----
    model_ae = Autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        hidden_layers=hidden_layers_list,
        dropout=dropout,
        verbose=False  
    )

    # ---- Print summary ----
    print("\n===== Autoencoder Architecture =====")
    dummy_input = torch.randn(5, X_scaled_train.shape[1])
    summary(model_ae, input_data=dummy_input, verbose=1)  # ✅ force print summary
    print("===================================\n")

    # ---- Train model ----
    trained_model, train_losses, val_losses = train_autoencoder_checkpoint(
        model=model_ae,
        X_train=X_scaled_train,
        X_val=X_scaled_val,
        epochs=epochs,
        checkpoint_every=checkpoint_every,
        early_stopping_patience=early_stopping_patience,
        run_dir=run_dir,
        loss_name=loss_name,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        scheduler_params=scheduler_params,
    )

    # ---- Plot learning curves ----
    plot_learning_curve(train_losses, val_losses)

    return trained_model, train_losses, val_losses


@timer
def run_lookup_threshold_pipeline(trained_model,
                                  X_scaled_train_all,
                                  train_all_df,
                                  y_train_all,
                                  true_fraud_list,
                                  loss_name="mae",
                                  percentile=95):
    
    # Step 1: Reconstruction error
    recon_error_train = get_reconstruction_error(
        trained_model, X_scaled_train_all, loss_name=loss_name
    )

    # Step 2: Lookup thresholds evaluation
    lookup_pct_threshold = evaluate_thresholds(
        x_scaled=X_scaled_train_all,
        train_all_df=train_all_df,
        y_train_all=y_train_all,
        recon_error=recon_error_train,
        true_fraud_list=true_fraud_list
    )

    # Step 3: Chosen threshold from normal data only
    chosen_threshold = np.percentile(
        recon_error_train[y_train_all == 0], percentile
    )

    # Step 4: Flag anomalies
    y_pred_train = flag_anomalies(recon_error_train, chosen_threshold)

    return recon_error_train, lookup_pct_threshold, chosen_threshold, y_pred_train


@timer
def run_evaluate_test_pipeline(trained_model, 
                               X_scaled_test, 
                               test_df, 
                               threshold, 
                               experiment_name="ae_test_1",
                               loss_name="mae"):

    # Step 1: Get reconstruction errors
    recon_error_test = get_reconstruction_error(trained_model, X_scaled_test, loss_name=loss_name)

    # Step 2: Flag anomalies using given threshold
    y_pred_test = flag_anomalies(recon_error_test, threshold)

    # Step 3: Pack results with metadata
    result_test_df = create_pack_results_date(test_df, y_pred_test, experiment_name=experiment_name)

    return recon_error_test, y_pred_test, result_test_df

@timer
def run_vis_embedding_pipeline(
    model,
    X_data,
    y_data,
    index_df,
    hover_col="sales_id",
    threshold=0.5,
    experiment_name="ae_all_1",
    loss_name="mae",
    plot_type="2d"  # choices: "2d", "3d", "both", "none"
):
    """
    Run pipeline for embeddings visualization and anomaly detection.
    
    Parameters
    ----------
    model : trained autoencoder
    X_data : np.array
        Scaled features input.
    y_data : array-like
        True labels or classes (optional).
    index_df : pd.DataFrame
        DataFrame with indices and metadata.
    hover_col : str, default="sales_id"
        Column to display on hover in plots.
    threshold : float, default=0.5
        Threshold for anomaly flagging.
    experiment_name : str, default="ae_all_1"
        Name tag for saving results.
    loss_name : str, default="mae"
        Loss function used for reconstruction error.
    plot_type : {"2d", "3d", "both", "none"}, default="2d"
        Which latent space plot to display. "none" disables plotting.
    
    Returns
    -------
    recon_error : np.array
        Reconstruction error for each sample.
    y_pred : np.array
        Binary anomaly flags.
    embeddings : np.array
        Latent embeddings from encoder.
    result_df : pd.DataFrame
        Packaged results with anomaly flags.
    """
    
    # Reconstruction error
    recon_error = get_reconstruction_error(model, X_data, loss_name=loss_name)

    # Flag anomalies
    y_pred = flag_anomalies(recon_error, threshold)

    # Extract embeddings
    embeddings = extract_embeddings(model, X_data)

    # Latent space visualization
    if plot_type.lower() in ["2d", "both"]:
        plot_latent_space_2d(
            embeddings=embeddings,
            y_test=y_data,
            y_pred=y_pred,
            index_df=index_df,
            hover_col=hover_col
        )
    if plot_type.lower() in ["3d", "both"]:
        plot_latent_space_3d(
            embeddings=embeddings,
            y_test=y_data,
            y_pred=y_pred,
            index_df=index_df,
            hover_col=hover_col
        )
    # if plot_type == "none", skip plotting

    # Pack results
    result_df = create_pack_results_date(index_df, y_pred, experiment_name=experiment_name)

    return recon_error, y_pred, embeddings, result_df

