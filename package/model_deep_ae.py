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



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@timer
def generate_mock_data(n_sales=300, n_rules=10, fraud_ratio=0.05, seed=42):
    np.random.seed(seed)
    
    timeunits = ['l3', 'l6', 'l9' , 'l12', 'l24','l36']
    sales_ids = np.arange(n_sales)
    all_data = []

    # Assign fraud at sales_id level (consistent across timeunits)
    is_fraud = np.random.choice([0, 1], size=n_sales, p=[1-fraud_ratio, fraud_ratio])
    sales_id_fraud_map = dict(zip(sales_ids, is_fraud))
    
    for t in timeunits:
        for s_id in sales_ids:
            row = {
                'sales_id': s_id,
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
    df_long = df.melt(id_vars=["sales_id", "timeunit", "flag_fraud"],
                      value_vars=rule_cols,
                      var_name="rule",
                      value_name="value")

    # Append timeunit to rule name
    df_long["rule_time"] = df_long["rule"] + "_" + df_long["timeunit"]

    # Pivot to wide format
    df_wide = df_long.pivot(
        index=["sales_id", "flag_fraud"],
        columns="rule_time",
        values="value"
    )

    # Flatten the columns (remove MultiIndex)
    df_wide.columns = df_wide.columns.tolist()

    # Bring index back as columns
    df_wide = df_wide.reset_index()

    return df_wide

@timer    
def subset_by_timeunit(df, timeunits,keep_cols_list=["sales_id", "flag_fraud"]):
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
def split_train_test(df, pk="sales_id", target="flag_fraud", test_size=0.2, random_state=42):
    """
    Split the dataframe into train/test sets (80-20) stratified by target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing pk, target, and features.
    pk : str
        Primary key column name.
    target : str
        Target column name (flag_fraud).
    test_size : float
        Proportion of data to use as test set.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
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
                                target="flag_fraud",
                                val_size=0.2, 
                                random_state=42, 
                                scale=False):
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
    """
    # Filter only non-fraud for training and validation
    train_nonfraud = train_df[train_df[target] == 0]

    # Split non-fraud training data into pure train and validation sets
    pure_train_nonfraud, val_nonfraud = train_test_split(
        train_nonfraud,
        test_size=val_size,
        random_state=random_state,
        stratify=train_nonfraud[target]  # stratify even if all zeros for consistency
    )

    pure_train_nonfraud = pure_train_nonfraud.reset_index(drop=True)
    val_nonfraud = val_nonfraud.reset_index(drop=True)

    # Separate features (drop primary key and target)
    X_train = pure_train_nonfraud.drop(columns=[pk, target])
    X_val = val_nonfraud.drop(columns=[pk, target])
    X_test = test_df.drop(columns=[pk, target])
    y_test = test_df[target].copy()

    # Optional scaling
    # zero mean, unit variance ((x - mean)/std). Works well for most cases.
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_val, X_test, y_test

class Autoencoder(nn.Module):
    """Autoencoder neural network with customizable architecture and optional dropout."""

    def __init__(self, input_dim, encoding_dim=4, hidden_layers=[8], 
                 activation=nn.ReLU, dropout=0.0, verbose=False):
        super(Autoencoder, self).__init__()
        self.verbose = verbose
        self.activation_cls = activation
        self.dropout = dropout

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

@timer
def train_autoencoder(model, 
                      X_train, 
                      X_val,
                      epochs=50, 
                      batch_size=32, 
                      lr=1e-3,
                      verbose=True):
    """
    Train autoencoder with explicit train and val sets (no internal splitting).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert to tensors
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # X_val = torch.tensor(X_val, dtype=torch.float32)
    X_train = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val.values if hasattr(X_val, "values") else X_val, dtype=torch.float32)


    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses

@timer
def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir, prefix="checkpoint"):
    """
    Save model checkpoint as .pkl artifact with training state.
    """
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    file_path = os.path.join(run_dir, f"{prefix}_epoch{epoch}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"✅ Saved checkpoint: {file_path}")
    
    
@timer
def load_checkpoint(file_path, model, optimizer):
    """
    Load model & optimizer state from checkpoint.
    Returns: start_epoch, train_losses, val_losses, run_dir
    """
    with open(file_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    model.load_state_dict(checkpoint_data["model_state"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state"])
    run_dir = os.path.dirname(file_path)

    print(f"🔄 Loaded checkpoint from {file_path} (epoch {checkpoint_data['epoch']})")

    return (checkpoint_data["epoch"], 
            checkpoint_data["train_losses"], 
            checkpoint_data["val_losses"],
            run_dir)


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
                                 verbose=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # Prepare data
    X_train = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val.values if hasattr(X_val, "values") else X_val, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False)

    # --- Flexible loss ---
    criterion = get_loss_function(loss_name)
    print(f"[INFO] Using loss function: {criterion.__class__.__name__}")
    
    # --- Flexible optimizer ---
    optimizer = get_optimizer(model, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)
    print(f"[INFO] Using optimizer: {optimizer.__class__.__name__}")

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
        os.makedirs(run_dir, exist_ok=True)

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

        # --- Save checkpoint ---
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir)

        # --- Early stopping ---
        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

    return model, train_losses, val_losses


@timer
def load_pickled_autoencoder(checkpoint_path, model=None, device=None):
    """
    Load a pickled autoencoder checkpoint saved with pickle.dump().
    """
    if device is None:
        device = torch.device('cpu')

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    train_losses = checkpoint.get("train_losses", None)
    val_losses = checkpoint.get("val_losses", None)
    last_epoch = checkpoint.get("epoch", None)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

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
# 1. Compute reconstruction error
def get_reconstruction_error(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, "values") else X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = model(X_test_tensor).cpu().numpy()

    recon_error = np.mean(( (X_test.values if hasattr(X_test, "values") else X_test) - X_pred) ** 2, axis=1)
    return recon_error

@timer
def plot_threshold_vs_metric_percentile(y_true, recon_error, percentiles=None, metric='accuracy'):
    """
    Plot predicted fraud count and chosen metric as threshold varies over percentiles.

    Parameters:
    - y_true: true labels (0=normal, 1=fraud)
    - recon_error: reconstruction errors
    - percentiles: list or np.array of percentiles to test (0-100). Default: 1 to 99
    - metric: one of ['accuracy', 'precision', 'recall', 'f0.5']

    Example:
    plot_threshold_vs_metric_percentile(y_test, recon_error, metric='precision')
    """
    if percentiles is None:
        percentiles = np.arange(1, 100)  # 1% to 99%

    thresholds = np.percentile(recon_error, percentiles)

    metric_values = []
    fraud_counts = []

    for thresh in thresholds:
        y_pred = (recon_error > thresh).astype(int)

        if metric == 'accuracy':
            val = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            val = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            val = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'f0.5':
            val = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric '{metric}'")

        metric_values.append(val)
        fraud_counts.append(y_pred.sum())

    fig, ax1 = plt.subplots(figsize=(10,6))

    color_count = 'tab:orange'
    ax1.set_xlabel('Threshold Percentile')
    ax1.set_ylabel('Count Predicted Fraud', color=color_count)
    ax1.bar(percentiles, fraud_counts, alpha=0.3, color=color_count)
    ax1.tick_params(axis='y', labelcolor=color_count)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color_metric = 'tab:blue'
    ax2.set_ylabel(metric.capitalize(), color=color_metric)
    ax2.plot(percentiles, metric_values, color=color_metric, label=metric.capitalize())
    ax2.tick_params(axis='y', labelcolor=color_metric)

    plt.title(f'Percentile Threshold vs Predicted Fraud Count and {metric.capitalize()}')
    fig.tight_layout()
    plt.show()


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
def detect_anomalies(model, X_test, y_test, threshold_quantile=95):
    '''
    reconstruction error — how well the autoencoder reconstructs (rebuilds) the input data.
    Calculated as mean squared error (MSE) between the original input and the reconstructed output for each sample.
    Low reconstruction error means the sample looks like normal training data → likely normal.
    High reconstruction error means the sample is not well reconstructed → likely anomaly/fraud.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, "values") else X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = model(X_test_tensor).cpu().numpy()

    recon_error = np.mean((X_test - X_pred) ** 2, axis=1)
    threshold = np.percentile(recon_error[y_test == 0], threshold_quantile)
    predicted_fraud = (recon_error > threshold).astype(int)

    print(f"Threshold (q={threshold_quantile}): {threshold:.4f}")
    print(classification_report(y_test, predicted_fraud))
    return recon_error, predicted_fraud

    
@timer
def tune_autoencoder_grid(X_train, X_val, param_grid, epochs=50, verbose=True):
    """
    Tune autoencoder hyperparameters using train and val data only.

    Select best model/config based on validation loss.

    Returns:
        best_model, best_config, best_val_loss
    """
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_val_loss = np.inf
    best_model = None
    best_config = None

    for i, cfg in enumerate(configs):
        if verbose:
            print(f"\nTesting config {i+1}/{len(configs)}: {cfg}")

        model = Autoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=cfg.get("encoding_dim", 4),
            hidden_layers=cfg.get("hidden_layers", [8]),
            activation=cfg.get("activation", nn.ReLU),
            verbose=False
        )

        model, train_losses, val_losses = train_autoencoder(
            model,
            X_train,
            X_val,
            epochs=epochs,
            batch_size=cfg.get("batch_size", 32),
            lr=cfg.get("lr", 1e-3),
            verbose=False
        )

        current_val_loss = val_losses[-1]  # or min(val_losses)

        if verbose:
            print(f"Config {i+1} final val loss: {current_val_loss:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = model
            best_config = cfg

    if verbose:
        print(f"\nBest config: {best_config}")
        print(f"Best validation loss: {best_val_loss:.4f}")

    return best_model, best_config, best_val_loss

@timer
def extract_embeddings(model, X):
    """
    Extract latent embeddings from the encoder of an Autoencoder.

    Parameters:
    - model: Autoencoder with `.encode()` method
    - X: np.array (input data)

    Returns:
    - embeddings: np.array of shape (n_samples, encoding_dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32).to(device)
    with torch.no_grad():
        embeddings = model.encode(X_tensor).cpu().numpy()

    return embeddings


@timer
def plot_latent_space(embeddings, 
                      y_test=None, 
                      y_pred=None, 
                      encoding_dim=2, 
                      label_names={0: "Normal", 1: "Anomaly"}):
    """
    Plot latent space (1D, 2D, or 3D) with optional ground truth and/or predicted labels.

    Parameters:
    - embeddings: np.array of shape (n_samples, encoding_dim)
    - y_test: np.array of ground truth labels (optional)
    - y_pred: np.array of predicted labels (optional)
    - encoding_dim: int (1, 2, or 3 supported)
    - label_names: dict, optional mapping of class labels for legend
    """

    def _scatter(ax, x, y=None, z=None, labels=None, title=None):
        c = labels if labels is not None else 'gray'
        if z is None:
            sc = ax.scatter(x[:, 0], x[:, 1] if x.shape[1] > 1 else [0]*len(x), c=c, cmap='coolwarm', alpha=0.6)
        else:
            sc = ax.scatter(x[:, 0], x[:, 1], z, c=c, cmap='coolwarm', alpha=0.6)
        ax.set_title(title)
        if encoding_dim >= 2:
            ax.set_xlabel("Latent dim 1")
            ax.set_ylabel("Latent dim 2")
        if encoding_dim == 3:
            ax.set_zlabel("Latent dim 3")
        return sc

    if encoding_dim not in [1, 2, 3]:
        print("Only 1D, 2D, or 3D latent spaces are supported. For higher dims, use t-SNE or PCA.")
        return

    if y_test is not None and y_pred is not None:
        # Show side-by-side plots for true vs predicted
        fig = plt.figure(figsize=(14, 5) if encoding_dim <= 2 else (16, 6))
        
        # First subplot: ground truth
        ax1 = fig.add_subplot(121, projection='3d' if encoding_dim == 3 else None)
        _scatter(ax1, embeddings, 
                 z=embeddings[:, 2] if encoding_dim == 3 else None, 
                 labels=y_test, 
                 title="Latent Space (Ground Truth)")

        # Second subplot: predicted labels
        ax2 = fig.add_subplot(122, projection='3d' if encoding_dim == 3 else None)
        sc = _scatter(ax2, embeddings, 
                      z=embeddings[:, 2] if encoding_dim == 3 else None, 
                      labels=y_pred, 
                      title="Latent Space (Predicted)")
        
        # Color bar only once
        cbar = fig.colorbar(sc, ax=[ax1, ax2], shrink=0.8)
        cbar.set_label('Label')
        plt.show()

    else:
        # Single plot: either y_test or y_pred or no label
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d' if encoding_dim == 3 else None)
        sc = _scatter(ax, embeddings, 
                      z=embeddings[:, 2] if encoding_dim == 3 else None,
                      labels=y_test if y_test is not None else y_pred, 
                      title="Latent Space")
        fig.colorbar(sc, label="Label")
        plt.show()

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

    color_map = {0: 'blue', 1: 'red'}
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
                    marker=dict(color=color, size=marker_size),
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
                        subplot_titles=("Latent Space (Ground Truth)", "Latent Space (Predicted)"))
    _add_scatter(fig, y_test, row=1, col=1, first_occurrence=True)
    _add_scatter(fig, y_pred, row=1, col=2, first_occurrence=False)

    fig.update_layout(height=500, width=1000)
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

    color_map = {0: 'blue', 1: 'red'}
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
                    marker=dict(color=color, size=marker_size),
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
                        subplot_titles=("Latent Space (Ground Truth)", "Latent Space (Predicted)"))
    _add_scatter3d(fig, y_test, row=1, col=1, first_occurrence=True)
    _add_scatter3d(fig, y_pred, row=1, col=2, first_occurrence=False)
    fig.update_layout(height=500, width=1000)
    fig.show()


        
@timer
def create_pack_results(test_df, y_pred, experiment_name, id_col='sales_id'
                        # ,date_col = 'expected_dt'
                       ):
    """
    Create a wide-format DataFrame with sales_id and one experiment column for predictions.

    Parameters:
    - test_df: pd.DataFrame with the test data (to get sales_id)
    - y_pred: array-like predicted labels
    - experiment_name: str, the column name for predictions (e.g., 'ae_1')
    - id_col: str, name of the ID column in test_df (default 'sales_id')

    Returns:
    - pd.DataFrame with columns ['sales_id', experiment_name]
    """
    df_results = pd.DataFrame({
        id_col: test_df[id_col].values,
        # date_col:test_df[date_col].values,
        experiment_name: y_pred
    })
    return df_results
    
@timer
def append_experiment_results(base_df, y_pred, experiment_name, id_col='sales_id'):
    """
    Append a new experiment's predictions as a column to an existing results DataFrame.

    Parameters:
    - base_df: pd.DataFrame with existing results, must have `id_col`
    - y_pred: array-like predicted labels for the new experiment
    - experiment_name: str, new column name (e.g., 'ae_2')
    - id_col: str, name of the ID column

    Returns:
    - pd.DataFrame updated with new experiment predictions column
    """
    new_df = pd.DataFrame({
        id_col: base_df[id_col].values,
        experiment_name: y_pred
    })

    # Merge on ID to keep existing columns and add new one
    merged_df = base_df.merge(new_df, on=id_col)
    return merged_df


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
def evaluate_thresholds(x_scaled, test_df,y_test, recon_error, true_fraud_list, 
                        exp_name ='default_all' ,               
                        percentiles=None,
                       ):
    if percentiles is None:
        percentiles = [10,20,30,40,50,60,70,80,85,90,95,97,99]

    df_lables_all = pd.DataFrame(index=test_df['sales_id'])
    for p in percentiles:
        # threshold = np.percentile(recon_error, p)
        threshold = np.percentile(recon_error[y_test == 0], p)
        y_pred = flag_anomalies(recon_error, threshold)

        experiment_name = 'ae_' + exp_name + f'_p{p}'
        df_labels_tmp = create_pack_results(test_df, y_pred, experiment_name=experiment_name)
        df_labels_tmp = df_labels_tmp.set_index('sales_id')

        df_lables_all[experiment_name] = df_labels_tmp[experiment_name]

    # Run evaluation on all thresholds at once
    results_df = evaluate_fraud_predictions(
        x_scaled=x_scaled,
        df_lables=df_lables_all.copy(),
        true_fraud_list=true_fraud_list
    )

    return results_df

################### Pipeline ########################
# ----------------- Function 1: Training Pipeline -----------------
@timer
def run_training_pipeline(
    model,
    X_scaled_train,
    X_scaled_val,
    X_scaled_test,
    test_df,
    y_test,
    true_fraud_list=None,
    seed: int = 42,
    epochs: int = 20,
    checkpoint_every: int = 10,
    early_stopping_patience: int = 10,
    run_dir: str = "model_checkpoint",
    loss_name: str = "mae",
    optimizer_name: str = "adam"
):
    """
    Train autoencoder, plot learning curve, create initial results, and evaluate thresholds.
    Returns: trained_model, train_losses, val_losses, df_lables, res_thresholds
    """
    set_seed(seed)

    trained_model, train_losses, val_losses = train_autoencoder_checkpoint(
        model=model,
        X_train=X_scaled_train,
        X_val=X_scaled_val,
        epochs=epochs,
        checkpoint_every=checkpoint_every,
        early_stopping_patience=early_stopping_patience,
        run_dir=run_dir,
        loss_name=loss_name,
        optimizer_name=optimizer_name
    )

    plot_learning_curve(train_losses, val_losses)

    y_pred_dummy = np.zeros_like(y_test)
    df_lables = create_pack_results(test_df, y_pred_dummy, experiment_name='ae_all_1')
    df_lables = df_lables.set_index('sales_id')
    
    recon_error = get_reconstruction_error(trained_model, X_scaled_test)
    
    res_thresholds = evaluate_thresholds(
        x_scaled=X_scaled_test,
        test_df=test_df,
        y_test=y_test,
        recon_error=recon_error,
        true_fraud_list=true_fraud_list
    )

    return trained_model, train_losses, val_losses, df_lables, res_thresholds


# ----------------- Function 2: Threshold & Plot Pipeline -----------------
@timer
def run_threshold_and_plot_pipeline(
    trained_model,
    X_scaled_test,
    test_df,
    y_test,
    threshold_percentile: float = 95,
    hover_col: str = 'sales_id'
):
    """
    Compute reconstruction error, select threshold, flag anomalies, extract embeddings,
    and plot 2D and 3D latent space.
    Returns: recon_error, threshold, y_pred, embeddings
    """
    # 1. Compute reconstruction error
    recon_error = get_reconstruction_error(trained_model, X_scaled_test)

    # 2. Compute threshold manually (percentile on normal data)
    threshold = np.percentile(recon_error[y_test == 0], threshold_percentile)
    print(f"Selected threshold ({threshold_percentile} percentile): {threshold:.4f}")

    # 3. Flag anomalies
    y_pred = flag_anomalies(recon_error, threshold)

    # 4. Extract embeddings
    embeddings = extract_embeddings(trained_model, X_scaled_test)

    # 5. Plot latent spaces
    plot_latent_space_2d(
        embeddings=embeddings,
        y_test=y_test,
        y_pred=y_pred,
        index_df=test_df,
        hover_col=hover_col
    )
    plot_latent_space_3d(
        embeddings=embeddings,
        y_test=y_test,
        y_pred=y_pred,
        index_df=test_df,
        hover_col=hover_col
    )

    return recon_error, threshold, y_pred, embeddings
