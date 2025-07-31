import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D  # Only used if 3D

# 1. Generate mock data
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


# 2. Subset and prepare data
def prepare_data(df, timeunit='l3', scale=True):
    """
    Subset the data by timeunit and optionally scale it.
    
    Parameters:
    - df: DataFrame
    - timeunit: str, e.g. 'l3'
    - scale: bool, whether to apply StandardScaler
    
    Returns:
    - X_train: training data (non-fraud only)
    - X_test: full test set
    - y_test: labels for test set
    - scaler: fitted scaler or None
    """
    subset_df = df[df['timeunit'] == timeunit].reset_index(drop=True)
    features = [col for col in subset_df.columns if col.startswith('rule_')]
    X = subset_df[features].values
    y = subset_df['flag_fraud'].values

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    X_train = X_scaled[y == 0]
    X_test = X_scaled
    y_test = y
    return X_train, X_test, y_test, scaler


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


class Autoencoder(nn.Module):
    """Autoencoder neural network with customizable architecture.

    This class implements a symmetric autoencoder in PyTorch,
    which compresses input data to a latent space (encoder)
    and reconstructs it back (decoder). Useful for dimensionality
    reduction, anomaly detection, and unsupervised learning.

    Attributes:
        encoder (nn.Sequential): Encoder layers.
        decoder (nn.Sequential): Decoder layers.
        verbose (bool): If True, prints tensor shapes during forward pass.
    """

    def __init__(self, input_dim, encoding_dim=4, hidden_layers=[8], activation=nn.ReLU, verbose=False):
        """
        Initializes the autoencoder.

        Args:
            input_dim (int): Number of features in the input.
            encoding_dim (int, optional): Size of the latent (compressed) space. Defaults to 4.
            hidden_layers (list of int, optional): Sizes of hidden layers. Same structure used for encoder and decoder (reversed). Defaults to [8].
            activation (nn.Module, optional): Activation function class (e.g., nn.ReLU, nn.Tanh). Defaults to nn.ReLU.
            verbose (bool, optional): Whether to print shapes during forward pass. Useful for debugging. Defaults to False.
        """
        super(Autoencoder, self).__init__()
        self.verbose = verbose
        self.activation_cls = activation

        # Encoder
        encoder = []
        prev_dim = input_dim
        for idx, h in enumerate(hidden_layers):
            encoder.append(nn.Linear(prev_dim, h))
            encoder.append(activation())
            prev_dim = h
        encoder.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = []
        prev_dim = encoding_dim
        for idx, h in enumerate(reversed(hidden_layers)):
            decoder.append(nn.Linear(prev_dim, h))
            decoder.append(activation())
            prev_dim = h
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        """Runs a forward pass through the encoder and decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Reconstructed tensor of the same shape as input.
        """
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
        """Encodes input into the latent space representation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Latent representation of shape (batch_size, encoding_dim).
        """
        if self.verbose:
            print(f"Input to encoder: {x.shape}")
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.verbose:
                print(f"Encoder layer {i}: {x.shape}")
        if self.verbose:
            print(f"Latent embedding: {x.shape}")
        return x


def train_autoencoder(model, 
                      X_train, 
                      epochs=50, 
                      batch_size=32, 
                      validation_split=0.1, 
                      lr=1e-3,
                      verbose=True):
    """
    Trains the PyTorch autoencoder on non-anomalous data.

    Parameters:
    - model: Autoencoder instance
    - X_train: np.array or torch.Tensor (normal data only)
    - epochs: int
    - batch_size: int
    - validation_split: float (e.g., 0.1)
    - lr: learning rate
    - verbose: show training progress

    Returns:
    - model: trained autoencoder
    - train_losses: list of training losses per epoch
    - val_losses: list of validation losses per epoch

    Note :
    Why do we train on only normal data?
        Because autoencoders learn to reconstruct what they see often.
        If it learns normal patterns:
        Low reconstruction error → likely normal
        High reconstruction error → likely anomaly
    """
    # PyTorch will throw an error if the model is on the CPU but the data is on the GPU (or vice versa).
    # PyTorch requires both the model and the data to be on the same device during training and inference.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert X_train to tensor object
    X_train = torch.tensor(X_train, dtype=torch.float32) 
    
    # Split into train/val
    # Training set : Learn patterns and reconstruct input
    # Validation set : Ensure generalization and detect overfitting. 
    # **Generalization is the ability of a model to perform well on new, unseen data 
    #     not just on the data it was trained on.
    val_size = int(len(X_train) * validation_split)
    train_size = len(X_train) - val_size
    train_data, val_data = torch.utils.data.random_split(X_train, [train_size, val_size])
    
    # DataLoader batches data into manageable chunks for training.
    # Shuffles training data each epoch to improve model learning and avoid bias.
    # Makes iteration easy with a simple for batch in loader loop.
    # Does not shuffle validation data to keep evaluation consistent.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    # Define loss function & optimizer
    # MSE Loss is used since autoencoders try to reconstruct inputs.
    # Adam Optimizer updates the weights efficiently.    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Lists to store losses for plotting learning curve
    train_losses = []
    val_losses = []

    # Training loop updates the model to reduce reconstruction error on training data.
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to GPU/CPU as needed
            optimizer.zero_grad()  # Clear old gradients from previous step
            output = model(batch)  # Forward pass: compute reconstruction
            loss = criterion(output, batch)  # Calculate MSE loss between output and input
            loss.backward() # Backpropagate the error to compute gradients.
            optimizer.step()  # Update weights based on gradients
            train_loss += loss.item() # Accumulate batch loss for reporting

        # Validation checks how well the model generalizes to unseen data, helping to detect overfitting.
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation (saves memory & speeds up computations)
            for batch in val_loader:
                batch = batch.to(device)  # Move batch data to GPU/CPU as appropriate
                output = model(batch)  # Forward pass: reconstruct input
                loss = criterion(output, batch)  # Calculate reconstruction loss
                val_loss += loss.item() # Accumulate batch loss for reporting

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses


def plot_learning_curve(train_losses, val_losses):
    """
    Plot training and validation loss curves.

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()




def detect_anomalies(model, X_test, y_test, threshold_quantile=95):
    """
    Detect anomalies based on reconstruction error from the trained model.

    Parameters:
    - model: trained PyTorch autoencoder
    - X_test: np.array
    - y_test: np.array
    - threshold_quantile: e.g., 95 for top 5% error to be flagged

    Returns:
    - recon_error: np.array of errors
    - predicted_fraud: np.array of binary predictions (1=fraud)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = model(X_test_tensor).cpu().numpy()

    recon_error = np.mean((X_test - X_pred) ** 2, axis=1)
    threshold = np.percentile(recon_error[y_test == 0], threshold_quantile)
    predicted_fraud = (recon_error > threshold).astype(int)

    print(f"Threshold (q={threshold_quantile}): {threshold:.4f}")
    print(classification_report(y_test, predicted_fraud))
    return recon_error, predicted_fraud


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

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        embeddings = model.encode(X_tensor).cpu().numpy()

    return embeddings



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

