#### pending!!!!!!!!!!!!!
import torch
import torch.nn as nn

class VAE(nn.Module):
    """Variational Autoencoder with customizable architecture and dropout."""
    
    def __init__(self, input_dim, hidden_layers=[16,8], latent_dim=4, 
                 activation=nn.ReLU, dropout=0.0):
        super(VAE, self).__init__()
        self.activation_cls = activation
        self.dropout = dropout

        # Encoder
        encoder = []
        prev_dim = input_dim
        for h in hidden_layers:
            encoder.append(nn.Linear(prev_dim, h))
            encoder.append(activation())
            if dropout > 0:
                encoder.append(nn.Dropout(dropout))
            prev_dim = h
        self.encoder = nn.Sequential(*encoder)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder = []
        prev_dim = latent_dim
        for h in reversed(hidden_layers):
            decoder.append(nn.Linear(prev_dim, h))
            decoder.append(activation())
            if dropout > 0:
                decoder.append(nn.Dropout(dropout))
            prev_dim = h
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

import torch
import torch.nn as nn

def vae_loss(x, x_recon, mu, logvar, reconstruction_loss_fn=nn.MSELoss()):
    recon_loss = reconstruction_loss_fn(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss


import torch
import os

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_losses": train_losses,
        "val_losses": val_losses
    }, path)

def load_checkpoint(path, model=None, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("train_losses", []), checkpoint.get("val_losses", []), os.path.dirname(path)


import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from checkpoint import save_checkpoint, load_checkpoint
from vae_loss import vae_loss

def train_vae(model, X_train, X_val,
              epochs=50, batch_size=32, lr=1e-3,
              checkpoint_every=100, checkpoint_path=None,
              run_dir=None, early_stopping_patience=None,
              weight_decay=0.0, verbose=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare data
    X_train = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val.values if hasattr(X_val, "values") else X_val, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 0
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, train_losses, val_losses, run_dir = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
    else:
        if run_dir is None:
            run_dir = f"vae_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            run_dir = os.path.join(run_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = vae_loss(batch, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_recon, mu, logvar = model(batch)
                loss = vae_loss(batch, x_recon, mu, logvar)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses, run_dir)

        # Early stopping
        if early_stopping_patience:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return model, train_losses, val_losses

import torch
from timer import timer  # assuming you already have a timer decorator

@timer
def extract_embeddings_vae(model, X):
    """
    Extract mean latent embeddings from a VAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        embeddings = mu.cpu().numpy()
    return embeddings

'''
from vae_model import VAE
from vae_train import train_vae
from vae_extract_embeddings import extract_embeddings_vae

# Define VAE
model = VAE(input_dim=20, hidden_layers=[16,8], latent_dim=4, dropout=0.2)

# Train
model, train_losses, val_losses = train_vae(
    model, X_train, X_val,
    epochs=100, batch_size=64,
    early_stopping_patience=10,
    weight_decay=1e-4
)

# Extract embeddings
embeddings = extract_embeddings_vae(model, X_train)

'''