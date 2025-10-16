import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from model import Autoencoder

def main():
    # Seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic data
    normal = np.random.normal(0, 1, (1000, 5))  # 5 metrics
    anom   = np.random.normal(5, 1, (50, 5))
    X = np.vstack([normal, anom])
    y = np.array([0]*1000 + [1]*50)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, _, _ = train_test_split(Xs, y, test_size=0.2, random_state=42)

    # DataLoader
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model     = Autoencoder(input_dim=10, hidden_dims=[128,64], latent_dim=16, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    epochs = 50
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch, in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:02d}/{epochs} — Loss: {avg:.4f}")

    # Save model weights and scaler
    torch.save(model.state_dict(), "model_weights.pth")
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)

if __name__ == "__main__":
    main()
