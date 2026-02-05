#!/usr/bin/env python3
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ----------------------------------------
# 1) Constants and hyperparameters
# ----------------------------------------
DATA_DIR = "/vol/csedu-nobackup/project/jlandsman/data"
TRAIN_FILE = os.path.join(DATA_DIR, "train_data_no_drift.npz")
VAL_FILE   = os.path.join(DATA_DIR, "val_data_no_drift.npz")

BATCH_SIZE   = 128
NUM_EPOCHS   = 60
LEARNING_RATE = 3e-4
STEP_SIZE    = 20     # scheduler step every 20 epochs
GAMMA        = 0.5    # lr *= gamma every step

LABELS = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol',
          'Nitrous oxide', 'Ethylene', 'Acetone', 'Isoprene',
          'Acetaldehyde', 'Ethanol']
NUM_GASES = len(LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ----------------------------------------
# 2) Model definitions
# ----------------------------------------
class GasConcentrationMLP1(nn.Module):
    def __init__(self, input_features=3600, num_gases=NUM_GASES, hidden_layers=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_gases))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GasConcentrationMLP2(nn.Module):
    def __init__(self, input_features=3600, num_gases=NUM_GASES, hidden_layers=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_gases))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GasConcentrationMLP3(nn.Module):
    def __init__(self, input_features=3600, num_gases=NUM_GASES, hidden_layers=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_gases))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ----------------------------------------
# 3) Load & scale data
# ----------------------------------------
def load_and_scale():
    # load
    tr = np.load(TRAIN_FILE)
    lntrans_train = tr["inputs_lntrans_1"].astype(np.float32)
    Y_train = np.delete(tr["c"], 7, axis=1).astype(np.float32)

    va = np.load(VAL_FILE)
    lntrans_val = va["inputs_lntrans_1"].astype(np.float32)
    Y_val = np.delete(va["c"], 7, axis=1).astype(np.float32)

    # scale X and Y to [0,1]
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()
    B, F = lntrans_train.shape

    lntrans_train_flat = lntrans_train.reshape(-1, F)
    lntrans_val_flat   = lntrans_val.reshape(-1, F)

    lntrans_train_scaled = x_scaler.fit_transform(lntrans_train_flat).reshape(B, F)
    lntrans_val_scaled   = x_scaler.transform(lntrans_val_flat).reshape(lntrans_val.shape)

    Y_train_scaled = y_scaler.fit_transform(Y_train)
    Y_val_scaled   = y_scaler.transform(Y_val)

    return (lntrans_train_scaled, Y_train_scaled,
            lntrans_val_scaled,   Y_val_scaled,
            y_scaler)

# ----------------------------------------
# 4) PyTorch DataLoader
# ----------------------------------------
def make_loaders(X_train, Y_train, X_val, Y_val):
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_ds   = TensorDataset(
        torch.from_numpy(X_val),   torch.from_numpy(Y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

# ----------------------------------------
# 5) Training & evaluation
# ----------------------------------------
def train_one(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    Y_true = []
    Y_pred = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb).cpu().numpy()
            Y_pred.append(out)
            Y_true.append(yb.numpy())
    Y_true = np.vstack(Y_true)
    Y_pred = np.vstack(Y_pred)
    return Y_true, Y_pred

def compute_metrics(Y_true, Y_pred, y_scaler):
    # inverse-scale
    Y_true_orig = y_scaler.inverse_transform(Y_true)
    Y_pred_orig = y_scaler.inverse_transform(Y_pred)
    results = {}
    for i, gas in enumerate(LABELS):
        y_t = Y_true_orig[:, i]
        y_p = Y_pred_orig[:, i]
        r2 = r2_score(y_t, y_p)
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        nrmse = rmse / (y_t.max() - y_t.min() + 1e-8)
        results[gas] = {"r2": r2, "nrmse": nrmse}
    return results

# ----------------------------------------
# 6) Main loop
# ----------------------------------------
if __name__ == "__main__":
    X_tr, Y_tr, X_va, Y_va, y_scaler = load_and_scale()
    train_loader, val_loader = make_loaders(X_tr, Y_tr, X_va, Y_va)

    models = {
        "MLP1": GasConcentrationMLP1().to(DEVICE),
        "MLP2": GasConcentrationMLP2().to(DEVICE),
        "MLP3": GasConcentrationMLP3().to(DEVICE),
    }

    criterion = nn.MSELoss()

    all_results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} for {NUM_EPOCHS} epochs ===")
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

        for epoch in range(1, NUM_EPOCHS+1):
            train_loss = train_one(model, train_loader, optimizer, criterion)
            scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f" Epoch {epoch:02d} — train loss: {train_loss:.5f}")

        print(" Evaluating on validation set...")
        Y_true, Y_pred = evaluate(model, val_loader)
        results = compute_metrics(Y_true, Y_pred, y_scaler)
        all_results[name] = results

    # ----------------------------------------
    # 7) Print a summary table
    # ----------------------------------------
    header = ["Model", "Gas", "R²", "NRMSE"]
    print("\n" + "-"*50)
    print(f"{header[0]:<6}   {header[1]:<16}   {header[2]:>6}   {header[3]:>6}")
    print("-"*50)
    for name, res in all_results.items():
        for gas, m in res.items():
            print(f"{name:<6} | {gas:<16} | {m['r2']:6.3f} | {m['nrmse']:6.3f}")
    print("-"*50)