import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

#--------- Model Definition ---------#
class GasConcentrationMLP(nn.Module):
    def __init__(self, input_features=3600, hidden_layers=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # single-output
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#--------- Dataset ---------#
class SpectraDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#--------- Utility Functions ---------#
def compute_nrmse(y_true, y_pred):
    # NRMSE: RMSE divided by range
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    data_range = y_true.max() - y_true.min()
    return rmse / data_range if data_range > 0 else np.nan

#--------- Main Training Script ---------#
def main():
    # Paths
    DATA_DIR   = "/vol/csedu-nobackup/project/jlandsman/data"
    TRAIN_FILE1 = os.path.join(DATA_DIR, "train_data_no_drift.npz")
    VAL_FILE1   = os.path.join(DATA_DIR, "val_data_no_drift.npz")
    TRAIN_FILE2 = os.path.join(DATA_DIR, "train_data_no_drift2.npz")
    VAL_FILE2   = os.path.join(DATA_DIR, "val_data_no_drift2.npz")
    TEST_FILE  = os.path.join(DATA_DIR, "test_data_no_drift2.npz")

    labels = [
        'Water', 'Carbon Dioxide', 'Methane', 'Ammonia',
        'Methanol', 'Nitrous oxide', 'Ethylene', 'CO',
        'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol'
    ]

    # ---- Load all data once ----
    def load_split(npz_path):
        arr = np.load(npz_path)
        return arr["inputs_lntrans_1"].astype(np.float32), arr["c"].astype(np.float32)

    X1_tr, C1_tr = load_split(TRAIN_FILE1)
    X2_tr, C2_tr = load_split(TRAIN_FILE2)
    X_train = np.concatenate([X1_tr, X2_tr], axis=0)
    C_train = np.concatenate([C1_tr, C2_tr], axis=0)

    X1_va, C1_va = load_split(VAL_FILE1)
    X2_va, C2_va = load_split(VAL_FILE2)
    X_val = np.concatenate([X1_va, X2_va], axis=0)
    C_val = np.concatenate([C1_va, C2_va], axis=0)

    X_test, C_test = load_split(TEST_FILE)

    input_scaler = MinMaxScaler()
    X_train = input_scaler.fit_transform(X_train)
    X_val   = input_scaler.transform(X_val)
    X_test  = input_scaler.transform(X_test)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For each gas, train its own model
    for gas_idx, gas_name in enumerate(labels):
        if gas_idx!=7:
            OUTPUT_DIR = f"mlp_pretrained_sep/{gas_name}"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"\nTraining model for: {gas_name}")

            # Prepare Y for this gas (shape [N,1])
            Y_train = C_train[:, gas_idx].reshape(-1,1)
            Y_val   = C_val[:,   gas_idx].reshape(-1,1)
            Y_test  = C_test[:,  gas_idx].reshape(-1,1)

            # Datasets & Loaders
            train_ds   = SpectraDataset(X_train, Y_train)
            val_ds     = SpectraDataset(X_val,   Y_val)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

            # Build model, loss, optimizer, scheduler
            model     = GasConcentrationMLP(input_features=X_train.shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )

            # Training with early stopping
            epochs = 200
            patience = 35
            best_val_loss = float('inf')
            no_improve = 0
            train_losses, val_losses = [], []

            for epoch in range(1, epochs+1):
                # -- train --
                model.train()
                total_train = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_train += loss.item() * xb.size(0)
                train_loss = total_train / len(train_loader.dataset)
                train_losses.append(train_loss)

                # -- validate --
                model.eval()
                total_val = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        total_val += criterion(model(xb), yb).item() * xb.size(0)
                val_loss = total_val / len(val_loader.dataset)
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                print(f"Epoch {epoch}/{epochs} — train: {train_loss:.6f}, val: {val_loss:.6f}")

                # checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(OUTPUT_DIR, f"mlp_weights_{gas_name}.pth")
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f" Early stopping after {epoch} epochs.")
                        break

            # Plot & save losses
            plt.figure()
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
            plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val')
            plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
            plt.close()

            # Final evaluation
            model.eval()
            with torch.no_grad():
                pred_val  = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
                pred_test = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

            r2_val   = r2_score(Y_val,  pred_val)
            nrmse_val = compute_nrmse(Y_val, pred_val)
            r2_test  = r2_score(Y_test, pred_test)
            nrmse_test = compute_nrmse(Y_test, pred_test)

            print(f"\n>> {gas_name} Results:")
            print(f"   Val —  R² = {r2_val:.4f}, NRMSE = {nrmse_val:.4f}")
            print(f"   Test — R² = {r2_test:.4f}, NRMSE = {nrmse_test:.4f}")

if __name__ == '__main__':
    main()