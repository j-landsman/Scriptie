import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

#--------- Sanity Check Model Definition ---------#
# This is the core of your MLP, but adapted to take a single channel.
# We are testing if this part of your architecture can learn from ideal data.
NUM_GASES = 11

class SanityCheckMLP(nn.Module):    
    def __init__(self, n_gases=NUM_GASES):
        super().__init__()
        # This model takes a single-channel spectrum
        self.flatten = nn.Flatten()
        
        # This is the exact same MLP structure you were using after the fusion layer
        self.sp1_conv = nn.Sequential(
            nn.Linear(3600, 512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_gases)
        )

    def forward(self, x):
        # x is expected to have shape (B, 1, 3600)
        x_flat = self.flatten(x)
        x_processed = self.sp1_conv(x_flat)
        return self.mlp(x_processed)

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
    n_gases = y_true.shape[1]
    nrmse_list = []
    for i in range(n_gases):
        mse = np.mean((y_true[:,i] - y_pred[:,i])**2)
        rmse = np.sqrt(mse)
        data_range = y_true[:,i].max() - y_true[:,i].min()
        nrmse = rmse / data_range if data_range > 0 else np.nan
        nrmse_list.append(nrmse)
    return nrmse_list

#--------- Main Training Script ---------#
def main():
    print("--- Running Sanity Check Script ---")
    print("Goal: Train a simple MLP on single-channel, perfectly clean data.")
    
    # Paths (identical to your script)
    DATA_DIR   = "/vol/csedu-nobackup/project/jlandsman/data"
    TRAIN_FILE1 = os.path.join(DATA_DIR, "train_data_no_drift.npz")
    VAL_FILE1   = os.path.join(DATA_DIR, "val_data_no_drift.npz")
    TRAIN_FILE2 = os.path.join(DATA_DIR, "train_data_no_drift2.npz")
    VAL_FILE2   = os.path.join(DATA_DIR, "val_data_no_drift2.npz")
    TEST_FILE  = os.path.join(DATA_DIR, "test_data_no_drift2.npz")
    OUTPUT_DIR = "mlp_pretrained/sanity_check" # Use a new output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Nitrous oxide', 'Ethylene', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol']

    # --- Load and prepare CLEAN data only ---
    tr1 = np.load(TRAIN_FILE1)
    X_train_clean1 = tr1["inputs_lntrans_1"].astype(np.float32) - tr1["inputs_lnbase_1"].astype(np.float32)
    Y_train1 = np.delete(tr1["c"], 7, axis=1).astype(np.float32)

    tr2 = np.load(TRAIN_FILE2)
    X_train_clean2 = tr2["inputs_lntrans_1"].astype(np.float32) - tr2["inputs_lnbase_1"].astype(np.float32)
    Y_train2 = np.delete(tr2["c"], 7, axis=1).astype(np.float32)

    X_train_clean = np.concatenate([X_train_clean1, X_train_clean2], axis=0)
    Y_train = np.concatenate([Y_train1, Y_train2], axis=0)

    va1 = np.load(VAL_FILE1)
    X_val_clean1 = va1["inputs_lntrans_1"].astype(np.float32) - va1["inputs_lnbase_1"].astype(np.float32)
    Y_val1 = np.delete(va1["c"], 7, axis=1).astype(np.float32)

    va2 = np.load(VAL_FILE2)
    X_val_clean2 = va2["inputs_lntrans_1"].astype(np.float32) - va2["inputs_lnbase_1"].astype(np.float32)
    Y_val2 = np.delete(va2["c"], 7, axis=1).astype(np.float32)

    X_val_clean = np.concatenate([X_val_clean1, X_val_clean2], axis=0)
    Y_val = np.concatenate([Y_val1, Y_val2], axis=0)

    te = np.load(TEST_FILE)
    X_test_clean = te["inputs_lntrans_1"].astype(np.float32) - te["inputs_lnbase_1"].astype(np.float32)
    Y_test = np.delete(te["c"], 7, axis=1).astype(np.float32)

    # --- Scaling ---
    x_scaler = MinMaxScaler(feature_range=(0,1), clip=True)
    x_scaler.fit(X_train_clean)
    y_scaler = MinMaxScaler().fit(Y_train)

    X_train_s = x_scaler.transform(X_train_clean)
    X_val_s   = x_scaler.transform(X_val_clean)
    X_test_s  = x_scaler.transform(X_test_clean)

    Y_train_scaled = y_scaler.transform(Y_train)
    Y_val_scaled   = y_scaler.transform(Y_val)

    # --- Add channel dimension for the model ---
    # The model's Flatten layer expects a shape like (B, Channels, Length)
    X_train = X_train_s[:, np.newaxis, :]
    X_val   = X_val_s[:, np.newaxis, :]
    X_test  = X_test_s[:, np.newaxis, :]

    print(f"Sanity check training data shape: {X_train.shape}")

    # Datasets & Loaders
    train_ds   = SpectraDataset(X_train, Y_train_scaled)
    val_ds     = SpectraDataset(X_val,   Y_val_scaled)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, loss, optimizer, scheduler
    model = SanityCheckMLP().to(device)
    criterion = nn.MSELoss()
    # Using a safer, lower learning rate as recommended for debugging
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop (identical to yours for fair comparison)
    epochs = 200
    early_stop_patience = 35
    epochs_no_improve   = 0
    best_loss           = float('inf')

    train_losses = []
    val_losses   = []
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                running_val_loss += criterion(preds, yb).item() * xb.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        # Checkpointing
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "sanity_check_mlp_weights.pth"))
            print("  Validation loss improved; model saved.")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs without improvement.")
                break

    # Plot losses
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sanity_check_loss_curve.png'))
    plt.close()

    # --- Final Evaluation ---
    print("\n--- Sanity Check Final Performance ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "sanity_check_mlp_weights.pth")))
    model.eval()
    
    # Validation Set
    with torch.no_grad():
        y_val_pred_scaled = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
    r2_val   = r2_score(Y_val, y_val_pred, multioutput='raw_values')
    nrmse_val = compute_nrmse(Y_val, y_val_pred)
    print("\nValidation Set Performance:")
    for i, (r2_i, nrmse_i) in enumerate(zip(r2_val, nrmse_val)):
        print(f"  {labels[i]:<15}: R² = {r2_i:.4f}, NRMSE = {nrmse_i:.4f}")

    # Test Set
    with torch.no_grad():
        y_test_pred_scaled = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)
    r2_test   = r2_score(Y_test, y_test_pred, multioutput='raw_values')
    nrmse_test = compute_nrmse(Y_test, y_test_pred)
    print("\nTest Set Performance:")
    for i, (r2_i, nrmse_i) in enumerate(zip(r2_test, nrmse_test)):
        print(f"  {labels[i]:<15}: R² = {r2_i:.4f}, NRMSE = {nrmse_i:.4f}")

if __name__ == '__main__':
    main()