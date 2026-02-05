import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error

class SpectraDataset(Dataset):
    def __init__(self, spectra, baselines=None):
        # spectra: (N, L) --> baselines: (N, L)
        self.spectra   = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)   # (N, 1, L)
        self.baselines = (torch.tensor(baselines, dtype=torch.float32).unsqueeze(1) # (N, 1, L)
                          if baselines is not None else None)                    

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        x = self.spectra[idx]
        y = self.baselines[idx] if self.baselines is not None else x
        return x, y

# Save model and plots
MODEL_SAVE_DIR = 'unet_trained_no_cbam'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Load data
train_np = np.load("/vol/csedu-nobackup/project/jlandsman/data/train_data.npz")
val_np   = np.load("/vol/csedu-nobackup/project/jlandsman/data/val_data.npz")
test_np  = np.load("/vol/csedu-nobackup/project/jlandsman/data/test_data.npz")

lntrans_train, lnbase_train = train_np["inputs_lntrans_1"], train_np["inputs_lnbase_1"]
lntrans_val,   lnbase_val   = val_np  ["inputs_lntrans_1"], val_np  ["inputs_lnbase_1"]
lntrans_test,  lnbase_test  = test_np ["inputs_lntrans_1"], test_np ["inputs_lnbase_1"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets & Loaders
train_ds = SpectraDataset(lntrans_train, lnbase_train)
val_ds   = SpectraDataset(lntrans_val,   lnbase_val)
test_ds  = SpectraDataset(lntrans_test,  lnbase_test)

batch_size = 128
base_train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=4
)
base_val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    pin_memory=True, num_workers=4
)
base_test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False,
    pin_memory=True, num_workers=4
)

# U-net model
class ChannelAttention1D(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_pool = x.mean(dim=2, keepdim=True)       # (B, C, 1)
        max_pool, _ = x.max(dim=2, keepdim=True)     # (B, C, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return self.sigmoid(attn) * x

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_pool = x.mean(dim=1, keepdim=True)       # (B, 1, L)
        max_pool, _ = x.max(dim=1, keepdim=True)     # (B, 1, L)
        concat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, L)
        attn = self.sigmoid(self.conv(concat))
        return attn * x

class CBAM1D(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention1D(channels, ratio)
        self.spatial_att = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, ks=3, pad=1):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = nn.Conv1d(in_ch, mid_ch, ks, padding=pad, dilation=1)
        self.bn1   = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, out_ch, ks, padding=3, dilation=3)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) # Align dimensions
        )
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class CBAMResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, ks=3, pad=1, ratio=16):
        super().__init__()
        self.res = ResBlock1D(in_ch, out_ch, mid_ch, ks, pad)
        self.cbam = CBAM1D(out_ch, ratio)

    def forward(self, x):
        out = self.res(x)
        out = self.cbam(out)
        return out

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool1d(2),
            ResBlock1D(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = ResBlock1D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(2) - x1.size(2)
        if diff != 0:
            x1 = F.pad(x1, [diff // 2, diff - diff//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetBaseline(nn.Module):
    # (B, 1, 3600) --> (B, 1, 3600)
    def __init__(self, n_channels=1, n_out_channels=1, base_ch=32):
        super().__init__()
        self.inc   = ResBlock1D(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*16)
        self.up1   = Up(base_ch*16, base_ch*8)
        self.up2   = Up(base_ch*8,  base_ch*4)
        self.up3   = Up(base_ch*4,  base_ch*2)
        self.up4   = Up(base_ch*2,  base_ch)
        self.outc  = OutConv(base_ch, n_out_channels)

    def forward(self, x):
        x1 = self.inc(x)         # (B, base_ch, 3600)
        x2 = self.down1(x1)      # (B, base_ch*2, 1800)
        x3 = self.down2(x2)      # (B, base_ch*4, 900)
        x4 = self.down3(x3)      # (B, base_ch*8, 450)
        x5 = self.down4(x4)      # (B, base_ch*16, 225)
        x  = self.up1(x5, x4)    # (B, base_ch*8, 450)
        x  = self.up2(x, x3)     # (B, base_ch*4,  900)
        x  = self.up3(x, x2)     # (B, base_ch*2, 1800)
        x  = self.up4(x, x1)     # (B, base_ch,   3600)
        return self.outc(x)      # (B, 1, 3600)

BASE_CHANNELS = 32
model     = UNetBaseline(n_channels=1, n_out_channels=1, base_ch=BASE_CHANNELS).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.3, patience=10, verbose=True
)

EPOCHS = 250
early_stop_patience = 25
epochs_no_improve   = 0
best_loss           = float('inf')

best_loss = float('inf')
history = {'train_loss':[], 'val_loss':[], 'test_loss':[], 'lr':[]}

for epoch in range(1, EPOCHS+1):
    # Training
    model.train()
    total_train_loss = 0.0
    for x_batch, y_batch in base_train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * x_batch.size(0)
    train_loss = total_train_loss / len(train_ds)
    history['train_loss'].append(train_loss)

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in base_val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            preds = model(x_batch)
            total_val_loss += criterion(preds, y_batch).item() * x_batch.size(0)
    val_loss = total_val_loss / len(val_ds)
    history['val_loss'].append(val_loss)

    # View test set evaluation every 10 epochs
    if epoch % 10 == 0:
        total_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in base_test_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                preds = model(x_batch)
                total_test_loss += criterion(preds, y_batch).item() * x_batch.size(0)
        test_loss = total_test_loss / len(test_ds)
        history['test_loss'].append(test_loss)
        print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}, Test={test_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.2e}")
    else:
        history['test_loss'].append(None)
        print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.2e}")

    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]['lr']
    history['lr'].append(lr)

    # Checkpointing
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "unet_weights.pth"))
        print("  Validation loss improved; model saved.")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print(f"\nEarly stopping triggered after {epoch} epochs without improvement.")
        break

# Plot losses (train/val/test)
n_epochs = len(history['train_loss'])
epochs = list(range(1, n_epochs+1))

train_vals = history['train_loss']
val_vals   = history['val_loss']

plt.figure(figsize=(10,6))
plt.plot(epochs, train_vals, label='Train Loss')
plt.plot(epochs, val_vals,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(os.path.join(MODEL_SAVE_DIR, "loss_curve.png"))
plt.close()

# Load model and eval on both validation and test sets
model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "unet_weights.pth"), map_location=device))
model.to(device).eval()

# Val metrics
all_preds_val = []
with torch.no_grad():
    for x_batch, y_batch in base_val_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        preds = model(x_batch)
        all_preds_val.append(preds.cpu().numpy())
preds_val_np = np.vstack(all_preds_val)       # (N_val, 1, L)
preds_val_np = preds_val_np.squeeze(1)        # (N_val, L)
true_val_np  = lnbase_val                     # (N_val, L)

r2_val  = r2_score(true_val_np, preds_val_np, multioutput='raw_values')
mse_val = mean_squared_error(true_val_np, preds_val_np, multioutput='raw_values')

print("=== Validation set metrics (per spectral point) ===")
for i, (r2_i, mse_i) in enumerate(zip(r2_val, mse_val)):
    print(f"Spectral index {i}: R²={r2_i:.4f}, MSE={mse_i:.4f}")

# Test metrics
all_preds_test = []
with torch.no_grad():
    for x_batch, y_batch in base_test_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        preds = model(x_batch)
        all_preds_test.append(preds.cpu().numpy())
preds_test_np = np.vstack(all_preds_test)     # (N_test, 1, L)
preds_test_np = preds_test_np.squeeze(1)      # (N_test, L)
true_test_np  = lnbase_test                    # (N_test, L)

r2_test  = r2_score(true_test_np, preds_test_np, multioutput='raw_values')
mse_test = mean_squared_error(true_test_np, preds_test_np, multioutput='raw_values')

print("\nTest set metrics (per spectral point)")
for i, (r2_i, mse_i) in enumerate(zip(r2_test, mse_test)):
    print(f"Spectral index {i}: R²={r2_i:.4f}, MSE={mse_i:.4f}")