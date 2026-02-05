from __future__ import annotations
import datetime, json
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics.functional import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import copy
from captum.attr import IntegratedGradients
import joblib

# region: Model Definitions (No changes needed here, except for the new MLP)

# ... [All U-Net related classes: ChannelAttention1D, SpatialAttention1D, CBAM1D, ResBlock1D, CBAMResBlock1D, Down, Up, OutConv, ResUNetBaseline] ...
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
        avg_pool = x.mean(dim=2, keepdim=True); max_pool, _ = x.max(dim=2, keepdim=True)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return self.sigmoid(attn) * x

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True); max_pool, _ = x.max(dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return attn * x

class CBAM1D(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__(); self.channel_att = ChannelAttention1D(channels, ratio); self.spatial_att = SpatialAttention1D(kernel_size)
    def forward(self, x):
        return self.spatial_att(self.channel_att(x))

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, ks=3, pad=1):
        super().__init__(); mid_ch = out_ch if mid_ch is None else mid_ch
        self.conv1 = nn.Conv1d(in_ch, mid_ch, ks, padding=pad, dilation=1); self.bn1 = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, out_ch, ks, padding=3, dilation=3); self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.skip(x); out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class CBAMResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, ks=3, pad=1, ratio=16):
        super().__init__(); self.res = ResBlock1D(in_ch, out_ch, mid_ch, ks, pad); self.cbam = CBAM1D(out_ch, ratio)
    def forward(self, x):
        return self.cbam(self.res(x))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.block = nn.Sequential(nn.MaxPool1d(2), CBAMResBlock1D(in_ch, out_ch))
    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.up = nn.ConvTranspose1d(in_ch, in_ch//2, kernel_size=2, stride=2); self.conv = CBAMResBlock1D(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1); diff = x2.size(2) - x1.size(2)
        if diff != 0: x1 = F.pad(x1, [diff // 2, diff - diff//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class ResUNetBaseline(nn.Module):
    def __init__(self, n_channels=1, n_out_channels=1, base_ch=32):
        super().__init__()
        self.inc = ResBlock1D(n_channels, base_ch); self.down1 = Down(base_ch, base_ch*2); self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8); self.down4 = Down(base_ch*8, base_ch*16); self.up1 = Up(base_ch*16, base_ch*8)
        self.up2 = Up(base_ch*8, base_ch*4); self.up3 = Up(base_ch*4, base_ch*2); self.up4 = Up(base_ch*2, base_ch); self.outc = OutConv(base_ch, n_out_channels)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1); return self.outc(x)

# --- NEW MLP with Fusion Layer ---
class MLPWithFusionLayer(nn.Module):
    def __init__(self, n_gases=11):
        super().__init__()
        # Input shape: (B, 2, 3600)
        self.channel_fuser = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        self.sp1_conv = nn.Sequential(
            nn.Linear(3600, 512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_gases)
        )

    def forward(self, x):
        fused_spectrum = self.channel_fuser(x)
        x_flat = self.flatten(fused_spectrum)
        x_processed = self.sp1_conv(x_flat)
        return self.mlp(x_processed)

class EarlyStop:
    def __init__(self, patience=8, mode="min"):
        self.best = np.inf; self.wait = 0; self.patience = patience; self.mode = mode
    def step(self, value):
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved: self.best = value; self.wait = 0; return False
        else: self.wait += 1; return self.wait >= self.patience
# endregion

def make_loader(X: np.ndarray, Y: np.ndarray, bs: int, shuffle: bool):
    return DataLoader(TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))), batch_size=bs, shuffle=shuffle, drop_last=shuffle)

def train_epoch(model, loader, criterion, optim, device):
    model.train(); run = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(); loss = criterion(model(x), y); loss.backward(); optim.step()
        run += loss.item() * x.size(0)
    return run / len(loader.dataset)

def evaluate_model(model, loader, criterion_func, device):
    model.eval(); running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion_func(model(x), y)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def eval_metrics(model, loader, device, y_scaler):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            trues.append(y.cpu())
            
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    
    preds_inv_np = y_scaler.inverse_transform(preds.numpy())
    trues_inv_np = y_scaler.inverse_transform(trues.numpy())

    preds_inv_t = torch.from_numpy(preds_inv_np)
    trues_inv_t = torch.from_numpy(trues_inv_np)

    r2 = r2_score(preds_inv_t, trues_inv_t, multioutput='raw_values')

    rmse = np.sqrt(np.mean((preds_inv_np - trues_inv_np)**2, axis=0))
    nrmse = rmse / (trues_inv_np.max(axis=0) - trues_inv_np.min(axis=0))
    return r2.cpu().numpy(), rmse, nrmse

def fine_tune_unet(unet, train_loader, val_loader, weights, args, device):
    """Helper function to fine-tune a U-Net model."""
    print(f"Fine-tuning U-Net: {unet.label}")
    def crit(pred, tgt): return ((pred - tgt)**2 * weights).mean()
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr_u, weight_decay=1e-5)
    sch = CosineAnnealingWarmRestarts(opt, 10, eta_min=1e-6)
    early_stop = EarlyStop(patience=8)
    for ep in range(1, args.epochs_u + 1):
        tl = train_epoch(unet, train_loader, crit, opt, device)
        sch.step()
        vl = evaluate_model(unet, val_loader, crit, device)
        print(f"  [{unet.label}] ep{ep:02d}/{args.epochs_u}  tr={tl:.4e}  va={vl:.4e}")
        if early_stop.step(vl):
            print(f"  Early stopping at epoch {ep}")
            break

def generate_dual_channel_data(loader, unet_orig, unet_smooth, scaler, device):
    """Generates the 2-channel input for the fusion MLP."""
    unet_orig.eval(); unet_smooth.eval()
    ch1_list, ch2_list, y_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            # Channel 1: "Perfectly" cleaned spectra
            clean1 = (x_dev - unet_orig(x_dev)).cpu().squeeze(1).numpy()
            # Channel 2: "Drift-free but noisy" spectra
            clean2 = (x_dev - unet_smooth(x_dev)).cpu().squeeze(1).numpy()
            
            ch1_list.append(clean1)
            ch2_list.append(clean2)
            y_list.append(y.numpy())

    ch1_scaled = scaler.transform(np.concatenate(ch1_list))
    ch2_scaled = scaler.transform(np.concatenate(ch2_list))
    
    # Stack into (N, 2, 3600)
    dual_channel_x = np.stack([ch1_scaled, ch2_scaled], axis=1)
    return dual_channel_x, np.concatenate(y_list)

def compute_blend_weights(unet_orig, unet_smooth, mlp, loader_noisy_noshuff, scaler, mse_val_gas, device, args):
    mlp.eval()
    target_gas_indices = torch.where(mse_val_gas > 0)[0].tolist()
    ig_attributor = IntegratedGradients(mlp)

    accum = torch.zeros(3600, device=device)
    total_samples = 0
    
    # Generate the 2-channel data needed for IG
    dual_channel_x, _ = generate_dual_channel_data(loader_noisy_noshuff, unet_orig, unet_smooth, scaler, device)
    
    # Process in batches to avoid memory issues
    ig_dataset = TensorDataset(torch.from_numpy(dual_channel_x.astype(np.float32)))
    ig_loader = DataLoader(ig_dataset, batch_size=args.bs)

    for (inputs,) in ig_loader:
        inputs = inputs.to(device)
        total_samples += inputs.size(0)
        baseline = torch.zeros_like(inputs)

        for gas_idx in target_gas_indices:
            attributions = ig_attributor.attribute(inputs, baselines=baseline, target=gas_idx, n_steps=256)
            # Sum attributions across batch and BOTH channels to get importance per wavelength
            weighted_ig = attributions.abs().sum(dim=0).sum(dim=0) * mse_val_gas[gas_idx].to(device)
            accum += weighted_ig

    weights = accum / total_samples
    weights = weights / (weights.max() + 1e-12)
    return weights

def plot_weights(w, path):
    """Plots the calculated loss weights against the wavelength."""
    wl = np.linspace(890, 1250, len(w))
    w_np = w.detach().cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.plot(wl, w_np)
    plt.title("Calculated Loss Weights for U-Net Fine-tuning")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized Weight")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(a.results_root)/datetime.datetime.now().strftime("%d%m_%H%M")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    print("Dual model test run, dropout 0.1")

    labels = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Nitrous oxide', 'Ethylene', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol']
    focus_gases = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Ethylene', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol']
    gas_mask = torch.tensor([g in focus_gases for g in labels], dtype=torch.bool, device=device)

    # --- Load and prepare all data ---
    tr = np.load(a.train); va = np.load(a.val); te = np.load(a.test)
    Xtr, Btr, Ytr = tr["inputs_lntrans_1"], tr["inputs_lnbase_1"], np.delete(tr["c"],7,1)
    Xva, Bva, Yva = va["inputs_lntrans_1"], va["inputs_lnbase_1"], np.delete(va["c"],7,1)
    Xte, Bte, Yte = te["inputs_lntrans_1"], te["inputs_lnbase_1"], np.delete(te["c"],7,1)
    
    # Create smoothed baseline targets
    Btr_smooth = savgol_filter(Btr, 51, 3, axis=1).copy()
    Bva_smooth = savgol_filter(Bva, 51, 3, axis=1).copy()

    # --- Create Scalers ---
    # IMPORTANT: Use a single shared scaler for MLP inputs, fit on ideal clean data
    ideal_clean_tr = Xtr - Btr
    shared_x_scaler = MinMaxScaler(feature_range=(0,1), clip=True).fit(ideal_clean_tr)
    joblib.dump(shared_x_scaler, outdir / 'shared_x_scaler.save')
    y_scaler = MinMaxScaler().fit(Ytr)
    joblib.dump(y_scaler, outdir / 'y_scaler.save')
    Ytr_s, Yva_s, Yte_s = y_scaler.transform(Ytr), y_scaler.transform(Yva), y_scaler.transform(Yte)

    # --- Create DataLoaders ---
    # Loaders for the two U-Nets
    tr_base_orig   = make_loader(Xtr[:,None,:], Btr[:,None,:], a.bs, True)
    va_base_orig   = make_loader(Xva[:,None,:], Bva[:,None,:], a.bs, False)
    tr_base_smooth = make_loader(Xtr[:,None,:], Btr_smooth[:,None,:], a.bs, True)
    va_base_smooth = make_loader(Xva[:,None,:], Bva_smooth[:,None,:], a.bs, False)
    # Non-shuffled loader for consistent data generation
    tr_noisy_noshuff = make_loader(Xtr[:,None,:], Ytr_s, a.bs, False)
    va_noisy_noshuff = make_loader(Xva[:,None,:], Yva_s, a.bs, False)
    te_noisy_noshuff = make_loader(Xte[:,None,:], Yte_s, a.bs, False)

    # --- Initialize Models ---
    unet_orig = ResUNetBaseline().to(device); unet_orig.load_state_dict(torch.load(a.unet_orig, map_location=device)); unet_orig.label = "UNet-Orig"
    unet_smooth = ResUNetBaseline().to(device); unet_smooth.load_state_dict(torch.load(a.unet_smooth, map_location=device)); unet_smooth.label = "UNet-Smooth"
    mlp_fused = MLPWithFusionLayer(n_gases=11).to(device) # Initialized from scratch
    weights = torch.ones(3600, device=device)

    for it in range(1, a.iters + 1):
        print(f"\n{'='*15} ITERATION {it}/{a.iters} {'='*15}")

        # --- A) Fine-tune both U-Nets (skip iter 1, as MLP is pre-training) ---
        if it > 1:
            fine_tune_unet(unet_orig, tr_base_orig, va_base_orig, weights, a, device)
            fine_tune_unet(unet_smooth, tr_base_smooth, va_base_smooth, weights, a, device)
            torch.save(unet_orig.state_dict(), outdir/f"unet_orig_iter{it}.pth")
            torch.save(unet_smooth.state_dict(), outdir/f"unet_smooth_iter{it}.pth")

        # --- B) Generate dual-channel data for the MLP ---
        print("Generating dual-channel data for MLP...")
        Xtr_reg, Ytr_reg = generate_dual_channel_data(tr_noisy_noshuff, unet_orig, unet_smooth, shared_x_scaler, device)
        Xva_reg, Yva_reg = generate_dual_channel_data(va_noisy_noshuff, unet_orig, unet_smooth, shared_x_scaler, device)
        Xte_reg, Yte_reg = generate_dual_channel_data(te_noisy_noshuff, unet_orig, unet_smooth, shared_x_scaler, device)
        
        tr_reg = DataLoader(TensorDataset(torch.from_numpy(Xtr_reg), torch.from_numpy(Ytr_reg)), batch_size=a.bs, shuffle=True)
        va_reg = DataLoader(TensorDataset(torch.from_numpy(Xva_reg), torch.from_numpy(Yva_reg)), batch_size=a.bs, shuffle=False)
        te_reg = DataLoader(TensorDataset(torch.from_numpy(Xte_reg), torch.from_numpy(Yte_reg)), batch_size=a.bs, shuffle=False)
        
        # --- C) Pre-train or fine-tune the Fusion MLP ---
        if it == 1:
            print("Pre-training Fusion MLP...")
            lr, epochs = a.lr_m_pretrain, a.epochs_m_pretrain
        else:
            print("Fine-tuning Fusion MLP...")
            lr, epochs = a.lr_m, a.epochs_m
            
        opt2 = torch.optim.AdamW(mlp_fused.parameters(), lr=lr, weight_decay=1e-5)
        sch2 = CosineAnnealingLR(opt2, epochs, eta_min=1e-6)
        best, best_r2 = copy.deepcopy(mlp_fused.state_dict()), -1e9
        
        for ep in range(1, epochs + 1):
            tl = train_epoch(mlp_fused, tr_reg, nn.MSELoss(), opt2, device); sch2.step()
            r2, _, _ = eval_metrics(mlp_fused, va_reg, device, y_scaler); r2m = float(np.mean(r2))
            if r2m > best_r2: best_r2 = r2m; best = copy.deepcopy(mlp_fused.state_dict())
            print(f"  [MLP] ep{ep:02d}/{epochs}  tr_loss={tl:.4e}  val_R2_mean={r2m:.4f}")
        mlp_fused.load_state_dict(best); torch.save(best, outdir/f"mlp_fused_iter{it}.pth")

        # --- D) Evaluate performance on the test set ---
        r2_t, rmse_t, nrmse_t = eval_metrics(mlp_fused, te_reg, device, y_scaler)
        print("\n" + "-"*20 + f" Performance on TEST set (iteration {it}) " + "-"*20)
        for name, r2_g, rmse_g, nrmse_g in zip(labels, r2_t, rmse_t, nrmse_t):
            print(f"  {name:15s} | R²={r2_g:6.4f} | RMSE={rmse_g:8.4f} | NRMSE={nrmse_g:7.4f}")
        json.dump({"R2":r2_t.tolist(), "RMSE":rmse_t.tolist(), "NRMSE":nrmse_t.tolist()}, open(outdir/f"metrics_iter{it}.json", "w"), indent=2)

        # --- E) Compute loss weights for the next iteration's U-Nets ---
        print("Computing loss weights for next iteration...")
        mlp_fused.eval(); pv, tv = [], []
        with torch.no_grad():
            for x, y in va_reg:
                pv.append(mlp_fused(x.to(device))); tv.append(y.to(device))
        pv = torch.cat(pv); tv = torch.cat(tv)
        mse_val = torch.mean((pv - tv)**2, 0)
        mse_val_focus = mse_val.clone(); mse_val_focus[~gas_mask] = 0.0
        print(f"MSE per gas on validation set (for weighting): {mse_val_focus.cpu().numpy()}")
        weights = compute_blend_weights(unet_orig, unet_smooth, mlp_fused, tr_noisy_noshuff, shared_x_scaler, mse_val_focus, device, a).detach()
        plot_weights(weights, outdir/f"w_iter{it}.png")

if __name__ == "__main__":
    class Args: pass
    a = Args()
    a.train = "/vol/csedu-nobackup/project/jlandsman/data/train_data.npz"
    a.val = "/vol/csedu-nobackup/project/jlandsman/data/val_data.npz"
    a.test = "/vol/csedu-nobackup/project/jlandsman/data/test_data.npz"
    a.unet_orig = "/vol/csedu-nobackup/project/jlandsman/trained_models/unet_4_32_v1.pth"
    a.unet_smooth = "/vol/csedu-nobackup/project/jlandsman/jobs/unet_trained_smooth/unet_weights.pth"
    a.results_root = "./full_proc_results"
    a.bs = 128
    a.iters = 10
    # U-Net fine-tuning params
    a.lr_u = 1e-4
    a.epochs_u = 20
    # MLP pre-training params (iter 1)
    a.lr_m_pretrain = 1e-3
    a.epochs_m_pretrain = 200
    # MLP fine-tuning params (iter > 1)
    a.lr_m = 1e-4
    a.epochs_m = 20

    main(a)