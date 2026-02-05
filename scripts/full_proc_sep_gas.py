from __future__ import annotations
import datetime, json
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics.functional import r2_score
from torch.func import jacrev, vmap
from sklearn.preprocessing import MinMaxScaler
import copy

# --- Model definitions (ResUNet, MLP, etc.) ---
# (Your model definitions are perfect and are included here without change)
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
        avg_pool = x.mean(dim=2, keepdim=True)
        max_pool, _ = x.max(dim=2, keepdim=True)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return self.sigmoid(attn) * x

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1)//2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool, _ = x.max(dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
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
            else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
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
            CBAMResBlock1D(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = CBAMResBlock1D(in_ch, out_ch)

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

class ResUNetBaseline(nn.Module):
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        return self.outc(x)

class GasConcentrationMLP(nn.Module):
    def __init__(self, input_features=3600, num_gases=1, hidden_layers=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_gases))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class EarlyStop:
    def __init__(self, patience=8, mode="min"):
        self.best = np.inf if mode == "min" else -np.inf
        self.wait = 0
        self.patience = patience
        self.mode = mode

    def step(self, value):
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience

# --- Helper Functions ---
def make_loader(X: np.ndarray, Y: np.ndarray, bs: int, shuffle: bool):
    return DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                      torch.as_tensor(Y, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle, drop_last=False)

def train_epoch(model, loader, criterion, optim, device, weights=None):
    model.train(); run = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        pred = model(x)
        if weights is not None: # For weighted U-Net loss
            loss = ((pred - y)**2 * weights).mean()
        else:
            loss = criterion(pred, y)
        loss.backward(); optim.step()
        run += loss.item() * x.size(0)
    return run / len(loader.dataset)

def evaluate_model(model, loader, criterion, device, weights=None):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if weights is not None:
                loss = ((pred - y)**2 * weights).mean()
            else:
                loss = criterion(pred, y)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def eval_metrics(model, loader, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu()); trues.append(y)
    preds = torch.cat(preds); trues = torch.cat(trues)
    # For single-output regression, these metrics are correct
    r2 = r2_score(trues, preds)
    rmse = torch.sqrt(torch.mean((preds-trues)**2))
    nrmse = rmse / (trues.max() - trues.min())
    return r2.item(), nrmse.item()

def compute_blend_weights(unet, mlp, train_loader_no_shuffle, x_scaler, mse_val_gas, device):
    unet.eval()
    mlp_cpu = copy.deepcopy(mlp).to(torch.device("cpu"))
    mlp_cpu.eval()
    accum = torch.zeros(3600)
    # Correctly shape MSE for broadcasting with single-gas Jacobian
    mse_bc = mse_val_gas.cpu().view(1, 1, 1)

    def mlp_f(x_flat: torch.Tensor) -> torch.Tensor:
        return mlp_cpu(x_flat.unsqueeze(0).unsqueeze(0)).squeeze(0)
    jac_fn = jacrev(mlp_f)

    total_samples = 0
    for noisy, _ in train_loader_no_shuffle:
        noisy = noisy.to(device)
        with torch.no_grad():
            baseline_pred = unet(noisy)
        cleaned = (noisy - baseline_pred).squeeze(1)
        cleaned_cpu = cleaned.cpu().detach()
        cleaned_norm = torch.as_tensor(x_scaler.transform(cleaned_cpu.numpy()), dtype=cleaned_cpu.dtype)
        total_samples += cleaned_norm.size(0)
        
        # J will be (B, 1, 3600) for a single-output MLP
        J = vmap(jac_fn)(cleaned_norm).abs()
        # weighted_sum is correct, summing over Batch and Gas (dim 1)
        weighted_sum = (J * mse_bc).sum((0, 1))
        accum += weighted_sum
        del J, cleaned, cleaned_cpu, cleaned_norm, baseline_pred

    weights = accum / total_samples
    weights = weights / (weights.max() + 1e-12)
    return weights.to(device)

def plot_weights(w, path):
    wl = np.linspace(890, 1250, len(w))
    w_np = w.detach().cpu().numpy()
    plt.figure(figsize=(10, 3)); plt.plot(wl, w_np);
    plt.xlabel("Wavelength (cm$^{-1}$)"); plt.ylabel("Weight"); plt.tight_layout()
    plt.savefig(path); plt.close()


def run_process_for_gas(a, x_scaler, gas_label: str, gas_idx: int, all_data: dict):
    """
    Runs the entire iterative fine-tuning pipeline for a single gas.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Setup Output Directory ---
    gas_outdir = Path(a.results_root) / gas_label
    timestamp_outdir = gas_outdir / datetime.datetime.now().strftime("%d%m_%H%M")
    timestamp_outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*20} PROCESSING: {gas_label.upper()} {'='*20}")
    print(f"Output will be saved to: {timestamp_outdir}")

    # --- 2. Prepare Data for the Current Gas ---
    Ytr = all_data['Ytr_all'][:, gas_idx].reshape(-1, 1)
    Yva = all_data['Yva_all'][:, gas_idx].reshape(-1, 1)
    Yte = all_data['Yte_all'][:, gas_idx].reshape(-1, 1)

    tr_base = make_loader(all_data['Xtr'][:, None, :], all_data['Btr'][:, None, :], a.bs, True)
    tr_base_noshuff = make_loader(all_data['Xtr'][:, None, :], all_data['Btr'][:, None, :], a.bs, False)
    va_base = make_loader(all_data['Xva'][:, None, :], all_data['Bva'][:, None, :], a.bs, False)

    # --- 3. Initialize Models for the Current Gas ---
    # Load the general-purpose pre-trained U-Net
    unet = ResUNetBaseline().to(device)
    unet.load_state_dict(torch.load(a.unet_path, map_location=device))

    # Load the specific pre-trained MLP for this gas
    mlp = GasConcentrationMLP(num_gases=1).to(device)
    mlp_path = Path(a.mlp_folder) / f"{gas_label}/mlp_weights_{gas_label}.pth"
    if not mlp_path.exists():
        print(f"---! WARNING: MLP for {gas_label} not found at {mlp_path}. Skipping this gas. !---")
        return
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))

    weights = torch.ones(3600, device=device)

    # --- 4. Iterative Fine-Tuning Loop ---
    for it in range(1, a.iters + 1):
        print(f"\n--- ITERATION {it}/{a.iters} for {gas_label} ---")

        # A) Fine-tune baseline U-Net (skips on first iteration)
        if it > 1:
            print("Fine-tuning U-Net...")
            # Freeze encoder for stable fine-tuning
            for name, param in unet.named_parameters():
                if 'down' in name or 'inc' in name: param.requires_grad = False
            
            trainable_params = [p for p in unet.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable_params, lr=a.lr_u, weight_decay=1e-5)
            sch = CosineAnnealingWarmRestarts(opt, 10, eta_min=1e-6)
            early_stop = EarlyStop(patience=8)
            best_unet_loss = float('inf')
            best_unet_state = None

            for ep in range(1, a.epochs_u + 1):
                tl = train_epoch(unet, tr_base, None, opt, device, weights=weights)
                vl = evaluate_model(unet, va_base, None, device, weights=weights)
                sch.step()
                print(f"[U-Net] Ep {ep:02d}/{a.epochs_u} | Train Loss: {tl:.4e} | Val Loss: {vl:.4e}")
                
                if vl < best_unet_loss:
                    best_unet_loss = vl
                    best_unet_state = copy.deepcopy(unet.state_dict())

                if early_stop.step(vl):
                    print(f"Early stopping U-Net at epoch {ep}")
                    break
            
            if best_unet_state: unet.load_state_dict(best_unet_state)
            torch.save(unet.state_dict(), timestamp_outdir / f"unet_iter{it}.pth")
            
            # Unfreeze layers for the next potential iteration
            for param in unet.parameters(): param.requires_grad = True


        # B) Generate "cleaned" spectra with the current U-Net
        def get_clean(loader):
            unet.eval(); outs = []
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(device); outs.append((x - unet(x)).cpu().squeeze(1))
            return torch.cat(outs).numpy()
        
        Xtr_c = get_clean(tr_base_noshuff)
        Xva_c = get_clean(va_base)
        Xte_c = get_clean(make_loader(all_data['Xte'][:, None, :], all_data['Bte'][:, None, :], a.bs, False))

        Xtr_c_s = x_scaler.transform(Xtr_c); Xva_c_s = x_scaler.transform(Xva_c); Xte_c_s = x_scaler.transform(Xte_c)

        # C) Fine-tune the gas-specific MLP
        print("Fine-tuning MLP...")
        tr_reg = make_loader(Xtr_c_s[:, None, :], Ytr, a.bs, True)
        va_reg = make_loader(Xva_c_s[:, None, :], Yva, a.bs, False)
        te_reg = make_loader(Xte_c_s[:, None, :], Yte, a.bs, False)
        
        opt2 = torch.optim.AdamW(mlp.parameters(), lr=a.lr_m, weight_decay=1e-5)
        sch2 = CosineAnnealingLR(opt2, a.epochs_m, eta_min=1e-6)
        early_stop_mlp = EarlyStop(patience=10, mode="max") # Stop on best R2
        best_mlp_state = copy.deepcopy(mlp.state_dict())

        for ep in range(1, a.epochs_m + 1):
            tl = train_epoch(mlp, tr_reg, nn.MSELoss(), opt2, device)
            sch2.step()
            r2, _ = eval_metrics(mlp, va_reg, device)
            print(f"[MLP] Ep {ep:02d}/{a.epochs_m} | Train Loss: {tl:.4e} | Val R²: {r2:.3f}")
            if early_stop_mlp.step(r2):
                # We save based on best R2, so we just load the best state at the end
                pass
            if r2 > early_stop_mlp.best: # a bit redundant with the EarlyStop object but safe
                best_mlp_state = copy.deepcopy(mlp.state_dict())

        mlp.load_state_dict(best_mlp_state)
        torch.save(best_mlp_state, timestamp_outdir / f"mlp_iter{it}.pth")

        # D) Evaluate and save final metrics for this iteration
        r2_t, nrmse_t = eval_metrics(mlp, te_reg, device)
        print("\n" + f"Performance on TEST set for {gas_label} (iteration {it}):")
        print(f"  R²={r2_t:6.3f}   NRMSE={nrmse_t:7.4f}")
        json.dump({"R2": r2_t, "NRMSE": nrmse_t},
                  (timestamp_outdir / f"metrics_iter{it}.json").open("w"), indent=2)

        # E) Compute new weights for the next U-Net fine-tuning step
        if it < a.iters:
            print("Computing new loss weights for next iteration...")
            mlp.eval()
            pv, tv = [], []
            with torch.no_grad():
                for x, y in va_reg:
                    pv.append(mlp(x.to(device))); tv.append(y.to(device))
            pv = torch.cat(pv); tv = torch.cat(tv)
            mse_val = torch.mean((pv - tv)**2, dim=0)
            
            weights = compute_blend_weights(unet, mlp, tr_base, x_scaler, mse_val, device).detach()
            np.save(timestamp_outdir / f"w_iter{it}.npy", weights.cpu().numpy())
            plot_weights(weights, timestamp_outdir / f"w_iter{it}.png")


if __name__ == "__main__":
    class Args: pass
    a = Args()
    a.train_data_path = "/vol/csedu-nobackup/project/jlandsman/data/train_data.npz"
    a.val_data_path = "/vol/csedu-nobackup/project/jlandsman/data/val_data.npz"
    a.test_data_path = "/vol/csedu-nobackup/project/jlandsman/data/test_data.npz"
    a.unet_path = "/vol/csedu-nobackup/project/jlandsman/trained_models/unet_4_32_v1.pth"
    a.mlp_folder = "/vol/csedu-nobackup/project/jlandsman/jobs/mlp_pretrained_sep/"
    a.results_root = "./full_proc_results_per_gas"
    a.bs = 128
    a.lr_u = 1e-5  # Lower LR for fine-tuning is recommended
    a.lr_m = 2e-4
    a.epochs_u = 15
    a.epochs_m = 20 # Increased epochs for MLP fine-tuning
    a.iters = 4

    # --- Load all data once to save time ---
    print("Loading all datasets into memory...")
    tr_all = np.load(a.train_data_path)
    va_all = np.load(a.val_data_path)
    te_all = np.load(a.test_data_path)
    all_data = {
        'Xtr': tr_all["inputs_lntrans_1"], 'Btr': tr_all["inputs_lnbase_1"], 'Ytr_all': tr_all["c"],
        'Xva': va_all["inputs_lntrans_1"], 'Bva': va_all["inputs_lnbase_1"], 'Yva_all': va_all["c"],
        'Xte': te_all["inputs_lntrans_1"], 'Bte': te_all["inputs_lnbase_1"], 'Yte_all': te_all["c"],
    }
    print("Data loading complete.")

    x_scaler = MinMaxScaler(feature_range=(0,1), clip=True)
    x_scaler.fit(np.concatenate([np.load("/vol/csedu-nobackup/project/jlandsman/data/train_data_no_drift.npz")["inputs_lntrans_1"].astype(np.float32), np.load("/vol/csedu-nobackup/project/jlandsman/data/train_data_no_drift2.npz")["inputs_lntrans_1"].astype(np.float32)], axis=0))

    # --- Define which gases to process ---
    ALL_LABELS = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Nitrous oxide', 'Ethylene', 'CO', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol']
    
    # You can customize this list to run for a subset of gases
    gases_to_process = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Nitrous oxide', 'Ethylene', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol'] 
    
    for gas_name in gases_to_process:
        if gas_name not in ALL_LABELS:
            print(f"Warning: '{gas_name}' not found in label list. Skipping.")
            continue
        
        gas_index = ALL_LABELS.index(gas_name)
        run_process_for_gas(a, x_scaler, gas_label=gas_name, gas_idx=gas_index, all_data=all_data)