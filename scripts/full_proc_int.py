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
import copy
from captum.attr import IntegratedGradients

# Model definitions
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

class GasConcentrationMLP(nn.Module):    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()    # <-- flatten (B,1,3600) → (B,3600)
        self.sp1_conv = nn.Sequential(
            nn.Linear(3600, 512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, abs1_):
        x = self.flatten(abs1_)       # now (B, 3600)
        x = self.sp1_conv(x)
        return self.mlp(x)            # outputs (B, 11)

#  Dataloaders
def make_loader(X: np.ndarray, Y: np.ndarray, bs: int, shuffle: bool):
    return DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                      torch.as_tensor(Y, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle, drop_last=False)

def train_epoch(model, loader, criterion, optim, device):
    model.train(); run = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optim.step()
        run += loss.item()*x.size(0)
    return run/len(loader.dataset)

def evaluate_model(model, loader, criterion_func, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion_func(pred, y)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def eval_metrics(model, loader, device, y_scaler):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu()); trues.append(y)
    preds = torch.cat(preds); trues = torch.cat(trues)
    preds_inv = torch.tensor(y_scaler.inverse_transform(preds))
    trues_inv = torch.tensor(y_scaler.inverse_transform(trues))
    r2 = r2_score(trues_inv, preds_inv, multioutput='raw_values')
    rmse = torch.sqrt(torch.mean((preds_inv-trues_inv)**2,0))
    nrmse = rmse/(trues_inv.max(0).values-trues_inv.min(0).values)
    return r2.numpy(), rmse.numpy(), nrmse.numpy()

def compute_blend_weights(
        unet, mlp,
        train_loader_no_shuffle, x_scaler,
        mse_val_gas: torch.Tensor,
        device: torch.device,
        baseline_steps: int = 256,       # IG path resolution
):
    unet.eval()
    mlp_dev = copy.deepcopy(mlp).to(device).eval()        # stay on single device
    target_gas_indices = torch.where(mse_val_gas > 0)[0].tolist()

    ig_attributor = IntegratedGradients(mlp_dev)

    accum = torch.zeros(3600, device=device)
    total_samples = 0

    for noisy, _ in train_loader_no_shuffle:
        noisy = noisy.to(device)
        total_samples += noisy.size(0)

        with torch.no_grad():
            cleaned_input = noisy - unet(noisy) # Shape: [B, 1, 3600]
        
        np_in = cleaned_input.squeeze(1).detach().cpu().numpy()
        np_scaled = x_scaler.transform(np_in)
        scaled_input = torch.from_numpy(np_scaled).unsqueeze(1).to(device)
        
        baseline = torch.zeros_like(scaled_input)
        for gas_idx in target_gas_indices:
            attributions_for_gas = ig_attributor.attribute(
                scaled_input,
                baselines=baseline,
                target=gas_idx,
                n_steps=baseline_steps
            )

            mse_for_this_gas = mse_val_gas[gas_idx].to(device)
            weighted_ig = attributions_for_gas.abs().sum(dim=0).squeeze() * mse_for_this_gas
            accum += weighted_ig

    # Final averaging and scaling to [0, 1]
    weights = accum / total_samples
    weights = weights / (weights.max() + 1e-12)
    return weights

def plot_weights(w, path):
    wl = np.linspace(890,1250,len(w))
    w_np = w.detach().cpu().numpy()
    plt.figure(figsize=(10,3)); plt.plot(wl,w_np); 
    plt.xlabel("Wavelength (cm$^{-1}$)"); plt.ylabel("Weight"); plt.tight_layout()
    plt.savefig(path); plt.close()

# Main
def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(a.results_root)/datetime.datetime.now().strftime("%d%m_%H%M")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    labels = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Nitrous oxide', 'Ethylene', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol']
    focus_gases = ['Water', 'Carbon Dioxide', 'Methane', 'Ammonia', 'Methanol', 'Acetone', 'Isoprene', 'Acetaldehyde', 'Ethanol']
    gas_mask = torch.tensor(
            [g in focus_gases for g in labels],    # Boolean list → BoolTensor
            dtype=torch.bool, device=device)
    # load data sets
    tr = np.load(a.train);  va = np.load(a.val);  te = np.load(a.test)
    Xtr, Btr, Ytr = tr["inputs_lntrans_1"], tr["inputs_lnbase_1"], np.delete(tr["c"],7,1)  # Remove CO for now
    Xva, Bva, Yva = va["inputs_lntrans_1"], va["inputs_lnbase_1"], np.delete(va["c"],7,1)
    Xte, Bte, Yte = te["inputs_lntrans_1"], te["inputs_lnbase_1"], np.delete(te["c"],7,1)

    x_scaler = MinMaxScaler(feature_range=(0,1), clip=True)
    x_scaler.fit(np.concatenate([np.load("/vol/csedu-nobackup/project/jlandsman/data/train_data_no_drift.npz")["inputs_lntrans_1"].astype(np.float32), np.load("/vol/csedu-nobackup/project/jlandsman/data/train_data_no_drift2.npz")["inputs_lntrans_1"].astype(np.float32)], axis=0))

    y_scaler = MinMaxScaler().fit(Ytr)

    Ytr_s, Yva_s, Yte_s = map(y_scaler.transform,(Ytr,Yva,Yte))

    # loaders
    tr_base   = make_loader(Xtr[:,None,:], Btr[:,None,:], a.bs, True)
    tr_base_noshuff = make_loader(Xtr[:,None,:], Btr[:,None,:], a.bs, False)
    va_base   = make_loader(Xva[:,None,:], Bva[:,None,:], a.bs, False)
    te_base   = make_loader(Xte[:,None,:], Bte[:,None,:], a.bs, False)

    # models
    unet = ResUNetBaseline().to(device); unet.load_state_dict(torch.load(a.unet,map_location=device))
    mlp  = GasConcentrationMLP().to(device); mlp.load_state_dict(torch.load(a.mlp,map_location=device))

    weights = torch.ones(3600, device=device)

    for it in range(1,a.iters+1):
        print(f"\n=== ITER {it}/{a.iters} ===")

        # --- A) fine-tune baseline (skip iter1)
        if it>1:
            def crit(pred,tgt,w=weights): return ((pred-tgt)**2*w).mean()
            opt = torch.optim.AdamW(unet.parameters(), lr=a.lr_u, weight_decay=1e-5)
            sch = CosineAnnealingWarmRestarts(opt, 10, eta_min=1e-6)
            for ep in range(1,a.epochs_u+1):
                tl = train_epoch(unet,tr_base,crit,opt,device)
                sch.step()
                vl = evaluate_model(unet, va_base, crit, device)
                print(f"[U-Net] ep{ep:02d}/{a.epochs_u}  tr={tl:.4e}  va={vl:.4e}")
            torch.save(unet.state_dict(), outdir/f"unet_iter{it}.pth")

        # --- B) generate clean spectra using NON-shuffled loaders to match order
        def get_clean(loader):
            unet.eval(); outs=[]
            with torch.no_grad():
                for x,_ in loader:
                    x=x.to(device); outs.append((x-unet(x)).cpu().squeeze(1))
            return torch.cat(outs).numpy()
        Xtr_c, Xva_c, Xte_c = map(get_clean,(tr_base_noshuff,va_base,te_base))
        Xtr_c_s = x_scaler.transform(Xtr_c); Xva_c_s = x_scaler.transform(Xva_c); Xte_c_s = x_scaler.transform(Xte_c)

        # --- loaders for regressor
        tr_reg = make_loader(Xtr_c_s[:,None,:], Ytr_s, a.bs, True)
        va_reg = make_loader(Xva_c_s[:,None,:], Yva_s, a.bs, False)
        te_reg = make_loader(Xte_c_s[:,None,:], Yte_s, a.bs, False)

        # --- C) fine-tune regressor
        opt2 = torch.optim.AdamW(mlp.parameters(), lr=a.lr_m, weight_decay=1e-5)
        sch2 = CosineAnnealingLR(opt2,a.epochs_m, eta_min=1e-6)
        best, best_r2 = None,-1e9
        for ep in range(1,a.epochs_m+1):
            tl = train_epoch(mlp,tr_reg,nn.MSELoss(),opt2,device); sch2.step()
            r2, _, _ = eval_metrics(mlp,va_reg,device,y_scaler); r2m=float(r2.mean())
            if r2m>best_r2: best_r2=r2m; best=mlp.state_dict()
            print(f"[MLP] ep{ep:02d}/{a.epochs_m}  tr_loss={tl:.4e}  val_R2={r2m:.3f}")
        mlp.load_state_dict(best); torch.save(best,outdir/f"mlp_iter{it}.pth")

        # --- D) test metrics
        r2_t,rmse_t,nrmse_t = eval_metrics(mlp,te_reg,device,y_scaler)
        print("\nPerformance on TEST set (iteration {}):".format(it))
        for name, r2_g, rmse_g, nrmse_g in zip(labels, r2_t, rmse_t, nrmse_t):
            print(f"  {name:15s}  R²={r2_g:6.3f}   RMSE={rmse_g:7.4f}   NRMSE={nrmse_g:7.4f}")
        json.dump({"R2":r2_t.tolist(),"RMSE":rmse_t.tolist(),"NRMSE":nrmse_t.tolist()},
                  open(outdir/f"metrics_iter{it}.json","w"), indent=2)

        # --- E) blend weights  (Gradients on TRAIN  ×  MSE on VAL)
        # MSE per gas on val
        mlp.eval(); pv, tv = [],[]
        with torch.no_grad():
            for x,y in va_reg:
                pv.append(mlp(x.to(device))); tv.append(y.to(device))
        pv=torch.cat(pv); tv=torch.cat(tv)
        mse_val=torch.mean((pv-tv)**2,0)             # (11,)
        mse_val_focus = mse_val.clone()
        mse_val_focus[~gas_mask] = 0.0
        print("mse_val:", mse_val.cpu().numpy())
        weights = compute_blend_weights(unet,mlp,tr_base,x_scaler,mse_val_focus,device).detach()
        np.save(outdir/f"w_iter{it}.npy",weights.detach().cpu().numpy())
        plot_weights(weights,outdir/f"w_iter{it}.png")

if __name__=="__main__":
    class Args:
        pass

    a = Args()
    a.train = "/vol/csedu-nobackup/project/jlandsman/data/train_data.npz"
    a.val = "/vol/csedu-nobackup/project/jlandsman/data/val_data.npz"
    a.test = "/vol/csedu-nobackup/project/jlandsman/data/test_data.npz"
    a.unet = "/vol/csedu-nobackup/project/jlandsman/trained_models/unet_4_32_v1.pth"
    a.mlp = "/vol/csedu-nobackup/project/jlandsman/jobs/full_proc_results/0707_2010/mlp_iter10.pth"
    a.y_scaler = '/vol/csedu-nobackup/project/jlandsman/trained_models/mlp_pretrained/conc_scaler.save'
    a.results_root = "./full_proc_results"
    a.bs = 128
    a.lr_u = 1e-4
    a.lr_m = 2e-4
    a.epochs_u = 15
    a.epochs_m = 12
    a.iters = 10

    main(a)