import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Executing Benchmarks on device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

class Conv2DFPEnsembleBlock:
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, R=5, lam=10.0, stride=1, padding=1, device='cpu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.R = R
        self.lam = lam
        self.stride = stride
        self.padding = padding
        self.device = device

        self.patch_dim = in_channels * kernel_size * kernel_size + 1
        self.gammas = [1.0 / R] * R

        self.Q_list = [torch.randn(self.patch_dim, out_channels, device=device) / (self.patch_dim ** 0.5) for _ in range(R)]
        self.U_list = [torch.randn(num_classes, out_channels, device=device) / (num_classes ** 0.5) for _ in range(R)]
        self.W = None

    def _extract_patches(self, X, spatial_step=1):
        N, C, H, W = X.shape
        unfold = F.unfold(X, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        patches = unfold.transpose(1, 2)

        if spatial_step > 1:
            patches = patches.reshape(N, H_out, W_out, -1)
            patches = patches[:, ::spatial_step, ::spatial_step, :]
            patches = patches.reshape(N, -1, C * self.kernel_size * self.kernel_size)

        ones = torch.ones((patches.shape[0], patches.shape[1], 1), device=self.device)
        patches = torch.cat([patches, ones], dim=-1)
        return patches, H_out, W_out

    def fit(self, X, Y_onehot, batch_size=256, spatial_subsample=2):
        t0 = time.time()
        t_target = 0.0
        N = X.shape[0]

        ATA = torch.zeros(self.patch_dim, self.patch_dim, device=self.device)
        ATZ = torch.zeros(self.patch_dim, self.out_channels, device=self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb = X[start:end].to(self.device)
            yb = Y_onehot[start:end].to(self.device)

            patches, _, _ = self._extract_patches(xb, spatial_step=spatial_subsample)
            B, L, D = patches.shape
            A_flat = patches.reshape(B * L, D)
            Y_flat = yb.unsqueeze(1).expand(B, L, self.num_classes).reshape(B * L, self.num_classes)

            tt0 = time.time()
            Z_ens = torch.zeros(B * L, self.out_channels, device=self.device)
            for r in range(self.R):
                Z_r = torch.sign(A_flat @ self.Q_list[r]) + torch.sign(Y_flat @ self.U_list[r])
                Z_ens += self.gammas[r] * Z_r
            t_target += time.time() - tt0

            ATA += A_flat.T @ A_flat
            ATZ += A_flat.T @ Z_ens

        reg = self.lam * (N / 1000.0) * torch.eye(self.patch_dim, device=self.device)
        reg[-1, -1] = 1e-4
        self.W = torch.linalg.solve(ATA + reg, ATZ)
        return t_target, time.time() - t0

    @torch.no_grad()
    def forward(self, X, batch_size=512):
        outputs = []
        for start in range(0, X.shape[0], batch_size):
            xb = X[start:min(start + batch_size, X.shape[0])].to(self.device)
            patches, H_out, W_out = self._extract_patches(xb)
            Z = patches @ self.W
            A = F.relu(Z).transpose(1, 2).reshape(xb.shape[0], self.out_channels, H_out, W_out)
            A = F.max_pool2d(A, kernel_size=2, stride=2)
            A = (A - A.mean(dim=(2, 3), keepdim=True)) / (A.std(dim=(2, 3), keepdim=True) + 1e-5)
            # Transfer intermediate feature maps to CPU to maintain strict O(1) GPU VRAM
            outputs.append(A.cpu())
        return torch.cat(outputs, dim=0)

class GenericFPEnsembleCNN:
    def __init__(self, in_channels, in_size, num_classes, R=5, conv_channels=(64, 128, 256), hidden_mlp=1024, lam_conv=5.0, lam_fc=20.0, device='cpu'):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.R = R
        self.lam_fc = lam_fc
        self.device = device
        self.gammas = [1.0 / R] * R

        self.conv1 = Conv2DFPEnsembleBlock(in_channels, conv_channels[0], 3, num_classes, R=R, lam=lam_conv, device=device)
        self.conv2 = Conv2DFPEnsembleBlock(conv_channels[0], conv_channels[1], 3, num_classes, R=R, lam=lam_conv, device=device)
        self.conv3 = Conv2DFPEnsembleBlock(conv_channels[1], conv_channels[2], 3, num_classes, R=R, lam=lam_conv, device=device)

        s = in_size
        for _ in range(3):
            s = s // 2
        self.final_spatial = s
        feat_dim = conv_channels[2] * (s * s + 2) + 1
        self.hidden_mlp = hidden_mlp

        self.Q_fc = [torch.randn(feat_dim, hidden_mlp, device=device) / (feat_dim ** 0.5) for _ in range(R)]
        self.U_fc = [torch.randn(num_classes, hidden_mlp, device=device) / (num_classes ** 0.5) for _ in range(R)]
        self.W_fc = None
        self.W_out = None

    def _pool_features_batch(self, C3_batch):
        C3 = C3_batch.to(self.device)
        flat = C3.reshape(C3.shape[0], -1)
        g_avg = C3.mean(dim=(2, 3))
        g_max = C3.amax(dim=(2, 3))
        feats = torch.cat([flat, g_avg, g_max], dim=1)
        feats = F.normalize(feats, p=2, dim=1)
        ones = torch.ones(feats.shape[0], 1, device=self.device)
        return torch.cat([feats, ones], dim=1)

    def fit(self, X_cpu, y_cpu, batch_size=512):
        t_start = time.time()
        N = X_cpu.shape[0]
        Y_onehot = F.one_hot(y_cpu, num_classes=self.num_classes).float()

        # Conv blocks with mini-batch accumulation and memory reclamation
        self.conv1.fit(X_cpu, Y_onehot, batch_size=batch_size, spatial_subsample=2)
        A1 = self.conv1.forward(X_cpu, batch_size=batch_size)
        torch.cuda.empty_cache()

        self.conv2.fit(A1, Y_onehot, batch_size=batch_size, spatial_subsample=1)
        A2 = self.conv2.forward(A1, batch_size=batch_size)
        del A1; torch.cuda.empty_cache()

        self.conv3.fit(A2, Y_onehot, batch_size=batch_size, spatial_subsample=1)
        A3 = self.conv3.forward(A2, batch_size=batch_size)
        del A2; torch.cuda.empty_cache()

        # Batch-wise Dense FP solve to keep VRAM low
        feat_dim = self.Q_fc[0].shape[0]
        ATA_fc = torch.zeros(feat_dim, feat_dim, device=self.device)
        ATZ_fc = torch.zeros(feat_dim, self.hidden_mlp, device=self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            F_b = self._pool_features_batch(A3[start:end])
            yb = Y_onehot[start:end].to(self.device)

            Z_fc_b = torch.zeros(F_b.shape[0], self.hidden_mlp, device=self.device)
            for r in range(self.R):
                Z_r = torch.sign(F_b @ self.Q_fc[r]) + torch.sign(yb @ self.U_fc[r])
                Z_fc_b += self.gammas[r] * Z_r

            ATA_fc += F_b.T @ F_b
            ATZ_fc += F_b.T @ Z_fc_b

        I_fc = self.lam_fc * torch.eye(feat_dim, device=self.device)
        I_fc[-1, -1] = 1e-4
        self.W_fc = torch.linalg.solve(ATA_fc + I_fc, ATZ_fc)

        # Batch-wise Readout solve
        readout_dim = self.hidden_mlp + 1
        ATA_out = torch.zeros(readout_dim, readout_dim, device=self.device)
        ATY_out = torch.zeros(readout_dim, self.num_classes, device=self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            F_b = self._pool_features_batch(A3[start:end])
            yb = Y_onehot[start:end].to(self.device)
            H_b = F.relu(F_b @ self.W_fc)
            H_b = torch.cat([H_b, torch.ones(H_b.shape[0], 1, device=self.device)], dim=1)

            ATA_out += H_b.T @ H_b
            ATY_out += H_b.T @ yb

        del A3; torch.cuda.empty_cache()
        I_out = self.lam_fc * torch.eye(readout_dim, device=self.device)
        I_out[-1, -1] = 1e-4
        self.W_out = torch.linalg.solve(ATA_out + I_out, ATY_out)

        return time.time() - t_start

    @torch.no_grad()
    def predict(self, X_cpu, batch_size=512):
        logits_list = []
        for start in range(0, X_cpu.shape[0], batch_size):
            end = min(start + batch_size, X_cpu.shape[0])
            xb = X_cpu[start:end]
            a1 = self.conv1.forward(xb, batch_size=batch_size)
            a2 = self.conv2.forward(a1, batch_size=batch_size)
            a3 = self.conv3.forward(a2, batch_size=batch_size)
            fb = self._pool_features_batch(a3)
            hb = F.relu(fb @ self.W_fc)
            hb = torch.cat([hb, torch.ones(hb.shape[0], 1, device=self.device)], dim=1)
            logits_b = hb @ self.W_out
            logits_list.append(logits_b.cpu())
        return torch.cat(logits_list, dim=0)

# Backpropagation Baseline
class BackpropCNN(nn.Module):
    def __init__(self, in_channels, in_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        
        s = in_size // 8
        feat_dim = 256 * s * s
        self.fc1 = nn.Linear(feat_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_dataset(name, root="./data"):
    os.makedirs(root, exist_ok=True)
    if name == 'fashion_mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        return train_set, test_set, 1, 28, 10
    elif name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        return train_set, test_set, 3, 32, 10
    elif name == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
        train_set = datasets.SVHN(root=root, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root=root, split='test', download=True, transform=transform)
        return train_set, test_set, 3, 32, 10

def run_all_benchmarks():
    dataset_names = ['fashion_mnist', 'cifar10', 'svhn']
    results = []

    for dname in dataset_names:
        print(f"\n==========================================")
        print(f"   EVALUATING DATASET: {dname.upper()}")
        print(f"==========================================")
        train_set, test_set, in_channels, in_size, num_classes = get_dataset(dname)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)
        X_train, y_train = next(iter(train_loader)) # Keep on CPU

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        X_test, y_test = next(iter(test_loader)) # Keep on CPU

        # 1. FP (R=1)
        print("--> Running Original FP (R=1)...")
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        model_fp1 = GenericFPEnsembleCNN(in_channels, in_size, num_classes, R=1, device=device)
        t_fp1 = model_fp1.fit(X_train, y_train, batch_size=256)
        preds_fp1 = model_fp1.predict(X_test, batch_size=256)
        acc_fp1 = (preds_fp1.argmax(dim=1) == y_test).float().mean().item() * 100
        vram_fp1 = (torch.cuda.max_memory_allocated() / (1024**2)) if device == 'cuda' else 0
        print(f"FP (R=1): Acc = {acc_fp1:.2f}%, Time = {t_fp1:.2f}s, Peak VRAM = {vram_fp1:.1f}MB")
        del model_fp1; torch.cuda.empty_cache()

        # 2. FP-Ensemble (R=5)
        print("--> Running Proposed FP-Ensemble (R=5)...")
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        model_fp5 = GenericFPEnsembleCNN(in_channels, in_size, num_classes, R=5, device=device)
        t_fp5 = model_fp5.fit(X_train, y_train, batch_size=256)
        preds_fp5 = model_fp5.predict(X_test, batch_size=256)
        acc_fp5 = (preds_fp5.argmax(dim=1) == y_test).float().mean().item() * 100
        probs_fp5 = F.softmax(preds_fp5, dim=1).numpy()
        auc_fp5 = roc_auc_score(y_test.numpy(), probs_fp5, multi_class='ovr') * 100
        vram_fp5 = (torch.cuda.max_memory_allocated() / (1024**2)) if device == 'cuda' else 0
        print(f"FP-Ensemble (R=5): Acc = {acc_fp5:.2f}%, AUC = {auc_fp5:.2f}%, Time = {t_fp5:.2f}s, Peak VRAM = {vram_fp5:.1f}MB")
        del model_fp5; torch.cuda.empty_cache()

        # 3. Standard Backprop CNN
        print("--> Running Standard Backprop CNN (Adam, 15 epochs)...")
        bp_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        bp_model = BackpropCNN(in_channels, in_size, num_classes).to(device)
        optimizer = torch.optim.Adam(bp_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        t_bp_start = time.time()
        bp_model.train()
        for epoch in range(15):
            for xb, yb in bp_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = bp_model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        t_bp = time.time() - t_bp_start

        bp_model.eval()
        with torch.no_grad():
            bp_test_logits = []
            for start in range(0, X_test.shape[0], 512):
                end = min(start + 512, X_test.shape[0])
                xb = X_test[start:end].to(device)
                bp_test_logits.append(bp_model(xb).cpu())
            bp_test_logits = torch.cat(bp_test_logits, dim=0)
            acc_bp = (bp_test_logits.argmax(dim=1) == y_test).float().mean().item() * 100
        vram_bp = (torch.cuda.max_memory_allocated() / (1024**2)) if device == 'cuda' else 0
        print(f"Backprop CNN: Acc = {acc_bp:.2f}%, Time = {t_bp:.2f}s, Peak VRAM = {vram_bp:.1f}MB")
        del bp_model; torch.cuda.empty_cache()

        results.append({
            "Dataset": dname.upper(),
            "Orig_FP_Acc": acc_fp1,
            "Orig_FP_Time": t_fp1,
            "FP_Ens_Acc": acc_fp5,
            "FP_Ens_AUC": auc_fp5,
            "FP_Ens_Time": t_fp5,
            "FP_Ens_VRAM_MB": vram_fp5,
            "Backprop_Acc": acc_bp,
            "Backprop_Time": t_bp,
            "Backprop_VRAM_MB": vram_bp,
            "Speedup_vs_Backprop": t_bp / t_fp5
        })

    df = pd.DataFrame(results)
    out_csv = "C:/Users/R RAMKUMAR/.gemini/antigravity/scratch/FP-Ensemble-SignLanguage-MNIST/results/comprehensive_benchmarks.csv"
    df.to_csv(out_csv, index=False)
    print("\n\n================ FINAL MULTI-BENCHMARK RESULTS ================")
    print(df.to_string())
    print(f"Saved results to: {out_csv}")

if __name__ == '__main__':
    run_all_benchmarks()
