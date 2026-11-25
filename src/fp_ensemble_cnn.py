"""
fp_ensemble_cnn.py
==================
Architecture 3 – Convolutional FP-Ensemble Network for Sign Language MNIST.

Implements:
  - Conv2DFPEnsembleBlock : patch-based convolutional layer trained via
                            closed-form Ridge Regression with ensemble targets.
  - FPEnsembleCNN         : three-block convolutional backbone followed by a
                            dense FP-Ensemble hidden layer and a Ridge readout.
"""

import time
import torch
import torch.nn.functional as F


class Conv2DFPEnsembleBlock:
    """
    A convolutional block trained without backpropagation using the
    FP-Ensemble closed-form solution.

    Training flow
    -------------
    1.  Extract sliding patches from the input feature map (im2col / unfold).
    2.  For each of the R random projection pairs (Q^(r), U^(r)), compute the
        sign-based ensemble target Z^(r) = sign(A Q^(r)) + sign(Y U^(r)).
    3.  Accumulate AᵀA and AᵀZ̃ in mini-batches (memory-efficient).
    4.  Solve the Ridge system: W = (AᵀA + λI)⁻¹ AᵀZ̃.

    Inference flow
    --------------
    1.  Extract full patches.
    2.  Apply W → ReLU → 2×2 max-pool → instance normalise.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size,
        num_classes, R=5, lam=10.0,
        stride=1, padding=1, device='cpu'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.R = R
        self.lam = lam
        self.stride = stride
        self.padding = padding
        self.device = device

        # patch_dim = C_in * k * k + 1 (bias appended)
        self.patch_dim = in_channels * kernel_size * kernel_size + 1
        self.gammas = [1.0 / R] * R

        # Random projection matrices
        self.Q_list = [
            torch.randn(self.patch_dim, out_channels, device=device)
            / (self.patch_dim ** 0.5)
            for _ in range(R)
        ]
        self.U_list = [
            torch.randn(num_classes, out_channels, device=device)
            / (num_classes ** 0.5)
            for _ in range(R)
        ]
        self.W = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_patches(self, X, spatial_step=1):
        """
        Unfold X into overlapping patches and optionally spatially subsample.

        Returns
        -------
        patches : (N, L, patch_dim) tensor with bias column appended.
        H_out, W_out : output spatial dimensions (before subsampling).
        """
        N, C, H, W = X.shape

        unfold = F.unfold(
            X,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = unfold.transpose(1, 2)   # (N, L, C*k*k)

        if spatial_step > 1:
            patches = patches.reshape(N, H_out, W_out, -1)
            patches = patches[:, ::spatial_step, ::spatial_step, :]
            patches = patches.reshape(N, -1, C * self.kernel_size * self.kernel_size)

        # Append bias column of ones
        ones = torch.ones(
            (patches.shape[0], patches.shape[1], 1), device=self.device
        )
        patches = torch.cat([patches, ones], dim=-1)

        return patches, H_out, W_out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, Y_onehot, batch_size=512, spatial_subsample=2):
        """
        Closed-form mini-batch fit.

        Parameters
        ----------
        X                : (N, C, H, W) input tensor.
        Y_onehot         : (N, num_classes) one-hot labels.
        batch_size       : samples per accumulation step.
        spatial_subsample: stride used when sampling patches for training.

        Returns
        -------
        (t_target, t_total) : timing floats (seconds).
        """
        t0 = time.time()
        t_target = 0.0
        N = X.shape[0]

        ATA = torch.zeros(self.patch_dim, self.patch_dim, device=self.device)
        ATZ = torch.zeros(self.patch_dim, self.out_channels, device=self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb = X[start:end]
            yb = Y_onehot[start:end]

            patches, _, _ = self._extract_patches(xb, spatial_step=spatial_subsample)
            B, L, D = patches.shape
            A_flat = patches.reshape(B * L, D)

            Y_flat = (
                yb.unsqueeze(1)
                .expand(B, L, self.num_classes)
                .reshape(B * L, self.num_classes)
            )

            tt0 = time.time()
            Z_ens = torch.zeros(B * L, self.out_channels, device=self.device)
            for r in range(self.R):
                Z_r = (
                    torch.sign(A_flat @ self.Q_list[r])
                    + torch.sign(Y_flat @ self.U_list[r])
                )
                Z_ens += self.gammas[r] * Z_r
            t_target += time.time() - tt0

            ATA += A_flat.T @ A_flat
            ATZ += A_flat.T @ Z_ens

        # Adaptive regularisation scaled by dataset size
        reg = self.lam * (N / 1000.0) * torch.eye(self.patch_dim, device=self.device)
        reg[-1, -1] = 1e-4   # softer regularisation on bias

        self.W = torch.linalg.solve(ATA + reg, ATZ)
        return t_target, time.time() - t0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, X, batch_size=1024):
        """
        Full-resolution inference: patch → W → ReLU → max-pool → normalise.
        """
        outputs = []
        for start in range(0, X.shape[0], batch_size):
            xb = X[start:min(start + batch_size, X.shape[0])]
            patches, H_out, W_out = self._extract_patches(xb)

            Z = patches @ self.W   # (B, L, out_channels)
            A = F.relu(Z).transpose(1, 2).reshape(
                xb.shape[0], self.out_channels, H_out, W_out
            )
            A = F.max_pool2d(A, kernel_size=2, stride=2)
            A = (A - A.mean(dim=(2, 3), keepdim=True)) / (
                A.std(dim=(2, 3), keepdim=True) + 1e-5
            )
            outputs.append(A)

        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------


class FPEnsembleCNN:
    """
    Three-block convolutional FP-Ensemble network.

    Architecture
    ------------
    Input (1×28×28)
      ↓  Conv2DFPEnsembleBlock(1   → 64,  3×3) + MaxPool2d(2)  → 14×14
      ↓  Conv2DFPEnsembleBlock(64  → 128, 3×3) + MaxPool2d(2)  →  7×7
      ↓  Conv2DFPEnsembleBlock(128 → 256, 3×3) + MaxPool2d(2)  →  3×3
      ↓  Pool features: flatten(3×3) + GAP + GMP  → 256×11 + 1 (bias)
      ↓  Dense FP-Ensemble hidden layer (1024 units, Ridge)
      ↓  Ridge readout → num_classes logits
    """

    def __init__(
        self, num_classes, R=5,
        conv_channels=(64, 128, 256),
        hidden_mlp=1024,
        lam_conv=5.0, lam_fc=20.0,
        device='cpu'
    ):
        self.num_classes = num_classes
        self.R = R
        self.lam_fc = lam_fc
        self.device = device
        self.gammas = [1.0 / R] * R

        # Convolutional blocks
        self.conv1 = Conv2DFPEnsembleBlock(
            1, conv_channels[0], 3, num_classes, R=R, lam=lam_conv, device=device
        )
        self.conv2 = Conv2DFPEnsembleBlock(
            conv_channels[0], conv_channels[1], 3, num_classes, R=R, lam=lam_conv, device=device
        )
        self.conv3 = Conv2DFPEnsembleBlock(
            conv_channels[1], conv_channels[2], 3, num_classes, R=R, lam=lam_conv, device=device
        )

        # Feature dimension after 3 pooling stages: 28 → 14 → 7 → 3
        # flat(3×3) + GAP + GMP = (9+2) × 256 channels  (+1 bias)
        feat_dim = conv_channels[2] * (3 * 3 + 2) + 1
        self.hidden_mlp = hidden_mlp

        # Dense projection matrices for the FC FP-Ensemble layer
        self.Q_fc = [
            torch.randn(feat_dim, hidden_mlp, device=device) / (feat_dim ** 0.5)
            for _ in range(R)
        ]
        self.U_fc = [
            torch.randn(num_classes, hidden_mlp, device=device) / (num_classes ** 0.5)
            for _ in range(R)
        ]

        self.W_fc = None   # Dense hidden weights
        self.W_out = None  # Ridge readout weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pool_features(self, C3):
        """
        Fuse spatial features via flatten + GAP + GMP,
        L2-normalise, and append a bias column.
        """
        flat = C3.reshape(C3.shape[0], -1)         # (N, C*H*W)
        g_avg = C3.mean(dim=(2, 3))                # (N, C)
        g_max = C3.amax(dim=(2, 3))               # (N, C)
        feats = torch.cat([flat, g_avg, g_max], dim=1)
        feats = F.normalize(feats, p=2, dim=1)
        ones = torch.ones(feats.shape[0], 1, device=self.device)
        return torch.cat([feats, ones], dim=1)

    def _ridge_solve(self, A, B, lam):
        """Solve the Ridge system W = (AᵀA + λI)⁻¹ AᵀB."""
        I = lam * torch.eye(A.shape[1], device=self.device)
        I[-1, -1] = 1e-4   # softer bias regularisation
        return torch.linalg.solve(A.T @ A + I, A.T @ B)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train all layers layer-by-layer in a single forward pass.

        Parameters
        ----------
        X : (N, 1, 28, 28) normalised image tensor.
        y : (N,) long tensor of class indices.

        Returns
        -------
        (total_target_time, total_train_time) : tuple[float, float]
        """
        t_start = time.time()
        Y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)
        total_target_time = 0.0

        # --- Convolutional layers ---
        tt, _ = self.conv1.fit(X, Y_onehot, spatial_subsample=2)
        total_target_time += tt
        A1 = self.conv1.forward(X)

        tt, _ = self.conv2.fit(A1, Y_onehot, spatial_subsample=1)
        total_target_time += tt
        A2 = self.conv2.forward(A1)

        tt, _ = self.conv3.fit(A2, Y_onehot, spatial_subsample=1)
        total_target_time += tt
        A3 = self.conv3.forward(A2)

        # --- Dense FP-Ensemble hidden layer ---
        F_vec = self._pool_features(A3)

        tt0 = time.time()
        Z_fc_ens = torch.zeros(F_vec.shape[0], self.hidden_mlp, device=self.device)
        for r in range(self.R):
            Z_r = (
                torch.sign(F_vec @ self.Q_fc[r])
                + torch.sign(Y_onehot @ self.U_fc[r])
            )
            Z_fc_ens += self.gammas[r] * Z_r
        total_target_time += time.time() - tt0

        self.W_fc = self._ridge_solve(F_vec, Z_fc_ens, self.lam_fc)

        H_fc = F.relu(F_vec @ self.W_fc)
        H_fc = torch.cat(
            [H_fc, torch.ones(H_fc.shape[0], 1, device=self.device)], dim=1
        )

        # --- Ridge readout ---
        self.W_out = self._ridge_solve(H_fc, Y_onehot, self.lam_fc)

        return total_target_time, time.time() - t_start

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X):
        """Return raw logits for the input batch."""
        A1 = self.conv1.forward(X)
        A2 = self.conv2.forward(A1)
        A3 = self.conv3.forward(A2)

        F_vec = self._pool_features(A3)
        H_fc = F.relu(F_vec @ self.W_fc)
        H_fc = torch.cat(
            [H_fc, torch.ones(H_fc.shape[0], 1, device=self.device)], dim=1
        )
        return H_fc @ self.W_out
