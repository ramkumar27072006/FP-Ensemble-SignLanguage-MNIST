"""
fp_ensemble_layer.py
====================
Architecture 3: Ensemble Random Projection Forward Projection (FP-Ensemble)
Dense MLP variant of the FP-Ensemble layer.

Reference: "Forward Projection with Ensemble Random Projections"
"""

import time
import torch
import torch.nn.functional as F


class FPEnsembleLayer:
    def __init__(self, in_features, out_features, num_classes, R=5, lam=1.0, weights=None, device='cpu'):
        """
        A single dense FP-Ensemble layer.

        Parameters
        ----------
        in_features  : int   – Input dimensionality.
        out_features : int   – Output (hidden) dimensionality.
        num_classes  : int   – Number of target classes.
        R            : int   – Number of independent random projection pairs.
        lam          : float – Ridge regularisation coefficient λ.
        weights      : list  – Optional fusion weights γ_r (must sum to 1).
                               Defaults to uniform 1/R.
        device       : str   – 'cpu' or 'cuda'.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.R = R
        self.lam = lam
        self.device = device

        # Fusion weights γ_r (Eq. 21-23)
        if weights is None:
            self.gammas = [1.0 / R] * R
        else:
            assert len(weights) == R and abs(sum(weights) - 1.0) < 1e-6, \
                "Weights must sum to 1"
            self.gammas = weights

        # R independent random projection pairs Q^(r), U^(r) (Eq. 17-18)
        self.Q_list = [
            torch.randn(in_features, out_features, device=device) / (in_features ** 0.5)
            for _ in range(R)
        ]
        self.U_list = [
            torch.randn(num_classes, out_features, device=device) / (num_classes ** 0.5)
            for _ in range(R)
        ]
        self.W = None

    def generate_ensemble_target(self, A_prev, Y_onehot):
        """
        Computes Z̃_ensemble = Σ_r γ_r · [sign(A_prev Q^(r)) + sign(Y U^(r))]
        (Eq. 4, 19-21).
        """
        Z_tilde_ensemble = torch.zeros(
            (A_prev.shape[0], self.out_features), device=self.device
        )
        for r in range(self.R):
            branch_target = (
                torch.sign(A_prev @ self.Q_list[r])
                + torch.sign(Y_onehot @ self.U_list[r])
            )
            Z_tilde_ensemble += self.gammas[r] * branch_target
        return Z_tilde_ensemble

    def fit(self, A_prev, Y_onehot):
        """
        Single closed-form Ridge Regression fit using the fused ensemble
        target (Eq. 11):
            W = (AᵀA + λI)⁻¹ Aᵀ Z̃_ensemble
        """
        t0 = time.time()
        Z_tilde = self.generate_ensemble_target(A_prev, Y_onehot)
        t_target = time.time() - t0

        I = torch.eye(self.in_features, device=self.device)
        ATA = A_prev.T @ A_prev + self.lam * I
        ATZ = A_prev.T @ Z_tilde
        self.W = torch.linalg.solve(ATA, ATZ)

        t_total = time.time() - t0
        return t_target, t_total

    def forward(self, A_prev, apply_relu=True):
        Z = A_prev @ self.W
        return F.relu(Z) if apply_relu else Z


class FPEnsembleMLP:
    def __init__(self, layer_dims, num_classes, R=5, lam=1.0, weights=None, device='cpu'):
        """
        Multi-layer dense FP-Ensemble network.

        Parameters
        ----------
        layer_dims  : list[int] – Sizes of each layer including input.
                                  E.g. [784, 1024, 512, 24].
        num_classes : int       – Number of output classes.
        R           : int       – Ensemble size per layer.
        lam         : float     – Ridge regularisation coefficient.
        weights     : list      – Optional per-projection fusion weights.
        device      : str       – 'cpu' or 'cuda'.
        """
        self.layers = []
        self.num_classes = num_classes
        self.device = device

        for i in range(len(layer_dims) - 1):
            self.layers.append(
                FPEnsembleLayer(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i + 1],
                    num_classes=num_classes,
                    R=R,
                    lam=lam,
                    weights=weights,
                    device=device
                )
            )

    def fit(self, X, y):
        """
        Layer-wise forward training in a single pass (no backpropagation).

        Returns
        -------
        (total_target_time, total_train_time) : tuple[float, float]
        """
        Y_onehot = F.one_hot(y, num_classes=self.num_classes).float().to(self.device)
        A = X.to(self.device)

        total_target_time = 0.0
        total_train_time = 0.0

        for idx, layer in enumerate(self.layers):
            t_target, t_layer = layer.fit(A, Y_onehot)
            total_target_time += t_target
            total_train_time += t_layer

            is_hidden = (idx < len(self.layers) - 1)
            A = layer.forward(A, apply_relu=is_hidden)

        return total_target_time, total_train_time

    @torch.no_grad()
    def predict(self, X):
        """Forward pass returning logits."""
        A = X.to(self.device)
        for idx, layer in enumerate(self.layers):
            is_hidden = (idx < len(self.layers) - 1)
            A = layer.forward(A, apply_relu=is_hidden)
        return A
