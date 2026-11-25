"""
train.py
========
Full training + evaluation script for FP-Ensemble CNN on Sign Language MNIST.

Usage (local)
-------------
    python train.py --data_dir "E:/C_Downloads/Sign Language MNIST" \
                    --ensemble_sizes 1 3 5 8 --seeds 0 1 2

Usage (Colab / no argparse)
----------------------------
    Set DATA_DIR at the top of the script and run all cells.
"""

import os
import sys
import time
import argparse
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# --- Allow running from project root or src/ ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fp_ensemble_cnn import FPEnsembleCNN

# ===========================================================================
# Configuration (edit here when not using CLI)
# ===========================================================================
DATA_DIR = "E:/C_Downloads/Sign Language MNIST"   # override with --data_dir
ENSEMBLE_SIZES = [1, 3, 5, 8]
SEEDS = [0, 1, 2]
CONV_CHANNELS = (64, 128, 256)
HIDDEN_MLP = 1024
LAM_CONV = 5.0
LAM_FC = 20.0
R_DEFAULT = 5


# ===========================================================================
# Data utilities
# ===========================================================================

def find_csvs(root):
    """Walk *root* and return (train_csv, test_csv) paths."""
    train_csv = test_csv = None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() == "sign_mnist_train.csv":
                train_csv = os.path.join(dirpath, f)
            elif f.lower() == "sign_mnist_test.csv":
                test_csv = os.path.join(dirpath, f)
    return train_csv, test_csv


def maybe_extract_zip(data_dir):
    """
    If CSV files are not present directly, look for a ZIP archive and
    extract it into *data_dir*.
    """
    train_csv, test_csv = find_csvs(data_dir)
    if train_csv and test_csv:
        return train_csv, test_csv

    # Try to find a ZIP file
    zip_path = None
    for f in os.listdir(data_dir):
        if f.lower().endswith(".zip"):
            zip_path = os.path.join(data_dir, f)
            break

    if zip_path is None:
        raise FileNotFoundError(
            f"No ZIP archive found in {data_dir!r}.\n"
            "Place 'Sign Language MNIST.zip' (or extracted CSVs) there."
        )

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path!r} is not a valid ZIP file.")

    extract_to = os.path.join(data_dir, "extracted")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupted ZIP entry: {bad}")
        z.extractall(extract_to)

    train_csv, test_csv = find_csvs(extract_to)
    if train_csv is None or test_csv is None:
        raise FileNotFoundError(
            "Could not find sign_mnist_train.csv / sign_mnist_test.csv "
            f"inside the ZIP extracted to {extract_to!r}."
        )
    return train_csv, test_csv


def preprocess_images(X):
    """Per-image zero-mean / unit-std normalisation."""
    imgs = X.reshape(-1, 1, 28, 28)
    mean = imgs.mean(axis=(2, 3), keepdims=True)
    std = imgs.std(axis=(2, 3), keepdims=True) + 1e-5
    return (imgs - mean) / std


def load_dataset(data_dir, device):
    """
    Load the Sign Language MNIST dataset from *data_dir*.

    Returns
    -------
    X_train, X_test  : (N, 1, 28, 28) float32 torch tensors on *device*.
    y_train_t, y_test_t : (N,) long tensors on *device*.
    y_test           : numpy array of raw test labels (for AUC).
    num_classes      : int.
    """
    train_csv, test_csv = maybe_extract_zip(data_dir)
    print(f"Train CSV : {train_csv}")
    print(f"Test  CSV : {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    y_train_raw = train_df["label"].values
    X_train_raw = train_df.iloc[:, 1:].values.astype(np.float32)
    y_test_raw = test_df["label"].values
    X_test_raw = test_df.iloc[:, 1:].values.astype(np.float32)

    # Map labels to contiguous indices (handles missing 'J'=9 & 'Z'=25)
    unique_labels = np.unique(y_train_raw)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[x] for x in y_train_raw])
    y_test = np.array([label_map[x] for x in y_test_raw])
    num_classes = len(unique_labels)

    X_train = torch.tensor(preprocess_images(X_train_raw), dtype=torch.float32, device=device)
    X_test = torch.tensor(preprocess_images(X_test_raw), dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}  |  Classes : {num_classes}")
    return X_train, X_test, y_train_t, y_test_t, y_test, num_classes


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(model, X_train, y_train_t, X_test, y_test_t, y_test_np):
    train_logits = model.predict(X_train)
    train_acc = (train_logits.argmax(dim=1) == y_train_t).float().mean().item() * 100

    test_logits = model.predict(X_test)
    test_acc = (test_logits.argmax(dim=1) == y_test_t).float().mean().item() * 100

    test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
    test_auc = roc_auc_score(y_test_np, test_probs, multi_class="ovr") * 100

    return train_acc, test_acc, test_auc


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_experiment(
    data_dir,
    ensemble_sizes=None,
    seeds=None,
    device_str=None,
    conv_channels=(64, 128, 256),
    hidden_mlp=1024,
    lam_conv=5.0,
    lam_fc=20.0,
):
    ensemble_sizes = ensemble_sizes or ENSEMBLE_SIZES
    seeds = seeds or SEEDS

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = device_str
    print(f"Using device: {device}\n")

    X_train, X_test, y_train_t, y_test_t, y_test_np, num_classes = load_dataset(
        data_dir, device
    )

    results = []

    for R in ensemble_sizes:
        label = "Original FP (R=1)" if R == 1 else f"FP-Ensemble (R={R})"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        train_accs, test_accs, test_aucs, t_targets, t_totals = [], [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            model = FPEnsembleCNN(
                num_classes=num_classes,
                R=R,
                conv_channels=conv_channels,
                hidden_mlp=hidden_mlp,
                lam_conv=lam_conv,
                lam_fc=lam_fc,
                device=device,
            )

            t_target, t_total = model.fit(X_train, y_train_t)
            train_acc, test_acc, test_auc = evaluate(
                model, X_train, y_train_t, X_test, y_test_t, y_test_np
            )

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_aucs.append(test_auc)
            t_targets.append(t_target)
            t_totals.append(t_total)

            print(
                f"  Seed {seed} | Train {train_acc:.2f}% | "
                f"Test {test_acc:.2f}% | AUC {test_auc:.2f}% | "
                f"Time {t_total:.2f}s"
            )

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        results.append({
            "Architecture": label,
            "R": R,
            "Train Acc (%)": f"{np.mean(train_accs):.2f} ± {np.std(train_accs):.2f}",
            "Test Acc (%)": f"{np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}",
            "Test AUC (%)": f"{np.mean(test_aucs):.2f} ± {np.std(test_aucs):.2f}",
            "Target Gen Time (s)": f"{np.mean(t_targets):.2f}",
            "Total Train Time (s)": f"{np.mean(t_totals):.2f}",
        })

    summary_df = pd.DataFrame(results)
    print("\n\n=== Final Comparative Results ===")
    print(summary_df.to_string(index=False))

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "results", "summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return summary_df


# ===========================================================================
# CLI entry-point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FP-Ensemble CNN on Sign Language MNIST"
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help="Path to the Sign Language MNIST folder (or ZIP parent folder)."
    )
    parser.add_argument(
        "--ensemble_sizes", nargs="+", type=int, default=ENSEMBLE_SIZES,
        help="List of R values to compare. E.g. --ensemble_sizes 1 3 5 8"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds for reproducibility."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cpu' or 'cuda'. Auto-detected if omitted."
    )
    parser.add_argument("--lam_conv", type=float, default=LAM_CONV)
    parser.add_argument("--lam_fc", type=float, default=LAM_FC)
    parser.add_argument("--hidden_mlp", type=int, default=HIDDEN_MLP)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        data_dir=args.data_dir,
        ensemble_sizes=args.ensemble_sizes,
        seeds=args.seeds,
        device_str=args.device,
        hidden_mlp=args.hidden_mlp,
        lam_conv=args.lam_conv,
        lam_fc=args.lam_fc,
    )
