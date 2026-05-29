import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def plot_loss(train_losses, val_losses, stationary_ratios, val_maes, val_maes_stationary, percentile_history, run_dir, show_plot=True):
    save_path = os.path.join(run_dir, "loss_plot.jpg")

    # 🔹 Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    # =========================
    # 🔹 Top plot: Loss curves
    # =========================

    # --- Training loss ---
    if len(train_losses) > 0:
        unique_lrs = sorted(set(lr for _, _, lr in train_losses))
        for lr in unique_lrs:
            lr_epochs = [e for (e, _, l) in train_losses if l == lr]
            lr_values = [v for (_, v, l) in train_losses if l == lr]
            ax1.plot(lr_epochs, lr_values, linewidth=1, label=f'Train (LR={lr:.1e})')
            ax1.text(lr_epochs[-1], lr_values[-1], f'{lr:.1e}', fontsize=6,
                     verticalalignment='bottom', horizontalalignment='left', rotation=90)

    # --- Validation loss ---
    if len(val_losses) > 0:
        unique_lrs = sorted(set(lr for _, _, lr in val_losses))
        for lr in unique_lrs:
            lr_epochs = [e for (e, _, l) in val_losses if l == lr]
            lr_values = [v for (_, v, l) in val_losses if l == lr]
            ax1.plot(lr_epochs, lr_values, linestyle='--', linewidth=1, label=f'Val (LR={lr:.1e})')
            ax1.text(lr_epochs[-1], lr_values[-1], f'{lr:.1e}', fontsize=6,
                     verticalalignment='bottom', horizontalalignment='left', rotation=90)

    ax1.set_ylabel("Loss")
    ax1.set_yscale('log')
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # =========================
    # 🔹 Bottom plot: Stationary ratio
    # =========================

    if len(stationary_ratios) > 0 and len(train_losses) > 0:
        # Use unique sorted epochs from training data
        epochs = sorted(set(e for (e, _, _) in train_losses))

        # Match lengths safely
        min_len = min(len(epochs), len(stationary_ratios))
        epochs = epochs[:min_len]
        values = stationary_ratios[:min_len]

        ax2.plot(epochs, values, color='purple', linewidth=1.5, label='Stationary Ratio')
        ax2.set_ylabel("Stationary Ratio")
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()

    # =========================
    # 🔹 Bottom plot: MAE curves
    # =========================

    if val_maes is not None and len(val_maes) > 0:
        unique_lrs = sorted(set(lr for _, _, lr in val_maes))
        for lr in unique_lrs:
            lr_epochs = [e for (e, _, l) in val_maes if l == lr]
            lr_values = [v for (_, v, l) in val_maes if l == lr]
            ax3.plot(lr_epochs, lr_values, linestyle='--', linewidth=1, label=f'Val MAE (LR={lr:.1e})')

    if val_maes_stationary is not None and len(val_maes_stationary) > 0:
        unique_lrs = sorted(set(lr for _, _, lr in val_maes_stationary))
        for lr in unique_lrs:
            lr_epochs = [e for (e, _, l) in val_maes_stationary if l == lr]
            lr_values = [v for (_, v, l) in val_maes_stationary if l == lr]
            ax3.plot(lr_epochs, lr_values, linestyle='--', linewidth=1, label=f'Val MAE Stationary (LR={lr:.1e})')

    ax3.set_ylabel("MAE")
    ax3.set_yscale('log')
    ax3.grid(True)
    ax3.legend()

    # =========================
    # 🔹 Percentile MAE Analysis (Line Plot)
    # =========================

    if percentile_history:

        # Step 1: group data by percentile
        percentile_series = {}

        for entry in percentile_history:
            epoch = entry["epoch"]
            values = entry["values"]

            for p, v in values.items():
                if p not in percentile_series:
                    percentile_series[p] = {"epochs": [], "values": []}

                percentile_series[p]["epochs"].append(epoch)
                percentile_series[p]["values"].append(v)

        # Step 2: sort by epoch (important for correct lines)
        for p in percentile_series:
            combined = sorted(
                zip(percentile_series[p]["epochs"], percentile_series[p]["values"])
            )
            epochs, values = zip(*combined)

            percentile_series[p]["epochs"] = epochs
            percentile_series[p]["values"] = values

        # Step 3: plot lines
        for p, data in percentile_series.items():
            ax4.plot(
                data["epochs"],
                data["values"],
                linestyle='--',
                linewidth=1,
                label=f'Percentile {p}'
            )

    ax4.set_ylabel("Percentile MAE")
    ax4.set_yscale('log')
    ax4.grid(True)
    ax4.legend()

    # =========================
    # 🔹 Save & show
    # =========================

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()

def plot_TD_loss(train_losses, stationary_losses, run_dir, show_plot=True):

    save_path = os.path.join(run_dir, "loss_plot.jpg")

    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True
    )

    # ========================================
    # TD Loss
    # ========================================

    ax1.plot(
        epochs,
        train_losses,
        color='tab:blue',
        linewidth=1.5,
        label='TD Loss'
    )

    ax1.set_ylabel("TD Loss")
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle=':')
    ax1.legend()

    # ========================================
    # Stationary Ratio
    # ========================================

    ax2.plot(
        epochs,
        stationary_losses,
        color='purple',
        linewidth=1.5,
        label='Stationary Loss'
    )

    ax2.set_ylabel("Stationary Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale('log')
    ax2.grid(True, which='both', linestyle=':')
    ax2.legend()

    fig.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    if show_plot:
        plt.show()

    plt.close(fig)

# def run_scaling(X=None, y=None, scaling_type=None, scaling_params=None,
#                 scaling_range_X=None, scaling_range_y=None, inverse=False):
#     """
#     Scale X and/or y using MinMaxnormalization.
#     Either X or y can be None if only one needs to be scaled/unscaled.

#     Args:
#         X (torch.Tensor, optional): Input features, shape (n_samples, n_features)
#         y (torch.Tensor, optional): Targets, shape (n_samples, 1)
#         scaling_type (str): "standardize" or "normalize"
#         scaling_params (dict, optional): Precomputed statistics (mean/std or min/max)
#         scaling_range_X (torch.Tensor or list, optional): Target range per feature for normalization
#         scaling_range_y (torch.Tensor or list, optional): Target range for y
#         inverse (bool): If True, performs inverse scaling

#     Returns:
#         X_scaled, y_scaled (torch.Tensor or None)
#     """

#     eps = 1e-8

#     # Only normalize is implemented here
#     if scaling_type == "normalize":
#         if scaling_params is None:
#             raise ValueError("scaling_params must be provided for normalization.")
#         if (X is not None and scaling_range_X is None) or (y is not None and scaling_range_y is None):
#             raise ValueError("scaling_range_X/y must be provided for normalization.")

#         # Extract min/max
#         X_min, X_max = scaling_params.get("X_min"), scaling_params.get("X_max")
#         y_min, y_max = scaling_params.get("y_min"), scaling_params.get("y_max")

#         # Forward scaling
#         if not inverse:
#             if X is not None:
#                 scaling_range_X = torch.tensor(scaling_range_X, dtype=X.dtype, device=X.device)
#                 X_scaled = (X - X_min) / (X_max - X_min + eps)
#                 X_scaled = X_scaled * (scaling_range_X[:, 1] - scaling_range_X[:, 0]) + scaling_range_X[:, 0]
#             else:
#                 X_scaled = None

#             if y is not None:
#                 scaling_range_y = torch.tensor(scaling_range_y, dtype=y.dtype, device=y.device)
#                 y_scaled = (y - y_min) / (y_max - y_min + eps)
#                 y_scaled = y_scaled * (scaling_range_y[1] - scaling_range_y[0]) + scaling_range_y[0]
#             else:
#                 y_scaled = None

#         # Inverse scaling
#         else:
#             if X is not None:
#                 scaling_range_X = torch.tensor(scaling_range_X, dtype=X.dtype, device=X.device)
#                 X_scaled = (X - scaling_range_X[:, 0]) / (scaling_range_X[:, 1] - scaling_range_X[:, 0] + eps)
#                 X_scaled = X_scaled * (X_max - X_min) + X_min
#             else:
#                 X_scaled = None

#             if y is not None:
#                 scaling_range_y = torch.tensor(scaling_range_y, dtype=y.dtype, device=y.device)
#                 y_scaled = (y - scaling_range_y[0]) / (scaling_range_y[1] - scaling_range_y[0] + eps)
#                 y_scaled = y_scaled * (y_max - y_min) + y_min
#             else:
#                 y_scaled = None

#     else:
#         raise ValueError(f"Unknown scaling_type: {scaling_type}")

#     return X_scaled, y_scaled

def run_scaling(
    X: torch.Tensor = None,
    y: torch.Tensor = None,
    scaling_type: str = "normalize",
    scaling_range_from_X=None,
    scaling_range_from_y=None,
    scaling_range_to_X=None,
    scaling_range_to_y=None,
    inverse: bool = False
):
    """
    Modified so that:
      scaling_range_from_X[0] = all feature mins
      scaling_range_from_X[1] = all feature maxs
      scaling_range_to_X[0]   = all feature output mins
      scaling_range_to_X[1]   = all feature output maxs
    """

    def scale_forward(data, from_min, from_max, to_min, to_max):
        return (data - from_min) / (from_max - from_min + 1e-12) * (to_max - to_min) + to_min

    def scale_inverse(data, from_min, from_max, to_min, to_max):
        return (data - to_min) / (to_max - to_min + 1e-12) * (from_max - from_min) + from_min

    X_out, y_out = None, None

    # -----------------------------------------------------
    # Process X
    # -----------------------------------------------------
    if X is not None:
        X = X.float()

        # NEW: interpret rows as shared scalar ranges
        from_min_X = torch.tensor(scaling_range_from_X[0], dtype=torch.float32)
        from_max_X = torch.tensor(scaling_range_from_X[1], dtype=torch.float32)

        # NEW: interpret to_min_X/to_max_X as shared bounds for all features
        to_min_value = scaling_range_to_X[0][0]  # first element only
        to_max_value = scaling_range_to_X[1][0]  # first element only
        to_min_X = torch.full_like(from_min_X, to_min_value)
        to_max_X = torch.full_like(from_max_X, to_max_value)

        if not inverse:
            X_out = scale_forward(X, from_min_X, from_max_X, to_min_X, to_max_X)
        else:
            X_out = scale_inverse(X, from_min_X, from_max_X, to_min_X, to_max_X)

    # -----------------------------------------------------
    # Process y
    # -----------------------------------------------------
    if y is not None:
        y = y.float()
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # single min/max
        from_min_y = torch.tensor([scaling_range_from_y[0]])
        from_max_y = torch.tensor([scaling_range_from_y[1]])

        to_min_y = torch.tensor([scaling_range_to_y[0]])
        to_max_y = torch.tensor([scaling_range_to_y[1]])

        if not inverse:
            y_out = scale_forward(y, from_min_y, from_max_y, to_min_y, to_max_y)
        else:
            y_out = scale_inverse(y, from_min_y, from_max_y, to_min_y, to_max_y)

        y_out = y_out.squeeze(1)

    return X_out, y_out