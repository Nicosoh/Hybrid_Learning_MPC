import os
import yaml

from TD.replay_buffer import ReplayBuffer
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from utils import get_num_config
from neural_network.utils import plot_TD_loss
from neural_network.losses import StationaryLoss

from neural_network.models import MODEL_REGISTRY

def train_model_TD(train_config, run_dir, capacity, data_path=None, seed=42):
    # === Set random seed ===
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    online_model_save_path = os.path.join(run_dir, "online_model/online_model.pt")
    target_model_save_path = os.path.join(run_dir, "target_model/target_model.pt")

    # === Config ===
    # Training
    learning_rate = get_num_config("TRAINING", "learning_rate", train_config)
    batch_size    = train_config.getint("TRAINING", "batch_size")
    gradient_steps = train_config.getint("TRAINING", "gradient_steps")
    target_tau = train_config.getfloat("TRAINING", "target_tau")
    target_update_frequency = train_config.getint("TRAINING", "target_update_frequency")

    # Overwrite data path if provided
    if data_path is not None:
        train_config.set("DATA", "data_path", data_path)

    # Data
    # dataset_class = train_config.get("DATA", "dataset_class")
    # log_space = train_config.getboolean("DATA", "log_space")

    # Model
    model_name = train_config.get("MODEL", "model_name")
    target_model_checkpoint_path = train_config.get("MODEL", "target_model_checkpoint_path")
    online_model_checkpoint_path = train_config.get("MODEL", "online_model_checkpoint_path")

    # Loss
    alpha = torch.tensor(train_config.getfloat("LOSS", "alpha"), dtype=torch.float32)

    # Config Model Name
    config_model_name = train_config.get("CONFIG_MODEL_NAME", "model_name")

    # Load Model config
    with open(f"configs/{config_model_name}config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    # === Create dataset + dataloader ===
    # Init empty replay buffer
    replay_buffer = ReplayBuffer(capacity=capacity)

    replay_buffer.reinstate_buffer(data_path)

    # === Update train_config.ini with dataset statistics ===
    train_config.set("DATA", "replay_buffer_size", str(replay_buffer.size))

    # Save updated config
    config_save_path = os.path.join(run_dir, "train_config.ini")
    with open(config_save_path, "w") as f:
        train_config.write(f)

    # === Model / optimizer / loss ===
    # Load model dynamically
    ModelClass = MODEL_REGISTRY[model_name]
    target_model = ModelClass(train_config).to(device)
    target_model.load_state_dict(torch.load(target_model_checkpoint_path, map_location=device))
    target_model.eval()

    ModelClass = MODEL_REGISTRY[model_name]
    online_model = ModelClass(train_config).to(device)
    online_model.load_state_dict(torch.load(online_model_checkpoint_path, map_location=device))

    optimizer = torch.optim.Adam(online_model.parameters(), lr=learning_rate)
    criterion = StationaryLoss(alpha=alpha)

    # ============================================
    # Logging
    # ============================================

    train_losses = []
    stationary_ratios = []

    pbar = tqdm(range(gradient_steps), desc="Training")

    # ============================================
    # Training
    # ============================================

    for gradient_step in pbar:

        online_model.train()

        # Sample batch
        X_t, X_tpn, cost_t, Xs, ys = replay_buffer.sample(batch_size)

        X_t = X_t.to(device)
        X_tpn = X_tpn.to(device)
        cost_t = cost_t.to(device)
        Xs = Xs.to(device)
        ys = ys.to(device)

        # TD target
        with torch.no_grad():
            next_pred = target_model(X_tpn)
            target = cost_t + next_pred

        # Forward
        optimizer.zero_grad()

        current_pred = online_model(X_t)
        preds_stationary = online_model(Xs)

        # Loss
        loss, loss1, loss2 = criterion(
            current_pred,
            target,
            preds_stationary,
            ys
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10.0)

        optimizer.step()

        # Soft update
        if gradient_step % target_update_frequency == 0:
            with torch.no_grad():
                for p, p_target in zip(online_model.parameters(), target_model.parameters()):
                    p_target.data.lerp_(p.data, target_tau)

        # ----------------------------
        # Logging (FIXED)
        # ----------------------------
        train_losses.append(loss.item())

        stationary_ratios.append(
            (loss2.item() / (loss1.item() + 1e-8))
        )

        pbar.set_postfix({
            "train_loss": f"{loss.item():.4e}",
            "stationary": f"{stationary_ratios[-1]:.4f}"
        })

    # ========================================
    # Save checkpoint at the end of training
    # ========================================

    torch.save(
        online_model.state_dict(),
        online_model_save_path
    )
    torch.save(
        target_model.state_dict(),
        target_model_save_path
    )
    pbar.close()

    # === Plot loss curves ===
    if data_path is not None:
        show_plot = False
    else:
        show_plot = True

    stationary_ratios_mean = float(np.mean(stationary_ratios))

    plot_TD_loss(train_losses, stationary_ratios, run_dir=run_dir, show_plot=show_plot)

    return train_losses[-1], stationary_ratios_mean