"""
Subprocess worker for a single TD loop.
Runs data collection + training for one iteration.
Writes metrics to JSON for main process to read.
Subprocess isolation ensures all torch/FX allocations are freed on exit.
"""
import os
import sys
import json
import glob
import yaml
import configparser
from datetime import datetime

# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TD.replay_buffer import ReplayBuffer
from neural_network.scripts import train_model_TD
from neural_network.datasets import DATASET_REGISTRY
from data_collection.data_utils import load_npz
from data_collection.data_collector import run_data_collector

def log_worker(log_path, message):
    """Append timestamped message to worker log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    with open(log_path, "a") as f:
        f.write(line)


def run_td_loop(loop_num, main_output_dir, model_name, data_config_path, 
                train_config_path, TD_config_path):
    """
    Run a single TD loop: data collection + training.
    
    Args:
        loop_num: int, loop number (0-indexed)
        main_output_dir: str, parent output directory
        model_name: str, model name
        data_config_path: str, path to data config
        train_config_path: str, path to train config
        TD_config_path: str, path to TD config
    
    Returns:
        dict: metrics {gt_cost, ctrl_cost, mse} or None on error
    """
    
    # Setup directories
    loop_dir = os.path.join(main_output_dir, f"loop_{loop_num+1}")
    os.makedirs(loop_dir, exist_ok=True)
    
    data_dir = os.path.join(loop_dir, "data")
    data_collection_dir = os.path.join(data_dir, "data_collection")
    replay_buffer_dir = os.path.join(data_dir, "replay_buffer")

    training_dir = os.path.join(loop_dir, "training")
    target_model_dir = os.path.join(loop_dir, "training/target_model")
    online_model_dir = os.path.join(loop_dir, "training/online_model")

    os.makedirs(data_collection_dir, exist_ok=True)
    os.makedirs(replay_buffer_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(target_model_dir, exist_ok=True)
    os.makedirs(online_model_dir, exist_ok=True)

    worker_log_path = os.path.join(loop_dir, "worker.log")
    metrics_path = os.path.join(loop_dir, "metrics.json")

    # -----------------------------
    # Redirect stdout / stderr
    # -----------------------------
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    log_f = open(worker_log_path, "a")

    sys.stdout = log_f
    sys.stderr = log_f

    try:
        log_worker(worker_log_path, f"Starting TD Loop {loop_num+1}")
        
        # Load TD config
        with open(TD_config_path, "r") as f:
            TD_config = yaml.safe_load(f)
        
        # Load Model config
        with open(f"configs/{model_name}config.yaml", "r") as f:
            model_config = yaml.safe_load(f)
        
        # Load Train config
        train_config = configparser.ConfigParser()
        train_config.read(train_config_path)
        dataset_class = train_config.get("DATA", "dataset_class")
        DatasetClass = DATASET_REGISTRY[dataset_class]
        
        # Configure for this loop
        if loop_num == 0:
            model_config["mpc"]["controller_name"] = TD_config["NN_controller_name"]
            model_config["mpc"]["terminal_cost"] = True
            model_config["NN"]["checkpoint_path"] = TD_config["pretrained_weights_path"]
            train_config.set("MODEL", "load_checkpoint", "True")
            train_config.set("MODEL", "target_model_checkpoint_path", TD_config["pretrained_weights_path"])
            train_config.set("MODEL", "online_model_checkpoint_path", TD_config["pretrained_weights_path"])
            train_config.set("MODEL", "offline_model_checkpoint_path", TD_config["pretrained_weights_path"])
        else:
            # Load from previous loop's best model and replay buffer
            prev_training_dir = os.path.join(main_output_dir, f"loop_{loop_num}", "training")
            prev_target_model_dir = os.path.join(prev_training_dir, "target_model")
            prev_online_model_dir = os.path.join(prev_training_dir, "online_model")

            prev_data_dir = os.path.join(main_output_dir, f"loop_{loop_num}", "data")
            prev_replay_buffer_dir = os.path.join(prev_data_dir, "replay_buffer")

            prev_train_config_path = os.path.join(main_output_dir, f"loop_{loop_num}", "training", "train_config.ini")
            prev_train_config = configparser.ConfigParser()
            prev_train_config.read(prev_train_config_path)
            prev_lam = prev_train_config.get("LOSS", "lam")
            ema_mse = prev_train_config.getfloat("LOSS", "ema_mse")
            ema_mse_tau = prev_train_config.getfloat("LOSS", "ema_mse_tau")

            online_pt_files = glob.glob(os.path.join(prev_online_model_dir, "*.pt"))
            target_pt_files = glob.glob(os.path.join(prev_target_model_dir, "*.pt"))
            prev_replay_buffer_npz_files = glob.glob(os.path.join(prev_replay_buffer_dir, "*.npz"))

            if len(online_pt_files) and len(target_pt_files) and len(prev_replay_buffer_npz_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one .pt/ .npz file in {prev_training_dir}, "
                    f"found {len(online_pt_files)} in online model dir and {len(target_pt_files)} in target model dir and {len(prev_replay_buffer_npz_files)} in replay buffer dir"
                )
            
            target_model_path = target_pt_files[0]
            online_model_path = online_pt_files[0]
            prev_replay_buffer_path = prev_replay_buffer_npz_files[0]
            train_config.set("LOSS", "lam", prev_lam)
            train_config.set("MODEL", "load_checkpoint", "True")
            train_config.set("MODEL", "target_model_checkpoint_path", target_model_path)
            train_config.set("MODEL", "offline_model_checkpoint_path", TD_config["pretrained_weights_path"])
            train_config.set("MODEL", "online_model_checkpoint_path", online_model_path)
            model_config["mpc"]["controller_name"] = TD_config["NN_controller_name"]
            model_config["mpc"]["terminal_cost"] = True
            model_config["NN"]["checkpoint_path"] = online_model_path

        # -----------------
        # Data Collection
        # -----------------
        log_worker(worker_log_path, "Starting data collection")
        run_data_collector(
            model_name,
            data_config_path=data_config_path,
            run_dir=data_collection_dir,
            config=model_config,
        )
        log_worker(worker_log_path, f"Data collection completed")
        
        # Find npz file
        npz_files = glob.glob(os.path.join(data_collection_dir, "*.npz"))
        if len(npz_files) != 1:
            raise RuntimeError(
                f"Expected exactly one .npz file in {data_collection_dir}, "
                f"found {len(npz_files)}"
            )
        data_collection_npz_path = npz_files[0]

        # Extract metrics
        log_worker(worker_log_path, "Extracting metrics")
        data = load_npz(data_collection_npz_path)
        
        ctrl_cost = []
        GT_cost = []
        sq_errors = []
        
        for run_key in data.keys():
            run_data = data[run_key]
            GT = np.array(run_data["GT_cost"])
            CTRL = np.array(run_data["terminal_cost"])
            
            # Pairwise validity mask
            valid_mask = np.isfinite(GT) & np.isfinite(CTRL)

            # Skip runs with no valid paired data
            if not np.any(valid_mask):
                continue

            GT_valid = GT[valid_mask]
            CTRL_valid = CTRL[valid_mask]

            GT_cost.extend(GT_valid)
            ctrl_cost.extend(CTRL_valid)
            sq_errors.extend((GT_valid - CTRL_valid) ** 2)
        
        gt_mean = float(np.nanmean(GT_cost)) if GT_cost else np.nan
        ctrl_mean = float(np.nanmean(ctrl_cost)) if ctrl_cost else np.nan
        mse_mean = float(np.nanmean(sq_errors)) if sq_errors else np.nan
        mse_std = float(np.nanstd(sq_errors)) if sq_errors else np.nan

        # Process mse_mean to use it as the "reward" signal for TD learning
        if loop_num == 0:
            new_ema_mse = mse_mean  # Initialize EMA with first loop's MSE
        else:
            new_ema_mse = ema_mse * (1 - ema_mse_tau) + mse_mean * ema_mse_tau

        train_config.set("LOSS", "current_mse", str(mse_mean))
        train_config.set("LOSS", "ema_mse", str(new_ema_mse))
        
        # Preprocess fresh data
        dataset = DatasetClass(train_config, model_config, data_collection_npz_path)
        
        # Load previous replay buffer and add new data
        replay_buffer = ReplayBuffer(TD_config["replay_buffer_capacity"])
        replay_buffer_path = os.path.join(replay_buffer_dir, "replay_buffer.npz")

        if loop_num == 0:
            replay_buffer.add_trajectory(dataset.X_t, dataset.X_tpn, dataset.cost_n, dataset.Xs, dataset.ys)
            replay_buffer.save_buffer(replay_buffer_path)
        elif loop_num > 0:
            replay_buffer.reinstate_buffer(prev_replay_buffer_path)
            replay_buffer.add_trajectory(dataset.X_t, dataset.X_tpn, dataset.cost_n, dataset.Xs, dataset.ys)
            replay_buffer.save_buffer(replay_buffer_path)

        # -----------------
        # Training
        # -----------------
        log_worker(worker_log_path, "Starting model training")
        main_losses_mean, td_losses_mean, offline_losses_mean, stationary_losses_mean, lam = train_model_TD(
            train_config,
            run_dir=training_dir,
            capacity=TD_config["replay_buffer_capacity"],
            data_path=replay_buffer_path,
            seed=np.random.randint(0, 100000)
        )
        
        metrics = {
            "loop": loop_num + 1,
            "gt_cost": gt_mean,
            "ctrl_cost": ctrl_mean,
            "mse": mse_mean,
            "mse_std": mse_std,
            "success": True,
            "main_loss": main_losses_mean,
            "td_loss": td_losses_mean,
            "offline_loss": offline_losses_mean,
            "stationary_loss": stationary_losses_mean,
            "lam": lam,
            "ema_mse": new_ema_mse
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        log_worker(worker_log_path, "Model training completed")
        log_worker(worker_log_path, f"Metrics: Train Loss={main_losses_mean:.4f}, TD Loss={td_losses_mean:.4f}, Offline Loss={offline_losses_mean:.4f}, Stationary Loss={stationary_losses_mean:.4f}")

    except Exception as e:
        log_worker(worker_log_path, f"ERROR: {e}")
        import traceback
        log_worker(worker_log_path, traceback.format_exc())
        
        metrics = {
            "loop": loop_num + 1,
            "success": False,
            "error": str(e),
            "main_loss": None,
            "td_loss": None,
            "offline_loss": None,
            "stationary_loss": None,
            "lam": None,
            "ema_mse": None
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    finally:
        # Restore stdout/stderr and close file
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_f.close()

    return metrics


if __name__ == "__main__":
    import numpy as np
    
    if len(sys.argv) < 7:
        print(f"Usage: {sys.argv[0]} <loop_num> <main_output_dir> <model_name> "
              "<data_config_path> <train_config_path> <TD_config_path>")
        sys.exit(1)
    
    loop_num = int(sys.argv[1])
    main_output_dir = sys.argv[2]
    model_name = sys.argv[3]
    data_config_path = sys.argv[4]
    train_config_path = sys.argv[5]
    TD_config_path = sys.argv[6]
    
    metrics = run_td_loop(
        loop_num, main_output_dir, model_name, data_config_path,
        train_config_path, TD_config_path
    )
    
    # Exit with 0 on success, 1 on failure
    sys.exit(0 if metrics and metrics.get("success") else 1)
