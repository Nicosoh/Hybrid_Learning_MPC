import torch
import numpy as np

from torch.utils.data import Dataset, random_split
from data_collection import load_npz
from utils import get_num_config

class PendulumDataset(Dataset):
    """
    Dataset for Pendulum:
    Inputs X = [qpos, qvel]
    Targets y = cost
    Handles multiple runs inside the data dictionary.

    Optional train/test split using `split_ratio`.
    """
    def __init__(self, config, run_dir, mode, test_config=None):
        self.run_dir = run_dir
        self.mode = mode
        self.log_space = config.getboolean("DATA", "log_space")

        # =============================================================
        #                     TRAIN MODE
        # =============================================================
        if mode == "train":
            # Load stationary point config 
            self.Xs_config = torch.tensor(get_num_config("LOSS", "x_s", config), dtype=torch.float32)
            self.ys_config = torch.tensor(get_num_config("LOSS", "y_s", config), dtype=torch.float32)
            
            # Load train data
            data_path = config.get("DATA", "data_path")
            data = load_npz(data_path)
            self.preprocess_data(data)  # sets self.X and self.y

            # Train/val split
            self.train_val_data()

        # =============================================================
        #                     TEST MODE
        # =============================================================
        elif mode == "test":

            if test_config is None:
                raise ValueError("test_config must be provided for test mode")

            # Load test data
            data_path = test_config.get("TEST", "test_data_path")
            data = load_npz(data_path)
            self.preprocess_data(data)

    def preprocess_data(self, data):
        X_list = []
        Xs_list = []
        y_list = []
        ys_list = []

        for run_key in data.keys():  # iterate over each run
            run_data = data[run_key]
            qpos = run_data["qpos"]
            qvel = run_data["qvel"]
            cost = run_data["total_cost"]

            if self.log_space:
                cost = np.log1p(run_data["total_cost"])
            else:
                cost = run_data["total_cost"]

            # Concatenate qpos and qvel
            X_run = np.concatenate([qpos, qvel], axis=1)
            X_list.append(X_run)

            # Ensure cost is 2D
            if len(cost.shape) == 1:
                y_run = cost[:, None]
            else:
                y_run = cost
            y_list.append(y_run)

            Xs_list.append(np.tile(self.Xs_config, (qpos.shape[0], 1)))
            ys_list.append(np.tile(self.ys_config, (qpos.shape[0], 1)))

        # Stack all runs together
        self.X = torch.from_numpy(np.vstack(X_list)).float()
        self.Xs = torch.from_numpy(np.vstack(Xs_list)).float()
        self.y = torch.from_numpy(np.vstack(y_list)).float()
        self.ys = torch.from_numpy(np.vstack(ys_list)).float()
    
    def train_val_data(self, val_split=0.2, seed=42):
        dataset_size = len(self.X)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.train_dataset, self.val_dataset = random_split(
            self, [train_size, val_size], generator=generator
        )
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Xs[idx], self.y[idx], self.ys[idx]