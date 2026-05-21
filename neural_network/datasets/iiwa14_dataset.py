import torch

import numpy as np

from torch.utils.data import Dataset, random_split
from data_collection import load_npz
from utils import get_num_config

class iiwa14_eeTracker(Dataset):
    """
    Dataset for TwoDofArm:
    Inputs X = [q1, ... q6, qvel1, ... qvel6, X_yref, Y_yref, Z_yref, X_ee, Y_ee, Z_ee] (Total 18 features)
    Targets y = cost
    Handles multiple runs inside the data dictionary.

    Optional train/test split using `split_ratio`.
    """
    def __init__(self, config, run_dir, mode, test_config=None):
        self.run_dir = run_dir
        self.mode = mode

        # =============================================================
        #                     TRAIN MODE
        # =============================================================
        if mode == "train":
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
            xyz = run_data["xyzpos"]
            cost = run_data["total_cost"]
                
            yref_pos = np.tile(run_data["yref_xyz"], (qpos.shape[0], 1))
            yref_q_run = np.tile(run_data["yref_q"], (qpos.shape[0], 1))

            # Concatenate qpos and qvel
            X_run = np.concatenate([qpos, qvel, yref_pos, xyz], axis=1)
            X_list.append(X_run)

            # Ensure cost is 2D
            if len(cost.shape) == 1:
                y_run = cost[:, None]
            else:
                y_run = cost
            y_list.append(y_run)

            # Stationary point arrays
            # While ee_pos is at yref and with zero velocity cost should be zero.
            Xs_run = np.concatenate([yref_q_run, np.zeros_like(qvel), yref_pos, yref_pos], axis=1)

            ys = np.zeros((1,))
            ys_run = np.tile(ys, (qpos.shape[0], 1))

            Xs_list.append(Xs_run)
            ys_list.append(ys_run)

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
    
class iiwa14_eeTracker_obs(iiwa14_eeTracker):
    """
    Dataset for TwoDofArm:
    Inputs X = [q1, ... q6, qvel1, ... qvel6, X_obstacle, Y_obstacle, Z_obstacle, R, P, Y, X_yref, Y_yref, Z_yref, X_ee, Y_ee, Z_ee] (Total 24 features)
    Targets y = cost
    Handles multiple runs inside the data dictionary.

    Optional train/test split using `split_ratio`.
    """

    def __init__(self, config, run_dir, mode, test_config=None):
        super().__init__(config, run_dir, mode, test_config)

    def preprocess_data(self, data):
        X_list = []
        Xs_list = []
        y_list = []
        ys_list = []

        for run_key in data.keys():  # iterate over each run
            run_data = data[run_key]
            qpos = run_data["qpos"]
            qvel = run_data["qvel"]
            xyz = run_data["xyzpos"]
            obs = run_data["obstacles"].item()["obs1"] # Assumes only 1 obstacle
            cost = run_data["total_cost"]

            center = np.tile(np.array(obs["center"]), (qpos.shape[0], 1))
            rpy = np.tile(np.array(obs["rpy"]), (qpos.shape[0], 1))
                
            yref_pos = np.tile(run_data["yref_xyz"], (qpos.shape[0], 1))
            yref_q_run = np.tile(run_data["yref_q"], (qpos.shape[0], 1))

            # Concatenate qpos and qvel
            X_run = np.concatenate([qpos, qvel, center, rpy, yref_pos, xyz], axis=1)
            X_list.append(X_run)

            # Ensure cost is 2D
            if len(cost.shape) == 1:
                y_run = cost[:, None]
            else:
                y_run = cost
            y_list.append(y_run)

            # Stationary point arrays
            # While ee_pos is at yref and with zero velocity cost should be zero.
            Xs_run = np.concatenate([yref_q_run, np.zeros_like(qvel), center, rpy, yref_pos, yref_pos], axis=1)

            ys = np.zeros((1,))
            ys_run = np.tile(ys, (qpos.shape[0], 1))

            Xs_list.append(Xs_run)
            ys_list.append(ys_run)

        # Stack all runs together
        self.X = torch.from_numpy(np.vstack(X_list)).float()
        self.Xs = torch.from_numpy(np.vstack(Xs_list)).float()
        self.y = torch.from_numpy(np.vstack(y_list)).float()
        self.ys = torch.from_numpy(np.vstack(ys_list)).float()