from networkx import config
import torch

import numpy as np

from torch.utils.data import Dataset, random_split
from data_collection import load_npz
from utils import get_num_config
from neural_network.cost import compute_l2_cost

class TwoDofArmDataset(Dataset):
    """
    Dataset for TwoDofArm:
    Inputs X = [qpos1, qpos2, qvel1, qvel2, yref_qpos1, yref_qpos2]
    Targets y = cost
    Handles multiple runs inside the data dictionary.

    Optional train/test split using `split_ratio`.
    """
    def __init__(self, config, run_dir, mode, test_config=None):
        self.run_dir = run_dir
        self.mode = mode
        self.log_space = config.getboolean("DATA", "log_space")
        self.config = config
        # =============================================================
        #                     TRAIN MODE
        # =============================================================
        if mode == "train":
            # Load stationary point config 
            # self.Xs_config = torch.tensor(get_num_config("LOSS", "x_s", config), dtype=torch.float32)
            # self.ys_config = torch.tensor(get_num_config("LOSS", "y_s", config), dtype=torch.float32)

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
            # Load stationary point config 
            # self.Xs_config = torch.tensor(get_num_config("LOSS", "x_s", config), dtype=torch.float32)
            # self.ys_config = torch.tensor(get_num_config("LOSS", "y_s", config), dtype=torch.float32)

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
            yref_q = np.tile(run_data["yref_xyz"], (qpos.shape[0], 1))

            # Concatenate qpos and qvel
            X_run = np.concatenate([qpos, qvel, yref_q], axis=1)
            X_list.append(X_run)

            # Ensure cost is 2D
            if len(cost.shape) == 1:
                y_run = cost[:, None]
            else:
                y_run = cost
            y_list.append(y_run)

            # Stationary point arrays
            Xs_q = run_data["yref_full"][-1][:qpos.shape[1]]
            Xs_v = np.zeros_like(Xs_q)
            Xs = np.concatenate([Xs_q, Xs_v])
            ys = np.zeros((1,))
            # Form duplicated array that follows stationary point/end objective
            Xs_run = np.concatenate([np.tile(Xs, (qpos.shape[0], 1)), yref_q], axis=1)
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

class TwoDofArmDataset_eeTracker(TwoDofArmDataset):
    """
    Dataset for TwoDofArm:
    Inputs X = [qpos1, qpos2, qvel1, qvel2, X_yref, Y_yref, Z_yref, X_ee, Y_ee, Z_ee]
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

            if self.log_space:
                cost = np.log1p(run_data["total_cost"])
            else:
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
            # Xs_pos = yref_pos
            # (ee pos, zero vel, yref pos) 
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

class TwoDofArmDataset_eeTracker_obs(TwoDofArmDataset):
    """
    Dataset for TwoDofArm:
    Inputs X = [qpos1, qpos2, qvel1, qvel2, X_obstacle, Y_obstacle, Z_obstacle, RPY[1], X_yref, Y_yref, Z_yref, X_ee, Y_ee, Z_ee]
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

            center = np.tile(np.array(obs["center"]), (qpos.shape[0], 1))
            rpy = np.tile(np.array(obs["rpy"])[1], (qpos.shape[0], 1))

            if self.log_space:
                cost = np.log1p(run_data["total_cost"])
            else:
                cost = run_data["total_cost"]
                
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
            # Xs_pos = yref_pos
            # (ee pos, zero vel, yref pos) 
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

class TwoDofArmDataset_eeTracker_TD(TwoDofArmDataset):
    """
    TD dataset for value function learning.

    Returns:
        X_t       : state at time t
        X_tp1     : state at time t+1
        cost_t    : stage cost
        Xs        : stationary states (goal)
    """

    def __init__(self, config, model_config, run_dir, mode, test_config=None):
        self.model_config = model_config
        super().__init__(config, run_dir, mode, test_config)

    def preprocess_data(self, data):

        X_t_list = []
        X_tp1_list = []
        cost_t_list = []

        Xs_list = []
        ys_list = []

        for run_key in data.keys():

            run_data = data[run_key]

            qpos = run_data["qpos"]
            qvel = run_data["qvel"]
            xyz = run_data["xyzpos"]

            # ====================================================
            # Cost
            # ====================================================
            cost = compute_l2_cost(
                config=self.model_config,
                xyz=xyz,
                qvel=qvel,
                yref_xyz=run_data["yref_xyz"],
                u=run_data["u_applied"])
            
            if self.log_space:
                cost = np.log1p(cost)

            # ====================================================
            # References
            # ====================================================
            yref = run_data["yref_xyz"]

            yref_q = np.tile(run_data["yref_q"], (qpos.shape[0], 1))
            yref_xyz = np.tile(yref, (qpos.shape[0], 1))

            # ====================================================
            # STATE: [qpos, qvel, goal, ee_pos]
            # ====================================================
            X = np.concatenate([qpos, qvel, yref_xyz, xyz], axis=1)

            # ====================================================
            # TD pairs
            # ====================================================
            X_t = X[:-1]
            X_tp1 = X[1:]
            cost_t = cost[:-1]
            
            X_t_list.append(X_t)
            X_tp1_list.append(X_tp1)
            cost_t_list.append(cost_t[:, None])

            # ====================================================
            # Stationary state (goal manifold)
            # ====================================================
            Xs_run = np.concatenate([
                yref_q,                  # q at goal
                np.zeros_like(qvel),     # zero velocity
                yref_xyz,                # goal conditioning
                yref_xyz                 # ee at goal
            ], axis=1)

            # repeat across time (important for stability)
            Xs_list.append(Xs_run[:-1])

            ys_run = np.zeros((qpos.shape[0] - 1, 1))

            ys_list.append(ys_run)

        # ====================================================
        # Stack everything
        # ====================================================
        self.X_t = torch.from_numpy(np.vstack(X_t_list)).float()
        self.X_tp1 = torch.from_numpy(np.vstack(X_tp1_list)).float()
        self.Xs = torch.from_numpy(np.vstack(Xs_list)).float()
        self.cost_t = torch.from_numpy(np.vstack(cost_t_list)).float()
        self.ys = torch.from_numpy(np.vstack(ys_list)).float()

    def train_val_data(self, val_split=0.2, seed=42):
        dataset_size = len(self.X_t)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.train_dataset, self.val_dataset = random_split(
            self, [train_size, val_size], generator=generator
        )
    
    def __len__(self):
        return self.X_t.shape[0]

    def __getitem__(self, idx):
        return self.X_t[idx], self.X_tp1[idx], self.Xs[idx], self.cost_t[idx], self.ys[idx]