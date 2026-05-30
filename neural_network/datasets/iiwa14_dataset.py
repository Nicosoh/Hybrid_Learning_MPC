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

class iiwa14Dataset_TD():
    """
    N-step TD dataset for value learning (NO discounting).

    Target structure:

        G_t^(n) = sum_{k=0}^{n-1} cost_{t+k}

    Training target:

        y_t = G_t^(n) + V(s_{t+n})

    Returned:
        X_t        : state at time t
        X_tpn      : state at time t+n
        cost_n     : cumulative n-step cost
        Xs         : stationary goal states
        ys         : terminal target (zeros)
    """

    def __init__(self, config, model_config, data_path):
        self.config = config
        self.model_config = model_config
        self.n_step = self.config.getint("TRAINING", "n_step")

        self.preprocess_data(load_npz(data_path))

    def preprocess_data(self, data):

        X_t_list = []
        X_tpn_list = []
        cost_n_list = []

        Xs_list = []
        ys_list = []

        n = self.n_step

        for run_key in data.keys():

            run_data = data[run_key]

            qpos = run_data["qpos"]
            qvel = run_data["qvel"]
            u = run_data["u_applied"]
            xyz = run_data["xyzpos"]
            yref_xyz = run_data["yref_xyz"] # (3,)
            yref_q = run_data["yref_q"] # (6,)
            # ====================================================
            # Stage cost
            # ====================================================
            Q_mat = np.array(self.model_config["mpc"]["Q_mat"])
            Q_mat_dot = np.array(self.model_config["mpc"]["Q_mat_dot"])
            R_mat = np.array(self.model_config["mpc"]["R_mat"])

            # Tracking & Vel Reg cost
            xyz_cost = np.sum(Q_mat * (xyz ** 2), axis=1)
            qvel_cost = np.sum(Q_mat_dot * (qvel ** 2), axis=1)

            # Control cost
            input_cost = np.sum(R_mat * (u ** 2), axis=1)

            # Total cost
            cost = xyz_cost + qvel_cost + input_cost

            # ====================================================
            # STATE:
            # [qpos, qvel]
            # ====================================================
            yref_xyz = np.tile(yref_xyz, (qpos.shape[0], 1)) # (T, 3)
            X = np.concatenate([qpos, qvel, yref_xyz, xyz], axis=1)

            # ====================================================
            # Stationary goal manifold
            # ====================================================
            yref_q = np.tile(yref_q, (qpos.shape[0], 1)) # (T, 6)
            Xs_run = np.concatenate([yref_q, np.zeros_like(qvel), yref_xyz, yref_xyz], axis=1)

            T = X.shape[0]

            # ====================================================
            # Build N-step TD samples
            # ====================================================
            for t in range(T - n):

                # --------------------------------------------
                # Current state
                # --------------------------------------------
                X_t = X[t]

                # --------------------------------------------
                # Bootstrap state
                # --------------------------------------------
                X_tpn = X[t + n]

                # --------------------------------------------
                # N-step cumulative cost
                # NO discounting
                # --------------------------------------------
                cumulative_cost = np.sum(cost[t:t+n])

                # --------------------------------------------
                # Goal stationary state
                # --------------------------------------------
                Xs = Xs_run[t]

                # --------------------------------------------
                # Terminal supervision target
                # V(goal) = 0
                # --------------------------------------------
                y = np.array([0.0], dtype=np.float32)

                # --------------------------------------------
                # Append
                # --------------------------------------------
                X_t_list.append(X_t)
                X_tpn_list.append(X_tpn)

                cost_n_list.append(np.array([cumulative_cost], dtype=np.float32))

                Xs_list.append(Xs)

                ys_list.append(y)

        # ====================================================
        # Stack everything
        # ====================================================
        self.X_t = torch.from_numpy(np.vstack(X_t_list)).float()
        self.X_tpn = torch.from_numpy(np.vstack(X_tpn_list)).float()
        self.cost_n = torch.from_numpy(np.vstack(cost_n_list)).float()
        self.Xs = torch.from_numpy(np.vstack(Xs_list)).float()
        self.ys = torch.from_numpy(np.vstack(ys_list)).float()

    def __len__(self):
        return self.X_t.shape[0]

    def __getitem__(self, idx):
        return (self.X_t[idx], self.X_tpn[idx], self.Xs[idx], self.cost_n[idx], self.ys[idx])