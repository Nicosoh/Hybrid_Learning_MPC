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
    
class PendulumDataset_TD():
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

            # ====================================================
            # Stage cost
            # ====================================================
            Q_mat = np.array(self.model_config["mpc"]["Q_mat"])
            R_mat = np.array(self.model_config["mpc"]["R_mat"])

            # Tracking & Vel Reg cost
            x = np.concatenate([qpos, qvel], axis=1)
            cost = np.sum(Q_mat * (x ** 2), axis=1)

            # Control cost
            input_cost = np.sum(R_mat * (u ** 2), axis=1)

            # Total cost
            cost = cost + input_cost

            # ====================================================
            # STATE:
            # [qpos, qvel]
            # ====================================================
            X = np.concatenate([qpos, qvel], axis=1)

            # ====================================================
            # Stationary goal manifold
            # ====================================================
            Xs_run = np.concatenate([np.zeros_like(qpos), np.zeros_like(qvel)], axis=1)

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