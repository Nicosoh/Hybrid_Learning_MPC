import numpy as np
import torch

from data_collection.data_utils import load_npz
from neural_network.cost.cost import compute_l2_cost


class ReplayBuffer:
    """
    Replay buffer for TD learning.

    Stores transitions:
        X_t      : state at time t
        X_tp1    : state at time t+1
        cost_t   : stage cost
        Xs       : stationary/goal state
        ys       : stationary cost (0 or log(1) if log space)
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)

        self.ptr = 0
        self.size = 0

        self.X_t = None
        self.X_tp1 = None
        self.cost_t = None
        self.Xs = None
        self.ys = None

    # ----------------------------------------------------
    # initialize storage once dimensions are known
    # ----------------------------------------------------
    def init_storage(self, sample):
        sample = np.asarray(sample)

        # robust shape inference
        state_dim = sample.shape[-1]

        self.X_t = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.X_tp1 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.cost_t = np.zeros((self.capacity, 1), dtype=np.float32)
        self.Xs = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.ys = np.zeros((self.capacity, 1), dtype=np.float32)

    # ----------------------------------------------------
    # add npz dataset
    # ----------------------------------------------------
    def add_npz_data(self, data_path, train_config, model_config):

        log_space = train_config.getboolean("DATA", "log_space")
        data = load_npz(data_path)

        for run_key in data.keys():

            run_data = data[run_key]

            qpos = run_data["qpos"]
            qvel = run_data["qvel"]
            xyz = run_data["xyzpos"]

            # ----------------------------
            # cost
            # ----------------------------
            cost = compute_l2_cost(
                config=model_config,
                xyz=xyz,
                qvel=qvel,
                yref_xyz=run_data["yref_xyz"],
                u=run_data["u_applied"]
            )

            if log_space:
                cost = np.log1p(cost)

            # ----------------------------
            # references
            # ----------------------------
            yref_xyz = np.tile(run_data["yref_xyz"], (qpos.shape[0], 1))
            yref_q = np.tile(run_data["yref_q"], (qpos.shape[0], 1))

            # ----------------------------
            # state
            # ----------------------------
            X = np.concatenate([qpos, qvel, yref_xyz, xyz], axis=1).astype(np.float32)

            # lazy init
            if self.X_t is None:
                self.init_storage(X)

            # ----------------------------
            # TD transitions
            # ----------------------------
            X_t = X[:-1]
            X_tp1 = X[1:]
            cost_t = cost[:-1, None]

            # ----------------------------
            # stationary state
            # ----------------------------
            Xs = np.concatenate([
                yref_q,
                np.zeros_like(qvel),
                yref_xyz,
                yref_xyz
            ], axis=1).astype(np.float32)

            Xs = Xs[:-1]

            ys = np.zeros_like(cost_t).astype(np.float32)

            # ----------------------------
            # add to buffer
            # ----------------------------
            self.add_trajectory(X_t, X_tp1, cost_t, Xs, ys)

    # ----------------------------------------------------
    # load buffer
    # ----------------------------------------------------
    def reinstate_buffer(self, path):

        data = np.load(path)

        self.X_t = data["X_t"].astype(np.float32)
        self.X_tp1 = data["X_tp1"].astype(np.float32)
        self.cost_t = data["cost_t"].astype(np.float32)
        self.Xs = data["Xs"].astype(np.float32)
        self.ys = data["ys"].astype(np.float32)

        self.ptr = int(data["ptr"][0])
        self.size = int(data["size"][0])
        self.capacity = int(data["capacity"][0])

    # ----------------------------------------------------
    # save buffer
    # ----------------------------------------------------
    def save_buffer(self, path):

        np.savez_compressed(
            path,
            X_t=self.X_t,
            X_tp1=self.X_tp1,
            cost_t=self.cost_t,
            Xs=self.Xs,
            ys=self.ys,
            ptr=np.array([self.ptr], dtype=np.int64),
            size=np.array([self.size], dtype=np.int64),
            capacity=np.array([self.capacity], dtype=np.int64),
        )

    # ----------------------------------------------------
    # add trajectory
    # ----------------------------------------------------
    def add_trajectory(self, X_t, X_tp1, cost_t, Xs, ys):

        n = X_t.shape[0]

        for i in range(n):
            self._add_single(
                X_t[i],
                X_tp1[i],
                cost_t[i],
                Xs[i],
                ys[i]
            )

    # ----------------------------------------------------
    # single transition
    # ----------------------------------------------------
    def _add_single(self, x_t, x_tp1, cost, xs, ys):

        i = self.ptr

        self.X_t[i] = x_t
        self.X_tp1[i] = x_tp1
        self.cost_t[i] = cost
        self.Xs[i] = xs
        self.ys[i] = ys

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ----------------------------------------------------
    # sample batch
    # ----------------------------------------------------
    def sample(self, batch_size):

        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.X_t[idx]).float(),
            torch.from_numpy(self.X_tp1[idx]).float(),
            torch.from_numpy(self.cost_t[idx]).float(),
            torch.from_numpy(self.Xs[idx]).float(),
            torch.from_numpy(self.ys[idx]).float(),
        )

    # ----------------------------------------------------
    # size
    # ----------------------------------------------------
    def __len__(self):
        return self.size