import numpy as np
import torch

class ReplayBuffer:
    """
    Replay buffer for TD learning.

    Stores transitions:
        X_t      : state at time t
        X_tpn    : state at time t+n
        cost_t   : stage cost
        Xs       : stationary/goal state
        ys       : stationary cost (0 or log(1) if log space)
    """

    def __init__(self, capacity):
        self.capacity = int(capacity)

        self.ptr = 0
        self.size = 0

        self.X_t = None
        self.X_tpn = None
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
        self.X_tpn = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.cost_t = np.zeros((self.capacity, 1), dtype=np.float32)
        self.Xs = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.ys = np.zeros((self.capacity, 1), dtype=np.float32)

    # ----------------------------------------------------
    # load buffer
    # ----------------------------------------------------
    def reinstate_buffer(self, path):

        data = np.load(path)

        self.X_t = data["X_t"].astype(np.float32)
        self.X_tpn = data["X_tpn"].astype(np.float32)
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
            X_tpn=self.X_tpn,
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
    def add_trajectory(self, X_t, X_tpn, cost_t, Xs, ys):
        if self.X_t is None:
            self.init_storage(X_t)

        n = X_t.shape[0]

        for i in range(n):
            self._add_single(
                X_t[i],
                X_tpn[i],
                cost_t[i],
                Xs[i],
                ys[i]
            )

    # ----------------------------------------------------
    # single transition
    # ----------------------------------------------------
    def _add_single(self, x_t, x_tpn, cost, xs, ys):

        i = self.ptr

        self.X_t[i] = x_t
        self.X_tpn[i] = x_tpn
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
            torch.from_numpy(self.X_tpn[idx]).float(),
            torch.from_numpy(self.cost_t[idx]).float(),
            torch.from_numpy(self.Xs[idx]).float(),
            torch.from_numpy(self.ys[idx]).float(),
        )

    # ----------------------------------------------------
    # size
    # ----------------------------------------------------
    def __len__(self):
        return self.size