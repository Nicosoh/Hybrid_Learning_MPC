import torch
import torch.nn as nn

class StationaryLoss(nn.Module):
    def __init__(self, alpha):
        """
        alpha : weight for the stationary-point penalty
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred_main, y_batch, pred_stationary, y_stationary):
        """
        pred_main        : V(x_batch) predictions       (n x 1)
        y_batch          : MPC/Bellman targets          (n x 1)
        pred_stationary  : V(X_s) predictions           (m x 1)
        y_stationary     : V(X_s) targets             (m x 1)
        """
        # 1. Main Bellman / MPC loss
        loss1 = self.mse(pred_main, y_batch)

        # 2. Stationary-point loss: enforce V(x_s) ≈ 0
        loss2 = self.alpha * self.mse(pred_stationary, y_stationary)

        # 3. Combined loss
        total_loss = loss1 + loss2

        return total_loss, loss1, loss2

class TDLoss(nn.Module):
    def __init__(self, alpha, lam):
        """
        alpha : weight for the stationary-point penalty
        lam : weight for the regularizer penalty
        """
        super().__init__()
        self.alpha = alpha
        self.lam = lam
        self.mse = nn.MSELoss()
    def forward(
        self,
        pred_main,
        y_batch,
        pred_stationary,
        y_stationary,
        pred_offline
    ):
        """
        pred_main       : V(x_batch)
        y_batch         : TD / MPC targets
        pred_stationary : V(x_s)
        y_stationary    : typically zeros or boundary targets
        pred_offline    : V_offline(x_batch)
        """

        # 1. Compute raw absolute MSE values (Inputs are log-space transformed from your loop!)
        raw_td_mse = self.mse(pred_main, y_batch)
        raw_trust_mse = self.mse(pred_main, pred_offline)

        # 2. Compute variance purely from the current batch data
        # Detached so denominators act strictly as normalization constants
        td_var = torch.var(y_batch).detach().clamp(min=1e-4)
        offline_var = torch.var(pred_offline).detach().clamp(min=1e-4)

        # 3. Normalize terms into unitless percentages of variance explained
        loss_td = raw_td_mse / td_var
        loss_offline = raw_trust_mse / offline_var

        # 4. Adaptive Lambda based on EMA of TD MSE
        loss_offline = self.lam * 0

        # 5. Stationary constraint (y_stationary is log1p(0) = 0)
        loss_stationary = self.alpha * self.mse(
            pred_stationary,
            y_stationary
        )
        print(f"lam: {self.lam}")
        # ----------------------------
        # 5. Total loss
        # ----------------------------
        total_loss = loss_td + loss_stationary

        return total_loss, loss_td, loss_offline, loss_stationary

class TDLossTR(nn.Module):
    def __init__(self, alpha, lam):

        super().__init__()
        self.alpha = alpha
        self.lam = lam
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_main,
        y_batch,
        pred_stationary,
        y_stationary,
        pred_offline
    ):
        """
        pred_main       : V(x_batch)
        y_batch         : TD / MPC targets
        pred_stationary : V(x_s)
        y_stationary    : typically zeros or boundary targets
        pred_offline    : V_offline(x_batch)
        """

        # 1. Compute raw absolute MSE values (Inputs are log-space transformed from your loop!)
        raw_td_mse = self.mse(pred_main, y_batch)
        raw_trust_mse = self.mse(pred_main, pred_offline)

        # 2. Compute variance purely from the current batch data
        # Detached so denominators act strictly as normalization constants
        td_var = torch.var(y_batch).detach().clamp(min=1e-4)
        offline_var = torch.var(pred_offline).detach().clamp(min=1e-4)

        # 3. Normalize terms into unitless percentages of variance explained
        loss_td = raw_td_mse / td_var
        loss_offline = raw_trust_mse / offline_var

        # 4. Adaptive Lambda based on EMA of TD MSE
        loss_offline = self.lam * loss_offline

        # 5. Stationary constraint (y_stationary is log1p(0) = 0)
        loss_stationary = self.alpha * self.mse(
            pred_stationary,
            y_stationary
        )
        print(f"lam: {self.lam}")
        # ----------------------------
        # 5. Total loss
        # ----------------------------
        total_loss = loss_td + loss_offline + loss_stationary

        return total_loss, loss_td, loss_offline, loss_stationary