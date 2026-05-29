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

    def forward(self, pred_main, y_batch, pred_stationary, y_stationary, pred_offline):
        """
        pred_main        : V(x_batch) predictions       (n x 1)
        y_batch          : MPC/Bellman targets          (n x 1)
        pred_stationary  : V(X_s) predictions           (m x 1)
        y_stationary     : V(X_s) targets             (m x 1)
        """
        # 1. Main Bellman / MPC loss
        loss1 = self.mse(pred_main, y_batch)

        # 2. Regularizer loss: keep V(x) as close to V_offline(x) as possible
        loss2 = self.lam *self.mse(pred_main, pred_offline)

        # 3. Stationary-point loss: enforce V(x_s) ≈ 0
        loss3 = self.alpha * self.mse(pred_stationary, y_stationary)

        # 4. Combined main_loss
        main_loss = loss1 + loss2

        # 5. Combined main_loss
        total_loss = loss1 + loss2 +  loss3

        return total_loss, main_loss, loss3

class TDLossless(nn.Module):
    def __init__(
        self,
        alpha,              # stationary constraint weight
        target_kl,        # desired drift from offline model
        lam,          # initial KL multiplier
        lam_lr,           # how fast lambda adapts
    ):
        super().__init__()
        self.alpha = alpha
        self.target_kl = target_kl
        self.lam_lr = lam_lr

        self.mse = nn.MSELoss()

        # learnable / adaptive regularization strength
        self.lam = lam

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

        # ----------------------------
        # 1. TD / Bellman loss
        # ----------------------------
        td_scale = y_batch.detach().abs().mean().clamp(min=1e-3)
        loss_td = self.mse(pred_main, y_batch) / td_scale

        # -------------------------
        # 2. Normalized KL surrogate
        # -------------------------
        eps = 1e-8

        mse_num = torch.mean(
            (pred_main - pred_offline) ** 2
        )

        mse_den = torch.mean(
            pred_offline ** 2
        ) + eps

        kl = mse_num / mse_den

        # -------------------------
        # adaptive trust region
        # -------------------------
        with torch.no_grad():

            self.lam *= torch.exp(
                self.lam_lr *
                (kl.detach() - self.target_kl)
            )

            # self.lam.clamp_(0.0, 1e12)

        loss_offline = self.lam * kl

        # ----------------------------
        # 4. Stationary constraint
        # ----------------------------
        loss_stationary = self.alpha * self.mse(
            pred_stationary,
            y_stationary
        )

        # ----------------------------
        # 5. Total loss
        # ----------------------------
        total_loss = loss_td + loss_offline + loss_stationary


        print("TD Loss: {:.4f}, KL: {:.4f}, Lambda: {:.4f}, Stationary Loss: {:.4f}".format(
            loss_td,
            kl.detach(),
            self.lam if not torch.is_tensor(self.lam) else self.lam.detach(),
            loss_stationary
        ))

        return total_loss, loss_td + loss_offline, loss_stationary

class TDLosslessV2(nn.Module):
    def __init__(
        self,
        alpha,              # stationary constraint weight
        target_kl,          # desired drift from offline model
        lam,                # initial KL multiplier
        lam_lr,             # how fast lambda adapts
    ):
        super().__init__()
        self.alpha = alpha
        self.target_kl = target_kl
        self.lam_lr = lam_lr

        self.mse = nn.MSELoss()

        # Kept exactly as your original code (handles raw float or tensor)
        self.lam = lam

    def forward(
        self,
        pred_main,
        y_batch,
        pred_stationary,
        y_stationary,
        pred_offline):

        """
        pred_main       : V(x_batch)
        y_batch         : TD / MPC targets
        pred_stationary : V(x_s)
        y_stationary    : typically zeros or boundary targets
        pred_offline    : V_offline(x_batch)
        """

        # 1. Compute raw absolute MSE values
        raw_td_mse = self.mse(pred_main, y_batch)
        raw_trust_mse = self.mse(pred_main, pred_offline)

        # 2. Compute variance purely from the current batch data
        # We detach so the normalization denominators don't interfere with gradients
        td_var = torch.var(y_batch).detach().clamp(min=1e-4)
        offline_var = torch.var(pred_offline).detach().clamp(min=1e-4)

        # 3. Normalize terms into unitless percentages of variance
        loss_td = raw_td_mse / td_var
        kl = raw_trust_mse / offline_var

        # 4. Adaptive trust region multiplier update
        with torch.no_grad():
            scale_factor = torch.exp(
                self.lam_lr * (kl.detach() - self.target_kl)
            )
            
            # Defensive check to handle whether self.lam was initialized as a float or a tensor
            if torch.is_tensor(self.lam):
                self.lam = (self.lam * scale_factor).clamp(min=1e-3, max=1e5)
            else:
                self.lam = max(1e-3, min(1e5, self.lam * scale_factor.item()))

        loss_offline = self.lam * kl

        # 5. Stationary constraint
        loss_stationary = self.alpha * self.mse(
            pred_stationary,
            y_stationary
        )

        # 6. Total loss assembly
        total_loss = loss_td + loss_offline + loss_stationary

        print("TD Loss: {:.4f}, KL: {:.4f}, Lambda: {:.4f}, Stationary Loss: {:.4f}".format(
            loss_td,
            kl.detach(),
            self.lam if not torch.is_tensor(self.lam) else self.lam.detach(),
            loss_stationary
        ))

        return total_loss, loss_td + loss_offline, loss_stationary
    
class TDLosslessV3(nn.Module):
    def __init__(
        self,
        alpha,              # stationary constraint weight (e.g., 0.1 - 1.0)
        target_kl,          # desired max drift from offline model (expressed as % of variance, e.g., 0.05)
        p_gain=2.0          # How aggressively lambda reacts when drifting (1.0 = linear, 2.0 = quadratic)
    ):
        super().__init__()
        self.alpha = alpha
        self.target_kl = target_kl
        self.p_gain = p_gain
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_main,
        y_batch,
        pred_stationary,
        y_stationary,
        pred_offline):
        
        # 1. Compute raw absolute MSE values (Inputs are log-space transformed from your loop!)
        raw_td_mse = self.mse(pred_main, y_batch)
        raw_trust_mse = self.mse(pred_main, pred_offline)

        # 2. Compute variance purely from the current batch data
        # Detached so denominators act strictly as normalization constants
        td_var = torch.var(y_batch).detach().clamp(min=1e-4)
        offline_var = torch.var(pred_offline).detach().clamp(min=1e-4)

        # 3. Normalize terms into unitless percentages of variance explained
        loss_td = raw_td_mse / td_var
        kl = raw_trust_mse / offline_var

        # 4. Instantaneous Proportional Lambda (No stateful tracking, zero lag!)
        with torch.no_grad():
            # How far are we from our allowed drift percentage?
            kl_ratio = kl / self.target_kl
            
            # Raise to p_gain power so lambda behaves like a stiff spring 
            # if the model aggressively oversteps the target_kl
            lam = torch.pow(kl_ratio, self.p_gain)

        # Apply the calculated lambda to the normalized KL loss
        loss_offline = lam * kl

        # 5. Stationary constraint (y_stationary is log1p(0) = 0)
        loss_stationary = self.alpha * self.mse(
            pred_stationary,
            y_stationary
        )

        # 6. Total loss assembly
        total_loss = loss_td + loss_offline + loss_stationary

        return total_loss, loss_td, loss_offline, loss_stationary
    
class TDLosslessV4(nn.Module):
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