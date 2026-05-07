import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_network.models.base_model import ScaleLayer
from neural_network.utils import run_scaling

MODEL_REGISTRY = {}

def register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls

# =================================================================
# Pendulum Models
# =================================================================

@register_model
class PendulumModel(nn.Module):
    def __init__(self, train_config):
        super().__init__()

        self.fc0 = ScaleLayer(2)
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * x**2        # Least Squares which mimics acados cost

        return x
    
@register_model
class PendulumModelAcados(PendulumModel):
    def __init__(self, train_config):
        super().__init__(train_config)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)                                                     # Output layer without activation ("scaling" layer)

        return x

@register_model
class PendulumModel_with_scaling(nn.Module):
    def __init__(self, train_config):
        super().__init__()

        # MLP
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

        # If scaling is required and is normalize
        self.apply_scaling_X = train_config.getboolean("DATA", "apply_scaling_X")
        self.apply_scaling_y = train_config.getboolean("DATA", "apply_scaling_y")
        self.scaling_type = train_config.get("DATA", "scaling_type")
        
        # Check scaling type
        if self.scaling_type != "normalize":
            raise NotImplementedError("Only 'normalize' scaling is implemented.")
        
        # Only load X scaling ranges if applicable
        if self.apply_scaling_X == True and self.scaling_type == "normalize":
            # Scaling ranges from config
            self.scaling_range_from_X = torch.tensor(ast.literal_eval(train_config.get("DATA", "scaling_range_from_X")), dtype=torch.float32)
            self.scaling_range_to_X = torch.tensor(ast.literal_eval(train_config.get("DATA", "scaling_range_to_X")), dtype=torch.float32)

        # Only load y scaling ranges if applicable
        if self.apply_scaling_y == True and self.scaling_type == "normalize":
            self.scaling_range_from_y = torch.tensor(ast.literal_eval(train_config.get("DATA", "scaling_range_from_y")), dtype=torch.float32)
            self.scaling_range_to_y = torch.tensor(ast.literal_eval(train_config.get("DATA", "scaling_range_to_y")), dtype=torch.float32)

    def forward(self, x):
        """
        If scaling_params is None:
            Training mode → model expects pre-scaled x
        
        If scaling_params is not None:
            ACADOS/Inference mode → model scales x internally, runs the MLP,
            then unscales the output.
        """

        # ---------------------------------------------------------
        # 1) Apply input scaling internally (ACADOS mode)
        # ---------------------------------------------------------
        if self.apply_scaling_X == True:
            # Only X is scaled; y is None
            x, _ = run_scaling(
                x, None,
                scaling_type=self.scaling_type,
                scaling_range_from_X=self.scaling_range_from_X,
                scaling_range_to_X=self.scaling_range_to_X,
                inverse=False
            )

        # ---------------------------------------------------------
        # 2) Neural network forward pass
        # ---------------------------------------------------------
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        y = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x ** 2, dim=-1, keepdim=True)

        # ---------------------------------------------------------
        # 3) Unscale output (ACADOS mode only)xs
        # ---------------------------------------------------------
        if self.apply_scaling_y == True:
            _, y = run_scaling(
                None, y,
                scaling_type=self.scaling_type,
                scaling_range_from_y=self.scaling_range_from_y,
                scaling_range_to_y=self.scaling_range_to_y,
                inverse=True
            )

        return y
    
# =================================================================
# TwoDofArm Models
# =================================================================

@register_model
class TwoDofArmModel(nn.Module):                                            # Without obstacles
    def __init__(self, train_config):
        super().__init__()

        self.fc0 = ScaleLayer(10)
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 64)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return torch.log1p(x)
        # return x

@register_model
class TwoDofArmModelAcados(TwoDofArmModel):                                            # Without obstacles
    def __init__(self, train_config):
        super().__init__(train_config)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        return x

@register_model
class TwoDofArmModel_obs(nn.Module):                                            # With obstacles
    def __init__(self, train_config):
        super().__init__()

        self.fc0 = ScaleLayer(14)
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 64)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return torch.log1p(x)
        # return x

@register_model
class TwoDofArmModelAcados_obs(TwoDofArmModel_obs):                                            # With obstacles
    def __init__(self, train_config):
        super().__init__(train_config)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        return x
    
# =================================================================
# iiwa14 Models
# =================================================================

@register_model
class iiwa14Model(nn.Module):                                            # Without obstacles
    def __init__(self, train_config):
        super().__init__()

        self.fc0 = ScaleLayer(18)
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 64)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return torch.log1p(x)
        # return x
    
@register_model
class iiwa14ModelAcados(iiwa14Model):                                            # Without obstacles
    def __init__(self, train_config):
        super().__init__(train_config)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        return x