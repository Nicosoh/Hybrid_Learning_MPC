import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_network.models.base_model import *
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
        self.fc_out = nn.Linear(64, 64)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

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
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        return x

@register_model
class PendulumModelSIREN(nn.Module):
    def __init__(self, train_config=None, omega_0=2.0):
        super().__init__()
        # Manual scaling to map to [-1,1]
        self.register_buffer('inv_ranges', torch.tensor([1.0 / 6.0, 1.0 / 5.0], dtype=torch.float32))
        
        # First SIREN layer (notices is_first=True)
        self.fc1 = SirenLayer(2, 64, is_first=True, omega_0=omega_0)
        
        # Hidden SIREN layers
        self.fc2 = SirenLayer(64, 64, is_first=False, omega_0=omega_0)
        self.fc3 = SirenLayer(64, 64, is_first=False, omega_0=omega_0)
        
        # Output layer (Linear, NO sine activation)
        self.fc_out = nn.Linear(64, 64)
        
        # Custom initialization for the final linear layer to match SIREN scheme
        with torch.no_grad():
            bounds = np.sqrt(6.0 / self.fc_out.in_features) / omega_0
            self.fc_out.weight.uniform_(-bounds, bounds)
            if self.fc_out.bias is not None:
                self.fc_out.bias.zero_()

    def forward(self, x):
        x = x * self.inv_ranges
        x = self.fc1(x)         # SIREN Layer 1
        x = self.fc2(x)         # SIREN Layer 2
        x = self.fc3(x)         # SIREN Layer 3
        x = self.fc_out(x)      # Linear output layer
        
        # Your custom Acados-mimicking least squares reduction
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)
        return x
    
@register_model
class PendulumModelSIRENAcados(PendulumModelSIREN):
    def __init__(self, train_config=None, omega_0=2.0):
        super().__init__(train_config=train_config, omega_0=omega_0)

    def forward(self, x):
        x = x * self.inv_ranges
        x = self.fc1(x)         # SIREN Layer 1
        x = self.fc2(x)         # SIREN Layer 2
        x = self.fc3(x)         # SIREN Layer 3
        x = self.fc_out(x)      # Linear output layer

        return x

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

        init_tanh_weights(self)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return x

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

        init_tanh_weights(self)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return x

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

@register_model
class TwoDofArmModelSIREN(nn.Module):
    def __init__(self, train_config=None, omega_0=30.0):
        super().__init__()
        
        # Define your exact asymmetric boundaries
        x_min = torch.tensor([-1.7, -2.5, -1.1, -1.1, -1.6, -1.0, 0.1, -1.6, -1.0, 0.1], dtype=torch.float32)
        x_max = torch.tensor([0.7,   2.2,  1.1,  1.1, -0.4,  1.0, 1.4, -0.4,  1.0, 1.4], dtype=torch.float32)
        
        # Precompute components of the Min-Max formula to keep the forward pass fast
        # scale = 2.0 / (max - min)
        # shift = (max + min) / (max - min)
        scale = 2.0 / (x_max - x_min)
        shift = (x_max + x_min) / (x_max - x_min)
        
        # Register them as buffers so they move with the model (CPU/GPU)
        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)
        
        # First SIREN layer
        self.fc1 = SirenLayer(10, 64, is_first=True, omega_0=omega_0)
        
        # Hidden SIREN layers
        self.fc2 = SirenLayer(64, 64, is_first=False, omega_0=omega_0)
        self.fc3 = SirenLayer(64, 64, is_first=False, omega_0=omega_0)
        self.fc_out = nn.Linear(64, 64)
        
        # Custom initialization for the final linear layer
        with torch.no_grad():
            bounds = np.sqrt(6.0 / self.fc_out.in_features) / omega_0
            self.fc_out.weight.uniform_(-bounds, bounds)
            if self.fc_out.bias is not None:
                self.fc_out.bias.zero_()

    def forward(self, x):
        # 1. Asymmetric Mapping: Maps any arbitrary unequal boundary strictly to [-1, 1]
        x = x * self.scale - self.shift
        
        # Force clamp to protect SIREN from any crazy out-of-bounds exploration data
        # x = torch.clamp(x, min=-1.0, max=1.0)
        
        # 2. Forward pass through SIREN architecture
        x = self.fc1(x)         
        x = self.fc2(x)         
        x = self.fc3(x)         
        x = self.fc_out(x)      
        
        # 3. Custom Acados-mimicking least squares reduction
        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)
        return x

@register_model
class TwoDofArmModelSIRENAcados(TwoDofArmModelSIREN):
    def __init__(self, train_config=None, omega_0=30.0):
        super().__init__(train_config=train_config, omega_0=omega_0)

    def forward(self, x):
        # 1. Asymmetric Mapping: Maps any arbitrary unequal boundary strictly to [-1, 1]
        x = x * self.scale - self.shift
        
        # Force clamp to protect SIREN from any crazy out-of-bounds exploration data
        # x = torch.clamp(x, min=-1.0, max=1.0)
        
        # 2. Forward pass through SIREN architecture
        x = self.fc1(x)         
        x = self.fc2(x)         
        x = self.fc3(x)         
        x = self.fc_out(x)      

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

        init_tanh_weights(self)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return x
    
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

@register_model
class iiwa14Model_obs(nn.Module):                                            # Without obstacles
    def __init__(self, train_config):
        super().__init__()

        self.fc0 = ScaleLayer(24)
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 64)

        init_tanh_weights(self)

    def forward(self, x):
        x = self.fc0(x)                                                     # Linear transformation without activation ("scaling" layer)
        x = F.tanh(self.fc1(x))                                             # Hidden layers with tanh activations
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc_out(x)                                                     # Output layer without activation ("scaling" layer)

        x = torch.tensor(0.5, dtype=x.dtype, device=x.device) * torch.sum(x**2, dim=1, keepdim=True)        # Least Squares which mimics acados cost

        return x
    
@register_model
class iiwa14ModelAcados_obs(iiwa14Model_obs):                                            # Without obstacles
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