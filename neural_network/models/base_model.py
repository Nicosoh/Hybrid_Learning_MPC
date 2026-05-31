import torch
import torch.nn as nn
import numpy as np

class ScaleLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale  # elementwise scaling
    
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer needs a different scaling factor
                bounds = 1.0 / self.linear.in_features
                self.linear.weight.uniform_(-bounds, bounds)
            else:
                # Hidden layers are scaled based on c / sqrt(in_features) * omega_0
                # SIREN paper uses c = 6 for sine activations
                bounds = np.sqrt(6.0 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bounds, bounds)
                
            # Initialize biases to zero
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        # We multiply by omega_0 before the sine to scale the input frequencies
        return torch.sin(self.omega_0 * self.linear(x))
    
def init_tanh_weights(model):
    """
    Applies Xavier initialization tailored for a network with Tanh hidden layers
    and a linear output layer.
    """
    def init_fn(m):
        if isinstance(m, nn.Linear):
            # Check if this specific submodule matches the model's output layer
            if m is getattr(model, 'fc_out', None):  
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_fn)