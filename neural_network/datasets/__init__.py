from .pendulum_dataset import PendulumDataset
from .twodofarm_dataset import *
from .iiwa14_dataset import *

DATASET_REGISTRY = {
    "PendulumDataset": PendulumDataset,
    "TwoDofArmDataset": TwoDofArmDataset,
    "TwoDofArmDataset_eeTracker": TwoDofArmDataset_eeTracker,
    "TwoDofArmDataset_eeTracker_TD": TwoDofArmDataset_eeTracker_TD,
    "TwoDofArmDataset_eeTracker_obs": TwoDofArmDataset_eeTracker_obs,
    "iiwa14_eeTracker": iiwa14_eeTracker,
}