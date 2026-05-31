from .pendulum_dataset import *
from .twodofarm_dataset import *
from .iiwa14_dataset import *

DATASET_REGISTRY = {
    "PendulumDataset": PendulumDataset,
    "PendulumDataset_TD": PendulumDataset_TD,
    "TwoDofArmDataset": TwoDofArmDataset,
    "TwoDofArmDataset_eeTracker": TwoDofArmDataset_eeTracker,
    "TwoDofArmDataset_eeTracker_obs": TwoDofArmDataset_eeTracker_obs,
    # "TwoDofArmDataset_eeTracker_TD": TwoDofArmDataset_eeTracker_TD,
    "iiwa14_eeTracker": iiwa14_eeTracker,
    "iiwa14_eeTracker_obs": iiwa14_eeTracker_obs,
    "iiwa14Dataset_TD": iiwa14Dataset_TD,
}