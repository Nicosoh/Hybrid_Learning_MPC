import yaml
import argparse
import os

from utils import *
from simulator import MujocoReplay
from controller import CONTROLLER_REGISTRY
from data_collection.data_utils import load_npz

def main(run_folder):
    if not os.path.isdir(run_folder):
        raise NotADirectoryError(f"Run folder does not exist: {run_folder}")

    # ----------------------------
    # Find YAML config
    # ----------------------------
    yaml_files = [
        f for f in os.listdir(run_folder)
        if f.endswith(".yaml")
    ]

    if len(yaml_files) == 0:
        raise FileNotFoundError(f"No YAML config found in {run_folder}")
    if len(yaml_files) > 1:
        raise RuntimeError(f"Multiple YAML configs found in {run_folder}: {yaml_files}")

    config_path = os.path.join(run_folder, yaml_files[0])

    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # ----------------------------
    # Load replay configuration
    # ----------------------------
    with open("configs/replay_config.yaml", "r") as f:
        replay_config = yaml.safe_load(f)

    # ----------------------------
    # Find NPZ logs
    # ----------------------------
    npz_files = [
        f for f in os.listdir(run_folder)
        if f.endswith(".npz")
    ]

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No NPZ log file found in {run_folder}")
    if len(npz_files) > 1:
        raise RuntimeError(f"Multiple NPZ files found in {run_folder}: {npz_files}")

    npz_path = os.path.join(run_folder, npz_files[0])

    logs_dict = load_npz(npz_path)["default"]

    # Load collision_config
    if model_config["collision"]["collision_avoidance_obstacle"] or model_config["collision"]["collision_avoidance_ground"]:                                                      # If enabled in config
        model_config = load_collision_config_for_replay(model_config)                                        # Load obstacles

    # ----------------------------
    # Run replay
    # ----------------------------
    replay = MujocoReplay(model_config, replay_config, logs_dict)
    replay.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a simulation from a run folder")
    parser.add_argument("run_folder", type=str, help="Path to the run folder")
    args = parser.parse_args()

    main(args.run_folder)