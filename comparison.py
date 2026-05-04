import yaml
import argparse
import os
import numpy as np

from utils import *
from simulator import MujocoComparisonReplay
from data_collection.data_utils import load_npz

def load_run_data(run_folder):
    """Helper to extract config and logs from a specific run directory."""
    if not os.path.isdir(run_folder):
        raise NotADirectoryError(f"Run folder does not exist: {run_folder}")

    # Find YAML
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

    # Find NPZ
    npz_files = [f for f in os.listdir(run_folder) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No NPZ log file found in {run_folder}")
    
    npz_path = os.path.join(run_folder, npz_files[0])
    logs_dict = load_npz(npz_path)["default"]

    # Load collision_config
    if model_config["collision"]["collision_avoidance_obstacle"] or \
       model_config["collision"]["collision_avoidance_ground"]:
        collision_config, model_config = load_collision_config(model_config)
    else:
        collision_config = None

    return model_config, logs_dict, collision_config

def main(run_folder_a, run_folder_b):
    # Load replay global settings
    with open("configs/replay_config.yaml", "r") as f:
        replay_config = yaml.safe_load(f)

    # Load both runs
    print(f"Loading Run A: {run_folder_a}")
    config_a, logs_a, coll_a = load_run_data(run_folder_a)
    
    print(f"Loading Run B: {run_folder_b}")
    config_b, logs_b, coll_b = load_run_data(run_folder_b)

    # Combine logs for the simulator
    # Note: This assumes your MujocoReplay class is updated to handle 
    # a list of logs or keys like 'qpos_a' and 'qpos_b'
    combined_logs = {
        "run_a": logs_a,
        "run_b": logs_b
    }

    replay = MujocoComparisonReplay(config_a, replay_config, combined_logs, coll_a)
    
    print("Starting Comparison Replay...")
    replay.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay two simulations side-by-side")
    parser.add_argument("folder_a", type=str, help="Path to the first run folder")
    parser.add_argument("folder_b", type=str, help="Path to the second run folder")
    args = parser.parse_args()

    main(args.folder_a, args.folder_b)