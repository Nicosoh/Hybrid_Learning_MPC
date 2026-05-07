import os
import time
import copy
import yaml
import argparse

from datetime import datetime
from utils import *
from simulator import MuJoCoSimulator
from controller import CONTROLLER_REGISTRY
from IK import InverseKinematicsSolver
from data_collection.data_utils import save_npz

def main(model_name, data_collection=False, output_dir=None, timestamp=None, data_config=None, config=None, worker_id=0):
    # Load configuration
    if config is None:
        with open(f"configs/{model_name}config.yaml", "r") as f:
            config = yaml.safe_load(f)

    # Base output directory
    output_dir = output_dir or "data"
    os.makedirs(output_dir, exist_ok=True)

    # Replace simulation time with max steps and set full_traj to true
    if data_collection:
        config["mujoco"]["sim_duration"] = config["mpc"]["mpc_timestep"]*data_config["max_steps"]
        config["mpc"]["full_traj"] = True
        config["mpc"]["solve_ocp"] = False
        config["mujoco"]["render"] = False

    # Create a subfolder with date, time and model name
    if timestamp is None: #takes timestamp from data collector if given
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{model_name}"
    run_dir = os.path.join(output_dir, folder_name)

    # Only create run directory if not in data collection mode and saving yaml is enabled
    if not data_collection and config["data"]["save_yaml"]:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving current run data to: {run_dir}")

    try:
        config_save_path = os.path.join(run_dir, f"{model_name}config.yaml")
        # ======== Collisions/obstacles ========
        if config["collision"]["collision_avoidance_obstacle"] or config["collision"]["collision_avoidance_ground"]:                                                      # If enabled in config
            config = load_collision_config(config)                                        # Load obstacles

        # Only save yaml if not in data collection mode and saving yaml is enabled
        if not data_collection and config["data"]["save_yaml"]:
            save_yaml(config=config, save_path=config_save_path)                                                # Save summary with updated obstacles

        # ======== x0/yref ======== 
        if not config["IK"]["IK_required"]: 
            x0 = load_x0(config)                                                                            # Load x0 (Starting position)
            config["mpc"]["x0"] = x0.tolist()                                                               # Save x0 to config
            yref = load_yref(config)                                                                        # Load yref
            config["mpc"]["yref"] = yref.tolist()                                                           # Save yref to config

        else:                                                                                               # If dealing with manipulators
            IK = InverseKinematicsSolver(config)
            x0_q = IK.load_x0()                                                                                    # Load valid x0 in IK solver
            config = IK.config
            # Only save yaml if not in data collection mode and saving yaml is enabled
            if not data_collection and config["data"]["save_yaml"]:
                save_yaml(config=config, save_path=config_save_path)

            if config["IK"]["point_reference"]:                                                             # If not point reference, convert to trajectory reference
                yref = IK.load_yref()
                config = IK.config
            else:
                yref, config = IK.IK_to_XYZ(yref)                                                           # Add to config for summary saving purpose
                
        # Only save yaml if not in data collection mode and saving yaml is enabled
        if not data_collection and config["data"]["save_yaml"]:
            save_yaml(config=config, save_path=config_save_path)                                                # Save summary with IK inputs

        controller = CONTROLLER_REGISTRY[config["mpc"]["controller_name"]](config, worker_id)        # Create MPCController
        
        if config["VI"]["ground_truth_controller"]:
            # Make a **deep copy** of config so we don't touch the original
            GT_config = copy.deepcopy(config)

            # Apply changes from VI -> this merges nested dicts
            for key, value in config["VI"]["changes"].items():
                if isinstance(value, dict) and key in GT_config:
                    GT_config[key].update(value)  # merge nested dict
                else:
                    GT_config[key] = value
                    
            # Create ground truth controller
            gt_controller = CONTROLLER_REGISTRY[GT_config["mpc"]["controller_name"]](GT_config, worker_id)
        else:
            gt_controller = None

        # Record start time
        start_time = time.time()

        # Create simulator object
        simulator = MuJoCoSimulator(config, yref, controller, gt_controller)

        # Run simulation
        simulator.run()

        # Record end time and print elapsed time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nTotal execution time: {elapsed:.2f} seconds")

        if data_collection: #quit here if data collection
            # List of keys you want to keep
            keys_to_keep = ['qpos', 'qvel', 'total_cost', 'xyzpos', 'yref_xyz', 'yref_q', 'terminal_cost', 'GT_cost', 'sq_dist', 'obstacles']

            # Delete everything else
            for key in list(simulator.logs.keys()):  # use list() to avoid runtime dict size change
                if key not in keys_to_keep:
                    del simulator.logs[key]
            return simulator.logs
        elif config["mpc"]["solve_ocp"]: #quit here if only solving for single OCP
             return ocp_plot(simulator, run_dir, config)

    except Exception as e:
        import traceback
        print(f"\nRun failed: {e}")

        if not data_collection and config["data"]["save_yaml"]:
            # Save the error info in the run folder
            error_log_path = os.path.join(run_dir, "error.log")
            with open(error_log_path, "w") as f:
                f.write("=== Simulation Run Failed ===\n")
                f.write(f"Timestamp: {datetime.now()}\n\n")
                f.write(f"Error: {e}\n\n")
                f.write("=== Traceback ===\n")
                traceback.print_exc(file=f)
        raise

    finally:
        if not data_collection:
            try:
                # Save video if frames were recorded
                if simulator.frames and not config["mpc"]["solve_ocp"]:
                    save_video(simulator.frames, output_dir=run_dir, fps=simulator.config["mujoco"]["sim_framerate"])

                if config["data"]["save_data"]:
                    save_npz(data=simulator.logs, output_dir=run_dir, filename=f"{timestamp}_logs")

                # Attempt to plot logs if available
                if "logs" in simulator.__dict__ and "time" in simulator.logs and not config["mpc"]["solve_ocp"]:
                    plot_signals(
                        time=simulator.logs["time"],
                        logs=simulator.logs,
                        model=simulator.model,
                        config=config,
                        output_dir=run_dir,
                    )
            except Exception as plot_err:
                import traceback
    
                print("\n=== Error inside finally block ===")
                print(f"Error type: {type(plot_err).__name__}")
                print(f"Error message: {plot_err}")
                print("Traceback:")
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config for a given model")
    parser.add_argument("model", type=str, help="Name of the model to load from config (e.g., 'cartpole')")
    args = parser.parse_args()

    main(args.model)