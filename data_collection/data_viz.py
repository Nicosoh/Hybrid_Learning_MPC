import argparse
import yaml
from data_collection import load_npz
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
import os
import glob
import mplcursors
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin

def main(model_name, data_path, run, samples):
    # Load config for the model
    with open(f"configs/{model_name}config.yaml", "r") as f: # 
        config = yaml.safe_load(f)

    # Find the .npz file in the data_path
    npz_files = glob.glob(os.path.join(data_path, "*.npz"))

    if len(npz_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .npz file in {data_path}, "
            f"found {len(npz_files)}"
        )

    npz_file = npz_files[0]

    # Directory to save plots
    plots_dir = os.path.join(data_path, "plots")

    # Load data
    all_logs = load_npz(npz_file)
    if config["IK"]["IK_required"]:
        plot_traj_xyz(all_logs, config=config, samples=samples, frame_name="attachment_site", save_dir=plots_dir)
        plot_xyz_targets(all_logs, samples=samples, save_dir=plots_dir)

    # Run plotting functions
    plot_traj(all_logs, save_dir=plots_dir, samples=samples, config=config, run_filter=run)
    plot_dist(all_logs, save_dir=plots_dir,  config=config)

def plot_traj_xyz(all_logs, config, samples, frame_name, save_dir=None):
    base_dir = "models_xml"
    model_path = config["model"]["model_path"]
    filename = os.path.join(base_dir, model_path)

    robot = RobotWrapper.BuildFromMJCF(filename=filename)
    frame_id = robot.model.getFrameId(frame_name)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    run_keys = sorted(all_logs.keys())
    if samples is not None and samples < len(run_keys):
        run_keys = random.sample(run_keys, samples)
    # If 'samples' is None, plot all runs (explicit)
    else:
        print(f"No sampling requested — plotting all {len(run_keys)} runs.")

    for run_key in run_keys:
        run_data = all_logs[run_key]
        qpos_traj = run_data["qpos"]  # shape (T, nq)

        xyz = np.zeros((qpos_traj.shape[0], 3))

        for t, qpos in enumerate(qpos_traj):
            pin.forwardKinematics(robot.model, robot.data, qpos)
            pin.updateFramePlacements(robot.model, robot.data)
            xyz[t] = robot.data.oMf[frame_id].translation

        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            alpha=0.6,
            linewidth=1.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Trajectories of frame '{frame_name}'")

    ax.view_init(elev=30, azim=45)
    ax.grid(True)
    set_axes_equal(ax)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{frame_name}_traj_3d.png"), dpi=200)

    plt.tight_layout()
    plt.show()

def plot_xyz_targets(all_logs, samples, save_dir=None):
    """
    Plots all target (yref) points and all trajectory start/paths 
    in two separate 3D subplots for all selected runs.
    """
    run_keys = sorted(all_logs.keys())
    if samples is not None and samples < len(run_keys):
        run_keys = random.sample(run_keys, samples)
    else:
        print(f"No sampling requested — plotting all {len(run_keys)} runs.")
    
    # Initialize the figure and axes OUTSIDE the loop
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # To avoid messy legends, we only add labels once
    first_run = True

    for run_key in run_keys:
        run_data = all_logs[run_key]
        
        yref = run_data["yref_xyz"]    # Shape (3,)
        xyz_traj = run_data["xyzpos"]  # Shape (T, 3)
        start_pos = xyz_traj[0]        # First position

        # --- Plot on Subplot 1: Target References ---
        ax.scatter(
            yref[0], yref[1], yref[2], 
            c='red', s=40, alpha=0.6,
            label='yref (Goal)' if first_run else ""
        )

        # --- Plot on Subplot 2: Start Positions and Paths ---
        # Plot path
        ax.plot(
            xyz_traj[:, 0], xyz_traj[:, 1], xyz_traj[:, 2], 
            alpha=0.3, linewidth=1
        )
        # Plot starting point
        ax.scatter(
            start_pos[0], start_pos[1], start_pos[2], 
            c='green', s=40, alpha=0.6,
            label='Initial Position' if first_run else ""
        )
        
        first_run = False

    # Formatting Subplot 1
    ax.set_title(f"Target References (yref)\nN={len(run_keys)} runs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    ax.legend()

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "all_xyz_targets_comparison.png"), dpi=200)
    
    plt.show()

def set_axes_equal(ax):
    """
    Make a 3D plot have equal scale on all axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_traj(
    all_logs,
    save_dir,
    samples=None,
    seed=44,
    config=None,
    run_filter=None,
    tstep=10,    # <- Plot predicted trajectories every tstep steps
    hstep=2,     # <- Subsample points within each horizon prediction
    ):

    os.makedirs(save_dir, exist_ok=True) # Create output directory if it doesn't exist

    run_keys = sorted(all_logs.keys())

    # Filter specific run
    if run_filter:
        if run_filter in run_keys:
            run_keys = [run_filter]
        else:
            print(f"Run '{run_filter}' not found in logs.")
            return
    elif samples is not None and samples < len(run_keys):
        random.seed(seed)
        run_keys = random.sample(run_keys, samples)
    # If 'samples' is None, plot all runs (explicit)
    else:
        print(f"No sampling requested — plotting all {len(run_keys)} runs.")

    # Get all plots
    qpos_plots = {
        plot_name: {"index": index, "unit": unit}
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qpos"
    }

    qvel_plots = {
        plot_name: {"index": index, "unit": unit}
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qvel"
    }

    input_plots = {
        plot_name: {"index": index, "unit": unit}
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "ctrl"
    }

    base_cmap = plt.get_cmap("plasma")

    # Normalize for color mapping
    max_t = max(all_logs[run_key]["qpos"].shape[0] for run_key in run_keys)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_t - 1)

    # === QPOS TRAJ ===
    if any("qpos_traj" in all_logs[r] for r in run_keys):
        valid_runs = [r for r in run_keys if "qpos_traj" in all_logs[r]]
        plot_trajectory_group(
            all_logs=all_logs,
            run_keys=valid_runs,
            var_dict=qpos_plots,
            var_source="qpos_traj",
            true_source="qpos",
            save_dir=save_dir,
            base_cmap=base_cmap,
            norm=norm,
            tstep=tstep,
            hstep=hstep,
        )
    else:
        print("Skipping qpos_traj — not found in any run.")

    # === QVEL TRAJ ===
    if any("qvel_traj" in all_logs[r] for r in run_keys):
        valid_runs = [r for r in run_keys if "qvel_traj" in all_logs[r]]
        plot_trajectory_group(
            all_logs=all_logs,
            run_keys=valid_runs,
            var_dict=qvel_plots,
            var_source="qvel_traj",
            true_source="qvel",
            save_dir=save_dir,
            base_cmap=base_cmap,
            norm=norm,
            tstep=tstep,
            hstep=hstep,
        )
    else:
        print("Skipping qvel_traj — not found in any run.")

    # === INPUT TRAJ ===
    if any("u_traj" in all_logs[r] for r in run_keys):
        valid_runs = [r for r in run_keys if "u_traj" in all_logs[r]]
        plot_trajectory_group(
            all_logs=all_logs,
            run_keys=valid_runs,
            var_dict=input_plots,
            var_source="u_traj",
            true_source="u_applied",
            save_dir=save_dir,
            base_cmap=base_cmap,
            norm=norm,
            tstep=tstep,
            hstep=hstep,
        )
    else:
        print("Skipping u_traj — not found in any run.")
    
    # === PLOT COST OVER TIME ===
    fig_cost, ax_cost = plt.subplots(figsize=(10, 4))
    for run_key in run_keys:
        cost = all_logs[run_key]["total_cost"]  # shape: (timesteps,)
        timesteps = np.arange(len(cost))
        ax_cost.plot(timesteps, cost, label=run_key, alpha=0.7)

    ax_cost.set_title("MPC Cost over Time")
    ax_cost.set_xlabel("Timestep")
    ax_cost.set_ylabel("Cost")
    ax_cost.grid(True)
    cursor = mplcursors.cursor(ax_cost.lines, hover=False)
    cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig_cost.tight_layout()
    fig_cost.savefig(os.path.join(save_dir, "Cost_over_time.png"))
    
    fig_input, ax_input = plt.subplots(figsize=(10, 4))

    plotted_any = False

    for run_key in run_keys:
        if "u_applied" not in all_logs[run_key]:
            print(f"Skipping {run_key} — no 'u_applied'")
            continue

        u_applied = all_logs[run_key]["u_applied"]
        timesteps = np.arange(u_applied.shape[0])

        for i in range(u_applied.shape[1]):
            ax_input.plot(
                timesteps,
                u_applied[:, i],
                label=f"{run_key} - u_applied {i}",
                alpha=0.7
            )

        plotted_any = True

    if plotted_any:
        ax_input.set_title("MPC Input over Time")
        ax_input.set_xlabel("Timestep")
        ax_input.set_ylabel("Input Value")
        ax_input.grid(True)

        cursor = mplcursors.cursor(ax_input.lines, hover=False)
        cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

        fig_input.tight_layout()
        fig_input.savefig(os.path.join(save_dir, "Input_over_time.png"))
    else:
        print("Skipping input plot — no runs had 'u_applied'")

    # Show all plots
    plt.show()

def plot_trajectory_group(
    all_logs,
    run_keys,
    var_dict,       # dict: {name: (idx, unit)}
    var_source,     # "x_traj" or "u_traj"
    true_source,    # "qpos" or "u_applied"
    save_dir,
    base_cmap,
    norm,
    tstep,
    hstep,
):
    """Generic plotter for x_traj or u_traj variables."""

    for name, info in var_dict.items():
        idx = info["index"]
        unit = info["unit"]
        fig, ax = plt.subplots(figsize=(10, 5))

        for run_key in run_keys:
            traj_all = all_logs[run_key][var_source]      # predicted trajectories
            true_vals = all_logs[run_key][true_source]    # actual applied values
            num_timesteps, horizon = traj_all.shape[:2]

            # Plot predicted horizons
            for t in range(0, num_timesteps, tstep):
                traj = traj_all[t, ::hstep, :]
                color = base_cmap(norm(t))
                time_axis = np.arange(t, t + horizon)[::hstep]
                ax.plot(time_axis, traj[:, idx], color=color, label=f"{run_key}_t{t}", alpha=0.1)

            # Plot true/applied values
            ax.plot(
                np.arange(len(true_vals)),
                true_vals[:, idx],
                color="black",
                linewidth=1,
                label=run_key,
                alpha=0.5,
            )

        ax.set_ylabel(f"{name} [{unit}]")
        ax.set_title(f"{name} over time\n(Predictions every {tstep} steps, horizon subsampled by {hstep})")
        ax.set_xlabel("Timestep")
        ax.grid(True)

        cursor = mplcursors.cursor(ax.lines, hover=False)
        cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{name}_trajectory.png"))

def plot_dist(
    all_logs,
    save_dir,
    config=None,
):
    os.makedirs(save_dir, exist_ok=True)

    run_keys = sorted(all_logs.keys())

    # Get qpos indices from config
    qpos_plots = {
        plot_name: index
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qpos"
    }

    qvel_plots = {
        plot_name: index
        for plot_name, (source, index, unit) in config["plots"].items()
        if source == "qvel"
    }

    # === HISTOGRAMS for qpos ===
    for state_name, idx in qpos_plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))

        all_values = []
        for run_key in run_keys:
            qpos = all_logs[run_key]["qpos"]  # shape: (timesteps, state_dim)
            all_values.append(qpos[:, idx])   # gather values for this axis

        all_values = np.concatenate(all_values)  # flatten across runs
        counts, bins, patches = ax.hist(all_values, bins=80, color="teal", alpha=0.75)
        ax.set_title(f"Distribution of {state_name}")
        ax.set_xlabel(state_name)
        ax.set_ylabel("Frequency")
        ax.grid(True)

        # --- Add count labels above bars ---
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            if count > 0:
                ax.text(
                    (bin_left + bin_right) / 2,  # center of the bar
                    count,                      # height position
                    f"{int(count)}",
                    ha="center", va="bottom",
                    fontsize=8,
                    rotation=90,                # vertical label for compactness
                )

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{state_name}_histogram.png"))
    
     # === HISTOGRAMS for qvel values ===
    for state_name, idx in qvel_plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))

        all_values = []
        for run_key in run_keys:
            qvel = all_logs[run_key]["qvel"]  # shape: (timesteps, state_dim)
            all_values.append(qvel[:, idx])   # gather values for this axis

        all_values = np.concatenate(all_values)  # flatten across runs
        counts, bins, patches = ax.hist(all_values, bins=80, color="teal", alpha=0.75)
        ax.set_title(f"Distribution of {state_name}")
        ax.set_xlabel(state_name)
        ax.set_ylabel("Frequency")
        ax.grid(True)

        # --- Add count labels above bars ---
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            if count > 0:
                ax.text(
                    (bin_left + bin_right) / 2,  # center of the bar
                    count,                      # height position
                    f"{int(count)}",
                    ha="center", va="bottom",
                    fontsize=8,
                    rotation=90,                # vertical label for compactness
                )

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{state_name}_histogram.png"))

    # === Linear HISTOGRAM for cost ===
    fig_cost_linear, ax_cost_linear = plt.subplots(figsize=(8, 4))
    all_costs = []
    for run_key in run_keys:
        cost = all_logs[run_key]["total_cost"]
        all_costs.append(cost)

    all_costs = np.concatenate(all_costs)
    counts, bins, patches = ax_cost_linear.hist(all_costs, bins=80, color="crimson", alpha=0.7)
    ax_cost_linear.set_title("Distribution of MPC Cost")
    ax_cost_linear.set_xlabel("Cost")
    ax_cost_linear.set_ylabel("Frequency")
    ax_cost_linear.grid(True)

    # --- Add count labels above bars ---
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            ax_cost_linear.text(
                (bin_left + bin_right) / 2,  # center of the bar
                count,                      # height position
                f"{int(count)}",
                ha="center", va="bottom",
                fontsize=8,
                rotation=90,                # vertical label for compactness
            )
    fig_cost_linear.tight_layout()
    fig_cost_linear.savefig(os.path.join(save_dir, "Cost_histogram.png"))

    # === Log HISTOGRAM for cost ===
    fig_cost_log, ax_cost_log = plt.subplots(figsize=(8, 4))

    all_costs = []
    for run_key in run_keys:
        cost = all_logs[run_key]["total_cost"]
        all_costs.append(cost)

    all_costs = np.concatenate(all_costs)

    # --- Avoid zeros (log scale can't handle 0) ---
    eps = 1e-8
    all_costs_safe = np.clip(all_costs, eps, None)

    # --- Create log-spaced bins ---
    bins = np.logspace(
        np.log10(all_costs_safe.min()),
        np.log10(all_costs_safe.max()),
        80
    )

    counts, bins, patches = ax_cost_log.hist(
        all_costs_safe,
        bins=bins,
        color="crimson",
        alpha=0.7
    )

    # --- Set x-axis to log scale ---
    ax_cost_log.set_xscale("log")

    ax_cost_log.set_title("Distribution of MPC Cost (Log Bins)")
    ax_cost_log.set_xlabel("Cost (log scale)")
    ax_cost_log.set_ylabel("Frequency")
    ax_cost_log.grid(True, which="both")

    # --- Add count labels (optional, but can get messy on log scale) ---
    for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            ax_cost_log.text(
                np.sqrt(bin_left * bin_right),  # geometric center for log scale
                count,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    fig_cost_log.tight_layout()
    fig_cost_log.savefig(os.path.join(save_dir, "Cost_histogram_log_bins.png"))

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data for a model")
    parser.add_argument("model", type=str, help="Model name to load from config.yaml")
    parser.add_argument("data_path", type=str, help="Path to the collected data directory")
    parser.add_argument("--run", type=str, default=None, help="Optional: specific run key (e.g., run_001)")
    parser.add_argument("--samples", type=int, default=None, help="Optional: number of samples to plot")
    args = parser.parse_args()

    # Call main with all args
    main(args.model, args.data_path, args.run, args.samples)