import re
import os
import ast
import json
import yaml
import mujoco
import numpy as np
import casadi as ca
import mediapy as media
import matplotlib.pyplot as plt

from casadi import vertcat
from scipy.spatial.transform import Rotation as R
from robot_descriptions.loaders.mujoco import load_robot_description

# ========== PLOTTING ==========
# def plot_signals(time, logs, model, config, output_dir, file_name="plot"):
#     """
#     Processes signals based on plots_config and plots them over time.
#     Now plots stage, terminal, and total costs on the same graph.
#     """
#     plots_config = config["plots"]
#     IK_required = config["IK"]["IK_required"]
#     output_xyz = config["IK"]["output_xyz"]

#     signals = {}
#     ylabel_units = {}

#     # Ensure numeric lists in logs are converted to numpy arrays
#     for key, val in logs.items():
#         if isinstance(val, list):
#             if len(val) == 0 or not isinstance(val[0], dict):
#                 logs[key] = np.array(val)

#     # Process yref
#     yref_full = logs.get("yref", None)
#     if yref_full is not None and IK_required:
#         # Convert list of dicts into array if necessary
#         yref_full = np.array([step["stage"][0] for step in yref_full])

#     # Offsets to access yref by type
#     source_offsets = {
#         "qpos": 0,
#         "qvel": model.nq,
#         "ctrl": model.nq + model.nv,
#         "u_applied": model.nq + model.nv,
#     }

#     # Separate cost keys to plot together
#     cost_keys = ["stage_cost", "terminal_cost", "total_cost"]

#     for name, (source, idx, unit) in plots_config.items():
#         if source == "xyzpos":
#             assert idx < 3
#             signals[name] = logs["xyzpos"][:, idx]

#         elif source == "qpos":
#             assert idx < model.nq
#             signals[name] = logs["qpos"][:, idx]

#         elif source == "qvel":
#             assert idx < model.nv
#             signals[name] = logs["qvel"][:, idx]

#         elif source in ["ctrl", "u_applied"]:
#             assert idx < model.nu
#             signals[name] = logs["u_applied"][:, idx]

#         elif source in cost_keys:
#             pass

#         else:
#             raise ValueError(f"Invalid signal source '{source}'")

#         ylabel_units[name] = unit

#     # --- Now do plotting ---
#     os.makedirs(output_dir, exist_ok=True)
#     full_path = os.path.join(output_dir, f"{file_name}.jpg")

#     # Prepare subplots
#     n = len(signals)
#     # Add 1 extra subplot for cost if any cost keys exist
#     if any(k in logs for k in cost_keys):
#         n += 1

#     dpi = 120
#     width, height = 800, 200 * n
#     figsize = (width / dpi, height / dpi)

#     fig, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)
#     if n == 1:
#         ax = [ax]

#     # Plot normal signals
#     for i, (name, values) in enumerate(signals.items()):
#         ax[i].plot(time, values, label="Actual")
#         source, idx, _ = plots_config[name]
#         if yref_full is not None and source in source_offsets:
#             yref_idx = source_offsets[source] + idx
#             if yref_idx < yref_full.shape[1]:
#                 ax[i].plot(time, yref_full[:, yref_idx], "--", label="Ref")
#         ax[i].set_title(name)
#         ax[i].set_ylabel(ylabel_units.get(name, ""))
#         ax[i].legend(loc="best")

#     # Plot cost on last subplot
#     if any(k in logs for k in cost_keys):
#         cost_ax = ax[-1]
#         for key in cost_keys:
#             if key in logs:
#                 cost_ax.plot(time, logs[key], label=key.replace("_", " ").capitalize())
#         cost_ax.set_title("MPC Cost")
#         cost_ax.set_ylabel("Cost")
#         cost_ax.legend(loc="best")

#     ax[-1].set_xlabel("Time (s)")
#     plt.tight_layout()
#     plt.savefig(full_path)
#     print(f"Plot saved to: {os.path.abspath(full_path)}")
#     plt.show()

def plot_signals(time, logs, model, config, output_dir, file_name="plot"):
    """
    Plot logged signals over time.

    Supports:
    - Joint-space plots (qpos, qvel, ctrl)
    - Task-space EE plots (xyzpos)
    - Selective reference plotting based on output_xyz flag
    - Multi-column subplot layout
    """

    plots_config = config["plots"]
    IK_required = config["IK"]["IK_required"]
    output_xyz = config["IK"]["output_xyz"]

    signals = {}
    ylabel_units = {}

    # -----------------------------
    # Convert list logs to numpy
    # -----------------------------
    for key, val in logs.items():
        if isinstance(val, list):
            if len(val) == 0 or not isinstance(val[0], dict):
                logs[key] = np.array(val)

    # -----------------------------
    # Process yref (time-varying ref)
    # -----------------------------
    yref_full = logs.get("yref", None)
    if yref_full is not None and IK_required:
        if not config["IK"]["point_reference"]:
            yref_full = np.array([step["stage"][0] for step in yref_full])
        else:
            yref_full = np.array(yref_full)

    # -----------------------------
    # Reference layout offsets
    # -----------------------------
    if output_xyz:
        source_offsets = {
            "xyzpos": 0,
            "qvel": 3,
            "ctrl": 3 + model.nv,
            "u_applied": 3 + model.nv,
        }
    else:
        source_offsets = {
            "qpos": 0,
            "qvel": model.nq,
            "ctrl": model.nq + model.nv,
            "u_applied": model.nq + model.nv,
        }

    # -----------------------------
    # Safety check (recommended)
    # -----------------------------
    if yref_full is not None:
        expected_dim = (
            3 + model.nv + model.nu if output_xyz
            else model.nq + model.nv + model.nu
        )
        assert yref_full.shape[1] == expected_dim, (
            f"yref_full has wrong dimension: "
            f"{yref_full.shape[1]} != {expected_dim}"
        )

    # -----------------------------
    # Cost keys
    # -----------------------------
    cost_keys = ["stage_cost", "terminal_cost", "total_cost"]
    has_cost = any(k in logs for k in cost_keys)

    # -----------------------------
    # Helper: reference gating
    # -----------------------------
    def should_plot_ref(source):
        if not output_xyz:
            return True
        return source != "qpos"

    # -----------------------------
    # Extract signals
    # -----------------------------
    for name, (source, idx, unit) in plots_config.items():

        if source == "xyzpos":
            assert idx < 3
            signals[name] = logs["xyzpos"][:, idx]

        elif source == "qpos":
            assert idx < model.nq
            signals[name] = logs["qpos"][:, idx]

        elif source == "qvel":
            assert idx < model.nv
            signals[name] = logs["qvel"][:, idx]

        elif source in ["ctrl", "u_applied"]:
            assert idx < model.nu
            signals[name] = logs["u_applied"][:, idx]
        
        elif source == "sq_dist":
            assert "sq_dist" in logs, "sq_dist not found in logs"
            assert idx < logs["sq_dist"].shape[1]
            signals[name] = logs["sq_dist"][:, idx]

        elif source in cost_keys:
            continue

        else:
            raise ValueError(f"Invalid signal source '{source}'")

        ylabel_units[name] = unit

    # -----------------------------
    # Plot layout (3 columns)
    # -----------------------------
    n_plots = len(signals) + (1 if has_cost else 0)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    dpi = 60
    fig_width = 1200 / dpi
    fig_height = (250 * n_rows) / dpi

    fig, ax = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        sharex=True,
    )

    ax = np.array(ax).reshape(-1)

    # -----------------------------
    # Plot signals
    # -----------------------------
    plot_idx = 0

    for name, values in signals.items():
        a = ax[plot_idx]
        a.plot(time, values, label="Actual")

        source, idx, _ = plots_config[name]

        if should_plot_ref(source) and yref_full is not None:

            # Task-space XYZ reference
            if source == "xyzpos" and output_xyz:
                yref_idx = source_offsets["xyzpos"] + idx
                a.plot(time, yref_full[:, yref_idx], "--", label="Ref")

            # Joint / velocity / control reference
            elif source in source_offsets:
                yref_idx = source_offsets[source] + idx
                if yref_idx < yref_full.shape[1]:
                    a.plot(time, yref_full[:, yref_idx], "--", label="Ref")

        a.set_title(name)
        a.set_ylabel(ylabel_units.get(name, ""))
        a.legend(loc="best")

        plot_idx += 1

    # -----------------------------
    # Plot costs
    # -----------------------------
    if has_cost:
        cost_ax = ax[plot_idx]
        for key in cost_keys:
            if key in logs:
                cost_ax.plot(time, logs[key], label=key.replace("_", " ").title())

        cost_ax.set_title("MPC Cost")
        cost_ax.set_ylabel("Cost")
        cost_ax.legend(loc="best")
        plot_idx += 1

    # -----------------------------
    # Hide unused subplots
    # -----------------------------
    for i in range(plot_idx, len(ax)):
        ax[i].axis("off")

    ax[0].figure.text(0.5, 0.04, "Time (s)", ha="center")

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{file_name}.jpg")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(full_path)
    print(f"Plot saved to: {os.path.abspath(full_path)}")
    plt.show()

def ocp_plot(simulator, output_dir, config,file_name="OCP_plot"):
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{file_name}.jpg")

    config = simulator.config
    logs = simulator.logs

    qpos_traj = logs["qpos_traj"][0]
    qvel_traj = logs["qvel_traj"][0]
    u_traj = logs["u_traj"][0]
    yref = logs["yref"]

    stage_cost = logs["stage_cost"]
    terminal_cost = logs["terminal_cost"]
    total_cost = logs["total_cost"]

    print(f"Total stage cost: {stage_cost}")
    print(f"Total terminal cost: {terminal_cost}")
    print(f"Total cost: {total_cost}")

    if config["IK"]["IK_required"]:
        if not config["IK"]["point_reference"]:
            yref = np.vstack([yref[0]["stage"], yref[0]["terminal"]])

    nq = simulator.model.nq
    
    dt = config["mpc"]["mpc_timestep"]
    T = qpos_traj.shape[0]  # Total time steps (N+1)
    time = np.arange(T) * dt  # Time axis for states

    time_u = np.arange(u_traj.shape[0]) * dt  # Time axis for control inputs

    if config["IK"]["IK_required"] and not config["IK"]["point_reference"]:
        yref_qpos = yref[:, :nq]
        yref_qvel = yref[:, nq:2*nq]
        yref_u = yref[:-1, 2*nq:]
    else:
        # Extract constant reference from first yref entry (ignore time)
        yref_qpos = np.tile(yref[0, : nq], (T, 1))       # shape (T, nq)
        yref_qvel = np.tile(yref[0, nq : 2 * nq], (T, 1))  # shape (T, nq)
        yref_u = np.tile(yref[0, 2 * nq :], (u_traj.shape[0], 1))  # shape (T-1, nu)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot positions
    axs[0].plot(time, qpos_traj, label="Trajectory")
    axs[0].plot(time, yref_qpos, "--", label="Reference")
    axs[0].set_ylabel("Positions")
    axs[0].legend([f"q{i}" for i in range(qpos_traj.shape[1])])
    axs[0].grid(True)

    # Plot velocities
    axs[1].plot(time, qvel_traj, label="Trajectory")
    axs[1].plot(time, yref_qvel, "--", label="Reference")
    axs[1].set_ylabel("Velocities")
    axs[1].legend([f"v{i}" for i in range(qvel_traj.shape[1])])
    axs[1].grid(True)

    # Plot control inputs
    axs[2].plot(time_u, u_traj, label="Control")
    axs[2].plot(time_u, yref_u, "--", label="Reference")
    axs[2].set_ylabel("Control Inputs")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend([f"u{i}" for i in range(u_traj.shape[1])])
    axs[2].grid(True)

    plt.tight_layout()

    plt.savefig(full_path)
    print(f"Plot saved to {full_path}")

    plt.show()

def expand_yref_over_time(yref, time):
    """
    Expand sparse yref definitions into a full yref array aligned with `time`.

    Parameters
    ----------
    yref_raw : np.ndarray
        Array of shape (T_ref, ny+1) where first column is time.
    time : np.ndarray
        Simulation time vector of shape (T_sim,).

    Returns
    -------
    yref_full : np.ndarray
        Array of shape (T_sim, ny), repeated using zero-order hold.
    """
    yref_times = yref[:, 0]
    yref_values = yref[:, 1:]

    yref_full = np.zeros((len(time), yref_values.shape[1]))
    current_index = 0

    for i, t in enumerate(time):
        # Move to the latest reference before or at time t
        while (current_index + 1 < len(yref_times)) and (yref_times[current_index + 1] <= t):
            current_index += 1
        yref_full[i] = yref_values[current_index]

    return yref_full

def save_video(frames, output_dir, file_name="video", fps=30):
    """
    Saves simulation frames as a video file with a running number.

    Parameters
    ----------
    frames : list or np.ndarray
        Video frames.
    output_dir : str
        Directory to save the video.
    file_name : str
        Filename.
    fps : int
        Frames per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{file_name}.mp4")

    media.write_video(full_path, frames, fps=fps)
    print(f"Video saved to: {os.path.abspath(full_path)}")

# ========== SUMMARY SAVING ==========

def save_yaml(config, save_path):
    with open(save_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

# ========== Loading yref ==========
def load_yref(config):
    if config["mpc"]["yref_random"]:
        return randomise_yref(config)
    else:
        return np.array(config["mpc"]["yref"])
    
def randomise_yref(config):
    yref_range = config["mpc"]["yref_range"]
    sampling = config["mpc"]["yref_sampling"]

    if sampling == "uniform":
        min_xyz = np.array(yref_range[0])
        max_xyz = np.array(yref_range[1])

        yref = np.random.uniform(low=min_xyz, high=max_xyz)
    else:
        raise ValueError(f"Unsupported yref_sampling method: {sampling}")
    
    print(f"Randomised yref state: {yref}")

    return yref

# ========== Loading x0 ==========
def load_x0(config):
    # Randomise inital state if specified
    if config["mpc"]["x0_random"]:
        x0 = randomise_x0(config)
        print(f"Randomised x0 state: {x0}")
    else:
        x0 = np.array(config["mpc"]["x0"])
    return x0

def randomise_x0(config):
    x0_range = config["mpc"]["x0_range"]
    sampling = config["mpc"]["x0_sampling"]

    if sampling == "uniform":
        min_xyz = np.array(x0_range[0])
        max_xyz = np.array(x0_range[1])

        x0 = np.random.uniform(low=min_xyz, high=max_xyz)
    else:
        raise ValueError(f"Unsupported x0_sampling method: {sampling}")
    return x0

# ========== Loading obstacles/collision setup ==========
def load_collision_config(config):
    collision_config = config["collision"]

    # ==========================================================
    # OBSTACLES
    # ==========================================================
    if collision_config["collision_avoidance_obstacle"]:
        if collision_config["obstacles_random"]:
            obstacles = randomise_obstacles(collision_config)
            collision_config["obstacles"] = to_yaml_safe(obstacles)
        # Since not random, expect obstacles to be fully defined in config
        else:
            required_keys = ["from", "to", "radius", "rpy", "center"]

            for name, obs in collision_config["obstacles"].items():
                for key in required_keys:
                    if key not in obs:
                        raise ValueError(f"Obstacle '{name}' missing '{key}'")

    # ==========================================================
    # GROUND
    # ==========================================================
    if collision_config["collision_avoidance_ground"]:
        if collision_config["ground_plane"] is None:
            raise ValueError("collision_avoidance_ground=True but no ground_plane defined")


    # Validate
    validate_collision_config(collision_config)

    # Replace original collision config
    config["collision"] = collision_config

    return config

def randomise_obstacles(collision_cfg):
    num_obs = collision_cfg["obstacles_num"]
    sampling = collision_cfg["obstacles_sampling"]

    # Updated config structure expectations:
    # center_min/max: [x, y, z] bounds for the center
    # length: float
    # radius_range: [min, max]
    center_range = collision_cfg["obstacle_center_range"]
    length = collision_cfg["obstacle_length"]
    radius = collision_cfg["obstacle_radius"]
    obstacle_RPY_range = collision_cfg["obstacle_RPY_range"]

    obstacles = {}

    if sampling != "uniform":
        raise ValueError(f"Unsupported obstacles_sampling method: {sampling}")

    for i in range(num_obs):
        obs_name = f"obs{i+1}"

        # 1. Sample the center location
        center = np.random.uniform(low=center_range[0], high=center_range[1])

        # 2. Sample Roll, Pitch, Yaw (in radians)
        # Usually: Roll [-pi, pi], Pitch [-pi/2, pi/2], Yaw [-pi, pi]
        rpy = np.random.uniform(
            low=obstacle_RPY_range[0], 
            high=obstacle_RPY_range[1]
        )

        # 3. Calculate direction vector from RPY
        # We assume the capsule's default axis is along X [1, 0, 0]
        rot = R.from_euler('xyz', rpy)
        direction = rot.apply([1, 0, 0]) 

        # 4. Calculate endpoints based on fixed length
        half_len = length / 2.0
        from_pt = center - (direction * half_len)
        to_pt   = center + (direction * half_len)

        obstacles[obs_name] = {
            "from": from_pt,
            "to": to_pt,
            "radius": float(radius),
            "rpy": rpy, # Useful for debugging or visualization
            "center": center
        }

    return obstacles

def validate_collision_config(collision):

    links = collision["links"]

    # ==========================================================
    # OBSTACLE VALIDATION
    # ==========================================================
    if collision["collision_avoidance_obstacle"]:

        obstacles = collision["obstacles"]

        for link_name, obs_name in collision["collision_pairs"].get("obstacle", []):
            if link_name not in links:
                raise KeyError(f"Unknown link in obstacle collision_pairs: {link_name}")
            if obs_name not in obstacles:
                raise KeyError(f"Unknown obstacle in collision_pairs: {obs_name}")

    # ==========================================================
    # GROUND VALIDATION
    # ==========================================================
    if collision["collision_avoidance_ground"]:

        if collision["ground_plane"] is None:
            raise ValueError("collision_avoidance_ground=True but no ground_plane defined")

        for link_name, ground_name in collision["collision_pairs"].get("ground", []):
            if link_name not in links:
                raise KeyError(f"Unknown link in ground collision_pairs: {link_name}")
            if ground_name != "ground":
                raise KeyError("Ground collision pair must use name 'ground'")

# Load model from xml file
def load_scene_from_xml(config):
    """Load a MuJoCo model and create associated data object."""
    base_dir = "models_xml"

    scene_path = config["model"]["scene_path"]
    filename = os.path.join(base_dir, scene_path)

    model = mujoco.MjModel.from_xml_path(filename)
    data = mujoco.MjData(model)

    apply_model_config(config, model)
    
    return model, data

def load_model_from_xml(config):
    """Load a MuJoCo model and create associated data object."""
    base_dir = "models_xml"

    scene_path = config["model"]["model_path"]
    filename = os.path.join(base_dir, scene_path)

    model = mujoco.MjModel.from_xml_path(filename)
    data = mujoco.MjData(model)

    apply_model_config(config, model)

    return model, data

# Apply model config only if loaded from xml
def apply_model_config(config, model):
    try:
        for body_name in config["model"]["mass"].keys():
            mass_value = config["model"]["mass"][body_name] # Respective mass value
            inertia_value = config["model"]["inertia"][body_name] # Respective inertia value
            # Find the body ID you want to modify
            body_id = model.body(name=body_name)
            # Assign the new mass & inertia value
            body_id.mass = mass_value
            body_id.inertia = inertia_value

            # Print to verify
            print(f"Updated body '{body_name}': mass={body_id.mass}, inertia={body_id.inertia}")
    except KeyError:
        print("No custom mass or inertia values found in config; using model defaults.")

def init_scene_options():
    """Initialize visualization options for rendering."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    scene_option.geomgroup[3] = 1 # Enable collision capsules
    scene_option.sitegroup[2] = 1 # Enable "attachement_site"
    scene_option.sitegroup[:2] = 0 # Disable all others
    # scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM

    # Show coordinate frames at sites
    scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE # Add x,y,z aces at sites``
    return scene_option

def get_yref_at_time(t_now, yref):
    """
    Get the most recent reference (no interpolation) for the given time.

    Args:
        t_now (float): Current time in seconds.

    Returns:
        ref (np.ndarray): Reference state at or before time t_now.
    """
    times = yref[:, 0]
    states = yref[:, 1:]

    # If before first timestamp, return the first reference
    if t_now <= times[0]:
        return states[0]
    
    # Find the last index where time <= t_now
    idx = np.searchsorted(times, t_now, side='right') - 1
    return states[idx]

def get_num_config(section, option, config):
    return ast.literal_eval(f"({config.get(section, option)})")

# ========== Convert items before saving to YAML ==========
def to_yaml_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_yaml_safe(v) for v in obj]
    else:
        return obj

# ========== Controller, Add Constrtaints ============
def clamp(x, lo=0.0, hi=1.0):
    return ca.fmin(ca.fmax(x, lo), hi)

def segment_segment_squared_distance():
    # Inputs
    p1 = ca.SX.sym("p1", 3)                 # Point 1 of segment 1
    q1 = ca.SX.sym("q1", 3)                 # Point 2 of segment 1
    p2 = ca.SX.sym("p2", 3)                 # Point 1 of segment 2
    q2 = ca.SX.sym("q2", 3)                 # Point 2 of segment 2

    eps = 1e-8

    d1 = q1 - p1                            # Direction vector of segment 1
    d2 = q2 - p2                            # Direction vector of segment 2
    r  = p1 - p2

    a = ca.dot(d1, d1)                      # Squared length of segment 1
    e = ca.dot(d2, d2)                      # Squared length of segment 2
    f = ca.dot(d2, r)
    c = ca.dot(d1, r)
    b = ca.dot(d1, d2)

    denom = a * e - b * b

    # Initialize
    s = ca.SX(0)                            # Position along segment 1
    t = ca.SX(0)                            # Position along segment 2

    # --- Both segments degenerate into points ---
    both_degenerate = ca.logic_and(a <= eps, e <= eps)

    s = ca.if_else(both_degenerate, 0.0, s)             # Set s = 0 if true
    t = ca.if_else(both_degenerate, 0.0, t)             # Set t = 0 if true

    # --- First segment degenerate into a point only ---
    first_degenerate = ca.logic_and(a <= eps, e > eps)
    t_fd = clamp(f / e)
    s = ca.if_else(first_degenerate, 0.0, s)            # Set s = 0 if true
    t = ca.if_else(first_degenerate, t_fd, t)           # Set t as the clamped point

    # --- Second segment degenerate into a point only ---
    second_degenerate = ca.logic_and(a > eps, e <= eps)
    s_sd = clamp(-c / a)
    s = ca.if_else(second_degenerate, s_sd, s)          # Set s as the clamped point
    t = ca.if_else(second_degenerate, 0.0, t)           # Set t = 0 if true

    # --- General case ---
    general = ca.logic_and(a > eps, e > eps)            # True is both is not a point

    s_gc = ca.if_else(                                  # If lines are not parallel (denom > 0)
        denom > eps,                                    # Then clamp segment 1
        clamp((b * f - c * e) / denom),                 # If not set s to zero.
        0.0
    )

    t_gc = (b * s_gc + f) / e                           # Compute closest point on segment 2 w.r.t s_gc.

    # Clamp t and recompute s if needed
    t_gc_clamped = clamp(t_gc)                          # Premptive clamping, does not matter if it already within [0,1]

    s_gc = ca.if_else(                                  
        t_gc < 0.0,                                     # If t is less than 0.0, outside of the segment.
        clamp(-c / a),                                  # Clamp t to 0.0 and recalculate s
        ca.if_else(                                     
            t_gc > 1.0,                                 # If t is more than 1.0, outside of segment
            clamp((b - c) / a),                         # Clamp t to 1.0 and recalculate s
            s_gc                                        # If both not true, then s_gc remains and done.
        )
    )

    s = ca.if_else(general, s_gc, s)
    t = ca.if_else(general, t_gc_clamped, t)

    # Closest points
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t

    dist2 = ca.dot(c1 - c2, c1 - c2)

    return ca.Function(
        "seg_seg_sq_dist",
        [p1, q1, p2, q2],
        [dist2, s, t, c1, c2],
        ["p1", "q1", "p2", "q2"],
        ["dist2", "s", "t", "c1", "c2"]
    )

def build_obstacle_collision_constraints(robot_sys, links, obstacles, collision_pairs, p_obs):
    constraints = []

    for link_name, obs_name in collision_pairs:
        seg_seg_sq_dist = segment_segment_squared_distance()    # ca function

        link = links[link_name]

        p1 = getattr(robot_sys, link["from"])   # CasADi 3-vector
        q1 = getattr(robot_sys, link["to"])     # CasADi 3-vector
        r1 = link["radius"]

        obs = obstacles[obs_name]
        # p2 = obs["from"]
        # q2 = obs["to"]
        p2 = p_obs[:3]
        q2 = p_obs[3:]
        r2 = obs["radius"]

        dist2, s, t, c1, c2 = seg_seg_sq_dist(p1, q1, p2, q2)

        min_dist2 = (r1 + r2)**2

        constraints.append(dist2 - min_dist2)

    return vertcat(*constraints)

def capsule_plane_distance():

    # Symbolic variables
    a = ca.MX.sym('a', 3)      # segment start
    b = ca.MX.sym('b', 3)      # segment end
    n = ca.MX.sym('n', 3)      # plane normal (assumed normalized)
    d = ca.MX.sym('d')         # plane offset
    r = ca.MX.sym('r')         # capsule radius

    # Signed distances
    da = ca.dot(n, a) - d
    db = ca.dot(n, b) - d

    # Check if segment crosses plane
    crosses = da * db <= 0

    # Distance if not crossing
    abs_da = ca.fabs(da)
    abs_db = ca.fabs(db)
    min_dist = ca.if_else(abs_da < abs_db, abs_da, abs_db)

    seg_dist = ca.if_else(crosses, 0, min_dist)

    # Capsule distance
    final_dist = ca.if_else(seg_dist - r > 0, seg_dist - r, 0)

    # Create CasADi function
    f = ca.Function(
        'capsule_plane_distance',
        [a, b, n, d, r],
        [final_dist]
    )

    return f

def capsule_plane_signed_distance():
    a = ca.MX.sym('a', 3)
    b = ca.MX.sym('b', 3)
    n = ca.MX.sym('n', 3)
    d = ca.MX.sym('d')
    r = ca.MX.sym('r')

    da = ca.dot(n, a) - d
    db = ca.dot(n, b) - d

    # Closest endpoint to plane (signed)
    # closest = ca.if_else(da < db, da, db)
    closest = ca.fmin(da, db)

    # Signed distance (can be negative = penetration)
    signed_dist = closest - r

    f = ca.Function(
        'capsule_plane_signed_distance',
        [a, b, n, d, r],
        [signed_dist]
    )

    return f

def build_ground_collision_constraints(robot_sys, links, ground_plane, collision_pairs):
    constraints = []

    for link_name, obs_name in collision_pairs:
        # cap_plane_dist = capsule_plane_distance()    # ca function
        cap_plane_dist = capsule_plane_signed_distance()

        link = links[link_name]

        p1 = getattr(robot_sys, link["from"])   # CasADi 3-vector
        q1 = getattr(robot_sys, link["to"])     # CasADi 3-vector
        r1 = link["radius"]

        n = ground_plane[:3]
        d = ground_plane[3]

        dist = cap_plane_dist(p1, q1, n, d, r1)

        constraints.append(dist)

    return vertcat(*constraints)

def plot_metrics(root_dir, plt_save_path):
    # ==========================
    # Configuration
    # ==========================
    # root_dir = "value_iteration/output/2026-02-13_16-58-33_TwoDofArm_VI"
    # plt_save_path = "metrics_plot.png"

    # ==========================
    # Collect metrics
    # ==========================
    loop_pattern = re.compile(r"loop_(\d+)")

    loops = []
    gt_cost = []
    ctrl_cost = []
    mse = []
    mse_std = []
    train_loss = []
    stationary_ratio_mean = []

    # Scan directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        if os.path.isdir(folder_path):
            match = loop_pattern.match(folder)
            if match:
                loop_number = int(match.group(1))
                metrics_path = os.path.join(folder_path, "metrics.json")
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                    
                    loops.append(loop_number)
                    gt_cost.append(data.get("gt_cost", np.nan))
                    ctrl_cost.append(data.get("ctrl_cost", np.nan))
                    mse.append(data.get("mse", np.nan))
                    mse_std.append(data.get("mse_std", np.nan))
                    train_loss.append(data.get("train_loss", np.nan))
                    stationary_ratio_mean.append(data.get("stationary_ratio_mean", np.nan))

    # ==========================
    # Sort everything by loop number
    # ==========================
    sorted_idx = np.argsort(loops)

    loops = np.array(loops)[sorted_idx]
    gt_cost = np.array(gt_cost)[sorted_idx]
    ctrl_cost = np.array(ctrl_cost)[sorted_idx]
    mse = np.array(mse)[sorted_idx]
    mse_std = np.array(mse_std)[sorted_idx]
    train_loss = np.array(train_loss)[sorted_idx]
    stationary_ratio_mean = np.array(stationary_ratio_mean)[sorted_idx]

    # ==========================
    # Plot setup
    # ==========================
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    def update_plot(x, ground_truth, controller, MSE, MSE_std, train_loss, stationary_ratio_mean):
        """Update and save plot with current metrics in separate subplots."""
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # Top subplot
        ax1.plot(x, ground_truth, label="Ground Truth", linewidth=0.8)
        ax1.plot(x, controller, label="Controller", linewidth=0.8)
        ax1.set_ylabel("Mean Cost")
        ax1.grid(True, which="both", linestyle=":", alpha=0.4)
        ax1.legend()
        
        # Middle subplot
        MSE = np.array(MSE)
        MSE_std = np.array(MSE_std)
        
        ax2.plot(x, MSE, label="MSE", color='green', linewidth=0.8)
        if len(MSE_std) == len(MSE):
            ax2.fill_between(x, MSE - MSE_std, MSE + MSE_std,
                            color='r', alpha=0.3, label="MSE ± std")
        
        ax2.set_ylabel("Mean Squared Error")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", linestyle=":", alpha=0.4)
        ax2.legend(loc='upper right')
        
        # Bottom subplot
        train_loss = np.array(train_loss)
        ax3.plot(x, train_loss, label="Train Loss",
                color='b', linewidth=0.8)
        ax3.set_xlabel("Value Iteration Loop")
        ax3.set_yscale("log")
        ax3.set_ylabel("Train Loss")
        ax3.grid(True, which="both", linestyle=":", alpha=0.4)
        ax3.legend(loc='upper right')

        # Bottom subplot
        stationary_ratio_mean = np.array(stationary_ratio_mean)
        ax4.plot(x, stationary_ratio_mean, label="Stationary Ratio Mean",
                color='b', linewidth=0.8)
        ax4.set_xlabel("Value Iteration Loop")
        ax4.set_yscale("log")
        ax4.set_ylabel("Stationary Ratio Mean")
        ax4.grid(True, which="both", linestyle=":", alpha=0.4)
        ax4.legend(loc='upper right')
        
        fig.tight_layout()
        plt.savefig(plt_save_path, dpi=400, bbox_inches='tight')
        print(f"Saved plot to {plt_save_path}")

    # ==========================
    # Run plotting
    # ==========================
    update_plot(loops, gt_cost, ctrl_cost, mse, mse_std, train_loss, stationary_ratio_mean)
    plt.close(fig)