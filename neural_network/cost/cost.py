import numpy as np

def compute_l2_cost(config, xyz, qvel, yref_xyz, u):
    """
    Compute L2 tracking + velocity + control cost.

    Args:
        config: configuration object containing cost weights
        xyz: (N, 3) end-effector positions
        qvel: (N, 2) joint velocities
        yref_xyz: (3,) reference position
        u: (N, 2) control input (optional)

    Returns:
        cost: (N,) stage cost
    """
    Q_mat = np.array(config["mpc"]["Q_mat"])
    Q_dot_mat = np.array(config["mpc"]["Q_dot_mat"])
    R_mat = np.array(config["mpc"]["R_mat"])

    # -------------------------
    # Tracking cost
    # -------------------------
    error = xyz - yref_xyz
    tracking_cost = np.sum(Q_mat * (error ** 2), axis=1)

    # -------------------------
    # Velocity cost
    # -------------------------
    vel_cost = np.sum(Q_dot_mat * (qvel ** 2), axis=1)

    # -------------------------
    # Control cost
    # -------------------------
    input_cost = np.sum(R_mat * (u ** 2), axis=1)

    # -------------------------
    # Total cost
    # -------------------------
    cost = tracking_cost + vel_cost + input_cost

    return cost