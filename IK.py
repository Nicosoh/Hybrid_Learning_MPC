import os
import sys
import numpy as np
import pinocchio as pin
import qpsolvers

import hppfcl as fcl
import meshcat_shapes
import pink

from pink.utils import process_collision_pairs
from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
from pinocchio.robot_wrapper import RobotWrapper
from pink.limits import AccelerationLimit

class InverseKinematicsSolver:
    def __init__(self, config):
        self.config = config
        self.collision_config = config["collision"]

        self.zone_idx = self.sample_zone()
        self.config["mpc"]["zone_idx"] = self.zone_idx

        # Extract IK config
        self.visualize = self.config["IK"]["visualize_IK"]
        self.max_iterations = self.config["IK"]["max_iterations"]
        self.stop_thres = self.config["IK"]["stop_threshold"]
        self.dt = config["mpc"]["mpc_timestep"]
        self.velocity_thres = self.config["IK"]["velocity_threshold"]
        self.limit_accel = self.config["IK"]["limit_acceleration"]
        self.d_min = self.config["IK"]["safety_distance"]
        self.output_xyz = self.config["IK"]["output_xyz"]
        self.attachment_site = "attachment_site"

        if self.output_xyz:
            self.traj_q = []
        self.traj = []      # Can be in joint space or Cartesian space

        self.load_model()

    def sample_zone(self):
        q_ranges = np.array(self.config["mpc"]["x0_range"])

        # Single zone → nothing to sample
        if not (q_ranges.ndim == 3 and q_ranges.shape[0] > 1):
            return None

        # ---- Check that zone_probability exists ----
        if "zone_probability" not in self.config["mpc"]:
            raise ValueError(
                f"zone_probability must be defined in the config when x0_range contains "
                f"{len(q_ranges)} zones."
            )

        probs = np.array(self.config["mpc"]["zone_probability"])

        # Validate length
        if len(probs) != len(q_ranges):
            raise ValueError(
                f"Length of zone_probability ({len(probs)}) does not match number of zones ({len(q_ranges)})"
            )

        # Check that probabilities sum to 1
        if not np.isclose(probs.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"zone_probability values must sum to 1. Currently sums to {probs.sum():.6f}"
            )

        zone_idx = np.random.choice(len(q_ranges), p=probs)
        print(f"[IK] Selected zone {zone_idx}")
        return int(zone_idx)
    
    def load_x0(self):
        # =========== Load x0 ===========
        self.x0_q = self.get_valid_q("x0_q", "x0_range")
        x0_v = self.randomise_vel()
        x0_q_save = np.hstack((self.x0_q, x0_v))                                                        # I need to include velocity sampling for this as well.
        self.config["mpc"]["x0_q"] = x0_q_save.tolist()                                                 # Save x0 in joint space to config for summary saving purpose
        self.config["mpc"]["x0"] = self.joint_to_xyz(self.x0_q, self.attachment_site).tolist()          # Save x0 in Cartesian space to config for summary saving purpose]

        if self.output_xyz:
            self.traj.append(self.joint_to_xyz(self.x0_q, self.attachment_site))                        # Record starting position
            self.traj_q.append(self.x0_q)
        else:
            self.traj.append(self.x0_q)                                                                 # Record starting position

        return x0_q_save
    
    def randomise_vel(self):
        xmax = np.array(self.config["mpc"]["xmax"])
        velocity_limit = self.config["IK"]["velocity_sampling_factor"] * xmax[xmax.shape[0]//2:]
        if self.config["mpc"]["x0_q_random"]:
            return np.random.uniform(low=-velocity_limit,
                                        high=velocity_limit)
        else:
            return np.array(self.config["mpc"]["x0_v"])
        
    def load_yref(self):
        # =========== Load x0 ===========
        self.yref_q = self.get_valid_q("yref_q", "yref_range")
        yref_q_save = np.hstack((self.yref_q, np.zeros((self.config["pin"]["nu"]))))                    # Velocity set as zero (For it to come to a stop)
        self.config["mpc"]["yref_q"] = yref_q_save.tolist()                                             # Save yref in joint space to config for summary saving purpose

        yref_save = self.joint_to_xyz(self.yref_q, self.attachment_site).tolist()                       # Convert to cartesian
        self.config["mpc"]["yref"] = yref_save                                                          # Save yref in Cartesian space to config for summary saving purpose
        yref = self.pad_yref(yref_save)                                                                 # Pad to the format of (x,y,z, q_dots ,x_goal,y_goal,z_goal)

        return yref
    
    def setup_tasks(self):
        # =========== Define configuration ===========
        self.configuration = pink.Configuration(model=self.robot.model, data=self.robot.data, q=self.x0_q, collision_model=self.robot.collision_model, collision_data=self.robot.collision_data)

        # =========== Define tasks ===========
        self.tasks = {
            "ee": FrameTask(
                self.attachment_site,
                position_cost=1.0, # Translation position
                orientation_cost=1e-3 # Pose of the end-effector
            ),
            "posture": PostureTask(cost=1e-2)} # Regularisation of angles
        
        # Initialize tasks from the current configuration
        for task in self.tasks.values():
            task.set_target_from_configuration(self.configuration)

        # Copy target format, set translation to x0
        self.T_target = self.tasks["ee"].transform_target_to_world.copy()

        # Set ee task target
        self.tasks["ee"].transform_target_to_world = self.T_target
        
        # Select QP solver
        self.solver = "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]

        if self.limit_accel:
            self.limits = (self.configuration.model.velocity_limit, self.accellimit)
        
        if self.visualize:
            self.start_viz()
            self.update_target_viz(self.T_target.np)  # currently only updating translation, if 6DOF then update
            self.update_robot_viz(self.configuration.q)
    
    def get_valid_q(self, q_name: str, q_range: str):
        q_ranges = np.array(self.config["mpc"][q_range])  # shape: (n_zones, 2, 3) or (2, 3)
        zone_idx = self.config["mpc"].get("zone_idx", None)

        # Select the active zone
        if q_ranges.ndim == 3 and q_ranges.shape[0] > 1:
            if zone_idx is None:
                raise ValueError(
                    f"zone_idx must be set for multi-zone {q_range}. "
                    f"Found {q_ranges.shape[0]} zones, but zone_idx is None."
                )
            q_range_sel = q_ranges[zone_idx]
        else:
            q_range_sel = q_ranges
            zone_idx = None
        
        attempt = 0
        while True:
            q = self.load_q(q_name)

            in_collision = self.collision_check(q)
            in_bbox = self.frame_within_bbox(q, self.attachment_site, q_range_sel)
            min_dist = self.distance_check(q)
            dist_ok = min_dist > self.d_min

            if (not in_collision) and in_bbox and dist_ok:
                print(
                    f"Loaded valid {q_name} configuration in {attempt+1} attempts\n"
                    f"  q:        {q}\n"
                    f"  q_range:  min={q_range_sel[0]}, max={q_range_sel[1]}\n"
                    f"  min_dist: {min_dist:.4f} (d_min={self.d_min})\n"
                )
                return q

            # If randomness disabled → fail immediately with details
            if not self.config["mpc"][f"{q_name}_random"]:
                reasons = []
                if in_collision:
                    reasons.append("collision_check failed (q is in collision)")
                if not in_bbox:
                    reasons.append("frame_within_bbox failed")
                if not dist_ok:
                    reasons.append(
                        f"distance_check failed (min_dist={min_dist:.4f} ≤ d_min={self.d_min})"
                    )

                reason_str = "\n  - ".join(reasons)

                raise ValueError(
                    f"Invalid {q_name} configuration:\n"
                    f"  q: {q}\n"
                    f"  Failed checks:\n"
                    f"  - {reason_str}"
                )
            
            attempt += 1
    
    def load_q(self, q_name: str):
        if self.config["mpc"][f"{q_name}_random"]:
            # Random initial configuration sampled from joint limits
            joint_sampling_factor = self.config["IK"]["joint_sampling_factor"]
            return np.random.uniform(low=self.robot.model.lowerPositionLimit * joint_sampling_factor,
                                        high=self.robot.model.upperPositionLimit * joint_sampling_factor)
        else:
            q = np.array(self.config["mpc"][f"{q_name}"])

            if q.shape[0] != self.robot.model.nq:
                q = q[:self.robot.model.nq]
                print(f"Warning: Loaded {q_name} has length {q.shape[0]}, but model has {self.robot.model.nq} joints. Truncating to first {self.robot.model.nq} values.")
            return q

    def frame_within_bbox(self, q, frame_name, bbox):
        """
        Check if a frame is within a world-axis-aligned bounding box.
        """
        pos = self.joint_to_xyz(q, frame_name)

        if np.all(pos >= bbox[0]) and np.all(pos <= bbox[1]):
            return True
        else:
            return False
    
    def joint_to_xyz(self, q, frame_name):
        """
        Convert joint positions to end-effector XYZ using Pinocchio FK.

        Args:
            q (np.array): Joint positions, shape (nq,)

        Returns:
            np.array: XYZ position of the 'self.attachment_site', shape (3,)
        """
        # 1. Forward kinematics
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        # 2. Get the frame ID of the attachment site
        frame_id = self.robot.model.getFrameId(frame_name)

        # 3. Return translation
        return self.robot.data.oMf[frame_id].translation.copy()
    
    def collision_check(self, q):
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateGeometryPlacements(
            self.robot.model,
            self.robot.data,
            self.robot.collision_model,
            self.robot.collision_data,
            q
        )

        return pin.computeCollisions(
            self.robot.collision_model,
            self.robot.collision_data,
            stop_at_first_collision=True
        )
    
    def distance_check(self, q):
        if self.collision_config["collision_avoidance_obstacle"] or self.collision_config["collision_avoidance_ground"]:
            pin.forwardKinematics(self.robot.model, self.robot.data, q)
            pin.updateGeometryPlacements(
                self.robot.model,
                self.robot.data,
                self.robot.collision_model,
                self.robot.collision_data,
                q
            )
            idx = pin.computeDistances(self.robot.collision_model,self.robot.collision_data)

            return self.robot.collision_data.distanceResults[idx].min_distance
        else:
            return 1e6

    def load_model(self):
        # Load model from XML
        base_dir = "models_xml"
        model_path = self.config["model"]["model_path"]
        filename = os.path.join(base_dir, model_path)

        try:
            # Load from MJCF
            self.robot = RobotWrapper.BuildFromMJCF(filename=filename)
            # Set velocitylimit
            xmax = np.array(self.config["mpc"]["xmax"])
            self.robot.model.velocityLimit = xmax[xmax.shape[0]//2:]
            if self.limit_accel:
                self.accellimit = AccelerationLimit(self.robot.model, np.array(self.config["IK"]["accelerationlimit"]))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{filename}' does not exist. Check your models_xml folder.")
        
        if self.collision_config["collision_avoidance_obstacle"]:
            self.add_obstacle_capsules()
        if self.collision_config["collision_avoidance_ground"]:
            self.add_ground_plane()

        if self.collision_config["collision_avoidance_obstacle"] or self.collision_config["collision_avoidance_ground"]:
            self.robot.collision_data = process_collision_pairs(
                self.robot.model,
                self.robot.collision_model,
                self.config["collision"]["srdf_path"]
                )
            
            # Collision barriers between self and obstacles
            collision_barrier = SelfCollisionBarrier(
                n_collision_pairs=len(self.robot.collision_model.collisionPairs),
                gain=1.0,
                safe_displacement_gain=1.0,
                d_min=self.d_min, # safety distance for collision
            )

            self.barriers = [collision_barrier]
        else:
            self.barriers = None

    def add_ground_plane(self):

        """
        Adds an infinite ground plane to the robot collision model.

        ground_plane format in config:
            [a, b, c, d]  -> ax + by + cz + d = 0
        """

        plane_params = self.collision_config["ground_plane"]
        a, b, c, d = plane_params

        normal = np.array([a, b, c], dtype=float)
        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            raise ValueError("Ground plane normal vector cannot be zero.")

        normal /= norm
        d /= norm  # normalize plane equation

        # Create FCL plane
        shape_plane = fcl.Plane(normal, d)

        # Identity placement (plane already defined in world frame)
        placement = pin.SE3.Identity()

        # Create geometry object attached to universe (joint 0)
        geom = pin.GeometryObject(
            "ground",
            0,  # universe joint
            placement,
            shape_plane
        )

        geom.meshColor = np.array([0.5, 0.5, 0.5, 0.3])  # semi-transparent gray

        # Add ONLY to collision model
        self.robot.collision_model.addGeometryObject(geom)
        self.robot.visual_model.addGeometryObject(geom)

    def add_obstacle_capsules(self):
        """
        Add obstacles as capsule geometries to robot's collision and visual models.
        
        obstacles_dict: 
            e.g. {"obs1": {"from": np.array([x0,y0,z0]), "to": np.array([x1,y1,z1]), "radius": r}}
        """
        obstacles_dict = self.collision_config["obstacles"]

        for name, obs in obstacles_dict.items():
            p0 = np.array(obs["from"])
            p1 = np.array(obs["to"])
            radius = float(obs["radius"])

            # Compute capsule length and placement
            vec = p1 - p0
            length = np.linalg.norm(vec)
            if length < 1e-8:
                length = 1e-6  # avoid zero-length capsule

            # Compute the placement SE3
            midpoint = (p0 + p1) / 2
            # Default capsule axis in Pinocchio/fcl is along z, so we need rotation
            z_axis = vec / length
            # Arbitrary choice of x_axis
            x_axis = np.array([1.0, 0.0, 0.0])
            if np.allclose(z_axis, x_axis):
                x_axis = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            R = np.column_stack([x_axis, y_axis, z_axis])
            placement = pin.SE3(R, midpoint)

            # Create FCL capsule
            shape_pole = fcl.Capsule(radius, length)

            # Create GeometryObject
            geom = pin.GeometryObject(name, 0, placement, shape_pole)
            geom.meshColor = np.array([1.0, 0.0, 0.0, 1.0])  # red for visualization

            # Add to models
            self.robot.collision_model.addGeometryObject(geom)
            self.robot.visual_model.addGeometryObject(geom)

    def start_viz(self):
        if self.visualize:
            # Launch Viewer
            self.viz = start_meshcat_visualizer(self.robot)
            self.viewer = self.viz.viewer

            meshcat_shapes.frame(self.viewer["target_frame"], opacity=0.3) # Target frame
            meshcat_shapes.frame(self.viewer["ee_frame"], opacity=1.0) # End-effector frame

    def update_target_viz(self, T_target):
        # Show goal position in Meshcat
        self.viewer["target_frame"].set_transform(T_target) # currently only updating translation, if 6DOF then update
    
    def update_ee_viz(self, T_ee):
        # Show goal position in Meshcat
        self.viewer["ee_frame"].set_transform(T_ee) # currently only updating translation, if 6DOF then update

    def update_robot_viz(self, q):
        self.viz.display(q)

    def pad_yref(self, yref=None):
        """
        Pad yref with zero velocity and control terms.

        Supports:
        - yref shape (ny,)       -> returns (ny + nv + nu,)
        - yref shape (T, ny)     -> returns (T, ny + nv + nu)
        """

        if yref is None:
            yref = self.traj

        yref = np.asarray(yref)

        # Track whether input was a single vector
        is_vector = (yref.ndim == 1)

        # Normalize to 2D: (T, ny)
        if is_vector:
            yref = yref.reshape(1, -1)

        T, ny = yref.shape

        nv = self.config["pin"]["nu"]   # your velocity dim
        nu = self.config["pin"]["nu"]   # your control dim

        yref_vel = np.zeros((T, nv))
        yref_u   = np.zeros((T, nu))

        yref_padded = np.hstack([yref, yref_vel, yref_u])

        # Return to original dimensionality
        if is_vector:
            return yref_padded.squeeze(axis=0)  # (7,)
        else:
            return yref_padded                  # (T, 7)

    def IK_to_XYZ(self, yref):
        # Setup task
        self.setup_tasks()

        # Now set actual end goal
        self.T_target.translation = yref

        if self.visualize:
            self.update_target_viz(self.T_target.np) # currently only updating translation, if 6DOF then update
            self.update_robot_viz(self.x0_q)

        # IK to goal
        self.IK_loop(record = True)
        
        self.traj = np.array(self.traj) # Convert to np.array (only positions without velocity)

        if self.output_xyz:
            self.traj_q = np.array(self.traj_q)
            sys.stdout.write("Joint trajectory of length: " + str(self.traj_q.shape) + "\n")

        sys.stdout.write("Trajectory of length: " + str(self.traj.shape) + "\n")

        yref_full = self.pad_yref()
        
        return yref_full, self.config # maybe return it as a dict of yref instead.
    
    def IK_loop(self, record: bool):
        iterations = 0
        vel_below_thresh = 0
        error = np.linalg.norm(self.T_target.translation - self.configuration.get_transform_frame_to_world(self.tasks["ee"].frame).translation)

        while error > self.stop_thres:
            # Integrate motion
            if self.limit_accel:
                velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver=self.solver, barriers=self.barriers, limits=self.limits, safety_break=False)
                self.configuration.integrate_inplace(velocity, self.dt)
                self.accellimit.set_last_integration(velocity, self.dt)
            else:
                velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver=self.solver, barriers=self.barriers, safety_break=False)
                self.configuration.integrate_inplace(velocity, self.dt)

            error = np.linalg.norm(self.T_target.translation - self.configuration.get_transform_frame_to_world(self.tasks["ee"].frame).translation)
            
            sys.stdout.write("\rvelocity: " + str(velocity) + " | error: {:.3e}".format(error))
            sys.stdout.flush()

            if self.visualize:
                # Update ee frame visual
                self.update_ee_viz(self.configuration.get_transform_frame_to_world(self.tasks["ee"].frame).np)
                # Display robot config
                self.update_robot_viz(self.configuration.q)

            if record:
                if self.output_xyz:
                    self.traj.append(self.joint_to_xyz(self.configuration.q, self.attachment_site))  # Record starting position
                    self.traj_q.append(self.configuration.q)
                else:
                    self.traj.append(self.configuration.q)                       # Record starting position


            iterations += 1
            if iterations > self.max_iterations:
                raise RuntimeError(f"IK did not converge within {self.max_iterations} steps. Current error: {error:.3e}")
            
            if np.all(np.abs(velocity) < self.velocity_thres):
                vel_below_thresh += 1
            else:
                vel_below_thresh = 0  # reset if condition breaks

            if vel_below_thresh >= 10:
                raise RuntimeError("Velocity below threshold for 10 consecutive iterations")

        sys.stdout.write("\nPosition reached\n")