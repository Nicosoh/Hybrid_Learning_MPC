from pyexpat import model
import mujoco
import numpy as np
from tqdm import tqdm
from utils import *
import mujoco.viewer
import time

class MuJoCoSimulator:
    def __init__(self, config, yref, controller, gt_controller=None):
        self.config = config
        self.yref = yref
        self.collision_config = config["collision"]
        self.controller = controller
        self.gt_controller = gt_controller

        # Extract config params
        self.N_horizon = self.config["mpc"]["N_horizon"]
        self.mpc_timestep = self.config["mpc"]["mpc_timestep"]
        self.IK_required = self.config["IK"]["IK_required"]
        self.point_reference = self.config["IK"]["point_reference"]

        self.model, self.data = load_scene_from_xml(self.config)                     # Create MuJoCo simulator object with loaded model
        # self.sanity_check()                                                 # Sanity check between model, controller, yref, x0
        self.model.opt.timestep = self.config["mujoco"]["sim_timestep"]     # Set simulation timestep

        if self.config["mujoco"]["render"]: # Set render options and create renderer
            self.scene_option = init_scene_options()
            height, width = self.config["mujoco"]["resolution"]
            self.renderer = mujoco.Renderer(self.model, height, width)
        
        # Reset model
        mujoco.mj_resetData(self.model, self.data)

    def sanity_check(self):
        # --- Sanity checks ---
        nq, nv, nu = self.model.nq, self.model.nv, self.model.nu
        nx_model = nq + nv
        nx_ctrl = self.controller.nx
        nu_ctrl = self.controller.nu
        x0 = self.config["mpc"]["x0"]
        solve_ocp = self.config["mpc"]["solve_ocp"]
        full_traj = self.config["mpc"]["full_traj"]

        # Check that controller state dim matches model state dim
        if nx_ctrl != nx_model:
            raise ValueError(
                f"State dimension mismatch: controller.nx={nx_ctrl}, "
                f"but model.nq+model.nv={nx_model}"
            )

        # Check that x0 length matches controller and model states
        if len(x0) != nx_ctrl:
            raise ValueError(
                f"x0 length mismatch: len(x0)={len(x0)}, expected {nx_ctrl} "
                f"(must match controller.nx and model.nq+nv)"
            )

        # Check that yref dimension matches controller outputs
        # yref can be shape (N, ny) or (ny,)
        if self.yref.ndim == 1:
            ny_yref = self.yref.shape[0]
        elif self.yref.ndim == 2:
            ny_yref = self.yref.shape[1]
        else:
            raise ValueError(f"Unexpected yref shape: {self.yref.shape}")

        if hasattr(self.controller, "ny"):
            if ny_yref != nx_ctrl+nu_ctrl:
                raise ValueError(
                    f"yref dimension mismatch: yref has {ny_yref}, "
                    f"controller expects nx + nu ={nx_ctrl+nu_ctrl}"
                )

        # Check that input dimensions match
        if nu_ctrl != nu:
            raise ValueError(
                f"Input dimension mismatch: controller.nu={nu_ctrl}, model.nu={nu}"
            )
        
        if solve_ocp and not full_traj:
            raise ValueError(
                "Input mismatch: solve_ocp=True requires full_traj=True."
            )

        print("Sanity checks passed: model, controller, x0, and yref are consistent.")

    def run(self):
        # Extract config for easier reading
        mpc_config = self.config["mpc"]
        mujoco_config = self.config["mujoco"]

        # Extract parameters from config for simulator
        sim_duration = mujoco_config["sim_duration"]
        verbose = mujoco_config["verbose"]
        render = mujoco_config["render"]
        sim_framerate = mujoco_config["sim_framerate"]
        sim_timestep = mujoco_config["sim_timestep"]

        # Extract parameters from config for MPC
        self.mpc_timestep = mpc_config["mpc_timestep"]
        early_termination_cost = mpc_config["early_termination_cost"]
        termination_thresh_cost = np.array(mpc_config["termination_cost"])
        early_termination_state = mpc_config["early_termination_state"]
        termination_thresh_state = np.array(mpc_config["termination_state"])
        solve_ocp = mpc_config["solve_ocp"]
        output_xyz = self.config["IK"]["output_xyz"]

        # Extract parameters for IK
        IK_required = self.config["IK"]["IK_required"]

        if IK_required:
            x0 = np.array(mpc_config["x0_q"])
        else:
            x0 = np.array(mpc_config["x0"])

        # Init variables for counting and so on...
        self.last_u = np.zeros(self.model.nu)
        self.next_mpc_time = 0.0
        
        # Empty lists for recording
        self.frames = []
        self.logs = {
            "time": [],
            "qpos": [],
            "qvel": [],
            "yref": [],
            "yref_full": self.yref,                 # Full reference trajectory could be in cartesian or joint space
            "yref_xyz": self.config["mpc"]["yref"], # Final end-effector position in XYZ
            "u_applied": [],
            "stage_cost": [],
            "terminal_cost": [],
            "total_cost": [],
            "qpos_traj": [],
            "qvel_traj": [],
            "u_traj": [],
            "sq_dist": [],
            "GT_cost": [],
            "GT_qpos_traj": [],
            "GT_qvel_traj": [],
            "xyz_traj": [],
        }
        
        if self.config["collision"]["collision_avoidance_obstacle"]:
            self.logs["obstacles"] = self.collision_config["obstacles"]
            self.obstacles = self.collision_config["obstacles"] 

        if self.IK_required:
            self.logs["yref_q"] = self.config["mpc"]["yref_q"][:self.model.nq]

        if output_xyz:
            self.logs["xyzpos"] = []                                                                # Empty list for End-effector positions in XYZ
            self.logs["GT_xyz_traj"] = []
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,"attachment_site")          # End-effector site id

        # Set initial position and velocity from x0
        self.data.qpos[:] = x0[:self.model.nq]
        self.data.qvel[:] = x0[self.model.nq:]

        mujoco.mj_forward(self.model, self.data) # Step forward to compute derived quantities
        print(f"Model has {self.model.nq} DoFs (qpos), {self.model.nv} velocities (qvel), and {self.model.nu} actuators.")
        print("Initial qpos:", self.data.qpos)
        print("Initial qvel:", self.data.qvel)

        # Calculate number of steps (for tqdm total)
        steps = int(sim_duration / sim_timestep)
        pbar = tqdm(total=steps, desc="Simulating")

        # Set update_initial_guess = True for the first time
        self.update_initial_guess = True

        # Main simulation loop
        while self.data.time < sim_duration:
            # Step simulation
            stage_cost, terminal_cost, total_cost ,qpos_traj, qvel_traj, u_traj, sq_dist, xyz_traj, GT_qpos_traj, GT_qvel_traj, GT_cost, GT_xyz_traj = self.step_sim()

            # Only log data if MPC actually produced new control
            if total_cost is not None and qpos_traj is not None and qvel_traj is not None and u_traj is not None:
                self.logs["time"].append(np.copy(self.data.time))
                self.logs["qpos"].append(np.copy(self.data.qpos))
                self.logs["qvel"].append(np.copy(self.data.qvel))
                self.logs["u_applied"].append(np.copy(self.data.ctrl))
                self.logs["stage_cost"].append(stage_cost)
                self.logs["terminal_cost"].append(terminal_cost)
                self.logs["total_cost"].append(total_cost)
                self.logs["qpos_traj"].append(qpos_traj)
                self.logs["qvel_traj"].append(qvel_traj)
                self.logs["u_traj"].append(u_traj)
                self.logs["sq_dist"].append(sq_dist)
                self.logs["GT_cost"].append(GT_cost)
                self.logs["GT_qpos_traj"].append(GT_qpos_traj)
                self.logs["GT_qvel_traj"].append(GT_qvel_traj)
                self.logs["xyz_traj"].append(xyz_traj)

                if output_xyz:
                    self.logs["xyzpos"].append(np.copy(self.data.site_xpos[site_id]))
                    self.logs["GT_xyz_traj"].append(GT_xyz_traj)

                if verbose:
                    print(
                        f"t = {self.data.time:.3f}s | "
                        f"qpos = {np.round(self.data.qpos, 3)} | "
                        f"qvel = {np.round(self.data.qvel, 3)} | "
                        f"ctrl = {np.round(self.data.ctrl, 3)} | "
                        f"cost = {np.round(total_cost, 5)}"
                    )
            
            # Render if enabled
            if render and len(self.frames) < self.data.time * sim_framerate:
                self.renderer.update_scene(self.data, scene_option=self.scene_option, camera=0)

                if output_xyz:
                    add_visual_sphere(self.renderer.scene, self.config["mpc"]["yref"], 0.03, rgba=(0.0, 1.0, 0.0, 0.2))  # For the end goal (green)
                    add_visual_sphere(self.renderer.scene, self.config["mpc"]["x0"], 0.03, rgba=(1.0, 1.0, 0.0, 0.2))  # For the start goal (yellow)
                if self.collision_config["collision_avoidance_obstacle"]:
                    # Add obstacle capsules to the scene (for now ignoring that over time it can shift aka static obstacles)
                    obstacles = self.collision_config["obstacles"]

                    for obs_name, obs in obstacles.items():
                        add_visual_capsule(self.renderer.scene, p1=obs["from"], p2=obs["to"], radius=obs["radius"], rgba=(0.8, 0.1, 0.1, 1))

                pixels = self.renderer.render()
                self.frames.append(pixels)
            
            if solve_ocp:
                pbar.write("Exiting simulation after one step.")
                break

            # Check for early termination condition
            if total_cost is not None:
                if total_cost < termination_thresh_cost and early_termination_cost:
                    pbar.write(f"Terminating early at t = {self.data.time:.2f}s with cost = {total_cost:.5f}")
                    break
                elif early_termination_state: #need to rethink about this part....
                    state_err = np.concatenate([self.data.qpos, self.data.qvel]) - self.yref[-1][:self.model.nq+self.model.nv]
                    if np.linalg.norm(state_err) < termination_thresh_state:
                        pbar.write(f"Terminating early at t = {self.data.time:.2f}s with state error = {np.linalg.norm(state_err):.5f}")
                        break

            # Update progress bar
            pbar.update(1)

        pbar.close()

        # Convert logs to arrays
        for key in self.logs.keys():
            self.logs[key] = np.array(self.logs[key])

    def step_sim(self):
        x = np.concatenate([self.data.qpos, self.data.qvel])
        total_cost = None
        stage_cost = None
        terminal_cost = None
        qpos_traj = None
        qvel_traj = None
        u_traj = None
        sq_dist = None
        GT_cost = None
        xyz_traj = None
        GT_qpos_traj = None
        GT_qvel_traj = None
        GT_xyz_traj = None

        # Only update MPC if needed
        if self.data.time >= self.next_mpc_time:
            try:
                if not self.IK_required or self.point_reference:
                    yref_now = self.yref
                    self.logs["yref"].append(yref_now)  # Keep track of the reference
                else:
                    yref_now = get_reference_for_horizon(self.yref, self.data.time, self.N_horizon, self.mpc_timestep)
                    self.logs["yref"].append(yref_now)

                # Run MPC to compute control input, cost, and trajectory
                self.last_u, stage_cost, terminal_cost, total_cost, qpos_traj, qvel_traj, u_traj, sq_dist, xyz_traj = self.controller(x, yref_now, self.config["mpc"]["full_traj"])

                if self.config["VI"]["ground_truth_controller"]:
                    GT_x = np.concatenate([qpos_traj[-1], qvel_traj[-1]])

                    if self.update_initial_guess:
                        self.gt_controller.update_initial_guess(GT_x)
                        self.update_initial_guess = False

                    _, GT_cost, _, _, GT_qpos_traj, GT_qvel_traj, _, _, GT_xyz_traj = self.gt_controller(GT_x, yref_now, self.config["mpc"]["full_traj"])

                    if np.isnan(GT_cost):
                        self.update_initial_guess = True

                # += to next mpc time step
                self.next_mpc_time += self.mpc_timestep

            except RuntimeError as e:
                raise RuntimeError(
                    f"MPC solver failed at t={self.data.time:.3f}s"
                ) from e

        # Apply last computed control
        self.data.ctrl[:] = self.last_u

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        return stage_cost, terminal_cost, total_cost, qpos_traj, qvel_traj, u_traj, sq_dist, xyz_traj, GT_qpos_traj, GT_qvel_traj, GT_cost, GT_xyz_traj
    
def add_visual_capsule(scene, p1, p2, radius, rgba):
    """Adds a visual-only capsule to an mjvScene (no physics)."""
    if scene.ngeom >= scene.maxgeom:
        return  # can't add more

    idx = scene.ngeom
    scene.ngeom += 1

    rgba = np.asarray(rgba, dtype=np.float32)

    mujoco.mjv_initGeom(
        scene.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),             # size (not used for capsules)
        np.zeros(3),             # position (not used for capsules)
        np.zeros(9),             # rotation (not used for capsules)
        rgba                     # color
    )

    # Set capsule endpoints and radius
    mujoco.mjv_connector(
        scene.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.array(p1, dtype=np.float32),
        np.array(p2, dtype=np.float32),
    )

def add_visual_sphere(scene, position, radius, rgba):
    """Adds a visual-only sphere to an mjvScene (no physics)."""
    if scene.ngeom >= scene.maxgeom:
        print("error adding sphere")
        return  # can't add more

    idx = scene.ngeom
    scene.ngeom += 1

    rgba = np.asarray(rgba, dtype=np.float32)
    position = np.asarray(position, dtype=np.float32)

    mujoco.mjv_initGeom(
        scene.geoms[idx],
        type = mujoco.mjtGeom.mjGEOM_SPHERE,
        size = np.array([radius, 0.0, 0.0], dtype=np.float32),  # size: radius in x
        pos = position,                                    # position
        mat=np.eye(3).flatten(),
        rgba = rgba                                         # color
    )

def get_reference_for_horizon(traj, t, N, mpc_dt):
    """
    Build Acados-compatible reference arrays over the horizon from full trajectory.

    Args:
        traj: ndarray shaped (T, nx)
        t: current continuous time [s]
        N: horizon length
        mpc_dt: MPC step size

    Returns:
        yref_stage: array of shape (N, nx)
        yref_terminal: array shape (nx,)
    """

    T, nx = traj.shape

    # convert continuous time to discrete index
    start_idx = int(np.round(t / mpc_dt))

    # indices for 0..N, clipped inside bounds
    idxs = start_idx + np.arange(N + 1)
    idxs = np.clip(idxs, 0, T - 1)

    # extract state references (N+1 of them)
    xrefs = traj[idxs]

    # stage references (state only)
    yref_stage = xrefs[:N]     # shape (N, nx)

    # terminal reference
    yref_terminal = xrefs[-1]  # shape (nx,)

    return {"stage": yref_stage, "terminal": yref_terminal}

class MujocoReplay:
    def __init__(self, model_config, replay_config, logs_dict):
        self.model_config = model_config
        self.replay_config = replay_config
        self.collision_config = model_config["collision"]

        self.model, self.data = load_scene_from_xml(self.model_config)                  # Create MuJoCo simulator object with loaded model
        self.model.opt.timestep = self.model_config["mpc"]["mpc_timestep"]     # Set simulation timestep

        self.qpos = logs_dict["qpos"]
        self.qvel = logs_dict["qvel"]
        self.xyz_traj = logs_dict["xyz_traj"]

        self.nframes = len(self.qpos)
        self.frame = 0
        self.playing = True
        self.speed = replay_config["playback_speed"]
        self.loop = replay_config["loop"]
        self.show_GT_controller = replay_config["show_GT_controller"]
        self.N_horizon = model_config["mpc"]["N_horizon"]
        self._last_time = None
        self.render_fps = self.replay_config["render_fps"]
        self._render_dt = 1.0 / self.render_fps
        self._accumulator = 0.0
        self.output_xyz = model_config["IK"]["output_xyz"]
        self.ground_truth_controller = model_config["VI"]["ground_truth_controller"]

        if self.ground_truth_controller:
            self.GT_xyz_traj = logs_dict["GT_xyz_traj"]

        self.KEY_SPACE = 32
        self.KEY_LEFT  = 263
        self.KEY_RIGHT = 262
        self.KEY_UP    = 265
        self.KEY_DOWN  = 264
    # ---------- input ----------

    def key_callback(self, keycode):
        if keycode == self.KEY_SPACE:
            self.playing = not self.playing
            print(f"Playing: {self.playing}")

        elif keycode == self.KEY_RIGHT:
            self._accumulator = 0.0
            self.frame += 1
            if self.loop:
                self.frame %= self.nframes
            else:
                self.frame = min(self.frame, self.nframes - 1)
            self.playing = False
            print(f"Frame: {self.frame}")

        elif keycode == self.KEY_LEFT:
            self._accumulator = 0.0
            self.frame -= 1
            if self.loop:
                self.frame %= self.nframes
            else:
                self.frame = max(self.frame, 0)
            self.playing = False
            print(f"Frame: {self.frame}")

        elif keycode == self.KEY_UP:
            self.speed *= 2.0
            print(f"Speed: {self.speed:.2f}x")

        elif keycode == self.KEY_DOWN:
            self.speed *= 0.5
            print(f"Speed: {self.speed:.2f}x")

    # ---------- playback ----------
    def advance(self, elapsed):
        if not self.playing:
            return

        self._accumulator += self.speed * elapsed

        while self._accumulator >= self.model.opt.timestep:
            self.frame += 1
            self._accumulator -= self.model.opt.timestep

            if self.loop:
                self.frame %= self.nframes
            else:
                if self.frame >= self.nframes - 1:
                    self.frame = self.nframes - 1
                    self.playing = False
                    break

    def apply_state(self):
        self.data.qpos[:] = self.qpos[self.frame]

        if self.qvel is not None:
            self.data.qvel[:] = self.qvel[self.frame]
        else:
            self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

    def _draw_path(self, viewer, horizon, rgba):
        """Refactored original path drawing logic."""
        if len(horizon) <= 50:
            indices = range(len(horizon))
        else:
            indices = np.linspace(0, len(horizon) - 1, 50).astype(int)

        for idx in indices:
            add_visual_sphere(viewer.user_scn, horizon[idx], radius=0.01, rgba=rgba)
            
    def viz_horizon(self, viewer):
        """Draws the predicted MPC horizon (Blue)."""
        self._draw_path(viewer, self.xyz_traj[self.frame], rgba=(0.0, 0.5, 1.0, 0.3))
    
    def viz_GT_horizon(self, viewer):
        """Update for the original MujocoReplay class."""
        if self.GT_xyz_traj is not None:
            self._draw_path(viewer, self.GT_xyz_traj[self.frame], rgba=(0.5, 0.0, 1.0, 0.3))
    
    # ---------- main loop ----------
    def run(self):
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
        ) as viewer:

            self._last_time = time.time()

            while viewer.is_running():
                loop_start = time.time()

                elapsed = loop_start - self._last_time
                self._last_time = loop_start

                viewer.user_scn.ngeom = 0
                
                self.advance(elapsed)
                self.apply_state()
                self.viz_horizon(viewer)

                if self.ground_truth_controller and self.show_GT_controller:
                    self.viz_GT_horizon(viewer)

                if self.output_xyz:
                    add_visual_sphere(viewer.user_scn, self.model_config["mpc"]["yref"], 0.03, rgba=(0.0, 1.0, 0.0, 0.2))  # For the end goal (green)
                    add_visual_sphere(viewer.user_scn, self.model_config["mpc"]["x0"], 0.03, rgba=(1.0, 1.0, 0.0, 0.2))  # For the start goal (yellow)
                
                if self.collision_config["collision_avoidance_obstacle"]:
                    # Add obstacle capsules to the scene (for now ignoring that over time it can shift aka static obstacles)
                    obstacles = self.collision_config["obstacles"]

                    for obs_name, obs in obstacles.items():
                        add_visual_capsule(viewer.user_scn, p1=obs["from"], p2=obs["to"], radius=obs["radius"], rgba=(0.8, 0.1, 0.1, 1))

                viewer.sync()

                # ---- render rate limiting ----
                sleep_time = self._render_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

class MujocoComparisonReplay(MujocoReplay):
    def __init__(self, model_config, replay_config, combined_logs):
        # Initialize the base class with Run A data
        super().__init__(model_config, replay_config, combined_logs["run_a"])

        # Replay config options
        self.save_video = replay_config["save_video"]

        # Store separate logs
        self.logs_a = combined_logs["run_a"]
        self.logs_b = combined_logs["run_b"]

        # Create the second data object for the Ghost
        self.data2 = mujoco.MjData(self.model)
        
        # Sync frames based on the shorter log
        self.nframes = min(len(self.logs_a["qpos"]), len(self.logs_b["qpos"]))

        # Ghost Visual Options
        self.vopt2 = mujoco.MjvOption()
        self.vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self.pert = mujoco.MjvPerturb()
        self.catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

        # Recording options
        self.video_path = "comparison_replay.mp4"
        self.frames = []

    def apply_state(self):
        """Updates both data objects for the current frame."""
        # Update Physical Robot (Run A)
        self.data.qpos[:] = self.logs_a["qpos"][self.frame]
        if "qvel" in self.logs_a:
            self.data.qvel[:] = self.logs_a["qvel"][self.frame]
        
        # Update Ghost Robot (Run B)
        self.data2.qpos[:] = self.logs_b["qpos"][self.frame]
        if "qvel" in self.logs_b:
            self.data2.qvel[:] = self.logs_b["qvel"][self.frame]
        
        # Step kinematics for both
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.data2)

    def run(self):
        """Main loop with off-screen rendering for video capture."""
        if self.save_video:
            video_renderer = mujoco.Renderer(self.model, height=480, width=640)

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback,
        ) as viewer:

            self._last_time = time.time()

            while viewer.is_running():
                loop_start = time.time()
                elapsed = loop_start - self._last_time if self._last_time else 0
                self._last_time = loop_start

                # Reset the custom geometry buffers
                viewer.user_scn.ngeom = 0
                
                self.advance(elapsed)
                self.apply_state()

                # Inner helper to populate any scene with our custom markers
                def add_visuals(target_scn):
                    # 1. Horizons (Run A: Blue, Run B: Orange)
                    if self.output_xyz:
                        self._draw_path(target_scn, self.logs_a["xyz_traj"][self.frame], rgba=(0.0, 0.5, 1.0, 0.3))
                        self._draw_path(target_scn, self.logs_b["xyz_traj"][self.frame], rgba=(1.0, 0.5, 0.0, 0.3))
                    
                    # 2. Ghost Robot (Run B)
                    mujoco.mjv_addGeoms(self.model, self.data2, self.vopt2, self.pert, self.catmask, target_scn)

                    # 3. Targets/Start
                    if self.output_xyz:
                        add_visual_sphere(target_scn, self.model_config["mpc"]["yref"], 0.03, rgba=(0.0, 1.0, 0.0, 0.2))
                        add_visual_sphere(target_scn, self.model_config["mpc"]["x0"], 0.03, rgba=(1.0, 1.0, 0.0, 0.2))
                    
                    # 4. GT Horizon if enabled
                    if self.ground_truth_controller and self.show_GT_controller:
                        if "GT_xyz_traj" in self.logs_a:
                            self._draw_path(target_scn, self.logs_a["GT_xyz_traj"][self.frame], rgba=(0.5, 0.0, 1.0, 0.3))
                        if "GT_xyz_traj" in self.logs_b:
                            self._draw_path(target_scn, self.logs_b["GT_xyz_traj"][self.frame], rgba=(1.0, 0.0, 0.5, 0.3))

                    # 5. Obstacles
                    if self.collision_config["collision_avoidance_obstacle"]:
                        for obs in self.collision_config["obstacles"].values():
                            add_visual_capsule(target_scn, p1=obs["from"], p2=obs["to"], radius=obs["radius"], rgba=(0.8, 0.1, 0.1, 1))

                # Populate the viewer for real-time feedback
                add_visuals(viewer.user_scn)

                # Update the video renderer's scene and capture the frame
                if self.save_video:
                    video_renderer.update_scene(self.data, camera=0) # Use MjvOption
                    add_visuals(video_renderer.scene)
                    
                    self.frames.append(video_renderer.render())

                viewer.sync()

                # Render Rate Limiting
                sleep_time = self._render_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Save video after window is closed
        if len(self.frames) > 0:
            print(f"Saving video to {self.video_path}...")
            media.write_video(self.video_path, self.frames, fps=self.render_fps)
            print("Video saved.")

    def viz_horizon(self, viewer):
        """Redirects to the refactored path drawer for both runs."""
        # This is called by advance() in the base class, but we handle 
        # actual scn population inside the run() loop for video sync.
        pass

    def viz_GT_horizon(self, viewer):
        """Redirects to the refactored path drawer for both runs."""
        # This is called by advance() in the base class, but we handle 
        # actual scn population inside the run() loop for video sync.
        pass

    def _draw_path(self, scn_or_viewer, horizon, rgba):
        """Standardized helper to draw a path into a scene or viewer."""
        # Determine if we are acting on the viewer's user_scn or a raw MjvScene
        target_scn = scn_or_viewer.user_scn if hasattr(scn_or_viewer, 'user_scn') else scn_or_viewer
        
        if len(horizon) <= 50:
            indices = range(len(horizon))
        else:
            indices = np.linspace(0, len(horizon) - 1, 50).astype(int)

        for idx in indices:
            add_visual_sphere(target_scn, horizon[idx], radius=0.01, rgba=rgba)