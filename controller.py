from colorlog import warning
import scipy.linalg

import numpy as np
import casadi as ca

from utils import *
from casadi import vertcat, SX
from pin_exporter import export_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver
from neural_network.torch_exporter import export_torch_model

CONTROLLER_REGISTRY = {}

def register_controller(cls):
    CONTROLLER_REGISTRY[cls.__name__] = cls
    return cls

@register_controller
class BaseMPCController:
    '''
    Class for a basic MPC Controller, also a base class for other MPC controllers.
    '''
    def __init__(self, config, worker_id=0):
        # Extract parameters from config
        self.config = config
        self.worker_id = worker_id
        self.use_RTI = config["mpc"]["use_RTI"]
        self.x0 = np.array(config["mpc"]["x0"])
        self.mpc_timestep = config["mpc"]["mpc_timestep"]
        self.terminal_cost = config["mpc"]["terminal_cost"]
        self.IK_required = config["IK"]["IK_required"]

        # Reassign x0 if IK is used
        if self.IK_required:
            self.x0 = np.array(config["mpc"]["x0_q"])

        # Init parameters if needed
        self.p_obs = None
        self.p_NN = None
        self.p = None

        # Setup MPC solver
        self.setup(config)

        # Warm start
        for _ in range(5):
            self.ocp_solver.solve_for_x0(x0_bar=self.x0, fail_on_nonzero_status=False, print_stats_on_failure=False) # It is ok to fail during the warmup phase

    def setup(self, config):
        mpc_config = config["mpc"]

        Fmax = mpc_config["Fmax"]
        xmax = mpc_config["xmax"]
        xmin = mpc_config["xmin"]
        N_horizon = mpc_config["N_horizon"]
        Tf = N_horizon * mpc_config["mpc_timestep"]  # Time horizon

        # Create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Call model creation function
        model, self.robot_sys = export_ode_model(config)
        model.name = model.name + str(self.worker_id)   # reassigned based on worker id
        ocp.model = model

        # Extract state and input dimensions
        nx = model.x.rows()
        nu = model.u.rows()

        # Generate obstacle collision constraints
        if config["collision"]["collision_avoidance_obstacle"] or config["collision"]["collision_avoidance_ground"]:
            self.add_hard_constraints(ocp, model, config["collision"])

        # Set stage cost module
        self.define_stage_cost(ocp, model, config)

        # Set terminal cost module
        if self.terminal_cost:
            self.define_terminal_cost(ocp, model, config)

        # Set input constraints
        ocp.constraints.lbu = -np.array(Fmax)
        ocp.constraints.ubu = np.array(Fmax)

        # Apply above to all inputs
        ocp.constraints.idxbu = np.arange(nu)

        # Set state constraints
        ocp.constraints.lbx = np.array(xmin)
        ocp.constraints.ubx = np.array(xmax)
        ocp.constraints.lbx_e = np.array(xmin)
        ocp.constraints.ubx_e = np.array(xmax)

        # Apply above to all states
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.idxbx_e = np.arange(nx)

        # Set initial constraint
        # ocp.constraints.x0 = self.x0
        ocp.constraints.x0 = np.zeros((1,nx))

        # set prediction horizon
        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = Tf # Total predicton time

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10
        ocp.solver_options.regularize_method = mpc_config["regularize_method"] # For the Hessian
        ocp.solver_options.levenberg_marquardt = mpc_config["levenberg_marquardt"]
        ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = mpc_config["nlp_solver_warm_start_first_qp_from_nlp"]
        ocp.solver_options.nlp_solver_warm_start_first_qp = mpc_config["nlp_solver_warm_start_first_qp"]
        ocp.solver_options.qp_solver_warm_start = mpc_config["qp_solver_warm_start"]

        if self.use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'
            ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
            ocp.solver_options.nlp_solver_max_iter = 6
        
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.nlp_solver_tol_stat = 1e-4
        # ocp.solver_options.qp_solver_iter_max
        
        # Finalize parameters, from obs if used, or/and NN input for yref and obs(center, rpy)
        if self.p_obs is not None and self.p_NN is not None:
            self.p = np.concatenate([self.obstacles_from_to, self.NN])
            ocp.model.p = ca.vertcat(self.p_obs, self.p_NN)
            ocp.parameter_values = np.zeros(ocp.model.p.shape[0])
        elif self.p_obs is not None:
            self.p = self.obstacles_from_to
            ocp.model.p = self.p_obs
            ocp.parameter_values = np.zeros(ocp.model.p.shape[0])
        elif self.p_NN is not None:
            self.p = self.NN
            ocp.model.p = self.p_NN
            ocp.parameter_values = np.zeros(ocp.model.p.shape[0])

        # Unique json and export directory
        solver_json = 'acados_json/acados_ocp_' + self.config["mpc"]["json_name"] + str(self.worker_id) + '.json'
        ocp.code_export_directory = 'acados_c_generated_code/worker' + str(self.worker_id)

        # Compile solver
        self.ocp_solver = AcadosOcpSolver(acados_ocp = ocp, json_file = solver_json, verbose=False, build=False, generate=False)

        # Extract vars
        self.nx = self.ocp_solver.acados_ocp.dims.nx
        self.nu = self.ocp_solver.acados_ocp.dims.nu
        self.N = self.ocp_solver.acados_ocp.dims.N

        # Update params which are set to be consistent to avoid recompilation
        self.update_initial_guess(self.x0)
        self.update_parameters()            # only used with NN or obstacles

    def update_initial_guess(self, x):
        x_inital_guess = np.tile(x, self.ocp_solver.acados_ocp.dims.N+1)
        self.ocp_solver.set_flat('x', x_inital_guess)

    def define_stage_cost(self, ocp, model, config):
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        Q_mat = np.diag(self.config["mpc"]["Q_mat"])
        R_mat = np.diag(self.config["mpc"]["R_mat"])

        ocp.cost.cost_type = 'NONLINEAR_LS'                 # Stage cost
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)  # Stage cost includes both states and input
        ocp.model.cost_y_expr = vertcat(model.x, model.u)   # Stage cost includes both states and input
        ocp.cost.yref  = np.zeros((ny, ))                   # Set stage references to match first entry of yref for all states and inputs

    def define_terminal_cost(self, ocp, model, config):
        Q_mat = np.diag(self.config["mpc"]["Q_mat"])
        self.ny_e = model.x.rows()
    
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = Q_mat                                # Terminal cost only inlcudes states
        ocp.model.cost_y_expr_e = model.x                   # Terminal cost only inlcudes states
        ocp.cost.yref_e = np.zeros((self.ny_e, ))                # Set terminal reference to match first entry of yref for states only

    def add_hard_constraints(self, ocp, model, collision_config):
        raise NotImplementedError("add_hard_constraints method not implemented in BaseMPCController.")
    
    def create_constraints_func(self, ocp, constraints):
        raise NotImplementedError("create_constraints_func method not implemented in BaseMPCController.")
    
    def update_parameters(self):
        warning("No parameters to update for this controller.")
        pass
    
    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", yref_now[:self.ny_e], api='new')  # Terminal reference (only x)
    
    def collect_traj(self, full_traj):
        qpos_traj = []
        qvel_traj = []
        u_traj = []

        if full_traj: # Extract full state, control trajectories
            for i in range(self.N):
                xi = self.ocp_solver.get(i, "x")
                ui = self.ocp_solver.get(i, "u")
                qpos_traj.append(xi[:self.nx//2])
                qvel_traj.append(xi[self.nx//2:])
                u_traj.append(ui)

            # Get final state (at step N)
            xN = self.ocp_solver.get(self.N, "x")
            qpos_traj.append(xN[:self.nx//2])
            qvel_traj.append(xN[self.nx//2:])

        qpos_traj = np.array(qpos_traj)
        qvel_traj = np.array(qvel_traj)
        u_traj = np.array(u_traj)

        # Convert q traj into cartesian traj
        if self.config["IK"]["output_xyz"]:
            xyz_traj = np.array(self.ee_fun(qpos_traj.T)).squeeze().T
        else:
            xyz_traj = 0

        return qpos_traj, qvel_traj, u_traj, xyz_traj
    
    def collect_cost(self, qpos_traj, qvel_traj, u_traj, yref_now):
        if qpos_traj.size == 0 or qvel_traj.size == 0 or u_traj.size == 0:
            raise ValueError("Empty trajectory detected, set full_traj=True.")

        # Full state matrix: X = [qpos | qvel]
        X = np.column_stack((qpos_traj, qvel_traj))  # shape (N+1, nx)
        U = u_traj                                   # shape (N, nu)

        stage_ref, terminal_ref = self.build_references(X, U, yref_now)

        stage_cost = self.compute_stage_cost(X, U, stage_ref)

        if self.terminal_cost:
            terminal_cost = self.compute_terminal_cost(X, terminal_ref)
        else:
            terminal_cost = 0.0

        total_cost = stage_cost + terminal_cost

        self.check_solver_cost(total_cost)

        return stage_cost, terminal_cost, total_cost
    
    def build_references(self, X, U, yref_now):
        nx, nu = X.shape[1], U.shape[1]

        if isinstance(yref_now, dict):
            stage_ref = yref_now["stage"]
            if self.terminal_cost:
                terminal_ref = yref_now["terminal"][:self.ny_e]
            else:
                terminal_ref = None
        else:
            stage_ref = np.tile(yref_now, (self.N, 1))
            if self.terminal_cost:
                terminal_ref = yref_now[:self.ny_e]
            else:
                terminal_ref = None

        return stage_ref, terminal_ref
    
    def compute_stage_cost(self, X, U, stage_ref):
        Q = np.diag(self.config["mpc"]["Q_mat"])
        R = np.diag(self.config["mpc"]["R_mat"])

        nx, nu = X.shape[1], U.shape[1]

        XU = np.hstack([X[:self.N], U[:self.N]])
        e = XU - stage_ref

        Q_full = np.block([
            [Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), R]
        ])

        stage_cost = 0.5 * self.mpc_timestep * np.sum(
            np.sum((e @ Q_full) * e, axis=1)
        )
        return stage_cost
    
    def compute_terminal_cost(self, X, terminal_ref):
        if self.terminal_cost:
            Q = np.diag(self.config["mpc"]["Q_mat"])
            e = X[self.N] - terminal_ref
            return 0.5 * e @ Q @ e
        else:
            return 0
    
    def check_solver_cost(self, total_cost):
        solver_cost = self.ocp_solver.get_cost()
        if not np.isclose(solver_cost, total_cost, rtol=1e-6, atol=1e-8):
            raise RuntimeError(
                f"Cost mismatch:\nsolver={solver_cost}\ncomputed={total_cost}"
            )

    def __call__(self, x, yref_now, full_traj):
        """Compute MPC input given MuJoCo state."""
        # Set yref
        self.set_yref(yref_now)

        if self.use_RTI:
            # Preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            
            status = self.ocp_solver.solve()
            if status != 0 and status != 2:
                if not self.config["mpc"]["json_name"] == "GT_controller":
                    raise RuntimeError(f"MPC solver returned status {status} in RTI phase 1")
                else:
                    return np.nan, np.nan, np.nan, np.nan, np.full((self.N+1, self.nx//2), np.nan), np.full((self.N+1, self.nx//2), np.nan), np.nan, np.nan, np.full((self.N+1, 3), np.nan)

            # Set initial state
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            # Feedback phase
            self.ocp_solver.options_set('rti_phase', 2)

            status = self.ocp_solver.solve()
            if status != 0 and status != 2:
                if not self.config["mpc"]["json_name"] == "GT_controller":
                    raise RuntimeError(f"MPC solver returned status {status} in RTI phase 2")
                else:
                    return np.nan, np.nan, np.nan, np.nan, np.full((self.N+1, self.nx//2), np.nan), np.full((self.N+1, self.nx//2), np.nan), np.nan, np.nan, np.full((self.N+1, 3), np.nan)

            # Get first control input
            u = self.ocp_solver.get(0, "u")
        
        else: # Without RTI (modofied solve_for_x0 to avoid min_step issue)
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            status = self.ocp_solver.solve()

            if status != 0 and status != 2:
                if not self.config["mpc"]["json_name"] == "GT_controller":
                    raise RuntimeError(f"MPC solver returned status {status} in SQP mode")
                else:
                    return np.nan, np.nan, np.nan, np.nan, np.full((self.N+1, self.nx//2), np.nan), np.full((self.N+1, self.nx//2), np.nan), np.nan, np.nan, np.full((self.N+1, 3), np.nan)
            
            u = self.ocp_solver.get(0, "u")
        
        # Collect traj
        qpos_traj, qvel_traj, u_traj, xyz_traj = self.collect_traj(full_traj)

        # Collect cost components
        stage_cost, terminal_cost, total_cost = self.collect_cost(qpos_traj, qvel_traj, u_traj, yref_now)

        # Collect distances between capsules
        if self.config["collision"]["collision_avoidance_obstacle"] or self.config["collision"]["collision_avoidance_ground"]:
            sq_dist = self.evaluate_distances(qpos_traj, qvel_traj)
        else:
            sq_dist = 0

        return u, stage_cost, terminal_cost, total_cost, qpos_traj, qvel_traj, u_traj, sq_dist, xyz_traj

@register_controller
class NNMPCController(BaseMPCController):
    '''
    Class for testing trained terminal value approximation with single endpoint reference. Can be used with or without terminal cost component.
    '''
    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)

        if not config["mpc"]["terminal_cost"]:
            raise ValueError("NNMPCController requires terminal cost to be True.")
    
    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", np.zeros((64,)), api='new')  # Terminal reference (only x)

    def define_terminal_cost(self, ocp, model, config):
        self.ny_e = model.x.rows()
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.eye(64)                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((64, ))                   # Set terminal reference to zero for NN output
        # Export trained NN model
        self.l4c_model = export_torch_model(config, self.worker_id)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(model.x))
        ocp.model.cost_y_expr_e = ca.transpose(y_sym)
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name
    
    def compute_terminal_cost(self, X, terminal_ref):
        # Terminal state (nx,)
        xN = X[self.N]

        # Evaluate NN numerically
        yN = np.asarray(self.l4c_model(xN)).squeeze()

        # NONLINEAR_LS terminal cost
        terminal_cost = 0.5 * np.sum(yN**2)
        
        return terminal_cost

@register_controller
class ManipulatorMPCController(BaseMPCController):
    '''
    Class for a basic MPC Controller with trajectory tracking and IK used for trajectory generation. Can be used with or without terminal cost component.
    Specifically for manipulator models or models which require IK. Tracking in joint space.
    '''
    def __init__(self, config, worker_id=0):        
        super().__init__(config, worker_id=worker_id)

    def add_hard_constraints(self, ocp, model, collision_config):
        # Generate collision constraints
        constraint_list = []

        if self.config["collision"]["collision_avoidance_obstacle"]:
            self.obs1_config = collision_config["obstacles"]["obs1"] # Assuming obs1 is the relevant obstacle for now

            # Extract obs1 info for parameters
            obs1_from = np.array(collision_config["obstacles"]["obs1"]["from"])
            obs1_to = np.array(collision_config["obstacles"]["obs1"]["to"])

            self.obstacles_from_to = np.concatenate([obs1_from, obs1_to])               # Property to save the actual location
            self.p_obs = ca.SX.sym('p_obs', self.obstacles_from_to.shape[0])            # Property to save the casadi symbols

            obstacle_constraints = build_obstacle_collision_constraints(
                self.robot_sys,
                collision_config["links"],
                collision_config["obstacles"],
                collision_config["collision_pairs"]["obstacle"],
                self.p_obs
            )
            constraint_list.append(obstacle_constraints)

        if self.config["collision"]["collision_avoidance_ground"]:
            ground_constraints = build_ground_collision_constraints(
                self.robot_sys,
                collision_config["links"],
                collision_config["ground_plane"],
                collision_config["collision_pairs"]["ground"],
            )
            constraint_list.append(ground_constraints)

        constraints = ca.vertcat(*constraint_list)

        print(f"Number of collision constraints: {constraints.shape[0]}")
        
        ocp.model.con_h_expr = constraints
        ocp.constraints.lh = np.zeros(constraints.shape[0])
        ocp.constraints.uh = 1e10 * np.ones(constraints.shape[0])

        ocp.model.con_h_expr_e = constraints
        ocp.constraints.lh_e = np.zeros(constraints.shape[0])
        ocp.constraints.uh_e = 1e10 * np.ones(constraints.shape[0])

        # Add slack only for GT controller. Reduce possibility of minstep failure
        # if self.config["mpc"]["json_name"] == "GT_controller":
        #     self.add_slack(ocp)

        self.create_constraints_func(ocp, constraints)

    # def add_slack(self, ocp):
    #     # idx of non-linear constraint to add slack to (for stage)
    #     ocp.constraints.idxsh = np.arange(ocp.model.con_h_expr.shape[0])
    #     # idx of non-linear constraint to add slack to (for terminal)
    #     ocp.constraints.idxsh_e = np.arange(ocp.model.con_h_expr_e.shape[0])

    #     # Slack penalty z for linear penalty & Z for quadratic penalty
    #     # Stage
    #     ocp.cost.zl = 500*np.ones(ocp.model.con_h_expr.shape[0])
    #     ocp.cost.Zl = 1000*np.ones(ocp.model.con_h_expr.shape[0])
    #     ocp.cost.zu = 500*np.ones(ocp.model.con_h_expr.shape[0])
    #     ocp.cost.Zu = 1000*np.ones(ocp.model.con_h_expr.shape[0])
    #     # Terminal
    #     ocp.cost.zl_e = 500*np.ones(ocp.model.con_h_expr_e.shape[0])
    #     ocp.cost.Zl_e = 1000*np.ones(ocp.model.con_h_expr_e.shape[0])
    #     ocp.cost.zu_e = 500*np.ones(ocp.model.con_h_expr_e.shape[0])
    #     ocp.cost.Zu_e = 1000*np.ones(ocp.model.con_h_expr_e.shape[0])

    def create_constraints_func(self, ocp, constraints):
        self.collision_constraint_fun = ca.Function(
            "collision_constraint_fun",
            [ocp.model.x, self.p_obs],
            [constraints],
            ["x", "p_obs"],
            ["h"]
            )
    
    def evaluate_distances(self, qpos_traj, qvel_traj):
        qpos = qpos_traj[0]
        qvel = qvel_traj[0]

        x = np.concatenate([qpos, qvel])

        sq_dist = np.array(self.collision_constraint_fun(x, self.p[:self.p_obs.shape[0]])).squeeze()

        return sq_dist
    
    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now["stage"][stage], api='new')
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", yref_now["terminal"][:self.ny_e], api='new')  # Terminal reference (only x)
    
    def update_parameters(self):
        # Stage & Terminal
        if self.p is not None:
            for stage in range(self.N+1):
                self.ocp_solver.set(stage, "p", self.p)                                             # Modify Goal/obstacle position

@register_controller
class NNManipulatorMPCController(ManipulatorMPCController):
    '''
    Class for testing trained terminal value approximation with trajectory tracking and IK used for trajectory generation.
    Specifically for manipulator models or models which require IK.
    '''

    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)

        if not config["mpc"]["terminal_cost"]:
            raise ValueError("NNMPCController requires terminal cost to be True.")

    def define_terminal_cost(self, ocp, model, config):
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.ones((1, 1))                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((1, ))                   # Set terminal reference to zero for NN output

        # Create parameters for goal, obstacle
        self.p = np.array(self.config["mpc"]["yref"])
        ocp.model.p = SX.sym('p', self.p.shape[0])  # Full yref as parameter
        ocp.parameter_values = np.zeros(self.p.shape[0])

        # Export trained NN model
        self.l4c_model = export_torch_model(config, self.worker_id)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(vertcat(model.x, ocp.model.p)))
        ocp.model.cost_y_expr_e = y_sym
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name
    
    def set_yref(self, yref_now):
        # Stage
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now["stage"][stage], api='new')
            self.ocp_solver.set(stage, "p", self.p)                                             # Modify Goal/obstacle position

        # Terminal
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", np.zeros((1,)), api='new')  # Terminal reference (only x)
            self.ocp_solver.set(self.N, "p", self.p)                                             # Modify Goal/obstacle position

    def compute_terminal_cost(self, X, terminal_ref):
        # Terminal state (nx,)
        xN = X[self.N]

        xN_p = np.concatenate([xN, self.p])

        # Evaluate NN numerically
        yN = np.asarray(self.l4c_model(xN_p)).squeeze()

        # NONLINEAR_LS terminal cost
        terminal_cost = 0.5 * yN**2
        return terminal_cost

@register_controller
class ManipulatorMPCController_eeTracker(ManipulatorMPCController):
    '''
    Same as ManipulatorMPCController but cost formulation is a hybrid of end-effector space tracking and joint velocities, input regularisation.
    to do: modify stage cost, modify set_yref??, modify terminal cost as well, modify compute stage cost and terminal cost.
    '''
    def __init__(self, config, worker_id=0):        
        super().__init__(config,  worker_id=worker_id)

        if not config["IK"]["output_xyz"]:
            raise ValueError("NNMPCController_eeTracker requires IK output to be in xyz format.")

    def define_stage_cost(self, ocp, model, config):
        nx = model.x.rows()
        nu = model.u.rows()

        self.build_FK_function(nx)

        ny = self.ee_expr.shape[0]+ nx//2 + nu # (for ee position tracking, joint velocity regularization, and input regularization respectively)

        Q_mat = np.diag(self.config["mpc"]["Q_mat"])
        Q_dot_mat = np.diag(self.config["mpc"]["Q_dot_mat"])
        R_mat = np.diag(self.config["mpc"]["R_mat"])

        q_dot = model.x[nx//2:]  # Joint velocities

        ocp.cost.cost_type = 'NONLINEAR_LS'                             # Stage cost
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, Q_dot_mat, R_mat)   # Stage cost includes both states and input
        ocp.model.cost_y_expr = vertcat(self.ee_expr, q_dot, model.u)             # Stage cost includes both states and input
        ocp.cost.yref  = np.zeros((ny, ))                               # Set stage references to match first entry of yref for all states and inputs

    def define_terminal_cost(self, ocp, model, config):
        Q_mat_e = np.diag(self.config["mpc"]["Q_mat_e"])
        # Q_mat = np.diag(self.config["mpc"]["Q_mat"])
        Q_dot_mat_e = np.diag(self.config["mpc"]["Q_dot_mat_e"])
        # Q_dot_mat = np.diag(self.config["mpc"]["Q_dot_mat"])

        nx = model.x.rows()
        self.ny_e = self.ee_expr.shape[0]+ nx//2

        q_dot = model.x[nx//2:]  # Joint velocities

        ocp.cost.cost_type_e = 'NONLINEAR_LS'                           # Terminal cost
        ocp.cost.W_e = scipy.linalg.block_diag(Q_mat_e, Q_dot_mat_e)        # Terminal cost only inlcudes states
        ocp.model.cost_y_expr_e = vertcat(self.ee_expr, q_dot)                    # Terminal cost only inlcudes states
        ocp.cost.yref_e = np.zeros((self.ny_e, ))                            # Set terminal reference to match first entry of yref for states only
    
    # def build_FK_function(self):
    #     # Create FK function from attachment_site SX
    #     self.ee_expr = getattr(self.robot_sys, "attachment_site")           # To feed to acados
    #     q_syms_list = ca.symvar(self.ee_expr)        # list of symbolic joints
    #     q_syms = ca.vertcat(*q_syms_list)      # stack into SX vector
    #     self.ee_fun = ca.Function("ee_fun", [q_syms], [self.ee_expr])  # FK function, To evaluate numerically

    def build_FK_function(self, nx):
        # # Get end-effector expression from robot system
        # self.ee_expr = getattr(self.robot_sys, "attachment_site")

        # # Full joint vector (nx/2 positional states)
        # nq = nx // 2
        # q_full = ca.SX.sym("q", nq)

        # # Get original symbolic variables inside expression
        # q_syms_list = ca.symvar(self.ee_expr)

        # # Sort to ensure deterministic order (important!)
        # q_syms_list = sorted(q_syms_list, key=lambda s: s.name())

        # # Substitute each original joint symbol with q_full[i]
        # expr = self.ee_expr
        # for i, q_old in enumerate(q_syms_list):
        #     expr = ca.substitute(expr, q_old, q_full[i])

        # # Build function with full joint vector
        # self.ee_fun = ca.Function("ee_fun", [q_full], [expr])

        # Create FK function from attachment_site SX
        self.ee_expr = getattr(self.robot_sys, "attachment_site")           # To feed to acados
        q_syms_list = ca.symvar(self.ee_expr)                               # list of symbolic joints
        q_syms = ca.vertcat(*q_syms_list)                                   # stack into SX vector
        self.ee_fun = ca.Function("ee_fun", [q_syms], [self.ee_expr])       # FK function, To evaluate numerically

    def compute_stage_cost(self, X, U, stage_ref):
        # --- Step 1: Prepare cost weights ---
        Q     = np.diag(self.config["mpc"]["Q_mat"])       # EE position weights
        Q_dot = np.diag(self.config["mpc"]["Q_dot_mat"])   # Joint velocity weights
        R     = np.diag(self.config["mpc"]["R_mat"])       # Control weights
        
        nx = X.shape[1]
        nq = nx // 2

        stage_cost = 0.0

        # --- Step 2: Loop through horizon ---
        for k in range(self.N):
            q     = X[k, :nq]
            qdot  = X[k, nq:]
            u     = U[k]

            # --- Evaluate FK using CasADi function ---
            ee_val = np.array(self.ee_fun(q)).squeeze()  # end-effector position

            # --- Stack stage output ---
            y = np.concatenate([ee_val, qdot, u])

            # --- Compute error to stage reference ---
            e = y - stage_ref[k]   # stage_ref should be (self.N, len(y)) array

            # --- Full stage weight ---
            W = scipy.linalg.block_diag(Q, Q_dot, R)

            # --- Accumulate cost ---
            stage_cost += 0.5 * self.mpc_timestep * e @ W @ e

        return stage_cost
    
    def compute_terminal_cost(self, X, terminal_ref):
            if not self.terminal_cost:
                return 0.0

            nx = X.shape[1]
            nq = nx // 2

            # --- Step 2: Extract final stage ---
            q_final = X[self.N, :nq]
            qdot_final = X[self.N, nq:]

            # --- Step 3: Evaluate end-effector FK ---
            ee_val = np.array(self.ee_fun(q_final)).squeeze()

            # --- Step 4: Stack terminal state for cost ---
            y_e = np.concatenate([ee_val, qdot_final])

            # --- Step 5: Compute terminal cost weights ---
            Q_mat_e = np.diag(self.config["mpc"]["Q_mat_e"])
            Q_dot_mat_e = np.diag(self.config["mpc"]["Q_dot_mat_e"])
            W_e = scipy.linalg.block_diag(Q_mat_e, Q_dot_mat_e)

            # --- Step 6: Compute error to terminal reference ---
            e = y_e - terminal_ref  # terminal_ref must match shape of y_e

            # --- Step 7: Return scalar terminal cost ---
            return 0.5 * e @ W_e @ e

@register_controller
class ManipulatorMPCController_eeTracker_point(ManipulatorMPCController_eeTracker):
    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)

    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')

        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", yref_now[:self.ny_e], api='new')  # Terminal reference (only x)

@register_controller
class NNManipulatorMPCController_eeTracker(ManipulatorMPCController_eeTracker, NNManipulatorMPCController):
    '''
    Class for testing trained terminal value approximation with stationary endpoint reference. No IK involved.
    
    Settings:
    - terminal_cost: true
    - model_name: "TwoDofArmModelAcados"
    - checkpoint_path: (set to trained model path)
    - IK_required: false during tesing, true during data generation

    '''
    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)
    
    def define_terminal_cost(self, ocp, model, config):
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.eye(64)                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((64, ))                   # Set terminal reference to zero for NN output

        # Extract joint velocities
        nx = model.x.rows()
        self.ny_e = self.ee_expr.shape[0]+ nx//2

        # Create parameters for goal, obstacle
        self.NN = np.array(self.config["mpc"]["yref"])
        self.p_NN = SX.sym('p', self.NN.shape[0])                    # Symbolic vars

        # Export trained NN model
        self.l4c_model = export_torch_model(config, self.worker_id)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(vertcat(model.x, self.p_NN, self.ee_expr)))
        ocp.model.cost_y_expr_e = ca.transpose(y_sym)
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name

    def compute_terminal_cost(self, X, terminal_ref):
        xN = X[self.N]

        # Extract q
        q_final = xN[:self.nx//2]

        # Compute EE position
        ee_val = np.array(self.ee_fun(q_final)).squeeze()

        # Build NN input (must match symbolic definition)
        xN_p = np.concatenate([xN, self.p[-self.p_NN.shape[0]:], ee_val])

        # Evaluate NN
        yN = np.asarray(self.l4c_model(xN_p)).squeeze()

        # NONLINEAR_LS terminal cost
        terminal_cost = 0.5 * np.sum(yN**2)

        return terminal_cost

@register_controller
class NNManipulatorMPCController_eeTracker_point(NNManipulatorMPCController_eeTracker):
    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)

        if not config["IK"]["point_reference"]:
            raise ValueError("NNManipulatorMPCController_eeTracker_point requires point_reference to be True.")
        
    def set_yref(self, yref_now):
        # Stage
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
            self.ocp_solver.set(stage, "p", self.p)                                             # Modify Goal/obstacle position

        # Terminal
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", np.zeros((64,)), api='new')                  # Terminal reference (only x)
            self.ocp_solver.set(self.N, "p", self.p)                                             # Modify Goal/obstacle position

@register_controller
class NNManipulatorMPCController_eeTracker_point_rndobs_TwoDofArm(NNManipulatorMPCController_eeTracker_point):
    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)

        if not config["collision"]["collision_avoidance_obstacle"]:
            raise ValueError("NNManipulatorMPCController_eeTracker_point_obs requires collision_avoidance_obstacle to be True.")

    def define_terminal_cost(self, ocp, model, config):
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.eye(64)                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((64, ))                   # Set terminal reference to zero for NN output

        # Extract joint velocities (Not actually used but required to build references)
        nx = model.x.rows()
        self.ny_e = self.ee_expr.shape[0]+ nx//2

        # Create parameters for goal, obstacle
        center = np.array(self.obs1_config["center"])
        rpy = np.array([self.obs1_config["rpy"][1]]) # Modify later for iiwa14
        yref = np.array(self.config["mpc"]["yref"])

        self.NN = np.concatenate([center, rpy, yref])               # Actual values 
        self.p_NN = SX.sym('p', self.NN.shape[0])                    # Symbolic vars

        # Export trained NN model
        self.l4c_model = export_torch_model(config, self.worker_id)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(vertcat(model.x, self.p_NN, self.ee_expr)))
        ocp.model.cost_y_expr_e = ca.transpose(y_sym)
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name

@register_controller
class NNManipulatorMPCController_eeTracker_point_rndobs_iiwa14(NNManipulatorMPCController_eeTracker_point_rndobs_TwoDofArm):
    def __init__(self, config, worker_id=0):
        super().__init__(config, worker_id=worker_id)

        if not config["collision"]["collision_avoidance_obstacle"]:
            raise ValueError("NNManipulatorMPCController_eeTracker_point_obs requires collision_avoidance_obstacle to be True.")

    def define_terminal_cost(self, ocp, model, config):
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.eye(64)                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((64, ))                   # Set terminal reference to zero for NN output

        # Extract joint velocities (Not actually used but required to build references)
        nx = model.x.rows()
        self.ny_e = self.ee_expr.shape[0]+ nx//2

        # Create parameters for goal, obstacle
        center = np.array(self.obs1_config["center"])
        rpy = np.array(self.obs1_config["rpy"]) # Modify later for iiwa14
        yref = np.array(self.config["mpc"]["yref"])

        self.NN = np.concatenate([center, rpy, yref])               # Actual values 
        self.p_NN = SX.sym('p', self.NN.shape[0])                    # Symbolic vars

        # Export trained NN model
        self.l4c_model = export_torch_model(config, self.worker_id)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(vertcat(model.x, self.p_NN, self.ee_expr)))
        ocp.model.cost_y_expr_e = ca.transpose(y_sym)
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name