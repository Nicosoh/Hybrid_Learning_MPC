import os
import sys
import yaml
import json
import shutil
import datetime
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from TD.replay_buffer import ReplayBuffer
from utils import save_yaml

def main():
    # Helper function to append to log file with timestamp
    def log_td(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        print(line, end="")
        with open(td_log_path, "a") as f:
            f.write(line)
    
    def update_plot(x, ground_truth, controller, MSE, MSE_std, main_losses, td_losses, offline_losses, stationary_losses, lam, ema_MSE):
        """Update and save plot with current metrics in separate subplots."""
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        ax7.clear()

        # Top subplot: ground truth vs controller
        ax1.plot(x, ground_truth, label="Ground Truth", linewidth=0.8)
        ax1.plot(x, controller, label="Controller", linewidth=0.8)
        ax1.set_ylabel("Mean Cost")
        ax1.grid(True, which="both", linestyle=":", alpha=0.4)
        ax1.legend()
        
        # Middle subplot: MSE ± std
        MSE = np.array(MSE)
        MSE_std = np.array(MSE_std)
        
        ax2.plot(x, MSE, label="MSE", color='green', linewidth=0.8)
        ax2.plot(x, ema_MSE, label="EMA MSE", color='blue', linewidth=0.8)
        if len(MSE_std) == len(MSE):
            ax2.fill_between(x, MSE - MSE_std, MSE + MSE_std, color='r', alpha=0.3, label="MSE ± std")
        
        ax2.set_ylabel("Mean Squared Error")
        ax2.tick_params(axis='y')
        ax2.set_yscale('log')
        ax2.grid(True, which="both", linestyle=":", alpha=0.4)
        ax2.legend(loc='upper right')
        
        # Bottom subplot: training loss
        main_losses = np.array(main_losses)
        ax3.plot(x, main_losses, label="Main Loss", color='b', linewidth=0.8)
        ax3.set_ylabel("Main Loss")
        ax3.grid(True, which="both", linestyle=":", alpha=0.4)
        ax3.legend(loc='upper right')
        ax3.set_yscale('log')

        # New subplot: TD loss
        td_losses = np.array(td_losses)
        ax4.plot(x, td_losses, label="TD Loss", color='g', linewidth=0.8)
        ax4.set_ylabel("TD Loss")
        ax4.grid(True, which="both", linestyle=":", alpha=0.4)
        ax4.legend(loc='upper right')
        ax4.set_yscale('log')

        # New subplot: offline loss
        offline_losses = np.array(offline_losses)
        ax5.plot(x, offline_losses, label="Offline Loss", color='orange', linewidth=0.8)
        ax5.set_ylabel("Offline Loss")
        ax5.grid(True, which="both", linestyle=":", alpha=0.4)
        ax5.legend(loc='upper right')
        ax5.set_yscale('log')

        # New subplot: stationary loss
        stationary_losses = np.array(stationary_losses)
        ax6.plot(x, stationary_losses, label="Stationary Loss", color='m', linewidth=0.8)
        ax6.set_ylabel("Stationary Loss")
        ax6.grid(True, which="both", linestyle=":", alpha=0.4)
        ax6.legend(loc='upper right')
        ax6.set_yscale('log')

        lam = np.array(lam)
        ax7.plot(x, lam, label="Lambda", color='c', linewidth=0.8)
        ax7.set_xlabel("TD Loop")
        ax7.set_ylabel("Lambda")
        ax7.grid(True, which="both", linestyle=":", alpha=0.4)
        ax7.legend(loc='upper right')
        ax7.set_yscale('log')

        fig.tight_layout()
        plt.savefig(plt_save_path, dpi=400, bbox_inches='tight')
    
    # Load TD config
    TD_config_path = "TD/TD_config.yaml"
    with open(TD_config_path, "r") as f:
        TD_config = yaml.safe_load(f)
    
    # Extract config from TD config
    model_name = TD_config["model_name"]
    data_config_path = TD_config["data_config_path"]
    loop0_data_config_path = TD_config["loop0_data_config_path"]
    train_config_path = TD_config["train_config_path"]
    TD_loops = TD_config["TD_loops"]
    resume_training = TD_config["resume_training"]
    
    # Check if resume training
    start_loop = 0
    main_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if resume_training:
        start_loop = TD_config["loop_to_resume_from"] - 1
        main_output_dir = TD_config["TD_dir"]
        suffix = f"_resume_from_{start_loop + 1}"
        
        # Delete the directory for the loop we are resuming from
        loop_dir_to_delete = os.path.join(main_output_dir, f"loop_{start_loop + 1}")

        if os.path.exists(loop_dir_to_delete):
            print(f"Deleting existing directory: {loop_dir_to_delete}")
            shutil.rmtree(loop_dir_to_delete)
    
    else:
        base_dir = "TD/output"
        main_output_dir = os.path.join(base_dir, f"{main_timestamp}_{model_name}_TD")
        os.makedirs(main_output_dir, exist_ok=True)
        suffix = ""

    # Load Model config
    with open(f"configs/{model_name}config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    # Paths to save files to
    plt_save_path = os.path.join(main_output_dir, f"TD_plot{suffix}.png")
    td_log_path   = os.path.join(main_output_dir, f"TD{suffix}.log")
    TD_yaml_save_path = os.path.join(main_output_dir, f"TD_config{suffix}.yaml")
    model_config_yaml_save_path = os.path.join(main_output_dir, f"Model_config{suffix}.yaml")

    save_yaml(TD_config, TD_yaml_save_path)
    save_yaml(model_config, model_config_yaml_save_path)
    
    with open(td_log_path, "w") as f:
        f.write("=== TD LOG (SUBPROCESS MODE) ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Started at: {main_timestamp}\n")
        f.write(f"TD loops: {TD_loops}\n")
        f.write("=" * 50 + "\n\n")
    
    # Setup plot
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(8, 14), sharex=True)
    
    x = []
    ground_truth = []
    controller = []
    MSE = []
    MSE_std = []
    ema_MSE = []
    main_losses = []
    td_losses = []
    offline_losses = []
    stationary_losses = []
    lam = []
    
    # Absolute path to the worker script
    TD_loop_worker_path = os.path.join(os.path.dirname(__file__), "TD_loop_worker.py")
        
    # TD Loop, Zero indexed, so minus one from the folder number
    for loop in range(start_loop, TD_loops):
        log_td(f"=== Starting TD Loop {loop+1}/{TD_loops} ===")

        # Spawn subprocess for this loop
        if loop == 0:
            cmd = [
                sys.executable,
                TD_loop_worker_path,
                str(loop),
                main_output_dir,
                model_name,
                loop0_data_config_path,
                train_config_path,
                TD_config_path,
                ]
        else:
            cmd = [
                sys.executable,
                TD_loop_worker_path,
                str(loop),
                main_output_dir,
                model_name,
                data_config_path,
                train_config_path,
                TD_config_path,
                ]
        
        try: # Try to run the subprocess
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            # If result is not successful
            if result.returncode != 0:
                log_td(f"Loop worker failed with return code {result.returncode}")
                if result.stdout:
                    log_td(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    log_td(f"STDERR:\n{result.stderr}")
                log_td(f"ERROR: Loop {loop+1} failed")
                continue
            
            # Path to metrics
            loop_dir = os.path.join(main_output_dir, f"loop_{loop+1}")
            metrics_path = os.path.join(loop_dir, "metrics.json")
            
            # Read metrics
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                if metrics.get("success"):
                    x.append(metrics["loop"])
                    ground_truth.append(metrics["gt_cost"])
                    controller.append(metrics["ctrl_cost"])
                    MSE.append(metrics["mse"])
                    MSE_std.append (metrics["mse_std"])
                    main_losses.append(metrics["main_loss"])
                    td_losses.append(metrics["td_loss"])
                    offline_losses.append(metrics["offline_loss"])
                    stationary_losses.append(metrics["stationary_loss"])
                    lam.append(metrics["lam"])
                    ema_MSE.append(metrics["ema_mse"])

                    log_td(f"Metrics: GT={metrics['gt_cost']:.4f}, CTRL={metrics['ctrl_cost']:.4f}, MSE={metrics['mse']:.4e}, TR_loss={metrics['main_loss']:.4f}, TD Loss={metrics['td_loss']:.4f}, Offline Loss={metrics['offline_loss']:.4f}, Stationary Loss={metrics['stationary_loss']:.4f}")
                else:
                    log_td(f"ERROR: {metrics.get('error', 'Unknown error')}")
            else:
                log_td(f"ERROR: No metrics file found at {metrics_path}")
            
            # Update plot after each loop
            update_plot(x, ground_truth, controller, MSE, MSE_std, main_losses, td_losses, offline_losses, stationary_losses, lam, ema_MSE)
            
        except Exception as e:
            log_td(f"ERROR spawning/running loop worker: {e}")
            import traceback
            log_td(traceback.format_exc())
        
        log_td(f"=== Finished TD Loop {loop+1} ===\n")
    
    log_td(f"\n=== TD COMPLETE ===")
    log_td(f"Total loops completed: {len(x)}/{TD_loops}")
    plt.savefig(plt_save_path, dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    main()