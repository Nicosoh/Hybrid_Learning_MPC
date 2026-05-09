import os
import sys
import yaml
import json
import shutil
import datetime
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from utils import save_yaml

def main():
    # Helper function to append to log file with timestamp
    def log_td(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        print(line, end="")
        with open(td_log_path, "a") as f:
            f.write(line)
    
    def update_plot(x, ground_truth, controller, MSE, MSE_std, train_loss, stationary_ratios):
        """Update and save plot with current metrics in separate subplots."""
        ax1.clear()
        ax2.clear()
        ax3.clear()  # new subplot for train loss
        ax4.clear()  # new subplot for stationary ratio
        
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
        if len(MSE_std) == len(MSE):
            ax2.fill_between(x, MSE - MSE_std, MSE + MSE_std, color='r', alpha=0.3, label="MSE ± std")
        
        ax2.set_ylabel("Mean Squared Error")
        ax2.tick_params(axis='y')
        ax2.set_yscale('log')
        ax2.grid(True, which="both", linestyle=":", alpha=0.4)
        ax2.legend(loc='upper right')
        
        # Bottom subplot: training loss
        train_loss = np.array(train_loss)
        ax3.plot(x, train_loss, label="Train Loss", color='b', linewidth=0.8)
        ax3.set_xlabel("TD Loop")
        ax3.set_ylabel("Train Loss")
        ax3.grid(True, which="both", linestyle=":", alpha=0.4)
        ax3.legend(loc='upper right')

        # New subplot: stationary ratio
        stationary_ratios = np.array(stationary_ratios)
        ax4.plot(x, stationary_ratios, label="Stationary Ratio", color='m', linewidth=0.8)
        ax4.set_xlabel("TD Loop")
        ax4.set_ylabel("Stationary Ratio")
        ax4.grid(True, which="both", linestyle=":", alpha=0.4)
        ax4.legend(loc='upper right')
        
        fig.tight_layout()
        plt.savefig(plt_save_path, dpi=400, bbox_inches='tight')
    
    # Load TD config
    TD_config_path = "TD/TD_config.yaml"
    with open(TD_config_path, "r") as f:
        TD_config = yaml.safe_load(f)
    
    # Extract config from TD config
    model_name = TD_config["model_name"]
    data_config_path = TD_config["data_config_path"]
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
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    
    x = []
    ground_truth = []
    controller = []
    MSE = []
    MSE_std = []
    train_loss = []
    stationary_ratios = []
    
    # Absolute path to the worker script
    TD_loop_worker_path = os.path.join(os.path.dirname(__file__), "TD_loop_worker.py")
        
    # TD Loop, Zero indexed, so minus one from the folder number
    for loop in range(start_loop, TD_loops):
        log_td(f"=== Starting TD Loop {loop+1}/{TD_loops} ===")
        
        # Spawn subprocess for this loop
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
                    MSE_std.append(metrics["mse_std"])
                    train_loss.append(metrics["train_loss"])
                    stationary_ratios.append(metrics["stationary_ratio_mean"])

                    log_td(f"Metrics: GT={metrics['gt_cost']:.4f}, CTRL={metrics['ctrl_cost']:.4f}, MSE={metrics['mse']:.4e}, TR_loss={metrics['train_loss']:.4f}, Stationary Ratio={metrics['stationary_ratio_mean']:.4f}")
                else:
                    log_td(f"ERROR: {metrics.get('error', 'Unknown error')}")
            else:
                log_td(f"ERROR: No metrics file found at {metrics_path}")
            
            # Update plot after each loop
            update_plot(x, ground_truth, controller, MSE, MSE_std, train_loss, stationary_ratios)
            
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