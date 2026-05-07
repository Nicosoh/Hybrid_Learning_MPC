import argparse

def worker(worker_id, run_indices, model_name, output_dir, data_config, config):
    import os
    import sys
    import traceback
    import random
    import numpy as np
    import time
    from datetime import datetime
    from main import main

    def tprint(*args, **kwargs):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}]", *args, **kwargs)

    os.environ["OMP_NUM_THREADS"] = "1"

    # Individual directory for each worker
    worker_output_dir = os.path.join(output_dir, f"worker_{worker_id}")
    os.makedirs(worker_output_dir, exist_ok=True)

    # Log file for this worker
    log_file_path = os.path.join(worker_output_dir, f"worker_{worker_id}.log")
    
    # Set unique seed combining time + worker_id for complete uniqueness
    run_seed = worker_id
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Save original stdout/stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    try:
        with open(log_file_path, "a") as log_file:
            # Redirect stdout/stderr to log file
            sys.stdout = log_file
            sys.stderr = log_file

            tprint(f"[Worker {worker_id}] starting with {len(run_indices)} runs")
            tprint(f"[Worker {worker_id}] PID: {os.getpid()}")
            sys.stdout.flush()

            all_logs = {}
            for run_index in run_indices:
                success = False
                retry_count = 0
                while not success:  # keep retrying until it succeeds
                    run_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}_run{run_index}"
                    try:
                        logs = main(
                            model_name,
                            data_collection=True,
                            output_dir=worker_output_dir,
                            timestamp=run_timestamp,
                            data_config=data_config,
                            config=config,
                            worker_id=worker_id,
                        )
                        all_logs[run_timestamp] = logs
                        success = True
                        tprint(f"[Worker {worker_id}] Finished run {run_index}")
                        sys.stdout.flush()

                    except Exception as e:
                        retry_count += 1
                        error_msg = f"[Worker {worker_id}] Run {run_index} attempt {retry_count} failed: {type(e).__name__}: {e}"
                        tprint(error_msg)
                        print(traceback.format_exc())
                        sys.stdout.flush()
                        
                        if retry_count >= 10:
                            raise RuntimeError(f"Run {run_index} failed after 10 attempts: {e}")

            tprint(f"[Worker {worker_id}] completed all assigned runs")
            sys.stdout.flush()

    except Exception as e:
        error_msg = f"[Worker {worker_id}] FATAL ERROR: {type(e).__name__}: {e}"
        tprint(error_msg)
        tprint(traceback.format_exc())
        sys.stdout.flush()
        
        raise  # Re-raise so ProcessPoolExecutor can see it

    finally:
        # Redirect stdout/stderr back to original
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return all_logs

def run_data_collector(model_name, data_config_path="data_collection/data_config.yaml", run_dir=None, config=None):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import yaml
    import time
    import os
    from utils import save_yaml
    from data_collection import save_npz
    from datetime import datetime

    def tprint(*args, **kwargs):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}]", *args, **kwargs)

    # Load data_collector config
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)["data_collector"]

    # Extract config
    n_runs = data_config["runs"]
    workers = data_config["workers"]

    # Define number of workers
    n_workers = min(mp.cpu_count(), workers)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Output dir
    if run_dir is None:
        output_dir = os.path.join("data", f"{timestamp}_{model_name}_data_collection")
    else:
        output_dir = run_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save config used
    save_yaml(data_config, os.path.join(output_dir, "data_config.yaml"))

    # 1. Properly group indices for the workers
    run_indices_per_worker = [[] for _ in range(n_workers)]
    for i in range(n_runs):
        run_indices_per_worker[i % n_workers].append(i)

    # 2. Package jobs (one job per worker)
    # Filter out empty lists if n_runs < n_workers
    jobs = [
        (w_id, indices, model_name, output_dir, data_config, config)
        for w_id, indices in enumerate(run_indices_per_worker) if indices
    ]

    start_time = time.time()
    all_logs = {}
    
    tprint(f"Starting {len(jobs)} parallel worker groups...")

    # 3. Submit grouped jobs
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, *job): job[0] for job in jobs}
        
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                if "error" in result:
                    tprint(f"Worker {worker_id} finished with errors.")
                    all_logs.update(result.get("partial_logs", {}))
                else:
                    all_logs.update(result)
                    tprint(f"Worker {worker_id} completed all assigned tasks.")
            except Exception as e:
                tprint(f"Worker {worker_id} process crashed: {e}")

    elapsed = time.time() - start_time

    # Save all_logs when done
    save_npz(f"{timestamp}_{model_name}_logs.npz", data=all_logs, output_dir=output_dir)

    tprint("=== Data collection finished ===")
    tprint(f"Total elapsed time: {elapsed:.2f} seconds")
    tprint(f"Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model by name")
    parser.add_argument("model", type=str, help="Name of the model to run")
    args = parser.parse_args()

    print(f"Starting data collection for model: {args.model}")
    run_data_collector(args.model)
    print("Done.")

# import sys
# import time
# import yaml
# import traceback
# import random
# import numpy as np
# import multiprocessing as mp
# from datetime import datetime
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import os
# import argparse

# # =========================================================
# # Worker (single run)
# # =========================================================
# def worker(worker_id, run_index, model_name, output_dir, data_config, config):
#     # Imports must be inside the worker for ProcessPool compatibility in "spawn" mode
#     import os
#     import sys
#     import traceback
#     import random
#     import numpy as np
#     from datetime import datetime
#     # Late import of main to ensure worker environment is set
#     try:
#         from main import main
#     except ImportError:
#         return {"error": "Could not import main", "run_index": run_index}

#     def tprint_log(file, *args, **kwargs):
#         ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(f"[{ts}]", *args, file=file, **kwargs)
#         file.flush()

#     # Isolate CPU usage
#     os.environ["OMP_NUM_THREADS"] = "1"

#     worker_output_dir = os.path.join(output_dir, f"worker_{worker_id}")
#     os.makedirs(worker_output_dir, exist_ok=True)
#     log_file_path = os.path.join(worker_output_dir, f"worker_{worker_id}.log")

#     # Seed based on run_index for variety
#     run_seed = worker_id
#     random.seed(run_seed)
#     np.random.seed(run_seed)

#     run_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}_run{run_index}"
    
#     try:
#         with open(log_file_path, "a") as log_file:
#             # We don't reassign sys.stdout globally to avoid silencing the scheduler.
#             # Instead, we catch the output if main() doesn't support a logger.
#             # If main() has its own prints, you can use a contextlib.redirect_stdout here.
#             from contextlib import redirect_stdout, redirect_stderr
            
#             with redirect_stdout(log_file), redirect_stderr(log_file):
#                 tprint_log(log_file, "=" * 60)
#                 tprint_log(log_file, f"[Worker {worker_id}] START run {run_index}")
#                 tprint_log(log_file, f"Timestamp: {run_timestamp}")
#                 tprint_log(log_file, "=" * 60)

#                 logs = main(
#                     model_name,
#                     data_collection=True,
#                     output_dir=worker_output_dir,
#                     timestamp=run_timestamp,
#                     data_config=data_config,
#                     config=config,
#                     worker_id=worker_id,
#                 )

#                 tprint_log(log_file, f"[Worker {worker_id}] SUCCESS run {run_index}\n")
                
#             return {run_timestamp: logs}

#     except Exception:
#         err = traceback.format_exc()
#         return {"error": err, "run_index": run_index}

# # =========================================================
# # Main scheduler
# # =========================================================
# def run_data_collector(model_name, data_config_path="data_collection/data_config.yaml", run_dir=None, config=None):
#     from utils import save_yaml
#     from data_collection import save_npz

#     def tprint(*args, **kwargs):
#         ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(f"[{ts}]", *args, **kwargs)

#     # Load data_collector config
#     if not os.path.exists(data_config_path):
#         print(f"Error: Config file {data_config_path} not found.")
#         return

#     with open(data_config_path, "r") as f:
#         data_config = yaml.safe_load(f)["data_collector"]

#     n_runs = data_config.get("runs", 1)
#     max_workers_cfg = data_config.get("workers", 1)
#     n_workers = min(mp.cpu_count(), max_workers_cfg)

#     timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
#     output_dir = run_dir if run_dir else os.path.join("data", f"{timestamp}_{model_name}_data_collection")
#     os.makedirs(output_dir, exist_ok=True)

#     save_yaml(data_config, os.path.join(output_dir, "data_config.yaml"))

#     start_time = time.time()
#     all_logs = {}
#     completed_runs = 0

#     tprint(f"Launching {n_workers} workers for {n_runs} runs")

#     # Initial job list
#     remaining_jobs = [
#         (i % n_workers, i, model_name, output_dir, data_config, config)
#         for i in range(n_runs)
#     ]

#     # Use ProcessPoolExecutor to truly bypass the GIL
#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         attempt = 0
#         max_rounds = 5 

#         while remaining_jobs and attempt < max_rounds:
#             attempt += 1
#             if attempt > 1:
#                 tprint(f"=== Scheduler round {attempt}, retrying {len(remaining_jobs)} failed jobs ===")

#             futures = {executor.submit(worker, *job): job for job in remaining_jobs}
#             next_round_jobs = []

#             for fut in as_completed(futures):
#                 job = futures[fut]
#                 run_index = job[1]

#                 try:
#                     result = fut.result()

#                     if result and "error" in result:
#                         tprint(f"Run {run_index} failed (see worker log) → retry next round")
#                         next_round_jobs.append(job)
#                     else:
#                         all_logs.update(result)
#                         completed_runs += 1
#                         tprint(f"Progress: [{completed_runs}/{n_runs}] runs completed.")

#                 except Exception as e:
#                     tprint(f"Run {run_index} crashed the process: {e}")
#                     next_round_jobs.append(job)

#             remaining_jobs = next_round_jobs
            
#     if remaining_jobs:
#         tprint(f"WARNING: {len(remaining_jobs)} runs failed after {max_rounds} attempts.")
#     else:
#         tprint("All runs completed successfully 🎉")

#     # Save results
#     save_npz(
#         f"{timestamp}_{model_name}_logs.npz",
#         data=all_logs,
#         output_dir=output_dir,
#     )

#     tprint("=== DONE ===")
#     tprint(f"Elapsed: {time.time() - start_time:.2f}s")
#     tprint(f"Results saved to: {output_dir}")


# # =========================================================
# # Entry point
# # =========================================================
# if __name__ == "__main__":
#     # Crucial for Windows/macOS and robustness with CUDA/Multiprocessing
#     mp.set_start_method("spawn", force=True)

#     parser = argparse.ArgumentParser(description="Run a model by name")
#     parser.add_argument("model", type=str, help="Name of the model to run")
#     args = parser.parse_args()

#     print(f"Starting data collection for model: {args.model}")
#     run_data_collector(args.model)