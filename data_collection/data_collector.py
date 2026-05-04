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

    # Split n_runs roughly evenly across workers
    run_indices_per_worker = [[] for _ in range(n_workers)]
    for i in range(n_runs):
        run_indices_per_worker[i % n_workers].append(i)

    start_time = time.time()
    all_logs = {}

    tprint(f"Launching {n_workers} workers for {n_runs} runs")

    ### === PHASE 1: First iteration (1 job per worker) === ###
    tprint("--- Phase 1: Running first iteration for each worker sequentially ---")
    first_runs_per_worker = [runs[0] for runs in run_indices_per_worker if runs]  # take the first run
    for worker_id, run_index in enumerate(first_runs_per_worker):
        logs = worker(
            worker_id=worker_id,
            run_indices=[run_index],  # single run
            model_name=model_name,
            output_dir=output_dir,
            data_config=data_config,
            config=config,
        )
        all_logs.update(logs)

    ### === PHASE 2: Remaining runs asynchronously === ###
    tprint("--- Phase 2: Running remaining jobs asynchronously ---")
    remaining_indices_per_worker = [
        runs[1:] if len(runs) > 1 else [] for runs in run_indices_per_worker
    ]

    # Only launch workers that have remaining runs
    workers_with_remaining = [
        (w_id, runs) for w_id, runs in enumerate(remaining_indices_per_worker) if runs
    ]

    # Submit to job pool
    if workers_with_remaining:
        with ProcessPoolExecutor(max_workers=len(workers_with_remaining)) as executor:
            futures = [
                executor.submit(
                    worker,
                    worker_id=w_id,
                    run_indices=runs,
                    model_name=model_name,
                    output_dir=output_dir,
                    data_config=data_config,
                    config=config,
                )
                for w_id, runs in workers_with_remaining
            ]

            # Check result
            for future in as_completed(futures):
                try:
                    logs = future.result()  # Returns log from main()
                    all_logs.update(logs)   # Add to all_logs
                    tprint(f"Collected logs. Total runs so far: {len(all_logs)}/{n_runs}")

                except Exception as e:
                    error_msg = f"ERROR: A worker failed with {type(e).__name__}: {e}"
                    tprint(error_msg)
                    tprint(f"Full traceback:")
                    import traceback
                    traceback.print_exc()
                    tprint(f"Continuing to collect results from other workers...")

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