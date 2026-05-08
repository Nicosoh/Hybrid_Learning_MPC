# Main calling function that passes arguments for all steps of the deep learning process.
# 1. Training
# 2. Evaluation
# 3. 

import argparse
import configparser
import os

from datetime import datetime
from neural_network.scripts import train_model, evaluate_model, train_model_TD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run different scripts based on the provided arguments"
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True,
                                       help="Script to run")

    # TRAIN_MODEL
    parser_train_model = subparsers.add_parser("train_model", help="Train the model")
    parser_train_model.add_argument(
        'config_path', type=str, help='Path to .ini config file for training'
    )

    # TRAIN_MODEL_TD
    parser_train_model_TD = subparsers.add_parser("train_model_TD", help="Train the model with TD")
    parser_train_model_TD.add_argument(
        'config_path', type=str, help='Path to .ini config file for training'
    )

    # EVALUATE_MODEL
    parser_evaluate_model = subparsers.add_parser("evaluate_model", help="Evaluate model")
    parser_evaluate_model.add_argument(
        'config_path', type=str, help='Path to .ini config file for testing'
    )

    args = parser.parse_args()

    # Create base output directory
    output_dir = "neural_network/output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{args.script_name}"
    run_dir = os.path.join(output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving current data to: {run_dir}")

    if args.script_name == 'train_model':
        config = configparser.ConfigParser()
        config.read(args.config_path)
        print(f'Training with config from {args.config_path}')
        train_loss, stationary_ratios_mean = train_model(config, run_dir)
    if args.script_name == 'train_model_TD':
        config = configparser.ConfigParser()
        config.read(args.config_path)
        print(f'Training with config from {args.config_path}')
        train_loss, stationary_ratios_mean = train_model_TD(config, run_dir)
    elif args.script_name == 'evaluate_model':
        evaluate_model(args.config_path, run_dir)