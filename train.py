import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import torch
import yaml

from src.data.dataset import FlowMatchingDataset
from src.models.networks import (
    MLP,
    ConditionalVelocityModel,
)
from src.training.trainer import FlowMatchingTrainer
from src.utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='Flow Matching Training')
    parser.add_argument('--config', type=str, default='./src/config/config_chengdu.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Directory to save checkpoints and results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override project.seed for an independent repeat')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional run-name prefix (letters, digits, dash, underscore)')
    return parser.parse_args()


def _git_value(*args):
    try:
        result = subprocess.run(
            ['git', *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_manifest(path, config_path, config, device, train_dataset, validation_dataset):
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {'visible_index': index, 'name': torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
    dataset_summary = None
    dataset_folder = getattr(train_dataset, 'input_folder', None)
    if dataset_folder is not None:
        dataset_summary_path = os.path.join(dataset_folder, 'dataset_summary.json')
        if os.path.exists(dataset_summary_path):
            with open(dataset_summary_path, encoding='utf-8') as stream:
                dataset_summary = json.load(stream)

    git_status = _git_value('status', '--porcelain')
    manifest = {
        'config_path': os.path.abspath(config_path),
        'seed': int(config.get('project', {}).get('seed', 42)),
        'deterministic': bool(config.get('project', {}).get('deterministic', True)),
        'git_commit': (
            _git_value('rev-parse', 'HEAD')
            or os.environ.get('TRAJFLOW_GIT_COMMIT')
        ),
        'git_dirty': None if git_status is None else bool(git_status),
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'device': str(device),
        'cuda_devices': cuda_devices,
        'train_samples': len(train_dataset),
        'validation_samples': len(validation_dataset) if validation_dataset is not None else 0,
        'split_file': getattr(train_dataset, 'split_indices_path', None),
        'dataset_summary': dataset_summary,
    }
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.seed is not None:
        config.setdefault('project', {})['seed'] = args.seed
    seed = int(config.get('project', {}).get('seed', 42))
    deterministic = bool(config.get('project', {}).get('deterministic', True))
    seed_everything(seed, deterministic=deterministic)

    # Validate model configuration
    # Check: if multiple models are enabled within flow_matching and ddpm, baseline, raise error
    enabled_models = []
    if config.get('baseline', {}).get('enabled', False):
        enabled_models.append('baseline')
    if config.get('flow_matching', {}).get('enabled', False):
        enabled_models.append('flow_matching')
    if config.get('ddpm', {}).get('enabled', False):
        enabled_models.append('ddpm')
    if len(enabled_models) == 0:
        raise ValueError("No model type enabled in configuration. Please enable one of 'baseline', 'flow_matching', or 'ddpm'.")
    if len(enabled_models) > 1:
        raise ValueError(f"Multiple model types enabled: {enabled_models}. Please enable only one.")

    # Check the model type
    if config.get('baseline', {}).get('enabled', False):
        print("Using Baseline model")
    elif config['flow_matching']['enabled']:
        print("Using Flow Matching model")
    elif config['ddpm']['enabled']:
        print("Using DDPM model")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.run_name is not None:
        if not args.run_name.replace('-', '').replace('_', '').isalnum():
            raise ValueError("run-name may contain only letters, digits, dash, and underscore")
        run_basename = f"{args.run_name}_seed{seed}_{timestamp}"
    else:
        run_basename = f"run_seed{seed}_{timestamp}"
    if config['project']['output_dir']:
        run_dir = os.path.join(config['project']['output_dir'], run_basename)
    else:
        run_dir = os.path.join(args.output, run_basename)
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Set device from config.training
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset_params = config['data']
    dataset = FlowMatchingDataset(config, mode='train')
    validation_dataset = None
    if config.get('training', {}).get('use_validation', False):
        validation_dataset = FlowMatchingDataset(config, mode='val')

    # Create model
    model_params = config['model']
    model_type = config['model']['type']
    if dataset_params['parametrized'] == True:
        M = dataset_params['parametrized_M']
    else:
        M = dataset_params['trajectory_length']
    input_dim = M * 2
    hidden_dim = model_params['hidden_dim']

    # Check if conditional flow is enabled
    conditional = config.get('condition', {}).get('enabled', False)

    if  not config.get('baseline', {}).get('enabled', False):
        if conditional:
            model = ConditionalVelocityModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                condition_dim=dataset.location_dim,
                embedding_dim=hidden_dim, #512
                dropout_prob=config['flow_matching']['dropout_prob'],
                config=config,
                dataset=dataset
            )
        else:
            model = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        baseline_type = config['baseline']['type']
        print(f"Creating {baseline_type.upper()} baseline model...")

        if baseline_type == 'vae':
            if conditional:
                from src.models.trajectory_vae import ConditionalTrajectoryVAE
                model = ConditionalTrajectoryVAE(config,dataset=dataset)
                print("Creating Conditional VAE baseline model...")
            else:
                from src.models.trajectory_vae import TrajectoryVAE
                model = TrajectoryVAE(config,dataset=dataset)
                print("Creating VAE baseline model...")
        elif baseline_type == 'gan':
            if conditional:
                from src.models.trajectory_gan import ConditionalTrajectoryGAN
                model = ConditionalTrajectoryGAN(config,dataset=dataset)
                print("Creating Conditional GAN baseline model...")
            else:
                from src.models.trajectory_gan import TrajectoryGAN
                model = TrajectoryGAN(config,dataset=dataset)
                print("Creating GAN baseline model...")
        elif baseline_type == 'markov':
            if conditional:
                from src.models.markov import (
                    ConditionalContinuousMarkovTrajectoryGenerator,
                )
                model = ConditionalContinuousMarkovTrajectoryGenerator(config=config, dataset=dataset)
                print("Creating Conditional Markov baseline model...")
            else:
                from src.models.markov import ContinuousMarkovTrajectoryGenerator
                model = ContinuousMarkovTrajectoryGenerator(config=config, dataset=dataset)
                print("Creating Markov baseline model...")
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    if config.get('training', {}).get('data_parallel', False):
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using DataParallel across {torch.cuda.device_count()} visible GPUs")
            model = torch.nn.DataParallel(model)
        else:
            print("data_parallel requested, but fewer than two CUDA devices are visible")

    write_run_manifest(
        os.path.join(run_dir, 'run_manifest.json'),
        args.config,
        config,
        device,
        dataset,
        validation_dataset,
    )

    # Create trainer
    trainer = FlowMatchingTrainer(
        config=config,
        model=model,
        dataset=dataset,
        save_dir=run_dir,
        device=device,
        validation_dataset=validation_dataset,
    )

    # Train model
    print("Starting training...")
    train_losses = trainer.train()

    # Plot training loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    if trainer.validation_losses and any(
        value is not None for value in trainer.validation_losses
    ):
        plt.plot(trainer.validation_losses, label='Validation')
        plt.plot(train_losses, label='Train')
        plt.legend()
    else:
        plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'training_loss.png'))
    plt.close()

    print(f"Training completed. Results saved to {run_dir}")

if __name__ == '__main__':
    main()
