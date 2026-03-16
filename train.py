import os
import argparse
import torch
import yaml
from datetime import datetime

from src.models.networks import MLP, CNN, TransformerVelocity, BiLSTMVelocity, ConditionalVelocityModel, TrajUnet
from src.data.dataset import FlowMatchingDataset
from src.training.trainer import FlowMatchingTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Flow Matching Training')
    parser.add_argument('--config', type=str, default='./src/config/config_chengdu.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Directory to save checkpoints and results')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config['project']['output_dir']:
        run_dir = os.path.join(config['project']['output_dir'], f"run_{timestamp}")
    else:
        run_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Set device from config.training
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Create dataset
    dataset_params = config['data']
    dataset = FlowMatchingDataset(config, mode='train')

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
                from src.models.markov import ConditionalContinuousMarkovTrajectoryGenerator
                model = ConditionalContinuousMarkovTrajectoryGenerator(config=config, dataset=dataset)
                print("Creating Conditional Markov baseline model...")
            else:
                from src.models.markov import ContinuousMarkovTrajectoryGenerator
                model = ContinuousMarkovTrajectoryGenerator(config=config, dataset=dataset)
                print("Creating Markov baseline model...")
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    # Create trainer
    trainer = FlowMatchingTrainer(
        config=config,
        model=model,
        dataset=dataset,
        save_dir=run_dir,
        device=device
    )

    # Train model
    print("Starting training...")
    train_losses = trainer.train()

    # Plot training loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
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
