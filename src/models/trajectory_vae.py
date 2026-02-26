import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.networks import WideAndDeep, SimpleMLPEmb, MLP


class ConditionalTrajectoryVAEEncoder(nn.Module):
    def __init__(self, config, condition_dim):
        super().__init__()

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.input_dim = self.M * 2
        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']
        self.hidden_dim = config['baseline']['model']['architecture']['hidden_dim']
        self.condition_dim = condition_dim

        if config['data']['od_finer'] == True:
            self.input_dim = self.input_dim + 4

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x, condition_embedded):
        x_cond = torch.cat([x, condition_embedded], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConditionalTrajectoryVAEDecoder(nn.Module):
    def __init__(self, config, condition_dim):
        super().__init__()

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.input_dim = self.M * 2
        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']
        self.hidden_dim = config['baseline']['model']['architecture']['hidden_dim']
        self.condition_dim = condition_dim

        if config['data']['od_finer'] == True:
            self.input_dim = self.input_dim + 4

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, z, condition_embedded):
        z_cond = torch.cat([z, condition_embedded], dim=1)
        return self.decoder(z_cond)


class ConditionalTrajectoryVAE(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.input_dim = self.M * 2
        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']
        self.hidden_dim = config['baseline']['model']['architecture']['hidden_dim']
        self.beta = config['baseline']['training']['vae_config']['beta']

        if config['data']['od_finer'] == True:
            self.input_dim = self.input_dim + 4

        # Determine if conditional from config
        self.conditional = config.get('condition', {}).get('enabled', False)

        # Get embedding of condition
        if self.conditional:
            embedding_dim = config['baseline']['model']['architecture'].get('embedding_dim', self.hidden_dim)
            self.condition_embedding = self._get_condition_embedding(
                config,
                embedding_dim=embedding_dim,
                hidden_dim=self.hidden_dim,
                dataset=dataset
            )
            self.condition_dim = embedding_dim
        else:
            self.condition_dim = 0

        # Initialize encoder and decoder
        self.encoder = ConditionalTrajectoryVAEEncoder(config, self.condition_dim)
        self.decoder = ConditionalTrajectoryVAEDecoder(config, self.condition_dim)

        # Optimizer
        lr = config['training']['learning_rate']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _get_condition_embedding(self, config, embedding_dim, hidden_dim, dataset):
        """Get the appropriate condition embedding module based on config"""
        embedding_type = config['condition']['embedding_type']
        location_emb_dim = dataset.location_dim

        if embedding_type == 'wide_and_deep':
            return WideAndDeep(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                               location_emb_dim=location_emb_dim, config=config)
        elif embedding_type == 'simple_mlp':
            return SimpleMLPEmb(input_dim=location_emb_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        else:
            # Default to SimpleMLPEmb
            return SimpleMLPEmb(input_dim=location_emb_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition):
        condition_embedded = self.condition_embedding(condition)
        mu, logvar = self.encoder(x, condition_embedded)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, condition_embedded)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= x.size(0) * x.size(1)  # Normalize by batch size and feature dim

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kld_loss

        return total_loss, recon_loss, kld_loss

    def train_step(self, batch_data):
        # Handle conditional data
        if isinstance(batch_data, tuple):
            real_trajectories, condition = batch_data
        else:
            raise ValueError("Conditional VAE requires both trajectory and condition data")

        self.optimizer.zero_grad()

        # Forward pass
        recon_trajectories, mu, logvar = self.forward(real_trajectories, condition)

        # Compute loss
        total_loss, recon_loss, kld_loss = self.loss_function(
            recon_trajectories, real_trajectories, mu, logvar
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item()
        }

    def generate(self, n_samples, condition, device='cuda'):
        """Generate conditional samples for inference"""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim, device=device)
            # Embed condition
            condition_embedded = self.condition_embedding(condition)
            # Generate samples
            samples = self.decoder(z, condition_embedded)
            return samples


class TrajectoryVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.input_dim = self.M * 2
        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']
        self.hidden_dim = config['baseline']['model']['architecture']['hidden_dim']

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TrajectoryVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.input_dim = self.M * 2
        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']
        self.hidden_dim = config['baseline']['model']['architecture']['hidden_dim']

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, z):
        return self.decoder(z)


class TrajectoryVAE(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']
        self.beta = config['baseline']['training']['vae_config']['beta']

        # Initialize encoder and decoder
        self.encoder = TrajectoryVAEEncoder(config)
        self.decoder = TrajectoryVAEDecoder(config)

        # Optimizer
        lr = config['training']['learning_rate']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= x.size(0) * x.size(1)  # Normalize by batch size and feature dim

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kld_loss

        return total_loss, recon_loss, kld_loss

    def train_step(self, batch_data):
        # Handle conditional vs non-conditional
        if isinstance(batch_data, tuple):
            real_trajectories = batch_data[0]  # Get trajectories, ignore condition
        else:
            real_trajectories = batch_data

        self.optimizer.zero_grad()

        # Forward pass
        recon_trajectories, mu, logvar = self.forward(real_trajectories)

        # Compute loss
        total_loss, recon_loss, kld_loss = self.loss_function(
            recon_trajectories, real_trajectories, mu, logvar
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item()
        }

    def generate(self, n_samples, device='cuda'):
        """Generate samples for inference"""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim, device=device)
            # Generate samples
            samples = self.decoder(z)
            return samples