# src/models/trajectory_gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.networks import WideAndDeep, SimpleMLPEmb, MLP


class ConditionalTrajectoryGenerator(nn.Module):
    def __init__(self, input_dim,latent_dim,hidden_dim,condition_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Generator network with condition
        generator_input_dim = self.latent_dim + self.condition_dim
        self.model = nn.Sequential(
            nn.Linear(generator_input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, z, condition):
        # Concatenate noise and condition
        z_cond = torch.cat([z, condition], dim=1)
        return self.model(z_cond)


class ConditionalTrajectoryDiscriminator(nn.Module):
    def __init__(self, input_dim,latent_dim,hidden_dim,condition_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Discriminator network with condition
        discriminator_input_dim = self.input_dim + self.condition_dim
        self.model = nn.Sequential(
            nn.Linear(discriminator_input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 1),
        )

    def _get_condition_dim(self, config):
        """Get condition dimension from config"""
        if config['condition']['condition_type'] == 'full':
            return 8  # Adjust based on your actual condition dimension
        elif config['condition']['condition_type'] == 'od':
            return 4  # Origin-destination only
        else:
            return 0

    def forward(self, x, condition):
        # Concatenate trajectory and condition
        x_cond = torch.cat([x, condition], dim=1)
        return self.model(x_cond)


class ConditionalTrajectoryGAN(nn.Module):
    def __init__(self, config,dataset):
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
        if self.config['data']['od_finer' ]== True:
            self.input_dim = self.input_dim + 4

        # Determine if conditional from config
        self.conditional = config.get('condition', {}).get('enabled', False)

        # get embedding of condition
        if self.conditional:
            embedding_dim = config['baseline']['model']['architecture'].get('embedding_dim', self.hidden_dim)
            self.condition_embedding = self._get_condition_embedding(
                config,
                embedding_dim=embedding_dim,
                hidden_dim=self.hidden_dim,
                dataset=dataset  # Pass dataset if needed for location_dim
            )
            self.condition_dim = embedding_dim
        else:
            self.condition_dim = 0

        # Initialize models
        self.generator = ConditionalTrajectoryGenerator(self.input_dim,self.latent_dim,self.hidden_dim,self.condition_dim)
        self.discriminator = ConditionalTrajectoryDiscriminator(self.input_dim,self.latent_dim,self.hidden_dim,self.condition_dim)

        # Optimizers
        g_lr = config['baseline']['training']['gan_config']['g_lr']
        d_lr = config['baseline']['training']['gan_config']['d_lr']

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

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

    def train_step(self, batch_data):
        # Handle conditional vs non-conditional data
        if isinstance(batch_data, tuple):
            real_trajectories, condition = batch_data
        else:
            raise ValueError("Conditional GAN requires both trajectory and condition data")

        batch_size = real_trajectories.size(0)
        device = real_trajectories.device

        # Ground truth labels
        valid = torch.ones(batch_size, 1, device=device, requires_grad=False)
        fake = torch.zeros(batch_size, 1, device=device, requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Sample noise
        z = torch.randn(batch_size, self.latent_dim, device=device)

        # Embed condition for generator
        condition_embedded = self.condition_embedding(condition)

        # Generate trajectories
        gen_trajectories = self.generator(z, condition_embedded)

        # Discriminator judges generated trajectories (for generator loss)
        validity_gen = self.discriminator(gen_trajectories, condition_embedded)

        # Generator loss (wants discriminator to think generated are real)
        g_loss = F.binary_cross_entropy_with_logits(validity_gen, valid)

        g_loss.backward()
        self.g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        # Re-embed condition for discriminator (separate computation graph)
        condition_embedded_d = self.condition_embedding(condition)

        # Discriminator on real trajectories
        real_validity = self.discriminator(real_trajectories, condition_embedded_d)
        real_loss = F.binary_cross_entropy_with_logits(real_validity, valid)

        # Discriminator on fake trajectories (DETACHED from generator)
        fake_validity = self.discriminator(gen_trajectories.detach(), condition_embedded_d)
        fake_loss = F.binary_cross_entropy_with_logits(fake_validity, fake)

        # Combined discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.d_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'total_loss': (g_loss.item() + d_loss.item()) / 2
        }
    def generate(self, n_samples, condition, device='cuda'):
        """Generate conditional samples for inference"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            # Embed the condition using the condition embedding model
            condition_embedded = self.condition_embedding(condition)
            samples = self.generator(z, condition_embedded)
            return samples

# Keep original classes for unconditional case
class TrajectoryGenerator(nn.Module):
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

        # Generator network
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, z):
        return self.model(z)


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.input_dim = self.M * 2
        self.hidden_dim = config['baseline']['model']['architecture']['hidden_dim']

        # Discriminator network
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)


class TrajectoryGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Get dimensions from config
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        self.latent_dim = config['baseline']['model']['architecture']['latent_dim']

        # Initialize models
        self.generator = TrajectoryGenerator(config)
        self.discriminator = TrajectoryDiscriminator(config)

        # Optimizers
        g_lr = config['baseline']['training']['gan_config']['g_lr']
        d_lr = config['baseline']['training']['gan_config']['d_lr']

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

    def train_step(self, batch_data):
        # Handle conditional vs non-conditional
        if isinstance(batch_data, tuple):
            real_trajectories = batch_data[0]  # Get trajectories, ignore condition
        else:
            real_trajectories = batch_data

        batch_size = real_trajectories.size(0)
        device = real_trajectories.device

        # Ground truth labels
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Sample noise
        z = torch.randn(batch_size, self.latent_dim, device=device)

        # Generate a batch of trajectories
        gen_trajectories = self.generator(z)

        # Discriminator output for generated trajectories
        validity = self.discriminator(gen_trajectories)

        # Generator loss
        g_loss = F.binary_cross_entropy_with_logits(validity, valid)

        g_loss.backward()
        self.g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()

        # Discriminator output for real and generated trajectories
        real_validity = self.discriminator(real_trajectories)
        fake_validity = self.discriminator(gen_trajectories.detach())

        # Discriminator loss
        real_loss = F.binary_cross_entropy_with_logits(real_validity, valid)
        fake_loss = F.binary_cross_entropy_with_logits(fake_validity, fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.d_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }

    def generate(self, n_samples, device='cuda'):
        """Generate samples for inference"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            samples = self.generator(z)
            return samples