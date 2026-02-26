import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pickle
import os
from typing import Optional, Tuple, Union
import jismesh.utils as ju


class ContinuousMarkovTrajectoryGenerator(nn.Module):
    """
    PyTorch-based Continuous Markov Model for trajectory generation using learned Gaussian Mixture Models
    """

    def __init__(self, config, dataset=None):
        super(ContinuousMarkovTrajectoryGenerator, self).__init__()
        self.config = config
        self.dataset = dataset
        self.model_params = config['baseline']['markov']

        # Model parameters
        self.n_components = self.model_params.get('n_components', 10)
        self.transition_components = self.model_params.get('transition_components', 5)

        # Trajectory parameters
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
            self.trajectory_length = config['data']['trajectory_length']
        else:
            self.M = config['data']['trajectory_length']
            self.trajectory_length = self.M

        self.input_dim = self.M * 2

        # Neural network for initial state distribution
        self.initial_net = nn.Sequential(
            nn.Linear(1, 128),  # Input is just a dummy for sampling
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.n_components)  # Mean and log_var for each component
        )

        # Neural networks for transition models at each step
        self.transition_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 128),  # Current state as input
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2 * self.transition_components)  # Mean and log_var for each component
            ) for _ in range(self.M - 1)
        ])

        # Component weights (learnable mixing coefficients)
        self.initial_weights = nn.Parameter(torch.randn(self.n_components))
        self.transition_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.transition_components))
            for _ in range(self.M - 1)
        ])

        self.is_trained = False
        self.device = None

    def forward(self, x=None, step=None):
        """
        Forward pass - used during training

        Args:
            x: Current state (for transitions) or None (for initial state)
            step: Which transition step (0 to M-2), None for initial state

        Returns:
            Distribution parameters
        """
        if step is None:
            # Initial state distribution
            dummy_input = torch.ones(1, 1).to(self.device)
            params = self.initial_net(dummy_input)
            means = params[:, :self.n_components].unsqueeze(0)
            log_vars = params[:, self.n_components:].unsqueeze(0)
            weights = F.softmax(self.initial_weights, dim=0)
            return means, log_vars, weights
        else:
            # Transition distribution
            params = self.transition_nets[step](x)
            batch_size = x.size(0)
            means = params[:, :self.transition_components].view(batch_size, self.transition_components)
            log_vars = params[:, self.transition_components:].view(batch_size, self.transition_components)
            weights = F.softmax(self.transition_weights[step], dim=0)
            return means, log_vars, weights

    def sample_from_gmm(self, means, log_vars, weights, n_samples=1):
        """
        Sample from a Gaussian Mixture Model

        Args:
            means: Component means [n_components] or [batch_size, n_components]
            log_vars: Component log variances [n_components] or [batch_size, n_components]
            weights: Component weights [n_components]
            n_samples: Number of samples

        Returns:
            Samples of shape [n_samples, feature_dim]
        """
        # Sample component indices
        component_indices = torch.multinomial(weights, n_samples, replacement=True)

        # Sample from selected components
        if means.dim() == 1:
            # Single set of components
            selected_means = means[component_indices]
            selected_vars = torch.exp(log_vars[component_indices])
            noise = torch.randn_like(selected_means)
            samples = selected_means + noise * torch.sqrt(selected_vars)
        else:
            # Batch of component sets
            batch_size = means.size(0)
            samples = torch.zeros(n_samples, batch_size, 2).to(self.device)

            for i in range(batch_size):
                comp_idx = torch.multinomial(weights, 1)[0]
                mean = means[i, comp_idx]
                var = torch.exp(log_vars[i, comp_idx])
                noise = torch.randn(2).to(self.device)
                samples[:, i, :] = mean + noise * torch.sqrt(var)

            if n_samples == 1:
                samples = samples.squeeze(0)

        return samples

    def compute_loss(self, trajectories):
        """
        Compute negative log-likelihood loss

        Args:
            trajectories: Batch of trajectories [batch_size, M, 2]

        Returns:
            Loss tensor
        """
        batch_size = trajectories.size(0)
        total_loss = 0.0

        # Loss for initial state
        initial_states = trajectories[:, 0, :]  # [batch_size, 2]
        means, log_vars, weights = self.forward()

        # Expand for batch processing
        means = means.expand(batch_size, -1)  # [batch_size, n_components]
        log_vars = log_vars.expand(batch_size, -1)  # [batch_size, n_components]

        # Compute likelihood for each component
        initial_ll = torch.zeros(batch_size, self.n_components).to(self.device)
        for k in range(self.n_components):
            var = torch.exp(log_vars[:, k])
            diff = initial_states - means[:, k].unsqueeze(-1)
            ll = -0.5 * (torch.sum(diff ** 2 / var.unsqueeze(-1), dim=-1) +
                         2 * torch.log(torch.sqrt(var)) + 2 * np.log(2 * np.pi))
            initial_ll[:, k] = ll

        # Weighted likelihood
        weighted_ll = initial_ll + torch.log(weights).unsqueeze(0)
        initial_loss = -torch.logsumexp(weighted_ll, dim=1).mean()
        total_loss += initial_loss

        # Loss for transitions
        for step in range(self.M - 1):
            current_states = trajectories[:, step, :]
            next_states = trajectories[:, step + 1, :]

            means, log_vars, weights = self.forward(current_states, step)

            # Compute transition likelihood
            trans_ll = torch.zeros(batch_size, self.transition_components).to(self.device)
            for k in range(self.transition_components):
                var = torch.exp(log_vars[:, k])
                diff = next_states - means[:, k].unsqueeze(-1)
                ll = -0.5 * (torch.sum(diff ** 2 / var.unsqueeze(-1), dim=-1) +
                             2 * torch.log(torch.sqrt(var)) + 2 * np.log(2 * np.pi))
                trans_ll[:, k] = ll

            weighted_ll = trans_ll + torch.log(weights).unsqueeze(0)
            trans_loss = -torch.logsumexp(weighted_ll, dim=1).mean()
            total_loss += trans_loss

        return total_loss

    def train_step(self, batch_data):
        """
        Training step for Markov model

        Args:
            batch_data: Batch of trajectory data

        Returns:
            loss: Training loss
        """
        if isinstance(batch_data, tuple):
            trajectories = batch_data[0]
        else:
            trajectories = batch_data

        # Ensure trajectories are in the right format
        if trajectories.dim() == 2:
            trajectories = trajectories.view(-1, self.M, 2)

        loss = self.compute_loss(trajectories)
        return loss

    def generate(self, n_samples, device='cpu'):
        """
        Generate trajectory samples

        Args:
            n_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            Generated trajectories of shape (n_samples, M*2)
        """
        self.device = device
        self.eval()

        with torch.no_grad():
            generated_trajectories = torch.zeros(n_samples, self.M, 2).to(device)

            # Sample initial states
            means, log_vars, weights = self.forward()
            initial_states = self.sample_from_gmm(
                means.squeeze(), log_vars.squeeze(), weights, n_samples
            )
            generated_trajectories[:, 0, :] = initial_states

            # Generate subsequent points using transition models
            for step in range(self.M - 1):
                current_states = generated_trajectories[:, step, :]
                means, log_vars, weights = self.forward(current_states, step)

                next_states = torch.zeros(n_samples, 2).to(device)
                for i in range(n_samples):
                    # Sample component
                    comp_idx = torch.multinomial(weights, 1)[0]
                    mean = means[i, comp_idx]
                    var = torch.exp(log_vars[i, comp_idx])
                    noise = torch.randn(2).to(device)
                    next_states[i] = mean + noise * torch.sqrt(var)

                generated_trajectories[:, step + 1, :] = next_states

        # Reshape to (n_samples, M*2)
        return generated_trajectories.view(n_samples, -1)


class ConditionalContinuousMarkovTrajectoryGenerator(nn.Module):
    """
    PyTorch-based Conditional Continuous Markov Model for trajectory generation with OD conditions
    """

    def __init__(self, config, dataset=None):
        super(ConditionalContinuousMarkovTrajectoryGenerator, self).__init__()
        self.config = config
        self.dataset = dataset
        self.model_params = config['baseline']['markov']

        # Model parameters
        self.n_components = self.model_params.get('n_components', 10)
        self.transition_components = self.model_params.get('transition_components', 5)

        # Trajectory parameters
        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
            self.trajectory_length = config['data']['trajectory_length']
        else:
            self.M = config['data']['trajectory_length']
            self.trajectory_length = self.M

        self.input_dim = self.M * 2

        # Condition parameters
        self.condition_dim = dataset.location_dim if dataset else 4
        self.condition_type = config['condition'].get('type', 'od')

        # OD finer support
        self.od_finer = config['data'].get('od_finer', False)
        if self.od_finer:
            self.od_finer_dim = 4  # [o_lat_mult, o_lon_mult, d_lat_mult, d_lon_mult]
            self.condition_dim += self.od_finer_dim

        # Neural network for conditional initial state distribution
        self.initial_net = nn.Sequential(
            nn.Linear(self.condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * self.n_components)  # Mean (2D) and log_var (2D) for each component
        )

        # Neural networks for conditional transition models
        self.transition_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 + self.condition_dim, 128),  # Current state + condition
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 4 * self.transition_components)  # Mean and log_var for each component
            ) for _ in range(self.M - 1)
        ])

        # Component weights (conditional)
        self.initial_weight_net = nn.Sequential(
            nn.Linear(self.condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_components)
        )

        self.transition_weight_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 + self.condition_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.transition_components)
            ) for _ in range(self.M - 1)
        ])

        self.is_trained = False
        self.device = None

    def forward(self, condition, x=None, step=None):
        """
        Forward pass with condition

        Args:
            condition: Condition tensor [batch_size, condition_dim]
            x: Current state (for transitions) [batch_size, 2]
            step: Which transition step (0 to M-2), None for initial state

        Returns:
            Distribution parameters
        """
        batch_size = condition.size(0)

        if step is None:
            # Initial state distribution
            params = self.initial_net(condition)  # [batch_size, 4 * n_components]
            means = params[:, :2 * self.n_components].view(batch_size, self.n_components, 2)
            log_vars = params[:, 2 * self.n_components:].view(batch_size, self.n_components, 2)
            weights = F.softmax(self.initial_weight_net(condition), dim=1)
            return means, log_vars, weights
        else:
            # Transition distribution
            input_tensor = torch.cat([x, condition], dim=1)
            params = self.transition_nets[step](input_tensor)
            means = params[:, :2 * self.transition_components].view(batch_size, self.transition_components, 2)
            log_vars = params[:, 2 * self.transition_components:].view(batch_size, self.transition_components, 2)
            weights = F.softmax(self.transition_weight_nets[step](input_tensor), dim=1)
            return means, log_vars, weights

    def sample_from_conditional_gmm(self, means, log_vars, weights):
        """
        Sample from conditional Gaussian Mixture Model

        Args:
            means: Component means [batch_size, n_components, 2]
            log_vars: Component log variances [batch_size, n_components, 2]
            weights: Component weights [batch_size, n_components]

        Returns:
            Samples of shape [batch_size, 2]
        """
        batch_size = means.size(0)
        samples = torch.zeros(batch_size, 2).to(self.device)

        for i in range(batch_size):
            # Sample component for this batch item
            comp_idx = torch.multinomial(weights[i], 1)[0]
            mean = means[i, comp_idx]  # [2]
            var = torch.exp(log_vars[i, comp_idx])  # [2]
            noise = torch.randn(2).to(self.device)
            samples[i] = mean + noise * torch.sqrt(var)

        return samples

    def compute_loss(self, trajectories, conditions):
        """
        Compute conditional negative log-likelihood loss

        Args:
            trajectories: Batch of trajectories [batch_size, M, 2]
            conditions: Batch of conditions [batch_size, condition_dim]

        Returns:
            Loss tensor
        """
        batch_size = trajectories.size(0)
        total_loss = 0.0

        # Loss for initial state
        initial_states = trajectories[:, 0, :]  # [batch_size, 2]
        means, log_vars, weights = self.forward(conditions)

        # Compute likelihood for each component
        initial_ll = torch.zeros(batch_size, self.n_components).to(self.device)
        for k in range(self.n_components):
            diff = initial_states.unsqueeze(1) - means[:, k:k + 1, :]  # [batch_size, 1, 2]
            var = torch.exp(log_vars[:, k, :])  # [batch_size, 2]
            ll = -0.5 * (torch.sum(diff.squeeze(1) ** 2 / var, dim=1) +
                         torch.sum(torch.log(var), dim=1) + 2 * np.log(2 * np.pi))
            initial_ll[:, k] = ll

        # Weighted likelihood
        weighted_ll = initial_ll + torch.log(weights + 1e-8)
        initial_loss = -torch.logsumexp(weighted_ll, dim=1).mean()
        total_loss += initial_loss

        # Loss for transitions
        for step in range(self.M - 1):
            current_states = trajectories[:, step, :]
            next_states = trajectories[:, step + 1, :]

            means, log_vars, weights = self.forward(conditions, current_states, step)

            # Compute transition likelihood
            trans_ll = torch.zeros(batch_size, self.transition_components).to(self.device)
            for k in range(self.transition_components):
                diff = next_states.unsqueeze(1) - means[:, k:k + 1, :]  # [batch_size, 1, 2]
                var = torch.exp(log_vars[:, k, :])  # [batch_size, 2]
                ll = -0.5 * (torch.sum(diff.squeeze(1) ** 2 / var, dim=1) +
                             torch.sum(torch.log(var), dim=1) + 2 * np.log(2 * np.pi))
                trans_ll[:, k] = ll

            weighted_ll = trans_ll + torch.log(weights + 1e-8)
            trans_loss = -torch.logsumexp(weighted_ll, dim=1).mean()
            total_loss += trans_loss

        return total_loss

    def train_step(self, batch_data):
        """
        Training step for conditional Markov model

        Args:
            batch_data: Tuple of (trajectories, conditions)

        Returns:
            loss: Training loss
        """
        if isinstance(batch_data, tuple):
            trajectories, conditions = batch_data
        else:
            raise ValueError("Conditional model requires (trajectories, conditions) tuple")

        # Ensure trajectories are in the right format
        if trajectories.dim() == 2:
            trajectories = trajectories.view(-1, self.M, 2)

        loss = self.compute_loss(trajectories, conditions)
        return loss

    def generate(self, n_samples, condition, device='cpu'):
        """
        Generate trajectory samples conditioned on given condition

        Args:
            n_samples: Number of samples to generate
            condition: Condition tensor [n_samples, condition_dim] or [1, condition_dim]
            device: Device to generate on

        Returns:
            Generated trajectories of shape (n_samples, M*2)
        """
        self.device = device
        self.eval()

        # Handle single condition for multiple samples
        if condition.size(0) == 1 and n_samples > 1:
            condition = condition.expand(n_samples, -1)
        elif condition.size(0) != n_samples:
            raise ValueError(f"Condition batch size {condition.size(0)} doesn't match n_samples {n_samples}")

        with torch.no_grad():
            generated_trajectories = torch.zeros(n_samples, self.M, 2).to(device)

            # Sample initial states
            means, log_vars, weights = self.forward(condition)
            initial_states = self.sample_from_conditional_gmm(means, log_vars, weights)
            generated_trajectories[:, 0, :] = initial_states

            # Generate subsequent points using transition models
            for step in range(self.M - 1):
                current_states = generated_trajectories[:, step, :]
                means, log_vars, weights = self.forward(condition, current_states, step)
                next_states = self.sample_from_conditional_gmm(means, log_vars, weights)
                generated_trajectories[:, step + 1, :] = next_states

        # Reshape to (n_samples, M*2)
        return generated_trajectories.view(n_samples, -1)

    def save_model(self, filepath):
        """Save the trained conditional model"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'M': self.M,
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
            'condition_type': self.condition_type,
            'od_finer': self.od_finer
        }, filepath)

    def load_model(self, filepath):
        """Load a trained conditional model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        self.M = checkpoint['M']
        self.input_dim = checkpoint['input_dim']
        self.condition_dim = checkpoint['condition_dim']
        self.condition_type = checkpoint['condition_type']
        self.od_finer = checkpoint.get('od_finer', False)