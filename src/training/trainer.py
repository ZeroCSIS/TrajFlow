import os
import time
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models.wrappers import WrappedModel, ConditionedVelocityModelWrapper
from src.utils.visualization import (
    visualize_velocity_field,
    visualize_density_comparison,
    visualize_trajectories
)
from src.data.transforms import para2point
from src.eval.inference import FlowMatchingInference
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FlowMatchingTrainer:
    def __init__(self, config, model, dataset, save_dir, device):
        """
        Flow Matching trainer

        Args:
            config: Configuration dictionary
            model: Model to train
            dataset: Dataset for training
            save_dir: Directory to save checkpoints and visualizations
            device: Device to train on
        """
        self.config = config
        self.model = model.to(device)
        self.dataset = dataset
        self.save_dir = save_dir
        self.device = device

        # Determine training type
        self.training_type = self._get_training_type(config)

        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']
        self.n_dim = self.M * 2
        self.traj_length = config['data']['trajectory_length']

        # Setup dataloader
        batch_size = config['data']['batch_size']
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=200
        )

        # Add early stopping parameters
        self.early_stop_patience = config.get('training', {}).get('early_stop_patience', 500)
        self.early_stop_delta = config.get('training', {}).get('early_stop_delta', 0.0001)
        self.early_stop_counter = 0
        self.best_loss = float('inf')

        # Setup path if needed
        self.setup_flow_path()

        # Pre-compute DDPM schedules if using DDPM
        if config.get('ddpm', {}).get('enabled', False):
            num_timesteps = config['ddpm']['num_diffusion_timesteps']
            beta_start = config['ddpm']['beta_start']
            beta_end = config['ddpm']['beta_end']
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
            self.alphas = 1 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _get_training_type(self, config):
        """Determine training type from config"""
        if config.get('baseline', {}).get('enabled', False):
            return 'baseline'
        elif config['flow_matching']['enabled']:
            return 'flow_matching'
        elif config['ddpm']['enabled']:
            return 'ddpm'
        else:
            raise ValueError("No training method specified in config")

    def setup_flow_path(self):
        """Set up flow path according to configuration"""
        flow_type = self.config['flow_matching']['flow_type']

        if flow_type == "standard":
            from flow_matching.path.scheduler import CondOTScheduler
            from flow_matching.path import AffineProbPath
            scheduler = CondOTScheduler()
            self.path = AffineProbPath(scheduler=scheduler)
            self.manifold = None
        elif flow_type == "discrete":
            from flow_matching.path.scheduler import StandardScheduler
            from flow_matching.path import DiscreteProbPath
            scheduler = StandardScheduler()
            self.path = DiscreteProbPath(scheduler=scheduler)
            self.manifold = None
        elif flow_type == "riemannian":
            self.setup_riemannian_flow()
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

    def setup_riemannian_flow(self):
        """Setup Riemannian flow components"""
        from flow_matching.path.scheduler import CondOTScheduler
        from flow_matching.path import GeodesicProbPath
        from flow_matching.utils.manifolds import FlatTorus, Euclidean
        from src.models.wrappers import ProjectToTangent

        # Determine manifold based on dataset type
        dataset_type = self.config['data']['dataset_type']
        if dataset_type == "naive_circle":
            # For naive_circle, use Euclidean manifold
            self.manifold = Euclidean()
        else:
            # For other datasets, use FlatTorus
            self.manifold = FlatTorus()

        # Wrap velocity field with projection to tangent space
        self.model = ProjectToTangent(self.model, self.manifold)

        # Set up geodesic path on manifold
        scheduler = CondOTScheduler()
        self.path = GeodesicProbPath(scheduler=scheduler, manifold=self.manifold)

    def train_step(self, batch):
        """Training step - choose between flow matching and DDPM"""
        if self.training_type == 'baseline':
            return self._train_step_baseline(batch)
        elif self.config['flow_matching']['enabled']:
            return self._train_step_flow_matching(batch)
        elif self.config['ddpm']['enabled']:
            return self._train_step_ddpm(batch)
        else:
            raise ValueError("Either flow_matching or ddpm must be enabled")

    def _train_step_ddpm(self, batch):
        """DDPM training step using the same neural network"""
        if self.config['condition']['enabled']:
            x_1, condition = batch
            x_1 = x_1.to(self.device)
            condition = condition.to(self.device)
        else:
            x_1 = batch.to(self.device)
            condition = None

        """Perform a single training step"""
        self.optimizer.zero_grad()
        # reshape x_1 to (batch_size, -1)
        x_1 = x_1.reshape(x_1.shape[0], -1)  # Flatten to [batch_size, n_dim]
        batch_size = x_1.shape[0]

        # Sample random timesteps for DDPM
        num_timesteps = self.config['ddpm']['num_diffusion_timesteps']
        t = torch.randint(num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_1)

        # Forward diffusion process: q(x_t | x_0)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        x_t = torch.sqrt(alpha_cumprod_t) * x_1 + torch.sqrt(1 - alpha_cumprod_t) * noise

        cfg_scale = self.config['condition'].get('cfg_scale', 1.0)
        dropout_prob = self.config['flow_matching'].get('dropout_prob', 0.1)
        if dropout_prob > 0:
            # Create wrapper with cfg_scale=0 for training (forces random dropout)
            wrapped_model = ConditionedVelocityModelWrapper(self.model, condition, cfg_scale)
            predicted_noise = wrapped_model(x_t, t.float() / num_timesteps)
        else:
            predicted_noise = self.model(x_t, t.float() / num_timesteps, condition)

        # Reshape predicted noise to match the shape of noise
        predicted_noise = predicted_noise.reshape_as(noise)
        # MSE loss between predicted and actual noise
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass and update
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_step_flow_matching(self, batch):

        # Move existing train_step logic here
        if self.config['condition']['enabled']:
            x_1, condition = batch
            x_1 = x_1.to(self.device)
            condition = condition.to(self.device)
        else:
            x_1 = batch.to(self.device)
            condition = None

        """Perform a single training step"""
        self.optimizer.zero_grad()
        batch_size = x_1.shape[0]

        # Generate random initial points
        x_0 = torch.randn_like(x_1).to(self.device)

        # Preprocess data if needed for manifold
        if self.manifold is not None:
            x_0, x_1 = self.preprocess_data_for_manifold(x_0, x_1)

        # Sample time
        t = torch.rand(batch_size).to(self.device)

        # Sample path and compute loss
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # Forward pass with conditional handling
        if condition is not None and self.config.get('condition', {}).get('enabled', False):
            # Apply classifier-free guidance dropout during training
            dropout_prob = self.config['flow_matching'].get('dropout_prob', 0.1)
            if dropout_prob > 0 and torch.rand(1).item() < dropout_prob:
                # With probability dropout_prob, don't pass condition (classifier-free training)
                # set condition to zeros
                zero_condition = torch.zeros_like(condition)
                pred_v = self.model(path_sample.x_t, path_sample.t, zero_condition)
            else:
                # Normal conditional forward pass
                pred_v = self.model(path_sample.x_t, path_sample.t, condition)

        else:
            # Unconditional case
            pred_v = self.model(path_sample.x_t, path_sample.t)

        # Compute loss
        # Reshape pred_v to match path_sample.dx_t
        if self.config['data']['od_finer'] == True:
            pred_v = pred_v.reshape(pred_v.size(0), -1)  # Flatten to [batch_size, n_dim]
            path_sample.dx_t = path_sample.dx_t.reshape(path_sample.dx_t.size(0), -1)  # Ensure same shape
        loss = torch.pow(pred_v - path_sample.dx_t, 2).mean()

        # Backward pass and update
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_step_baseline(self, batch_data):
        """Training step for baseline models with proper conditional support"""
        if self.config['condition']['enabled']:
            x_1, condition = batch_data
            x_1 = x_1.to(self.device)
            condition = condition.to(self.device)
            batch_size = x_1.size(0)
            x_1 = x_1.view(batch_size, -1)

            # Pass both trajectory and condition to the conditional model
            loss_dict = self.model.train_step((x_1, condition))
        else:
            x_1 = batch_data.to(self.device)
            batch_size = x_1.size(0)
            x_1 = x_1.view(batch_size, -1)

            # Pass only trajectory to unconditional model
            loss_dict = self.model.train_step(x_1)

        # Extract loss value from dictionary
        if isinstance(loss_dict, dict):
            if 'loss' in loss_dict:
                return loss_dict['loss']
            elif 'total_loss' in loss_dict:
                return loss_dict['total_loss']
            elif 'g_loss' in loss_dict:
                return loss_dict['g_loss']
            else:
                return next(iter(loss_dict.values()))
        elif loss_dict is None:
            return 0.0
        else:
            return loss_dict

    def preprocess_data_for_manifold(self, x_0, x_1):
        """Preprocess data to fit on chosen manifold if needed"""
        # For Euclidean manifold, wrap points using expmap
        if self.manifold is not None:
            def wrap(manifold, samples):
                center = torch.zeros_like(samples)
                return manifold.expmap(center, samples)

            x_0 = wrap(self.manifold, x_0)
            x_1 = wrap(self.manifold, x_1)

        return x_0, x_1

    def train(self):
        """Train the model"""
        num_epochs = self.config['training']['num_epochs']
        print_every = self.config['training']['print_every']
        save_every = self.config['training']['save_every']
        is_conditional = self.config['condition']['enabled']

        total_iterations = num_epochs * len(self.dataloader)

        if print_every is None:
            print_every = max(1, total_iterations // 10)

        iteration = 0
        best_loss = float('inf')
        train_losses = []

        # Create visualization directory
        os.makedirs(self.save_dir, exist_ok=True)
        vis_dir = os.path.join(os.path.dirname(self.save_dir), 'visualizations', os.path.basename(self.save_dir))
        os.makedirs(vis_dir, exist_ok=True)
        self.model_dir = os.path.join(os.path.dirname(self.save_dir), 'models', os.path.basename(self.save_dir))
        os.makedirs(self.model_dir, exist_ok=True)

        for epoch in range(num_epochs):
            epoch_loss = 0
            start_time = time.time()

            for batch_idx, batch_data in enumerate(self.dataloader):
                # Handle both conditional and unconditional cases
                if is_conditional:
                    # Assuming batch_data is a tuple of (x_1, condition)
                    x_1, condition = batch_data
                    x_1 = x_1.to(self.device)
                    condition = condition.to(self.device)
                    loss = self.train_step([x_1, condition])
                else:
                    # Unconditional case - batch_data is just x_1
                    x_1 = batch_data.to(self.device)
                    loss = self.train_step(x_1)

                epoch_loss += loss

                if iteration % print_every == 0:
                    log_message = f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(self.dataloader)}, Iteration {iteration}/{total_iterations}, Loss: {loss:.6f}"
                    print(log_message)

                    # Save loss log to file
                    log_file = os.path.join(self.save_dir, 'training_log.txt')
                    with open(log_file, 'a') as f:
                        f.write(log_message + '\n')
                iteration += 1

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            train_losses.append(avg_epoch_loss)

            # scheduler step for learning rate adjustment
            self.scheduler.step(avg_epoch_loss)

            # Early stopping check
            if avg_epoch_loss < self.best_loss - self.early_stop_delta:
                self.best_loss = avg_epoch_loss
                self.early_stop_counter = 0
                self.save_checkpoint(os.path.join(self.model_dir, 'best_model.pt'))
                print(f"New best model saved with loss: {self.best_loss:.6f}")
            else:
                self.early_stop_counter += 1
                print(f"No improvement: {self.early_stop_counter}/{self.early_stop_patience}")
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Save checkpoint if better than previous best
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_checkpoint(os.path.join(self.model_dir, 'best_model.pt'))
                print(f"New best model saved with loss: {best_loss:.6f}")

            # Regular checkpoint saving
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pt'))

                # Generate visualizations
                if self.config['training']['viz_bool']:
                    self.inference_and_visualize(epoch, vis_dir,1, is_conditional)

        # Final checkpoint
        self.save_checkpoint(os.path.join(self.model_dir, 'final_model.pt'))
        print(f"Training completed. Final loss: {train_losses[-1]:.6f}")

        return train_losses
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['config']

    def inference_and_visualize(self, epoch, vis_dir, viz_num = 1, is_conditional=False):
        """Generate visualizations during training"""
        # Create epoch-specific directory
        epoch_dir = os.path.join(vis_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Sample some data points for visualization
        batch_data = next(iter(self.dataloader))

        if is_conditional:
            # Handle conditional data (x_1, condition)
            x_1_sample, condition_sample = batch_data
            x_1_sample = x_1_sample[:viz_num].to(self.device)
            condition_sample = condition_sample[:viz_num].to(self.device)
        else:
            # Handle unconditional data
            x_1_sample = batch_data[:viz_num].to(self.device)
            condition_sample = None

        # Get trajectory samples by solving ODE
        inference = FlowMatchingInference(config=self.config,
                                          model=self.model,
                                          dataset=self.dataset,
                                          save_dir=epoch_dir,
                                          device=self.device)

        # Sample with condition if in conditional mode
        if is_conditional:
            sol = inference.sample(n_samples=viz_num, n_steps=10, method='em', condition=condition_sample)
        else:
            sol = inference.sample(n_samples=viz_num, n_steps=10, method='em')

        # Convert to numpy for visualization
        sol_np = [s.detach().cpu().numpy() for s in sol][-1]
        ground_truth_np = x_1_sample.detach().cpu().numpy()
        if is_conditional:
            condition_sample_np = condition_sample.detach().cpu().numpy()

        if self.config['data']['parametrized']:
            from src.data.transforms import para2point_batch
            # For parametrized data, use para2point
            traj_length = self.config['data']['trajectory_length']
            if self.config['data'].get('od_finer', False):
                # Extract trajectory segments and OD finer parameters
                sol_traj_dim = self.M * 2
                gt_traj_dim = self.M * 2

                # For generated trajectories
                sol_traj = sol_np[:, :sol_traj_dim]  # Extract trajectory points
                sol_od_finer = sol_np[:, sol_traj_dim:]  # Extract OD finer parameters

                # For ground truth
                gt_traj = ground_truth_np[:, :gt_traj_dim]  # Extract trajectory points
                gt_od_finer = ground_truth_np[:, gt_traj_dim:]  # Extract OD finer parameters

                # Convert trajectory points to full trajectories
                sol_np_points = para2point_batch(sol_traj, N_new=traj_length,
                                                 method=self.config['data'].get('para_method', 'rdp_k'))
                ground_truth_np_points = para2point_batch(gt_traj, N_new=traj_length,
                                                          method=self.config['data'].get('para_method', 'rdp_k'))

                # Store for denormalization
                sol_np = sol_np_points.reshape(sol_np_points.shape[0], -1)
                ground_truth_np = ground_truth_np_points.reshape(ground_truth_np_points.shape[0], -1)

                # Save OD finer parameters for denormalization
                od_finer_params_sol = sol_od_finer
                od_finer_params_gt = gt_od_finer
            else:
                # Standard processing without OD finer parameters
                sol_np = para2point_batch(sol_np, N_new=traj_length,
                                          method=self.config['data'].get('para_method', 'rdp_k'))
                ground_truth_np = para2point_batch(ground_truth_np, N_new=traj_length,
                                                   method=self.config['data'].get('para_method', 'rdp_k'))
                od_finer_params_sol = None
                od_finer_params_gt = None
        else:
            from src.data.transforms import para2point_batch
            # For parametrized data, use para2point
            traj_length = self.config['data']['trajectory_length']
            if self.config['data'].get('od_finer', False):
                # Extract trajectory segments and OD finer parameters
                sol_traj_dim = self.M * 2
                gt_traj_dim = self.M * 2

                # For generated trajectories
                sol_traj = sol_np[:, :sol_traj_dim]  # Extract trajectory points
                sol_od_finer = sol_np[:, sol_traj_dim:]  # Extract OD finer parameters

                # For ground truth
                gt_traj = ground_truth_np[:, :gt_traj_dim]  # Extract trajectory points
                gt_od_finer = ground_truth_np[:, gt_traj_dim:]  # Extract OD finer parameters

                # Store for denormalization
                sol_np = sol_traj.reshape(sol_traj.shape[0], -1)
                ground_truth_np = gt_traj.reshape(gt_traj.shape[0], -1)

                # Save OD finer parameters for denormalization
                od_finer_params_sol = sol_od_finer
                od_finer_params_gt = gt_od_finer
            else:
                od_finer_params_sol = None
                od_finer_params_gt = None

        # Create before-denormalization visualizations
        before_dir = os.path.join(epoch_dir, 'before_denormalization')
        os.makedirs(before_dir, exist_ok=True)
        visualize_trajectories(sol_np, ground_truth_np, self.traj_length,
                               self.config['data']['parametrized'],
                               before_dir,
                               traj_length=self.config['data']['trajectory_length'])
        visualize_density_comparison(sol_np, ground_truth_np, self.traj_length,
                                     before_dir,
                                     traj_length=self.config['data']['trajectory_length'])

        # Denormalize if needed
        if (is_conditional and self.config['data'].get('od_finer', False) and
                self.config['visualization'].get('norm1by12origialvis', False)):
            print("Denormalizing data for visualization...")
            # Denormalize the trajectories
            sol_np_denorm = inference.denormalize_trajectories([sol_np],
                                                               condition_sample_np,
                                                               self.dataset,
                                                               od_finer_params_sol)[0]
            ground_truth_np_denorm = \
            inference.denormalize_trajectories([ground_truth_np],
                                               condition_sample_np,
                                               self.dataset,
                                               od_finer_params_gt)[0]

            # Create after-denormalization visualizations
            after_dir = os.path.join(epoch_dir, 'after_denormalization')
            os.makedirs(after_dir, exist_ok=True)
            visualize_trajectories(sol_np_denorm, ground_truth_np_denorm, self.traj_length, self.config['data']['parametrized'],
                                   after_dir)
            visualize_density_comparison(sol_np_denorm, ground_truth_np_denorm, self.traj_length, after_dir)


        # Optionally visualize velocity field if 2D
        if self.n_dim == 2:
            visualize_velocity_field(self.model, self.device, epoch_dir)
