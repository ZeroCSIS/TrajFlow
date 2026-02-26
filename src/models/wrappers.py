import torch
from torch import nn
from flow_matching.utils import ModelWrapper

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t,**extras)

class ProjectToTangent(nn.Module):
    """Wraps a velocity field model to project onto tangent space of a manifold"""
    def __init__(self, vecfield, manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x, t):
        x = self.manifold.projx(x)
        v = self.vecfield(x, t)
        v = self.manifold.proju(x, v)
        return v


class ConditionedVelocityModelWrapper(nn.Module):
    """Wrapper for conditional sampling with classifier-free guidance"""

    def __init__(self, velocity_model, condition, cfg_scale=1.0):
        super().__init__()
        self.velocity_model = velocity_model
        self.condition = condition
        self.cfg_scale = cfg_scale

    def forward(self, x, t, **kwargs):
        """Forward pass with classifier-free guidance"""
        # For cfg_scale = 1.0, just use regular conditioned model
        # For training stage, keep it 1.0
        if self.cfg_scale == 1.0:
            return self.velocity_model(x, t, c=self.condition, **kwargs)

        # For cfg_scale > 1.0, compute both conditional and unconditional
        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # Duplicate inputs for conditional and unconditional passes
        x_doubled = torch.cat([x, x], dim=0)
        t_doubled = torch.cat([t, t], dim=0)
        c_doubled = torch.cat([self.condition, self.condition], dim=0)

        # Create force_drop_ids (0=keep condition, 1=drop condition)
        force_drop_ids = torch.cat([
            torch.zeros(batch_size, dtype=torch.long, device=x.device),
            torch.ones(batch_size, dtype=torch.long, device=x.device)
        ], dim=0)

        # Single forward pass with doubled batch
        v_doubled = self.velocity_model(x_doubled, t_doubled, c=c_doubled, force_drop_ids=force_drop_ids, **kwargs)

        # Split results
        v_cond, v_null = v_doubled.chunk(2, dim=0)

        # Apply CFG: v = (1-w)*v_null + w*v_cond
        guided_velocity = (1 - self.cfg_scale) * v_null + self.cfg_scale * v_cond

        return guided_velocity