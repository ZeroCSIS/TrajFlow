import torch
import torch.nn.functional as F

def flow_matching_loss(pred_v, target_v):
    """
    Compute the flow matching loss

    Args:
        pred_v: Predicted velocity
        target_v: Target velocity

    Returns:
        loss: Mean squared error between predicted and target velocity
    """
    return F.mse_loss(pred_v, target_v)

def conditional_flow_matching_loss(pred_v, target_v, condition):
    """
    Compute the conditional flow matching loss

    Args:
        pred_v: Predicted velocity
        target_v: Target velocity
        condition: Conditioning vector

    Returns:
        loss: Mean squared error between predicted and target velocity
    """
    # Basic implementation - can be extended for more complex conditioning
    return F.mse_loss(pred_v, target_v)

def wasserstein_distance_loss(generated, real):
    """
    Approximate Wasserstein distance as a loss function

    Args:
        generated: Generated samples
        real: Real samples

    Returns:
        loss: Approximation of Wasserstein distance
    """
    # Simple approximation using sorted values
    # More accurate approximation would require OT algorithms
    generated_sorted, _ = torch.sort(generated, dim=0)
    real_sorted, _ = torch.sort(real, dim=0)

    return F.mse_loss(generated_sorted, real_sorted)