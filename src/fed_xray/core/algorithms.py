"""
Fed-XRay Advanced Federated Optimization Algorithms
====================================================
Implements:
1. FedAvg (McMahan et al., 2017)
2. FedProx (Li et al., 2020) - Proximal Regularization
3. SCAFFOLD (Karimireddy et al., 2020) - Control Variates
4. FedDyn (Acar et al., 2021) - Dynamic Risk Regularization
5. MOON (Li et al., 2021) - Model-Contrastive Learning
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


def compute_fedprox_loss(
    model: nn.Module, 
    global_weights: Dict[str, torch.Tensor], 
    mu: float
) -> torch.Tensor:
    """
    Compute FedProx proximal regularization penalty:
    L_prox = (mu / 2) * sum || w - w_global ||_2^2
    
    Args:
        model: Local PyTorch model
        global_weights: Global parameter tensors
        mu: Proximal regularization parameter
        
    Returns:
        Scalar penalty tensor
    """
    if mu <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
        
    proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in global_weights:
            global_param = global_weights[name].to(param.device)
            proximal_term += torch.sum((param - global_param) ** 2)
            
    return (mu / 2.0) * proximal_term


def compute_feddyn_loss(
    model: nn.Module,
    global_weights: Dict[str, torch.Tensor],
    prev_grads: Dict[str, torch.Tensor],
    alpha: float
) -> torch.Tensor:
    """
    Compute FedDyn dynamic regularization loss:
    L_dyn = - <grad_F(w_global), w> + (alpha / 2) * || w - w_global ||_2^2
    
    Args:
        model: Local PyTorch model
        global_weights: Global parameter tensors
        prev_grads: Local historical gradients
        alpha: Dynamic penalty parameter
        
    Returns:
        Scalar loss tensor
    """
    device = next(model.parameters()).device
    if alpha <= 0:
        return torch.tensor(0.0, device=device)
        
    linear_term = torch.tensor(0.0, device=device)
    quad_term = torch.tensor(0.0, device=device)
    
    for name, param in model.named_parameters():
        if name in global_weights:
            w_glob = global_weights[name].to(device)
            quad_term += torch.sum((param - w_glob) ** 2)
            
            if name in prev_grads:
                g = prev_grads[name].to(device)
                linear_term += torch.sum(g * param)
                
    return -linear_term + (alpha / 2.0) * quad_term


def compute_moon_contrastive_loss(
    local_rep: torch.Tensor,
    global_rep: torch.Tensor,
    prev_rep: Optional[torch.Tensor],
    temperature: float = 0.5,
    mu: float = 1.0
) -> torch.Tensor:
    """
    Compute MOON model-contrastive loss:
    L_con = - mu * log( exp(sim(z_loc, z_glob)/tau) / [exp(sim(z_loc, z_glob)/tau) + exp(sim(z_loc, z_prev)/tau)] )
    
    Args:
        local_rep: Latent representation from current model
        global_rep: Latent representation from global model
        prev_rep: Latent representation from previous local model round
        temperature: Temperature hyperparameter tau
        mu: Contrastive loss weight
        
    Returns:
        Scalar contrastive loss tensor
    """
    if mu <= 0:
        return torch.tensor(0.0, device=local_rep.device)
        
    cos = nn.CosineSimilarity(dim=-1)
    
    # Positive similarity: local with global
    sim_pos = cos(local_rep, global_rep) / temperature
    
    if prev_rep is not None:
        # Negative similarity: local with previous local
        sim_neg = cos(local_rep, prev_rep) / temperature
        logits = torch.stack([sim_pos, sim_neg], dim=1) # (batch, 2)
        labels = torch.zeros(local_rep.size(0), dtype=torch.long, device=local_rep.device)
        loss = F.cross_entropy(logits, labels)
    else:
        loss = -torch.mean(sim_pos)
        
    return mu * loss


class ScaffoldController:
    """
    SCAFFOLD Control Variate Coordinator.
    Maintains client control variates (c_k) and server control variate (c).
    """
    
    def __init__(self, model: nn.Module) -> None:
        self.server_controls: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
        }
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {}

    def get_client_controls(self, client_id: int, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Retrieve or initialize client control variates."""
        if client_id not in self.client_controls:
            self.client_controls[client_id] = {
                name: torch.zeros_like(param.data)
                for name, param in model.named_parameters()
            }
        return self.client_controls[client_id]

    def update_server_controls(
        self, 
        client_deltas: List[Dict[str, torch.Tensor]], 
        num_total_clients: int
    ) -> None:
        """Update global server control variates: c = c + (1/K) * sum(delta_c_k)."""
        for key in self.server_controls.keys():
            delta_sum = torch.zeros_like(self.server_controls[key])
            for delta in client_deltas:
                if key in delta:
                    delta_sum += delta[key].to(self.server_controls[key].device)
            self.server_controls[key] += delta_sum / max(num_total_clients, 1)
