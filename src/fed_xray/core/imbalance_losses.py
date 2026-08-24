"""
Fed-XRay: Advanced Imbalance Loss Functions for Federated Medical AI
====================================================================
Implements:
1. Dynamic Adaptive Focal Loss (DAFL)
2. Bayesian Balanced Softmax Loss (Logit Adjustment)
3. Class-Balanced Loss (L_CB, Effective Number of Samples)
4. Label-Distribution-Aware Margin (LDAM) Loss
5. Missing-Class Prototype Margin Regularization (L_repel)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Union


class DynamicAdaptiveFocalLoss(nn.Module):
    """
    Dynamic Adaptive Focal Loss (DAFL).
    
    L_DAFL(p_t, c, t) = -alpha_c(t) * (1 - p_t)^gamma_c(t) * log(p_t)
    alpha_c(t) = ((1 - beta) / (1 - beta^N_c)) * (max_j N_j / N_c)^delta(t)
    delta(t) = delta_0 * (1 - t / T)
    gamma_c(t) = gamma_base + mu * (1 - Rec_c^(t-1))
    """
    
    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor],
        current_round: int = 1,
        total_rounds: int = 10,
        beta: float = 0.999,
        delta_0: float = 0.5,
        gamma_base: float = 1.0,
        mu: float = 1.0,
        recalls: Optional[Dict[int, float]] = None
    ) -> None:
        super().__init__()
        self.class_counts = torch.tensor(class_counts, dtype=torch.float32)
        self.num_classes = len(class_counts)
        self.current_round = max(1, current_round)
        self.total_rounds = max(1, total_rounds)
        self.beta = beta
        self.delta_0 = delta_0
        self.gamma_base = gamma_base
        self.mu = mu
        self.recalls = recalls or {c: 0.5 for c in range(self.num_classes)}
        
        # Effective number of samples weighting
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        effective_num = torch.clamp(effective_num, min=1e-8)
        weights_cb = (1.0 - self.beta) / effective_num
        weights_cb = weights_cb / weights_cb.sum() * self.num_classes
        
        # Dynamic exponent delta(t)
        t_factor = max(0.0, 1.0 - (self.current_round / self.total_rounds))
        delta_t = self.delta_0 * t_factor
        
        max_count = torch.max(self.class_counts)
        ratio = max_count / torch.clamp(self.class_counts, min=1.0)
        imbalance_term = torch.pow(ratio, delta_t)
        
        self.alpha_c = weights_cb * imbalance_term
        self.alpha_c = self.alpha_c / self.alpha_c.sum() * self.num_classes
        
        # Dynamic focusing parameter gamma_c(t)
        gamma_list = []
        for c in range(self.num_classes):
            rec = self.recalls.get(c, 0.5)
            g_c = self.gamma_base + self.mu * (1.0 - rec)
            gamma_list.append(g_c)
        self.gamma_c = torch.tensor(gamma_list, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_c.to(logits.device)
        gamma = self.gamma_c.to(logits.device)
        
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        target_alpha = alpha[targets]
        target_gamma = gamma[targets]
        
        focal_weight = target_alpha * torch.pow(1.0 - pt, target_gamma)
        loss = focal_weight * ce_loss
        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    """
    Bayesian Balanced Softmax (Logit Adjustment) Loss.
    L_BSM(x, y) = -log ( exp(z_y + log pi_y) / sum_j exp(z_j + log pi_j) )
    """
    def __init__(self, class_counts: Union[List[int], torch.Tensor]) -> None:
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        freq = counts / counts.sum()
        self.log_prior = torch.log(torch.clamp(freq, min=1e-8))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prior = self.log_prior.to(logits.device)
        adjusted_logits = logits + log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets)


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples (L_CB).
    L_CB(z, y) = ((1 - beta) / (1 - beta^n_y)) * L_CE(z, y)
    """
    def __init__(self, class_counts: Union[List[int], torch.Tensor], beta: float = 0.999) -> None:
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-8)
        self.weights = weights / weights.sum() * len(class_counts)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weights)


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.
    Margin delta_j = C / (n_j^(1/4))
    """
    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor],
        max_m: float = 0.5,
        s: float = 30.0
    ) -> None:
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.clamp(counts, min=1.0)))
        m_list = m_list * (max_m / torch.max(m_list))
        self.m_list = m_list
        self.s = s

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        m_list = self.m_list.to(logits.device)
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        
        batch_m = m_list[targets]
        batch_m = batch_m.view(-1, 1)
        
        output = torch.where(index, logits - batch_m, logits)
        return F.cross_entropy(self.s * output, targets)


class PrototypeRepelLoss(nn.Module):
    """
    Missing-Class Prototype Margin Regularization.
    L_repel = sum_{c != y_i} max(0, margin - ||f(x_i) - p_c||_2)
    """
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        global_prototypes: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        
        for i in range(features.size(0)):
            feat_i = features[i]
            y_i = targets[i].item()
            
            for c, proto in global_prototypes.items():
                if c != y_i:
                    proto = proto.to(features.device)
                    dist = torch.norm(feat_i - proto, p=2)
                    loss += torch.clamp(self.margin - dist, min=0.0)
                    count += 1
                    
        return loss / max(count, 1)
