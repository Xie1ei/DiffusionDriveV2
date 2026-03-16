from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score_para
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

# input1: encoding_feature : [B, N, D_model]  agent_feature + map_feature + static_feature
# input2 :Anchor : [N_anchor, T, 2]
# 需要设计一个轨迹预测头， 基于输入的encding_feature、 Anchor信息， 预测多条轨迹 [B, N_anchor, n_mode, T, 2], 以及每条轨迹对应的概率 [B, N_anchor, N_mode]
# 需要设计模拟损失函数，GRPO-like损失函数
# GRPO 中的规则Reward 可以 `from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer` 中通过PDMScorer计算



class RegPlanningDecoder(nn.Module):
    """Deterministic trajectory head with a scoring network."""

    def __init__(
        self,
        d_model: int,
        num_modes: int,
        num_poses: int,
        hidden_dim: int = 256,
        il_weight: float = 1.0,
        grpo_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.num_poses = num_poses
        self.il_weight = il_weight
        self.grpo_weight = grpo_weight

        self.anchor_encoder = nn.Sequential(
            nn.Linear(num_poses * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.backbone = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        out_dim = num_modes * num_poses * 2
        self.delta_head = nn.Linear(hidden_dim, out_dim)
        self.score_head = nn.Sequential(
            nn.Linear(num_poses * 2 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        encoding_feature: torch.Tensor,
        anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            traj: [B, N_anchor, M, T, 2]
            score: [B, N_anchor, M]
        """
        bsz = encoding_feature.shape[0]
        num_anchor = anchor.shape[0]

        anchor_flat = anchor.reshape(num_anchor, self.num_poses * 2)
        anchor_feat = self.anchor_encoder(anchor_flat)[None, :, :].expand(bsz, num_anchor, -1)
        attn_out, _ = self.cross_attn(anchor_feat, encoding_feature, encoding_feature)
        feat = self.backbone(torch.cat([anchor_feat, attn_out], dim=-1))

        delta = self.delta_head(feat).view(bsz, num_anchor, self.num_modes, self.num_poses, 2)
        anchor_b = anchor[None, :, None, :, :].expand(bsz, num_anchor, self.num_modes, self.num_poses, 2)
        traj = anchor_b + delta

        traj_flat = traj.view(bsz, num_anchor, self.num_modes, self.num_poses * 2)
        feat_exp = feat[:, :, None, :].expand(bsz, num_anchor, self.num_modes, feat.shape[-1])
        score_in = torch.cat([traj_flat, feat_exp], dim=-1)
        score = self.score_head(score_in).squeeze(-1)
        return traj, score

    def compute_losses(
        self,
        traj: torch.Tensor,
        score: torch.Tensor,
        gt_traj: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        metric_cache: Optional[Sequence[MetricCache]] = None,
        future_sampling: Optional[TrajectorySampling] = None,
        simulator: Optional[PDMSimulator] = None,
        scorer: Optional[PDMScorer] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        if gt_traj is not None:
            losses["il_loss"] = self._soft_reg_loss(traj, gt_traj) * self.il_weight

        if reward is None and metric_cache is not None:
            if future_sampling is None or simulator is None or scorer is None:
                raise ValueError("future_sampling/simulator/scorer are required when metric_cache is provided.")
            _, reward = self.compute_pdm_reward(traj, metric_cache, future_sampling, simulator, scorer)

        if reward is not None:
            grpo_loss = self._grpo_loss(score, reward)
            losses["grpo_loss"] = grpo_loss * self.grpo_weight

        if losses:
            losses["loss"] = torch.stack(list(losses.values())).sum()
        return losses

    def compute_pdm_reward(
        self,
        traj: torch.Tensor,
        metric_cache: Sequence[MetricCache],
        future_sampling: TrajectorySampling,
        simulator: PDMSimulator,
        scorer: PDMScorer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PDM score as GRPO reward.
        traj: [B, N_anchor, M, T, 2]
        returns reward, advantage: [B, N_anchor, M]
        """
        if traj.shape[-2] != future_sampling.num_poses:
            raise ValueError("Trajectory length must match future_sampling.num_poses.")
        if len(metric_cache) != traj.shape[0]:
            raise ValueError("metric_cache length must match batch size.")

        bsz, n_anchor, n_mode, t_steps, _ = traj.shape
        rewards = torch.zeros((bsz, n_anchor, n_mode), device=traj.device, dtype=traj.dtype)
        no_collision = torch.zeros_like(rewards)
        drivable_area = torch.zeros_like(rewards)
        progress = torch.zeros_like(rewards)
        comfort = torch.zeros_like(rewards)
        ttc = torch.zeros_like(rewards)

        traj_np = traj.detach().cpu().numpy()
        for b in range(bsz):
            flat_xy = traj_np[b].reshape(n_anchor * n_mode, t_steps, 2)
            flat_xyh = self._xy_to_xyh(flat_xy)
            traj_list = [Trajectory(flat_xyh[i], future_sampling) for i in range(flat_xyh.shape[0])]
            results, _ = pdm_score_para(metric_cache[b], traj_list, future_sampling, simulator, scorer)
            scores = np.array([r.score for r in results], dtype=np.float32).reshape(n_anchor, n_mode)
            nc = np.array([r.no_at_fault_collisions for r in results], dtype=np.float32).reshape(n_anchor, n_mode)
            da = np.array([r.drivable_area_compliance for r in results], dtype=np.float32).reshape(n_anchor, n_mode)
            pr = np.array([r.ego_progress for r in results], dtype=np.float32).reshape(n_anchor, n_mode)
            cm = np.array([r.comfort for r in results], dtype=np.float32).reshape(n_anchor, n_mode)
            tc = np.array([r.time_to_collision_within_bound for r in results], dtype=np.float32).reshape(
                n_anchor, n_mode
            )
            rewards[b] = torch.from_numpy(scores).to(device=traj.device, dtype=traj.dtype)
            no_collision[b] = torch.from_numpy(nc).to(device=traj.device, dtype=traj.dtype)
            drivable_area[b] = torch.from_numpy(da).to(device=traj.device, dtype=traj.dtype)
            progress[b] = torch.from_numpy(pr).to(device=traj.device, dtype=traj.dtype)
            comfort[b] = torch.from_numpy(cm).to(device=traj.device, dtype=traj.dtype)
            ttc[b] = torch.from_numpy(tc).to(device=traj.device, dtype=traj.dtype)

        # normalize per anchor over modes
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True).clamp_min(1e-4)
        advantage = (rewards - mean) / std
        gate = (no_collision > 0.5) & (drivable_area > 0.5)
        weight = (
            (0.5 + 0.5 * no_collision)
            * (0.5 + 0.5 * drivable_area)
            * (0.5 + 0.5 * progress)
            * (0.5 + 0.5 * comfort)
            * (0.5 + 0.5 * ttc)
        )
        advantage = advantage * weight * gate.float()
        return rewards, advantage

    @staticmethod
    def _xy_to_xyh(xy: np.ndarray) -> np.ndarray:
        dx = np.diff(xy[..., 0], axis=1, prepend=xy[..., :1, 0])
        dy = np.diff(xy[..., 1], axis=1, prepend=xy[..., :1, 1])
        heading = np.arctan2(dy, dx)
        return np.concatenate([xy, heading[..., None]], axis=-1)

    @staticmethod
    def _soft_reg_loss(traj: torch.Tensor, gt_traj: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        gt_rep = gt_traj[:, None, None, :, :].expand_as(traj)
        l1 = (traj - gt_rep).abs().mean(dim=(-1, -2))
        weight = torch.softmax(-l1 / tau, dim=-1)
        return (weight * l1).mean()

    @staticmethod
    def _grpo_loss(score: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        if reward.shape != score.shape:
            raise ValueError("reward shape must match score shape")
        reward = reward.detach()
        return -(score * reward).mean()
