from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig
# from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
# from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrivev2.pluto.modules.agent_encoder import AgentEncoder
from navsim.agents.diffusiondrivev2.pluto.modules.map_encoder import MapEncoder
from navsim.agents.diffusiondrivev2.pluto.modules.static_objects_encoder import StaticObjectsEncoder
from navsim.agents.diffusiondrivev2.pluto.layers.transformer import TransformerEncoderLayer
from navsim.agents.diffusiondrivev2.pluto.layers.fourier_embedding import FourierEmbedding
from navsim.agents.diffusiondrivev2.pluto.layers.mlp_layer import MLPLayer
from typing import Any, List, Dict, Optional, Union, Tuple
import math
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.evaluate.pdm_score import pdm_score, pdm_score_para
import itertools, os
import lzma
import pickle
import concurrent.futures as cf, os #, cloudpickle
import threading
import multiprocessing as mp
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex as WIdx,
)
import matplotlib as mpl
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.agents.diffusiondrivev2.query_traj_head import AnchorQueryTrajHead
def _pairwise_subscores(scorer):
    """
    从已调用过 score_proposals 的 PDMScorer 中
    拆出 7 个子指标和最终分数，全部 shape=(G,)，顺序与 proposal id 对齐
    返回 dict[str, np.ndarray]
    """
    mm   = scorer._multi_metrics                # (3, N)
    wm   = scorer._weighted_metrics.copy()      # <<< 一定要 copy !
    prod = mm.prod(axis=0)                      # (N,)

    wcoef  = scorer._config.weighted_metrics_array
    thresh = scorer._config.progress_distance_threshold
    prog_raw = scorer._progress_raw             # (N,)

    # ---------- progress 归一化（与 _pairwise_scores 完全一致） ----------
    raw_prog    = prog_raw * prod
    raw_prog_gt = raw_prog[0]
    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)
    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    # ---------- 加权指标 ----------
    wscore = (wm * wcoef[:, None]).sum(axis=0) / wcoef.sum()

    return {
        "no_collision"  : mm[MultiMetricIndex.NO_COLLISION,        1:].copy(),
        "drivable_area" : mm[MultiMetricIndex.DRIVABLE_AREA,       1:].copy(), # 乘法版
        "progress"      : wm[WeightedMetricIndex.PROGRESS,         1:].copy(),
        "ttc"           : wm[WeightedMetricIndex.TTC,              1:].copy(),
        "comfort"       : wm[WeightedMetricIndex.COMFORTABLE,      1:].copy(),
        "dir_weighted"  : wm[WeightedMetricIndex.DRIVING_DIRECTION,1:].copy(),
        "final"         : prod[1:] * wscore[1:],                               # 总分
    }
def _pairwise_scores(scorer) -> np.ndarray:
    """
    使用 scorer 在 batch 模式下缓存的中间结果，
    重新计算“GT (索引0) vs 每条候选”的得分。
    返回 shape = (N-1,)  float32。
    """
    # --- 取中间量 ---------------------------------------------------
    mm   = scorer._multi_metrics            # (M_mul, N)
    wm   = scorer._weighted_metrics.copy()  # (M_wgt, N)  (复制以便我们改进程)
    prog_raw = scorer._progress_raw         # (N,)
    weight_coef = scorer._config.weighted_metrics_array  # (M_wgt,)

    N = mm.shape[1]                         # proposals = 1(GT) + G
    assert N >= 2, "Need at least GT + 1 proposal"

    # --- 计算乘法指标乘积 ------------------------------------------
    multi_prod = mm.prod(axis=0)            # (N,)

    # --- 重新归一化 progress，每条候选只与 GT 对标 ------------------
    raw_prog    = prog_raw * multi_prod     # (N,)
    raw_prog_gt = raw_prog[0]

    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])           # (G,)
    thresh      = scorer._config.progress_distance_threshold

    # 若 max_pair > thresh → 按比例归一；否则看 collision 情况
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(multi_prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)                                         # (G,)

    # 把 progress 行（WeightedMetricIndex.PROGRESS）替换成新的
    wm[WIdx.PROGRESS, 1:] = norm_prog

    # --- 计算 weighted_metric_scores（与 _aggregate_scores 同式） ----
    weighted_scores = (wm[:, 1:] * weight_coef[:, None]).sum(axis=0)
    weighted_scores /= weight_coef.sum()                         # (G,)

    # --- 最终得分 = 乘法指标 × 加权指标 -----------------------------
    final_scores = multi_prod[1:] * weighted_scores              # (G,)

    return final_scores.astype(np.float32)                       # (G,)

def _pdm_worker(args):
    cache, traj_np = args
    # if isinstance(cache, str): 
    with lzma.open(cache, "rb") as f:
        metric_cache = pickle.load(f)
    # else:
    #     metric_cache = cache
    results = pdm_score_para(
        metric_cache=metric_cache,
        model_trajectory=traj_np,                # (G, T, C)
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,                    # 全局对象，见 initializer
        scorer=SCORER,
    )
    scores = _pairwise_scores(SCORER)
    subscores  = _pairwise_subscores(SCORER)
    return scores.astype(np.float32), metric_cache, subscores  # (G,)

def _init_pool(sim_cfg, scorer_cfg):
    global SIMULATOR, SCORER
    SIMULATOR = instantiate(sim_cfg)
    SCORER    = instantiate(scorer_cfg)

    
class V2TransfuserModel_TS(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        # self._query_splits = [
        #     1,
        #     config.num_bounding_boxes,
        # ]

        self._config = config

        # Vector Encoder --> pluto-like
        self.pos_emb = FourierEmbedding(3, config.tf_d_model, 64)
        self.agent_encoder = AgentEncoder(
            state_channel = config.state_channel,
            history_channel = config.history_channel,
            dim = config.tf_d_model,
            hist_steps = config.history_steps, # 21
            use_ego_history=True,
            drop_path = 0.2,
            state_attn_encoder = True,
            state_dropout=0.75
        )

        self.map_encoder = MapEncoder(
            dim = config.tf_d_model,
            polygon_channel = config.polygon_channel,
            use_lane_boundary = config.use_lane_boundary
        )
        
        self.static_object_encoder = StaticObjectsEncoder(dim = config.tf_d_model)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim = config.tf_d_model, num_heads=config.tf_num_head, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, config.drop_path, config.tf_num_layers)]
        )

        self.norm = nn.LayerNorm(config.tf_d_model)
        # self._ego_query = nn.Parameter(torch.randn(1, 1, config.tf_d_model) * 0.02)
        # self._agent_queries = nn.Parameter(
        #     torch.randn(1, config.num_bounding_boxes, config.tf_d_model) * 0.02
        # )
        self._query_cross_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )

        # TODO: Add agent predictor
        # self._agent_predictor = AgentHead(
        #     config.tf_d_model,
        #     config.tf_d_model * 2
        # )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor] = None,
        eta: float = 0.0,
        metric_cache=None,
        cal_pdm: bool = True,
        token=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Vector-based forward pass.

        The model consumes vector encodings (agent/map/static) and predicts a trajectory from the fused
        `encoding_feature` with shape [B, N, d_model].
        """
        # --- build token positions for FourierEmbedding (x, y, heading) ---
        
        agent_pos: torch.Tensor = features["agent"]["position"][:, :, self._config.history_steps - 2]
        agent_heading: torch.Tensor = features["agent"]["heading"][:, :, self._config.history_steps - 2]
        agent_mask: torch.Tensor = features["agent"]["valid_mask"][:, :, : self._config.history_steps - 2]

        polygon_center: torch.Tensor = features["map"]["polygon_center"]
        polygon_mask: torch.Tensor = features["map"]["valid_mask"]

        N_agent = agent_pos.shape[1]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        # --- encoders ---
        agent_encoding = self.agent_encoder(features)
        map_encoding = self.map_encoder(features)
        static_encoding, static_pos, static_key_padding = self.static_object_encoder(features)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        encoding_feature = torch.cat([agent_encoding, map_encoding, static_encoding], dim=1)  # [B, N, D]

        # --- add pos embedding + transformer encoder ---
        pos = torch.cat([pos, static_pos], dim=1)
        encoding_feature = encoding_feature + self.pos_emb(pos)
        for blk in self.encoder_blocks:
            encoding_feature = blk(
                encoding_feature, key_padding_mask=key_padding_mask, return_attn_weights=False
            )
        encoding_feature = self.norm(encoding_feature)

        batch_size = encoding_feature.shape[0]

        agent_feature = encoding_feature[:, 1:N_agent, ...]
        # TODO: Add agent prediction head
        # agent_pre = self._agent_predictor(agent_feature)
        # output.update(agent_pre)
        
        if targets is None:
            targets = {"trajectory" : features["agent"]["target"][:, 0 , 4:41:5, :]}


        output: Dict[str, torch.Tensor] = {}
        pred = self._trajectory_head(
            encoding_feature,
            key_padding_mask,
            targets=targets,
            metric_cache=metric_cache,
            cal_pdm=cal_pdm,
        )
        output.update(pred)

        # agents = self._agent_head(agents_query)
        # output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        # num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        # self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 3),
        )

        # self._mlp_label = nn.Sequential(
        #     nn.Linear(self._d_model, 1),
        # )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., 0:2] = agent_states[..., 0:2].tanh() * 32
        agent_states[..., 2] = agent_states[..., 2].tanh() * np.pi

        # agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states} #, "agent_labels": agent_labels}



class TrajectoryHead(nn.Module):
    """Anchor-query trajectory head with the original PDM advantage shaping."""

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        plan_anchor_path: str,
        config: TransfuserConfig,
    ):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.num_groups = config.num_groups

        from pathlib import Path

        plan_path = Path(plan_anchor_path)
        candidates: List[Path] = []
        if plan_path.is_absolute():
            candidates.append(plan_path)
        else:
            base_path = os.environ.get("NAVSIM_DEVKIT_ROOT")
            if base_path:
                base = Path(base_path).resolve()
                candidates.append(base.parent / plan_anchor_path)
                candidates.append(base / plan_anchor_path)

            # Project root: .../DiffusionDriveV2 (navsim/agents/diffusiondrivev2/this_file.py)
            project_root = Path(__file__).resolve().parents[3]
            candidates.append(project_root / plan_anchor_path)
            candidates.append(Path(__file__).resolve().parent / plan_anchor_path)

        full_path = next((p for p in candidates if p.exists()), None)
        if full_path is None:
            raise FileNotFoundError(
                f"Cannot find plan anchors '{plan_anchor_path}'. Tried: {[str(p) for p in candidates]}"
            )
        plan_anchor = np.load(str(full_path))
        if plan_anchor.ndim != 3 or plan_anchor.shape[-1] != 2:
            raise ValueError(f"Expected plan anchors with shape [K, T, 2], got {plan_anchor.shape}.")
        if plan_anchor.shape[1] != num_poses:
            raise ValueError(
                f"Anchor num_poses ({plan_anchor.shape[1]}) must match config trajectory_sampling.num_poses ({num_poses})."
            )
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # [N_anchor, T, 2]
        self.num_anchors = int(plan_anchor.shape[0])
        self.ego_fut_mode = self.num_anchors
        self.query_traj_head = AnchorQueryTrajHead(
            embed_dim=d_model,
            num_anchors=self.num_anchors,
            num_modes=self.num_groups,
            timestamp=num_poses,
            nhead=config.tf_num_head,
            dropout=config.tf_dropout,
        )

        base_path = os.environ.get("NAVSIM_DEVKIT_ROOT")
        if not base_path:
            base_path = str(Path(__file__).resolve().parents[3])
        pdm_cfg_path = Path(base_path) / "planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
        if not pdm_cfg_path.exists():
            pdm_cfg_path = Path(__file__).resolve().parents[3] / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
        pdm_cfg = OmegaConf.load(str(pdm_cfg_path))
        self.simulator_cfg = pdm_cfg.simulator
        self.scorer_cfg = pdm_cfg.scorer

        self._pdm_pool = None
        self.metric_caches = {}
        self.simulator: Optional[PDMSimulator] = None
        self.scorer: Optional[PDMScorer] = None
        self._pdm_available = False
        self._pdm_init_error: Optional[str] = None
        try:
            self._pdm_pool = cf.ProcessPoolExecutor(
                max_workers=16,
                mp_context=mp.get_context("spawn"),
                initializer=_init_pool,
                initargs=(self.simulator_cfg, self.scorer_cfg),
            )
            self.simulator = instantiate(self.simulator_cfg)
            self.scorer = instantiate(self.scorer_cfg)
            self._pdm_available = True
        except Exception as e:
            self._pdm_init_error = str(e)

    def _can_use_pdm(self, metric_cache) -> bool:
        return self._pdm_available and metric_cache is not None

    def _fallback_loss_dict(
        self,
        trajectory: torch.Tensor,
        all_trajectories: torch.Tensor,
        trajectory_probabilities: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]],
        reason: str,
    ) -> Dict[str, torch.Tensor]:
        device = trajectory.device
        if targets is not None and "trajectory" in targets:
            loss = F.l1_loss(trajectory, targets["trajectory"])
        else:
            loss = trajectory.new_tensor(0.0)
        out = {
            "loss": loss,
            "reward": torch.zeros((), device=device, dtype=trajectory.dtype),
            "sub_rewards": {"pdm_enabled": 0.0},
            "trajectory": trajectory,
            "all_trajectories": all_trajectories,
            "trajectory_probabilities": trajectory_probabilities,
            "pdm_disabled_reason": reason,
        }
        return out

    def forward(
        self,
        encoding_feature,
        encoding_key_padding_mask,
        targets=None,
        metric_cache=None,
        cal_pdm=True,
    ) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.forward_train_rl(
                encoding_feature,
                encoding_key_padding_mask,
                targets,
                metric_cache,
                cal_pdm=cal_pdm,
            )
        return self.forward_test_rl(
            encoding_feature,
            encoding_key_padding_mask,
            targets,
            metric_cache,
        )

    def _predict_candidates(
        self,
        encoding_feature: torch.Tensor,
        encoding_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_out = self.query_traj_head(
            encoding_feature,
            self.plan_anchor,
            scene_key_padding_mask=encoding_key_padding_mask,
        )
        traj_xy = head_out["traj_pred"]
        bs = traj_xy.shape[0]
        traj_xyh = self.bezier_xyyaw(traj_xy.reshape(bs, -1, self._num_poses, 2))
        traj_xyh = traj_xyh.reshape(bs, self.num_anchors, self.num_groups, self._num_poses, 3)
        logits = head_out["logits"]
        return traj_xy, traj_xyh, logits

    def _select_best_trajectory(
        self,
        traj_xyh: torch.Tensor,
        score_flat: torch.Tensor,
    ) -> torch.Tensor:
        bs = traj_xyh.shape[0]
        traj_flat = traj_xyh.reshape(bs, -1, self._num_poses, 3)
        best_idx = score_flat.argmax(dim=-1)
        batch_idx = torch.arange(bs, device=traj_xyh.device)
        return traj_flat[batch_idx, best_idx]

    def _compute_imitation_loss(
        self,
        traj_xy: torch.Tensor,
        logits: torch.Tensor,
        gt_traj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _, _, steps, _ = traj_xy.shape
        gt_xy = gt_traj[..., :2]
        gt_expanded = gt_xy.unsqueeze(1).unsqueeze(2)
        dist = torch.norm(traj_xy - gt_expanded, dim=-1).mean(dim=-1)
        dist_flat = dist.reshape(bs, -1)
        best_idx = torch.argmin(dist_flat, dim=1)
        batch_idx = torch.arange(bs, device=traj_xy.device)
        best_traj = traj_xy.reshape(bs, -1, steps, 2)[batch_idx, best_idx]
        loss_reg = F.smooth_l1_loss(best_traj, gt_xy)
        loss_cls = F.cross_entropy(logits.reshape(bs, -1), best_idx)
        return loss_reg + loss_cls, best_idx

    def _compute_grpo_loss(
        self,
        logits: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        bs = logits.shape[0]
        log_probs = F.log_softmax(logits.reshape(bs, -1), dim=-1)
        adv_detached = advantages.detach()
        return -(log_probs * adv_detached).mean()

    def _compute_pdm_outputs(
        self,
        traj_xyh: torch.Tensor,
        metric_cache,
        include_gt: bool,
        targets: Optional[Dict[str, torch.Tensor]],
    ):
        bs = traj_xyh.shape[0]
        traj_to_score = traj_xyh.reshape(bs, -1, self._num_poses, 3)
        if include_gt:
            if targets is None or "trajectory" not in targets:
                raise ValueError("targets['trajectory'] is required when include_gt=True")
            target_traj = targets["trajectory"].unsqueeze(1)
            traj_to_score = torch.cat((traj_to_score, target_traj), dim=1)

        reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(traj_to_score, metric_cache)

        batched_sub = None
        if sub_rewards_group:
            keys = sub_rewards_group[0].keys()
            batched_sub = {
                k: torch.as_tensor(
                    np.vstack([d[k] for d in sub_rewards_group]),
                    device=reward_group.device,
                    dtype=reward_group.dtype,
                )
                for k in keys
            }

        reward_gt = None
        if include_gt:
            reward_gt = reward_group[:, -1:].unsqueeze(-1)
            reward_group = reward_group[:, :-1]
            if batched_sub is not None:
                batched_sub = {k: v[:, :-1] for k, v in batched_sub.items()}

        reward_group = reward_group.reshape(bs, self.num_anchors, self.num_groups)
        if batched_sub is not None:
            batched_sub = {
                k: v.reshape(bs, self.num_anchors, self.num_groups) for k, v in batched_sub.items()
            }
        return reward_group, reward_gt, batched_sub, metric_cache

    def _compute_probabilities(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = logits.shape[0]
        prob_flat = F.softmax(logits.reshape(bs, -1), dim=-1)
        return prob_flat.reshape(bs, self.num_anchors, self.num_groups), prob_flat

    def _compute_advantage_from_pdm(
        self,
        reward_group: torch.Tensor,
        reward_gt: torch.Tensor,
        batched_sub: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        reward_group = reward_group.permute(0, 2, 1)
        reward_gt = reward_gt.expand(-1, self.num_groups, -1)
        mean_grouped_rewards = reward_group.mean(dim=1)
        std_grouped_rewards = reward_group.std(dim=1)
        advantages = (reward_group - mean_grouped_rewards.unsqueeze(1)) / (
            std_grouped_rewards.unsqueeze(1) + 1e-4
        )

        mask_positive = reward_group > (reward_gt - 1e-6)
        advantages = advantages.clamp(min=0) * mask_positive.float()

        sub_rewards_mean = {"pdm_enabled": 1.0}
        if batched_sub is not None:
            for k, v_full in batched_sub.items():
                v = v_full.permute(0, 2, 1)
                if k == "no_collision" or k == "drivable_area":
                    zero_mask = v != 1
                    advantages = torch.where(zero_mask, torch.full_like(advantages, -1.0), advantages)
                sub_rewards_mean[k] = v.mean().item()

        pos_cnt = mask_positive.sum(dim=(1, 2), keepdim=True)
        pos_sum = (reward_group * mask_positive.float()).sum(dim=(1, 2), keepdim=True)
        mean_pos = pos_sum / pos_cnt.clamp(min=1)
        mean_all = reward_group.mean(dim=(1, 2), keepdim=True)
        batch_reward = torch.where(pos_cnt > 0, mean_pos, mean_all)
        reward = batch_reward.squeeze(-1).mean()

        advantages = advantages.permute(0, 2, 1).reshape(reward_group.shape[0], -1).detach()
        return advantages, reward, sub_rewards_mean

    def get_pdm_score_para(self, trajectory, metric_cache_path):
        B, G = trajectory.shape[:2]
        traj_np = trajectory.detach().cpu().numpy()
        futures = [
            self._pdm_pool.submit(
                _pdm_worker,
                (metric_cache_path[b], traj_np[b]),
            )
            for b in range(B)
        ]
        scores_np = np.vstack([f.result()[0] for f in futures])
        metric_cache = [f.result()[1] for f in futures]
        sub_scores = [f.result()[2] for f in futures]
        return torch.from_numpy(scores_np).to(trajectory.device), metric_cache, sub_scores

    def forward_train_rl(
        self,
        encoding_feature,
        encoding_key_padding_mask,
        targets,
        metric_cache,
        cal_pdm,
    ) -> Dict[str, torch.Tensor]:
        traj_xy, traj_xyh, logits = self._predict_candidates(
            encoding_feature,
            encoding_key_padding_mask,
        )
        bs = traj_xy.shape[0]
        trajectory_probabilities, prob_flat = self._compute_probabilities(logits)
        trajectory = self._select_best_trajectory(traj_xyh, prob_flat)
        il_loss, _ = self._compute_imitation_loss(traj_xy, logits, targets["trajectory"])
        if (not cal_pdm) or (not self._can_use_pdm(metric_cache)):
            reason = self._pdm_init_error or "metric_cache_missing_or_cal_pdm_disabled"
            out = self._fallback_loss_dict(
                trajectory,
                traj_xyh,
                trajectory_probabilities,
                targets,
                reason,
            )
            out["loss"] = il_loss
            return out

        reward_group, reward_gt, batched_sub, metric_cache = self._compute_pdm_outputs(
            traj_xyh,
            metric_cache,
            include_gt=True,
            targets=targets,
        )
        advantages, reward, sub_rewards_mean = self._compute_advantage_from_pdm(
            reward_group,
            reward_gt,
            batched_sub,
        )
        grpo_loss = self._compute_grpo_loss(logits, advantages)
        loss = il_loss + 0.5 * grpo_loss

        return {
            "loss": loss,
            "trajectory": trajectory,
            "all_trajectories": traj_xyh,
            "trajectory_probabilities": trajectory_probabilities,
            "trajectory_pdm_scores": reward_group,
            "reward": reward,
            "sub_rewards": sub_rewards_mean,
        }

    def forward_test_rl(
        self,
        encoding_feature,
        encoding_key_padding_mask,
        targets,
        metric_cache,
    ) -> Dict[str, torch.Tensor]:
        traj_xy, traj_xyh, logits = self._predict_candidates(
            encoding_feature,
            encoding_key_padding_mask,
        )
        bs = traj_xy.shape[0]
        trajectory_probabilities, prob_flat = self._compute_probabilities(logits)
        trajectory = self._select_best_trajectory(traj_xyh, prob_flat)
        if not self._can_use_pdm(metric_cache):
            reason = self._pdm_init_error or "metric_cache_missing"
            return self._fallback_loss_dict(
                trajectory,
                traj_xyh,
                trajectory_probabilities,
                targets,
                reason,
            )

        reward_group, _, sub_rewards_group, metric_cache = self._compute_pdm_outputs(
            traj_xyh,
            metric_cache,
            include_gt=False,
            targets=targets,
        )
        reward_flat = reward_group.reshape(bs, -1)
        prob_flat = trajectory_probabilities.reshape(bs, -1)
        trajectory = self._select_best_trajectory(traj_xyh, prob_flat)
        batch_idx = torch.arange(bs, device=traj_xyh.device)
        reward_best = reward_flat[batch_idx, prob_flat.argmax(dim=-1)]

        if targets is not None and "trajectory" in targets:
            trajectory_loss = F.l1_loss(trajectory, targets["trajectory"])
        else:
            trajectory_loss = trajectory.new_tensor(0.0)

        sub_scores_mean = {"pdm_enabled": 1.0}
        if sub_rewards_group is not None:
            sub_scores_mean.update({k: v.mean().item() for k, v in sub_rewards_group.items()})
        return {
            "loss": trajectory_loss,
            "reward": reward_best.mean(),
            "sub_rewards": sub_scores_mean,
            "trajectory": trajectory,
            "all_trajectories": traj_xyh,
            "trajectory_probabilities": trajectory_probabilities,
            "trajectory_pdm_scores": reward_group,
        }

    def bezier_xyyaw(self, xy8: torch.Tensor) -> torch.Tensor:
        assert xy8.shape[-2:] == (8, 2), "Input must be (B,G,8,2)"
        device, dtype = xy8.device, xy8.dtype

        origin = torch.zeros_like(xy8[..., :1, :])
        ctrl = torch.cat([origin, xy8], dim=-2)
        n = ctrl.shape[-2] - 1

        delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]
        binom = torch.tensor([math.comb(n - 1, i) for i in range(n)], device=device, dtype=dtype)
        t = torch.arange(1, n + 1, device=device, dtype=dtype) / n

        t_pow = t.view(-1, 1) ** torch.arange(0, n, device=device, dtype=dtype)
        one_pow = (1 - t).view(-1, 1) ** torch.arange(n - 1, -1, -1, device=device, dtype=dtype)
        basis = binom * t_pow * one_pow

        delta_exp = delta.unsqueeze(2)
        basis_exp = basis.view(1, 1, 8, 8, 1)
        deriv = n * (delta_exp * basis_exp).sum(dim=3)

        dx, dy = deriv[..., 0], deriv[..., 1]
        yaw = torch.atan2(dy, dx).unsqueeze(-1)
        return torch.cat([xy8, yaw], dim=-1)
