from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
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
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
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

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

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
        self._ego_query = nn.Parameter(torch.randn(1, 1, config.tf_d_model) * 0.02)
        self._agent_queries = nn.Parameter(
            torch.randn(1, config.num_bounding_boxes, config.tf_d_model) * 0.02
        )
        self._query_cross_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )

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
        ego_query = self._ego_query.expand(batch_size, -1, -1)
        ego_query = ego_query + self._query_cross_attention(
            ego_query,
            encoding_feature,
            encoding_feature,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        agents_query = self._agent_queries.expand(batch_size, -1, -1)
        agents_query = agents_query + self._query_cross_attention(
            agents_query,
            encoding_feature,
            encoding_feature,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        output: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            old_pred = self._trajectory_head(
                ego_query,
                agents_query,
                encoding_feature,
                key_padding_mask,
                None,
                None,
                None,
                targets=targets,
                global_img=None,
                eta=eta,
                old_pred=None,
                metric_cache=metric_cache,
                cal_pdm=cal_pdm,
                token=token,
            )
        if self.training and old_pred.get("advantages") is not None:
            pred = self._trajectory_head(
                ego_query,
                agents_query,
                encoding_feature,
                key_padding_mask,
                None,
                None,
                None,
                targets=targets,
                global_img=None,
                eta=eta,
                old_pred=old_pred,
                metric_cache=metric_cache,
                cal_pdm=cal_pdm,
                token=token,
            )
            if "reward" not in pred:
                pred["reward"] = old_pred["reward"]
            if "sub_rewards" not in pred:
                pred["sub_rewards"] = old_pred["sub_rewards"]
        else:
            pred = old_pred
        output.update(pred)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
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

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape
        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_encoding_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, config.tf_d_model)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                encoding_feature,
                encoding_key_padding_mask,
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = traj_feature + self.dropout(
            self.cross_encoding_attention(
                traj_feature,
                encoding_feature,
                encoding_feature,
                key_padding_mask=encoding_key_padding_mask,
                need_weights=False,
            )[0]
        )
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        traj_feature = traj_feature.view(traj_feature.shape[0], -1, 20, traj_feature.shape[-1])
        bs,num_groups, _, _ = traj_feature.shape
        traj_feature = traj_feature.view(-1, 20, traj_feature.shape[-1])
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg = poses_reg.view(bs, 20*num_groups, 8, 3)
        poses_cls = poses_cls.view(bs, -1, 20)
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                encoding_feature,
                encoding_key_padding_mask,
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature,
                traj_points,
                encoding_feature,
                encoding_key_padding_mask,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class DDIMScheduler_with_logprob(DDIMScheduler):
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0, # 1.0 for ddpm, 0.0 for ddim
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )
        # # to prevent OOB on gather
        # prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** (0.5)).clamp_(min=1e-10)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp_(min=0) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if prev_sample_mean is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
            )

        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        else:
            std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        if prev_sample is None:
            # 乘性噪声
            variance_noise_horizon = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0

            variance_noise_mul = torch.cat((variance_noise_horizon,variance_noise_vert),dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1,1,model_output.shape[2],1)

            # 加性噪声
            variance_noise_x = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_y = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_add = torch.cat((variance_noise_x,variance_noise_y),dim=-1)
            variance_noise_add = variance_noise_add.repeat(1,1,model_output.shape[2],1)

            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add

        std_dev_t_mul = torch.clip(std_dev_t, min=0.1)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul**2))
            - torch.log(std_dev_t_mul)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )   

        log_prob = log_prob.sum(dim=(-2, -1))
        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)



class TrajectoryHead(nn.Module):
    """Diffusion trajectory head conditioned on vector tokens."""

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
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20
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
        )  # [20, T, 2]

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
            num_train_timesteps=1000,
            steps_offset=1,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        self.sigmoid = nn.Sigmoid()
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 1)

        self.loss_computer = LossComputer(config)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.targets = []
        self.num_draw = 0
        self._score_buf = []

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
        diffusion_output: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]],
        reason: str,
        all_diffusion_output: Optional[torch.Tensor] = None,
        log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = diffusion_output.device
        if targets is not None and "trajectory" in targets:
            best_traj = diffusion_output[:, 0] if diffusion_output.dim() == 4 else diffusion_output
            loss = F.l1_loss(best_traj, targets["trajectory"])
        else:
            loss = diffusion_output.new_tensor(0.0)
        out = {
            "loss": loss,
            "reward": torch.zeros((), device=device, dtype=diffusion_output.dtype),
            "sub_rewards": {"pdm_enabled": 0.0},
            "trajectory": diffusion_output[:, 0] if diffusion_output.dim() == 4 else diffusion_output,
            "pdm_disabled_reason": reason,
            "advantages": None,
        }
        if all_diffusion_output is not None:
            out["all_diffusion_output"] = all_diffusion_output
        if log_probs is not None:
            out["log_probs"] = log_probs
        return out

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = odo_info_fut_x / 50
        odo_info_fut_y = odo_info_fut_y / 20
        odo_info_fut_head = odo_info_fut_head / 1.57
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = odo_info_fut_x * 50
        odo_info_fut_y = odo_info_fut_y * 20
        odo_info_fut_head = odo_info_fut_head * 1.57
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def forward(
        self,
        ego_query,
        agents_query,
        encoding_feature,
        encoding_key_padding_mask,
        status_encoding,
        status_feature,
        camera_feature,
        targets=None,
        global_img=None,
        eta=0.0,
        old_pred=None,
        metric_cache=None,
        cal_pdm=True,
        token=None,
    ) -> Dict[str, torch.Tensor]:
        if self.training:
            if old_pred is not None:
                return self.get_rlloss(
                    ego_query,
                    agents_query,
                    encoding_feature,
                    encoding_key_padding_mask,
                    status_encoding,
                    status_feature,
                    camera_feature,
                    targets,
                    global_img,
                    eta,
                    old_pred,
                )
            return self.forward_train_rl(
                ego_query,
                agents_query,
                encoding_feature,
                encoding_key_padding_mask,
                status_encoding,
                status_feature,
                camera_feature,
                targets,
                global_img,
                eta,
                metric_cache,
                cal_pdm=cal_pdm,
                token=token,
            )
        return self.forward_test_rl(
            ego_query,
            agents_query,
            encoding_feature,
            encoding_key_padding_mask,
            status_encoding,
            status_feature,
            camera_feature,
            targets,
            global_img,
            metric_cache,
            eta=0.0,
        )

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
        ego_query,
        agents_query,
        encoding_feature,
        encoding_key_padding_mask,
        status_encoding,
        status_feature,
        camera_feature,
        targets,
        global_img,
        eta,
        metric_cache,
        cal_pdm,
        token,
    ) -> Dict[str, torch.Tensor]:
        step_num = 10
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = self.num_groups
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])
        diffusion_output = self.norm_odo(plan_anchor)

        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusionrl_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps
        )

        all_log_probs = []
        all_diffusion_output = [diffusion_output]

        for k in roll_timesteps:
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, num_groups * self.ego_fut_mode, -1)

            timesteps = k.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                encoding_feature,
                encoding_key_padding_mask,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg = poses_reg_list[-1]
            x_start = self.norm_odo(poses_reg[..., :2])
            prev_sample, log_prob, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=eta,
            )
            diffusion_output = prev_sample
            all_log_probs.append(log_prob)
            all_diffusion_output.append(prev_sample)

        all_log_probs = torch.stack(all_log_probs, dim=-1)
        all_log_probs = all_log_probs.view(bs, num_groups, self.ego_fut_mode, all_log_probs.shape[-1])
        all_diffusion_output = torch.stack(all_diffusion_output, dim=-1)

        diffusion_output = self.denorm_odo(diffusion_output)
        diffusion_output = self.bezier_xyyaw(diffusion_output)
        if (not cal_pdm) or (not self._can_use_pdm(metric_cache)):
            reason = self._pdm_init_error or "metric_cache_missing_or_cal_pdm_disabled"
            return self._fallback_loss_dict(
                diffusion_output,
                targets,
                reason,
                all_diffusion_output=all_diffusion_output,
            )

        target_traj = targets["trajectory"].unsqueeze(1)
        diffusion_output_with_gt = torch.cat((diffusion_output, target_traj), dim=1)
        reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(
            diffusion_output_with_gt, metric_cache
        )
        reward_gt = reward_group[:, -1:].unsqueeze(-1)
        reward_group = reward_group[:, :-1]

        keys = sub_rewards_group[0].keys()
        batched_sub = {
            k: torch.as_tensor(
                np.vstack([d[k] for d in sub_rewards_group]),
                device=reward_group.device,
                dtype=reward_group.dtype,
            )
            for k in keys
        }

        reward_group = reward_group.view(bs, num_groups, self.ego_fut_mode)
        mean_grouped_rewards = reward_group.mean(dim=1)
        std_grouped_rewards = reward_group.std(dim=1)
        advantages = (reward_group - mean_grouped_rewards.unsqueeze(1)) / (
            std_grouped_rewards.unsqueeze(1) + 1e-4
        )

        mask_positive = reward_group > (reward_gt - 1e-6)
        advantages = advantages.clamp(min=0) * mask_positive.float()

        for k, v_full in batched_sub.items():
            v = v_full[:, :-1].view(bs, num_groups, self.ego_fut_mode)
            if k == "no_collision" or k == "drivable_area":
                zero_mask = v != 1
                advantages = torch.where(zero_mask, torch.full_like(advantages, -1.0), advantages)

        pos_cnt = mask_positive.sum(dim=(1, 2), keepdim=True)
        pos_sum = (reward_group * mask_positive.float()).sum(dim=(1, 2), keepdim=True)
        mean_pos = pos_sum / pos_cnt.clamp(min=1)
        mean_all = reward_group.mean(dim=(1, 2), keepdim=True)
        batch_reward = torch.where(pos_cnt > 0, mean_pos, mean_all)
        reward = batch_reward.squeeze(-1).mean()

        sub_rewards_mean = {"pdm_enabled": 1.0}
        for k, v_full in batched_sub.items():
            v = v_full[:, :-1].view(bs, num_groups, self.ego_fut_mode)
            mean_all_k = v.mean(dim=1, keepdim=True)
            sub_rewards_mean[k] = mean_all_k.mean().item()

        advantages = advantages.view(bs, num_groups * self.ego_fut_mode)
        advantages = advantages.detach().unsqueeze(-1).repeat(1, 1, step_num)
        discount = torch.tensor([0.8 ** (step_num - i - 1) for i in range(step_num)]).to(
            advantages.device
        )
        advantages = advantages * discount

        return {
            "all_diffusion_output": all_diffusion_output,
            "advantages": advantages,
            "reward": reward,
            "sub_rewards": sub_rewards_mean,
        }

    def forward_test_rl(
        self,
        ego_query,
        agents_query,
        encoding_feature,
        encoding_key_padding_mask,
        status_encoding,
        status_feature,
        camera_feature,
        targets,
        global_img,
        metric_cache,
        eta=0.0,
    ) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = 4
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])
        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps
        )
        all_diffusion_output = [diffusion_output]
        all_log_probs = []
        ego_fut_mode = diffusion_output.shape[1]
        for k in roll_timesteps:
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k[None].to(diffusion_output.device) if len(k.shape) == 0 else k.to(diffusion_output.device)
            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                encoding_feature,
                encoding_key_padding_mask,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg = poses_reg_list[-1]
            x_start = self.norm_odo(poses_reg[..., :2])
            diffusion_output, log_prob, diffusion_output_mean = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=0.0,
            )
            all_diffusion_output.append(diffusion_output)
            all_log_probs.append(log_prob)

        diffusion_output = self.denorm_odo(diffusion_output_mean)
        diffusion_output = self.bezier_xyyaw(diffusion_output)

        all_diffusion_output = torch.stack(all_diffusion_output, dim=-1)
        all_log_probs = torch.stack(all_log_probs, dim=-1)
        if not self._can_use_pdm(metric_cache):
            reason = self._pdm_init_error or "metric_cache_missing"
            return self._fallback_loss_dict(
                diffusion_output,
                targets,
                reason,
                all_diffusion_output=all_diffusion_output,
                log_probs=all_log_probs,
            )

        reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(diffusion_output, metric_cache)
        reward_group = reward_group.max(dim=-1)[0]

        target_traj = targets["trajectory"].unsqueeze(1)
        trajectory_loss = F.l1_loss(diffusion_output, target_traj)

        keys = sub_rewards_group[0].keys()
        sub_rewards_group = {k: np.vstack([d[k] for d in sub_rewards_group]) for k in keys}
        sub_scores_mean = {"pdm_enabled": 1.0}
        sub_scores_mean.update({k: v.mean().item() for k, v in sub_rewards_group.items()})
        return {
            "loss": trajectory_loss,
            "reward": reward_group.mean(),
            "sub_rewards": sub_scores_mean,
            "all_diffusion_output": all_diffusion_output,
            "log_probs": all_log_probs,
        }

    def get_rlloss(
        self,
        ego_query,
        agents_query,
        encoding_feature,
        encoding_key_padding_mask,
        status_encoding,
        status_feature,
        camera_feature,
        targets,
        global_img,
        eta,
        old_pred,
    ):
        old_diffusion_output = old_pred["all_diffusion_output"]
        advantages = old_pred["advantages"]

        chains = old_diffusion_output[..., :-1]
        chains_prev = old_diffusion_output[..., 1:]

        step_num = 10
        bs = chains.shape[0]
        device = chains.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        all_log_probs = []
        poses_reg_steps_list = []
        for i, k in enumerate(roll_timesteps):
            diffusion_output = chains[..., i]
            ego_fut_mode = diffusion_output.shape[1]
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                encoding_feature,
                encoding_key_padding_mask,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg_steps_list.append(poses_reg_list)
            poses_reg = poses_reg_list[-1]
            x_start = self.norm_odo(poses_reg[..., :2])
            _, log_prob, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=eta,
                prev_sample=chains_prev[..., i],
            )
            all_log_probs.append(log_prob)

        all_log_probs = torch.stack(all_log_probs, dim=-1)
        per_token_logps = all_log_probs.view(bs, self.num_groups * self.ego_fut_mode, -1)
        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages

        mask_nz = per_token_loss != 0
        RL_loss_b = (per_token_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp_min(1)
        RL_loss_b = RL_loss_b.mean(dim=-1)

        IL_loss_b = torch.zeros_like(RL_loss_b)
        target_traj = targets["trajectory"].unsqueeze(1).repeat(1, self.num_groups * self.ego_fut_mode, 1, 1)
        for poses_reg_list in poses_reg_steps_list:
            for poses_reg in poses_reg_list:
                traj_l1 = F.l1_loss(
                    poses_reg[..., :2], target_traj[..., :2], reduction="none"
                )
                IL_loss_b += traj_l1.mean()

        IL_loss_b /= len(poses_reg_steps_list) * len(poses_reg_steps_list[0])
        has_positive = (advantages > 0).any(dim=2).any(dim=1)
        il_weight = torch.where(
            has_positive == 0,
            torch.tensor(1.0, device=RL_loss_b.device),
            torch.tensor(0.1, device=RL_loss_b.device),
        )
        loss_b = RL_loss_b + il_weight * IL_loss_b
        loss = loss_b.mean()
        return {"loss": loss}

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
