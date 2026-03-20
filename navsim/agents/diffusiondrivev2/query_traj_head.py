import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorQueryTrajHead(nn.Module):
    def __init__(self, embed_dim, num_anchors, num_modes, timestamp, nhead=4, dropout=0.1):
        super().__init__()
        self.N_anchor = num_anchors   # 锚点数量 (e.g., 6)
        self.M = num_modes            # 模式数量
        self.T = timestamp            # 时间步长 (e.g., 80)
        self.D = embed_dim            # 特征维度
        
        # 1. Anchor Encoder: [N_anchor, T, 2] -> [N_anchor, D]
        # 将几何轨迹映射到特征空间，作为Query的初始值
        self.anchor_encoder = nn.Sequential(
            nn.Linear(self.T * 2, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D)
        )
        
        # 2. Mode Embedding: [M, D]
        # 可学习的风格向量 (如: 左变道、直行、保守、激进等)
        self.mode_embedding = nn.Embedding(self.M, self.D)
        
        # 3. Interaction Layer (Cross-Attention)
        # Query: Anchor Queries, Key/Value: Scene Features
        # 这里的d_model必须与embed_dim一致
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.D, 
            nhead=nhead, 
            dim_feedforward=self.D * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.interaction_layer = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # 4. Prediction Heads
        # 输入是交互后的特征 [B, N_anchor * M, D]
        self.reg_head = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.T * 2)
        )
        # TODO cls head needed to be add layer
        self.cls_head = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.LayerNorm(self.D),
            nn.ReLU(),
            nn.Linear(self.D, 1)
        )
        

    def forward(self, feat, anchor, scene_key_padding_mask=None):
        """
        Inputs:
        - feat:  [B, N_scene, D]  (Encoded agents + map items)
        - anchor: [N_anchor, T, 2] (Pre-defined trajectory anchors)
        
        Outputs:
        - traj_pred: [B, N_anchor, M, T, 2]
        - logits:    [B, N_anchor, M]
        """
        B, N_scene, D = feat.shape
        anchor = anchor.to(device=feat.device, dtype=feat.dtype)
        
        # --- Step 1: Encode Anchors to Queries ---
        # anchor: [N_anchor, T, 2] -> [N_anchor, T*2]
        anchor_flat = anchor.reshape(self.N_anchor, -1)
        # anchor_query: [N_anchor, D]
        anchor_query_base = self.anchor_encoder(anchor_flat)
        
        # --- Step 2: Generate Multi-Modal Queries ---
        # mode_embed: [M, D]
        mode_embed = self.mode_embedding.weight
        
        # Query = Anchor_Base + Mode_Style
        # [N_anchor, 1, D] + [1, M, D] -> [N_anchor, M, D]
        query_nm = anchor_query_base.unsqueeze(1) + mode_embed.unsqueeze(0)
        
        # Reshape for Attention: [B, N_anchor * M, D]
        # 需要扩展到Batch维度
        query = query_nm.reshape(1, self.N_anchor * self.M, self.D).expand(B, -1, -1)
        
        # --- Step 3: Cross-Attention Interaction ---
        # Query关注场景特征Feat
        # Query shape: [B, N_anchor * M, D]
        # Memory shape: [B, N_scene, D]
        # Output shape: [B, N_anchor * M, D]
        interaction_feat = self.interaction_layer(
            tgt=query,
            memory=feat,
            memory_key_padding_mask=scene_key_padding_mask,
        )
        
        # --- Step 4: Prediction ---
        # Regression: [B, N_anchor * M, D] -> [B, N_anchor * M, T*2]
        offset_flat = self.reg_head(interaction_feat)
        offset = offset_flat.reshape(B, self.N_anchor, self.M, self.T, 2)
        
        # Classification: [B, N_anchor * M, D] -> [B, N_anchor * M, 1] -> [B, N_anchor, M]
        logits = self.cls_head(interaction_feat).reshape(B, self.N_anchor, self.M)
        
        # Apply offset to original anchor
        # anchor: [N_anchor, T, 2] -> [1, N_anchor, 1, T, 2] (broadcastable)
        traj_pred = anchor.unsqueeze(0).unsqueeze(2) + offset
        
        return {
            "traj_pred": traj_pred,
            "logits": logits,
            "offset": offset
        }

    def compute_loss(self, output, gt_traj, advantage):
        """
        计算损失 (IL + GRPO)
        - gt_traj: [B, T, 2]
        - advantage: [B, N_anchor * M]
        """
        traj_pred = output['traj_pred'] # [B, N_anchor, M, T, 2]
        logits = output['logits']       # [B, N_anchor, M]
        B, N_anc, M, T, _ = traj_pred.shape
        
        # --- 1. Imitation Learning Loss ---
        # 计算 GT 与所有预测轨迹的距离
        # gt_traj: [B, T, 2] -> [B, 1, 1, T, 2]
        gt_expanded = gt_traj.unsqueeze(1).unsqueeze(2)
        dist = torch.norm(traj_pred - gt_expanded, dim=(-1, -2)) # [B, N_anchor, M]
        
        # 找到最佳匹配
        dist_flat = dist.view(B, -1) # [B, N_anchor * M]
        best_idx = torch.argmin(dist_flat, dim=1) # [B]
        
        # Regression Loss (仅对最佳匹配计算)
        batch_idx = torch.arange(B, device=traj_pred.device)
        best_trajs = traj_pred.view(B, -1, T, 2)[batch_idx, best_idx] # [B, T, 2]
        loss_reg = F.smooth_l1_loss(best_trajs, gt_traj)
        
        # Classification Loss (让最佳匹配的概率最大)
        loss_cls = F.cross_entropy(logits.view(B, -1), best_idx)
        
        L_il = loss_reg + loss_cls
        
        # --- 2. GRPO-like Loss ---
        # 对 Advantage 进行归一化 (Group Relative)
        # advantage shape: [B, N_anchor * M]
        adv_mean = advantage.mean(dim=1, keepdim=True)
        adv_std = advantage.std(dim=1, keepdim=True) + 1e-8
        adv_norm = (advantage - adv_mean) / adv_std
        
        # 计算 Policy Log Prob
        prob = F.softmax(logits.view(B, -1), dim=-1)
        log_prob = torch.log(prob + 1e-8)
        
        # GRPO Loss: 鼓励高 Advantage 的轨迹
        L_grpo = - (adv_norm * log_prob).mean()
        
        loss = L_il + 0.5 * L_grpo
        return loss, {"loss_il": L_il.item(), "loss_grpo": L_grpo.item()}
