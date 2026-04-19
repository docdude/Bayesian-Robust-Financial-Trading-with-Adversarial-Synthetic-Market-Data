"""
Self-contained DQN model architecture.

Combines embed.py, qnet.py, and policy.py into a single file so the
Agent nn.Module can be reconstructed from a state dict without the full repo.

External deps: torch, timm (for Mlp), einops (for rearrange).
"""
import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import Mlp


# ═══════════════════════════════════════════════════════════════════════════
#  Embeddings (from embed.py)
# ═══════════════════════════════════════════════════════════════════════════


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        w = torch.zeros(input_dim, embed_dim).float()
        w.require_grad = False
        position = torch.arange(0, input_dim).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        ).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(input_dim, embed_dim)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type="fixed", temporals_name=("day", "weekday", "month")):
        super().__init__()
        self.temporals_name = list(temporals_name)
        size_maps = {"day": 32, "weekday": 7, "month": 13, "hour": 25, "minute": 61}
        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        self.embed = nn.ModuleDict()
        for item in self.temporals_name:
            self.embed[item] = Embed(size_maps[item], embed_dim)

    def forward(self, x):
        x = x.long()
        embeds = []
        for index, item in enumerate(self.temporals_name):
            embeds.append(self.embed[item](x[:, :, index]))
        return torch.stack(embeds, dim=-2).sum(dim=-2)


class TimesEmbed(nn.Module):
    def __init__(self, *, timestamps=10, input_dim=156, embed_dim=128,
                 embed_type="fixed", temporals_name=("day", "weekday", "month"), **kwargs):
        super().__init__()
        self.temporal_dim = len(temporals_name)
        self.feature_dim = input_dim - self.temporal_dim
        self.value_embedding = TokenEmbedding(self.feature_dim, embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim, temporals_name=temporals_name,
                                                     embed_type=embed_type)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        b, c, d, f = x.shape
        x = rearrange(x, "b c d f -> (b c) d f", b=b, c=c)
        feature = x[..., : -self.temporal_dim]
        temporal = x[..., -self.temporal_dim :]
        x = (
            self.value_embedding(feature)
            + self.temporal_embedding(temporal)
            + self.position_embedding(feature)
        )
        x = rearrange(x, "(b c) d f -> b c d f", b=b, c=c)
        return x.mean(dim=-2)


# ═══════════════════════════════════════════════════════════════════════════
#  QNet & QuantileBelief (from qnet.py)
# ═══════════════════════════════════════════════════════════════════════════


def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class QuantileBelief(nn.Module):
    def __init__(self, *, input_dim=156, timestamps=10, embed_dim=128, depth=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), output_dim=3,
                 temporals_name=("day", "weekday", "month"), device=torch.device("cpu"),
                 **kwargs):
        super().__init__()
        self.temporal_dim = len(temporals_name)
        self.value_embedding = TokenEmbedding(input_dim - self.temporal_dim, embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim, temporals_name=temporals_name,
                                                     embed_type="fixed")
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                            num_layers=depth, batch_first=True)
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, output_dim, bias=True)
        self.apply(_init_weights)
        self.to(device)

    def forward(self, x):
        feature = x[..., : -self.temporal_dim]
        temporal = x[..., -self.temporal_dim :]
        x = self.value_embedding(feature) + self.temporal_embedding(temporal)
        x, _ = self.lstm(x)
        x = self.norm(x)
        return self.decoder_pred(x)


class QNet(nn.Module):
    def __init__(self, *, input_dim=156, timestamps=10, embed_method="mean",
                 embed_dim=128, depth=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 cls_embed=False, output_dim=3,
                 temporals_name=("day", "weekday", "month"),
                 use_quantile_belief=False, quantile_heads_num=0,
                 device=torch.device("cpu"), **kwargs):
        super().__init__()
        self.use_quantile_belief = use_quantile_belief
        self.patch_embed = TimesEmbed(timestamps=timestamps, input_dim=input_dim,
                                       embed_dim=embed_dim, embed_method=embed_method,
                                       temporals_name=temporals_name)
        self.blocks = nn.ModuleList([
            Mlp(in_features=embed_dim, hidden_features=embed_dim,
                act_layer=nn.Tanh, out_features=embed_dim)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, output_dim, bias=True)
        if use_quantile_belief:
            self.belief_embedding = nn.Embedding(quantile_heads_num, embed_dim)
        self.apply(_init_weights)
        self.to(device)

    def forward(self, x, quantile_belief=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_quantile_belief and quantile_belief is not None:
            x = x + self.belief_embedding(quantile_belief).unsqueeze(1)
        x = self.decoder_pred(x).squeeze(-1).squeeze(1)
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  Agent (from policy.py)
# ═══════════════════════════════════════════════════════════════════════════


class Agent(nn.Module):
    """Full DQN Agent with Q-network, target network, optional NFSP + quantile belief."""

    def __init__(self, *, input_dim=153, timestamps=30, embed_dim=64, depth=1,
                 action_dim=3, temporals_name=("day", "weekday", "month"),
                 device=torch.device("cpu"), use_quantile_belief=True,
                 quantile_heads_num=5, use_nfsp=True, **kwargs):
        super().__init__()
        qnet_kwargs = dict(
            input_dim=input_dim, timestamps=timestamps, embed_dim=embed_dim,
            depth=depth, output_dim=action_dim, temporals_name=temporals_name,
            use_quantile_belief=use_quantile_belief,
            quantile_heads_num=quantile_heads_num, device=device,
        )
        self.q_network = QNet(**qnet_kwargs).to(device)
        self.target_network = QNet(**qnet_kwargs).to(device)
        self.use_nfsp = use_nfsp
        if use_nfsp:
            self.q_network_nfsp = QNet(**qnet_kwargs).to(device)
        self.use_quantile_belief = use_quantile_belief
        if use_quantile_belief:
            self.quantile_belief_network = QuantileBelief(
                input_dim=input_dim, timestamps=timestamps, embed_dim=embed_dim,
                depth=depth, output_dim=quantile_heads_num,
                temporals_name=temporals_name, device=device,
            ).to(device)
        self.device = device

    def forward(self, *args, **kwargs):
        pass
