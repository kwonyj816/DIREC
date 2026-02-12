import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class TargetAwareAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        # Multi-head attention
        super(TargetAwareAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.attn_temper = 0.5
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = emb_dim // num_heads

        # Q, K, V projection
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)

        # Output projection
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, target_item_embedding: torch.Tensor,  # (batch, emb_dim)
                source_item_embeddings: list[torch.Tensor], # list of (Num_item_i, emb_dim)
                source_reviews: list[torch.Tensor]          # list of (Num_item_i, emb_dim)
            ) -> torch.Tensor:
        # Compute target-aware preference and attention weights.
        batch_size = target_item_embedding.size(0)
        emb_dim = target_item_embedding.size(1)

        lengths = [x.size(0) for x in source_item_embeddings]
        max_len = max(lengths)

        padded_keys = torch.zeros((batch_size, max_len, emb_dim), device=target_item_embedding.device)
        padded_values = torch.zeros((batch_size, max_len, emb_dim), device=target_item_embedding.device)
        key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=target_item_embedding.device)  # True = masked

        for i in range(batch_size):
            len_i = lengths[i]
            padded_keys[i, :len_i, :] = source_item_embeddings[i]
            padded_values[i, :len_i, :] = source_reviews[i]
            key_padding_mask[i, :len_i] = False

        # Q, K, V projections
        q = self.q_proj(target_item_embedding).unsqueeze(1)  # (batch, 1, emb_dim)
        k = self.k_proj(padded_keys)                        # (batch, max_len, emb_dim)
        v = self.v_proj(padded_values)                      # (batch, max_len, emb_dim)

        def reshape_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.head_dim ** 0.5) * self.attn_temper)
        attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, emb_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output + target_item_embedding.unsqueeze(1)
        attn_output = attn_output.squeeze(1)

        normed_output = self.norm(attn_output)  # (batch, emb_dim)

        return normed_output, attn_weights


class RatingPredictor(nn.Module):
    def __init__(self, emb_dim, n_user, n_item, dropout=0.1):
        # MLP rating head with item bias.
        super(RatingPredictor, self).__init__()
        hidden_dims = [emb_dim, emb_dim // 4, emb_dim // 16]
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dims[-1], 1, bias=False))
        self.mlp = nn.Sequential(*layers)

        self.item_bias = nn.Embedding(n_item, 1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, embedding, item_ids):
        # Predict ratings from embeddings.
        """
        embedding: Tensor of shape (batch_size, emb_dim)
        item_ids: Tensor of shape (batch_size,) - LongTensor
        """
        base_rating = self.mlp(embedding).squeeze(-1)             # (batch_size,)
        item_b = self.item_bias(item_ids).squeeze(-1)             # (batch_size,)
        return base_rating + item_b                      # (batch_size,)
    


class TimeCondMLP(nn.Module):
    def __init__(self, emb_dim: int):
        # MLP that mixes embedding with condition.
        super().__init__()
        self.double_nn = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Combine x and condition embedding through a shared MLP.
        """
        x: (batch, emb_dims)
        t: (batch, time_dims)
        """
        x = self.double_nn(torch.cat([x, t], dim=-1))
        return x


class MLP_conditional(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        condition: bool = False,
    ):
        # Stack time-conditioned MLP blocks with optional conditioning.
        """
        emb_dim: output embedding dimension
        """
        super().__init__()
        self.condition = condition
        self.emb_dim = emb_dim
        self.time_cond_mlp_1 = TimeCondMLP(emb_dim)
        self.time_cond_mlp_2 = TimeCondMLP(emb_dim)
        self.time_cond_mlp_3 = TimeCondMLP(emb_dim)
        self.norm_x = nn.LayerNorm(emb_dim)
        self.norm_t = nn.LayerNorm(emb_dim)

    def pos_encoding(self, t: torch.Tensor) -> torch.Tensor:
        # Build sinusoidal positional encoding for timesteps.
        """
        positional encoding
        t: (batch, 1)
        returns: (batch, emb_dim)
        """
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.emb_dim, 2, device=one_param(self).device).float() / self.emb_dim)
        )
        sinusoid_in = t.repeat(1, self.emb_dim // 2) * inv_freq
        pos_enc = torch.cat([torch.sin(sinusoid_in), torch.cos(sinusoid_in)], dim=-1)
        return pos_enc

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_aware_pref: torch.Tensor = None
    ) -> torch.Tensor:
        # Predict denoised embedding with optional conditioning.
        """
        x: (batch, emb_dim)
        t: (batch,) single scalar timestep
        target_aware_pref: (batch, emb_dim)
        """
        t = t.unsqueeze(-1)             # (batch, 1)
        t_emb = self.pos_encoding(t)    # (batch, emb_dim)
        
        if self.condition and target_aware_pref is not None:
            t_emb = t_emb + target_aware_pref
        
        x = self.norm_x(x)
        t_emb = self.norm_t(t_emb)

        x = self.time_cond_mlp_1(x, t_emb)
        x = self.time_cond_mlp_2(x, t_emb)
        out_emb = self.time_cond_mlp_3(x, t_emb)

        return out_emb