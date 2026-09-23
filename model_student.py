"""Attention-free MLP student for the MCM-DM history encoder.

Drop-in replacement for transformer_st.Transformer_ST: same call signature
(event_loc, event_time) and same outputs ([B, L, 3*d], non_pad_mask), so the
diffusion head, VLM projector, ST-Morse encoder and fusion gate are untouched.

Teacher: 4-layer x 3-stack self-attention encoder + 3 LSTMs  (1.46 M params)
Student: causal window flatten -> shared MLP trunk -> 3 heads (~0.15 M params)

A position-wise MLP alone has no history, so the causal receptive field is made
explicit: every position sees the last W events, gathered by a padded unfold
(no attention, no recurrence, one matmul chain).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def causal_windows(x, W):
    """x: [B, L, C] -> [B, L, W*C]; slot w of position i holds event i-W+1+w.

    Left-padded with zeros so position i never sees i+1 (strict causality).
    """
    B, L, C = x.shape
    xp = F.pad(x.transpose(1, 2), (W - 1, 0))        # [B, C, L+W-1]
    win = xp.unfold(2, W, 1)                          # [B, C, L, W]
    return win.permute(0, 2, 1, 3).reshape(B, L, C * W)


class MLPHistoryEncoder(nn.Module):
    def __init__(self, d_model=64, window=16, hidden=256, n_hidden=2,
                 dropout=0.1, device=None, loc_dim=2):
        super().__init__()
        self.d_model = d_model
        self.window  = window
        self.loc_dim = loc_dim
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # per-event channels: absolute time, inter-event dt, loc dims, validity
        self.n_chan = 2 + loc_dim + 1
        in_dim = self.n_chan * window + d_model

        layers, prev = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
            prev = hidden
        self.trunk = nn.Sequential(*layers)
        # three heads mirroring the teacher's (enc_t, enc_loc, enc_out) streams
        self.head_t   = nn.Linear(hidden, d_model)
        self.head_loc = nn.Linear(hidden, d_model)
        self.head_out = nn.Linear(hidden, d_model)

    def temporal_enc(self, time, non_pad_mask):
        """Identical to Encoder_ST.temporal_enc so the student sees the same
        absolute-time representation as the teacher."""
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time):
        non_pad_mask = get_non_pad_mask(event_time)              # [B, L, 1]

        dt = event_time - F.pad(event_time, (1, 0))[:, :-1]      # inter-event time
        feats = torch.cat([event_time.unsqueeze(-1),
                           dt.unsqueeze(-1),
                           event_loc,
                           non_pad_mask], dim=-1) * non_pad_mask  # [B, L, n_chan]

        h = torch.cat([causal_windows(feats, self.window),
                       self.temporal_enc(event_time, non_pad_mask)], dim=-1)
        h = self.trunk(h)

        enc_t   = self.head_t(h)   * non_pad_mask
        enc_loc = self.head_loc(h) * non_pad_mask
        enc_out = self.head_out(h) * non_pad_mask
        return torch.cat((enc_t, enc_loc, enc_out), dim=-1), non_pad_mask
