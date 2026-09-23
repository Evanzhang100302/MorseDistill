import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

PAD = 0


# ─── Attention & FFN ────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        for w in [self.w_qs, self.w_ks, self.w_vs]:
            nn.init.xavier_uniform_(w.weight)
        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output)) + residual
        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = self.dropout(self.w_2(self.dropout(F.gelu(self.w_1(x))))) + residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                           dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn  = PositionwiseFeedForward(d_model, d_inner,
                                                dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask
        return enc_output, enc_slf_attn


# ─── Mask helpers ────────────────────────────────────────────────────────────

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD).unsqueeze(1).expand(-1, len_q, -1, -1)
    return padding_mask

def get_subsequent_mask(seq, dim=2):
    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1
    ).permute(1, 2, 0).unsqueeze(0).expand(sz_b, -1, -1, -1)
    return subsequent_mask


# ─── RNN layer ───────────────────────────────────────────────────────────────

class RNN_layers(nn.Module):
    def __init__(self, d_model, d_rnn):
        super().__init__()
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu().clamp(min=1)
        packed  = nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        out     = nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0], batch_first=True)[0]
        return self.projection(out)


# ─── Encoder_ST ──────────────────────────────────────────────────────────────

class Encoder_ST(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v,
                 dropout, device, loc_dim, CosSin=False):
        super().__init__()
        self.d_model = d_model
        self.loc_dim = loc_dim
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)], device=device)

        def mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        self.event_emb_temporal = mlp(1)
        self.event_emb_loc      = mlp(loc_dim)

        def layer_stack():
            return nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
                for _ in range(n_layers)])
        self.layer_stack          = layer_stack()
        self.layer_stack_loc      = layer_stack()
        self.layer_stack_temporal = layer_stack()

    def temporal_enc(self, time, non_pad_mask):
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, non_pad_mask):
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask = (slf_attn_mask_keypad.type_as(slf_attn_mask_subseq) +
                         slf_attn_mask_subseq).gt(0)[:, :, :, 0]

        enc_t   = self.temporal_enc(event_time, non_pad_mask)
        enc_loc = self.event_emb_loc(event_loc)
        enc_out = enc_t + enc_loc

        for i in range(len(self.layer_stack)):
            enc_loc, _ = self.layer_stack_loc[i](enc_loc, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            enc_t,   _ = self.layer_stack_temporal[i](enc_t, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            enc_out, _ = self.layer_stack[i](enc_out, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_out, enc_t, enc_loc


# ─── Transformer_ST (main model) ─────────────────────────────────────────────

class Transformer_ST(nn.Module):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64,
                 dropout=0.1, device=None, loc_dim=2, CosSin=False):
        super().__init__()
        self.encoder = Encoder_ST(d_model=d_model, d_inner=d_inner, n_layers=n_layers,
                                  n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                                  device=device, loc_dim=loc_dim, CosSin=CosSin)
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.rnn          = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_spatial  = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        non_pad_mask = get_non_pad_mask(event_time)
        enc_out, enc_t, enc_loc = self.encoder(event_loc, event_time, non_pad_mask)
        enc_out = self.rnn(enc_out, non_pad_mask)
        enc_t   = self.rnn_temporal(enc_t, non_pad_mask)
        enc_loc = self.rnn_spatial(enc_loc, non_pad_mask)
        return torch.cat((enc_t, enc_loc, enc_out), dim=-1), non_pad_mask


# ─── Dataset & DataLoader ─────────────────────────────────────────────────────

class EventData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.time      = [[e[0] for e in inst] for inst in data]
        self.time_norm = [[e[1] for e in inst] for inst in data]
        self.lng       = [[e[2] for e in inst] for inst in data]
        self.lat       = [[e[3] for e in inst] for inst in data]
        self.length    = len(data)

    def __len__(self): return self.length
    def __getitem__(self, idx):
        return self.time[idx], self.time_norm[idx], self.lng[idx], self.lat[idx]


def _pad(insts):
    max_len = max(len(i) for i in insts)
    return torch.tensor(
        np.array([i + [PAD] * (max_len - len(i)) for i in insts]), dtype=torch.float32)

def _collate(insts):
    time, time_norm, lng, lat = zip(*insts)
    return _pad(time), _pad(time_norm), _pad(lng), _pad(lat)

def get_dataloader(data, batch_size, D=2, shuffle=True):
    ds = EventData(data)
    return torch.utils.data.DataLoader(
        ds, num_workers=2, batch_size=batch_size,
        collate_fn=_collate, shuffle=shuffle)
