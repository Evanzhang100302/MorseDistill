import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
import numpy as np
from tqdm.auto import tqdm
from einops import reduce

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x): return x is not None
def default(val, d): return val if exists(val) else (d() if callable(d) else d)
def identity(t, *a, **k): return t

def extract(arr, timesteps, broadcast_shape):
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def normalize_to_neg_one_to_one(img): return img * 2 - 1
def unnormalize_to_zero_to_one(t): return (t + 1) * 0.5
def mean_flat(tensor): return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = next(o for o in (mean1, logvar1, mean2, logvar2) if isinstance(o, torch.Tensor))
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2)**2) * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(z, mean, log_std):
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, max_beta=0.01):
    return torch.linspace(1e-4, max_beta, timesteps)

class ImageProjector(nn.Module):
    def __init__(self, input_dim=1536, img_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, img_dim)
        )
    def forward(self, x):
        return self.proj(x)

class MorseTransformer(nn.Module):
    """
    Encode all tile Morse features [n_tiles, 2] via Transformer.
    Outputs per-tile embeddings [n_tiles, out_dim].
    Input is fixed (precomputed morse features), weights are trainable.
    """
    def __init__(self, in_dim=2, d_model=64, nhead=4, num_layers=2, out_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, tile_features):
        # tile_features: [n_tiles, 2]
        x = self.input_proj(tile_features.unsqueeze(0))  # [1, n_tiles, d_model]
        x = self.transformer(x)                           # [1, n_tiles, d_model]
        x = self.output_proj(x.squeeze(0))               # [n_tiles, out_dim]
        return x


class STMorseEncoder(nn.Module):
    """
    Encode all spatio-temporal Morse cells [n_cells, in_dim] via Transformer.
    Outputs per-cell embeddings [n_cells, out_dim].

    Unified replacement for MorseTransformer (spatial-only, tile graph) plus
    TemporalMorseEncoder (per-sequence temporal critical times): cells already
    live in the joint (dt, lon, lat) space, so one encoder covers both axes.
    Input is fixed (precomputed ST Morse features), weights are trainable.
    """
    def __init__(self, in_dim=2, d_model=64, nhead=4, num_layers=2, out_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, cell_features):
        # cell_features: [n_cells, in_dim]
        x = self.input_proj(cell_features.unsqueeze(0))   # [1, n_cells, d_model]
        x = self.transformer(x)                            # [1, n_cells, d_model]
        return self.output_proj(x.squeeze(0))              # [n_cells, out_dim]


class ModalFusionGate(nn.Module):
    """Gate network to fuse image and Morse embeddings (TimeVLM-style)."""
    def __init__(self, dim=64):
        super().__init__()
        self.norm_img   = nn.LayerNorm(dim)
        self.norm_morse = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, img_emb, morse_emb):
        # Normalize both to same scale before gating
        img_n   = self.norm_img(img_emb)
        morse_n = self.norm_morse(morse_emb)
        combined = torch.cat([img_n, morse_n], dim=-1)  # [N, 1, dim*2]
        weights  = self.gate(combined)                   # [N, 1, 2]
        fused = weights[..., 0:1] * img_n + weights[..., 1:2] * morse_n
        return fused

class TemporalMorseEncoder(nn.Module):
    """Encodes per-sequence temporal Morse critical nodes into a fixed-size embedding."""
    def __init__(self, top_n=3, d_model=64, out_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, crit_times):
        # crit_times: [B, K, 1] padded critical inter-event times
        x = self.input_proj(crit_times)   # [B, K, d_model]
        x = self.transformer(x)            # [B, K, d_model]
        x = x.mean(dim=1)                 # [B, d_model]
        return self.output_proj(x)         # [B, out_dim]


class TriModalFusionGate(nn.Module):
    """Gate network to fuse VLM, Spatial Morse, and Temporal Morse embeddings."""
    def __init__(self, dim=64):
        super().__init__()
        self.norm_vlm      = nn.LayerNorm(dim)
        self.norm_spatial  = nn.LayerNorm(dim)
        self.norm_temporal = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, vlm_emb, spatial_morse_emb, temporal_morse_emb):
        # all inputs: [N, 1, dim]
        v = self.norm_vlm(vlm_emb)
        s = self.norm_spatial(spatial_morse_emb)
        t = self.norm_temporal(temporal_morse_emb)
        weights = self.gate(torch.cat([v, s, t], dim=-1))  # [N, 1, 3]
        return weights[..., 0:1] * v + weights[..., 1:2] * s + weights[..., 2:3] * t


class KDEFusion(nn.Module):
    """Fuse MorseTransformer output with KDE critical nodes via cross-attention."""
    def __init__(self, dim=64):
        super().__init__()
        self.kde_proj = nn.Linear(3, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

    def forward(self, morse_emb, kde_nodes):
        # morse_emb: [N, 1, dim] or [n_tiles, dim]
        # kde_nodes: [M, 3] (lon_nm, lat_nm, density_nm)
        kv = self.kde_proj(kde_nodes).unsqueeze(0)  # [1, M, dim]
        if morse_emb.dim() == 2:
            q = morse_emb.unsqueeze(0)  # [1, n_tiles, dim]
        else:
            q = morse_emb
        Q = self.to_q(q)
        K = self.to_k(kv)
        V = self.to_v(kv)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) * self.scale, dim=-1)
        out = torch.matmul(attn, V)
        out = self.out_proj(out).squeeze(0)  # [n_tiles, dim]
        return self.norm(morse_emb.squeeze(0) + out)



class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, q, kv):
        # The conditioning carries a single token per event (kv is [N, 1, dim]),
        # so the softmax runs over one key and is identically 1: attn @ V == V.
        # Skipping Q/K and the two bmms is therefore bit-exact, not an
        # approximation, and it is what the 500-step sampling loop pays for --
        # six of these run per denoising step. to_q/to_k are kept so existing
        # checkpoints load unchanged.
        if kv.shape[-2] == 1:
            out = self.to_v(kv)
        else:
            Q = self.to_q(q)
            K = self.to_k(kv)
            V = self.to_v(kv)
            attn = torch.softmax(torch.bmm(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
            out = torch.bmm(attn, V)
        return self.norm(q + self.out_proj(out))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ST_Diffusion_MM(nn.Module):
    def __init__(self, n_steps, dim, num_units=64, self_condition=False,
                 condition=True, cond_dim=64, img_dim=64):
        super().__init__()
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition
        sinu_pos_emb = SinusoidalPosEmb(num_units)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(num_units, num_units), nn.GELU(), nn.Linear(num_units, num_units))
        self.linears_spatial = nn.ModuleList([
            nn.Linear(dim-1, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units)])
        self.linears_temporal = nn.ModuleList([
            nn.Linear(1, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units)])
        self.output_spatial  = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, dim-1))
        self.output_temporal = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 1))
        self.linear_t = nn.Sequential(nn.Linear(num_units*2, num_units), nn.ReLU(), nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 2))
        self.linear_s = nn.Sequential(nn.Linear(num_units*2, num_units), nn.ReLU(), nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 2))
        self.cond_all      = nn.Sequential(nn.Linear(cond_dim*3, num_units), nn.ReLU(), nn.Linear(num_units, num_units))
        self.cond_temporal = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        self.cond_spatial  = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        self.cond_joint    = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        self.img_proj = nn.Linear(img_dim, num_units)
        self.cross_attn_spatial  = nn.ModuleList([CrossAttention(num_units) for _ in range(3)])
        self.cross_attn_temporal = nn.ModuleList([CrossAttention(num_units) for _ in range(3)])

    def _split_cond(self, cond):
        h = cond.shape[-1] // 3
        return cond[:,:,:h], cond[:,:,h:2*h], cond[:,:,2*h:]

    def get_attn(self, x, t, x_self_cond=None, cond=None, img_cond=None):
        cond_merged = self.cond_all(cond)
        t_emb = self.time_mlp(t).unsqueeze(1)
        cond_all = torch.cat((cond_merged, t_emb), dim=-1)
        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(1)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(1)
        return alpha_s, alpha_t

    def forward(self, x, t, x_self_cond=None, cond=None, img_cond=None):
        x_spatial  = x[:,:,1:].clone()
        x_temporal = x[:,:,:1].clone()
        cond_temporal, cond_spatial, cond_joint = self._split_cond(cond)
        cond_merged = self.cond_all(cond)
        t_emb = self.time_mlp(t).unsqueeze(1)
        cond_all = torch.cat((cond_merged, t_emb), dim=-1)
        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(1).unsqueeze(2)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(1).unsqueeze(2)
        img_kv = self.img_proj(img_cond) if img_cond is not None else torch.zeros_like(t_emb)
        for idx in range(3):
            x_spatial  = self.linears_spatial[2*idx](x_spatial)
            x_temporal = self.linears_temporal[2*idx](x_temporal)
            x_spatial  += t_emb
            x_temporal += t_emb
            x_spatial  += self.cond_joint[idx](cond_joint) + self.cond_spatial[idx](cond_spatial)
            x_temporal += self.cond_joint[idx](cond_joint) + self.cond_temporal[idx](cond_temporal)
            x_spatial  = self.cross_attn_spatial[idx](x_spatial,  img_kv)
            x_temporal = self.cross_attn_temporal[idx](x_temporal, img_kv)
            x_spatial  = self.linears_spatial[2*idx+1](x_spatial)
            x_temporal = self.linears_temporal[2*idx+1](x_temporal)
        x_spatial  = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)
        x_output = torch.cat((x_temporal, x_spatial), dim=1)
        x_output_t = (x_output * alpha_t).sum(dim=1, keepdim=True)
        x_output_s = (x_output * alpha_s).sum(dim=1, keepdim=True)
        return torch.cat((self.output_temporal(x_output_t), self.output_spatial(x_output_s)), dim=-1)

class GaussianDiffusion_MM(nn.Module):
    def __init__(self, model, *, seq_length, timesteps=1000, sampling_timesteps=None,
                 loss_type='l2', objective='pred_noise', beta_schedule='cosine',
                 p2_loss_weight_gamma=0., p2_loss_weight_k=1, ddim_sampling_eta=1.):
        super().__init__()
        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition
        self.seq_length = seq_length
        self.objective = objective
        self.loss_type = loss_type
        betas = cosine_beta_schedule(timesteps) if beta_schedule == 'cosine' else linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.dim_weight = None
        rb = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        rb('betas', betas)
        rb('alphas_cumprod', alphas_cumprod)
        rb('alphas_cumprod_prev', alphas_cumprod_prev)
        rb('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        rb('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        rb('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        rb('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        rb('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        rb('posterior_variance', posterior_variance)
        rb('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=posterior_variance[1])))
        rb('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        rb('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        rb('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    @property
    def loss_fn(self):
        if self.loss_type == 'l2': return F.mse_loss
        if self.loss_type == 'l1': return F.l1_loss
        raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        pm  = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
               extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        pv  = extract(self.posterior_variance, t, x_t.shape)
        plv = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return pm, pv, plv

    def q_mean_variance(self, x_start, t):
        mean     = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_var  = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_var

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, cond=None, img_cond=None):
        model_output = self.model(x, t, x_self_cond, cond=cond, img_cond=img_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = maybe_clip(self.predict_start_from_noise(x, t, pred_noise))
        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True, cond=None, img_cond=None):
        preds = self.model_predictions(x, t, x_self_cond, cond=cond, img_cond=img_cond)
        x_start = preds.pred_x_start
        if clip_denoised: x_start.clamp_(-1., 1.)
        model_mean, pv, plv = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, pv, plv, x_start

    @torch.no_grad()
    def p_sample(self, x, t, x_self_cond=None, cond=None, img_cond=None):
        b = x.shape[0]
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, cond=cond, img_cond=img_cond)
        noise = torch.randn_like(x) if t > 0 else 0.
        return model_mean + (0.5 * model_log_variance).exp() * noise, x_start

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, img_cond=None, clip_denoised=True):
        batch, device = shape[0], self.betas.device
        # NOTE: MCM-DM reversed the two slices independently, which pairs each t
        # with a LARGER t_next; then 1 - alpha/alpha_next < 0 and sigma is NaN.
        # The bug was latent because sampling always ran with
        # sampling_timesteps == timesteps, i.e. the p_sample_loop branch.
        times = list(reversed(torch.linspace(-1, self.num_timesteps-1,
                                             steps=self.sampling_timesteps+1).int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised, cond=cond, img_cond=img_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            if time_next < 0:
                img = x_start; continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha/alpha_next) * (1-alpha_next) / (1-alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * torch.randn_like(img)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, img_cond=None):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, cond=cond, img_cond=img_cond)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, img_cond=None):
        fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return fn((batch_size, self.channels, self.seq_length), cond=cond, img_cond=img_cond)

    def set_dim_weight(self, w):
        """Per-dimension weights for the (dt, lon, lat) axes of the epsilon loss.

        The three axes have very different spreads once normalised: on the US
        state sets 99% of the true dt falls in the bottom 1.5% of its range, so
        an unweighted MSE is dominated by lon/lat and the model never learns to
        collapse dt. Measured on Florida, the model predicts a median dt of 8.23
        days where the truth is 0.00, and 88% of its RMSE is that bias.
        """
        self.dim_weight = None if w is None else torch.as_tensor(
            w, dtype=torch.float32, device=self.betas.device)

    def p_losses(self, x_start, t, noise=None, cond=None, img_cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t, None, cond=cond, img_cond=img_cond)
        target = noise if self.objective == 'pred_noise' else x_start
        loss = self.loss_fn(model_out, target, reduction='none')
        if getattr(self, 'dim_weight', None) is not None:
            loss = loss * self.dim_weight
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def _vb_terms_bpd(self, x_start, x_t, t, clip_denoised=True, cond=None, img_cond=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x_t, t=t, clip_denoised=clip_denoised, cond=cond, img_cond=img_cond)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl_all = mean_flat(kl) / np.log(np.e)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5*model_log_variance)
        decoder_nll_all = mean_flat(decoder_nll) / np.log(np.e)
        kl_temporal = mean_flat(kl[:,:,:1]) / np.log(np.e)
        kl_spatial  = mean_flat(kl[:,:,-(self.seq_length-1):]) / np.log(np.e)
        decoder_nll_temporal = mean_flat(decoder_nll[:,:,:1]) / np.log(np.e)
        decoder_nll_spatial  = mean_flat(decoder_nll[:,:,-(self.seq_length-1):]) / np.log(np.e)
        output          = torch.where(t==0, decoder_nll_all,      kl_all)
        output_temporal = torch.where(t==0, decoder_nll_temporal, kl_temporal)
        output_spatial  = torch.where(t==0, decoder_nll_spatial,  kl_spatial)
        return output, output_temporal, output_spatial, pred_xstart

    def _prior_bpd(self, x_start):
        b = x_start.shape[0]
        t = torch.tensor([self.num_timesteps-1]*b, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
        return (mean_flat(kl_prior) / np.log(np.e),
                mean_flat(kl_prior[:,:,:1]) / np.log(np.e),
                mean_flat(kl_prior[:,:,-(self.seq_length-1):]) / np.log(np.e))

    def NLL_cal(self, x_start, cond, img_cond=None, noise=None, clip_denoised=True):
        x_start = normalize_to_neg_one_to_one(x_start)
        device, b = x_start.device, x_start.shape[0]
        vb_all, vb_t_all, vb_s_all = [], [], []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t]*b, device=device)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                vb, vbt, vbs, _ = self._vb_terms_bpd(x_start, x_t, t_batch, clip_denoised, cond=cond, img_cond=img_cond)
            vb_all.append(vb.unsqueeze(1))
            vb_t_all.append(vbt.unsqueeze(1))
            vb_s_all.append(vbs.unsqueeze(1))
        vb_all  = torch.sum(torch.cat(vb_all,  dim=-1), dim=-1)
        vb_t_all = torch.sum(torch.cat(vb_t_all, dim=-1), dim=-1)
        vb_s_all = torch.sum(torch.cat(vb_s_all, dim=-1), dim=-1)
        prior_all, prior_t, prior_s = self._prior_bpd(x_start)
        return (vb_all+prior_all).sum().item(), (vb_t_all+prior_t).sum().item(), (vb_s_all+prior_s).sum().item()

    def forward(self, img, cond, img_cond=None, *args, **kwargs):
        b, c, n, device = *img.shape, img.device
        assert n == self.seq_length
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, cond=cond, img_cond=img_cond, *args, **kwargs)

class Model_all_MM(nn.Module):
    def __init__(self, transformer, diffusion, img_projector, proj_dim=128):
        super().__init__()
        self.transformer   = transformer
        self.diffusion     = diffusion
        self.img_projector = img_projector
        # MCL projection heads: 把 history 和 image embedding 映射到同一对比学习空间
        self.hist_proj = nn.Sequential(
            nn.Linear(64, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.img_proj_mcl = nn.Sequential(
            nn.Linear(64, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
