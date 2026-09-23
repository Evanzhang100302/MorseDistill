"""Distil the MCM-DM history transformer into an attention-free MLP.

Teacher: Transformer_ST (frozen, loaded from train_teacher.py checkpoints)
Student: MLPHistoryEncoder, everything downstream initialised from the teacher.

Losses
  L_diff      student's own diffusion epsilon loss
  L_kd_out    match the teacher's predicted epsilon on the SAME (x_t, t)
  L_kd_cond   match the teacher's history embedding (per stream: t / loc / out)
  L_kd_fuse   match the teacher's fused VLM + ST-Morse condition
  L_kd_pred   PREDICTION level: match what the teacher's noise prediction implies
              about the event itself, x0_hat, in 2-norm
  L_cnodes_Y    prediction level restricted to the Morse complex's critical cells
  L_cnodes_H    feature level restricted the same way
  L_kd_rel    match the teacher's pairwise-distance STRUCTURE (RKD-D), not its
              values -- a scale-invariant constraint that, unlike the three
              element-wise terms above, does not pin the student to the teacher
  L_topo_st   unified ST Morse topology anchor
  L_mcl       history <-> condition contrastive term (as in the teacher)
"""
import os, pickle, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from einops import reduce
import setproctitle

from transformer_st import Transformer_ST, get_dataloader
from model_student import MLPHistoryEncoder
from model_mm import (ST_Diffusion_MM, GaussianDiffusion_MM, Model_all_MM,
                      ImageProjector, STMorseEncoder, ModalFusionGate,
                      normalize_to_neg_one_to_one, extract)
from st_morse import lat_lon_to_tile, dt_bin


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',          type=int, default=1234)
    p.add_argument('--dataset',       type=str, required=True)
    p.add_argument('--teacher_dir',   type=str, required=True)
    p.add_argument('--total_epochs',  type=int, default=100)
    p.add_argument('--warmup_cond_epochs', type=int, default=5,
                   help='stage A: encoder-only pretraining on L_kd_cond')
    p.add_argument('--window',        type=int, default=16, help='student causal window')
    p.add_argument('--hidden',        type=int, default=256)
    p.add_argument('--n_hidden',      type=int, default=2)
    p.add_argument('--lambda_out',    type=float, default=1.0)
    p.add_argument('--lambda_cond',   type=float, default=1.0)
    p.add_argument('--lambda_fuse',   type=float, default=0.5)
    p.add_argument('--lambda_pred',   type=float, default=0.0,
                   help='prediction-level KD: 2-norm between the teacher and student estimates of\n'
                        'the clean event x0_hat (0 = off)')
    p.add_argument('--lambda_rel',    type=float, default=0.0,
                   help='relation-based KD on the history embedding (0 = off, the pre-2026-08-24 behaviour)')
    p.add_argument('--rel_max_n',     type=int, default=512,
                   help='anchors subsampled per batch for the O(n^2) distance matrix')
    p.add_argument('--lambda_ms',     type=float, default=0.0,
                   help='multi-scale KD on the full [B,L,64] history sequence (0 = off)')
    p.add_argument('--ms_scales',     type=int, default=3)
    p.add_argument('--ms_window',     type=int, default=2)
    p.add_argument('--lambda_cnodes_y', type=float, default=0.0,
                   help='squared 2-norm between teacher/student x0_hat, over events in a\n'
                        'critical ST-Morse cell only (0 = off)')
    p.add_argument('--lambda_cnodes_h', type=float, default=0.0,
                   help='squared 2-norm between teacher/student STMorseEncoder embeddings\n'
                        'at the critical cells (0 = off)')
    p.add_argument('--cnodes_reduce', type=str, default='sum', choices=['sum', 'mean'],
                   help="L_cnodes reduction. 'sum' follows the equation literally; 'mean' "
                        "removes the dependence on how many critical-cell events a batch "
                        "happens to hold, so lambda becomes dataset-independent.")
    p.add_argument('--cnodes_tmax',   type=float, default=1.0,
                   help='apply L_cnodes_Y only to samples with t < tmax*T; x0_hat is '
                        'unreliable near t=T where 1/sqrt(alpha_bar_t) blows up')
    p.add_argument('--alpha',         type=float, default=0.05)
    p.add_argument('--beta_topo_st',  type=float, default=0.001)
    p.add_argument('--topo_w_time',   type=float, default=-1.0)
    p.add_argument('--no_kd',         action='store_true', help='ablation: train the MLP from scratch')
    p.add_argument('--dim_weight',    action='store_true',
                   help='scale the (dt, lon, lat) axes of the diffusion loss by 1/std, measured on the\n                        training split in the [-1,1] space the loss lives in. Without it the two\n                        spatial axes dominate and dt is never learned (0 = off).')
    p.add_argument('--no_st_morse',   action='store_true',
                   help='ablation: zero out the ST-Morse branch of the condition')
    p.add_argument('--dual_morse',    action='store_true',
                   help='two ST-Morse branches fused by the gate: one over ALL cells (indexed '
                        'by the event own cell) and one over the CRITICAL cells only (indexed '
                        'by the nearest critical cell). Enables the two Morse-guided losses '
                        'and drops L_fuse.')
    p.add_argument('--lambda_mpred',  type=float, default=1.0,
                   help='--dual_morse: weight of ||eps_S - eps_T||^2 restricted to events that '
                        'sit in a critical cell (Morse-guided prediction level)')
    p.add_argument('--lambda_mfeat',  type=float, default=1.0,
                   help='--dual_morse: weight of ||e_crit_S - e_crit_T||^2 (Morse-guided '
                        'feature level)')
    p.add_argument('--no_vlm',        action='store_true',
                   help='ablation: drop the VLM tile embedding, leaving ST Morse as the only\n'
                        'conditioning modality. Must match how the teacher was trained.')
    p.add_argument('--no_warmstart',  action='store_true',
                   help='keep every KD loss but do NOT copy the teacher weights into the downstream\n'
                        'modules. Useful where the teacher itself is the weak part: the warm start\n'
                        'then hands the student the teacher\'s bad solution to start from.')
    p.add_argument('--freeze_head',   action='store_true',
                   help='ablation: train only the student encoder, keep teacher head frozen')
    p.add_argument('--batch_size',    type=int, default=64)
    p.add_argument('--timesteps',     type=int, default=500)
    p.add_argument('--samplingsteps', type=int, default=500)
    p.add_argument('--objective',     type=str, default='pred_noise')
    p.add_argument('--loss_type',     type=str, default='l2')
    p.add_argument('--beta_schedule', type=str, default='cosine')
    p.add_argument('--cuda_id',       type=str, default='0')
    p.add_argument('--img_dim',       type=int, default=64)
    p.add_argument('--zoom',          type=int, default=7)
    p.add_argument('--emb_file',      type=str, required=True)
    p.add_argument('--st_morse_file', type=str, required=True)
    p.add_argument('--st_in_dim',     type=int, default=2)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--max_val_seqs',  type=int, default=200)
    p.add_argument('--eval_every',    type=int, default=10)
    p.add_argument('--cond_from_history', action='store_true',
                   help='index VLM tile / ST cell by the last observed event, not the target; '
                        'must match how the teacher was trained')
    p.add_argument('--exp_name',      type=str, default='student')
    args = p.parse_args()
    args.cuda = torch.cuda.is_available()
    args.dim  = 2
    return args


opt = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)
device = torch.device('cuda:0' if opt.cuda else 'cpu')


def mcl_loss(hist_emb, img_emb, temperature=0.1):
    h = F.normalize(hist_emb, dim=-1)
    v = F.normalize(img_emb,  dim=-1)
    sim = torch.mm(h, v.T) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def setup_init():
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    if opt.cuda: torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalization(x, MAX, MIN): return (x - MIN) / (MAX - MIN)


def load_data():
    data_root = f'dataset/{opt.dataset}'
    def read(split):
        with open(f'{data_root}/{split}.pkl', 'rb') as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        # The first event of a sequence has no predecessor, so upstream filled this
        # column with its ABSOLUTE time. That value is never read -- targets are
        # indexed at tgt>=1 and the ST Morse lookup at c=tgt -- but it does set
        # MAX[1], and on Wildfire_CA it inflates the dt range from 1 day to 3995,
        # squeezing every real gap into 0.025% of the [0,1] span. Use 0 instead.
        d = [[[i[0], i[0]-u[idx-1][0] if idx > 0 else 0.0] + i[1:]
              for idx, i in enumerate(u)] for u in d]
        return d
    train_data, val_data, test_data = read('data_train'), read('data_val'), read('data_test')
    data_all = train_data + val_data + test_data
    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1); Min.append(0)
    def norm(d):
        return [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in d]
    val_data = val_data[:opt.max_val_seqs]
    return (get_dataloader(norm(train_data), opt.batch_size, D=opt.dim, shuffle=True),
            get_dataloader(norm(val_data),   min(len(val_data), 1000),  D=opt.dim, shuffle=False),
            get_dataloader(norm(test_data),  min(len(test_data), 1000), D=opt.dim, shuffle=False),
            (Max, Min), data_all)


def build_emb_lookup(data_all, emb_cache, zoom):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw, lat_raw = float(event[2]), float(event[3])
            key = (zoom,) + lat_lon_to_tile(lat_raw, lon_raw, zoom)
            if key in emb_cache:
                val = emb_cache[key]
                lookup[(round(lon_raw, 5), round(lat_raw, 5))] = val["embedding"] if isinstance(val, dict) else val
    return lookup


def batch_to_model(batch, encoders, emb_lookup, st, MAX, MIN):
    """Same event-pair extraction as the teacher script, but runs several history
    encoders on the identical batch so their conditions are directly comparable."""
    event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
    event_loc = torch.cat((lng.unsqueeze(2), lat.unsqueeze(2)), dim=-1)

    encs, mask = [], None
    for enc in encoders:
        e, m = enc(event_loc, event_time_origin)
        encs.append(e)
        mask = m
    B = mask.shape[0]

    st_zoom, st_edges, st_key2idx, st_miss = st['zoom'], st['edges'], st['key2idx'], st['miss_idx']
    bi, ei, ti, img_l, cell_l = [], [], [], [], []
    zero_emb = torch.zeros(1536)
    for b in range(B):
        length = int(mask[b].sum().item())
        if length <= 1:
            continue
        lng_raw = (lng[b, :length].cpu() * (MAX[2]-MIN[2])) + MIN[2]
        lat_raw = (lat[b, :length].cpu() * (MAX[3]-MIN[3])) + MIN[3]
        dt_nm   = event_time[b, :length].cpu()
        for e in range(length - 1):
            tgt = e + 1
            bi.append(b); ei.append(e); ti.append(tgt)
            c = e if opt.cond_from_history else tgt      # which event the condition is read at
            lon_k, lat_k = round(lng_raw[c].item(), 5), round(lat_raw[c].item(), 5)
            img_l.append(emb_lookup.get((lon_k, lat_k), zero_emb).unsqueeze(0))
            tx, ty = lat_lon_to_tile(lat_raw[c].item(), lng_raw[c].item(), st_zoom)
            cell_l.append(st_key2idx.get((tx, ty, dt_bin(dt_nm[c].item(), st_edges)), st_miss))

    if not bi:
        return (None,) * 5 + ((None, None),)
    bi = torch.tensor(bi, device=device); ei = torch.tensor(ei, device=device)
    ti = torch.tensor(ti, device=device)
    conds = [e[bi, ei].unsqueeze(1) for e in encs]
    return (event_time[bi, ti].reshape(-1, 1, 1),
            event_loc[bi, ti].reshape(-1, 1, opt.dim),
            conds,
            torch.cat(img_l, 0).unsqueeze(1).to(device),
            torch.tensor(cell_l, dtype=torch.long, device=device),
            (encs, mask))


def multiscale_seq_loss(e_s, e_t, mask, n_scales, window):
    """Multi-Scale Distillation (TimeDistill), adapted to an event sequence.

    Average-pools the history representation along the event axis and matches
    student to teacher at every resolution. This stays an MSE, so unlike
    rkd_dist_loss it preserves the teacher's coordinate system -- which matters
    here because the denoiser is warm-started from the teacher and reads `cond`
    in that frame -- while the coarse scales stop demanding that the student
    reproduce the teacher's per-event noise.

    Padding is handled by pooling the mask alongside the features: dividing the
    pooled features by the pooled mask turns AvgPool's fixed 1/window into a
    mean over the valid positions only.
    """
    m = mask.to(e_s.dtype)                             # encoders return [B, L, 1]
    if m.dim() == 2:
        m = m.unsqueeze(-1)
    s = (e_s * m).transpose(1, 2)                      # [B, C, L]
    t = (e_t * m).transpose(1, 2)
    w = m.transpose(1, 2)                              # [B, 1, L]

    def masked_mse(a, b, valid):
        return ((a - b) ** 2 * valid).sum() / valid.sum().clamp(min=1.) / a.shape[1]

    total = masked_mse(s, t, w)
    pool = nn.AvgPool1d(kernel_size=window, ceil_mode=True)
    for _ in range(n_scales):
        if s.shape[-1] < window:
            break
        s, t, w = pool(s), pool(t), pool(w)
        valid = (w > 1e-6).to(s.dtype)
        denom = w.clamp(min=1e-6)
        total = total + masked_mse(s / denom, t / denom, valid)
    return total / (n_scales + 1)


def cnodes_y_loss(diffusion, x_t, t, eps_S, eps_T, crit_mask, reduce='sum', t_cut=None):
    """L_cnodes_Y  (L^{Y;Morse}) -- prediction level, critical cells only.

        || Y_hat_teacher[critical] - Y_hat_student[critical] ||^2

    Y_hat is x0_hat, the model's estimate of the event (dt, lon, lat), taken at
    the events whose ST cell the Morse complex marks critical. Unlike the plain
    prediction-level term, restricting to a SUBSET changes which samples are
    matched rather than merely reweighting them by timestep, so this carries
    information L_kd_out does not.

    The norm is a SUM of squares over the selected critical-cell events, as in
    the equation -- not a mean. That makes the term's magnitude grow with how
    many critical-cell events a batch happens to contain, so lambda_cnodes_y has
    to be set far below 1 to keep it comparable with L_kd_out.
    """
    if crit_mask.sum() == 0:
        return torch.zeros((), device=x_t.device)
    # clamp to the data range, as ddim_trajectory does at sampling time: x0_hat
    # carries a 1/sqrt(a_t) factor that diverges as t -> T, and unclamped the
    # summed norm reaches 1e6 and swamps every other term.
    sel = crit_mask
    if t_cut is not None:
        sel = sel & (t < t_cut)
        if sel.sum() == 0:
            return torch.zeros((), device=x_t.device)
    y_S = diffusion.predict_start_from_noise(x_t, t, eps_S).clamp(-1., 1.)[sel]
    y_T = diffusion.predict_start_from_noise(x_t, t, eps_T).clamp(-1., 1.)[sel]
    d2 = (y_S - y_T) ** 2
    return d2.mean() if reduce == 'mean' else d2.sum()


def cnodes_h_loss(H_S, H_T, crit_idx, reduce='sum'):
    """L_cnodes_H  (L^{H;Morse}) -- feature level, critical cells only.

        || H_teacher[critical] - H_student[critical] ||^2

    H is the STMorseEncoder's per-cell embedding. Teacher and student share the
    encoder architecture and width, so no Regressor (Eq. 5) is needed to align
    dimensions. Summed, not averaged: the critical-cell count is fixed per
    dataset, so this is a constant rescaling of lambda_cnodes_h -- but the
    constant is large (n_crit x 64), so lambda must be scaled down to match.
    """
    if crit_idx.numel() == 0:
        return torch.zeros((), device=H_S.device)
    d2 = (H_S[crit_idx] - H_T[crit_idx]) ** 2
    return d2.mean() if reduce == 'mean' else d2.sum()


def pred_level_loss(diffusion, x_t, t, eps_S, eps_T):
    """Prediction-level KD: match the teacher on the EVENT, not on the noise.

    L_kd_out compares the two noise predictions; this compares what those
    predictions imply about the clean event,
        x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t),
    which is the quantity the sampler actually emits and the metrics score.

    The two differ only by the factor sqrt((1 - a_t) / a_t), but that factor
    diverges as t -> T (a_T ~ 0 under the cosine schedule), so an unclamped
    2-norm here is dominated by the noisiest timesteps. Both estimates are
    therefore clamped to the [-1, 1] data range first -- exactly what
    ddim_trajectory does at inference, so the loss scores the model in the
    space it is actually evaluated in.
    """
    x0_S = diffusion.predict_start_from_noise(x_t, t, eps_S).clamp(-1., 1.)
    x0_T = diffusion.predict_start_from_noise(x_t, t, eps_T).clamp(-1., 1.)
    return torch.linalg.vector_norm(x0_S - x0_T, ord=2, dim=-1).mean()


def rkd_dist_loss(f_s, f_t, max_n):
    """Relation-based KD (Park et al. 2019, RKD-D).

    Aligns the geometry of the batch -- the matrix of pairwise distances --
    instead of the embeddings themselves. Each matrix is divided by its own
    off-diagonal mean, so the loss is blind to a global rescaling of either
    feature space. That is deliberately much looser than the element-wise MSE
    in L_kd_cond: the student has to reproduce which events the teacher places
    near each other, not the teacher's exact coordinates (and therefore not the
    teacher's per-event error).
    """
    s = f_s.reshape(f_s.shape[0], -1)
    t = f_t.reshape(f_t.shape[0], -1)
    if s.shape[0] > max_n:                       # the O(n^2) matrix is the cost here
        idx = torch.randperm(s.shape[0], device=s.device)[:max_n]
        s, t = s[idx], t[idx]
    n = s.shape[0]
    if n < 2:
        return torch.zeros((), device=s.device)
    d_s, d_t = torch.cdist(s, s), torch.cdist(t, t)
    off = n * n - n                              # the diagonal is exactly zero
    d_s = d_s / (d_s.sum() / off + 1e-8)
    d_t = d_t / (d_t.sum() / off + 1e-8)
    return F.smooth_l1_loss(d_s, d_t)


def st_topo_loss(pred_xstart, crit_pm1, w_time):
    p = pred_xstart[:, 0, :3].clone(); p[:, 0] = p[:, 0] * w_time
    c = crit_pm1.clone();              c[:, 0] = c[:, 0] * w_time
    return torch.cdist(p.unsqueeze(0), c.unsqueeze(0)).squeeze(0).min(dim=-1).values.mean()


if __name__ == '__main__':
    setup_init()
    setproctitle.setproctitle(f'KD-student-{opt.dataset}')

    emb_cache = torch.load(opt.emb_file, map_location='cpu', weights_only=False)
    stf  = torch.load(opt.st_morse_file, map_location='cpu', weights_only=False)
    meta = stf['meta']
    trainloader, valloader, testloader, (MAX, MIN), data_all = load_data()
    for i in (1, 2, 3):
        assert abs(meta['MAX'][i] - MAX[i]) < 1e-6 and abs(meta['MIN'][i] - MIN[i]) < 1e-6

    emb_lookup = build_emb_lookup(data_all, emb_cache, opt.zoom)
    n_cells = len(stf['cell_keys'])
    cell_features = torch.cat([stf['cell_features'][:, :opt.st_in_dim],
                               torch.zeros(1, opt.st_in_dim)], dim=0).to(device)
    st = {'zoom': meta['zoom'], 'edges': meta['dt_edges'],
          'key2idx': {k: i for i, k in enumerate(stf['cell_keys'])}, 'miss_idx': n_cells}
    if opt.dim_weight:
        cols = []
        for _mi in (1, 2, 3):
            _v = np.array([i[_mi] for u in data_all for i in u], dtype=np.float64)
            _v = (_v - MIN[_mi]) / (MAX[_mi] - MIN[_mi]) * 2 - 1
            cols.append(max(_v.std(), 1e-6))
        dim_w = torch.tensor([1.0 / c for c in cols], dtype=torch.float32, device=device)
        dim_w = dim_w / dim_w.mean()
        print(f'[dim_weight] per-axis std (dt, lon, lat) = ({cols[0]:.4f}, {cols[1]:.4f}, '
              f'{cols[2]:.4f})  ->  weights = ({dim_w[0]:.3f}, {dim_w[1]:.3f}, {dim_w[2]:.3f})', flush=True)
    else:
        dim_w = None
    crit_pm1 = stf['critical_centers'].to(device) * 2.0 - 1.0
    # indices into cell_features of the cells the Morse complex marks critical.
    # cell_features has one extra padded row (miss_idx) for events whose cell is
    # absent from the complex; that row is never critical.
    crit_idx = torch.nonzero(stf['is_critical'].to(device) > 0.5, as_tuple=False).squeeze(-1)
    is_crit_cell = torch.cat([stf['is_critical'].to(device) > 0.5,
                              torch.zeros(1, dtype=torch.bool, device=device)])
    w_time = meta['w_time'] if opt.topo_w_time < 0 else opt.topo_w_time

    # --dual_morse: a second branch whose input set is the critical cells alone. Each cell is
    # re-indexed to its nearest critical cell under the same anisotropic metric the k-NN graph
    # was built with, so an event in a non-critical cell still receives the embedding of the
    # hotspot it belongs to. Mirrors the block in train_teacher.py exactly.
    crit_features, cell2crit = None, None
    if opt.dual_morse:
        _C = stf['cell_centers'].clone().float(); _C[:, 0] *= w_time
        cell2crit = torch.cat([torch.cdist(_C, _C[crit_idx.cpu()]).argmin(dim=-1),
                               torch.tensor([crit_idx.numel()])]).to(device)
        crit_features = torch.cat([stf['cell_features'][crit_idx.cpu(), :opt.st_in_dim],
                                   torch.zeros(1, opt.st_in_dim)], dim=0).to(device)
        print(f'[dual_morse] branch A: {n_cells} cells   branch B: {crit_idx.numel()} critical '
              f'cells   (L_fuse disabled)', flush=True)

    mk_diffusion = lambda: GaussianDiffusion_MM(
        ST_Diffusion_MM(n_steps=opt.timesteps, dim=1+opt.dim, condition=True,
                        cond_dim=64, img_dim=opt.img_dim).to(device),
        seq_length=1+opt.dim, timesteps=opt.timesteps, sampling_timesteps=opt.samplingsteps,
        loss_type=opt.loss_type, objective=opt.objective, beta_schedule=opt.beta_schedule).to(device)

    # ── teacher (frozen) ────────────────────────────────────────────────────
    T_enc = Transformer_ST(d_model=64, d_rnn=256, d_inner=128, n_layers=4, n_head=4,
                           d_k=16, d_v=16, dropout=0.1, device=device,
                           loc_dim=opt.dim, CosSin=True).to(device)
    T = Model_all_MM(T_enc, mk_diffusion(), ImageProjector(1536, opt.img_dim).to(device))
    T.hist_proj, T.img_proj_mcl = T.hist_proj.to(device), T.img_proj_mcl.to(device)
    T.load_state_dict(torch.load(opt.teacher_dir + '/model_best.pkl', map_location=device))
    T_st   = STMorseEncoder(in_dim=opt.st_in_dim, out_dim=opt.img_dim).to(device)
    T_gate = ModalFusionGate(dim=opt.img_dim).to(device)
    T_st.load_state_dict(torch.load(opt.teacher_dir + '/st_morse_best.pkl', map_location=device))
    T_gate.load_state_dict(torch.load(opt.teacher_dir + '/gate_best.pkl', map_location=device))
    T_st_crit = None
    if opt.dual_morse:
        T_st_crit = STMorseEncoder(in_dim=opt.st_in_dim, out_dim=opt.img_dim).to(device)
        T_st_crit.load_state_dict(
            torch.load(opt.teacher_dir + '/st_morse_crit_best.pkl', map_location=device))
    for m in (T, T_st, T_gate) + ((T_st_crit,) if opt.dual_morse else ()):
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad_(False)

    # ── student: MLP encoder, everything downstream warm-started from teacher ─
    S_enc = MLPHistoryEncoder(d_model=64, window=opt.window, hidden=opt.hidden,
                              n_hidden=opt.n_hidden, dropout=0.1,
                              device=device, loc_dim=opt.dim).to(device)
    S = Model_all_MM(S_enc, mk_diffusion(), ImageProjector(1536, opt.img_dim).to(device))
    S.hist_proj, S.img_proj_mcl = S.hist_proj.to(device), S.img_proj_mcl.to(device)
    S_st   = STMorseEncoder(in_dim=opt.st_in_dim, out_dim=opt.img_dim).to(device)
    S_gate = ModalFusionGate(dim=opt.img_dim).to(device)
    S_st_crit = (STMorseEncoder(in_dim=opt.st_in_dim, out_dim=opt.img_dim).to(device)
                 if opt.dual_morse else None)
    if not opt.no_kd and not opt.no_warmstart:
        tsd = torch.load(opt.teacher_dir + '/model_best.pkl', map_location=device)
        S.load_state_dict({k: v for k, v in tsd.items() if not k.startswith('transformer.')},
                          strict=False)
        S_st.load_state_dict(torch.load(opt.teacher_dir + '/st_morse_best.pkl', map_location=device))
        S_gate.load_state_dict(torch.load(opt.teacher_dir + '/gate_best.pkl', map_location=device))
        if opt.dual_morse:
            S_st_crit.load_state_dict(
                torch.load(opt.teacher_dir + '/st_morse_crit_best.pkl', map_location=device))

    n_par = lambda m: sum(p.numel() for p in m.parameters())
    _extra = n_par(S_st_crit) if opt.dual_morse else 0
    tot_T = n_par(T) + n_par(T_st) + n_par(T_gate) + _extra
    tot_S = n_par(S) + n_par(S_st) + n_par(S_gate) + _extra
    print(f'[teacher] encoder={n_par(T_enc):,}  total={tot_T:,}')
    print(f'[student] encoder={n_par(S_enc):,}  total={tot_S:,}  '
          f'(encoder {n_par(T_enc)/n_par(S_enc):.1f}x smaller, total {tot_T/tot_S:.2f}x)')
    print(f'[cfg] window={opt.window} kd={not opt.no_kd} warmstart={not (opt.no_kd or opt.no_warmstart)} '
          f'freeze_head={opt.freeze_head} '
          f'ST cells={n_cells} critical={crit_pm1.shape[0]}')

    head_params = [p_ for m in (S.diffusion, S.img_projector, S.hist_proj, S.img_proj_mcl) for p_ in m.parameters()] \
                  + list(S_st.parameters()) + list(S_gate.parameters()) \
                  + (list(S_st_crit.parameters()) if opt.dual_morse else [])
    if opt.freeze_head:
        for p_ in head_params:
            p_.requires_grad_(False)
        groups = [{"params": S_enc.parameters(), "lr": opt.lr}]
    else:
        groups = [{"params": S_enc.parameters(), "lr": opt.lr},
                  {"params": head_params,        "lr": opt.lr / 10}]
    optimizer = AdamW(groups, betas=(0.9, 0.99))

    model_path = f'./checkpoints/{opt.dataset}_{opt.exp_name}_seed{opt.seed}/'
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'./logs/{opt.dataset}_{opt.exp_name}_seed{opt.seed}', flush_secs=5)

    def make_cond(img_nm, cell_idx, st_enc, gate, projector, st_enc_crit=None):
        """Returns (fused condition, all-cell branch, critical branch).

        Under --dual_morse the gate fuses the two Morse branches and the VLM slot is
        unused; otherwise it fuses the VLM projection with the single Morse branch, and
        the third return value is None.
        """
        img_proj = (torch.zeros(img_nm.shape[0], 1, opt.img_dim, device=device)
                    if opt.no_vlm else projector(img_nm))
        st_emb = (torch.zeros(img_proj.shape[0], 1, opt.img_dim, device=device)
                  if opt.no_st_morse else st_enc(cell_features)[cell_idx].unsqueeze(1))
        if opt.dual_morse:
            crit_emb = st_enc_crit(crit_features)[cell2crit[cell_idx]].unsqueeze(1)
            return gate(st_emb, crit_emb), st_emb, crit_emb
        return gate(img_proj, st_emb), st_emb, None

    step, early_stop, min_loss_val = 0, 0, 1e20
    warmup_steps = 5

    import time as _t
    for itr in range(opt.total_epochs):
        _ep0 = _t.perf_counter()
        if opt.cuda: torch.cuda.reset_peak_memory_stats(device)
        stage_a = itr < opt.warmup_cond_epochs and not opt.no_kd

        if itr % opt.eval_every == 0:
            S.eval(); S_st.eval(); S_gate.eval()
            if opt.dual_morse: S_st_crit.eval()
            with torch.no_grad():
                loss_val, dist_s, total_num = 0., 0., 0
                for batch in valloader:
                    t_nm, l_nm, conds, img_nm, cell_idx, (seqs, seq_mask) = batch_to_model(
                        batch, [S.transformer], emb_lookup, st, MAX, MIN)
                    if t_nm is None: continue
                    imgc, _, _ = make_cond(img_nm, cell_idx, S_st, S_gate,
                                           S.img_projector, S_st_crit)
                    x_input = torch.cat((t_nm, l_nm), dim=-1)
                    loss_val += S.diffusion(x_input, conds[0], img_cond=imgc).item() * t_nm.shape[0]
                    sampled = S.diffusion.sample(batch_size=t_nm.shape[0], cond=conds[0], img_cond=imgc)
                    scale = torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])
                    real = (l_nm[:, 0, :2].cpu() + torch.tensor([MIN[2:]])) * scale
                    gen  = (sampled[:, 0, 1:3].cpu() + torch.tensor([MIN[2:]])) * scale
                    dist_s += torch.sqrt(((real-gen)**2).sum(dim=-1)).sum().item()
                    total_num += t_nm.shape[0]
                if total_num > 0:
                    lv, ds = loss_val / total_num, dist_s / total_num
                    print(f'  [val] epoch={itr} loss={lv:.4f} dist_spatial={ds:.4f}', flush=True)
                    writer.add_scalar('Eval/loss_val', lv, itr)
                    writer.add_scalar('Eval/distance_spatial', ds, itr)
                    if loss_val > min_loss_val:
                        early_stop += 1
                    else:
                        early_stop = 0
                        torch.save(S.state_dict(),      model_path + 'model_best.pkl')
                        torch.save(S_st.state_dict(),   model_path + 'st_morse_best.pkl')
                        torch.save(S_gate.state_dict(), model_path + 'gate_best.pkl')
                        if opt.dual_morse:
                            torch.save(S_st_crit.state_dict(),
                                       model_path + 'st_morse_crit_best.pkl')
                    min_loss_val = min(min_loss_val, loss_val)

        base_lr = opt.lr*(itr+1)/warmup_steps if itr < warmup_steps \
                  else opt.lr - (opt.lr-5e-5)*(itr-warmup_steps)/opt.total_epochs
        for gi, gp in enumerate(optimizer.param_groups):
            gp['lr'] = base_lr if gi == 0 else base_lr / 10

        S.train(); S_st.train(); S_gate.train()
        if opt.dual_morse: S_st_crit.train()
        agg = {k: 0. for k in ('diff', 'out', 'cond', 'fuse', 'pred', 'cnY', 'cnH',
                               'rel', 'ms', 'topo', 'mcl', 'mpred', 'mfeat')}
        total_num = 0
        for batch in trainloader:
            t_nm, l_nm, conds, img_nm, cell_idx, (seqs, seq_mask) = batch_to_model(
                batch, [S.transformer, T.transformer], emb_lookup, st, MAX, MIN)
            if t_nm is None: continue
            cond_S, cond_T = conds

            if opt.no_kd:
                L_cond = L_rel = L_ms = torch.tensor(0., device=device)
            else:
                L_cond = F.mse_loss(cond_S, cond_T)
                L_rel  = (rkd_dist_loss(cond_S, cond_T.detach(), opt.rel_max_n)
                          if opt.lambda_rel > 0 else torch.tensor(0., device=device))
                L_ms   = (multiscale_seq_loss(seqs[0], seqs[1].detach(), seq_mask,
                                              opt.ms_scales, opt.ms_window)
                          if opt.lambda_ms > 0 else torch.tensor(0., device=device))
            if stage_a:                                  # stage A: encoder only
                loss = L_cond + opt.lambda_rel * L_rel + opt.lambda_ms * L_ms
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(S_enc.parameters(), 1.)
                optimizer.step()
                agg['cond'] += L_cond.item() * t_nm.shape[0]
                agg['rel']  += L_rel.item()  * t_nm.shape[0]
                agg['ms']   += L_ms.item()   * t_nm.shape[0]
                total_num += t_nm.shape[0]; step += 1
                continue

            imgc_S, eall_S, ecrit_S = make_cond(img_nm, cell_idx, S_st, S_gate,
                                                S.img_projector, S_st_crit)
            x_input = torch.cat((t_nm, l_nm), dim=-1)
            x_pm1   = normalize_to_neg_one_to_one(x_input)
            t_rand  = torch.randint(0, opt.timesteps, (x_pm1.shape[0],), device=device)
            noise   = torch.randn_like(x_pm1)
            x_t     = S.diffusion.q_sample(x_pm1, t_rand, noise=noise)

            eps_S = S.diffusion.model(x_t, t_rand, None, cond=cond_S, img_cond=imgc_S)
            se = F.mse_loss(eps_S, noise, reduction='none')
            if dim_w is not None:
                se = se * dim_w
            l = reduce(se, 'b ... -> b (...)', 'mean')
            L_diff = (l * extract(S.diffusion.p2_loss_weight, t_rand, l.shape)).mean()

            if opt.no_kd:
                L_out = L_fuse = L_pred = L_cnY = L_cnH = torch.tensor(0., device=device)
                L_mpred = L_mfeat = torch.tensor(0., device=device)
            else:
                with torch.no_grad():
                    imgc_T, eall_T, ecrit_T = make_cond(img_nm, cell_idx, T_st, T_gate,
                                                        T.img_projector, T_st_crit)
                    eps_T  = T.diffusion.model(x_t, t_rand, None, cond=cond_T, img_cond=imgc_T)
                L_out  = F.mse_loss(eps_S, eps_T)
                if opt.dual_morse:
                    # feature level splits in two: the all-cell branch joins L_cond as the
                    # plain KD feature term, the critical branch is the Morse-guided one.
                    L_cond  = L_cond + F.mse_loss(eall_S, eall_T)
                    L_mfeat = F.mse_loss(ecrit_S, ecrit_T)
                    # prediction level, restricted to events sitting in a critical cell
                    _m = is_crit_cell[cell_idx]
                    L_mpred = (F.mse_loss(eps_S[_m], eps_T[_m]) if _m.any()
                               else torch.tensor(0., device=device))
                    L_fuse = torch.tensor(0., device=device)      # superseded by the split
                else:
                    L_mpred = L_mfeat = torch.tensor(0., device=device)
                    L_fuse = F.mse_loss(imgc_S, imgc_T)
                L_pred = (pred_level_loss(S.diffusion, x_t, t_rand, eps_S, eps_T)
                          if opt.lambda_pred > 0 else torch.tensor(0., device=device))
                if opt.lambda_cnodes_y > 0:
                    L_cnY = cnodes_y_loss(S.diffusion, x_t, t_rand, eps_S, eps_T,
                                       is_crit_cell[cell_idx], opt.cnodes_reduce,
                                       None if opt.cnodes_tmax >= 1.0
                                       else int(opt.cnodes_tmax * opt.timesteps))
                else:
                    L_cnY = torch.tensor(0., device=device)
                if opt.lambda_cnodes_h > 0:
                    with torch.no_grad():
                        H_T = T_st(cell_features)
                    L_cnH = cnodes_h_loss(S_st(cell_features), H_T, crit_idx, opt.cnodes_reduce)
                else:
                    L_cnH = torch.tensor(0., device=device)

            if opt.alpha > 0:
                L_mcl = mcl_loss(S.hist_proj(cond_S[:, 0, :64]),
                                 S.img_proj_mcl(imgc_S[:, 0, :]))
            else:
                L_mcl = torch.tensor(0., device=device)

            if opt.beta_topo_st > 0 and crit_pm1.shape[0] > 0:
                pred_x0 = S.diffusion.predict_start_from_noise(x_t, t_rand, eps_S)
                L_topo  = st_topo_loss(pred_x0, crit_pm1, w_time)
            else:
                L_topo = torch.tensor(0., device=device)

            loss = (L_diff
                    + opt.lambda_out  * L_out
                    + opt.lambda_cond * L_cond
                    + opt.lambda_rel  * L_rel
                    + opt.lambda_ms   * L_ms
                    + opt.lambda_fuse * L_fuse
                    + opt.lambda_pred * L_pred
                    + opt.lambda_cnodes_y * L_cnY
                    + opt.lambda_cnodes_h * L_cnH
                    + opt.lambda_mpred * L_mpred
                    + opt.lambda_mfeat * L_mfeat
                    + opt.alpha       * L_mcl
                    + opt.beta_topo_st * L_topo)

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for g in optimizer.param_groups for p_ in g['params']], 1.)
            optimizer.step()

            n = t_nm.shape[0]; total_num += n
            for k, v in zip(('diff', 'out', 'cond', 'fuse', 'pred', 'cnY', 'cnH',
                             'rel', 'ms', 'topo', 'mcl', 'mpred', 'mfeat'),
                            (L_diff, L_out, L_cond, L_fuse, L_pred, L_cnY, L_cnH,
                             L_rel, L_ms, L_topo, L_mcl, L_mpred, L_mfeat)):
                agg[k] += v.item() * n
            writer.add_scalar('Train/loss_step', loss.item(), step)
            step += 1

        d = {k: v / max(total_num, 1) for k, v in agg.items()}
        for k, v in d.items():
            writer.add_scalar(f'Train/L_{k}', v, itr)
        print(f'EPOCH_TIME {itr} {_t.perf_counter() - _ep0:.4f}', flush=True)
        if opt.cuda:
            print(f'EPOCH_MEM {itr} {torch.cuda.max_memory_allocated(device)/2**20:.2f}', flush=True)
        tag = '[A]' if stage_a else '[B]'
        print(f'epoch {itr} {tag} diff={d["diff"]:.4f} kd_out={d["out"]:.4f} '
              f'kd_cond={d["cond"]:.4f} kd_fuse={d["fuse"]:.4f} kd_pred={d["pred"]:.4f} cnodes_Y={d["cnY"]:.5f} cnodes_H={d["cnH"]:.5f} kd_rel={d["rel"]:.4f} kd_ms={d["ms"]:.4f} topo={d["topo"]:.4f} mcl={d["mcl"]:.4f} '
              f'm_pred={d["mpred"]:.5f} m_feat={d["mfeat"]:.5f}', flush=True)
        if opt.cuda:
            torch.cuda.empty_cache()
