"""Evaluate a teacher (Transformer_ST) or student (MLPHistoryEncoder) checkpoint.

Metrics follow MCM-DM's test_morse_pure.py exactly (MAE/RMSE temporal, MAE
spatial, averaged over --n_samples draws) and add the efficiency numbers the
distillation claim rests on: parameter count and sampling latency.
"""
import os, pickle, random, argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F

from transformer_st import Transformer_ST, get_dataloader
from model_student import MLPHistoryEncoder
from model_mm import (ST_Diffusion_MM, GaussianDiffusion_MM, Model_all_MM,
                      ImageProjector, STMorseEncoder, ModalFusionGate)
from st_morse import lat_lon_to_tile, dt_bin

p = argparse.ArgumentParser()
p.add_argument('--dataset',       type=str, required=True)
p.add_argument('--ckpt_dir',      type=str, required=True)
p.add_argument('--encoder',       type=str, default='transformer', choices=['transformer', 'mlp'])
p.add_argument('--window',        type=int, default=16)
p.add_argument('--hidden',        type=int, default=256)
p.add_argument('--n_hidden',      type=int, default=2)
p.add_argument('--emb_file',      type=str, required=True)
p.add_argument('--st_morse_file', type=str, required=True)
p.add_argument('--st_in_dim',     type=int, default=2)
p.add_argument('--zoom',          type=int, default=7)
p.add_argument('--timesteps',     type=int, default=500)
p.add_argument('--samplingsteps', type=int, default=500)
p.add_argument('--n_samples',     type=int, default=3)
p.add_argument('--ddim_eta',      type=float, default=1.0,
               help='0 = deterministic DDIM; required for step-distilled students, which are trained on the eta=0 map')
p.add_argument('--img_dim',       type=int, default=64)
p.add_argument('--objective',     type=str, default='pred_noise')
p.add_argument('--loss_type',     type=str, default='l2')
p.add_argument('--beta_schedule', type=str, default='cosine')
p.add_argument('--cuda_id',       type=str, default='0')
p.add_argument('--seed',          type=int, default=42)
p.add_argument('--dump_errs',     type=str, default='',
               help='diagnostic only: save per-event spatial errors to this .npy')
p.add_argument('--no_st_morse',   action='store_true',
               help='ablation: zero out the ST-Morse branch (must match training)')
p.add_argument('--dual_morse',    action='store_true',
               help='two ST-Morse branches fused by the gate (must match training)')
p.add_argument('--no_vlm',        action='store_true',
               help='must match how the checkpoint was trained')
p.add_argument('--cond_from_history', action='store_true',
               help='must match how the checkpoint was trained')
p.add_argument('--train_seed',    type=int, default=-1,
               help='seed the checkpoint was TRAINED with; recorded for bookkeeping (eval seed is --seed)')
p.add_argument('--out_json',      type=str, default='')
args = p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DIM = 2
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def normalization(x, MAX, MIN): return (x - MIN) / (MAX - MIN)


def load_data():
    root = f'dataset/{args.dataset}'
    def read(split):
        with open(f'{root}/{split}.pkl', 'rb') as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        # see train_distill.py: the first event's dt column held its absolute
        # time, which set MAX[1] and inflated the dt range (3995x on Wildfire_CA).
        return [[[i[0], i[0]-u[idx-1][0] if idx > 0 else 0.0] + i[1:]
                 for idx, i in enumerate(u)] for u in d]
    tr, va, te = read('data_train'), read('data_val'), read('data_test')
    data_all = tr + va + te
    Max, Min = [], []
    for m in range(DIM + 2):
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1); Min.append(0)
    norm = lambda d: [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in d]
    return get_dataloader(norm(te), min(len(te), 1000), D=DIM, shuffle=False), (Max, Min), data_all


emb_cache = torch.load(args.emb_file, map_location='cpu', weights_only=False)
stf  = torch.load(args.st_morse_file, map_location='cpu', weights_only=False)
meta = stf['meta']
testloader, (MAX, MIN), data_all = load_data()

emb_lookup = {}
for seq in data_all:
    for ev in seq:
        lon_raw, lat_raw = float(ev[2]), float(ev[3])
        key = (args.zoom,) + lat_lon_to_tile(lat_raw, lon_raw, args.zoom)
        if key in emb_cache:
            v = emb_cache[key]
            emb_lookup[(round(lon_raw, 5), round(lat_raw, 5))] = v["embedding"] if isinstance(v, dict) else v

n_cells = len(stf['cell_keys'])
cell_features = torch.cat([stf['cell_features'][:, :args.st_in_dim],
                           torch.zeros(1, args.st_in_dim)], dim=0).to(device)
st = {'zoom': meta['zoom'], 'edges': meta['dt_edges'],
      'key2idx': {k: i for i, k in enumerate(stf['cell_keys'])}, 'miss_idx': n_cells}

if args.encoder == 'transformer':
    enc = Transformer_ST(d_model=64, d_rnn=256, d_inner=128, n_layers=4, n_head=4,
                         d_k=16, d_v=16, dropout=0.1, device=device, loc_dim=DIM, CosSin=True).to(device)
else:
    enc = MLPHistoryEncoder(d_model=64, window=args.window, hidden=args.hidden,
                            n_hidden=args.n_hidden, dropout=0.1, device=device, loc_dim=DIM).to(device)

denoiser  = ST_Diffusion_MM(n_steps=args.timesteps, dim=1+DIM, condition=True,
                            cond_dim=64, img_dim=args.img_dim).to(device)
diffusion = GaussianDiffusion_MM(denoiser, seq_length=1+DIM, timesteps=args.timesteps,
                                 sampling_timesteps=args.samplingsteps, loss_type=args.loss_type,
                                 objective=args.objective, beta_schedule=args.beta_schedule,
                                 ddim_sampling_eta=args.ddim_eta).to(device)
Model = Model_all_MM(enc, diffusion, ImageProjector(1536, args.img_dim).to(device))
Model.hist_proj, Model.img_proj_mcl = Model.hist_proj.to(device), Model.img_proj_mcl.to(device)
Model.load_state_dict(torch.load(f'{args.ckpt_dir}/model_best.pkl', map_location=device))
st_enc = STMorseEncoder(in_dim=args.st_in_dim, out_dim=args.img_dim).to(device)
gate   = ModalFusionGate(dim=args.img_dim).to(device)
st_enc.load_state_dict(torch.load(f'{args.ckpt_dir}/st_morse_best.pkl', map_location=device))
gate.load_state_dict(torch.load(f'{args.ckpt_dir}/gate_best.pkl', map_location=device))

# --dual_morse: rebuild the critical branch exactly as training did. The nearest-critical
# map is derived from the same cached centres and the same anisotropy weight, so no extra
# state has to travel with the checkpoint.
st_enc_crit, crit_features, cell2crit = None, None, None
if args.dual_morse:
    _ci = torch.nonzero(stf['is_critical'] > 0.5, as_tuple=False).squeeze(-1)
    _wt = meta['w_time']
    _C = stf['cell_centers'].clone().float(); _C[:, 0] *= _wt
    cell2crit = torch.cat([torch.cdist(_C, _C[_ci]).argmin(dim=-1),
                           torch.tensor([_ci.numel()])]).to(device)
    crit_features = torch.cat([stf['cell_features'][_ci, :args.st_in_dim],
                               torch.zeros(1, args.st_in_dim)], dim=0).to(device)
    st_enc_crit = STMorseEncoder(in_dim=args.st_in_dim, out_dim=args.img_dim).to(device)
    st_enc_crit.load_state_dict(
        torch.load(f'{args.ckpt_dir}/st_morse_crit_best.pkl', map_location=device))
    st_enc_crit.eval()
    print(f'[dual_morse] branch A: {n_cells} cells   branch B: {_ci.numel()} critical cells')

Model.eval(); st_enc.eval(); gate.eval()

n_par = lambda m: sum(pp.numel() for pp in m.parameters())
_crit_par = n_par(st_enc_crit) if args.dual_morse else 0
params = {'encoder': n_par(enc), 'denoiser': n_par(denoiser),
          'st_morse': n_par(st_enc) + _crit_par, 'img_projector': n_par(Model.img_projector),
          'total': n_par(Model) + n_par(st_enc) + n_par(gate) + _crit_par}


def batch_to_model(batch):
    eto, et, lng, lat = map(lambda x: x.to(device), batch)
    loc = torch.cat((lng.unsqueeze(2), lat.unsqueeze(2)), dim=-1)
    e_out, mask = Model.transformer(loc, eto)
    bi, ei, ti, img_l, cell_l = [], [], [], [], []
    zero = torch.zeros(1536)
    for b in range(mask.shape[0]):
        L = int(mask[b].sum().item())
        if L <= 1: continue
        lon_r = (lng[b, :L].cpu() * (MAX[2]-MIN[2])) + MIN[2]
        lat_r = (lat[b, :L].cpu() * (MAX[3]-MIN[3])) + MIN[3]
        dt_nm = et[b, :L].cpu()
        for e in range(L - 1):
            tgt = e + 1
            bi.append(b); ei.append(e); ti.append(tgt)
            c = e if args.cond_from_history else tgt
            img_l.append(emb_lookup.get((round(lon_r[c].item(), 5), round(lat_r[c].item(), 5)), zero).unsqueeze(0))
            tx, ty = lat_lon_to_tile(lat_r[c].item(), lon_r[c].item(), st['zoom'])
            cell_l.append(st['key2idx'].get((tx, ty, dt_bin(dt_nm[c].item(), st['edges'])), st['miss_idx']))
    if not bi:
        return (None,) * 4
    bi = torch.tensor(bi, device=device); ei = torch.tensor(ei, device=device); ti = torch.tensor(ti, device=device)
    return (et[bi, ti].reshape(-1, 1, 1), loc[bi, ti].reshape(-1, 1, DIM),
            e_out[bi, ei].unsqueeze(1), torch.cat(img_l, 0).unsqueeze(1).to(device),
            torch.tensor(cell_l, dtype=torch.long, device=device))


mae_s, mae_t, rmse_t, total = 0., 0., 0., 0
lat_ms, lat_n = 0., 0
_ncalls = 0
_evt_per_call = []
_err_dump = []
# peak device memory over the whole sampling pass, for the efficiency table
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
with torch.no_grad():
    for batch in testloader:
        out = batch_to_model(batch)
        if out[0] is None: continue
        t_nm, loc_nm, enc_nm, img_nm, cell_idx = out
        img_proj = (torch.zeros(img_nm.shape[0], 1, args.img_dim, device=device)
                    if args.no_vlm else Model.img_projector(img_nm))
        st_emb = (torch.zeros(img_proj.shape[0], 1, args.img_dim, device=device)
                  if args.no_st_morse else st_enc(cell_features)[cell_idx].unsqueeze(1))
        if args.dual_morse:
            crit_emb = st_enc_crit(crit_features)[cell2crit[cell_idx]].unsqueeze(1)
            imgc = gate(st_emb, crit_emb)
        else:
            imgc = gate(img_proj, st_emb)

        gt, gs = [], []
        for _ in range(args.n_samples):
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            sampled = Model.diffusion.sample(batch_size=t_nm.shape[0], cond=enc_nm, img_cond=imgc)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_ms += (time.perf_counter() - t0) * 1e3; lat_n += t_nm.shape[0]
            _ncalls += 1; _evt_per_call.append(int(t_nm.shape[0]))
            gt.append(sampled[:, 0, :1].cpu()); gs.append(sampled[:, 0, -2:].cpu())
        gen_t = torch.stack(gt, 0).mean(0); gen_s = torch.stack(gs, 0).mean(0)

        real_t = (t_nm[:, 0, :].cpu() * (MAX[1]-MIN[1])) + MIN[1]
        gen_t  = (gen_t * (MAX[1]-MIN[1])) + MIN[1]
        mae_t  += torch.abs(real_t - gen_t).sum().item()
        rmse_t += ((real_t - gen_t) ** 2).sum().item()

        sc = torch.tensor([MAX[2]-MIN[2], MAX[3]-MIN[3]]); off = torch.tensor([MIN[2], MIN[3]])
        real_s = (loc_nm[:, 0, :].cpu() * sc) + off
        gen_s  = (gen_s * sc) + off
        _e = torch.sqrt(((real_s - gen_s) ** 2).sum(dim=-1))
        mae_s += _e.sum().item()
        if args.dump_errs: _err_dump.append(_e.numpy())
        total += t_nm.shape[0]

if args.dump_errs and _err_dump:
    np.save(args.dump_errs, np.concatenate(_err_dump))
    print(f"-> per-event spatial errors: {args.dump_errs}")

res = {
    'dataset': args.dataset, 'encoder': args.encoder, 'ckpt_dir': args.ckpt_dir,
    'seed': args.seed, 'train_seed': args.train_seed, 'samplingsteps': args.samplingsteps, 'n_samples': args.n_samples,
    'ddim_eta': args.ddim_eta,
    'test_events': total,
    'mae_temporal':  mae_t / total,
    'rmse_temporal': (rmse_t / total) ** 0.5,
    'mae_spatial':   mae_s / total,
    'n_sample_calls': _ncalls,
    'events_per_call': (sum(_evt_per_call)/len(_evt_per_call)) if _evt_per_call else None,
    'params': params,
    'peak_mem_mib': (torch.cuda.max_memory_allocated(device) / 2**20
                     if torch.cuda.is_available() else None),
    'sampling_ms_per_1k_events': lat_ms / lat_n * 1000,
}
print(f"Dataset       : {args.dataset}   encoder: {args.encoder}")
print(f"Test events   : {total}")
print(f"MAE  temporal : {res['mae_temporal']:.4f}")
print(f"RMSE temporal : {res['rmse_temporal']:.4f}")
print(f"MAE  spatial  : {res['mae_spatial']:.4f}")
print(f"Params        : encoder={params['encoder']:,}  total={params['total']:,}")
print(f"Sampling      : {res['sampling_ms_per_1k_events']:.1f} ms / 1k events "
      f"({args.samplingsteps} steps)")
if args.out_json:
    with open(args.out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'-> {args.out_json}')
