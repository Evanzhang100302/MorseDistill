"""MCM-DM teacher with a single unified spatio-temporal Morse function.

Differences vs MCM-DM's train_mm_morse_temporal.py:
  * one ST Morse path (STMorseEncoder over joint (dt, lon, lat) cells) instead of
    MorseTransformer (spatial) + TemporalMorseEncoder (temporal)
  * ModalFusionGate (VLM + ST Morse) instead of TriModalFusionGate
  * one topology loss L_topo_st in the full 3-D prediction space, weight
    --beta_topo_st, instead of beta_topo + beta_topo_t
  * the per-sequence O(N^2) temporal-Morse python loop is gone

This is the teacher for the transformer -> MLP distillation; it keeps
Transformer_ST intact.
"""
import os, math, pickle, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import setproctitle

from transformer_st import Transformer_ST, get_dataloader
from model_mm import (ST_Diffusion_MM, GaussianDiffusion_MM, Model_all_MM,
                      ImageProjector, STMorseEncoder, ModalFusionGate,
                      normalize_to_neg_one_to_one)
from st_morse import lat_lon_to_tile, dt_bin


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',          type=int, default=1234)
    p.add_argument('--dataset',       type=str, default='eMAS_smoke')
    p.add_argument('--total_epochs',  type=int, default=100)
    p.add_argument('--alpha',         type=float, default=0.05,  help='MCL loss weight')
    p.add_argument('--beta_topo_st',  type=float, default=0.001, help='unified ST topo loss weight')
    p.add_argument('--topo_w_time',   type=float, default=-1.0,
                   help='temporal anisotropy in the topo loss; <0 uses the value stored in the feature file')
    p.add_argument('--batch_size',    type=int, default=64)
    p.add_argument('--timesteps',     type=int, default=500)
    p.add_argument('--samplingsteps', type=int, default=500)
    p.add_argument('--objective',     type=str, default='pred_noise')
    p.add_argument('--loss_type',     type=str, default='l2')
    p.add_argument('--beta_schedule', type=str, default='cosine')
    p.add_argument('--cuda_id',       type=str, default='0')
    p.add_argument('--img_dim',       type=int, default=64)
    p.add_argument('--zoom',          type=int, default=7,  help='VLM tile zoom')
    p.add_argument('--emb_file',      type=str, required=True)
    p.add_argument('--st_morse_file', type=str, required=True)
    p.add_argument('--st_in_dim',     type=int, default=2,
                   help='2 = (morse_value, is_critical) as in MCM-DM; 3 adds cell density')
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--no_vlm',        action='store_true')
    p.add_argument('--dim_weight',    action='store_true',
                   help='scale the (dt, lon, lat) axes of the diffusion loss by 1/std, measured on the\n                        training split in the [-1,1] space the loss lives in. Without it the two\n                        spatial axes dominate and dt is never learned (0 = off).')
    p.add_argument('--no_st_morse',   action='store_true')
    p.add_argument('--dual_morse',    action='store_true',
                   help='two ST-Morse branches: one over ALL cells (indexed by the event own '
                        'cell) and one over the CRITICAL cells only (indexed by the nearest '
                        'critical cell under the graph metric d_w). Their outputs are fused by '
                        'the gate, which replaces the now-unused VLM slot. Off = single branch '
                        'over all cells, as before.')
    p.add_argument('--max_val_seqs',  type=int, default=200, help='cap val sequences to keep eval fast')
    p.add_argument('--eval_every',    type=int, default=10)
    p.add_argument('--cond_from_history', action='store_true',
                   help='index VLM tile / ST cell by the last observed event, not the target '
                        '(MCM-DM uses the target, which leaks its own coordinates)')
    p.add_argument('--exp_name',      type=str, default='teacher')
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
    train_data = read('data_train')
    val_data   = read('data_val')
    test_data  = read('data_test')
    data_all   = train_data + val_data + test_data
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


def batch_to_model(batch, transformer, emb_lookup, st, MAX, MIN):
    """Returns (dt_nm, loc_nm, hist_cond, vlm_emb, st_cell_idx) for every
    (history -> next event) pair in the batch."""
    event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
    event_loc = torch.cat((lng.unsqueeze(2), lat.unsqueeze(2)), dim=-1)
    enc_out, mask = transformer(event_loc, event_time_origin)
    B = mask.shape[0]

    st_zoom, st_edges, st_key2idx = st['zoom'], st['edges'], st['key2idx']
    st_miss_idx = st['miss_idx']

    enc_l, time_l, loc_l, img_l, cell_l = [], [], [], [], []
    zero_emb = torch.zeros(1536)
    for b in range(B):
        length = int(mask[b].sum().item())
        if length <= 1:
            continue
        lng_raw = (lng[b, :length].cpu() * (MAX[2]-MIN[2])) + MIN[2]
        lat_raw = (lat[b, :length].cpu() * (MAX[3]-MIN[3])) + MIN[3]
        dt_nm   = event_time[b, :length].cpu()
        for e in range(length - 1):
            tgt = e + 1                       # the event being predicted
            enc_l.append(enc_out[b][e].unsqueeze(0))
            time_l.append(event_time[b][tgt].unsqueeze(0))
            loc_l.append(event_loc[b][tgt].unsqueeze(0))
            c = e if opt.cond_from_history else tgt      # which event the condition is read at
            lon_k, lat_k = round(lng_raw[c].item(), 5), round(lat_raw[c].item(), 5)
            img_l.append(emb_lookup.get((lon_k, lat_k), zero_emb).unsqueeze(0))
            tx, ty = lat_lon_to_tile(lat_raw[c].item(), lng_raw[c].item(), st_zoom)
            ckey = (tx, ty, dt_bin(dt_nm[c].item(), st_edges))
            cell_l.append(st_key2idx.get(ckey, st_miss_idx))

    if not enc_l:
        return (None,) * 5
    return (torch.cat(time_l, 0).reshape(-1, 1, 1),
            torch.cat(loc_l,  0).reshape(-1, 1, opt.dim),
            torch.cat(enc_l,  0).reshape(-1, 1, enc_out.shape[-1]),
            torch.cat(img_l,  0).unsqueeze(1).to(device),
            torch.tensor(cell_l, dtype=torch.long, device=device))


def st_topo_loss(pred_xstart, crit_centers_pm1, w_time):
    """min distance from each predicted (dt, lon, lat) to the ST critical set.

    Both sides live in the diffusion's [-1, 1] space: pred_x_start comes out of a
    model trained on normalize_to_neg_one_to_one(x), and crit_centers_pm1 has
    already been mapped from [0, 1] to [-1, 1]. MCM-DM compared a [-1, 1]
    prediction against [0, 1] critical nodes, which biased the anchor by a
    constant offset.
    """
    p = pred_xstart[:, 0, :3].clone()
    p[:, 0] = p[:, 0] * w_time
    c = crit_centers_pm1.clone()
    c[:, 0] = c[:, 0] * w_time
    return torch.cdist(p.unsqueeze(0), c.unsqueeze(0)).squeeze(0).min(dim=-1).values.mean()


if __name__ == '__main__':
    setup_init()
    setproctitle.setproctitle(f'KD-teacher-{opt.dataset}')

    print(f'Loading VLM tile embeddings: {opt.emb_file}')
    emb_cache = torch.load(opt.emb_file, map_location='cpu', weights_only=False)
    print(f'Loading ST Morse features:   {opt.st_morse_file}')
    stf = torch.load(opt.st_morse_file, map_location='cpu', weights_only=False)
    meta = stf['meta']

    trainloader, valloader, testloader, (MAX, MIN), data_all = load_data()

    # the feature file must have been built from the same normalization
    for i in (1, 2, 3):
        assert abs(meta['MAX'][i] - MAX[i]) < 1e-6 and abs(meta['MIN'][i] - MIN[i]) < 1e-6, \
            f'ST Morse file was built with different MAX/MIN at index {i} -- rebuild it'

    emb_lookup = build_emb_lookup(data_all, emb_cache, opt.zoom)
    n_cells = len(stf['cell_keys'])
    # extra all-zero row absorbs events whose cell was dropped / unseen
    cell_features = torch.cat([stf['cell_features'][:, :opt.st_in_dim],
                               torch.zeros(1, opt.st_in_dim)], dim=0).to(device)
    st = {'zoom': meta['zoom'], 'edges': meta['dt_edges'],
          'key2idx': {k: i for i, k in enumerate(stf['cell_keys'])},
          'miss_idx': n_cells}
    crit_centers = stf['critical_centers'].to(device)
    crit_centers_pm1 = crit_centers * 2.0 - 1.0        # [0,1] -> diffusion's [-1,1] space
    w_time = meta['w_time'] if opt.topo_w_time < 0 else opt.topo_w_time

    # --dual_morse: a second encoder sees only the critical cells. Every cell is re-indexed
    # to its nearest critical cell under the same anisotropic metric the k-NN graph was built
    # with, so an event in a non-critical cell still gets the embedding of the hotspot it
    # belongs to. Computed here rather than cached in the .pt so existing feature files stay
    # valid; the cdist is at most n_cells x n_crit.
    crit_features, cell2crit = None, None
    if opt.dual_morse:
        ci = (stf['is_critical'] == 1).nonzero().squeeze(-1)
        C = stf['cell_centers'].clone().float(); C[:, 0] *= w_time
        cell2crit = torch.cat([torch.cdist(C, C[ci]).argmin(dim=-1),
                               torch.tensor([len(ci)])]).to(device)   # miss_idx -> pad row
        crit_features = torch.cat([stf['cell_features'][ci, :opt.st_in_dim],
                                   torch.zeros(1, opt.st_in_dim)], dim=0).to(device)

    print(f'  VLM tiles: {len(emb_cache)}  |  ST cells: {n_cells}  |  '
          f'ST critical: {crit_centers.shape[0]}  |  topo w_time={w_time}'
          + (f'  |  dual_morse: all={n_cells} crit={crit_features.shape[0] - 1}'
             if opt.dual_morse else ''))

    model_path = f'./models/{opt.dataset}_{opt.exp_name}_seed{opt.seed}/'
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'./logs/{opt.dataset}_{opt.exp_name}_seed{opt.seed}', flush_secs=5)

    transformer = Transformer_ST(d_model=64, d_rnn=256, d_inner=128, n_layers=4,
                                 n_head=4, d_k=16, d_v=16, dropout=0.1,
                                 device=device, loc_dim=opt.dim, CosSin=True).to(device)
    denoiser = ST_Diffusion_MM(n_steps=opt.timesteps, dim=1+opt.dim, condition=True,
                               cond_dim=64, img_dim=opt.img_dim).to(device)
    diffusion = GaussianDiffusion_MM(denoiser, seq_length=1+opt.dim,
                                     timesteps=opt.timesteps, sampling_timesteps=opt.samplingsteps,
                                     loss_type=opt.loss_type, objective=opt.objective,
                                     beta_schedule=opt.beta_schedule).to(device)
    img_projector = ImageProjector(input_dim=1536, img_dim=opt.img_dim).to(device)
    st_encoder    = STMorseEncoder(in_dim=opt.st_in_dim, d_model=64, nhead=4,
                                   num_layers=2, out_dim=opt.img_dim).to(device)
    st_encoder_crit = (STMorseEncoder(in_dim=opt.st_in_dim, d_model=64, nhead=4,
                                      num_layers=2, out_dim=opt.img_dim).to(device)
                       if opt.dual_morse else None)
    fusion_gate   = ModalFusionGate(dim=opt.img_dim).to(device)
    Model = Model_all_MM(transformer, diffusion, img_projector)
    Model.hist_proj    = Model.hist_proj.to(device)
    Model.img_proj_mcl = Model.img_proj_mcl.to(device)

    if opt.dim_weight:
        cols = []
        for _mi in (1, 2, 3):
            _v = np.array([i[_mi] for u in data_all for i in u], dtype=np.float64)
            _v = (_v - MIN[_mi]) / (MAX[_mi] - MIN[_mi]) * 2 - 1
            cols.append(max(_v.std(), 1e-6))
        dw = torch.tensor([1.0 / c for c in cols], dtype=torch.float32, device=device)
        dw = dw / dw.mean()
        print(f'[dim_weight] per-axis std (dt, lon, lat) = ({cols[0]:.4f}, {cols[1]:.4f}, '
              f'{cols[2]:.4f})  ->  weights = ({dw[0]:.3f}, {dw[1]:.3f}, {dw[2]:.3f})', flush=True)
        Model.diffusion.set_dim_weight(dw)

    st_mods = [st_encoder] + ([st_encoder_crit] if opt.dual_morse else [])
    n_par = lambda m: sum(p.numel() for p in m.parameters())
    print(f'  params: transformer={n_par(transformer):,}  diffusion={n_par(denoiser):,}  '
          f'st_morse={sum(n_par(m) for m in st_mods):,}'
          f'{" (2 branches)" if opt.dual_morse else ""}  '
          f'total={n_par(Model)+sum(n_par(m) for m in st_mods)+n_par(fusion_gate):,}')

    optimizer = AdamW(
        [{"params": Model.parameters(), "lr": opt.lr}]
        + [{"params": m.parameters(), "lr": opt.lr / 10} for m in st_mods]
        + [{"params": fusion_gate.parameters(), "lr": opt.lr / 10}],
        betas=(0.9, 0.99))

    warmup_steps = 5
    step, early_stop, min_loss_val = 0, 0, 1e20

    def make_cond(img_nm, cell_idx):
        if opt.no_vlm:
            img_proj = torch.zeros(img_nm.shape[0], 1, opt.img_dim, device=device)
        else:
            img_proj = Model.img_projector(img_nm)
        if opt.no_st_morse:
            st_emb = torch.zeros(img_proj.shape[0], 1, opt.img_dim, device=device)
        else:
            st_emb = st_encoder(cell_features)[cell_idx].unsqueeze(1)
        if opt.dual_morse:
            # the gate's first slot carries the all-cell branch, its second the critical
            # branch -- the VLM slot is unused under --no_vlm, so no third input is needed
            crit_emb = st_encoder_crit(crit_features)[cell2crit[cell_idx]].unsqueeze(1)
            return fusion_gate(st_emb, crit_emb)
        return fusion_gate(img_proj, st_emb)

    for itr in range(opt.total_epochs):
        if itr % opt.eval_every == 0:
            Model.eval(); fusion_gate.eval()
            for _m in st_mods: _m.eval()
            with torch.no_grad():
                loss_val, dist_s, total_num = 0., 0., 0
                for batch in valloader:
                    t_nm, l_nm, enc_nm, img_nm, cell_idx = batch_to_model(
                        batch, Model.transformer, emb_lookup, st, MAX, MIN)
                    if t_nm is None: continue
                    img_proj = make_cond(img_nm, cell_idx)
                    x_input = torch.cat((t_nm, l_nm), dim=-1)
                    loss = Model.diffusion(x_input, enc_nm, img_cond=img_proj)
                    loss_val += loss.item() * t_nm.shape[0]
                    sampled = Model.diffusion.sample(batch_size=t_nm.shape[0],
                                                     cond=enc_nm, img_cond=img_proj)
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
                        if early_stop >= 100:
                            print('Early stopping.'); break
                    else:
                        early_stop = 0
                        torch.save(Model.state_dict(),       model_path + 'model_best.pkl')
                        torch.save(st_encoder.state_dict(),  model_path + 'st_morse_best.pkl')
                        torch.save(fusion_gate.state_dict(), model_path + 'gate_best.pkl')
                        if opt.dual_morse:
                            torch.save(st_encoder_crit.state_dict(),
                                       model_path + 'st_morse_crit_best.pkl')
                    min_loss_val = min(min_loss_val, loss_val)

        base_lr = opt.lr*(itr+1)/warmup_steps if itr < warmup_steps \
                  else opt.lr - (opt.lr-5e-5)*(itr-warmup_steps)/opt.total_epochs
        for gi, gp in enumerate(optimizer.param_groups):
            gp['lr'] = base_lr if gi == 0 else base_lr / 10
        writer.add_scalar('Stats/lr', base_lr, itr)

        Model.train(); fusion_gate.train()
        for _m in st_mods: _m.train()
        loss_epoch, total_num = 0., 0
        for batch in trainloader:
            t_nm, l_nm, enc_nm, img_nm, cell_idx = batch_to_model(
                batch, Model.transformer, emb_lookup, st, MAX, MIN)
            if t_nm is None: continue

            img_proj  = make_cond(img_nm, cell_idx)
            x_input   = torch.cat((t_nm, l_nm), dim=-1)
            loss_diff = Model.diffusion(x_input, enc_nm, img_cond=img_proj)

            loss_mcl = mcl_loss(Model.hist_proj(enc_nm[:, 0, :64]),
                                Model.img_proj_mcl(img_proj[:, 0, :]))

            if opt.beta_topo_st > 0 and crit_centers.shape[0] > 0:
                t_rand = torch.randint(0, opt.timesteps, (x_input.shape[0],), device=device)
                x_pm1  = normalize_to_neg_one_to_one(x_input)
                x_t    = Model.diffusion.q_sample(x_pm1, t_rand, noise=torch.randn_like(x_pm1))
                preds  = Model.diffusion.model_predictions(x_t, t_rand, cond=enc_nm, img_cond=img_proj)
                loss_topo = st_topo_loss(preds.pred_x_start, crit_centers_pm1, w_time)
            else:
                loss_topo = torch.tensor(0.0, device=device)

            loss = loss_diff + opt.alpha * loss_mcl + opt.beta_topo_st * loss_topo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()

            loss_epoch += loss.item() * t_nm.shape[0]
            total_num  += t_nm.shape[0]
            writer.add_scalar('Train/loss_step',   loss.item(),      step)
            writer.add_scalar('Train/loss_topo_st', loss_topo.item(), step)
            step += 1

        ep_loss = loss_epoch / max(total_num, 1)
        writer.add_scalar('Train/loss_epoch', ep_loss, itr)
        print(f'epoch {itr}: train_loss={ep_loss:.4f}', flush=True)
        if opt.cuda:
            torch.cuda.empty_cache()
