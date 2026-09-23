"""Build the unified spatio-temporal Morse complex for a dataset.

Pipeline
  events (dt_nm, lon_nm, lat_nm)
    -> ST cells  (tile_x, tile_y, dt_quantile_bin)
    -> k-NN graph in normalized ST space (anisotropy w on the temporal axis)
    -> clique complex -> discrete Morse function -> critical simplices
    -> st_morse_features_{DATASET}_z{ZOOM}_b{BINS}.pt

Replaces build_morse_features.py (spatial only, keyed on VLM tile embeddings)
and the per-sequence temporal Morse computed inside the old training loop.

Usage:
  python build_st_morse_features.py --dataset eMAS_smoke --zoom 7 \
      --n_bins 16 --k 5 --w_time 1.0
"""
import argparse
import math
import os
import pickle

import numpy as np
import networkx as nx
import torch

from morse_function import (
    build_clique_complex,
    construct_discrete_morse_function,
    identify_critical_simplices,
)
from st_morse import (
    lat_lon_to_tile,
    tile_to_lonlat,
    dt_bin,
    initialize_st_vertex_weights,
    st_pairwise_dist,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    type=str, required=True)
    p.add_argument('--data_root',  type=str, default='dataset')
    p.add_argument('--zoom',       type=int, required=True)
    p.add_argument('--n_bins',     type=int,   default=16,  help='# quantile bins on dt')
    p.add_argument('--k',          type=int,   default=5,   help='k-NN neighbours per ST cell')
    p.add_argument('--w_time',     type=float, default=1.0, help='temporal anisotropy weight')
    p.add_argument('--min_count',  type=int,   default=1,   help='drop cells with fewer events')
    p.add_argument('--weight_mode', type=str,  default='density', choices=['density', 'degree'])
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--out_file',   type=str,   default='')
    return p.parse_args()


def load_events(dataset, data_root):
    """Mirror load_data() in the training script: [t, dt, lon, lat] + global MAX/MIN."""
    def read(split):
        with open(f'{data_root}/{dataset}/{split}.pkl', 'rb') as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        # see train_distill.py: the first event's dt column held its absolute time.
        d = [[[i[0], i[0] - u[idx - 1][0] if idx > 0 else 0.0] + i[1:]
              for idx, i in enumerate(u)] for u in d]
        return d

    data_all = read('data_train') + read('data_val') + read('data_test')
    Max, Min = [], []
    for m in range(4):                     # dim=2 -> [t, dt, lon, lat]
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1)
            Min.append(0)
    return data_all, Max, Min


def main():
    args = get_args()
    out_file = args.out_file or \
        f'st_morse_features_{args.dataset}_z{args.zoom}_b{args.n_bins}.pt'

    data_all, MAX, MIN = load_events(args.dataset, args.data_root)
    n_ev = sum(len(u) for u in data_all)
    print(f'[data] {args.dataset}: {len(data_all)} sequences, {n_ev} events')
    print(f'[data] dt  range [{MIN[1]:.6g}, {MAX[1]:.6g}]')
    print(f'[data] lon range [{MIN[2]:.6g}, {MAX[2]:.6g}]')
    print(f'[data] lat range [{MIN[3]:.6g}, {MAX[3]:.6g}]')

    # ── events -> normalized ST coordinates ─────────────────────────────────
    span = [None, MAX[1] - MIN[1], MAX[2] - MIN[2], MAX[3] - MIN[3]]
    span = [s if (s is None or s > 0) else 1.0 for s in span]

    dts, lons_raw, lats_raw, lon_nms, lat_nms = [], [], [], [], []
    for seq in data_all:
        for ev in seq:
            dts.append((ev[1] - MIN[1]) / span[1])
            lons_raw.append(float(ev[2]))
            lats_raw.append(float(ev[3]))
            lon_nms.append((ev[2] - MIN[2]) / span[2])
            lat_nms.append((ev[3] - MIN[3]) / span[3])
    dts = np.asarray(dts, dtype=np.float64)

    # ── quantile bin edges on dt (heavy-tailed -> quantiles, not equal width) ─
    qs = np.linspace(0, 100, args.n_bins + 1)[1:-1]
    edges = sorted(set(np.percentile(dts, qs).tolist()))
    print(f'[bins] {len(edges) + 1} effective dt bins (requested {args.n_bins})')

    # ── aggregate events into ST cells ──────────────────────────────────────
    cells = {}
    for i in range(len(dts)):
        tx, ty = lat_lon_to_tile(lats_raw[i], lons_raw[i], args.zoom)
        key = (tx, ty, dt_bin(dts[i], edges))
        c = cells.get(key)
        if c is None:
            cells[key] = c = {'count': 0, 'sum': np.zeros(3)}
        c['count'] += 1
        c['sum'] += (dts[i], lon_nms[i], lat_nms[i])

    if args.min_count > 1:
        cells = {k: v for k, v in cells.items() if v['count'] >= args.min_count}
    keys = sorted(cells.keys())
    N = len(keys)
    if N < 2:
        raise SystemExit(f'only {N} ST cells -- lower --zoom or --n_bins')
    counts = np.array([cells[k]['count'] for k in keys], dtype=np.float64)
    # cell centroid = mean of member events, so critical nodes sit on real data
    centers = np.stack([cells[k]['sum'] / cells[k]['count'] for k in keys])
    print(f'[cells] {N} non-empty ST cells, '
          f'events/cell min={counts.min():.0f} med={np.median(counts):.0f} max={counts.max():.0f}')

    # ── k-NN graph in normalized ST space ───────────────────────────────────
    dist = st_pairwise_dist(centers, args.w_time)
    np.fill_diagonal(dist, np.inf)
    k = min(args.k, N - 1)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in np.argsort(dist[i])[:k]:
            G.add_edge(i, int(j), weight=1.0 / (dist[i][j] + 1e-6))
    print(f'[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (k={k}, w_time={args.w_time})')

    # ── discrete Morse ──────────────────────────────────────────────────────
    density = {i: math.log1p(counts[i]) for i in range(N)}
    g = initialize_st_vertex_weights(G, density, mode=args.weight_mode, seed=args.seed)
    np.random.seed(args.seed)                 # morse_function draws eps from np.random
    complex_dict = build_clique_complex(G, max_dimension=1)
    f, Flag = construct_discrete_morse_function(G, complex_dict, g, max_dimension=1)
    IsCritical = identify_critical_simplices(complex_dict, f, max_dimension=1)

    morse_vals = np.array([f.get(frozenset([i]), 0.0) for i in range(N)])
    is_crit = np.array([int(IsCritical.get(frozenset([i]), False)) for i in range(N)])
    n_crit = int(is_crit.sum())
    print(f'[morse] critical cells: {n_crit}/{N} ({n_crit / N:.1%})')
    print(f'[morse] f range [{morse_vals.min():.3f}, {morse_vals.max():.3f}]')
    if n_crit:
        print(f'[morse] mean events/cell  critical={counts[is_crit == 1].mean():.1f}  '
              f'non-critical={counts[is_crit == 0].mean():.1f}')

    # ── per-cell feature matrix ─────────────────────────────────────────────
    def minmax(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    # col 0/1 mirror the old MorseTransformer input exactly; col 2 is extra
    features = np.stack([minmax(morse_vals), is_crit.astype(np.float64), minmax(counts)], axis=1)

    crit_centers = centers[is_crit == 1]
    if len(crit_centers) == 0:
        # degenerate graph -> fall back to the densest cells so the topo loss stays defined
        top = np.argsort(-counts)[:5]
        crit_centers = centers[top]
        print(f'[morse] WARNING no critical cells, falling back to {len(top)} densest cells')

    payload = {
        'cell_keys':       keys,
        'cell_features':   torch.tensor(features, dtype=torch.float32),
        'cell_centers':    torch.tensor(centers, dtype=torch.float32),
        'cell_counts':     torch.tensor(counts, dtype=torch.float32),
        'is_critical':     torch.tensor(is_crit, dtype=torch.long),
        'critical_centers': torch.tensor(crit_centers, dtype=torch.float32),  # (dt_nm, lon_nm, lat_nm)
        'meta': {
            'dataset': args.dataset, 'zoom': args.zoom, 'n_bins': args.n_bins,
            'dt_edges': edges, 'k': k, 'w_time': args.w_time,
            'weight_mode': args.weight_mode, 'min_count': args.min_count,
            'seed': args.seed, 'MAX': MAX, 'MIN': MIN,
        },
    }
    torch.save(payload, out_file)
    print(f'[save] -> {out_file}  ({os.path.getsize(out_file) / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
