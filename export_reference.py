"""Dump the learned quantization contract of a trained model.

Writes <out>.json with, per quantization point: keep_negative k, integer bits i,
fractional bits f (element-wise max for heterogeneous weights), i.e. everything
a future firmware loader needs to derive ap_fixed<k+i+f, k+i> types; and
<out>.npz with the float kernels/biases plus kernels pre-snapped to their own
grid with RND_CONV (what weights.h would contain).

This is the HGQ2 analogue of nPELICAN-fpga/model_loader.py's --quant path; the
weights.h emission itself stays in nPELICAN-fpga and is future work (Phase 4 in
PLAN.md).

Usage:
    python export_reference.py --weights model/hgq2_nano.weights.h5 --out model/contract
"""
import argparse
import json
import os

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np

from pelican_hgq2 import build_model, preset_config


def kif_of(quantizer_layer):
    """(k, i, f) numpy arrays of a hgq Quantizer (or QDense .iq/.kq)."""
    q = quantizer_layer.quantizer
    k = np.asarray(q.k, dtype=np.int32) if not callable(q.k) else np.asarray(q.k())
    i = np.rint(np.asarray(q.i)).astype(np.int32)
    f = np.rint(np.asarray(q.f)).astype(np.int32)
    return k, i, f


def snap(w, k, i, f):
    """Round-to-nearest-even onto the 2^-f grid with saturation (RND_CONV/SAT)."""
    step = 2.0 ** -f.astype(np.float64)
    scaled = w / step
    q = np.round(scaled)  # numpy rounds half to even == RND_CONV
    hi = 2.0 ** (i.astype(np.float64)) / step - 1
    lo = np.where(k > 0, -(2.0 ** i.astype(np.float64)) / step, 0.0)
    return np.clip(q, lo, hi) * step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--n-hidden', type=int, default=2)
    ap.add_argument('--nmax', type=int, default=22)
    ap.add_argument('--beta', type=float, default=0.0,
                    help='must match the beta the checkpoint was trained with '
                         '(EBOP tracking adds variables to the weight file)')
    ap.add_argument('--preset', choices=['init24', 'w6p12'], default='init24',
                    help='must match training (w6p12 adds the pmu quantizer, '
                         'which changes the variable set)')
    ap.add_argument('--out', default='model/contract')
    args = ap.parse_args()

    model = build_model(n_hidden=args.n_hidden, nmax=args.nmax,
                        qcfg=preset_config(args.preset, beta=args.beta))
    model.load_weights(args.weights)

    points = {
        'input': model.input_quant,
        'post_agg_2to2': model.mixing_2to2.iq,
        'w_2to2': model.mixing_2to2.kq,
        'act': model.act_quant,
        'post_agg_2to0': model.mixing_2to0.iq,
        'w_2to0': model.mixing_2to0.kq,
        'output': model.output_quant,
    }
    if model.pmu_quant is not None:
        points['pmu'] = model.pmu_quant

    contract = {}
    for name, layer in points.items():
        k, i, f = kif_of(layer)
        contract[name] = {
            'k': int(k.max()), 'i': int(i.max()), 'f': int(f.max()),
            'bits': int(k.max() + i.max() + f.max()),
            'shape': list(np.broadcast(k, i, f).shape),
            'ap_fixed': f'ap_{"" if k.max() else "u"}fixed<{int(k.max()+i.max()+f.max())},'
                        f'{int(k.max()+i.max())}>',
        }

    w1 = np.asarray(model.mixing_2to2.kernel)          # [6, H] (torch: [H, 6].T)
    w2 = np.asarray(model.mixing_2to0.kernel)          # [2H, 1]
    k1, i1, f1 = kif_of(model.mixing_2to2.kq)
    k2, i2, f2 = kif_of(model.mixing_2to0.kq)
    arrays = {
        'w1_2to2': w1, 'w1_2to2_snapped': snap(w1, k1, i1, f1),
        'w2_2to0': w2, 'w2_2to0_snapped': snap(w2, k2, i2, f2),
        'b1': np.asarray(model.bias_diag.bias),
        'b1_diag': np.asarray(model.bias_diag.diag_bias),
        'b2': np.asarray(model.mixing_2to0.bias),
        'bn1_gamma': np.asarray(model.bn1.gamma),
        'bn1_beta': np.asarray(model.bn1.beta),
        'bn1_mean': np.asarray(model.bn1.moving_mean),
        'bn1_var': np.asarray(model.bn1.moving_var),
        'bn2_gamma': np.asarray(model.bn2.gamma),
        'bn2_beta': np.asarray(model.bn2.beta),
        'bn2_mean': np.asarray(model.bn2.moving_mean),
        'bn2_var': np.asarray(model.bn2.moving_var),
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out + '.json', 'w') as fh:
        json.dump(contract, fh, indent=2)
    np.savez(args.out + '.npz', **arrays)
    print(json.dumps(contract, indent=2))
    print(f'wrote {args.out}.json and {args.out}.npz')


if __name__ == '__main__':
    main()
