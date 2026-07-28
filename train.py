"""Train nanoPELICAN-HGQ2. Run with KERAS_BACKEND=jax (enforced below).

The torch model emits [-w, w] and trains with CrossEntropy; that equals
BCE-from-logits on 2w, which is what nano_bce implements, so losses are
comparable across the two codebases.

Example (smoke run on the sibling repo's sample data):
    python train.py --data-dir ../PELICAN-nano/data/sample_data --epochs 2
"""
import argparse
import json
import os

os.environ.setdefault('KERAS_BACKEND', 'jax')

import keras
import numpy as np

from pelican_hgq2 import build_model, preset_config
from pelican_hgq2.data import load_split


def nano_bce(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, 2.0 * y_pred, from_logits=True)


def parse_args():
    p = argparse.ArgumentParser(description='nanoPELICAN HGQ2 training')
    p.add_argument('--data-dir', default='../PELICAN-nano/data/sample_data')
    p.add_argument('--n-hidden', type=int, default=2)
    p.add_argument('--nave', type=float, default=49.0)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--drop-rate', type=float, default=0.05)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--nmax', type=int, default=20,
                   help='leading-pT constituent cap before beams (firmware '
                        'NPARTICLES=20); wider h5 files are truncated on read')
    p.add_argument('--limit', type=int, default=None,
                   help='cap events per split (smoke tests)')
    p.add_argument('--float', dest='quant', action='store_false',
                   help='float baseline, no quantizers')
    p.add_argument('--beta', type=float, default=0.0,
                   help='EBOP resource-regularization strength (0 = off)')
    p.add_argument('--preset', choices=['init24', 'w6p12', 'w6p12f'], default='init24',
                   help='bitwidth starting point: Brevitas-24bit grids, or the '
                        'hand-tuned w6a6i6p12 operating point (adds the pmu '
                        'quantizer)')
    p.add_argument('--no-het-weights', dest='het_weights', action='store_false',
                   help='per-tensor instead of per-element weight bitwidths')
    p.add_argument('--out', default='model/hgq2_nano.weights.h5')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    train_x, train_y = load_split(os.path.join(args.data_dir, 'train.h5'),
                                  scale=args.scale, nmax=args.nmax,
                                  limit=args.limit)
    valid_x, valid_y = load_split(os.path.join(args.data_dir, 'valid.h5'),
                                  scale=args.scale, nmax=args.nmax,
                                  limit=args.limit)

    qcfg = None
    if args.quant:
        qcfg = preset_config(args.preset, beta=args.beta,
                             het_weights=args.het_weights)
    model = build_model(n_hidden=args.n_hidden, nmax=train_x.shape[1],
                        nave=args.nave, drop_rate=args.drop_rate, qcfg=qcfg)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=nano_bce,
        metrics=[keras.metrics.AUC(from_logits=True, name='auc')],
    )
    model.summary()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    hist = model.fit(
        train_x, train_y,
        validation_data=(valid_x, valid_y),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[keras.callbacks.ModelCheckpoint(
            args.out, monitor='val_auc', mode='max',
            save_best_only=True, save_weights_only=True)],
    )

    # Best-val-AUC checkpoint alone hides the converged bitwidth state when the
    # EBOP term degrades AUC over training (the beta=1e-6 epoch-1 artifact) —
    # always keep the final-epoch weights too.
    stem = args.out[:-len('.weights.h5')] if args.out.endswith('.weights.h5') \
        else os.path.splitext(args.out)[0]
    model.save_weights(stem + '.final.weights.h5')

    best = float(np.max(hist.history.get('val_auc', [0.0])))
    print(f'best val AUC: {best:.4f}')
    with open(os.path.splitext(args.out)[0] + '.history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f)


if __name__ == '__main__':
    main()
