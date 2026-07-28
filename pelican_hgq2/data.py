"""Toptag HDF5 loading, ported from PELICAN-nano's collate_fn defaults.

Reference semantics (src/dataloaders/collate.py with the training defaults
scale=1.0, beam_mass=0.0, add_beams=True):
  - jets are scaled by `scale`; the two beam spurions are NOT scaled
  - beam_mass=0 gives beams (1, 0, 0, +-1) prepended along the particle axis
  - masks are derived from E != 0 (the model re-derives them internally)
"""
from math import sqrt

import h5py
import numpy as np


def load_split(path, scale=1.0, add_beams=True, beam_mass=0.0, nmax=None,
               limit=None, seed=0):
    """Returns (pmu [B, 2+N, 4] float32, labels [B] float32).

    `limit` takes a seeded RANDOM subsample — the toptag h5 files are sorted by
    label, so a head slice would be single-class (AUC undefined).

    `nmax` caps constituents to the leading nmax (toptag files are pT-sorted, so
    this is the leading-pT cap; firmware NPARTICLES=20). The cap is applied
    inside the h5 read so 200-wide files never load fully into memory."""
    csl = slice(None) if nmax is None else slice(None, nmax)
    with h5py.File(path, 'r') as f:
        n = f['Pmu'].shape[0]
        if limit is not None and limit < n:
            idx = np.sort(np.random.default_rng(seed).choice(n, limit,
                                                             replace=False))
            pmu = f['Pmu'][idx, csl].astype('float32')
            y = f['is_signal'][idx].astype('float32')
        else:
            pmu = f['Pmu'][:, csl].astype('float32')
            y = f['is_signal'][:].astype('float32')
    pmu = pmu * np.float32(scale)
    if add_beams:
        p = 1.0
        e = sqrt(p ** 2 + beam_mass ** 2)
        beams = np.array([[e, 0, 0, p], [e, 0, 0, -p]], dtype=np.float32)
        beams = np.broadcast_to(beams, (pmu.shape[0], 2, 4))
        pmu = np.concatenate([beams, pmu], axis=1)
    return pmu, y
