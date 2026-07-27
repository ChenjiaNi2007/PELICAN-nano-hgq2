"""Generate golden parity vectors from the PyTorch float reference model.

Run with the SIBLING repo's venv (torch + the PELICAN-nano source tree):

    ../PELICAN-nano/.venv/bin/python scripts/make_golden.py

Writes tests/data/golden.npz: a random padded batch, the float logits from
PELICANNano (n_hidden=2, relu, batchnorm='b', config 's'/'s', add_beams), and
every parameter needed to load the same network into the Keras port. The Keras
float model must reproduce the logits (tests/test_parity_float.py).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', '..', 'PELICAN-nano')
sys.path.insert(0, REPO)

from src.models.pelican_nano import PELICANNano  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

N_HIDDEN, NMAX, B = 2, 20, 16

model = PELICANNano(
    N_HIDDEN, activate_agg=False, activate_lin=True, activation='relu',
    add_beams=True, config='s', config_out='s', average_nobj=49,
    # NB: activate_agg_out=False is the TRAINING default (args.py), not the
    # constructor default (True). The firmware has no ReLU on R either.
    factorize=False, masked=True, activate_agg_out=False, activate_lin_out=False,
    dropout=True, batchnorm='b', device=torch.device('cpu'), dtype=torch.float32,
)

# Randomize every float parameter/buffer so the parity test exercises BN
# statistics and affine params, not just fresh-init identity BN.
with torch.no_grad():
    for name, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * 0.5)
    for name, b in model.named_buffers():
        if b.dtype.is_floating_point:
            if 'running_var' in name:
                b.copy_(torch.rand_like(b) + 0.5)
            else:
                b.copy_(torch.randn_like(b) * 0.5)
model.eval()

# Random massless jet constituents, zero-padded, beams prepended (beam_mass=0).
nobj = np.random.randint(4, NMAX + 1, size=B)
pmu = np.zeros((B, NMAX, 4), dtype=np.float32)
for e in range(B):
    p3 = np.random.randn(nobj[e], 3).astype(np.float32) * 20.0
    pmu[e, :nobj[e], 1:] = p3
    pmu[e, :nobj[e], 0] = np.linalg.norm(p3, axis=-1)
beams = np.array([[1, 0, 0, 1], [1, 0, 0, -1]], dtype=np.float32)
pmu_full = np.concatenate([np.broadcast_to(beams, (B, 2, 4)), pmu], axis=1)

pt = torch.from_numpy(pmu_full)
particle_mask = pt[..., 0] != 0
edge_mask = particle_mask.unsqueeze(1) & particle_mask.unsqueeze(2)
data = {'Pmu': pt, 'particle_mask': particle_mask, 'edge_mask': edge_mask}

with torch.no_grad():
    out = model(data)['predict']            # [B, 2] = [-w, w]
logit = out[:, 1:2].numpy()

eq22 = model.net2to2.eq_layers[0]
bn1 = model.net2to2.message_layers[0].normlayer
bn2 = model.msg_2to0.normlayer
eq20 = model.agg_2to0

np.savez(
    os.path.join(HERE, '..', 'tests', 'data', 'golden.npz'),
    pmu=pmu_full, logit=logit,
    w1=eq22.mixing.weight.detach().numpy(),          # [H, 6]
    b1=eq22.bias.detach().numpy(),                   # [H]
    b1_diag=eq22.diag_bias.detach().numpy(),         # [H]
    w2=eq20.mixing.weight.detach().numpy(),          # [1, 2H]
    b2=eq20.mixing.bias.detach().numpy(),            # [1]
    bn1_gamma=bn1.weight.detach().numpy(),
    bn1_beta=bn1.bias.detach().numpy(),
    bn1_mean=bn1.running_mean.numpy(),
    bn1_var=bn1.running_var.numpy(),
    bn2_gamma=bn2.weight.detach().numpy(),
    bn2_beta=bn2.bias.detach().numpy(),
    bn2_mean=bn2.running_mean.numpy(),
    bn2_var=bn2.running_var.numpy(),
)
print('wrote tests/data/golden.npz; logits:', logit.ravel())
