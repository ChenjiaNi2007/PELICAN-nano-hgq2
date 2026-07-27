"""Float Keras port must reproduce the torch reference logits bit-close.

Golden vectors come from scripts/make_golden.py (run in the PELICAN-nano venv).
If this test fails after a layer edit, the port has diverged from the reference
math — fix the port, do not regenerate the goldens to match.
"""
import os

import numpy as np
import pytest

from pelican_hgq2 import build_model

GOLDEN = os.path.join(os.path.dirname(__file__), 'data', 'golden.npz')


def load_torch_weights(model, g):
    model.mixing_2to2.kernel.assign(g['w1'].T)      # torch [H,6] -> keras [6,H]
    model.bias_diag.bias.assign(g['b1'])
    model.bias_diag.diag_bias.assign(g['b1_diag'])
    model.mixing_2to0.kernel.assign(g['w2'].T)      # torch [1,2H] -> keras [2H,1]
    model.mixing_2to0.bias.assign(g['b2'])
    for bn, tag in ((model.bn1, 'bn1'), (model.bn2, 'bn2')):
        bn.gamma.assign(g[f'{tag}_gamma'])
        bn.beta.assign(g[f'{tag}_beta'])
        bn.moving_mean.assign(g[f'{tag}_mean'])
        bn.moving_var.assign(g[f'{tag}_var'])


@pytest.mark.skipif(not os.path.exists(GOLDEN),
                    reason='golden.npz missing - run scripts/make_golden.py')
def test_float_parity_vs_torch():
    g = np.load(GOLDEN)
    model = build_model(n_hidden=g['w1'].shape[0], nmax=g['pmu'].shape[1],
                        qcfg=None)
    load_torch_weights(model, g)
    logit = np.asarray(model(g['pmu'], training=False))
    np.testing.assert_allclose(logit, g['logit'], rtol=1e-4, atol=1e-5)
