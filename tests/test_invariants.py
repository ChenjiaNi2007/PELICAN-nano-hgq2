"""Non-negotiable invariants: permutation invariance, masking, quantizer
granularity, and exact-zero representability (workspace CLAUDE.md)."""
import numpy as np
import pytest

from pelican_hgq2 import HGQ2Config, build_model


def random_batch(b=6, nmax=22, seed=1):
    rng = np.random.default_rng(seed)
    nobj = rng.integers(4, nmax - 2, size=b, endpoint=True)
    pmu = np.zeros((b, nmax, 4), dtype=np.float32)
    pmu[:, 0] = [1, 0, 0, 1]
    pmu[:, 1] = [1, 0, 0, -1]
    for e in range(b):
        p3 = rng.standard_normal((nobj[e], 3)).astype(np.float32) * 20.0
        pmu[e, 2:2 + nobj[e], 1:] = p3
        pmu[e, 2:2 + nobj[e], 0] = np.linalg.norm(p3, axis=-1)
    return pmu


@pytest.fixture(scope='module', params=['float', 'quant'])
def model(request):
    qcfg = HGQ2Config() if request.param == 'quant' else None
    return build_model(qcfg=qcfg)


def test_permutation_invariance(model):
    pmu = random_batch()
    perm = np.random.default_rng(2).permutation(pmu.shape[1])
    out = np.asarray(model(pmu, training=False))
    out_p = np.asarray(model(pmu[:, perm], training=False))
    np.testing.assert_allclose(out, out_p, rtol=1e-5, atol=1e-6)


def test_padding_independence(model):
    """Extending the padded region must not change the logits."""
    pmu = random_batch(nmax=20)
    padded = np.concatenate(
        [pmu, np.zeros((pmu.shape[0], 4, 4), dtype=np.float32)], axis=1)
    out = np.asarray(model(pmu, training=False))
    out_pad = np.asarray(model(padded, training=False))
    np.testing.assert_allclose(out, out_pad, rtol=1e-5, atol=1e-6)


def test_datalane_quantizers_are_per_tensor():
    """A learned bitwidth per (i, j) position would break permutation
    equivariance; every activation quantizer must hold scalar k/i/f."""
    model = build_model(qcfg=HGQ2Config())
    lanes = [model.input_quant, model.act_quant, model.output_quant,
             model.mixing_2to2.iq, model.mixing_2to0.iq]
    for lane in lanes:
        for var in lane.weights:
            assert int(np.prod(var.shape)) == 1, (
                f'{lane.name}/{var.name} has shape {var.shape}; datalane '
                'quantizers must be per-tensor')


def test_quantized_zero_is_exact():
    model = build_model(qcfg=HGQ2Config())
    zeros = np.zeros((2, 22, 4), dtype=np.float32)
    out = np.asarray(model(zeros, training=False))
    b2 = np.asarray(model.mixing_2to0.bias)
    # all-masked event: everything upstream of the final bias is exactly 0,
    # so the logit is exactly the (quantized) bias
    assert np.all(np.abs(out - b2) <= 2.0 ** -10)
