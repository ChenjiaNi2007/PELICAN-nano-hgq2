"""End-to-end smoke: quant model trains a couple of steps on synthetic data,
loss is finite, EBOP regularization contributes when beta > 0, and BN moving
statistics actually update under JAX's stateless training."""
import keras
import numpy as np

from pelican_hgq2 import HGQ2Config, build_model
from train import nano_bce


def synthetic(b=64, nmax=22, seed=3):
    rng = np.random.default_rng(seed)
    pmu = np.zeros((b, nmax, 4), dtype=np.float32)
    pmu[:, 0] = [1, 0, 0, 1]
    pmu[:, 1] = [1, 0, 0, -1]
    n = rng.integers(4, nmax - 2, size=b)
    for e in range(b):
        p3 = rng.standard_normal((n[e], 3)).astype(np.float32) * 20.0
        pmu[e, 2:2 + n[e], 1:] = p3
        pmu[e, 2:2 + n[e], 0] = np.linalg.norm(p3, axis=-1)
    y = rng.integers(0, 2, size=b).astype(np.float32)
    return pmu, y


def test_fit_smoke_and_bn_updates():
    keras.utils.set_random_seed(0)
    pmu, y = synthetic()
    model = build_model(qcfg=HGQ2Config(beta=1e-6))
    bn_mean_before = np.asarray(model.bn1.moving_mean).copy()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=nano_bce)
    hist = model.fit(pmu, y, epochs=2, batch_size=32, verbose=0)
    losses = hist.history['loss']
    assert np.all(np.isfinite(losses))
    bn_mean_after = np.asarray(model.bn1.moving_mean)
    assert not np.allclose(bn_mean_before, bn_mean_after), (
        'BN moving statistics did not update during fit (JAX stateless '
        'training is dropping non-trainable variable updates)')


def test_ebops_present_when_beta_positive():
    model = build_model(qcfg=HGQ2Config(beta=1e-6))
    pmu, y = synthetic(b=8)
    model(pmu, training=True)
    assert len(model.losses) > 0, 'expected EBOP regularization losses'
