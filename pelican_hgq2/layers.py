"""Keras-3 ports of the nanoPELICAN building blocks.

Each function/layer is a line-by-line port of the PyTorch reference in
../PELICAN-nano/src (see that repo's CLAUDE.md for the math). The float path must
stay numerically equivalent to the torch float model; tests/test_parity_float.py
gates this against golden vectors produced by scripts/make_golden.py.

Conventions (identical to torch reference):
  - tensors are channel-last: x [B, N, N, C], masks [B, N, N, 1]
  - aggregation is sum-then-divide by the FIXED hyperparameter nave (=average_nobj,
    49 for toptag), never by the per-event multiplicity ("normalize-late")
"""
import keras
from keras import ops


def minkowski_dots(pmu):
    """d_ij = p_i . p_j with metric (+,-,-,-). pmu [B, N, 4] -> [B, N, N]."""
    g = ops.einsum('bim,bjm->bij', pmu, pmu)
    t = ops.einsum('bi,bj->bij', pmu[..., 0], pmu[..., 0])
    return 2.0 * t - g


def eq2to2_ops(x, nave):
    """The 6 nano permutation-equivariant 2->2 aggregators ('s' config).

    x [B, N, N, C] -> [B, N, N, C*6], flattened so that basis index b is fastest
    (matches torch permute(0,3,4,1,2).reshape and the firmware w1_2to2[h*6+b]
    element order). Op order is the torch eops_2_to_2 order:
      0: identity   1: diag(colsum)  2: colsum[j] rows  3: colsum[i] cols
      4: totsum     5: diag(totsum)
    """
    n = x.shape[1]
    shape = ops.shape(x)
    colsum = ops.sum(x, axis=1) / nave              # [B, N, C], indexed by j
    totsum = ops.sum(x, axis=(1, 2)) / nave ** 2    # [B, C]
    eye = ops.eye(n, dtype=x.dtype)[None, :, :, None]   # [1, N, N, 1]

    op1 = x
    op2 = eye * colsum[:, None, :, :]
    op3 = ops.broadcast_to(colsum[:, None, :, :], shape)
    op4 = ops.broadcast_to(colsum[:, :, None, :], shape)
    op5 = ops.broadcast_to(totsum[:, None, None, :], shape)
    op6 = eye * totsum[:, None, None, :]

    stacked = ops.stack([op1, op2, op3, op4, op5, op6], axis=-1)  # [B,N,N,C,6]
    return ops.reshape(stacked, (shape[0], n, n, x.shape[-1] * 6))


def eq2to0_ops(x, nave):
    """The 2 nano 2->0 aggregators: total sum and trace.

    x [B, N, N, C] -> [B, C*2] with aggregator index a fastest (firmware
    w2_2to0[h*2+a] order: a=0 sum, a=1 trace).
    """
    tot = ops.sum(x, axis=(1, 2)) / nave ** 2                    # [B, C]
    trace = ops.sum(ops.diagonal(x, axis1=1, axis2=2), axis=-1) / nave  # [B, C]
    stacked = ops.stack([tot, trace], axis=-1)                   # [B, C, 2]
    return ops.reshape(stacked, (ops.shape(x)[0], x.shape[-1] * 2))


class MaskedBatchNorm(keras.layers.Layer):
    """Port of MaskedBatchNorm2d: masked statistics, padded entries forced to 0.

    call(x [B,N,N,C], mask [B,N,N,1] bool). Training uses biased batch variance
    for normalization and stores the unbiased variance in the moving average,
    exactly like the torch reference. BatchNorm parameters stay float in the
    quant model (firmware invariant: BN is explicit, never folded).
    """

    def __init__(self, momentum=0.1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        c = input_shape[-1]
        self.gamma = self.add_weight(name='gamma', shape=(c,), initializer='ones')
        self.beta = self.add_weight(name='beta', shape=(c,), initializer='zeros')
        self.moving_mean = self.add_weight(
            name='moving_mean', shape=(c,), initializer='zeros', trainable=False)
        self.moving_var = self.add_weight(
            name='moving_var', shape=(c,), initializer='ones', trainable=False)

    def call(self, x, mask, training=False):
        mask_f = ops.cast(mask, x.dtype)
        if training:
            n = ops.sum(mask_f)
            p = mask_f / n
            mean = ops.sum(p * x, axis=(0, 1, 2))
            var = ops.sum(p * x ** 2, axis=(0, 1, 2)) - mean ** 2
            m = self.momentum
            self.moving_mean.assign(m * mean + (1.0 - m) * self.moving_mean)
            self.moving_var.assign(
                m * var * n / (n - 1.0) + (1.0 - m) * self.moving_var)
        else:
            mean, var = self.moving_mean, self.moving_var
        y = (x - mean) / ops.sqrt(var + self.epsilon)
        y = y * self.gamma + self.beta
        return ops.where(ops.cast(mask, 'bool'), y, ops.zeros_like(y))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(momentum=self.momentum, epsilon=self.epsilon)
        return cfg


class BiasDiag(keras.layers.Layer):
    """Adds bias (everywhere) + diag_bias (diagonal only), port of Eq2to2's
    explicit bias/diag_bias parameters (the mixing Dense runs without bias).
    Biases stay float during training; export snaps them to the pre-activation
    grid (same policy as Brevitas D6 float-bias)."""

    def build(self, input_shape):
        c = input_shape[-1]
        self.bias = self.add_weight(name='bias', shape=(c,), initializer='zeros')
        self.diag_bias = self.add_weight(
            name='diag_bias', shape=(c,), initializer='zeros')

    def call(self, x):
        n = x.shape[1]
        eye = ops.eye(n, dtype=x.dtype)[None, :, :, None]
        return x + self.bias + eye * self.diag_bias
