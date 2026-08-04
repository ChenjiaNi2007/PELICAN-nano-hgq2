"""nanoPELICAN in Keras 3 with HGQ2 quantization (JAX backend recommended).

Quantization points mirror the Brevitas QAT placement one-to-one (see
../nPELICAN-fpga docs and PELICAN-nano/CLAUDE.md):

  pmu_quant (optional) -> dots -> input_quant -> BN1 -> eq2to2 ops
  -> [post_agg_2to2 = QDense input quantizer] -> QDense(no bias) -> +bias/diag
  -> ReLU -> act_quant -> mask -> BN2 -> eq2to0 ops
  -> [post_agg_2to0 = QDense input quantizer] -> QDense(bias) -> output_quant

CRITICAL: all activation ("datalane") quantizers are forced to per-tensor
granularity (heterogeneous_axis=()). HGQ2's default is heterogeneous over all
non-batch axes, i.e. a learned bitwidth per (i, j) position — that would break
permutation invariance. Weight quantizers may be per-element (constants in the
unrolled firmware). Rounding/overflow use RND_CONV/SAT, matching the firmware's
AP_RND_CONV/AP_SAT contract.
"""
from dataclasses import dataclass, field
from math import sqrt
from typing import Optional, Tuple

import keras
from keras import ops

from .layers import (
    BiasDiag, MaskedBatchNorm, eq2to0_ops, eq2to2_ops, minkowski_dots,
)

KIF = Tuple[bool, int, int]  # (keep_negative, integer bits, fractional bits)


@dataclass
class HGQ2Config:
    """Initial (k, i, f) per quantization point; HGQ2 learns i/f from there.

    Defaults start on the current Brevitas 24-bit checkpoint's grid
    (ap_fixed<24, B-k> table in the workspace CLAUDE.md) so training begins at a
    known-good precision and the EBOP term shrinks bitwidths from above.
    """
    enabled: bool = True
    beta: float = 0.0                 # EBOP regularization strength; 0 = off
    round_mode: str = 'RND_CONV'      # matches firmware AP_RND_CONV
    overflow_mode: str = 'SAT'        # matches firmware AP_SAT
    het_weights: bool = True          # per-element weight bitwidths
    pmu_bits: Optional[KIF] = None    # None = float momenta into the dots
    freeze_pmu: bool = False          # pin pmu bits (no MACs -> no EBOP pressure,
                                      # so trainable pmu bits only ever grow)
    freeze_input: bool = False        # pin the d_ij grid likewise
    cap_bits: bool = False            # upper-bound every i/f at its init value:
                                      # bits may only shrink from the start point
    bit_penalty: float = 0.0          # MonoL1 on every i/f; 0 = keep hgq's
                                      # default 1e-8 (negligible in practice)
    input_bits: KIF = (True, 11, 12)
    post_agg_2to2_bits: KIF = (True, 3, 20)
    act_bits: KIF = (False, 3, 21)
    post_agg_2to0_bits: KIF = (True, 1, 22)
    output_bits: KIF = (True, 3, 20)
    w_2to2_bits: KIF = (True, 0, 23)
    w_2to0_bits: KIF = (True, 3, 20)
    i_bound: int = 16                 # learnable-integer-bits constraint |i| <= bound
    f_bound: int = 26                 # learnable-fractional-bits constraint


def _quantizer_config(qcfg: HGQ2Config, kif: KIF, place: str, het: bool,
                      frozen: bool = False):
    from hgq.config import QuantizerConfig
    from hgq.constraints import MinMax
    from hgq.regularizers import MonoL1

    k0, i0, f0 = kif
    # cap_bits pins the *upper* bound at the starting budget; the lower bound
    # stays wide open, so a capped run can only ever cost less than its preset.
    i_hi = float(i0) if qcfg.cap_bits else float(qcfg.i_bound)
    f_hi = float(f0) if qcfg.cap_bits else float(qcfg.f_bound)
    kwargs = dict(
        q_type='kif', place=place, k0=bool(k0), i0=float(i0), f0=float(f0),
        round_mode=qcfg.round_mode, overflow_mode=qcfg.overflow_mode,
        ic=MinMax(-qcfg.i_bound, i_hi),
        fc=MinMax(-qcfg.f_bound, f_hi),
        trainable=not frozen,
    )
    if qcfg.bit_penalty > 0.0 and not frozen:
        # hgq puts a MonoL1(1e-8) on every i/f by default — signed sum, so it
        # pushes bits down without the |i| sign flip a plain L1 would have.
        # It is the ONLY resource gradient the standalone quantizer lanes
        # (pmu, d_ij, ReLU out, logit) ever see, since EBOP counts only
        # multiplications inside Q-layers, and at 1e-8 it is ~4 orders of
        # magnitude too weak to hold them down: both w6p12 runs let pmu drift
        # from <12,10> to <22,12> against no opposition. Raising it is what
        # gives those lanes a real cost.
        pen = MonoL1(qcfg.bit_penalty)
        kwargs['ir'] = pen
        kwargs['fr'] = pen
    if het:
        kwargs['homogeneous_axis'] = ()      # every element gets its own bits
    else:
        kwargs['heterogeneous_axis'] = ()    # single (k,i,f) for the tensor
    return QuantizerConfig(**kwargs)


class PelicanNanoHGQ(keras.Model):
    """nanoPELICAN: LinEq2->2 -> ReLU -> LinEq2->0 on the Gram matrix of dots.

    call(pmu [B, N, 4]) -> logit w [B, 1]; the torch model's 2-class output is
    [-w, w], so CE there equals BCE-from-logits on 2w here (see losses in
    train.py). Masks are derived from E != 0 exactly like the torch collate.
    """

    def __init__(self, n_hidden=2, nave=49.0, drop_rate=0.05,
                 qcfg: Optional[HGQ2Config] = None, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.nave = float(nave)
        self.qcfg = qcfg
        self.quant = qcfg is not None and qcfg.enabled

        init22 = keras.initializers.RandomNormal(0.0, sqrt(2.0 / 6.0))
        init20 = keras.initializers.RandomNormal(0.0, sqrt(4.0 / (2.0 * n_hidden)))

        self.pmu_quant = None
        self.input_quant = None
        self.act_quant = None
        self.output_quant = None

        if self.quant:
            from hgq.config import LayerConfigScope
            from hgq.layers import QDense, Quantizer

            q = qcfg
            act = lambda kif: _quantizer_config(q, kif, 'datalane', het=False)
            wgt = lambda kif: _quantizer_config(q, kif, 'weight', het=q.het_weights)
            frz = lambda kif, frozen: _quantizer_config(
                q, kif, 'datalane', het=False, frozen=frozen)
            with LayerConfigScope(enable_ebops=q.beta > 0, beta0=q.beta):
                if q.pmu_bits is not None:
                    self.pmu_quant = Quantizer(
                        frz(q.pmu_bits, q.freeze_pmu), name='pmu_quant')
                self.input_quant = Quantizer(
                    frz(q.input_bits, q.freeze_input), name='input_quant')
                self.mixing_2to2 = QDense(
                    n_hidden, use_bias=False, kernel_initializer=init22,
                    iq_conf=act(q.post_agg_2to2_bits), kq_conf=wgt(q.w_2to2_bits),
                    name='mixing_2to2')
                self.act_quant = Quantizer(act(q.act_bits), name='act_quant')
                self.mixing_2to0 = QDense(
                    1, use_bias=True, kernel_initializer=init20,
                    iq_conf=act(q.post_agg_2to0_bits), kq_conf=wgt(q.w_2to0_bits),
                    name='mixing_2to0')
                self.output_quant = Quantizer(act(q.output_bits), name='output_quant')
        else:
            self.mixing_2to2 = keras.layers.Dense(
                n_hidden, use_bias=False, kernel_initializer=init22,
                name='mixing_2to2')
            self.mixing_2to0 = keras.layers.Dense(
                1, use_bias=True, kernel_initializer=init20, name='mixing_2to0')

        self.bn1 = MaskedBatchNorm(name='bn1')
        self.bn2 = MaskedBatchNorm(name='bn2')
        self.bias_diag = BiasDiag(name='bias_diag')
        self.drop1 = keras.layers.Dropout(drop_rate)
        self.drop2 = keras.layers.Dropout(drop_rate)
        self.drop_out = keras.layers.Dropout(drop_rate)

    def call(self, pmu, training=False):
        mask_p = ops.not_equal(pmu[..., 0], 0.0)                       # [B, N]
        edge = ops.logical_and(mask_p[:, :, None], mask_p[:, None, :])
        edge = edge[..., None]                                         # [B,N,N,1]

        if self.pmu_quant is not None:
            pmu = self.pmu_quant(pmu, training=training)
        d = minkowski_dots(pmu)[..., None]                             # [B,N,N,1]
        if self.input_quant is not None:
            d = self.input_quant(d, training=training)

        x = self.bn1(d, edge, training=training)
        x = self.drop1(x, training=training)
        t = eq2to2_ops(x, self.nave)                                   # [B,N,N,6]
        y = self.mixing_2to2(t, training=training)
        y = self.bias_diag(y)
        y = ops.relu(y)
        if self.act_quant is not None:
            y = self.act_quant(y, training=training)
        y = y * ops.cast(edge, y.dtype)

        z = self.bn2(y, edge, training=training)
        z = self.drop2(z, training=training)
        r = eq2to0_ops(z, self.nave)                                   # [B, 2H]
        w = self.mixing_2to0(r, training=training)
        w = self.drop_out(w, training=training)
        if self.output_quant is not None:
            w = self.output_quant(w, training=training)
        return w


def preset_config(preset: str, beta: float = 0.0, het_weights: bool = True,
                  bit_penalty: float = 0.0) -> HGQ2Config:
    """Named starting points for the bitwidth optimization.

    'init24'  — the Brevitas 24-bit checkpoint grids (original defaults).
    'w6p12'   — warm start at the hand-tuned resource optimum: the grids
                fpga_model_qat_w6a6i6p12_best.pt actually learned, read off with
                `model_loader.py --quant` (scale 2^-f per quantizer, i = B-k-f).
                Earlier revisions guessed the integer parts from the 24-bit
                beta=3e-7 run and were up to 5 binades too fine on post_agg_2to2
                and act — same bit count, wrong window. Results in results/
                predating 2026-08-04 used those wrong placements.
    'w6p12f'  — w6p12 with the boundary grids FROZEN (pmu at <12,10>, d_ij at
                its 6-bit grid). Standalone quantizers feed no MACs, so EBOP
                never pushes back on them and trainable boundary bits only ever
                grow (both w6p12 runs ballooned pmu to <22,12>); the hand flow
                proved these boundaries sufficient at AUC 0.9519, so pin them
                and let HGQ2 optimize the interior only.
    'w6p12c'  — w6p12 CAPPED: every lane starts at the hand-tuned budget and may
                only shrink (i/f upper-bounded at their init). Guarantees the
                result costs no more than the hand model on any lane, and shows
                where the model gives bits back. Pair with --bit-penalty, since
                with EBOP alone only the four Q-layer lanes feel any pressure
                and the rest just sit at the cap.
    """
    if preset == 'init24':
        return HGQ2Config(beta=beta, het_weights=het_weights,
                          bit_penalty=bit_penalty)
    if preset in ('w6p12', 'w6p12f', 'w6p12c'):
        frozen = preset == 'w6p12f'
        return HGQ2Config(
            beta=beta, het_weights=het_weights, bit_penalty=bit_penalty,
            cap_bits=preset == 'w6p12c',
            # (k, i, f) with f = -log2(learned scale); every lane is 6 bits
            # except pmu at 12. act_layer is signed in the checkpoint despite
            # being a ReLU output, so k=1 there too.
            pmu_bits=(True, 9, 2),            # ap_fixed<12,10>, scale 2^-2
            freeze_pmu=frozen, freeze_input=frozen,
            input_bits=(True, 9, -4),         # ap_fixed<6,10>, scale 2^4
            post_agg_2to2_bits=(True, 1, 4),  # ap_fixed<6,2>,  scale 2^-4
            act_bits=(True, 2, 3),            # ap_fixed<6,3>,  scale 2^-3
            post_agg_2to0_bits=(True, -1, 6),  # ap_fixed<6,0>,  scale 2^-6
            output_bits=(True, 2, 3),         # ap_fixed<6,3>,  scale 2^-3
            w_2to2_bits=(True, 0, 5),         # ap_fixed<6,1>,  scale 2^-5
            w_2to0_bits=(True, 2, 3),         # ap_fixed<6,3>,  scale 2^-3
        )
    raise ValueError(f'unknown preset {preset!r}')


def build_model(n_hidden=2, nmax=22, nave=49.0, drop_rate=0.05,
                qcfg: Optional[HGQ2Config] = None) -> PelicanNanoHGQ:
    model = PelicanNanoHGQ(n_hidden=n_hidden, nave=nave, drop_rate=drop_rate,
                           qcfg=qcfg)
    model(ops.zeros((1, nmax, 4)))  # build weights
    return model
