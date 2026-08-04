"""The 'w6p12c' preset must be a one-way door: every lane starts at the
hand-tuned w6a6i6p12 budget and can only ever shrink from there, so any run
using it is guaranteed to cost no more than the hand model on any lane."""
import keras
import numpy as np
import pytest

from pelican_hgq2 import build_model, contract_of, preset_config, quant_points
from train import nano_bce

from test_training_smoke import synthetic

HAND_BITS = {'pmu': 12, 'input': 6, 'post_agg_2to2': 6, 'w_2to2': 6,
             'act': 6, 'post_agg_2to0': 6, 'w_2to0': 6, 'output': 6}


def test_capped_preset_starts_at_hand_budget():
    model = build_model(qcfg=preset_config('w6p12c'))
    assert {n: c['bits'] for n, c in contract_of(model).items()} == HAND_BITS


@pytest.mark.parametrize('preset', ['w6p12', 'w6p12c'])
def test_only_capped_preset_bounds_bits_from_above(preset):
    """Each i/f constraint's upper bound equals its own init iff capped."""
    model = build_model(qcfg=preset_config(preset))
    capped = []
    for name, layer in quant_points(model).items():
        q = layer.quantizer
        for var in (q._i, q._f):
            init = float(np.asarray(var).max())
            hi = var.constraint.max_value
            # constraint is what the optimizer applies after each step
            clamped = float(np.asarray(var.constraint(np.asarray(var) + 8.0)).max())
            capped.append(hi == init and clamped == init)
    assert all(capped) if preset == 'w6p12c' else not any(capped)


@pytest.mark.parametrize('lane', ['pmu_quant', 'input_quant', 'act_quant',
                                  'output_quant'])
def test_bit_penalty_reaches_the_standalone_lanes(lane):
    """EBOP is blind to non-MAC quantizers; the bit penalty must not be.

    pmu/d_ij/ReLU-out/logit are standalone Quantizer layers with no
    multiplications inside them, so beta leaves them with an identically zero
    resource loss — this is why both w6p12 runs let pmu balloon to <22,12>.
    """
    def lams(model):
        q = getattr(model, lane).quantizer
        return [q._i.regularizer.l1, q._f.regularizer.l1]

    for beta in (0.0, 1e-6):
        m = build_model(qcfg=preset_config('w6p12c', beta=beta))
        assert lams(m) == [1e-8, 1e-8], (
            f'{lane} bit pressure changed with beta — EBOP is supposed to be '
            'blind to standalone quantizers')
    assert lams(build_model(qcfg=preset_config(
        'w6p12c', bit_penalty=1e-3))) == [1e-3, 1e-3]


def test_capped_run_only_ever_shrinks():
    # lr/lambda are cranked far above the training defaults so a few dozen
    # steps move the bit variables measurably; the point is direction, not
    # the operating point.
    keras.utils.set_random_seed(0)
    pmu, y = synthetic()
    model = build_model(qcfg=preset_config('w6p12c', bit_penalty=1e-2))
    model.compile(optimizer=keras.optimizers.Adam(0.1), loss=nano_bce)
    model.fit(pmu, y, epochs=5, batch_size=16, verbose=0)
    after = {n: c['bits'] for n, c in contract_of(model).items()}
    assert all(after[n] <= HAND_BITS[n] for n in HAND_BITS), after
    assert any(after[n] < HAND_BITS[n] for n in HAND_BITS), (
        f'bit penalty produced no reduction at all: {after}')
